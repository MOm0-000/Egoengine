"""TACO world-GT ingestion and two-hand MINK candidates, without physics claims."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import mink
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from ..evaluation.taco_surface import (
    MANO21_SOURCE, MANO_TIP_VERTICES, _load_model_data,
    load_taco_mano_sequence, reconstruct_taco_mano,
)
from .mink import (_FrameDisplacementLimit, _StrictCollisionLimit,
                   _enable_planning_collision_masks, _explicit_collision_groups,
                   _joint_velocity_limits)
from .collision_audit import audit_intrahand_trajectory, explicit_hand_pairs, distances
from .schema import validate_human_reference

SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIPS = (4, 8, 12, 16, 20)
DIPS = (3, 7, 11, 15, 19)
MANO_DISTALS = (15, 3, 6, 12, 9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_frame(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    z = np.array(direction, dtype=np.float64, copy=True)
    if np.linalg.norm(z) < 1e-8:
        raise ValueError("degenerate distal bone direction")
    z /= np.linalg.norm(z)
    x = np.array(normal, dtype=np.float64, copy=True)
    x -= np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("degenerate palm normal")
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def mano_fingertip_frames(pose_path: Path, shape_path: Path, model_path: Path,
                         side: str, expected_joints: np.ndarray) -> tuple[np.ndarray, dict]:
    """Transport fixed neutral fingertip axes with the released MANO rotations."""
    import smplx
    import torch
    from smplx.utils import Struct
    from smplx.lbs import batch_rigid_transform

    poses, _, _, keys = load_taco_mano_sequence(pose_path, shape_path)
    _, recovered, _, _, reconstructed_keys = reconstruct_taco_mano(
        pose_path, shape_path, model_path, side=side)
    expected = np.asarray(expected_joints, dtype=float)
    if expected.shape != recovered.shape or not np.isfinite(expected).all():
        raise ValueError("released joint reference must match reconstructed MANO shape")
    if keys != reconstructed_keys or [int(k) for k in keys] != list(range(1, len(expected) + 1)):
        raise ValueError("MANO pose keys do not match released joint rows")
    error = float(np.linalg.norm(recovered - expected, axis=-1).max())
    if error > 1e-6:
        raise ValueError(f"MANO rotations do not reproduce released joint GT: {error} m")

    # Calibration is a model convention: zero betas and zero joint rotations,
    # never the episode's first pose or a per-frame palm-normal projection.
    layer = smplx.MANO("unused", data_struct=Struct(**_load_model_data(model_path)),
                       is_rhand=side == "right", use_pca=False, flat_hand_mean=True,
                       create_transl=False)
    with torch.no_grad():
        neutral = layer(global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45),
                        betas=torch.zeros(1, 10))
    joints = torch.cat([neutral.joints, neutral.vertices[:, MANO_TIP_VERTICES[side]]], 1)
    points = joints[0, MANO21_SOURCE].numpy().astype(float)
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    if side == "left":
        normal = -normal
    calibration = np.stack([geometric_frame(normal, points[t] - points[d]) for t, d in zip(TIPS, DIPS)])
    rotations = Rotation.from_rotvec(poses.astype(float).reshape(-1, 3)).as_matrix().reshape(-1, 16, 3, 3)
    with torch.no_grad():
        _, transforms = batch_rigid_transform(
            torch.from_numpy(rotations), torch.zeros(len(poses), 16, 3, dtype=torch.float64), layer.parents)
    global_distal = transforms.numpy()[:, MANO_DISTALS, :3, :3]
    frames = global_distal @ calibration
    directions = expected[:, TIPS] - expected[:, DIPS]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    direction_error = np.rad2deg(np.arccos(np.clip((directions * frames[..., 2]).sum(-1), -1, 1)))
    step = Rotation.from_matrix((frames[:-1].swapaxes(-1, -2) @ frames[1:]).reshape(-1, 3, 3)).magnitude()
    report = dict(source="released_MANO_local_rotations_via_smplx_rigid_FK",
        neutral_calibration="MANO_v1.2_zero_betas_flat_mean_identity_pose; palm/distal_axes",
        calibration_fitted_to_episode=False, per_frame_palm_projection=False,
        distal_joint_indices=list(MANO_DISTALS), neutral_distal_frames=calibration.tolist(),
        reconstructed_joint_max_error_m=error,
        distal_axis_vs_observed_bone_p95_deg=np.percentile(direction_error, 95, axis=0).tolist(),
        distal_axis_vs_observed_bone_max_deg=direction_error.max(axis=0).tolist(),
        max_frame_rotation_step_deg=float(np.rad2deg(step).max()),
        source_frame_keys=list(keys),
        inputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in (pose_path, shape_path, model_path)])
    return frames, report


def pose7(transform: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], q[[3, 0, 1, 2]]]


def differentiate(model: mujoco.MjModel, qpos: np.ndarray, dt: float) -> np.ndarray:
    if len(qpos) < 2 or dt <= 0:
        raise ValueError("differentiation requires at least two frames and positive dt")
    qvel = np.empty((len(qpos), model.nv))
    for i in range(len(qpos)):
        first, last = max(0, i - 1), min(len(qpos) - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], (last - first) * dt, qpos[first], qpos[last])
    return qvel


def prepare(dev4: Path, scene: Path, sequence: str, episode: str, output: Path,
            *, mano_model_dir: Path) -> Path:
    """Preserve source rows; fail on mismatched modalities instead of truncating."""
    if output.exists():
        raise FileExistsError(output)
    hands_dir = dev4 / "hand_poses/Hand_Poses" / sequence
    object_dir = dev4 / "object_poses/Object_Poses" / sequence
    joints_path = hands_dir / "hand_joints.npy"
    joints = np.load(joints_path, allow_pickle=False).astype(np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3) or not np.isfinite(joints).all():
        raise ValueError("TACO joints must be finite (T,2,21,3), left then right")
    n = len(joints)
    inputs = [joints_path, scene]
    translation_errors = {}
    for hand_index, side in enumerate(("left", "right")):
        path = hands_dir / f"{side}_hand.pkl"
        with path.open("rb") as stream:
            data = pickle.load(stream)
        keys = sorted(data, key=int)
        if [int(key) for key in keys] != list(range(1, n + 1)):
            raise ValueError(f"{side} PKL source frame IDs do not match joint rows")
        translation = np.stack([np.asarray(data[key]["hand_trans"]) for key in keys])
        error = float(np.linalg.norm(translation - joints[:, hand_index, 0], axis=-1).max())
        if error > 1e-5:
            raise ValueError(f"{side} hand order/world frame disagreement: {error} m")
        translation_errors[side] = error
        inputs.append(path)

    objects, meshes = [], []
    for role in ("tool", "target"):
        paths = sorted(object_dir.glob(f"{role}_*.npy"))
        if len(paths) != 1:
            raise ValueError(f"expected exactly one {role} GT file")
        poses = np.load(paths[0], allow_pickle=False).astype(np.float64)
        if poses.shape != (n, 4, 4):
            raise ValueError(f"{role} frame count differs from hand GT")
        objects.append(poses)
        object_id = paths[0].stem.split("_", 1)[1]
        mesh_path = dev4 / "object_models/object_models_released" / f"{object_id}_cm.obj"
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.apply_scale(0.01)
        meshes.append(mesh)
        inputs.extend([paths[0], mesh_path])
    objects = np.stack(objects, axis=1)
    video = dev4 / "rgb" / f"{episode}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    if count != n or not np.isclose(fps, 30.0):
        raise ValueError(f"RGB has {count} frames/{fps} fps, GT has {n}; explicit alignment needed")
    inputs.append(video)

    # Same fixed pseudo-base convention as the copied freeze_taco_base_frame.
    target_world = meshes[1].vertices @ objects[0, 1, :3, :3].T + objects[0, 1, :3, 3]
    transform = np.eye(4)
    transform[:2, 3] = [0.6, 0.0] - objects[0, :, :2, 3].mean(axis=0)
    transform[2, 3] = 0.72 - target_world[:, 2].min()
    joints_sim = joints[:, [1, 0]] + transform[:3, 3]
    objects_sim = np.einsum("ij,tojk->toik", transform, objects)

    model = mujoco.MjModel.from_xml_path(str(scene))
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    wrists = np.tile(np.eye(4), (n, 2, 1, 1))
    tips = np.tile(np.eye(4), (n, 2, 5, 1, 1))
    orientation_audit = {}
    for h, side in enumerate(SIDES):
        shape_path = hands_dir / f"{side}_hand_shape.pkl"
        model_path = mano_model_dir / f"MANO_{side.upper()}.pkl"
        distal_frames, orientation_audit[side] = mano_fingertip_frames(
            hands_dir / f"{side}_hand.pkl", shape_path, model_path, side, joints[:, 1 - h])
        inputs.extend([shape_path, model_path])
        palm = model.site(f"{side}_palm").id
        root = model.body(f"{side}_hand_link").id
        middle = model.body(f"{side}_hand_mid_link1").id
        robot_frame = geometric_frame(zero.site_xmat[palm].reshape(3, 3)[:, 0],
                                      zero.xpos[middle] - zero.xpos[root])
        wrist_offset = robot_frame.T @ zero.xmat[root].reshape(3, 3)
        local_tip_frames = []
        for finger in FINGERS:
            site = model.site(f"{side}_{finger}_tip").id
            distal = zero.site_xpos[site] - zero.xpos[model.site_bodyid[site]]
            local_tip_frames.append(zero.site_xmat[site].reshape(3, 3).T @
                                    geometric_frame(robot_frame[:, 0], distal))
        for frame in range(n):
            points = joints_sim[frame, h]
            normal = np.cross(points[5] - points[0], points[17] - points[0])
            if side == "left":
                normal = -normal
            human_frame = geometric_frame(normal, points[9] - points[0])
            wrists[frame, h, :3, :3] = human_frame @ wrist_offset
            wrists[frame, h, :3, 3] = points[0]
            for k, tip in enumerate(TIPS):
                tips[frame, h, k, :3, :3] = transform[:3, :3] @ distal_frames[frame, k] @ local_tip_frames[k].T
                tips[frame, h, k, :3, 3] = points[tip]
    arrays = dict(frame_indices=np.arange(n), timestamps_s=np.arange(n) / fps,
                  hand_order=np.asarray(SIDES), object_roles=np.asarray(["tool", "target"]),
                  T_sim_world=transform, joint_positions_sim=joints_sim,
                  T_sim_wrist_target=wrists, T_sim_fingertip_target=tips,
                  T_sim_object_reference=objects_sim, valid_hand=np.ones((n, 2), dtype=bool),
                  valid_fingertip_orientation=np.ones((n, 2, 5), dtype=bool),
                  confidence_hand=np.ones((n, 2)), confidence_fingertip=np.ones((n, 2, 5)))
    arrays["fingertip_orientation_source"] = np.asarray("MANO_rotational_FK_fixed_neutral_calibration_v1")
    validate_human_reference(arrays)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    path = output / "human_reference.npz"
    np.savez_compressed(path, **arrays)
    report = dict(status="GT_input_not_robot_demonstration", episode=episode, frames=n, fps=fps,
                  hand_order=list(SIDES), object_roles=["tool", "target"],
                  hand_pkl_joint_translation_max_error_m=translation_errors,
                  T_sim_world=transform.tolist(), depth_not_used=True,
                  coordinate_source="TACO world metric GT; fixed shared transform, not per-frame camera coordinates",
                  axis_convention="object pair center at x=0.6,y=0; +Z up; target initial bottom at table z=0.72",
                  orientation_source="MANO rotational FK for fingertips; unchanged landmark wrist targets; fixed model calibrations are not published author offsets",
                  fingertip_orientation_audit=orientation_audit,
                  table_alignment_independently_calibrated=False,
                  code_inputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in
                               (Path(__file__), Path(__file__).parents[1] / "evaluation/taco_surface.py")],
                  inputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in inputs])
    (output / "input_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return path


def retarget(scene: Path, human_path: Path, settings: dict, output: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("expected two 18-DoF floating hands and two passive free objects")
    with np.load(human_path, allow_pickle=False) as data:
        human = dict(data)
    validate_human_reference(human)
    dt = float(np.diff(human["timestamps_s"])[0])
    n = len(human["frame_indices"])
    primal_tolerance = float(settings.get("solver_primal_tolerance", 1e-9))
    dual_tolerance = float(settings.get("solver_dual_tolerance", 1e-9))
    planning_buffer = float(settings.get("planning_collision_buffer_m", 0.0))
    accepted_min_distance = float(
        settings.get("accepted_min_self_collision_distance_m", -1e-6)
    )
    acceptance_slack = float(
        settings.get("collision_acceptance_numerical_slack_m", 1e-12)
    )
    local_projection_buffer = float(
        settings.get("local_surface_guard_projection_buffer_m", 0.0)
    )
    if (not np.isfinite(primal_tolerance) or primal_tolerance <= 0.0
            or not np.isfinite(dual_tolerance) or dual_tolerance <= 0.0):
        raise ValueError("solver tolerances must be positive and finite")
    if not np.isfinite(planning_buffer) or planning_buffer < 0.0:
        raise ValueError("planning collision buffer must be finite and nonnegative")
    if not np.isfinite(accepted_min_distance) or accepted_min_distance > 0.0:
        raise ValueError("accepted minimum self-collision distance must be finite and nonpositive")
    if (not np.isfinite(acceptance_slack) or acceptance_slack < 0.0
            or acceptance_slack > 1e-9):
        raise ValueError("collision acceptance numerical slack must be in [0, 1e-9] m")
    if (not np.isfinite(local_projection_buffer) or local_projection_buffer < 0.0
            or local_projection_buffer > 1e-6):
        raise ValueError("local surface-guard projection buffer must be in [0, 1e-6] m")
    local_minimum_distance = accepted_min_distance + local_projection_buffer
    if local_minimum_distance > 0.0:
        raise ValueError("local surface-guard projection target must remain nonpositive")
    config = mink.Configuration(model)
    tasks = []
    for side in SIDES:
        tasks.append(mink.FrameTask(f"{side}_hand_link", "body", position_cost=0.0,
                                   orientation_cost=settings["wrist_orientation_cost"], lm_damping=1e-3))
        tasks.extend(mink.FrameTask(f"{side}_{finger}_tip", "site",
                                   position_cost=settings["fingertip_position_cost"],
                                   orientation_cost=settings["fingertip_orientation_cost"], lm_damping=1e-3)
                     for finger in FINGERS)
    hand_geoms = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("collision_hand_")]
    pairs = explicit_hand_pairs(model)
    if not pairs:
        raise ValueError("retargeting requires an audited explicit runtime self-collision contract")
    _enable_planning_collision_masks(model, hand_geoms, [])
    groups = _explicit_collision_groups(model, mujoco, hand_geom_ids=set(hand_geoms))
    local_guard_token = "_palm_thumb_surface_guard_"
    local_groups = [group for group in groups if any(
        local_guard_token in name for side in group for name in side
    )]
    standard_groups = [group for group in groups if group not in local_groups]
    collisions = []
    if standard_groups:
        inner = mink.CollisionAvoidanceLimit(
            model, standard_groups,
            minimum_distance_from_collisions=planning_buffer,
            collision_detection_distance=0.02,
            include_explicit_pairs=True,
        )
        collision = _StrictCollisionLimit(
            inner, mujoco, minimum_distance=planning_buffer,
            depenetration_step=0.002,
        )
        collision.enabled = True
        collisions.append(collision)
    if local_groups:
        # Tiny convex mesh pairs can return an exact zero without a positive
        # separation witness.  A +2 um planning buffer would activate hundreds
        # of such non-contact rows.  Keep the same runtime pairs in MINK, but
        # activate this local family only after actual penetration and enforce
        # the independently declared -1 um acceptance bound.
        local_inner = mink.CollisionAvoidanceLimit(
            model, local_groups,
            minimum_distance_from_collisions=local_minimum_distance,
            collision_detection_distance=0.0,
            include_explicit_pairs=True,
        )
        local_collision = _StrictCollisionLimit(
            local_inner, mujoco, minimum_distance=local_minimum_distance,
            depenetration_step=0.002, deepest_invalid_only=True,
        )
        local_collision.enabled = True
        collisions.append(local_collision)
    constrained_pairs = set().union(*(set(limit.geom_id_pairs) for limit in collisions))
    if constrained_pairs != set(pairs):
        raise ValueError("MINK filtered out runtime self pairs; IK/physics contract differs")
    collision_pairs = tuple(sorted(constrained_pairs))
    velocity_map = _joint_velocity_limits(model, mujoco, settings["velocity_limits"])
    displacement = _FrameDisplacementLimit(model, mujoco, velocity_map, mink.Constraint)
    limits = [mink.ConfigurationLimit(model), *collisions, displacement]
    locks = [mink.DofFreezingTask(model, list(range(36, 48)))]
    qpos = np.empty((n, model.nq))
    tip_errors = np.empty((n, 2, 5))
    wrist_errors = np.empty((n, 2))
    self_distances = np.empty(n)
    worst_pairs = []
    previous = None
    substeps = int(settings["max_iterations_per_frame"])
    if substeps < 1:
        raise ValueError("max_iterations_per_frame must be positive")

    def fail(frame, reason):
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "failed_kinematic_prefix.npz",
                            qpos=np.vstack([qpos[:frame], config.q]),
                            frame_indices=human["frame_indices"][:frame + 1],
                            failed_frame=frame)
        report = dict(status="failed_kinematic_candidate", failed_frame=frame,
                      reason=reason, scene=str(scene.resolve()), scene_sha256=sha256(scene),
                      self_min_distance_m=float(distances(model, config.data, pairs).min()),
                      strict_gate_passed=False, rl_validation_completed=False)
        (output / "retarget_failure.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(reason)

    def solve(tracking_tasks):
        try:
            return mink.solve_ik(config, tracking_tasks, dt / substeps, solver="daqp",
                                 damping=1e-5, limits=limits, constraints=locks,
                                 primal_tol=primal_tolerance, dual_tol=dual_tolerance)
        except Exception as error:
            fail(frame, f"QP failed at frame {frame}: {error}")

    for frame in range(n):
        q = config.q.copy()
        for h, side in enumerate(SIDES):
            objadr = int(model.joint(f"{side}_object_joint").qposadr[0])
            q[objadr:objadr + 7] = pose7(human["T_sim_object_reference"][frame, h])
            wrist = human["T_sim_wrist_target"][frame, h]
            if frame == 0:
                q[h * 18:h * 18 + 3] = wrist[:3, 3]
                angles = Rotation.from_matrix(wrist[:3, :3]).as_euler("ZXY")
                q[h * 18 + 3:h * 18 + 6] = angles * [1, 1, -1]
            for k, target in enumerate([wrist, *human["T_sim_fingertip_target"][frame, h]]):
                tasks[h * 6 + k].set_target(mink.SE3.from_matrix(target))
        config.update(q)
        displacement.set_previous(previous, dt if previous is not None else None)
        for _ in range(max(80, substeps) if frame == 0 else substeps):
            velocity = solve(tasks)
            config.integrate_inplace(velocity, dt / substeps)
        for _ in range(128):
            if (distances(model, config.data, pairs).min()
                    >= accepted_min_distance - acceptance_slack):
                break
            velocity = solve(())
            config.integrate_inplace(velocity, dt / substeps)
        else:
            fail(frame, f"self-collision feasibility projection failed at frame {frame}")
        qpos[frame] = config.q
        previous = config.q.copy()
        for h, side in enumerate(SIDES):
            for k, finger in enumerate(FINGERS):
                site = model.site(f"{side}_{finger}_tip").id
                tip_errors[frame, h, k] = np.linalg.norm(config.data.site_xpos[site] - human["T_sim_fingertip_target"][frame, h, k, :3, 3])
            body = model.body(f"{side}_hand_link").id
            relative = config.data.xmat[body].reshape(3, 3).T @ human["T_sim_wrist_target"][frame, h, :3, :3]
            wrist_errors[frame, h] = Rotation.from_matrix(relative).magnitude()
        self_distances[frame] = min((mujoco.mj_geomDistance(model, config.data, a, b, 0.05, None)
                                    for a, b in collision_pairs), default=0.05)
        if frame == 0 or self_distances[frame] < self_distances[:frame].min():
            worst_pairs = sorted((dict(geom1=model.geom(a).name, geom2=model.geom(b).name,
                                       distance_m=float(mujoco.mj_geomDistance(model, config.data, a, b, 0.05, None)))
                                  for a, b in collision_pairs), key=lambda item: item["distance_m"])[:12]
        if frame % 25 == 0:
            print(f"MINK {frame + 1}/{n}: tip mean {tip_errors[frame].mean():.4f} m; self clearance {self_distances[frame]:.6f} m", flush=True)
    qvel = differentiate(model, qpos, dt)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("nonfinite MINK candidate")
    intrahand_audit = audit_intrahand_trajectory(model, qpos)
    addresses = np.array([int(model.joint(name).qposadr[0]) for name in velocity_map])
    speeds = np.array(list(velocity_map.values()))
    ranges = np.array([model.joint(name).range for name in velocity_map])
    joint_margin = np.minimum(qpos[:, addresses] - ranges[:, 0],
                              ranges[:, 1] - qpos[:, addresses]).min(axis=1)
    velocity_ratio = np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt * speeds)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "robot_reference.npz", qpos=qpos, qvel=qvel, ctrl=qpos[:, :36],
                        frequency=1.0 / dt, frame_indices=human["frame_indices"],
                        timestamps_s=human["timestamps_s"], hand_order=np.asarray(SIDES),
                        object_roles=np.asarray(["tool", "target"]),
                        fingertip_position_error_m=tip_errors, wrist_orientation_error_rad=wrist_errors,
                        self_collision_distance_m=self_distances, joint_limit_min_margin=joint_margin,
                        frame_velocity_max_ratio=velocity_ratio.max(axis=1))
    intrahand_path = output / "intrahand_collision_audit.json"
    intrahand_artifact = dict(
        scene=str(scene.resolve()), scene_sha256=sha256(scene),
        human_reference=str(human_path.resolve()),
        human_reference_sha256=sha256(human_path), **intrahand_audit)
    intrahand_path.write_text(json.dumps(intrahand_artifact, indent=2) + "\n")
    report = dict(status="MINK_kinematic_candidate_not_physics_validated", frames=n,
                  scene=str(scene.resolve()), scene_sha256=sha256(scene),
                  human_reference=str(human_path.resolve()), human_reference_sha256=sha256(human_path),
                  mink_module=str(Path(mink.__file__).resolve()), inherited_settings=settings,
                  effective_settings=dict(source_dt_s=dt, qp_integration_dt_s=dt / substeps,
                                          iterations_per_frame=substeps, first_frame_iterations=max(80, substeps),
                                          wrist_orientation_cost=settings["wrist_orientation_cost"],
                                          fingertip_position_cost=settings["fingertip_position_cost"],
                                          fingertip_orientation_cost=settings["fingertip_orientation_cost"],
                                          posture_cost=0.0,
                                          planning_collision_buffer_m=planning_buffer,
                                          standard_self_collision_pair_count=len(standard_groups),
                                          local_surface_guard_pair_count=len(local_groups),
                                          local_surface_guard_collision_detection_distance_m=(
                                              0.0 if local_groups else None
                                          ),
                                          local_surface_guard_minimum_distance_m=(
                                              local_minimum_distance if local_groups else None
                                          ),
                                          local_surface_guard_projection_buffer_m=(
                                              local_projection_buffer if local_groups else None
                                          ),
                                          local_surface_guard_deepest_invalid_only=bool(local_groups),
                                          accepted_min_self_collision_distance_m=accepted_min_distance,
                                          collision_acceptance_numerical_slack_m=acceptance_slack),
                  solver_primal_tolerance=primal_tolerance,
                  solver_dual_tolerance=dual_tolerance,
                  wrist_position_cost=0.0, object_dofs_locked_during_ik=True,
                  object_dofs_actuated_in_physics=False, hand_order=list(SIDES),
                  fingertip_mean_error_m=tip_errors.mean(axis=(0, 2)).tolist(),
                  fingertip_max_error_m=tip_errors.max(axis=(0, 2)).tolist(),
                  wrist_mean_error_rad=wrist_errors.mean(axis=0).tolist(),
                  self_collision_min_distance_m=float(self_distances.min()),
                  self_penetrating_frames=int((self_distances < -5e-5).sum()),
                  self_collision_pair_count=len(pairs),
                  joint_limit_min_margin=float(joint_margin.min()),
                  joint_limit_violating_frames=int((joint_margin < -1e-6).sum()),
                  frame_velocity_max_ratio=float(velocity_ratio.max()),
                  frame_velocity_violating_intervals=int((velocity_ratio.max(axis=1) > 1 + 1e-6).sum()),
                  kinematic_model_feasible=bool((self_distances >= accepted_min_distance - acceptance_slack).all()
                                                and (joint_margin >= -1e-6).all()
                                                and (velocity_ratio <= 1 + 1e-6).all()),
                  collision_policy="explicit_runtime_pairs; source_intrahand_full_interhand_and_local_surface_guard",
                  self_collision_resolution="strict separating QP, fixed per-frame velocity envelope",
                  model_self_collision_passed=bool((self_distances >= -1e-6 - acceptance_slack).all()),
                  intrahand_collision_audit_path=str(intrahand_path.resolve()),
                  intrahand_collision_summary=dict(
                      pair_count=intrahand_audit["pair_count"],
                      counts=intrahand_audit["counts"],
                      by_classification=intrahand_audit["by_classification"],
                      min_distance_m=intrahand_audit["min_distance_m"],
                      penetrating_frames=intrahand_audit["penetrating_frames"],
                      tolerance_m=intrahand_audit["tolerance_m"]),
                  complete_intrahand_geometry_coverage=False,
                  worst_self_collision_pairs=worst_pairs,
                  strict_gate_passed=False, rl_validation_completed=False)
    (output / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
