"""Trace reference errors without changing poses, geometry, or running physics."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

from audit_taco_initialization import visual_meshes
from egoengine_repro.evaluation.taco_surface import (
    _load_model_data, load_taco_mano_sequence, reconstruct_taco_mano,
)
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from egoengine_repro.retarget.schema import validate_human_reference

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
RUN = ROOT / "runs/taco_pour_bimanual_gt_v1"
DATA = ROOT / "data/taco_v1/pour_bowl_plate"
SEQUENCE = "(pour in some, bowl, plate)/20230927_017"
MANO = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIPS = np.array([4, 8, 12, 16, 20])


def rotation_steps_degrees(rotations):
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim < 3 or rotations.shape[-2:] != (3, 3) or len(rotations) < 2:
        raise ValueError("need at least two (...,3,3) rotations")
    if not np.isfinite(rotations).all():
        raise ValueError("nonfinite rotations")
    if (not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError("rotations must be in SO(3)")
    relative = rotations[:-1].swapaxes(-1, -2) @ rotations[1:]
    return np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(relative.shape[:-2])


def mano_global_rotations(poses, parents):
    poses = np.asarray(poses, dtype=float)
    parents = np.asarray(parents)
    if poses.ndim != 2 or poses.shape[1] != 48 or not np.isfinite(poses).all():
        raise ValueError("finite (T,48) MANO poses required")
    if parents.shape != (16,) or parents[0] != -1:
        raise ValueError("expected MANO wrist-rooted topology")
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 16, 3, 3)
    result = local.copy()
    for joint in range(1, 16):
        parent = int(parents[joint])
        if parent != parents[joint] or not 0 <= parent < joint:
            raise ValueError("parents must precede their children")
        result[:, joint] = result[:, parent] @ local[:, joint]
    return result


def palm_height_decomposition(vertices, target, actual, table_z):
    """A rigid-mesh arithmetic decomposition, not a proposed robot candidate."""
    target_vertices = vertices @ target[:3, :3].T + target[:3, 3]
    actual_vertices = vertices @ actual[:3, :3].T + actual[:3, 3]
    target_clearance = float(target_vertices[:, 2].min() - table_z)
    actual_clearance = float(actual_vertices[:, 2].min() - table_z)
    delta_z = float(actual[2, 3] - target[2, 3])
    return dict(
        palm_at_input_wrist_clearance_m=target_clearance,
        wrist_vertical_displacement_m=delta_z,
        rotation_contribution_m=actual_clearance - target_clearance - delta_z,
        saved_reference_palm_clearance_m=actual_clearance,
        interpretation="height decomposition only; input wrist placement is not a feasible initialization",
    )


def inspect():
    hands = DATA / "hand_poses/Hand_Poses" / SEQUENCE
    objects = DATA / "object_poses/Object_Poses" / SEQUENCE
    paths = [SCENE, RUN / "human_reference.npz", RUN / "robot_reference.npz",
             RUN / "retarget_report.json", hands / "hand_joints.npy"]
    paths.extend(hands / f"{side}_hand{suffix}.pkl" for side in SIDES for suffix in ("", "_shape"))
    paths.extend(MANO / f"MANO_{side.upper()}.pkl" for side in SIDES)
    paths.extend(objects.glob("*.npy"))
    paths.extend((DATA / "object_models/object_models_released").glob("*.obj"))
    paths.extend((DATA / "camera").rglob("*.npy"))
    paths.extend((DATA / "camera").rglob("*.txt"))
    inputs = [artifact(p) for p in paths] + scene_mesh_artifacts(SCENE)
    with np.load(RUN / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as source:
        robot = dict(source)
    validate_human_reference(human)
    raw_joints = np.load(hands / "hand_joints.npy", allow_pickle=False).astype(float)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    qpos = validate_qpos(model, robot["qpos"], trajectory=True)
    if not np.array_equal(human["frame_indices"], robot["frame_indices"]):
        raise ValueError("human/robot timeline mismatch")
    if not np.array_equal(human["hand_order"], SIDES):
        raise ValueError("unexpected hand order")
    transform = human["T_sim_world"]
    if not np.array_equal(transform[:3, :3], np.eye(3)):
        raise ValueError("this audit targets the preserved translation-only alignment")
    np.testing.assert_allclose(raw_joints[:, [1, 0]] + transform[:3, 3],
                               human["joint_positions_sim"], atol=1e-12, rtol=0)
    meshes, _ = visual_meshes(SCENE, model)
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    floor = model.geom("floor").id
    np.testing.assert_allclose(zero.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-12)
    table_z = float(zero.geom_xpos[floor, 2])
    data = mujoco.MjData(model)
    trace, results, orientation = [], {}, {}

    # Independent PKL -> MANO FK -> 21-joint reconstruction checks topology,
    # metric scale, and the wrist-origin convention against the released array.
    for hi, side in enumerate(SIDES):
        model_path = MANO / f"MANO_{side.upper()}.pkl"
        vertices, joints, _, regions, keys = reconstruct_taco_mano(
            hands / f"{side}_hand.pkl", hands / f"{side}_hand_shape.pkl", model_path, side=side)
        if [int(key) for key in keys] != list(range(1, len(qpos) + 1)):
            raise ValueError("PKL frame IDs mismatch")
        joint_error = np.linalg.norm(joints - raw_joints[:, 1 - hi], axis=-1)
        if joint_error.max() > 1e-6:
            raise ValueError("MANO rotational convention is not corroborated by released joints")
        vertices_sim = vertices.astype(float) + transform[:3, 3]
        results[side] = dict(
            mano_joint_reconstruction_max_error_m=float(joint_error.max()),
            initial_human_mesh_clearance_m=float(vertices_sim[0, :, 2].min() - table_z),
            initial_human_palm_region_clearance_m=float(vertices_sim[0, regions == 0, 2].min() - table_z),
            initial_human_vertex_penetration_count_50um=int((vertices_sim[0, :, 2] < table_z - 5e-5).sum()),
            initial_human_joint_min_clearance_m=float(human["joint_positions_sim"][0, hi, :, 2].min() - table_z),
            human_mesh_min_clearance_all_frames_m=float(vertices_sim[:, :, 2].min() - table_z),
        )
        if side == "left":
            plate = trimesh.load_mesh(DATA / "object_models/object_models_released/135_cm.obj", process=True)
            plate.apply_scale(.01)
            if not plate.is_watertight or plate.volume <= 0:
                raise ValueError("signed containment requires the closed, oriented native plate")
            plate_pose = np.load(objects / "target_135.npy", allow_pickle=False)[0].astype(float)
            local = (vertices[0].astype(float) - plate_pose[:3, 3]) @ plate_pose[:3, :3]
            signed = closed_mesh_signed_distance(plate, local)
            if not np.isfinite(signed).all():
                raise ValueError("nonfinite MANO/plate vertex distances")
            results[side]["initial_human_plate_vertex_check"] = dict(
                vertices_tested=len(signed), inside_vertices_50um=int((signed > 5e-5).sum()),
                maximum_signed_distance_m=float(signed.max()),
                minimum_vertex_surface_distance_m=float(np.abs(signed).min()),
                whole_surface_nonintersection_certified=False,
                coordinate_source="native centimeter mesh scaled once; original world object pose and reconstructed MANO",
            )

        poses, _, _, _ = load_taco_mano_sequence(hands / f"{side}_hand.pkl", hands / f"{side}_hand_shape.pkl")
        topology = np.asarray(_load_model_data(model_path)["kintree_table"])
        np.testing.assert_array_equal(topology[1], np.arange(16))
        parents = topology[0].astype(np.int64)
        parents[0] = -1
        global_rotations = mano_global_rotations(poses, parents)
        distal_step = rotation_steps_degrees(global_rotations[:, [15, 3, 6, 12, 9]])
        stored_step = rotation_steps_degrees(human["T_sim_fingertip_target"][:, hi, :, :3, :3])
        points = human["joint_positions_sim"][:, hi]
        bone = points[:, TIPS] - points[:, TIPS - 1]
        bone /= np.linalg.norm(bone, axis=-1, keepdims=True)
        bone_step = np.rad2deg(np.arccos(np.clip((bone[:-1] * bone[1:]).sum(-1), -1, 1)))
        normal = np.cross(points[:, 5] - points[:, 0], points[:, 17] - points[:, 0])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        condition = np.sqrt(np.maximum(0, 1 - (bone * normal[:, None]).sum(-1) ** 2))
        events = []
        for frame, finger in np.argwhere(stored_step > 90):
            events.append(dict(source_rows_zero_based=[int(frame), int(frame + 1)],
                source_keys_one_based=[keys[frame], keys[frame + 1]], finger=FINGERS[finger],
                constructed_orientation_step_deg=float(stored_step[frame, finger]),
                mano_distal_rotation_step_deg=float(distal_step[frame, finger]),
                observed_bone_direction_step_deg=float(bone_step[frame, finger]),
                projected_normal_relative_norm=condition[frame:frame + 2, finger].tolist()))

        # A fixed axis calibration cancels out of angular step magnitudes.
        # This comparison does not prescribe unpublished robot/MANO offsets.
        root_step = rotation_steps_degrees(global_rotations[:, 0])
        wrist_step = rotation_steps_degrees(human["T_sim_wrist_target"][:, hi, :3, :3])
        orientation[side] = dict(events_over_90deg=events,
            stored_orientation_max_step_deg=float(stored_step.max()),
            mano_distal_max_step_deg=float(distal_step.max()),
            wrist_step_max_difference_deg=float(np.abs(root_step - wrist_step).max()),
            minimum_projected_normal_relative_norm=float(condition.min()),
            fixed_frame_calibration_independent_step_comparison=True,
            stored_orientation_valid_flags_all_true=bool(human["valid_fingertip_orientation"][:, hi].all()))

    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        row = dict(source_row_zero_based=frame, hands={})
        for hi, side in enumerate(SIDES):
            body = model.body(f"{side}_hand_link").id
            geom = model.geom(f"{side}_hand_link_visual").id
            target = human["T_sim_wrist_target"][frame, hi]
            actual = np.eye(4)
            actual[:3, :3] = data.xmat[body].reshape(3, 3)
            actual[:3, 3] = data.xpos[body]
            record = palm_height_decomposition(meshes[geom].vertices, target, actual, table_z)
            errors = np.stack([data.site_xpos[model.site(f"{side}_{finger}_tip").id]
                - human["T_sim_fingertip_target"][frame, hi, k, :3, 3] for k, finger in enumerate(FINGERS)])
            record.update(wrist_translation_difference_m=(actual[:3, 3] - target[:3, 3]).tolist(),
                          fingertip_centroid_residual_m=errors.mean(0).tolist())
            seed = mujoco.MjData(model)
            seed.qpos[:] = q
            seed.qpos[hi * 18:hi * 18 + 3] = target[:3, 3]
            seed.qpos[hi * 18 + 3:hi * 18 + 6] = Rotation.from_matrix(target[:3, :3]).as_euler("ZXY") * [1, 1, -1]
            mujoco.mj_forward(model, seed)
            record["wrist_euler_seed_fk_error_rad"] = float(Rotation.from_matrix(
                target[:3, :3].T @ seed.xmat[body].reshape(3, 3)).magnitude())
            row["hands"][side] = record
        trace.append(row)
    for side in SIDES:
        results[side]["initial_robot_palm"] = trace[0]["hands"][side]
        results[side]["wrist_euler_seed_fk_max_error_rad"] = max(row["hands"][side]["wrist_euler_seed_fk_error_rad"] for row in trace)

    verify_artifacts(inputs)
    return dict(status="input_logic_diagnostic_not_retarget_fix_or_reset", frames=len(qpos),
        inputs=inputs, table_z_m=table_z, source_table_z_assumed_m=table_z - float(transform[2, 3]),
        table_alignment_independently_calibrated=False,
        alignment_method="source target first mesh minimum -> horizontal infinite table",
        physics_steps=0, pose_optimization_performed=False, original_inputs_modified=False,
        formal_initialization_proposed=False, full_so3_gt_used_by_current_retarget=False,
        results=results, orientation=orientation, trace=trace,
        limitations=[
            "MANO is reconstructed annotation geometry, not measured human skin or a measured tabletop.",
            "Initial palm decomposition isolates translation/rotation, not why the optimizer selected its finger configuration.",
            "Temporal orientation defects do not prove they caused the first-frame table penetration.",
            "Independent table normal, height and finite footprint are not supplied by the current alignment.",
            "No new collision geometry or collision-pair classification is established here.",
        ],
        code=[artifact(Path(__file__)), artifact(ROOT / "src/egoengine_repro/evaluation/taco_surface.py"),
              artifact(ROOT / "src/egoengine_repro/retarget/taco_bimanual.py")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "frames", "results", "orientation")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
