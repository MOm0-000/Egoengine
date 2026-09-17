"""Inspect remaining position errors and query the existing QP without stepping."""

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
import mink
from audit_taco_initialization import visual_meshes
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.mink import (
    _FrameDisplacementLimit, _StrictCollisionLimit, _enable_planning_collision_masks,
    _explicit_collision_groups, _joint_velocity_limits,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_v1"
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def inspect():
    metadata = json.loads((RUN / "retarget_report.json").read_text())
    scene = Path(metadata["scene"])
    if artifact(scene)["sha256"] != metadata["scene_sha256"]:
        raise ValueError("scene changed since reference generation")
    inputs = [artifact(RUN / name) for name in ("human_reference.npz", "robot_reference.npz", "retarget_report.json")]
    inputs.extend([artifact(scene), *scene_mesh_artifacts(scene)])
    with np.load(RUN / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as source:
        robot = dict(source)
    model = mujoco.MjModel.from_xml_path(str(scene))
    qpos = validate_qpos(model, robot["qpos"], trajectory=True)
    meshes, _ = visual_meshes(scene, model)
    settings = metadata["inherited_settings"]
    config = mink.Configuration(model)
    tasks, site_geometry = [], {}
    for side in SIDES:
        tasks.append(mink.FrameTask(f"{side}_hand_link", "body", position_cost=0,
            orientation_cost=settings["wrist_orientation_cost"], lm_damping=1e-3))
        site_geometry[side] = {}
        for finger in FINGERS:
            tasks.append(mink.FrameTask(f"{side}_{finger}_tip", "site",
                position_cost=settings["fingertip_position_cost"],
                orientation_cost=settings["fingertip_orientation_cost"], lm_damping=1e-3))
            site = model.site(f"{side}_{finger}_tip").id
            body = int(model.site_bodyid[site])
            geom = next(g for g in meshes if model.geom_bodyid[g] == body)
            point = model.site_pos[site]
            closest, distances, _ = trimesh.proximity.closest_point(meshes[geom], point[None])
            site_geometry[side][finger] = dict(local_site_m=point.tolist(),
                nearest_native_surface_distance_m=float(distances[0]), nearest_surface_point_m=closest[0].tolist(),
                native_mesh_watertight=bool(meshes[geom].is_watertight),
                interpretation="unsigned native-mesh offset; not evidence of inside/outside or manufacturer pad intent")
    position = np.empty((len(qpos), 2, 5))
    orientation = np.empty_like(position)
    loss_rows = []
    for row, q in enumerate(qpos):
        config.update(q)
        position_loss, orientation_loss = 0., 0.
        for hi, side in enumerate(SIDES):
            targets = [human["T_sim_wrist_target"][row, hi], *human["T_sim_fingertip_target"][row, hi]]
            for k, target in enumerate(targets):
                task = tasks[hi * 6 + k]
                task.set_target(mink.SE3.from_matrix(target))
                weighted = task.compute_error(config) * task.cost
                position_loss += float(weighted[:3] @ weighted[:3])
                orientation_loss += float(weighted[3:] @ weighted[3:])
                if k:
                    site = model.site(f"{side}_{FINGERS[k - 1]}_tip").id
                    position[row, hi, k - 1] = np.linalg.norm(config.data.site_xpos[site] - target[:3, 3])
                    orientation[row, hi, k - 1] = Rotation.from_matrix(config.data.site_xmat[site].reshape(3, 3).T @ target[:3, :3]).magnitude()
        loss_rows.append(dict(source_row_zero_based=row, weighted_se3_translation_squared=position_loss,
                              weighted_rotation_squared=orientation_loss))
    np.testing.assert_allclose(position, robot["fingertip_position_error_m"], atol=1e-12, rtol=0)

    geoms = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("collision_hand_")]
    _enable_planning_collision_masks(model, geoms, [])
    groups = _explicit_collision_groups(model, mujoco, hand_geom_ids=set(geoms))
    inner = mink.CollisionAvoidanceLimit(model, groups, minimum_distance_from_collisions=0., collision_detection_distance=.02)
    collision = _StrictCollisionLimit(inner, mujoco, minimum_distance=0., depenetration_step=.002)
    collision.enabled = True
    velocity = _joint_velocity_limits(model, mujoco, settings["velocity_limits"])
    displacement = _FrameDisplacementLimit(model, mujoco, velocity, mink.Constraint)
    limits = [mink.ConfigurationLimit(model), collision, displacement]
    locks = [mink.DofFreezingTask(model, list(range(36, 48)))]
    dt = float(np.diff(human["timestamps_s"])[0])
    qp_dt = dt / settings["max_iterations_per_frame"]
    probes = []
    for row in (0, 20, 50, 100, 150, 197):
        config.update(qpos[row])
        displacement.set_previous(qpos[row - 1] if row else None, dt if row else None)
        for hi in range(2):
            for k, target in enumerate([human["T_sim_wrist_target"][row, hi], *human["T_sim_fingertip_target"][row, hi]]):
                tasks[hi * 6 + k].set_target(mink.SE3.from_matrix(target))
        before = config.q.copy()
        delta = qp_dt * mink.solve_ik(config, tasks, qp_dt, solver="daqp", damping=1e-5,
            limits=limits, constraints=locks, primal_tol=1e-9, dual_tol=1e-9)
        np.testing.assert_array_equal(config.q, before)
        probes.append(dict(source_row_zero_based=row, unapplied_wrist_translation_increment_m=delta[[0,1,2,18,19,20]].tolist(),
            maximum_unapplied_finger_increment_rad=float(np.abs(delta[np.r_[6:18,24:36]]).max()),
            state_integrated=False))
    summary = {}
    for hi, side in enumerate(SIDES):
        fingers = {}
        for k, finger in enumerate(FINGERS):
            values = position[:, hi, k]
            fingers[finger] = dict(mean_m=float(values.mean()), median_m=float(np.median(values)),
                p95_m=float(np.percentile(values, 95)), maximum_m=float(values.max()),
                orientation_mean_deg=float(np.rad2deg(orientation[:, hi, k]).mean()))
        bounds = []
        for joint in range(hi * 18 + 6, hi * 18 + 18):
            values = qpos[:, model.jnt_qposadr[joint]]
            bounds.append(dict(joint=model.joint(joint).name,
                lower_bound_rows=int((np.abs(values - model.jnt_range[joint, 0]) < 1e-5).sum()),
                upper_bound_rows=int((np.abs(values - model.jnt_range[joint, 1]) < 1e-5).sum()),
                bound_activity_tolerance_rad=1e-5))
        summary[side] = dict(fingers=fingers, joint_bound_activity=bounds)
    verify_artifacts(inputs)
    return dict(status="remaining_centimeter_error_unresolved", frames=len(qpos), inputs=inputs,
        per_hand=summary, native_tracking_sites=site_geometry, weighted_loss_rows=loss_rows,
        same_qp_unapplied_probes=probes,
        local_cost_convention=dict(position_cost=settings["fingertip_position_cost"],
            fingertip_rotation_cost=settings["fingertip_orientation_cost"], wrist_rotation_cost=settings["wrist_orientation_cost"],
            costs_multiply_residual_before_squaring=True,
            rotation_deg_equal_cost_to_pure_1cm_translation=float(np.rad2deg(.01 * settings["fingertip_position_cost"] / settings["fingertip_orientation_cost"])),
            author_coefficients_recovered=False),
        limitations=["A stationary first-frame QP is not a global or position-only optimum.",
                     "Different point semantics and active joint bounds do not quantify the irreducible morphology error.",
                     "No weights, tracking sites, reference states or collision geometry were modified by this audit."],
        code=[artifact(Path(__file__)), artifact(ROOT / "external/mink/src/mink/tasks/task.py"),
              artifact(ROOT / "external/mink/src/mink/tasks/frame_task.py")])


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
    print(json.dumps(dict(per_hand=report["per_hand"], probes=report["same_qp_unapplied_probes"],
                         cost=report["local_cost_convention"]), indent=2))


if __name__ == "__main__":
    main()
