"""Run the read-only object-local contact mapping candidate.

This experiment keeps the formal MINK references untouched.  It projects the
human fingertip observations onto each held object's physical collision mesh,
uses those patches and inward normals as bounded contact guidance, and writes
an independent comparison report for the original and corrected qpos traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]

from egoengine_repro.action.contact_nominal import (  # noqa: E402
    ContactNominalConfig,
    FINGER_DIRECTION_AXES,
    _finger_geometry_gaps,
    improve_nominal_with_gt_contact,
)
from egoengine_repro.action.contact_wrench import CollisionSurfaceProjector  # noqa: E402
from egoengine_repro.action.replay import MujocoReplayBackend  # noqa: E402
from egoengine_repro.retarget.collision_audit import collision_families, distances  # noqa: E402


TASKS = {
    "pour": {
        "scene": ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml",
        "run": ROOT / "runs/taco_pour_bimanual_gt_v1",
    },
    "brush": {
        "scene": ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_source_contacts_mass.xml",
        "run": ROOT / "runs/taco_brush_bimanual_gt_v4",
    },
}
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load(task: str) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    spec = TASKS[task]
    with np.load(spec["run"] / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    with np.load(spec["run"] / "robot_reference.npz", allow_pickle=False) as source:
        robot = dict(source)
    return spec["scene"], human, robot


def _project_contacts(
    scene: Path, human: dict[str, np.ndarray], robot: dict[str, np.ndarray],
    side: str, hand_index: int, threshold_m: float,
) -> dict[str, np.ndarray | float | int | str]:
    """Create object-local patch centres/normals from the human observations."""
    backend = MujocoReplayBackend(
        scene, object_joint_name=f"{side}_object_joint", hand_order=(side,),
    )
    projector = CollisionSurfaceProjector(backend)
    count = len(robot["qpos"])
    points = np.empty((count, len(FINGERS), 3), dtype=np.float64)
    normals = np.empty_like(points)
    projection_distance = np.empty((count, len(FINGERS)), dtype=np.float64)
    for row, qpos in enumerate(robot["qpos"]):
        backend.data.qpos[:] = qpos
        backend.data.qvel[:] = 0.0
        backend.mujoco.mj_forward(backend.model, backend.data)
        for finger in range(len(FINGERS)):
            sample = projector.project_world(
                human["T_sim_fingertip_target"][row, hand_index, finger, :3, 3],
            )
            points[row, finger] = sample.position_world_m
            # The mapper uses the inward approach direction, so the contact
            # solver receives the object's outward normal and negates it.
            normals[row, finger] = sample.outward_normal_world
            projection_distance[row, finger] = sample.distance_m
    contact = projection_distance <= float(threshold_m)
    return {
        "contact": contact,
        "points_world_m": points,
        "normals_world": normals,
        "projection_distance_m": projection_distance,
        "threshold_m": float(threshold_m),
        "active_count": int(contact.sum()),
        "contact_definition": "human MANO fingertip target within threshold of held-object physical collision surface",
    }


def _summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _measure(
    scene: Path, human: dict[str, np.ndarray], qpos: np.ndarray,
    contact_by_side: dict[str, dict[str, np.ndarray | float | int | str]],
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    count = len(qpos)
    tip_error = np.empty((count, 2, 5), dtype=np.float64)
    patch_error = np.full_like(tip_error, np.nan)
    normal_angle = np.full_like(tip_error, np.nan)
    wrist_position = np.empty((count, 2), dtype=np.float64)
    wrist_orientation = np.empty_like(wrist_position)
    gaps = np.empty((count, 2, 5), dtype=np.float64)
    actual_contact = np.zeros_like(gaps, dtype=bool)
    groups = collision_families(model)
    collision_min = {name: np.empty(count, dtype=np.float64) for name in groups}
    side_backends = {
        side: MujocoReplayBackend(
            scene, object_joint_name=f"{side}_object_joint", hand_order=(side,),
        ) for side in SIDES
    }
    actuated_addresses = []
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if name.startswith(("right_hand_", "left_hand_")):
            actuated_addresses.append(int(model.jnt_qposadr[joint]))
    actuated_addresses = np.asarray(sorted(set(actuated_addresses)), dtype=np.int64)
    for row, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for name, pairs in groups.items():
            values = distances(model, data, pairs)
            collision_min[name][row] = float(values.min()) if len(values) else np.inf
        for hi, side in enumerate(SIDES):
            object_body = model.body(f"{side}_object").id
            object_rotation = data.xmat[object_body].reshape(3, 3)
            object_position = data.xpos[object_body]
            evidence = contact_by_side[side]
            active = np.asarray(evidence["contact"], dtype=bool)[row]
            points = np.asarray(evidence["points_world_m"], dtype=np.float64)[row]
            normals = np.asarray(evidence["normals_world"], dtype=np.float64)[row]
            wrist_body = model.body(f"{side}_hand_link").id
            wrist_target = human["T_sim_wrist_target"][row, hi]
            wrist_position[row, hi] = np.linalg.norm(data.xpos[wrist_body] - wrist_target[:3, 3])
            wrist_orientation[row, hi] = Rotation.from_matrix(
                data.xmat[wrist_body].reshape(3, 3).T @ wrist_target[:3, :3],
            ).magnitude()
            for finger, finger_name in enumerate(FINGERS):
                site = model.site(f"{side}_{finger_name}_tip").id
                position = data.site_xpos[site]
                target = human["T_sim_fingertip_target"][row, hi, finger]
                tip_error[row, hi, finger] = np.linalg.norm(position - target[:3, 3])
                if active[finger]:
                    patch_error[row, hi, finger] = np.linalg.norm(position - points[finger])
                    column, sign = FINGER_DIRECTION_AXES[finger]
                    axis = data.site_xmat[site].reshape(3, 3)[:, column] * sign
                    axis /= np.linalg.norm(axis)
                    # The target approach axis points into the object, opposite
                    # to its outward surface normal.
                    normal_angle[row, hi, finger] = np.arccos(np.clip(
                        float(np.dot(axis, -normals[finger])), -1.0, 1.0,
                    ))
            side_backend = side_backends[side]
            side_backend.data.qpos[:] = state
            side_backend.data.qvel[:] = 0.0
            mujoco.mj_forward(side_backend.model, side_backend.data)
            gaps[row, hi] = _finger_geometry_gaps(side_backend)
            actual_contact[row, hi] = gaps[row, hi] <= 0.001
    label = np.stack([
        np.asarray(contact_by_side[side]["contact"], dtype=bool) for side in SIDES
    ], axis=1)
    tp = int(np.logical_and(actual_contact, label).sum())
    predicted = int(actual_contact.sum())
    observed = int(label.sum())
    dt = float(np.diff(human["timestamps_s"])[0]) if count > 1 else 1.0
    acceleration = np.diff(qpos[:, actuated_addresses], n=2, axis=0) / (dt * dt) if count > 2 else np.empty((0, len(actuated_addresses)))
    joint_margins = []
    for joint in range(model.njnt):
        if not bool(model.jnt_limited[joint]) or int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        address = int(model.jnt_qposadr[joint])
        low, high = model.jnt_range[joint]
        joint_margins.append(np.minimum(qpos[:, address] - low, high - qpos[:, address]))
    margins = np.concatenate(joint_margins) if joint_margins else np.asarray([np.inf])
    return {
        "fingertip_target_error_m": {
            side: _summary(tip_error[:, hi]) for hi, side in enumerate(SIDES)
        },
        "contact_patch_position_error_m": {
            side: _summary(patch_error[:, hi], np.asarray(contact_by_side[side]["contact"], dtype=bool))
            for hi, side in enumerate(SIDES)
        },
        "contact_approach_angle_rad": {
            side: _summary(normal_angle[:, hi], np.asarray(contact_by_side[side]["contact"], dtype=bool))
            for hi, side in enumerate(SIDES)
        },
        "wrist_position_error_m": {side: _summary(wrist_position[:, hi]) for hi, side in enumerate(SIDES)},
        "wrist_orientation_error_rad": {side: _summary(wrist_orientation[:, hi]) for hi, side in enumerate(SIDES)},
        "finger_object_gap_m": {side: _summary(gaps[:, hi]) for hi, side in enumerate(SIDES)},
        "contact_proxy": {
            "threshold_m": 0.001,
            "true_positive": tp,
            "predicted_count": predicted,
            "observed_count": observed,
            "precision": float(tp / predicted) if predicted else None,
            "recall": float(tp / observed) if observed else None,
            "label_definition": "human projected patch active within contact-evidence threshold; robot physical hand-object gap <= 1 mm",
        },
        "collision_min_distance_m": {
            name: _summary(values) for name, values in collision_min.items()
        },
        "joint_limit_margin_mixed_units": {
            "minimum": float(margins.min()),
            "violating_samples": int((margins < -1.0e-6).sum()),
        },
        "actuated_qpos_acceleration_norm": _summary(
            np.linalg.norm(acceleration, axis=1) if len(acceleration) else np.empty(0),
        ),
    }


def run(task: str, output: Path, threshold_m: float = 0.005) -> dict:
    if output.exists():
        raise FileExistsError(output)
    scene, human, robot = _load(task)
    base = np.asarray(robot["qpos"], dtype=np.float64)
    candidate = base.copy()
    contacts: dict[str, dict[str, np.ndarray | float | int | str]] = {}
    side_results = {}
    # The TACO references currently contain pre-existing hand/floor and
    # hand/object penetration.  Preserve those violations for this read-only
    # comparison, while rejecting any trial that makes a collision family or
    # non-contact finger gap worse than its own baseline frame.
    config = ContactNominalConfig(
        iterations_per_frame=4,
        lookahead_frames=6,
        preserve_baseline_violations=True,
    )
    for hi, side in enumerate(SIDES):
        print(f"[{task}] projecting {side} contact patches", flush=True)
        contacts[side] = _project_contacts(scene, human, robot, side, hi, threshold_m)
        backend = MujocoReplayBackend(
            scene, object_joint_name=f"{side}_object_joint", hand_order=(side,),
        )
        result = improve_nominal_with_gt_contact(
            backend, candidate,
            contact=np.asarray(contacts[side]["contact"], dtype=bool),
            contact_positions_world_m=np.asarray(contacts[side]["points_world_m"], dtype=np.float64),
            contact_normals_world_m=np.asarray(contacts[side]["normals_world"], dtype=np.float64),
            object_transforms_world=human["T_sim_object_reference"][:, hi],
            config=config,
        )
        candidate = result.qpos
        print(
            f"[{task}] corrected {side}: {contacts[side]['active_count']} active samples, "
            f"tip {np.nanmean(result.tip_error_before_m) * 1000:.3f} -> "
            f"{np.nanmean(result.tip_error_after_m) * 1000:.3f} mm",
            flush=True,
        )
        side_results[side] = {
            "active_contact_count": int(contacts[side]["active_count"]),
            "mean_tip_error_before_m": float(np.nanmean(result.tip_error_before_m)),
            "mean_tip_error_after_m": float(np.nanmean(result.tip_error_after_m)),
            "mean_gap_before_m": float(np.mean(result.geom_gap_before_m)),
            "mean_gap_after_m": float(np.mean(result.geom_gap_after_m)),
            "correction_max_abs": float(np.max(np.abs(result.correction_qpos))),
        }
    print(f"[{task}] measuring baseline", flush=True)
    before = _measure(scene, human, base, contacts)
    print(f"[{task}] measuring candidate", flush=True)
    after = _measure(scene, human, candidate, contacts)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "contact_evidence.npz",
        **{f"{side}_contact": np.asarray(contacts[side]["contact"], dtype=bool) for side in SIDES},
        **{f"{side}_points_world_m": np.asarray(contacts[side]["points_world_m"], dtype=np.float64) for side in SIDES},
        **{f"{side}_normals_world": np.asarray(contacts[side]["normals_world"], dtype=np.float64) for side in SIDES},
        **{f"{side}_projection_distance_m": np.asarray(contacts[side]["projection_distance_m"], dtype=np.float64) for side in SIDES},
    )
    np.savez_compressed(output / "candidate_qpos.npz", qpos=candidate, frame_indices=robot["frame_indices"], timestamps_s=robot["timestamps_s"])
    report = {
        "status": "read_only_contact_geometry_candidate",
        "task": task,
        "scene": str(scene.resolve()),
        "baseline_run": str(TASKS[task]["run"].resolve()),
        "formal_reference_modified": False,
        "rl_validation_completed": False,
        "candidate_contract": "object-local collision-surface patch centre plus inward approach-axis guidance; bounded robot-only correction",
        "contact_evidence": {
            side: {
                "threshold_m": contacts[side]["threshold_m"],
                "active_count": contacts[side]["active_count"],
                "frame_count": len(robot["frame_indices"]),
                "definition": contacts[side]["contact_definition"],
            } for side in SIDES
        },
        "fixed_candidate_config": {
            "iterations_per_frame": config.iterations_per_frame,
            "lookahead_frames": config.lookahead_frames,
            "tip_target_weight": config.tip_target_weight,
            "normal_contact_weight": config.normal_contact_weight,
            "geometry_contact_weight": config.geometry_contact_weight,
            "baseline_weight": config.baseline_weight,
            "temporal_weight": config.temporal_weight,
            "preserve_baseline_violations": config.preserve_baseline_violations,
            "baseline_preservation_tolerance_m": config.baseline_preservation_tolerance_m,
        },
        "side_results": side_results,
        "before": before,
        "after": after,
        "comparison_note": "contact precision/recall uses projected human proximity as an engineering label, not TACO tactile ground truth",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-threshold-mm", type=float, default=5.0)
    args = parser.parse_args()
    report = run(args.task, args.output, threshold_m=args.contact_threshold_mm / 1000.0)
    print(json.dumps({"task": args.task, "before": report["before"], "after": report["after"], "side_results": report["side_results"]}, indent=2))


if __name__ == "__main__":
    main()
