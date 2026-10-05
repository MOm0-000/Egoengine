#!/usr/bin/env python3
"""Run the single authorized Brush native-support MINK candidate and audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
import qpsolvers
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    audit_intrahand_trajectory,
    distances,
    explicit_hand_pairs,
)
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    NativeSupportPlaneLimit,
    build_native_support_geoms,
    geom_world_vertices,
    independent_native_floor_rows,
    native_hand_visual_geom_ids,
)
from egoengine_repro.retarget.taco_bimanual import retarget  # noqa: E402
from egoengine_repro.retarget.mink import _joint_velocity_limits  # noqa: E402
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


SCHEMA = "taco_brush_environment_aware_mink_v1"
CLASSES = {
    "ENVIRONMENT_AWARE_MINK_FRAME0_INFEASIBLE",
    "ENVIRONMENT_AWARE_MINK_TRAJECTORY_INFEASIBLE",
    "ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED",
    "ENVIRONMENT_AWARE_MINK_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS",
    "ENVIRONMENT_AWARE_MINK_STATIC_GATE_PASS",
    "FUNCTIONAL_ERROR",
}
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=json_default,
    ) + "\n")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
    ).strip()


def load_contract(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected candidate schema")
    if git_head(ROOT.parent) != cfg["baseline_commit"]:
        raise ValueError("candidate must start from its pinned baseline")
    if cfg["authorization"]["new_retarget_candidates"] != 1:
        raise ValueError("exactly one retarget candidate must be authorized")
    forbidden = ("physics", "replay", "mpc", "rl", "promotion", "chunk_commit")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("runtime work is forbidden by this candidate contract")
    if not all(cfg["frozen_from_baseline"].values()):
        raise ValueError("all baseline settings must remain frozen")
    if not all(cfg["hard_rules"].values()):
        raise ValueError("all candidate hard rules must remain active")
    return cfg


def support_plane(infra_run: Path) -> Plane:
    contract = yaml.safe_load((infra_run / "support_surface_contract.yaml").read_text())
    simulator = contract["simulator"]
    return Plane(
        normal=simulator["normal"],
        offset=simulator["offset_m"],
        frame=simulator["frame"],
    )


def floor_code_audit(output: Path) -> None:
    query = subprocess.run(
        ["rg", "-n", "floor_collision_avoidance", "rl/src", "rl/scripts", "rl/configs"],
        cwd=ROOT.parent, text=True, capture_output=True, check=False,
    )
    all_references = [line for line in query.stdout.splitlines() if line]
    callsites = [
        line for line in all_references
        if not line.startswith("rl/scripts/run_taco_brush_environment_aware_mink_v1.py:")
    ]
    write_json(output / "floor_constraint_code_audit.json", {
        "schema": "taco_brush_floor_constraint_code_audit_v1",
        "legacy_proxy_path": "floor_collision_avoidance",
        "active_callsites": callsites,
        "audit_self_references_excluded": len(all_references) - len(callsites),
        "active_use_count": len(callsites),
        "decision": "RETAINED_ACTIVE_GENERIC_PROXY_ONLY",
        "brush_candidate_pass_gate": "NATIVE_VISUAL_SUPPORT_PLANE_ONLY",
        "proxy_used_as_pass_gate": False,
        "reason": "generic runtime configurations still reference the proxy path",
    })


def validate_limit(
    model: Any, plane: Plane, ids: list[int], baseline_qpos: np.ndarray,
    output: Path,
) -> NativeSupportPlaneLimit:
    limit = NativeSupportPlaneLimit(
        model, mujoco, plane, ids,
        minimum_clearance_m=0.0,
        activation_distance_m=None,
        gain=0.85,
        depenetration_step_m=0.002,
    )
    rng = np.random.default_rng(20230927)
    hull_errors: list[float] = []
    for geom in limit.geoms:
        for _ in range(5):
            rotation = Rotation.random(random_state=rng).as_matrix()
            translation = rng.normal(size=3)
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            full = (geom.full_vertices @ rotation.T + translation) @ normal
            hull = (geom.support_vertices @ rotation.T + translation) @ normal
            hull_errors.append(float(abs(full.min() - hull.min())))

    jacobian_errors: list[float] = []
    for frame in rng.choice(len(baseline_qpos), size=6, replace=False):
        import mink
        configuration = mink.Configuration(model, q=baseline_qpos[frame])
        for row in limit.rows(configuration)[::5]:
            velocity = rng.normal(size=model.nv)
            velocity[36:] = 0.0
            velocity /= np.linalg.norm(velocity)
            analytic = float(row["normal_jacobian"] @ velocity)
            epsilon = 1e-7
            finite_values = []
            for sign in (-1.0, 1.0):
                q = baseline_qpos[frame].copy()
                mujoco.mj_integratePos(model, q, velocity, sign * epsilon)
                probe = mink.Configuration(model, q=q)
                match = next(
                    candidate for candidate in limit.rows(probe)
                    if candidate["geom"].geom_id == row["geom"].geom_id
                )
                finite_values.append(match["distance_m"])
            finite = (finite_values[1] - finite_values[0]) / (2.0 * epsilon)
            jacobian_errors.append(float(abs(analytic - finite)))

    import mink
    depenetration = mink.Configuration(model, q=baseline_qpos[0])
    initial = limit.minimum_distance(depenetration)
    history = [initial]
    for _ in range(16):
        constraint = limit.compute_qp_inequalities(depenetration, 1.0 / 240.0)
        delta = qpsolvers.solve_qp(
            np.eye(model.nv), np.zeros(model.nv), constraint.G, constraint.h,
            solver="daqp", primal_tol=1e-9, dual_tol=1e-9,
        )
        if delta is None:
            raise RuntimeError("depenetration validation QP is infeasible")
        depenetration.integrate_inplace(delta, 1.0)
        history.append(limit.minimum_distance(depenetration))

    validation = {
        "schema": "native_support_limit_validation_v1",
        "status": "PASS",
        "always_active": True,
        "activation_distance_m": None,
        "tunneling_argument": "all native visual links constrained on every QP",
        "hull_support_random_pose_trials": len(hull_errors),
        "hull_support_max_abs_error_m": max(hull_errors),
        "jacobian_finite_difference_trials": len(jacobian_errors),
        "jacobian_max_abs_error": max(jacobian_errors),
        "depenetration_initial_distance_m": initial,
        "depenetration_final_distance_m": history[-1],
        "depenetration_history_m": history,
        "full_mesh_hull_equivalence_passed": max(hull_errors) <= 1e-12,
        "jacobian_finite_difference_passed": max(jacobian_errors) <= 2e-6,
        "depenetration_passed": history[-1] >= -1e-8 and history[-1] > initial,
    }
    if not all(validation[key] for key in (
        "full_mesh_hull_equivalence_passed",
        "jacobian_finite_difference_passed",
        "depenetration_passed",
    )):
        validation["status"] = "FAIL"
        write_json(output / "native_support_limit_validation.json", validation)
        raise RuntimeError("native support limit validation failed")
    write_json(output / "native_support_limit_validation.json", validation)
    write_json(output / "native_support_geom_manifest.json", limit.manifest())
    return limit


def native_floor_audit(
    model: Any, plane: Plane, ids: list[int], qpos: np.ndarray, tolerance: float,
    output: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = independent_native_floor_rows(model, mujoco, plane, qpos, ids)
    frame_minimum = np.full(len(qpos), np.inf)
    for row in rows:
        frame = int(row["frame"])
        frame_minimum[frame] = min(frame_minimum[frame], row["minimum_signed_distance_m"])
    by_side = {
        side: min(row["minimum_signed_distance_m"] for row in rows if row["side"] == side)
        for side in SIDES
    }
    by_link = {
        name: min(row["minimum_signed_distance_m"] for row in rows if row["geom"] == name)
        for name in sorted({row["geom"] for row in rows})
    }
    penetrating_frames = sorted({
        int(row["frame"]) for row in rows
        if row["minimum_signed_distance_m"] < -tolerance
    })
    report = {
        "schema": "taco_brush_native_floor_audit_v1",
        "audit_geometry": "full native material mesh vertices",
        "support_plane": plane.to_dict(),
        "tolerance_m": tolerance,
        "unknown_count": 0,
        "global_minimum_signed_distance_m": min(row["minimum_signed_distance_m"] for row in rows),
        "per_side_minimum_signed_distance_m": by_side,
        "per_link_minimum_signed_distance_m": by_link,
        "penetrating_frame_count": len(penetrating_frames),
        "penetrating_frames": penetrating_frames,
        "worst_20_frame_link_pairs": sorted(
            rows, key=lambda row: row["minimum_signed_distance_m"]
        )[:20],
        "status": "PASS" if not penetrating_frames else "FAIL",
    }
    write_json(output / "native_floor_audit.json", report)
    return report, frame_minimum


def self_collision_audit(
    model: Any, qpos: np.ndarray, baseline_run: Path,
    minimum: float, slack: float, output: Path,
) -> dict[str, Any]:
    pairs = explicit_hand_pairs(model)
    data = mujoco.MjData(model)
    values = np.empty((len(qpos), len(pairs)), dtype=np.float64)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = distances(model, data, pairs)
    full = audit_intrahand_trajectory(model, qpos)
    baseline_full = json.loads(
        (baseline_run / "mink_baseline/intrahand_collision_audit.json").read_text()
    )
    explicit_min = float(values.min())
    report = {
        "schema": "taco_brush_environment_aware_self_collision_audit_v1",
        "explicit_runtime_pair_count": len(pairs),
        "explicit_minimum_distance_m": explicit_min,
        "accepted_minimum_distance_m": minimum,
        "acceptance_numerical_slack_m": slack,
        "explicit_violating_frame_count": int(
            np.any(values < minimum - slack, axis=1).sum()
        ),
        "certified_intrahand_audit": full,
        "baseline_intrahand_summary": {
            "min_distance_m": baseline_full["min_distance_m"],
            "penetrating_frames": baseline_full["penetrating_frames"],
            "by_classification": baseline_full["by_classification"],
        },
        "status": "PASS" if explicit_min >= minimum - slack else "FAIL",
    }
    write_json(output / "self_collision_audit.json", report)
    return report


def kinematic_errors(model: Any, qpos: np.ndarray, human: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(qpos)
    position = np.empty((count, 2, 5), dtype=np.float64)
    orientation = np.empty_like(position)
    wrist = np.empty((count, 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for hand, side in enumerate(SIDES):
            body = model.body(f"{side}_hand_link").id
            actual_wrist = data.xmat[body].reshape(3, 3)
            target_wrist = human["T_sim_wrist_target"][frame, hand, :3, :3]
            wrist[frame, hand] = Rotation.from_matrix(
                actual_wrist.T @ target_wrist
            ).magnitude()
            for finger_index, finger in enumerate(FINGERS):
                site = model.site(f"{side}_{finger}_tip").id
                target = human["T_sim_fingertip_target"][frame, hand, finger_index]
                position[frame, hand, finger_index] = np.linalg.norm(
                    data.site_xpos[site] - target[:3, 3]
                )
                actual = data.site_xmat[site].reshape(3, 3)
                orientation[frame, hand, finger_index] = Rotation.from_matrix(
                    actual.T @ target[:3, :3]
                ).magnitude()
    return {"position": position, "orientation": orientation, "wrist": wrist}


def fidelity_audit(
    model: Any, baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
    human: dict[str, np.ndarray], baseline_floor: np.ndarray,
    candidate_floor: np.ndarray, output: Path,
) -> dict[str, Any]:
    base_errors = kinematic_errors(model, baseline["qpos"], human)
    cand_errors = kinematic_errors(model, candidate["qpos"], human)
    fields = [
        "frame", "side", "finger",
        "baseline_fingertip_position_error_m", "candidate_fingertip_position_error_m",
        "delta_fingertip_position_error_m",
        "baseline_fingertip_orientation_error_rad", "candidate_fingertip_orientation_error_rad",
        "delta_fingertip_orientation_error_rad",
        "baseline_wrist_orientation_error_rad", "candidate_wrist_orientation_error_rad",
        "delta_wrist_orientation_error_rad",
        "baseline_frame_velocity_ratio", "candidate_frame_velocity_ratio",
        "baseline_joint_limit_margin", "candidate_joint_limit_margin",
        "baseline_native_floor_minimum_m", "candidate_native_floor_minimum_m",
    ]
    path = output / "candidate_minus_baseline_fidelity.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for frame in range(len(candidate["qpos"])):
            for hand, side in enumerate(SIDES):
                for finger_index, finger in enumerate(FINGERS):
                    bp = base_errors["position"][frame, hand, finger_index]
                    cp = cand_errors["position"][frame, hand, finger_index]
                    bo = base_errors["orientation"][frame, hand, finger_index]
                    co = cand_errors["orientation"][frame, hand, finger_index]
                    bw = base_errors["wrist"][frame, hand]
                    cw = cand_errors["wrist"][frame, hand]
                    writer.writerow({
                        "frame": frame, "side": side, "finger": finger,
                        "baseline_fingertip_position_error_m": bp,
                        "candidate_fingertip_position_error_m": cp,
                        "delta_fingertip_position_error_m": cp - bp,
                        "baseline_fingertip_orientation_error_rad": bo,
                        "candidate_fingertip_orientation_error_rad": co,
                        "delta_fingertip_orientation_error_rad": co - bo,
                        "baseline_wrist_orientation_error_rad": bw,
                        "candidate_wrist_orientation_error_rad": cw,
                        "delta_wrist_orientation_error_rad": cw - bw,
                        "baseline_frame_velocity_ratio": "" if frame == 0 else baseline["frame_velocity_max_ratio"][frame - 1],
                        "candidate_frame_velocity_ratio": "" if frame == 0 else candidate["frame_velocity_max_ratio"][frame - 1],
                        "baseline_joint_limit_margin": baseline["joint_limit_min_margin"][frame],
                        "candidate_joint_limit_margin": candidate["joint_limit_min_margin"][frame],
                        "baseline_native_floor_minimum_m": baseline_floor[frame],
                        "candidate_native_floor_minimum_m": candidate_floor[frame],
                    })
    summaries = {}
    for hand, side in enumerate(SIDES):
        for finger_index, finger in enumerate(FINGERS):
            key = f"{side}_{finger}"
            summaries[key] = {}
            for metric in ("position", "orientation"):
                base = base_errors[metric][:, hand, finger_index]
                cand = cand_errors[metric][:, hand, finger_index]
                summaries[key][metric] = {
                    "baseline_mean": float(base.mean()),
                    "candidate_mean": float(cand.mean()),
                    "baseline_p95": float(np.quantile(base, 0.95)),
                    "candidate_p95": float(np.quantile(cand, 0.95)),
                    "baseline_max": float(base.max()),
                    "candidate_max": float(cand.max()),
                }
    report = {
        "schema": "taco_brush_candidate_fidelity_summary_v1",
        "paired_frames": len(candidate["qpos"]),
        "per_hand_finger": summaries,
        "wrist_orientation": {
            side: {
                "baseline_mean_rad": float(base_errors["wrist"][:, hand].mean()),
                "candidate_mean_rad": float(cand_errors["wrist"][:, hand].mean()),
                "baseline_p95_rad": float(np.quantile(base_errors["wrist"][:, hand], 0.95)),
                "candidate_p95_rad": float(np.quantile(cand_errors["wrist"][:, hand], 0.95)),
                "baseline_max_rad": float(base_errors["wrist"][:, hand].max()),
                "candidate_max_rad": float(cand_errors["wrist"][:, hand].max()),
            }
            for hand, side in enumerate(SIDES)
        },
    }
    write_json(output / "fidelity_summary.json", report)
    return report


def external_static_blockers(model: Any, plane: Plane, qpos: np.ndarray, output: Path) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    values = {}
    for role, name in (("brush", "right_object_visual"), ("bowl", "left_object_visual")):
        geom_id = model.geom(name).id
        geom = build_native_support_geoms(model, mujoco, [geom_id])[0]
        minimum = float(plane.signed_distance(
            geom_world_vertices(data, geom, full=True)
        ).min())
        values[role] = {
            "geom": name,
            "frame0_minimum_signed_distance_m": minimum,
            "controlled_by_this_candidate": False,
        }
    report = {
        "schema": "taco_brush_remaining_external_static_blockers_v1",
        "support_plane": plane.to_dict(),
        "objects": values,
        "brush_penetration_retained_without_posthoc_lift": values["brush"]["frame0_minimum_signed_distance_m"] < 0.0,
        "status": "EXTERNAL_STATIC_BLOCKER_REMAINS" if values["brush"]["frame0_minimum_signed_distance_m"] < -5e-5 else "PASS",
    }
    write_json(output / "remaining_external_static_blockers.json", report)
    return report


def render_visuals(
    model: Any, baseline_qpos: np.ndarray, candidate_qpos: np.ndarray,
    frames: list[int], output: Path,
) -> None:
    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    views = {
        "side": dict(azimuth=90.0, elevation=-12.0, distance=0.52, lookat=[0.6, 0.0, 0.79]),
        "front": dict(azimuth=180.0, elevation=-12.0, distance=0.52, lookat=[0.6, 0.0, 0.79]),
        "thumb_table_closeup": dict(azimuth=115.0, elevation=-5.0, distance=0.25, lookat=[0.6, 0.0, 0.74]),
    }
    records = []
    for frame in frames:
        for view, spec in views.items():
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = spec["azimuth"]
            camera.elevation = spec["elevation"]
            camera.distance = spec["distance"]
            camera.lookat[:] = spec["lookat"]
            images = []
            for label, qpos in (("BASELINE", baseline_qpos), ("CANDIDATE", candidate_qpos)):
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                image = renderer.render().copy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"{label} frame {frame}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                images.append(image)
            panel = np.concatenate(images, axis=1)
            filename = f"frame_{frame:03d}_{view}_baseline_vs_candidate.jpg"
            if not cv2.imwrite(str(visual_dir / filename), panel):
                raise RuntimeError(f"failed to write {filename}")
            records.append((frame, view, filename))
    renderer.close()
    lines = ["# Baseline vs environment-aware MINK", "",
             "All panels render the real MuJoCo floor and native simulator geometry; no 2-D table grid is drawn.", ""]
    for frame, view, filename in records:
        lines.append(f"- frame {frame}, {view}: [image](visuals/{filename})")
    (output / "VISUAL_INDEX.md").write_text("\n".join(lines) + "\n")


def deleted_manifest(cfg: dict[str, Any], output: Path) -> None:
    entries = []
    for source in cfg["cleanup"]["trash_if_unreferenced"]:
        name = Path(source).name
        destination = Path(
            f"rl/TRASH/taco_brush_environment_aware_mink_v1/retired_runs/{name}"
        )
        entries.append({
            "source": source,
            "destination": str(destination),
            "active_source_absent": not (ROOT.parent / source).exists(),
            "trash_destination_present": (ROOT.parent / destination).exists(),
            "active_callsite_count_before_move": 0,
        })
    if not all(row["active_source_absent"] and row["trash_destination_present"] for row in entries):
        raise RuntimeError("retired Brush runs were not moved to the declared TRASH location")
    write_json(output / "deleted_code_manifest.json", {
        "schema": "taco_brush_environment_aware_deleted_code_manifest_v1",
        "moved_to_trash": entries,
        "legacy_floor_proxy": "retained because active generic call sites exist",
        "yellow_grid_chain": "already deleted at baseline b1ae493",
    })


def hash_tree(output: Path) -> None:
    lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()
                       and p.name != "server_artifacts.sha256"):
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(lines) + "\n")


def finish_failure(output: Path, failed_frame: int, reason: str) -> int:
    if failed_frame == 0:
        classification = "ENVIRONMENT_AWARE_MINK_FRAME0_INFEASIBLE"
    elif "native support-plane feasibility failed" in reason:
        classification = "ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED"
    else:
        classification = "ENVIRONMENT_AWARE_MINK_TRAJECTORY_INFEASIBLE"
    write_json(output / "decision.json", {
        "schema": "taco_brush_environment_aware_mink_decision_v1",
        "classification": classification,
        "allowed_classes": sorted(CLASSES),
        "failed_frame": failed_frame,
        "reason": reason,
        "runtime_counts": {"physics": 0, "replay": 0, "mpc": 0, "rl": 0,
                           "new_retarget_candidates": 1, "promotion": 0,
                           "chunk_commit": 0},
    })
    (output / "summary.md").write_text(
        "# TACO Brush environment-aware MINK v1\n\n"
        f"- Classification: `{classification}`\n"
        f"- Failed frame: `{failed_frame}`\n"
        f"- Reason: `{reason}`\n"
        "- No physics, Replay, MPC, RL, promotion, or chunk commit was run.\n"
    )
    hash_tree(output)
    return 0


def prefix_reference_metrics(
    model: Any, qpos: np.ndarray, dt: float, settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    velocity_map = _joint_velocity_limits(
        model, mujoco, settings["velocity_limits"]
    )
    addresses = np.array([
        int(model.joint(name).qposadr[0]) for name in velocity_map
    ])
    speeds = np.array(list(velocity_map.values()))
    ranges = np.array([model.joint(name).range for name in velocity_map])
    margin = np.minimum(
        qpos[:, addresses] - ranges[:, 0],
        ranges[:, 1] - qpos[:, addresses],
    ).min(axis=1)
    velocity = np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt * speeds)
    return {
        "qpos": qpos,
        "joint_limit_min_margin": margin,
        "frame_velocity_max_ratio": velocity.max(axis=1),
    }


def postprocess_existing_failure(
    cfg: dict[str, Any], config_path: Path, output: Path,
) -> int:
    baseline_run = Path(cfg["paths"]["baseline_run"]).resolve(strict=True)
    infra_run = Path(cfg["paths"]["infra_run"]).resolve(strict=True)
    scene = Path(cfg["paths"]["scene"]).resolve(strict=True)
    failure = json.loads((output / "retarget_failure.json").read_text())
    if failure.get("status") != "failed_kinematic_candidate":
        raise ValueError("existing output is not a frozen failed candidate")
    floor_code_audit(output)
    failed_frame = int(failure["failed_frame"])
    with np.load(output / "failed_kinematic_prefix.npz", allow_pickle=False) as archive:
        prefix_qpos = archive["qpos"].copy()
    if len(prefix_qpos) != failed_frame + 1:
        raise ValueError("failed prefix length does not match failed frame")
    with np.load(baseline_run / "robot_reference.npz", allow_pickle=False) as archive:
        baseline = dict(archive)
    with np.load(baseline_run / "human_reference.npz", allow_pickle=False) as archive:
        human = dict(archive)
    baseline_report = json.loads((baseline_run / "retarget_report.json").read_text())
    settings = baseline_report["inherited_settings"]
    dt = float(np.diff(human["timestamps_s"])[0])
    model = mujoco.MjModel.from_xml_path(str(scene))
    candidate = prefix_reference_metrics(
        model=model, qpos=prefix_qpos, dt=dt, settings=settings,
    )
    plane = support_plane(infra_run)
    ids = native_hand_visual_geom_ids(model, mujoco)
    tolerance = float(cfg["environment_constraint"]["validation_tolerance_m"])
    floor_report, candidate_floor = native_floor_audit(
        model, plane, ids, prefix_qpos, tolerance, output,
    )
    _, baseline_floor = native_floor_audit(
        model, plane, ids, baseline["qpos"][:len(prefix_qpos)], tolerance,
        output / "_baseline_floor_tmp",
    )
    temporary = output / "_baseline_floor_tmp/native_floor_audit.json"
    temporary.unlink()
    temporary.parent.rmdir()
    self_report = self_collision_audit(
        model, prefix_qpos, baseline_run,
        float(baseline_report["effective_settings"]["accepted_min_self_collision_distance_m"]),
        float(baseline_report["effective_settings"]["collision_acceptance_numerical_slack_m"]),
        output,
    )
    fidelity = fidelity_audit(
        model,
        {key: value[:len(prefix_qpos)] if getattr(value, "ndim", 0) else value
         for key, value in baseline.items()},
        candidate,
        human,
        baseline_floor,
        candidate_floor,
        output,
    )
    external = external_static_blockers(model, plane, prefix_qpos, output)
    worst_frame = int(np.argmin(candidate_floor))
    visual_frames = sorted({
        frame for frame in (0, 1, 5, 10, worst_frame, failed_frame)
        if frame < len(prefix_qpos)
    })
    render_visuals(
        model, baseline["qpos"][:len(prefix_qpos)], prefix_qpos,
        visual_frames, output,
    )
    report = {
        "schema": "taco_brush_environment_aware_failed_retarget_report_v1",
        "status": "failed_kinematic_candidate",
        "classification": "ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED",
        "failed_frame": failed_frame,
        "accepted_complete_frames_before_failure": failed_frame,
        "audited_prefix_includes_failed_state": True,
        "robot_reference_npz_written": False,
        "robot_reference_absence_reason": "no complete 209-frame feasible candidate exists",
        "reason": failure["reason"],
        "native_floor_prefix_audit": floor_report,
        "self_collision_prefix_status": self_report["status"],
        "joint_limit_min_margin": float(candidate["joint_limit_min_margin"].min()),
        "joint_limit_violating_frames": int(
            (candidate["joint_limit_min_margin"] < -1e-6).sum()
        ),
        "frame_velocity_max_ratio": float(
            candidate["frame_velocity_max_ratio"].max()
        ),
        "frame_velocity_violating_intervals": int(
            (candidate["frame_velocity_max_ratio"] > 1.0 + 1e-6).sum()
        ),
        "fidelity_summary": fidelity,
        "external_static_blockers": external,
        "strict_gate_passed": False,
    }
    write_json(output / "retarget_report.json", report)
    classification = "ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED"
    decision = {
        "schema": "taco_brush_environment_aware_mink_decision_v1",
        "classification": classification,
        "allowed_classes": sorted(CLASSES),
        "failed_frame": failed_frame,
        "accepted_complete_frames_before_failure": failed_frame,
        "failed_state_native_floor_minimum_m": floor_report["global_minimum_signed_distance_m"],
        "native_floor_tolerance_m": tolerance,
        "second_candidate_authorized": False,
        "robot_reference_written": False,
        "required_success_only_artifacts_absent": ["robot_reference.npz"],
        "promotion_authorized": False,
        "runtime_counts": {"physics": 0, "replay": 0, "mpc": 0, "rl": 0,
                           "new_retarget_candidates": 1, "promotion": 0,
                           "chunk_commit": 0},
    }
    write_json(output / "decision.json", decision)
    (output / "summary.md").write_text("\n".join([
        "# TACO Brush environment-aware MINK v1", "",
        f"- Classification: `{classification}`",
        "- Extension: `LOCAL_ENVIRONMENT_NONPENETRATION_EXTENSION`",
        f"- Accepted complete frames before failure: `{failed_frame}/209`",
        f"- Failed frame: `{failed_frame}`",
        f"- Failed-state native floor minimum: `{floor_report['global_minimum_signed_distance_m']:.9f} m`",
        f"- Frozen material tolerance: `{tolerance:.9f} m`",
        f"- Explicit self-collision prefix status: `{self_report['status']}`",
        f"- Remaining brush-floor clearance: `{external['objects']['brush']['frame0_minimum_signed_distance_m']:.9f} m`", "",
        "The one authorized candidate passed frame 0, but its linearized QP constraint missed the independently audited native material support-plane gate at frame 13. The QP itself did not report infeasibility. No tolerance, target, weight, table, or object pose was changed, and no second candidate was run.",
        "", "No complete `robot_reference.npz` is emitted because a 209-frame feasible candidate does not exist. The frozen failed state is retained in `failed_kinematic_prefix.npz`.",
    ]) + "\n")
    manifest_inputs = [
        config_path, Path(__file__),
        ROOT / "src/egoengine_repro/retarget/support_plane_limit.py",
        ROOT / "src/egoengine_repro/retarget/taco_bimanual.py",
        ROOT / "src/egoengine_repro/scene/support_surface.py",
        scene, baseline_run / "human_reference.npz",
        baseline_run / "robot_reference.npz",
        baseline_run / "retarget_report.json",
        infra_run / "support_surface_contract.yaml",
    ]
    write_json(output / "input_manifest.json", {
        "schema": "taco_brush_environment_aware_input_manifest_v1",
        "artifacts": [artifact(path) for path in manifest_inputs],
    })
    hash_tree(output)
    print(json.dumps(decision, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--postprocess-existing-failure", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = load_contract(config_path)
    baseline_run = Path(cfg["paths"]["baseline_run"]).resolve(strict=True)
    infra_run = Path(cfg["paths"]["infra_run"]).resolve(strict=True)
    scene = Path(cfg["paths"]["scene"]).resolve(strict=True)
    output = Path(cfg["paths"]["output"])
    if args.postprocess_existing_failure:
        if not output.is_dir():
            raise FileNotFoundError(output)
        return postprocess_existing_failure(cfg, config_path, output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    deleted_manifest(cfg, output)
    write_json(output / "source_pins.json", {
        "schema": "taco_brush_environment_aware_source_pins_v1",
        "active_repository": {"path": str(ROOT.parent), "commit": git_head(ROOT.parent)},
        "baseline_commit": cfg["baseline_commit"],
        "scene": artifact(scene),
        "baseline_robot_reference": artifact(baseline_run / "robot_reference.npz"),
        "baseline_human_reference": artifact(baseline_run / "human_reference.npz"),
        "support_surface_contract": artifact(infra_run / "support_surface_contract.yaml"),
    })
    floor_code_audit(output)
    plane = support_plane(infra_run)
    model = mujoco.MjModel.from_xml_path(str(scene))
    ids = native_hand_visual_geom_ids(model, mujoco)
    with np.load(baseline_run / "robot_reference.npz", allow_pickle=False) as archive:
        baseline = dict(archive)
    with np.load(baseline_run / "human_reference.npz", allow_pickle=False) as archive:
        human = dict(archive)
    validate_limit(model, plane, ids, baseline["qpos"], output)

    baseline_report = json.loads((baseline_run / "retarget_report.json").read_text())
    settings = dict(baseline_report["inherited_settings"])
    environment = cfg["environment_constraint"]
    settings["native_support_plane_limit"] = {
        "normal": plane.normal.tolist(), "offset_m": plane.offset,
        "frame": plane.frame,
        "minimum_clearance_m": environment["minimum_clearance_m"],
        "activation_distance_m": environment["activation_distance_m"],
        "gain": environment["gain"],
        "depenetration_step_m": environment["depenetration_step_m"],
        "validation_tolerance_m": environment["validation_tolerance_m"],
    }
    write_json(output / "candidate_delta_contract.json", {
        "schema": "taco_brush_environment_aware_candidate_delta_v1",
        "extension": "LOCAL_ENVIRONMENT_NONPENETRATION_EXTENSION",
        "baseline_settings": baseline_report["inherited_settings"],
        "candidate_settings_without_extension": {
            key: value for key, value in settings.items()
            if key != "native_support_plane_limit"
        },
        "unchanged_settings_exact": (
            baseline_report["inherited_settings"]
            == {key: value for key, value in settings.items()
                if key != "native_support_plane_limit"}
        ),
        "only_added_setting": {"native_support_plane_limit": settings["native_support_plane_limit"]},
        "forbidden_changes": {
            "human_targets": False, "z_offset": False, "table": False,
            "objects": False, "post_correction": False,
        },
    })

    try:
        retarget(scene, baseline_run / "human_reference.npz", settings, output)
    except RuntimeError as error:
        failure = json.loads((output / "retarget_failure.json").read_text())
        return finish_failure(output, int(failure["failed_frame"]), str(error))

    with np.load(output / "robot_reference.npz", allow_pickle=False) as archive:
        candidate = dict(archive)
    tolerance = float(environment["validation_tolerance_m"])
    floor_report, candidate_floor = native_floor_audit(
        model, plane, ids, candidate["qpos"], tolerance, output,
    )
    _, baseline_floor = native_floor_audit(
        model, plane, ids, baseline["qpos"], tolerance,
        output / "_baseline_floor_tmp",
    )
    temporary = output / "_baseline_floor_tmp/native_floor_audit.json"
    temporary.unlink()
    temporary.parent.rmdir()
    self_report = self_collision_audit(
        model, candidate["qpos"], baseline_run,
        float(baseline_report["effective_settings"]["accepted_min_self_collision_distance_m"]),
        float(baseline_report["effective_settings"]["collision_acceptance_numerical_slack_m"]),
        output,
    )
    fidelity = fidelity_audit(
        model, baseline, candidate, human, baseline_floor, candidate_floor, output,
    )
    external = external_static_blockers(model, plane, candidate["qpos"], output)

    report_path = output / "retarget_report.json"
    report = json.loads(report_path.read_text())
    report["candidate_delta_contract"] = "candidate_delta_contract.json"
    report["native_floor_audit"] = "native_floor_audit.json"
    report["fidelity_summary"] = fidelity
    report["strict_gate_passed"] = bool(
        floor_report["status"] == "PASS"
        and self_report["status"] == "PASS"
        and report["joint_limit_violating_frames"] == 0
        and report["frame_velocity_violating_intervals"] == 0
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if floor_report["status"] != "PASS":
        classification = "ENVIRONMENT_AWARE_MINK_FLOOR_CONSTRAINT_MISSED"
    elif not report["strict_gate_passed"]:
        classification = "ENVIRONMENT_AWARE_MINK_TRAJECTORY_INFEASIBLE"
    elif external["status"] != "PASS":
        classification = "ENVIRONMENT_AWARE_MINK_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS"
    else:
        classification = "ENVIRONMENT_AWARE_MINK_STATIC_GATE_PASS"

    worst_frame = int(np.argmin(candidate_floor))
    frames = sorted({0, 1, 5, 10, worst_frame, len(candidate["qpos"]) - 1})
    render_visuals(model, baseline["qpos"], candidate["qpos"], frames, output)
    decision = {
        "schema": "taco_brush_environment_aware_mink_decision_v1",
        "classification": classification,
        "allowed_classes": sorted(CLASSES),
        "candidate_frames": len(candidate["qpos"]),
        "native_hand_floor_closed": floor_report["status"] == "PASS",
        "self_collision_gate_passed": self_report["status"] == "PASS",
        "joint_gate_passed": report["joint_limit_violating_frames"] == 0,
        "velocity_gate_passed": report["frame_velocity_violating_intervals"] == 0,
        "external_static_blocker_remains": external["status"] != "PASS",
        "promotion_authorized": False,
        "runtime_counts": {"physics": 0, "replay": 0, "mpc": 0, "rl": 0,
                           "new_retarget_candidates": 1, "promotion": 0,
                           "chunk_commit": 0},
    }
    write_json(output / "decision.json", decision)
    summary = [
        "# TACO Brush environment-aware MINK v1", "",
        f"- Classification: `{classification}`",
        "- Extension: `LOCAL_ENVIRONMENT_NONPENETRATION_EXTENSION`",
        f"- Candidate frames: `{len(candidate['qpos'])}/209`",
        f"- Native hand-floor minimum: `{floor_report['global_minimum_signed_distance_m']:.9f} m`",
        f"- Hand-floor penetrating frames beyond tolerance: `{floor_report['penetrating_frame_count']}`",
        f"- Explicit self-collision minimum: `{self_report['explicit_minimum_distance_m']:.9f} m`",
        f"- Frame velocity max ratio: `{report['frame_velocity_max_ratio']:.9f}`",
        f"- Joint-limit minimum margin: `{report['joint_limit_min_margin']:.9f}`",
        f"- Remaining brush-floor clearance: `{external['objects']['brush']['frame0_minimum_signed_distance_m']:.9f} m`", "",
        "The candidate changes only the native hand-material support-plane constraint. The brush is not controlled or lifted. No physics, Replay, MPC, RL, promotion, or chunk commit was run.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    manifest_inputs = [
        config_path, Path(__file__),
        ROOT / "src/egoengine_repro/retarget/support_plane_limit.py",
        ROOT / "src/egoengine_repro/retarget/taco_bimanual.py",
        ROOT / "src/egoengine_repro/scene/support_surface.py",
        scene, baseline_run / "human_reference.npz",
        baseline_run / "robot_reference.npz",
        baseline_run / "retarget_report.json",
        infra_run / "support_surface_contract.yaml",
    ]
    write_json(output / "input_manifest.json", {
        "schema": "taco_brush_environment_aware_input_manifest_v1",
        "artifacts": [artifact(path) for path in manifest_inputs],
    })
    hash_tree(output)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
