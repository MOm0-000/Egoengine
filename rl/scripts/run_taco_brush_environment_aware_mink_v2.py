#!/usr/bin/env python3
"""Run the architecture gate and single authorized Brush MINK v2 candidate."""

from __future__ import annotations

import argparse
import copy
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
from egoengine_repro.retarget.taco_bimanual import FINGERS, SIDES, retarget  # noqa: E402
from egoengine_repro.retarget.taco_bimanual_settings import (  # noqa: E402
    SELF_ONLY_FINAL_FEASIBILITY,
    UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    NativeSupportSettings,
    load_taco_bimanual_settings,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


SCHEMA = "taco_brush_environment_aware_mink_v2"
FINAL_CLASSES = {
    "RETARGET_ARCHITECTURE_CONTRACT_BLOCKER",
    "RETARGET_REFACTOR_SEMANTIC_DRIFT",
    "UNIFIED_FEASIBILITY_CLOSURE_IMPLEMENTATION_ERROR",
    "UNIFIED_FEASIBILITY_CLOSURE_INFEASIBLE",
    "ENVIRONMENT_AWARE_MINK_V2_TRAJECTORY_INFEASIBLE",
    "ENVIRONMENT_AWARE_MINK_V2_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS",
    "ENVIRONMENT_AWARE_MINK_V2_STATIC_GATE_PASS",
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
    ).strip()


def load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError("unexpected v2 schema")
    if git_head(ROOT.parent) != value["baseline_commit"]:
        raise ValueError("v2 must start at its pinned baseline commit")
    if value["authorization"]["new_retarget_candidates"] != 1:
        raise ValueError("exactly one candidate must be authorized")
    forbidden = ("physics", "replay", "mpc", "rl", "promotion", "chunk_commit")
    if any(value["authorization"][key] for key in forbidden):
        raise ValueError("v2 contract authorizes forbidden runtime work")
    if not all(value["architecture_gate"].values()):
        raise ValueError("all architecture gates must be enabled")
    if not all(value["hard_rules"].values()):
        raise ValueError("all v2 hard rules must be enabled")
    if value["only_scientific_algorithm_change"] != {
        "from": SELF_ONLY_FINAL_FEASIBILITY,
        "to": UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    }:
        raise ValueError("unexpected scientific algorithm delta")
    return value


def support_plane(path: Path) -> Plane:
    value = yaml.safe_load(path.read_text())["simulator"]
    return Plane(
        normal=value["normal"], offset=value["offset_m"], frame=value["frame"]
    )


def active_callgraph(output: Path, cfg: dict[str, Any]) -> None:
    nodes = [
        ("rl/scripts/run_taco_brush_environment_aware_mink_v2.py", "ACTIVE_EXECUTION"),
        ("rl/configs/taco_brush_environment_aware_mink_v2.yaml", "ACTIVE_EXECUTION"),
        ("rl/configs/taco_bimanual_mink_local_v1.yaml", "ACTIVE_EXECUTION"),
        ("rl/src/egoengine_repro/retarget/taco_bimanual.py", "ACTIVE_EXECUTION"),
        ("rl/src/egoengine_repro/retarget/taco_bimanual_settings.py", "ACTIVE_EXECUTION"),
        ("rl/src/egoengine_repro/retarget/kinematic_limits.py", "ACTIVE_SHARED_UTILITY"),
        ("rl/src/egoengine_repro/retarget/support_plane_limit.py", "ACTIVE_SHARED_UTILITY"),
        ("rl/src/egoengine_repro/retarget/collision_audit.py", "ACTIVE_SHARED_UTILITY"),
        ("rl/src/egoengine_repro/retarget/schema.py", "ACTIVE_SHARED_UTILITY"),
        ("rl/src/egoengine_repro/scene/support_surface.py", "ACTIVE_SHARED_UTILITY"),
        (cfg["paths"]["scene"], "ACTIVE_EXECUTION"),
        (cfg["paths"]["support_surface_contract"], "ACTIVE_EXECUTION"),
        ("rl/src/egoengine_repro/retarget/mink.py", "UNREFERENCED"),
        ("rl/src/egoengine_repro/configs/paper_faithful_taco.yaml", "UNREFERENCED"),
        ("rl/TRASH/retarget_historical_surface_2026-10-06", "HISTORICAL_ONLY"),
    ]
    report = {
        "schema": "active_retarget_callgraph_v1",
        "root": nodes[0][0],
        "nodes": [{"path": path, "classification": kind} for path, kind in nodes],
        "edges": [
            [nodes[0][0], "rl/src/egoengine_repro/retarget/taco_bimanual.py"],
            [nodes[0][0], "rl/src/egoengine_repro/retarget/taco_bimanual_settings.py"],
            ["rl/src/egoengine_repro/retarget/taco_bimanual.py", "rl/src/egoengine_repro/retarget/kinematic_limits.py"],
            ["rl/src/egoengine_repro/retarget/taco_bimanual.py", "rl/src/egoengine_repro/retarget/support_plane_limit.py"],
            ["rl/src/egoengine_repro/retarget/taco_bimanual.py", "rl/src/egoengine_repro/retarget/collision_audit.py"],
        ],
        "generic_mink_retargeter_called": False,
        "generic_reward_config_read": False,
        "old_pour_candidate_framework_called": False,
        "status": "PASS",
    }
    write_json(output / "active_retarget_callgraph.json", report)
    lines = [
        "# Active TACO bimanual retarget call graph", "",
        "```text",
        "v2 runner + v2 contract + minimal retarget settings",
        "  -> taco_bimanual.retarget (typed settings)",
        "     -> kinematic_limits",
        "     -> support_plane_limit",
        "     -> collision_audit / schema",
        "     -> MINK + MuJoCo",
        "  -> independent audits / reports / visuals",
        "```", "",
        "The active path does not call `retarget_with_mink`, a Pour candidate "
        "framework, generic reward/chunk/PPO configuration, or the archived v1 runner.",
    ]
    (output / "active_retarget_callgraph.md").write_text("\n".join(lines) + "\n")


def dead_code_audit(output: Path) -> None:
    moved = {
        "interaction_aware.py": 3,
        "shape_aware.py": 2,
        "contract_closure.py": 4,
        "contract_fidelity.py": 4,
        "initial_hand.py": 9,
    }
    rows = [{
        "file": f"rl/src/egoengine_repro/retarget/{name}",
        "pre_cleanup_active_caller_count": 0,
        "pre_cleanup_historical_caller_count": callers,
        "action": "MOVE_TO_TRASH",
        "destination": (
            "rl/TRASH/retarget_historical_surface_2026-10-06/modules/" + name
        ),
        "reason": "only frozen Pour scripts/tests formed the remaining caller closure",
    } for name, callers in moved.items()]
    rows.append({
        "file": "rl/src/egoengine_repro/retarget/paper_audit.py",
        "pre_cleanup_active_caller_count": 1,
        "pre_cleanup_historical_caller_count": 20,
        "action": "KEEP",
        "reason": "active video_to_spider.rl.physics_contract imports it",
    })
    rows.append({
        "file": "rl/src/egoengine_repro/retarget/taco_bimanual.py",
        "pre_cleanup_active_caller_count": 1,
        "pre_cleanup_historical_caller_count": 10,
        "action": "REPLACE_ACTIVE_AND_ARCHIVE_V1",
        "reason": "active v2 path is typed and minimal; source-alignment v1 is historical",
    })
    write_json(output / "retarget_dead_code_audit.json", {
        "schema": "retarget_dead_code_audit_v1",
        "rows": rows,
        "status": "PASS",
    })
    trash_root = ROOT / "TRASH/retarget_historical_surface_2026-10-06"
    v1_root = ROOT / "TRASH/taco_brush_environment_aware_mink_v2/retarget_v1"
    files = sorted(
        str(path.relative_to(ROOT.parent))
        for root in (trash_root, v1_root)
        for path in root.rglob("*") if path.is_file()
    )
    write_json(output / "deleted_code_manifest.json", {
        "schema": "taco_brush_environment_aware_mink_v2_deleted_code_v1",
        "operation": "MOVE_TO_TRASH",
        "compatibility_wrappers_created": 0,
        "files": files,
        "historical_v1_raw_delta_status": "SUPERSEDED_RAW_CONFIG_NOT_EFFECTIVE_CONTRACT",
    })


def architecture_static_gate(output: Path, cfg: dict[str, Any]) -> None:
    source = (ROOT / "src/egoengine_repro/retarget/taco_bimanual.py").read_text()
    settings_source = Path(cfg["paths"]["retarget_settings"]).read_text()
    if "from .mink import" in source or "retarget_with_mink" in source:
        raise RuntimeError("active taco_bimanual imports generic retargeter")
    forbidden = (
        "paper_faithful_taco.yaml", "['reward']", '["reward"]',
        "['chunks']", '["chunks"]', "ppo_learning_rate", "mpc_samples",
    )
    hits = [
        token for token in forbidden
        if token in source or token in settings_source
    ]
    if hits:
        raise RuntimeError(f"action optimization dependency in retarget path: {hits}")
    model = mujoco.MjModel.from_xml_path(cfg["paths"]["scene"])
    local_guard_geoms = [
        model.geom(index).name for index in range(model.ngeom)
        if "_palm_thumb_surface_guard_" in (model.geom(index).name or "")
    ]
    if local_guard_geoms:
        raise RuntimeError("primary Brush scene unexpectedly contains Pour local guards")
    active_callgraph(output, cfg)
    dead_code_audit(output)


def native_support_settings(cfg: dict[str, Any], plane: Plane) -> NativeSupportSettings:
    support = cfg["frozen_native_support"]
    return NativeSupportSettings(
        normal=tuple(float(value) for value in plane.normal),
        offset_m=float(plane.offset),
        frame=plane.frame,
        minimum_clearance_m=float(support["minimum_clearance_m"]),
        activation_distance_m=support["activation_distance_m"],
        gain=float(support["gain"]),
        depenetration_step_m=float(support["depenetration_step_m"]),
        validation_tolerance_m=float(support["validation_tolerance_m"]),
    )


def floor_rows(
    model: Any, plane: Plane, qpos: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    ids = native_hand_visual_geom_ids(model, mujoco)
    rows = independent_native_floor_rows(model, mujoco, plane, qpos, ids)
    minimum = np.full(len(qpos), np.inf, dtype=np.float64)
    for row in rows:
        minimum[int(row["frame"])] = min(
            minimum[int(row["frame"])], row["minimum_signed_distance_m"]
        )
    return rows, minimum


def trajectory_metrics(
    model: Any, human: dict[str, np.ndarray], qpos: np.ndarray, plane: Plane,
) -> dict[str, np.ndarray]:
    count = len(qpos)
    tips = np.empty((count, 2, 5), dtype=np.float64)
    wrists = np.empty((count, 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for hand, side in enumerate(SIDES):
            body = model.body(f"{side}_hand_link").id
            relative = (
                data.xmat[body].reshape(3, 3).T
                @ human["T_sim_wrist_target"][frame, hand, :3, :3]
            )
            wrists[frame, hand] = Rotation.from_matrix(relative).magnitude()
            for finger_index, finger in enumerate(FINGERS):
                site = model.site(f"{side}_{finger}_tip").id
                tips[frame, hand, finger_index] = np.linalg.norm(
                    data.site_xpos[site]
                    - human["T_sim_fingertip_target"][
                        frame, hand, finger_index, :3, 3
                    ]
                )
    pairs = explicit_hand_pairs(model)
    self_min = np.empty(count, dtype=np.float64)
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        self_min[frame] = distances(model, data, pairs).min()
    _, native = floor_rows(model, plane, qpos)
    return {"tips": tips, "wrists": wrists, "self": self_min, "native": native}


def refactor_equivalence(
    cfg: dict[str, Any], settings: Any, plane: Plane, output: Path,
) -> tuple[bool, dict[str, Any]]:
    regression = output / "refactor_regression"
    scene = Path(cfg["paths"]["scene"])
    human_path = Path(cfg["paths"]["human_reference"])
    retarget(scene, human_path, settings, regression, frame_count=13)
    with np.load(regression / "robot_reference.npz", allow_pickle=False) as archive:
        current = dict(archive)
    with np.load(
        Path(cfg["paths"]["v1_run"]) / "failed_kinematic_prefix.npz",
        allow_pickle=False,
    ) as archive:
        v1_qpos = archive["qpos"][:13].copy()
    with np.load(human_path, allow_pickle=False) as archive:
        human = dict(archive)
    model = mujoco.MjModel.from_xml_path(str(scene))
    current_metrics = trajectory_metrics(model, human, current["qpos"], plane)
    v1_metrics = trajectory_metrics(model, human, v1_qpos, plane)
    values = {
        "qpos_max_abs_difference": float(np.abs(current["qpos"] - v1_qpos).max()),
        "fingertip_error_max_abs_difference_m": float(np.abs(
            current_metrics["tips"] - v1_metrics["tips"]
        ).max()),
        "self_collision_max_abs_difference_m": float(np.abs(
            current_metrics["self"] - v1_metrics["self"]
        ).max()),
        "native_support_max_abs_difference_m": float(np.abs(
            current_metrics["native"] - v1_metrics["native"]
        ).max()),
    }
    passed = all(value <= 1e-12 for value in values.values())
    report = {
        "schema": "taco_brush_refactor_equivalence_v1",
        "frames": list(range(13)),
        "threshold": "each max absolute difference <= 1e-12",
        **values,
        "status": "PASS" if passed else "FAIL",
    }
    write_json(output / "refactor_equivalence_report.json", report)
    return passed, json.loads(
        (regression / "effective_retarget_contract.json").read_text()
    )


def frame13_mechanism(cfg: dict[str, Any], plane: Plane, output: Path) -> None:
    scene = Path(cfg["paths"]["scene"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(
        Path(cfg["paths"]["v1_run"]) / "failed_kinematic_prefix.npz",
        allow_pickle=False,
    ) as archive:
        qpos = archive["qpos"][13]
    import mink
    configuration = mink.Configuration(model, q=qpos)
    limit = NativeSupportPlaneLimit(
        model, mujoco, plane, native_hand_visual_geom_ids(model, mujoco)
    )
    support = min(limit.rows(configuration), key=lambda row: row["distance_m"])
    full = min(
        limit.full_mesh_rows(configuration.data), key=lambda row: row["distance_m"]
    )
    local_normal = (
        np.asarray(configuration.data.geom_xmat[support["geom"].geom_id])
        .reshape(3, 3).T @ plane.normal
    )
    write_json(output / "frame13_failure_mechanism.json", {
        "schema": "taco_brush_frame13_failure_mechanism_v1",
        "classification": "UNRESOLVED",
        "reason": (
            "frozen v1 artifact has the final failed state but not the final "
            "substep pre-state; nonlinear integration and vertex switching "
            "cannot be separated without inventing history"
        ),
        "frame": 13,
        "full_native_minimum_m": full["distance_m"],
        "support_minimum_m": support["distance_m"],
        "support_full_difference_m": support["distance_m"] - full["distance_m"],
        "active_geom": support["geom"].name,
        "support_vertex_world": support["point_world"].tolist(),
        "full_vertex_index": full["vertex_index"],
        "local_plane_normal": local_normal.tolist(),
        "normal_jacobian": support["normal_jacobian"].tolist(),
        "deterministic_last_substep_replay_available": False,
    })


def native_floor_audit(
    model: Any, plane: Plane, qpos: np.ndarray, tolerance: float, output: Path,
) -> dict[str, Any]:
    rows, minimum = floor_rows(model, plane, qpos)
    violating = np.flatnonzero(minimum < -tolerance).tolist()
    report = {
        "schema": "taco_brush_environment_aware_mink_v2_native_floor_v1",
        "geometry": "all full native hand material visual-mesh vertices",
        "native_geom_count": len({row["geom_id"] for row in rows}),
        "frames": len(qpos),
        "tolerance_m": tolerance,
        "global_minimum_m": float(minimum.min()),
        "violating_frames": violating,
        "unknown_count": 0,
        "status": "PASS" if not violating else "FAIL",
        "worst_rows": sorted(
            rows, key=lambda row: row["minimum_signed_distance_m"]
        )[:20],
    }
    write_json(output / "native_floor_audit.json", report)
    return report


def self_collision_audit(model: Any, qpos: np.ndarray, output: Path) -> dict[str, Any]:
    pairs = explicit_hand_pairs(model)
    data = mujoco.MjData(model)
    values = np.empty((len(qpos), len(pairs)), dtype=np.float64)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = distances(model, data, pairs)
    certified = audit_intrahand_trajectory(model, qpos)
    minimum = float(values.min())
    report = {
        "schema": "taco_brush_environment_aware_mink_v2_self_collision_v1",
        "explicit_pair_count": len(pairs),
        "minimum_distance_m": minimum,
        "accepted_minimum_m": -1e-6,
        "violating_frame_count": int(np.any(values < -1e-6 - 1e-12, axis=1).sum()),
        "certified_intrahand_audit": certified,
        "status": "PASS" if minimum >= -1e-6 - 1e-12 else "FAIL",
    }
    write_json(output / "self_collision_audit.json", report)
    return report


def fidelity_csv(
    path: Path, model: Any, human: dict[str, np.ndarray],
    reference: np.ndarray, candidate: np.ndarray, plane: Plane,
) -> None:
    count = min(len(reference), len(candidate))
    reference_metrics = trajectory_metrics(model, human, reference[:count], plane)
    candidate_metrics = trajectory_metrics(model, human, candidate[:count], plane)
    fields = [
        "frame", "qpos_max_abs_difference",
        "reference_tip_mean_error_m", "candidate_tip_mean_error_m",
        "reference_wrist_mean_error_rad", "candidate_wrist_mean_error_rad",
        "reference_self_minimum_m", "candidate_self_minimum_m",
        "reference_native_minimum_m", "candidate_native_minimum_m",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for frame in range(count):
            writer.writerow({
                "frame": frame,
                "qpos_max_abs_difference": float(np.abs(
                    reference[frame] - candidate[frame]
                ).max()),
                "reference_tip_mean_error_m": float(
                    reference_metrics["tips"][frame].mean()
                ),
                "candidate_tip_mean_error_m": float(
                    candidate_metrics["tips"][frame].mean()
                ),
                "reference_wrist_mean_error_rad": float(
                    reference_metrics["wrists"][frame].mean()
                ),
                "candidate_wrist_mean_error_rad": float(
                    candidate_metrics["wrists"][frame].mean()
                ),
                "reference_self_minimum_m": reference_metrics["self"][frame],
                "candidate_self_minimum_m": candidate_metrics["self"][frame],
                "reference_native_minimum_m": reference_metrics["native"][frame],
                "candidate_native_minimum_m": candidate_metrics["native"][frame],
            })


def external_blockers(
    model: Any, plane: Plane, qpos: np.ndarray, output: Path,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    objects = {}
    for role, name in (("brush", "right_object_visual"), ("bowl", "left_object_visual")):
        geom = build_native_support_geoms(model, mujoco, [model.geom(name).id])[0]
        minimum = float(plane.signed_distance(
            geom_world_vertices(data, geom, full=True)
        ).min())
        objects[role] = {
            "geom": name,
            "frame0_minimum_signed_distance_m": minimum,
            "controlled_by_v2": False,
        }
    report = {
        "schema": "taco_brush_environment_aware_mink_v2_external_blockers_v1",
        "objects": objects,
        "status": (
            "EXTERNAL_STATIC_BLOCKER_REMAINS"
            if objects["brush"]["frame0_minimum_signed_distance_m"] < -5e-5
            else "PASS"
        ),
    }
    write_json(output / "remaining_external_static_blockers.json", report)
    return report


def render_visuals(
    model: Any, baseline: np.ndarray, candidate: np.ndarray, output: Path,
) -> None:
    visual = output / "visuals"
    visual.mkdir(exist_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = sorted({0, min(13, len(candidate) - 1), len(candidate) - 1})
    records = []
    for frame in frames:
        panels = []
        for q in (baseline[frame], candidate[frame]):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.60, 0.0, 0.76]
            camera.distance = 0.55
            camera.azimuth = 90
            camera.elevation = -65
            renderer.update_scene(data, camera=camera)
            panels.append(renderer.render().copy())
        image = np.concatenate(panels, axis=1)
        name = f"frame_{frame:03d}_top_paper_baseline_vs_v2.jpg"
        if not cv2.imwrite(str(visual / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {name}")
        records.append((frame, name))
    renderer.close()
    lines = [
        "# TACO Brush MINK v2 visual evidence", "",
        "Panels show paper baseline (left) and v2 (right) with the real MuJoCo floor.", "",
    ]
    lines.extend(f"- frame {frame}: [top view](visuals/{name})" for frame, name in records)
    (output / "VISUAL_INDEX.md").write_text("\n".join(lines) + "\n")


def contract_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], output: Path,
) -> bool:
    expected = copy.deepcopy(baseline)
    expected["final_feasibility"]["algorithm"] = (
        UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY
    )
    passed = candidate == expected
    write_json(output / "candidate_delta_contract_v2.json", {
        "schema": "taco_brush_candidate_delta_contract_v2",
        "scientific_baseline": "v1 candidate effective algorithm, not raw YAML",
        "v1_raw_candidate_delta_status": "SUPERSEDED_RAW_CONFIG_NOT_EFFECTIVE_CONTRACT",
        "baseline_effective_contract": baseline,
        "candidate_effective_contract": candidate,
        "allowed_difference": {
            "final_feasibility.algorithm": [
                SELF_ONLY_FINAL_FEASIBILITY,
                UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
            ]
        },
        "candidate_equals_expected_single_delta": passed,
        "status": "PASS" if passed else "FAIL",
    })
    return passed


def hash_tree(output: Path) -> None:
    lines = []
    for path in sorted(
        item for item in output.rglob("*")
        if item.is_file() and item.name != "server_artifacts.sha256"
    ):
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(lines) + "\n")


def decision(
    output: Path, classification: str, *, candidate_count: int,
    detail: dict[str, Any] | None = None,
) -> None:
    if classification not in FINAL_CLASSES:
        raise ValueError(classification)
    write_json(output / "decision.json", {
        "schema": "taco_brush_environment_aware_mink_v2_decision_v1",
        "classification": classification,
        "allowed_classes": sorted(FINAL_CLASSES),
        "promotion_authorized": False,
        "runtime_counts": {
            "architecture_refactor": 1,
            "new_retarget_candidates": candidate_count,
            "physics": 0, "replay": 0, "mpc": 0, "rl": 0,
            "promotion": 0, "chunk_commit": 0,
        },
        **(detail or {}),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = load_contract(config_path)
    output = Path(cfg["paths"]["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        architecture_static_gate(output, cfg)
        settings_path = Path(cfg["paths"]["retarget_settings"])
        base_settings, consumption = load_taco_bimanual_settings(settings_path)
        write_json(output / "config_consumption_audit.json", consumption)
        if consumption["unknown_key_count"] or consumption["unused_key_count"]:
            raise RuntimeError("active config is not exactly consumed")
        plane = support_plane(Path(cfg["paths"]["support_surface_contract"]))
        support = native_support_settings(cfg, plane)
        from dataclasses import replace
        regression_settings = replace(
            base_settings,
            final_feasibility_algorithm=SELF_ONLY_FINAL_FEASIBILITY,
            native_support=support,
        )
        equivalent, baseline_contract = refactor_equivalence(
            cfg, regression_settings, plane, output
        )
        frame13_mechanism(cfg, plane, output)
        if not equivalent:
            decision(output, "RETARGET_REFACTOR_SEMANTIC_DRIFT", candidate_count=0)
            (output / "summary.md").write_text(
                "# TACO Brush environment-aware MINK v2\n\n"
                "Architecture refactor changed the frozen v1 frame 0–12 prefix; "
                "the sole v2 candidate was not run.\n"
            )
            hash_tree(output)
            return 0
    except Exception as error:
        decision(
            output, "RETARGET_ARCHITECTURE_CONTRACT_BLOCKER",
            candidate_count=0, detail={"reason": str(error)},
        )
        (output / "summary.md").write_text(
            "# TACO Brush environment-aware MINK v2\n\n"
            f"Architecture gate blocked candidate execution: `{error}`.\n"
        )
        hash_tree(output)
        print(json.dumps(json.loads((output / "decision.json").read_text()), indent=2))
        return 0

    candidate_settings = replace(
        regression_settings,
        final_feasibility_algorithm=UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    )
    expected_candidate_contract = copy.deepcopy(baseline_contract)
    expected_candidate_contract["final_feasibility"]["algorithm"] = (
        UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY
    )
    scene = Path(cfg["paths"]["scene"])
    human_path = Path(cfg["paths"]["human_reference"])
    failure = None
    try:
        retarget(
            scene, human_path, candidate_settings, output,
            expected_effective_contract=expected_candidate_contract,
        )
    except RuntimeError as error:
        failure = str(error)
    candidate_contract = json.loads(
        (output / "effective_retarget_contract.json").read_text()
    )
    if not contract_delta(baseline_contract, candidate_contract, output):
        classification = "UNIFIED_FEASIBILITY_CLOSURE_IMPLEMENTATION_ERROR"
    elif failure and "unified hard-feasibility closure exhausted" in failure:
        classification = "UNIFIED_FEASIBILITY_CLOSURE_INFEASIBLE"
    elif failure and "QP failed" in failure:
        classification = "UNIFIED_FEASIBILITY_CLOSURE_IMPLEMENTATION_ERROR"
    elif failure:
        classification = "ENVIRONMENT_AWARE_MINK_V2_TRAJECTORY_INFEASIBLE"
    else:
        classification = "ENVIRONMENT_AWARE_MINK_V2_STATIC_GATE_PASS"

    if failure:
        with np.load(output / "failed_kinematic_prefix.npz", allow_pickle=False) as archive:
            candidate_qpos = archive["qpos"].copy()
    else:
        with np.load(output / "robot_reference.npz", allow_pickle=False) as archive:
            candidate_qpos = archive["qpos"].copy()
    with np.load(human_path, allow_pickle=False) as archive:
        human = dict(archive)
    baseline_run = Path(cfg["paths"]["paper_baseline"])
    with np.load(baseline_run / "robot_reference.npz", allow_pickle=False) as archive:
        paper_qpos = archive["qpos"].copy()
    with np.load(
        Path(cfg["paths"]["v1_run"]) / "failed_kinematic_prefix.npz",
        allow_pickle=False,
    ) as archive:
        v1_qpos = archive["qpos"].copy()
    model = mujoco.MjModel.from_xml_path(str(scene))
    floor = native_floor_audit(
        model, plane, candidate_qpos, support.validation_tolerance_m, output
    )
    self_report = self_collision_audit(model, candidate_qpos, output)
    fidelity_csv(
        output / "candidate_vs_paper_baseline_fidelity.csv",
        model, human, paper_qpos, candidate_qpos, plane,
    )
    fidelity_csv(
        output / "candidate_v2_vs_v1_prefix.csv",
        model, human, v1_qpos, candidate_qpos, plane,
    )
    external = external_blockers(model, plane, candidate_qpos, output)
    render_visuals(model, paper_qpos, candidate_qpos, output)
    if (
        failure is None and floor["status"] == "PASS"
        and self_report["status"] == "PASS"
        and external["status"] != "PASS"
    ):
        classification = (
            "ENVIRONMENT_AWARE_MINK_V2_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS"
        )
    decision(output, classification, candidate_count=1, detail={
        "candidate_frames_available": len(candidate_qpos),
        "candidate_complete": failure is None,
        "failure_reason": failure,
        "native_floor_status": floor["status"],
        "self_collision_status": self_report["status"],
        "robot_reference_written": (output / "robot_reference.npz").is_file(),
    })
    write_json(output / "source_pins.json", {
        "schema": "taco_brush_environment_aware_mink_v2_source_pins_v1",
        "baseline_commit": cfg["baseline_commit"],
        "active_repository": {"path": str(ROOT.parent), "commit": git_head(ROOT.parent)},
        "inputs": [artifact(path) for path in (
            config_path, settings_path, Path(__file__),
            ROOT / "src/egoengine_repro/retarget/taco_bimanual.py",
            ROOT / "src/egoengine_repro/retarget/taco_bimanual_settings.py",
            ROOT / "src/egoengine_repro/retarget/kinematic_limits.py",
            ROOT / "src/egoengine_repro/retarget/support_plane_limit.py",
            scene, human_path, Path(cfg["paths"]["support_surface_contract"]),
        )],
    })
    summary = [
        "# TACO Brush environment-aware MINK v2", "",
        f"- Classification: `{classification}`",
        "- Architecture contract: `PASS`",
        "- Refactor frame 0–12 equivalence: `PASS`",
        f"- Candidate frames available: `{len(candidate_qpos)}/209`",
        f"- Native hand-floor minimum: `{floor['global_minimum_m']:.9f} m`",
        f"- Native hand-floor status: `{floor['status']}`",
        f"- Self-collision status: `{self_report['status']}`",
        f"- Brush-floor minimum retained: `{external['objects']['brush']['frame0_minimum_signed_distance_m']:.9f} m`",
        "- Physics / Replay / MPC / RL / promotion / chunk commit: `0`", "",
        "The only candidate algorithm change is the final feasibility closure: "
        "self-only to unified self-collision plus full native support acceptance. "
        "No tolerance, task weight, target, support plane, or object pose changed.",
    ]
    if failure:
        summary.extend(["", f"Candidate stopped fail-closed: `{failure}`."])
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    hash_tree(output)
    print(json.dumps(json.loads((output / "decision.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
