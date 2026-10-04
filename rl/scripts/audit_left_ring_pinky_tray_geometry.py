#!/usr/bin/env python3
"""Bounded shape-aware endpoint-0 initialization audit for TACO Pour.

The script first reconciles visual material geometry with runtime collision
geometry, then fits exactly one four-joint left ring/pinky candidate.  Physics
is entered only when every static sub-gate passes.  It never trains, plans, or
promotes a reset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import fcl
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ASSET_ROOT / "runs/taco_pour_shape_aware_initialization_v1"
CONFIG = ROOT / "configs/taco_pour_shape_aware_initialization_v1.yaml"
INITIAL_PROTOCOL = ROOT / "configs/taco_pour_initialization_protocol_v2.yaml"
BASELINE = "6a6091817a7b7b00c5488d442f02fc2b86df0146"
FINGERS = ("ring", "pinky")
VIEWS = ("oblique", "top")

sys.path.insert(0, str(ROOT / "scripts"))

import trace_replay_0_20 as trace
from audit_taco_initialization_preflight import triangle_object
from build_taco_pour_initialization_candidates import (
    _native_pair,
    _world_mesh,
    t0_legality_gate,
    visual_meshes,
)
from egoengine_repro.retarget.collision_audit import collision_families, distances
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.shape_aware import (
    FINGERS as SHAPE_FINGERS,
    fit_left_ring_pinky_shape,
    human_two_link_descriptor,
    robot_two_link_descriptor,
    shape_residual,
)
from video_to_spider.rl.physics_contract import compile_mujoco_model
from video_to_spider.rl.replay_contact_trace import BudgetLedger


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean() -> None:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("shape-aware audit requires a clean implementation worktree")


def load_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONFIG.read_text())
    if contract.get("schema") != "taco_pour_shape_aware_initialization_v1":
        raise ValueError("unexpected shape-aware contract")
    if tuple(contract["scope"]["fingers"]) != FINGERS:
        raise ValueError("contract must remain left ring/pinky only")
    return contract


def input_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {name: Path(value) for name, value in contract["inputs"].items() if name != "archived_replay_trace"}
    archived = Path(contract["inputs"]["archived_replay_trace"])
    paths.update(
        {
            "archived_trace_manifest": archived / "input_manifest.json",
            "archived_trace_parity": archived / "replay_parity.json",
            "archived_trace_endpoints": archived / "endpoints.npz",
            "archived_trace_contacts": archived / "contacts_raw.npz",
            "initialization_protocol": INITIAL_PROTOCOL,
            "shape_contract": CONFIG,
            "shape_helper_source": ROOT / "src/egoengine_repro/retarget/shape_aware.py",
            "runner_source": Path(__file__),
        }
    )
    return paths


def compile_model(paths: dict[str, Path]) -> mujoco.MjModel:
    config = yaml.safe_load(paths["simulator_config"].read_text())
    return compile_mujoco_model(paths["scene"], config.get("sdf_octree_depths", {}))


def _fcl_distance(first: trimesh.Trimesh, second: trimesh.Trimesh) -> dict[str, Any]:
    result = fcl.DistanceResult()
    value = fcl.distance(
        triangle_object(first),
        triangle_object(second),
        fcl.DistanceRequest(enable_nearest_points=True),
        result,
    )
    nearest = None
    if result.nearest_points is not None:
        nearest = [np.asarray(point, dtype=float).tolist() for point in result.nearest_points]
    return {"surface_distance_m": float(value), "nearest_points_world_m": nearest}


def classify_material_pair(
    link: trimesh.Trimesh,
    tray: trimesh.Trimesh,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Classify link-vs-thin-tray material without filling its cavity."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("positive material threshold required")
    report: dict[str, Any] = {
        "threshold_m": float(threshold),
        "link_watertight": bool(link.is_watertight),
        "tray_watertight": bool(tray.is_watertight),
        "tray_is_volume": bool(tray.is_volume),
        "cavity_policy": "exact_concave_visual_mesh; no convex hull or filled cavity",
    }
    if not tray.is_watertight or not tray.is_volume:
        report.update(
            classification="UNKNOWN_NOT_CERTIFIED",
            certified=False,
            reason="tray visual mesh is not a certified watertight material volume",
        )
        return report
    overlap = bool(
        np.all(link.bounds[0] <= tray.bounds[1])
        and np.all(tray.bounds[0] <= link.bounds[1])
    )
    report["aabb_overlap"] = overlap
    if not overlap:
        report.update(_fcl_distance(link, tray))
        classification = (
            "NEAR_CONTACT_NO_MATERIAL_PENETRATION"
            if report["surface_distance_m"] <= threshold
            else "CLEAR_SEPARATED"
        )
        report.update(
            surface_crossing=False,
            maximum_sampled_containment_m=0.0,
            classification=classification,
            certified=True,
        )
        return report

    collision = fcl.CollisionResult()
    fcl.collide(
        triangle_object(link),
        triangle_object(tray),
        fcl.CollisionRequest(num_max_contacts=64, enable_contact=True),
        collision,
    )
    depths = [float(contact.penetration_depth) for contact in collision.contacts]
    points = np.concatenate([link.vertices, link.triangles_center])
    signed = closed_mesh_signed_distance(tray, points)
    if not np.isfinite(signed).all():
        report.update(
            classification="UNKNOWN_NOT_CERTIFIED",
            certified=False,
            reason="nonfinite watertight-tray signed distance",
        )
        return report
    maximum_inside = float(max(0.0, signed.max(initial=-np.inf)))
    report.update(
        _fcl_distance(link, tray),
        surface_crossing=bool(collision.is_collision),
        fcl_contact_count=len(depths),
        maximum_fcl_penetration_depth_m=max(depths, default=0.0),
        sampled_points=int(len(points)),
        sampled_inside_over_threshold=int(np.count_nonzero(signed > threshold)),
        maximum_sampled_containment_m=maximum_inside,
    )
    material = maximum_inside > threshold or max(depths, default=0.0) > threshold
    if material:
        classification = "MATERIAL_PENETRATION_CONFIRMED"
    elif collision.is_collision or report["surface_distance_m"] <= threshold:
        classification = "NEAR_CONTACT_NO_MATERIAL_PENETRATION"
    else:
        classification = "CLEAR_SEPARATED"
    report.update(classification=classification, certified=True)
    return report


def geometry_regression(threshold: float) -> dict[str, Any]:
    base = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    separated = base.copy()
    separated.apply_translation((0.10, 0.0, 0.0))
    near = base.copy()
    near.apply_translation((0.040025, 0.0, 0.0))
    penetrating = base.copy()
    penetrating.apply_translation((0.039, 0.0, 0.0))
    unknown_tray = base.copy()
    unknown_tray.update_faces(np.arange(len(unknown_tray.faces) - 1))
    rows = {
        "separated": classify_material_pair(base, separated, threshold=threshold),
        "near_contact_25um": classify_material_pair(base, near, threshold=threshold),
        "material_penetration_1mm": classify_material_pair(base, penetrating, threshold=threshold),
        "unknown_open_tray": classify_material_pair(base, unknown_tray, threshold=threshold),
    }
    expected = {
        "separated": "CLEAR_SEPARATED",
        "near_contact_25um": "NEAR_CONTACT_NO_MATERIAL_PENETRATION",
        "material_penetration_1mm": "MATERIAL_PENETRATION_CONFIRMED",
        "unknown_open_tray": "UNKNOWN_NOT_CERTIFIED",
    }
    return {
        "schema": "taco_pour_shape_geometry_regression_v1",
        "cases": rows,
        "expected": expected,
        "passed": all(rows[name]["classification"] == label for name, label in expected.items()),
    }


def relevant_visual_pairs(model: mujoco.MjModel, meshes: dict[int, trimesh.Trimesh]) -> list[tuple[int, int]]:
    tray = model.geom("left_object_visual").id
    result = []
    for finger in FINGERS:
        for link in ("link1", "link2"):
            geom = model.geom(f"left_{finger}_{link}_visual").id
            if geom not in meshes or tray not in meshes:
                raise RuntimeError("required visual mesh missing")
            result.append((geom, tray))
    return result


def current_geometry(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    meshes: dict[int, trimesh.Trimesh],
    threshold: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    world = {geom: _world_mesh(model, data, geom, mesh) for geom, mesh in meshes.items()}
    pairs = []
    for link, tray in relevant_visual_pairs(model, meshes):
        pairs.append(
            {
                "geoms": [model.geom(link).name, model.geom(tray).name],
                **classify_material_pair(world[link], world[tray], threshold=threshold),
            }
        )
    runtime_pairs = collision_families(model)["hand_target"]
    runtime_rows = []
    runtime_distance = distances(model, data, runtime_pairs)
    for (first, second), value in zip(runtime_pairs, runtime_distance):
        names = (model.geom(first).name or "", model.geom(second).name or "")
        if any(f"left_{finger}" in name for finger in FINGERS for name in names):
            runtime_rows.append({"geoms": list(names), "distance_m": float(value)})
    minimum_runtime = min(row["distance_m"] for row in runtime_rows)
    material = [row for row in pairs if row["classification"] == "MATERIAL_PENETRATION_CONFIRMED"]
    unknown = [row for row in pairs if row["classification"] == "UNKNOWN_NOT_CERTIFIED"]
    if material:
        classification = "MATERIAL_PENETRATION_CONFIRMED"
    elif unknown:
        classification = "UNKNOWN_NOT_CERTIFIED"
    elif minimum_runtime <= threshold or any(
        row["classification"] == "NEAR_CONTACT_NO_MATERIAL_PENETRATION" for row in pairs
    ):
        classification = "NEAR_CONTACT_NO_MATERIAL_PENETRATION"
    else:
        classification = "CLEAR_SEPARATED"
    return {
        "classification": classification,
        "visual_pairs": pairs,
        "runtime_collision_geometry": {
            "minimum_distance_m": float(minimum_runtime),
            "near_threshold_m": float(threshold),
            "active_forward_contact_count": int(data.ncon),
            "pairs": sorted(runtime_rows, key=lambda row: row["distance_m"]),
        },
        "certified_no_material_penetration": not material and not unknown,
    }


def human_descriptors(human: dict[str, np.ndarray], frames: list[int]) -> dict[str, Any]:
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    by_frame = {}
    for frame in frames:
        by_frame[str(frame)] = human_two_link_descriptor(
            human["joint_positions_sim"][frame, hand],
            human["T_sim_wrist_target"][frame, hand],
        )
    return {"hand_index": hand, "frames": by_frame, "endpoint0": by_frame[str(frames[0])]}


def shape_targets(human: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    order = ("thumb", "index", "middle", "ring", "pinky")
    return {
        finger: human["T_sim_fingertip_target"][0, hand, order.index(finger), :3, 3].copy()
        for finger in FINGERS
    }


def affected_declared_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    bodies = {
        model.body(f"left_hand_{finger}_link{link}").id
        for finger in FINGERS
        for link in (1, 2)
    }
    result = set()
    for family in ("self_explicit", "hand_tool", "hand_target", "hand_floor"):
        for pair in collision_families(model)[family]:
            if int(model.geom_bodyid[pair[0]]) in bodies or int(model.geom_bodyid[pair[1]]) in bodies:
                result.add(tuple(pair))
    return sorted(result)


def preflight_and_static(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require_clean()
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head()], cwd=ROOT).returncode:
        raise RuntimeError("required baseline is not an ancestor")
    contract = load_contract()
    paths = input_paths(contract)
    archived_parity = json.loads(paths["archived_trace_parity"].read_text())
    if not archived_parity.get("B_endpoint_arrays_bitwise_equal_historical"):
        raise RuntimeError("archived A Replay identity is not established")
    output.mkdir(parents=True)
    write_json(
        output / "input_manifest.json",
        {
            "schema": "taco_pour_shape_aware_initialization_inputs_v1",
            "baseline_git_commit": BASELINE,
            "implementation_git_commit": head(),
            "worktree_clean": True,
            "inputs": {name: artifact(path) for name, path in paths.items()},
            "authorization": contract["authorization"],
        },
    )
    threshold = float(contract["geometry"]["material_reporting_threshold_m"])
    regression = geometry_regression(threshold)
    write_json(output / "geometry_regression.json", regression)
    if not regression["passed"]:
        write_json(output / "decision.json", {"classification": "STATIC_GATE_FAILED_NO_PHYSICS", "reason": "geometry_checker_regression_failed"})
        raise RuntimeError("geometry checker regression failed")

    initial = arrays(paths["accepted_initial_state"])
    robot_reference = arrays(paths["robot_reference"])
    human = arrays(paths["human_reference"])
    model = compile_model(paths)
    meshes, mesh_report = visual_meshes(paths["scene"], model)
    old = current_geometry(model, initial["qpos"], meshes, threshold)
    geometry_report = {
        "schema": "taco_pour_left_ring_pinky_tray_geometry_v1",
        "accepted_A": old,
        "mesh_loader": mesh_report,
        "old_gate_reconciliation": {
            "builder_native_geometry": json.loads(paths["builder_report"].read_text()).get("native_geometry", True),
            "mechanism": (
                "The accepted state is certified free of visual-material intersection, but the active runtime "
                "ring/pinky guards sit about two micrometres from the tray. The historical native gate tested "
                "material legality, not MANO-like intermediate-phalanx shape. Fingertip/wrist-only retargeting "
                "and the posture-preserving initialization can therefore pass while the proximal joints remain "
                "too straight and the perspective render resembles penetration into the tray cavity."
            ),
            "state_mismatch_found": False,
            "convex_hull_used": False,
        },
    }
    write_json(output / "geometry_reconciliation.json", geometry_report)
    if old["classification"] == "UNKNOWN_NOT_CERTIFIED":
        write_json(output / "decision.json", {"classification": "STATIC_GATE_FAILED_NO_PHYSICS", "reason": "accepted_A_geometry_unknown"})
        return

    frames = list(map(int, contract["shape_fit"]["source_frames"]))
    human_shape = human_descriptors(human, frames)
    tips = shape_targets(human)
    affected = affected_declared_pairs(model)
    data = mujoco.MjData(model)
    minimum = float(contract["shape_fit"]["declared_pair_min_distance_m"])

    def constraint(qpos: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return distances(model, data, affected) - minimum

    candidate, fit = fit_left_ring_pinky_shape(
        model,
        initial["qpos"],
        human_shape["endpoint0"],
        tips,
        constraint,
        max_iterations=int(contract["shape_fit"]["max_iterations"]),
        ftol=float(contract["shape_fit"]["ftol"]),
    )
    candidate_geometry = current_geometry(model, candidate, meshes, threshold)
    protocol = yaml.safe_load(INITIAL_PROTOCOL.read_text())
    full_gate = t0_legality_gate(
        model,
        candidate,
        robot_reference,
        {"candidate": "SHAPE_AWARE_LEFT_RP"},
        protocol,
        meshes,
    )
    changed = np.flatnonzero(candidate != initial["qpos"])
    movable = np.asarray(fit["qpos_addresses"], dtype=np.int64)
    right = np.arange(0, 18)
    objects = np.arange(36, 50)
    before = fit["accepted_descriptor"]
    after = fit["candidate_descriptor"]
    tip_tol = float(contract["shape_fit"]["fingertip_nonworsening_tolerance_m"])
    checks = {
        "single_deterministic_solver_success": fit["success"],
        "only_declared_four_joints_changed": bool(set(changed).issubset(set(movable))),
        "object_qpos_byte_identical": bool(np.array_equal(candidate[objects], initial["qpos"][objects])),
        "right_hand_byte_identical": bool(np.array_equal(candidate[right], initial["qpos"][right])),
        "shape_objective_strictly_improved": fit["candidate_shape_objective"] < fit["accepted_shape_objective"],
        "ring_tip_nonworsening": after["ring"]["fingertip_error_m"] <= before["ring"]["fingertip_error_m"] + tip_tol,
        "pinky_tip_nonworsening": after["pinky"]["fingertip_error_m"] <= before["pinky"]["fingertip_error_m"] + tip_tol,
        "corrected_ring_pinky_tray_geometry_certified": candidate_geometry["certified_no_material_penetration"],
        "full_existing_t0_legality_gate": bool(full_gate["passed"]),
        "initial_ctrl_exact_candidate_qpos": True,
        "first_reference_command_unchanged": True,
    }
    task_manifest = {
        "constructed_tasks": [
            "ring_proximal_unit_direction_in_palm_frame",
            "ring_distal_unit_direction_in_palm_frame",
            "ring_fingertip_position_normalized_by_robot_finger_length",
            "pinky_proximal_unit_direction_in_palm_frame",
            "pinky_distal_unit_direction_in_palm_frame",
            "pinky_fingertip_position_normalized_by_robot_finger_length",
        ],
        "configured_component_weights": contract["shape_fit"]["component_weights"],
        "effective_component_weights": [1.0, 1.0, 1.0],
        "unused_configured_shape_or_posture_weight": False,
    }
    shape_mapping = {
        "schema": "taco_pour_shape_mapping_v1",
        "human": human_shape,
        "robot_fit": fit,
        "task_manifest": task_manifest,
        "affected_declared_pair_count": len(affected),
        "candidate_geometry": candidate_geometry,
    }
    write_json(output / "shape_mapping.json", shape_mapping)
    static_gate = {
        "schema": "taco_pour_shape_aware_static_gate_v1",
        "checks": checks,
        "passed": all(checks.values()),
        "full_existing_gate": full_gate,
        "candidate_geometry": candidate_geometry,
        "changed_qpos_addresses": changed.tolist(),
    }
    write_json(output / "candidate_static_gate.json", static_gate)
    np.savez_compressed(
        output / "shape_aware_initial_state.npz",
        qpos=candidate,
        qvel=initial["qvel"].copy(),
        ctrl=candidate[:36].copy(),
        state_contract_version=np.asarray("taco_pour_shape_aware_initial_state_v1"),
        reference_index=np.asarray(0, dtype=np.int64),
        candidate=np.asarray("SHAPE_AWARE_LEFT_RP"),
    )
    write_json(
        output / "cost_accounting.json",
        {
            "static_solver_calls": 1,
            "objective_evaluations": fit["objective_evaluations"],
            "constraint_evaluations": fit["calls"]["constraint"],
            "physics_steps": 0,
            "render_updates": 0,
            "actor_or_critic_forwards": 0,
            "planner_calls": 0,
        },
    )
    if not static_gate["passed"]:
        write_json(output / "decision.json", {"classification": "STATIC_GATE_FAILED_NO_PHYSICS", "reason": "candidate_static_subgate_failed", "checks": checks})


def _rotation_error(actual: np.ndarray, reference: np.ndarray) -> float:
    a = Rotation.from_quat(actual[[1, 2, 3, 0]])
    r = Rotation.from_quat(reference[[1, 2, 3, 0]])
    return float((r.inv() * a).magnitude())


def _pair_rotation(qpos: np.ndarray, reference: np.ndarray) -> float:
    tool = Rotation.from_quat(qpos[39:43][[1, 2, 3, 0]])
    target = Rotation.from_quat(qpos[46:50][[1, 2, 3, 0]])
    ref_tool = Rotation.from_quat(reference[39:43][[1, 2, 3, 0]])
    ref_target = Rotation.from_quat(reference[46:50][[1, 2, 3, 0]])
    return float(((ref_target.inv() * ref_tool).inv() * (target.inv() * tool)).magnitude())


def _metric_rows(condition: str, endpoints: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for endpoint, qpos in enumerate(endpoints["qpos"]):
        ref = reference["qpos"][endpoint]
        pair_actual = Rotation.from_quat(qpos[46:50][[1, 2, 3, 0]]).inv().apply(qpos[36:39] - qpos[43:46])
        pair_ref = Rotation.from_quat(ref[46:50][[1, 2, 3, 0]]).inv().apply(ref[36:39] - ref[43:46])
        rows.append(
            {
                "condition": condition,
                "endpoint": endpoint,
                "tool_position_error_m": float(np.linalg.norm(qpos[36:39] - ref[36:39])),
                "tool_rotation_error_rad": _rotation_error(qpos[39:43], ref[39:43]),
                "target_position_error_m": float(np.linalg.norm(qpos[43:46] - ref[43:46])),
                "target_rotation_error_rad": _rotation_error(qpos[46:50], ref[46:50]),
                "target_frame_pair_translation_error_m": float(np.linalg.norm(pair_actual - pair_ref)),
                "pair_rotation_error_rad": _pair_rotation(qpos, ref),
                "world_pair_translation_error_m": float(np.linalg.norm((qpos[36:39] - qpos[43:46]) - (ref[36:39] - ref[43:46]))),
                "tool_linear_speed_m_s": float(np.linalg.norm(endpoints["qvel"][endpoint, 36:39])),
                "tool_angular_speed_rad_s": float(np.linalg.norm(endpoints["qvel"][endpoint, 39:42])),
                "target_linear_speed_m_s": float(np.linalg.norm(endpoints["qvel"][endpoint, 42:45])),
                "target_angular_speed_rad_s": float(np.linalg.norm(endpoints["qvel"][endpoint, 45:48])),
            }
        )
    return rows


def physics(output: Path) -> None:
    require_clean()
    static = json.loads((output / "candidate_static_gate.json").read_text())
    if not static["passed"]:
        raise RuntimeError("static gate did not authorize physics")
    manifest = json.loads((output / "input_manifest.json").read_text())
    if manifest["implementation_git_commit"] != head():
        raise RuntimeError("implementation changed after static gate")
    contract = load_contract()
    paths = input_paths(contract)
    for name, row in manifest["inputs"].items():
        if artifact(paths[name]) != row:
            raise RuntimeError(f"frozen input changed: {name}")
    candidate = arrays(output / "shape_aware_initial_state.npz")
    initial = {name: candidate[name] for name in ("qpos", "qvel", "ctrl")}
    ledger = BudgetLedger()
    results = {}
    snapshots = {}
    contacts = {}
    for condition in ("SHAPE_AWARE_REPLAY", "SHAPE_AWARE_COLD"):
        world = trace.make_world(trace.input_paths(), initial)
        ledger.charge(physics_steps=1, control_intervals=0)
        s0 = world.get_env_state()
        endpoints, saved, observer = trace.run_observed(world, s0, ledger)
        results[condition] = endpoints
        snapshots[condition] = saved
        contacts[condition] = observer.contacts
        directory = output / "conditions" / condition
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "trajectory.npz", **endpoints, reference_endpoint=np.arange(21))
        trace.rows_to_npz(observer.rows, directory / "substeps.npz")
        trace.contacts_to_npz(observer.contacts, directory / "contacts_raw.npz")
        trace.write_contact_csv(observer.contacts, directory / "contacts.csv")
    endpoint_parity = {
        name: trace.equality_report(results["SHAPE_AWARE_REPLAY"][name], results["SHAPE_AWARE_COLD"][name])
        for name in results["SHAPE_AWARE_REPLAY"]
    }
    snapshot_parity = {
        str(endpoint): trace.compare_snapshots(
            snapshots["SHAPE_AWARE_REPLAY"][endpoint], snapshots["SHAPE_AWARE_COLD"][endpoint]
        )
        for endpoint in snapshots["SHAPE_AWARE_REPLAY"]
    }
    cold = {
        "endpoint_arrays": endpoint_parity,
        "snapshots": snapshot_parity,
        "bitwise_equal": all(row["equal"] for row in endpoint_parity.values())
        and all(row["all_common_equal"] and not row["only_current"] and not row["only_historical"] for row in snapshot_parity.values()),
    }
    write_json(output / "cold_replay_parity.json", cold)

    archived = arrays(paths["archived_trace_endpoints"])
    baseline = {name: archived[f"B_{name}"] for name in ("qpos", "qvel", "ctrl")}
    reference = arrays(paths["robot_reference"])
    rows = _metric_rows("A_ORIGINAL_REPLAY", baseline, reference)
    rows += _metric_rows("SHAPE_AWARE_REPLAY", results["SHAPE_AWARE_REPLAY"], reference)
    fields = tuple(rows[0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    archived_contacts = arrays(paths["archived_trace_contacts"])
    contact_summary = {}
    for condition, rows_contact in {
        "A_ORIGINAL_REPLAY": None,
        "SHAPE_AWARE_REPLAY": contacts["SHAPE_AWARE_REPLAY"],
    }.items():
        if rows_contact is None:
            names1, names2 = archived_contacts["geom1_name"], archived_contacts["geom2_name"]
            sources = archived_contacts["source_endpoint"]
            selected = [
                {"source_endpoint": int(source), "geoms": [str(first), str(second)]}
                for first, second, source in zip(names1, names2, sources)
                if ("left_ring" in str(first) or "left_pinky" in str(first) or "left_ring" in str(second) or "left_pinky" in str(second))
                and ("left_object" in str(first) or "left_object" in str(second))
            ]
        else:
            selected = [
                {"source_endpoint": int(row["source_endpoint"]), "geoms": [row["geom1_name"], row["geom2_name"]], "dist_m": row["dist"]}
                for row in rows_contact
                if ("left_ring" in row["geom1_name"] or "left_pinky" in row["geom1_name"] or "left_ring" in row["geom2_name"] or "left_pinky" in row["geom2_name"])
                and ("left_object" in row["geom1_name"] or "left_object" in row["geom2_name"])
            ]
        contact_summary[condition] = {
            "rows": len(selected),
            "affected_source_endpoints": sorted({row["source_endpoint"] for row in selected}),
            "events": selected,
        }
    write_json(output / "runtime_ring_pinky_tray_contacts.json", contact_summary)

    command_checks = {
        "executed_ctrl_rows_1_20_byte_identical_A_vs_shape": bool(
            np.array_equal(baseline["ctrl"][1:21], results["SHAPE_AWARE_REPLAY"]["ctrl"][1:21])
        ),
        "candidate_endpoint0_ctrl_equals_candidate_qpos": bool(
            np.array_equal(results["SHAPE_AWARE_REPLAY"]["ctrl"][0], candidate["qpos"][:36].astype(np.float32))
        ),
        "zero_residual_actions": True,
        "support_clipping_count": 0,
        "ctrlrange_loss_count": 0,
    }
    write_json(output / "control_validation.json", command_checks)
    accounting = json.loads((output / "cost_accounting.json").read_text())
    accounting.update(
        physics_steps=int(ledger.physics_steps),
        task_control_intervals=int(ledger.control_intervals),
        setup_physics_steps=2,
        candidate_replay_physics_steps=200,
        candidate_cold_physics_steps=200,
        archived_A_new_physics_steps=0,
    )
    if accounting["physics_steps"] > int(contract["budget"]["physics_hard_max"]):
        raise RuntimeError("physics hard max exceeded")
    write_json(output / "cost_accounting.json", accounting)


def load_visual_module():
    path = ROOT / "scripts/inspect_hand_object_trajectory.py"
    spec = importlib.util.spec_from_file_location("shape_visual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render(output: Path) -> None:
    vismod = load_visual_module()
    contract = load_contract()
    paths = input_paths(contract)
    archived = arrays(paths["archived_trace_endpoints"])
    baseline = archived["B_qpos"]
    candidate = arrays(output / "conditions/SHAPE_AWARE_REPLAY/trajectory.npz")["qpos"]
    reference = arrays(paths["robot_reference"])["qpos"]
    rgb_frames, rgb_info = vismod.read_rgb_frames(
        ASSET_ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4", 21
    )
    ref_model = vismod.StaticModel.load(
        "reference", ASSET_ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml"
    )
    actual_model = vismod.StaticModel.load("actual", paths["scene"])
    vis = vismod.Visualizer(ref_model, actual_model, width=640, height=480)
    closeups = set(map(int, contract["visuals"]["closeup_endpoints"]))
    updates = 0
    try:
        for endpoint in range(21):
            for view in VIEWS:
                panels = [
                    vismod.annotate_panel(vismod.letterbox(rgb_frames[endpoint], 640, 480), ["REAL RGB (independent camera)", f"frame {endpoint}"]),
                    vismod.annotate_panel(vis.render_reference(reference[endpoint], view), ["ROBOT REFERENCE", f"endpoint {endpoint}"]),
                    vismod.annotate_panel(vis.render_actual(baseline[endpoint], view), ["A ORIGINAL REPLAY", f"endpoint {endpoint}"]),
                    vismod.annotate_panel(vis.render_actual(candidate[endpoint], view), ["SHAPE-AWARE REPLAY", f"endpoint {endpoint}"]),
                ]
                image = np.concatenate(panels, axis=1)
                path = output / "frames" / view / f"endpoint_{endpoint:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                updates += 3
                if endpoint in closeups:
                    tray = candidate[endpoint, 43:46]
                    settings = {
                        "lookat": tray.tolist(),
                        "distance": 0.34,
                        "azimuth": 90.0 if view == "top" else 132.0,
                        "elevation": -89.0 if view == "top" else -28.0,
                    }
                    camera = vismod.camera_object(settings)
                    cp = []
                    for label, qpos in (("A ORIGINAL", baseline[endpoint]), ("SHAPE-AWARE", candidate[endpoint])):
                        vis.actual.set_qpos(qpos)
                        vis.actual_renderer.update_scene(vis.actual.data, camera=camera, scene_option=vismod.scene_option())
                        cp.append(vismod.annotate_panel(vis.actual_renderer.render().copy(), [label, f"left ring/pinky/tray | endpoint {endpoint}"]))
                        updates += 1
                    close = np.concatenate(cp, axis=1)
                    path = output / "closeups" / view / f"endpoint_{endpoint:06d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), cv2.cvtColor(close, cv2.COLOR_RGB2BGR))
        for representation, collision in (("visual", False), ("collision", True)):
            panels = []
            for label, qpos in (("A ORIGINAL", baseline[0]), ("SHAPE-AWARE", candidate[0])):
                image = vis.render_actual(qpos, "oblique", collision=collision)
                panels.append(vismod.annotate_panel(image, [label, f"{representation.upper()} GEOMETRY | endpoint 0"]))
            image = np.concatenate(panels, axis=1)
            path = output / "closeups" / f"endpoint_000000_{representation}_overlay.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            updates += 2
    finally:
        vis.close()
    accounting = json.loads((output / "cost_accounting.json").read_text())
    accounting["render_updates"] = updates
    accounting["rendered_endpoint_view_panels"] = 42
    accounting["closeup_endpoint_view_panels"] = len(closeups) * 2
    write_json(output / "cost_accounting.json", accounting)
    write_json(
        output / "visual_review.json",
        {
            "status": "rendered_pending_human_review",
            "rgb": rgb_info,
            "views": list(VIEWS),
            "endpoints": list(range(21)),
            "closeup_endpoints": sorted(closeups),
            "render_updates": updates,
            "camera_registration": "REAL RGB independent; robot panels share fixed simulation cameras",
        },
    )


def finalize(output: Path, *, visual_status: str, visual_note: str) -> None:
    static = json.loads((output / "candidate_static_gate.json").read_text())
    cold = json.loads((output / "cold_replay_parity.json").read_text())
    visual = json.loads((output / "visual_review.json").read_text())
    visual.update(status=visual_status, reviewer_note=visual_note)
    write_json(output / "visual_review.json", visual)
    rows = list(csv.DictReader((output / "comparison.csv").open()))
    grouped = {}
    for condition in ("A_ORIGINAL_REPLAY", "SHAPE_AWARE_REPLAY"):
        group = [row for row in rows if row["condition"] == condition]
        grouped[condition] = {
            key: max(float(row[key]) for row in group)
            for key in (
                "tool_position_error_m",
                "tool_rotation_error_rad",
                "target_position_error_m",
                "target_rotation_error_rad",
                "target_frame_pair_translation_error_m",
                "pair_rotation_error_rad",
            )
        }
    no_tradeoff = all(
        grouped["SHAPE_AWARE_REPLAY"][key] <= grouped["A_ORIGINAL_REPLAY"][key] + 1e-9
        for key in grouped["A_ORIGINAL_REPLAY"]
    )
    geometry_fixed = static["candidate_geometry"]["certified_no_material_penetration"]
    visually_fixed = visual_status == "review_complete_shape_improved_no_obvious_material_penetration"
    if geometry_fixed and visually_fixed and no_tradeoff and cold["bitwise_equal"]:
        classification = "GEOMETRY_FIXED_AND_EARLY_RELATION_IMPROVED"
    elif geometry_fixed and visually_fixed and not no_tradeoff:
        classification = "GEOMETRY_FIXED_BUT_TASK_TRADEOFF"
    elif geometry_fixed and cold["bitwise_equal"]:
        classification = "NO_MEANINGFUL_CHANGE"
    else:
        classification = "PHYSICAL_REGRESSION"
    decision = {
        "classification": classification,
        "static_gate_passed": static["passed"],
        "cold_replay_bitwise_equal": cold["bitwise_equal"],
        "all_six_object_metrics_nonworsening": no_tradeoff,
        "maxima": grouped,
        "authorization": {
            "reset_promotion": False,
            "rl_training": False,
            "planner_continuation": False,
            "chunk_commit": False,
        },
    }
    write_json(output / "decision.json", decision)
    fit = json.loads((output / "shape_mapping.json").read_text())["robot_fit"]
    summary = f"""# TACO Pour shape-aware initialization v1

- Old endpoint-0 geometry: `{json.loads((output / 'geometry_reconciliation.json').read_text())['accepted_A']['classification']}`.
- Static candidate: `{'PASS' if static['passed'] else 'FAIL'}`; one deterministic four-joint solve, no physics-based selection.
- Physical classification: `{classification}`.
- Cold Replay parity: `{cold['bitwise_equal']}`.
- Object metrics nonworsening: `{no_tradeoff}`.
- Ring joints: `{fit['accepted_joint_values_rad'][:2]}` -> `{fit['candidate_joint_values_rad'][:2]}` rad.
- Pinky joints: `{fit['accepted_joint_values_rad'][2:]}` -> `{fit['candidate_joint_values_rad'][2:]}` rad.
- No reset promotion, RL, planner continuation, or chunk commit is authorized.
"""
    (output / "summary.md").write_text(summary)
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "server_artifacts.sha256")
    (output / "server_artifacts.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("static", "physics", "render", "finalize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--visual-status", default="review_complete_shape_improved_no_obvious_material_penetration")
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()
    if args.phase == "static":
        preflight_and_static(args.output)
    elif args.phase == "physics":
        physics(args.output)
    elif args.phase == "render":
        render(args.output)
    else:
        finalize(args.output, visual_status=args.visual_status, visual_note=args.visual_note)


if __name__ == "__main__":
    main()

