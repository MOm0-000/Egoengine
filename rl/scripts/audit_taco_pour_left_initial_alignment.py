#!/usr/bin/env python3
"""Bounded left-only initial-alignment diagnostic; never trains or commits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ASSET_ROOT / "runs/taco_pour_left_initial_alignment_v1"
BASELINE = "9c595ec2ad2e11cff3cc7f2f8083cebd2bff34b1"
STATIC_SOLVE_SERIALIZATION_FAILURE_COMMIT = "8a8ce76c79a694f38ef4477ff9b033fc4a4243a2"
CONFIG = ROOT / "configs/taco_pour_left_initial_alignment_v1.yaml"
INITIAL_PROTOCOL = ROOT / "configs/taco_pour_initialization_protocol_v2.yaml"
SUMMARY_ENDPOINTS = (0, 1, 5, 10, 14, 15, 16, 20)

sys.path.insert(0, str(ROOT / "scripts"))

import trace_replay_0_20 as trace
from build_taco_pour_initialization_candidates import t0_legality_gate, visual_meshes
from egoengine_repro.retarget.initial_hand import solve_left_reference_aligned_initial
from video_to_spider.rl.physics_contract import compile_mujoco_model
from video_to_spider.rl.replay_contact_trace import BudgetLedger, startup_control_sequences


def write_json(path: Path, value: Any) -> None:
    trace.write_json(path, value)


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean() -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("left-alignment diagnostic requires a clean worktree")


def require_baseline() -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE, head()], cwd=ROOT, check=False
    ).returncode:
        raise RuntimeError(f"required baseline {BASELINE} is not an ancestor")


def input_paths() -> dict[str, Path]:
    paths = trace.startup_input_paths()
    paths.update({
        "left_alignment_contract": CONFIG,
        "initialization_protocol": INITIAL_PROTOCOL,
        "initialization_comparison": ASSET_ROOT / "runs/taco_pour_initialization_protocol_v2/comparison.json",
        "initialization_builder_report": ASSET_ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/builder_report.json",
        "early_contact_findings": ASSET_ROOT / "runs/taco_pour_early_contact_origin_v1/findings.md",
        # The lightweight narrative is Git-tracked; only the large startup
        # arrays remain under the server asset root.
        "startup_summary": ROOT / "runs/taco_pour_startup_transition_isolation_v1/summary.md",
        "startup_comparison": ASSET_ROOT / "runs/taco_pour_startup_transition_isolation_v1/comparison.csv",
        "startup_hold_endpoints": ASSET_ROOT / "runs/taco_pour_startup_transition_isolation_v1/conditions/HOLD_1/endpoints.npz",
        "startup_hold_substeps": ASSET_ROOT / "runs/taco_pour_startup_transition_isolation_v1/conditions/HOLD_1/substeps.npz",
        "startup_hold_contacts": ASSET_ROOT / "runs/taco_pour_startup_transition_isolation_v1/conditions/HOLD_1/contacts_raw.npz",
        "initial_hand_source": ROOT / "src/egoengine_repro/retarget/initial_hand.py",
        "runner_source": Path(__file__),
    })
    return paths


def preflight(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require_clean()
    require_baseline()
    config = yaml.safe_load(CONFIG.read_text())
    if config.get("schema") != "taco_pour_left_initial_alignment_v1":
        raise RuntimeError("unexpected left-alignment contract")
    paths = input_paths()
    resolved = {name: trace.artifact(path) for name, path in paths.items()}
    expected = config["inputs"]["expected_hashes"]
    checks = {
        "accepted_A_initial_state": resolved["initial_state"]["sha256"],
        "runtime_scene": resolved["scene"]["sha256"],
        "robot_reference": resolved["reference"]["sha256"],
        "simulator_config": resolved["simulator_config"]["sha256"],
        "historical_trajectory": resolved["historical_trajectory"]["sha256"],
    }
    for name, digest in checks.items():
        if digest != expected[name]:
            raise RuntimeError(f"frozen input hash changed: {name}")
    parity = json.loads(paths["trace_replay_parity"].read_text())
    if not parity.get("B_endpoint_arrays_bitwise_equal_historical"):
        raise RuntimeError("Replay 0->20 identity is not established")
    output.mkdir(parents=True)
    write_json(output / "resolved_inputs.json", {
        "schema": "taco_pour_left_initial_alignment_inputs_v1",
        "baseline_git_commit": BASELINE,
        "implementation_git_commit": head(),
        "inputs": resolved,
        "expected_hashes_verified": True,
        "historical_s0_provenance": json.loads(paths["trace_manifest"].read_text())["s0_source"],
    })
    write_json(output / "budget_ledger.json", {
        "schema": "taco_pour_left_initial_alignment_budget_v1",
        "status": "reserved_before_static_solve",
        "qp_calls": 0,
        "geometry_checks": 0,
        "task_control_intervals": 0,
        "task_physics_steps": 0,
        "setup_physics_steps": 0,
        "limits": config["budget"],
    })
    write_json(output / "status.json", {
        "schema": "taco_pour_left_initial_alignment_v1",
        "status": "preflight_complete_no_solver_or_physics_executed",
        "implementation_git_commit": head(),
    })


def _compile(paths: dict[str, Path]):
    simulator = yaml.safe_load(paths["simulator_config"].read_text())
    return compile_mujoco_model(paths["scene"], simulator.get("sdf_octree_depths", {}))


def _tip_arrays(states: list[dict[str, Any]], key: str) -> np.ndarray:
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    return np.asarray([[row[key][finger] for finger in fingers] for row in states])


def static_solve(output: Path) -> None:
    require_clean()
    require_baseline()
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "preflight_complete_no_solver_or_physics_executed":
        raise RuntimeError("static solve requires unused preflight")
    resolved = json.loads((output / "resolved_inputs.json").read_text())
    if resolved["implementation_git_commit"] != head():
        raise RuntimeError("implementation changed after preflight")
    paths = input_paths()
    for name, row in resolved["inputs"].items():
        if trace.artifact(paths[name]) != row:
            raise RuntimeError(f"input changed after preflight: {name}")
    contract = yaml.safe_load(CONFIG.read_text())
    init_protocol = yaml.safe_load(INITIAL_PROTOCOL.read_text())
    initial = trace.load_initial(paths)
    reference = arrays(paths["reference"])
    # The solver may temporarily alter collision masks for MINK planning.  All
    # native/runtime acceptance checks therefore use a separately compiled,
    # untouched validation model.
    model = _compile(paths)
    solver_cfg = contract["static_solver"]
    trace_qpos, report = solve_left_reference_aligned_initial(
        model,
        initial["qpos"],
        reference["qpos"][0],
        init_protocol["hand_solver"]["velocity_limits"],
        max_qp_calls=int(solver_cfg["max_qp_calls"]),
        numerical_dt=float(solver_cfg["numerical_dt_s"]),
        planning_collision_buffer=float(init_protocol["hand_solver"]["planning_collision_buffer_m"]),
        solver_primal_tolerance=float(solver_cfg["primal_tolerance"]),
        solver_dual_tolerance=float(solver_cfg["dual_tolerance"]),
        depenetration_step=float(init_protocol["hand_solver"]["depenetration_step_m"]),
    )
    states = report["states"]
    if not report["locked_coordinates_byte_identical"]:
        raise RuntimeError("static trajectory changed a locked coordinate")
    validation_model = _compile(paths)
    meshes, _ = visual_meshes(paths["scene"], validation_model)
    baseline = states[0]
    baseline_gate64 = t0_legality_gate(
        validation_model, trace_qpos[0], reference, {"candidate": "accepted_A"}, init_protocol, meshes
    )
    reference32 = {key: value.copy() for key, value in reference.items()}
    reference32["qpos"] = reference32["qpos"].astype(np.float32).astype(np.float64)
    baseline_gate32 = t0_legality_gate(
        validation_model, trace_qpos[0].astype(np.float32).astype(np.float64), reference32,
        {"candidate": "accepted_A_runtime_float32"}, init_protocol, meshes,
    )
    if not baseline_gate64["passed"] or not baseline_gate32["passed"]:
        raise RuntimeError("reused t0 legality gate contradicts accepted A")

    selection_cfg = solver_cfg["candidate_selection"]
    guard = float(selection_cfg["position_numerical_guard_m"])
    merit_guard = float(selection_cfg["merit_numerical_guard_relative"]) * max(
        1.0, abs(float(baseline["weighted_task_merit"]))
    )
    order = sorted(range(1, len(states)), key=lambda i: (states[i]["weighted_task_merit"], i))
    validations = []
    selected = None
    for index in order[: int(solver_cfg["max_noninitial_geometry_checks"])]:
        row = states[index]
        cheap = {
            "lower_weighted_task_merit": row["weighted_task_merit"] < baseline["weighted_task_merit"] - merit_guard,
            "lower_tip_position_rms": row["tip_position_rms_m"] < baseline["tip_position_rms_m"] - guard,
            "ring_nonworsening": row["tip_position_error_m"]["ring"] <= baseline["tip_position_error_m"]["ring"] + guard,
            "pinky_nonworsening": row["tip_position_error_m"]["pinky"] <= baseline["tip_position_error_m"]["pinky"] + guard,
        }
        validation: dict[str, Any] = {"iteration": index, "static_checks": cheap}
        if all(cheap.values()):
            q64 = trace_qpos[index]
            q32 = q64.astype(np.float32).astype(np.float64)
            gate64 = t0_legality_gate(
                validation_model, q64, reference, {"candidate": "left_aligned_float64"}, init_protocol, meshes
            )
            gate32 = t0_legality_gate(
                validation_model, q32, reference32, {"candidate": "left_aligned_runtime_float32"}, init_protocol, meshes
            )
            validation["float64_gate"] = gate64
            validation["runtime_float32_gate"] = gate32
            validation["full_geometry_passed"] = bool(gate64["passed"] and gate32["passed"])
            if validation["full_geometry_passed"]:
                selected = index
                validations.append(validation)
                break
        else:
            validation["full_geometry_passed"] = False
            validation["geometry_not_run_reason"] = "static_selection_preconditions_failed"
        validations.append(validation)

    np.savez_compressed(
        output / "solver_trace.npz",
        qpos=trace_qpos,
        weighted_task_merit=np.asarray([row["weighted_task_merit"] for row in states]),
        tip_position_error_m=_tip_arrays(states, "tip_position_error_m"),
        tip_orientation_error_rad=_tip_arrays(states, "tip_orientation_error_rad"),
        wrist_orientation_error_rad=np.asarray([row["wrist_orientation_error_rad"] for row in states]),
        minimum_declared_distance_m=np.asarray([row["minimum_declared_distance_m"] for row in states]),
    )
    budget = json.loads((output / "budget_ledger.json").read_text())
    budget["qp_calls"] = int(report["qp_calls"])
    budget["geometry_checks"] = int(sum("float64_gate" in row for row in validations) * 2 + 2)
    budget["status"] = "static_solve_complete"
    write_json(output / "budget_ledger.json", budget)
    write_json(output / "cost_report.json", {
        "objective": report["objective_provenance"],
        "seed_metrics": baseline,
        "states": states,
        "solver": {key: value for key, value in report.items() if key != "states"},
    })
    selection = {
        "schema": "taco_pour_left_initial_alignment_selection_v1",
        "physics_outcomes_used_for_selection": False,
        "baseline_iteration": 0,
        "ordered_noninitial_iterations": order,
        "validations": validations,
        "selected_iteration": selected,
        "status": "SELECTED_LEGAL_DISTINCT_STATE" if selected is not None else "NO_DISTINCT_LEGAL_LEFT_INITIAL_STATE",
    }
    if selected is None:
        write_json(output / "selection.json", selection)
        status["status"] = "NO_DISTINCT_LEGAL_LEFT_INITIAL_STATE"
        status["static_solver"] = selection
        write_json(status_path, status)
        return
    candidate = trace_qpos[selected]
    if not np.array_equal(candidate[report["locked_qpos_addresses"]], initial["qpos"][report["locked_qpos_addresses"]]):
        raise RuntimeError("selected candidate changed right hand or object coordinates")
    np.savez_compressed(
        output / "initial_left_aligned.npz",
        qpos=candidate,
        qvel=initial["qvel"],
        ctrl=np.concatenate([initial["ctrl"][:18], candidate[18:36]]),
        reference_index=np.asarray(0, dtype=np.int64),
    )
    selection["selected_metrics"] = states[selected]
    selection["selected_state"] = trace.artifact(output / "initial_left_aligned.npz")
    selection["right_hand_qpos_byte_identical"] = bool(np.array_equal(candidate[:18], initial["qpos"][:18]))
    selection["objects_qpos_byte_identical"] = bool(np.array_equal(candidate[36:], initial["qpos"][36:]))
    control_delta = candidate[:36] - reference["ctrl"][1]
    control_rows = []
    for actuator_id in range(validation_model.nu):
        joint_id = int(validation_model.actuator_trnid[actuator_id, 0])
        joint_type = int(validation_model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            unit = "m"
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            unit = "rad"
        else:
            raise RuntimeError("initial hand control maps to a non-scalar joint")
        control_rows.append({
            "actuator": validation_model.actuator(actuator_id).name,
            "joint": validation_model.joint(joint_id).name,
            "side": "right" if actuator_id < 18 else "left",
            "unit": unit,
            "candidate_initial_ctrl": float(candidate[actuator_id]),
            "reference_ctrl_endpoint1": float(reference["ctrl"][1, actuator_id]),
            "delta": float(control_delta[actuator_id]),
        })
    selection["initial_control_to_reference_endpoint1"] = {
        "per_actuator": control_rows,
        "left_translation_max_abs_m": float(np.max(np.abs(control_delta[18:21]))),
        "left_angular_max_abs_rad": float(np.max(np.abs(control_delta[21:36]))),
    }
    write_json(output / "selection.json", selection)
    status["status"] = "static_candidate_frozen_pending_physics"
    status["selected_iteration"] = selected
    status["selected_state_sha256"] = selection["selected_state"]["sha256"]
    write_json(status_path, status)


def recover_selection(output: Path) -> None:
    """Recover a frozen selection after the known final JSON failure."""
    require_clean()
    require_baseline()
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    resolved_path = output / "resolved_inputs.json"
    resolved = json.loads(resolved_path.read_text())
    if resolved["implementation_git_commit"] != STATIC_SOLVE_SERIALIZATION_FAILURE_COMMIT:
        raise RuntimeError("recovery is only valid for the recorded serialization failure")
    if status.get("status") != "preflight_complete_no_solver_or_physics_executed":
        raise RuntimeError("static selection recovery requires the untouched failed status")
    required = ("solver_trace.npz", "cost_report.json", "initial_left_aligned.npz")
    if any(not (output / name).is_file() for name in required) or (output / "selection.json").exists():
        raise RuntimeError("static recovery artifacts are incomplete or selection already exists")
    budget = json.loads((output / "budget_ledger.json").read_text())
    if budget.get("qp_calls") != 128 or budget.get("geometry_checks") != 244:
        raise RuntimeError("failed static attempt does not have the recorded bounded ledger")

    paths = input_paths()
    for name, row in resolved["inputs"].items():
        if name == "runner_source":
            continue
        if trace.artifact(paths[name]) != row:
            raise RuntimeError(f"input changed before static recovery: {name}")
    solver_trace = arrays(output / "solver_trace.npz")
    candidate_state = arrays(output / "initial_left_aligned.npz")
    report = json.loads((output / "cost_report.json").read_text())
    trace_qpos = solver_trace["qpos"]
    hits = np.flatnonzero(np.all(trace_qpos == candidate_state["qpos"][None, :], axis=1))
    if len(hits) != 1 or int(hits[0]) == 0:
        raise RuntimeError("frozen candidate is not one distinct solver-trace state")
    selected = int(hits[0])
    states = report["states"]
    order = sorted(range(1, len(states)), key=lambda i: (states[i]["weighted_task_merit"], i))
    rank = order.index(selected)
    if 2 + 2 * (rank + 1) != budget["geometry_checks"]:
        raise RuntimeError("frozen candidate rank contradicts the persisted geometry-check count")
    baseline = states[0]
    candidate_metrics = states[selected]
    contract = yaml.safe_load(CONFIG.read_text())
    selection_cfg = contract["static_solver"]["candidate_selection"]
    guard = float(selection_cfg["position_numerical_guard_m"])
    merit_guard = float(selection_cfg["merit_numerical_guard_relative"]) * max(
        1.0, abs(float(baseline["weighted_task_merit"]))
    )
    static_checks = {
        "lower_weighted_task_merit": candidate_metrics["weighted_task_merit"] < baseline["weighted_task_merit"] - merit_guard,
        "lower_tip_position_rms": candidate_metrics["tip_position_rms_m"] < baseline["tip_position_rms_m"] - guard,
        "ring_nonworsening": candidate_metrics["tip_position_error_m"]["ring"] <= baseline["tip_position_error_m"]["ring"] + guard,
        "pinky_nonworsening": candidate_metrics["tip_position_error_m"]["pinky"] <= baseline["tip_position_error_m"]["pinky"] + guard,
    }
    if not all(static_checks.values()):
        raise RuntimeError("frozen candidate no longer satisfies static selection guards")

    initial_protocol = yaml.safe_load(INITIAL_PROTOCOL.read_text())
    reference = arrays(paths["reference"])
    validation_model = _compile(paths)
    meshes, _ = visual_meshes(paths["scene"], validation_model)
    q64 = candidate_state["qpos"]
    reference32 = {key: value.copy() for key, value in reference.items()}
    reference32["qpos"] = reference32["qpos"].astype(np.float32).astype(np.float64)
    gate64 = t0_legality_gate(
        validation_model, q64, reference,
        {"candidate": "left_aligned_recovery_float64"}, initial_protocol, meshes,
    )
    gate32 = t0_legality_gate(
        validation_model, q64.astype(np.float32).astype(np.float64), reference32,
        {"candidate": "left_aligned_recovery_runtime_float32"}, initial_protocol, meshes,
    )
    if not gate64["passed"] or not gate32["passed"]:
        raise RuntimeError("frozen candidate did not reproduce its legality result")

    initial = trace.load_initial(paths)
    if not np.array_equal(q64[:18], initial["qpos"][:18]) or not np.array_equal(q64[36:], initial["qpos"][36:]):
        raise RuntimeError("recovered candidate changed right hand or objects")
    control_delta = q64[:36] - reference["ctrl"][1]
    control_rows = []
    for actuator_id in range(validation_model.nu):
        joint_id = int(validation_model.actuator_trnid[actuator_id, 0])
        joint_type = int(validation_model.jnt_type[joint_id])
        unit = "m" if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE) else "rad"
        control_rows.append({
            "actuator": validation_model.actuator(actuator_id).name,
            "joint": validation_model.joint(joint_id).name,
            "side": "right" if actuator_id < 18 else "left",
            "unit": unit,
            "candidate_initial_ctrl": float(q64[actuator_id]),
            "reference_ctrl_endpoint1": float(reference["ctrl"][1, actuator_id]),
            "delta": float(control_delta[actuator_id]),
        })
    selected_artifact = trace.artifact(output / "initial_left_aligned.npz")
    selection = {
        "schema": "taco_pour_left_initial_alignment_selection_v1",
        "status": "SELECTED_LEGAL_DISTINCT_STATE",
        "physics_outcomes_used_for_selection": False,
        "baseline_iteration": 0,
        "selected_iteration": selected,
        "selected_rank_zero_based": rank,
        "ordered_noninitial_iterations": order,
        "selected_metrics": candidate_metrics,
        "selected_state": selected_artifact,
        "right_hand_qpos_byte_identical": True,
        "objects_qpos_byte_identical": True,
        "validations": [{
            "iteration": selected,
            "static_checks": static_checks,
            "float64_gate": gate64,
            "runtime_float32_gate": gate32,
            "full_geometry_passed": True,
            "evidence_recovered_after_json_serialization_failure": True,
        }],
        "selection_recovery": {
            "static_solver_implementation_git_commit": STATIC_SOLVE_SERIALIZATION_FAILURE_COMMIT,
            "recovery_implementation_git_commit": head(),
            "mink_rerun": False,
            "ranking_rerun": False,
            "physics_executed_before_freeze": False,
            "preceding_ranked_states_rejected_before_serialization_failure": rank,
            "original_geometry_checks": budget["geometry_checks"],
            "recovery_geometry_checks": 2,
            "lost_detail": "per-state rejected gate payloads were not durable before the JSON failure",
        },
        "initial_control_to_reference_endpoint1": {
            "per_actuator": control_rows,
            "left_translation_max_abs_m": float(np.max(np.abs(control_delta[18:21]))),
            "left_angular_max_abs_rad": float(np.max(np.abs(control_delta[21:36]))),
        },
    }
    write_json(output / "selection.json", selection)
    budget["recovery_geometry_checks"] = 2
    budget["total_geometry_checks_all_attempts"] = budget["geometry_checks"] + 2
    budget["status"] = "static_solve_and_serialization_recovery_complete"
    write_json(output / "budget_ledger.json", budget)
    resolved["preflight_implementation_git_commit"] = resolved["implementation_git_commit"]
    resolved["implementation_git_commit"] = head()
    resolved["inputs"]["runner_source"] = trace.artifact(paths["runner_source"])
    write_json(resolved_path, resolved)
    status.update({
        "status": "static_candidate_frozen_pending_physics",
        "static_solver_implementation_git_commit": STATIC_SOLVE_SERIALIZATION_FAILURE_COMMIT,
        "implementation_git_commit": head(),
        "selected_iteration": selected,
        "selected_state_sha256": selected_artifact["sha256"],
        "selection_recovered_without_mink_or_physics_rerun": True,
    })
    write_json(status_path, status)


def _candidate_initial(output: Path) -> dict[str, np.ndarray]:
    values = arrays(output / "initial_left_aligned.npz")
    return {name: values[name] for name in ("qpos", "qvel", "ctrl")}


def _controls(world, candidate: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = world.ctrl_ref[:21].detach().cpu().numpy().astype(np.float32)
    initial_ctrl = candidate["ctrl"].astype(np.float32)
    plan = startup_control_sequences(initial_ctrl, reference)
    zero = plan["original_residual_action"]
    if not np.array_equal(zero, np.zeros_like(zero)):
        raise RuntimeError("Replay action is not zero residual")
    hold = plan["hold_1_residual_action"]
    support_ok = bool(np.isfinite(hold).all() and np.max(np.abs(hold)) <= 1.0)
    result = {
        "replay_action_sha256": trace._array_sha256(zero),
        "replay_ctrl_sha256": trace._array_sha256(reference[1:21]),
        "hold_legally_representable": support_ok,
        "hold_maximum_absolute_normalized_action": float(np.max(np.abs(hold))),
    }
    realized_hold = None
    if support_ok:
        actual = world._apply_residual(
            torch.as_tensor(reference[1:2]), torch.as_tensor(hold)
        ).detach().cpu().numpy()
        error = float(np.max(np.abs(actual.astype(np.float64) - initial_ctrl[None].astype(np.float64))))
        support_ok = error <= 2e-7
        result["hold_maximum_encoding_error"] = error
        result["hold_legally_representable"] = support_ok
        if support_ok:
            realized_hold = actual.astype(np.float32)
    return {
        "replay_action": zero,
        "replay_ctrl": reference[1:21],
        "hold_action": hold if support_ok else np.empty((0, 36), dtype=np.float32),
        "hold_ctrl": realized_hold if support_ok else np.empty((0, 36), dtype=np.float32),
    }, result


def physics(output: Path) -> None:
    require_clean()
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "static_candidate_frozen_pending_physics":
        raise RuntimeError("physics requires one frozen legal candidate")
    if status["implementation_git_commit"] != head():
        raise RuntimeError("implementation changed after preflight")
    paths = input_paths()
    resolved = json.loads((output / "resolved_inputs.json").read_text())
    for name, row in resolved["inputs"].items():
        if trace.artifact(paths[name]) != row:
            raise RuntimeError(f"input changed before physics: {name}")
    candidate = _candidate_initial(output)
    original_initial = trace.load_initial(paths)
    original_s0 = trace.torch_gzip_read(paths["trace_s0_reconstructed"])
    ledger = BudgetLedger(limit_physics_steps=1000, limit_control_intervals=81)
    ledger.require_capacity(physics_steps=612, control_intervals=61)

    world = trace.make_world(paths, candidate)
    ledger.charge(physics_steps=1, control_intervals=0)
    candidate_s0 = world.get_env_state()
    trace.torch_gzip_write(output / "s0_left_aligned.pt.gz", candidate_s0)
    controls, control_report = _controls(world, candidate)
    np.savez_compressed(output / "control_sequences.npz", **controls)
    write_json(output / "control_validation.json", control_report)

    primitive = {
        "right_qpos_equal": bool(np.array_equal(
            candidate_s0["qpos"].cpu().numpy()[0, :18], original_s0["qpos"].cpu().numpy()[0, :18]
        )),
        "object_qpos_equal": bool(np.array_equal(
            candidate_s0["qpos"].cpu().numpy()[0, 36:], original_s0["qpos"].cpu().numpy()[0, 36:]
        )),
        "all_qvel_equal": bool(np.array_equal(candidate_s0["qvel"].cpu().numpy(), original_s0["qvel"].cpu().numpy())),
        "right_ctrl_equal": bool(np.array_equal(
            candidate_s0["ctrl"].cpu().numpy()[0, :18], original_s0["ctrl"].cpu().numpy()[0, :18]
        )),
        "left_ctrl_equals_candidate_left_qpos": bool(np.array_equal(
            candidate_s0["ctrl"].cpu().numpy()[0, 18:], candidate_s0["qpos"].cpu().numpy()[0, 18:36]
        )),
    }
    if not all(primitive.values()):
        raise RuntimeError(f"new s0 primitive invariants failed: {primitive}")
    snapshot_delta = trace.compare_snapshots(candidate_s0, original_s0)
    write_json(output / "s0_rebuild_report.json", {
        "primitive_invariants": primitive,
        "complete_snapshot_field_count": len(candidate_s0),
        "changed_fields": sorted(snapshot_delta["unequal_fields"]),
        "derived_fields_expected_to_change": True,
        "old_contact_or_solver_cache_reused": False,
    })

    zero = controls["replay_action"]
    expected = controls["replay_ctrl"]
    original = trace._run_startup_condition(world, original_s0, "ORIGINAL", zero, expected, ledger)
    trace._save_condition(output, "ORIGINAL", *original)
    parity = trace._historical_original_parity(output, paths, *original[:3])
    if not parity["all_bitwise_equal"]:
        status["status"] = "original_identity_failed_stopped"
        write_json(status_path, status)
        raise RuntimeError("ORIGINAL no longer reproduces historical Replay")

    if control_report["hold_legally_representable"]:
        hold = trace._run_startup_condition(
            world, candidate_s0, "LEFT_ALIGNED_HOLD_1",
            controls["hold_action"], controls["hold_ctrl"], ledger,
        )
        trace._save_condition(output, "LEFT_ALIGNED_HOLD_1", *hold)
    else:
        hold = None
        write_json(output / "conditions/LEFT_ALIGNED_HOLD_1/result.json", {
            "condition": "LEFT_ALIGNED_HOLD_1", "status": "SKIPPED",
            "reason": "initial control cannot be represented without clipping",
            "physics_steps": 0,
        })

    replay = trace._run_startup_condition(
        world, candidate_s0, "LEFT_ALIGNED_REPLAY", zero, expected, ledger,
    )
    trace._save_condition(output, "LEFT_ALIGNED_REPLAY", *replay)

    cold_world = trace.make_world(paths, candidate)
    ledger.charge(physics_steps=1, control_intervals=0)
    cold = trace._run_startup_condition(
        cold_world, candidate_s0, "LEFT_ALIGNED_COLD", zero, expected, ledger,
    )
    trace._save_condition(output, "LEFT_ALIGNED_COLD", *cold)
    replay_rows, replay_contacts, replay_efc = trace._observer_arrays(replay[2])
    cold_rows, cold_contacts, cold_efc = trace._observer_arrays(cold[2])
    cold_parity = {
        "endpoints": trace._compare_array_maps(replay[0], cold[0]),
        "substeps": trace._compare_array_maps(replay_rows, cold_rows),
        "contacts": trace._compare_array_maps(replay_contacts, cold_contacts),
        "efc_force": trace._compare_array_maps(replay_efc, cold_efc),
        "full_terminal_snapshot": trace.compare_snapshots(
            replay[1][max(replay[1])], cold[1][max(cold[1])]
        ),
    }
    cold_parity["all_bitwise_equal"] = bool(
        cold_parity["endpoints"]["all_equal"]
        and cold_parity["substeps"]["all_equal"]
        and cold_parity["contacts"]["all_equal"]
        and cold_parity["efc_force"]["all_equal"]
        and cold_parity["full_terminal_snapshot"]["all_common_equal"]
        and not cold_parity["full_terminal_snapshot"]["only_current"]
        and not cold_parity["full_terminal_snapshot"]["only_historical"]
    )
    write_json(output / "cold_replay_parity.json", cold_parity)
    if not cold_parity["all_bitwise_equal"]:
        raise RuntimeError("new candidate cold replay is not bitwise deterministic")

    budget = json.loads((output / "budget_ledger.json").read_text())
    budget.update({
        "status": "physics_complete",
        "task_control_intervals": ledger.control_intervals,
        "task_physics_steps": ledger.control_intervals * 10,
        "setup_physics_steps": ledger.physics_steps - ledger.control_intervals * 10,
        "all_in_physics_steps": ledger.physics_steps,
        "actor_or_critic_forwards": 0,
        "optimizer_updates": 0,
    })
    write_json(output / "budget_ledger.json", budget)
    status["status"] = "physics_complete_analysis_pending"
    status["conditions"] = {
        "ORIGINAL": original[3],
        "LEFT_ALIGNED_HOLD_1": hold[3] if hold is not None else {"status": "SKIPPED"},
        "LEFT_ALIGNED_REPLAY": replay[3],
        "LEFT_ALIGNED_COLD": cold[3],
    }
    write_json(status_path, status)


def _pose(qpos: np.ndarray, offset: int) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = qpos[offset : offset + 3]
    quat = qpos[offset + 3 : offset + 7]
    transform[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return transform


def _metrics(qpos: np.ndarray, qvel: np.ndarray, reference: np.ndarray, s0: np.ndarray) -> dict[str, Any]:
    base = trace._object_metrics(qpos, reference)
    relative = np.linalg.inv(_pose(qpos, 43)) @ _pose(qpos, 36)
    relative_ref = np.linalg.inv(_pose(reference, 43)) @ _pose(reference, 36)
    base.update({
        "tool_displacement_from_s0_m": float(np.linalg.norm(qpos[36:39] - s0[36:39])),
        "tool_rotation_from_s0_rad": trace.rotation_error(qpos[39:43], s0[39:43]),
        "tool_linear_speed_m_s": float(np.linalg.norm(qvel[36:39])),
        "tool_angular_speed_rad_s": float(np.linalg.norm(qvel[39:42])),
        "target_displacement_from_s0_m": float(np.linalg.norm(qpos[43:46] - s0[43:46])),
        "target_rotation_from_s0_rad": trace.rotation_error(qpos[46:50], s0[46:50]),
        "target_linear_speed_m_s": float(np.linalg.norm(qvel[42:45])),
        "target_angular_speed_rad_s": float(np.linalg.norm(qvel[45:48])),
        "object_frame_pair_translation_error_m": float(np.linalg.norm(relative[:3, 3] - relative_ref[:3, 3])),
        "object_frame_pair_rotation_error_rad": float(Rotation.from_matrix(relative_ref[:3, :3].T @ relative[:3, :3]).magnitude()),
        "right_wrist_translation_tracking_rms_m": float(np.sqrt(np.mean(np.square(qpos[0:3] - reference[0:3])))),
        "right_angular_tracking_rms_rad": float(np.sqrt(np.mean(np.square(qpos[3:18] - reference[3:18])))),
        "left_wrist_translation_tracking_rms_m": float(np.sqrt(np.mean(np.square(qpos[18:21] - reference[18:21])))),
        "left_angular_tracking_rms_rad": float(np.sqrt(np.mean(np.square(qpos[21:36] - reference[21:36])))),
    })
    return base


def analyze(output: Path) -> None:
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") not in (
        "physics_complete_analysis_pending",
        "physics_and_analysis_complete_visual_review_pending",
    ):
        raise RuntimeError("analysis requires completed physics")
    paths = input_paths()
    reference = arrays(paths["reference"])
    labels = ("ORIGINAL", "LEFT_ALIGNED_REPLAY", "LEFT_ALIGNED_COLD")
    conditions = {
        label: arrays(output / "conditions" / label / "endpoints.npz") for label in labels
    }
    contacts = {
        label: arrays(output / "conditions" / label / "contacts_raw.npz") for label in labels
    }
    hold_label = "LEFT_ALIGNED_HOLD_1"
    hold_endpoints = output / "conditions" / hold_label / "endpoints.npz"
    if hold_endpoints.is_file():
        conditions[hold_label] = arrays(hold_endpoints)
        contacts[hold_label] = arrays(output / "conditions" / hold_label / "contacts_raw.npz")
    rows = []
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    for label in labels:
        endpoint_metrics[label] = {}
        s0 = conditions[label]["qpos"][0]
        for endpoint in range(21):
            if endpoint >= len(conditions[label]["qpos"]):
                rows.append({"condition": label, "endpoint": endpoint, "availability": "N/A"})
                endpoint_metrics[label][str(endpoint)] = None
                continue
            metric = _metrics(
                conditions[label]["qpos"][endpoint], conditions[label]["qvel"][endpoint],
                reference["qpos"][endpoint], s0,
            )
            endpoint_metrics[label][str(endpoint)] = metric
            rows.append({"condition": label, "endpoint": endpoint, "availability": "measured", **metric})
    fields = sorted({key for row in rows for key in row})
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    pair_specs = {
        "left_ring_target": ("left_hand:ring", "target", "target"),
        "left_pinky_target": ("left_hand:pinky", "target", "target"),
        "target_floor": ("target", "floor", "target"),
    }
    contact_summary = {
        label: {name: trace._pair_contact_stats(contacts[label], *spec) for name, spec in pair_specs.items()}
        for label in contacts
    }
    original_hold = {
        "endpoints": trace.artifact(paths["startup_hold_endpoints"]),
        "substeps": trace.artifact(paths["startup_hold_substeps"]),
        "contacts": trace.artifact(paths["startup_hold_contacts"]),
    }
    historical_hold_arrays = arrays(paths["startup_hold_endpoints"])
    original_hold["endpoint_metrics"] = {
        str(endpoint): _metrics(
            historical_hold_arrays["qpos"][endpoint], historical_hold_arrays["qvel"][endpoint],
            reference["qpos"][endpoint], historical_hold_arrays["qpos"][0],
        )
        for endpoint in range(len(historical_hold_arrays["qpos"]))
    }
    if hold_label in conditions:
        hold_values = conditions[hold_label]
        original_hold["left_aligned_hold_endpoint_metrics"] = {
            str(endpoint): _metrics(
                hold_values["qpos"][endpoint], hold_values["qvel"][endpoint],
                reference["qpos"][endpoint], hold_values["qpos"][0],
            )
            for endpoint in range(len(hold_values["qpos"]))
        }
    else:
        original_hold["left_aligned_hold_endpoint_metrics"] = None
    early = {}
    early_complete = {}
    for label in ("ORIGINAL", "LEFT_ALIGNED_REPLAY"):
        measured = [endpoint_metrics[label][str(i)] for i in range(11)]
        early_complete[label] = all(row is not None for row in measured)
        if early_complete[label]:
            early[label] = {
                "target_position_error_mean_m": float(np.mean([row["target_position_error_m"] for row in measured])),
                "target_position_error_max_m": float(np.max([row["target_position_error_m"] for row in measured])),
                "target_rotation_error_mean_rad": float(np.mean([row["target_rotation_error_rad"] for row in measured])),
                "pair_position_error_mean_m": float(np.mean([row["pair_position_error_m"] for row in measured])),
                "object_frame_pair_translation_error_mean_m": float(np.mean([row["object_frame_pair_translation_error_m"] for row in measured])),
            }
        else:
            early[label] = None
    summary_metrics = {
        label: {str(endpoint): endpoint_metrics[label][str(endpoint)] for endpoint in SUMMARY_ENDPOINTS}
        for label in labels
    }
    candidate_better_early = bool(
        all(early_complete.values())
        and early["ORIGINAL"] is not None
        and early["LEFT_ALIGNED_REPLAY"] is not None
        and
        early["LEFT_ALIGNED_REPLAY"]["target_position_error_mean_m"]
        < early["ORIGINAL"]["target_position_error_mean_m"]
        and early["LEFT_ALIGNED_REPLAY"]["target_rotation_error_mean_rad"]
        <= early["ORIGINAL"]["target_rotation_error_mean_rad"]
        and early["LEFT_ALIGNED_REPLAY"]["object_frame_pair_translation_error_mean_m"]
        < early["ORIGINAL"]["object_frame_pair_translation_error_mean_m"]
    )
    early_components = {
        "target_position_mean_improved": bool(
            all(early_complete.values())
            and early["LEFT_ALIGNED_REPLAY"]["target_position_error_mean_m"]
            < early["ORIGINAL"]["target_position_error_mean_m"]
        ),
        "target_position_max_improved": bool(
            all(early_complete.values())
            and early["LEFT_ALIGNED_REPLAY"]["target_position_error_max_m"]
            < early["ORIGINAL"]["target_position_error_max_m"]
        ),
        "target_rotation_mean_improved": bool(
            all(early_complete.values())
            and early["LEFT_ALIGNED_REPLAY"]["target_rotation_error_mean_rad"]
            < early["ORIGINAL"]["target_rotation_error_mean_rad"]
        ),
        "object_frame_pair_translation_mean_improved": bool(
            all(early_complete.values())
            and early["LEFT_ALIGNED_REPLAY"]["object_frame_pair_translation_error_mean_m"]
            < early["ORIGINAL"]["object_frame_pair_translation_error_mean_m"]
        ),
    }
    original20 = endpoint_metrics["ORIGINAL"]["20"]
    aligned20 = endpoint_metrics["LEFT_ALIGNED_REPLAY"]["20"]
    endpoint20_better = bool(
        original20 is not None and aligned20 is not None
        and aligned20["target_position_error_m"] < original20["target_position_error_m"]
        and aligned20["target_rotation_error_rad"] <= original20["target_rotation_error_rad"]
    )
    conclusion = (
        "INCOMPLETE_PREFIX_REPORTED_WITH_NA"
        if not all(early_complete.values()) else
        "EARLY_AND_ENDPOINT20_IMPROVEMENT_SUPPORTS_LEFT_INITIALIZATION_REDESIGN"
        if candidate_better_early and endpoint20_better else
        "EARLY_LOCAL_EFFECT_NOT_PERSISTENT" if candidate_better_early else
        "MIXED_EARLY_RESPONSE_WITH_ENDPOINT20_POSITION_ROTATION_IMPROVEMENT"
        if endpoint20_better and any(early_components.values()) else
        "REFERENCE_CLOSER_LEFT_INITIALIZATION_DID_NOT_IMPROVE_EARLY_RELATION"
    )
    analysis = {
        "schema": "taco_pour_left_initial_alignment_analysis_v1",
        "analysis_git_commit": head(),
        "endpoint_metrics": endpoint_metrics,
        "summary_endpoints": summary_metrics,
        "early_0_10": early,
        "early_0_10_complete": early_complete,
        "early_0_10_component_comparison": early_components,
        "contact_pairs": contact_summary,
        "historical_original_hold_1": original_hold,
        "conclusion": conclusion,
        "candidate_better_early_joint_relation_gate": candidate_better_early,
        "candidate_better_at_endpoint20_position_and_rotation": endpoint20_better,
        "boundaries": {
            "deterministic_single_candidate": True,
            "not_a_deployable_reset": True,
            "not_full_task_validation": True,
            "no_chunk_commit": True,
        },
    }
    write_json(output / "summary.json", analysis)
    selection = json.loads((output / "selection.json").read_text())
    selected_validation = next(
        (row for row in selection["validations"]
         if row["iteration"] == selection["selected_iteration"]),
        None,
    )
    early_original = early["ORIGINAL"]
    early_aligned = early["LEFT_ALIGNED_REPLAY"]
    findings = [
        "# Left initial alignment v1 findings",
        "",
        f"**Conclusion: `{conclusion}`.**",
        "",
        "The candidate was selected from one deterministic static left-only MINK trajectory before any physics result was observed. Right-hand/object initial primitives, initial velocities, Replay commands, scene and physics were unchanged; left qpos and its consistent initial position target changed together.",
        "",
        f"- Selected solver iteration: `{selection['selected_iteration']}`.",
        "- This is a bounded initialization attribution, not a reset promotion, downstream feasibility result, or RL authorization.",
    ]
    if early_original is not None and early_aligned is not None:
        findings.extend([
            f"- Early target position mean, ORIGINAL → LEFT_ALIGNED: `{early_original['target_position_error_mean_m']:.9f} → {early_aligned['target_position_error_mean_m']:.9f} m`.",
            f"- Early target position maximum: `{early_original['target_position_error_max_m']:.9f} → {early_aligned['target_position_error_max_m']:.9f} m`.",
            f"- Early target rotation mean: `{early_original['target_rotation_error_mean_rad']:.9f} → {early_aligned['target_rotation_error_mean_rad']:.9f} rad`.",
            f"- Early object-frame pair translation mean: `{early_original['object_frame_pair_translation_error_mean_m']:.9f} → {early_aligned['object_frame_pair_translation_error_mean_m']:.9f} m`.",
        ])
    else:
        findings.append("- The 0→10 early window is incomplete; unavailable endpoints are N/A and were not counted as passing.")
    if original20 is not None and aligned20 is not None:
        findings.append(
            f"- Endpoint20 target position error: `{original20['target_position_error_m']:.9f} → {aligned20['target_position_error_m']:.9f} m`."
        )
    else:
        findings.append("- Endpoint20 is unavailable for at least one condition and was not counted as passing.")
    if selected_validation is not None:
        findings.append(
            f"- Selected iteration passed both float64 and runtime-float32 geometry gates: "
            f"`{selected_validation.get('full_geometry_passed', False)}`."
        )
    (output / "findings.md").write_text("\n".join(findings) + "\n")
    status["status"] = "physics_and_analysis_complete_visual_review_pending"
    status["conclusion"] = conclusion
    write_json(status_path, status)


def hashes(output: Path) -> None:
    rows = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows[str(path.relative_to(output))] = trace.sha256(path)
    lines = [f"{digest}  {name}" for name, digest in rows.items()]
    (output / "server_artifacts.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("preflight", "solve", "recover-selection", "physics", "analyze", "hashes"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "preflight": preflight(args.output)
    elif args.phase == "solve": static_solve(args.output)
    elif args.phase == "recover-selection": recover_selection(args.output)
    elif args.phase == "physics": physics(args.output)
    elif args.phase == "analyze": analyze(args.output)
    else: hashes(args.output)


if __name__ == "__main__":
    main()
