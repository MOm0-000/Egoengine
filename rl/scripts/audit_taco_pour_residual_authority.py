"""Read-only audit of residual use, state gap, and actuator control headroom."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts


SATURATION_TOLERANCE = 2e-7
WINDOWS = {
    "full_endpoint_21_to_53": np.arange(33),
    "failure_tail_10_endpoint_44_to_53": np.arange(23, 33),
    "failure_tail_5_endpoint_49_to_53": np.arange(28, 33),
}
GROUPS = {
    "both_wrist_translation": [0, 1, 2, 18, 19, 20],
    "both_wrist_rotation": [3, 4, 5, 21, 22, 23],
    "both_fingers": [*range(6, 18), *range(24, 36)],
    "right_wrist_translation": [0, 1, 2],
    "right_wrist_rotation": [3, 4, 5],
    "right_fingers": [*range(6, 18)],
    "left_wrist_translation": [18, 19, 20],
    "left_wrist_rotation": [21, 22, 23],
    "left_fingers": [*range(24, 36)],
}


def _sha256_record(path: Path) -> dict:
    return artifact(path.resolve(strict=True))


def _absolute_stats(values: np.ndarray) -> dict:
    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("statistics require finite, nonempty values")
    quantiles = np.quantile(values, (0.50, 0.90, 0.95, 0.99))
    return {
        "sample_count": int(values.size),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _plain_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("statistics require finite, nonempty values")
    quantiles = np.quantile(values, (0.00, 0.10, 0.50, 0.90, 1.00))
    return {
        "sample_count": int(values.size),
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def _count(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {
        "component_samples": int(mask.size),
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "steps_with_any": int(mask.any(axis=1).sum()),
        "step_count": int(mask.shape[0]),
        "step_fraction_with_any": float(mask.any(axis=1).mean()),
    }


def _expected_actuators() -> list[str]:
    return [
        *(f"R_forearm_{axis}_position" for axis in ("tx", "ty", "tz", "roll", "pitch", "yaw")),
        "right_thumb_bend_position", "right_thumb_rota1_position", "right_thumb_rota2_position",
        "right_index_bend_position", "right_index_joint1_position", "right_index_joint2_position",
        "right_middle_joint1_position", "right_middle_joint2_position",
        "right_ring_joint1_position", "right_ring_joint2_position",
        "right_pinky_joint1_position", "right_pinky_joint2_position",
        *(f"L_forearm_{axis}_position" for axis in ("tx", "ty", "tz", "roll", "pitch", "yaw")),
        "left_thumb_bend_position", "left_thumb_rota1_position", "left_thumb_rota2_position",
        "left_index_bend_position", "left_index_joint1_position", "left_index_joint2_position",
        "left_middle_joint1_position", "left_middle_joint2_position",
        "left_ring_joint1_position", "left_ring_joint2_position",
        "left_pinky_joint1_position", "left_pinky_joint2_position",
    ]


def _window_stats(
    rows: np.ndarray,
    indices: np.ndarray,
    requested: np.ndarray,
    effective: np.ndarray,
    state_gap: np.ndarray,
    reference_ctrl: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    residual_limit: float,
) -> dict:
    requested = requested[rows][:, indices]
    effective = effective[rows][:, indices]
    state_gap = state_gap[rows][:, indices]
    reference_ctrl = reference_ctrl[rows][:, indices]
    lower = lower[indices]
    upper = upper[indices]
    saturated = np.abs(requested) >= residual_limit - SATURATION_TOLERANCE
    truncated = np.abs(effective - requested) > SATURATION_TOLERANCE
    blocked = (
        (np.abs(requested) > SATURATION_TOLERANCE)
        & (np.abs(effective) <= SATURATION_TOLERANCE)
    )
    negative_headroom = np.maximum(reference_ctrl - lower, 0.0)
    positive_headroom = np.maximum(upper - reference_ctrl, 0.0)
    bidirectional_headroom = np.minimum(negative_headroom, positive_headroom)
    return {
        "endpoints": [int(rows[0] + 21), int(rows[-1] + 21)],
        "requested_residual_absolute": _absolute_stats(requested),
        "requested_residual_at_0p05_limit": _count(saturated),
        "source_qpos_to_next_reference_target_absolute": _absolute_stats(state_gap),
        "residual_limit_over_state_gap_p95": (
            None if np.quantile(np.abs(state_gap), 0.95) == 0.0
            else float(residual_limit / np.quantile(np.abs(state_gap), 0.95))
        ),
        "ctrlrange": {
            "negative_headroom": _plain_stats(negative_headroom),
            "positive_headroom": _plain_stats(positive_headroom),
            "minimum_bidirectional_headroom": _plain_stats(bidirectional_headroom),
            "negative_headroom_below_0p05": _count(
                negative_headroom < residual_limit - SATURATION_TOLERANCE
            ),
            "positive_headroom_below_0p05": _count(
                positive_headroom < residual_limit - SATURATION_TOLERANCE
            ),
        },
        "range_truncated_requested_residual": _count(truncated),
        "fully_blocked_requested_residual": _count(blocked),
        "effective_residual_absolute_after_ctrlrange": _absolute_stats(effective),
        "authority_lost_absolute": _absolute_stats(effective - requested),
    }


def run(source_report_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_report = json.loads(source_report_path.read_text())
    chunks = source_report.get("chunks", [])
    if len(chunks) != 1 or chunks[0].get("start") != 20:
        raise ValueError("expected the frozen endpoint-20 3+1 experiment")
    matches = [row for row in chunks[0]["validation_traces"] if row.get("mode") == "rl"]
    if len(matches) != 1:
        raise ValueError("expected exactly one deterministic CPU PPO trace")
    trace = matches[0]
    steps = trace["steps"]
    endpoints = [int(row["endpoint"]) for row in steps]
    if (
        endpoints != list(range(21, 54))
        or trace.get("validated_steps") != 32
        or trace.get("first_failure", {}).get("endpoint") != 53
    ):
        raise ValueError("source CPU trace is not the frozen 32/40 result")

    snapshots = source_report["input_contract_snapshots"]
    config_path = Path(snapshots["simulator_config"]["snapshot_path"])
    action_profile_path = Path(snapshots["action_profile"]["snapshot_path"])
    boundary_path = Path(snapshots["resume_boundary"]["snapshot_path"])
    for name, path in (
        ("simulator_config", config_path),
        ("action_profile", action_profile_path),
        ("resume_boundary", boundary_path),
    ):
        record = snapshots[name]
        actual = _sha256_record(path)
        if actual["sha256"] != record["sha256"]:
            raise ValueError(f"frozen {name} snapshot hash differs")

    config = yaml.safe_load(config_path.read_text())
    action_profile = yaml.safe_load(action_profile_path.read_text())
    residual_limit = float(action_profile["mapping"]["residual_clip_rad"])
    if residual_limit != 0.05 or float(action_profile["mapping"]["residual_scale"]) != 0.05:
        raise ValueError("expected the frozen 0.05 scaled residual profile")
    scene_path = Path(config["model_path"]).resolve(strict=True)
    reference_path = Path(config["data_path"]).resolve(strict=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    if names != _expected_actuators() or model.nu != 36:
        raise ValueError("actuator layout differs from the audited XHand contract")
    joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int64)
    qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    if not np.array_equal(qpos_addresses, np.arange(36)) or not np.array_equal(model.actuator_gear[:, 0], np.ones(36)):
        raise ValueError("control target and qpos coordinates are not directly comparable")
    if not np.all(model.actuator_ctrllimited):
        raise ValueError("all audited actuators must have explicit control ranges")
    lower = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
    upper = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)

    boundary_raw = gzip.decompress(boundary_path.read_bytes())
    if source_report["incoming_boundary"]["uncompressed_pt_sha256"] != __import__("hashlib").sha256(boundary_raw).hexdigest():
        raise ValueError("uncompressed endpoint-20 boundary hash differs")
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    if (
        boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v2"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
        or tuple(boundary["qpos"].shape) != (1, 50)
    ):
        raise ValueError("invalid endpoint-20 complete boundary")

    with np.load(reference_path, allow_pickle=False) as arrays:
        ctrl_reference = np.asarray(arrays["ctrl"], dtype=np.float64)
    if ctrl_reference.shape != (198, 36):
        raise ValueError("unexpected formal reference shape")
    requested = np.asarray([row["applied_residual"] for row in steps], dtype=np.float64)
    commanded = np.asarray([row["commanded_ctrl"] for row in steps], dtype=np.float64)
    runtime_reference = commanded - requested
    frozen_reference = ctrl_reference[np.asarray(endpoints)]
    reference_match_max = float(np.max(np.abs(runtime_reference - frozen_reference)))
    if reference_match_max > SATURATION_TOLERANCE:
        raise ValueError("trace reference target differs from the frozen reference")
    source_qpos = np.vstack([
        boundary["qpos"][0, :36].detach().cpu().numpy(),
        *[np.asarray(row["endpoint_qpos"][:36], dtype=np.float64) for row in steps[:-1]],
    ]).astype(np.float64)
    if source_qpos.shape != (33, 36):
        raise ValueError("failed to reconstruct source states for every trace step")
    state_gap = runtime_reference - source_qpos

    clipped_reference = np.clip(runtime_reference, lower, upper)
    reference_range_violation = np.maximum(lower - runtime_reference, 0.0) + np.maximum(runtime_reference - upper, 0.0)
    if float(reference_range_violation.max()) > SATURATION_TOLERANCE:
        raise ValueError("reference target meaningfully violates actuator ctrlrange")
    effective_command = np.clip(commanded, lower, upper)
    effective = effective_command - clipped_reference

    group_windows = {
        window: {
            group: _window_stats(
                rows, np.asarray(indices), requested, effective, state_gap,
                runtime_reference, lower, upper, residual_limit,
            )
            for group, indices in GROUPS.items()
        }
        for window, rows in WINDOWS.items()
    }

    coordinate_audit = []
    for index, name in enumerate(names):
        unit = "m" if index in GROUPS["both_wrist_translation"] else "rad"
        coordinate_audit.append({
            "index": index,
            "actuator": name,
            "unit": unit,
            "ctrlrange": [float(lower[index]), float(upper[index])],
            "windows": {
                window: _window_stats(
                    rows, np.asarray([index]), requested, effective, state_gap,
                    runtime_reference, lower, upper, residual_limit,
                )
                for window, rows in WINDOWS.items()
            },
        })

    step_audit = []
    for row_index, source_row in enumerate(steps):
        groups = {}
        for group in ("both_wrist_translation", "both_wrist_rotation", "both_fingers"):
            indices = np.asarray(GROUPS[group])
            saturation = np.abs(requested[row_index, indices]) >= residual_limit - SATURATION_TOLERANCE
            truncation = np.abs(effective[row_index, indices] - requested[row_index, indices]) > SATURATION_TOLERANCE
            groups[group] = {
                "component_count": int(len(indices)),
                "saturated_components": int(saturation.sum()),
                "range_truncated_components": int(truncation.sum()),
                "requested_residual_abs_max": float(np.abs(requested[row_index, indices]).max()),
                "source_qpos_to_next_reference_abs_p95": float(np.quantile(np.abs(state_gap[row_index, indices]), 0.95)),
                "source_qpos_to_next_reference_abs_max": float(np.abs(state_gap[row_index, indices]).max()),
            }
        step_audit.append({
            "source_endpoint": int(source_row["control_interval"]),
            "outcome_endpoint": int(source_row["endpoint"]),
            "objective_score": float(source_row["objective_score"][0]),
            "terminated": bool(source_row["terminated"]),
            "groups": groups,
        })

    full = group_windows["full_endpoint_21_to_53"]
    tail = group_windows["failure_tail_10_endpoint_44_to_53"]
    report = {
        "schema": "taco_pour_residual_authority_audit_v1",
        "status": "all_groups_use_limit_finger_ctrlrange_further_truncates_no_new_scale_selected",
        "scope": {
            "source": "frozen deterministic CPU PPO validation trace from the 3+1 experiment",
            "training_executed": False,
            "policy_inference_executed": False,
            "physics_rollout_executed": False,
            "simulator_environments_created": 0,
            "source_trace_modified": False,
            "new_action_scale_selected": False,
            "next_training_authorized": False,
        },
        "trace_contract": {
            "source_endpoint": 20,
            "last_outcome_endpoint": 53,
            "step_count": 33,
            "validated_steps_before_failure": 32,
            "first_failure_endpoint": 53,
            "first_failure_score": float(steps[-1]["objective_score"][0]),
            "runtime_reference_matches_npz_max_abs": reference_match_max,
            "source_qpos_provenance": "endpoint-20 complete boundary followed by the prior recorded CPU endpoint qpos",
            "next_reference_target_provenance": "commanded_ctrl minus recorded requested applied_residual, checked against robot_reference ctrl[outcome_endpoint]",
        },
        "measurement_contract": {
            "requested_residual": "recorded residual added to the next reference target before actuator ctrlrange",
            "policy_limit": residual_limit,
            "saturation_tolerance": SATURATION_TOLERANCE,
            "effective_residual": "clip(commanded_ctrl, ctrlrange) minus clip(reference_ctrl, ctrlrange)",
            "state_gap": "next reference control target minus current source-state actuated qpos",
            "state_gap_is_required_optimal_residual": False,
            "ctrlrange_headroom": "distance from next reference target to each actuator control bound",
        },
        "headline": {
            "full_requested_saturation_fraction": {
                "wrist_translation": full["both_wrist_translation"]["requested_residual_at_0p05_limit"]["fraction"],
                "wrist_rotation": full["both_wrist_rotation"]["requested_residual_at_0p05_limit"]["fraction"],
                "fingers": full["both_fingers"]["requested_residual_at_0p05_limit"]["fraction"],
            },
            "tail10_requested_saturation_fraction": {
                "wrist_translation": tail["both_wrist_translation"]["requested_residual_at_0p05_limit"]["fraction"],
                "wrist_rotation": tail["both_wrist_rotation"]["requested_residual_at_0p05_limit"]["fraction"],
                "fingers": tail["both_fingers"]["requested_residual_at_0p05_limit"]["fraction"],
            },
            "full_source_qpos_to_next_reference_p95": {
                "wrist_translation_m": full["both_wrist_translation"]["source_qpos_to_next_reference_target_absolute"]["p95"],
                "wrist_rotation_rad": full["both_wrist_rotation"]["source_qpos_to_next_reference_target_absolute"]["p95"],
                "fingers_rad": full["both_fingers"]["source_qpos_to_next_reference_target_absolute"]["p95"],
            },
            "tail10_source_qpos_to_next_reference_p95": {
                "wrist_translation_m": tail["both_wrist_translation"]["source_qpos_to_next_reference_target_absolute"]["p95"],
                "wrist_rotation_rad": tail["both_wrist_rotation"]["source_qpos_to_next_reference_target_absolute"]["p95"],
                "fingers_rad": tail["both_fingers"]["source_qpos_to_next_reference_target_absolute"]["p95"],
            },
            "full_range_truncated_components": {
                "wrist_translation": full["both_wrist_translation"]["range_truncated_requested_residual"]["count"],
                "wrist_rotation": full["both_wrist_rotation"]["range_truncated_requested_residual"]["count"],
                "fingers": full["both_fingers"]["range_truncated_requested_residual"]["count"],
            },
            "finger_steps_with_range_truncation": full["both_fingers"]["range_truncated_requested_residual"]["steps_with_any"],
            "finger_fully_blocked_components": full["both_fingers"]["fully_blocked_requested_residual"]["count"],
        },
        "decision": {
            "case_A_angular_limited_while_translation_rarely_limited_supported": False,
            "case_B_no_category_materially_uses_limit_supported": False,
            "finding": "all three categories materially use the 0.05 limit; finger commands are additionally restricted by actuator ctrlrange",
            "uniform_mixed_unit_scale_is_clean_design": False,
            "mixed_unit_scale_proven_as_endpoint53_primary_cause": False,
            "increasing_only_angular_scale_supported": False,
            "increasing_finger_scale_without_ctrlrange_handling_supported": False,
            "new_group_scale_values_selected": False,
            "next_training_authorized": False,
        },
        "group_windows": group_windows,
        "coordinate_audit": coordinate_audit,
        "step_audit": step_audit,
        "actuator_contract": {
            "count": 36,
            "names": names,
            "qpos_addresses": qpos_addresses.tolist(),
            "position_transmission_gear": model.actuator_gear[:, 0].tolist(),
            "all_ctrl_limited": bool(np.all(model.actuator_ctrllimited)),
            "reference_max_ctrlrange_numerical_excess": float(reference_range_violation.max()),
        },
        "limitations": [
            "The recorded residual is the requested command offset before actuator ctrlrange; the effective value here is the model-defined clipped target offset.",
            "Current qpos-to-next-reference target error includes servo lag, contact constraints, and reference motion; it is not an optimal residual label.",
            "Saturation proves that a bound is active, not that increasing it would improve object tracking.",
            "This single deterministic CPU trajectory cannot establish the primary cause of endpoint-53 failure.",
        ],
        "preserved_inputs": [
            _sha256_record(source_report_path),
            _sha256_record(config_path),
            _sha256_record(action_profile_path),
            _sha256_record(boundary_path),
            _sha256_record(scene_path),
            _sha256_record(reference_path),
        ],
        "audit_code": [_sha256_record(Path(__file__))],
    }
    verify_artifacts(report["preserved_inputs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        type=Path,
        default=ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_residual_authority_audit_v1/report.json",
    )
    args = parser.parse_args()
    report = run(args.source_report.resolve(), args.output.resolve())
    print(json.dumps(report["headline"], indent=2))


if __name__ == "__main__":
    main()
