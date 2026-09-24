"""Describe ctrlrange loss in one frozen, non-promotable 3+1 PPO run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 2e-7
FINGER_INDICES = np.asarray([*range(6, 18), *range(24, 36)], dtype=np.int64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    bottom = float(np.abs(denominator).sum())
    return None if bottom == 0.0 else float(np.abs(numerator).sum() / bottom)


def _scope(
    selected: np.ndarray,
    requested: np.ndarray,
    effective: np.ndarray,
    lost: np.ndarray,
    *,
    residual_limit: float,
) -> dict:
    req = requested[selected]
    eff = effective[selected]
    loss = lost[selected]
    truncated = np.abs(loss) > TOLERANCE
    requested_nonzero = np.abs(req) > TOLERANCE
    blocked = requested_nonzero & (np.abs(eff) <= TOLERANCE)
    at_limit = np.abs(req) >= residual_limit - TOLERANCE
    finger_truncated = truncated[:, FINGER_INDICES]
    return {
        "world_step_samples": int(selected.sum()),
        "component_samples": int(req.size),
        "truncated_component_count": _count(truncated),
        "fully_blocked_component_count": _count(blocked),
        "world_steps_with_any_finger_truncation": _count(
            finger_truncated.any(axis=1)
        ),
        "absolute_lost_over_absolute_requested": _ratio(loss, req),
        "absolute_requested_sum": float(np.abs(req).sum()),
        "absolute_lost_sum": float(np.abs(loss).sum()),
        "truncated_components_by_requested_direction": {
            "positive": _count(truncated & (req > TOLERANCE)),
            "negative": _count(truncated & (req < -TOLERANCE)),
        },
        "fully_blocked_components_by_requested_direction": {
            "positive": _count(blocked & (req > TOLERANCE)),
            "negative": _count(blocked & (req < -TOLERANCE)),
        },
        "ctrlrange_truncation_path": {
            "requested_at_residual_limit_then_truncated": _count(
                truncated & at_limit
            ),
            "requested_below_residual_limit_but_truncated": _count(
                truncated & ~at_limit
            ),
            "requested_at_residual_limit_then_fully_blocked": _count(
                blocked & at_limit
            ),
            "requested_below_residual_limit_but_fully_blocked": _count(
                blocked & ~at_limit
            ),
        },
    }


def run(source: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source = source.resolve(strict=True)
    report = json.loads(source.read_text())
    diagnostic_contract = report.get("diagnostic_no_commit", {}).get("contract", {})
    local_settings = report.get("local_settings", {})
    if (
        report.get("schema") != "taco_pour_ctrlrange_training_distribution_run_v1"
        or report.get("status") != "diagnostic_complete_no_commit"
        or report.get("promotion", {}).get("allowed") is not False
        or report.get("promotion", {}).get("chunk_committed") is not False
        or report.get("promotion", {}).get("committed_boundary_written") is not False
        or report.get("promotion", {}).get("optimized_trajectory_written") is not False
        or report.get("incoming_boundary_restored_after_diagnostic") is not True
        or report.get("committed_reference_index") != 20
        or report.get("tracking_variant") != "tool_only"
        or diagnostic_contract.get("schema")
            != "taco_pour_ctrlrange_training_distribution_v1"
        or diagnostic_contract.get("training", {}).get("fixed_reset_endpoints")
            != [20, 20, 20, 46]
        or local_settings.get("worlds") != 4
        or local_settings.get("ppo_epochs") != 8
        or local_settings.get("training_start_distribution") != [20, 20, 20, 46]
    ):
        raise ValueError("source is not the frozen completed no-commit diagnostic")
    if (source.parent / "optimized_trajectory.npz").exists():
        raise ValueError("diagnostic output unexpectedly contains an optimized trajectory")
    if list(source.parent.glob("committed_boundary_*")):
        raise ValueError("diagnostic output unexpectedly contains a committed boundary")

    runs = report.get("diagnostic", {}).get("training_runs", [])
    if len(runs) != 1:
        raise ValueError("expected exactly one PPO training run")
    visitation = runs[0].get("training_visitation", {})
    manifest_path = Path(visitation.get("path", "")).resolve(strict=True)
    if _sha256(manifest_path) != visitation.get("sha256"):
        raise ValueError("training visitation manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    action = manifest.get("action_contract", {})
    epochs = manifest.get("epochs", [])
    if (
        manifest.get("schema") != "taco_ppo_training_visitation_v3"
        or manifest.get("status") != "complete"
        or len(action.get("actuator_names", [])) != 36
        or len(action.get("coordinate_units", [])) != 36
        or len(epochs) != 8
        or float(action.get("residual_scale", -1)) != 0.05
        or float(action.get("residual_clip", -1)) != 0.05
    ):
        raise ValueError("training visitation differs from the frozen v3 contract")

    combined: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "source_endpoint", "outcome_endpoint", "requested_residual",
            "effective_residual_after_ctrlrange", "residual_lost_to_ctrlrange",
        )
    }
    epoch_index = []
    world_index = []
    rollout_step = []
    per_epoch_artifacts = []
    for expected_epoch, row in enumerate(epochs, start=1):
        if row.get("epoch") != expected_epoch or row.get("sample_count") != 160:
            raise ValueError("each epoch must contain exactly 4 x 40 ordered samples")
        visits_path = Path(row["visits"]["path"])
        if not visits_path.is_absolute():
            visits_path = manifest_path.parent / visits_path
        visits_path = visits_path.resolve(strict=True)
        if _sha256(visits_path) != row["visits"]["sha256"]:
            raise ValueError(f"epoch {expected_epoch} visit artifact hash differs")
        with np.load(visits_path, allow_pickle=False) as arrays:
            data = {name: np.asarray(arrays[name]) for name in combined}
        if (
            data["source_endpoint"].shape != (160,)
            or data["outcome_endpoint"].shape != (160,)
            or any(data[name].shape != (160, 36) for name in (
                "requested_residual", "effective_residual_after_ctrlrange",
                "residual_lost_to_ctrlrange",
            ))
        ):
            raise ValueError(f"epoch {expected_epoch} has an unexpected v3 array shape")
        if not np.array_equal(
            data["requested_residual"],
            data["effective_residual_after_ctrlrange"]
            + data["residual_lost_to_ctrlrange"],
        ):
            raise ValueError(f"epoch {expected_epoch} residual identity differs")
        source_by_step_world = data["source_endpoint"].reshape(40, 4)
        if source_by_step_world[0].tolist() != [20, 20, 20, 46]:
            raise ValueError(
                f"epoch {expected_epoch} does not start with the frozen 3+1 assignment"
            )
        for name, value in data.items():
            combined[name].append(value)
        epoch_index.append(np.full(160, expected_epoch, dtype=np.int32))
        # PpoTrainingTrace concatenates the four-world batch once per rollout step.
        world_index.append(np.tile(np.arange(4, dtype=np.int32), 40))
        rollout_step.append(np.repeat(np.arange(40, dtype=np.int32), 4))
        per_epoch_artifacts.append({
            "epoch": expected_epoch,
            "path": str(visits_path),
            "sha256": row["visits"]["sha256"],
        })

    data = {name: np.concatenate(values, axis=0) for name, values in combined.items()}
    epoch_index = np.concatenate(epoch_index)
    world_index = np.concatenate(world_index)
    rollout_step = np.concatenate(rollout_step)
    requested = data["requested_residual"].astype(np.float64, copy=False)
    effective = data["effective_residual_after_ctrlrange"].astype(np.float64, copy=False)
    lost = data["residual_lost_to_ctrlrange"].astype(np.float64, copy=False)
    if len(requested) != 1280:
        raise ValueError("diagnostic must contain exactly 1280 world-step samples")

    names = action["actuator_names"]
    units = action["coordinate_units"]
    residual_limit = float(action["residual_clip"])
    all_rows = np.ones(1280, dtype=bool)
    anchor_rows = world_index < 3
    tail_rows = world_index == 3
    endpoint_rows = np.isin(data["outcome_endpoint"], np.arange(46, 54))
    scopes = {
        "all_training_samples": _scope(
            all_rows, requested, effective, lost, residual_limit=residual_limit
        ),
        "anchor_worlds_0_to_2": _scope(
            anchor_rows, requested, effective, lost, residual_limit=residual_limit
        ),
        "tail_world_3": _scope(
            tail_rows, requested, effective, lost, residual_limit=residual_limit
        ),
        "outcome_endpoints_46_to_53": _scope(
            endpoint_rows, requested, effective, lost, residual_limit=residual_limit
        ),
    }
    per_endpoint = {
        str(endpoint): _scope(
            data["outcome_endpoint"] == endpoint,
            requested, effective, lost, residual_limit=residual_limit,
        )
        for endpoint in range(46, 54)
    }

    truncated = np.abs(lost) > TOLERANCE
    blocked = (np.abs(requested) > TOLERANCE) & (np.abs(effective) <= TOLERANCE)
    per_actuator = []
    for index, (name, unit) in enumerate(zip(names, units, strict=True)):
        item_truncated = truncated[:, index]
        item_blocked = blocked[:, index]
        item_requested = requested[:, index]
        item_lost = lost[:, index]
        at_limit = np.abs(item_requested) >= residual_limit - TOLERANCE
        per_actuator.append({
            "index": index,
            "actuator": name,
            "unit": unit,
            "lost_nonzero": {
                "total": _count(item_truncated),
                "positive_request": _count(item_truncated & (item_requested > TOLERANCE)),
                "negative_request": _count(item_truncated & (item_requested < -TOLERANCE)),
            },
            "fully_blocked": {
                "total": _count(item_blocked),
                "positive_request": _count(item_blocked & (item_requested > TOLERANCE)),
                "negative_request": _count(item_blocked & (item_requested < -TOLERANCE)),
            },
            "absolute_lost_over_absolute_requested": _ratio(
                item_lost, item_requested
            ),
            "truncated_at_residual_limit": _count(item_truncated & at_limit),
            "truncated_below_residual_limit": _count(item_truncated & ~at_limit),
        })

    per_epoch = []
    for epoch in range(1, 9):
        selected = epoch_index == epoch
        finger_truncated = truncated[selected][:, FINGER_INDICES].any(axis=1)
        epoch_world = world_index[selected]
        epoch_rollout = rollout_step[selected]
        per_epoch.append({
            "epoch": epoch,
            "world_steps_with_any_finger_truncation": _count(finger_truncated),
            "anchor_world_steps_with_any_finger_truncation": _count(
                finger_truncated & (epoch_world < 3)
            ),
            "tail_world_steps_with_any_finger_truncation": _count(
                finger_truncated & (epoch_world == 3)
            ),
            "rollout_time_slots_with_any_world_finger_truncation": int(
                len(np.unique(epoch_rollout[finger_truncated]))
            ),
        })

    traces = report["diagnostic"]["validation_traces"]
    cpu_description = {
        trace["mode"]: {
            "validated_steps": trace["validated_steps"],
            "first_failure": trace["first_failure"],
            "feasible": trace["feasible"],
        }
        for trace in traces
    }
    result = {
        "schema": "taco_pour_training_ctrlrange_distribution_audit_v1",
        "status": "descriptive_measurement_complete_no_severity_threshold",
        "scope": {
            "diagnostic_only": True,
            "algorithm_changed": False,
            "promotion_allowed": False,
            "chunk_committed": False,
            "performance_comparison_allowed": False,
            "new_scale_selected": False,
            "severity_threshold_defined": False,
            "worlds": 4,
            "epochs": 8,
            "samples": 1280,
            "seed": 0,
            "fixed_reset_endpoints": [20, 20, 20, 46],
            "world_row_order_contract": (
                "within each rollout step, rows are worlds 0,1,2,3"
            ),
            "numeric_tolerance": TOLERANCE,
        },
        "inputs": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "run_report": {"path": str(source), "sha256": _sha256(source)},
            "training_visitation_manifest": {
                "path": str(manifest_path), "sha256": _sha256(manifest_path)
            },
            "epoch_visit_artifacts": per_epoch_artifacts,
        },
        "residual_semantics": manifest["logging_semantics"],
        "scopes": scopes,
        "per_outcome_endpoint_46_to_53": per_endpoint,
        "per_epoch": per_epoch,
        "per_actuator_and_requested_direction": per_actuator,
        "CPU_validation_descriptive_only": cpu_description,
        "interpretation_guard": (
            "The CPU score is recorded for run completeness only. GPU training is "
            "nondeterministic, so it is not performance or causal evidence against the "
            "previous 3+1 run."
        ),
        "descriptive_findings": {
            "ctrlrange_loss_present_throughout_training": True,
            "all_eight_epochs_have_finger_truncation": all(
                row["world_steps_with_any_finger_truncation"] > 0
                for row in per_epoch
            ),
            "every_epoch_rollout_slot_has_at_least_one_affected_world": all(
                row["rollout_time_slots_with_any_world_finger_truncation"] == 40
                for row in per_epoch
            ),
            "loss_is_not_confined_to_tail_world": (
                scopes["anchor_worlds_0_to_2"][
                    "world_steps_with_any_finger_truncation"
                ] > 0
            ),
            "below_residual_limit_truncation_exists": (
                scopes["all_training_samples"]["ctrlrange_truncation_path"][
                    "requested_below_residual_limit_but_truncated"
                ] > 0
            ),
            "causal_task_failure_claimed": False,
            "action_space_redesign_selected": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "runs/taco_pour_ctrlrange_training_distribution_v1/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "runs/taco_pour_ctrlrange_training_distribution_v1"
            / "ctrlrange_distribution_audit.json"
        ),
    )
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
