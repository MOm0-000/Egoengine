#!/usr/bin/env python3
"""Compare the single frozen residual-scale experiment with the old 8-epoch policy."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OLD_TRACE = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "checkpoint_8ep_closed_loop_v2.json.gz"
)
OLD_BOUNDARY = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "endpoint20_complete_boundary.pt.gz"
)
NEW_RUN = ROOT / "runs/taco_pour_normalized_action_scale_v1/report.json.gz"
NEW_REPEAT = ROOT / (
    "runs/taco_pour_normalized_action_scale_v1/cpu_closed_loop_validation.json.gz"
)
NEW_BOUNDARY = ROOT / (
    "runs/taco_pour_normalized_action_scale_v1/input_contracts/resume_boundary.pt.gz"
)
OUTPUT = ROOT / "runs/taco_pour_normalized_action_scale_v1/comparison.json"


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial_rows(report: dict, mode: str) -> list[dict]:
    return report["repetitions"][0]["trials"][mode]["trace"]["steps"]


def action_metrics(rows: list[dict]) -> dict:
    raw = np.asarray([row["raw_residual_action"] for row in rows], np.float64)
    applied = np.asarray([row["applied_residual"] for row in rows], np.float64)
    saturated = np.isclose(np.abs(applied), 0.05, atol=1e-7, rtol=0.0)
    normalized_at_bound = np.isclose(np.abs(raw), 1.0, atol=1e-7, rtol=0.0)
    delta = np.diff(applied, axis=0)
    return {
        "steps": len(rows),
        "raw_abs_mean": float(np.abs(raw).mean()),
        "normalized_output_at_bound_fraction": float(normalized_at_bound.mean()),
        "applied_abs_mean_rad": float(np.abs(applied).mean()),
        "applied_saturation_fraction": float(saturated.mean()),
        "per_step_saturated_dimensions_min_median_max": [
            int(saturated.sum(axis=1).min()),
            float(np.median(saturated.sum(axis=1))),
            int(saturated.sum(axis=1).max()),
        ],
        "adjacent_delta_abs_max_rad": float(np.abs(delta).max()),
        "adjacent_delta_l2_max_rad": float(np.linalg.norm(delta, axis=1).max()),
    }


def tracking_metrics(rows: list[dict]) -> dict:
    score = np.asarray([row["objective_score"][0] for row in rows], np.float64)
    last = rows[-1]
    position = float(last["position_error_m"][0])
    rotation = float(last["rotation_error_rad"][0])
    return {
        "validated_steps": len(rows) - int(bool(last["terminated"])),
        "first_endpoint": int(rows[0]["endpoint"]),
        "last_endpoint": int(last["endpoint"]),
        "score_mean": float(score.mean()),
        "last_position_error_m": position,
        "last_rotation_error_rad": rotation,
        "last_score": float(last["objective_score"][0]),
        "last_position_squared_contribution": (position / 0.12) ** 2,
        "last_rotation_squared_contribution": (rotation / 1.5) ** 2,
    }


def exact_trace_equal(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    keys = (
        "endpoint_qpos", "endpoint_qvel", "commanded_ctrl", "objective_score",
        "position_error_m", "rotation_error_rad", "raw_residual_action",
        "applied_residual",
    )
    return all(
        a["endpoint"] == b["endpoint"]
        and all(np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in keys)
        for a, b in zip(left, right, strict=True)
    )


def main() -> None:
    import torch
    from video_to_spider.rl.replay_rl import _snapshot_value_equal

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old_report = load_json(OLD_TRACE)
    new_run = load_json(NEW_RUN)
    new_repeat = load_json(NEW_REPEAT)
    old_rl = trial_rows(old_report, "rl")
    old_replay = trial_rows(old_report, "replay")
    matching_chunks = [chunk for chunk in new_run["chunks"] if chunk["start"] == 20]
    if len(matching_chunks) != 1:
        raise ValueError(f"expected one endpoint-20 chunk, found {len(matching_chunks)}")
    new_chunk = matching_chunks[0]
    new_replay = next(row["steps"] for row in new_chunk["validation_traces"] if row["mode"] == "replay")
    new_rl = next(row["steps"] for row in new_chunk["validation_traces"] if row["mode"] == "rl")

    old_state = torch.load(
        io.BytesIO(gzip.decompress(OLD_BOUNDARY.read_bytes())),
        map_location="cpu", weights_only=False,
    )
    new_state = torch.load(
        io.BytesIO(gzip.decompress(NEW_BOUNDARY.read_bytes())),
        map_location="cpu", weights_only=False,
    )
    shared_keys = sorted(set(old_state) & set(new_state))
    unequal_state_keys = [
        key for key in shared_keys if not _snapshot_value_equal(old_state[key], new_state[key])
    ]

    old_by_endpoint = {int(row["endpoint"]): row for row in old_rl}
    new_by_endpoint = {int(row["endpoint"]): row for row in new_rl}
    replay_by_endpoint = {int(row["endpoint"]): row for row in new_replay}
    common = sorted(set(old_by_endpoint) & set(new_by_endpoint))
    old_scores = np.asarray([old_by_endpoint[e]["objective_score"][0] for e in common])
    new_scores = np.asarray([new_by_endpoint[e]["objective_score"][0] for e in common])
    replay_common = sorted(set(replay_by_endpoint) & set(new_by_endpoint))
    replay_scores = np.asarray([
        replay_by_endpoint[e]["objective_score"][0] for e in replay_common
    ])
    scaled_scores = np.asarray([
        new_by_endpoint[e]["objective_score"][0] for e in replay_common
    ])
    repeat_signatures = [
        row["trials"]["rl"]["trajectory_signature_sha256"]
        for row in new_repeat["repetitions"]
    ]

    old_action = action_metrics(old_rl)
    new_action = action_metrics(new_rl)
    old_tracking = tracking_metrics(old_rl)
    new_tracking = tracking_metrics(new_rl)
    validated_step_change = new_tracking["validated_steps"] - old_tracking["validated_steps"]
    scaled_passed = new_tracking["validated_steps"] == 40
    tracking_improved = validated_step_change > 0
    promote = (
        scaled_passed
        and new_action["applied_saturation_fraction"]
        < old_action["applied_saturation_fraction"]
        and len(set(repeat_signatures)) == 1
    )
    if promote:
        status = "interface_improved_and_tracking_gate_passed"
        interpretation = (
            "Linear scaling restored continuous action magnitudes, reduced runtime clipping, "
            "and the frozen 8-epoch policy passed the deterministic CPU gate in all repeats."
        )
    elif tracking_improved:
        status = "interface_improved_tracking_improved_but_gate_failed"
        interpretation = (
            "Linear scaling restored continuous action magnitudes and reduced runtime clipping. "
            "The frozen 8-epoch policy passed more intervals but still failed the strict CPU gate."
        )
    else:
        status = "interface_improved_tracking_not_improved_candidate_not_promoted"
        interpretation = (
            "Linear scaling restored continuous action magnitudes and reduced runtime clipping, "
            "but the one frozen 8-epoch policy did not improve the strict CPU tracking result. "
            "The candidate repairs the action representation defect but is not sufficient as a "
            "standalone PPO improvement under otherwise unchanged settings."
        )
    report = {
        "schema": "taco_pour_residual_action_scale_experiment_v1",
        "status": status,
        "scope": {
            "single_controlled_training_run": True,
            "epochs": 8,
            "seed": 0,
            "only_intended_algorithm_change": "residual_scale 1.0 -> 0.05",
            "full_horizon_executed": False,
        },
        "artifacts": {
            "old_cpu_trace": {"path": str(OLD_TRACE), "sha256": sha256(OLD_TRACE)},
            "new_formal_run": {"path": str(NEW_RUN), "sha256": sha256(NEW_RUN)},
            "new_cpu_repeats": {"path": str(NEW_REPEAT), "sha256": sha256(NEW_REPEAT)},
            "old_boundary": {"path": str(OLD_BOUNDARY), "sha256": sha256(OLD_BOUNDARY)},
            "new_boundary": {"path": str(NEW_BOUNDARY), "sha256": sha256(NEW_BOUNDARY)},
        },
        "frozen_contract": {
            "residual_action": new_run["residual_action"],
            "local_settings": new_run["local_settings"],
            "input_contract_snapshots": new_run["input_contract_snapshots"],
            "actor_transfer_bitwise_equal": new_chunk["training_runs"][0]["actor_transfer_bitwise_equal"],
            "endpoint20_state_entry_count": len(old_state),
            "endpoint20_state_keys_equal": set(old_state) == set(new_state),
            "endpoint20_state_unequal_keys": unequal_state_keys,
            "replay_trace_bitwise_equal": exact_trace_equal(old_replay, new_replay),
        },
        "old_scale_1_epoch8": {"tracking": old_tracking, "action": old_action},
        "new_scale_005_epoch8": {"tracking": new_tracking, "action": new_action},
        "paired_common_endpoints": {
            "range": [common[0], common[-1]],
            "count": len(common),
            "old_mean_score": float(old_scores.mean()),
            "new_mean_score": float(new_scores.mean()),
            "new_minus_old_mean_score": float((new_scores - old_scores).mean()),
            "new_lower_score_count": int((new_scores < old_scores).sum()),
        },
        "scaled_policy_vs_same_run_replay": {
            "range": [replay_common[0], replay_common[-1]],
            "count": len(replay_common),
            "replay_mean_score": float(replay_scores.mean()),
            "scaled_policy_mean_score": float(scaled_scores.mean()),
            "scaled_minus_replay_mean_score": float(
                (scaled_scores - replay_scores).mean()
            ),
            "scaled_lower_score_count": int((scaled_scores < replay_scores).sum()),
            "last_endpoint_scaled_minus_replay_score": float(
                scaled_scores[-1] - replay_scores[-1]
            ),
        },
        "deterministic_cpu_repeat": {
            "repetitions": len(repeat_signatures),
            "trajectory_signature_sha256": repeat_signatures,
            "bitwise_repeatable": len(set(repeat_signatures)) == 1,
            "validated_steps": [
                row["trials"]["rl"]["validated_steps"]
                for row in new_repeat["repetitions"]
            ],
        },
        "effects": {
            "applied_saturation_percentage_point_change": 100.0 * (
                new_action["applied_saturation_fraction"]
                - old_action["applied_saturation_fraction"]
            ),
            "max_adjacent_action_jump_change_rad": (
                new_action["adjacent_delta_abs_max_rad"]
                - old_action["adjacent_delta_abs_max_rad"]
            ),
            "validated_step_change": validated_step_change,
            "failure_tail_55_59_comparison_available": False,
            "failure_tail_55_59_unavailable_reason": (
                "the scaled candidate terminated at endpoint 50"
            ),
            "endpoint50_replay": {
                "position_error_m": replay_by_endpoint[50]["position_error_m"][0],
                "rotation_error_rad": replay_by_endpoint[50]["rotation_error_rad"][0],
                "score": replay_by_endpoint[50]["objective_score"][0],
            },
            "endpoint50_old_scale_1": {
                "position_error_m": old_by_endpoint[50]["position_error_m"][0],
                "rotation_error_rad": old_by_endpoint[50]["rotation_error_rad"][0],
                "score": old_by_endpoint[50]["objective_score"][0],
            },
            "endpoint50_scaled": {
                "position_error_m": new_by_endpoint[50]["position_error_m"][0],
                "rotation_error_rad": new_by_endpoint[50]["rotation_error_rad"][0],
                "score": new_by_endpoint[50]["objective_score"][0],
            },
        },
        "decision": {
            "promote_scaled_mapping": promote,
            "run_more_epochs_or_seeds": False,
            "claim_interface_diagnosis_was_false": False,
            "interpretation": interpretation,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "endpoint20_state_unequal_keys": unequal_state_keys,
        "replay_trace_bitwise_equal": report["frozen_contract"]["replay_trace_bitwise_equal"],
        "cpu_repeat": report["deterministic_cpu_repeat"],
        "effects": report["effects"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
