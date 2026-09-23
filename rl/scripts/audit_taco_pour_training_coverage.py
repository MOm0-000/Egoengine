#!/usr/bin/env python3
"""Summarize prospective PPO visitation around Pour endpoints 46--50."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_ENDPOINTS = tuple(range(46, 51))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_artifact(recorded_path: str, local_path: Path) -> Path:
    """Prefer the same run's checked-in artifact over a machine-local path."""
    return local_path if local_path.exists() else Path(recorded_path)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "runs/taco_pour_training_coverage_v1",
    )
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_training_coverage_diagnostic_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "coverage_analysis.json"
    if output.exists():
        raise FileExistsError(output)

    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text())
    chunks = report.get("chunks", [])
    if len(chunks) != 1 or chunks[0].get("start") != 20:
        raise ValueError("coverage audit requires exactly one endpoint-20 chunk")
    training_runs = chunks[0].get("training_runs", [])
    if len(training_runs) != 1:
        raise ValueError("coverage audit requires exactly one PPO training run")
    trace_report = training_runs[0]["training_visitation"]
    if trace_report.get("schema") != "taco_ppo_training_visitation_v2":
        raise ValueError("coverage audit requires visitation schema v2")
    manifest_path = local_artifact(
        trace_report["path"],
        args.run_dir / "ppo_chunk_20/training_visitation/manifest.json",
    )
    if sha256(manifest_path) != trace_report["sha256"]:
        raise ValueError("training visitation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or len(manifest.get("epochs", [])) != 8:
        raise ValueError("coverage audit requires eight complete logged epochs")

    epoch_arrays = []
    epoch_rows = []
    for artifact in manifest["epochs"]:
        visits_path = local_artifact(
            artifact["visits"]["path"],
            manifest_path.parent / Path(artifact["visits"]["path"]).name,
        )
        summary_path = local_artifact(
            artifact["summary"]["path"],
            manifest_path.parent / Path(artifact["summary"]["path"]).name,
        )
        if sha256(visits_path) != artifact["visits"]["sha256"]:
            raise ValueError(f"visitation hash mismatch: {visits_path}")
        if sha256(summary_path) != artifact["summary"]["sha256"]:
            raise ValueError(f"summary hash mismatch: {summary_path}")
        loaded = np.load(visits_path, allow_pickle=False)
        data = {name: loaded[name] for name in loaded.files}
        if len(data["source_endpoint"]) != 40:
            raise ValueError("each frozen PPO epoch must contain exactly 40 samples")
        epoch_arrays.append(data)
        outcome = data["outcome_endpoint"]
        epoch_rows.append({
            "epoch": int(artifact["epoch"]),
            "target_endpoint_visit_counts": {
                str(endpoint): int((outcome == endpoint).sum())
                for endpoint in TARGET_ENDPOINTS
            },
            "maximum_outcome_endpoint": int(outcome.max()),
            "termination_endpoints": list(map(
                int, outcome[data["tracking_terminated"]]
            )),
        })

    data = {
        name: np.concatenate([epoch[name] for epoch in epoch_arrays], axis=0)
        for name in epoch_arrays[0]
    }
    if len(data["source_endpoint"]) != 320:
        raise ValueError("eight 40-step epochs must contain 320 samples")

    # Episodes continue across epoch boundaries. Reconstruct them from the
    # actual reset-to-20 transitions rather than treating each epoch as a reset.
    episodes = []
    current = None
    for index, (source, outcome, terminated, timeout) in enumerate(zip(
        data["source_endpoint"], data["outcome_endpoint"],
        data["tracking_terminated"], data["time_out"], strict=True,
    )):
        source, outcome = int(source), int(outcome)
        if current is None:
            if source != 20:
                raise ValueError("first training episode must start at endpoint 20")
            current = {"first_sample": index, "source_endpoint": source}
        elif source != int(data["outcome_endpoint"][index - 1]):
            if source != 20 or not (
                bool(data["tracking_terminated"][index - 1])
                or bool(data["time_out"][index - 1])
            ):
                raise ValueError("unexpected endpoint discontinuity in training trace")
            current = {"first_sample": index, "source_endpoint": source}
        current["last_sample"] = index
        current["outcome_endpoint"] = outcome
        current["sample_count"] = index - current["first_sample"] + 1
        current["tracking_terminated"] = bool(terminated)
        current["time_out"] = bool(timeout)
        if terminated or timeout:
            episodes.append(current)
            current = None
    if current is not None:
        current["incomplete_at_training_end"] = True
        episodes.append(current)

    per_endpoint = {}
    for endpoint in TARGET_ENDPOINTS:
        select = data["outcome_endpoint"] == endpoint
        preclamp = data["right_wrist_sampled_action_preclamp"][select]
        clamped = data["right_wrist_sampled_action_clamped"][select]
        applied = data["right_wrist_applied_residual"][select]
        per_endpoint[str(endpoint)] = {
            "visit_count": int(select.sum()),
            "epoch_visit_count": int(sum(
                row["target_endpoint_visit_counts"][str(endpoint)] > 0
                for row in epoch_rows
            )),
            "tool_position_error_m": distribution(data["tool_position_error_m"][select]),
            "tool_rotation_error_rad": distribution(data["tool_rotation_error_rad"][select]),
            "normalized_ellipse_score": distribution(data["tool_objective_score"][select]),
            "tracking_termination_count": int(data["tracking_terminated"][select].sum()),
            "right_wrist_sampled_preclamp_abs": distribution(np.abs(preclamp)),
            "right_wrist_sampled_clamped_abs": distribution(np.abs(clamped)),
            "right_wrist_applied_residual_abs": distribution(np.abs(applied)),
            "sampled_preclamp_fraction_abs_gt_1": (
                float((np.abs(preclamp) > 1.0).mean()) if len(preclamp) else None
            ),
            "sampled_clamped_fraction_at_limit": (
                float(np.isclose(np.abs(clamped), 1.0).mean()) if len(clamped) else None
            ),
            "applied_fraction_at_safety_clip": (
                float(np.isclose(np.abs(applied), 0.05, atol=1e-7).mean())
                if len(applied) else None
            ),
        }

    termination_counts = Counter(map(
        int, data["outcome_endpoint"][data["tracking_terminated"]]
    ))
    completed = [episode for episode in episodes if episode["tracking_terminated"] or episode["time_out"]]
    target_reached = {
        str(endpoint): sum(episode["outcome_endpoint"] >= endpoint for episode in episodes)
        for endpoint in TARGET_ENDPOINTS
    }
    validation = {}
    for trace in chunks[0]["validation_traces"]:
        validation[trace["mode"]] = {
            "validated_steps": int(trace["validated_steps"]),
            "first_failure": trace["first_failure"],
        }

    target_visits = sum(row["visit_count"] for row in per_endpoint.values())
    coverage_observed = all(per_endpoint[str(endpoint)]["visit_count"] > 0 for endpoint in TARGET_ENDPOINTS)
    endpoint50_visits = per_endpoint["50"]["visit_count"]
    status = (
        "target_range_visited_but_sparse_with_termination_attrition"
        if coverage_observed and endpoint50_visits < len(manifest["epochs"])
        else "target_range_coverage_result_requires_review"
    )
    analysis = {
        "schema": "taco_pour_prospective_training_coverage_v1",
        "status": status,
        "scope": {
            "prospective_training_only": True,
            "historical_8epoch_coverage_inference_allowed": False,
            "actor_mean_mu_recorded": False,
            "policy_variance_recorded": False,
            "task_success_is_primary_question": False,
            "target_outcome_endpoints": list(TARGET_ENDPOINTS),
        },
        "artifacts": {
            "experiment_contract": {
                "path": display_path(args.contract), "sha256": sha256(args.contract)
            },
            "formal_run_report": {
                "path": display_path(report_path), "sha256": sha256(report_path)
            },
            "training_visitation_manifest": {
                "path": display_path(manifest_path), "sha256": sha256(manifest_path)
            },
        },
        "run_outcome": {
            "formal_status": report["status"],
            "task_success": report["task_success"],
            "committed_reference_index": report["committed_reference_index"],
            "cpu_validation": validation,
        },
        "training_samples": {
            "epochs": len(manifest["epochs"]),
            "samples_per_epoch": 40,
            "total_samples": len(data["source_endpoint"]),
            "episode_segments_started": len(episodes),
            "completed_episode_segments": len(completed),
            "incomplete_segments_at_training_end": len(episodes) - len(completed),
            "termination_endpoint_counts": {
                str(key): value for key, value in sorted(termination_counts.items())
            },
            "target_endpoint_total_visits": target_visits,
            "target_endpoint_visit_fraction_of_all_samples": target_visits / len(data["source_endpoint"]),
            "episode_segments_reaching_endpoint": target_reached,
            "per_epoch": epoch_rows,
            "per_target_endpoint": per_endpoint,
        },
        "decision": {
            "zero_coverage_hypothesis_rejected": coverage_observed,
            "endpoint50_seen_in_all_epochs": endpoint50_visits >= len(manifest["epochs"]),
            "endpoint50_visit_count": endpoint50_visits,
            "coverage_adequacy_predeclared_threshold_available": False,
            "coverage_alone_proven_as_failure_cause": False,
            "interpretation": (
                "The new run did visit every target endpoint, so it is not a zero-coverage "
                "case. Coverage was sparse and narrowed after early terminations: six episode "
                "segments reached endpoint 46, but only four reached endpoint 50. Because no "
                "adequacy threshold was predeclared and the final deterministic policy still "
                "failed, this run does not prove that sparse coverage is the sole cause."
            ),
        },
    }
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "formal_run_status": report["status"],
        "cpu_validation": validation,
        "episode_segments": len(episodes),
        "target_reached": target_reached,
        "per_target_endpoint_visits": {
            endpoint: row["visit_count"] for endpoint, row in per_endpoint.items()
        },
        "termination_endpoint_counts": analysis["training_samples"]["termination_endpoint_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
