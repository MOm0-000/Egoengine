#!/usr/bin/env python3
"""Summarize the single frozen 3+1 tail-curriculum experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def validation_summary(report: dict) -> dict:
    traces = report["chunks"][0]["validation_traces"]
    return {
        trace["mode"]: {
            "validated_steps": int(trace["validated_steps"]),
            "first_failure": trace["first_failure"],
        }
        for trace in traces
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "runs/taco_pour_tail_curriculum_3plus1_v1",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "runs/taco_pour_multiworld_training_v1/report.json",
    )
    args = parser.parse_args()
    output = args.run_dir / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text())
    baseline = json.loads(args.baseline.read_text())
    visitation = args.run_dir / "ppo_chunk_20/training_visitation"
    files = sorted(visitation.glob("epoch_*_visits.npz"))
    if len(files) != 8:
        raise ValueError("frozen experiment requires exactly eight visitation files")

    targets = tuple(range(46, 51))
    total_samples = 0
    target_visits = 0
    epochs_with_50 = 0
    per_endpoint = {str(endpoint): 0 for endpoint in targets}
    per_world_samples = [0, 0, 0, 0]
    per_world_target_visits = [0, 0, 0, 0]
    terminations = [
        {"any": 0, "tracking": 0, "timeout": 0} for _ in range(4)
    ]
    epoch_rows = []
    for epoch, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["source_endpoint"], dtype=np.int32)
            outcome = np.asarray(data["outcome_endpoint"], dtype=np.int32)
            tracking = np.asarray(data["tracking_terminated"], dtype=bool)
            timeout = np.asarray(data["time_out"], dtype=bool)
        if len(outcome) != 160 or any(len(row) != len(outcome) for row in (source, tracking, timeout)):
            raise ValueError(f"epoch {epoch} does not contain the frozen 160 samples")
        world = np.arange(len(outcome), dtype=np.int32) % 4
        is_target = np.isin(outcome, targets)
        total_samples += len(outcome)
        target_visits += int(is_target.sum())
        epochs_with_50 += int(np.any(outcome == 50))
        for endpoint in targets:
            per_endpoint[str(endpoint)] += int((outcome == endpoint).sum())
        for index in range(4):
            select = world == index
            per_world_samples[index] += int(select.sum())
            per_world_target_visits[index] += int((select & is_target).sum())
            terminations[index]["tracking"] += int((select & tracking).sum())
            terminations[index]["timeout"] += int((select & timeout).sum())
            terminations[index]["any"] += int((select & (tracking | timeout)).sum())
        epoch_rows.append({
            "epoch": epoch,
            "first_source_endpoint_by_world": source[:4].tolist(),
            "target_46_to_50_visits": int(is_target.sum()),
            "endpoint_50_visits": int((outcome == 50).sum()),
            "termination_counts_by_world": [
                {
                    "tracking": int(((world == index) & tracking).sum()),
                    "timeout": int(((world == index) & timeout).sum()),
                    "any": int(((world == index) & (tracking | timeout)).sum()),
                }
                for index in range(4)
            ],
        })

    if total_samples != 1280:
        raise ValueError("frozen experiment did not produce exactly 1280 samples")
    curriculum = report["chunks"][0]["training_runs"][0]["tail_curriculum"]
    starts = curriculum["world_start_endpoints"]
    if starts != [20, 20, 20, 46] or len(curriculum["epochs_prepared"]) != 8:
        raise ValueError("runtime curriculum audit differs from the frozen contract")
    if any(
        row["world_start_endpoints"] != starts
        or row["normalization_unchanged_during_prefix_replay"] is not True
        for row in curriculum["epochs_prepared"]
    ):
        raise ValueError("an epoch changed assignment or normalization during prefix replay")

    cpu = validation_summary(report)
    baseline_cpu = validation_summary(baseline)
    result = {
        "schema": "taco_pour_tail_curriculum_3plus1_comparison_v1",
        "status": (
            "CPU_40_of_40_passed" if cpu.get("rl", {}).get("validated_steps") == 40
            else "CPU_40_of_40_failed"
        ),
        "controlled_change": (
            "fixed_training_start_distribution_[20,20,20,20]_to_[20,20,20,46]"
        ),
        "frozen_budget": {
            "worlds": 4, "epochs": 8, "samples_per_epoch": 160,
            "total_samples": total_samples, "seed": 0,
        },
        "tail_visitation": {
            "endpoint_46_to_50_total_visits": target_visits,
            "fraction_of_1280": target_visits / total_samples,
            "per_endpoint": per_endpoint,
            "epochs_with_endpoint_50_visit": epochs_with_50,
            "per_world_total_samples": per_world_samples,
            "per_world_endpoint_46_to_50_visits": per_world_target_visits,
        },
        "terminations": {
            "per_world": terminations,
            "three_endpoint20_worlds": {
                key: sum(row[key] for row in terminations[:3])
                for key in ("any", "tracking", "timeout")
            },
            "one_endpoint46_world": terminations[3],
        },
        "deterministic_CPU_validation": cpu,
        "prior_four_endpoint20_world_baseline": {
            "deterministic_CPU_validation": baseline_cpu,
            "tail_visits": 96,
            "tail_visit_fraction": 0.075,
        },
        "per_epoch": epoch_rows,
        "runtime_contract": {
            "world_start_endpoints": starts,
            "normalization_unchanged_during_every_prefix_replay": True,
            "tail_physics_generated_by_current_actor": False,
            "CPU_acceptance_always_started_at_endpoint": 20,
            "curriculum_rollout_directly_committed": False,
        },
        "artifacts": {
            "formal_report": {"path": relative(report_path), "sha256": sha256(report_path)},
            "baseline_report": {"path": relative(args.baseline), "sha256": sha256(args.baseline)},
            "visitation_manifest": {
                "path": relative(visitation / "manifest.json"),
                "sha256": sha256(visitation / "manifest.json"),
            },
        },
        "interpretation_limits": {
            "tail_start_is_off_policy": True,
            "hidden_refresh_makes_tail_physics_on_policy": False,
            "single_controlled_run_not_a_seed_sweep": True,
            "result_proves_tail_sparsity_is_root_cause": False,
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "tail_visits": target_visits,
        "tail_fraction": target_visits / total_samples,
        "epochs_with_endpoint_50": epochs_with_50,
        "CPU": cpu,
    }, indent=2))


if __name__ == "__main__":
    main()
