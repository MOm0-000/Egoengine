#!/usr/bin/env python3
"""Summarize the frozen v4 100k seed-robustness extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v4"
OUTPUT = RUN / "seed_robustness_100k"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def report_path(candidate: str, seed: int) -> Path:
    return (
        RUN
        / "training"
        / f"candidate_{candidate}"
        / f"seed_{seed}"
        / "100k"
        / "report.json"
    )


def summarize_run(candidate: str, seed: int) -> dict[str, object]:
    path = report_path(candidate, seed)
    report = json.loads(path.read_text())
    at_0 = report["validations"]["0"]["summary"]
    at_100k = report["validations"]["62"]["summary"]
    curve = report["learning_curve"]["62"]
    validation_path = Path(
        report["validations"]["62"]["trajectory"]["path"]
    )
    with np.load(validation_path) as arrays:
        sigma = np.asarray(arrays["actor_sigma"][-1], dtype=np.float64)
    pretraining = report.get("pretraining_validation")
    return {
        "candidate": candidate,
        "seed": seed,
        "step0": at_0,
        "at_100k": at_100k,
        "delta_validated_intervals": int(
            at_100k["successful_intervals"] - at_0["successful_intervals"]
        ),
        "mean_exact_post_update_KL": curve["mean_post_update_KL"],
        "max_exact_post_update_KL": curve["max_post_update_KL"],
        "mean_ratio_outside_0p8_1p2_fraction": curve[
            "mean_ratio_outside_0p8_1p2_fraction"
        ],
        "mean_deterministic_effective_residual_RMS": curve[
            "mean_deterministic_effective_residual_RMS"
        ],
        "mean_deterministic_action_bound_fraction": curve[
            "mean_deterministic_action_bound_fraction"
        ],
        "mean_training_reward": curve["mean_training_reward"],
        "final_sigma": {
            "minimum": float(sigma.min()),
            "mean": float(sigma.mean()),
            "maximum": float(sigma.max()),
            "per_action_dimension": sigma.tolist(),
        },
        "physics_steps": report["training"]["simulation_physics_steps"],
        "actor_updates": report["training"]["actor_updates"],
        "checkpoint_roundtrip_exact": report["checkpoints"]["62"][
            "payload_roundtrip_exact"
        ],
        "all_canonical_identity_gates_passed": curve[
            "all_pre_optimizer_identity_checks_passed"
        ],
        "maximum_rollout_to_canonical_ratio_error": curve[
            "maximum_rollout_to_canonical_ratio_error"
        ],
        "maximum_canonical_optimizer_ratio_error": curve[
            "maximum_canonical_optimizer_ratio_identity_error"
        ],
        "pretraining_validation": pretraining,
        "report": artifact(path),
    }


def candidate_aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    at_0 = [int(row["step0"]["successful_intervals"]) for row in rows]
    at_100k = [int(row["at_100k"]["successful_intervals"]) for row in rows]
    delta = [int(row["delta_validated_intervals"]) for row in rows]
    return {
        "seeds_beating_Replay_at_100k": [
            int(row["seed"])
            for row in rows
            if int(row["at_100k"]["successful_intervals"]) > 30
        ],
        "seeds_improving_from_own_step0": [
            int(row["seed"])
            for row in rows
            if int(row["delta_validated_intervals"]) > 0
        ],
        "median_validated_intervals_at_0": float(median(at_0)),
        "median_validated_intervals_at_100k": float(median(at_100k)),
        "median_delta": float(median(delta)),
        "at_least_35_at_100k_count": sum(value >= 35 for value in at_100k),
        "all_seeds_improve_from_own_step0": all(value > 0 for value in delta),
    }


def main() -> None:
    rows = {
        candidate: [summarize_run(candidate, seed) for seed in (0, 1, 2)]
        for candidate in ("B", "A")
    }
    aggregates = {
        candidate: candidate_aggregate(candidate_rows)
        for candidate, candidate_rows in rows.items()
    }
    b = aggregates["B"]
    a = aggregates["A"]
    if b["all_seeds_improve_from_own_step0"] and b[
        "at_least_35_at_100k_count"
    ] >= 2:
        decision_case = "case_1_B_reproducible_improvement"
        next_authorization = (
            "candidate_B_only_three_seeds_may_continue_from_100k_to_500k_"
            "after_materializing_a_separate_frozen_extension_contract"
        )
    elif len(b["seeds_improving_from_own_step0"]) in (1, 2):
        decision_case = "case_2_B_partial_seed_improvement"
        next_authorization = "none_pending_seed_sensitivity_review"
    elif all(
        int(row["step0"]["successful_intervals"]) >= 35 for row in rows["A"]
    ) and all(int(row["delta_validated_intervals"]) < 0 for row in rows["A"]):
        decision_case = "case_3_A_good_random_step0_but_training_degrades"
        next_authorization = "none_pending_initialization_interpretation"
    elif all(
        aggregate["all_seeds_improve_from_own_step0"]
        for aggregate in aggregates.values()
    ):
        decision_case = "case_4_A_and_B_stably_improve"
        next_authorization = (
            "both_candidates_may_continue_after_a_separate_frozen_500k_contract"
        )
    else:
        decision_case = "none_of_the_prefrozen_cases"
        next_authorization = "none_pending_review"

    comparison = {
        "schema": "taco_pour_algorithmic_seed_robustness_comparison_v4",
        "status": "seed_robustness_100k_completed_no_chunk_commit",
        "classification": (
            "local_algorithmic_reproduction_seed_robustness_after_"
            "canonical_old_policy_likelihood_fix"
        ),
        "paper_faithful": False,
        "seeds": [0, 1, 2],
        "new_runs": ["B1", "B2", "A1", "A2"],
        "seed0_rerun": False,
        "candidates": rows,
        "aggregates": aggregates,
        "prefrozen_decision": {
            "case": decision_case,
            "next_authorization": next_authorization,
            "automatic_500k_execution_allowed": False,
        },
        "significance_test_executed": False,
        "hyperparameter_change_executed": False,
        "intermediate_checkpoint_selection_executed": False,
        "chunk_commit_written": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    lines = [
        "# TACO Pour v4 100k Seed Robustness",
        "",
        "| Candidate | Seed | Step 0 | 100k | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in ("B", "A"):
        for row in rows[candidate]:
            lines.append(
                f"| {candidate} | {row['seed']} | "
                f"{row['step0']['successful_intervals']}/40 | "
                f"{row['at_100k']['successful_intervals']}/40 | "
                f"{row['delta_validated_intervals']:+d} |"
            )
    lines.extend([
        "",
        "| Candidate | Seed | First failure | Score @40 | Score @50 | "
        "Score @60 | Mean KL | Max KL | Ratio outside |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for candidate in ("B", "A"):
        for row in rows[candidate]:
            milestone = row["at_100k"]
            lines.append(
                f"| {candidate} | {row['seed']} | "
                f"{milestone['first_failure_endpoint']} | "
                f"{milestone['endpoint40_score']:.6f} | "
                f"{milestone['endpoint50_score']:.6f} | "
                f"{milestone['endpoint60_score']:.6f} | "
                f"{row['mean_exact_post_update_KL']:.6f} | "
                f"{row['max_exact_post_update_KL']:.6f} | "
                f"{row['mean_ratio_outside_0p8_1p2_fraction']:.6f} |"
            )
    lines.extend([
        "",
        "| Candidate | Seed | Residual RMS | Bound fraction | "
        "Training reward | Final sigma min/mean/max |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for candidate in ("B", "A"):
        for row in rows[candidate]:
            sigma = row["final_sigma"]
            lines.append(
                f"| {candidate} | {row['seed']} | "
                f"{row['mean_deterministic_effective_residual_RMS']:.6f} | "
                f"{row['mean_deterministic_action_bound_fraction']:.6f} | "
                f"{row['mean_training_reward']:.6f} | "
                f"{sigma['minimum']:.6f} / {sigma['mean']:.6f} / "
                f"{sigma['maximum']:.6f} |"
            )
    lines.extend([
        "",
        "| Candidate | Seeds beating Replay | Seeds improving from step 0 | "
        "Median step 0 | Median 100k | Median delta |",
        "|---|---|---|---:|---:|---:|",
    ])
    for candidate in ("B", "A"):
        aggregate = aggregates[candidate]
        lines.append(
            f"| {candidate} | "
            f"{aggregate['seeds_beating_Replay_at_100k']} | "
            f"{aggregate['seeds_improving_from_own_step0']} | "
            f"{aggregate['median_validated_intervals_at_0']:.0f}/40 | "
            f"{aggregate['median_validated_intervals_at_100k']:.0f}/40 | "
            f"{aggregate['median_delta']:+.0f} |"
        )
    lines.extend([
        "",
        f"Prefrozen decision: `{decision_case}`.",
        "",
        "No 500k/1M run, hyperparameter change, checkpoint selection, "
        "significance test, or chunk commit was executed.",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
