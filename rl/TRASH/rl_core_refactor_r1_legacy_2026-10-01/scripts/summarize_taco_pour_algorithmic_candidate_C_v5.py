#!/usr/bin/env python3
"""Summarize the pre-registered three-seed Candidate-C 400k ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v5"
TRAINING_ROOT = RUN_ROOT / "training" / "candidate_C"
V4_COMPARISON = (
    ROOT
    / "runs/taco_pour_algorithmic_reproduction_training_v4"
    / "B_budget_extension_500k"
    / "comparison.json"
)
OUTPUT = RUN_ROOT / "comparison.json"
SUMMARY = RUN_ROOT / "summary.md"
MILESTONES = (0, 62, 125, 188, 250)
LABELS = {0: "0", 62: "100k", 125: "200k", 188: "300k", 250: "400k"}
MASS_FLOOR = 1.0e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def point_metrics(seed_root: Path, epoch: int) -> dict[str, object] | None:
    if epoch == 0:
        return None
    path = seed_root / "update_audit" / f"epoch_{epoch:04d}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text())["update"]
    return {
        "minimum_truncated_normalization_mass": float(
            row["minimum_truncated_normalization_mass"]
        ),
        "p01_truncated_normalization_mass": float(
            row["p01_truncated_normalization_mass"]
        ),
        "median_truncated_normalization_mass": float(
            row["median_truncated_normalization_mass"]
        ),
        "raw_mu_outside_support_fraction": float(
            row["raw_mu_outside_support_fraction"]
        ),
        "maximum_support_violation_in_sigma": float(
            row["maximum_support_violation_in_sigma"]
        ),
        "p95_support_violation_in_sigma": float(
            row["p95_support_violation_in_sigma"]
        ),
        "deterministic_action_bound_fraction": float(
            row["deterministic_action_bound_fraction"]
        ),
        "deterministic_effective_residual_RMS": float(
            row["deterministic_effective_residual_RMS"]
        ),
        "exact_post_update_KL_mean": float(
            row["exact_old_to_new_truncated_policy_KL"]["mean"]
        ),
        "ratio_outside_0p8_1p2_fraction": float(
            row["post_optimizer_ratio_outside_0p8_1p2_fraction"]
        ),
        "training_reward_mean": float(row["training_reward_mean"]),
        "actor_mean_regularization": row["actor_mean_regularization"],
        "canonical_optimizer_ratio_max_error": float(
            row["pre_optimizer_identity"]["canonical_optimizer_identity"][
                "max_abs_ratio_minus_one"
            ]
        ),
        "rollout_to_canonical_ratio_max_error": float(
            row["pre_optimizer_identity"]["rollout_vs_canonical"][
                "max_abs_ratio_minus_one"
            ]
        ),
    }


def validation_row(
    report: dict[str, Any], seed_root: Path, epoch: int
) -> dict[str, object]:
    row = report["validations"][str(epoch)]
    summary = row["summary"]
    result: dict[str, object] = {
        "epoch": epoch,
        "budget_label": LABELS[epoch],
        "physics_steps": int(row["simulation_physics_steps"]),
        "successful_intervals": int(summary["successful_intervals"]),
        "first_failure_endpoint": summary["first_failure_endpoint"],
        "forty_of_forty": bool(summary["forty_of_forty"]),
        "beats_Replay": bool(summary["beats_Replay"]),
        "meaningful_refinement": bool(summary["meaningful_refinement"]),
        "endpoint40": summary["endpoint40"],
        "endpoint50": summary["endpoint50"],
        "endpoint60": summary["endpoint60"],
        "first_failure": summary.get("first_failure"),
        "validation_residual_RMS": float(summary["validation_residual_RMS"]),
        "validation_action_bound_fraction": float(
            summary["validation_action_bound_fraction"]
        ),
        "final_sigma": summary["final_sigma"],
        "trajectory": row["trajectory"],
        "checkpoint": row["checkpoint"],
        "point_training_metrics": point_metrics(seed_root, epoch),
    }
    if epoch == 0:
        result["step0_checks"] = row["step0_checks"]
        result["all_step0_checks_passed"] = bool(
            row["all_step0_checks_passed"]
        )
    return result


def failure_is_mass_floor(report: dict[str, Any]) -> bool:
    message = str(report.get("exception", {}).get("message", ""))
    return "normalization mass" in message and "below" in message


def failure_mass(report: dict[str, Any]) -> float | None:
    match = re.search(
        r"normalization mass ([0-9.eE+-]+) is below",
        str(report.get("exception", {}).get("message", "")),
    )
    return float(match.group(1)) if match else None


def monotonic_explosion(values: list[float]) -> bool:
    return len(values) > 1 and all(b > a for a, b in zip(values, values[1:]))


def main() -> None:
    v4 = json.loads(V4_COMPARISON.read_text())
    seeds: dict[str, dict[str, object]] = {}
    completed: list[int] = []
    failed: list[int] = []
    mass_floor_failures: list[int] = []
    strict_events: list[dict[str, int]] = []
    global_min_mass = float("inf")

    for seed in (0, 1, 2):
        seed_root = TRAINING_ROOT / f"seed_{seed}"
        report_path = seed_root / "report.json"
        failure_path = seed_root / "failure_report.json"
        if report_path.is_file() == failure_path.is_file():
            raise RuntimeError(
                f"seed {seed} must have exactly one completion/failure report"
            )
        path = report_path if report_path.is_file() else failure_path
        report = json.loads(path.read_text())
        if report["candidate"] != "C" or int(report["seed"]) != seed:
            raise RuntimeError(f"seed {seed} identity differs")
        if not report["pretraining_validation"]["all_step0_checks_passed"]:
            raise RuntimeError(f"seed {seed} step0 gate did not pass")
        curve = {
            LABELS[epoch]: validation_row(report, seed_root, epoch)
            for epoch in MILESTONES
            if str(epoch) in report["validations"]
        }
        for epoch in MILESTONES:
            if str(epoch) in report["validations"] and bool(
                report["validations"][str(epoch)]["summary"]["forty_of_forty"]
            ):
                strict_events.append({"seed": seed, "epoch": epoch})

        update_rows = sorted(
            (seed_root / "update_audit").glob("epoch_*_summary.json")
        )
        observed_masses = [
            float(json.loads(item.read_text())["update"][
                "minimum_truncated_normalization_mass"
            ])
            for item in update_rows
        ]
        seed_min_mass = min(observed_masses, default=None)
        if seed_min_mass is not None:
            global_min_mass = min(global_min_mass, seed_min_mass)

        base: dict[str, object] = {
            "seed": seed,
            "status": report["status"],
            "step0_successful_intervals": 30,
            "step0_first_failure_endpoint": 51,
            "completed_actor_updates": int(
                report["training"]["completed_actor_updates"]
            ),
            "observed_minimum_truncated_normalization_mass": seed_min_mass,
            "learning_curve": curve,
            "report": artifact(path),
            "chunk_commit_written": bool(report["chunk_commit_written"]),
            "intermediate_checkpoint_selected": bool(
                report["intermediate_checkpoint_selected"]
            ),
        }
        if report_path.is_file():
            if (
                report["status"] != "completed_fixed_400k_no_chunk_commit"
                or int(report["training"]["completed_actor_updates"]) != 250
                or set(report["validations"])
                != {str(value) for value in MILESTONES}
            ):
                raise RuntimeError(f"seed {seed} completion contract differs")
            completed.append(seed)
            bounds = [
                float(curve[label]["validation_action_bound_fraction"])
                for label in ("100k", "200k", "300k", "400k")
            ]
            base.update({
                "final_400k_successful_intervals": int(
                    curve["400k"]["successful_intervals"]
                ),
                "delta_step0_to_400k": int(
                    curve["400k"]["successful_intervals"]
                ) - 30,
                "validation_bound_fraction_monotonically_increases": (
                    monotonic_explosion(bounds)
                ),
            })
        else:
            failed.append(seed)
            if failure_is_mass_floor(report):
                mass_floor_failures.append(seed)
            offending_mass = failure_mass(report)
            if offending_mass is not None:
                global_min_mass = min(global_min_mass, offending_mass)
            base.update({
                "final_400k_successful_intervals": None,
                "delta_step0_to_400k": None,
                "exception": report["exception"],
                "failure_normalization_mass": offending_mass,
                "eligible_as_400k_outcome": False,
            })
        seeds[str(seed)] = base

    all_complete = len(completed) == 3
    median_by_budget: dict[str, float | None] = {}
    for label in ("0", "100k", "200k", "300k", "400k"):
        values = [
            int(seeds[str(seed)]["learning_curve"][label]["successful_intervals"])
            for seed in (0, 1, 2)
            if label in seeds[str(seed)]["learning_curve"]
        ]
        median_by_budget[label] = float(median(values)) if len(values) == 3 else None

    post_100 = [
        median_by_budget[label] for label in ("100k", "200k", "300k", "400k")
    ]
    continuously_declines = bool(
        all(value is not None for value in post_100)
        and all(
            float(b) < float(a)
            for a, b in zip(post_100, post_100[1:])
        )
    )
    no_mass_failure = not mass_floor_failures and global_min_mass >= MASS_FLOOR
    no_bound_explosion = bool(
        all_complete
        and not any(
            bool(seeds[str(seed)]["validation_bound_fraction_monotonically_increases"])
            for seed in completed
        )
    )
    # v4 did not log the v5 raw-mu support metrics. The categorical comparison
    # (0/3 B runs vs 3/3 C runs reaching 400k) and the shared validation-bound
    # occupancy are therefore the cross-candidate evidence; no raw-mu value is
    # fabricated for B.
    structural_stability = all_complete and no_mass_failure
    c1 = structural_stability and no_bound_explosion
    c2 = bool(all_complete and (
        float(median_by_budget["400k"]) <= 30 or continuously_declines
    ))
    c3 = bool(mass_floor_failures)
    c4 = bool(strict_events)
    final_values = [
        int(seeds[str(seed)]["final_400k_successful_intervals"])
        for seed in completed
    ]
    active_baseline = bool(
        c1 and all_complete and float(median(final_values)) > 30
    )

    comparison: dict[str, object] = {
        "schema": "taco_pour_algorithmic_candidate_C_comparison_v5",
        "status": (
            "completed_fixed_400k_ablation_no_chunk_commit"
            if all_complete
            else "candidate_C_failed_closed_before_complete_fixed_budget"
        ),
        "paper_faithful": False,
        "classification": (
            "local_algorithmic_reproduction_H2S2R_LSTM_mean_regularization_ablation"
        ),
        "candidate": "C",
        "Replay_successful_intervals": 30,
        "only_algorithm_change_from_B": {
            "bounds_loss_coef": 0.005,
            "bound_loss_type": "regularisation",
            "linear_LR_enabled": False,
        },
        "seeds": seeds,
        "aggregate": {
            "completed_seeds": completed,
            "failed_seeds": failed,
            "mass_floor_failure_seeds": mass_floor_failures,
            "all_three_reached_400k": all_complete,
            "global_minimum_truncated_normalization_mass": (
                global_min_mass if global_min_mass != float("inf") else None
            ),
            "median_validated_intervals": median_by_budget,
            "seeds_beating_Replay_at_400k": [
                seed
                for seed in completed
                if int(seeds[str(seed)]["final_400k_successful_intervals"]) > 30
            ],
            "strict_success_events": strict_events,
            "seed_robust_strict_success": len({row["seed"] for row in strict_events}) >= 2,
            "active_baseline": active_baseline,
        },
        "v4_comparison": {
            "artifact": artifact(V4_COMPARISON),
            "B_seeds_reaching_400k": [],
            "B_mass_floor_failure_seeds": [0, 2],
            "B_numerical_gate_failure_seeds": [1],
            "B_validation_bound_fraction": {
                seed: {
                    label: row["validation_action_bound_fraction"]
                    for label, row in v4["seeds"][seed]["learning_curve"].items()
                }
                for seed in ("0", "1", "2")
            },
            "raw_mu_support_metrics_available_in_v4": False,
            "raw_mu_cross_candidate_values_invented": False,
        },
        "prefrozen_classification": {
            "C1_mean_regularization_fixes_structural_instability": {
                "triggered": c1,
                "structural_stability_core_satisfied": structural_stability,
                "no_validation_bound_fraction_monotonic_explosion": no_bound_explosion,
                "relative_support_evidence": (
                    "C does not reach 400k for 3/3 seeds: seed2 repeats the "
                    "normalization-mass failure; v4 has no directly comparable "
                    "raw-mu metric"
                    if not all_complete
                    else "C reaches 400k for 3/3 seeds without the v4 mass-floor "
                    "failure; v4 has no directly comparable raw-mu metric"
                ),
                "active_baseline": active_baseline,
            },
            "C2_numerical_stability_fixed_but_performance_drifts": {
                "triggered": c2,
                "evaluable_at_fixed_400k_budget": all_complete,
                "median_400k_at_most_30": bool(
                    all_complete and float(median_by_budget["400k"]) <= 30
                ),
                "median_continuously_declines_after_100k": continuously_declines,
                "diagnostic_median_through_300k": {
                    label: median_by_budget[label]
                    for label in ("100k", "200k", "300k")
                },
            },
            "C3_mass_floor_still_occurs": {
                "triggered": c3,
                "mass_floor_failure_seeds": mass_floor_failures,
                "regularization_coefficient_sweep_allowed": False,
            },
            "C4_strict_task_success": {
                "triggered": c4,
                "events": strict_events,
                "seed_robust": len({row["seed"] for row in strict_events}) >= 2,
                "automatic_chunk_commit_allowed": False,
            },
        },
        "forbidden_actions_executed": [],
        "chunk_commit_written": False,
    }
    OUTPUT.write_text(json.dumps(comparison, indent=2) + "\n")

    lines = [
        "# Candidate C v5 fixed 400k summary",
        "",
        "Candidate C is a local, non-paper-faithful ablation. Relative to Candidate B, "
        "the only algorithm change is H2S2R-LSTM-style actor-mean regularization "
        "(`bounds_loss_coef=0.005`); the learning rate remains constant.",
        "",
        "| seed | 0 | 100k | 200k | 300k | 400k | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (0, 1, 2):
        row = seeds[str(seed)]
        curve = row["learning_curve"]
        cells = [
            str(curve.get(label, {}).get("successful_intervals", "—"))
            for label in ("0", "100k", "200k", "300k", "400k")
        ]
        lines.append(f"| {seed} | {' | '.join(cells)} | {row['status']} |")
    lines.extend([
        "",
        "## Frozen classification",
        "",
        f"- C1 structural stabilization: **{c1}**",
        f"- C2 performance drift: **{'not evaluable at 400k' if not all_complete else c2}**",
        f"- C3 mass-floor recurrence: **{c3}**",
        f"- C4 strict 40/40 event: **{c4}**",
        f"- Active Candidate-C baseline under the frozen rule: **{active_baseline}**",
        "",
        "No intermediate best checkpoint was selected, no v4 checkpoint was resumed, "
        "and no chunk was committed.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "comparison": artifact(OUTPUT),
        "summary": artifact(SUMMARY),
        "classification": comparison["prefrozen_classification"],
    }, indent=2))


if __name__ == "__main__":
    main()
