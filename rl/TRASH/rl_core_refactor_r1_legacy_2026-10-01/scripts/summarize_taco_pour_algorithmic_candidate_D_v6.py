#!/usr/bin/env python3
"""Summarize the pre-registered three-seed Candidate-D 400k ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v6"
TRAINING_ROOT = RUN_ROOT / "training" / "candidate_D"
V5_COMPARISON = (
    ROOT / "runs/taco_pour_algorithmic_reproduction_training_v5/comparison.json"
)
OUTPUT = RUN_ROOT / "comparison.json"
SUMMARY = RUN_ROOT / "summary.md"
MILESTONES = (0, 62, 125, 188, 250)
LABELS = {0: "0", 62: "100k", 125: "200k", 188: "300k", 250: "400k"}
MASS_FLOOR = 1.0e-12


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def point_metrics(seed_root: Path, epoch: int) -> dict[str, object] | None:
    if epoch == 0:
        return None
    row = json.loads(
        (seed_root / "update_audit" / f"epoch_{epoch:04d}_summary.json").read_text()
    )["update"]
    return {
        name: row[name]
        for name in (
            "minimum_truncated_normalization_mass",
            "p01_truncated_normalization_mass",
            "median_truncated_normalization_mass",
            "raw_location_abs_mean",
            "raw_location_abs_p95",
            "raw_location_abs_max",
            "tanh_saturation_fraction",
            "bounded_mu_outside_support_count",
            "bounded_mu_at_support_fraction",
            "bounded_mu_normalized_support_position",
            "minimum_distance_to_support_boundary_in_sigma",
            "deterministic_action_bound_fraction",
            "deterministic_effective_residual_RMS",
            "training_reward_mean",
        )
    } | {
        "exact_post_update_KL_mean": row[
            "exact_old_to_new_truncated_policy_KL"
        ]["mean"],
        "ratio_outside_0p8_1p2_fraction": row[
            "post_optimizer_ratio_outside_0p8_1p2_fraction"
        ],
        "sigma": row["sigma_post_optimizer"],
        "canonical_optimizer_ratio_max_error": row["pre_optimizer_identity"][
            "canonical_optimizer_identity"
        ]["max_abs_ratio_minus_one"],
        "rollout_to_canonical_ratio_max_error": row["pre_optimizer_identity"][
            "rollout_vs_canonical"
        ]["max_abs_ratio_minus_one"],
    }


def validation_row(report: dict[str, Any], seed_root: Path, epoch: int):
    row = report["validations"][str(epoch)]
    summary = row["summary"]
    result = {
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
        "validation_residual_RMS": summary["validation_residual_RMS"],
        "validation_action_bound_fraction": summary[
            "validation_action_bound_fraction"
        ],
        "bounded_mu_outside_support_count": summary[
            "bounded_mu_outside_support_count"
        ],
        "final_sigma": summary["final_sigma"],
        "trajectory": row["trajectory"],
        "checkpoint": row["checkpoint"],
        "point_training_metrics": point_metrics(seed_root, epoch),
    }
    if epoch == 0:
        result["step0_checks"] = row["step0_checks"]
        result["all_step0_checks_passed"] = row["all_step0_checks_passed"]
    return result


def failure_mass(report: dict[str, Any]) -> float | None:
    match = re.search(
        r"normalization mass ([0-9.eE+-]+) is below",
        str(report.get("exception", {}).get("message", "")),
    )
    return float(match.group(1)) if match else None


def main() -> None:
    seeds: dict[str, dict[str, Any]] = {}
    completed: list[int] = []
    failed: list[int] = []
    mass_failures: list[int] = []
    strict_events: list[dict[str, int]] = []
    global_min_mass = float("inf")
    max_outside = 0

    for seed in (0, 1, 2):
        root = TRAINING_ROOT / f"seed_{seed}"
        report_path = root / "report.json"
        failure_path = root / "failure_report.json"
        if report_path.is_file() == failure_path.is_file():
            raise RuntimeError(f"seed {seed} must have one terminal report")
        path = report_path if report_path.is_file() else failure_path
        report = json.loads(path.read_text())
        if report["candidate"] != "D" or int(report["seed"]) != seed:
            raise RuntimeError(f"seed {seed} identity differs")
        if not report["pretraining_validation"]["all_step0_checks_passed"]:
            raise RuntimeError(f"seed {seed} step0 gate failed")
        curve = {
            LABELS[epoch]: validation_row(report, root, epoch)
            for epoch in MILESTONES
            if str(epoch) in report["validations"]
        }
        for epoch in MILESTONES:
            if str(epoch) in report["validations"] and report["validations"][
                str(epoch)
            ]["summary"]["forty_of_forty"]:
                strict_events.append({"seed": seed, "epoch": epoch})
        rows = [
            json.loads(item.read_text())["update"]
            for item in sorted((root / "update_audit").glob("epoch_*_summary.json"))
        ]
        masses = [float(row["minimum_truncated_normalization_mass"]) for row in rows]
        seed_min = min(masses, default=None)
        seed_max_outside = max(
            (int(row["bounded_mu_outside_support_count"]) for row in rows),
            default=0,
        )
        max_outside = max(max_outside, seed_max_outside)
        if seed_min is not None:
            global_min_mass = min(global_min_mass, seed_min)
        base = {
            "seed": seed,
            "status": report["status"],
            "completed_actor_updates": report["training"]["completed_actor_updates"],
            "observed_minimum_truncated_normalization_mass": seed_min,
            "maximum_bounded_mu_outside_support_count": seed_max_outside,
            "learning_curve": curve,
            "report": artifact(path),
            "chunk_commit_written": report["chunk_commit_written"],
            "intermediate_checkpoint_selected": report[
                "intermediate_checkpoint_selected"
            ],
        }
        if report_path.is_file():
            if report["training"]["completed_actor_updates"] != 250:
                raise RuntimeError(f"seed {seed} completion budget differs")
            completed.append(seed)
            base["final_400k_successful_intervals"] = curve["400k"][
                "successful_intervals"
            ]
            base["delta_step0_to_400k"] = (
                curve["400k"]["successful_intervals"] - 30
            )
        else:
            failed.append(seed)
            offending = failure_mass(report)
            if offending is not None:
                mass_failures.append(seed)
                global_min_mass = min(global_min_mass, offending)
            base.update(
                final_400k_successful_intervals=None,
                delta_step0_to_400k=None,
                exception=report["exception"],
                failure_normalization_mass=offending,
                eligible_as_400k_outcome=False,
            )
        seeds[str(seed)] = base

    all_complete = len(completed) == 3
    medians: dict[str, float | None] = {}
    for label in ("0", "100k", "200k", "300k", "400k"):
        values = [
            seeds[str(seed)]["learning_curve"][label]["successful_intervals"]
            for seed in (0, 1, 2)
            if label in seeds[str(seed)]["learning_curve"]
        ]
        medians[label] = float(median(values)) if len(values) == 3 else None
    d1 = bool(
        all_complete
        and max_outside == 0
        and not mass_failures
        and global_min_mass >= MASS_FLOOR
    )
    final_values = [
        int(seeds[str(seed)]["final_400k_successful_intervals"])
        for seed in completed
    ]
    beats = [seed for seed in completed if final_values[completed.index(seed)] > 30]
    d2 = bool(d1 and float(median(final_values)) > 30 and len(beats) >= 2)
    systematic_drift = bool(
        all_complete
        and all(
            int(seeds[str(seed)]["learning_curve"]["400k"]["successful_intervals"])
            < max(
                int(seeds[str(seed)]["learning_curve"]["100k"]["successful_intervals"]),
                int(seeds[str(seed)]["learning_curve"]["200k"]["successful_intervals"]),
            )
            for seed in (0, 1, 2)
        )
    )
    d3 = bool(d1 and (float(median(final_values)) <= 30 or systematic_drift))
    d4 = bool(mass_failures and max_outside == 0)
    d5 = bool(strict_events)
    active = d2

    comparison = {
        "schema": "taco_pour_algorithmic_candidate_D_comparison_v6",
        "status": (
            "completed_fixed_400k_ablation_no_chunk_commit"
            if all_complete
            else "candidate_D_failed_closed_before_complete_fixed_budget"
        ),
        "paper_faithful": False,
        "classification": (
            "local_algorithmic_reproduction_support_anchored_bounded_mean_ablation"
        ),
        "candidate": "D",
        "Replay_successful_intervals": 30,
        "only_algorithm_change_from_B": (
            "support_anchored_bounded_distribution_mean"
        ),
        "seeds": seeds,
        "aggregate": {
            "completed_seeds": completed,
            "failed_seeds": failed,
            "mass_floor_failure_seeds": mass_failures,
            "all_three_reached_400k": all_complete,
            "global_minimum_truncated_normalization_mass": (
                None if global_min_mass == float("inf") else global_min_mass
            ),
            "maximum_bounded_mu_outside_support_count": max_outside,
            "median_validated_intervals": medians,
            "seeds_beating_Replay_at_400k": beats,
            "strict_success_events": strict_events,
            "seed_robust_strict_success": len({row["seed"] for row in strict_events}) >= 2,
            "active_baseline": active,
        },
        "v5_evidence": {
            "artifact": artifact(V5_COMPARISON),
            "C3_mass_floor_recurrence": True,
            "v5_result_rewritten": False,
        },
        "prefrozen_classification": {
            "D1_raw_mean_support_collapse_mechanism_solved": {"triggered": d1},
            "D2_structurally_stable_and_task_learning_useful": {
                "triggered": d2,
                "active_baseline": active,
            },
            "D3_structurally_stable_but_performance_drifts": {
                "triggered": d3,
                "systematic_post_early_milestone_drift": systematic_drift,
                "learning_rate_schedule_automatically_authorized": False,
            },
            "D4_bounded_mean_but_mass_floor_occurs": {
                "triggered": d4,
                "mass_floor_failure_seeds": mass_failures,
                "next_distribution_requires_separate_review": True if d4 else None,
            },
            "D5_strict_task_success": {
                "triggered": d5,
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
        "# Candidate D v6 fixed 400k summary",
        "",
        "Candidate D is a local, non-paper-faithful ablation. Relative to Candidate B, "
        "the only algorithm change is a support-anchored bounded distribution mean; "
        "mean regularization and LR scheduling are disabled.",
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
        "Common-milestone medians: `" + " → ".join(
            "—" if medians[label] is None else str(int(medians[label]))
            for label in ("0", "100k", "200k", "300k", "400k")
        ) + "`.",
        "",
        f"Global minimum normalization mass: `{comparison['aggregate']['global_minimum_truncated_normalization_mass']}`; "
        f"maximum bounded-mean outside-support count: `{max_outside}`.",
        "",
        "## Frozen classification",
        "",
        f"- D1 structural support fix: **{d1}**",
        f"- D2 stable and useful learning: **{d2}**",
        f"- D3 stable but performance drift: **{d3}**",
        f"- D4 in-support mass-floor failure: **{d4}**",
        f"- D5 strict 40/40 event: **{d5}**",
        f"- Active Candidate-D baseline: **{active}**",
        "",
        "No intermediate checkpoint was selected, no earlier candidate was resumed, "
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
