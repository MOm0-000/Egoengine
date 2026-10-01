#!/usr/bin/env python3
"""Summarize the pre-registered three-seed Candidate-B 500k extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "runs/taco_pour_algorithmic_reproduction_training_v4"
    / "B_budget_extension_500k"
)
OUTPUT = RUN_ROOT / "comparison.json"
SUMMARY = RUN_ROOT / "summary.md"
MILESTONES = (62, 125, 188, 250, 313)
LABELS = {
    62: "100k",
    125: "200k",
    188: "300k",
    250: "400k",
    313: "500k",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_report(seed: int) -> tuple[Path, dict[str, object]]:
    path = RUN_ROOT / f"seed_{seed}" / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed seed report: {path}")
    report = json.loads(path.read_text())
    if report["status"] != "completed_fixed_500k_no_chunk_commit":
        raise RuntimeError(f"seed {seed} is not a completed fixed-budget run")
    if int(report["seed"]) != seed or report["candidate"] != "B":
        raise RuntimeError(f"seed {seed} report identity differs")
    if set(report["validations"]) != {str(value) for value in MILESTONES}:
        raise RuntimeError(f"seed {seed} fixed validation set differs")
    if report["chunk_commit_written"] or report["automatic_1m_executed"]:
        raise RuntimeError(f"seed {seed} performed a forbidden post-500k action")
    if not report["resume_preflight"]["source_validation_all_arrays_exact"]:
        raise RuntimeError(f"seed {seed} source reproduction was not exact")
    return path, report


def failed_validation_row(row: dict[str, object]) -> dict[str, object]:
    summary = row["summary"]
    return {
        "epoch": int(row["epoch"]),
        "budget_label": LABELS[int(row["epoch"])],
        "physics_steps": int(row["physics_steps"]),
        "successful_intervals": int(summary["successful_intervals"]),
        "first_failure_endpoint": summary["first_failure_endpoint"],
        "forty_of_forty": bool(summary["forty_of_forty"]),
        "beats_Replay": bool(summary["beats_Replay"]),
        "meaningful_refinement": bool(summary["meaningful_refinement"]),
        "endpoint40": summary["endpoint40"],
        "endpoint50": summary["endpoint50"],
        "endpoint60": summary["endpoint60"],
        "first_failure": summary.get("first_failure"),
        "final_sigma": summary["final_sigma"],
        "validation_residual_RMS": float(summary["validation_residual_RMS"]),
        "validation_action_bound_fraction": float(
            summary["validation_action_bound_fraction"]
        ),
        "trajectory": row["trajectory"],
        "checkpoint": None,
        "training_metrics": None,
    }


def validation_row(report: dict[str, object], epoch: int) -> dict[str, object]:
    validation = report["validations"][str(epoch)]
    summary = validation["summary"]
    checkpoint = validation.get(
        "checkpoint", validation.get("source_100k_checkpoint")
    )
    if checkpoint is None:
        raise RuntimeError(f"epoch {epoch} validation has no bound checkpoint")
    return {
        "epoch": epoch,
        "budget_label": LABELS[epoch],
        "physics_steps": int(
            report["learning_curve"][str(epoch)]["physics_steps"]
        ),
        "successful_intervals": int(summary["successful_intervals"]),
        "first_failure_endpoint": summary["first_failure_endpoint"],
        "forty_of_forty": bool(summary["forty_of_forty"]),
        "beats_Replay": bool(summary["beats_Replay"]),
        "meaningful_refinement": bool(summary["meaningful_refinement"]),
        "endpoint40": summary["endpoint40"],
        "endpoint50": summary["endpoint50"],
        "endpoint60": summary["endpoint60"],
        "first_failure": summary.get("first_failure"),
        "final_sigma": summary["final_sigma"],
        "validation_residual_RMS": float(summary["validation_residual_RMS"]),
        "validation_action_bound_fraction": float(
            summary["validation_action_bound_fraction"]
        ),
        "trajectory": validation["trajectory"],
        "checkpoint": checkpoint,
        "training_metrics": report["learning_curve"][str(epoch)],
    }


def main() -> None:
    seeds: dict[str, dict[str, object]] = {}
    values_at_500k: list[int] = []
    strict_seeds: list[int] = []
    failed_seeds: list[int] = []
    for seed in (0, 1, 2):
        report_path = RUN_ROOT / f"seed_{seed}" / "report.json"
        failure_path = RUN_ROOT / f"seed_{seed}" / "failure_report.json"
        if not report_path.is_file():
            if not failure_path.is_file():
                raise FileNotFoundError(
                    f"seed {seed} has neither completion nor failure report"
                )
            failure = json.loads(failure_path.read_text())
            if failure["status"] != "failed_closed_before_fixed_500k_budget":
                raise RuntimeError(f"seed {seed} failure status differs")
            milestone_rows = failure["fixed_milestone_validations"]
            curve = {
                LABELS[int(epoch)]: failed_validation_row(row)
                for epoch, row in milestone_rows.items()
            }
            at_100 = int(curve["100k"]["successful_intervals"])
            failed_seeds.append(seed)
            seeds[str(seed)] = {
                "seed": seed,
                "status": "failed_closed_before_fixed_500k_budget",
                "step0_successful_intervals": 30,
                "step0_first_failure_endpoint": 51,
                "source_100k_successful_intervals": at_100,
                "final_500k_successful_intervals": None,
                "delta_100k_to_500k": None,
                "delta_step0_to_500k": None,
                "failure_report": artifact(failure_path),
                "failure": failure["exception"],
                "learning_curve": curve,
                "last_completed_actor_update_epoch": failure[
                    "last_completed_actor_update_epoch"
                ],
                "physics_steps_at_exception": failure[
                    "physics_steps_at_exception"
                ],
            }
            continue

        report_path, report = load_report(seed)
        curve = {
            LABELS[epoch]: validation_row(report, epoch)
            for epoch in MILESTONES
        }
        at_100 = int(curve["100k"]["successful_intervals"])
        at_500 = int(curve["500k"]["successful_intervals"])
        values_at_500k.append(at_500)
        if bool(curve["500k"]["forty_of_forty"]):
            strict_seeds.append(seed)
        seeds[str(seed)] = {
            "seed": seed,
            "status": "completed_fixed_500k_no_chunk_commit",
            "step0_successful_intervals": 30,
            "step0_first_failure_endpoint": 51,
            "source_100k_successful_intervals": at_100,
            "final_500k_successful_intervals": at_500,
            "delta_100k_to_500k": at_500 - at_100,
            "delta_step0_to_500k": at_500 - 30,
            "report": artifact(report_path),
            "resume_preflight": report["resume_preflight"],
            "learning_curve": curve,
            "total_actor_updates": report["training"]["total_actor_updates"],
            "total_physics_steps": report["training"]["total_physics_steps"],
        }

    values_at_100k = [
        int(seeds[str(seed)]["source_100k_successful_intervals"])
        for seed in (0, 1, 2)
    ]
    completed_seeds = [
        seed
        for seed in (0, 1, 2)
        if seeds[str(seed)]["final_500k_successful_intervals"] is not None
    ]
    beating_replay = [
        seed
        for seed in completed_seeds
        if int(seeds[str(seed)]["final_500k_successful_intervals"]) > 30
    ]
    improving_from_100k = [
        seed
        for seed in (0, 1, 2)
        if seeds[str(seed)]["delta_100k_to_500k"] is not None
        and int(seeds[str(seed)]["delta_100k_to_500k"]) > 0
    ]
    fixed_budget_evaluable = not failed_seeds
    median_500k = (
        float(median(values_at_500k)) if fixed_budget_evaluable else None
    )
    all_above_replay = len(beating_replay) == 3
    b500_1 = bool(
        fixed_budget_evaluable and all_above_replay and median_500k >= 35
    )
    b500_2 = bool(
        fixed_budget_evaluable
        and len(beating_replay) >= 2
        and len(beating_replay) < 3
    )
    b500_3 = bool(fixed_budget_evaluable and median_500k <= 36)
    b500_4 = bool(fixed_budget_evaluable and strict_seeds)
    classifications = {
        "B500_1_budget_resolves_seed_sensitivity": {
            "triggered": b500_1,
            "evaluable": fixed_budget_evaluable,
            "condition": "all_3_seeds_gt_30_and_median_at_least_35",
        },
        "B500_2_partial_robustness_remains": {
            "triggered": b500_2,
            "evaluable": fixed_budget_evaluable,
            "condition": "at_least_2_of_3_gt_30_and_at_least_1_seed_le_30",
        },
        "B500_3_overtraining_or_policy_drift": {
            "triggered": b500_3,
            "evaluable": fixed_budget_evaluable,
            "condition": "median_500k_le_median_100k_which_is_36",
        },
        "B500_4_strict_success_exists": {
            "triggered": b500_4,
            "evaluable": fixed_budget_evaluable,
            "condition": "any_seed_is_40_of_40",
            "strict_success_seeds": strict_seeds,
            "seed_robust_strict_success": len(strict_seeds) >= 2,
        },
    }
    if failed_seeds:
        interpretation = (
            "All three fixed Candidate-B continuations failed closed before "
            "500k: two crossed the frozen truncated-Gaussian normalization-"
            "mass floor and one crossed the frozen rollout-to-canonical "
            "likelihood tolerance. Seed sensitivity is not resolved, no "
            "intermediate milestone is a fallback, and no automatic 1M "
            "continuation is authorized."
        )
    elif b500_1:
        interpretation = (
            "The fixed 500k budget resolves the observed 100k seed "
            "sensitivity under the local Candidate-B contract. No automatic "
            "1M continuation is authorized."
        )
    elif b500_2:
        interpretation = (
            "Candidate B remains seed-sensitive at 500k; no 1M continuation "
            "is authorized before a separate update-stability decision."
        )
    elif b500_3:
        interpretation = (
            "The median fixed-budget outcome does not exceed the 100k median; "
            "continued training shows overtraining or policy-drift evidence."
        )
    else:
        interpretation = (
            "The fixed 500k outcome does not match a stronger pre-registered "
            "robustness case; no automatic follow-on is authorized."
        )

    comparison = {
        "schema": "taco_pour_algorithmic_candidate_B_500k_comparison_v4",
        "status": (
            "fixed_budget_extension_failed_closed_no_automatic_follow_on"
            if failed_seeds
            else "completed_fixed_budget_comparison_no_automatic_follow_on"
        ),
        "paper_faithful": False,
        "candidate": "B",
        "Replay_successful_intervals": 30,
        "seed0_was_not_rerun_at_100k": True,
        "Candidate_A_continued": False,
        "seeds": seeds,
        "aggregate": {
            "successful_intervals_at_100k": values_at_100k,
            "successful_intervals_at_500k": values_at_500k,
            "completed_500k_seeds": completed_seeds,
            "failed_closed_before_500k_seeds": failed_seeds,
            "fixed_budget_three_seed_classification_evaluable": (
                fixed_budget_evaluable
            ),
            "median_successful_intervals_at_step0": 30.0,
            "median_successful_intervals_at_100k": float(
                median(values_at_100k)
            ),
            "median_successful_intervals_at_500k": median_500k,
            "median_delta_100k_to_500k": (
                float(
                    median(
                        [
                            int(seeds[str(seed)]["delta_100k_to_500k"])
                            for seed in (0, 1, 2)
                        ]
                    )
                )
                if fixed_budget_evaluable
                else None
            ),
            "seeds_beating_Replay_at_500k": beating_replay,
            "seeds_beating_Replay_at_500k_count": len(beating_replay),
            "seeds_improving_from_own_100k": improving_from_100k,
            "seeds_improving_from_own_100k_count": len(
                improving_from_100k
            ),
            "strict_success_seeds": strict_seeds,
        },
        "prefrozen_classifications": classifications,
        "interpretation": interpretation,
        "automatic_1m_authorized": False,
        "automatic_1m_executed": False,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_authorized": False,
        "chunk_commit_written": False,
        "next_step_requires_separate_authorization": True,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(comparison, indent=2) + "\n")

    lines = [
        "# Candidate B fixed 500k seed extension",
        "",
        "No hyperparameter, policy, physics, or objective setting changed. "
        "Each seed continued in place from its exact epoch-62 checkpoint.",
        "",
        "| seed | step 0 | 100k | 200k | 300k | 400k | 500k | "
        "100k→500k | failure@500k |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (0, 1, 2):
        row = seeds[str(seed)]
        curve = row["learning_curve"]
        if row["status"] != "completed_fixed_500k_no_chunk_commit":
            def observed(label: str) -> str:
                value = curve.get(label)
                return (
                    f"{value['successful_intervals']}/40"
                    if value is not None
                    else "not reached"
                )

            lines.append(
                f"| {seed} | 30/40 | "
                f"{observed('100k')} | "
                f"{observed('200k')} | "
                f"{observed('300k')} | "
                "fail-closed | fail-closed | n/a | n/a |"
            )
            continue
        failure = curve["500k"]["first_failure_endpoint"]
        lines.append(
            f"| {seed} | 30/40 | "
            f"{curve['100k']['successful_intervals']}/40 | "
            f"{curve['200k']['successful_intervals']}/40 | "
            f"{curve['300k']['successful_intervals']}/40 | "
            f"{curve['400k']['successful_intervals']}/40 | "
            f"{curve['500k']['successful_intervals']}/40 | "
            f"{row['delta_100k_to_500k']:+d} | {failure} |"
        )
    lines.extend(
        [
            "",
            f"500k median: **{median_500k if median_500k is not None else 'not evaluable'}**; "
            f"complete 500k trajectories: **{len(completed_seeds)}/3**; strict successes: "
            f"**{len(strict_seeds)}/3**.",
            "",
            "Pre-registered cases:",
        ]
    )
    for name, row in classifications.items():
        state = (
            str(row["triggered"]).lower()
            if row["evaluable"]
            else "not evaluable (fixed 500k budget not completed)"
        )
        lines.append(f"- `{name}`: `{state}`")
    if failed_seeds:
        lines.extend(["", "Fail-closed records:"])
        for seed in failed_seeds:
            row = seeds[str(seed)]
            failure = row["failure"]
            lines.append(
                f"- seed {seed}: epoch {row['last_completed_actor_update_epoch']} "
                f"last complete update; failure at {row['physics_steps_at_exception']:,} "
                f"physics steps — `{failure['type']}: {failure['message']}`"
            )
    lines.extend(
        [
            "",
            interpretation,
            "",
            "No automatic 1M continuation, checkpoint selection, or chunk "
            "commit was executed or authorized.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(json.dumps({"comparison": artifact(OUTPUT), "summary": artifact(SUMMARY)}, indent=2))


if __name__ == "__main__":
    main()
