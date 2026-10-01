#!/usr/bin/env python3
"""Close Candidate G using only its predeclared milestones and decision rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/taco_pour_candidate_G_policy_warm_start_v1"
CONTRACT = ROOT / "configs/taco_pour_candidate_G_policy_warm_start_v1.yaml"
MILESTONES = (0, 62, 125, 188, 250)
HISTORICAL_FRESH_D = {
    "0": [9, 9, 9],
    "62": [17, 19, 20],
    "125": [20, 20, 19],
    "188": [20, 20, 20],
    "250": [14, 20, 17],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validation_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    return {
        "epoch": int(row["epoch"]),
        "physics_steps": int(row["simulation_physics_steps"]),
        "successful_intervals": int(summary["successful_intervals"]),
        "first_failure_endpoint": summary["first_failure_endpoint"],
        "forty_of_forty": bool(summary["forty_of_forty"]),
        "endpoint60": summary["endpoint60"],
        "endpoint70": summary["endpoint70"],
        "endpoint80": summary["endpoint80"],
        "validation_residual_RMS": summary["validation_residual_RMS"],
        "validation_action_bound_fraction": summary[
            "validation_action_bound_fraction"
        ],
        "final_sigma": summary["final_sigma"],
        "checkpoint": row["checkpoint"],
        "trajectory": row["trajectory"],
        "independent_exact_revalidation": row.get(
            "independent_exact_revalidation"
        ),
    }


def aggregate_coverage(report: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(report["training_visitation"]["path"])
    manifest = load_json(manifest_path)
    source_counts: dict[int, int] = {}
    outcome_counts: dict[int, int] = {}
    termination_counts: dict[int, int] = {}
    sample_count = 0
    for epoch in manifest["epochs"]:
        summary = load_json(Path(epoch["summary"]["path"]))
        sample_count += int(summary["sample_count"])
        for key, value in summary["source_endpoint_visit_counts"].items():
            source_counts[int(key)] = source_counts.get(int(key), 0) + int(value)
        for key, value in summary["outcome_endpoint_visit_counts"].items():
            outcome_counts[int(key)] = outcome_counts.get(int(key), 0) + int(value)
        for key, value in summary["tracking_termination_endpoint_counts"].items():
            termination_counts[int(key)] = termination_counts.get(int(key), 0) + int(value)

    source60_attempts = source_counts.get(60, 0)
    endpoint61_outcomes = outcome_counts.get(61, 0)
    endpoint61_terminations = termination_counts.get(61, 0)
    endpoint61_passes = endpoint61_outcomes - endpoint61_terminations
    if endpoint61_passes < 0:
        raise RuntimeError("endpoint61 termination count exceeds its outcomes")

    deep = {}
    for offset in (21, 25, 30, 35):
        visits = sum(
            count for endpoint, count in source_counts.items()
            if endpoint - 40 >= offset
        )
        deep[str(offset)] = {
            "visits": visits,
            "fraction_of_training_samples": visits / sample_count,
        }
    return {
        "sample_count": sample_count,
        "source60_attempts": source60_attempts,
        "endpoint61_outcomes": endpoint61_outcomes,
        "endpoint61_tracking_terminations": endpoint61_terminations,
        "endpoint61_passes": endpoint61_passes,
        "conditional_endpoint61_pass_rate": (
            endpoint61_passes / source60_attempts if source60_attempts else None
        ),
        "outcome_endpoint_reach_counts": {
            str(endpoint): outcome_counts.get(endpoint, 0)
            for endpoint in (65, 70, 75, 80)
        },
        "deep_source_visit_fraction_by_k": deep,
        "maximum_source_endpoint": max(source_counts),
        "maximum_outcome_endpoint": max(outcome_counts),
        "tracking_termination_count": sum(termination_counts.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--repository-tests", type=int, required=True)
    parser.add_argument("--repository-subtests", type=int, required=True)
    parser.add_argument("--repository-warnings", type=int, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    preflight_path = run_root / "preflight_G0/report.json"
    preflight = load_json(preflight_path)
    if preflight.get("status") != "passed_training_authorized" or not all(
        preflight.get("checks", {}).values()
    ):
        raise RuntimeError("Candidate-G G0 is not a complete pass")

    reports: dict[int, dict[str, Any]] = {}
    seed_rows: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        path = run_root / "training" / f"seed_{seed}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = load_json(path)
        reports[seed] = report
        if report.get("candidate") != "G" or report.get("seed") != seed:
            raise RuntimeError(f"Candidate-G report lineage differs for seed {seed}")
        if report.get("chunk_commit_written") is not False:
            raise RuntimeError(f"Candidate-G seed {seed} wrote a chunk commit")
        if report.get("status") not in {
            "completed_400k_without_strict_success_no_chunk_commit",
            "strict_success_independently_revalidated_no_chunk_commit",
        }:
            raise RuntimeError(f"Candidate-G seed {seed} did not close cleanly")
        validations = report["validations"]
        if set(map(int, validations)) != set(MILESTONES):
            raise RuntimeError(f"Candidate-G seed {seed} lacks a fixed milestone")
        step0 = validations["0"]
        if (
            not step0.get("all_step0_checks_passed")
            or step0["summary"]["successful_intervals"] != 20
            or step0["summary"]["first_failure_endpoint"] != 61
        ):
            raise RuntimeError(f"Candidate-G seed {seed} step-zero identity failed")
        epochs_prepared = report["boundary_context_audit"]["epochs_prepared"]
        if len(epochs_prepared) != 251:
            raise RuntimeError(f"Candidate-G seed {seed} context rebuild count differs")
        if not all(
            row["normalization_unchanged_during_prefix_replay"]
            and row["prefix_observations"] == 20
            and all(row["restore"]["physics_bitwise_equal"])
            for row in epochs_prepared
        ):
            raise RuntimeError(f"Candidate-G seed {seed} boundary context audit failed")
        if not report["learning_curve"]["250"]["cumulative"][
            "all_v6_identity_checks_passed"
        ]:
            raise RuntimeError(f"Candidate-G seed {seed} likelihood identity failed")

        milestone_rows = {
            str(epoch): validation_row(validations[str(epoch)])
            for epoch in MILESTONES
        }
        seed_rows.append({
            "seed": seed,
            "status": report["status"],
            "inheritance": report["inheritance"],
            "training": report["training"],
            "milestones": milestone_rows,
            "coverage": aggregate_coverage(report),
            "report": artifact(path),
            "chunk_commit_written": False,
            "intermediate_checkpoint_selected": False,
        })

    strict_events = []
    for seed, report in reports.items():
        for epoch, row in report["validations"].items():
            if row["summary"]["forty_of_forty"]:
                exact = row.get("independent_exact_revalidation", {})
                if not exact.get("all_arrays_bitwise_equal"):
                    raise RuntimeError(
                        f"seed {seed} epoch {epoch} strict result lacks exact replay"
                    )
                strict_events.append({"seed": seed, "epoch": int(epoch)})

    final_intervals = [
        reports[seed]["validations"]["250"]["summary"]["successful_intervals"]
        for seed in (0, 1, 2)
    ]
    seeds_at_least_30 = [
        seed for seed, intervals in enumerate(final_intervals) if intervals >= 30
    ]
    all_complete = all(
        reports[seed]["training"]["completed_actor_updates"] == 250
        and reports[seed]["training"]["simulation_physics_steps"] == 400000
        and reports[seed]["training"]["simulation_control_intervals"] == 40000
        for seed in (0, 1, 2)
    )
    if strict_events:
        classification = "G1"
        label = "STRICT_NEXT_WINDOW_SUCCESS"
        action = "stop_request_separate_promotion_no_commit"
    elif all_complete and len(seeds_at_least_30) >= 2:
        classification = "G2"
        label = "REPEATED_PARTIAL_WINDOW_EXTENSION_NO_COMMIT"
        action = "stop_review_only_no_automatic_extension"
    elif all_complete:
        classification = "G3"
        label = "NO_PREDECLARED_SUSTAINED_EXTENSION"
        action = "stop_this_specific_actor_only_warm_start_package"
    else:
        raise RuntimeError("Candidate-G cannot be classified by its frozen rules")

    median_intervals = {
        str(epoch): median(
            reports[seed]["validations"][str(epoch)]["summary"][
                "successful_intervals"
            ]
            for seed in (0, 1, 2)
        )
        for epoch in MILESTONES
    }
    comparison = {
        "schema": "taco_pour_candidate_G_policy_warm_start_comparison_v1",
        "status": "completed_prefrozen_three_seed_adjudication",
        "paper_faithful": False,
        "classification": "local_policy_warm_start_with_recurrent_boundary_context",
        "candidate": "G",
        "contract": artifact(CONTRACT),
        "preflight": artifact(preflight_path),
        "warm_start_scope": (
            "source20 Candidate-D seed2 epoch125 actor/RMS/sigma plus causally "
            "reencoded endpoint40 recurrent context; fresh critic and optimizers"
        ),
        "baselines": {
            "zero_residual_Replay": {
                "successful_intervals": 9,
                "first_failure_endpoint": 50,
            },
            "pure_previous_policy_carryover": {
                "successful_intervals": 20,
                "first_failure_endpoint": 61,
            },
            "historical_fresh_source40_candidate_D": {
                "milestone_successful_intervals_by_seed": HISTORICAL_FRESH_D,
                "same_new_training_budget": True,
                "equal_total_data_cost": False,
            },
        },
        "seeds": seed_rows,
        "aggregate": {
            "completed_seeds": [0, 1, 2],
            "all_three_completed_400k": all_complete,
            "strict_success_events": strict_events,
            "final_successful_intervals": final_intervals,
            "final_seeds_at_least_30": seeds_at_least_30,
            "median_successful_intervals_by_milestone": median_intervals,
            "prefrozen_classification": classification,
            "prefrozen_label": label,
        },
        "cost_accounting": {
            "candidate_G_new_training": {
                "physics_steps": sum(
                    report["training"]["simulation_physics_steps"]
                    for report in reports.values()
                ),
                "control_intervals": sum(
                    report["training"]["simulation_control_intervals"]
                    for report in reports.values()
                ),
            },
            "candidate_G_fixed_milestone_validation": {
                "rollouts": sum(len(report["validations"]) for report in reports.values()),
                "physics_steps": sum(len(report["validations"]) for report in reports.values()) * 400,
                "control_intervals": sum(len(report["validations"]) for report in reports.values()) * 40,
            },
            "candidate_G_exact_revalidation": {
                "rollouts": len(strict_events),
                "physics_steps": len(strict_events) * 400,
                "control_intervals": len(strict_events) * 40,
            },
            "candidate_G_training_context_reencoding": {
                "context_rebuilds": sum(
                    len(report["boundary_context_audit"]["epochs_prepared"])
                    for report in reports.values()
                ),
                "prefix_observation_forwards": sum(
                    len(report["boundary_context_audit"]["epochs_prepared"]) * 20
                    for report in reports.values()
                ),
                "physics_steps": 0,
            },
            "candidate_G_validation_context_reencoding": {
                "context_rebuilds": sum(len(report["validations"]) for report in reports.values()),
                "prefix_observation_forwards": sum(len(report["validations"]) * 20 for report in reports.values()),
                "physics_steps": 0,
            },
            "preflight_G0_derived_from_frozen_code_path": {
                "physics_steps": 5200,
                "control_intervals": 520,
                "prefix_observation_forwards": 160,
                "runtime_counter_was_not_serialized": True,
            },
            "historical_donor_training_and_selection": {
                "candidate_D_seed_count": 3,
                "fixed_budget_physics_steps_per_seed": 400000,
                "total_historical_training_physics_steps": 1200000,
                "charged_as_new_candidate_G_training": False,
                "equal_total_data_cost_claim_allowed": False,
            },
        },
        "decision": {
            "code": classification,
            "label": label,
            "action": action,
            "G2_final_threshold": 30,
            "G2_threshold_is_local_reporting_only": True,
            "chunk_commit_written": False,
        },
        "verification": {
            "repository_tests": args.repository_tests,
            "repository_subtests": args.repository_subtests,
            "warnings": args.repository_warnings,
        },
        "summarizer_sha256": sha256(Path(__file__)),
    }
    comparison_path = run_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    decision = {
        "schema": "taco_pour_candidate_G_policy_warm_start_decision_v1",
        "status": "completed_prefrozen_adjudication",
        "classification": classification,
        "label": label,
        "prefrozen_precedence": ["G0", "G1", "G2", "G3"],
        "all_three_seeds_completed_declared_budget": all_complete,
        "strict_success_events": strict_events,
        "final_successful_intervals": final_intervals,
        "final_seeds_at_least_30": seeds_at_least_30,
        "action": action,
        "scope": (
            "this actor-only warm-start package at the fixed budget; not all "
            "warm-start variants and not mathematical task infeasibility"
        ),
        "comparison": artifact(comparison_path),
        "promotion_request_authorized": classification == "G1",
        "automatic_budget_or_seed_extension_authorized": False,
        "per_endpoint_repair_authorized": False,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
        "chunk_commit_written": False,
    }
    decision_path = run_root / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")

    summary = {
        "schema": "taco_pour_candidate_G_policy_warm_start_summary_v1",
        "status": "complete",
        "classification": classification,
        "label": label,
        "milestone_successful_intervals_by_seed": {
            str(seed): {
                str(epoch): reports[seed]["validations"][str(epoch)]["summary"][
                    "successful_intervals"
                ]
                for epoch in MILESTONES
            }
            for seed in (0, 1, 2)
        },
        "final_successful_intervals": final_intervals,
        "median_successful_intervals_by_milestone": median_intervals,
        "comparison": artifact(comparison_path),
        "decision": artifact(decision_path),
        "strict_success": bool(strict_events),
        "chunk_commit_written": False,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    table = []
    for seed in (0, 1, 2):
        values = [
            reports[seed]["validations"][str(epoch)]["summary"][
                "successful_intervals"
            ]
            for epoch in MILESTONES
        ]
        table.append(
            f"| {seed} | " + " | ".join(map(str, values)) +
            f" | {reports[seed]['status']} |"
        )
    summary_md = "\n".join([
        "# Candidate G — policy warm start with recurrent boundary context",
        "",
        f"Formal classification: **{classification} — {label}**.",
        "",
        "Candidate G is a local, non-paper-faithful actor/RMS warm-start experiment. "
        "The critic and both optimizers are fresh, and the endpoint-40 RNN context is "
        "causally rebuilt from the committed source20–39 observation prefix.",
        "",
        "| seed | epoch 0 | 62 / 99.2k | 125 / 200k | 188 / 300.8k | 250 / 400k | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
        *table,
        "",
        "Milestone medians: `" + " → ".join(
            str(median_intervals[str(epoch)]) for epoch in MILESTONES
        ) + "`.",
        "",
        f"Final seeds reaching the local G2 reporting threshold (>=30/40): `{seeds_at_least_30}`.",
        "",
        "No arbitrary intermediate checkpoint was selected and no endpoint40→60 chunk was committed. "
        "The endpoint20→40 chunk remains the only committed chunk.",
        "",
        f"Repository verification: {args.repository_tests} tests + "
        f"{args.repository_subtests} subtests ({args.repository_warnings} warnings).",
        "",
    ])
    (run_root / "summary.md").write_text(summary_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
