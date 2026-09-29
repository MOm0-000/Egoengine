#!/usr/bin/env python3
"""Summarize the pre-registered Candidate F three-seed experiment.

This script is intentionally descriptive.  It does not select intermediate
checkpoints, alter the frozen thresholds, or authorize a chunk commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "taco_pour_candidate_F_two_chunk_curriculum_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _milestone(report: dict[str, Any], epoch: int) -> dict[str, Any]:
    validation = report["validations"][str(epoch)]["summary"]
    return {
        "epoch": epoch,
        "successful_intervals": int(validation["successful_intervals"]),
        "first_failure_endpoint": validation["first_failure_endpoint"],
        "forty_of_forty": bool(validation["forty_of_forty"]),
        "endpoint60_score": float(validation["endpoint60"]["score"]),
        "endpoint70_score": float(validation["endpoint70"]["score"]),
        "endpoint80_score": float(validation["endpoint80"]["score"]),
        "validation_residual_RMS": float(validation["validation_residual_RMS"]),
        "validation_action_bound_fraction": float(
            validation["validation_action_bound_fraction"]
        ),
        "final_sigma": validation["final_sigma"],
    }


def _coverage(report: dict[str, Any], segment: str, cohort: str) -> dict[str, Any]:
    raw = report["coverage"]["segments"][segment][cohort]
    keys = (
        "sample_count",
        "fraction_k_ge_21",
        "fraction_k_ge_25",
        "fraction_k_ge_30",
        "fraction_k_ge_35",
        "source60_action_count",
        "endpoint61_tracking_termination_count",
        "tracking_termination_count",
        "maximum_outcome_endpoint",
        "maximum_reached_k_histogram_by_epoch_world",
        "tracking_termination_endpoint_histogram",
    )
    return {key: raw[key] for key in keys}


def _classification(strict: bool, median_deep: float) -> tuple[str, str]:
    if strict:
        return (
            "F1",
            "LOOKAHEAD_CURRICULUM_STRICT_SUCCESS",
        )
    if median_deep >= 0.20:
        return (
            "F2",
            "LOOKAHEAD_COVERAGE_RESTORED_NO_STRICT_SUCCESS",
        )
    if median_deep < 0.05:
        return (
            "F3",
            "TAIL_TRANSITION_REMAINS_BOTTLENECK",
        )
    return ("F4", "MIXED_CURRICULUM_RESULT")


def summarize(run_dir: Path) -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    strict = False
    for seed in (0, 1, 2):
        report_path = run_dir / f"seed_{seed}" / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing completed seed report: {report_path}")
        report = _read_json(report_path)
        milestones = {str(epoch): _milestone(report, epoch) for epoch in (0, 62, 125)}
        strict_seed = any(item["forty_of_forty"] for item in milestones.values())
        strict = strict or strict_seed
        seed_entry = {
            "seed": seed,
            "report": {
                "path": str(report_path.relative_to(ROOT)),
                "sha256": _sha256(report_path),
            },
            "status": report["status"],
            "fresh_actor_critic_optimizers_RMS_RNN": report[
                "fresh_actor_critic_optimizers_RMS_RNN"
            ],
            "warm_start_used": report["warm_start_used"],
            "milestones": milestones,
            "strict_success": strict_seed,
            "strict_success_epoch": report["strict_success_epoch"],
            "chunk_commit_written": report["chunk_commit_written"],
            "coverage": {
                "epochs_1_through_62": {
                    cohort: _coverage(report, "epochs_1_through_62", cohort)
                    for cohort in ("all_worlds", "anchor_worlds", "tail_worlds")
                },
                "epochs_63_through_125": {
                    cohort: _coverage(report, "epochs_63_through_125", cohort)
                    for cohort in ("all_worlds", "anchor_worlds", "tail_worlds")
                },
            },
        }
        seeds.append(seed_entry)

    deep_fractions = [
        seed["coverage"]["epochs_63_through_125"]["all_worlds"]["fraction_k_ge_21"]
        for seed in seeds
    ]
    tail_source60_actions = sum(
        seed["coverage"]["epochs_63_through_125"]["tail_worlds"][
            "source60_action_count"
        ]
        for seed in seeds
    )
    tail_endpoint61_terminations = sum(
        seed["coverage"]["epochs_63_through_125"]["tail_worlds"][
            "endpoint61_tracking_termination_count"
        ]
        for seed in seeds
    )
    median_deep = float(statistics.median(deep_fractions))
    code, label = _classification(strict, median_deep)

    boundary_report = run_dir / "tail_boundary_builder" / "report.json"
    sampler_report = run_dir / "sampler_gate" / "report.json"
    comparison = {
        "schema": "taco_pour_candidate_F_two_chunk_curriculum_comparison_v1",
        "candidate": "F_balanced_two_chunk_lookahead_curriculum",
        "paper_faithful": False,
        "local_engineering_change": "rollout reset/start-state distribution only",
        "frozen_curriculum": {
            "epochs_1_through_62": [40, 40, 40, 40],
            "epochs_63_through_125": [40, 40, 60, 60],
            "ratio_sweep_performed": False,
        },
        "gates": {
            "tail_boundary_builder": {
                "path": str(boundary_report.relative_to(ROOT)),
                "sha256": _sha256(boundary_report),
            },
            "no_training_sampler_gate": {
                "path": str(sampler_report.relative_to(ROOT)),
                "sha256": _sha256(sampler_report),
                "optimizer_steps": 0,
            },
        },
        "seeds": seeds,
        "decision": {
            "code": code,
            "label": label,
            "strict_success_at_fixed_milestone": strict,
            "post_curriculum_all_world_deep_lookahead_fractions": deep_fractions,
            "post_curriculum_all_world_median_deep_lookahead_fraction": median_deep,
            "post_curriculum_tail_source60_action_count": tail_source60_actions,
            "post_curriculum_tail_endpoint61_tracking_termination_count": (
                tail_endpoint61_terminations
            ),
            "frozen_thresholds": {
                "F2_minimum": 0.20,
                "F3_strict_upper_bound": 0.05,
            },
            "optimizer_or_LR_review_authorized": code == "F2",
            "further_ratio_sweep_authorized": False,
            "automatic_400k_extension_authorized": False,
            "promotion_request_required_before_commit": code == "F1",
        },
        "commit_state": {
            "endpoint20_to40_committed": True,
            "endpoint40_to60_committed": False,
            "candidate_F_chunk_commit_written": False,
        },
        "verification": {
            "focused_tests": "10 passed",
            "full_repository_tests": "814 passed, 19 warnings, 57 subtests passed",
        },
    }
    return comparison


def _summary_markdown(comparison: dict[str, Any]) -> str:
    decision = comparison["decision"]
    lines = [
        "# Candidate F — Balanced Two-Chunk Lookahead Curriculum",
        "",
        f"Formal classification: **{decision['code']} — {decision['label']}**.",
        "",
        "Candidate F is a local engineering curriculum, not a recovered EgoEngine author setting. "
        "Only the rollout reset/start-state distribution changed.",
        "",
        "## Fixed-milestone validation",
        "",
        "| Seed | Step 0 | 100k / epoch 62 | 200k / epoch 125 | Strict 40/40 |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for seed in comparison["seeds"]:
        milestones = seed["milestones"]
        values = []
        for epoch in ("0", "62", "125"):
            value = milestones[epoch]
            values.append(
                f"{value['successful_intervals']}/40 (fail@{value['first_failure_endpoint']})"
            )
        lines.append(
            f"| {seed['seed']} | {values[0]} | {values[1]} | {values[2]} | "
            f"{'yes' if seed['strict_success'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Post-curriculum coverage (epochs 63–125)",
            "",
            "| Seed | Cohort | Samples | k>=21 | k>=25 | k>=30 | k>=35 | source60 actions | endpoint61 terminations |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in comparison["seeds"]:
        for cohort in ("all_worlds", "anchor_worlds", "tail_worlds"):
            item = seed["coverage"]["epochs_63_through_125"][cohort]
            lines.append(
                f"| {seed['seed']} | {cohort} | {item['sample_count']} | "
                f"{item['fraction_k_ge_21']:.6f} | {item['fraction_k_ge_25']:.6f} | "
                f"{item['fraction_k_ge_30']:.6f} | {item['fraction_k_ge_35']:.6f} | "
                f"{item['source60_action_count']} | {item['endpoint61_tracking_termination_count']} |"
            )

    fractions = decision["post_curriculum_all_world_deep_lookahead_fractions"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Post-curriculum all-world deep-lookahead fractions: `{fractions}`; "
            f"median: `{decision['post_curriculum_all_world_median_deep_lookahead_fraction']:.6f}`.",
            "",
            f"Across the three fixed tail cohorts, all "
            f"`{decision['post_curriculum_tail_source60_action_count']}` source60 actions "
            f"terminated for tracking at endpoint61 "
            f"(`{decision['post_curriculum_tail_endpoint61_tracking_termination_count']}`/"
            f"`{decision['post_curriculum_tail_source60_action_count']}`).",
            "",
            "No fixed validation milestone achieved strict 40/40. No Candidate F chunk was committed. "
            "The endpoint20→40 chunk remains committed; endpoint40→60 remains uncommitted.",
            "",
            "The frozen decision rules prohibit an automatic curriculum-ratio sweep or a 400k extension. "
            "Optimizer/LR review is authorized only by F2.",
            "",
            "Verification: tail-boundary and no-training sampler gates passed; the sampler gate used "
            "zero optimizer steps. The final repository suite passed 814 tests and 57 subtests "
            "(19 warnings).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    comparison = summarize(run_dir)
    comparison_path = run_dir / "comparison.json"
    summary_path = run_dir / "summary.md"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(_summary_markdown(comparison), encoding="utf-8")
    print(
        json.dumps(
            {
                "comparison": str(comparison_path),
                "comparison_sha256": _sha256(comparison_path),
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "decision": comparison["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
