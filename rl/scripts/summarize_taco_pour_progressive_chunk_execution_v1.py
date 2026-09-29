#!/usr/bin/env python3
"""Summarize the frozen endpoint-40 progressive solve without selecting a run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/taco_pour_progressive_chunk_execution_v1"
MILESTONES = (("0", "0"), ("100k", "62"), ("200k", "125"), ("300.8k", "188"), ("400k", "250"))


def artifact(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def main() -> None:
    promotion_path = ROOT / "runs/taco_pour_candidate_D_strict_success_promotion_v1/report.json"
    replay_path = RUN_ROOT / "chunk_source_40/replay/report.json"
    reports = {
        seed: RUN_ROOT / f"chunk_source_40/candidate_D/seed_{seed}/report.json"
        for seed in (0, 1, 2)
    }
    for path in (promotion_path, replay_path, *reports.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    promotion = json.loads(promotion_path.read_text())
    replay = json.loads(replay_path.read_text())
    seeds: dict[str, object] = {}
    curves: dict[int, list[int]] = {}
    strict_events: list[dict[str, int]] = []
    total_steps = 0
    for seed, path in reports.items():
        report = json.loads(path.read_text())
        if (
            report["status"] != "completed_400k_without_strict_success_no_chunk_commit"
            or report["seed"] != seed
            or report["source_endpoint"] != 40
            or report["lookahead_endpoint"] != 80
            or report["training"]["simulation_physics_steps"] != 400000
            or report["chunk_commit_written"] is not False
        ):
            raise ValueError(f"seed {seed} is not a completed fail-closed restart")
        total_steps += int(report["training"]["simulation_physics_steps"])
        curve: dict[str, object] = {}
        for label, epoch in MILESTONES:
            validation = report["validations"][epoch]
            summary = validation["summary"]
            successful = int(summary["successful_intervals"])
            curves.setdefault(int(epoch), []).append(successful)
            if summary["forty_of_forty"]:
                strict_events.append({"seed": seed, "epoch": int(epoch)})
            curve[label] = {
                "epoch": int(epoch),
                "physics_steps": int(validation["simulation_physics_steps"]),
                "successful_intervals": successful,
                "first_failure_endpoint": summary["first_failure_endpoint"],
                "forty_of_forty": bool(summary["forty_of_forty"]),
                "bounded_mu_outside_support_count": int(
                    summary["bounded_mu_outside_support_count"]
                ),
                "trajectory": validation["trajectory"],
                "checkpoint": validation["checkpoint"],
            }
        seeds[str(seed)] = {
            "seed": seed,
            "status": report["status"],
            "report": artifact(path),
            "fresh_actor_critic_optimizers_RMS_RNN": report[
                "fresh_actor_critic_optimizers_RMS_RNN"
            ],
            "warm_start_used": report["warm_start_used"],
            "learning_curve": curve,
        }

    medians = {
        label: median(curves[int(epoch)]) for label, epoch in MILESTONES
    }
    comparison = {
        "schema": "taco_pour_progressive_chunk_execution_comparison_v1",
        "status": "endpoint40_candidate_D_restart_protocol_exhausted_no_chunk_commit",
        "paper_faithful": False,
        "classification": "local_progressive_chunk_generation_failure_evidence",
        "committed_prefix": {"source_endpoint": 20, "target_endpoint": 40},
        "promotion": artifact(promotion_path),
        "promotion_status": promotion["status"],
        "endpoint40_Replay": {
            "report": artifact(replay_path),
            "successful_intervals": replay["summary"]["successful_intervals"],
            "first_failure_endpoint": replay["summary"]["first_failure_endpoint"],
            "chunk_commit_written": replay["chunk_commit_written"],
        },
        "candidate_D_restarts": seeds,
        "fixed_milestone_medians": medians,
        "strict_fixed_milestone_events": strict_events,
        "strict_success_found": not not strict_events,
        "restart_seed_order": [0, 1, 2],
        "restart_protocol_exhausted": True,
        "chunk_40_60_committed": False,
        "generation_solver_physics_steps_after_endpoint40": total_steps,
        "eligible_for_paper_cost_comparison": False,
        "next_action": (
            "stop_progression_and_review_optimizer_only_under_separate_authorization"
        ),
        "automatic_LR_schedule_executed": False,
        "non_strict_checkpoint_selected": False,
    }
    comparison_path = RUN_ROOT / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    rows = []
    for seed in (0, 1, 2):
        curve = seeds[str(seed)]["learning_curve"]
        rows.append(
            f"| {seed} | "
            + " | ".join(str(curve[label]["successful_intervals"]) for label, _ in MILESTONES)
            + " | no strict success |"
        )
    summary = "\n".join(
        [
            "# Progressive Candidate-D chunk solve: endpoint 40",
            "",
            "The promoted source20 solver committed only endpoint21–40. Zero-residual Replay from the exact endpoint40 boundary passed 9/40 and wrote no endpoint60 commit.",
            "",
            "| seed | step0 | 100k | 200k | 300.8k | 400k | result |",
            "|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "Fixed-milestone medians: "
            + " → ".join(str(medians[label]) for label, _ in MILESTONES)
            + ".",
            "",
            "No seed reached strict 40/40 at any predeclared fixed milestone. The three-restart protocol consumed 1,200,000 training physics steps after endpoint40; no non-strict checkpoint was selected and endpoint41–60 remains uncommitted.",
            "",
            "The frozen protocol now permits review of an optimizer change, but does not authorize or execute one automatically. These local runs are not eligible for paper Cost comparison.",
        ]
    )
    (RUN_ROOT / "summary.md").write_text(summary + "\n")
    print(json.dumps({"comparison": artifact(comparison_path), "strict_events": strict_events}, indent=2))


if __name__ == "__main__":
    main()
