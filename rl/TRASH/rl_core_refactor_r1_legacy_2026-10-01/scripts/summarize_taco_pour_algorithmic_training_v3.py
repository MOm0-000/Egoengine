#!/usr/bin/env python3
"""Build the immutable-facing A/B learning-curve comparison for benchmark v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v3"
MILESTONES = (("0", "0"), ("100k", "62"), ("500k", "313"), ("1M", "625"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(candidate: str, seed: int = 0) -> tuple[Path, dict] | None:
    path = RUN / "training" / f"candidate_{candidate}" / f"seed_{seed}" / "report.json"
    if not path.is_file():
        return None
    return path, json.loads(path.read_text())


def milestone(report: dict, epoch: str) -> dict:
    validation = report["validations"][epoch]
    return {
        "simulation_physics_steps": validation["simulation_physics_steps"],
        "validation": validation["summary"],
        "training_dynamics_cumulative": report["learning_curve"][epoch],
        "checkpoint_roundtrip_exact": validation["checkpoint"][
            "payload_roundtrip_exact"
        ],
    }


def interpretation(a_final: int | None, b_final: int | None) -> dict:
    if a_final is None or b_final is None:
        return {
            "status": "awaiting_both_seed0_candidates",
            "undertraining_hypothesis_supported": None,
            "replay_preserving_initialization_hypothesis_supported": None,
            "candidate_C_discussion_allowed": False,
        }
    if a_final > 30:
        status = "candidate_A_exceeds_Replay_supports_undertraining"
    elif b_final > 30:
        status = "only_candidate_B_exceeds_Replay_supports_replay_preserving_initialization"
    elif max(a_final, b_final) <= 30:
        status = "both_seed0_candidates_plateau_at_or_below_Replay"
    else:
        status = "training_objective_vs_deterministic_validation_requires_review"
    return {
        "status": status,
        "undertraining_hypothesis_supported": a_final > 30,
        "replay_preserving_initialization_hypothesis_supported": (
            a_final <= 30 and b_final > 30
        ),
        "candidate_C_discussion_allowed": a_final <= 30 and b_final <= 30,
    }


def main() -> None:
    b0_path = RUN / "candidate_B/seed_0/report.json"
    if not b0_path.is_file():
        raise FileNotFoundError("v3 Candidate-B zero-step report is missing")
    b0 = json.loads(b0_path.read_text())
    candidates: dict[str, dict | None] = {}
    report_artifacts: dict[str, dict] = {}
    final_counts: dict[str, int | None] = {}
    for candidate in ("A", "B"):
        loaded = load_candidate(candidate)
        if loaded is None:
            candidates[candidate] = None
            final_counts[candidate] = None
            continue
        path, report = loaded
        candidates[candidate] = {
            label: milestone(report, epoch) for label, epoch in MILESTONES
        }
        final_counts[candidate] = int(
            report["validations"]["625"]["summary"]["successful_intervals"]
        )
        report_artifacts[candidate] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    conditional: dict[str, dict] = {}
    for candidate in ("A", "B"):
        rows = {}
        for seed in (1, 2):
            loaded = load_candidate(candidate, seed)
            if loaded is not None:
                path, report = loaded
                rows[str(seed)] = {
                    "report": {"path": str(path.resolve()), "sha256": sha256(path)},
                    "final_validation": report["validations"]["625"]["summary"],
                }
        if rows:
            conditional[candidate] = rows

    decision = interpretation(final_counts["A"], final_counts["B"])
    complete = candidates["A"] is not None and candidates["B"] is not None
    comparison = {
        "schema": "taco_pour_algorithmic_reproduction_training_comparison_v3",
        "status": "seed0_A_B_complete" if complete else "seed0_training_in_progress",
        "classification": (
            "local_algorithmic_reproduction_after_replay_identity_and_numeric_"
            "likelihood_gate_fix"
        ),
        "paper_faithful": False,
        "Replay": b0["replay"],
        "Candidate_B_at_0": b0["candidate_B_step0"],
        "candidates": candidates,
        "candidate_reports": report_artifacts,
        "conditional_multiseed": conditional,
        "interpretation": decision,
        "strict_success_requires": "40_of_40",
        "chunk_commit_written": False,
        "old_checkpoint_warm_start_used": False,
    }
    path = RUN / "comparison.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(comparison, indent=2) + "\n")
    temporary.replace(path)

    lines = [
        "# TACO Pour Algorithmic Reproduction Training Benchmark v3",
        "",
        f"- Replay: `{b0['replay']['successful_intervals']}/40`, first failure "
        f"endpoint `{b0['replay']['first_failure_endpoint']}`.",
        f"- Candidate B at 0: `{b0['candidate_B_step0']['successful_intervals']}/40`.",
    ]
    for candidate in ("A", "B"):
        row = candidates[candidate]
        if row is None:
            lines.append(f"- Candidate {candidate}: not completed.")
            continue
        for label in ("100k", "500k", "1M"):
            summary = row[label]["validation"]
            lines.append(
                f"- Candidate {candidate}@{label}: "
                f"`{summary['successful_intervals']}/40`, first failure endpoint "
                f"`{summary['first_failure_endpoint']}`."
            )
    lines.extend([
        "",
        f"Interpretation: `{decision['status']}`.",
        "No chunk was committed.",
        "",
    ])
    (RUN / "summary.md").write_text("\n".join(lines))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
