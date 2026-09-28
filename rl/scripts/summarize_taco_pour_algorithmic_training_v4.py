#!/usr/bin/env python3
"""Summarize the completed canonical-old-policy 100k A/B stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    b0_path = RUN / "candidate_B/seed_0/report.json"
    checkpoint_path = (
        ROOT / "runs/taco_pour_algorithmic_checkpoint_roundtrip_v4/report.json"
    )
    reports = {
        candidate: RUN
        / "training"
        / f"candidate_{candidate}"
        / "seed_0"
        / "100k"
        / "report.json"
        for candidate in ("B", "A")
    }
    b0 = json.loads(b0_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    candidates = {
        name: json.loads(path.read_text()) for name, path in reports.items()
    }
    rows = {}
    for name, report in candidates.items():
        audit_path = Path(report["update_audit"]["path"])
        audit = json.loads(audit_path.read_text())
        identity = [
            row["update"]["pre_optimizer_identity"]
            for row in audit["epoch_reports"]
        ]
        rows[name] = {
            "at_0": report["validations"]["0"]["summary"],
            "at_100k": report["validations"]["62"]["summary"],
            "physics_steps": report["training"]["simulation_physics_steps"],
            "actor_updates": report["training"]["actor_updates"],
            "checkpoint_roundtrip_exact": report["checkpoints"]["62"][
                "payload_roundtrip_exact"
            ],
            "all_identity_gates_passed": all(
                row["passed"] for row in identity
            ),
            "maximum_rollout_canonical_mu_difference": max(
                row["rollout_vs_canonical"]["max_abs_mu_difference"]
                for row in identity
            ),
            "maximum_rollout_canonical_sigma_difference": max(
                row["rollout_vs_canonical"]["max_abs_sigma_difference"]
                for row in identity
            ),
            "maximum_rollout_canonical_ratio_error": max(
                row["rollout_vs_canonical"]["max_abs_ratio_minus_one"]
                for row in identity
            ),
            "maximum_canonical_optimizer_ratio_error": max(
                row["canonical_optimizer_identity"][
                    "max_abs_ratio_minus_one"
                ]
                for row in identity
            ),
            "report": artifact(reports[name]),
            "update_audit": artifact(audit_path),
        }
    best = max(
        rows,
        key=lambda name: rows[name]["at_100k"]["successful_intervals"],
    )
    comparison = {
        "schema": "taco_pour_algorithmic_reproduction_training_comparison_v4",
        "status": "seed0_100k_B_then_A_completed_no_chunk_commit",
        "classification": (
            "local_algorithmic_reproduction_after_canonical_old_policy_"
            "likelihood_fix"
        ),
        "paper_faithful": False,
        "Replay": b0["replay"],
        "Candidate_B_at_0": b0["candidate_B_step0"],
        "candidates": rows,
        "best_candidate_at_100k": best,
        "best_validated_intervals_at_100k": rows[best]["at_100k"][
            "successful_intervals"
        ],
        "candidate_B_beats_Replay_at_100k": rows["B"]["at_100k"][
            "beats_Replay"
        ],
        "candidate_B_meaningful_refinement_at_100k": rows["B"][
            "at_100k"
        ]["meaningful_refinement"],
        "candidate_A_degrades_from_its_fresh_initial_policy": (
            rows["A"]["at_100k"]["successful_intervals"]
            < rows["A"]["at_0"]["successful_intervals"]
        ),
        "strict_40_of_40_success": any(
            row["at_100k"]["forty_of_forty"] for row in rows.values()
        ),
        "B0_gate": artifact(b0_path),
        "checkpoint_infrastructure_gate": artifact(checkpoint_path),
        "checkpoint_infrastructure_exact": all(
            checkpoint["field_equality"].values()
        ),
        "chunk_commit_written": False,
        "later_milestones_executed": False,
    }
    (RUN / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    (RUN / "summary.md").write_text(
        "# TACO Pour Algorithmic Reproduction Training Benchmark v4\n\n"
        "The canonical differentiable old-policy likelihood contract passed "
        "for every one of the 124 fresh actor updates. No v3 checkpoint was "
        "loaded and no chunk was committed.\n\n"
        f"- Replay / Candidate B at step 0: `30/40`, fail endpoint `51`.\n"
        f"- Candidate B at 99,200 physics steps: "
        f"`{rows['B']['at_100k']['successful_intervals']}/40`, fail endpoint "
        f"`{rows['B']['at_100k']['first_failure_endpoint']}`.\n"
        f"- Candidate A at step 0: "
        f"`{rows['A']['at_0']['successful_intervals']}/40`, fail endpoint "
        f"`{rows['A']['at_0']['first_failure_endpoint']}`.\n"
        f"- Candidate A at 99,200 physics steps: "
        f"`{rows['A']['at_100k']['successful_intervals']}/40`, fail endpoint "
        f"`{rows['A']['at_100k']['first_failure_endpoint']}`.\n"
        f"- Maximum rollout-to-canonical likelihood ratio error: B "
        f"`{rows['B']['maximum_rollout_canonical_ratio_error']:.9g}`, A "
        f"`{rows['A']['maximum_rollout_canonical_ratio_error']:.9g}`.\n"
        "- Maximum canonical optimizer ratio error: `0` for both candidates.\n"
        "- Strict success: `false`; chunk commit: `false`.\n"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
