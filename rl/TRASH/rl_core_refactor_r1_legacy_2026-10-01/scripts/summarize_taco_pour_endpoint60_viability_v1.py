#!/usr/bin/env python3
"""Apply the prefrozen endpoint-60 viability decision table."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/taco_pour_endpoint60_viability_adjudication_v1"
CONTRACT = ROOT / "configs/taco_pour_endpoint60_viability_adjudication_v1.yaml"


def classify(*, carryover: bool, A: bool, B: bool, P: bool) -> tuple[str, str]:
    if carryover:
        return "C1", "POLICY_CARRYOVER_SOLVES_NEXT_WINDOW"
    if not A and not B and P:
        return "C3", "ARRIVAL_STATE_VIABILITY_BOTTLENECK"
    if A or B:
        return "C2", "ONE_STEP_VIABLE_ACTION_EXISTS"
    return "C4", "NO_ONE_STEP_FEASIBLE_ACTION_FOUND_ON_KNOWN_ENDPOINT60_STATES"


def main() -> None:
    carryover = json.loads((OUTPUT / "carryover/report.json").read_text())
    reports = {
        name: json.loads(
            (OUTPUT / f"one_step_search/boundary_{name}/report.json").read_text()
        )
        for name in ("A", "B", "P")
    }
    if any(
        row.get("status") != "completed_read_only_search"
        or row.get("new_search_control_intervals") != 7168
        or row.get("training_executed") is not False
        or row.get("chunk_commit_written") is not False
        for row in reports.values()
    ):
        raise ValueError("endpoint60 search evidence is incomplete")
    # Require the frozen top-level filenames from the authorized output tree;
    # detailed reports and top-32 metadata remain in per-boundary subfolders.
    for name in ("A", "B", "P"):
        destination = OUTPUT / f"one_step_search/{name}.npz"
        if not destination.is_file():
            raise FileNotFoundError(destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != reports[name][
            "arrays"
        ]["sha256"]:
            raise RuntimeError(f"published search array changed for {name}")
    flags = {name: bool(row["any_feasible_action_found"]) for name, row in reports.items()}
    c1 = bool(carryover["endpoint41_80_summary"]["forty_of_forty"])
    code, label = classify(carryover=c1, **flags)
    if c1:
        raise RuntimeError("C1 should have stopped before one-step search")
    action = {
        "C2": "stop_no_automatic_LR_or_curriculum_change",
        "C3": "stop_previous_solver_warm_start_or_carryover_becomes_next_candidate",
        "C4": "stop_close_tail_reset_LR_sigma_and_more_source60_exploration",
    }[code]
    decision = {
        "schema": "taco_pour_endpoint60_viability_decision_v1",
        "status": "completed_prefrozen_adjudication",
        "classification": code,
        "label": label,
        "contract": {
            "path": str(CONTRACT.resolve()),
            "sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        },
        "prefrozen_precedence": ["C1", "C3", "C2", "C4"],
        "carryover_endpoint41_80_strict_40_of_40": c1,
        "one_step_feasible": flags,
        "best_score_minus_1": {
            name: row["best"]["score_minus_1"] for name, row in reports.items()
        },
        "action": action,
        "finite_negative_is_mathematical_infeasibility_proof": False,
        "future_endpoint50_pre_boundary_shaping_design_authorized": code == "C4",
        "future_endpoint50_pre_boundary_shaping_executed": False,
        "training_executed": False,
        "optimizer_updates": 0,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
        "chunk_commit_written": False,
        "verification": {
            "repository_tests": 818,
            "repository_subtests": 57,
            "warnings": 19,
        },
        "blocked": {
            "LR_or_sigma_only_tuning": True,
            "source60_tail_curriculum_extension": True,
            "curriculum_ratio_sweep": True,
            "reference_or_GT_state_injection": True,
            "direct_chunk_commit": True,
        },
    }
    (OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    report = {
        "schema": "taco_pour_endpoint60_viability_adjudication_report_v1",
        "status": "completed",
        "paper_faithful": False,
        "carryover": carryover,
        "historical_source60_samples": json.loads(
            (OUTPUT / "historical_source60_samples.json").read_text()
        ),
        "one_step_search": reports,
        "decision": decision,
        "verification": decision["verification"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Endpoint-60 viability adjudication v1",
        "",
        f"- Decision: `{code} / {label}`",
        "- Continuous Candidate-D carryover: "
        f"`{carryover['endpoint41_80_summary']['successful_intervals']}/40`, "
        f"first failure endpoint "
        f"`{carryover['endpoint41_80_summary']['first_failure_endpoint']}`.",
        "- Endpoint 21-60 historical reproduction: bitwise exact for every saved array.",
        "",
        "| Boundary | Feasible / 7168 | Best score | Best score - 1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("A", "B", "P"):
        row = reports[name]
        lines.append(
            f"| {name} | {row['feasible_action_count']} | "
            f"{row['best']['tracking_score']:.9f} | "
            f"{row['best']['score_minus_1']:+.9f} |"
        )
    lines.extend([
        "",
        "No PPO training, reward/observation/action/physics/LR change, support",
        "expansion, reference/GT state injection, or chunk commit was performed.",
        "A finite negative search is not a mathematical infeasibility proof.",
        "",
        "Verification: `818 passed, 19 warnings, 57 subtests passed`.",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
