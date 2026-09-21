#!/usr/bin/env python3
"""Re-score an existing Pour trace under unresolved threshold mappings.

This is an audit only.  It does not select a replacement objective or change
the Replay/PPO result that produced the trace.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POSITION_THRESHOLD_M = 0.12
ROTATION_THRESHOLD_RAD = 1.5


def trace_steps(report: dict) -> tuple[list[dict], str]:
    stitched = report.get("stitched_trajectory_validation")
    if stitched is not None:
        return stitched["steps"], "independently_replayed_stitched_trajectory"
    if "trace" in report:
        return report["trace"]["steps"], "standalone_validation_trace"

    steps: list[dict] = []
    for chunk in report.get("chunks", []):
        start = int(chunk["start"])
        committed_end = int(chunk.get("committed_end", start))
        selected_mode = chunk.get("mode")
        if committed_end <= start or selected_mode is None:
            continue
        selected = [
            trace for trace in chunk["validation_traces"]
            if trace["mode"] == selected_mode and trace["feasible"]
        ]
        if len(selected) != 1:
            raise ValueError(f"cannot resolve selected trace at chunk {start}")
        steps.extend(selected[0]["steps"][:committed_end - start])
    return steps, "committed_chunk_prefix"


def candidate_definitions(report: dict) -> list[dict]:
    tracking = report["objective"]["tracking"]
    return [
        {
            "name": "current_runtime_local_unpublished",
            "formula": "sqrt(lambda_p * ep^2 + lambda_R * eR^2) <= C",
            "lambda_p": float(tracking["lambda_p"]),
            "lambda_R": float(tracking["lambda_R"]),
            "C": float(tracking["C"]),
            "paper_formula_compatible": True,
            "adopted": True,
        },
        {
            "name": "axis_intercept_normalized_ellipse",
            "formula": "sqrt((ep / 0.12)^2 + (eR / 1.5)^2) <= 1",
            "lambda_p": 1.0 / POSITION_THRESHOLD_M**2,
            "lambda_R": 1.0 / ROTATION_THRESHOLD_RAD**2,
            "C": 1.0,
            "paper_formula_compatible": True,
            "adopted": False,
        },
        {
            "name": "corner_normalized_ellipse",
            "formula": "sqrt((ep / 0.12)^2 + (eR / 1.5)^2) <= sqrt(2)",
            "lambda_p": 1.0 / POSITION_THRESHOLD_M**2,
            "lambda_R": 1.0 / ROTATION_THRESHOLD_RAD**2,
            "C": math.sqrt(2.0),
            "paper_formula_compatible": True,
            "adopted": False,
        },
        {
            "name": "independent_threshold_box",
            "formula": "ep <= 0.12 and eR <= 1.5",
            "paper_formula_compatible": False,
            "adopted": False,
        },
    ]


def passes(candidate: dict, position: float, rotation: float) -> bool:
    if candidate["name"] == "independent_threshold_box":
        return position <= POSITION_THRESHOLD_M and rotation <= ROTATION_THRESHOLD_RAD
    error = math.sqrt(
        candidate["lambda_p"] * position**2
        + candidate["lambda_R"] * rotation**2
    )
    return error <= candidate["C"]


def score(candidate: dict, steps: list[dict]) -> dict:
    roles = steps[0]["tracked_object_roles"]
    per_role = {}
    all_step_passes = []
    for role_index, role in enumerate(roles):
        flags = [
            passes(candidate, row["position_error_m"][role_index], row["rotation_error_rad"][role_index])
            for row in steps
        ]
        failures = [row for row, passed in zip(steps, flags) if not passed]
        per_role[role] = {
            "valid_steps": sum(flags),
            "violation_steps": len(failures),
            "first_violation_endpoint": failures[0]["endpoint"] if failures else None,
            "maximum_position_error_m": max(row["position_error_m"][role_index] for row in steps),
            "maximum_rotation_error_rad": max(row["rotation_error_rad"][role_index] for row in steps),
        }
        all_step_passes.append(flags)

    combined = [all(flags[index] for flags in all_step_passes) for index in range(len(steps))]
    failures = [row for row, passed in zip(steps, combined) if not passed]
    return {
        "definition": candidate,
        "all_tracked_objects": {
            "valid_steps": sum(combined),
            "violation_steps": len(failures),
            "first_violation_endpoint": failures[0]["endpoint"] if failures else None,
            "full_trace_feasible": all(combined),
        },
        "per_object_role": per_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    report = json.loads(args.input_report.read_text())
    steps, source = trace_steps(report)
    if not steps:
        raise ValueError("input report contains no executed trace steps")
    endpoints = [int(row["endpoint"]) for row in steps]
    if endpoints != list(range(1, len(steps) + 1)):
        raise ValueError("audit requires one contiguous trace beginning at endpoint 1")

    result = {
        "schema": "taco_pour_objective_mapping_sensitivity_v1",
        "status": "diagnostic_only_no_mapping_selected",
        "source_report": str(args.input_report.resolve()),
        "trace_source": source,
        "trace_steps": len(steps),
        "published_example_thresholds": {
            "position_m": POSITION_THRESHOLD_M,
            "rotation_rad": ROTATION_THRESHOLD_RAD,
        },
        "paper_limit": (
            "EgoEngine Appendix C publishes the weighted-error formula and the Pour example "
            "thresholds, but not lambda_p, lambda_R, C, or their mapping."
        ),
        "candidates": [score(candidate, steps) for candidate in candidate_definitions(report)],
        "decision": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "trace_steps": len(steps),
        "candidate_results": {
            row["definition"]["name"]: row["all_tracked_objects"]
            for row in result["candidates"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
