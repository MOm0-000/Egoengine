"""Evidence regressions for the single authorized Brush MINK v2 candidate.

These tests deliberately read the immutable run artifacts.  They protect the
scientific gates that motivated v2 without invoking a second retarget
candidate from the default test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "rl/runs/taco_brush_environment_aware_mink_v2"


def _json(name: str) -> dict:
    return json.loads((RUN / name).read_text())


def _jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (RUN / name).read_text().splitlines()
        if line.strip()
    ]


def test_v2_effective_contract_is_runtime_derived_and_action_free() -> None:
    contract = _json("effective_retarget_contract.json")
    audit = _json("config_consumption_audit.json")

    assert contract["tasks"]["wrist"]["position"] == "ABSENT"
    assert contract["tasks"]["posture"] == "ABSENT"
    assert contract["hard_limits"]["configuration_limit"] == "ConfigurationLimit"
    assert contract["hard_limits"]["self_collision"]["present"] is True
    assert contract["hard_limits"]["frame_displacement_limit"] == "FrameDisplacementLimit"
    assert contract["hard_limits"]["native_support"]["present"] is True
    assert contract["action_optimization_config_dependencies"] == []
    assert audit["unknown_key_count"] == audit["unused_key_count"] == 0
    assert audit["declared_key_count"] == audit["consumed_key_count"] == 11


def test_v2_refactor_equivalence_precedes_scientific_delta() -> None:
    report = _json("refactor_equivalence_report.json")
    assert report["status"] == "PASS"
    assert report["frames"] == list(range(13))
    assert report["qpos_max_abs_difference"] == 0.0
    assert report["fingertip_error_max_abs_difference_m"] == 0.0
    assert report["self_collision_max_abs_difference_m"] == 0.0
    assert report["native_support_max_abs_difference_m"] == 0.0


def test_v2_closes_frame13_native_mesh_regression() -> None:
    trace = _jsonl("feasibility_closure_trace.jsonl")
    acceptance = {row["frame"]: row for row in _jsonl(
        "per_frame_native_acceptance.jsonl"
    )}
    row = next(
        item for item in trace
        if item["frame"] == 13 and item["iteration"] == 0
    )

    assert row["before"]["native_minimum_m"] == pytest.approx(
        -0.00010077589910284512, abs=1e-15
    )
    assert row["before"]["native_pass"] is False
    assert row["after"]["native_minimum_m"] == pytest.approx(
        -0.000015116381758661923, abs=1e-15
    )
    assert row["after"]["native_pass"] is True
    assert acceptance[13]["accepted"] is True
    assert acceptance[13]["closure_iterations"] == 1


def test_v2_rechecks_the_active_full_mesh_support_vertex_each_frame() -> None:
    acceptance = _jsonl("per_frame_native_acceptance.jsonl")
    active = [
        (row["native_geom"], row["native_vertex_index"])
        for row in acceptance
    ]
    switches = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(active, active[1:]))
        if before != after
    ]

    assert len(acceptance) == 209
    assert len(switches) == 88
    assert active[17] == ("left_thumb_rota2_visual", 834)
    assert active[18] == ("left_thumb_rota2_visual", 1626)
    assert all(row["native_pass"] for row in acceptance)


def test_v2_closes_simultaneous_self_and_native_support_failures() -> None:
    trace = _jsonl("feasibility_closure_trace.jsonl")
    simultaneous = [
        row for row in trace
        if not row["before"]["self_pass"]
        and not row["before"]["native_pass"]
    ]

    assert [row["frame"] for row in simultaneous] == [14, 16]
    assert all(row["after"]["all_pass"] for row in simultaneous)


def test_v2_preserves_frame_displacement_after_repeated_closure() -> None:
    trace = _jsonl("feasibility_closure_trace.jsonl")
    acceptance = _jsonl("per_frame_native_acceptance.jsonl")
    repeated = [row for row in acceptance if row["closure_iterations"] > 1]

    assert max(row["closure_iterations"] for row in repeated) == 5
    assert all(row["frame_displacement_pass"] for row in repeated)
    assert all(row["before"]["frame_displacement_pass"] for row in trace)
    assert all(row["after"]["frame_displacement_pass"] for row in trace)


def test_v2_stops_before_physics_and_retains_external_brush_blocker() -> None:
    decision = _json("decision.json")
    blockers = _json("remaining_external_static_blockers.json")

    assert decision["classification"] == (
        "ENVIRONMENT_AWARE_MINK_V2_HAND_FLOOR_CLOSED_EXTERNAL_BLOCKER_REMAINS"
    )
    assert decision["candidate_complete"] is True
    assert decision["candidate_frames_available"] == 209
    assert decision["runtime_counts"] == {
        "architecture_refactor": 1,
        "new_retarget_candidates": 1,
        "physics": 0,
        "replay": 0,
        "mpc": 0,
        "rl": 0,
        "promotion": 0,
        "chunk_commit": 0,
    }
    assert blockers["objects"]["brush"]["frame0_minimum_signed_distance_m"] == pytest.approx(
        -0.0013113098847677973, abs=1e-15
    )
