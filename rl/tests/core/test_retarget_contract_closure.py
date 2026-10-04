from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from egoengine_repro.retarget.contract_closure import (
    final_classification,
    objective_semantic_unit_tests,
    prior_native_stricter_breakdown,
    project_world_points,
)


ROOT = Path(__file__).resolve().parents[2]


def test_closure_contract_is_strictly_zero_runtime():
    value = yaml.safe_load((ROOT / "configs/taco_pour_retarget_contract_closure_v2.yaml").read_text())
    assert value["schema"] == "taco_pour_retarget_contract_closure_v2"
    assert value["expected_baseline"] == "32e0bbe27ac2c3c6844d409654f34f4fe6e13f50"
    assert not any(value["authorization"].values())
    assert value["frames"]["rgb_reconciliation"] == [0, 14, 15, 16, 17, 20]


def test_prior_collision_recount_parses_rows_instead_of_summary():
    pair = ["left_hand_link_visual", "left_thumb_rota1_visual"]
    rows = [{
        "classification": "NATIVE_STRICTER_THAN_PROXY", "state": "OLD", "endpoint": "20",
        "candidate_relevant_findings": json.dumps({
            "left_hand_object": [], "left_hand_table": ["left_ring_link2_visual"],
            "left_omitted_nonadjacent": [pair], "left_unclassified_omitted_nonadjacent": [],
        }),
    }, {
        "classification": "PROXY_AND_NATIVE_AGREE_LEGAL", "state": "OTHER", "endpoint": "0",
        "candidate_relevant_findings": "this row must not be parsed",
    }]
    report = prior_native_stricter_breakdown(rows)
    assert report["native_stricter_row_count"] == 1
    assert report["rows_containing_palm_thumb_pair"] == 1
    assert report["left_hand_table_row_count"] == 1
    assert report["left_hand_table_finding_count"] == 1
    assert report["other_omitted_self_pair_count"] == 0


def test_objective_components_pass_their_own_duties():
    rows = objective_semantic_unit_tests()
    expected = {
        "fingertip_position_error_m", "fingertip_orientation_error_rad",
        "wrist_to_tip_vector_error_m", "thumb_to_tip_vector_error_m",
        "direct_proximal_orientation_error_rad", "direct_distal_orientation_error_rad",
        "near_interaction_position_error_m", "near_interaction_orientation_error_rad",
        "wrist_tray_position_error_m", "wrist_orientation_error_rad",
        "joint_nominal_deviation_rad_l2", "temporal_joint_change_rad_l2",
    }
    assert {row["component"] for row in rows} == expected
    assert all(row["classification"] == "SEMANTICALLY_CERTIFIED" for row in rows)
    assert all(all(row["checks"].values()) for row in rows)


def test_released_camera_projection_has_no_implicit_inverse_or_offset():
    points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    pixels, depth = project_world_points(points, intrinsic, transform)
    assert np.allclose(pixels, [[50.0, 40.0], [100.0, 40.0]])
    assert np.allclose(depth, [2.0, 2.0])
    with pytest.raises(ValueError, match="behind"):
        project_world_points([[0, 0, -1]], intrinsic, transform)


def test_final_decision_fails_closed_and_never_hides_multiple_blockers():
    assert final_classification(collision_closed=True, objective_closed=True, source_rgb_closed=True) == (
        "RETARGET_CONTRACT_V2_CERTIFIED_NO_CANDIDATE", []
    )
    classification, blockers = final_classification(
        collision_closed=False, objective_closed=True, source_rgb_closed=False
    )
    assert classification == "MULTIPLE_CONTRACT_BLOCKERS_REMAIN"
    assert blockers == ["COLLISION_SEMANTICS_BLOCKER_REMAINS", "SOURCE_RGB_ALIGNMENT_BLOCKER"]


def test_closure_runner_contains_no_runtime_or_candidate_generation():
    source = (ROOT / "scripts/audit_taco_pour_retarget_contract_closure_v2.py").read_text()
    assert "mj_step(" not in source
    assert ".step(" not in source
    assert "scipy.optimize" not in source
    assert "minimize(" not in source
    assert '"physics": 0' in source
    assert '"candidate": 0' in source
    assert '"promotion": 0' in source


def test_frozen_closure_evidence_closes_a_and_b_but_fails_closed_on_c():
    evidence = ROOT / "runs/taco_pour_retarget_contract_closure_v2"
    decision = json.loads((evidence / "decision.json").read_text())
    phases = json.loads((evidence / "phase_contract.json").read_text())
    overlap = json.loads((evidence / "structural_overlap_policy.json").read_text())
    objective = json.loads((evidence / "objective_component_semantics.json").read_text())
    rgb = json.loads((evidence / "rgb_reconciliation.json").read_text())
    assert decision["classification"] == "SOURCE_RGB_ALIGNMENT_BLOCKER"
    assert decision["blockers"] == ["SOURCE_RGB_ALIGNMENT_BLOCKER"]
    assert phases["phase_A_collision"]["closed"] is True
    assert phases["phase_A_collision"]["summary"]["runtime_collision_representation_complete"] is False
    assert phases["phase_A_collision"]["summary"]["uncategorized_mismatch_count"] == 0
    assert phases["phase_B_objective"]["closed"] is True
    assert phases["phase_C_source_rgb"]["closed"] is False
    assert not any(phases["runtime_counts"].values())
    assert overlap["pair_classification"]["classification"] == "TRUE_SELF_COLLISION"
    assert overlap["structural_overlap_allowlist"] == []
    assert objective["certified"] is True
    assert rgb["pixel_reprojection_error"] is None
    assert rgb["manual_offset_applied"] is False
    assert not (evidence / "retarget_v2_contract.yaml").exists()

    hashes = {
        relative: expected for expected, relative in (
            line.split(maxsplit=1)
            for line in (evidence / "server_artifacts.sha256").read_text().splitlines()
        )
    }
    for relative in ("decision.json", "phase_contract.json", "summary.md", "collision_semantics_v2.csv"):
        digest = __import__("hashlib").sha256((evidence / relative).read_bytes()).hexdigest()
        assert digest == hashes[relative]
