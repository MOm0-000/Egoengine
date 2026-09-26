import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source46_state_entry_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source46_state_entry_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_source46_gate_is_read_only_and_trace_bound():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert contract["frozen_half_LR_candidate"]["remains_blocked"] is True
    assert report["regression_gate"] == {
        "formal_Replay_trace_bitwise_reproduced": True,
        "formal_PPO_trace_bitwise_reproduced": True,
        "restored_source46_suffix_bitwise_reproduced": True,
        "restored_source46_physical_records_exact": True,
        "formal_Replay_successful_intervals": 30,
        "formal_PPO_successful_intervals": 28,
        "formal_PPO_endpoint49_score": 1.1142933368682861,
    }


def test_source46_branch_results_match_predeclared_gate():
    expected = {
        "zero_R_forearm_ty_residual_only": (False, 0.9580519199371338, 50),
        "zero_right_wrist_translation": (False, 0.9875005483627319, 50),
        "zero_complete_right_wrist": (False, 0.9733787775039673, 50),
        "zero_entire_right_hand_residual": (True, 1.0295884609222412, 49),
        "zero_full_36d_residual": (True, 1.0445936918258667, 49),
    }
    report = _report()
    assert {row["name"] for row in report["branches"]} == set(expected)
    for row in report["branches"]:
        contact, score, failure = expected[row["name"]]
        assert row["untargeted_action_dimensions_bitwise_unchanged"] is True
        assert row["endpoint_48_has_live_right_hand_tool_contact"] is contact
        assert row["endpoint_49_score"] == score
        assert row["first_failure_endpoint_or_60"] == failure
        assert row["passes_predeclared_gate"] is False
        if contact:
            assert row["outcome_states"]["48"]["contact"][
                "active_right_tool_fingers"
            ] == ["index"]
        else:
            assert row["outcome_states"]["48"]["contact"][
                "active_right_tool_fingers"
            ] == []


def test_state_entry_metrics_are_separate_and_do_not_use_positive_geom_gap():
    report = _report()
    assert report["measurement_boundaries"] == {
        "positive_mesh_surface_gap_computed": False,
        "mj_geomDistance_used": False,
        "no_contact_pose_surrogates": [
            "right_fingertips_in_tool_frame",
            "right_wrist_pose_relative_to_tool",
            "right_wrist_velocity_relative_to_tool",
        ],
        "endpoint47_state_distance_is_not_a_success_criterion": True,
        "mixed_unit_combined_distance_reported": False,
    }
    for row in report["branches"]:
        for label in (
            "endpoint_47_distance_to_formal_Replay",
            "endpoint_47_distance_to_formal_PPO",
        ):
            distance = row[label]
            assert distance["combined_mixed_unit_distance"] is None
            assert distance["tool_position_l2_m"] >= 0.0
            assert distance["tool_rotation_geodesic_rad"] >= 0.0
            assert distance["right_wrist_tool_relative_linear_velocity_l2_m_s"] >= 0.0
            assert distance["right_wrist_tool_relative_angular_velocity_l2_rad_s"] >= 0.0
            assert distance["right_fingertip_tool_frame_distance_m"]["maximum"] >= 0.0
        assert len(row["outcome_states"]["47"]["right_fingertips"][
            "positions_in_tool_frame_m"
        ]) == 5
    arrays = Path(report["branch_arrays"]["path"])
    assert report["branch_arrays"]["sha256"] == _sha256(arrays)


def test_protocol_moves_to_source45_and_keeps_half_lr_blocked():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "final_OFF_reversal_does_not_restore_endpoint57_source56_semantic_authority_attribution_required"
    assert report["decision"]["passing_branches"] == []
    assert report["decision"]["gate_passed"] is False
    assert report["decision"]["actor_LR_5e_minus_5_unblocked"] is False
    assert report["decision"]["next_read_only_or_candidate_direction"] == (
        "move_read_only_state_entry_attribution_to_source_45"
    )
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source46_state_entry_gate"]
    assert gate["passing_branches"] == []
    assert gate["new_training_authorized"] is False
    assert gate["half_LR_candidate_unblocked"] is False
    assert gate["next_read_only_direction"] == "completed_by_source45_state_entry_gate"
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
