import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_binary_translation_oracle_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_binary_translation_oracle_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_binary_oracle_is_read_only_and_regresses_known_evidence():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["paper_faithful"] is False
    assert report["classification"] == (
        "local_engineering_oracle_shield_not_EgoEngine_RL"
    )
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["runtime"]["training_allowed"] is False
    assert report["regression_gate"] == {
        "formal_Replay_trace_bitwise_reproduced": True,
        "formal_PPO_trace_bitwise_reproduced": True,
        "source45_translation_prefix_arrays_bitwise_reproduced": True,
        "source50_R_forearm_ty_branch_arrays_bitwise_reproduced": True,
    }


def test_each_decision_obeys_the_predeclared_one_step_rule():
    report = _report()
    assert report["selection_contract"] == {
        "right_wrist_translation_indices": [0, 1, 2],
        "right_wrist_translation_actuators": [
            "R_forearm_tx_position",
            "R_forearm_ty_position",
            "R_forearm_tz_position",
        ],
        "actor_forward_calls_per_source": 1,
        "same_pre_step_snapshot_and_post_forward_hidden_for_ON_OFF": True,
        "criterion": "one_step_active_object_tracking_score_and_termination_only",
        "contact_used": False,
        "source_index_used": False,
        "manual_threshold_used": False,
        "future_beyond_one_step_used": False,
    }
    for row in report["decision_trace"]:
        assert row["actor_forward_calls"] == 1
        assert row["pre_forward_hidden_shared"] is True
        assert row["post_forward_hidden_shared"] is True
        assert row["contact_used_for_selection"] is False
        on, off = row["ON"], row["OFF"]
        if on["terminated"] != off["terminated"]:
            expected = "OFF" if not off["terminated"] else "ON"
        elif off["objective_score"] < on["objective_score"]:
            expected = "OFF"
        else:
            expected = "ON"
        assert row["selected"] == expected
        assert row["ON_minus_OFF_separate_physical_deltas"][
            "combined_mixed_unit_distance"
        ] is None


def test_oracle_extends_to_36_but_is_not_sparse_and_fails_at_57():
    report = _report()
    result = report["oracle_result"]
    assert result["successful_intervals"] == 36
    assert result["forty_of_forty"] is False
    assert result["first_failure_endpoint"] == 57
    assert result["Replay_successful_intervals"] == 30
    assert result["formal_PPO_successful_intervals"] == 28
    assert result["OFF_count"] == 19
    assert result["ON_count"] == 18
    assert result["OFF_sources"] == [
        27, 31, 35, 36, 37, 38, 42, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55,
    ]
    assert result["longest_consecutive_OFF_sources"] == list(range(44, 56))
    assert result["source45_selected_OFF"] is True
    assert result["source50_selected_OFF"] is True
    failure = result["failure_transition"]
    assert failure["source_endpoint"] == 56
    assert failure["outcome_endpoint"] == 57
    assert failure["ON_score"] == failure["OFF_score"] == 1.004638910293579
    assert failure["ON_and_OFF_both_terminate"] is True
    assert failure["endpoint57_ellipse_squared_contributions"] == {
        "position": 0.8251121540348055,
        "rotation": 0.18418713835863956,
    }
    assert failure["source_contact_fingers"] == []
    delta = failure["ON_minus_OFF_separate_physical_deltas"]
    assert delta["right_wrist_world_position_l2_m"] == 0.006331534267210198
    assert delta["tool_freejoint_qpos_l2"] == 1.7701254979472804e-08
    assert result["learned_gate_training_authorized"] is False
    assert result["chunk_acceptance_or_commit_authorized"] is False


def test_protocol_moves_to_endpoint57_attribution_and_keeps_training_blocked():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "source57_temporal_hold_insufficient_active_lag_correction_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["binary_translation_oracle_gate"]
    assert gate["successful_intervals"] == 36
    assert gate["first_failure_endpoint"] == 57
    assert gate["automatically_selects_source45_OFF"] is True
    assert gate["automatically_selects_source50_OFF"] is True
    assert gate["sparse_gate_supported"] is False
    assert gate["new_PPO_training_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert gate["next_read_only_direction"] == "completed_by_last_OFF_reversal_gate"
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_source57_temporal_hold_gate"
    )
    assert half_lr["latest_state_entry_evidence"][
        "binary_translation_oracle_report"
    ]["sha256"] == _sha256(REPORT)
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
