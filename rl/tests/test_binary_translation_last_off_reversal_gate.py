import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_binary_translation_last_off_reversal_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_binary_translation_last_off_reversal_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_last_off_reversal_is_read_only_and_reproduces_the_baseline_bitwise():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["learned_gate_training_allowed"] is False
    regression = report["regression_gate"]
    assert regression["baseline_decision_arrays_bitwise_reproduced"] is True
    assert regression["baseline_successful_intervals"] == 36
    assert regression["baseline_first_failure_endpoint"] == 57
    assert regression["forced_source_complete_snapshots_captured"] == [52, 53, 54, 55]
    assert regression["forced_source_pre_forward_hidden_captured"] == [52, 53, 54, 55]


def test_each_branch_reverses_only_one_predeclared_off_decision():
    report = _report()
    assert report["intervention_contract"] == {
        "forced_sources": [52, 53, 54, 55],
        "one_reversal_per_branch": True,
        "forced_action": "translation_ON_complete_frozen_PPO_action",
        "from_next_source": "same_one_step_binary_translation_oracle",
        "actor_forward_calls_per_source": 1,
        "same_pre_step_snapshot_and_post_forward_hidden_for_ON_OFF": True,
        "contact_used_for_selection": False,
        "future_beyond_one_step_used": False,
    }
    arrays = np.load(RUN / "last_off_reversal_branches.npz")
    for branch in report["branches"]:
        source = branch["forced_ON_source"]
        assert branch["forced_source_baseline_greedy_selection"] == "OFF"
        decisions = branch["downstream_decisions"]
        assert decisions[0]["source_endpoint"] == source
        assert decisions[0]["selected"] == "ON"
        assert decisions[0]["selection_overridden_once"] is True
        assert all(row["selection_overridden_once"] is False for row in decisions[1:])
        assert all(row["actor_forward_calls"] == 1 for row in decisions)
        selected_off = arrays[f"force_source{source}_ON__selected_off"]
        assert bool(selected_off[0]) is False


def test_no_reversal_survives_57_but_source55_restores_authority_without_feasibility():
    report = _report()
    by_source = {row["forced_ON_source"]: row for row in report["branches"]}
    assert {
        source: (row["successful_intervals"], row["first_failure_endpoint"])
        for source, row in by_source.items()
    } == {
        52: (34, 55),
        53: (34, 55),
        54: (35, 56),
        55: (36, 57),
    }
    assert all(row["survives_endpoint57"] is False for row in by_source.values())
    assert all(row["immediate_ON_minus_OFF_score_cost"] > 0 for row in by_source.values())
    source56 = by_source[55]["source56_diagnostic"]
    assert source56["source_contact"]["active_finger_roles"] == []
    assert source56["ON"]["right_hand_tool_contact"]["active_finger_roles"] == ["pinky"]
    assert source56["OFF"]["right_hand_tool_contact"]["active_finger_roles"] == []
    assert source56["ON"]["terminated"] is True
    assert source56["OFF"]["terminated"] is True
    assert source56["ON"]["objective_score"] == 1.0040502548217773
    assert source56["OFF"]["objective_score"] == 1.0030568838119507
    assert source56["ON_minus_OFF_tool_pose_separate_deltas"] == {
        "tool_position_l2_m": 0.00017124311813405613,
        "tool_rotation_distance_rad": 0.004397958873654247,
        "mixed_position_quaternion_norm_reported": False,
    }
    decision = report["decision"]
    assert decision["any_branch_survives_endpoint57"] is False
    assert decision["one_step_greedy_myopia_supported"] is False
    assert decision[
        "source55_reversal_changes_source56_tool_pose_and_contact"
    ] is True
    assert decision["source55_reversal_restores_endpoint57_feasibility"] is False


def test_protocol_moves_to_source56_semantic_attribution_and_keeps_training_blocked():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "unit_gain_active_lag_correction_insufficient_no_parameter_sweep_authorized"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["binary_translation_last_OFF_reversal_gate"]
    assert gate["any_branch_survives_endpoint57"] is False
    assert gate["source55_reversal"]["source56_ON_OFF_tool_position_l2_m"] == (
        0.00017124311813405613
    )
    assert gate["source55_reversal"][
        "source56_ON_OFF_tool_rotation_distance_rad"
    ] == 0.004397958873654247
    assert gate["source55_reversal"][
        "both_source56_candidates_terminate_at_endpoint57"
    ] is True
    assert gate["new_PPO_training_authorized"] is False
    assert gate["learned_gate_training_authorized"] is False
    assert gate["half_LR_candidate_unblocked"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert gate["next_read_only_direction"] == (
        "completed_by_source56_semantic_action_gate"
    )
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_source57_active_lag_gate"
    )
    assert half_lr["latest_state_entry_evidence"][
        "binary_translation_last_OFF_reversal_report"
    ]["sha256"] == _sha256(REPORT)
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
