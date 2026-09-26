import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source56_semantic_action_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source56_semantic_action_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_source56_gate_is_read_only_and_reproduces_both_parent_chains():
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
    assert report["regression_gate"] == {
        "baseline_oracle_arrays_bitwise_reproduced": True,
        "source55_ON_reversal_arrays_bitwise_reproduced": True,
        "source56_actor_forward_calls": 1,
        "same_source56_snapshot_and_post_forward_hidden_for_all_candidates": True,
    }


def test_anchor_outputs_are_exact_and_translation_off_is_not_a_new_branch():
    report = _report()
    anchors = {row["name"]: row for row in report["anchors"]}
    assert set(anchors) == {"complete_PPO_translation_ON", "translation_OFF"}
    assert anchors["complete_PPO_translation_ON"]["endpoint57"][
        "objective_score"
    ] == 1.0040502548217773
    assert anchors["translation_OFF"]["endpoint57"][
        "objective_score"
    ] == 1.0030568838119507
    assert all(row["passes_endpoint57"] is False for row in anchors.values())
    arrays = np.load(RUN / "source56_semantic_branches.npz")
    assert arrays["complete_PPO_translation_ON__endpoint"].tolist() == [57]
    assert arrays["translation_OFF__endpoint"].tolist() == [57]


def test_all_semantic_subspaces_fail_with_position_dominated_error():
    report = _report()
    branches = {row["name"]: row for row in report["branches"]}
    assert {
        name: row["endpoint57"]["objective_score"]
        for name, row in branches.items()
    } == {
        "zero_right_wrist_rotation": 1.004225492477417,
        "zero_right_fingers": 1.0041028261184692,
        "zero_complete_right_wrist": 1.0030564069747925,
        "zero_entire_right_hand": 1.0030568838119507,
        "zero_full_36d_residual": 1.0030559301376343,
    }
    assert all(row["passes_endpoint57"] is False for row in branches.values())
    assert all(row["successful_intervals"] == 36 for row in branches.values())
    assert all(row["first_failure_endpoint"] == 57 for row in branches.values())
    assert all(
        row["endpoint57_position_contribution"]
        > row["endpoint57_rotation_contribution"]
        for row in branches.values()
    )
    assert branches["zero_right_wrist_rotation"]["endpoint57"][
        "right_hand_tool_contact"
    ]["active_finger_roles"] == ["pinky"]
    assert branches["zero_right_fingers"]["endpoint57"][
        "right_hand_tool_contact"
    ]["active_finger_roles"] == ["pinky"]
    assert branches["zero_full_36d_residual"]["endpoint57"][
        "right_hand_tool_contact"
    ]["active_finger_roles"] == []
    assert report["decision"]["passing_semantic_branches"] == []
    assert report["decision"]["any_semantic_branch_passes_endpoint57"] is False


def test_protocol_moves_attribution_earlier_and_keeps_all_training_blocked():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "tail_mode_state_criterion_and_endpoint58_active_correction_unresolved"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source56_semantic_action_gate"]
    assert gate["passing_semantic_branches"] == []
    assert gate["failure_is_position_dominated"] is True
    assert gate["source56_action_subspace_selection_exhausted"] is True
    assert gate["learned_gate_training_authorized"] is False
    assert gate["new_PPO_training_authorized"] is False
    assert gate["half_LR_candidate_unblocked"] is False
    assert gate["reward_change_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert gate["next_read_only_direction"] == (
        "completed_by_tail_semantic_suppression_oracle"
    )
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_tail_mode_transition_audit"
    )
    assert half_lr["latest_state_entry_evidence"][
        "source56_semantic_action_report"
    ]["sha256"] == _sha256(REPORT)
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
