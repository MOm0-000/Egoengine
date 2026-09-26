import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_tail_semantic_suppression_oracle_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_tail_semantic_suppression_oracle_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_tail_oracle_is_read_only_and_reproduces_the_binary_oracle_prefix():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert report["regression_gate"] == {
        "complete_binary_oracle_arrays_bitwise_reproduced": True,
        "source20_through_43_untouched": True,
        "source44_complete_snapshot_and_pre_forward_hidden_captured": True,
    }
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["actor_learning_rate_5e_minus_5_training_allowed"] is False
    assert contract["runtime"]["learned_gate_training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False


def test_tail_oracle_selects_the_declared_modes_and_applies_ties_conservatively():
    report = _report()
    expected = [
        "zero_complete_right_wrist",
        "zero_complete_right_wrist",
        "zero_right_wrist_translation",
        "zero_right_wrist_translation",
        "zero_full_36d_residual",
        "zero_right_wrist_translation",
        "zero_right_wrist_translation",
        "zero_right_wrist_translation",
        "zero_full_36d_residual",
        "zero_entire_right_hand",
        "zero_right_wrist_translation",
        "complete_PPO",
    ]
    assert report["result"]["selected_modes_44_to_55"] == expected
    assert [row["source_endpoint"] for row in report["tail_decisions"]] == list(
        range(44, 56)
    )
    assert [row["selected"] for row in report["tail_decisions"]] == expected
    assert report["result"]["selected_mode_counts"] == {
        "zero_complete_right_wrist": 2,
        "zero_right_wrist_translation": 6,
        "zero_full_36d_residual": 2,
        "zero_entire_right_hand": 1,
        "complete_PPO": 1,
    }

    source53 = report["tail_decisions"][9]
    source55 = report["tail_decisions"][11]
    source53_scores = {
        row["name"]: row["outcome"]["objective_score"]
        for row in source53["candidates"]
    }
    assert source53_scores["zero_entire_right_hand"] == source53_scores[
        "zero_full_36d_residual"
    ]
    assert source53["selection_reason"] == (
        "exact_score_tie_prefer_fewer_zeroed_dimensions"
    )
    assert source55["selected"] == "complete_PPO"
    assert source55["selection_reason"] == (
        "exact_score_tie_prefer_fewer_zeroed_dimensions"
    )


def test_tail_state_entry_makes_source56_recoverable_but_still_fails_endpoint58():
    report = _report()
    source56 = report["source56_fork"]
    names = report["selection_contract"]["candidate_order"]
    assert source56["passing_candidates"] == names
    assert source56["selected_for_continuation"] == "complete_PPO"
    outcomes = {row["name"]: row["outcome"] for row in source56["candidates"]}
    assert all(row["terminated"] is False for row in outcomes.values())
    assert all(row["objective_score"] < 1.0 for row in outcomes.values())
    assert outcomes["complete_PPO"]["objective_score"] == 0.9918596148490906
    assert outcomes["complete_PPO"]["ellipse_squared_contributions"] == {
        "position": 0.8299110422371901,
        "rotation": 0.153874388054501,
    }
    continuation = source56["binary_oracle_continuation"]
    assert len(continuation) == 1
    assert continuation[0]["source_endpoint"] == 57
    assert continuation[0]["selected"] == "ON"
    assert continuation[0]["ON_score"] == 1.0227254629135132
    assert continuation[0]["OFF_score"] == 1.023302674293518
    assert report["result"]["successful_intervals"] == 37
    assert report["result"]["forty_of_forty"] is False
    assert report["result"]["first_failure_endpoint"] == 58

    arrays = np.load(RUN / "tail_semantic_oracle_trace.npz")
    assert arrays["tail_source_endpoint"].tolist() == list(range(44, 56))
    assert arrays["source56_candidate_terminated"].tolist() == [False] * 7
    assert np.all(arrays["source56_candidate_score"] < 1.0)


def test_protocol_keeps_training_and_commit_blocked_after_source56_recovery():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "gate_A_active_corrective_parameterization_decision_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["tail_semantic_suppression_oracle"]
    assert gate["source56"]["all_seven_candidates_pass_endpoint57"] is True
    assert gate["continuation"]["first_failure_endpoint"] == 58
    assert gate["continuation"]["successful_intervals"] == 37
    assert gate["forty_of_forty"] is False
    assert gate["semantic_suppression_family_exhausted"] is False
    assert gate["learned_gate_training_authorized"] is False
    assert gate["new_PPO_training_authorized"] is False
    assert gate["half_LR_candidate_unblocked"] is False
    assert gate["reward_change_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_gate_A_action_feasibility_classification"
    )
    assert half_lr["latest_state_entry_evidence"]["tail_semantic_oracle_report"][
        "sha256"
    ] == _sha256(REPORT)
    assert report["result"]["PPO_retraining_authorized"] is False
    assert report["result"]["chunk_acceptance_or_commit_authorized"] is False
