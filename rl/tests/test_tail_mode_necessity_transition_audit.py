import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_tail_mode_necessity_transition_audit_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_tail_mode_necessity_transition_audit_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_audit_is_read_only_and_parent_evidence_is_hash_bound():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_audit"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert report["parent_evidence"]["saved_results_only_for_mode_margins"] is True
    assert report["parent_evidence"][
        "selected_path_and_existing_binary_continuation_exactly_reproduced_for_transition"
    ] is True
    assert contract["mode_necessity"]["numerical_tolerance_for_merging_modes"] is None
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["learned_gate_training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False


def test_exact_margins_do_not_promote_near_ties_to_finger_or_bilateral_causality():
    mode = _report()["mode_necessity"]
    rows = {row["source_endpoint"]: row for row in mode["sources"]}
    assert mode["numerical_tolerance_used_to_merge_modes"] is None
    assert mode["near_tie_promoted_to_causal_label"] is False
    assert mode["learned_gate_labels_generated"] is False
    assert mode["source44_complete_wrist_improvement_vs_translation"] == (
        0.00217515230178833
    )
    assert mode["source45_complete_wrist_improvement_vs_translation"] == (
        0.006376802921295166
    )
    assert rows[48]["winner_runner_up_absolute_margin"] == 2.980232238769531e-07
    assert rows[52]["winner_runner_up_absolute_margin"] == 2.384185791015625e-06
    assert rows[53]["winner_runner_up_absolute_margin"] == 0.0
    assert rows[53]["exact_minimum_score_candidates"] == [
        "zero_entire_right_hand",
        "zero_full_36d_residual",
    ]
    assert rows[55]["winner_runner_up_absolute_margin"] == 0.0
    assert len(rows[55]["exact_minimum_score_candidates"]) == 6
    assert mode["distinct_finger_causality_established"] is False
    assert mode["distinct_bilateral_causality_established"] is False


def test_transition_reproduction_finds_z_lag_then_y_drift_and_force_collapse():
    transition = _report()["transition_audit"]
    first = transition["source56_to_57"]
    second = transition["source57_to_58"]
    assert first["dominant_absolute_position_error_growth_axis"] == "z"
    assert first["actual_tool_displacement_world_mm"] == [
        0.03904104232788086,
        0.2776980400085449,
        0.050008296966552734,
    ]
    assert first["reference_tool_displacement_world_mm"] == [
        -1.4603137969970703,
        0.44063106179237366,
        4.490375518798828,
    ]
    assert first["absolute_position_error_growth_mm"][2] == 4.440367221832275
    assert first["source_contact"]["active_finger_roles"] == []
    assert first["outcome_contact"]["active_finger_roles"] == ["pinky"]
    assert first["outcome_contact"]["sum_normal_force"] == 10.689228057861328

    assert second["dominant_absolute_position_error_growth_axis"] == "y"
    assert second["absolute_position_error_growth_mm"] == [
        0.38254261016845703,
        4.2178817093372345,
        2.483844757080078,
    ]
    assert second["outcome_contact"]["active_finger_roles"] == ["pinky"]
    assert second["outcome_contact"]["sum_normal_force"] == 0.17438175529241562
    assert second["squared_contribution_change"]["position"] > 0
    assert second["squared_contribution_change"]["rotation"] < 0

    actor = transition["actor_source56_to_57"]
    assert len(actor["source56_normalized_action"]) == 36
    assert len(actor["source57_normalized_action"]) == 36
    assert actor["source56_normalized_action"][1] == 0.6558910012245178
    assert actor["source57_normalized_action"][1] == 0.6731809973716736
    assert transition["source57_existing_binary_scores"] == {
        "ON": 1.0227254629135132,
        "OFF": 1.023302674293518,
        "selected": "ON",
    }

    arrays = np.load(RUN / "transition_trace.npz")
    assert arrays["source"].tolist() == list(range(44, 56))
    assert arrays["source56_action"].shape == (36,)
    assert arrays["source57_action"].shape == (36,)


def test_protocol_keeps_all_algorithm_changes_blocked_after_refined_attribution():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "tail_mode_state_criterion_and_endpoint58_active_correction_unresolved"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["tail_mode_necessity_transition_audit"]
    assert gate["numerical_tolerance_used_to_merge_modes"] is None
    assert gate["distinct_finger_causality_established"] is False
    assert gate["distinct_bilateral_causality_established"] is False
    assert gate["source57_new_semantic_gate_executed"] is False
    assert gate["actor_LR_5e_minus_5_unblocked"] is False
    assert gate["learned_gate_training_authorized"] is False
    assert gate["reward_change_authorized"] is False
    assert gate["PPO_retraining_authorized"] is False
    assert gate["chunk_acceptance_or_commit_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_tail_mode_transition_audit"
    )
    assert half_lr["latest_state_entry_evidence"][
        "tail_mode_necessity_transition_report"
    ]["sha256"] == _sha256(REPORT)
    assert report["decision"]["source57_new_semantic_gate_executed"] is False
    assert report["decision"]["PPO_retraining_authorized"] is False
    assert report["decision"]["chunk_acceptance_or_commit_authorized"] is False
