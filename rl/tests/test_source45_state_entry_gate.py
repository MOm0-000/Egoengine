import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source45_state_entry_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source45_state_entry_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_source45_gate_is_read_only_and_exactly_reproduced():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["primary_passing_condition"]["contact_required"] is False
    assert contract["runtime"]["training_allowed"] is False
    assert report["regression_gate"] == {
        "formal_Replay_trace_bitwise_reproduced": True,
        "formal_PPO_trace_bitwise_reproduced": True,
        "restored_source45_suffix_bitwise_reproduced": True,
        "restored_source45_physical_records_exact": True,
        "restored_formal_source46_actor_decision_exact": True,
        "formal_Replay_successful_intervals": 30,
        "formal_PPO_successful_intervals": 28,
    }


def test_source45_translation_and_wrist_branches_pass_endpoint50():
    expected = {
        "zero_R_forearm_ty_residual_only": (False, 1.0591870546340942, 50),
        "zero_right_wrist_translation": (True, 0.9937561750411987, 51),
        "zero_complete_right_wrist": (True, 0.9861631989479065, 51),
        "zero_entire_right_hand_residual": (False, 1.0342196226119995, 50),
        "zero_full_36d_residual": (False, 1.0388888120651245, 50),
    }
    report = _report()
    for row in report["branches"]:
        passes, score, failure = expected[row["name"]]
        assert row["untargeted_action_dimensions_bitwise_unchanged"] is True
        assert row["endpoint_50_survives"] is passes
        assert row["endpoint_50_score"] == score
        assert row["first_failure_endpoint_or_60"] == failure
        assert row["passes_predeclared_primary_gate"] is passes
        assert row["contact_used_as_acceptance_condition"] is False
    assert report["decision"]["passing_branches"] == [
        "zero_right_wrist_translation",
        "zero_complete_right_wrist",
    ]
    assert report["decision"]["narrowest_passing_branch"] == (
        "zero_right_wrist_translation"
    )


def test_contact_is_neither_required_nor_sufficient_and_mu_y_does_not_recover():
    report = _report()
    correction = report["source46_interpretation_correction"]
    assert correction["source46_is_still_tracking_controllable"] is True
    assert correction[
        "endpoint48_contact_is_necessary_for_short_horizon_tracking_feasibility"
    ] is False
    assert correction[
        "endpoint48_contact_is_sufficient_for_short_horizon_tracking_feasibility"
    ] is False
    by_name = {row["name"]: row for row in report["branches"]}
    translation = by_name["zero_right_wrist_translation"]
    assert all(
        not row["has_live_right_hand_tool_contact"]
        for row in translation["contact_secondary_diagnostic"].values()
    )
    for state in translation["outcome_states"].values():
        assert state["contact"]["unmapped_right_hand_contact_geoms"] == []
        assert "all compiled contacts" in state["contact"]["semantics"]
    for name, expected_delta in (
        ("zero_right_wrist_translation", 0.062278151512145996),
        ("zero_complete_right_wrist", 0.010316014289855957),
    ):
        group = by_name[name]["source46_actor_delta_vs_formal"][
            "highlighted_groups"
        ]["R_forearm_ty"]
        assert group["mu_delta"] == [expected_delta]
        assert group["deterministic_action_delta"] == [0.0]
        assert group["counterfactual_deterministic_action"] == [1.0]
    assert report["measurement_boundaries"][
        "contact_is_secondary_and_does_not_affect_gate"
    ] is True


def test_protocol_selects_translation_direction_without_authorizing_training():
    report = _report()
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "tail_semantic_suppression_creates_recoverable_source56_state_requires_mode_characterization"
    assert report["decision"]["gate_passed"] is True
    assert report["decision"]["new_training_authorized"] is False
    assert report["decision"]["actor_LR_5e_minus_5_unblocked"] is False
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source45_state_entry_gate"]
    assert gate["narrowest_passing_branch"] == "zero_right_wrist_translation"
    assert gate["contact_is_secondary_not_acceptance"] is True
    assert gate["new_training_authorized"] is False
    assert gate["next_candidate_direction"] == (
        "right_wrist_translation_temporal_scale_or_gating"
    )
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_tail_semantic_oracle"
    )
    assert half_lr["latest_state_entry_evidence"][
        "half_LR_selected_for_next_experiment"
    ] is False
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
