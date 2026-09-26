import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_source47_semantic_action_subspace_gate_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_source47_semantic_action_subspace_gate_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_semantic_gate_is_read_only_and_trace_bound():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert contract["runtime"]["training_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert report["regression_gate"] == {
        "formal_Replay_trace_bitwise_reproduced": True,
        "formal_PPO_trace_bitwise_reproduced": True,
        "restored_source47_suffix_bitwise_reproduced": True,
        "formal_Replay_successful_intervals": 30,
        "formal_PPO_successful_intervals": 28,
        "formal_PPO_endpoint49_score": 1.1142933368682861,
    }


def test_no_source47_semantic_branch_preserves_contact_or_survives_endpoint49():
    report = _report()
    expected = {
        "zero_right_wrist_translation": 1.0902138948440552,
        "zero_right_wrist_rotation": 1.1122767925262451,
        "zero_right_fingers": 1.1150819063186646,
        "zero_right_wrist_translation_and_rotation": 1.0883245468139648,
        "zero_entire_right_hand_residual": 1.088700294494629,
    }
    assert {row["name"] for row in report["branches"]} == set(expected)
    for row in report["branches"]:
        assert row["left_hand_action_bitwise_unchanged"] is True
        assert row["endpoint_48"][
            "has_effective_live_right_hand_tool_contact"
        ] is False
        assert row["endpoint_48"]["right_hand_tool_contact"][
            "active_right_tool_fingers"
        ] == []
        assert row["endpoint_48"]["right_hand_tool_contact"][
            "live_mjwp_contacts"
        ] == []
        assert row["endpoint_49"]["objective_score"] == expected[row["name"]]
        assert row["endpoint_49"]["terminated"] is True
        assert row["first_failure_endpoint_or_60"] == 49
        assert row["passes_predeclared_gate"] is False
    assert report["decision"]["passing_branches"] == []
    assert report["decision"]["gate_passed"] is False
    assert report["decision"]["new_training_authorized"] is False
    assert report["decision"]["next_read_only_direction"] == (
        "move_read_only_state_entry_attribution_to_source_46"
    )


def test_training_credit_favors_contact_preservation_relatively_without_bonus():
    grouped = _report()["training_credit_source46_47"][
        "by_source_and_next_step_contact"
    ]
    expected_counts = {"46": (19, 4), "47": (17, 3)}
    for source, (no_count, contact_count) in expected_counts.items():
        no_contact = grouped[source]["next_step_no_contact"]
        contact = grouped[source]["next_step_contact"]
        assert no_contact["sample_count"] == no_count
        assert contact["sample_count"] == contact_count
        assert contact["return"]["mean"] > no_contact["return"]["mean"]
        assert contact["raw_advantage"]["mean"] > no_contact["raw_advantage"]["mean"]
        assert contact["normalized_advantage"]["mean"] > no_contact[
            "normalized_advantage"
        ]["mean"]
        assert contact["paper_bonus_contact_condition_count"] == 0
        assert no_contact["paper_bonus_contact_condition_count"] == 0
        assert "thumb" not in "+".join(contact[
            "right_tool_finger_pattern_counts"
        ])


def test_protocol_moves_to_source46_without_unblocking_half_lr():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "source57_temporal_hold_insufficient_active_lag_correction_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    gate = protocol["evaluation"]["source47_semantic_action_subspace_gate"]
    assert gate["passing_branches"] == []
    assert gate["gate_passed"] is False
    assert gate["new_training_authorized"] is False
    assert gate["next_read_only_direction"] == "source46_state_entry_attribution"
    assert half_lr["execution_gate"]["fresh_training_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_source57_temporal_hold_gate"
    )
