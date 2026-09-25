import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_reference_timing_action_frame_audit_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_reference_timing_action_frame_audit_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_audit_is_read_only_and_reproduces_formal_traces():
    report = _report()
    assert report["schema"] == "taco_pour_reference_timing_action_frame_audit_v1"
    assert report["status"] == "completed_read_only_semantic_audit"
    assert report["paper_faithful"] is False
    assert report["scope"] == {
        "actor_frozen": True,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "normalization_frozen": True,
        "optimizer_steps": 0,
        "reward_objective_scale_and_bound_unchanged": True,
        "training_performed": False,
    }
    gate = report["regression_gate"]
    assert gate["formal_replay_trace_exactly_reproduced"]
    assert gate["formal_ppo_trace_exactly_reproduced"]
    assert gate["replay_validated_intervals"] == 30
    assert gate["ppo_validated_intervals"] == 27
    assert gate["actor_state_unchanged"]
    assert gate["normalization_state_unchanged"]


def test_timing_ledger_keeps_corrected_physics_and_reward_alignment():
    rows = _report()["timing_ledger"]
    for source in range(40, 48):
        row = rows[str(source)]
        assert row == {
            "action_interval": [source, source + 1],
            "actor_reference_ctrl_endpoint": source + 1,
            "actor_reference_ctrl_preview_endpoint": source + 2,
            "base_command_reference_endpoint": source + 1,
            "goal_object_anchors_reference_endpoint": source + 1,
            "hand_object_and_contact_observation_endpoint": source,
            "residual_corrects_command_endpoint": source + 1,
            "returned_observation_goal_reference_endpoint": source + 2,
            "reward_reference_endpoint": source + 1,
            "simulator_state_endpoint_after_action": source + 1,
            "simulator_state_endpoint_before_action": source,
        }


def test_minus_one_actor_context_is_material_but_not_a_success():
    report = _report()
    timing = report["reference_forward_counterfactual"]
    assert timing["selected_nonzero_shift"] == -1
    assert abs(timing["selected_relative_improvement"] - 0.11544809062326543) < 1e-12
    assert timing["sum_positive_mu_excess"] == {
        "-1": 3.7242162227630615,
        "0": 4.210285663604736,
        "1": 4.686591625213623,
    }
    assert len(report["closed_loop_counterfactuals"]) == 1
    candidate = report["closed_loop_counterfactuals"][0]
    assert candidate["candidate"] == "diagnostic_actor_reference_shift_-1"
    assert candidate["validated_intervals"] == 30
    assert candidate["first_failure"]["endpoint"] == 51
    assert candidate["diagnostic_only"]
    assert candidate["promotion_allowed"] is False
    assert candidate["commit_allowed"] is False
    assert report["findings"]["formal_success_or_promotion_claimed"] is False


def test_hidden_context_does_not_remove_tail_mean_saturation():
    report = _report()
    rows = report["hidden_isolation"]
    for source in range(43, 48):
        assert all(
            not row["inside_state_support"] for row in rows[str(source)].values()
        )
    assert report["findings"]["zero_hidden_returns_mu_inside_support_sources"] == [
        40, 41, 42
    ]
    assert report["findings"]["replay_hidden_returns_mu_inside_support_sources"] == [
        40, 41, 42
    ]


def test_local_frame_is_geometrically_less_bad_but_rejected_by_support_gate():
    report = _report()
    frame = report["frame_geometry"]
    axis = frame["base_translation_axis_gate"]
    assert axis["robot_base_translation_frame_equals_world"]
    assert axis["maximum_absolute_difference_from_identity"] < 3e-11
    projection = frame["mean_signed_projection_sources_43_to_46"]
    assert projection["world_or_robot_base"] < projection["current_palm_site"] < 0
    assert projection["world_or_robot_base"] < projection["reference_palm_site"] < 0
    assert frame["best_local_frame"] == "reference_palm_site"
    assert frame["best_projection_improvement_over_world"] > 0.57
    assert frame["all_best_frame_transformed_actions_inside_existing_support"] is False
    assert frame["selected_frame_for_closed_loop"] is None
    for source in range(42, 48):
        assert not frame["transformed_action_support"]["reference_palm_site"][
            str(source)
        ]["inside_existing_normalized_support"]


def test_arrays_contract_and_active_protocol_are_hash_bound():
    report = _report()
    assert _sha256(CONTRACT) == report["contract"]["sha256"]
    artifact = report["full_arrays"]
    path = RUN / Path(artifact["path"]).name
    assert _sha256(path) == artifact["sha256"]
    with np.load(path) as arrays:
        assert arrays["source_endpoints"].tolist() == list(range(40, 48))
        assert arrays["ppo_raw_observation"].shape == (8, 236)
        assert arrays["ppo_actor_mu"].shape == (8, 36)
        assert arrays["timing_shift_-1_actor_mu"].shape == (8, 36)
        assert arrays["timing_shift_0_actor_mu"].shape == (8, 36)
        assert arrays["timing_shift_1_actor_mu"].shape == (8, 36)

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    blocker = (
        "reference_timing_minus1_is_material_but_unpublished_and_"
        "local_frame_candidate_violates_frozen_action_support"
    )
    assert protocol["blocking_checks"] == [blocker]
    assert protocol["training_ready_scope"] == blocker
    assert Path(
        protocol["audit_results"]["reference_timing_action_frame_audit"]
    ).parts[-3:] == (
        "runs",
        "taco_pour_reference_timing_action_frame_audit_v1",
        "report.json",
    )
    assert protocol["training_ready"] is False
