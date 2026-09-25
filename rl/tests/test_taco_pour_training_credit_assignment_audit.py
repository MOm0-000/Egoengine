import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_training_credit_assignment_audit_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_training_credit_assignment_audit_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_credit_audit_is_read_only_and_reproduces_every_formal_baseline():
    report = _report()
    assert report["schema"] == "taco_pour_training_credit_assignment_audit_v1"
    assert report["status"] == "completed_read_only_training_credit_audit"
    assert report["paper_faithful"] is False
    assert report["scope"] == {
        "actor_frozen": True,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "normalization_frozen": True,
        "optimizer_steps": 0,
        "reward_objective_scale_bound_and_formal_timing_unchanged": True,
        "training_performed": False,
    }
    gate = report["regression_gate"]
    assert gate["formal_replay_trace_exactly_reproduced"]
    assert gate["formal_ppo_trace_exactly_reproduced"]
    assert gate["formal_plus_y_suffix_exactly_reproduced"] == {
        "43": True, "44": True, "45": True, "46": True
    }
    assert gate["replay_validated_intervals"] == 30
    assert gate["ppo_validated_intervals"] == 27
    assert gate["timing_minus1_validated_intervals"] == 30
    assert gate["actor_state_unchanged"] and gate["normalization_state_unchanged"]


def test_timing_minus_one_is_suppression_not_refinement():
    row = _report()["timing_minus1_replay_alignment"]
    assert row["same_failure_endpoint_and_reason_as_replay"]
    assert row["no_validated_interval_beyond_replay"]
    assert row["closer_to_replay_than_formal_ppo_on_common_endpoints"]
    assert row["classified_as_suppression_candidate"]
    assert row["classified_as_refinement_candidate"] is False
    assert row["mean_timing_minus1_qpos_l2_to_replay_common_ppo"] < row[
        "mean_formal_ppo_qpos_l2_to_replay_common"
    ]
    assert row["rows"]["51"]["replay"]["tracking_score"] > 1.0
    assert row["rows"]["51"]["timing_minus1"]["tracking_score"] > 1.0


def test_formal_plus_y_is_counterfactually_suboptimal_at_all_four_sources():
    rows = _report()["one_action_counterfactual"]["by_source"]
    expected_best = {"43": (-1.0, 50), "44": (-1.0, 51),
                     "45": (-1.0, 56), "46": (-0.75, 51)}
    for source, (best_action, best_endpoint) in expected_best.items():
        row = rows[source]
        assert row["formal_mean_is_counterfactually_suboptimal"]
        assert row["formal_saturated_action"]["one_step_action_y"] == 1.0
        assert row["formal_saturated_action"]["termination_endpoint_or_60"] == 48
        assert row["best_action_y"] == best_action
        assert row["best_termination_endpoint_or_60"] == best_endpoint
        assert row[
            "actions_better_than_formal_by_termination_then_discounted_return"
        ] == [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]


def test_zero_y_rescues_do_not_pass_the_full_window_or_replay_boundary():
    result = _report()["one_action_counterfactual"]
    assert result["any_zero_y_rescue_completes_40_step_window"] is False
    assert result["any_zero_y_rescue_passes_endpoint_51"] is False
    rescues = result["full_zero_y_rescues"]
    assert {source: row["termination_endpoint_or_60"] for source, row in rescues.items()} == {
        "44": 49, "45": 51, "46": 49
    }
    assert all(not row["full_remaining_horizon_passed"] for row in rescues.values())


def test_historical_advantage_value_and_return_are_not_recoverable():
    credit = _report()["historical_training_credit"]
    assert credit["exact_historical_advantage_value_return_available"] is False
    assert credit["exact_historical_credit_reconstruction_allowed"] is False
    assert credit["checkpoint_contains_rollout_buffer"] is False
    assert credit["per_epoch_actor_or_critic_checkpoints_available"] is False
    assert credit["final_critic_backfill_rejected"]
    assert credit["missing_credit_fields"] == [
        "advantage", "critic_observation", "critic_value", "raw_observation",
        "return", "reward", "rnn_hidden",
    ]
    assert {key: row["visits"] for key, row in credit["tail_source_summary"].items()} == {
        "43": 34, "44": 31, "45": 28, "46": 25, "47": 23
    }


def test_artifacts_and_active_protocol_are_hash_bound():
    report = _report()
    assert _sha256(CONTRACT) == report["contract"]["sha256"]
    arrays_path = RUN / Path(report["full_arrays"]["path"]).name
    assert _sha256(arrays_path) == report["full_arrays"]["sha256"]
    with np.load(arrays_path) as arrays:
        for source in (43, 44, 45, 46):
            key = f"source_{source}_y_+1.00_qpos"
            assert key in arrays
            assert arrays[key].shape[1] == 50
            assert f"source_{source}_y_-1.00_commanded_ctrl" in arrays

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    blocker = (
        "final_actor_world_y_mean_is_counterfactually_suboptimal_but_"
        "historical_ppo_credit_was_not_logged"
    )
    assert protocol["blocking_checks"] == [blocker]
    assert protocol["training_ready_scope"] == blocker
    assert Path(protocol["audit_results"]["training_credit_assignment_audit"]).parts[-3:] == (
        "runs", "taco_pour_training_credit_assignment_audit_v1", "report.json"
    )
    assert protocol["training_ready"] is False
