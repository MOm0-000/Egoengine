import gzip
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/taco_pour_normalized_policy_diagnostics_v1/report.json.gz"


def load_report():
    with gzip.open(REPORT, "rt") as stream:
        return json.load(stream)


def test_checkpoint_transfer_includes_normalization_and_rnn_contract():
    report = load_report()
    assert report["scope"]["training_executed"] is False
    assert report["scope"]["physics_rollout_executed"] is False
    assert report["inference_contract"]["mismatch_found"] is False
    assert report["inference_contract"]["rnn"]["all_checks_pass"] is True
    for checkpoint in report["checkpoints"].values():
        assert checkpoint["missing_normalization_keys"] == []
        assert checkpoint["cpu_strict_load"]["missing_keys"] == []
        assert checkpoint["cpu_strict_load"]["unexpected_keys"] == []
        assert checkpoint["cpu_strict_load"]["all_tensors_bitwise_equal"] is True
        assert checkpoint["recurrent_state"]["default_state_is_zero"] is True
        assert checkpoint["normalization"]["running_mean_std"]["finite"] is True
        assert checkpoint["normalization"]["running_mean_std"]["strictly_positive_variance"] is True


def test_failure_tail_decomposition_and_reward_reconstruction():
    report = load_report()
    trace = report["cpu_traces"]["epoch_8"]
    assert trace["validated_steps"] == 38
    assert [row["endpoint"] for row in trace["failure_tail"]] == [55, 56, 57, 58, 59]
    for row in trace["failure_tail"]:
        assert math.isclose(
            row["ellipse_score"] ** 2,
            row["position_squared_contribution"] + row["rotation_squared_contribution"],
            rel_tol=0.0,
            abs_tol=2e-7,
        )
        assert row["independent_threshold_pass"] is True
    assert trace["reward"]["reward_reconstruction_max_abs_error"] < 3e-8
    assert trace["reward"]["contact_flags_or_forces_present_in_saved_trace"] is False


def test_action_saturation_drives_the_predeclared_next_change_family():
    report = load_report()
    for trace in report["cpu_traces"].values():
        assert trace["action"]["saturated_value_fraction"] > 0.95
        assert trace["action"]["steps_with_any_saturated_dimension"] == trace["action"]["steps"]
    assert report["evidence_assessment"]["action_saturation_or_switching"]["level"] == "strong"
    assert report["evidence_assessment"]["contact_or_lift_reward_conflict_at_epoch8_failure_tail"][
        "mechanical_masking_detected"
    ] is False
    assert report["paper_constraint"]["taco_action_smoothness_reward"] == "disabled"
    assert report["paper_constraint"]["published_residual_scale_or_range"] is None
    assert report["decision"]["next_single_controlled_change_family"] == "action unit/range contract"
    assert report["decision"]["proposed_mapping_is_published_by_egoengine"] is False
    assert report["decision"]["implementation_change_executed"] is False


def test_8_and_16_epoch_checkpoints_are_not_claimed_as_one_learning_curve():
    report = load_report()
    comparison = report["policy_comparison"]
    assert comparison["causal_epoch_ablation"] is False
    assert report["evidence_assessment"]["same_policy_degraded_from_epoch8_to_epoch16"][
        "level"
    ] == "not_identifiable"
    assert report["checkpoints"]["epoch_8"]["checkpoints_in_same_run_directory"] == [
        "last_ep_8_rew__8.038462_.pth"
    ]
    assert report["checkpoints"]["epoch_16"]["checkpoints_in_same_run_directory"] == [
        "last_ep_16_rew__5.7482166_.pth"
    ]
