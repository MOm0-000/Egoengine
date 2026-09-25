import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/taco_pour_normalized_action_scale_v1/comparison.json"


def test_action_scale_experiment_changed_only_the_declared_mapping():
    report = json.loads(REPORT.read_text())
    assert report["scope"]["only_intended_algorithm_change"] == "residual_scale 1.0 -> 0.05"
    frozen = report["frozen_contract"]
    assert frozen["endpoint20_state_entry_count"] == 362
    assert frozen["endpoint20_state_keys_equal"] is True
    assert frozen["endpoint20_state_unequal_keys"] == []
    assert frozen["replay_trace_bitwise_equal"] is True
    assert frozen["actor_transfer_bitwise_equal"] is True


def test_scaled_mapping_reduced_saturation_and_reports_tracking_result():
    report = json.loads(REPORT.read_text())
    old = report["old_scale_1_epoch8"]
    new = report["new_scale_005_epoch8"]
    assert old["tracking"]["validated_steps"] == 38
    assert old["action"]["applied_saturation_fraction"] > 0.95
    assert new["action"]["applied_saturation_fraction"] < 0.50
    assert new["action"]["adjacent_delta_abs_max_rad"] < old["action"]["adjacent_delta_abs_max_rad"]
    assert report["effects"]["validated_step_change"] == (
        new["tracking"]["validated_steps"] - old["tracking"]["validated_steps"]
    )
    assert report["decision"]["promote_scaled_mapping"] == (
        new["tracking"]["validated_steps"] == 40
    )
    paired = report["scaled_policy_vs_same_run_replay"]
    assert paired["count"] == 30
    assert paired["scaled_lower_score_count"] == 25
    assert report["effects"]["failure_tail_55_59_comparison_available"] is False


def test_scaled_policy_cpu_result_is_bitwise_repeatable():
    repeat = json.loads(REPORT.read_text())["deterministic_cpu_repeat"]
    assert repeat["repetitions"] == 3
    assert repeat["bitwise_repeatable"] is True
    assert len(set(repeat["validated_steps"])) == 1
