import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/taco_pour_action_attribution_v1/report.json"


def load_report():
    return json.loads(REPORT.read_text())


def test_prelimit_audit_reproduces_both_frozen_cpu_traces():
    report = load_report()
    assert report["scope"]["training_executed"] is False
    assert report["trace_reproduction"] == {
        "old_bitwise_equal": True,
        "new_bitwise_equal": True,
    }


def test_actuator_groups_come_from_names_and_preserve_physical_units():
    contract = load_report()["actuator_contract"]
    assert len(contract["names"]) == 36
    assert contract["groups"]["right_wrist_translation"] == [0, 1, 2]
    assert contract["groups"]["right_wrist_rotation"] == [3, 4, 5]
    assert contract["units"]["right_wrist_translation"] == "m"
    assert contract["units"]["right_wrist_rotation"] == "rad"
    assert contract["single_scalar_action_scale_spans_mixed_units"] is True


def test_critical_tail_is_wrong_direction_not_a_clipped_request_for_more_translation():
    report = load_report()
    diagnosis = report["diagnosis"]["critical_right_wrist_translation"]
    assert diagnosis["new_network_mu_fraction_abs_gt_1"] == 0.0
    assert diagnosis["new_network_mu_abs_max"] < 1.0
    assert diagnosis["old_vs_new_flattened_action_cosine"] < -0.49
    assert diagnosis["old_vs_new_signed_sum_cosine"] < -0.59
    assert diagnosis["same_sign_fraction"] == 0.4
    assert report["diagnosis"]["decision"]["increase_or_sweep_scalar_action_scale_next"] is False
    assert report["diagnosis"]["decision"]["late_wrong_direction_explanation_supported"] is True


def test_new_policy_has_moderate_global_prelimit_overflow_but_not_extreme_values():
    network = load_report()["diagnosis"]["network_output"]
    assert 0.27 < network["new_global_fraction_abs_gt_1_endpoint40_50"] < 0.271
    assert 0.01 < network["new_global_fraction_abs_gt_2_endpoint40_50"] < 0.011
    assert network["new_global_abs_max_endpoint40_50"] < 2.5
    assert network["critical_right_wrist_translation_is_action_limited"] is False
