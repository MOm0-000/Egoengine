import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/endpoint46_50_local_control_sensitivity_v1/report.json"


def load_report():
    return json.loads(REPORT.read_text())


def test_probe_respects_transition_indexing_and_does_not_train():
    report = load_report()
    scope = report["scope"]
    assert scope["training_executed"] is False
    assert scope["policy_modified"] is False
    assert scope["physics_modified"] is False
    assert scope["source_state_endpoints"] == [45, 46, 47, 48, 49]
    assert scope["outcome_endpoints"] == [46, 47, 48, 49, 50]
    assert scope["epsilon_m"] == [0.0005, 0.001]
    assert scope["diagnostic_residual_clip_m"] == 0.051


def test_both_policies_have_complete_symmetric_xyz_probes():
    report = load_report()
    for policy in report["policies"].values():
        assert len(policy["records"]) == 5
        for row in policy["records"]:
            for epsilon in ("0.0005", "0.0010"):
                probes = row["policy_centered_probe"]["probes"][epsilon]
                assert set(probes) == {"x", "y", "z"}
                assert all(set(axis) == {"plus", "minus"} for axis in probes.values())


def test_one_step_contact_sensitivity_is_not_scale_stable_enough_for_a_direction_claim():
    report = load_report()
    old = report["policies"]["old_scale_1"]["summary"]
    new = report["policies"]["new_scale_005"]["summary"]
    assert old["gradient_sign_stability_fraction"] == 0.5333333333333333
    assert new["gradient_sign_stability_fraction"] == 0.4
    assert old["gradient_cosine_between_epsilons_mean"] < 0.28
    assert new["gradient_cosine_between_epsilons_mean"] < 0.0
    diagnosis = report["diagnosis"]
    assert diagnosis["stable_local_favorable_direction_established"] is False
    assert diagnosis["new_policy_wrong_direction_causally_confirmed"] is False
    assert diagnosis["old_policy_local_direction_certified"] is False
    assert diagnosis["decisions"]["scale_sweep_authorized"] is False
    assert diagnosis["decisions"]["training_authorized"] is False
