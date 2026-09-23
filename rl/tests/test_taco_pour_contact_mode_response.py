import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/endpoint45_49_contact_mode_response_v1"


def reports():
    summary = json.loads((RUN / "summary.json").read_text())
    with gzip.open(RUN / "report.json.gz", "rt") as stream:
        full = json.load(stream)
    return summary, full


def test_artifact_hash_and_frozen_scope_are_exact():
    summary, report = reports()
    artifact = RUN / "report.json.gz"
    assert summary["full_report"]["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert report["scope"]["paired_common_support"] == [45, 46, 47]
    assert report["scope"]["old_policy_only_extension"] == [48, 49]
    assert report["scope"]["new_policy_missing_followup_actions"] == [48, 49]
    assert report["scope"]["response_horizon_control_intervals"] == 3
    assert report["scope"]["physics_substeps_per_response"] == 30
    assert report["scope"]["formal_action_clip_retained"] is True
    assert report["scope"]["training_executed"] is False
    assert report["scope"]["policy_called_during_probes"] is False


def test_contact_observer_and_unperturbed_baselines_are_exact():
    _, report = reports()
    for policy in report["policies"].values():
        observer = policy["substep_observer_audit"]
        assert observer["observer_is_bitwise_transparent"] is True
        assert observer["mismatching_fields"] == []
        assert observer["captured_physics_substeps"] == 10
        for record in policy["paired_common_support"] + policy["old_policy_only_extension"]:
            repeat = record["baseline_repeatability"]
            assert repeat["bitwise_numeric_and_exact_contact_sequence"] is True
            assert repeat["first_contact_mode_sha256"] == repeat["second_contact_mode_sha256"]
            assert len(record["baseline"]["physics_substeps"]) == 30


def test_only_common_support_enters_pairwise_diagnosis():
    _, report = reports()
    old = report["policies"]["old_scale_1"]
    new = report["policies"]["new_scale_005"]
    assert [row["source_endpoint"] for row in old["paired_common_support"]] == [45, 46, 47]
    assert [row["source_endpoint"] for row in new["paired_common_support"]] == [45, 46, 47]
    assert [row["source_endpoint"] for row in old["old_policy_only_extension"]] == [48, 49]
    assert new["old_policy_only_extension"] == []
    assert new["new_policy_missing_followup_actions"] == [48, 49]
    assert report["diagnosis"]["old_policy_only_extension_excluded_from_pairwise_claims"] is True


def test_every_feasible_perturbation_switches_exact_contact_mode():
    _, report = reports()
    old = report["policies"]["old_scale_1"]["paired_summary"]
    new = report["policies"]["new_scale_005"]["paired_summary"]
    assert old["difference_scheme_counts"] == {
        "one_sided_minus": 12,
        "one_sided_plus": 6,
    }
    assert new["difference_scheme_counts"] == {"central": 18}
    for summary in (old, new):
        assert summary["contact_stable_local_estimate_count"] == 0
        assert summary["contact_changed_variant_count"] == summary[
            "feasible_signed_variant_count"
        ]
        assert summary["pair_set_changed_variant_count"] == summary[
            "feasible_signed_variant_count"
        ]
        assert summary["multiplicity_only_changed_variant_count"] == 0
        assert summary["baseline_three_step_repeatable_at_every_source"] is True
    diagnosis = report["diagnosis"]
    assert diagnosis["descriptive_response_delay_evidence"] is False
    assert diagnosis["strict_contact_stable_local_linear_support"] is False
    assert diagnosis["stop_further_local_linear_probing_of_frozen_policies"] is True
    assert diagnosis["scale_sweep_authorized"] is False
    assert diagnosis["training_authorized"] is False
