import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_corrected_policy_decision_attribution_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_corrected_policy_decision_attribution_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_policy_decision_attribution_is_read_only_and_trace_bound():
    report = _report()
    assert report["schema"] == "taco_pour_corrected_policy_decision_attribution_v1"
    assert report["status"] == "completed_read_only_attribution"
    assert report["paper_faithful"] is False
    assert report["scope"] == {
        "actor_frozen": True,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "normalization_frozen": True,
        "objective_reward_and_residual_bound_unchanged": True,
        "optimizer_steps": 0,
        "training_performed": False,
    }
    gate = report["regression_gate"]
    assert gate["formal_cpu_trace_exactly_reproduced"]
    assert gate["validated_intervals"] == 27
    assert gate["first_failure"]["endpoint"] == 48
    assert gate["restored_complete_state_count"] == 15


def test_formal_cpu_is_already_deterministic_without_sample_or_squash():
    report = _report()
    comparison = report["deterministic_vs_current"]
    assert comparison["distinct_rollout_exists"] is False
    assert comparison["current_formal_cpu_is_deterministic"] is True
    assert comparison["current_formal_cpu_selector"] == "clip_mu_to_state_bounds"
    assert comparison["stochastic_sample_used"] is False
    assert comparison["squash_transform_used"] is False

    for row in report["policy_decisions"].values():
        policy = row["policy"]
        assert policy["stochastic_sample"] is None
        assert policy["pre_squash_action"] is None
        assert policy["post_squash_action"] is None


def test_wrist_y_mean_crosses_support_at_source_43_and_remains_saturated():
    decisions = _report()["policy_decisions"]
    for source in (40, 41, 42):
        row = decisions[str(source)]["right_wrist_y"]
        assert row["actor_mu"] < row["state_high"]
        assert row["deterministic_action"] < 1.0
    for source in (43, 44, 45, 46, 47):
        row = decisions[str(source)]["right_wrist_y"]
        assert row["actor_mu"] > row["state_high"]
        assert row["deterministic_action"] == 1.0
        assert abs(row["effective_residual_after_ctrlrange_m"] - 0.05) < 2e-8
    assert decisions["47"]["right_wrist_y"]["actor_mu"] > 2.13


def test_one_zero_y_decision_at_44_45_or_46_recovers_endpoint_48():
    rows = _report()["single_step_zero_y"]["analysis"]
    expected = {
        "44": (0.9559440612792969, -0.013238832354545593),
        "45": (0.9171217083930969, -0.019431568682193756),
        "46": (0.9847465753555298, -0.008574016392230988),
    }
    for source, (score, bowl_y_change) in expected.items():
        row = rows[source]
        assert row["first_action_y_before"] == 1.0
        assert row["first_action_y_after"] == 0.0
        assert abs(row["endpoint_48_score"] - score) < 1e-12
        assert abs(row["change_from_baseline_bowl_y_m"] - bowl_y_change) < 1e-12
        assert row["endpoint_48_score"] < 1.0
        assert row["first_tracking_failure_endpoint"] is None


def test_control_transmission_collapses_between_sources_46_and_47():
    rows = _report()["control_transmission_probe"]["analysis"]
    source_46 = rows["46"]
    source_47 = rows["47"]
    assert source_46["baseline_source_contact"]["active_right_tool_fingers"] == [
        "index"
    ]
    assert source_47["baseline_source_contact"]["active_right_tool_fingers"] == []
    assert source_46["endpoint_48_response_ratio"] > 0.47
    assert source_47["endpoint_48_response_ratio"] < 0.005
    assert abs(source_46["endpoint_48_bowl_y_delta_m"]) > 0.005
    assert abs(source_47["endpoint_48_bowl_y_delta_m"]) < 0.0001
    assert source_46["smooth_jacobian_claim"] is False
    assert source_47["smooth_jacobian_claim"] is False

    live = source_46["baseline_source_contact"]["live_mjwp_contacts"]
    assert live
    assert all(row["finger"] == "index" for row in live)
    assert all(row["contact_distance_m"] <= 0.0 for row in live)
    assert all(row["normal_force"] > 0.0 for row in live)


def test_policy_decision_artifacts_are_hash_bound_and_not_resumable():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert _sha256(CONTRACT) == report["contract"]["sha256"]
    assert contract["runtime"]["checkpoint_resume_for_training_allowed"] is False
    assert contract["snapshots"]["diagnostic_only"] is True
    assert contract["snapshots"]["resume_authorized"] is False
    for endpoint in range(42, 48):
        row = report["source_snapshots"][str(endpoint)]
        path = RUN / "source_snapshots" / Path(row["path"]).name
        assert row["snapshot_field_count"] == 362
        assert row["diagnostic_only"] is True
        assert row["resume_authorized"] is False
        assert _sha256(path) == row["artifact_sha256"]

    checks = report["global_checks"]
    assert checks["zero_probes_reproduce_formal_endpoint_48"]
    assert checks["all_interventions_remain_within_formal_action_support"]
    assert checks["complete_physics_and_rnn_state_restored_for_every_branch"]
    assert checks["maximum_ctrlrange_loss"] == 0.0


def test_protocol_promotes_only_policy_input_and_action_frame_decision():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    blocker = (
        "corrected_policy_mean_saturation_attributed_"
        "input_and_action_frame_decision_required"
    )
    assert protocol["blocking_checks"] == [blocker]
    audit_path = Path(
        protocol["audit_results"]["corrected_policy_decision_attribution"]
    )
    assert audit_path.parts[-3:] == (
        "runs",
        "taco_pour_corrected_policy_decision_attribution_v1",
        "report.json",
    )
    assert protocol["training_ready"] is False
