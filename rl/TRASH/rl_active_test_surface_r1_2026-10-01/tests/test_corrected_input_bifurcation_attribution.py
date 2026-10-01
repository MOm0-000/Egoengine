import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_corrected_input_bifurcation_attribution_v1"
REPORT = RUN / "report.json"
CONTRACT = ROOT / "configs/taco_pour_corrected_input_bifurcation_attribution_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_input_attribution_is_read_only_and_reproduces_both_formal_traces():
    report = _report()
    assert report["schema"] == "taco_pour_corrected_input_bifurcation_attribution_v1"
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
    assert gate["formal_replay_trace_exactly_reproduced"]
    assert gate["formal_ppo_trace_exactly_reproduced"]
    assert gate["replay_validated_intervals"] == 30
    assert gate["ppo_validated_intervals"] == 27
    assert gate["actor_state_unchanged"]
    assert gate["normalization_state_unchanged"]
    assert gate["reference_context_identical_between_paths_at_all_sources"]


def test_actor_saturates_on_replay_and_ppo_paths_not_only_bad_ppo_state():
    report = _report()
    findings = report["findings"]
    assert findings["ppo_mu_y_outside_support_sources"] == [43, 44, 45, 46, 47]
    assert findings["replay_path_mu_y_outside_support_sources"] == [43, 44, 45, 46, 47]
    assert findings["tail_saturation_is_unique_to_ppo_state_path"] is False
    assert findings[
        "replay_observation_with_common_ppo_hidden_has_higher_mu_y_than_ppo_at_sources_42_to_47"
    ]

    rows = report["path_forward_comparison"]["rows"]
    assert abs(rows["40"]["ppo_path_mu_y"] - 0.49808362126350403) < 1e-12
    assert abs(rows["47"]["ppo_path_mu_y"] - 2.1384003162384033) < 1e-12
    for source in range(42, 48):
        row = rows[str(source)]
        assert row["replay_obs_ppo_hidden_mu_y"] > row["ppo_path_mu_y"]


def test_successful_single_step_rescue_does_not_make_later_mean_self_correct():
    report = _report()
    expected_scores = {
        "44": 0.9559440612792969,
        "45": 0.9171217083930969,
        "46": 0.9847465753555298,
    }
    for source, expected_score in expected_scores.items():
        row = report["single_step_rescue_followup"][source]
        assert row["subsequent_mu_all_remain_above_state_high"]
        assert row["subsequent_mu_returns_below_state_high_before_48"] is False
        assert abs(row["endpoint_48"]["outcome_score"] - expected_score) < 1e-12
        assert row["endpoint_48"]["outcome_score"] < 1.0
        assert all(step["mu_exceeds_high"] for step in row["steps"])


def test_group_substitution_uses_fixed_hidden_and_reference_is_path_invariant():
    report = _report()
    rows = report["observation_group_substitution"]
    for source in range(42, 47):
        row = rows[str(source)]
        assert row["fixed_hidden"] == "ppo_path_pre_forward_hidden"
        reference = row["substitutions"]["reference_context"]
        assert reference["raw_group_l2_difference"] == 0.0
        assert reference["right_wrist_y_mu_change"] == 0.0
        assert row["substitutions"]["current_object_anchors"][
            "right_wrist_y_mu_change"
        ] > 0.0


def test_normalization_does_not_clip_and_finite_differences_are_not_physics_claims():
    report = _report()
    assert report["findings"][
        "normalization_clipped_dimension_count_across_recorded_path_states"
    ] == 0
    assert report["findings"]["empirical_training_support_claim_available"] is False
    assert report["findings"][
        "maximum_predicted_mu_y_change_for_one_mm_local_perturbation"
    ] < 0.0015
    sensitivity = report["local_actor_input_sensitivity"]
    assert sensitivity["fixed_hidden"] == "ppo_path_pre_forward_hidden"
    assert sensitivity["causal_physics_gradient"] is False
    for source_rows in sensitivity["rows"].values():
        for row in source_rows.values():
            assert row["causal_physics_gradient"] is False
            assert row["central_difference_delta_m"] == 0.001


def test_full_actor_inputs_outputs_bounds_and_hidden_are_hash_bound():
    report = _report()
    artifact = report["observation_layout"]["full_arrays"]
    path = RUN / Path(artifact["path"]).name
    assert _sha256(path) == artifact["sha256"]
    assert _sha256(CONTRACT) == report["contract"]["sha256"]
    with np.load(path) as arrays:
        assert arrays["ppo_raw_observation"].shape == (8, 236)
        assert arrays["replay_raw_observation"].shape == (8, 236)
        assert arrays["ppo_actor_input"].shape == (8, 236)
        assert arrays["ppo_actor_mu"].shape == (8, 36)
        assert arrays["ppo_actor_logstd"].shape == (8, 36)
        assert arrays["ppo_state_low"].shape == (8, 36)
        assert arrays["ppo_state_high"].shape == (8, 36)
        assert arrays["ppo_requested_residual"].shape == (8, 36)
        np.testing.assert_array_equal(
            arrays["ppo_requested_residual"],
            0.05 * arrays["ppo_deterministic_action"],
        )
