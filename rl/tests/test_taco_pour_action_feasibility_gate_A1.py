import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_action_feasibility_gate_A1_v1"
CONTRACT = ROOT / "configs/taco_pour_action_feasibility_gate_A1_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads((RUN / "report.json").read_text())


def test_gate_A1_replays_only_existing_stage1_actions_with_exact_parent_semantics():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert report["status"] == "completed_final_read_only_gate_A"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["policy_optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    assert report["stage_1_local_controllability"]["new_action_search_executed"] is False
    assert report["stage_1_local_controllability"]["replayed_saved_action_counts"] == {
        "current_support": 240,
        "rho_le_3": 512,
    }
    regression = report["regression_gate"]
    assert regression["current_support_saved_first_step_scores_bitwise_reproduced_as_float32"]
    assert regression["rho_saved_first_step_scores_bitwise_reproduced_as_float32"]
    assert regression["rho_float64_provenance_values_round_to_saved_float32"]
    assert regression["rho_float64_provenance_value_count"] == 512
    rho_contract = contract["stage_1_local_controllability"]["datasets"][1]
    assert rho_contract["response_replay_residual_clip"]["method"] == (
        "exactly_regenerate_float64_rho_from_parent_CEM_provenance"
    )


def test_both_local_surrogates_are_informative_on_predeclared_holdouts():
    models = _report()["stage_1_local_controllability"]["models"]
    expected_rows = {"current_support": 48, "rho_le_3": 64}
    for name, row in models.items():
        held = row["held_out"]
        assert held["held_out_rows"] == expected_rows[name]
        assert held["informative"] is True
        assert held["position"]["RMSE"] < held["position"]["train_mean_baseline_RMSE"]
        assert held["velocity"]["RMSE"] < held["velocity"]["train_mean_baseline_RMSE"]
        assert row["coefficient_shape"] == [36, 6]


def test_single_current_support_QP_action_fails_exact_endpoint58_validation():
    qp = _report()["stage_1_local_controllability"]["QP_exact_replay"]
    assert len(qp["action"]) == 36
    assert qp["solver_success"] is True
    assert qp["predicted_position_error_m"] == 0.10908050654658302
    assert qp["objective_score"] == 1.0178940296173096
    assert qp["terminated"] is True
    assert qp["finite"] is True
    assert qp["passed_endpoint58"] is False
    artifact = np.load(RUN / "surrogate_QP_action.npz")
    assert artifact["action"].shape == (36,)
    assert bool(artifact["passed"]) is False


def test_structured_feedback_has_no_three_step_feasible_candidate():
    report = _report()
    feedback = report["stage_2_structured_feedback"]
    assert feedback["executed"] is True
    assert feedback["feasible_candidate_count"] == 0
    assert feedback["any_feasible_candidate"] is False
    best = feedback["best_candidate"]
    assert best["kp"] == 1.0
    assert best["kv_seconds"] == 0.03300682384120368
    np.testing.assert_array_equal(
        best["scores"],
        [1.0175656080245972, 1.0489952564239502, 1.08027184009552],
    )
    assert best["feasible"] is False
    assert best["bitwise_repeatable_on_CPU"] is True
    candidates = np.load(RUN / "structured_feedback_candidates.npz")
    assert candidates["kp"].shape == (192,)
    assert int(candidates["feasible"].sum()) == 0


def test_gate_A_is_closed_and_protocol_moves_to_low_level_controllability():
    report = _report()
    decision = report["decision"]
    blocker = "low_level_impedance_or_force_aware_contact_controllability_required"
    assert decision["Gate_A_closed"] is True
    assert decision["finite_search_failure_is_mathematical_infeasibility_proof"] is False
    assert decision["gate_B_observation_sufficiency_read_only_allowed"] is False
    assert decision["gate_C_policy_representability_allowed"] is False
    assert decision["gate_D_fresh_PPO_allowed"] is False
    assert decision["additional_Gate_A_search_authorized"] is False
    assert decision["pure_learned_suppression_gate_training_authorized"] is False
    assert decision["actor_LR_5e_minus_5_unblocked"] is False
    assert decision["PPO_retraining_authorized"] is False
    assert decision["chunk_acceptance_or_commit_authorized"] is False
    assert decision["next_blocker"] == blocker

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    assert protocol["evidence_policy"]["current_revision"] == blocker
    gate = protocol["evaluation"]["action_feasibility_gate_A1"]
    assert gate["gate_A_closed"] is True
    assert gate["structured_object_motion_feedback"]["feasible_candidates"] == 0
    assert gate["next_blocker"] == blocker
    assert half_lr["status"] == (
        "frozen_blocked_after_gate_A1_closed_negative_at_declared_resolution"
    )


def test_gate_A1_artifact_hashes_and_response_dataset_are_self_consistent():
    report = _report()
    for row in report["artifacts"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert row["sha256"] == _sha256(path)
    arrays = np.load(RUN / "local_controllability_responses.npz")
    assert arrays["current_action"].shape == (240, 36)
    assert arrays["rho_action"].shape == (512, 36)
    assert arrays["rho_value_float32"].shape == (512,)
    assert arrays["rho_value_float64"].shape == (512,)
    assert arrays["rho_replay_residual_clip_m"].shape == (512,)
    np.testing.assert_array_equal(
        arrays["rho_value_float64"].astype(np.float32),
        arrays["rho_value_float32"],
    )
    np.testing.assert_allclose(
        arrays["rho_replay_residual_clip_m"],
        0.05 * arrays["rho_value_float64"],
        rtol=0.0,
        atol=0.0,
    )
