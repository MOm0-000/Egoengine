import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
A0_RUN = ROOT / "runs/taco_pour_action_feasibility_cem_v1"
RHO_RUN = ROOT / "runs/taco_pour_action_feasibility_minimum_rho_v1"
FRAME_RUN = ROOT / "runs/taco_pour_action_feasibility_reference_object_frame_v1"
A0_CONTRACT = ROOT / "configs/taco_pour_action_feasibility_cem_v1.yaml"
RHO_CONTRACT = ROOT / "configs/taco_pour_action_feasibility_minimum_rho_v1.yaml"
FRAME_CONTRACT = ROOT / "configs/taco_pour_action_feasibility_reference_object_frame_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_gate_A0_has_positive_controls_and_a_negative_source57_finite_search():
    report = _json(A0_RUN / "report.json")
    assert report["status"] == "completed_read_only_gate_A0"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["policy_optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["contract"]["sha256"] == _sha256(A0_CONTRACT)
    rows = {row["name"]: row for row in report["benchmarks"]}
    assert rows["formal_source45"]["feasible_CEM_sequence_count"] == 232
    assert rows["prefix_source50"]["feasible_CEM_sequence_count"] == 186
    assert rows["tail_source57"]["feasible_CEM_sequence_count"] == 0
    np.testing.assert_array_equal(
        rows["tail_source57"]["best_sequence"]["scores"],
        [1.002278447151184, 1.0133832693099976, 1.0380656719207764],
    )
    assert report["regression_gate"]["all_best_sequences_bitwise_repeatable_on_CPU"]
    assert report["decision"]["finite_optimizer_failure_is_mathematical_infeasibility_proof"] is False
    arrays = np.load(A0_RUN / "action_feasibility_candidates.npz")
    assert arrays["action"].shape == (720, 3, 36)
    assert int(arrays["feasible"].sum()) == 418


def test_gate_A_rho_is_joint_not_a_manual_sweep_and_finds_no_solution_through_three():
    report = _json(RHO_RUN / "report.json")
    contract = yaml.safe_load(RHO_CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate_A_rho"
    assert report["contract"]["sha256"] == _sha256(RHO_CONTRACT)
    assert report["support_parameter"]["lower"] == 1.0
    assert report["support_parameter"]["upper"] == 3.0
    assert report["optimizer"]["candidate_sequence_count"] == 512
    assert report["decision"]["feasible_candidate_count"] == 0
    assert report["decision"]["expanded_support_feasible_sequence_found"] is False
    assert report["decision"]["smallest_observed_feasible_rho"] is None
    assert report["decision"]["reported_rho_is_mathematical_minimum"] is False
    assert report["decision"]["automatic_upper_bound_expansion_authorized"] is False
    assert report["decision"]["manual_rho_or_axis_sweep_authorized"] is False
    assert contract["stopping_rule"]["manual_rho_sweep_allowed"] is False
    best = report["best_observed_candidate"]
    assert best["rho"] == 2.3921969949729545
    np.testing.assert_array_equal(
        best["scores"],
        [0.9932945966720581, 1.0009766817092896, 1.0196046829223633],
    )
    assert best["bitwise_repeatable_on_CPU"] is True
    arrays = np.load(RHO_RUN / "minimum_rho_candidates.npz")
    assert arrays["action"].shape == (512, 3, 36)
    assert int(arrays["feasible"].sum()) == 0


def test_reference_object_frame_gate_changes_only_translation_basis_and_is_negative():
    report = _json(FRAME_RUN / "report.json")
    contract = yaml.safe_load(FRAME_CONTRACT.read_text())
    assert report["status"] == "completed_read_only_gate_A_reference_object_frame"
    assert report["contract"]["sha256"] == _sha256(FRAME_CONTRACT)
    parameterization = report["translation_parameterization"]
    assert parameterization["normalized_action_indices"] == [0, 1, 2]
    assert parameterization["input_coordinate_frame"] == (
        "command_endpoint_reference_tool_frame"
    )
    assert parameterization["output_coordinate_frame"] == "world_frame_actuator_command"
    assert parameterization["local_component_scale_m"] == 0.05
    assert parameterization["world_component_hard_clip_before_ctrlrange"] is False
    assert parameterization["all_other_33_dimensions_use_current_world_frame_state_feasible_support"] is True
    assert report["decision"]["feasible_candidate_count"] == 0
    assert report["decision"]["reference_object_frame_feasible_sequence_found"] is False
    assert report["decision"]["finite_optimizer_failure_is_mathematical_infeasibility_proof"] is False
    assert report["decision"]["additional_frame_or_scale_search_authorized"] is False
    assert report["best_observed_candidate"]["invalid_actuator_bound_count"] == 0
    np.testing.assert_array_equal(
        report["best_observed_candidate"]["scores"],
        [0.9990851283073425, 1.0100034475326538, 1.0314950942993164],
    )
    assert report["regression_gate"]["best_candidate_bitwise_repeatable_on_CPU"]
    assert contract["stopping_rule"]["frame_scale_or_axis_sweep_allowed"] is False


def test_reference_object_frame_dataset_is_complete_and_transform_is_nontrivial():
    candidates = np.load(FRAME_RUN / "reference_object_frame_candidates.npz")
    assert candidates["local_action"].shape == (240, 3, 36)
    assert candidates["world_action"].shape == (240, 3, 36)
    assert int(candidates["invalid_actuator_bound_count"].sum()) == 0
    assert int(candidates["feasible"].sum()) == 0
    assert np.any(candidates["local_action"][:, :, :3] != candidates["world_action"][:, :, :3])
    np.testing.assert_array_equal(
        candidates["local_action"][:, :, 3:], candidates["world_action"][:, :, 3:]
    )
    best = np.load(FRAME_RUN / "reference_object_frame_best_sequence.npz")
    assert best["world_action"].shape == (3, 36)
    assert bool(best["feasible"]) is False


def test_gate_A_keeps_observation_policy_and_training_gates_closed():
    frame = _json(FRAME_RUN / "report.json")
    decision = frame["decision"]
    assert decision["next_blocker"] == "gate_A_active_corrective_parameterization_decision_required"
    assert decision["gate_B_observation_sufficiency_allowed"] is False
    assert decision["gate_C_policy_representability_allowed"] is False
    assert decision["gate_D_fresh_PPO_allowed"] is False
    assert decision["actor_LR_5e_minus_5_unblocked"] is False
    assert decision["learned_gate_training_authorized"] is False
    assert decision["reward_change_authorized"] is False
    assert decision["PPO_retraining_authorized"] is False
    assert decision["chunk_acceptance_or_commit_authorized"] is False


def test_protocol_and_half_lr_candidate_record_the_gate_A_classification():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    half_lr = yaml.safe_load((
        ROOT / "configs/taco_pour_postfix_single_actor_pass_lr_half_candidate_v1.yaml"
    ).read_text())
    blocker = "gate_A_active_corrective_parameterization_decision_required"
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == blocker
    assert protocol["blocking_checks"] == [blocker]
    assert protocol["evidence_policy"]["current_revision"] == blocker
    gate = protocol["evaluation"]["action_feasibility_reference_object_frame_gate"]
    assert gate["feasible_sequences"] == 0
    assert gate["additional_frame_or_scale_search_authorized"] is False
    assert gate["gate_B_observation_sufficiency_allowed"] is False
    assert gate["PPO_retraining_authorized"] is False
    assert half_lr["status"] == (
        "frozen_blocked_not_selected_after_gate_A_action_feasibility_classification"
    )
    evidence = half_lr["latest_state_entry_evidence"]
    assert evidence["action_feasibility_gate_A0_report"]["sha256"] == _sha256(
        A0_RUN / "report.json"
    )
    assert evidence["action_feasibility_minimum_rho_report"]["sha256"] == _sha256(
        RHO_RUN / "report.json"
    )
    assert evidence["action_feasibility_reference_object_frame_report"]["sha256"] == _sha256(
        FRAME_RUN / "report.json"
    )
