import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_low_level_contact_controllability_gate_L_v1"
CONTRACT = ROOT / "configs/taco_pour_low_level_contact_controllability_gate_L_v1.yaml"
BLOCKER = "physics_reference_and_XHand_contact_actuator_model_review_required"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads((RUN / "report.json").read_text())


def test_gate_L_contract_is_finite_read_only_and_not_paper_parameter_recovery():
    contract = yaml.safe_load(CONTRACT.read_text())
    report = _report()
    assert contract["schema"] == "taco_pour_low_level_contact_controllability_gate_L_v1"
    assert contract["status"] == "authorized_finite_read_only_gate_L"
    assert contract["paper_faithful"] is False
    assert report["status"] == "completed_finite_read_only_gate_L"
    assert report["paper_faithful"] is False
    assert report["training_executed"] is False
    assert report["policy_optimizer_steps"] == 0
    assert report["chunk_commit_written"] is False
    assert report["source57"]["same_model_snapshot_fields_bitwise_restored"] is True
    assert report["diagnostic_model"][
        "formal_scene_model_is_unchanged_during_L0_and_L1"
    ] is True


def test_free_space_equivalence_uses_all_actuators_and_matches_formal_response():
    report = _report()["free_space_equivalence"]
    assert report["passed"] is True
    assert report["rollout_count"] == 72
    assert report["physics_substeps_per_rollout"] == 10
    assert report["effort_saturation_count"] == 0
    assert report["maximum_qpos_difference"] == 5.960464477539063e-08
    assert report["maximum_qvel_difference"] == 4.032626748085022e-07
    assert {row["direction"] for row in report["rows"]} == {-1, 1}
    assert len({row["actuator_index"] for row in report["rows"]}) == 36

    arrays = np.load(RUN / "free_space_equivalence.npz")
    assert arrays["formal_qpos"].shape == (72, 10, 50)
    assert arrays["impedance_qpos"].shape == (72, 10, 50)
    assert arrays["formal_qvel"].shape == (72, 10, 48)
    assert arrays["impedance_qvel"].shape == (72, 10, 48)
    assert np.max(np.abs(arrays["formal_qpos"] - arrays["impedance_qpos"])) == report[
        "maximum_qpos_difference"
    ]
    assert np.max(np.abs(arrays["formal_qvel"] - arrays["impedance_qvel"])) == report[
        "maximum_qvel_difference"
    ]

    scene = ET.parse(RUN / "free_space_scene.xml").getroot()
    assert len(scene.findall("./contact/*")) == 0
    assert len(scene.findall("./actuator/general")) == 36


def test_L0_and_conditional_L1_both_fail_the_fixed_three_step_gate_safely():
    report = _report()
    l0 = report["L0"]
    assert l0["executed"] is True
    assert l0["task_gain_search_executed"] is False
    assert l0["any_anchor_feasible"] is False
    np.testing.assert_array_equal(
        l0["anchors"]["formal_frozen_PPO"]["scores"],
        [1.0248640775680542, 1.0675162076950073, 1.109575867652893],
    )
    np.testing.assert_array_equal(
        l0["anchors"]["gate_A1_best_position_velocity_feedback"]["scores"],
        [1.0176074504852295, 1.049155354499817, 1.0826141834259033],
    )

    l1 = report["L1"]
    assert l1["executed"] is True
    assert l1["probe"]["identifiable"] is True
    assert l1["probe"]["force_target_n"] == 10.689228266477585
    assert l1["probe"]["abs_dFn_dxn_N_per_m"] == 72.73908704519272
    assert l1["probe"]["kf_dimensionless"] == 13.747766718307876
    assert l1["task_gain_or_force_target_search_executed"] is False
    assert l1["any_anchor_feasible"] is False
    np.testing.assert_array_equal(
        l1["anchors"]["formal_frozen_PPO"]["scores"],
        [1.0383347272872925, 1.1346522569656372, 1.1964021921157837],
    )
    np.testing.assert_array_equal(
        l1["anchors"]["gate_A1_best_position_velocity_feedback"]["scores"],
        [1.0266213417053223, 1.0595113039016724, 1.093111276626587],
    )
    for stage in (l0, l1):
        for anchor in stage["anchors"].values():
            assert anchor["feasible_three_of_three"] is False
            assert anchor["all_applied_effort_within_URDF_limits"] is True
            assert anchor["maximum_applied_effort_fraction"] <= 1.0
            assert anchor["bitwise_repeatable_on_CPU"] is True


def test_rollout_arrays_record_generalized_effort_without_legacy_torque_names():
    arrays = np.load(RUN / "gate_L_rollouts.npz")
    assert len(arrays.files) == 28
    assert not any("torque" in key for key in arrays.files)
    for stage in ("L0", "L1"):
        for anchor in (
            "formal_frozen_PPO",
            "gate_A1_best_position_velocity_feedback",
        ):
            prefix = f"{stage}_{anchor}"
            assert arrays[f"{prefix}_scores"].shape == (3,)
            assert arrays[f"{prefix}_requested_effort"].shape == (30, 36)
            assert arrays[f"{prefix}_applied_effort"].shape == (30, 36)
            assert arrays[f"{prefix}_force_correction"].shape == (30, 3)


def test_gate_L_closes_controller_search_and_protocol_moves_to_physics_review():
    report = _report()
    decision = report["decision"]
    assert decision["Gate_L_closed"] is True
    assert decision["classification"] == (
        "low_level_contact_controllability_not_demonstrated_under_the_two_"
        "predeclared_controllers"
    )
    assert decision["finite_negative_result_is_mathematical_infeasibility_proof"] is False
    assert decision["RL_architecture_gates_remain_closed"] is True
    assert decision["Gate_B_allowed_now"] is False
    assert decision["Gate_C_allowed_now"] is False
    assert decision["Gate_D_or_PPO_retraining_allowed_now"] is False
    assert decision["next_blocker"] == BLOCKER
    assert report["stopping_rule"][
        "additional_source57_controller_gain_contact_force_axis_frame_or_per_source_sweep_authorized"
    ] is False

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == BLOCKER
    assert protocol["blocking_checks"] == [BLOCKER]
    assert protocol["evidence_policy"]["current_revision"] == BLOCKER
    gate = protocol["evaluation"]["low_level_contact_controllability_gate_L"]
    assert gate["L0_effort_limited_joint_impedance"]["feasible_anchors"] == 0
    assert gate["L1_force_aware_impedance"]["feasible_anchors"] == 0
    assert gate["next_blocker"] == BLOCKER
    assert protocol["audit_results"]["low_level_contact_controllability_gate_L"] == str(
        RUN / "report.json"
    )


def test_gate_L_artifact_hashes_are_reconstructible():
    report = _report()
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    for row in report["artifacts"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert row["sha256"] == _sha256(path)
