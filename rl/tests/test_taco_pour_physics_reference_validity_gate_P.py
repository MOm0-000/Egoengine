import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_physics_reference_validity_gate_P_v1"
CONTRACT = ROOT / "configs/taco_pour_physics_reference_validity_gate_P_v1.yaml"
BLOCKER = "physics_reference_and_XHand_contact_actuator_model_review_required"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads((RUN / "report.json").read_text())


def test_gate_P_contract_is_finite_read_only_and_closed_in_declared_order():
    contract = yaml.safe_load(CONTRACT.read_text())
    report = _report()
    assert contract["schema"] == "taco_pour_physics_reference_validity_gate_P_v1"
    assert contract["runtime"]["order"] == ["P1", "P2", "P0", "P3", "P4"]
    assert report["status"] == "completed_finite_read_only_gate_P_closed"
    assert report["paper_faithful"] is False
    assert report["runtime"]["training_allowed"] is False
    assert report["runtime"]["model_parameter_mutation_allowed"] is False
    assert report["runtime"]["mass_friction_effort_sweep_allowed"] is False
    assert report["decision"]["Gate_P_closed"] is True
    assert report["decision"]["Gate_B_or_C_allowed"] is False
    assert report["decision"]["PPO_retraining_allowed"] is False
    assert report["decision"]["chunk_commit_allowed"] is False
    assert report["P5"]["status"] == "external_author_information_required"


def test_P1_reports_declared_effort_saturation_without_false_force_claim():
    p1 = _report()["P1"]
    assert p1["anchors"]["formal_frozen_PPO"]["saturated_requests"] == 678
    assert p1["anchors"]["formal_frozen_PPO"]["total_requests"] == 1080
    assert p1["anchors"]["gate_A1_best_position_velocity_feedback"]["saturated_requests"] == 683
    assert p1["all_observed_task_saturation_is_on_finger_actuators"] is True
    assert p1["source57_measured_two_constraint_force_sum_n"] == 5.344611614942551
    assert p1["source57_normal_force_capacity"]["finger_only"][
        "maximum_aggregate_normal_force_n"
    ] == 41.472550753723155
    assert p1["measured_sum_exceeds_finger_only_aggregate_bound"] is False


def test_P2_identifies_same_patch_duplicate_constraints_without_volume_overlap():
    p2 = _report()["P2"]
    assert p2["part9_part23_intersection_volume_m3"] == 0.0
    assert p2["parts_have_no_positive_volume_overlap"] is True
    assert p2["source57_constraints_resolve_to_same_visual_surface_patch"] is True
    assert p2["source57_summed_force_is_valid_unique_physical_force_target"] is False
    metrics = p2["sources"]["57"]["pair_metrics"]
    assert metrics["contact_point_separation_m"] == 0.00034350547683191637
    assert metrics["contact_normal_angle_deg"] == 2.0075465781817767


def test_P0_records_backend_parity_failure_at_all_predeclared_sources():
    p0 = _report()["P0"]
    assert p0["all_sources_passed"] is False
    assert set(p0["sources"]) == {"45", "50", "57"}
    assert all(not row["passed"] for row in p0["sources"].values())
    assert any(not row["exact_match"] for row in p0["sources"]["45"]["contact_pair_sets"])
    assert all(row["exact_match"] for row in p0["sources"]["50"]["contact_pair_sets"])
    assert all(row["exact_match"] for row in p0["sources"]["57"]["contact_pair_sets"])
    assert p0["sources"]["45"]["maximum_differences"]["bowl_position_m"] > 0.0005
    assert p0["sources"]["57"]["maximum_differences"]["bowl_position_m"] < 0.0005
    assert p0["sources"]["57"]["maximum_differences"]["qvel"] > 0.05


def test_P3_and_P4_keep_finite_claim_boundaries_explicit():
    report = _report()
    p3 = report["P3"]
    assert p3["endpoints_without_any_reachable_finger"] == []
    assert p3["endpoints_without_reachable_thumb_plus_non_thumb"] == []
    assert "not a global IK proof" in p3["finite_sampling_statement"]
    assert len(p3["endpoints"]) == 41

    p4 = report["P4"]
    assert p4["current_friction_coefficient"] == 1.0
    assert p4["all_current_model_endpoints_feasible"] is False
    assert p4["infeasible_endpoints"] == [48, 55, 59, 60]
    assert "not a proof" in p4["finite_model_statement"]
    assert len(p4["endpoints"]) == 17


def test_gate_P_arrays_hashes_and_protocol_are_reconstructible():
    report = _report()
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    for row in report["artifacts"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert row["sha256"] == _sha256(path)
    assert np.load(RUN / "P0_native_MJWP_parity.npz")["source57_MJWP_qpos"].shape == (10, 50)
    assert (RUN / "P1_actuation_saturation_heatmap.png").stat().st_size > 100_000
    assert np.load(RUN / "P3_reference_contact_reachability.npz")["endpoint"].min() == 20
    assert np.load(RUN / "P4_reference_wrench_feasibility.npz")["required_wrench_object_frame"].shape == (17, 6)

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == BLOCKER
    assert protocol["blocking_checks"] == [BLOCKER]
    gate = protocol["evaluation"]["physics_reference_validity_gate_P"]
    assert gate["structural_issues"] == report["decision"]["structural_issues"]
    assert gate["Gate_B_C_allowed"] is False
    assert gate["next_blocker"] == BLOCKER
    assert protocol["audit_results"]["physics_reference_validity_gate_P"] == str(
        RUN / "report.json"
    )
