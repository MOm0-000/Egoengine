import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_simulator_contract_adjudication_gate_Q_v1"
CONTRACT = ROOT / "configs/taco_pour_simulator_contract_adjudication_gate_Q_v1.yaml"
FINAL_BLOCKER = (
    "blocked_pending_unpublished_simulator_contact_actuation_object_physics_"
    "and_objective_details"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads((RUN / "report.json").read_text())


def test_gate_Q_contract_is_final_finite_and_fail_closed():
    contract = yaml.safe_load(CONTRACT.read_text())
    report = _report()
    assert contract["schema"] == "taco_pour_simulator_contract_adjudication_gate_Q_v1"
    assert contract["runtime"]["order"] == ["Q1", "Q2", "Q3"]
    assert contract["stopping_rule"]["Gate_Q_is_final_physics_gate"] is True
    assert contract["stopping_rule"]["additional_physics_gate_allowed"] is False
    assert report["status"] == "completed_final_finite_simulator_contract_adjudication_closed"
    assert report["paper_faithful"] is False
    assert report["decision"]["Gate_Q_closed"] is True
    assert report["decision"]["Gate_Q_is_final_physics_gate"] is True
    assert report["decision"]["additional_physics_or_parity_gate_allowed"] is False
    assert report["decision"]["PPO_retraining_allowed"] is False
    assert report["decision"]["chunk_commit_allowed"] is False


def test_Q1_observes_intrinsic_formal_finger_effort_exceedance():
    q1 = _report()["Q1"]
    expected = {
        "Replay_zero_residual": (51, 3486, 267.06607818603516),
        "frozen_single_pass_PPO": (49, 3530, 278.0171775817871),
        "frozen_binary_prefix_plus_tail_semantic_oracle": (
            58, 3526, 285.6385040283203,
        ),
    }
    for name, (failure, count, maximum) in expected.items():
        row = q1["paths"][name]
        focus = row["focus_sources_44_59"]
        assert row["first_tracking_failure_endpoint"] == failure
        assert focus["physics_substeps"] == 160
        assert focus["requested_over_limit"] == count
        assert focus["actual_over_limit"] == count
        assert focus["maximum_actual_to_limit_ratio"] == maximum
        assert focus["maximum_abs_actual_minus_affine_law_n"] < 3.3e-4
        wrist_groups = [key for key in focus["groups"] if "wrist" in key]
        assert all(focus["groups"][key]["actual_over_limit"] == 0 for key in wrist_groups)
        assert sum(
            group["actual_over_limit"] for key, group in focus["groups"].items()
            if "wrist" not in key
        ) == count
    assert q1["formal_actuator_changed"] is False


def test_Q2_rejects_all_three_predeclared_collision_backend_contracts():
    q2 = _report()["Q2"]
    assert set(q2["variants"]) == {
        "current_mixed", "convex_only", "external_exact_SDF_only",
    }
    assert q2["passing_variants"] == []
    assert q2["unique_selected_variant"] is None
    assert q2["collision_backend_contract_frozen"] is False
    assert q2["backend_dependent_physics_acknowledged"] is True
    expected = {
        "current_mixed": (3009, 20),
        "convex_only": (3005, 20),
        "external_exact_SDF_only": (1537, 0),
    }
    for name, (pairs, duplicates) in expected.items():
        row = q2["variants"][name]
        assert row["compiled_pair_count"] == pairs
        assert row["visual_surface_patch_duplicate_count"] == duplicates
        assert row["parity"]["all_sources_passed"] is False
        assert row["source45_exact_pair_set_all_substeps"] is False
        assert row["freeze_condition_passed"] is False
        assert Path(row["xml"]).is_file()
        assert str(RUN) in row["xml"]


def test_Q3_supersedes_finite_P4_and_is_feasible_at_all_reference_endpoints():
    q3 = _report()["Q3"]
    assert q3["collision_variant"] == "current_mixed"
    assert q3["compiled_cone"] == "pyramidal"
    assert q3["artificial_16_sided_cone_used"] is False
    assert q3["all_endpoints_feasible"] is True
    assert q3["infeasible_endpoints"] == []
    assert set(q3["endpoints"]) == {str(endpoint) for endpoint in range(44, 61)}
    hand_counts = []
    for row in q3["endpoints"].values():
        assert row["contact_cone_feasible_without_hand_effort_limits"] is True
        assert row["feasible"] is True
        assert row["passive_unique_patch_candidates"] == 0
        assert row["pyramidal_rays"] == 4 * row["hand_unique_patch_candidates"]
        hand_counts.append(row["hand_unique_patch_candidates"])
    assert min(hand_counts) == 89
    assert max(hand_counts) == 95
    assert "qfrc_inverse_plus_qfrc_constraint" in q3["required_wrench_method"]


def test_gate_Q_artifacts_and_protocol_are_reconstructible():
    report = _report()
    assert report["contract"]["sha256"] == _sha256(CONTRACT)
    for row in report["artifacts"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert row["sha256"] == _sha256(path)
    q1 = np.load(RUN / "Q1_formal_actuator_demand.npz")
    assert q1["Replay_zero_residual_actual"].shape == (400, 36)
    assert q1["frozen_single_pass_PPO_requested_ratio"].shape == (400, 36)
    q2 = np.load(RUN / "Q2_collision_backend_parity.npz")
    assert q2["current_mixed_source57_MJWP_qpos"].shape == (10, 50)
    q3 = np.load(RUN / "Q3_complete_contact_reference_wrench.npz")
    assert q3["required_inverse_dynamics_free_joint_force"].shape == (17, 6)
    assert bool(q3["feasible"].all())
    assert (RUN / "Q1_formal_actuator_demand_heatmap.png").stat().st_size > 100_000

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == FINAL_BLOCKER
    assert protocol["blocking_checks"] == [FINAL_BLOCKER]
    gate = protocol["evaluation"]["simulator_contract_adjudication_gate_Q"]
    assert gate["Q2"]["uniquely_frozen_collision_backend_contract"] is None
    assert gate["Q3"]["all_endpoints_feasible_with_declared_effort_limits"] is True
    assert gate["Gate_Q_is_final_physics_gate"] is True
    assert gate["Gate_B_C_reopened"] is False
    assert gate["exact_reproduction_status"] == FINAL_BLOCKER
    assert protocol["audit_results"]["simulator_contract_adjudication_gate_Q"] == str(
        RUN / "report.json"
    )
