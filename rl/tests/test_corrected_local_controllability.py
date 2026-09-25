import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/taco_pour_corrected_local_controllability_v1/report.json"
CONTRACT = ROOT / "configs/taco_pour_corrected_local_controllability_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_local_controllability_is_frozen_read_only_evidence():
    report = _report()
    assert report["schema"] == "taco_pour_corrected_local_controllability_v1"
    assert report["status"] == "completed_read_only_counterfactual"
    assert report["paper_faithful"] is False
    assert report["scope"] == {
        "actor_frozen": True,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "diagnostic_actions_may_exceed_formal_residual_support": True,
        "diagnostic_continuation_after_tracking_failure": True,
        "objective_and_reward_unchanged": True,
        "optimizer_steps": 0,
        "training_performed": False,
    }
    baseline = report["baseline_regression"]
    assert baseline["formal_trace_exactly_reproduced"]
    assert baseline["validated_intervals"] == 27
    assert baseline["first_failure"] == {
        "control_interval": 47,
        "endpoint": 48,
        "object_roles": ["tool"],
        "reason": "tracking_boundary",
    }

    checks = report["global_checks"]
    assert checks["all_sources_restored_from_complete_state_and_rnn"]
    assert checks["zero_perturbation_branches_reproduce_formal_endpoint_48"]
    assert checks["material_ctrlrange_loss_count_gt_1e_8"] == 0
    assert checks["maximum_ctrlrange_loss"] == 0.0
    assert report["single_axis_sweep"]["branch_count"] == 144
    assert report["axis_ablation"]["branch_count"] == 32


def test_positive_wrist_y_authority_is_contradicted_before_endpoint_47():
    report = _report()
    analysis = report["single_axis_sweep"]["analysis"]
    expected_remove_y_mm = {"44": 94.078, "45": 87.886, "46": 98.743}

    for source, expected_mm in expected_remove_y_mm.items():
        row = analysis[source]["right_wrist_y"]
        fractions = row["fractions"]
        y_errors = [vector[1] for vector in row["final_bowl_position_error_vectors_m"]]
        assert fractions == [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        assert abs(y_errors[0] * 1000.0 - expected_mm) < 0.001
        assert y_errors[0] < y_errors[4]
        assert y_errors[-1] > y_errors[4]
        assert row["small_probe_direction_reducing_abs_bowl_y"] == "negative_0.25"

    source_45_y = [
        vector[1]
        for vector in analysis["45"]["right_wrist_y"][
            "final_bowl_position_error_vectors_m"
        ]
    ]
    assert source_45_y == sorted(source_45_y)
    assert analysis["45"]["right_wrist_y"]["signed_bowl_y_monotonic_over_full_grid"]


def test_drop_y_ablation_works_early_but_is_too_late_at_endpoint_47():
    analysis = _report()["axis_ablation"]["analysis"]
    expected_scores = {
        44: 0.783473551273346,
        45: 0.8270218372344971,
        46: 0.984746515750885,
        47: 1.065616488456726,
    }
    for source, expected_score in expected_scores.items():
        row = analysis[f"source_{source}/drop_axis/right_wrist_y"]
        assert abs(row["final_score"] - expected_score) < 1e-12
        if source < 47:
            assert row["final_score"] < 1.0
            assert row["first_tracking_failure_endpoint"] is None
        else:
            assert row["final_score"] > 1.0
            assert row["first_tracking_failure_endpoint"] == 48


def test_endpoint_47_wrist_moves_with_weak_bowl_y_transmission():
    analysis = _report()["single_axis_sweep"]["analysis"]["47"]
    for axis, qpos_index, maximum_bowl_y_span_m in (
        ("right_wrist_x", 0, 0.0007),
        ("right_wrist_y", 1, 0.0001),
        ("right_wrist_z", 2, 0.0001),
    ):
        row = analysis[axis]
        wrist_values = [qpos[qpos_index] for qpos in row["final_right_wrist_qpos"]]
        bowl_y = [vector[1] for vector in row["final_bowl_position_error_vectors_m"]]
        assert max(wrist_values) - min(wrist_values) > 0.014
        assert max(bowl_y) - min(bowl_y) < maximum_bowl_y_span_m
        assert all(
            "right-tool" not in contact
            for contacts in row["final_active_contacts"]
            for contact in contacts
        )


def test_diagnostic_snapshots_are_hash_bound_and_cannot_resume_training():
    report = _report()
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["runtime"]["checkpoint_resume_for_training_allowed"] is False
    assert contract["snapshots"]["diagnostic_only"] is True
    assert contract["snapshots"]["resume_authorized"] is False
    assert _sha256(CONTRACT) == report["contract"]["sha256"]

    for endpoint in (44, 45, 46, 47):
        row = report["source_snapshots"][str(endpoint)]
        path = (
            ROOT
            / "runs/taco_pour_corrected_local_controllability_v1/source_snapshots"
            / Path(row["path"]).name
        )
        assert row["snapshot_field_count"] == 362
        assert row["diagnostic_only"] is True
        assert row["resume_authorized"] is False
        assert _sha256(path) == row["artifact_sha256"]


def test_protocol_preserves_local_controllability_audit_after_later_audits():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["blocking_checks"] == [protocol["training_ready_scope"]]
    audit_path = Path(protocol["audit_results"]["corrected_local_controllability"])
    assert audit_path.parts[-3:] == (
        "runs",
        "taco_pour_corrected_local_controllability_v1",
        "report.json",
    )
    assert protocol["training_ready"] is False
