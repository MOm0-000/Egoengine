"""Regression checks for the non-promotable PPO mu/sigma attribution run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_policy_distribution_attribution_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_transparency_gate_is_bitwise_and_full_width():
    report = json.loads(
        (ROOT / "runs/taco_pour_training_trace_transparency_v4/report.json").read_text()
    )
    assert report["schema"] == "taco_pour_training_trace_transparency_v4"
    assert report["status"] == "logging_transparency_gate_passed"
    assert all(report["initial_state_bitwise_equal"].values())
    assert all(report["final_state_bitwise_equal"].values())
    contract = report["training_trace_contract"]
    assert contract["schema"] == "taco_ppo_training_visitation_v4"
    assert contract["sample_count"] == 4
    assert all(shape == [4, 36] for shape in contract["full_action_array_shapes"].values())
    assert contract["actor_sigma_strictly_positive"] is True
    assert contract["sampled_action_clamp_exact"] is True
    assert contract["extra_actor_forward_for_logging"] is False
    script = ROOT / "scripts/audit_taco_pour_training_trace_transparency_v4.py"
    assert _sha256(script) == report["inputs"]["audit_script"]["sha256"]


def test_attribution_run_is_complete_but_cannot_promote():
    report = json.loads((RUN / "report.json").read_text())
    assert report["schema"] == "taco_pour_policy_distribution_attribution_run_v1"
    assert report["status"] == "diagnostic_complete_no_commit"
    assert report["incoming_boundary_restored_after_diagnostic"] is True
    assert report["committed_reference_index"] == 20
    assert report["task_success"] is False
    assert report["promotion"] == {
        "allowed": False,
        "chunk_committed": False,
        "committed_boundary_written": False,
        "optimized_trajectory_written": False,
        "task_success_evidence": False,
        "CPU_score_performance_comparison_allowed": False,
        "reason": (
            "GPU training is nondeterministic; this run only measures the frozen "
            "diagnostic training distribution."
        ),
    }
    assert not list(RUN.glob("**/optimized_trajectory.npz"))
    assert not list(RUN.glob("**/committed_boundary*"))
    trace = report["diagnostic"]["training_runs"][0]["training_visitation"]
    assert trace["schema"] == "taco_ppo_training_visitation_v4"
    assert trace["status"] == "complete"
    assert len(trace["epochs"]) == 8
    assert sum(epoch["sample_count"] for epoch in trace["epochs"]) == 1280
    assert trace["logging_semantics"]["actor_distribution_parameters"].endswith(
        "no extra actor forward pass is performed"
    )


def test_v4_raw_arrays_and_attribution_are_bound_to_evidence():
    report = json.loads((RUN / "report.json").read_text())
    trace = report["diagnostic"]["training_runs"][0]["training_visitation"]
    required = (
        "sampled_action_preclamp",
        "sampled_action_clamped",
        "actor_mu",
        "actor_sigma",
        "reference_ctrl",
        "requested_residual",
        "effective_residual_after_ctrlrange",
        "residual_lost_to_ctrlrange",
    )
    for epoch in trace["epochs"]:
        path = RUN / "ppo_diagnostic_chunk_20/training_visitation" / Path(
            epoch["visits"]["path"]
        ).name
        assert _sha256(path) == epoch["visits"]["sha256"]
        with np.load(path, allow_pickle=False) as raw:
            assert all(raw[name].shape == (160, 36) for name in required)
            np.testing.assert_array_equal(
                raw["sampled_action_clamped"],
                np.clip(raw["sampled_action_preclamp"], -1.0, 1.0),
            )
            assert (raw["actor_sigma"] > 0).all()
            np.testing.assert_array_equal(
                raw["requested_residual"],
                raw["effective_residual_after_ctrlrange"]
                + raw["residual_lost_to_ctrlrange"],
            )

    audit = json.loads((RUN / "policy_distribution_attribution_audit.json").read_text())
    assert audit["schema"] == "taco_pour_policy_distribution_attribution_audit_v1"
    assert audit["status"] == "completed_descriptive_no_promotion"
    script = ROOT / "scripts/audit_taco_pour_policy_distribution_attribution.py"
    assert _sha256(script) == audit["inputs"]["audit_script"]["sha256"]
    finger = audit["grouped_attribution"]["all"]["both_fingers"]
    assert finger["component_count"] == 30720
    assert finger["mu_outside_feasible"]["count"] == 7048
    assert finger["officially_clamped_mu_outside_actuator_feasible"]["count"] == 2369
    observed = finger["observed_layers"]
    assert observed["official_clamp_changed_component_count"] == 12557
    assert observed["ctrlrange_lost_component_count"] == 2147
    assert observed["fully_blocked_component_count"] == 1986
    assert observed[
        "ctrlrange_lost_while_officially_clamped_mu_outside_actuator_feasible"
    ] == 1552
    assert observed[
        "ctrlrange_lost_while_officially_clamped_mu_inside_actuator_feasible"
    ] == 595
    numeric = audit["contracts"]["reference_ctrlrange_numeric_contract"]
    assert numeric["outside_tolerance_component_count"] == 0
    assert numeric["maximum_violation"] < numeric["tolerance"]
