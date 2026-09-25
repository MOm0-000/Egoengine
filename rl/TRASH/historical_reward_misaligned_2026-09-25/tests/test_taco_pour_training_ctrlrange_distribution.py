"""Evidence checks for the non-promotable 3+1 ctrlrange diagnostic."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_ctrlrange_training_distribution_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_is_hash_bound_complete_and_cannot_be_promoted():
    report = json.loads((RUN / "report.json").read_text())
    assert report["schema"] == "taco_pour_ctrlrange_training_distribution_run_v1"
    assert report["status"] == "diagnostic_complete_no_commit"
    assert report["tracking_variant"] == "tool_only"
    assert report["local_settings"]["training_start_distribution"] == [20, 20, 20, 46]
    assert report["task_success"] is False
    assert report["committed_reference_index"] == 20
    assert report["incoming_boundary_restored_after_diagnostic"] is True
    assert report["promotion"] == {
        "allowed": False,
        "chunk_committed": False,
        "committed_boundary_written": False,
        "optimized_trajectory_written": False,
        "task_success_evidence": False,
        "CPU_score_performance_comparison_allowed": False,
        "reason": (
            "GPU training is nondeterministic; this run only measures the v3 "
            "training distribution."
        ),
    }
    assert not (RUN / "optimized_trajectory.npz").exists()
    assert not list(RUN.glob("committed_boundary_*"))
def test_v3_trace_contains_exact_frozen_budget_and_residual_identity():
    report = json.loads((RUN / "report.json").read_text())
    visitation = report["diagnostic"]["training_runs"][0]["training_visitation"]
    manifest_path = Path(visitation["path"])
    assert _sha256(manifest_path) == visitation["sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "taco_ppo_training_visitation_v3"
    assert len(manifest["epochs"]) == 8
    total = 0
    for epoch, artifact in enumerate(manifest["epochs"], start=1):
        assert artifact["epoch"] == epoch
        assert artifact["sample_count"] == 160
        path = Path(artifact["visits"]["path"])
        assert _sha256(path) == artifact["visits"]["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["source_endpoint"].reshape(40, 4)[0].tolist() == [20, 20, 20, 46]
            np.testing.assert_array_equal(
                arrays["requested_residual"],
                arrays["effective_residual_after_ctrlrange"]
                + arrays["residual_lost_to_ctrlrange"],
            )
        total += artifact["sample_count"]
    assert total == 1280


def test_distribution_measurement_matches_raw_training_trace():
    audit = json.loads((RUN / "ctrlrange_distribution_audit.json").read_text())
    assert audit["schema"] == "taco_pour_training_ctrlrange_distribution_audit_v1"
    assert audit["status"] == "descriptive_measurement_complete_no_severity_threshold"
    scope = audit["scopes"]["all_training_samples"]
    assert scope["world_step_samples"] == 1280
    assert scope["truncated_component_count"] == 2271
    assert scope["fully_blocked_component_count"] == 2099
    assert scope["world_steps_with_any_finger_truncation"] == 1114
    assert scope["absolute_lost_over_absolute_requested"] == 0.04780075004962982
    assert scope["ctrlrange_truncation_path"] == {
        "requested_at_residual_limit_then_truncated": 929,
        "requested_below_residual_limit_but_truncated": 1342,
        "requested_at_residual_limit_then_fully_blocked": 839,
        "requested_below_residual_limit_but_fully_blocked": 1260,
    }
    assert audit["scopes"]["anchor_worlds_0_to_2"][
        "world_steps_with_any_finger_truncation"
    ] == 800
    assert audit["scopes"]["tail_world_3"][
        "world_steps_with_any_finger_truncation"
    ] == 314
    assert audit["scopes"]["outcome_endpoints_46_to_53"][
        "world_steps_with_any_finger_truncation"
    ] == 396
    assert all(
        row["rollout_time_slots_with_any_world_finger_truncation"] == 40
        for row in audit["per_epoch"]
    )
    assert audit["descriptive_findings"] == {
        "ctrlrange_loss_present_throughout_training": True,
        "all_eight_epochs_have_finger_truncation": True,
        "every_epoch_rollout_slot_has_at_least_one_affected_world": True,
        "loss_is_not_confined_to_tail_world": True,
        "below_residual_limit_truncation_exists": True,
        "causal_task_failure_claimed": False,
        "action_space_redesign_selected": False,
    }
