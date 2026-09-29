"""Strict-success promotion and progressive chunk execution evidence tests."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "configs/taco_pour_candidate_D_strict_success_promotion_v1.yaml"
PROMOTION_RUNNER = ROOT / "scripts/promote_taco_pour_candidate_D_strict_success_v1.py"
PROMOTION_ROOT = ROOT / "runs/taco_pour_candidate_D_strict_success_promotion_v1"
PROGRESSION = ROOT / "configs/taco_pour_progressive_chunk_execution_v1.yaml"
REPLAY_RUNNER = ROOT / "scripts/run_taco_pour_progressive_replay_v1.py"
SOLVER = ROOT / "configs/taco_pour_candidate_D_chunk_local_solver_v1.yaml"
SOLVER_RUNNER = ROOT / "scripts/run_taco_pour_candidate_D_chunk_local_v1.py"
PROGRESSIVE_COMPARISON = (
    ROOT / "runs/taco_pour_progressive_chunk_execution_v1/comparison.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_snapshot(path: Path):
    return torch.load(
        io.BytesIO(gzip.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )


def test_promotion_contract_binds_unique_prefrozen_milestone():
    contract = yaml.safe_load(PROMOTION.read_text())
    assert contract["schema"] == "taco_pour_candidate_D_strict_success_promotion_v1"
    assert contract["promotion"] == {
        "candidate": "D",
        "seed": 2,
        "epoch": 125,
        "source_endpoint": 20,
        "target_endpoint": 40,
        "lookahead_endpoint": 60,
        "validation_intervals": 40,
        "committed_intervals": 20,
        "commit_only_after_exact_revalidation": True,
        "promotion_reason": "prefrozen_D5_fixed_milestone_strict_40_of_40",
        "algorithmic_chunk_generation": True,
        "paper_cost_comparison_eligible": False,
        "retraining_allowed": False,
        "checkpoint_substitution_allowed": False,
        "threshold_relaxation_allowed": False,
    }
    assert contract["implementation_contract"]["promotion_runner_sha256"] == _sha256(
        PROMOTION_RUNNER
    )
    for row in contract["inputs"].values():
        assert _sha256(Path(row["path"])) == row["sha256"]


def test_materialized_promotion_is_exact_and_commits_only_first_half():
    report_path = PROMOTION_ROOT / "report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    assert report["status"] == "promoted_exact_40_of_40_chunk_20_40_committed"
    assert report["exact_revalidation"]["all_historical_arrays_bitwise_equal"] is True
    assert all(report["exact_revalidation"]["array_checks"].values())
    assert report["exact_revalidation"]["summary"]["successful_intervals"] == 40
    assert report["exact_revalidation"]["summary"]["first_failure_endpoint"] is None
    assert report["exact_revalidation"]["bounded_mu_outside_support_count"] == 0
    assert report["committed_intervals"] == 20
    assert report["committed_endpoints"] == [21, 40]
    assert report["lookahead_validated"] is True
    assert report["lookahead_committed"] is False
    assert report["paper_cost_comparison_eligible"] is False

    historical = np.load(
        ROOT
        / "runs/taco_pour_algorithmic_reproduction_training_v6/training/"
        "candidate_D/seed_2/validations/epoch_0125.npz"
    )
    promoted = np.load(PROMOTION_ROOT / "promoted_validation.npz")
    for name in historical.files:
        assert promoted[name].dtype == historical[name].dtype
        assert promoted[name].shape == historical[name].shape
        assert promoted[name].tobytes() == historical[name].tobytes()

    committed = np.load(PROMOTION_ROOT / "committed_chunk_20_40.npz")
    assert committed["source_endpoint"].tolist() == list(range(20, 40))
    assert committed["endpoint"].tolist() == list(range(21, 41))
    assert committed["ctrl"].shape[0] == 20
    assert committed["qpos"].tobytes() == promoted["qpos"][:20].tobytes()
    boundary = _load_snapshot(PROMOTION_ROOT / "committed_boundary_endpoint_40.pt.gz")
    assert np.asarray(boundary["time_indices"]).tolist() == [40]
    assert boundary["mujoco_warp_version"] == "3.13.0"
    assert report["boundary"]["restore_bitwise_exact"] is True


def test_progressive_replay_is_zero_residual_and_fail_closed_when_materialized():
    contract = yaml.safe_load(PROGRESSION.read_text())
    assert contract["active_stage"]["candidate_D_policy_allowed"] is False
    assert contract["active_stage"]["RNN_state_allowed"] is False
    assert contract["implementation_contract"]["replay_runner_sha256"] == _sha256(
        REPLAY_RUNNER
    )
    report_path = (
        ROOT
        / "runs/taco_pour_progressive_chunk_execution_v1/chunk_source_40/replay/report.json"
    )
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    assert report["candidate_D_policy_loaded"] is False
    assert report["RNN_state_used"] is False
    assert report["summary"]["deterministic_residual_bitwise_zero"] is True
    if report["summary"]["forty_of_forty"]:
        assert report["status"] == "replay_passed_chunk_40_60_committed"
        assert report["lookahead_committed"] is False
    else:
        assert report["status"] == "replay_failed_no_chunk_commit"
        assert report["chunk_commit_written"] is False
        assert report["next_action"] == "fresh_candidate_D_chunk_local_solve_required"


def test_chunk_local_solver_contract_is_fresh_fixed_milestone_only():
    contract = yaml.safe_load(SOLVER.read_text())
    row = contract["candidate_D"]
    assert row["source_endpoint"] == 40
    assert row["restart_seed_order"] == [0, 1, 2]
    assert row["fresh_actor_critic_optimizer_RMS_RNN"] is True
    assert row["actor_learning_rate"] == 1.0e-4
    assert row["actor_mini_epochs"] == 1
    assert row["fixed_milestone_epochs"] == [62, 125, 188, 250]
    assert row["first_strict_milestone_action"] == (
        "stop_no_commit_request_separate_promotion"
    )
    assert row["warm_start_allowed"] is False
    assert row["non_strict_checkpoint_selection_allowed"] is False
    assert row["LR_schedule_allowed"] is False
    assert contract["implementation_contract"]["chunk_solver_runner_sha256"] == _sha256(
        SOLVER_RUNNER
    )


def test_three_restart_progressive_evidence_stops_without_non_strict_commit():
    if not PROGRESSIVE_COMPARISON.is_file():
        return
    comparison = json.loads(PROGRESSIVE_COMPARISON.read_text())
    assert comparison["status"] == (
        "endpoint40_candidate_D_restart_protocol_exhausted_no_chunk_commit"
    )
    assert comparison["restart_seed_order"] == [0, 1, 2]
    assert comparison["restart_protocol_exhausted"] is True
    assert comparison["strict_fixed_milestone_events"] == []
    assert comparison["strict_success_found"] is False
    assert comparison["chunk_40_60_committed"] is False
    assert comparison["non_strict_checkpoint_selected"] is False
    assert comparison["automatic_LR_schedule_executed"] is False
    assert comparison["generation_solver_physics_steps_after_endpoint40"] == 1_200_000
    assert comparison["fixed_milestone_medians"] == {
        "0": 9,
        "100k": 19,
        "200k": 20,
        "300.8k": 20,
        "400k": 17,
    }
    expected = {
        "0": [9, 17, 20, 20, 14],
        "1": [9, 19, 20, 20, 20],
        "2": [9, 20, 19, 20, 17],
    }
    labels = ["0", "100k", "200k", "300.8k", "400k"]
    for seed, curve in expected.items():
        row = comparison["candidate_D_restarts"][seed]
        assert row["fresh_actor_critic_optimizers_RMS_RNN"] is True
        assert row["warm_start_used"] is False
        assert [row["learning_curve"][label]["successful_intervals"] for label in labels] == curve
        assert all(
            row["learning_curve"][label]["forty_of_forty"] is False
            for label in labels
        )
        assert all(
            row["learning_curve"][label]["bounded_mu_outside_support_count"] == 0
            for label in labels
        )
