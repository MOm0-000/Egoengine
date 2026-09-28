import hashlib
import json
from pathlib import Path

import torch
import yaml

from video_to_spider.rl.algorithmic_benchmark import validate_budget_plan


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v4.yaml"
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v4"
CHECKPOINT_GATE = ROOT / "runs/taco_pour_algorithmic_checkpoint_roundtrip_v4"
PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_contract_freezes_canonical_policy_and_original_learning_settings():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == (
        "taco_pour_algorithmic_reproduction_training_benchmark_v4"
    )
    assert contract["classification"] == (
        "local_algorithmic_reproduction_after_canonical_old_policy_likelihood_fix"
    )
    assert contract["v3_evidence_immutable"] is True
    assert contract["v3_checkpoint_resume_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert contract["runtime"]["old_checkpoint_warm_start_allowed"] is False
    assert contract["runtime"]["order"][:2] == [
        "candidate_B_step0_gate",
        "checkpoint_roundtrip_gate",
    ]
    task = contract["frozen_task"]
    assert task["actor_mini_epochs"] == 1
    assert task["critic_mini_epochs"] == 4
    assert task["actor_learning_rate"] == 1.0e-4
    assert task["observation_dimension"] == 236
    gate = contract["quality_gates"]["pre_optimizer_likelihood_identity"]
    epsilon = torch.finfo(torch.float32).eps
    assert gate["rollout_to_canonical_ratio_atol"] == 128 * epsilon
    assert gate["canonical_ratio_atol"] == epsilon
    assert gate["mu_difference_is_diagnostic_only"] is True
    assert gate["sigma_difference_is_diagnostic_only"] is True
    milestones = validate_budget_plan(
        contract["training_budget"]["seed0_milestones"]
    )
    assert [row["epoch"] for row in milestones] == [62, 313, 625]


def test_v4_is_the_active_algorithmic_evidence():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    v3 = protocol["evaluation"]["algorithmic_reproduction_training_benchmark_v3"]
    v4 = protocol["evaluation"]["algorithmic_reproduction_training_benchmark_v4"]
    assert v3["active_algorithmic_evidence"] is False
    assert v3["superseded_as_active_evidence_by"] == (
        "algorithmic_reproduction_training_benchmark_v4"
    )
    assert v4["active_algorithmic_evidence"] is True
    assert v4["status"] == "seed0_100k_B_then_A_completed_no_chunk_commit"
    assert v4["canonical_old_policy_gate"]["all_124_actor_updates_passed"] is True
    assert v4["seed0_100k"]["candidate_B"] == {
        "successful_intervals": 36,
        "first_failure_endpoint": 57,
    }
    assert v4["seed0_100k"]["candidate_A"] == {
        "successful_intervals": 29,
        "first_failure_endpoint": 50,
    }
    assert v4["chunk_commit_written"] is False


def test_v4_implementation_hashes_are_bound_to_current_files():
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "state_feasible_distribution_sha256": (
            ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"
        ),
        "algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training.py"
        ),
        "benchmark_runner_sha256": (
            ROOT / "scripts/run_taco_pour_algorithmic_reproduction_training_v3.py"
        ),
        "checkpoint_roundtrip_runner_sha256": (
            ROOT / "scripts/audit_taco_pour_algorithmic_checkpoint_roundtrip_v3.py"
        ),
        "training_runner_sha256": (
            ROOT / "scripts/run_taco_pour_algorithmic_training_v3.py"
        ),
    }
    for key, path in paths.items():
        assert contract["implementation_contract"][key] == _sha256(path)


def test_v4_B0_and_checkpoint_gates_when_materialized():
    if not RUN.exists() or not CHECKPOINT_GATE.exists():
        return
    b0 = json.loads((RUN / "candidate_B/seed_0/report.json").read_text())
    assert b0["status"] == "passed_training_may_start"
    assert b0["candidate_B_step0"]["successful_intervals"] == 30
    assert b0["candidate_B_step0"]["first_failure_endpoint"] == 51
    assert b0["replay"] == b0["candidate_B_step0"]
    assert all(b0["checks"].values())
    checkpoint = json.loads((CHECKPOINT_GATE / "report.json").read_text())
    assert checkpoint["status"] == "passed_no_optimizer_step"
    assert all(checkpoint["field_equality"].values())


def test_v4_seed0_100k_stage_is_complete_and_never_commits_a_chunk():
    comparison = json.loads((RUN / "comparison.json").read_text())
    assert comparison["status"] == (
        "seed0_100k_B_then_A_completed_no_chunk_commit"
    )
    assert comparison["chunk_commit_written"] is False
    assert comparison["strict_40_of_40_success"] is False
    candidate_b = comparison["candidates"]["B"]
    candidate_a = comparison["candidates"]["A"]
    assert candidate_b["at_0"]["successful_intervals"] == 30
    assert candidate_b["at_100k"]["successful_intervals"] == 36
    assert candidate_b["at_100k"]["first_failure_endpoint"] == 57
    assert candidate_b["at_100k"]["meaningful_refinement"] is True
    assert candidate_a["at_0"]["successful_intervals"] == 36
    assert candidate_a["at_100k"]["successful_intervals"] == 29
    assert candidate_a["at_100k"]["first_failure_endpoint"] == 50
    for candidate in (candidate_a, candidate_b):
        assert candidate["physics_steps"] == 99_200
        assert candidate["actor_updates"] == 62
        assert candidate["checkpoint_roundtrip_exact"] is True
        assert candidate["all_identity_gates_passed"] is True
        assert candidate["maximum_canonical_optimizer_ratio_error"] == 0.0
        assert candidate["maximum_rollout_canonical_ratio_error"] <= (
            128 * torch.finfo(torch.float32).eps
        )


def test_v4_A_exceeds_retired_mu_gate_without_likelihood_mismatch():
    comparison = json.loads((RUN / "comparison.json").read_text())
    candidate_a = comparison["candidates"]["A"]
    epsilon = torch.finfo(torch.float32).eps
    assert candidate_a["maximum_rollout_canonical_mu_difference"] > 8 * epsilon
    assert candidate_a["maximum_rollout_canonical_ratio_error"] <= 128 * epsilon
    assert candidate_a["maximum_canonical_optimizer_ratio_error"] == 0.0
