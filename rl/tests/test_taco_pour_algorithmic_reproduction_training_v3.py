import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from video_to_spider.rl.algorithmic_benchmark import validate_budget_plan
from video_to_spider.rl.algorithmic_training import exact_truncated_normal_kl


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v3.yaml"
RUN = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v3"
CHECKPOINT_GATE = ROOT / "runs/taco_pour_algorithmic_checkpoint_roundtrip_v3"
NUMERIC_RECLASSIFICATION = (
    ROOT / "runs/taco_pour_likelihood_identity_numeric_reclassification_v3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_contract_freezes_numeric_gate_and_original_training_budget():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["schema"] == "taco_pour_algorithmic_reproduction_training_benchmark_v3"
    assert contract["classification"] == (
        "local_algorithmic_reproduction_after_replay_identity_and_numeric_likelihood_gate_fix"
    )
    assert contract["v1_evidence_immutable"] is True
    assert contract["v2_evidence_immutable"] is True
    assert contract["v2_checkpoint_resume_allowed"] is False
    assert contract["runtime"]["chunk_commit_allowed"] is False
    assert contract["runtime"]["old_checkpoint_warm_start_allowed"] is False
    assert contract["frozen_task"]["residual_execution_identity"] == (
        "original_formal_reference_plus_residual"
    )
    assert contract["frozen_task"]["actor_mini_epochs"] == 1
    assert contract["frozen_task"]["actor_learning_rate"] == 1.0e-4
    gate = contract["quality_gates"]["pre_optimizer_likelihood_identity"]
    epsilon = torch.finfo(torch.float32).eps
    assert gate["mu_atol"] == 8 * epsilon
    assert gate["sigma_atol"] == 8 * epsilon
    assert gate["ratio_atol"] == 128 * epsilon
    assert gate["semantic_identity_hard_fail"] == 1.0e-4
    milestones = validate_budget_plan(contract["training_budget"]["seed0_milestones"])
    assert [row["epoch"] for row in milestones] == [62, 313, 625]


def test_v3_B0_is_exact_replay_and_authorizes_training():
    report = json.loads((RUN / "candidate_B/seed_0/report.json").read_text())
    assert report["status"] == "passed_training_may_start"
    assert report["optimizer_updates"] == 0
    assert report["candidate_B_step0"]["successful_intervals"] == 30
    assert report["candidate_B_step0"]["first_failure_endpoint"] == 51
    assert report["replay"] == report["candidate_B_step0"]
    assert all(report["checks"].values())
    assert report["maximum_abs_actor_mu"] == 0.0
    assert report["maximum_abs_deterministic_action"] == 0.0
    assert report["maximum_abs_command_difference"] == 0.0
    assert report["maximum_abs_qpos_difference"] == 0.0
    assert report["maximum_abs_qvel_difference"] == 0.0
    arrays = np.load(RUN / "candidate_B/seed_0/step0_trajectory.npz")
    for field in ("ctrl", "qpos", "qvel", "score"):
        assert arrays[f"candidate_B_{field}"].tobytes() == arrays[f"replay_{field}"].tobytes()


def test_real_fresh_agent_checkpoint_roundtrip_is_complete_and_exact():
    report_path = CHECKPOINT_GATE / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed_no_optimizer_step"
    assert report["real_fresh_agent"] is True
    assert report["optimizer_updates"] == 0
    assert all(report["field_equality"].values())
    assert report["chunk_commit_allowed"] is False
    artifact = Path(report["artifact"]["path"])
    assert artifact == CHECKPOINT_GATE / "fresh_agent_roundtrip.pt.gz"
    assert report["artifact"]["artifact_sha256"] == _sha256(artifact)
    assert report["artifact"]["eligible_for_warm_start"] is False


def test_archived_v2_numeric_evidence_passes_only_C_D_reclassification():
    report = json.loads(
        (NUMERIC_RECLASSIFICATION / "report.json").read_text()
    )
    assert report["numerical_C_D"][
        "policy_output_numerical_equivalence_passed"
    ] is True
    assert report["numerical_C_D"]["semantic_identity_hard_fail"] is False
    assert report["full_v3_A_B_C_D_gate_replayed"] is False
    assert report["v2_training_status_changed"] is False
    assert report["warm_start_allowed"] is False


def test_v3_candidate_A_failure_is_formal_fail_closed_and_archived():
    report = json.loads((RUN / "candidate_A/seed_0/report.json").read_text())
    assert report["status"] == (
        "halted_fail_closed_pre_optimizer_likelihood_identity_epoch32"
    )
    assert report["completed_actor_updates"] == 31
    assert report["optimizer_update_executed_in_failed_epoch"] is False
    assert report["milestone_100k_reached"] is False
    assert report["candidate_B_training_started"] is False
    identity = report["pre_optimizer_identity"]
    assert identity["actor_parameter_hash_equal"] is True
    assert identity["actor_RMS_hash_equal"] is True
    assert identity["normalization_version_equal"] is True
    assert identity["max_abs_mu_difference"] == 10 * torch.finfo(torch.float32).eps
    assert identity["max_abs_mu_difference"] > identity["mu_tolerance"]
    assert identity["ratio_max_abs_error_from_one"] < identity["ratio_tolerance"]
    assert identity["numerical_identity_passed"] is False
    for key in (
        "failure_evidence",
        "last_completed_update_summary",
        "training_visitation_manifest",
    ):
        artifact = report[key]
        path = Path(artifact["path"])
        assert "TRASH" in path.parts
        assert artifact["repository_retention"] == "local_TRASH_only"
        assert artifact["available_in_clean_checkout"] is False
        assert len(artifact["sha256"]) == 64


def test_exact_truncated_KL_is_zero_for_identical_distributions_and_positive_otherwise():
    mu = torch.tensor([[0.1, -0.3], [0.8, 0.0]], dtype=torch.float32)
    sigma = torch.tensor([[0.7, 1.1], [0.5, 0.9]], dtype=torch.float32)
    low = torch.tensor([[-1.0, -0.2], [0.0, -1.0]], dtype=torch.float32)
    high = torch.tensor([[0.9, 1.0], [1.0, 0.4]], dtype=torch.float32)
    same = exact_truncated_normal_kl(mu, sigma, mu, sigma, low, high)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=2.0e-12, rtol=0.0)
    moved = exact_truncated_normal_kl(mu, sigma, mu + 0.2, sigma * 0.9, low, high)
    assert torch.all(moved > 0.0)
