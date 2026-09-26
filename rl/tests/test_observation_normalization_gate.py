import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/taco_pour_observation_normalization_gate_v1/report.json"
CONTRACT = ROOT / "configs/taco_pour_frozen_rollout_observation_normalization_v1.yaml"
ELIGIBILITY = ROOT / "configs/taco_pour_ppo_checkpoint_eligibility_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_lr_zero_likelihood_gate_passes_without_training_or_commit():
    report = json.loads(REPORT.read_text())
    assert report["status"] == "passed"
    assert report["task_level_training_executed"] is False
    assert report["chunk_commit_written"] is False
    assert report["checkpoint_written"] is False
    assert report["dry_run"]["samples"] == 160
    assert report["dry_run"]["mini_epochs"] == 4
    assert report["repeat_forward"][
        "pre_optimizer_ratio_max_abs_error_from_one"
    ] == 0.0
    assert report["final_ratio_max_abs_error_from_one"] == 0.0
    assert report["critic_raw_output_max_abs_error"] == 0.0
    assert all(report["checks"].values())


def test_observation_normalization_snapshot_is_immutable_during_ppo_passes():
    report = json.loads(REPORT.read_text())
    audit = report["normalization"]
    assert audit["schema"] == "frozen_rollout_observation_normalization_v1"
    assert audit["current_version"] == 0
    assert len(audit["epochs"]) == 1
    epoch = audit["epochs"][0]
    assert epoch["statistics_committed_after_updates"] is False
    assert epoch["before"] == epoch["after"]
    assert audit["pre_optimizer_likelihood_identity_checks"] == [{
        "epoch": 1,
        "samples": 160,
        "maximum_abs_error_from_one": 0.0,
        "tolerance": 5e-06,
    }]
    critic = audit["pre_optimizer_critic_value_identity_checks"][0]
    assert critic["samples"] == 160
    assert critic["maximum_abs_error"] <= critic["tolerance"]


def test_contract_binds_gate_and_does_not_authorize_fresh_training():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["gate"]["report_sha256"] == _sha256(REPORT)
    assert contract["gate"]["status"] == "passed"
    assert contract["promotion"]["implementation_blocker_resolved"] is True
    assert contract["promotion"]["fresh_task_level_ppo_authorized_by_this_contract"] is False
    assert contract["promotion"]["old_misaligned_checkpoint_resume_allowed"] is False


def test_every_pre_fix_checkpoint_is_behavioral_audit_only():
    policy = yaml.safe_load(ELIGIBILITY.read_text())
    assert policy["default"] == {
        "eligible_for_warm_start": False,
        "eligible_for_algorithm_comparison": False,
    }
    assert policy["post_fix_checkpoint"] == {
        "exists": False,
        "fresh_training_required": True,
        "old_checkpoint_resume_allowed": False,
    }
    for row in policy["affected_runs"].values():
        assert row["ppo_obsnorm_likelihood_misaligned"] is True
        assert row["eligible_for_warm_start"] is False
        assert row["eligible_for_algorithm_comparison"] is False
        assert row["behavioral_audit_only"] is True
