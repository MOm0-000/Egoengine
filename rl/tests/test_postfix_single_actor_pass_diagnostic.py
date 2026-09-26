import gzip
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_postfix_single_actor_pass_diagnostic_v1"
AUDIT = ROOT / "runs/taco_pour_postfix_single_actor_pass_audit_v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_single_actor_pass_is_the_only_training_change_and_commits_nothing():
    report = json.loads((RUN / "report.json").read_text())
    manifest = json.loads((RUN / "credit_audit/manifest.json").read_text())
    assert report["schema"] == "taco_pour_postfix_single_actor_pass_candidate_v1"
    assert report["status"] == "completed_postfix_diagnostic_no_commit"
    assert report["actor_mini_epochs"] == 1
    assert report["critic_mini_epochs"] == 4
    assert report["training"]["total_samples"] == 1280
    assert report["training"]["old_actor_or_checkpoint_resume"] is False
    assert manifest["actor_updates"] == 8
    assert manifest["epochs"] == 8
    assert report["chunk_commit_written"] is False
    assert report["task_success_claimed"] is False


def test_strict_cpu_result_and_normalization_contract_are_exact():
    report = json.loads((RUN / "report.json").read_text())
    assert report["replay_validation"]["validated_steps"] == 31
    assert report["replay_validation"]["first_failure"]["endpoint"] == 51
    assert report["ppo_validation"]["validated_steps"] == 29
    assert report["ppo_validation"]["first_failure"]["endpoint"] == 49
    assert report["training_audit"]["actor_transfer_bitwise_equal"] is True
    checks = report["training_audit"]["observation_normalization"]
    assert checks["current_version"] == 8
    assert [
        row["maximum_abs_error_from_one"]
        for row in checks["pre_optimizer_likelihood_identity_checks"]
    ] == [0.0] * 8


def test_read_only_audit_reproduces_trace_and_reports_post_update_kl():
    report = json.loads((AUDIT / "report.json").read_text())
    assert report["schema"] == "taco_pour_postfix_single_actor_pass_audit_v1"
    assert report["status"] == "completed_read_only_no_algorithm_change"
    assert report["training_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["source_run"]["formal_trace_bitwise_reproduced"] == {
        "Replay": True,
        "PPO": True,
    }
    assert len(report["update_dynamics"]["updates"]) == 8
    post = report["update_dynamics"][
        "post_optimizer_fixed_probe_exact_KL_mean_by_update"
    ]
    assert post["max"] == 0.039736180923459874
    stochastic = report["deterministic_vs_stochastic"]["stochastic"]
    assert stochastic["passed_endpoint_40_count"] == 31
    assert stochastic["full_40_step_pass_count"] == 0


def test_checkpoint_is_retained_only_as_ineligible_compressed_evidence():
    transport = json.loads((RUN / "checkpoint_transport.json").read_text())
    artifact = RUN / transport["retained_compressed_artifact"]["path"]
    raw = artifact.read_bytes()
    assert _sha256(raw) == transport["retained_compressed_artifact"]["sha256"]
    assert _sha256(gzip.decompress(raw)) == transport["retained_compressed_artifact"][
        "gzip_payload_sha256"
    ]
    assert transport["eligible_for_warm_start"] is False
    assert transport["eligible_for_chunk_commit"] is False


def test_protocol_records_failure_and_requires_a_new_decision():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    assert protocol["training_ready_scope"] == (
        "binary_translation_oracle_36_of_40_endpoint57_failure_attribution_required"
    )
    run = protocol["evaluation"]["postfix_single_actor_pass_diagnostic"]
    assert run["sole_change"] == {"actor_mini_epochs": {"baseline": 4, "candidate": 1}}
    assert run["deterministic_PPO"] == {
        "successful_intervals": 28,
        "first_failure_endpoint": 49,
        "committed": False,
    }
    assert run["rerun_authorized"] is False
    assert run["chunk_commit_written"] is False
