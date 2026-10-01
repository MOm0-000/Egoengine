import gzip
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_postfix_fresh_ppo_credit_instrumented_v1"
REPORT = RUN / "report.json"
MANIFEST = RUN / "credit_audit/manifest.json"
TRANSPORT = RUN / "checkpoint_transport.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_postfix_run_is_fresh_valid_evidence_but_commits_nothing():
    report = json.loads(REPORT.read_text())
    assert report["schema"] == "taco_pour_postfix_fresh_ppo_credit_instrumented_v1"
    assert report["status"] == "completed_postfix_diagnostic_no_commit"
    assert report["valid_postfix_algorithm_evidence"] is True
    assert report["chunk_commit_written"] is False
    assert report["task_success_claimed"] is False
    assert report["training"]["old_actor_or_checkpoint_resume"] is False
    assert report["training"]["fixed_reset_endpoints"] == [20, 20, 20, 20]
    assert report["replay_validation"]["validated_steps"] == 31
    assert report["replay_validation"]["first_failure"]["endpoint"] == 51
    assert report["ppo_validation"]["validated_steps"] == 20
    assert report["ppo_validation"]["first_failure"]["endpoint"] == 40
    assert report["training_audit"]["actor_transfer_bitwise_equal"] is True


def test_every_epoch_uses_one_frozen_normalization_version():
    report = json.loads(REPORT.read_text())
    audit = report["training_audit"]["observation_normalization"]
    assert audit["current_version"] == 8
    assert len(audit["epochs"]) == 8
    assert [
        row["version_used_for_rollout_and_updates"] for row in audit["epochs"]
    ] == list(range(8))
    assert [
        row["maximum_abs_error_from_one"]
        for row in audit["pre_optimizer_likelihood_identity_checks"]
    ] == [0.0] * 8
    for index, row in enumerate(audit["epochs"]):
        assert row["statistics_committed_after_updates"] is True
        assert row["before"] == row["frozen_before_commit"]
        assert row["before"] != row["after"]
        if index:
            assert row["before"] == audit["epochs"][index - 1]["after"]
    assert max(
        row["maximum_abs_error"]
        for row in audit["pre_optimizer_critic_value_identity_checks"]
    ) <= 2e-5


def test_credit_chain_and_compressed_checkpoint_are_hash_bound():
    report = json.loads(REPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "taco_corrected_fresh_ppo_credit_instrumentation_v2"
    assert manifest["actor_updates"] == 32
    assert len(manifest["normalization_reports"]) == 8
    assert report["training_audit"]["credit_audit"]["sha256"] == _sha256(
        MANIFEST.read_bytes()
    )

    transport = json.loads(TRANSPORT.read_text())
    compressed = RUN / transport["retained_compressed_artifact"]["path"]
    assert _sha256(compressed.read_bytes()) == transport[
        "retained_compressed_artifact"
    ]["sha256"]
    assert _sha256(gzip.decompress(compressed.read_bytes())) == transport[
        "retained_compressed_artifact"
    ]["gzip_payload_sha256"]
    assert transport["eligible_for_warm_start"] is False
    assert transport["chunk_commit_written"] is False


def test_protocol_and_checkpoint_eligibility_keep_the_result_fail_closed():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["training_ready"] is False
    assert protocol["blocking_checks"] == [protocol["training_ready_scope"]]
    run = protocol["evaluation"]["postfix_fresh_ppo_credit_instrumented"]
    assert run["status"] == (
        "completed_valid_algorithm_evidence_strict_CPU_gate_failed_no_commit"
    )
    assert run["behavioral_audit_checkpoint_loaded"] is False
    assert run["checkpoint_resume_allowed"] is False
    eligibility = yaml.safe_load(
        (ROOT / "configs/taco_pour_ppo_checkpoint_eligibility_v1.yaml").read_text()
    )
    post_fix = eligibility["post_fix_checkpoint"]
    assert post_fix["valid_postfix_algorithm_evidence"] is True
    assert post_fix["eligible_for_warm_start"] is False
    assert post_fix["chunk_commit_written"] is False
