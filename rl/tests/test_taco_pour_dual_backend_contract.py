import hashlib
import gzip
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/taco_pour_gpu_train_cpu_validate_v1.yaml"
REPORT = ROOT / "runs/taco_pour_dual_backend_contract_v1/report.json"
RUNNER_REPORT = ROOT / "runs/taco_pour_dual_backend_runner_smoke_v1/report.json"


def test_gpu_cannot_accept_or_commit_cpu_validation_state():
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["training_backend"]["device"] == "cuda:0"
    assert contract["training_backend"]["may_decide_acceptance"] is False
    assert contract["training_backend"]["may_provide_committed_state"] is False
    assert contract["validation_backend"]["device"] == "cpu"
    assert contract["validation_backend"]["policy_inference_device"] == "cpu"
    assert contract["validation_backend"]["may_decide_acceptance"] is True
    assert contract["validation_backend"]["may_provide_committed_state"] is True
    assert contract["scheduler"]["required_validated_intervals"] == 40
    assert contract["scheduler"]["commit_control_intervals"] == 20
    assert contract["scheduler"]["commit_source"] == "cpu_validation_endpoint_20"


def test_real_cpu_to_gpu_snapshot_transfer_audit_is_bound_and_passed():
    report = json.loads(REPORT.read_text())
    assert report["status"] == "dual_backend_transfer_gate_passed"
    assert report["scope"] == {
        "training_executed": False,
        "ppo_budget_changed": False,
        "full_horizon_executed": False,
        "cpu_control_intervals_executed": 1,
        "gpu_control_intervals_executed": 0,
    }
    assert report["backend_contract"]["sha256"] == hashlib.sha256(
        CONTRACT.read_bytes()
    ).hexdigest()
    assert report["verified_transfer_count"] == 2
    assert [row["reference_endpoint"] for row in report["cpu_to_gpu_transfers"]] == [0, 1]
    assert all(row["warp_state_field_count"] == 342 for row in report["cpu_to_gpu_transfers"])
    assert all(row["cpu_to_gpu_bitwise_equal"] for row in report["cpu_to_gpu_transfers"])

    training = report["backend_runtime_records"]["training"]
    validation = report["backend_runtime_records"]["validation"]
    assert training["device"] == "cuda:0"
    assert validation["device"] == "cpu"
    assert training["physics_contract_sha256"] == validation["physics_contract_sha256"]
    assert training["warp_state_keys_sha256"] == validation["warp_state_keys_sha256"]
    assert training["runtime_contract_sha256"] != validation["runtime_contract_sha256"]


def test_formal_runner_commits_only_cpu_state_and_snapshots_its_inputs():
    report = json.loads(RUNNER_REPORT.read_text())
    assert report["status"] == "chunk_budget_reached_not_full_task_success"
    assert report["task_success"] is False
    assert report["committed_reference_index"] == 20
    assert len(report["chunks"]) == 1
    chunk = report["chunks"][0]
    assert chunk["mode"] == "replay"
    assert chunk["trials"][0]["validated_steps"] == 40
    assert chunk["training_runs"] == []
    assert chunk["commit_state_backend"] == "cpu"
    assert chunk["committed_boundary"]["source_backend"] == "cpu_validation"
    assert chunk["committed_boundary"]["reference_endpoint"] == 20
    boundary = Path(chunk["committed_boundary"]["path"])
    boundary_raw = gzip.decompress(boundary.read_bytes())
    assert hashlib.sha256(boundary.read_bytes()).hexdigest() == (
        chunk["committed_boundary"]["artifact_sha256"]
    )
    assert hashlib.sha256(boundary_raw).hexdigest() == (
        chunk["committed_boundary"]["uncompressed_pt_sha256"]
    )
    assert report["simulation_work"]["gpu_training_backend"] == {
        "control_intervals": 0,
        "physics_steps": 0,
        "verified_cpu_snapshot_transfers": 2,
    }
    assert report["simulation_work"]["cpu_validation_backend"] == {
        "control_intervals": 40,
        "physics_steps": 400,
    }
    for row in report["input_contract_snapshots"].values():
        snapshot = Path(row["snapshot_path"])
        assert snapshot.is_file()
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == row["sha256"]
    assert (
        report["objective"]["protocol_sha256"]
        == report["input_contract_snapshots"]["replay_rl_protocol"]["sha256"]
    )
    assert (
        report["backend_contract"]["sha256"]
        == report["input_contract_snapshots"]["dual_backend_contract"]["sha256"]
    )
