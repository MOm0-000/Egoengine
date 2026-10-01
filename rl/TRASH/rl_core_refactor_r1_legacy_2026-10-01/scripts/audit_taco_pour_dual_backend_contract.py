#!/usr/bin/env python3
"""Verify exact CPU-commit transfer into the non-authoritative GPU backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import (
    _backend_runtime_record,
    load_accepted_initialization,
    load_dual_backend_contract,
    verify_runtime_model,
)
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml",
    )
    parser.add_argument(
        "--initialization-report",
        type=Path,
        default=ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/replay_rl_protocol.yaml",
    )
    parser.add_argument(
        "--objective-profile",
        type=Path,
        default=ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml",
    )
    parser.add_argument(
        "--observation-profile",
        type=Path,
        default=ROOT / "configs/taco_pour_observation_local_236d_v1.yaml",
    )
    parser.add_argument(
        "--backend-contract",
        type=Path,
        default=ROOT / "configs/taco_pour_gpu_train_cpu_validate_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_dual_backend_contract_v1/report.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _load_ego_config,
        _load_reference,
        torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action

    contract, contract_artifact = load_dual_backend_contract(args.backend_contract)
    objective = load_runtime_objective(
        args.protocol,
        args.objective_profile,
        tracking_variant="tool_only",
        # This is a no-training state-transfer audit.  The formal training gate
        # is intentionally closed after the single corrected PPO authorization
        # was consumed, but that must not prevent revalidating backend transfer.
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        args.protocol,
        args.observation_profile,
        require_run_ready=False,
    )
    initial, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )

    def make_env(device, *, asymmetric_critic):
        config = _load_ego_config(str(args.config), device)
        reference = _load_reference(
            config.data_path, device, expected_frequency=30
        )
        env = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=asymmetric_critic,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
            ),
        )
        verify_runtime_model(
            env.env.model_cpu, initialization["validated_physics_contract"]
        )
        values = [
            torch.as_tensor(initial[name][None], device=device, dtype=torch.float32)
            for name in ("qpos", "qvel", "ctrl")
        ]
        env._write_state(*values, np.array([True]))
        env._last_ctrl = values[2].clone()
        env._check_capacity()
        return env

    cpu_env = make_env(
        contract["validation_backend"]["device"], asymmetric_critic=False
    )
    gpu_env = make_env(
        contract["training_backend"]["device"], asymmetric_critic=True
    )
    cpu = MJWPChunkBackend(cpu_env)
    gpu = MJWPChunkBackend(gpu_env)
    transfers = []
    for endpoint in (0, 1):
        if endpoint == 1:
            if not cpu.step(replay_action(cpu, 0), 0):
                raise RuntimeError("CPU Replay failed before endpoint 1")
        boundary = cpu.snapshot()
        gpu.restore(boundary)
        gpu.verify_restored_snapshot(boundary)
        transfers.append({
            "reference_endpoint": endpoint,
            "snapshot_schema": boundary["snapshot_schema"],
            "warp_state_field_count": len(boundary["warp_state_keys"]),
            "cpu_to_gpu_bitwise_equal": True,
        })

    physics_sha256 = initialization["validated_physics_contract"][
        "physics_contract_sha256"
    ]
    report = {
        "schema": "taco_pour_dual_backend_contract_audit_v1",
        "status": "dual_backend_transfer_gate_passed",
        "scope": {
            "training_executed": False,
            "ppo_budget_changed": False,
            "full_horizon_executed": False,
            "cpu_control_intervals_executed": 1,
            "gpu_control_intervals_executed": 0,
            "formal_training_gate_opened": False,
        },
        "backend_contract": contract_artifact,
        "physics_contract_sha256": physics_sha256,
        "objective_profile_sha256": objective.profile_sha256,
        "observation_profile_sha256": observation.profile_sha256,
        "backend_runtime_records": {
            "training": _backend_runtime_record(
                gpu_env,
                role="policy_optimization_only",
                policy_inference_device="cuda:0",
                physics_contract_sha256=physics_sha256,
            ),
            "validation": _backend_runtime_record(
                cpu_env,
                role="replay_policy_acceptance_and_commit",
                policy_inference_device="cpu",
                physics_contract_sha256=physics_sha256,
            ),
        },
        "cpu_to_gpu_transfers": transfers,
        "verified_transfer_count": gpu.verified_restore_count,
        "decision": {
            "gpu_may_decide_acceptance": False,
            "gpu_may_provide_committed_state": False,
            "cpu_40_of_40_required_for_commit": True,
            "commit_state_source": "cpu_validation_endpoint_20",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "verified_transfer_count": report["verified_transfer_count"],
        "training_runtime_contract_sha256": report["backend_runtime_records"]["training"]["runtime_contract_sha256"],
        "validation_runtime_contract_sha256": report["backend_runtime_records"]["validation"]["runtime_contract_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
