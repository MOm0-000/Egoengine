#!/usr/bin/env python3
"""Prove deterministic CPU Replay from the accepted reward-aligned endpoint 0."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _trace_signature(trace: dict) -> str:
    digest = hashlib.sha256()
    for row in trace["steps"]:
        for name, dtype in (
            ("endpoint_qpos", np.float32),
            ("endpoint_qvel", np.float32),
            ("commanded_ctrl", np.float32),
            ("objective_score", np.float64),
        ):
            digest.update(np.asarray(row[name], dtype=dtype).tobytes())
        digest.update(
            np.asarray(
                [
                    row["command_reference_endpoint"],
                    row["reward_reference_endpoint"],
                    row["next_observation_goal_reference_endpoint"],
                ],
                dtype=np.int32,
            ).tobytes()
        )
    digest.update(
        json.dumps(trace["first_failure"], sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def main() -> None:
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
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
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
        "--action-profile",
        type=Path,
        default=ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/taco_pour_cpu_backend_repeatability_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _load_ego_config,
        _load_reference,
        torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action

    objective = load_runtime_objective(
        args.protocol,
        args.objective_profile,
        tracking_variant="tool_only",
        require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    initial, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )

    def make_env():
        config = _load_ego_config(str(args.config), "cpu")
        reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
        env = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=False,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
        )
        verify_runtime_model(
            env.env.model_cpu, initialization["validated_physics_contract"]
        )
        tensors = [
            torch.as_tensor(initial[name][None], dtype=torch.float32)
            for name in ("qpos", "qvel", "ctrl")
        ]
        env._write_state(*tensors, np.array([True]))
        env._last_ctrl = tensors[2].clone()
        env._check_capacity()
        return env

    source_env = make_env()
    source_backend = MJWPChunkBackend(source_env)
    boundary = source_backend.snapshot()
    if boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned":
        raise RuntimeError("CPU evidence requires the reward-aligned snapshot schema")

    def run(env, repeat: int) -> dict:
        backend = MJWPChunkBackend(env)
        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        backend.begin_trial(f"cpu_replay_repeat_{repeat}", 0, 40)
        valid = 0
        error = None
        try:
            for index in range(40):
                if not backend.step(replay_action(backend, index), index):
                    break
                valid += 1
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            backend.end_trial(valid == 40, valid, error=error)
        trace = backend.validation_traces[-1]
        return {
            "repeat": repeat,
            "validated_steps": trace["validated_steps"],
            "first_failure": trace["first_failure"],
            "trajectory_signature_sha256": _trace_signature(trace),
        }

    same = [run(source_env, repeat) for repeat in range(1, 4)]
    fresh = [run(make_env(), repeat) for repeat in range(1, 4)]
    same_ok = len({row["trajectory_signature_sha256"] for row in same}) == 1
    fresh_ok = len({row["trajectory_signature_sha256"] for row in fresh}) == 1
    cross_ok = same[0]["trajectory_signature_sha256"] == fresh[0]["trajectory_signature_sha256"]
    passed = (
        same_ok
        and fresh_ok
        and cross_ok
        and all(row["validated_steps"] == 40 for row in same + fresh)
    )

    report = {
        "schema": "taco_pour_cpu_backend_repeatability_v1",
        "status": "repeatability_gate_passed" if passed else "repeatability_gate_failed",
        "paper_faithful": False,
        "scope": {
            "training_executed": False,
            "optimizer_updates": 0,
            "device": "cpu",
            "start_endpoint": 0,
            "actions": "Replay zero residual",
            "reward_alignment": "outcome t+1 versus reference t+1",
        },
        "snapshot_schema": boundary["snapshot_schema"],
        "same_environment_mode_repeatable": {"replay": same_ok},
        "fresh_environment_mode_repeatable": {"replay": fresh_ok},
        "same_environment_repetitions": same,
        "fresh_environment_repetitions": fresh,
        "same_and_fresh_signatures_equal": cross_ok,
        "objective": objective.as_report(),
        "observation": observation.as_report(),
        "residual_action": residual_report,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path.read_bytes())}
            for name, path in (
                ("protocol", args.protocol),
                ("objective_profile", args.objective_profile),
                ("observation_profile", args.observation_profile),
                ("action_profile", args.action_profile),
                ("config", args.config),
                ("initialization_report", args.initialization_report),
            )
        },
    }
    args.output_dir.mkdir(parents=True)
    encoded = json.dumps(report, indent=2).encode() + b"\n"
    compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
    (args.output_dir / "report.json.gz").write_bytes(compressed)
    print(json.dumps({
        "status": report["status"],
        "same_environment_mode_repeatable": report["same_environment_mode_repeatable"],
        "fresh_environment_mode_repeatable": report["fresh_environment_mode_repeatable"],
        "same_and_fresh_signatures_equal": cross_ok,
    }, indent=2))


if __name__ == "__main__":
    main()
