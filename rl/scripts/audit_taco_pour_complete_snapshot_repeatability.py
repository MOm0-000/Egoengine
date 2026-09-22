#!/usr/bin/env python3
"""Rebuild endpoint 20 with the complete MJWP state and replay frozen actions."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def state_difference(expected: dict, actual: dict) -> list[str]:
    mismatches = []
    if expected.keys() != actual.keys():
        return ["snapshot keys differ"]
    for key, value in expected.items():
        other = actual[key]
        if hasattr(value, "numpy"):
            left = value.numpy()
            right = other.numpy()
            equal = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        elif isinstance(value, np.ndarray):
            equal = value.dtype == other.dtype and value.shape == other.shape and value.tobytes() == other.tobytes()
        else:
            equal = value == other
        if not equal:
            mismatches.append(key)
    return mismatches


def trace_signature(trace: dict) -> str:
    chunks = []
    for row in trace["steps"]:
        chunks.extend((
            np.asarray(row["endpoint_qpos"], np.float32).tobytes(),
            np.asarray(row["endpoint_qvel"], np.float32).tobytes(),
            np.asarray(row["objective_score"], np.float32).tobytes(),
        ))
    failure = json.dumps(trace["first_failure"], sort_keys=True, separators=(",", ":")).encode()
    return sha256(b"".join(chunks) + failure)


def run_trials(env, boundary: dict, actions: dict, repeat: int, backend_type) -> dict:
    env.set_env_state(boundary)
    restored = env.get_env_state()
    mismatches = state_difference(boundary, restored)
    backend = backend_type(env)
    trials = {}
    for mode in ("replay", "rl"):
        backend.restore(boundary)
        mode_actions = actions[f"{mode}_actions"]
        backend.begin_trial(f"{mode}_repeat_{repeat}", 20, 20 + len(mode_actions))
        valid_steps = 0
        error = None
        try:
            for offset, action in enumerate(mode_actions):
                if not backend.step(action[None], 20 + offset):
                    break
                valid_steps += 1
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            backend.end_trial(valid_steps == len(mode_actions), valid_steps, error=error)
        trace = backend.validation_traces[-1]
        trials[mode] = {
            "action_count": len(mode_actions),
            "validated_steps": trace["validated_steps"],
            "first_failure": trace["first_failure"],
            "trajectory_signature_sha256": trace_signature(trace),
            "trace": trace,
        }
    return {
        "repeat": repeat,
        "restored_state_bitwise_equal": not mismatches,
        "restored_state_mismatches": mismatches,
        "trials": trials,
    }


def repeatability(repetitions: list[dict]) -> dict[str, bool]:
    return {
        mode: len({row["trials"][mode]["trajectory_signature_sha256"] for row in repetitions}) == 1
        for mode in ("replay", "rl")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    source_manifest_raw = args.source_manifest.read_bytes()
    source_manifest = json.loads(source_manifest_raw)
    if source_manifest.get("schema") != "taco_pour_normalized_ellipse_repeatability_gate_v1":
        raise ValueError("unexpected frozen-action manifest")
    action_record = source_manifest["actions"]
    actions_raw = Path(action_record["artifact_path"]).read_bytes()
    if sha256(actions_raw) != action_record["artifact_sha256"]:
        raise ValueError("frozen action hash mismatch")
    with np.load(io.BytesIO(actions_raw), allow_pickle=False) as saved:
        actions = {name: np.asarray(saved[name]) for name in saved.files}

    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, _load_ego_config, _load_reference, torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    initial, initialization = load_accepted_initialization(args.initialization_report, args.config)
    contracts = source_manifest["contracts"]
    if objective.profile_sha256 != contracts["objective_profile_sha256"]:
        raise ValueError("objective profile differs from frozen actions")
    if observation.profile_sha256 != contracts["observation_profile_sha256"]:
        raise ValueError("observation profile differs from frozen actions")
    if initialization["physics_contract"]["physics_contract_sha256"] != contracts["physics_contract_sha256"]:
        raise ValueError("physics contract differs from frozen actions")

    def make_env():
        config = _load_ego_config(str(args.config), args.device)
        reference = _load_reference(config.data_path, args.device, expected_frequency=30)
        env = MJWPVectorEnv(
            config, reference, num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=True,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
            ),
        )
        verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
        return config, reference, env

    config, reference, source_env = make_env()
    initial_tensors = [
        torch.as_tensor(initial[name][None], device=args.device, dtype=torch.float32)
        for name in ("qpos", "qvel", "ctrl")
    ]
    source_env._write_state(*initial_tensors, np.array([True]))
    source_env._last_ctrl = initial_tensors[2].clone()
    source_env._check_capacity()
    source_backend = MJWPChunkBackend(source_env)
    for step in range(20):
        zero = np.zeros((1, 36), dtype=np.float32)
        if not source_backend.step(zero, step):
            raise RuntimeError(f"fresh Replay failed before endpoint 20 at transition {step}")
    boundary = source_backend.snapshot()
    if int(boundary["time_indices"][0]) != 20:
        raise RuntimeError("rebuilt boundary is not endpoint 20")

    args.output_dir.mkdir(parents=True)
    boundary_buffer = io.BytesIO()
    torch.save(boundary, boundary_buffer)
    boundary_raw = boundary_buffer.getvalue()
    boundary_gzip = gzip.compress(boundary_raw, compresslevel=9, mtime=0)
    boundary_path = args.output_dir / "endpoint20_complete_boundary.pt.gz"
    boundary_path.write_bytes(boundary_gzip)

    same_environment_repetitions = [
        run_trials(source_env, boundary, actions, repeat, MJWPChunkBackend)
        for repeat in range(1, 4)
    ]

    fresh_environment_repetitions = []
    for repeat in range(3):
        repeat_config, repeat_reference, env = make_env()
        fresh_environment_repetitions.append(
            run_trials(env, boundary, actions, repeat + 1, MJWPChunkBackend)
        )
        del env, repeat_reference, repeat_config
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    same_environment_repeatable = repeatability(same_environment_repetitions)
    fresh_environment_repeatable = repeatability(fresh_environment_repetitions)
    repetitions = same_environment_repetitions + fresh_environment_repetitions
    passed = (
        all(row["restored_state_bitwise_equal"] for row in repetitions)
        and all(same_environment_repeatable.values())
        and all(fresh_environment_repeatable.values())
    )
    report = {
        "schema": "taco_pour_complete_snapshot_repeatability_v4",
        "status": "repeatability_gate_passed" if passed else "repeatability_gate_failed",
        "scope": {
            "training_executed": False,
            "full_horizon_executed": False,
            "device": args.device,
            "boundary_source": "first fresh accepted-reset Replay rollout; no candidate selection",
            "frozen_actions": "actions from the prior 8-epoch run; performance is diagnostic because the boundary was rebuilt",
            "acceptance": "complete restored state must be bitwise equal and each action sequence must have one exact trajectory signature across three same-environment and three fresh-environment repeats",
        },
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": sha256(source_manifest_raw),
        },
        "boundary": {
            "artifact_path": str(boundary_path.resolve()),
            "artifact_sha256": sha256(boundary_gzip),
            "uncompressed_pt_sha256": sha256(boundary_raw),
            "snapshot_schema": boundary["snapshot_schema"],
            "mujoco_warp_version": boundary["mujoco_warp_version"],
            "warp_state_field_count": len(boundary["warp_state_keys"]),
            "reference_endpoint": 20,
        },
        "objective": objective.as_report(),
        "observation": observation.as_report(),
        "initialization": initialization,
        "same_environment_mode_repeatable": same_environment_repeatable,
        "fresh_environment_mode_repeatable": fresh_environment_repeatable,
        "same_environment_repetitions": same_environment_repetitions,
        "fresh_environment_repetitions": fresh_environment_repetitions,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "warp_state_field_count": report["boundary"]["warp_state_field_count"],
        "same_environment_mode_repeatable": same_environment_repeatable,
        "fresh_environment_mode_repeatable": fresh_environment_repeatable,
        "same_environment_repetitions": [
            {
                "repeat": row["repeat"],
                "restored_state_bitwise_equal": row["restored_state_bitwise_equal"],
                "replay_validated_steps": row["trials"]["replay"]["validated_steps"],
                "replay_failure": row["trials"]["replay"]["first_failure"],
                "rl_validated_steps": row["trials"]["rl"]["validated_steps"],
                "rl_failure": row["trials"]["rl"]["first_failure"],
            }
            for row in same_environment_repetitions
        ],
        "fresh_environment_repetitions": [
            {
                "repeat": row["repeat"],
                "restored_state_bitwise_equal": row["restored_state_bitwise_equal"],
                "replay_validated_steps": row["trials"]["replay"]["validated_steps"],
                "replay_failure": row["trials"]["replay"]["first_failure"],
                "rl_validated_steps": row["trials"]["rl"]["validated_steps"],
                "rl_failure": row["trials"]["rl"]["first_failure"],
            }
            for row in fresh_environment_repetitions
        ],
    }, indent=2))

    del source_backend, source_env, reference, config


if __name__ == "__main__":
    main()
