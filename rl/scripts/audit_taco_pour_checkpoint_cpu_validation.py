#!/usr/bin/env python3
"""Validate a trained recurrent PPO checkpoint closed-loop on deterministic CPU MJWP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def trace_signature(trace: dict) -> str:
    chunks = []
    for row in trace["steps"]:
        chunks.extend((
            np.asarray(row["endpoint_qpos"], np.float32).tobytes(),
            np.asarray(row["endpoint_qvel"], np.float32).tobytes(),
            np.asarray(row["raw_residual_action"], np.float32).tobytes(),
            np.asarray(row["objective_score"], np.float32).tobytes(),
        ))
    failure = json.dumps(trace["first_failure"], sort_keys=True, separators=(",", ":")).encode()
    return sha256(b"".join(chunks) + failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, PpoAgent, _build_network_config,
        _build_ppo_config, _load_ego_config, _load_reference, torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    boundary_artifact = args.boundary.read_bytes()
    boundary_raw = gzip.decompress(boundary_artifact)
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    checkpoint_raw = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False)
    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    _, initialization = load_accepted_initialization(args.initialization_report, args.config)
    config = _load_ego_config(str(args.config), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=False,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,),
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
        ),
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    if boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v2":
        raise ValueError("checkpoint validation requires a complete v2 boundary")
    if int(boundary["time_indices"][0]) != 20:
        raise ValueError("checkpoint validation requires endpoint 20")

    ppo_config = _build_ppo_config(
        num_envs=1,
        horizon_length=40,
        seq_length=4,
        max_epochs=8,
        learning_rate=1e-4,
        device="cpu",
        asymmetric_critic=None,
    )
    with tempfile.TemporaryDirectory(prefix="egoengine_ppo_cpu_validation_") as temp_dir:
        agent = PpoAgent(
            experiment_dir=Path(temp_dir),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
        )
        agent.model.load_state_dict(checkpoint["model"])
        agent.set_eval()
        backend = MJWPChunkBackend(env)
        repetitions = []
        for repeat in range(1, 4):
            trials = {}
            for mode in ("replay", "rl"):
                backend.restore(boundary)
                agent.rnn_states = [
                    state.to(agent.device).zero_()
                    for state in agent.model.get_default_rnn_state()
                ]
                backend.begin_trial(f"{mode}_cpu_repeat_{repeat}", 20, 60)
                valid_steps = 0
                for reference_step in range(20, 60):
                    if mode == "replay":
                        action = np.zeros((1, 36), dtype=np.float32)
                    else:
                        values = agent.get_action_values(agent.obs_to_tensors(backend.observation()))
                        agent.rnn_states = values["rnn_states"]
                        action = agent.preprocess_actions(values["mus"])
                    if not backend.step(action, reference_step):
                        break
                    valid_steps += 1
                backend.end_trial(valid_steps == 40, valid_steps)
                trace = backend.validation_traces[-1]
                trials[mode] = {
                    "validated_steps": trace["validated_steps"],
                    "first_failure": trace["first_failure"],
                    "trajectory_signature_sha256": trace_signature(trace),
                    "trace": trace,
                }
            repetitions.append({"repeat": repeat, "trials": trials})
        agent.writer.close()

    repeatable = {
        mode: len({row["trials"][mode]["trajectory_signature_sha256"] for row in repetitions}) == 1
        for mode in ("replay", "rl")
    }
    passed = all(repeatable.values()) and all(
        row["trials"]["rl"]["validated_steps"] == 40 for row in repetitions
    )
    report = {
        "schema": "taco_pour_checkpoint_cpu_closed_loop_validation_v2",
        "status": "checkpoint_validation_passed" if passed else "checkpoint_validation_failed",
        "scope": {
            "device": "cpu",
            "closed_loop_recurrent_policy": True,
            "deterministic_mean_action": True,
            "repetitions": 3,
            "training_executed": False,
            "full_horizon_executed": False,
        },
        "boundary": {
            "artifact_path": str(args.boundary.resolve()),
            "artifact_sha256": sha256(boundary_artifact),
            "uncompressed_pt_sha256": sha256(boundary_raw),
            "reference_endpoint": 20,
        },
        "checkpoint": {
            "artifact_path": str(args.checkpoint.resolve()),
            "artifact_sha256": sha256(checkpoint_raw),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_frame": int(checkpoint["frame"]),
            "checkpoint_env_state_not_restored": True,
        },
        "objective": objective.as_report(),
        "observation": observation.as_report(),
        "initialization": initialization,
        "bitwise_trajectory_repeatable": repeatable,
        "repetitions": repetitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "bitwise_trajectory_repeatable": repeatable,
        "repetitions": [
            {
                "repeat": row["repeat"],
                "replay_validated_steps": row["trials"]["replay"]["validated_steps"],
                "replay_failure": row["trials"]["replay"]["first_failure"],
                "rl_validated_steps": row["trials"]["rl"]["validated_steps"],
                "rl_failure": row["trials"]["rl"]["first_failure"],
            }
            for row in repetitions
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
