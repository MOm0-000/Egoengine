#!/usr/bin/env python3
"""Run the accepted reset through the first Replay lookahead, without PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--tracking-variant", choices=("tool_only", "tool_and_target"), default="tool_only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant=args.tracking_variant, require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    initial, provenance = load_accepted_initialization(args.initialization_report, args.config)

    from run_mjwp_ppo import _load_ego_config, _load_reference, MJWPVectorEnv, MJWPVectorEnvConfig, torch
    from run_taco_replay_rl import verify_runtime_model
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action

    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=True,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,) if args.tracking_variant == "tool_only" else None,
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
        ),
    )
    verify_runtime_model(env.env.model_cpu, provenance["validated_physics_contract"])
    tensors = [
        torch.as_tensor(initial[name][None], device="cuda:0", dtype=torch.float32)
        for name in ("qpos", "qvel", "ctrl")
    ]
    env._write_state(*tensors, np.array([True]))
    env._last_ctrl = tensors[2].clone()
    env._check_capacity()
    backend = MJWPChunkBackend(env)
    boundary = backend.snapshot()
    backend.begin_trial("replay", 0, 40)
    valid_steps = 0
    error = None
    try:
        for index in range(40):
            if not backend.step(replay_action(backend, index), index):
                break
            valid_steps += 1
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        backend.end_trial(valid_steps == 40, valid_steps, error=error)
        backend.restore(boundary)
        trace = backend.validation_traces[0]
        report = {
            "schema": "taco_pour_first40_replay_audit_v1",
            "status": "replay_feasible" if valid_steps == 40 else "replay_failed",
            "ppo_executed": False,
            "tracking_variant": args.tracking_variant,
            "objective": objective.as_report(),
            "observation": observation.as_report(),
            "initialization": provenance,
            "trace": trace,
            "simulation_control_intervals": env.simulation_control_intervals,
            "simulation_physics_steps": env.simulation_physics_steps,
            "boundary_restored_after_audit": int(env.time_indices[0]) == 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "status": report["status"],
            "validated_steps": valid_steps,
            "first_failure": trace["first_failure"],
            "simulation_physics_steps": report["simulation_physics_steps"],
        }, indent=2))


if __name__ == "__main__":
    main()
