#!/usr/bin/env python3
"""Replay one saved optimized trajectory from its accepted reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--tracking-variant", choices=("tool_only", "tool_and_target"), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source_report = json.loads(args.source_report.read_text())
    raw = args.trajectory.read_bytes()
    trajectory_hash = hashlib.sha256(raw).hexdigest()
    if source_report.get("optimized_trajectory", {}).get("sha256") != trajectory_hash:
        raise ValueError("trajectory hash does not match its source report")
    with np.load(args.trajectory, allow_pickle=False) as saved:
        qpos_saved = np.asarray(saved["qpos"], dtype=np.float32)
        qvel_saved = np.asarray(saved["qvel"], dtype=np.float32)
        ctrl_saved = np.asarray(saved["ctrl"], dtype=np.float32)
        actions = np.asarray(saved["raw_residual_action"], dtype=np.float32)
        modes = np.asarray(saved["mode"])
    if qpos_saved.shape != (198, 50) or qvel_saved.shape != (198, 48):
        raise ValueError("saved trajectory must contain all 198 Pour endpoints")
    if ctrl_saved.shape != (198, 36) or actions.shape != (197, 36) or modes.shape != (197,):
        raise ValueError("saved trajectory control/action dimensions are invalid")

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant=args.tracking_variant, require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    initial, provenance = load_accepted_initialization(args.initialization_report, args.config)
    if not (
        np.array_equal(qpos_saved[0], np.asarray(initial["qpos"], dtype=np.float32))
        and np.array_equal(qvel_saved[0], np.asarray(initial["qvel"], dtype=np.float32))
        and np.array_equal(ctrl_saved[0], np.asarray(initial["ctrl"], dtype=np.float32))
    ):
        raise ValueError("saved endpoint 0 is not the accepted initialization")

    from run_mjwp_ppo import _load_ego_config, _load_reference, MJWPVectorEnv, MJWPVectorEnvConfig, torch
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)
    expected_ctrl = reference[2].cpu().numpy().copy()
    expected_ctrl[0, :36] = np.asarray(initial["ctrl"], dtype=np.float32)
    expected_ctrl[1:, :36] += np.clip(actions, -0.05, 0.05)
    if not np.allclose(ctrl_saved, expected_ctrl[:, :36], atol=2e-7, rtol=0):
        raise ValueError("saved ctrl is inconsistent with reference plus clipped residual")

    env = MJWPVectorEnv(
        config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=True,
            max_episode_length=197,
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
    backend.begin_trial("saved_trajectory", 0, 197)
    valid_steps = 0
    error = None
    try:
        for index, action in enumerate(actions):
            if not backend.step(action[None], index):
                break
            valid_steps += 1
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        feasible = valid_steps == 197
        backend.end_trial(feasible, valid_steps, error=error)
        trace = backend.validation_traces[0]
        replayed_qpos = np.asarray([initial["qpos"]] + [row["endpoint_qpos"] for row in trace["steps"]])
        replayed_qvel = np.asarray([initial["qvel"]] + [row["endpoint_qvel"] for row in trace["steps"]])
        count = len(replayed_qpos)
        report = {
            "schema": "taco_pour_saved_trajectory_validation_v1",
            "status": "full_horizon_replay_validated" if feasible else "saved_trajectory_replay_failed",
            "tracking_variant": args.tracking_variant,
            "trajectory": {"path": str(args.trajectory.resolve()), "sha256": trajectory_hash},
            "source_report": str(args.source_report.resolve()),
            "initialization": provenance,
            "objective": objective.as_report(),
            "observation": observation.as_report(),
            "mode_counts": {name: int(np.sum(modes == name)) for name in np.unique(modes)},
            "trace": trace,
            "saved_state_replay_difference": {
                "compared_endpoints": count,
                "qpos_max_abs": float(np.max(np.abs(replayed_qpos - qpos_saved[:count]))),
                "qvel_max_abs": float(np.max(np.abs(replayed_qvel - qvel_saved[:count]))),
                "exact_state_reproduction_expected": False,
            },
            "simulation_control_intervals": env.simulation_control_intervals,
            "simulation_physics_steps": env.simulation_physics_steps,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "status": report["status"],
            "validated_steps": valid_steps,
            "first_failure": trace["first_failure"],
            "qpos_max_abs_vs_saved": report["saved_state_replay_difference"]["qpos_max_abs"],
        }, indent=2))


if __name__ == "__main__":
    main()
