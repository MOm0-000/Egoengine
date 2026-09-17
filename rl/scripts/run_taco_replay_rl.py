"""Guarded full-source Replay→PPO runner; no implicit reset or task promotion."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_accepted_initialization(report_path, config_path):
    """Read the measured candidate, never accept an old qpos-only diagnostic."""
    report = json.loads(report_path.read_text())
    if not report.get("accepted_for_replay_rl", False):
        raise ValueError("initialization has not passed model and passive-object release checks")
    config = yaml.safe_load(config_path.read_text())
    for key, config_key in (("scene", "model_path"), ("reference", "data_path")):
        path = Path(config[config_key]).resolve(strict=True)
        if str(path) != report[key]["path"] or hashlib.sha256(path.read_bytes()).hexdigest() != report[key]["sha256"]:
            raise ValueError(f"accepted initialization belongs to another {key}")
    path = Path(report["initial_state"]["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["initial_state"]["sha256"]:
        raise ValueError("initial state changed after validation")
    with np.load(path, allow_pickle=False) as source:
        initial = dict(source)
    if int(initial["reference_index"]) != 0:
        raise ValueError("this runner preserves the full source trajectory from row 0")
    for name, shape in (("qpos", (50,)), ("qvel", (48,)), ("ctrl", (36,))):
        if initial[name].shape != shape or not np.isfinite(initial[name]).all():
            raise ValueError(f"initial {name} must be finite with shape {shape}")
    with np.load(config["data_path"], allow_pickle=False) as reference:
        if not np.array_equal(initial["qpos"][36:], reference["qpos"][0, 36:]):
            raise ValueError("initialization moved the source object's initial pose")
    return initial, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/taco_pour_bimanual_ppo.yaml")
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--tracking-variant", choices=("tool_only", "tool_and_target"), default="tool_and_target")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.max_chunks < 1:
        raise ValueError("positive epoch/chunk budgets required")
    initial, provenance = load_accepted_initialization(args.initialization_report, args.config)

    # Refuse unaccepted initialization before allocating a GPU or loading PPO.
    from run_mjwp_ppo import _load_ego_config, _load_reference, MJWPVectorEnv, MJWPVectorEnvConfig, torch
    from egoengine_repro.action.replay_rl import solve_chunk
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action, train_chunk_ppo

    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)
    env = MJWPVectorEnv(config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(reference_start_index=0, asymmetric_critic=True,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,) if args.tracking_variant == "tool_only" else None))
    tensors = [torch.as_tensor(initial[k][None], device="cuda:0", dtype=torch.float32)
               for k in ("qpos", "qvel", "ctrl")]
    env._write_state(*tensors, np.array([True]))
    env._last_ctrl = tensors[2].clone()
    env._check_capacity()
    backend = MJWPChunkBackend(env)
    args.output.mkdir(parents=True)
    result = dict(status="running", initialization=provenance,
                  tracking_variant=args.tracking_variant,
                  local_settings=dict(worlds=1, ppo_epochs=args.epochs, ppo_horizon=40,
                      deterministic_mean_validation=True, fresh_policy_per_failed_chunk=True,
                      tracking_boundary=env.tracking_boundary),
                  source_frames=len(reference[0]), control_intervals=len(reference[0]) - 1,
                  chunks=[], task_success=False)
    start = 0
    try:
        for _ in range(args.max_chunks):
            chunk = solve_chunk(backend, replay_action,
                lambda current, first, end: train_chunk_ppo(current, first, end,
                    args.output / f"ppo_chunk_{first}", epochs=args.epochs),
                start=start, total_steps=len(reference[0]) - 1)
            result["chunks"].append(asdict(chunk))
            if chunk.mode is None:
                result["status"] = "both_modes_failed_within_local_budget"
                break
            start = chunk.committed_end
            torch.save(backend.snapshot(), args.output / "committed_boundary.pt")
            if start == len(reference[0]) - 1:
                result.update(status="full_horizon_tracking_feasible", task_success=True)
                break
        else:
            result["status"] = "chunk_budget_reached_not_full_task_success"
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["simulation_control_intervals"] = env.simulation_control_intervals
        result["simulation_physics_steps"] = env.simulation_physics_steps
        result["committed_reference_index"] = int(env.time_indices[0])
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
