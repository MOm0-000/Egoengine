#!/usr/bin/env python3
"""Replay frozen Replay/PPO window actions from the exact endpoint-20 state."""

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


def checked_bytes(record: dict, *, compressed: bool = False) -> bytes:
    raw = Path(record["artifact_path"]).read_bytes()
    if sha256(raw) != record["artifact_sha256"]:
        raise ValueError(f"artifact hash mismatch: {record['artifact_path']}")
    if compressed:
        unpacked = gzip.decompress(raw)
        expected = record.get("uncompressed_pt_sha256") or record.get("uncompressed_json_sha256")
        if expected is not None and sha256(unpacked) != expected:
            raise ValueError(f"uncompressed hash mismatch: {record['artifact_path']}")
        return unpacked
    return raw


def curve_comparison(trace: dict, endpoints: np.ndarray, scores: np.ndarray) -> dict:
    source = {int(endpoint): float(score) for endpoint, score in zip(endpoints, scores, strict=True)}
    actual = {int(row["endpoint"]): float(row["objective_score"][0]) for row in trace["steps"]}
    common = sorted(source.keys() & actual.keys())
    differences = [abs(actual[endpoint] - source[endpoint]) for endpoint in common]
    return {
        "common_endpoints": len(common),
        "first_common_endpoint": common[0] if common else None,
        "last_common_endpoint": common[-1] if common else None,
        "mean_absolute_score_difference": float(np.mean(differences)) if differences else None,
        "maximum_absolute_score_difference": max(differences) if differences else None,
        "selected_endpoints": {
            str(endpoint): {
                "source_score": source.get(endpoint),
                "replayed_score": actual.get(endpoint),
            }
            for endpoint in (48, 56)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != "taco_pour_normalized_ellipse_repeatability_gate_v1":
        raise ValueError("unsupported gate manifest")
    plan = manifest["execution_plan"]
    if plan != {
        "fresh_environment_per_repetition": True,
        "repetitions": 3,
        "trial_order_per_repetition": ["replay", "rl"],
        "seed_sweep": False,
        "training": False,
        "full_horizon": False,
    }:
        raise ValueError("execution plan differs from the frozen three-repeat gate")
    checked_bytes(manifest["source_report"], compressed=True)
    checked_bytes(manifest["variant_trace_report"], compressed=True)
    boundary_raw = checked_bytes(manifest["boundary"], compressed=True)
    actions_raw = checked_bytes(manifest["actions"])

    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, _load_ego_config, _load_reference, torch,
    )
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    if int(np.asarray(boundary["time_indices"])[0]) != 20:
        raise ValueError("boundary artifact is not endpoint 20")
    with np.load(io.BytesIO(actions_raw), allow_pickle=False) as saved:
        action_data = {name: np.asarray(saved[name]) for name in saved.files}

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    _, initialization = load_accepted_initialization(args.initialization_report, args.config)
    contracts = manifest["contracts"]
    if objective.profile_sha256 != contracts["objective_profile_sha256"]:
        raise ValueError("objective profile differs from the frozen action source")
    if observation.profile_sha256 != contracts["observation_profile_sha256"]:
        raise ValueError("observation profile differs from the frozen action source")
    if initialization["physics_contract"]["physics_contract_sha256"] != contracts["physics_contract_sha256"]:
        raise ValueError("physics contract differs from the frozen action source")

    repetitions = []
    for repeat in range(3):
        config = _load_ego_config(str(args.config), "cuda:0")
        reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)
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
        env.set_env_state(boundary)
        env._check_capacity()
        backend = MJWPChunkBackend(env)
        trials = {}
        for mode in ("replay", "rl"):
            backend.restore(boundary)
            actions = action_data[f"{mode}_actions"]
            endpoints = action_data[f"{mode}_endpoints"]
            scores = action_data[f"{mode}_scores"]
            backend.begin_trial(f"{mode}_repeat_{repeat + 1}", 20, 20 + len(actions))
            valid_steps = 0
            error = None
            try:
                for offset, action in enumerate(actions):
                    if not backend.step(action[None], 20 + offset):
                        break
                    valid_steps += 1
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
                raise
            finally:
                backend.end_trial(valid_steps == len(actions), valid_steps, error=error)
            trial = backend.validation_traces[-1]
            trials[mode] = {
                "action_count": len(actions),
                "validated_steps": trial["validated_steps"],
                "first_failure": trial["first_failure"],
                "curve_comparison": curve_comparison(trial, endpoints, scores),
                "trace": trial,
            }
        replay_failure = trials["replay"]["first_failure"]
        rl_failure = trials["rl"]["first_failure"]
        paired_pass = (
            trials["replay"]["validated_steps"] == 27
            and replay_failure is not None and replay_failure["endpoint"] == 48
            and trials["rl"]["validated_steps"] == 35
            and rl_failure is not None and rl_failure["endpoint"] == 56
            and trials["rl"]["validated_steps"] - trials["replay"]["validated_steps"] == 8
        )
        repetitions.append({
            "repeat": repeat + 1,
            "paired_acceptance_passed": paired_pass,
            "trials": trials,
        })
        del backend, env, reference, config
        gc.collect()
        torch.cuda.empty_cache()

    passed = all(row["paired_acceptance_passed"] for row in repetitions)
    report = {
        "schema": "taco_pour_normalized_ellipse_action_repeatability_v1",
        "status": "repeatability_gate_passed" if passed else "repeatability_gate_failed",
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256(manifest_raw),
        },
        "objective": objective.as_report(),
        "observation": observation.as_report(),
        "initialization": initialization,
        "offline_variant_reduction": manifest["offline_variant_reduction"],
        "acceptance": manifest["acceptance"],
        "repetitions": repetitions,
        "training_executed": False,
        "full_horizon_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "repetitions": [
            {
                "repeat": row["repeat"],
                "passed": row["paired_acceptance_passed"],
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
