#!/usr/bin/env python3
"""Locate the first nondeterministic MJWP physics substep after endpoint 20."""

from __future__ import annotations

import argparse
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


def digest(value) -> str:
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, _load_ego_config, _load_reference, torch,
    )

    boundary_artifact = args.boundary.read_bytes()
    boundary_raw = gzip.decompress(boundary_artifact)
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    _, initialization = load_accepted_initialization(args.initialization_report, args.config)
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
    ctrl = env._reference_ctrls(np.array([20]), offset=1)

    repetitions = []
    snapshots = []
    for repeat in range(3):
        env.set_env_state(boundary)
        rows = []
        saved = []
        for substep in range(1, int(config.ctrl_steps) + 1):
            env._mjwp.step_env(config, env.env, ctrl)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            state = env.get_env_state()
            saved.append(state)
            nacon = int(state["nacon"].numpy()[0])
            worldid = state["contact.worldid"].numpy()
            geom = state["contact.geom"].numpy()
            active = np.flatnonzero(worldid == 0)[:nacon]
            pairs = sorted(tuple(map(int, geom[index])) for index in active)
            rows.append({
                "substep": substep,
                "qpos_sha256": digest(state["qpos"]),
                "qvel_sha256": digest(state["qvel"]),
                "nacon": nacon,
                "nefc": int(state["nefc"].numpy()[0]),
                "contact_geom_pairs": pairs,
            })
        repetitions.append({"repeat": repeat + 1, "substeps": rows})
        snapshots.append(saved)

    comparison = []
    for substep in range(int(config.ctrl_steps)):
        differing = [
            key for key in boundary["warp_state_keys"]
            if len({digest(snapshots[repeat][substep][key]) for repeat in range(3)}) != 1
        ]
        qpos = np.stack([snapshots[repeat][substep]["qpos"].numpy() for repeat in range(3)])
        qvel = np.stack([snapshots[repeat][substep]["qvel"].numpy() for repeat in range(3)])
        pair_sets = [set(repetitions[repeat]["substeps"][substep]["contact_geom_pairs"]) for repeat in range(3)]
        comparison.append({
            "substep": substep + 1,
            "differing_warp_field_count": len(differing),
            "first_differing_fields": differing[:30],
            "maximum_qpos_spread": float(np.ptp(qpos, axis=0).max()),
            "maximum_qvel_spread": float(np.ptp(qvel, axis=0).max()),
            "contact_pair_sets_equal_ignoring_order": pair_sets[0] == pair_sets[1] == pair_sets[2],
        })

    first = next((row for row in comparison if row["differing_warp_field_count"]), None)
    report = {
        "schema": "taco_pour_mjwp_first_interval_repeatability_v1",
        "status": "bitwise_repeatable" if first is None else "nondeterministic_physics_step",
        "boundary": {
            "artifact_path": str(args.boundary.resolve()),
            "artifact_sha256": hashlib.sha256(boundary_artifact).hexdigest(),
            "uncompressed_pt_sha256": hashlib.sha256(boundary_raw).hexdigest(),
            "snapshot_schema": boundary["snapshot_schema"],
            "reference_endpoint": int(boundary["time_indices"][0]),
        },
        "action": "zero residual over reference ctrl[21]",
        "same_environment": True,
        "device": args.device,
        "restored_before_each_repeat": True,
        "physics_substeps_per_control_interval": int(config.ctrl_steps),
        "first_divergent_substep": None if first is None else first["substep"],
        "comparison": comparison,
        "repetitions": repetitions,
        "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "first_divergent_substep": report["first_divergent_substep"],
        "comparison": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
