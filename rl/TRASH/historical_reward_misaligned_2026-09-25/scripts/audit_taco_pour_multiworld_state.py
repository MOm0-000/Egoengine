#!/usr/bin/env python3
"""Gate four independent GPU training worlds before multi-world PPO."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairwise_max(values: list[np.ndarray]) -> float:
    return max(
        float(np.max(np.abs(values[left] - values[right])))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def snapshot_mismatches(expected: dict, actual: dict) -> list[str]:
    from video_to_spider.rl.replay_rl import _snapshot_value_equal

    if expected.keys() != actual.keys():
        return ["<snapshot keys differ>"]
    return [
        key for key in expected
        if not _snapshot_value_equal(expected[key], actual[key])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml",
    )
    parser.add_argument(
        "--initialization-report", type=Path,
        default=ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml",
    )
    parser.add_argument(
        "--objective-profile", type=Path,
        default=ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml",
    )
    parser.add_argument(
        "--observation-profile", type=Path,
        default=ROOT / "configs/taco_pour_observation_local_236d_v1.yaml",
    )
    parser.add_argument(
        "--action-profile", type=Path,
        default=ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
    )
    parser.add_argument(
        "--boundary", type=Path,
        default=(
            ROOT / "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1"
            / "endpoint20_complete_boundary.pt.gz"
        ),
    )
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.worlds != 4:
        raise ValueError("the frozen gate requires exactly four worlds")

    import torch
    from run_mjwp_ppo import _load_ego_config, _load_reference
    from run_taco_replay_rl import load_accepted_initialization
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.mjwp_env import (
        IndependentMJWPTrainingEnv,
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
    )
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.physics_contract import verify_runtime_model
    from video_to_spider.rl.replay_rl import MJWPIndependentTrainingBackend

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    action, action_report = load_residual_action_profile(args.action_profile)
    initial, provenance = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(
        config.data_path, "cuda:0", expected_frequency=30,
    )
    children = []
    for _ in range(args.worlds):
        child = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=True,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=action,
            ),
            seed=args.seed,
        )
        verify_runtime_model(
            child.env.model_cpu, provenance["validated_physics_contract"]
        )
        tensors = [
            torch.as_tensor(initial[name][None], device="cuda:0", dtype=torch.float32)
            for name in ("qpos", "qvel", "ctrl")
        ]
        child._write_state(*tensors, np.array([True]))
        child._last_ctrl = tensors[2].clone()
        children.append(child)

    vector = IndependentMJWPTrainingEnv(children)
    backend = MJWPIndependentTrainingBackend(vector)
    boundary_raw = gzip.decompress(args.boundary.read_bytes())
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    if np.asarray(boundary["time_indices"]).tolist() != [20]:
        raise ValueError("gate boundary must be the exact endpoint-20 state")
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    exact_states = backend.snapshot()
    exact_mismatches = [snapshot_mismatches(boundary, state) for state in exact_states]
    active_counts = [{
        "contact": int(state["nacon"][0]),
        "constraint": int(state["nefc"][0]),
        "previous_contact": int(state["prev.nacon"][0]),
        "previous_constraint": int(state["prev.nefc"][0]),
    } for state in exact_states]

    for child in children:
        child.set_chunk_reset(start=20, end=60)

    hand_dof = children[0].env_cfg.residual.hand_dof
    zero = np.zeros((1, hand_dof), dtype=np.float32)
    for child in children:
        child.step(zero, auto_reset=False)
    same_action_qpos = [
        child._mjwp.get_qpos(child.ego_cfg, child.env).detach().cpu().numpy()[0]
        for child in children
    ]
    same_action_qvel = [
        child._mjwp.get_qvel(child.ego_cfg, child.env).detach().cpu().numpy()[0]
        for child in children
    ]
    same_action_qpos_spread = pairwise_max(same_action_qpos)
    same_action_qvel_spread = pairwise_max(same_action_qvel)

    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    for child in children:
        child.set_chunk_reset(start=20, end=60)
    chunk_boundary = children[0]._chunk_reset_state
    rng = np.random.default_rng(args.seed)
    sampled_actions = np.clip(
        rng.normal(size=(args.worlds, hand_dof)), -1.0, 1.0
    ).astype(np.float32)
    _, rewards, dones, info = vector.step(sampled_actions)
    after_independent = backend.snapshot()
    independent_qpos = [state["qpos"][0].numpy() for state in after_independent]
    independent_qvel = [state["qvel"][0].numpy() for state in after_independent]

    before_isolated_reset = backend.snapshot()
    children[0].reset()
    after_isolated_reset = backend.snapshot()
    reset_world_mismatches = snapshot_mismatches(chunk_boundary, after_isolated_reset[0])
    untouched_mismatches = [
        snapshot_mismatches(before_isolated_reset[index], after_isolated_reset[index])
        for index in range(1, args.worlds)
    ]

    exact_pass = not any(exact_mismatches)
    isolation_pass = not reset_world_mismatches and not any(untouched_mismatches)
    short_pass = bool(
        np.isfinite(rewards).all()
        and np.isfinite(np.stack(independent_qpos)).all()
        and np.isfinite(np.stack(independent_qvel)).all()
        and len({row.tobytes() for row in sampled_actions}) == args.worlds
        and not dones.any()
        and np.asarray(info["outcome_reference_endpoint"]).tolist() == [21] * args.worlds
    )
    passed = exact_pass and isolation_pass and short_pass
    report = {
        "schema": "taco_pour_multiworld_state_gate_v1",
        "status": "passed" if passed else "failed",
        "worlds": args.worlds,
        "layout": "independent_one_world_mjwp_instances",
        "reason_for_layout": (
            "each world retains an independent packed contact and constraint buffer; "
            "no partial packed-buffer array copy is used"
        ),
        "source_boundary": {
            "path": str(args.boundary.resolve()),
            "artifact_sha256": sha256(args.boundary),
            "uncompressed_pt_sha256": hashlib.sha256(boundary_raw).hexdigest(),
            "reference_endpoint": 20,
        },
        "contracts": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256(args.protocol)},
            "objective": {"path": str(args.objective_profile.resolve()), "sha256": sha256(args.objective_profile)},
            "observation": {"path": str(args.observation_profile.resolve()), "sha256": sha256(args.observation_profile)},
            "action": {"path": str(args.action_profile.resolve()), "sha256": sha256(args.action_profile)},
        },
        "exact_start_state": {
            "passed": exact_pass,
            "all_worlds_bitwise_equal": exact_pass,
            "snapshot_keys_per_world": len(boundary),
            "warp_state_fields_per_world": len(boundary["warp_state_keys"]),
            "mismatches_by_world": exact_mismatches,
            "active_contact_and_constraint_counts": active_counts,
        },
        "same_action_baseline": {
            "action": "zero_residual",
            "control_intervals": 1,
            "qpos_pairwise_max_abs_spread": same_action_qpos_spread,
            "qvel_pairwise_max_abs_spread": same_action_qvel_spread,
            "gpu_bitwise_repeatability_claimed": False,
        },
        "short_rollout": {
            "passed": short_pass,
            "control_intervals": 1,
            "sample_seed": args.seed,
            "independent_action_rows": True,
            "reset_or_domain_noise_enabled": False,
            "outcome_endpoints": np.asarray(info["outcome_reference_endpoint"]).tolist(),
            "rewards": np.asarray(rewards).tolist(),
            "terminated": np.asarray(dones).tolist(),
            "qpos_pairwise_max_abs_spread": pairwise_max(independent_qpos),
            "qvel_pairwise_max_abs_spread": pairwise_max(independent_qvel),
            "attribution_limit": (
                "independent sampled actions are the only configured input difference, "
                "but GPU atomic-order nondeterminism remains possible"
            ),
        },
        "isolated_reset": {
            "passed": isolation_pass,
            "reset_world_matches_complete_boundary": not reset_world_mismatches,
            "other_worlds_bitwise_unchanged": not any(untouched_mismatches),
            "reset_world_mismatches": reset_world_mismatches,
            "untouched_world_mismatches": untouched_mismatches,
        },
        "action_contract": action_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "warp_state_fields_per_world": len(boundary["warp_state_keys"]),
        "exact_start_state": exact_pass,
        "isolated_reset": isolation_pass,
        "same_action_qpos_spread": same_action_qpos_spread,
        "independent_action_qpos_spread": pairwise_max(independent_qpos),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
