#!/usr/bin/env python3
"""No-training gate for the Replay-to-RL reward/reference transition contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actor_array(observation):
    return observation["obs"] if isinstance(observation, dict) else observation


def _apply_initial_state(env, initial, torch) -> None:
    tensors = [
        torch.as_tensor(initial[name][None], device=str(env.ego_cfg.device), dtype=torch.float32)
        for name in ("qpos", "qvel", "ctrl")
    ]
    env._write_state(*tensors, np.array([True]))
    env._last_ctrl = tensors[2].clone()
    env._check_capacity()


def _expected_goal_anchors(env, endpoints):
    import torch
    from video_to_spider.rl.mjwp_env import _object_pose_parts, _transform_anchors_torch

    qpos = env.qpos_ref[np.asarray(endpoints, dtype=np.int64)].to(str(env.ego_cfg.device))
    objects = _object_pose_parts(qpos, int(env.ego_cfg.nq_obj))
    return torch.cat(
        [_transform_anchors_torch(pose[0], pose[1], env.anchors) for pose in objects],
        dim=1,
    ).reshape(len(endpoints), -1).cpu().numpy()


def _recompute_errors(env, endpoints):
    from egoengine_repro.action.paper_rewards import object_tracking
    from video_to_spider.rl.mjwp_env import _object_pose_parts

    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)
    current = _object_pose_parts(qpos, int(env.ego_cfg.nq_obj))
    goal = env._reference_object_poses_at_absolute_endpoints(
        np.asarray(endpoints, dtype=np.int64)
    )
    scores = [
        object_tracking(
            current[index][0], current[index][1], goal[index][0], goal[index][1],
            env.objective.tracking,
        )
        for index in env.tracked_object_indices
    ]
    return (
        np.stack([score.position_error.detach().cpu().numpy() for score in scores], axis=1),
        np.stack([score.rotation_error.detach().cpu().numpy() for score in scores], axis=1),
    )


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


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
        "--gate-contract", type=Path,
        default=ROOT / "configs/taco_pour_reward_alignment_gate_v1.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_reward_alignment_gate_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    contract = yaml.safe_load(args.gate_contract.read_text())
    if contract.get("schema") != "taco_pour_reward_alignment_gate_v1":
        raise ValueError("unsupported reward-alignment gate contract")
    if contract.get("training", {}).get("optimizer_updates") != 0:
        raise ValueError("reward-alignment gate must forbid optimizer updates")

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_and_target", require_run_ready=False,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    initial, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )

    def make_world(device: str, *, max_episode_length: int = 197):
        config = _load_ego_config(str(args.config), device)
        reference = _load_reference(config.data_path, device, expected_frequency=30.0)
        env = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=True,
                max_episode_length=max_episode_length,
                tracked_object_indices=None,
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=0,
        )
        verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
        _apply_initial_state(env, initial, torch)
        return env

    zero = np.zeros((1, 36), dtype=np.float32)

    single = make_world("cpu")
    accepted_initial_snapshot = single.get_env_state()
    source = int(single.time_indices[0])
    pre_obs = _actor_array(single.current_observation())
    pre_goal = _expected_goal_anchors(single, [source + 1])
    pre_goal_error = _max_abs(pre_obs[:, 126:144], pre_goal)
    next_obs, _, _, single_info = single.step(zero, auto_reset=False)
    outcome = int(single.time_indices[0])
    correct_position, correct_rotation = _recompute_errors(single, [outcome])
    legacy_position, legacy_rotation = _recompute_errors(single, [outcome + 1])
    next_goal = _expected_goal_anchors(single, [outcome + 1])
    next_goal_error = _max_abs(_actor_array(next_obs)[:, 126:144], next_goal)
    runtime_position_error = np.asarray(single_info["object_position_error"])
    runtime_rotation_error = np.asarray(single_info["object_rotation_error"])
    correct_error = max(
        _max_abs(runtime_position_error, correct_position),
        _max_abs(runtime_rotation_error, correct_rotation),
    )
    legacy_error = max(
        _max_abs(runtime_position_error, legacy_position),
        _max_abs(runtime_rotation_error, legacy_rotation),
    )

    class LegacyRewardTargetEnv(MJWPVectorEnv):
        def _reference_object_poses_at_absolute_endpoints(self, endpoints):
            return super()._reference_object_poses_at_absolute_endpoints(
                np.asarray(endpoints, dtype=np.int64) + 1
            )

    def make_legacy_world():
        config = _load_ego_config(str(args.config), "cpu")
        reference = _load_reference(config.data_path, "cpu", expected_frequency=30.0)
        env = LegacyRewardTargetEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=True,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=None,
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=0,
        )
        verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
        _apply_initial_state(env, initial, torch)
        return env

    single.set_env_state(accepted_initial_snapshot)
    fixed = single
    legacy = make_legacy_world()
    physical_checks = []
    reward_difference_seen = False
    auxiliary_equal = True
    for _ in range(5):
        _, fixed_reward, _, fixed_info = fixed.step(zero, auto_reset=False)
        _, legacy_reward, _, legacy_info = legacy.step(zero, auto_reset=False)
        reward_difference_seen |= not np.array_equal(fixed_reward, legacy_reward)
        auxiliary_equal &= all(
            np.array_equal(fixed_info[name], legacy_info[name])
            for name in ("contact_score", "lift_reward", "contact_flags")
        )
        for name, left, right in (
            ("qpos", fixed._mjwp.get_qpos(fixed.ego_cfg, fixed.env), legacy._mjwp.get_qpos(legacy.ego_cfg, legacy.env)),
            ("qvel", fixed._mjwp.get_qvel(fixed.ego_cfg, fixed.env), legacy._mjwp.get_qvel(legacy.ego_cfg, legacy.env)),
            ("ctrl", fixed._last_ctrl, legacy._last_ctrl),
        ):
            equal = torch.equal(left, right)
            physical_checks.append({"endpoint": int(fixed.time_indices[0]), "field": name, "bitwise_equal": equal})

    fixed.set_env_state(accepted_initial_snapshot)
    fixed.episode_lengths[:] = 1
    fixed.step(zero, auto_reset=False)
    terminal_same_time_position, terminal_same_time_rotation = _recompute_errors(
        fixed, [1]
    )
    fixed.set_env_state(accepted_initial_snapshot)
    fixed.episode_lengths[:] = 1
    _, terminal_reward, terminal_done, terminal_info = fixed.step(zero, auto_reset=True)
    terminal_position = np.asarray(terminal_info["object_position_error"])
    terminal_preserved = (
        bool(terminal_done[0])
        and int(terminal_info["reward_reference_endpoint"][0]) == 1
        and _max_abs(terminal_position, terminal_same_time_position) == 0.0
        and _max_abs(
            terminal_info["object_rotation_error"], terminal_same_time_rotation
        ) == 0.0
        and np.array_equal(terminal_reward, terminal_info["reward"])
    )

    starts = [20, 20, 20, 46]
    worlds = [make_world("cuda:0") for _ in starts]
    for world, target in zip(worlds, starts, strict=True):
        for _ in range(target):
            world.step(zero, auto_reset=False)
    multi = IndependentMJWPTrainingEnv(worlds)
    multi_obs_before = _actor_array(multi.current_observation())
    # Expected anchors need the common reference but one row per independent world.
    multi_pre_goal = _expected_goal_anchors(worlds[0], np.asarray(starts) + 1)
    multi_pre_goal_error = _max_abs(multi_obs_before[:, 126:144], multi_pre_goal)
    multi_next, _, _, multi_info = multi.step(np.zeros((4, 36), dtype=np.float32))
    multi_next_goal = _expected_goal_anchors(worlds[0], np.asarray(starts) + 2)
    multi_next_goal_error = _max_abs(
        _actor_array(multi_next)[:, 126:144], multi_next_goal
    )
    multi_correct_rows = []
    for world, endpoint in zip(worlds, np.asarray(starts) + 1, strict=True):
        pos, rot = _recompute_errors(world, [endpoint])
        multi_correct_rows.append((pos[0], rot[0]))
    multi_position = np.stack([row[0] for row in multi_correct_rows])
    multi_rotation = np.stack([row[1] for row in multi_correct_rows])
    multi_reward_error = max(
        _max_abs(multi_info["object_position_error"], multi_position),
        _max_abs(multi_info["object_rotation_error"], multi_rotation),
    )

    checks = {
        "runtime_reward_matches_outcome_endpoint_reference": correct_error == 0.0,
        "legacy_outcome_plus_one_reference_is_negative_control": legacy_error > 0.0,
        "actor_pre_step_goal_remains_t_plus_1": pre_goal_error == 0.0,
        "returned_next_observation_goal_remains_t_plus_2": next_goal_error == 0.0,
        "fixed_action_physics_is_bitwise_unchanged_by_reward_target": all(
            row["bitwise_equal"] for row in physical_checks
        ),
        "contact_lift_and_ctrl_are_unchanged_by_reward_target": auxiliary_equal,
        "reward_changes_under_legacy_negative_control": reward_difference_seen,
        "terminal_info_retains_pre_reset_same_time_reward": terminal_preserved,
        "single_world_endpoint_provenance_passes": (
            single_info["command_reference_endpoint"].tolist() == [1]
            and single_info["reward_reference_endpoint"].tolist() == [1]
            and single_info["next_observation_goal_reference_endpoint"].tolist() == [2]
        ),
        "four_world_20_20_20_46_endpoint_provenance_passes": (
            multi_info["source_reference_endpoint"].tolist() == starts
            and multi_info["outcome_reference_endpoint"].tolist() == [21, 21, 21, 47]
            and multi_info["command_reference_endpoint"].tolist() == [21, 21, 21, 47]
            and multi_info["reward_reference_endpoint"].tolist() == [21, 21, 21, 47]
            and multi_info["next_observation_goal_reference_endpoint"].tolist() == [22, 22, 22, 48]
            and multi_pre_goal_error == 0.0
            and multi_next_goal_error == 0.0
            and multi_reward_error == 0.0
        ),
    }
    required = set(contract["required_checks"])
    if not required.issubset(checks):
        raise RuntimeError(f"gate implementation omitted checks: {sorted(required - set(checks))}")
    passed = all(checks.values())
    report = {
        "schema": "taco_pour_reward_alignment_gate_v1",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "optimizer_updates": 0,
        "task_level_training_executed": False,
        "chunk_commit_written": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("gate_contract", args.gate_contract),
                ("protocol", args.protocol),
                ("objective_profile", args.objective_profile),
                ("observation_profile", args.observation_profile),
                ("action_profile", args.action_profile),
                ("simulator_config", args.config),
                ("initialization_report", args.initialization_report),
            )
        },
        "runtime": {
            "snapshot_schema": fixed.get_env_state()["snapshot_schema"],
            "training_trace_schema": "taco_ppo_training_visitation_v5",
            "residual_action": residual_report,
        },
        "implementation": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("mjwp_environment", ROOT / "src/video_to_spider/rl/mjwp_env.py"),
                ("chunk_backend", ROOT / "src/video_to_spider/rl/replay_rl.py"),
                ("training_trace", ROOT / "src/video_to_spider/rl/training_trace.py"),
                ("formal_runner", ROOT / "scripts/run_taco_replay_rl.py"),
                ("gate_script", Path(__file__)),
            )
        },
        "single_world": {
            "source_endpoint": source,
            "outcome_endpoint": outcome,
            "same_time_recompute_max_abs_error": correct_error,
            "legacy_plus_one_recompute_max_abs_error": legacy_error,
            "actor_pre_step_goal_max_abs_error": pre_goal_error,
            "next_observation_goal_max_abs_error": next_goal_error,
        },
        "fixed_vs_legacy_reward_target": {
            "steps": 5,
            "physics": physical_checks,
            "reward_difference_seen": reward_difference_seen,
            "contact_lift_equal": auxiliary_equal,
        },
        "terminal_autoreset": {
            "done": bool(terminal_done[0]),
            "reward_reference_endpoint": int(terminal_info["reward_reference_endpoint"][0]),
            "terminal_position_recompute_max_abs_error": _max_abs(
                terminal_position, terminal_same_time_position
            ),
            "terminal_rotation_recompute_max_abs_error": _max_abs(
                terminal_info["object_rotation_error"], terminal_same_time_rotation
            ),
        },
        "multiworld": {
            "source_endpoints": starts,
            "outcome_endpoints": multi_info["outcome_reference_endpoint"].tolist(),
            "same_time_recompute_max_abs_error": multi_reward_error,
            "actor_pre_step_goal_max_abs_error": multi_pre_goal_error,
            "next_observation_goal_max_abs_error": multi_next_goal_error,
        },
        "decision": {
            "formal_reward_alignment_validated": passed,
            "old_endpoint20_boundary_reuse_allowed": False,
            "corrected_replay_must_rebase_from_endpoint": 0,
            "old_policy_resume_allowed": False,
        },
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("reward-alignment gate failed")


if __name__ == "__main__":
    main()
