#!/usr/bin/env python3
"""Gate natural tail physics + recurrent-state restore without training PPO."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


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
        "--boundary", type=Path,
        default=(
            ROOT / "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1"
            / "endpoint20_complete_boundary.pt.gz"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=(
            ROOT / "runs/taco_pour_multiworld_training_v1/ppo_chunk_20/nn"
            / "last_ep_8_rew__8.024075_.pth"
        ),
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
        "--reset-contract", type=Path,
        default=ROOT / "configs/taco_pour_tail_curriculum_reset_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        PpoAgent,
        _build_asymmetric_critic_config,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.curriculum_reset import (
        capture_physics_rnn_boundary,
        refresh_boundary_rnn_for_actor,
        restore_physics_rnn_boundaries,
    )
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import _snapshot_value_equal

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)

    def make_world():
        world = MJWPVectorEnv(
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
                residual=residual,
            ),
            seed=0,
        )
        verify_runtime_model(
            world.env.model_cpu, initialization["validated_physics_contract"]
        )
        return world

    def make_agent(env, worlds: int, output: Path):
        agent = PpoAgent(
            experiment_dir=output,
            ppo_config=_build_ppo_config(
                num_envs=worlds,
                horizon_length=40,
                seq_length=4,
                max_epochs=8,
                learning_rate=1e-4,
                device="cpu",
                asymmetric_critic=_build_asymmetric_critic_config(worlds * 40),
            ),
            network_config=_build_network_config(4),
            env=env,
        )
        agent.init_tensors()
        agent.model.load_state_dict(checkpoint["model"])
        agent.set_eval()
        return agent

    boundary_artifact = args.boundary.read_bytes()
    start_boundary = torch.load(
        io.BytesIO(gzip.decompress(boundary_artifact)),
        map_location="cpu",
        weights_only=False,
    )
    if int(start_boundary["time_indices"][0]) != 20:
        raise ValueError("curriculum gate requires the exact endpoint-20 boundary")
    checkpoint_raw = args.checkpoint.read_bytes()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False
    )

    target_endpoints = (46, 47, 48, 49, 50)
    with tempfile.TemporaryDirectory(prefix="egoengine_tail_reset_gate_") as temp:
        temp = Path(temp)
        source_env = make_world()
        source_agent = make_agent(source_env, 1, temp / "source")
        source_env.set_env_state(start_boundary)
        source_agent.rnn_states = [
            state.to("cpu").zero_()
            for state in source_agent.model.get_default_rnn_state()
        ]
        source_agent.dones.zero_()
        source_agent.current_rewards.zero_()
        source_agent.current_shaped_rewards.zero_()
        source_agent.current_lengths.zero_()
        source_agent.obs = source_agent.obs_to_tensors(
            source_env.current_observation()
        )

        actions = []
        prefix_observations = []
        boundaries = []
        rollout_rows = []
        for source_endpoint in range(20, max(target_endpoints)):
            prefix_observations.append(source_env.current_observation())
            values = source_agent.get_action_values(source_agent.obs)
            source_agent.rnn_states = values["rnn_states"]
            action = np.asarray(
                source_agent.preprocess_actions(values["mus"]), dtype=np.float32
            )
            obs, reward, done, info = source_env.step(action, auto_reset=False)
            source_agent.obs = source_agent.obs_to_tensors(obs)
            source_agent.dones = torch.as_tensor(done, dtype=torch.uint8)
            shaped = torch.as_tensor(reward[:, None], dtype=torch.float32)
            source_agent.current_rewards += shaped
            source_agent.current_shaped_rewards += shaped
            source_agent.current_lengths += 1
            actions.append(action.copy())
            outcome_endpoint = int(info["outcome_reference_endpoint"][0])
            rollout_rows.append({
                "source_endpoint": source_endpoint,
                "outcome_endpoint": outcome_endpoint,
                "terminated": bool(done[0]),
                "tool_objective_score": float(
                    info["object_tracking_error_per_object"][0, 0]
                ),
            })
            if done[0]:
                raise ValueError(
                    f"frozen policy terminated before tail capture at endpoint {outcome_endpoint}"
                )
            if outcome_endpoint in target_endpoints:
                prefix = np.concatenate(actions, axis=0).astype(np.float32)
                boundaries.append(capture_physics_rnn_boundary(
                    source_agent,
                    source_env,
                    rollout_start_endpoint=20,
                    observation_prefix=prefix_observations,
                    provenance={
                        "generation_device": "cpu",
                        "natural_simulator_rollout": True,
                        "gt_state_injected": False,
                        "deterministic_mean_policy": True,
                        "source_boundary_sha256": hashlib.sha256(
                            boundary_artifact
                        ).hexdigest(),
                        "actor_checkpoint_sha256": hashlib.sha256(
                            checkpoint_raw
                        ).hexdigest(),
                        "prefix_action_sha256": hashlib.sha256(
                            prefix.tobytes()
                        ).hexdigest(),
                        "prefix_control_intervals": len(actions),
                    },
                ))

        if [row["reference_endpoint"] for row in boundaries] != list(target_endpoints):
            raise ValueError("natural rollout did not produce every requested tail boundary")

        same_actor_refreshed = [
            refresh_boundary_rnn_for_actor(source_agent, boundary)
            for boundary in boundaries
        ]
        same_actor_refresh_equal = [
            all(_snapshot_value_equal(left, right) for left, right in zip(
                boundary["rnn_states"], refreshed["rnn_states"], strict=True
            ))
            for boundary, refreshed in zip(
                boundaries, same_actor_refreshed, strict=True
            )
        ]

        buffer = io.BytesIO()
        torch.save({
            "schema": "taco_pour_tail_curriculum_boundaries_v1",
            "boundaries": boundaries,
        }, buffer)
        boundary_output = args.output_dir / "paired_boundaries.pt.gz"
        boundary_output.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))

        batch_env = IndependentMJWPTrainingEnv([make_world() for _ in range(4)])
        batch_agent = make_agent(batch_env, 4, temp / "batch")
        selected = [boundaries[index] for index in (0, 1, 2, 4)]
        restore_audit = restore_physics_rnn_boundaries(
            batch_agent, batch_env, selected
        )

        def one_interval():
            values = batch_agent.get_action_values(batch_agent.obs)
            actions_now = np.asarray(
                batch_agent.preprocess_actions(values["mus"]), dtype=np.float32
            )
            batch_agent.rnn_states = values["rnn_states"]
            obs_now, rewards_now, dones_now, info_now = batch_env.step(actions_now)
            return {
                "actions": actions_now.copy(),
                "rnn_states": tuple(
                    state.detach().cpu().clone() for state in batch_agent.rnn_states
                ),
                "physics": batch_env.get_env_state(),
                "observation": obs_now,
                "rewards": rewards_now.copy(),
                "dones": dones_now.copy(),
                "info": info_now,
            }

        first = one_interval()
        second_restore_audit = restore_physics_rnn_boundaries(
            batch_agent, batch_env, selected
        )
        second = one_interval()
        repeat_equal = {
            key: _snapshot_value_equal(first[key], second[key])
            for key in first
        }

        changed_parameter_rejected = False
        refreshed_memory_accepted = False
        refreshed_physics_unchanged = False
        first_parameter = next(source_agent.model.parameters())
        original_parameter = first_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.reshape(-1)[0] += 1e-5
        try:
            try:
                restore_physics_rnn_boundaries(
                    source_agent, source_env, [boundaries[0]]
                )
            except ValueError as error:
                changed_parameter_rejected = (
                    "different actor/normalization" in str(error)
                )
            refreshed_for_changed_actor = refresh_boundary_rnn_for_actor(
                source_agent, boundaries[0]
            )
            refreshed_physics_unchanged = _snapshot_value_equal(
                boundaries[0]["physics_state"],
                refreshed_for_changed_actor["physics_state"],
            )
            changed_actor_restore = restore_physics_rnn_boundaries(
                source_agent, source_env, [refreshed_for_changed_actor]
            )
            refreshed_memory_accepted = bool(
                changed_actor_restore["physics_bitwise_equal"] == [True]
                and changed_actor_restore["observation_bitwise_equal"] is True
            )
        finally:
            with torch.no_grad():
                first_parameter.copy_(original_parameter)

        source_agent.writer.close()
        batch_agent.writer.close()

    warp_fields = [
        len(boundary["physics_state"]["warp_state_keys"])
        for boundary in boundaries
    ]
    passed = bool(
        warp_fields == [342] * len(boundaries)
        and restore_audit["physics_bitwise_equal"] == [True] * 4
        and restore_audit["observation_bitwise_equal"] is True
        and second_restore_audit["physics_bitwise_equal"] == [True] * 4
        and same_actor_refresh_equal == [True] * len(boundaries)
        and all(repeat_equal.values())
        and changed_parameter_rejected
        and refreshed_memory_accepted
        and refreshed_physics_unchanged
    )
    report = {
        "schema": "taco_pour_tail_curriculum_reset_gate_v1",
        "status": "passed" if passed else "failed",
        "scope": {
            "training_executed": False,
            "algorithm_changed": False,
            "device": "cpu",
            "purpose": "paired physics/RNN restore engineering gate only",
        },
        "contracts": {
            "config": {"path": project_path(args.config), "sha256": sha256(args.config)},
            "protocol": {"path": project_path(args.protocol), "sha256": sha256(args.protocol)},
            "tail_curriculum_reset": {
                "path": project_path(args.reset_contract),
                "sha256": sha256(args.reset_contract),
            },
            "objective": objective.as_report(),
            "observation": observation.as_report(),
            "residual_action": residual_report,
        },
        "inputs": {
            "start_boundary": {
                "path": project_path(args.boundary),
                "sha256": hashlib.sha256(boundary_artifact).hexdigest(),
                "reference_endpoint": 20,
            },
            "actor_checkpoint": {
                "path": project_path(args.checkpoint),
                "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
                "epoch": int(checkpoint["epoch"]),
                "frame": int(checkpoint["frame"]),
            },
        },
        "natural_rollout": {
            "gt_state_injected": False,
            "deterministic_mean_policy": True,
            "rows": rollout_rows,
            "captured_endpoints": list(target_endpoints),
        },
        "paired_boundary_artifact": {
            "path": project_path(boundary_output),
            "sha256": sha256(boundary_output),
            "count": len(boundaries),
            "snapshot_keys_per_boundary": [
                len(boundary["physics_state"]) for boundary in boundaries
            ],
            "warp_state_fields_per_boundary": warp_fields,
            "rnn_state_shapes_per_boundary": [
                [list(state.shape) for state in boundary["rnn_states"]]
                for boundary in boundaries
            ],
            "actor_state_sha256": boundaries[0]["actor_state_sha256"],
            "actor_hash_scope": boundaries[0]["actor_hash_scope"],
            "input_normalization_keys": boundaries[0][
                "actor_normalization_state_keys"
            ],
        },
        "four_world_restore": restore_audit,
        "four_world_second_restore": second_restore_audit,
        "deterministic_one_interval_repeat": {
            "bitwise_equal": repeat_equal,
            "all_bitwise_equal": all(repeat_equal.values()),
        },
        "stale_actor_memory_rejection": {
            "changed_actor_parameter_rejected": changed_parameter_rejected,
            "passed": changed_parameter_rejected,
        },
        "actor_update_memory_refresh": {
            "method": "replay_natural_observation_prefix_under_current_actor",
            "same_actor_replay_matches_saved_RNN_bitwise": same_actor_refresh_equal,
            "changed_actor_old_memory_rejected": changed_parameter_rejected,
            "changed_actor_refreshed_memory_accepted": refreshed_memory_accepted,
            "physical_state_unchanged_by_refresh": refreshed_physics_unchanged,
            "passed": bool(
                same_actor_refresh_equal == [True] * len(boundaries)
                and changed_parameter_rejected
                and refreshed_memory_accepted
                and refreshed_physics_unchanged
            ),
        },
        "semantic_limit": {
            "saved_memory_reusable_after_actor_update_without_refresh": False,
            "implemented_refresh_behavior": (
                "replay each saved natural observation prefix through the current actor"
            ),
            "physical_tail_state_generated_by_current_actor_after_update": False,
            "exploration_rng_restored": False,
            "CPU_acceptance_still_starts_at_endpoint": 20,
            "CPU_acceptance_required_steps": 40,
        },
        "PPO_training_authorized": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "captured_endpoints": list(target_endpoints),
        "warp_fields": warp_fields,
        "four_world_rnn_shapes": restore_audit["rnn_state_shapes"],
        "repeat_equal": repeat_equal,
        "changed_actor_parameter_rejected": changed_parameter_rejected,
        "changed_actor_refreshed_memory_accepted": refreshed_memory_accepted,
        "PPO_training_authorized": False,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
