#!/usr/bin/env python3
"""Zero-optimizer gate for the first corrected-reward four-world PPO fallback."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "--distribution-profile", type=Path,
        default=ROOT / "configs/taco_pour_state_feasible_truncated_gaussian_v1.yaml",
    )
    parser.add_argument(
        "--source-boundary", type=Path,
        default=(ROOT / "runs/taco_pour_corrected_replay_rebase_v1/tool_only"
                 / "committed_boundary_endpoint_20.pt.gz"),
    )
    parser.add_argument(
        "--reward-alignment-gate", type=Path,
        default=ROOT / "runs/taco_pour_reward_alignment_gate_v1/report.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_corrected_ppo_gate_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch
    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _build_asymmetric_critic_config,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import (
        MJWPIndependentTrainingBackend,
        _model_state_sha256,
        _snapshot_value_equal,
    )
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
        truncated_normal_log_prob,
    )

    alignment = json.loads(args.reward_alignment_gate.read_text())
    if (
        alignment.get("status") != "passed"
        or not all(alignment.get("checks", {}).values())
        or alignment.get("decision", {}).get("corrected_replay_must_rebase_from_endpoint") != 0
    ):
        raise ValueError("corrected PPO gate requires the passed reward-alignment gate")
    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    spec, profile = load_truncated_gaussian_profile(args.distribution_profile)
    if not np.isclose(residual.residual_scale, spec.residual_scale):
        raise ValueError("residual and truncated-Gaussian profiles disagree")
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)

    artifact = args.source_boundary.read_bytes()
    raw = gzip.decompress(artifact)
    source = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if (
        source.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(source["time_indices"]).tolist() != [20]
    ):
        raise ValueError("corrected PPO gate requires the new endpoint-20 v3 boundary")

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
        world.set_env_state(source)
        return world

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with tempfile.TemporaryDirectory(prefix="egoengine_corrected_ppo_gate_") as temp:
        env = IndependentMJWPTrainingEnv([make_world() for _ in range(4)])
        backend = MJWPIndependentTrainingBackend(env)
        backend.verify_restored_snapshot(source)
        exact_start = all(
            _snapshot_value_equal(world.get_env_state(), source)
            for world in env.worlds
        )
        env.set_chunk_reset(start=20, end=60)
        boundary = env.worlds[0]._chunk_reset_state
        before_reset = env.get_env_states()
        env.worlds[0].reset()
        isolated_reset = (
            _snapshot_value_equal(env.worlds[0].get_env_state(), boundary)
            and all(
                _snapshot_value_equal(env.worlds[index].get_env_state(), before_reset[index])
                for index in range(1, 4)
            )
        )
        env.set_env_state(source)
        env.set_chunk_reset(start=20, end=60)

        ppo_config = replace(
            _build_ppo_config(
                num_envs=4,
                horizon_length=40,
                seq_length=4,
                max_epochs=8,
                learning_rate=1e-4,
                device="cuda:0",
                asymmetric_critic=_build_asymmetric_critic_config(160),
            ),
            clip_actions=False,
        )
        agent = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
        )
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.epoch_num = 1
        env.enable_training_trace(args.output_dir / "training_visitation")
        initial_states = {
            "model": copy.deepcopy(agent.model.state_dict()),
            "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
            "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": copy.deepcopy(
                agent.asymmetric_critic_net.optimizer.state_dict()
            ),
        }
        actor_before = _model_state_sha256(agent.model.state_dict())
        completed = False
        try:
            env.set_train_info(0, agent)
            starts = [int(world.time_indices[0]) for world in env.worlds]
            batch = agent.play_steps_rnn()
            completed = True
        finally:
            trace = env.finalize_training_trace(completed=completed)
            agent.writer.close()

        final_states = {
            "model": copy.deepcopy(agent.model.state_dict()),
            "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
            "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": copy.deepcopy(
                agent.asymmetric_critic_net.optimizer.state_dict()
            ),
        }
        state_unchanged = {
            name: _snapshot_value_equal(initial_states[name], final_states[name])
            for name in initial_states
        }
        actor_after = _model_state_sha256(agent.model.state_dict())

        actions = batch["actions"]
        lows = batch["action_lows"]
        highs = batch["action_highs"]
        old_neglogp = batch["neglogpacs"]
        recomputed = -truncated_normal_log_prob(
            actions, batch["mus"], batch["sigmas"], lows, highs,
            minimum_mass=spec.minimum_normalization_mass,
        ).sum(dim=-1)
        ratio = torch.exp(old_neglogp - recomputed)
        with torch.no_grad():
            network = agent.evaluate_ppo_distribution({
                "actions": actions,
                "action_lows": lows,
                "action_highs": highs,
                "obs": batch["obses"],
                "dones": batch["dones"],
                "rnn_states": batch["rnn_states"],
            })
        network_ratio = torch.exp(old_neglogp - network["neglogp"])

        mu = batch["mus"].detach().clone().requires_grad_(True)
        sigma = batch["sigmas"].detach().clone().requires_grad_(True)
        gradient_loss = -truncated_normal_log_prob(
            actions.detach(), mu, sigma, lows.detach(), highs.detach(),
            minimum_mass=spec.minimum_normalization_mass,
        ).mean()
        mu_grad, sigma_grad = torch.autograd.grad(gradient_loss, (mu, sigma))

        manifest = json.loads(Path(trace["path"]).read_text())
        parts = []
        for epoch in manifest["epochs"]:
            with np.load(epoch["visits"]["path"], allow_pickle=False) as loaded:
                parts.append({name: np.asarray(loaded[name]) for name in loaded.files})
        visits = {
            name: np.concatenate([part[name] for part in parts], axis=0)
            for name in parts[0]
        }
        official_clamp_changes = int((
            visits["sampled_action_preclamp"] != visits["sampled_action_clamped"]
        ).sum())
        lost_components = int((visits["residual_lost_to_ctrlrange"] != 0.0).sum())
        temporal_trace = bool(
            np.array_equal(
                visits["command_reference_endpoint"], visits["outcome_endpoint"]
            )
            and np.array_equal(
                visits["reward_reference_endpoint"], visits["outcome_endpoint"]
            )
            and np.array_equal(
                visits["next_observation_goal_reference_endpoint"],
                visits["outcome_endpoint"] + 1,
            )
        )
        optimizer_rejected = False
        try:
            agent.train_actor_critic({})
        except RuntimeError as error:
            optimizer_rejected = "optimizer training is forbidden" in str(error)

        env.set_env_state(source)
        restored = all(
            _snapshot_value_equal(world.get_env_state(), source)
            for world in env.worlds
        )
        action_audit = env.state_feasible_action_audit()

    checks = {
        "four_worlds_restore_corrected_boundary_bitwise": exact_start,
        "isolated_world_reset_preserves_other_worlds": isolated_reset,
        "world_starts_are_20_20_20_20": starts == [20, 20, 20, 20],
        "rollout_sample_count_is_160": len(actions) == 160,
        "all_actions_inside_state_bounds": bool(((actions >= lows) & (actions <= highs)).all()),
        "official_action_clamp_is_noop": official_clamp_changes == 0,
        "actuator_ctrlrange_loss_is_zero": lost_components == 0,
        "same_policy_logprob_is_bitwise_equal": torch.equal(old_neglogp, recomputed),
        "same_policy_ratio_is_bitwise_one": torch.equal(ratio, torch.ones_like(ratio)),
        "network_recomputed_mu_is_bitwise_equal": torch.equal(batch["mus"], network["mu"]),
        "network_recomputed_sigma_is_bitwise_equal": torch.equal(batch["sigmas"], network["sigma"]),
        "network_recomputed_logprob_is_bitwise_equal": torch.equal(old_neglogp, network["neglogp"]),
        "network_recomputed_ratio_is_bitwise_one": torch.equal(
            network_ratio, torch.ones_like(network_ratio)
        ),
        "truncated_logprob_gradients_are_finite": bool(
            torch.isfinite(mu_grad).all() and torch.isfinite(sigma_grad).all()
        ),
        "actor_critic_and_optimizers_unchanged": all(state_unchanged.values()),
        "actor_hash_unchanged": actor_before == actor_after,
        "training_trace_v5_temporal_endpoints_pass": temporal_trace,
        "optimizer_path_remains_fail_closed": optimizer_rejected,
        "incoming_boundary_restored_after_gate": restored,
    }
    passed = all(checks.values())
    implementation_paths = {
        "state_feasible_distribution": ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
        "mjwp_environment": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "ppo_chunk_adapter": ROOT / "src/video_to_spider/rl/replay_rl.py",
        "formal_runner": ROOT / "scripts/run_taco_replay_rl.py",
        "gate_script": Path(__file__),
    }
    report = {
        "schema": "taco_pour_corrected_ppo_gate_v1",
        "status": "passed" if passed else "failed",
        "paper_faithful": False,
        "PPO_optimizer_steps": 0,
        "task_level_training_executed": False,
        "chunk_commit_written": False,
        "profile": profile,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("source_boundary", args.source_boundary),
                ("reward_alignment_gate", args.reward_alignment_gate),
                ("protocol", args.protocol),
                ("objective_profile", args.objective_profile),
                ("observation_profile", args.observation_profile),
                ("action_profile", args.action_profile),
                ("distribution_profile", args.distribution_profile),
                ("simulator_config", args.config),
                ("initialization_report", args.initialization_report),
            )
        },
        "implementation": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in implementation_paths.items()
        },
        "frozen_rollout": {
            "world_start_endpoints": starts,
            "worlds": 4,
            "horizon": 40,
            "samples": int(len(actions)),
            "seed": 0,
            "optimizer_updates": 0,
        },
        "rollout_buffer": {
            "actions_shape": list(actions.shape),
            "action_lows_shape": list(lows.shape),
            "action_highs_shape": list(highs.shape),
            "same_policy_ratio_max_abs_error": float((ratio - 1).abs().max()),
            "network_recomputed_ratio_max_abs_error": float(
                (network_ratio - 1).abs().max()
            ),
        },
        "environment_action_semantics": {
            "official_clamp_changed_components": official_clamp_changes,
            "actual_ctrlrange_nonzero_lost_components": lost_components,
            "reference_contract_audit": action_audit,
        },
        "training_trace": trace,
        "residual_action": residual_report,
        "checks": checks,
        "decision": {
            "engineering_gate_passed": passed,
            "single_corrected_ppo_experiment_may_be_authorized": passed,
            "chunk_commit_requires_deterministic_CPU_40_of_40": True,
            "tail_curriculum_allowed": False,
            "old_policy_resume_allowed": False,
        },
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("corrected PPO integration gate failed")


if __name__ == "__main__":
    main()
