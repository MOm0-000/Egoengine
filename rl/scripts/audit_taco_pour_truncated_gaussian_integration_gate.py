#!/usr/bin/env python3
"""Four-world, zero-optimizer gate for the feasible truncated Gaussian."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(path: Path) -> str:
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
        "--source-boundary", type=Path,
        default=(ROOT / "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1"
                 / "endpoint20_complete_boundary.pt.gz"),
    )
    parser.add_argument(
        "--paired-boundaries", type=Path,
        default=ROOT / "runs/taco_pour_tail_curriculum_reset_gate_v1/paired_boundaries.pt.gz",
    )
    parser.add_argument(
        "--curriculum-contract", type=Path,
        default=ROOT / "configs/taco_pour_tail_curriculum_3plus1_v1.yaml",
    )
    parser.add_argument(
        "--distribution-profile", type=Path,
        default=ROOT / "configs/taco_pour_state_feasible_truncated_gaussian_v1.yaml",
    )
    parser.add_argument(
        "--offline-gate", type=Path,
        default=ROOT / "runs/taco_pour_truncated_gaussian_offline_gate_v1/report.json",
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
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_truncated_gaussian_integration_gate_v1",
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
    from video_to_spider.rl.replay_rl import _model_state_sha256, _snapshot_value_equal
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
        truncated_normal_log_prob,
    )

    offline = json.loads(args.offline_gate.read_text())
    spec, profile = load_truncated_gaussian_profile(args.distribution_profile)
    curriculum_contract = yaml.safe_load(args.curriculum_contract.read_text())
    if curriculum_contract.get("schema") != "taco_pour_tail_curriculum_3plus1_v1":
        raise ValueError("unsupported curriculum contract")
    if (
        offline.get("schema") != "taco_pour_truncated_gaussian_offline_gate_v1"
        or offline.get("status") != "passed"
        or offline.get("decision", {}).get("optimizer_training_authorized") is not False
        or offline.get("profile", {}).get("profile_sha256")
        != profile["profile_sha256"]
    ):
        raise ValueError("the exact candidate profile lacks a passed offline gate")

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_report = load_residual_action_profile(args.action_profile)
    if not np.isclose(residual.residual_scale, spec.residual_scale):
        raise ValueError("residual action profile and distribution scale differ")
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)

    source_raw = args.source_boundary.read_bytes()
    source = torch.load(
        io.BytesIO(gzip.decompress(source_raw)), map_location="cpu", weights_only=False
    )
    if int(np.asarray(source["time_indices"])[0]) != 20:
        raise ValueError("source boundary must be endpoint 20")
    paired_raw = args.paired_boundaries.read_bytes()
    paired = torch.load(
        io.BytesIO(gzip.decompress(paired_raw)), map_location="cpu", weights_only=False
    )
    tail_rows = [
        row for row in paired.get("boundaries", ())
        if int(row.get("reference_endpoint", -1)) == 46
    ]
    if paired.get("schema") != "taco_pour_tail_curriculum_boundaries_v1" or len(tail_rows) != 1:
        raise ValueError("paired artifact must contain exactly one endpoint-46 boundary")
    tail = tail_rows[0]

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
    with tempfile.TemporaryDirectory(prefix="egoengine_truncated_gaussian_gate_") as temp:
        env = IndependentMJWPTrainingEnv([make_world() for _ in range(4)])
        env.set_chunk_reset(start=20, end=60)
        env.enable_tail_curriculum(
            tail,
            anchor_endpoint=20,
            tail_endpoint=46,
            window_end_endpoint=60,
        )
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

        initial = {
            "model": copy.deepcopy(agent.model.state_dict()),
            "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
            "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": copy.deepcopy(
                agent.asymmetric_critic_net.optimizer.state_dict()
            ),
        }
        actor_hash_before = _model_state_sha256(agent.model.state_dict())
        completed = False
        try:
            env.set_train_info(0, agent)
            starts = [int(world.time_indices[0]) for world in env.worlds]
            batch = agent.play_steps_rnn()
            completed = True
        finally:
            trace = env.finalize_training_trace(completed=completed)
            agent.writer.close()

        final = {
            "model": copy.deepcopy(agent.model.state_dict()),
            "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
            "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
            "critic_optimizer": copy.deepcopy(
                agent.asymmetric_critic_net.optimizer.state_dict()
            ),
        }
        unchanged = {
            key: _snapshot_value_equal(initial[key], final[key]) for key in initial
        }
        actor_hash_after = _model_state_sha256(agent.model.state_dict())

        actions = batch["actions"]
        lows = batch["action_lows"]
        highs = batch["action_highs"]
        old_neglogp = batch["neglogpacs"]
        recomputed = -truncated_normal_log_prob(
            actions,
            batch["mus"],
            batch["sigmas"],
            lows,
            highs,
            minimum_mass=spec.minimum_normalization_mass,
        ).sum(dim=-1)
        ratio = torch.exp(old_neglogp - recomputed)
        with torch.no_grad():
            network_result = agent.evaluate_ppo_distribution({
                "actions": actions,
                "action_lows": lows,
                "action_highs": highs,
                "obs": batch["obses"],
                "dones": batch["dones"],
                "rnn_states": batch["rnn_states"],
            })
        actor_hash_after_network_recompute = _model_state_sha256(
            agent.model.state_dict()
        )
        network_ratio = torch.exp(old_neglogp - network_result["neglogp"])
        actions_in_bounds = bool(((actions >= lows) & (actions <= highs)).all().item())
        exact_logprob = bool(torch.equal(old_neglogp, recomputed))
        exact_ratio = bool(torch.equal(ratio, torch.ones_like(ratio)))
        network_exact_mu = bool(torch.equal(batch["mus"], network_result["mu"]))
        network_exact_sigma = bool(torch.equal(batch["sigmas"], network_result["sigma"]))
        network_exact_logprob = bool(torch.equal(
            old_neglogp, network_result["neglogp"]
        ))
        network_exact_ratio = bool(torch.equal(
            network_ratio, torch.ones_like(network_ratio)
        ))
        buffer_shapes = {
            "actions": list(actions.shape),
            "action_lows": list(lows.shape),
            "action_highs": list(highs.shape),
        }

        trace_manifest = json.loads(Path(trace["path"]).read_text())
        visit_parts = []
        for epoch in trace_manifest["epochs"]:
            with np.load(epoch["visits"]["path"], allow_pickle=False) as raw:
                visit_parts.append({name: np.asarray(raw[name]) for name in raw.files})
        visits = {
            name: np.concatenate([part[name] for part in visit_parts], axis=0)
            for name in visit_parts[0]
        }
        sample_count = int(visits["sampled_action_preclamp"].shape[0])
        official_clamp_changes = int((
            visits["sampled_action_preclamp"]
            != visits["sampled_action_clamped"]
        ).sum())
        lost = np.abs(visits["residual_lost_to_ctrlrange"])
        exact_lost_components = int((lost != 0.0).sum())
        lost_components = int((lost > spec.reference_snap_tolerance).sum())
        max_lost = float(lost.max(initial=0.0))

        optimizer_rejected = False
        try:
            agent.train_actor_critic({})
        except RuntimeError as error:
            optimizer_rejected = "optimizer training is forbidden" in str(error)

        env.set_env_state(source)
        restored_endpoints = [int(world.time_indices[0]) for world in env.worlds]
        restored_states_equal = all(
            _snapshot_value_equal(world.get_env_state(), source)
            for world in env.worlds
        )
        action_audit = env.state_feasible_action_audit()

    checks = {
        "world_starts_are_frozen_3plus1": starts == [20, 20, 20, 46],
        "rollout_sample_count_is_160": sample_count == 160,
        "rollout_bounds_stored_as_160x36": buffer_shapes == {
            "actions": [160, 36],
            "action_lows": [160, 36],
            "action_highs": [160, 36],
        },
        "all_actions_inside_stored_state_bounds": actions_in_bounds,
        "official_clamp_changed_zero_components": official_clamp_changes == 0,
        "actual_ctrlrange_loss_exactly_zero": exact_lost_components == 0,
        "same_policy_logprob_bitwise_equal": exact_logprob,
        "same_policy_ratio_bitwise_one": exact_ratio,
        "network_recomputed_mu_bitwise_equal": network_exact_mu,
        "network_recomputed_sigma_bitwise_equal": network_exact_sigma,
        "network_recomputed_logprob_bitwise_equal": network_exact_logprob,
        "network_recomputed_ratio_bitwise_one": network_exact_ratio,
        "actor_critic_and_optimizers_unchanged": all(unchanged.values()),
        "actor_hash_unchanged": actor_hash_before == actor_hash_after,
        "network_recompute_did_not_change_actor_or_normalization": (
            actor_hash_after_network_recompute == actor_hash_after
        ),
        "optimizer_path_fail_closed": optimizer_rejected,
        "incoming_boundary_restored_after_gate": (
            restored_endpoints == [20, 20, 20, 20] and restored_states_equal
        ),
    }
    passed = all(checks.values())
    report = {
        "schema": "taco_pour_truncated_gaussian_integration_gate_v1",
        "status": "passed" if passed else "failed",
        "paper_faithful": False,
        "PPO_optimizer_steps": 0,
        "task_level_training_executed": False,
        "chunk_commit_written": False,
        "profile": profile,
        "inputs": {
            "offline_gate": {
                "path": _project_path(args.offline_gate),
                "sha256": _sha256(args.offline_gate),
            },
            "source_boundary": {
                "path": _project_path(args.source_boundary),
                "sha256": hashlib.sha256(source_raw).hexdigest(),
            },
            "paired_boundaries": {
                "path": _project_path(args.paired_boundaries),
                "sha256": hashlib.sha256(paired_raw).hexdigest(),
            },
            "curriculum_contract": {
                "path": _project_path(args.curriculum_contract),
                "sha256": _sha256(args.curriculum_contract),
            },
            "script": {
                "path": _project_path(Path(__file__)),
                "sha256": _sha256(Path(__file__)),
            },
        },
        "frozen_rollout": {
            "world_start_endpoints": starts,
            "worlds": 4,
            "horizon": 40,
            "samples": sample_count,
            "seed": 0,
            "optimizer_updates": 0,
        },
        "rollout_buffer": {
            "shapes": buffer_shapes,
            "all_actions_inside_state_bounds": actions_in_bounds,
            "same_policy_logprob_bitwise_equal": exact_logprob,
            "same_policy_ratio_max_abs_error": float((ratio - 1.0).abs().max().item()),
            "network_recomputed_mu_bitwise_equal": network_exact_mu,
            "network_recomputed_sigma_bitwise_equal": network_exact_sigma,
            "network_recomputed_logprob_bitwise_equal": network_exact_logprob,
            "network_recomputed_ratio_max_abs_error": float(
                (network_ratio - 1.0).abs().max().item()
            ),
        },
        "environment_action_semantics": {
            "official_clamp_changed_components": official_clamp_changes,
            "actual_ctrlrange_nonzero_lost_components": exact_lost_components,
            "actual_ctrlrange_lost_components_above_2e_7": lost_components,
            "maximum_abs_residual_lost_to_ctrlrange": max_lost,
            "reference_contract_audit": action_audit,
        },
        "immutability": {
            "actor_sha256_before": actor_hash_before,
            "actor_sha256_after": actor_hash_after,
            "actor_sha256_after_network_recompute": (
                actor_hash_after_network_recompute
            ),
            "state_equal": unchanged,
            "all_four_physics_states_restored_bitwise_to_incoming_boundary": (
                restored_states_equal
            ),
        },
        "training_trace": trace,
        "residual_action": residual_report,
        "checks": checks,
        "decision": {
            "engineering_gate_passed": passed,
            "optimizer_training_authorized": False,
            "task_level_PPO_authorized": False,
            "chunk_commit_authorized": False,
            "next_step_requires_explicit_user_authorization": True,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "samples": sample_count,
        "buffer_shapes": buffer_shapes,
        "official_clamp_changed_components": official_clamp_changes,
        "actual_ctrlrange_nonzero_lost_components": exact_lost_components,
        "actual_ctrlrange_lost_components_above_2e_7": lost_components,
        "maximum_abs_residual_lost_to_ctrlrange": max_lost,
        "checks": checks,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
