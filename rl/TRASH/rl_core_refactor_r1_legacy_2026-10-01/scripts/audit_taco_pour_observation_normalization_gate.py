#!/usr/bin/env python3
"""No-learning gate for frozen observation normalization in Pour PPO."""

from __future__ import annotations

import argparse
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
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def maximum_difference(left, right) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


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
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_observation_normalization_gate_v1",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--commit-observation-stats-after-epoch", action="store_true"
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

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
    from video_to_spider.rl.replay_rl import MJWPIndependentTrainingBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    class GateAgent(StateFeasibleTruncatedGaussianPpoAgent):
        def __init__(self, *agent_args, **agent_kwargs):
            super().__init__(*agent_args, **agent_kwargs)
            self.repeat_forward = None
            self.actor_update_calls = 0
            self.critic_raw_before = None
            self.critic_probe_states = None

        def _critic_raw_value(self, states):
            critic = self.asymmetric_critic_net
            normalized = critic.model.norm_obs(states, update_stats=False)
            with torch.no_grad():
                value, _ = critic.model.a2c_network({
                    "obs": normalized,
                    "rnn_states": None,
                })
            return value

        def prepare_dataset(self, batch_dict):
            states = self.asymmetric_critic_net._preproc_obs(batch_dict["states"])
            self.critic_probe_states = states.detach().clone()
            self.critic_raw_before = self._critic_raw_value(states).detach().clone()
            super().prepare_dataset(batch_dict)

        def train_actor_critic(self, input_dict):
            if self.actor_update_calls == 0:
                normalization_before = self._normalization_pair()
                outputs = [self.evaluate_ppo_distribution(input_dict) for _ in range(3)]
                normalization_after = self._normalization_pair()
                ratio = torch.exp(
                    input_dict["old_logp_actions"] - outputs[0]["neglogp"]
                )
                self.repeat_forward = {
                    "normalization_before": normalization_before,
                    "normalization_after": normalization_after,
                    "mu_max_abs_difference": maximum_difference(
                        outputs[0]["mu"], outputs[1]["mu"]
                    ),
                    "sigma_max_abs_difference": maximum_difference(
                        outputs[0]["sigma"], outputs[1]["sigma"]
                    ),
                    "neglogp_max_abs_difference": maximum_difference(
                        outputs[0]["neglogp"], outputs[2]["neglogp"]
                    ),
                    "value_max_abs_difference": maximum_difference(
                        outputs[0]["values"], outputs[2]["values"]
                    ),
                    "pre_optimizer_ratio_max_abs_error_from_one": float(
                        (ratio - 1.0).abs().max().detach().cpu()
                    ),
                    "samples": int(ratio.numel()),
                }
            self.actor_update_calls += 1
            return super().train_actor_critic(input_dict)

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(args.action_profile)
    base_spec, distribution = load_truncated_gaussian_profile(
        args.distribution_profile
    )
    spec = replace(base_spec, optimizer_training_authorized=True)
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    source_bytes = args.source_boundary.read_bytes()
    source = torch.load(
        io.BytesIO(gzip.decompress(source_bytes)),
        map_location="cpu",
        weights_only=False,
    )
    if (
        source.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(source["time_indices"]).tolist() != [20]
    ):
        raise ValueError("normalization gate requires corrected endpoint-20 boundary")

    def make_world():
        config = _load_ego_config(str(args.config), "cuda:0")
        reference = _load_reference(
            config.data_path, "cuda:0", expected_frequency=30
        )
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
    env = IndependentMJWPTrainingEnv([make_world() for _ in range(4)])
    backend = MJWPIndependentTrainingBackend(env)
    backend.verify_restored_snapshot(source)
    env.set_chunk_reset(start=20, end=60)
    critic_config = replace(
        _build_asymmetric_critic_config(160), learning_rate=0.0
    )
    ppo_config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=args.epochs,
            learning_rate=0.0,
            device="cuda:0",
            asymmetric_critic=critic_config,
        ),
        clip_actions=False,
        print_stats=False,
    )

    with tempfile.TemporaryDirectory(prefix="egoengine_obsnorm_gate_") as temp:
        agent = GateAgent(
            experiment_dir=Path(temp),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
            commit_observation_stats_after_epoch=(
                args.commit_observation_stats_after_epoch
            ),
        )
        agent.init_tensors()
        agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs
        actor_parameters_before = parameter_sha256(agent.model)
        critic_parameters_before = parameter_sha256(agent.asymmetric_critic_net.model)
        normalization_before = agent._normalization_pair()
        for _ in range(args.epochs):
            agent.update_epoch()
            agent.train_epoch()
        normalization_after = agent._normalization_pair()
        actor_parameters_after = parameter_sha256(agent.model)
        critic_parameters_after = parameter_sha256(agent.asymmetric_critic_net.model)
        normalization_audit = agent.observation_normalization_audit()
        if args.commit_observation_stats_after_epoch:
            final_ratio_error = None
            critic_raw_error = None
        else:
            final_input = agent.dataset[0]
            final_distribution = agent.evaluate_ppo_distribution(final_input)
            final_ratio_error = float((
                torch.exp(
                    final_input["old_logp_actions"]
                    - final_distribution["neglogp"]
                ) - 1.0
            ).abs().max().detach().cpu())
            critic_raw_after = agent._critic_raw_value(
                agent.critic_probe_states
            ).detach()
            critic_raw_error = maximum_difference(
                agent.critic_raw_before, critic_raw_after
            )
        if agent.writer is not None:
            agent.writer.close()

    repeat = agent.repeat_forward
    epochs = normalization_audit["epochs"]
    likelihood_checks = normalization_audit[
        "pre_optimizer_likelihood_identity_checks"
    ]
    critic_checks = normalization_audit[
        "pre_optimizer_critic_value_identity_checks"
    ]
    common_checks = {
        "complete_formal_rollout_has_160_samples": repeat["samples"] == 160,
        "expected_actor_optimizer_calls_executed_at_lr_zero": (
            agent.actor_update_calls == 4 * args.epochs
        ),
        "actor_parameters_bitwise_unchanged": (
            actor_parameters_before == actor_parameters_after
        ),
        "critic_parameters_bitwise_unchanged": (
            critic_parameters_before == critic_parameters_after
        ),
        "repeat_forward_does_not_change_rms": (
            repeat["normalization_before"] == repeat["normalization_after"]
        ),
        "repeat_forward_mu_is_bitwise_stable": (
            repeat["mu_max_abs_difference"] == 0.0
        ),
        "repeat_forward_sigma_is_bitwise_stable": (
            repeat["sigma_max_abs_difference"] == 0.0
        ),
        "repeat_forward_logprob_is_bitwise_stable": (
            repeat["neglogp_max_abs_difference"] == 0.0
        ),
        "repeat_forward_value_is_bitwise_stable": (
            repeat["value_max_abs_difference"] == 0.0
        ),
        "ratio_is_one_before_first_optimizer_call": (
            repeat["pre_optimizer_ratio_max_abs_error_from_one"] <= 5.0e-6
        ),
        "every_epoch_first_ratio_is_one": (
            len(likelihood_checks) == args.epochs
            and all(
                row["maximum_abs_error_from_one"] <= row["tolerance"]
                for row in likelihood_checks
            )
        ),
        "every_epoch_critic_value_recomputation_passed": (
            len(critic_checks) == args.epochs
            and all(
                row["maximum_abs_error"] <= row["tolerance"]
                for row in critic_checks
            )
        ),
    }
    if args.commit_observation_stats_after_epoch:
        mode_checks = {
            "normalization_version_advanced_once_per_epoch": (
                normalization_audit["current_version"] == args.epochs
            ),
            "rms_changed_only_at_each_epoch_tail": (
                len(epochs) == args.epochs
                and all(
                    row["statistics_committed_after_updates"] is True
                    and row["before"] == row["frozen_before_commit"]
                    and row["frozen_before_commit"] != row["after"]
                    for row in epochs
                )
            ),
            "next_rollout_uses_previous_committed_version": (
                all(
                    epochs[index]["before"] == epochs[index - 1]["after"]
                    and epochs[index]["version_used_for_rollout_and_updates"]
                    == index
                    for index in range(1, len(epochs))
                )
                and epochs[0]["version_used_for_rollout_and_updates"] == 0
            ),
            "actor_and_critic_rms_changed_after_commits": (
                normalization_before != normalization_after
            ),
        }
    else:
        mode_checks = {
            "actor_and_critic_observation_rms_unchanged": (
                normalization_before == normalization_after
            ),
            "ratio_is_one_after_full_lr_zero_dry_run": (
                final_ratio_error is not None and final_ratio_error <= 5.0e-6
            ),
            "critic_raw_output_is_stable_after_lr_zero_dry_run": (
                critic_raw_error is not None and critic_raw_error <= 2.0e-5
            ),
            "normalization_version_did_not_advance_in_gate": (
                normalization_audit["current_version"] == 0
            ),
        }
    checks = {**common_checks, **mode_checks}
    report = {
        "schema": (
            "taco_pour_observation_normalization_commit_gate_v1"
            if args.commit_observation_stats_after_epoch
            else "taco_pour_observation_normalization_likelihood_gate_v1"
        ),
        "status": "passed" if all(checks.values()) else "failed",
        "paper_faithful": False,
        "task_level_training_executed": False,
        "chunk_commit_written": False,
        "checkpoint_written": False,
        "purpose": (
            "Verify that rollout old likelihoods and every PPO recomputation use "
            "one immutable actor/critic observation-normalization snapshot."
        ),
        "inputs": {
            "simulator_config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "initialization_report": {"path": str(args.initialization_report.resolve()), "sha256": sha256(args.initialization_report)},
            "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256(args.protocol)},
            "objective_profile": {"path": str(args.objective_profile.resolve()), "sha256": sha256(args.objective_profile)},
            "observation_profile": {"path": str(args.observation_profile.resolve()), "sha256": sha256(args.observation_profile)},
            "action_profile": {"path": str(args.action_profile.resolve()), "sha256": sha256(args.action_profile)},
            "distribution_profile": {"path": str(args.distribution_profile.resolve()), "sha256": sha256(args.distribution_profile)},
            "source_boundary": {"path": str(args.source_boundary.resolve()), "sha256": sha256(args.source_boundary)},
        },
        "dry_run": {
            "device": "cuda:0",
            "worlds": 4,
            "horizon": 40,
            "epochs": args.epochs,
            "samples_per_epoch": 160,
            "samples": 160 * args.epochs,
            "actor_learning_rate": 0.0,
            "critic_learning_rate": 0.0,
            "mini_epochs": 4,
            "observation_stats_commit_enabled": (
                args.commit_observation_stats_after_epoch
            ),
            "distribution_profile": distribution,
        },
        "repeat_forward": repeat,
        "final_ratio_max_abs_error_from_one": final_ratio_error,
        "critic_raw_output_max_abs_error": critic_raw_error,
        "normalization": normalization_audit,
        "checks": checks,
        "decision": (
            "The implementation gate passed; this no-training report does not "
            "itself authorize a fresh task-level PPO."
            if all(checks.values())
            else "Fresh task-level PPO is forbidden because this gate failed."
        ),
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "pre_optimizer_ratio_max_abs_error": repeat[
            "pre_optimizer_ratio_max_abs_error_from_one"
        ],
        "post_dryrun_ratio_max_abs_error": final_ratio_error,
        "critic_raw_output_max_abs_error": critic_raw_error,
        "failed_checks": [name for name, value in checks.items() if not value],
    }, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
