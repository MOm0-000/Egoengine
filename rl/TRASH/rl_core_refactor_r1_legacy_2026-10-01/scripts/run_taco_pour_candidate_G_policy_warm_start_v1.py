#!/usr/bin/env python3
"""Run Candidate G: donor actor plus causally rebuilt endpoint-40 memory."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]

from run_taco_pour_algorithmic_candidate_C_v5 import (  # noqa: E402
    MILESTONES,
    artifact,
    exact_equal,
    load_checkpoint,
    restore_rng_states,
    sha256,
)
from run_taco_pour_algorithmic_candidate_D_v6 import metric_summary  # noqa: E402
from run_taco_pour_candidate_D_chunk_local_v1 import _validation_summary  # noqa: E402


CONTRACT = ROOT / "configs/taco_pour_candidate_G_policy_warm_start_v1.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_candidate_G_policy_warm_start_v1"


def _tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes()
    return np.asarray(left).tobytes() == np.asarray(right).tobytes()


def _state_hash(values: Any) -> str:
    from video_to_spider.rl.state_feasible_truncated_gaussian import _tensor_tree_sha256

    return _tensor_tree_sha256(values)


def _named_parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_contract(path: Path, run_root: Path) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text())
    if (
        contract.get("schema") != "taco_pour_candidate_G_policy_warm_start_v1"
        or contract.get("status")
        != "authorized_fixed_actor_only_warm_start_400k_per_seed"
        or contract.get("paper_faithful") is not False
        or contract.get("execution_authorized_now") is not True
        or contract.get("chunk_commit_authorized") is not False
        or contract.get("optimization", {}).get("seeds") != [0, 1, 2]
        or contract.get("warm_start", {}).get("donor")
        != {"candidate": "D", "source_endpoint": 20, "seed": 2, "epoch": 125}
        or Path(contract["output_directory"]).resolve() != run_root.resolve()
    ):
        raise ValueError("Candidate-G authorization contract changed")
    expected = [
        {"epoch": epoch, "physics_steps": physics, "control_intervals": controls}
        for epoch, (_, physics, controls) in MILESTONES.items()
    ]
    if contract.get("budget", {}).get("milestones") != expected:
        raise ValueError("Candidate-G milestones changed")
    return contract


def _verify_inputs(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Candidate-G input changed: {name}")
    return inputs


def _verify_implementation(contract: dict[str, Any]) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "policy_warm_start": ROOT / "src/video_to_spider/rl/policy_warm_start.py",
        "state_feasible_distribution": ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
        "bounded_mean_training": ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py",
        "curriculum_reset": ROOT / "src/video_to_spider/rl/curriculum_reset.py",
        "MJWP_environment": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "training_trace": ROOT / "src/video_to_spider/rl/training_trace.py",
        "official_PPO_agent": ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py",
        "PPO_config_builder": ROOT / "scripts/run_mjwp_ppo.py",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    frozen = contract.get("implementation_contract", {})
    if frozen.get("status") != "hashes_frozen_before_sampling":
        raise ValueError("Candidate-G implementation hashes are not frozen")
    if frozen.get("hashes") != observed:
        raise ValueError("Candidate-G implementation changed after hash freeze")
    return observed


class Runtime:
    def __init__(self, contract: dict[str, Any], inputs: dict[str, Path]) -> None:
        from run_mjwp_ppo import _load_ego_config, _load_reference
        from run_taco_replay_rl import load_accepted_initialization
        from video_to_spider.rl.action_contract import load_residual_action_profile
        from video_to_spider.rl.algorithmic_training_v6 import load_support_anchored_profile
        from video_to_spider.rl.objective_contract import load_runtime_objective
        from video_to_spider.rl.observation_contract import load_runtime_observation

        self.contract = contract
        self.inputs = inputs
        self.objective = load_runtime_objective(
            inputs["protocol_at_authorization"],
            inputs["objective_profile"],
            tracking_variant="tool_only",
            require_run_ready=False,
        )
        self.observation = load_runtime_observation(
            inputs["protocol_at_authorization"],
            inputs["observation_profile"],
            require_run_ready=False,
        )
        self.residual, self.residual_report = load_residual_action_profile(
            inputs["action_profile"]
        )
        self.distribution, self.distribution_report = load_support_anchored_profile(
            inputs["bounded_mean_distribution_profile"]
        )
        _, self.initialization = load_accepted_initialization(
            inputs["initialization_report"], inputs["simulator_config"]
        )
        self.boundary = load_checkpoint(inputs["source_boundary"])
        self.context = load_checkpoint(inputs["boundary_context"])
        self.donor = load_checkpoint(inputs["donor_checkpoint"])
        self.cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
        self.reference = _load_reference(
            self.cpu_config.data_path, "cpu", expected_frequency=30
        )

    def make_world(self, *, asymmetric: bool, seed: int):
        from run_mjwp_ppo import MJWPVectorEnv, MJWPVectorEnvConfig
        from run_taco_replay_rl import verify_runtime_model

        world = MJWPVectorEnv(
            self.cpu_config,
            self.reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=asymmetric,
                max_episode_length=len(self.reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=self.objective,
                observation=self.observation,
                residual=self.residual,
            ),
            seed=seed,
        )
        verify_runtime_model(
            world.env.model_cpu,
            self.initialization["validated_physics_contract"],
        )
        world.set_env_state(self.boundary)
        world.set_chunk_reset(start=40, end=80)
        return world

    def ppo_config(self, *, worlds: int, lr: float, critic_lr: float, freeze_critic: bool = False):
        from run_mjwp_ppo import _build_asymmetric_critic_config, _build_ppo_config

        critic = replace(
            _build_asymmetric_critic_config(40 * worlds),
            learning_rate=critic_lr,
            freeze_critic=freeze_critic,
        )
        return replace(
            _build_ppo_config(
                num_envs=worlds,
                horizon_length=40,
                seq_length=4,
                max_epochs=250,
                learning_rate=lr,
                device="cpu",
                asymmetric_critic=critic,
                actor_mini_epochs=1,
            ),
            clip_actions=False,
            bounds_loss_coef=0.0,
            bound_loss_type="regularisation",
            lr_schedule=None,
            schedule_type="legacy",
            kl_threshold=None,
        )

    def make_agent(
        self,
        *,
        seed: int,
        output: Path,
        training: bool,
        trace: bool,
        lr: float = 1.0e-4,
        critic_lr: float = 5.0e-5,
        freeze_critic: bool = False,
        candidate_g: bool = True,
    ):
        from run_mjwp_ppo import _build_network_config
        from video_to_spider.rl.algorithmic_training_v6 import SupportAnchoredBoundedMeanPpoAgent
        from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
        from video_to_spider.rl.policy_warm_start import (
            CandidateGWarmStartPpoAgent,
            inherit_actor_only,
        )
        from video_to_spider.rl.state_feasible_truncated_gaussian import (
            CanonicalOldPolicyGateSpec,
            LikelihoodIdentityGateSpec,
        )

        worlds = 4 if training else 1
        physical = [
            self.make_world(asymmetric=training, seed=seed + index)
            for index in range(worlds)
        ]
        env = IndependentMJWPTrainingEnv(physical) if training else physical[0]
        if training:
            env.set_chunk_reset(start=40, end=80)
            env.enable_boundary_recurrent_context(
                self.context, start_endpoint=40, window_end_endpoint=80
            )
            if trace:
                env.enable_training_trace(
                    output / "training_visitation",
                    include_world_index=True,
                    run_id=f"candidate_G_seed_{seed}",
                    training_seed=seed,
                )
        cls = CandidateGWarmStartPpoAgent if candidate_g else SupportAnchoredBoundedMeanPpoAgent
        kwargs: dict[str, Any] = {}
        if candidate_g:
            kwargs.update(run_id=f"candidate_G_seed_{seed}", training_seed=seed)
        agent = cls(
            experiment_dir=output / "ppo",
            ppo_config=self.ppo_config(
                worlds=worlds, lr=lr, critic_lr=critic_lr,
                freeze_critic=freeze_critic,
            ),
            network_config=_build_network_config(worlds),
            env=env,
            distribution_spec=self.distribution,
            likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
            canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
            audit_dir=output / "update_audit",
            **kwargs,
        )
        inheritance = inherit_actor_only(agent, self.donor)
        return env, agent, inheritance


def _validation_arrays(runtime: Runtime, payload: dict[str, Any], *, seed: int) -> dict[str, np.ndarray]:
    from run_mjwp_ppo import _build_network_config
    from video_to_spider.rl.algorithmic_training_v6 import SupportAnchoredBoundedMeanPpoAgent
    from video_to_spider.rl.curriculum_reset import refresh_boundary_rnn_for_actor
    from video_to_spider.rl.policy_warm_start import inherit_actor_only
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
    )

    world = runtime.make_world(asymmetric=False, seed=seed)
    agent = SupportAnchoredBoundedMeanPpoAgent(
        experiment_dir=Path(payload["_validation_dir"]),
        ppo_config=replace(runtime.ppo_config(worlds=1, lr=1.0e-4, critic_lr=5.0e-5), asymmetric_critic=None),
        network_config=_build_network_config(1),
        env=world,
        distribution_spec=runtime.distribution,
        likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
        canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
        audit_dir=Path(payload["_validation_dir"]) / "audit_unused",
    )
    agent.model.load_state_dict(payload["actor"], strict=True)
    agent._observation_normalization_version = int(payload["observation_normalization_version"])
    refreshed = refresh_boundary_rnn_for_actor(agent, runtime.context)
    agent.rnn_states = [state.to("cpu").clone() for state in refreshed["rnn_states"]]
    agent.set_eval()
    world.set_env_state(world._chunk_reset_state)
    rows: dict[str, list[Any]] = {name: [] for name in (
        "endpoint", "terminated", "tracking_score", "position_error",
        "rotation_error", "ctrl", "qpos", "qvel", "contact_flags",
        "deterministic_action", "raw_location", "bounded_mu", "actor_sigma",
        "action_low", "action_high", "reward", "tracking_reward",
        "contact_bonus", "lift_reward",
    )}
    for source in range(40, 80):
        obs = agent.obs_to_tensors(world.current_observation())
        result = agent.get_deterministic_action_values(obs)
        agent.rnn_states = result["rnn_states"]
        action = agent.preprocess_actions(result["deterministic_actions"])
        _, reward, _, info = world.step(action, auto_reset=False)
        rows["endpoint"].append(source + 1)
        rows["terminated"].append(bool(info["terminated"][0]))
        rows["tracking_score"].append(float(info["object_tracking_error"][0]))
        rows["position_error"].append(float(info["object_position_error"][0, 0]))
        rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
        rows["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
        rows["qpos"].append(world._mjwp.get_qpos(runtime.cpu_config, world.env)[0].detach().cpu().numpy().copy())
        rows["qvel"].append(world._mjwp.get_qvel(runtime.cpu_config, world.env)[0].detach().cpu().numpy().copy())
        rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
        rows["deterministic_action"].append(np.asarray(action[0]).copy())
        rows["raw_location"].append(result["raw_locations"][0].detach().cpu().numpy().copy())
        rows["bounded_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
        rows["actor_sigma"].append(result["sigmas"][0].detach().cpu().numpy().copy())
        rows["action_low"].append(result["action_lows"][0].detach().cpu().numpy().copy())
        rows["action_high"].append(result["action_highs"][0].detach().cpu().numpy().copy())
        rows["reward"].append(float(reward[0]))
        rows["tracking_reward"].append(float(info["aggregate_tracking_reward"][0]))
        rows["contact_bonus"].append(float(info["aggregate_contact_bonus"][0]))
        rows["lift_reward"].append(float(info["lift_reward"][0]))
    arrays = {
        name: np.asarray(values, dtype=np.int32) if name == "endpoint" else np.asarray(values)
        for name, values in rows.items()
    }
    arrays["actor_mu"] = arrays["bounded_mu"]
    if agent.writer is not None:
        agent.writer.close()
    return arrays


def _compare_step0(runtime: Runtime, arrays: dict[str, np.ndarray]) -> dict[str, bool]:
    with np.load(runtime.inputs["carryover_baseline"], allow_pickle=False) as baseline:
        select = baseline["endpoint"] >= 41
        checks = {}
        for name in (
            "endpoint", "terminated", "tracking_score", "position_error",
            "rotation_error", "ctrl", "qpos", "qvel", "contact_flags",
            "deterministic_action", "raw_location", "bounded_mu", "actor_sigma",
            "action_low", "action_high", "reward", "tracking_reward",
            "contact_bonus", "lift_reward", "actor_mu",
        ):
            checks[f"{name}_bitwise_equal"] = (
                arrays[name].tobytes() == baseline[name][select].tobytes()
            )
    summary = _validation_summary(arrays)
    checks["is_20_of_40_fail61"] = (
        summary["successful_intervals"] == 20
        and summary["first_failure_endpoint"] == 61
        and float(arrays["tracking_score"][19]) == 0.9899907112121582
        and float(arrays["tracking_score"][20]) == 1.0139968395233154
    )
    return checks


def run_preflight(runtime: Runtime, run_root: Path, implementations: dict[str, str]) -> dict[str, Any]:
    from video_to_spider.rl.curriculum_reset import refresh_boundary_rnn_for_actor
    from video_to_spider.rl.replay_rl import _snapshot_value_equal

    output = run_root / "preflight_G0"
    if output.exists():
        report = json.loads((output / "report.json").read_text())
        if report.get("status") != "passed_training_authorized":
            raise RuntimeError("existing Candidate-G preflight is not passing")
        return report
    output.mkdir(parents=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    env, agent, inheritance = runtime.make_agent(
        seed=0, output=output / "identity", training=True, trace=True,
        lr=0.0, critic_lr=0.0, freeze_critic=True, candidate_g=True,
    )
    agent.init_tensors()
    agent.last_mean_rewards = -100500
    agent.obs = agent.env_reset()
    agent.curr_frames = agent.batch_size_envs
    refreshed = refresh_boundary_rnn_for_actor(agent, runtime.context)
    donor_h40_equal = all(
        _tensor_equal(left, right)
        for left, right in zip(refreshed["rnn_states"], runtime.context["rnn_states"], strict=True)
    )

    # One isolated lr=0 optimizer pass exercises canonical likelihood identity.
    agent.epoch_num = 1
    agent.train_epoch()
    identity = agent._likelihood_identity_checks[-1]
    canonical_pass = bool(identity["passed"])
    if agent.writer is not None:
        agent.writer.close()

    # A separate no-optimizer shadow performs two rollout/RMS/context-refresh cycles.
    random.seed(101); np.random.seed(101); torch.manual_seed(101)
    shadow_env, shadow, _ = runtime.make_agent(
        seed=0, output=output / "shadow", training=True, trace=False,
        lr=0.0, critic_lr=0.0, freeze_critic=True, candidate_g=True,
    )
    shadow.init_tensors(); shadow.obs = shadow.env_reset(); shadow.curr_frames = shadow.batch_size_envs
    shadow_rows = []
    parameter_hash = _named_parameter_hash(shadow.model)
    for epoch in (1, 2):
        shadow.epoch_num = epoch
        shadow_env.set_train_info(shadow.frame, shadow)
        context_before = _state_hash(shadow_env.rollout_reset_rnn_states())
        with torch.no_grad():
            batch = shadow.play_steps_rnn()
        shadow.prepare_dataset(batch)
        if shadow.optimizer.state_dict()["state"]:
            raise RuntimeError("no-optimizer shadow unexpectedly initialized Adam state")
        shadow.model.update_obs_stats(shadow._rollout_actor_observations_for_rms)
        shadow._observation_normalization_version += 1
        shadow._rollout_actor_observations_for_rms = None
        shadow._rollout_critic_observations_for_rms = None
        shadow.dataset.update_values_dict(None)
        shadow_env._prepare_boundary_recurrent_context_epoch(shadow)
        context_after = _state_hash(shadow_env.rollout_reset_rnn_states())
        shadow_rows.append({
            "epoch": epoch,
            "context_before_RMS_commit": context_before,
            "context_after_RMS_commit_and_refresh": context_after,
            "normalization_version_after": shadow._observation_normalization_version,
            "optimizer_state_empty": shadow.optimizer.state_dict()["state"] == {},
        })
    shadow_parameters_unchanged = parameter_hash == _named_parameter_hash(shadow.model)
    if shadow.writer is not None:
        shadow.writer.close()

    # Exact step-zero continuation, validated by a fresh one-world agent.
    payload = deepcopy(runtime.donor)
    payload["_validation_dir"] = str(output / "step0_validation_agent")
    arrays = _validation_arrays(runtime, payload, seed=0)
    trajectory = output / "step0_continuation.npz"
    np.savez_compressed(trajectory, **arrays)
    step0 = _compare_step0(runtime, arrays)

    # Provider restores all four worlds identically and changes only selected RNN slots.
    env._prepare_boundary_recurrent_context_epoch(agent)
    world_states = env.get_env_states()
    four_world_equal = all(
        _snapshot_value_equal(world_states[0], state) for state in world_states[1:]
    )
    before = [state.clone() for state in agent.rnn_states]
    selected = [state.clone().add_(3.0) for state in before]
    agent.rnn_states = env.reset_rnn_states_after_done(selected, torch.tensor([[1], [3]]))
    reset = env.rollout_reset_rnn_states()
    partial_rnn = all(
        torch.equal(state[:, [1, 3]], target[:, [1, 3]])
        and torch.equal(state[:, [0, 2]], source[:, [0, 2]] + 3.0)
        for state, target, source in zip(agent.rnn_states, reset, before, strict=True)
    )
    physics_context_equal = _snapshot_value_equal(
        runtime.boundary, runtime.context["physics_state"]
    )
    checks = {
        "input_hashes_match": True,
        "implementation_hashes_frozen": True,
        "committed_and_context_physics_bitwise_equal": physics_context_equal,
        "donor_h40_bitwise_equal_after_prefix_reencoding": donor_h40_equal,
        "actor_only_inheritance": all(inheritance["checks"].values()),
        "step0_all_arrays_bitwise_equal": all(step0.values()),
        "four_world_physics_equal": four_world_equal,
        "partial_RNN_reset_selected_worlds_only": partial_rnn,
        "canonical_ratio_identity": canonical_pass,
        "two_shadow_epochs_no_optimizer": all(
            row["optimizer_state_empty"] for row in shadow_rows
        ),
        "shadow_actor_parameters_unchanged": shadow_parameters_unchanged,
    }
    report = {
        "schema": "taco_pour_candidate_G_preflight_G0_v1",
        "status": "passed_training_authorized" if all(checks.values()) else "failed_closed",
        "checks": checks,
        "inheritance": inheritance,
        "step0_checks": step0,
        "step0_summary": _validation_summary(arrays),
        "step0_trajectory": artifact(trajectory),
        "canonical_identity": identity,
        "shadow_RMS_context_refresh": shadow_rows,
        "implementation_hashes": implementations,
        "optimizer_steps_in_no_optimizer_shadow": 0,
        "chunk_commit_written": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed_training_authorized":
        raise RuntimeError(f"Candidate-G G0 failed: {checks}")
    return report


def train_seed(
    runtime: Runtime,
    contract: dict[str, Any],
    run_root: Path,
    seed: int,
    preflight: dict[str, Any],
    implementations: dict[str, str],
) -> dict[str, Any]:
    from video_to_spider.rl.algorithmic_benchmark import (
        build_checkpoint_payload,
        validate_checkpoint_payload,
        write_checkpoint,
    )

    output = run_root / "training" / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env, agent, inheritance = runtime.make_agent(
        seed=seed, output=output, training=True, trace=True,
    )
    agent.init_tensors(); agent.last_mean_rewards = -100500
    agent.obs = agent.env_reset(); agent.curr_frames = agent.batch_size_envs
    env._prepare_boundary_recurrent_context_epoch(agent)

    input_dir = output / "input_contracts"; input_dir.mkdir()
    frozen_files = {"candidate_G_contract": CONTRACT, **runtime.inputs}
    for name, path in frozen_files.items():
        suffix = "".join(path.suffixes) or ".bin"
        (input_dir / f"{name}{suffix}").write_bytes(path.read_bytes())

    config_hashes = {
        "candidate_G_contract": sha256(CONTRACT),
        **implementations,
        "source_boundary": sha256(runtime.inputs["source_boundary"]),
        "donor_checkpoint": sha256(runtime.inputs["donor_checkpoint"]),
        "boundary_context": sha256(runtime.inputs["boundary_context"]),
    }
    checkpoints: dict[int, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}

    def validate_payload(epoch: int, payload: dict[str, Any]) -> dict[str, Any]:
        saved_rng = deepcopy(payload["rng_states"])
        local = deepcopy(payload)
        local["_validation_dir"] = str(output / "validation_agents" / f"epoch_{epoch:04d}")
        arrays = _validation_arrays(runtime, local, seed=seed)
        restore_rng_states(saved_rng)
        path = output / "validations" / f"epoch_{epoch:04d}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        summary = _validation_summary(arrays)
        row: dict[str, Any] = {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": summary,
            "trajectory": artifact(path),
            "checkpoint": checkpoints[epoch],
            "RNN_context": "recomputed_from_sources20_through39_with_checkpoint_actor_RMS",
            "chunk_commit_written": False,
        }
        if epoch == 0:
            row["step0_checks"] = _compare_step0(runtime, arrays)
            row["all_step0_checks_passed"] = all(row["step0_checks"].values())
        if summary["forty_of_forty"]:
            second = deepcopy(payload)
            second["_validation_dir"] = str(
                output / "validation_agents" / f"epoch_{epoch:04d}_exact_replay"
            )
            second_arrays = _validation_arrays(runtime, second, seed=seed)
            exact = {
                name: arrays[name].tobytes() == second_arrays[name].tobytes()
                for name in arrays
            }
            second_path = output / "validations" / f"epoch_{epoch:04d}_exact_replay.npz"
            np.savez_compressed(second_path, **second_arrays)
            row["independent_exact_revalidation"] = {
                "all_arrays_bitwise_equal": all(exact.values()),
                "per_array": exact,
                "summary": _validation_summary(second_arrays),
                "trajectory": artifact(second_path),
            }
        validations[str(epoch)] = row
        return row

    def save_milestone(epoch: int) -> dict[str, Any]:
        _, physics, controls = MILESTONES[epoch]
        observed = (env.simulation_physics_steps, env.simulation_control_intervals)
        if observed != (physics, controls):
            raise RuntimeError(f"Candidate-G milestone {epoch} counters disagree: {observed}")
        payload = build_checkpoint_payload(
            agent,
            # The immutable benchmark serializer predates Candidate G and only
            # accepts its historical A/B enum.  Candidate D/F use the same
            # compatibility path: serialize the common payload, then bind the
            # actual candidate and lineage before hashing/writing it.
            candidate="B",
            seed=seed,
            simulation_physics_steps=physics,
            simulation_control_intervals=controls,
            config_hashes=config_hashes,
        )
        payload.update({
            "candidate": "G",
            "candidate_lineage": "source20_D_seed2_epoch125_actor_only_plus_reencoded_h40",
            "donor_candidate": "D",
            "donor_seed": 2,
            "donor_epoch": 125,
            "chunk_source_endpoint": 40,
            "boundary_context_source_endpoint": 20,
            "boundary_context_observation_count": 20,
            "chunk_commit_allowed": False,
        })
        path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
        checkpoint = write_checkpoint(path, payload)
        restored = load_checkpoint(path)
        validate_checkpoint_payload(restored)
        if not exact_equal(payload, restored):
            raise RuntimeError(f"Candidate-G checkpoint {epoch} roundtrip differs")
        checkpoints[epoch] = {
            **checkpoint,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
        row = validate_payload(epoch, restored)
        if epoch == 0 and not row["all_step0_checks_passed"]:
            raise RuntimeError("Candidate-G seed failed previous-policy identity")
        return restored

    started = time.time(); failure: BaseException | None = None
    success_epoch: int | None = None; trace = None; update_audit = None
    try:
        save_milestone(0)
        while int(agent.epoch_num) < 250:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in MILESTONES:
                save_milestone(epoch)
                row = validations[str(epoch)]
                exact = row.get("independent_exact_revalidation", {})
                if row["summary"]["forty_of_forty"] and exact.get("all_arrays_bitwise_equal"):
                    success_epoch = epoch
                    break
            if epoch == 1 or epoch % 10 == 0 or epoch in MILESTONES:
                print(json.dumps({
                    "candidate": "G", "seed": seed, "epoch": epoch,
                    "physics_steps": env.simulation_physics_steps,
                    "elapsed_seconds": time.time() - started,
                    "latest_update": agent.update_reports[-1],
                    "latest_validation": validations.get(str(epoch), {}).get("summary"),
                }), flush=True)
        trace = env.finalize_training_trace(completed=True)
        update_audit = agent.finalize_algorithmic_audit()
    except BaseException as error:
        failure = error
        try:
            trace = env.finalize_training_trace(completed=False)
        except Exception as trace_error:
            error.add_note(f"trace finalization also failed: {trace_error}")
    finally:
        if agent.writer is not None:
            agent.writer.close()

    common = {
        "schema": "taco_pour_candidate_G_policy_warm_start_training_v1",
        "paper_faithful": False,
        "candidate": "G",
        "seed": seed,
        "inheritance": inheritance,
        "preflight": {"path": str((run_root / "preflight_G0/report.json").resolve()), "status": preflight["status"]},
        "training": {
            "worlds": 4,
            "completed_actor_updates": len(agent.update_reports),
            "simulation_physics_steps": env.simulation_physics_steps,
            "simulation_control_intervals": env.simulation_control_intervals,
            "wall_seconds": time.time() - started,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
        },
        "validations": validations,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "training_visitation": trace,
        "update_audit": update_audit,
        "boundary_context_audit": env.boundary_recurrent_context_audit(),
        "residual_action": runtime.residual_report,
        "action_distribution": runtime.distribution_report,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_written": False,
    }
    if failure is not None:
        report = {
            **common,
            "status": "failed_closed_G0_or_runtime_contract",
            "exception": {
                "type": type(failure).__name__,
                "message": str(failure),
                "traceback": "".join(traceback.format_exception(failure)),
            },
            "eligible_for_promotion": False,
        }
        (output / "failure_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise failure
    if success_epoch is not None:
        report = {
            **common,
            "status": "strict_success_independently_revalidated_no_chunk_commit",
            "strict_success_epoch": success_epoch,
            "eligible_for_separate_promotion": True,
            "stop_all_unstarted_seeds": True,
        }
    else:
        report = {
            **common,
            "status": "completed_400k_without_strict_success_no_chunk_commit",
            "strict_success_epoch": None,
            "eligible_for_separate_promotion": False,
            "stop_all_unstarted_seeds": False,
        }
    report["learning_curve"] = {
        str(epoch): {
            "epoch": epoch,
            "physics_steps": MILESTONES[epoch][1],
            "cumulative": metric_summary(agent.update_reports[:epoch]),
            "validation": validations.get(str(epoch), {}).get("summary"),
        }
        for epoch in MILESTONES if epoch == 0 or epoch <= len(agent.update_reports)
    }
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "seed": seed,
        "strict_success_epoch": success_epoch,
        "validations": {key: row["summary"] for key, row in validations.items()},
        "report": artifact(path),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    contract = _load_contract(args.contract, args.run_root)
    inputs = _verify_inputs(contract)
    implementations = _verify_implementation(contract)
    runtime = Runtime(contract, inputs)
    preflight = run_preflight(runtime, args.run_root, implementations)
    if args.stage == "preflight":
        print(json.dumps(preflight, indent=2))
        return
    if args.seed not in contract["budget"]["seed_order"]:
        raise ValueError("training requires one of the frozen seeds 0, 1, 2")
    train_seed(runtime, contract, args.run_root, int(args.seed), preflight, implementations)


if __name__ == "__main__":
    main()
