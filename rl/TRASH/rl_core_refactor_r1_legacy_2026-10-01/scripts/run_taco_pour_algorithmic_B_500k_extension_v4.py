#!/usr/bin/env python3
"""Continue one verified Candidate-B v4 trajectory from 100k to 500k."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
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
CONTRACT = ROOT / "configs/taco_pour_algorithmic_B_500k_extension_v4.yaml"
RUN_ROOT = (
    ROOT
    / "runs/taco_pour_algorithmic_reproduction_training_v4"
    / "B_budget_extension_500k"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(
        io.BytesIO(gzip.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )


def exact_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    cuda_states = states.get("torch_cuda_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def first_failure(terminated: list[bool], endpoints: list[int]) -> int | None:
    return next(
        (
            endpoint
            for endpoint, failed in zip(endpoints, terminated, strict=True)
            if failed
        ),
        None,
    )


def validation_summary(
    terminated: list[bool],
    endpoints: list[int],
    scores: list[float],
    position: list[float],
    rotation: list[float],
) -> dict[str, object]:
    failure = first_failure(terminated, endpoints)
    successful = 40 if failure is None else failure - endpoints[0]

    def endpoint_row(endpoint: int) -> dict[str, float]:
        index = endpoints.index(endpoint)
        position_error = float(position[index])
        rotation_error = float(rotation[index])
        return {
            "score": float(scores[index]),
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "position_squared_contribution": float((position_error / 0.12) ** 2),
            "rotation_squared_contribution": float((rotation_error / 1.5) ** 2),
        }

    result: dict[str, object] = {
        "successful_intervals": int(successful),
        "first_failure_endpoint": failure,
        "forty_of_forty": failure is None,
        "beats_Replay": successful > 30,
        "meaningful_refinement": successful >= 35,
        "endpoint40": endpoint_row(40),
        "endpoint50": endpoint_row(50),
        "endpoint60": endpoint_row(60),
    }
    if failure is not None:
        result["first_failure"] = endpoint_row(failure)
    return result


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("training metric segment is empty")

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))

    return {
        "actor_updates": len(rows),
        "mean_exact_post_update_KL": float(
            np.mean(
                [
                    float(row["exact_old_to_new_truncated_policy_KL"]["mean"])
                    for row in rows
                ]
            )
        ),
        "max_exact_post_update_KL": float(
            np.max(
                [
                    float(row["exact_old_to_new_truncated_policy_KL"]["max"])
                    for row in rows
                ]
            )
        ),
        "mean_ratio_outside_0p8_1p2_fraction": mean(
            "post_optimizer_ratio_outside_0p8_1p2_fraction"
        ),
        "mean_deterministic_effective_residual_RMS": mean(
            "deterministic_effective_residual_RMS"
        ),
        "mean_deterministic_action_bound_fraction": mean(
            "deterministic_action_bound_fraction"
        ),
        "mean_training_reward": mean("training_reward_mean"),
        "maximum_rollout_to_canonical_ratio_error": float(
            np.max(
                [
                    float(
                        row["pre_optimizer_identity"]
                        .get("rollout_vs_canonical", {})
                        .get(
                            "max_abs_ratio_minus_one",
                            row["pre_optimizer_identity"][
                                "ratio_max_abs_error_from_one"
                            ],
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "maximum_canonical_optimizer_ratio_error": float(
            np.max(
                [
                    float(
                        row["pre_optimizer_identity"]
                        .get("canonical_optimizer_identity", {})
                        .get(
                            "max_abs_ratio_minus_one",
                            row["pre_optimizer_identity"][
                                "maximum_abs_error_from_one"
                            ],
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "all_pre_optimizer_identity_checks_passed": all(
            bool(row["pre_optimizer_identity"]["numerical_identity_passed"])
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in (0, 1, 2):
        raise ValueError("Candidate-B 500k extension is frozen to seeds 0, 1, 2")

    contract = yaml.safe_load(args.contract.read_text())
    extension = contract.get("candidate_B_500k_extension", {})
    if (
        contract.get("schema")
        != "taco_pour_algorithmic_candidate_B_500k_extension_v4"
        or contract.get("status") != "authorized_candidate_B_500k_continuation"
        or contract.get("paper_faithful") is not False
        or extension.get("candidate") != "B"
        or extension.get("seeds") != [0, 1, 2]
        or int(extension.get("source_epoch", -1)) != 62
        or int(extension.get("target_epoch", -1)) != 313
        or int(extension.get("additional_physics_steps_per_seed", -1)) != 401_600
        or bool(extension.get("candidate_A_continuation_allowed", True))
        or bool(extension.get("intermediate_checkpoint_selection_allowed", True))
        or bool(extension.get("chunk_commit_allowed", True))
        or bool(extension.get("automatic_1m_allowed", True))
    ):
        raise ValueError("v4 Candidate-B 500k continuation is not authorized")
    if Path(contract["output_directory"]).resolve() != args.run_root.resolve():
        raise ValueError("500k extension output root differs from the contract")

    frozen_contracts: dict[str, Path] = {}
    for name, row in contract["frozen_contracts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"frozen contract changed: {name}")
        frozen_contracts[name] = path
    base = yaml.safe_load(frozen_contracts["base_v4"].read_text())
    seed_extension = yaml.safe_load(
        frozen_contracts["seed_robustness_v4"].read_text()
    )
    for key in ("frozen_task", "quality_gates", "inputs"):
        if seed_extension[key] != base[key]:
            raise ValueError(f"seed extension changed frozen v4 field {key}")
    if base["frozen_task"]["actor_learning_rate"] != 1.0e-4:
        raise ValueError("Candidate-B continuation changed actor learning rate")
    if base["frozen_task"]["actor_mini_epochs"] != 1:
        raise ValueError("Candidate-B continuation changed actor mini-epochs")
    if base["frozen_task"]["critic_mini_epochs"] != 4:
        raise ValueError("Candidate-B continuation changed critic mini-epochs")

    source = contract["source_100k"][f"seed{args.seed}"]
    source_paths = {
        name: Path(source[name]["path"])
        for name in ("report", "checkpoint", "validation")
    }
    for name, path in source_paths.items():
        if not path.is_file() or sha256(path) != source[name]["sha256"]:
            raise ValueError(f"seed {args.seed} source {name} changed")
    source_report = json.loads(source_paths["report"].read_text())
    source_payload = load_checkpoint(source_paths["checkpoint"])

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
    from video_to_spider.rl.algorithmic_benchmark import (
        build_checkpoint_payload,
        validate_checkpoint_payload,
        write_checkpoint,
    )
    from video_to_spider.rl.algorithmic_training import AlgorithmicBenchmarkPpoAgent
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    validate_checkpoint_payload(source_payload)
    expected_source = {
        "candidate": "B",
        "seed": args.seed,
        "simulation_physics_steps": 99_200,
        "simulation_control_intervals": 9_920,
        "agent_epoch": 62,
        "agent_frame": 9_920,
    }
    for name, expected in expected_source.items():
        if source_payload[name] != expected:
            raise ValueError(
                f"seed {args.seed} source checkpoint {name} changed: "
                f"{source_payload[name]} != {expected}"
            )
    source_summary = source_report["validations"]["62"]["summary"]
    expected_intervals = int(source["expected_successful_intervals"])
    expected_failure = int(source["expected_first_failure_endpoint"])
    if (
        int(source_summary["successful_intervals"]) != expected_intervals
        or int(source_summary["first_failure_endpoint"]) != expected_failure
    ):
        raise ValueError("source report does not match the frozen 100k outcome")

    paths = {name: Path(row["path"]) for name, row in base["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != base["inputs"][name]["sha256"]:
            raise ValueError(f"frozen training input changed: {name}")
    expected_implementation = {
        "state_feasible_distribution_sha256": (
            ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"
        ),
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": (
            ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py"
        ),
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training.py"
        ),
        "continuation_runner_sha256": Path(__file__).resolve(),
    }
    for name, path in expected_implementation.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"frozen implementation changed: {name}")

    milestones = {
        int(row["epoch"]): row for row in contract["fixed_milestones"]
    }
    expected_milestones = {
        62: (99_200, 9_920, "100k"),
        125: (200_000, 20_000, "200k"),
        188: (300_800, 30_080, "300k"),
        250: (400_000, 40_000, "400k"),
        313: (500_800, 50_080, "500k"),
    }
    observed_milestones = {
        epoch: (
            int(row["physics_steps"]),
            int(row["control_intervals"]),
            str(row["name"]),
        )
        for epoch, row in milestones.items()
    }
    if observed_milestones != expected_milestones:
        raise ValueError("Candidate-B fixed learning-curve milestones changed")

    objective = load_runtime_objective(
        paths["protocol_at_authorization"],
        paths["objective_profile"],
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol_at_authorization"],
        paths["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(
        paths["action_profile"]
    )
    distribution, distribution_report = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    distribution = replace(distribution, optimizer_training_authorized=True)
    gate_row = base["quality_gates"]["pre_optimizer_likelihood_identity"]
    identity_gate = LikelihoodIdentityGateSpec.float32_ulp_aware_v1()
    canonical_gate = CanonicalOldPolicyGateSpec(
        rollout_to_canonical_ratio_atol=float(
            gate_row["rollout_to_canonical_ratio_atol"]
        ),
        canonical_ratio_atol=float(gate_row["canonical_ratio_atol"]),
        semantic_identity_hard_fail=float(gate_row["semantic_identity_hard_fail"]),
    )
    canonical_gate.validate()
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    boundary = load_checkpoint(paths["source_boundary"])
    if (
        boundary.get("snapshot_schema")
        != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("continuation requires the exact endpoint-20 boundary")

    cpu_config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    cpu_reference = _load_reference(
        cpu_config.data_path, "cpu", expected_frequency=30
    )

    def make_world(*, asymmetric: bool, world_seed: int) -> MJWPVectorEnv:
        world = MJWPVectorEnv(
            cpu_config,
            cpu_reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=asymmetric,
                max_episode_length=len(cpu_reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=world_seed,
        )
        verify_runtime_model(
            world.env.model_cpu, initialization["validated_physics_contract"]
        )
        world.set_env_state(boundary)
        return world

    output = args.run_root / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_dir = output / "input_contracts"
    input_dir.mkdir()
    frozen_inputs = {
        "extension_contract": args.contract,
        "base_v4_contract": frozen_contracts["base_v4"],
        "seed_robustness_v4_contract": frozen_contracts[
            "seed_robustness_v4"
        ],
        "seed_robustness_comparison": frozen_contracts[
            "seed_robustness_comparison"
        ],
        "source_100k_report": source_paths["report"],
        "source_100k_validation": source_paths["validation"],
        **paths,
    }
    for name, path in frozen_inputs.items():
        suffix = "".join(path.suffixes) or ".bin"
        (input_dir / f"{name}{suffix}").write_bytes(path.read_bytes())

    training_env = IndependentMJWPTrainingEnv(
        [
            make_world(asymmetric=True, world_seed=args.seed + index)
            for index in range(4)
        ]
    )
    training_env.set_chunk_reset(start=20, end=60)
    training_env.enable_training_trace(output / "training_visitation")
    ppo_config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=313,
            learning_rate=1.0e-4,
            device="cpu",
            asymmetric_critic=_build_asymmetric_critic_config(160),
            actor_mini_epochs=1,
        ),
        clip_actions=False,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = AlgorithmicBenchmarkPpoAgent(
        experiment_dir=output / "ppo",
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=training_env,
        distribution_spec=distribution,
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
        audit_dir=output / "update_audit",
    )
    validation_world = make_world(asymmetric=False, world_seed=args.seed)
    validation_config = replace(
        _build_ppo_config(
            num_envs=1,
            horizon_length=40,
            seq_length=4,
            max_epochs=1,
            learning_rate=1.0e-4,
            device="cpu",
            asymmetric_critic=None,
            actor_mini_epochs=1,
        ),
        clip_actions=False,
    )
    validation_agent = StateFeasibleTruncatedGaussianPpoAgent(
        experiment_dir=output / "validation_agent",
        ppo_config=validation_config,
        network_config=_build_network_config(4),
        env=validation_world,
        distribution_spec=distribution,
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
    )

    checkpoints: dict[int, dict[str, object]] = {
        62: {
            **artifact(source_paths["checkpoint"]),
            "source_100k_checkpoint": True,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
    }
    validations: dict[str, dict[str, object]] = {}
    resume_audit: dict[str, object]
    trace = None
    update_audit = None
    training_completed = False
    started = time.time()

    def validate_payload(
        *, epoch: int, payload: dict[str, Any], output_path: Path
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        validation_agent.model.load_state_dict(payload["actor"], strict=True)
        validation_agent.set_eval()
        validation_agent.rnn_states = [
            state.to("cpu").zero_()
            for state in validation_agent.model.get_default_rnn_state()
        ]
        validation_world.set_env_state(boundary)
        endpoints: list[int] = []
        terminated: list[bool] = []
        scores: list[float] = []
        position: list[float] = []
        rotation: list[float] = []
        actions: list[np.ndarray] = []
        actor_mu: list[np.ndarray] = []
        actor_sigma: list[np.ndarray] = []
        for source_endpoint in range(20, 60):
            obs = validation_agent.obs_to_tensors(
                validation_world.current_observation()
            )
            result = validation_agent.get_deterministic_action_values(obs)
            validation_agent.rnn_states = result["rnn_states"]
            action = validation_agent.preprocess_actions(
                result["deterministic_actions"]
            )
            _, _, _, info = validation_world.step(action, auto_reset=False)
            endpoints.append(source_endpoint + 1)
            terminated.append(bool(info["terminated"][0]))
            scores.append(float(info["object_tracking_error"][0]))
            position.append(float(info["object_position_error"][0, 0]))
            rotation.append(float(info["object_rotation_error"][0, 0]))
            actions.append(np.asarray(action[0]).copy())
            actor_mu.append(result["mus"][0].detach().cpu().numpy().copy())
            actor_sigma.append(
                result["sigmas"][0].detach().cpu().numpy().copy()
            )
        arrays = {
            "endpoint": np.asarray(endpoints, dtype=np.int32),
            "terminated": np.asarray(terminated),
            "tracking_score": np.asarray(scores),
            "position_error": np.asarray(position),
            "rotation_error": np.asarray(rotation),
            "deterministic_action": np.asarray(actions),
            "actor_mu": np.asarray(actor_mu),
            "actor_sigma": np.asarray(actor_sigma),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        summary = validation_summary(
            terminated, endpoints, scores, position, rotation
        )
        sigma = arrays["actor_sigma"].astype(np.float64)
        summary["final_sigma"] = {
            "minimum": float(sigma.min()),
            "mean": float(sigma.mean()),
            "maximum": float(sigma.max()),
        }
        summary["validation_residual_RMS"] = float(
            np.sqrt(np.mean(np.square(arrays["deterministic_action"] * 0.05)))
        )
        summary["validation_action_bound_fraction"] = float(
            np.mean(np.isclose(np.abs(arrays["deterministic_action"]), 1.0))
        )
        row = {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": summary,
            "trajectory": artifact(output_path),
            "chunk_commit_written": False,
        }
        return row, arrays

    def save_milestone(epoch: int) -> tuple[dict[str, Any], dict[str, object]]:
        row = milestones[epoch]
        expected_physics = int(row["physics_steps"])
        expected_controls = int(row["control_intervals"])
        observed = (
            int(training_env.simulation_physics_steps),
            int(training_env.simulation_control_intervals),
        )
        if observed != (expected_physics, expected_controls):
            raise RuntimeError(
                f"milestone {epoch} counters disagree: {observed}"
            )
        payload = build_checkpoint_payload(
            agent,
            candidate="B",
            seed=args.seed,
            simulation_physics_steps=expected_physics,
            simulation_control_intervals=expected_controls,
            config_hashes={
                "extension_contract": sha256(args.contract),
                "source_100k_checkpoint": sha256(source_paths["checkpoint"]),
                "continuation_runner": sha256(Path(__file__)),
                "algorithmic_training_module": sha256(
                    ROOT / "src/video_to_spider/rl/algorithmic_training.py"
                ),
            },
        )
        checkpoint_path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
        checkpoint = write_checkpoint(checkpoint_path, payload)
        restored = load_checkpoint(checkpoint_path)
        validate_checkpoint_payload(restored)
        if not exact_equal(payload, restored):
            raise RuntimeError(f"milestone {epoch} checkpoint roundtrip differs")
        checkpoints[epoch] = {
            **checkpoint,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
        rng = deepcopy(payload["rng_states"])
        validation, _ = validate_payload(
            epoch=epoch,
            payload=restored,
            output_path=output / "validations" / f"epoch_{epoch:04d}.npz",
        )
        restore_rng_states(rng)
        validation["checkpoint"] = checkpoints[epoch]
        validations[str(epoch)] = validation
        return restored, validation

    try:
        agent.init_tensors()
        agent.model.load_state_dict(source_payload["actor"], strict=True)
        if agent.asymmetric_critic_net is None:
            raise RuntimeError("continuation requires the asymmetric critic")
        agent.asymmetric_critic_net.load_state_dict(
            source_payload["critic"], strict=True
        )
        agent.optimizer.load_state_dict(source_payload["actor_optimizer"])
        agent.asymmetric_critic_net.optimizer.load_state_dict(
            source_payload["critic_optimizer"]
        )
        agent._observation_normalization_version = int(
            source_payload["observation_normalization_version"]
        )
        agent.epoch_num = int(source_payload["agent_epoch"])
        agent.frame = int(source_payload["agent_frame"])
        agent.rnn_states = [
            state.to("cpu").clone()
            for state in source_payload["agent_rnn_states"]
        ]
        agent.obs = deepcopy(source_payload["agent_observation"])
        agent.dones = source_payload["agent_dones"].to("cpu").clone()
        agent.curr_frames = agent.batch_size_envs
        agent.mean_rewards = agent.last_mean_rewards = -100500
        training_env.set_env_state(source_payload["environment"])
        physics_per_world = int(source_payload["simulation_physics_steps"]) // 4
        controls_per_world = int(
            source_payload["simulation_control_intervals"]
        ) // 4
        if (
            physics_per_world * 4
            != int(source_payload["simulation_physics_steps"])
            or controls_per_world * 4
            != int(source_payload["simulation_control_intervals"])
        ):
            raise RuntimeError("source counters are not divisible across four worlds")
        for world in training_env.worlds:
            world.simulation_physics_steps = physics_per_world
            world.simulation_control_intervals = controls_per_world
        restore_rng_states(source_payload["rng_states"])

        recaptured = build_checkpoint_payload(
            agent,
            candidate="B",
            seed=args.seed,
            simulation_physics_steps=99_200,
            simulation_control_intervals=9_920,
            config_hashes=source_payload["config_hashes"],
        )
        resume_fields = [
            "actor",
            "critic",
            "actor_optimizer",
            "critic_optimizer",
            "observation_normalization_version",
            "simulation_physics_steps",
            "simulation_control_intervals",
            "agent_epoch",
            "agent_frame",
            "agent_rnn_states",
            "agent_observation",
            "agent_dones",
            "environment",
            "rng_states",
        ]
        resume_field_equality = {
            name: exact_equal(source_payload[name], recaptured[name])
            for name in resume_fields
        }
        if not all(resume_field_equality.values()):
            raise RuntimeError(
                f"seed {args.seed} checkpoint restore differs: "
                f"{resume_field_equality}"
            )

        pre_rng = deepcopy(source_payload["rng_states"])
        prevalidation, pre_arrays = validate_payload(
            epoch=62,
            payload=source_payload,
            output_path=output
            / "precontinuation_validation"
            / "epoch_0062.npz",
        )
        restore_rng_states(pre_rng)
        with np.load(source_paths["validation"]) as frozen:
            source_validation_exact = {
                name: name in pre_arrays
                and exact_equal(pre_arrays[name], frozen[name])
                for name in frozen.files
            }
            source_validation_keyset_equal = set(frozen.files) == set(
                pre_arrays
            )
        summary = prevalidation["summary"]
        source_outcome_exact = (
            int(summary["successful_intervals"]) == expected_intervals
            and int(summary["first_failure_endpoint"]) == expected_failure
        )
        resume_audit = {
            "schema": "taco_pour_candidate_B_500k_resume_preflight_v4",
            "status": "passed_before_new_optimizer_update",
            "seed": args.seed,
            "new_optimizer_updates": 0,
            "source_checkpoint": artifact(source_paths["checkpoint"]),
            "source_report": artifact(source_paths["report"]),
            "resume_field_equality": resume_field_equality,
            "source_validation_keyset_equal": source_validation_keyset_equal,
            "source_validation_array_equality": source_validation_exact,
            "source_validation_all_arrays_exact": (
                source_validation_keyset_equal
                and all(source_validation_exact.values())
            ),
            "source_outcome_exact": source_outcome_exact,
            "summary": summary,
            "trajectory": prevalidation["trajectory"],
        }
        if not (
            resume_audit["source_validation_all_arrays_exact"]
            and resume_audit["source_outcome_exact"]
        ):
            raise RuntimeError(
                f"seed {args.seed} failed exact 100k deterministic replay"
            )
        resume_dir = output / "precontinuation_validation"
        (resume_dir / "report.json").write_text(
            json.dumps(resume_audit, indent=2) + "\n"
        )
        validations["62"] = {
            **prevalidation,
            "source_100k_checkpoint": checkpoints[62],
            "exact_reproduction_before_new_optimizer_update": True,
        }

        while int(agent.epoch_num) < 313:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in milestones and epoch > 62:
                _, validation = save_milestone(epoch)
                print(
                    json.dumps(
                        {
                            "seed": args.seed,
                            "epoch": epoch,
                            "physics_steps": training_env.simulation_physics_steps,
                            "validation": validation["summary"],
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
            elif epoch % 10 == 0:
                print(
                    json.dumps(
                        {
                            "seed": args.seed,
                            "epoch": epoch,
                            "physics_steps": training_env.simulation_physics_steps,
                            "elapsed_seconds": time.time() - started,
                            "latest_update": agent.update_reports[-1],
                        }
                    ),
                    flush=True,
                )
        training_completed = True
        trace = training_env.finalize_training_trace(completed=True)
        update_audit = agent.finalize_algorithmic_audit()
    except BaseException as error:
        try:
            training_env.finalize_training_trace(completed=False)
        except Exception as trace_error:
            error.add_note(
                f"training trace finalization also failed: {trace_error}"
            )
        raise
    finally:
        if agent.writer is not None:
            agent.writer.close()
        if validation_agent.writer is not None:
            validation_agent.writer.close()

    if not training_completed or trace is None or update_audit is None:
        raise RuntimeError("Candidate-B continuation did not complete 500k")

    source_manifest_path = Path(source_report["update_audit"]["path"])
    if (
        not source_manifest_path.is_file()
        or sha256(source_manifest_path) != source_report["update_audit"]["sha256"]
    ):
        raise RuntimeError("source 100k update audit manifest changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_updates = [
        row["update"] for row in source_manifest["epoch_reports"]
    ]
    if len(source_updates) != 62 or len(agent.update_reports) != 251:
        raise RuntimeError("continuation update count does not equal 62 + 251")
    all_updates = source_updates + agent.update_reports
    if [int(row["epoch"]) for row in all_updates] != list(range(1, 314)):
        raise RuntimeError("combined update evidence is not epoch-contiguous")

    previous_epoch = 0
    learning_curve: dict[str, dict[str, object]] = {}
    for epoch in (62, 125, 188, 250, 313):
        cumulative = all_updates[:epoch]
        segment = all_updates[previous_epoch:epoch]
        learning_curve[str(epoch)] = {
            "epoch": epoch,
            "physics_steps": int(milestones[epoch]["physics_steps"]),
            "cumulative": metric_summary(cumulative),
            "since_previous_milestone": {
                "start_epoch_exclusive": previous_epoch,
                "end_epoch_inclusive": epoch,
                **metric_summary(segment),
            },
        }
        previous_epoch = epoch

    report = {
        "schema": "taco_pour_algorithmic_candidate_B_500k_continuation_v4",
        "status": "completed_fixed_500k_no_chunk_commit",
        "classification": (
            "local_candidate_B_fixed_budget_continuation_from_verified_100k"
        ),
        "paper_faithful": False,
        "candidate": "B",
        "seed": args.seed,
        "backend": "CPU_MuJoCo_Warp_training_and_validation",
        "source_epoch": 62,
        "target_epoch": 313,
        "continuation_from_verified_100k_checkpoint": True,
        "fresh_restart": False,
        "historical_or_cross_run_warm_start": False,
        "resume_preflight": resume_audit,
        "training": {
            "worlds": 4,
            "source_physics_steps": 99_200,
            "additional_physics_steps": 401_600,
            "total_physics_steps": training_env.simulation_physics_steps,
            "total_control_intervals": training_env.simulation_control_intervals,
            "source_actor_updates": 62,
            "additional_actor_updates": len(agent.update_reports),
            "total_actor_updates": 313,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
            "wall_seconds": time.time() - started,
        },
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "validations": validations,
        "learning_curve": learning_curve,
        "continuation_training_visitation": trace,
        "continuation_update_audit": update_audit,
        "source_100k_update_audit": artifact(source_manifest_path),
        "candidate_A_continued": False,
        "intermediate_checkpoint_selection_executed": False,
        "hyperparameter_change_executed": False,
        "automatic_1m_executed": False,
        "chunk_commit_written": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "seed": args.seed,
                "validations": {
                    epoch: row["summary"]
                    for epoch, row in validations.items()
                },
                "report": artifact(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
