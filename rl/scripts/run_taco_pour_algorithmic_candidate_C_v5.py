#!/usr/bin/env python3
"""Run one fresh 400k Candidate-C mean-regularization ablation seed."""

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
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v5.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v5"
MILESTONES = {
    0: ("0", 0, 0),
    62: ("100k", 99_200, 9_920),
    125: ("200k", 200_000, 20_000),
    188: ("300k", 300_800, 30_080),
    250: ("400k", 400_000, 40_000),
}


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
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def validation_summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    endpoints = arrays["endpoint"].astype(int).tolist()
    terminated = arrays["terminated"].astype(bool).tolist()
    failure = next(
        (endpoint for endpoint, done in zip(endpoints, terminated, strict=True) if done),
        None,
    )
    successful = 40 if failure is None else failure - endpoints[0]

    def endpoint_row(endpoint: int) -> dict[str, float]:
        index = endpoints.index(endpoint)
        position = float(arrays["position_error"][index])
        rotation = float(arrays["rotation_error"][index])
        return {
            "score": float(arrays["tracking_score"][index]),
            "position_error_m": position,
            "rotation_error_rad": rotation,
            "position_squared_contribution": float((position / 0.12) ** 2),
            "rotation_squared_contribution": float((rotation / 1.5) ** 2),
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
        "validation_residual_RMS": float(
            np.sqrt(np.mean(np.square(arrays["deterministic_action"] * 0.05)))
        ),
        "validation_action_bound_fraction": float(
            np.mean(np.isclose(np.abs(arrays["deterministic_action"]), 1.0))
        ),
        "final_sigma": {
            "minimum": float(arrays["actor_sigma"].min()),
            "mean": float(arrays["actor_sigma"].mean()),
            "maximum": float(arrays["actor_sigma"].max()),
        },
    }
    if failure is not None:
        result["first_failure"] = endpoint_row(failure)
    return result


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {"actor_updates": 0}

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    return {
        "actor_updates": len(rows),
        "mean_exact_post_update_KL": float(np.mean([
            float(row["exact_old_to_new_truncated_policy_KL"]["mean"])
            for row in rows
        ])),
        "max_exact_post_update_KL": float(np.max([
            float(row["exact_old_to_new_truncated_policy_KL"]["max"])
            for row in rows
        ])),
        "mean_ratio_outside_0p8_1p2_fraction": float(np.mean(values(
            "post_optimizer_ratio_outside_0p8_1p2_fraction"
        ))),
        "mean_deterministic_action_bound_fraction": float(np.mean(values(
            "deterministic_action_bound_fraction"
        ))),
        "latest_deterministic_action_bound_fraction": values(
            "deterministic_action_bound_fraction"
        )[-1],
        "mean_deterministic_effective_residual_RMS": float(np.mean(values(
            "deterministic_effective_residual_RMS"
        ))),
        "mean_training_reward": float(np.mean(values("training_reward_mean"))),
        "minimum_truncated_normalization_mass": float(np.min(values(
            "minimum_truncated_normalization_mass"
        ))),
        "minimum_p01_truncated_normalization_mass": float(np.min(values(
            "p01_truncated_normalization_mass"
        ))),
        "median_of_median_truncated_normalization_mass": float(np.median(values(
            "median_truncated_normalization_mass"
        ))),
        "mean_raw_mu_outside_support_fraction": float(np.mean(values(
            "raw_mu_outside_support_fraction"
        ))),
        "latest_raw_mu_outside_support_fraction": values(
            "raw_mu_outside_support_fraction"
        )[-1],
        "maximum_support_violation_in_sigma": float(np.max(values(
            "maximum_support_violation_in_sigma"
        ))),
        "maximum_p95_support_violation_in_sigma": float(np.max(values(
            "p95_support_violation_in_sigma"
        ))),
        "mean_unscaled_actor_mean_regularization": float(np.mean([
            float(row["actor_mean_regularization"][
                "unscaled_mean_sum_squared_mu"
            ]) for row in rows
        ])),
        "maximum_rollout_to_canonical_ratio_error": float(np.max([
            float(row["pre_optimizer_identity"]["rollout_vs_canonical"][
                "max_abs_ratio_minus_one"
            ]) for row in rows
        ])),
        "maximum_canonical_optimizer_ratio_error": float(np.max([
            float(row["pre_optimizer_identity"]["canonical_optimizer_identity"][
                "max_abs_ratio_minus_one"
            ]) for row in rows
        ])),
        "all_v5_identity_checks_passed": all(
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
        raise ValueError("Candidate C is frozen to seeds 0, 1, 2")

    contract = yaml.safe_load(args.contract.read_text())
    candidate = contract.get("candidate_C", {})
    if (
        contract.get("schema")
        != "taco_pour_algorithmic_reproduction_training_benchmark_v5"
        or contract.get("status") != "authorized_candidate_C_fresh_400k"
        or contract.get("paper_faithful") is not False
        or candidate.get("seeds") != [0, 1, 2]
        or candidate.get("initialization") != "replay_preserving"
        or float(candidate.get("bounds_loss_coef", -1)) != 0.005
        or candidate.get("bound_loss_type") != "regularisation"
        or bool(candidate.get("linear_LR_allowed", True))
        or bool(candidate.get("warm_start_allowed", True))
        or bool(candidate.get("chunk_commit_allowed", True))
    ):
        raise ValueError("v5 Candidate C is not authorized")
    if Path(contract["output_directory"]).resolve() != args.run_root.resolve():
        raise ValueError("Candidate-C output root differs from the contract")
    expected_milestones = [
        {"name": name, "epoch": epoch, "physics_steps": physics,
         "control_intervals": controls}
        for epoch, (name, physics, controls) in MILESTONES.items()
    ]
    if contract.get("fixed_milestones") != expected_milestones:
        raise ValueError("v5 milestones changed")

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
        replay_preserving_initialization,
        validate_checkpoint_payload,
        write_checkpoint,
    )
    from video_to_spider.rl.algorithmic_training_v5 import (
        MeanRegularizedAlgorithmicPpoAgent,
    )
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    frozen = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in frozen.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Candidate-C input changed: {name}")
    implementations = {
        "state_feasible_distribution_sha256": (
            ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"
        ),
        "v4_algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training.py"
        ),
        "v5_algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training_v5.py"
        ),
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": (
            ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py"
        ),
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "training_runner_sha256": Path(__file__).resolve(),
    }
    for name, path in implementations.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"Candidate-C implementation changed: {name}")

    base = yaml.safe_load(frozen["base_v4_contract"].read_text())
    for name in (
        "simulator_config", "initialization_report", "protocol_at_authorization",
        "objective_profile", "observation_profile", "action_profile",
        "distribution_profile", "source_boundary",
    ):
        if base["inputs"][name] != contract["inputs"][name]:
            raise ValueError(f"Candidate C changed frozen v4 input {name}")
    upstream = yaml.safe_load(frozen["H2S2R_LSTM_config"].read_text())
    if (
        float(upstream["ppo"]["bounds_loss_coef"]) != 0.005
        or upstream["ppo"]["bound_loss_type"] != "regularisation"
    ):
        raise ValueError("public H2S2R LSTM mean regularization provenance changed")

    objective = load_runtime_objective(
        frozen["protocol_at_authorization"], frozen["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        frozen["protocol_at_authorization"], frozen["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(
        frozen["action_profile"]
    )
    distribution, distribution_report = load_truncated_gaussian_profile(
        frozen["distribution_profile"]
    )
    distribution = replace(distribution, optimizer_training_authorized=True)
    identity_gate = LikelihoodIdentityGateSpec.float32_ulp_aware_v1()
    canonical_gate = CanonicalOldPolicyGateSpec.v1()
    _, initialization = load_accepted_initialization(
        frozen["initialization_report"], frozen["simulator_config"]
    )
    boundary = load_checkpoint(frozen["source_boundary"])
    if (
        boundary.get("snapshot_schema")
        != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("Candidate C requires the exact endpoint-20 boundary")

    cpu_config = _load_ego_config(str(frozen["simulator_config"]), "cpu")
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

    output = args.run_root / "training" / "candidate_C" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_dir = output / "input_contracts"
    input_dir.mkdir()
    for name, path in {"v5_contract": args.contract, **frozen}.items():
        suffix = "".join(path.suffixes) or ".bin"
        (input_dir / f"{name}{suffix}").write_bytes(path.read_bytes())

    training_env = IndependentMJWPTrainingEnv([
        make_world(asymmetric=True, world_seed=args.seed + index)
        for index in range(4)
    ])
    training_env.set_chunk_reset(start=20, end=60)
    training_env.enable_training_trace(output / "training_visitation")
    ppo_config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=250,
            learning_rate=1.0e-4,
            device="cpu",
            asymmetric_critic=_build_asymmetric_critic_config(160),
            actor_mini_epochs=1,
        ),
        clip_actions=False,
        bounds_loss_coef=0.005,
        bound_loss_type="regularisation",
        lr_schedule=None,
        schedule_type="legacy",
        kl_threshold=None,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = MeanRegularizedAlgorithmicPpoAgent(
        experiment_dir=output / "ppo",
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=training_env,
        distribution_spec=distribution,
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
        audit_dir=output / "update_audit",
    )
    initialization_audit = replay_preserving_initialization(
        agent, sigma_multiplier=0.25
    )
    config_hashes = {
        "v5_contract": sha256(args.contract),
        "training_runner": sha256(Path(__file__)),
        "v5_algorithmic_training": sha256(
            ROOT / "src/video_to_spider/rl/algorithmic_training_v5.py"
        ),
        "H2S2R_LSTM_config": sha256(frozen["H2S2R_LSTM_config"]),
    }

    validation_world = make_world(asymmetric=False, world_seed=args.seed)
    validation_config = replace(
        _build_ppo_config(
            num_envs=1, horizon_length=40, seq_length=4, max_epochs=1,
            learning_rate=1.0e-4, device="cpu", asymmetric_critic=None,
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

    checkpoints: dict[int, dict[str, object]] = {}
    validations: dict[str, dict[str, object]] = {}
    started = time.time()
    trace = None
    update_audit = None
    pretraining_validation: dict[str, object] | None = None

    def validate_payload(
        *, epoch: int, payload: dict[str, Any], step0: bool = False
    ) -> dict[str, object]:
        validation_agent.model.load_state_dict(payload["actor"], strict=True)
        validation_agent.set_eval()
        validation_agent.rnn_states = [
            state.to("cpu").zero_()
            for state in validation_agent.model.get_default_rnn_state()
        ]
        validation_world.set_env_state(boundary)
        rows: dict[str, list[Any]] = {
            "endpoint": [], "terminated": [], "tracking_score": [],
            "position_error": [], "rotation_error": [], "ctrl": [],
            "qpos": [], "qvel": [], "contact_flags": [],
            "deterministic_action": [], "actor_mu": [], "actor_sigma": [],
        }
        for source in range(20, 60):
            obs = validation_agent.obs_to_tensors(
                validation_world.current_observation()
            )
            result = validation_agent.get_deterministic_action_values(obs)
            validation_agent.rnn_states = result["rnn_states"]
            action = validation_agent.preprocess_actions(
                result["deterministic_actions"]
            )
            _, _, _, info = validation_world.step(action, auto_reset=False)
            rows["endpoint"].append(source + 1)
            rows["terminated"].append(bool(info["terminated"][0]))
            rows["tracking_score"].append(float(info["object_tracking_error"][0]))
            rows["position_error"].append(float(info["object_position_error"][0, 0]))
            rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            rows["ctrl"].append(
                validation_world._last_ctrl[0].detach().cpu().numpy().copy()
            )
            rows["qpos"].append(
                validation_world._mjwp.get_qpos(
                    validation_world.ego_cfg, validation_world.env
                )[0].detach().cpu().numpy().copy()
            )
            rows["qvel"].append(
                validation_world._mjwp.get_qvel(
                    validation_world.ego_cfg, validation_world.env
                )[0].detach().cpu().numpy().copy()
            )
            rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            rows["deterministic_action"].append(np.asarray(action[0]).copy())
            rows["actor_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
            rows["actor_sigma"].append(
                result["sigmas"][0].detach().cpu().numpy().copy()
            )
        arrays = {
            name: np.asarray(values, dtype=np.int32)
            if name == "endpoint" else np.asarray(values)
            for name, values in rows.items()
        }
        path = output / "validations" / f"epoch_{epoch:04d}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        summary = validation_summary(arrays)
        row: dict[str, object] = {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": summary,
            "trajectory": artifact(path),
            "checkpoint": checkpoints[epoch],
            "chunk_commit_written": False,
        }
        if step0:
            with np.load(frozen["Replay_step0_trajectory"]) as replay:
                checks = {
                    "mean_residual_within_tolerance": bool(
                        np.max(np.abs(arrays["actor_mu"])) <= 1.0e-7
                    ),
                    "deterministic_action_within_tolerance": bool(
                        np.max(np.abs(arrays["deterministic_action"])) <= 1.0e-7
                    ),
                    "command_bitwise_equals_Replay": bool(
                        arrays["ctrl"].tobytes() == replay["replay_ctrl"].tobytes()
                    ),
                    "qpos_bitwise_equals_Replay": bool(
                        arrays["qpos"].tobytes() == replay["replay_qpos"].tobytes()
                    ),
                    "qvel_bitwise_equals_Replay": bool(
                        arrays["qvel"].tobytes() == replay["replay_qvel"].tobytes()
                    ),
                    "score_bitwise_equals_Replay": bool(
                        arrays["tracking_score"].tobytes()
                        == replay["replay_score"].tobytes()
                    ),
                    "termination_bitwise_equals_Replay": bool(
                        arrays["terminated"].tobytes()
                        == replay["replay_terminated"].tobytes()
                    ),
                    "contact_flags_bitwise_equal_Replay": bool(
                        arrays["contact_flags"].tobytes()
                        == replay["replay_contact_flags"].tobytes()
                    ),
                    "is_30_of_40_fail51": bool(
                        summary["successful_intervals"] == 30
                        and summary["first_failure_endpoint"] == 51
                    ),
                }
            row["step0_checks"] = checks
            row["all_step0_checks_passed"] = all(checks.values())
        validations[str(epoch)] = row
        return row

    def save_milestone(epoch: int) -> dict[str, Any]:
        _, physics, controls = MILESTONES[epoch]
        observed = (
            int(training_env.simulation_physics_steps),
            int(training_env.simulation_control_intervals),
        )
        if observed != (physics, controls):
            raise RuntimeError(f"milestone {epoch} counters disagree: {observed}")
        payload = build_checkpoint_payload(
            agent,
            # The immutable v4 capture helper accepts only its historical A/B
            # labels. Candidate C inherits B's replay-preserving
            # initialization; relabel the complete payload before the v5
            # validator/writer sees it rather than broadening the v4 helper.
            candidate="B",
            seed=args.seed,
            simulation_physics_steps=physics,
            simulation_control_intervals=controls,
            config_hashes=config_hashes,
        )
        payload["candidate"] = "C"
        payload["candidate_lineage"] = "B_replay_preserving_plus_mean_regularization"
        path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
        checkpoint = write_checkpoint(path, payload)
        restored = load_checkpoint(path)
        validate_checkpoint_payload(restored)
        if not exact_equal(payload, restored):
            raise RuntimeError(f"milestone {epoch} checkpoint roundtrip differs")
        checkpoints[epoch] = {
            **checkpoint,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
        rng = deepcopy(payload["rng_states"])
        validation = validate_payload(epoch=epoch, payload=restored, step0=epoch == 0)
        restore_rng_states(rng)
        if epoch == 0 and not validation["all_step0_checks_passed"]:
            raise RuntimeError("Candidate C failed the per-seed Replay identity gate")
        return restored

    failure: BaseException | None = None
    try:
        agent.init_tensors()
        agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs
        save_milestone(0)
        pretraining_validation = validations["0"]
        while int(agent.epoch_num) < 250:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in MILESTONES:
                save_milestone(epoch)
            if epoch == 1 or epoch % 10 == 0 or epoch in MILESTONES:
                print(json.dumps({
                    "candidate": "C",
                    "seed": args.seed,
                    "epoch": epoch,
                    "physics_steps": training_env.simulation_physics_steps,
                    "elapsed_seconds": time.time() - started,
                    "latest_update": agent.update_reports[-1],
                    "latest_validation": validations.get(str(epoch), {}).get("summary"),
                }), flush=True)
        trace = training_env.finalize_training_trace(completed=True)
        update_audit = agent.finalize_algorithmic_audit()
    except BaseException as error:
        failure = error
        try:
            trace = training_env.finalize_training_trace(completed=False)
        except Exception as trace_error:
            error.add_note(f"training trace finalization also failed: {trace_error}")
    finally:
        if agent.writer is not None:
            agent.writer.close()
        if validation_agent.writer is not None:
            validation_agent.writer.close()

    learning_curve = {
        str(epoch): {
            "epoch": epoch,
            "physics_steps": MILESTONES[epoch][1],
            "cumulative": metric_summary(agent.update_reports[:epoch]),
            "validation": validations.get(str(epoch), {}).get("summary"),
        }
        for epoch in MILESTONES
        if epoch == 0 or epoch <= len(agent.update_reports)
    }

    common = {
        "classification": contract["classification"],
        "paper_faithful": False,
        "candidate": "C",
        "seed": args.seed,
        "backend": "CPU_MuJoCo_Warp_training_and_validation",
        "fresh_actor_critic_optimizers_RMS": True,
        "warm_start_used": False,
        "initialization": initialization_audit,
        "pretraining_validation": pretraining_validation,
        "training": {
            "worlds": 4,
            "target_epochs": 250,
            "completed_actor_updates": len(agent.update_reports),
            "critic_updates_per_epoch": 4,
            "simulation_physics_steps": training_env.simulation_physics_steps,
            "simulation_control_intervals": training_env.simulation_control_intervals,
            "wall_seconds": time.time() - started,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
            "lr_schedule": None,
            "bounds_loss_coef": 0.005,
            "bound_loss_type": "regularisation",
        },
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "validations": validations,
        "learning_curve": learning_curve,
        "training_visitation": trace,
        "H2S2R_mean_regularization_provenance": artifact(
            frozen["H2S2R_LSTM_config"]
        ),
        "linear_LR_executed": False,
        "half_LR_executed": False,
        "adaptive_LR_executed": False,
        "KL_penalty_executed": False,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_written": False,
    }
    if failure is None:
        if trace is None or update_audit is None or len(agent.update_reports) != 250:
            raise RuntimeError("Candidate C did not complete its fixed budget")
        report = {
            "schema": "taco_pour_algorithmic_candidate_C_training_v5",
            "status": "completed_fixed_400k_no_chunk_commit",
            **common,
            "update_audit": update_audit,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "status": report["status"],
            "seed": args.seed,
            "validations": {
                epoch: row["summary"] for epoch, row in validations.items()
            },
            "report": artifact(report_path),
        }, indent=2))
        return

    failure_report = {
        "schema": "taco_pour_algorithmic_candidate_C_failclosed_v5",
        "status": "failed_closed_before_fixed_400k_budget",
        **common,
        "exception": {
            "type": type(failure).__name__,
            "message": str(failure),
            "traceback": "".join(traceback.format_exception(failure)),
        },
        "last_completed_actor_update_epoch": len(agent.update_reports),
        "eligible_as_400k_outcome": False,
        "retry_executed": False,
        "threshold_relaxed": False,
    }
    failure_path = output / "failure_report.json"
    failure_path.write_text(json.dumps(failure_report, indent=2) + "\n")
    print(json.dumps({
        "status": failure_report["status"],
        "seed": args.seed,
        "last_completed_actor_update_epoch": len(agent.update_reports),
        "exception": failure_report["exception"],
        "failure_report": artifact(failure_path),
    }, indent=2), file=sys.stderr)
    raise failure


if __name__ == "__main__":
    main()
