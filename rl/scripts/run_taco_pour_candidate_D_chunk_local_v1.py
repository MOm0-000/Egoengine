#!/usr/bin/env python3
"""Run a fresh Candidate-D solve for the endpoint-40 progressive chunk."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
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


CONTRACT = ROOT / "configs/taco_pour_candidate_D_chunk_local_solver_v1.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_progressive_chunk_execution_v1"


def _validation_summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    endpoints = arrays["endpoint"].astype(int).tolist()
    failure_index = next(
        (index for index, done in enumerate(arrays["terminated"].tolist()) if done),
        None,
    )
    successful = 40 if failure_index is None else failure_index
    failure = None if failure_index is None else endpoints[failure_index]

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

    summary: dict[str, object] = {
        "successful_intervals": successful,
        "first_failure_endpoint": failure,
        "forty_of_forty": failure_index is None,
        "endpoint60": endpoint_row(60),
        "endpoint70": endpoint_row(70),
        "endpoint80": endpoint_row(80),
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
        summary["first_failure"] = endpoint_row(failure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    candidate = contract.get("candidate_D", {})
    if (
        contract.get("schema") != "taco_pour_candidate_D_chunk_local_solver_v1"
        or contract.get("status") != "authorized_fresh_chunk_source40"
        or contract.get("paper_faithful") is not False
        or args.seed not in candidate.get("restart_seed_order", [])
        or candidate.get("initialization") != "replay_preserving"
        or float(candidate.get("initial_sigma_multiplier", -1)) != 0.25
        or candidate.get("actor_mini_epochs") != 1
        or float(candidate.get("actor_learning_rate", -1)) != 1.0e-4
        or candidate.get("first_strict_milestone_action")
        != "stop_no_commit_request_separate_promotion"
        or bool(candidate.get("warm_start_allowed", True))
        or bool(candidate.get("non_strict_checkpoint_selection_allowed", True))
    ):
        raise ValueError("chunk-local Candidate-D contract changed")
    output = args.run_root / "chunk_source_40" / "candidate_D" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen chunk-local input changed: {name}")
    implementations = {
        "state_feasible_distribution_sha256": (
            ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py"
        ),
        "v6_algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
        ),
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": (
            ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py"
        ),
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "chunk_solver_runner_sha256": Path(__file__).resolve(),
    }
    for name, path in implementations.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"chunk-local implementation changed: {name}")

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
    from video_to_spider.rl.algorithmic_training_v6 import (
        SupportAnchoredBoundedMeanPpoAgent,
        load_support_anchored_profile,
    )
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
    )

    objective = load_runtime_objective(
        inputs["protocol_at_authorization"],
        inputs["objective_profile"],
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        inputs["protocol_at_authorization"],
        inputs["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(inputs["action_profile"])
    distribution, distribution_report = load_support_anchored_profile(
        inputs["bounded_mean_distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    boundary = load_checkpoint(inputs["source_boundary"])
    if (
        boundary.get("mujoco_warp_version") != "3.13.0"
        or np.asarray(boundary["time_indices"]).tolist() != [40]
    ):
        raise ValueError("chunk-local solve requires exact endpoint-40 boundary")
    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)

    def make_world(*, asymmetric: bool, seed: int) -> MJWPVectorEnv:
        world = MJWPVectorEnv(
            cpu_config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=asymmetric,
                max_episode_length=len(reference[0]) - 1,
                tracked_object_indices=(0,),
                object_roles=("tool", "target"),
                objective=objective,
                observation=observation,
                residual=residual,
            ),
            seed=seed,
        )
        verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
        world.set_env_state(boundary)
        return world

    training_env = IndependentMJWPTrainingEnv([
        make_world(asymmetric=True, seed=args.seed + index) for index in range(4)
    ])
    training_env.set_chunk_reset(start=40, end=80)
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
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        lr_schedule=None,
        schedule_type="legacy",
        kl_threshold=None,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = SupportAnchoredBoundedMeanPpoAgent(
        experiment_dir=output / "ppo",
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=training_env,
        distribution_spec=distribution,
        likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
        canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
        audit_dir=output / "update_audit",
    )
    initialization_audit = replay_preserving_initialization(agent, sigma_multiplier=0.25)
    config_hashes = {
        "chunk_solver_contract": sha256(args.contract),
        "chunk_solver_runner": sha256(Path(__file__)),
        "v6_algorithmic_training": sha256(
            ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
        ),
        "source_boundary": sha256(inputs["source_boundary"]),
    }

    validation_world = make_world(asymmetric=False, seed=args.seed)
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
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        lr_schedule=None,
    )
    validation_agent = SupportAnchoredBoundedMeanPpoAgent(
        experiment_dir=output / "validation_agent",
        ppo_config=validation_config,
        network_config=_build_network_config(4),
        env=validation_world,
        distribution_spec=distribution,
        likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
        canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
        audit_dir=output / "validation_audit_unused",
    )

    input_dir = output / "input_contracts"
    input_dir.mkdir()
    for name, path in {"solver_contract": args.contract, **inputs}.items():
        suffix = "".join(path.suffixes) or ".bin"
        (input_dir / f"{name}{suffix}").write_bytes(path.read_bytes())

    checkpoints: dict[int, dict[str, object]] = {}
    validations: dict[str, dict[str, object]] = {}

    def validate_payload(*, epoch: int, payload: dict[str, Any]) -> dict[str, object]:
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
            "deterministic_action": [], "raw_location": [], "bounded_mu": [],
            "actor_sigma": [], "action_low": [], "action_high": [],
        }
        for source in range(40, 80):
            obs = validation_agent.obs_to_tensors(validation_world.current_observation())
            result = validation_agent.get_deterministic_action_values(obs)
            validation_agent.rnn_states = result["rnn_states"]
            action = validation_agent.preprocess_actions(result["deterministic_actions"])
            _, _, _, info = validation_world.step(action, auto_reset=False)
            rows["endpoint"].append(source + 1)
            rows["terminated"].append(bool(info["terminated"][0]))
            rows["tracking_score"].append(float(info["object_tracking_error"][0]))
            rows["position_error"].append(float(info["object_position_error"][0, 0]))
            rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            rows["ctrl"].append(validation_world._last_ctrl[0].detach().cpu().numpy().copy())
            rows["qpos"].append(
                validation_world._mjwp.get_qpos(cpu_config, validation_world.env)[0]
                .detach().cpu().numpy().copy()
            )
            rows["qvel"].append(
                validation_world._mjwp.get_qvel(cpu_config, validation_world.env)[0]
                .detach().cpu().numpy().copy()
            )
            rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            rows["deterministic_action"].append(np.asarray(action[0]).copy())
            rows["raw_location"].append(result["raw_locations"][0].detach().cpu().numpy().copy())
            rows["bounded_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
            rows["actor_sigma"].append(result["sigmas"][0].detach().cpu().numpy().copy())
            rows["action_low"].append(result["action_lows"][0].detach().cpu().numpy().copy())
            rows["action_high"].append(result["action_highs"][0].detach().cpu().numpy().copy())
        arrays = {
            name: np.asarray(values, dtype=np.int32) if name == "endpoint" else np.asarray(values)
            for name, values in rows.items()
        }
        arrays["actor_mu"] = arrays["bounded_mu"]
        path = output / "validations" / f"epoch_{epoch:04d}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        summary = _validation_summary(arrays)
        summary["bounded_mu_outside_support_count"] = int(np.count_nonzero(
            (arrays["bounded_mu"] < arrays["action_low"])
            | (arrays["bounded_mu"] > arrays["action_high"])
        ))
        row: dict[str, object] = {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": summary,
            "trajectory": artifact(path),
            "checkpoint": checkpoints[epoch],
            "chunk_commit_written": False,
        }
        if epoch == 0:
            with np.load(inputs["Replay_40_80_trajectory"]) as replay:
                checks = {
                    "raw_location_bitwise_zero": bool(
                        arrays["raw_location"].tobytes()
                        == np.zeros_like(arrays["raw_location"]).tobytes()
                    ),
                    "bounded_mu_bitwise_zero": bool(
                        arrays["bounded_mu"].tobytes()
                        == np.zeros_like(arrays["bounded_mu"]).tobytes()
                    ),
                    "deterministic_action_bitwise_zero": bool(
                        arrays["deterministic_action"].tobytes()
                        == np.zeros_like(arrays["deterministic_action"]).tobytes()
                    ),
                    "command_bitwise_equals_Replay": bool(
                        arrays["ctrl"].tobytes() == replay["ctrl"].tobytes()
                    ),
                    "qpos_bitwise_equals_Replay": bool(
                        arrays["qpos"].tobytes() == replay["qpos"].tobytes()
                    ),
                    "qvel_bitwise_equals_Replay": bool(
                        arrays["qvel"].tobytes() == replay["qvel"].tobytes()
                    ),
                    "score_bitwise_equals_Replay": bool(
                        arrays["tracking_score"].tobytes()
                        == replay["tracking_score"].tobytes()
                    ),
                    "termination_bitwise_equals_Replay": bool(
                        arrays["terminated"].tobytes() == replay["terminated"].tobytes()
                    ),
                    "contact_flags_bitwise_equal_Replay": bool(
                        arrays["contact_flags"].tobytes()
                        == replay["contact_flags"].tobytes()
                    ),
                    "is_9_of_40_fail50": bool(
                        summary["successful_intervals"] == 9
                        and summary["first_failure_endpoint"] == 50
                    ),
                }
            row["step0_checks"] = checks
            row["all_step0_checks_passed"] = all(checks.values())
        validations[str(epoch)] = row
        return row

    def save_milestone(epoch: int) -> dict[str, Any]:
        _, physics, controls = MILESTONES[epoch]
        if (
            int(training_env.simulation_physics_steps),
            int(training_env.simulation_control_intervals),
        ) != (physics, controls):
            raise RuntimeError(f"milestone {epoch} counters disagree")
        payload = build_checkpoint_payload(
            agent,
            candidate="B",
            seed=args.seed,
            simulation_physics_steps=physics,
            simulation_control_intervals=controls,
            config_hashes=config_hashes,
        )
        payload["candidate"] = "D"
        payload["candidate_lineage"] = "fresh_chunk_local_replay_preserving_bounded_mean"
        payload["chunk_source_endpoint"] = 40
        path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
        checkpoint_row = write_checkpoint(path, payload)
        restored = load_checkpoint(path)
        validate_checkpoint_payload(restored)
        if not exact_equal(payload, restored):
            raise RuntimeError(f"milestone {epoch} checkpoint roundtrip differs")
        checkpoints[epoch] = {
            **checkpoint_row,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
        rng = deepcopy(payload["rng_states"])
        row = validate_payload(epoch=epoch, payload=restored)
        restore_rng_states(rng)
        if epoch == 0 and not row["all_step0_checks_passed"]:
            raise RuntimeError("chunk-local Candidate D failed Replay identity gate")
        return restored

    started = time.time()
    trace = None
    update_audit = None
    success_epoch: int | None = None
    failure: BaseException | None = None
    try:
        agent.init_tensors()
        agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs
        save_milestone(0)
        while int(agent.epoch_num) < 250:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in MILESTONES:
                save_milestone(epoch)
                if validations[str(epoch)]["summary"]["forty_of_forty"]:
                    success_epoch = epoch
                    break
            if epoch == 1 or epoch % 10 == 0 or epoch in MILESTONES:
                print(json.dumps({
                    "candidate": "D", "seed": args.seed, "source_endpoint": 40,
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
            error.add_note(f"trace finalization also failed: {trace_error}")
    finally:
        if agent.writer is not None:
            agent.writer.close()
        if validation_agent.writer is not None:
            validation_agent.writer.close()

    common = {
        "schema": "taco_pour_candidate_D_chunk_local_training_v1",
        "paper_faithful": False,
        "classification": "fresh_progressive_chunk_local_candidate_D_solve",
        "candidate": "D",
        "seed": args.seed,
        "source_endpoint": 40,
        "lookahead_endpoint": 80,
        "fresh_actor_critic_optimizers_RMS_RNN": True,
        "warm_start_used": False,
        "initialization": initialization_audit,
        "pretraining_validation": validations.get("0"),
        "training": {
            "worlds": 4,
            "completed_actor_updates": len(agent.update_reports),
            "simulation_physics_steps": training_env.simulation_physics_steps,
            "simulation_control_intervals": training_env.simulation_control_intervals,
            "wall_seconds": time.time() - started,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
            "lr_schedule": None,
            "bounds_loss_coef": 0.0,
        },
        "validations": validations,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "training_visitation": trace,
        "update_audit": update_audit,
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_written": False,
    }
    if failure is not None:
        report = {
            **common,
            "status": "failed_closed",
            "exception": {
                "type": type(failure).__name__,
                "message": str(failure),
                "traceback": "".join(traceback.format_exception(failure)),
            },
            "eligible_for_promotion": False,
        }
        path = output / "failure_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise failure
    if success_epoch is not None:
        report = {
            **common,
            "status": "strict_success_fixed_milestone_stop_no_chunk_commit",
            "strict_success_epoch": success_epoch,
            "strict_success_checkpoint": checkpoints[success_epoch],
            "strict_success_validation": validations[str(success_epoch)],
            "eligible_for_separate_promotion": True,
            "restart_required": False,
        }
    else:
        report = {
            **common,
            "status": "completed_400k_without_strict_success_no_chunk_commit",
            "strict_success_epoch": None,
            "eligible_for_separate_promotion": False,
            "restart_required": args.seed != candidate["restart_seed_order"][-1],
            "next_seed": candidate["restart_seed_order"][args.seed + 1]
            if args.seed + 1 < len(candidate["restart_seed_order"])
            else None,
        }
    report["learning_curve"] = {
        str(epoch): {
            "epoch": epoch,
            "physics_steps": MILESTONES[epoch][1],
            "cumulative": metric_summary(agent.update_reports[:epoch]),
            "validation": validations.get(str(epoch), {}).get("summary"),
        }
        for epoch in MILESTONES
        if epoch == 0 or epoch <= len(agent.update_reports)
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "seed": args.seed,
        "strict_success_epoch": success_epoch,
        "validations": {key: row["summary"] for key, row in validations.items()},
        "report": artifact(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
