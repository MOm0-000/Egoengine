#!/usr/bin/env python3
"""Run one fresh 400k Candidate-D bounded-mean ablation seed."""

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
    validation_summary,
)


CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v6.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v6"


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
        "mean_raw_location_abs_mean": float(np.mean(values(
            "raw_location_abs_mean"
        ))),
        "maximum_raw_location_abs_max": float(np.max(values(
            "raw_location_abs_max"
        ))),
        "latest_raw_location_abs_p95": values("raw_location_abs_p95")[-1],
        "mean_tanh_saturation_fraction": float(np.mean(values(
            "tanh_saturation_fraction"
        ))),
        "latest_tanh_saturation_fraction": values(
            "tanh_saturation_fraction"
        )[-1],
        "maximum_bounded_mu_outside_support_count": int(max(values(
            "bounded_mu_outside_support_count"
        ))),
        "maximum_bounded_mu_at_support_fraction": float(np.max(values(
            "bounded_mu_at_support_fraction"
        ))),
        "minimum_distance_to_support_boundary_in_sigma": float(np.min(values(
            "minimum_distance_to_support_boundary_in_sigma"
        ))),
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
        "all_v6_identity_checks_passed": all(
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
        raise ValueError("Candidate D is frozen to seeds 0, 1, 2")

    contract = yaml.safe_load(args.contract.read_text())
    candidate = contract.get("candidate_D", {})
    if (
        contract.get("schema")
        != "taco_pour_algorithmic_reproduction_training_benchmark_v6"
        or contract.get("status") != "authorized_candidate_D_fresh_400k"
        or contract.get("paper_faithful") is not False
        or candidate.get("seeds") != [0, 1, 2]
        or candidate.get("initialization") != "replay_preserving"
        or float(candidate.get("bounds_loss_coef", -1.0)) != 0.0
        or candidate.get("location_parameterization")
        != "support_anchored_piecewise_tanh"
        or bool(candidate.get("linear_LR_allowed", True))
        or bool(candidate.get("warm_start_allowed", True))
        or bool(candidate.get("chunk_commit_allowed", True))
    ):
        raise ValueError("v6 Candidate D is not authorized")
    if Path(contract["output_directory"]).resolve() != args.run_root.resolve():
        raise ValueError("Candidate-D output root differs from the contract")
    expected_milestones = [
        {
            "name": name,
            "epoch": epoch,
            "physics_steps": physics,
            "control_intervals": controls,
        }
        for epoch, (name, physics, controls) in MILESTONES.items()
    ]
    if contract.get("fixed_milestones") != expected_milestones:
        raise ValueError("v6 milestones changed")

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

    frozen = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in frozen.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Candidate-D input changed: {name}")
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
        "v6_algorithmic_training_sha256": (
            ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
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
            raise ValueError(f"Candidate-D implementation changed: {name}")

    objective = load_runtime_objective(
        frozen["protocol_at_authorization"],
        frozen["objective_profile"],
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        frozen["protocol_at_authorization"],
        frozen["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(
        frozen["action_profile"]
    )
    distribution, distribution_report = load_support_anchored_profile(
        frozen["bounded_mean_distribution_profile"]
    )
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
        raise ValueError("Candidate D requires the exact endpoint-20 boundary")

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

    output = args.run_root / "training" / "candidate_D" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_dir = output / "input_contracts"
    input_dir.mkdir()
    for name, path in {"v6_contract": args.contract, **frozen}.items():
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
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
        audit_dir=output / "update_audit",
    )
    initialization_audit = replay_preserving_initialization(
        agent, sigma_multiplier=0.25
    )
    config_hashes = {
        "v6_contract": sha256(args.contract),
        "training_runner": sha256(Path(__file__)),
        "v6_algorithmic_training": sha256(
            ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"
        ),
        "bounded_mean_distribution_profile": sha256(
            frozen["bounded_mean_distribution_profile"]
        ),
    }

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
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
        audit_dir=output / "validation_audit_unused",
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
            "endpoint": [],
            "terminated": [],
            "tracking_score": [],
            "position_error": [],
            "rotation_error": [],
            "ctrl": [],
            "qpos": [],
            "qvel": [],
            "contact_flags": [],
            "deterministic_action": [],
            "raw_location": [],
            "bounded_mu": [],
            "actor_sigma": [],
            "action_low": [],
            "action_high": [],
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
            rows["raw_location"].append(
                result["raw_locations"][0].detach().cpu().numpy().copy()
            )
            rows["bounded_mu"].append(
                result["mus"][0].detach().cpu().numpy().copy()
            )
            rows["actor_sigma"].append(
                result["sigmas"][0].detach().cpu().numpy().copy()
            )
            rows["action_low"].append(
                result["action_lows"][0].detach().cpu().numpy().copy()
            )
            rows["action_high"].append(
                result["action_highs"][0].detach().cpu().numpy().copy()
            )
        arrays = {
            name: np.asarray(values, dtype=np.int32)
            if name == "endpoint"
            else np.asarray(values)
            for name, values in rows.items()
        }
        # Compatibility name used by the immutable v5 summary helper.
        arrays["actor_mu"] = arrays["bounded_mu"]
        path = output / "validations" / f"epoch_{epoch:04d}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        summary = validation_summary(arrays)
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
        if step0:
            with np.load(frozen["Replay_step0_trajectory"]) as replay:
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
                    "bounded_mu_outside_support_count_zero": (
                        summary["bounded_mu_outside_support_count"] == 0
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
            candidate="B",
            seed=args.seed,
            simulation_physics_steps=physics,
            simulation_control_intervals=controls,
            config_hashes=config_hashes,
        )
        payload["candidate"] = "D"
        payload["candidate_lineage"] = (
            "B_replay_preserving_plus_support_anchored_bounded_mean"
        )
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
        validation = validate_payload(
            epoch=epoch, payload=restored, step0=epoch == 0
        )
        restore_rng_states(rng)
        if epoch == 0 and not validation["all_step0_checks_passed"]:
            raise RuntimeError("Candidate D failed the per-seed Replay identity gate")
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
                    "candidate": "D",
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
        "candidate": "D",
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
            "bounds_loss_coef": 0.0,
            "location_parameterization": "support_anchored_piecewise_tanh",
        },
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "validations": validations,
        "learning_curve": learning_curve,
        "training_visitation": trace,
        "linear_LR_executed": False,
        "mean_regularization_executed": False,
        "half_LR_executed": False,
        "adaptive_LR_executed": False,
        "KL_penalty_executed": False,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_written": False,
    }
    if failure is None:
        if trace is None or update_audit is None or len(agent.update_reports) != 250:
            raise RuntimeError("Candidate D did not complete its fixed budget")
        report = {
            "schema": "taco_pour_algorithmic_candidate_D_training_v6",
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
        "schema": "taco_pour_algorithmic_candidate_D_failclosed_v6",
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
