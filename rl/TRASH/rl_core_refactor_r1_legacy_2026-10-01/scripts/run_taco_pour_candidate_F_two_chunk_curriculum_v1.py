#!/usr/bin/env python3
"""Run one fresh Candidate-F balanced two-chunk curriculum seed."""

from __future__ import annotations

import argparse
from collections import Counter
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
from run_taco_pour_candidate_D_chunk_local_v1 import _validation_summary  # noqa: E402


CONTRACT = ROOT / "configs/taco_pour_candidate_F_two_chunk_curriculum_v1.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_candidate_F_two_chunk_curriculum_v1"


def _coverage_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "taco_ppo_training_visitation_v6_world_indexed":
        raise ValueError("Candidate F requires the world-indexed v6 visitation schema")
    epochs = {int(row["epoch"]): Path(row["visits"]["path"]) for row in manifest["epochs"]}
    segments = {
        "epochs_1_through_62": range(1, 63),
        "epochs_63_through_125": range(63, 126),
    }
    cohorts = {
        "all_worlds": (0, 1, 2, 3),
        "anchor_worlds": (0, 1),
        "tail_worlds": (2, 3),
    }
    result: dict[str, Any] = {
        "schema": "taco_pour_candidate_F_world_indexed_coverage_v1",
        "relative_offset": "k = source_endpoint - 40",
        "cohort_note": (
            "world indices 2/3 are ordinary endpoint40 anchors before epoch63 and "
            "fixed endpoint60 tail worlds from epoch63 onward"
        ),
        "segments": {},
    }
    for segment_name, epoch_range in segments.items():
        segment: dict[str, Any] = {}
        present_epochs = [epoch for epoch in epoch_range if epoch in epochs]
        for cohort_name, world_ids in cohorts.items():
            sources: list[np.ndarray] = []
            outcomes: list[np.ndarray] = []
            terminated: list[np.ndarray] = []
            maxima: list[int] = []
            termination_hist: Counter[int] = Counter()
            source60_actions = 0
            endpoint61_terminations = 0
            for epoch in present_epochs:
                with np.load(epochs[epoch], allow_pickle=False) as data:
                    world_index = np.asarray(data["world_index"], np.int32)
                    select = np.isin(world_index, np.asarray(world_ids, np.int32))
                    source = np.asarray(data["source_endpoint"], np.int32)[select]
                    outcome = np.asarray(data["outcome_endpoint"], np.int32)[select]
                    term = np.asarray(data["tracking_terminated"], bool)[select]
                    selected_worlds = world_index[select]
                    sources.append(source)
                    outcomes.append(outcome)
                    terminated.append(term)
                    source60_actions += int(np.count_nonzero(source == 60))
                    endpoint61_terminations += int(
                        np.count_nonzero((outcome == 61) & term)
                    )
                    termination_hist.update(map(int, outcome[term]))
                    for world_id in world_ids:
                        world_source = source[selected_worlds == world_id]
                        if len(world_source):
                            maxima.append(int(np.max(world_source - 40)))
            source_all = np.concatenate(sources) if sources else np.empty(0, np.int32)
            outcome_all = np.concatenate(outcomes) if outcomes else np.empty(0, np.int32)
            term_all = np.concatenate(terminated) if terminated else np.empty(0, bool)
            k = source_all - 40
            segment[cohort_name] = {
                "epochs_present": present_epochs,
                "sample_count": int(len(k)),
                "fraction_k_ge_21": float(np.mean(k >= 21)) if len(k) else None,
                "fraction_k_ge_25": float(np.mean(k >= 25)) if len(k) else None,
                "fraction_k_ge_30": float(np.mean(k >= 30)) if len(k) else None,
                "fraction_k_ge_35": float(np.mean(k >= 35)) if len(k) else None,
                "source60_action_count": source60_actions,
                "endpoint61_tracking_termination_count": endpoint61_terminations,
                "maximum_reached_k_histogram_by_epoch_world": {
                    str(key): value for key, value in sorted(Counter(maxima).items())
                },
                "tracking_termination_endpoint_histogram": {
                    str(key): value for key, value in sorted(termination_hist.items())
                },
                "tracking_termination_count": int(term_all.sum()),
                "maximum_outcome_endpoint": int(outcome_all.max()) if len(outcome_all) else None,
            }
        result["segments"][segment_name] = segment
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    candidate = contract.get("candidate_F", {})
    curriculum = contract.get("curriculum", {})
    if (
        contract.get("schema") != "taco_pour_candidate_F_two_chunk_curriculum_v1"
        or contract.get("status") != "authorized_fixed_two_chunk_curriculum_200k"
        or contract.get("paper_faithful") is not False
        or candidate.get("fresh_seeds") != [0, 1, 2]
        or args.seed not in candidate["fresh_seeds"]
        or candidate.get("maximum_epoch") != 125
        or candidate.get("maximum_physics_steps_per_seed") != 200_000
        or candidate.get("actor_mini_epochs") != 1
        or float(candidate.get("actor_learning_rate", -1)) != 1.0e-4
        or candidate.get("critic_mini_epochs") != 4
        or bool(candidate.get("warm_start_allowed", True))
        or curriculum.get("activation_epoch") != 63
        or curriculum.get("epochs_63_through_125_world_starts") != [40, 40, 60, 60]
    ):
        raise ValueError("Candidate-F contract changed")
    output = args.run_root / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Candidate-F input changed: {name}")
    boundary_root = args.run_root / "tail_boundary_builder"
    sampler_report_path = args.run_root / "sampler_gate/report.json"
    sampler_report = json.loads(sampler_report_path.read_text())
    if (
        sampler_report.get("status") != "passed_no_training_sampler_gate"
        or sampler_report.get("candidate_F_training_authorized") is not True
        or sampler_report.get("optimizer_steps") != 0
    ):
        raise ValueError("Candidate-F sampler gate did not authorize training")
    tail_boundaries = tuple(
        load_checkpoint(boundary_root / f"seed{seed}_endpoint60.pt.gz")
        for seed in (0, 1)
    )

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
        inputs["protocol_at_authorization"], inputs["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        inputs["protocol_at_authorization"], inputs["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(inputs["action_profile"])
    distribution, distribution_report = load_support_anchored_profile(
        inputs["bounded_mean_distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    source_boundary = load_checkpoint(inputs["source_boundary_endpoint40"])
    if np.asarray(source_boundary["time_indices"]).tolist() != [40]:
        raise ValueError("Candidate F requires the exact committed endpoint40 boundary")
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
        world.set_env_state(source_boundary)
        return world

    training_env = IndependentMJWPTrainingEnv([
        make_world(asymmetric=True, seed=args.seed + index) for index in range(4)
    ])
    training_env.set_chunk_reset(start=40, end=80)
    training_env.enable_fixed_two_chunk_curriculum(
        tail_boundaries,
        anchor_endpoint=40,
        tail_endpoint=60,
        window_end_endpoint=80,
        activation_epoch=63,
    )
    training_env.enable_training_trace(
        output / "training_visitation", include_world_index=True
    )
    ppo_config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=125,
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

    validation_world = make_world(asymmetric=False, seed=args.seed)
    validation_config = replace(
        _build_ppo_config(
            num_envs=1, horizon_length=40, seq_length=4, max_epochs=1,
            learning_rate=1.0e-4, device="cpu", asymmetric_critic=None,
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
    frozen_inputs = {
        "candidate_F_contract": args.contract,
        "sampler_gate_report": sampler_report_path,
        "tail_boundary_builder_report": boundary_root / "report.json",
        **inputs,
    }
    for name, path in frozen_inputs.items():
        suffix = "".join(path.suffixes) or ".bin"
        (input_dir / f"{name}{suffix}").write_bytes(path.read_bytes())

    config_hashes = {
        "candidate_F_contract": sha256(args.contract),
        "candidate_F_runner": sha256(Path(__file__)),
        "sampler_gate_report": sha256(sampler_report_path),
        "tail_boundary_A": sha256(boundary_root / "seed0_endpoint60.pt.gz"),
        "tail_boundary_B": sha256(boundary_root / "seed1_endpoint60.pt.gz"),
        "v6_algorithmic_training": sha256(ROOT / "src/video_to_spider/rl/algorithmic_training_v6.py"),
        "MJWP_environment": sha256(ROOT / "src/video_to_spider/rl/mjwp_env.py"),
        "training_trace": sha256(ROOT / "src/video_to_spider/rl/training_trace.py"),
        "source_boundary": sha256(inputs["source_boundary_endpoint40"]),
    }
    checkpoints: dict[int, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}

    def validate_payload(epoch: int, payload: dict[str, Any]) -> dict[str, Any]:
        validation_agent.model.load_state_dict(payload["actor"], strict=True)
        validation_agent.set_eval()
        validation_agent.rnn_states = [
            state.to("cpu").zero_()
            for state in validation_agent.model.get_default_rnn_state()
        ]
        validation_world.set_env_state(source_boundary)
        rows: dict[str, list[Any]] = {
            "endpoint": [], "terminated": [], "tracking_score": [],
            "position_error": [], "rotation_error": [], "ctrl": [],
            "qpos": [], "qvel": [], "contact_flags": [],
            "deterministic_action": [], "raw_location": [], "bounded_mu": [],
            "actor_sigma": [], "action_low": [], "action_high": [],
        }
        for source in range(40, 80):
            obs = validation_agent.obs_to_tensors(validation_world.current_observation())
            values = validation_agent.get_deterministic_action_values(obs)
            validation_agent.rnn_states = values["rnn_states"]
            action = validation_agent.preprocess_actions(values["deterministic_actions"])
            _, _, _, info = validation_world.step(action, auto_reset=False)
            rows["endpoint"].append(source + 1)
            rows["terminated"].append(bool(info["terminated"][0]))
            rows["tracking_score"].append(float(info["object_tracking_error"][0]))
            rows["position_error"].append(float(info["object_position_error"][0, 0]))
            rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            rows["ctrl"].append(validation_world._last_ctrl[0].detach().cpu().numpy().copy())
            rows["qpos"].append(validation_world._mjwp.get_qpos(cpu_config, validation_world.env)[0].detach().cpu().numpy().copy())
            rows["qvel"].append(validation_world._mjwp.get_qvel(cpu_config, validation_world.env)[0].detach().cpu().numpy().copy())
            rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            rows["deterministic_action"].append(np.asarray(action[0]).copy())
            rows["raw_location"].append(values["raw_locations"][0].detach().cpu().numpy().copy())
            rows["bounded_mu"].append(values["mus"][0].detach().cpu().numpy().copy())
            rows["actor_sigma"].append(values["sigmas"][0].detach().cpu().numpy().copy())
            rows["action_low"].append(values["action_lows"][0].detach().cpu().numpy().copy())
            rows["action_high"].append(values["action_highs"][0].detach().cpu().numpy().copy())
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
        row: dict[str, Any] = {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": summary,
            "trajectory": artifact(path),
            "checkpoint": checkpoints[epoch],
            "chunk_commit_written": False,
            "validation_source_endpoint": 40,
            "validation_initial_RNN": "zero",
            "curriculum_tail_used_for_acceptance": False,
        }
        if epoch == 0:
            with np.load(inputs["Replay_40_80_trajectory"], allow_pickle=False) as replay:
                checks = {
                    "raw_location_bitwise_zero": arrays["raw_location"].tobytes() == np.zeros_like(arrays["raw_location"]).tobytes(),
                    "bounded_mu_bitwise_zero": arrays["bounded_mu"].tobytes() == np.zeros_like(arrays["bounded_mu"]).tobytes(),
                    "deterministic_action_bitwise_zero": arrays["deterministic_action"].tobytes() == np.zeros_like(arrays["deterministic_action"]).tobytes(),
                    "command_bitwise_equals_Replay": arrays["ctrl"].tobytes() == replay["ctrl"].tobytes(),
                    "qpos_bitwise_equals_Replay": arrays["qpos"].tobytes() == replay["qpos"].tobytes(),
                    "qvel_bitwise_equals_Replay": arrays["qvel"].tobytes() == replay["qvel"].tobytes(),
                    "score_bitwise_equals_Replay": arrays["tracking_score"].tobytes() == replay["tracking_score"].tobytes(),
                    "termination_bitwise_equals_Replay": arrays["terminated"].tobytes() == replay["terminated"].tobytes(),
                    "contact_flags_bitwise_equal_Replay": arrays["contact_flags"].tobytes() == replay["contact_flags"].tobytes(),
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
            raise RuntimeError(f"Candidate-F milestone {epoch} counters disagree: {observed}")
        payload = build_checkpoint_payload(
            agent, candidate="B", seed=args.seed,
            simulation_physics_steps=physics,
            simulation_control_intervals=controls,
            config_hashes=config_hashes,
        )
        payload["candidate"] = "F"
        payload["candidate_lineage"] = "fresh_balanced_two_chunk_curriculum"
        payload["chunk_source_endpoint"] = 40
        payload["curriculum_activation_epoch"] = 63
        path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
        checkpoint_row = write_checkpoint(path, payload)
        restored = load_checkpoint(path)
        validate_checkpoint_payload(restored)
        if not exact_equal(payload, restored):
            raise RuntimeError(f"Candidate-F checkpoint {epoch} roundtrip differs")
        checkpoints[epoch] = {
            **checkpoint_row,
            "payload_roundtrip_exact": True,
            "checkpoint_validation_passed": True,
        }
        rng = deepcopy(payload["rng_states"])
        row = validate_payload(epoch, restored)
        restore_rng_states(rng)
        if epoch == 0 and not row["all_step0_checks_passed"]:
            raise RuntimeError("Candidate F failed the Replay identity gate")
        return restored

    started = time.time()
    trace = None
    update_audit = None
    coverage = None
    success_epoch: int | None = None
    failure: BaseException | None = None
    try:
        agent.init_tensors()
        agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs
        save_milestone(0)
        while int(agent.epoch_num) < 125:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in (62, 125):
                save_milestone(epoch)
                if validations[str(epoch)]["summary"]["forty_of_forty"]:
                    success_epoch = epoch
                    break
            if epoch == 1 or epoch % 10 == 0 or epoch in (62, 63, 125):
                print(json.dumps({
                    "candidate": "F", "seed": args.seed, "epoch": epoch,
                    "physics_steps": training_env.simulation_physics_steps,
                    "curriculum_active": epoch >= 63,
                    "elapsed_seconds": time.time() - started,
                    "latest_update": agent.update_reports[-1],
                    "latest_validation": validations.get(str(epoch), {}).get("summary"),
                }), flush=True)
        trace = training_env.finalize_training_trace(completed=True)
        coverage = _coverage_summary(trace)
        coverage_path = output / "coverage_summary.json"
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
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
        "schema": "taco_pour_candidate_F_two_chunk_curriculum_training_v1",
        "paper_faithful": False,
        "classification": "local_balanced_two_chunk_lookahead_curriculum",
        "candidate": "F",
        "seed": args.seed,
        "fresh_actor_critic_optimizers_RMS_RNN": True,
        "warm_start_used": False,
        "initialization": initialization_audit,
        "training": {
            "worlds": 4,
            "epochs_completed": len(agent.update_reports),
            "simulation_physics_steps": training_env.simulation_physics_steps,
            "simulation_control_intervals": training_env.simulation_control_intervals,
            "wall_seconds": time.time() - started,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
            "epoch_1_62_world_starts": [40, 40, 40, 40],
            "epoch_63_125_world_starts": [40, 40, 60, 60],
        },
        "validations": validations,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "training_visitation": trace,
        "coverage": coverage,
        "curriculum_audit": training_env.fixed_two_chunk_curriculum_audit(),
        "update_audit": update_audit,
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "intermediate_checkpoint_selected": False,
        "chunk_commit_written": False,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
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
            "eligible_for_separate_promotion": True,
            "stop_all_later_seeds": True,
        }
    else:
        report = {
            **common,
            "status": "completed_200k_without_strict_success_no_chunk_commit",
            "strict_success_epoch": None,
            "eligible_for_separate_promotion": False,
            "stop_all_later_seeds": False,
        }
    report["learning_curve"] = {
        str(epoch): {
            "epoch": epoch,
            "physics_steps": MILESTONES[epoch][1],
            "cumulative": metric_summary(agent.update_reports[:epoch]),
            "validation": validations.get(str(epoch), {}).get("summary"),
        }
        for epoch in (0, 62, 125)
        if epoch == 0 or epoch <= len(agent.update_reports)
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "seed": args.seed,
        "strict_success_epoch": success_epoch,
        "validations": {key: row["summary"] for key, row in validations.items()},
        "curriculum_deep_fraction": (
            coverage["segments"]["epochs_63_through_125"]["all_worlds"]["fraction_k_ge_21"]
            if coverage is not None else None
        ),
        "report": artifact(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
