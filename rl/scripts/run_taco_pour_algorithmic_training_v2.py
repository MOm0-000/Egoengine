#!/usr/bin/env python3
"""Run one frozen CPU candidate from the replay-identity benchmark v2."""

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
import time

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
CONTRACT = ROOT / "configs/taco_pour_algorithmic_reproduction_training_v2.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_algorithmic_reproduction_training_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def first_failure(terminated: list[bool], endpoints: list[int]) -> int | None:
    return next((endpoint for endpoint, failed in zip(endpoints, terminated, strict=True) if failed), None)


def validation_summary(
    terminated: list[bool], endpoints: list[int], scores: list[float]
) -> dict[str, object]:
    failure = first_failure(terminated, endpoints)
    successful = 40 if failure is None else failure - endpoints[0]
    return {
        "successful_intervals": int(successful),
        "first_failure_endpoint": failure,
        "forty_of_forty": failure is None,
        "beats_Replay": successful > 30,
        "meaningful_refinement": successful >= 35,
        "endpoint40_score": float(scores[endpoints.index(40)]),
        "endpoint50_score": float(scores[endpoints.index(50)]),
        "endpoint60_score": float(scores[endpoints.index(60)]),
    }


def load_checkpoint(path: Path) -> dict:
    return torch.load(
        io.BytesIO(gzip.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("additional seeds are not authorized before the seed-0 trigger")

    contract = yaml.safe_load(CONTRACT.read_text())
    if (
        contract.get("schema") != "taco_pour_algorithmic_reproduction_training_benchmark_v2"
        or contract.get("status") != "authorized_staged_B0_then_A_B_training"
        or contract.get("paper_faithful") is not False
    ):
        raise ValueError("benchmark v2 is not authorized")
    b0_path = RUN_ROOT / "candidate_B/seed_0/report.json"
    b0 = json.loads(b0_path.read_text())
    if b0.get("status") != "passed_training_may_start" or not all(b0["checks"].values()):
        raise RuntimeError("exact Candidate-B B0 gate did not authorize training")
    checkpoint_gate_path = (
        ROOT / "runs/taco_pour_algorithmic_checkpoint_roundtrip_v2/report.json"
    )
    checkpoint_gate = json.loads(checkpoint_gate_path.read_text())
    if (
        checkpoint_gate.get("status") != "passed_no_optimizer_step"
        or not all(checkpoint_gate["field_equality"].values())
    ):
        raise RuntimeError("real fresh-agent checkpoint gate did not pass")

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
        validate_budget_plan,
        validate_checkpoint_payload,
        write_checkpoint,
    )
    from video_to_spider.rl.algorithmic_training import AlgorithmicBenchmarkPpoAgent
    from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen training input changed: {name}")
    expected_implementation = {
        "state_feasible_distribution_sha256": ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py",
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "benchmark_runner_sha256": ROOT / "scripts/run_taco_pour_algorithmic_reproduction_training_v2.py",
    }
    for name, path in expected_implementation.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"frozen implementation changed: {name}")
    milestones = validate_budget_plan(contract["training_budget"]["seed0_milestones"])
    milestone_by_epoch = {int(row["epoch"]): row for row in milestones}
    final_epoch = max(milestone_by_epoch)

    objective = load_runtime_objective(
        paths["protocol_at_authorization"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol_at_authorization"], paths["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(paths["action_profile"])
    distribution, distribution_report = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    distribution = replace(distribution, optimizer_training_authorized=True)
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    boundary = load_checkpoint(paths["source_boundary"])
    if (
        boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("training requires the exact corrected endpoint-20 boundary")

    # Config/reference inputs are immutable and MJWPVectorEnv derives its own
    # per-world dataclass copy. Load these large read-only artifacts once so the
    # five independent CPU worlds do not repeat identical disk parsing.
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
        verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
        world.set_env_state(boundary)
        return world

    output = RUN_ROOT / "training" / f"candidate_{args.candidate}" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "input_contracts").mkdir()
    for name, path in {
        "benchmark_contract": CONTRACT,
        "B0_report": b0_path,
        "checkpoint_gate_report": checkpoint_gate_path,
        **paths,
    }.items():
        target = output / "input_contracts" / f"{name}{''.join(path.suffixes) or '.bin'}"
        target.write_bytes(path.read_bytes())

    training_env = IndependentMJWPTrainingEnv([
        make_world(asymmetric=True, world_seed=args.seed + index)
        for index in range(4)
    ])
    training_env.set_chunk_reset(start=20, end=60)
    training_env.enable_training_trace(output / "training_visitation")
    config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=40,
            seq_length=4,
            max_epochs=final_epoch,
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
        ppo_config=config,
        network_config=_build_network_config(4),
        env=training_env,
        distribution_spec=distribution,
        audit_dir=output / "update_audit",
    )
    initialization_audit = (
        {"type": "fresh_formal_random_initialization", "seed": args.seed}
        if args.candidate == "A"
        else replay_preserving_initialization(agent, sigma_multiplier=0.25)
    )
    config_hashes = {
        "benchmark_contract": sha256(CONTRACT),
        "B0_report": sha256(b0_path),
        "checkpoint_gate_report": sha256(checkpoint_gate_path),
        "training_runner": sha256(Path(__file__)),
        "algorithmic_training_module": sha256(
            ROOT / "src/video_to_spider/rl/algorithmic_training.py"
        ),
    }
    checkpoints: dict[int, dict[str, object]] = {}
    training_completed = False
    trace = None
    update_audit = None
    started = time.time()
    try:
        agent.init_tensors()
        agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs

        def save_milestone(epoch: int, physics: int, controls: int) -> None:
            observed_controls = int(training_env.simulation_control_intervals)
            observed_physics = int(training_env.simulation_physics_steps)
            if (observed_physics, observed_controls) != (physics, controls):
                raise RuntimeError(
                    "training counters disagree at milestone "
                    f"{epoch}: {(observed_physics, observed_controls)}"
                )
            payload = build_checkpoint_payload(
                agent,
                candidate=args.candidate,
                seed=args.seed,
                simulation_physics_steps=physics,
                simulation_control_intervals=controls,
                config_hashes=config_hashes,
            )
            checkpoint_path = output / "checkpoints" / f"epoch_{epoch:04d}.pt.gz"
            checkpoints[epoch] = write_checkpoint(checkpoint_path, payload)

        save_milestone(0, 0, 0)
        while int(agent.epoch_num) < final_epoch:
            epoch = agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            agent.frame += int(agent.curr_frames)
            if epoch in milestone_by_epoch:
                row = milestone_by_epoch[epoch]
                save_milestone(
                    epoch,
                    int(row["actual_physics_steps"]),
                    int(row["actual_control_intervals"]),
                )
            if epoch == 1 or epoch % 10 == 0 or epoch in milestone_by_epoch:
                print(json.dumps({
                    "candidate": args.candidate,
                    "epoch": epoch,
                    "physics_steps": training_env.simulation_physics_steps,
                    "elapsed_seconds": time.time() - started,
                    "latest_update": agent.update_reports[-1],
                }), flush=True)
        training_completed = True
        trace = training_env.finalize_training_trace(completed=True)
        update_audit = agent.finalize_algorithmic_audit()
    except BaseException as error:
        try:
            training_env.finalize_training_trace(completed=False)
        except Exception as trace_error:
            error.add_note(f"training trace finalization also failed: {trace_error}")
        raise
    finally:
        if agent.writer is not None:
            agent.writer.close()

    if not training_completed or trace is None or update_audit is None:
        raise RuntimeError("training did not complete its fixed budget")

    # Validate only after the entire optimizer run, so validation cannot perturb
    # the training RNG, environment, recurrent state, or normalization schedule.
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
    )

    def validate(epoch: int) -> dict[str, object]:
        checkpoint_path = Path(checkpoints[epoch]["path"])
        payload = load_checkpoint(checkpoint_path)
        validate_checkpoint_payload(payload)
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
        for source in range(20, 60):
            obs = validation_agent.obs_to_tensors(validation_world.current_observation())
            result = validation_agent.get_deterministic_action_values(obs)
            validation_agent.rnn_states = result["rnn_states"]
            action = validation_agent.preprocess_actions(result["deterministic_actions"])
            _, _, _, info = validation_world.step(action, auto_reset=False)
            endpoints.append(source + 1)
            terminated.append(bool(info["terminated"][0]))
            scores.append(float(info["object_tracking_error"][0]))
            position.append(float(info["object_position_error"][0, 0]))
            rotation.append(float(info["object_rotation_error"][0, 0]))
            actions.append(np.asarray(action[0]).copy())
            actor_mu.append(result["mus"][0].detach().cpu().numpy().copy())
            actor_sigma.append(result["sigmas"][0].detach().cpu().numpy().copy())
        arrays_path = output / "validations" / f"epoch_{epoch:04d}.npz"
        arrays_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            arrays_path,
            endpoint=np.asarray(endpoints, dtype=np.int32),
            terminated=np.asarray(terminated),
            tracking_score=np.asarray(scores),
            position_error=np.asarray(position),
            rotation_error=np.asarray(rotation),
            deterministic_action=np.asarray(actions),
            actor_mu=np.asarray(actor_mu),
            actor_sigma=np.asarray(actor_sigma),
        )
        return {
            "epoch": epoch,
            "simulation_physics_steps": int(payload["simulation_physics_steps"]),
            "summary": validation_summary(terminated, endpoints, scores),
            "trajectory": artifact(arrays_path),
            "checkpoint": checkpoints[epoch],
            "chunk_commit_written": False,
        }

    validations = {str(epoch): validate(epoch) for epoch in (0, 62, 313, 625)}
    if validation_agent.writer is not None:
        validation_agent.writer.close()
    report = {
        "schema": "taco_pour_algorithmic_reproduction_training_candidate_v2",
        "status": "completed_fixed_seed0_budget_no_chunk_commit",
        "classification": "local_algorithmic_reproduction_after_replay_identity_fix",
        "paper_faithful": False,
        "candidate": args.candidate,
        "seed": args.seed,
        "backend": "CPU_MuJoCo_Warp_training_and_validation",
        "fresh_actor_critic_optimizers_RMS": True,
        "warm_start_used": False,
        "initialization": initialization_audit,
        "training": {
            "worlds": 4,
            "epochs": final_epoch,
            "actor_updates": final_epoch,
            "critic_updates": final_epoch * 4,
            "simulation_physics_steps": training_env.simulation_physics_steps,
            "simulation_control_intervals": training_env.simulation_control_intervals,
            "wall_seconds": time.time() - started,
            "actor_learning_rate": 1.0e-4,
            "actor_mini_epochs": 1,
            "critic_mini_epochs": 4,
        },
        "residual_action": residual_report,
        "action_distribution": distribution_report,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "validations": validations,
        "training_visitation": trace,
        "update_audit": update_audit,
        "B0_report": artifact(b0_path),
        "checkpoint_infrastructure_gate": artifact(checkpoint_gate_path),
        "chunk_commit_written": False,
        "conditional_multiseed_triggered": any(
            row["summary"]["successful_intervals"] > 30
            for epoch, row in validations.items() if epoch == "625"
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "candidate": args.candidate,
        "validations": {key: value["summary"] for key, value in validations.items()},
        "report": artifact(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
