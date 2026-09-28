#!/usr/bin/env python3
"""Replay-identity repair and staged Pour A/B benchmark v3 runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def relocated_artifact(path: Path, final_path: Path) -> dict[str, object]:
    return {
        "path": str(final_path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _first_failure(terminated: list[bool], endpoints: list[int]) -> int | None:
    return next((endpoint for endpoint, failed in zip(endpoints, terminated, strict=True) if failed), None)


def _validation_summary(terminated: list[bool], endpoints: list[int], scores: list[float]) -> dict[str, object]:
    failure = _first_failure(terminated, endpoints)
    successful = 40 if failure is None else failure - endpoints[0]
    return {
        "successful_intervals": int(successful),
        "first_failure_endpoint": failure,
        "forty_of_forty": failure is None,
        "endpoint40_score": float(scores[endpoints.index(40)]),
        "endpoint50_score": float(scores[endpoints.index(50)]),
        "endpoint60_score": float(scores[endpoints.index(60)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_algorithmic_reproduction_training_v3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/taco_pour_algorithmic_reproduction_training_v3",
    )
    parser.add_argument("--stage", choices=("B0", "train"), default="B0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = yaml.safe_load(args.contract.read_text())
    schema = str(contract.get("schema", ""))
    version = schema.rsplit("_v", 1)[-1] if "_v" in schema else ""
    if (
        schema not in {
            "taco_pour_algorithmic_reproduction_training_benchmark_v3",
            "taco_pour_algorithmic_reproduction_training_benchmark_v4",
        }
        or contract.get("status") != "authorized_fresh_B0_roundtrip_then_A_B_training"
        or contract.get("paper_faithful") is not False
        or Path(contract.get("output_directory", "")) != args.output_dir.resolve()
    ):
        raise ValueError("algorithmic benchmark contract is not authorized")
    if args.stage == "train":
        raise RuntimeError(
            "training is fail-closed until a separately produced B0 report passes; "
            "run --stage B0 first"
        )

    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.algorithmic_benchmark import (
        replay_preserving_initialization,
        validate_budget_plan,
    )
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen benchmark input changed: {name}")
    validate_budget_plan(contract["training_budget"]["seed0_milestones"])
    implementation = contract["implementation_contract"]
    expected_implementation = {
        "state_feasible_distribution_sha256": ROOT / "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "official_PPO_agent_sha256": ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py",
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "benchmark_runner_sha256": Path(__file__).resolve(),
    }
    for key, path in expected_implementation.items():
        if sha256(path) != implementation[key]:
            raise ValueError(f"benchmark implementation changed: {key}")

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
    gate_row = contract["quality_gates"]["pre_optimizer_likelihood_identity"]
    if version == "4":
        identity_gate = LikelihoodIdentityGateSpec.float32_ulp_aware_v1()
        canonical_gate = CanonicalOldPolicyGateSpec(
            rollout_to_canonical_ratio_atol=float(
                gate_row["rollout_to_canonical_ratio_atol"]
            ),
            canonical_ratio_atol=float(gate_row["canonical_ratio_atol"]),
            semantic_identity_hard_fail=float(
                gate_row["semantic_identity_hard_fail"]
            ),
        )
        canonical_gate.validate()
    else:
        identity_gate = LikelihoodIdentityGateSpec(
            mu_atol=float(gate_row["mu_atol"]),
            sigma_atol=float(gate_row["sigma_atol"]),
            ratio_atol=float(gate_row["ratio_atol"]),
            semantic_identity_hard_fail=float(
                gate_row["semantic_identity_hard_fail"]
            ),
        )
        identity_gate.validate()
        canonical_gate = None
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    boundary = torch.load(
        io.BytesIO(gzip.decompress(paths["source_boundary"].read_bytes())),
        map_location="cpu", weights_only=False,
    )
    if (
        boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("benchmark requires the exact corrected endpoint-20 boundary")

    def make_world() -> MJWPVectorEnv:
        config = _load_ego_config(str(paths["simulator_config"]), "cpu")
        reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
        world = MJWPVectorEnv(
            config,
            reference,
            num_envs=1,
            env_config=MJWPVectorEnvConfig(
                reference_start_index=0,
                asymmetric_critic=False,
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
        world.set_env_state(boundary)
        return world

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_env = make_world()
    candidate_env = make_world()
    ppo_config = _build_ppo_config(
        num_envs=1,
        horizon_length=40,
        seq_length=4,
        max_epochs=1,
        learning_rate=1.0e-4,
        device="cpu",
        asymmetric_critic=None,
        actor_mini_epochs=1,
    )
    ppo_config = replace(ppo_config, clip_actions=False)
    temporary_agent_dir = Path(tempfile.mkdtemp(prefix=".benchmark_B0_agent_", dir=ROOT / "runs"))
    agent = StateFeasibleTruncatedGaussianPpoAgent(
        experiment_dir=temporary_agent_dir,
        ppo_config=ppo_config,
        network_config=_build_network_config(4),
        env=candidate_env,
        distribution_spec=distribution,
        likelihood_identity_gate=identity_gate,
        canonical_old_policy_gate=canonical_gate,
    )
    initialization_audit = replay_preserving_initialization(
        agent, sigma_multiplier=0.25
    )
    agent.set_eval()
    agent.rnn_states = [
        state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
    ]

    endpoints: list[int] = []
    replay_terminated: list[bool] = []
    candidate_terminated: list[bool] = []
    replay_scores: list[float] = []
    candidate_scores: list[float] = []
    replay_ctrl: list[np.ndarray] = []
    candidate_ctrl: list[np.ndarray] = []
    replay_qpos: list[np.ndarray] = []
    candidate_qpos: list[np.ndarray] = []
    replay_qvel: list[np.ndarray] = []
    candidate_qvel: list[np.ndarray] = []
    actor_mu: list[np.ndarray] = []
    actor_sigma: list[np.ndarray] = []
    deterministic_action: list[np.ndarray] = []
    action_low: list[np.ndarray] = []
    action_high: list[np.ndarray] = []
    replay_position: list[float] = []
    replay_rotation: list[float] = []
    candidate_position: list[float] = []
    candidate_rotation: list[float] = []
    contacts: list[np.ndarray] = []
    replay_contacts: list[np.ndarray] = []

    for source in range(20, 60):
        _, _, _, replay_info = replay_env.step(
            np.zeros((1, 36), dtype=np.float32), auto_reset=False
        )
        observation_tensor = agent.obs_to_tensors(candidate_env.current_observation())
        output = agent.get_deterministic_action_values(observation_tensor)
        agent.rnn_states = output["rnn_states"]
        action = agent.preprocess_actions(output["deterministic_actions"])
        _, _, _, candidate_info = candidate_env.step(action, auto_reset=False)

        endpoint = source + 1
        endpoints.append(endpoint)
        replay_terminated.append(bool(replay_info["terminated"][0]))
        candidate_terminated.append(bool(candidate_info["terminated"][0]))
        replay_scores.append(float(replay_info["object_tracking_error"][0]))
        candidate_scores.append(float(candidate_info["object_tracking_error"][0]))
        replay_position.append(float(replay_info["object_position_error"][0, 0]))
        replay_rotation.append(float(replay_info["object_rotation_error"][0, 0]))
        candidate_position.append(float(candidate_info["object_position_error"][0, 0]))
        candidate_rotation.append(float(candidate_info["object_rotation_error"][0, 0]))
        replay_contacts.append(np.asarray(replay_info["contact_flags"][0]).copy())
        contacts.append(np.asarray(candidate_info["contact_flags"][0]).copy())
        replay_ctrl.append(replay_env._last_ctrl[0].detach().cpu().numpy().copy())
        candidate_ctrl.append(candidate_env._last_ctrl[0].detach().cpu().numpy().copy())
        replay_qpos.append(
            replay_env._mjwp.get_qpos(replay_env.ego_cfg, replay_env.env)[0]
            .detach().cpu().numpy().copy()
        )
        candidate_qpos.append(
            candidate_env._mjwp.get_qpos(candidate_env.ego_cfg, candidate_env.env)[0]
            .detach().cpu().numpy().copy()
        )
        replay_qvel.append(
            replay_env._mjwp.get_qvel(replay_env.ego_cfg, replay_env.env)[0]
            .detach().cpu().numpy().copy()
        )
        candidate_qvel.append(
            candidate_env._mjwp.get_qvel(candidate_env.ego_cfg, candidate_env.env)[0]
            .detach().cpu().numpy().copy()
        )
        actor_mu.append(output["mus"][0].detach().cpu().numpy().copy())
        actor_sigma.append(output["sigmas"][0].detach().cpu().numpy().copy())
        deterministic_action.append(np.asarray(action[0]).copy())
        action_low.append(output["action_lows"][0].detach().cpu().numpy().copy())
        action_high.append(output["action_highs"][0].detach().cpu().numpy().copy())

    arrays = {
        "endpoint": np.asarray(endpoints, dtype=np.int32),
        "replay_terminated": np.asarray(replay_terminated),
        "candidate_B_terminated": np.asarray(candidate_terminated),
        "replay_score": np.asarray(replay_scores),
        "candidate_B_score": np.asarray(candidate_scores),
        "replay_position_error": np.asarray(replay_position),
        "replay_rotation_error": np.asarray(replay_rotation),
        "candidate_B_position_error": np.asarray(candidate_position),
        "candidate_B_rotation_error": np.asarray(candidate_rotation),
        "replay_ctrl": np.asarray(replay_ctrl),
        "candidate_B_ctrl": np.asarray(candidate_ctrl),
        "replay_qpos": np.asarray(replay_qpos),
        "candidate_B_qpos": np.asarray(candidate_qpos),
        "replay_qvel": np.asarray(replay_qvel),
        "candidate_B_qvel": np.asarray(candidate_qvel),
        "actor_mu": np.asarray(actor_mu),
        "actor_sigma": np.asarray(actor_sigma),
        "deterministic_action": np.asarray(deterministic_action),
        "action_low": np.asarray(action_low),
        "action_high": np.asarray(action_high),
        "replay_contact_flags": np.asarray(replay_contacts),
        "candidate_B_contact_flags": np.asarray(contacts),
    }
    replay_summary = _validation_summary(replay_terminated, endpoints, replay_scores)
    candidate_summary = _validation_summary(
        candidate_terminated, endpoints, candidate_scores
    )
    command_delta = np.abs(arrays["candidate_B_ctrl"] - arrays["replay_ctrl"])
    qpos_delta = np.abs(arrays["candidate_B_qpos"] - arrays["replay_qpos"])
    qvel_delta = np.abs(arrays["candidate_B_qvel"] - arrays["replay_qvel"])
    differing_rows = np.flatnonzero(np.any(command_delta != 0.0, axis=1))
    step0_contract = contract["candidates"]["B"]["step0_gate"]
    checks = {
        "mean_residual_within_tolerance": bool(
            np.max(np.abs(arrays["actor_mu"]))
            <= float(step0_contract["mean_max_abs_tolerance"])
        ),
        "deterministic_action_within_tolerance": bool(
            np.max(np.abs(arrays["deterministic_action"]))
            <= float(step0_contract["deterministic_action_max_abs_tolerance"])
        ),
        "deterministic_command_bitwise_equals_formal_Replay": bool(
            arrays["candidate_B_ctrl"].tobytes() == arrays["replay_ctrl"].tobytes()
        ),
        "candidate_qpos_bitwise_equals_formal_Replay": bool(
            arrays["candidate_B_qpos"].tobytes() == arrays["replay_qpos"].tobytes()
        ),
        "candidate_qvel_bitwise_equals_formal_Replay": bool(
            arrays["candidate_B_qvel"].tobytes() == arrays["replay_qvel"].tobytes()
        ),
        "tracking_score_bitwise_equals_formal_Replay": bool(
            arrays["candidate_B_score"].tobytes() == arrays["replay_score"].tobytes()
        ),
        "termination_sequence_equals_formal_Replay": bool(
            np.array_equal(
                arrays["candidate_B_terminated"], arrays["replay_terminated"]
            )
        ),
        "contact_flags_equal_formal_Replay": bool(
            np.array_equal(
                arrays["candidate_B_contact_flags"], arrays["replay_contact_flags"]
            )
        ),
        "formal_Replay_is_30_of_40_fail51": bool(
            replay_summary["successful_intervals"] == 30
            and replay_summary["first_failure_endpoint"] == 51
        ),
        "candidate_B_step0_is_30_of_40_fail51": bool(
            candidate_summary["successful_intervals"] == 30
            and candidate_summary["first_failure_endpoint"] == 51
        ),
    }
    passed = all(checks.values())

    temporary_output = Path(tempfile.mkdtemp(prefix=".algorithmic_benchmark_", dir=ROOT / "runs"))
    try:
        shutil.copy2(args.contract, temporary_output / "contract.yaml")
        b0 = temporary_output / "candidate_B" / "seed_0"
        b0.mkdir(parents=True)
        arrays_path = b0 / "step0_trajectory.npz"
        np.savez_compressed(arrays_path, **arrays)
        b0_report = {
            "schema": f"taco_pour_replay_preserving_candidate_B_step0_gate_v{version}",
            "status": "passed_training_may_start" if passed else "failed_training_forbidden",
            "paper_faithful": False,
            "backend": "CPU_MuJoCo_Warp",
            "optimizer_updates": 0,
            "simulation_physics_steps_for_training": 0,
            "initialization": initialization_audit,
            "replay": replay_summary,
            "candidate_B_step0": candidate_summary,
            "checks": checks,
            "maximum_abs_actor_mu": float(np.max(np.abs(arrays["actor_mu"]))),
            "maximum_abs_deterministic_action": float(
                np.max(np.abs(arrays["deterministic_action"]))
            ),
            "maximum_abs_command_difference": float(command_delta.max()),
            "command_rows_differing": int(len(differing_rows)),
            "first_differing_command_endpoint": (
                None if not len(differing_rows)
                else int(arrays["endpoint"][differing_rows[0]])
            ),
            "maximum_abs_qpos_difference": float(qpos_delta.max()),
            "maximum_abs_qvel_difference": float(qvel_delta.max()),
            "mechanism": (
                "The repaired action contract always executes the original formal "
                "reference plus residual. Bound construction may preserve but never "
                "worsen a machine-scale baseline ctrlrange violation; it cannot alter "
                "the executed base command."
            ),
            "trajectory": relocated_artifact(
                arrays_path,
                args.output_dir / "candidate_B/seed_0/step0_trajectory.npz",
            ),
            "training_A_or_B_started": False,
            "chunk_commit_written": False,
            "likelihood_identity_gate": gate_row,
        }
        b0_report_path = b0 / "report.json"
        b0_report_path.write_text(json.dumps(b0_report, indent=2) + "\n")

        a0 = temporary_output / "candidate_A" / "seed_0"
        a0.mkdir(parents=True)
        (a0 / "report.json").write_text(json.dumps({
            "schema": f"taco_pour_algorithmic_candidate_A_seed0_v{version}",
            "status": "not_started_due_candidate_B_step0_gate_failure" if not passed else "authorized_not_started",
            "optimizer_updates": 0,
            "simulation_physics_steps_for_training": 0,
            "chunk_commit_written": False,
        }, indent=2) + "\n")

        comparison = {
            "schema": f"taco_pour_algorithmic_reproduction_training_comparison_v{version}",
            "status": "B0_failed_no_training" if not passed else "B0_passed_training_authorized",
            "Replay_validated_intervals": replay_summary["successful_intervals"],
            "Replay_first_failure_endpoint": replay_summary["first_failure_endpoint"],
            "A_at_100k": None,
            "A_at_500k": None,
            "A_at_1M": None,
            "B_at_0": candidate_summary["successful_intervals"],
            "B_at_100k": None,
            "B_at_500k": None,
            "B_at_1M": None,
            "best_candidate": None,
            "first_budget_exceeding_Replay": None,
            "best_validated_intervals": replay_summary["successful_intervals"],
            "undertraining_hypothesis_supported": None,
            "replay_preserving_initialization_hypothesis_supported": False if not passed else None,
            "structured_architecture_experiment_authorized": False,
            "training_started": False,
            "failure_reason": None if passed else "Candidate B zero-step does not reproduce formal Replay",
            "candidate_B_step0_report": relocated_artifact(
                b0_report_path,
                args.output_dir / "candidate_B/seed_0/report.json",
            ),
        }
        comparison_path = temporary_output / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
        (temporary_output / "summary.md").write_text(
            f"# TACO Pour Algorithmic Reproduction Training Benchmark v{version}\n\n"
            "Candidate B's mandatory zero-step gate was evaluated before any "
            "optimizer update.\n\n"
            f"- Formal Replay: `{replay_summary['successful_intervals']}/40`, "
            f"first failure endpoint `{replay_summary['first_failure_endpoint']}`.\n"
            f"- Candidate B at zero physics training steps: "
            f"`{candidate_summary['successful_intervals']}/40`, first failure "
            f"endpoint `{candidate_summary['first_failure_endpoint']}`.\n"
            f"- Maximum command difference: `{float(command_delta.max()):.9g}`.\n"
            f"- Maximum qpos difference: `{float(qpos_delta.max()):.9g}`.\n\n"
            + (
                "The B0 gate passed; A/B training is authorized by this stage.\n"
                if passed else
                "The B0 gate failed. Per the frozen contract, Candidate A and B "
                "training did not start, no checkpoint was produced, and no chunk "
                "was committed.\n"
            )
        )
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        temporary_output.replace(args.output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        if agent.writer is not None:
            agent.writer.close()
        shutil.rmtree(temporary_agent_dir, ignore_errors=True)

    print(json.dumps({
        "status": "B0_passed" if passed else "B0_failed_training_forbidden",
        "Replay": replay_summary,
        "Candidate_B_step0": candidate_summary,
        "checks": checks,
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
