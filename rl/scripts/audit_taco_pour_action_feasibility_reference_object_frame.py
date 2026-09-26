#!/usr/bin/env python3
"""Gate-A reference-object-frame translation feasibility comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_corrected_endpoint44_48_failure_attribution import _actuator_metadata  # noqa: E402
from audit_taco_pour_action_feasibility_cem import (  # noqa: E402
    action_to_latent,
    latent_to_action,
)
from audit_taco_pour_action_feasibility_minimum_rho import capture_source57  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    load_contract as load_tail_contract,
)


SOURCE = 57
HORIZON = 3


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_action_feasibility_reference_object_frame_v1":
        raise ValueError("unsupported reference-object-frame contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("reference-object-frame gate is not authorized")
    runtime = contract["runtime"]
    parameterization = contract["translation_parameterization"]
    optimizer = contract["optimizer"]
    stopping = contract["stopping_rule"]
    required_runtime = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_updates_policy": False,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "reference_timing_changed": False,
        "nontranslation_action_semantics_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    if (
        any(runtime.get(key) != value for key, value in required_runtime.items())
        or contract.get("paper_faithful") is not False
        or contract["source_state"] != {
            "name": "tail_source57",
            "source_endpoint": 57,
            "provenance": "selected_tail_semantic_oracle_path_through_source56",
            "exact_gate_A0_state_reproduction_required": True,
        }
        or parameterization["normalized_action_indices"] != [0, 1, 2]
        or parameterization["input_coordinate_frame"]
        != "command_endpoint_reference_tool_frame"
        or parameterization["output_coordinate_frame"]
        != "world_frame_actuator_command"
        or parameterization["reference_index_offset_from_source"] != 1
        or parameterization["local_component_interval"] != [-1.0, 1.0]
        or parameterization["local_component_scale_m"] != 0.05
        or parameterization["world_component_hard_clip_before_ctrlrange"] is not False
        or parameterization["actuator_ctrlrange_remains_hard_constraint"] is not True
        or parameterization["invalid_ctrlrange_candidates_are_rejected_not_projected"] is not True
        or parameterization["all_other_33_dimensions_use_current_world_frame_state_feasible_support"] is not True
        or optimizer["algorithm"] != "deterministic_CEM"
        or optimizer["horizon_control_intervals"] != HORIZON
        or optimizer["population"] != 24
        or optimizer["elites"] != 6
        or optimizer["iterations"] != 5
        or optimizer["candidate_sequence_count"] != 240
        or [row["name"] for row in optimizer["restarts"]]
        != ["transformed_gate_A0_best_centered", "zero_residual_centered"]
        or optimizer["stop_on_first_feasible"] is not False
        or any(value is not False for value in stopping.values())
        or contract["frozen_decisions"] != {
            "actor_learning_rate_5e_minus_5": "blocked",
            "learned_gate": "blocked",
            "reward_change": "blocked",
            "observation_change": "blocked_pending_gate_B",
            "policy_architecture_change": "blocked_pending_gate_C",
            "PPO_retraining": "blocked_pending_gate_D",
            "chunk_commit": "blocked",
        }
    ):
        raise ValueError("reference-object-frame definition changed")
    paths: dict[str, Path] = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], np.float64)


def world_translation_ctrl_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    reference = env._reference_ctrls(env.time_indices, offset=1)
    reference, _ = env._snap_state_feasible_reference(reference)
    contract = env._state_feasible_action_contract
    if contract is None:
        raise RuntimeError("state-feasible action contract is not enabled")
    controlled = reference[0, contract["indices"]].detach().cpu().numpy().astype(np.float64)
    limited = contract["limited"].detach().cpu().numpy().astype(bool)
    safe_lower = contract["safe_lower_f32"].detach().cpu().numpy().astype(np.float64)
    safe_upper = contract["safe_upper_f32"].detach().cpu().numpy().astype(np.float64)
    extent = math.sqrt(3.0)
    low = np.full(3, -extent, np.float64)
    high = np.full(3, extent, np.float64)
    for index in range(3):
        if limited[index]:
            low[index] = max(low[index], (safe_lower[index] - controlled[index]) / 0.05)
            high[index] = min(high[index], (safe_upper[index] - controlled[index]) / 0.05)
        if low[index] > high[index]:
            raise RuntimeError("reference-object-frame world translation support is empty")
    return low, high


def support_sequence(*, backend, snapshot: dict, rotations: np.ndarray,
                     original_residual) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    lows = []
    highs = []
    translation_world_lows = []
    translation_world_highs = []
    for offset in range(HORIZON):
        low_t, high_t = backend.env.current_normalized_action_bounds()
        low = low_t[0].detach().cpu().numpy().astype(np.float64)
        high = high_t[0].detach().cpu().numpy().astype(np.float64)
        world_low, world_high = world_translation_ctrl_bounds(backend.env)
        lows.append(low)
        highs.append(high)
        translation_world_lows.append(world_low)
        translation_world_highs.append(world_high)
        backend.step(np.zeros((1, 36), np.float32), SOURCE + offset)
    backend.env.env_cfg.residual = original_residual
    return {
        "low": np.asarray(lows),
        "high": np.asarray(highs),
        "translation_world_low": np.asarray(translation_world_lows),
        "translation_world_high": np.asarray(translation_world_highs),
        "rotation_world_from_reference_tool": rotations,
    }


def latent_to_parameterized_actions(latent: np.ndarray, support: dict) -> tuple[np.ndarray, np.ndarray]:
    local_actions = np.empty_like(latent, dtype=np.float64)
    world_actions = np.empty_like(latent, dtype=np.float64)
    for offset in range(HORIZON):
        local_actions[offset, :3] = latent[offset, :3]
        local_actions[offset, 3:] = latent_to_action(
            latent[offset, 3:], support["low"][offset, 3:], support["high"][offset, 3:]
        )
        world_actions[offset] = local_actions[offset]
        world_actions[offset, :3] = (
            support["rotation_world_from_reference_tool"][offset]
            @ local_actions[offset, :3]
        )
    return local_actions.astype(np.float32), world_actions.astype(np.float32)


def rollout(*, backend, snapshot: dict, latent: np.ndarray, support: dict,
            original_residual) -> dict:
    local_actions, world_actions = latent_to_parameterized_actions(latent, support)
    below = world_actions[:, :3] < support["translation_world_low"]
    above = world_actions[:, :3] > support["translation_world_high"]
    invalid_count = int((below | above).sum())
    if invalid_count:
        return {
            "local_actions": local_actions,
            "world_actions": world_actions,
            "scores": np.full(HORIZON, 1.0e6, np.float64),
            "finite": np.ones(HORIZON, np.bool_),
            "terminated": np.ones(HORIZON, np.bool_),
            "qpos": np.zeros((HORIZON, 50), np.float32),
            "qvel": np.zeros((HORIZON, 48), np.float32),
            "invalid_actuator_bound_count": invalid_count,
            "nonfinite_endpoint_count": 0,
            "tracking_boundary_violation_count": HORIZON,
            "maximum_objective_score": 1.0e6,
            "sum_objective_score": 3.0e6,
            "feasible": False,
        }

    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    backend.env.env_cfg.residual = replace(
        original_residual, residual_clip=0.05 * math.sqrt(3.0)
    )
    scores = []
    finite_rows = []
    terminated = []
    qpos_rows = []
    qvel_rows = []
    for offset in range(HORIZON):
        backend.step(world_actions[offset][None], SOURCE + offset)
        score = float(np.asarray(backend.last_info["object_tracking_error"])[0])
        qpos = backend.env._mjwp.get_qpos(
            backend.env.ego_cfg, backend.env.env
        )[0].detach().cpu().numpy()
        qvel = backend.env._mjwp.get_qvel(
            backend.env.ego_cfg, backend.env.env
        )[0].detach().cpu().numpy()
        scores.append(score)
        finite_rows.append(bool(
            np.isfinite(score) and np.isfinite(qpos).all() and np.isfinite(qvel).all()
        ))
        terminated.append(bool(np.asarray(backend.last_info["terminated"])[0]))
        qpos_rows.append(qpos)
        qvel_rows.append(qvel)
    backend.env.env_cfg.residual = original_residual
    scores_array = np.asarray(scores, np.float64)
    nonfinite = int((~np.asarray(finite_rows, np.bool_)).sum())
    violations = int((scores_array >= 1.0).sum()) if nonfinite == 0 else HORIZON
    finite_scores = scores_array[np.isfinite(scores_array)]
    maximum = float(finite_scores.max()) if finite_scores.size else float("inf")
    total = float(finite_scores.sum()) if finite_scores.size else float("inf")
    return {
        "local_actions": local_actions,
        "world_actions": world_actions,
        "scores": scores_array,
        "finite": np.asarray(finite_rows, np.bool_),
        "terminated": np.asarray(terminated, np.bool_),
        "qpos": np.asarray(qpos_rows, np.float32),
        "qvel": np.asarray(qvel_rows, np.float32),
        "invalid_actuator_bound_count": 0,
        "nonfinite_endpoint_count": nonfinite,
        "tracking_boundary_violation_count": violations,
        "maximum_objective_score": maximum,
        "sum_objective_score": total,
        "feasible": nonfinite == 0 and violations == 0,
    }


def rank_key(row: dict) -> tuple:
    return (
        row["invalid_actuator_bound_count"],
        row["nonfinite_endpoint_count"],
        row["tracking_boundary_violation_count"],
        row["maximum_objective_score"],
        row["sum_objective_score"],
    )


def write_summary(path: Path, report: dict) -> None:
    decision = report["decision"]
    best = report["best_observed_candidate"]
    path.write_text(
        "# Gate A: reference-object-frame comparison\n\n"
        f"- feasible candidates: {decision['feasible_candidate_count']}\n"
        f"- feasible sequence found: {decision['reference_object_frame_feasible_sequence_found']}\n"
        f"- best scores: {best['scores']}\n"
        f"- best world translations: {best['world_actions'][0:3]}\n"
        f"- next blocker: {decision['next_blocker']}\n\n"
        "Only the right-wrist translation basis changed. The three local components\n"
        "use the fixed 0.05 m scale and the command-endpoint reference-tool frame.\n"
        "No frame, scale or axis sweep, policy training, acceptance or commit occurred.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_action_feasibility_reference_object_frame_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_action_feasibility_reference_object_frame_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_a0 = json.loads(paths["gate_A0_report"].read_text())
    gate_rho = json.loads(paths["gate_A_rho_report"].read_text())
    if gate_a0["decision"]["source57_current_support_feasible_sequence_found"] is not False:
        raise ValueError("frame comparison requires negative Gate A0")
    if gate_rho["decision"]["expanded_support_feasible_sequence_found"] is not False:
        raise ValueError("frame comparison requires negative Gate A-rho")
    _, parent_paths, _ = load_tail_contract(paths["tail_contract"])
    tail_report = json.loads(paths["tail_report"].read_text())

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
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    objective = load_runtime_objective(
        parent_paths["protocol"], parent_paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        parent_paths["protocol"], parent_paths["observation_profile"],
        require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(parent_paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(parent_paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        parent_paths["initialization_report"], parent_paths["simulator_config"]
    )
    config = _load_ego_config(str(parent_paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    reference_qvel = reference[1].detach().cpu().numpy().astype(np.float64)
    rotations = np.asarray([
        quaternion_wxyz_to_matrix(reference_qpos[SOURCE + offset + 1, 39:43])
        for offset in range(HORIZON)
    ])
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
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
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    actuator_names, _, _ = _actuator_metadata(
        env.env.model_cpu, tuple(env.env_cfg.residual.hand_control_indices)
    )
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(parent_paths["boundary"])
    checkpoint = load_gzip_torch(parent_paths["checkpoint"])
    temporary = tempfile.TemporaryDirectory(prefix=".reference_object_frame_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temporary.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        source = capture_source57(
            backend=backend,
            policy=policy,
            boundary=boundary,
            parent_paths=parent_paths,
            tail_report=tail_report,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        expected_state = next(
            row["source_state"] for row in gate_a0["benchmarks"]
            if row["name"] == "tail_source57"
        )
        if source["source_state"] != expected_state:
            raise RuntimeError("source57 state differs from Gate A0")
        original_residual = backend.env.env_cfg.residual
        support = support_sequence(
            backend=backend,
            snapshot=source["snapshot"],
            rotations=rotations,
            original_residual=original_residual,
        )
        a0_arrays = np.load(paths["gate_A0_best_sequences"])
        tail_index = list(a0_arrays["benchmark"]).index("tail_source57")
        a0_world = np.asarray(a0_arrays["action"][tail_index], np.float64)
        transformed_a0 = np.empty_like(a0_world)
        transformed_a0[:, :3] = np.asarray([
            rotations[offset].T @ a0_world[offset, :3]
            for offset in range(HORIZON)
        ])
        transformed_a0[:, :3] = np.clip(transformed_a0[:, :3], -1.0, 1.0)
        transformed_a0[:, 3:] = a0_world[:, 3:]
        transformed_a0_latent = np.empty_like(transformed_a0)
        transformed_a0_latent[:, :3] = transformed_a0[:, :3]
        transformed_a0_latent[:, 3:] = action_to_latent(
            transformed_a0[:, 3:], support["low"][:, 3:], support["high"][:, 3:]
        )
        zero_latent = np.empty_like(transformed_a0_latent)
        zero_latent[:, :3] = 0.0
        zero_latent[:, 3:] = action_to_latent(
            np.zeros((HORIZON, 33)), support["low"][:, 3:], support["high"][:, 3:]
        )

        optimizer = contract["optimizer"]
        dataset = {key: [] for key in (
            "restart_index", "iteration", "sample_index", "latent", "local_action",
            "world_action", "score", "finite", "terminated", "invalid", "feasible",
        )}
        global_best = None
        feasible_count = 0
        restart_reports = []
        for restart_index, restart in enumerate(optimizer["restarts"]):
            rng = np.random.default_rng(int(restart["seed"]))
            mean = (
                transformed_a0_latent.copy()
                if restart["name"] == "transformed_gate_A0_best_centered"
                else zero_latent.copy()
            )
            std = np.full_like(mean, float(optimizer["initial_standard_deviation"]))
            iteration_reports = []
            for iteration in range(int(optimizer["iterations"])):
                samples = np.clip(
                    rng.normal(mean, std, size=(int(optimizer["population"]),) + mean.shape),
                    -1.0,
                    1.0,
                )
                samples[0] = mean
                samples[1] = transformed_a0_latent
                samples[2] = zero_latent
                rows = []
                for sample_index, latent in enumerate(samples):
                    outcome = rollout(
                        backend=backend,
                        snapshot=source["snapshot"],
                        latent=latent,
                        support=support,
                        original_residual=original_residual,
                    )
                    row = {
                        **outcome,
                        "latent": latent.astype(np.float32),
                        "restart_index": restart_index,
                        "iteration": iteration,
                        "sample_index": sample_index,
                    }
                    rows.append(row)
                    feasible_count += int(row["feasible"])
                    if global_best is None or rank_key(row) < rank_key(global_best):
                        global_best = row
                    dataset["restart_index"].append(restart_index)
                    dataset["iteration"].append(iteration)
                    dataset["sample_index"].append(sample_index)
                    dataset["latent"].append(row["latent"])
                    dataset["local_action"].append(row["local_actions"])
                    dataset["world_action"].append(row["world_actions"])
                    dataset["score"].append(row["scores"])
                    dataset["finite"].append(row["finite"])
                    dataset["terminated"].append(row["terminated"])
                    dataset["invalid"].append(row["invalid_actuator_bound_count"])
                    dataset["feasible"].append(row["feasible"])
                order = sorted(range(len(rows)), key=lambda index: rank_key(rows[index]))
                elite_latent = np.asarray([
                    rows[index]["latent"] for index in order[: int(optimizer["elites"])]
                ], np.float64)
                weight = float(optimizer["elite_update_new_weight"])
                mean = (1.0 - weight) * mean + weight * elite_latent.mean(axis=0)
                std = np.maximum(
                    float(optimizer["minimum_standard_deviation"]),
                    (1.0 - weight) * std + weight * elite_latent.std(axis=0),
                )
                best_iteration = rows[order[0]]
                iteration_reports.append({
                    "iteration": iteration,
                    "feasible_sequences": int(sum(row["feasible"] for row in rows)),
                    "invalid_actuator_bound_sequences": int(sum(
                        row["invalid_actuator_bound_count"] > 0 for row in rows
                    )),
                    "best_scores": best_iteration["scores"].tolist(),
                    "best_maximum_score": best_iteration["maximum_objective_score"],
                })
                print(
                    f"reference-object-frame restart={restart['name']} iter={iteration} "
                    f"feasible={iteration_reports[-1]['feasible_sequences']} "
                    f"invalid={iteration_reports[-1]['invalid_actuator_bound_sequences']} "
                    f"best_max={best_iteration['maximum_objective_score']:.9f}",
                    flush=True,
                )
            restart_reports.append({
                "name": restart["name"],
                "seed": restart["seed"],
                "iterations": iteration_reports,
            })
        if global_best is None:
            raise RuntimeError("reference-object-frame CEM produced no candidates")
        repeats = [
            rollout(
                backend=backend,
                snapshot=source["snapshot"],
                latent=global_best["latent"],
                support=support,
                original_residual=original_residual,
            )
            for _ in range(2)
        ]
        repeatable = all(
            np.array_equal(repeats[0][key], repeats[1][key])
            for key in (
                "scores", "local_actions", "world_actions", "qpos", "qvel",
                "finite", "terminated",
            )
        )
        if not repeatable:
            raise RuntimeError("best reference-object-frame candidate is not CPU repeatable")
        backend.env.env_cfg.residual = original_residual
        policy.writer.close()
    finally:
        temporary.cleanup()

    found = bool(feasible_count)
    next_blocker = (
        "gate_A_reference_object_frame_candidate_requires_action_interface_decision"
        if found else
        "gate_A_active_corrective_parameterization_decision_required"
    )
    best = {
        "scores": global_best["scores"].tolist(),
        "local_actions": global_best["local_actions"].tolist(),
        "world_actions": global_best["world_actions"].tolist(),
        "finite": global_best["finite"].tolist(),
        "terminated": global_best["terminated"].tolist(),
        "invalid_actuator_bound_count": global_best["invalid_actuator_bound_count"],
        "tracking_boundary_violation_count": global_best["tracking_boundary_violation_count"],
        "maximum_objective_score": global_best["maximum_objective_score"],
        "sum_objective_score": global_best["sum_objective_score"],
        "feasible": global_best["feasible"],
        "bitwise_repeatable_on_CPU": repeatable,
    }
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate_A_reference_object_frame",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "policy_optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "gate_A0_source57_state_exactly_reproduced": True,
            "tail_selected_sources44_through56_exactly_reproduced": True,
            "best_candidate_bitwise_repeatable_on_CPU": repeatable,
        },
        "actuator_names": list(actuator_names),
        "source57_state": source["source_state"],
        "translation_parameterization": contract["translation_parameterization"],
        "reference_rotations_world_from_tool": rotations.tolist(),
        "support": {
            "nontranslation_low": support["low"][:, 3:].tolist(),
            "nontranslation_high": support["high"][:, 3:].tolist(),
            "translation_world_ctrl_low": support["translation_world_low"].tolist(),
            "translation_world_ctrl_high": support["translation_world_high"].tolist(),
        },
        "optimizer": optimizer,
        "restarts": restart_reports,
        "best_observed_candidate": best,
        "decision": {
            "feasible_candidate_count": feasible_count,
            "reference_object_frame_feasible_sequence_found": found,
            "finite_optimizer_failure_is_mathematical_infeasibility_proof": False,
            "action_frame_change_authorized_for_training": False,
            "additional_frame_or_scale_search_authorized": False,
            "gate_B_observation_sufficiency_allowed": False,
            "gate_C_policy_representability_allowed": False,
            "gate_D_fresh_PPO_allowed": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "learned_gate_training_authorized": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": next_blocker,
        },
        "measurement_boundaries": {
            "right_wrist_translation_frame_only_changed": True,
            "local_component_scale_remains_0_05_m": True,
            "all_other_33_dimensions_unchanged": True,
            "no_frame_scale_or_axis_sweep": True,
            "no_policy_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    candidate_path = args.output / contract["artifacts"]["candidate_dataset"]
    np.savez_compressed(
        candidate_path,
        restart_index=np.asarray(dataset["restart_index"], np.int16),
        iteration=np.asarray(dataset["iteration"], np.int16),
        sample_index=np.asarray(dataset["sample_index"], np.int16),
        latent=np.asarray(dataset["latent"], np.float32),
        local_action=np.asarray(dataset["local_action"], np.float32),
        world_action=np.asarray(dataset["world_action"], np.float32),
        score=np.asarray(dataset["score"], np.float32),
        finite=np.asarray(dataset["finite"], np.bool_),
        terminated=np.asarray(dataset["terminated"], np.bool_),
        invalid_actuator_bound_count=np.asarray(dataset["invalid"], np.int16),
        feasible=np.asarray(dataset["feasible"], np.bool_),
    )
    best_path = args.output / contract["artifacts"]["best_sequence"]
    np.savez_compressed(
        best_path,
        latent=np.asarray(global_best["latent"], np.float32),
        local_action=np.asarray(global_best["local_actions"], np.float32),
        world_action=np.asarray(global_best["world_actions"], np.float32),
        score=np.asarray(global_best["scores"], np.float32),
        qpos=np.asarray(global_best["qpos"], np.float32),
        qvel=np.asarray(global_best["qvel"], np.float32),
        feasible=np.asarray(global_best["feasible"], np.bool_),
    )
    report["artifacts"] = {
        "candidate_dataset": {
            "path": str(candidate_path.resolve()),
            "sha256": sha256(candidate_path),
        },
        "best_sequence": {
            "path": str(best_path.resolve()),
            "sha256": sha256(best_path),
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "best_observed_candidate": best,
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
