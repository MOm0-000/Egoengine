#!/usr/bin/env python3
"""Final Gate-A local controllability and structured-feedback audit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.optimize import lsq_linear
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_corrected_endpoint44_48_failure_attribution import _actuator_metadata  # noqa: E402
from audit_taco_pour_action_feasibility_minimum_rho import capture_source57  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    load_contract as load_tail_contract,
)


SOURCE = 57
HORIZON = 3
TOOL_QPOS = slice(36, 43)
TOOL_QVEL = slice(36, 42)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_action_feasibility_gate_A1_v1":
        raise ValueError("unsupported Gate A1 contract")
    if contract.get("status") != "authorized_final_read_only_gate_A":
        raise ValueError("Gate A1 is not authorized")
    runtime = contract["runtime"]
    stage1 = contract["stage_1_local_controllability"]
    stage2 = contract["stage_2_structured_feedback"]
    optimizer = stage2["optimizer"]
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
        "action_frame_changed": False,
        "current_action_support_only": True,
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
        or stage1["new_action_search_allowed"] is not False
        or stage1["replay_saved_first_actions_only"] is not True
        or stage1["regression"]["model"] != "affine_ridge"
        or stage1["regression"]["fixed_ridge_coefficient"] != 1.0e-6
        or stage1["datasets"][0]["response_replay_residual_clip"] != {
            "method": "fixed_current_support",
            "value_m": 0.05,
        }
        or stage1["datasets"][1]["response_replay_residual_clip"] != {
            "method": "exactly_regenerate_float64_rho_from_parent_CEM_provenance",
            "equation": "residual_clip_m = 0.05 * rho_float64",
            "validation": "all_512_regenerated_rho_values_round_to_saved_float32_rho",
        }
        or stage1["constrained_QP"]["model_source"] != "current_support_only"
        or stage1["constrained_QP"]["exact_simulator_validation_actions"] != 1
        or stage2["execute_only_if_stage_1_QP_exact_replay_fails"] is not True
        or stage2["feedback_applies_sources"] != [57, 58, 59]
        or stage2["gains"]["kp_interval"] != [0.0, 1.0]
        or stage2["gains"]["kv_interval_seconds"] != [0.0, 0.03333333333333333]
        or optimizer["algorithm"] != "deterministic_two_parameter_CEM"
        or optimizer["population"] != 16
        or optimizer["elites"] != 4
        or optimizer["iterations"] != 6
        or optimizer["candidate_count"] != 192
        or optimizer["stop_on_first_feasible"] is not False
        or contract["stopping_rule"] != {
            "Gate_A_closes_after_this_contract": True,
            "additional_axis_gain_frame_rho_or_per_source_search_allowed": False,
            "pure_learned_suppression_gate_training_allowed": False,
            "half_actor_learning_rate_training_allowed": False,
            "policy_training_allowed": False,
            "chunk_commit_allowed": False,
        }
        or contract["frozen_decisions"] != {
            "actor_learning_rate_5e_minus_5": "blocked",
            "learned_suppression_gate": "blocked",
            "reward_change": "blocked",
            "observation_change": "blocked_until_Gate_A1_classification",
            "policy_architecture_change": "blocked_until_Gate_A1_classification",
            "PPO_retraining": "blocked",
            "chunk_commit": "blocked",
        }
    ):
        raise ValueError("Gate A1 definition changed")
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


def read_physics_state(env) -> tuple[np.ndarray, np.ndarray]:
    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy()
    qvel = env._mjwp.get_qvel(env.ego_cfg, env.env)[0].detach().cpu().numpy()
    return qpos, qvel


def replay_first_action(*, backend, snapshot: dict, action: np.ndarray,
                        residual_clip: float, original_residual) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    backend.env.env_cfg.residual = replace(original_residual, residual_clip=residual_clip)
    backend.step(np.asarray(action, np.float32)[None], SOURCE)
    qpos, qvel = read_physics_state(backend.env)
    score = float(np.asarray(backend.last_info["object_tracking_error"])[0])
    terminated = bool(np.asarray(backend.last_info["terminated"])[0])
    backend.env.env_cfg.residual = original_residual
    return {
        "score": score,
        "terminated": terminated,
        "qpos": qpos,
        "qvel": qvel,
        "finite": bool(
            np.isfinite(score) and np.isfinite(qpos).all() and np.isfinite(qvel).all()
        ),
    }


def collect_saved_responses(*, name: str, actions: np.ndarray, saved_scores: np.ndarray,
                            restart: np.ndarray, iteration: np.ndarray,
                            sample: np.ndarray, rho: np.ndarray,
                            response_replay_residual_clip_m: np.ndarray,
                            backend, snapshot: dict, source_position: np.ndarray,
                            original_residual) -> dict:
    clips = np.asarray(response_replay_residual_clip_m, np.float64)
    if clips.shape != (len(actions),) or not np.isfinite(clips).all():
        raise RuntimeError(f"invalid response replay clips for {name}")
    responses = []
    score_reproduced = []
    for index, action in enumerate(actions):
        result = replay_first_action(
            backend=backend,
            snapshot=snapshot,
            action=action,
            residual_clip=float(clips[index]),
            original_residual=original_residual,
        )
        expected = np.float32(saved_scores[index])
        actual = np.float32(result["score"])
        score_reproduced.append(bool(actual == expected))
        responses.append(np.concatenate([
            np.asarray(result["qpos"][36:39], np.float64) - source_position,
            np.asarray(result["qvel"][36:39], np.float64),
        ]))
        if (index + 1) % 64 == 0 or index + 1 == len(actions):
            print(f"response-replay dataset={name} {index + 1}/{len(actions)}", flush=True)
    if not all(score_reproduced):
        bad = np.flatnonzero(~np.asarray(score_reproduced))
        raise RuntimeError(f"saved first-step scores changed in {name}: {bad[:8].tolist()}")
    return {
        "name": name,
        "action": np.asarray(actions, np.float64),
        "response": np.asarray(responses, np.float64),
        "saved_score": np.asarray(saved_scores, np.float64),
        "restart_index": np.asarray(restart, np.int16),
        "iteration": np.asarray(iteration, np.int16),
        "sample_index": np.asarray(sample, np.int16),
        "rho": np.asarray(rho, np.float64),
        "response_replay_residual_clip_m": clips,
        "first_step_scores_bitwise_reproduced_as_float32": True,
    }


def regenerate_parent_float64_rho(parent_contract: dict, arrays) -> np.ndarray:
    """Recover the exact rho used by the hash-bound parent CEM rollouts."""
    optimizer = parent_contract["optimizer"]
    support = parent_contract["support_parameter"]
    population = int(optimizer["population"])
    horizon = int(optimizer["horizon_control_intervals"])
    action_dim = int(arrays["action"].shape[-1])
    lower = float(support["lower"])
    upper = float(support["upper"])
    weight = float(optimizer["elite_update_new_weight"])
    minimum = float(optimizer["minimum_standard_deviation"])
    exact: list[float] = []
    base = 0

    def rank_key(index: int, rho_value: float) -> tuple:
        scores = np.asarray(arrays["score"][index], np.float64)
        finite = np.asarray(arrays["finite"][index], bool)
        nonfinite = int((~finite).sum())
        violations = int((scores >= 1.0).sum()) if nonfinite == 0 else horizon
        finite_scores = scores[np.isfinite(scores)]
        maximum = float(finite_scores.max()) if finite_scores.size else float("inf")
        total = float(finite_scores.sum()) if finite_scores.size else float("inf")
        if bool(arrays["feasible"][index]):
            return (0, rho_value, maximum, total)
        return (1, nonfinite, violations, maximum, total, rho_value)

    for restart_index, restart in enumerate(optimizer["restarts"]):
        rng = np.random.default_rng(int(restart["seed"]))
        mean_rho = 2.0 * (float(restart["initial_rho"]) - lower) / (upper - lower) - 1.0
        std_rho = float(optimizer["initial_rho_latent_standard_deviation"])
        for iteration_index in range(int(optimizer["iterations"])):
            # The parent drew action normals before rho. Their values are not
            # needed here, but advancing the same RNG stream is required.
            rng.normal(size=(population, horizon, action_dim))
            rho_latent = np.clip(
                rng.normal(mean_rho, std_rho, size=population), -1.0, 1.0
            )
            rho_latent[0] = mean_rho
            rho_latent[1] = -1.0
            rho_latent[2] = -1.0
            rho_values = lower + 0.5 * (rho_latent + 1.0) * (upper - lower)

            indices = np.arange(base, base + population)
            if not (
                np.all(arrays["restart_index"][indices] == restart_index)
                and np.all(arrays["iteration"][indices] == iteration_index)
                and np.array_equal(arrays["sample_index"][indices], np.arange(population))
            ):
                raise RuntimeError("parent rho candidate ordering changed")
            exact.extend(rho_values.tolist())
            order = sorted(
                range(population),
                key=lambda local: rank_key(int(indices[local]), float(rho_values[local])),
            )
            elite_rho = rho_latent[order[: int(optimizer["elites"])]]
            mean_rho = float((1.0 - weight) * mean_rho + weight * elite_rho.mean())
            std_rho = float(max(
                minimum,
                (1.0 - weight) * std_rho + weight * elite_rho.std(),
            ))
            base += population

    reconstructed = np.asarray(exact, np.float64)
    if base != len(arrays["rho"]) or not np.array_equal(
        reconstructed.astype(np.float32), np.asarray(arrays["rho"], np.float32)
    ):
        raise RuntimeError("float64 parent rho provenance reconstruction failed")
    return reconstructed


def split_final_iteration_by_restart(restart: np.ndarray, iteration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test = np.zeros(len(restart), dtype=bool)
    for restart_index in np.unique(restart):
        mask = restart == restart_index
        test |= mask & (iteration == iteration[mask].max())
    return ~test, test


def fit_affine_ridge(dataset: dict, ridge: float) -> tuple[dict, dict]:
    x = dataset["action"]
    y = dataset["response"]
    train, test = split_final_iteration_by_restart(
        dataset["restart_index"], dataset["iteration"]
    )

    def fit(x_fit: np.ndarray, y_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_mean = x_fit.mean(axis=0)
        y_mean = y_fit.mean(axis=0)
        centered_x = x_fit - x_mean
        centered_y = y_fit - y_mean
        weights = np.linalg.solve(
            centered_x.T @ centered_x + ridge * np.eye(x_fit.shape[1]),
            centered_x.T @ centered_y,
        )
        intercept = y_mean - x_mean @ weights
        return weights, intercept, y_mean

    weights, intercept, baseline = fit(x[train], y[train])
    prediction = intercept + x[test] @ weights
    baseline_prediction = np.broadcast_to(baseline, prediction.shape)

    def metrics(columns: slice) -> dict:
        residual = prediction[:, columns] - y[test, columns]
        baseline_residual = baseline_prediction[:, columns] - y[test, columns]
        squared = float(np.square(residual).sum())
        baseline_squared = float(np.square(baseline_residual).sum())
        return {
            "RMSE": float(np.sqrt(np.square(residual).mean())),
            "train_mean_baseline_RMSE": float(np.sqrt(np.square(baseline_residual).mean())),
            "R2_vs_train_mean": float(1.0 - squared / baseline_squared),
        }

    heldout = {
        "train_rows": int(train.sum()),
        "held_out_rows": int(test.sum()),
        "held_out_rule": "final_CEM_iteration_of_each_restart",
        "position": metrics(slice(0, 3)),
        "velocity": metrics(slice(3, 6)),
    }
    heldout["informative"] = bool(
        heldout["position"]["RMSE"]
        < heldout["position"]["train_mean_baseline_RMSE"]
        and heldout["velocity"]["RMSE"]
        < heldout["velocity"]["train_mean_baseline_RMSE"]
    )
    all_weights, all_intercept, _ = fit(x, y)
    model = {
        "weights": all_weights,
        "intercept": all_intercept,
        "ridge_coefficient": ridge,
    }
    return heldout, model


def solve_current_support_QP(*, model: dict, low: np.ndarray, high: np.ndarray,
                             target_position_delta: np.ndarray) -> dict:
    position_matrix = model["weights"][:, :3].T
    rhs = target_position_delta - model["intercept"][:3]
    solution = lsq_linear(
        position_matrix,
        rhs,
        bounds=(np.asarray(low, np.float64), np.asarray(high, np.float64)),
        method="trf",
        tol=1.0e-12,
        lsmr_tol=1.0e-12,
        max_iter=1000,
    )
    action = np.asarray(solution.x, np.float32)
    predicted = model["intercept"] + action.astype(np.float64) @ model["weights"]
    return {
        "action": action,
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "solver_cost": float(solution.cost),
        "predicted_response": predicted,
        "predicted_position_error_m": float(np.linalg.norm(
            predicted[:3] - target_position_delta
        )),
    }


def gain_to_latent(kp: float, kv: float, kp_bounds: tuple[float, float],
                   kv_bounds: tuple[float, float]) -> np.ndarray:
    return np.asarray([
        2.0 * (kp - kp_bounds[0]) / (kp_bounds[1] - kp_bounds[0]) - 1.0,
        2.0 * (kv - kv_bounds[0]) / (kv_bounds[1] - kv_bounds[0]) - 1.0,
    ])


def latent_to_gain(latent: np.ndarray, kp_bounds: tuple[float, float],
                   kv_bounds: tuple[float, float]) -> tuple[float, float]:
    kp = kp_bounds[0] + 0.5 * (float(latent[0]) + 1.0) * (kp_bounds[1] - kp_bounds[0])
    kv = kv_bounds[0] + 0.5 * (float(latent[1]) + 1.0) * (kv_bounds[1] - kv_bounds[0])
    return kp, kv


def rollout_feedback(*, backend, policy, snapshot: dict, hidden, kp: float, kv: float,
                     reference_qpos: np.ndarray, reference_qvel: np.ndarray) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    policy.rnn_states = clone_hidden(hidden)
    scores = []
    finite_rows = []
    terminated = []
    actor_actions = []
    actions = []
    physical_feedback = []
    position_errors = []
    velocity_errors = []
    projected_components = []
    qpos_rows = []
    qvel_rows = []
    for offset in range(HORIZON):
        source = SOURCE + offset
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        actor_action = np.asarray(
            policy.preprocess_actions(result["deterministic_actions"])[0], np.float32
        )
        low = result["action_lows"][0].detach().cpu().numpy()
        high = result["action_highs"][0].detach().cpu().numpy()
        qpos, qvel = read_physics_state(backend.env)
        position_error = reference_qpos[source + 1, 36:39] - qpos[36:39]
        velocity_error = reference_qvel[source + 1, 36:39] - qvel[36:39]
        feedback = kp * position_error + kv * velocity_error
        candidate = actor_action.copy()
        before_projection = candidate[:3].astype(np.float64) + feedback / 0.05
        candidate[:3] = np.clip(before_projection, low[:3], high[:3]).astype(np.float32)
        projected = ~np.isclose(candidate[:3], before_projection, rtol=0.0, atol=1.0e-7)
        backend.step(candidate[None], source)
        next_qpos, next_qvel = read_physics_state(backend.env)
        score = float(np.asarray(backend.last_info["object_tracking_error"])[0])
        finite = bool(
            np.isfinite(score)
            and np.isfinite(next_qpos).all()
            and np.isfinite(next_qvel).all()
        )
        scores.append(score)
        finite_rows.append(finite)
        terminated.append(bool(np.asarray(backend.last_info["terminated"])[0]))
        actor_actions.append(actor_action)
        actions.append(candidate)
        physical_feedback.append(feedback)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        projected_components.append(projected)
        qpos_rows.append(next_qpos)
        qvel_rows.append(next_qvel)
    score_array = np.asarray(scores, np.float64)
    nonfinite = int((~np.asarray(finite_rows, np.bool_)).sum())
    violations = int((score_array >= 1.0).sum()) if nonfinite == 0 else HORIZON
    finite_scores = score_array[np.isfinite(score_array)]
    maximum = float(finite_scores.max()) if finite_scores.size else float("inf")
    total = float(finite_scores.sum()) if finite_scores.size else float("inf")
    return {
        "kp": float(kp),
        "kv_seconds": float(kv),
        "scores": score_array,
        "finite": np.asarray(finite_rows, np.bool_),
        "terminated": np.asarray(terminated, np.bool_),
        "actor_actions": np.asarray(actor_actions, np.float32),
        "actions": np.asarray(actions, np.float32),
        "physical_feedback_m": np.asarray(physical_feedback, np.float32),
        "position_error_m": np.asarray(position_errors, np.float32),
        "velocity_error_m_s": np.asarray(velocity_errors, np.float32),
        "projected_translation_components": np.asarray(projected_components, np.bool_),
        "qpos": np.asarray(qpos_rows, np.float32),
        "qvel": np.asarray(qvel_rows, np.float32),
        "nonfinite_endpoint_count": nonfinite,
        "tracking_boundary_violation_count": violations,
        "maximum_objective_score": maximum,
        "sum_objective_score": total,
        "normalized_gain_norm": float((kp * kp) + ((kv * 30.0) ** 2)),
        "feasible": nonfinite == 0 and violations == 0,
    }


def feedback_rank(row: dict) -> tuple:
    return (
        row["nonfinite_endpoint_count"],
        row["tracking_boundary_violation_count"],
        row["maximum_objective_score"],
        row["sum_objective_score"],
        row["normalized_gain_norm"],
    )


def optimize_feedback(*, contract: dict, backend, policy, snapshot: dict, hidden,
                      reference_qpos: np.ndarray, reference_qvel: np.ndarray) -> tuple[dict, dict]:
    optimizer = contract["stage_2_structured_feedback"]["optimizer"]
    gains = contract["stage_2_structured_feedback"]["gains"]
    kp_bounds = tuple(float(value) for value in gains["kp_interval"])
    kv_bounds = tuple(float(value) for value in gains["kv_interval_seconds"])
    no_feedback = gain_to_latent(0.0, 0.0, kp_bounds, kv_bounds)
    full_feedback = gain_to_latent(1.0, 1.0 / 30.0, kp_bounds, kv_bounds)
    dataset = {key: [] for key in (
        "restart_index", "iteration", "sample_index", "latent", "kp", "kv_seconds",
        "score", "finite", "terminated", "actor_action", "action",
        "physical_feedback_m", "position_error_m", "velocity_error_m_s",
        "projected_translation_components", "feasible",
    )}
    global_best = None
    feasible_count = 0
    restart_reports = []
    for restart_index, restart in enumerate(optimizer["restarts"]):
        rng = np.random.default_rng(int(restart["seed"]))
        mean = gain_to_latent(
            float(restart["initial_kp"]), float(restart["initial_kv_seconds"]),
            kp_bounds, kv_bounds,
        )
        std = np.full(2, float(optimizer["initial_normalized_standard_deviation"]))
        iteration_reports = []
        for iteration in range(int(optimizer["iterations"])):
            samples = np.clip(
                rng.normal(mean, std, size=(int(optimizer["population"]), 2)),
                -1.0,
                1.0,
            )
            samples[0] = mean
            samples[1] = no_feedback
            samples[2] = full_feedback
            rows = []
            for sample_index, latent in enumerate(samples):
                kp, kv = latent_to_gain(latent, kp_bounds, kv_bounds)
                outcome = rollout_feedback(
                    backend=backend,
                    policy=policy,
                    snapshot=snapshot,
                    hidden=hidden,
                    kp=kp,
                    kv=kv,
                    reference_qpos=reference_qpos,
                    reference_qvel=reference_qvel,
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
                if global_best is None or feedback_rank(row) < feedback_rank(global_best):
                    global_best = row
                mapping = {
                    "restart_index": restart_index,
                    "iteration": iteration,
                    "sample_index": sample_index,
                    "latent": row["latent"],
                    "kp": row["kp"],
                    "kv_seconds": row["kv_seconds"],
                    "score": row["scores"],
                    "finite": row["finite"],
                    "terminated": row["terminated"],
                    "actor_action": row["actor_actions"],
                    "action": row["actions"],
                    "physical_feedback_m": row["physical_feedback_m"],
                    "position_error_m": row["position_error_m"],
                    "velocity_error_m_s": row["velocity_error_m_s"],
                    "projected_translation_components": row["projected_translation_components"],
                    "feasible": row["feasible"],
                }
                for key, value in mapping.items():
                    dataset[key].append(value)
            order = sorted(range(len(rows)), key=lambda index: feedback_rank(rows[index]))
            elite_latent = np.asarray([
                rows[index]["latent"] for index in order[: int(optimizer["elites"])]
            ], np.float64)
            weight = float(optimizer["elite_update_new_weight"])
            mean = (1.0 - weight) * mean + weight * elite_latent.mean(axis=0)
            std = np.maximum(
                float(optimizer["minimum_normalized_standard_deviation"]),
                (1.0 - weight) * std + weight * elite_latent.std(axis=0),
            )
            best = rows[order[0]]
            iteration_reports.append({
                "iteration": iteration,
                "feasible_candidates": int(sum(row["feasible"] for row in rows)),
                "best_kp": best["kp"],
                "best_kv_seconds": best["kv_seconds"],
                "best_scores": best["scores"].tolist(),
                "best_maximum_score": best["maximum_objective_score"],
            })
            print(
                f"feedback restart={restart['name']} iter={iteration} "
                f"feasible={iteration_reports[-1]['feasible_candidates']} "
                f"kp={best['kp']:.6f} kv={best['kv_seconds']:.8f} "
                f"best_max={best['maximum_objective_score']:.9f}",
                flush=True,
            )
        restart_reports.append({
            "name": restart["name"],
            "seed": restart["seed"],
            "iterations": iteration_reports,
        })
    if global_best is None:
        raise RuntimeError("feedback CEM produced no candidate")
    repeats = [
        rollout_feedback(
            backend=backend,
            policy=policy,
            snapshot=snapshot,
            hidden=hidden,
            kp=global_best["kp"],
            kv=global_best["kv_seconds"],
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
        )
        for _ in range(2)
    ]
    repeatable = all(
        np.array_equal(repeats[0][key], repeats[1][key])
        for key in (
            "scores", "actor_actions", "actions", "physical_feedback_m",
            "qpos", "qvel", "finite", "terminated",
        )
    )
    if not repeatable:
        raise RuntimeError("best structured-feedback candidate is not CPU repeatable")
    result = {
        "executed": True,
        "feasible_candidate_count": feasible_count,
        "any_feasible_candidate": bool(feasible_count),
        "restarts": restart_reports,
        "best_candidate": {
            "kp": global_best["kp"],
            "kv_seconds": global_best["kv_seconds"],
            "scores": global_best["scores"].tolist(),
            "finite": global_best["finite"].tolist(),
            "terminated": global_best["terminated"].tolist(),
            "actor_actions": global_best["actor_actions"].tolist(),
            "actions": global_best["actions"].tolist(),
            "physical_feedback_m": global_best["physical_feedback_m"].tolist(),
            "position_error_m": global_best["position_error_m"].tolist(),
            "velocity_error_m_s": global_best["velocity_error_m_s"].tolist(),
            "projected_translation_components": global_best[
                "projected_translation_components"
            ].tolist(),
            "tracking_boundary_violation_count": global_best[
                "tracking_boundary_violation_count"
            ],
            "maximum_objective_score": global_best["maximum_objective_score"],
            "sum_objective_score": global_best["sum_objective_score"],
            "feasible": global_best["feasible"],
            "bitwise_repeatable_on_CPU": repeatable,
        },
        "_best": global_best,
    }
    return result, dataset


def write_summary(path: Path, report: dict) -> None:
    stage1 = report["stage_1_local_controllability"]
    stage2 = report["stage_2_structured_feedback"]
    lines = [
        "# Gate A1: local controllability and structured feedback",
        "",
        f"- current-support surrogate informative: {stage1['models']['current_support']['held_out']['informative']}",
        f"- rho<=3 surrogate informative: {stage1['models']['rho_le_3']['held_out']['informative']}",
        f"- exact QP endpoint58 score: {stage1['QP_exact_replay']['objective_score']}",
        f"- exact QP passed: {stage1['QP_exact_replay']['passed_endpoint58']}",
        f"- structured feedback executed: {stage2['executed']}",
    ]
    if stage2["executed"]:
        lines.extend([
            f"- feedback feasible candidates: {stage2['feasible_candidate_count']}",
            f"- feedback best gains: kp={stage2['best_candidate']['kp']}, kv={stage2['best_candidate']['kv_seconds']}",
            f"- feedback best scores: {stage2['best_candidate']['scores']}",
        ])
    lines.extend([
        f"- Gate A classification: {report['decision']['gate_A_classification']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
        "Gate A is closed by this contract. Finite negative results are not",
        "mathematical infeasibility proofs. No policy training or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_action_feasibility_gate_A1_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_action_feasibility_gate_A1_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_a0 = json.loads(paths["gate_A0_report"].read_text())
    gate_rho = json.loads(paths["gate_A_rho_report"].read_text())
    gate_frame = json.loads(paths["gate_A_frame_report"].read_text())
    if any((
        gate_a0["decision"]["source57_current_support_feasible_sequence_found"],
        gate_rho["decision"]["expanded_support_feasible_sequence_found"],
        gate_frame["decision"]["reference_object_frame_feasible_sequence_found"],
    )):
        raise ValueError("Gate A1 requires the three preceding negative finite searches")
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
    temporary = tempfile.TemporaryDirectory(prefix=".gate_A1_", dir=ROOT / "runs")
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
        source_hidden = clone_hidden(policy.rnn_states)
        expected_state = next(
            row["source_state"] for row in gate_a0["benchmarks"]
            if row["name"] == "tail_source57"
        )
        if source["source_state"] != expected_state:
            raise RuntimeError("source57 state differs from Gate A0")
        source_position = np.asarray(source["source_state"]["tool"]["position_world_m"])
        original_residual = backend.env.env_cfg.residual

        a0 = np.load(paths["gate_A0_candidates"])
        a0_mask = a0["benchmark"] == "tail_source57"
        current = collect_saved_responses(
            name="current_support",
            actions=a0["action"][a0_mask, 0],
            saved_scores=a0["score"][a0_mask, 0],
            restart=a0["restart_index"][a0_mask],
            iteration=a0["iteration"][a0_mask],
            sample=a0["sample_index"][a0_mask],
            rho=np.ones(int(a0_mask.sum())),
            response_replay_residual_clip_m=np.full(int(a0_mask.sum()), 0.05),
            backend=backend,
            snapshot=source["snapshot"],
            source_position=source_position,
            original_residual=original_residual,
        )
        rho_arrays = np.load(paths["gate_A_rho_candidates"])
        rho_parent_contract = yaml.safe_load(paths["gate_A_rho_contract"].read_text())
        exact_rho = regenerate_parent_float64_rho(rho_parent_contract, rho_arrays)
        rho_dataset = collect_saved_responses(
            name="rho_le_3",
            actions=rho_arrays["action"][:, 0],
            saved_scores=rho_arrays["score"][:, 0],
            restart=rho_arrays["restart_index"],
            iteration=rho_arrays["iteration"],
            sample=rho_arrays["sample_index"],
            rho=exact_rho,
            response_replay_residual_clip_m=0.05 * exact_rho,
            backend=backend,
            snapshot=source["snapshot"],
            source_position=source_position,
            original_residual=original_residual,
        )
        ridge = float(contract["stage_1_local_controllability"]["regression"]["fixed_ridge_coefficient"])
        current_metrics, current_model = fit_affine_ridge(current, ridge)
        rho_metrics, rho_model = fit_affine_ridge(rho_dataset, ridge)

        source57_row = next(
            row for row in gate_a0["benchmarks"] if row["name"] == "tail_source57"
        )
        qp = solve_current_support_QP(
            model=current_model,
            low=np.asarray(source57_row["state_feasible_low"][0], np.float64),
            high=np.asarray(source57_row["state_feasible_high"][0], np.float64),
            target_position_delta=reference_qpos[58, 36:39] - source_position,
        )
        qp_exact = replay_first_action(
            backend=backend,
            snapshot=source["snapshot"],
            action=qp["action"],
            residual_clip=0.05,
            original_residual=original_residual,
        )
        qp_passed = bool(qp_exact["finite"] and qp_exact["score"] < 1.0)
        if qp_passed:
            feedback = {
                "executed": False,
                "reason": "stage_1_QP_exact_replay_passed",
                "feasible_candidate_count": 0,
                "any_feasible_candidate": False,
            }
            feedback_dataset = None
        else:
            feedback, feedback_dataset = optimize_feedback(
                contract=contract,
                backend=backend,
                policy=policy,
                snapshot=source["snapshot"],
                hidden=source_hidden,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
            )
        policy.writer.close()
    finally:
        temporary.cleanup()

    feedback_passed = bool(feedback.get("any_feasible_candidate", False))
    if qp_passed:
        classification = "current_raw_36d_interface_has_demonstrated_one_step_authority"
        gate_b_allowed = True
        next_blocker = "gate_B_observation_sufficiency_contract_required"
    elif feedback_passed:
        classification = "structured_object_motion_feedback_demonstrates_three_step_feasibility"
        gate_b_allowed = True
        next_blocker = "gate_B_observation_sufficiency_contract_required_with_feedback_architecture_candidate"
    else:
        classification = (
            "tested_raw_surrogate_QP_and_structured_feedback_candidates_do_not_"
            "demonstrate_source57_feasibility"
        )
        gate_b_allowed = False
        next_blocker = "low_level_impedance_or_force_aware_contact_controllability_required"

    qp_report = {
        "action": qp["action"].tolist(),
        "solver_success": qp["solver_success"],
        "solver_status": qp["solver_status"],
        "solver_message": qp["solver_message"],
        "solver_cost": qp["solver_cost"],
        "predicted_response": qp["predicted_response"].tolist(),
        "predicted_position_error_m": qp["predicted_position_error_m"],
        "actual_endpoint58_tool_position_delta_world_m": (
            np.asarray(qp_exact["qpos"][36:39], np.float64) - source_position
        ).tolist(),
        "actual_endpoint58_tool_linear_velocity_world_m_s": qp_exact["qvel"][36:39].tolist(),
        "objective_score": qp_exact["score"],
        "terminated": qp_exact["terminated"],
        "finite": qp_exact["finite"],
        "passed_endpoint58": qp_passed,
    }
    feedback_serializable = {key: value for key, value in feedback.items() if key != "_best"}
    report = {
        "schema": contract["schema"],
        "status": "completed_final_read_only_gate_A",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "policy_optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "gate_A0_source57_state_exactly_reproduced": True,
            "tail_selected_sources44_through56_exactly_reproduced": True,
            "current_support_saved_first_step_scores_bitwise_reproduced_as_float32": current[
                "first_step_scores_bitwise_reproduced_as_float32"
            ],
            "rho_saved_first_step_scores_bitwise_reproduced_as_float32": rho_dataset[
                "first_step_scores_bitwise_reproduced_as_float32"
            ],
            "rho_float64_provenance_values_round_to_saved_float32": bool(
                np.array_equal(
                    np.asarray(rho_dataset["rho"], np.float64).astype(np.float32),
                    np.asarray(rho_arrays["rho"], np.float32),
                )
            ),
            "rho_float64_provenance_value_count": int(len(rho_dataset["rho"])),
            "best_feedback_candidate_bitwise_repeatable_on_CPU": (
                feedback_serializable.get("best_candidate", {}).get(
                    "bitwise_repeatable_on_CPU"
                ) if feedback["executed"] else None
            ),
        },
        "actuator_names": list(actuator_names),
        "source57_state": source["source_state"],
        "stage_1_local_controllability": {
            "new_action_search_executed": False,
            "replayed_saved_action_counts": {
                "current_support": len(current["action"]),
                "rho_le_3": len(rho_dataset["action"]),
            },
            "models": {
                "current_support": {
                    "held_out": current_metrics,
                    "coefficient_shape": list(current_model["weights"].shape),
                },
                "rho_le_3": {
                    "held_out": rho_metrics,
                    "coefficient_shape": list(rho_model["weights"].shape),
                },
            },
            "QP_model_source": "current_support_only",
            "QP_exact_replay": qp_report,
        },
        "stage_2_structured_feedback": feedback_serializable,
        "decision": {
            "Gate_A_closed": True,
            "gate_A_classification": classification,
            "finite_search_failure_is_mathematical_infeasibility_proof": False,
            "gate_B_observation_sufficiency_read_only_allowed": gate_b_allowed,
            "gate_C_policy_representability_allowed": False,
            "gate_D_fresh_PPO_allowed": False,
            "additional_Gate_A_search_authorized": False,
            "pure_learned_suppression_gate_training_authorized": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": next_blocker,
        },
        "measurement_boundaries": {
            "stage_1_replays_existing_actions_only": True,
            "stage_1_validates_exactly_one_new_QP_action": True,
            "stage_2_optimizes_only_two_shared_scalar_gains": feedback["executed"],
            "current_action_support_only": True,
            "no_axis_gain_frame_rho_or_per_source_sweep": True,
            "no_policy_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    response_path = args.output / contract["artifacts"]["response_dataset"]
    np.savez_compressed(
        response_path,
        current_action=current["action"].astype(np.float32),
        current_response=current["response"].astype(np.float32),
        current_restart_index=current["restart_index"],
        current_iteration=current["iteration"],
        current_sample_index=current["sample_index"],
        current_score=current["saved_score"].astype(np.float32),
        rho_action=rho_dataset["action"].astype(np.float32),
        rho_response=rho_dataset["response"].astype(np.float32),
        rho_restart_index=rho_dataset["restart_index"],
        rho_iteration=rho_dataset["iteration"],
        rho_sample_index=rho_dataset["sample_index"],
        rho_value_float32=np.asarray(rho_arrays["rho"], np.float32),
        rho_value_float64=rho_dataset["rho"].astype(np.float64),
        rho_replay_residual_clip_m=rho_dataset[
            "response_replay_residual_clip_m"
        ].astype(np.float64),
        rho_score=rho_dataset["saved_score"].astype(np.float32),
        current_model_weights=current_model["weights"].astype(np.float64),
        current_model_intercept=current_model["intercept"].astype(np.float64),
        rho_model_weights=rho_model["weights"].astype(np.float64),
        rho_model_intercept=rho_model["intercept"].astype(np.float64),
    )
    qp_path = args.output / contract["artifacts"]["QP_action"]
    np.savez_compressed(
        qp_path,
        action=qp["action"],
        predicted_response=qp["predicted_response"],
        actual_qpos=np.asarray(qp_exact["qpos"], np.float32),
        actual_qvel=np.asarray(qp_exact["qvel"], np.float32),
        actual_score=np.asarray(qp_exact["score"], np.float32),
        passed=np.asarray(qp_passed, np.bool_),
    )
    feedback_candidates_path = args.output / contract["artifacts"]["feedback_candidates"]
    feedback_best_path = args.output / contract["artifacts"]["feedback_best"]
    if feedback_dataset is None:
        np.savez_compressed(feedback_candidates_path, executed=np.asarray(False, np.bool_))
        np.savez_compressed(feedback_best_path, executed=np.asarray(False, np.bool_))
    else:
        np.savez_compressed(
            feedback_candidates_path,
            **{
                key: np.asarray(values)
                for key, values in feedback_dataset.items()
            },
        )
        best = feedback["_best"]
        np.savez_compressed(
            feedback_best_path,
            executed=np.asarray(True, np.bool_),
            kp=np.asarray(best["kp"], np.float64),
            kv_seconds=np.asarray(best["kv_seconds"], np.float64),
            score=np.asarray(best["scores"], np.float32),
            actor_action=np.asarray(best["actor_actions"], np.float32),
            action=np.asarray(best["actions"], np.float32),
            physical_feedback_m=np.asarray(best["physical_feedback_m"], np.float32),
            qpos=np.asarray(best["qpos"], np.float32),
            qvel=np.asarray(best["qvel"], np.float32),
            feasible=np.asarray(best["feasible"], np.bool_),
        )
    report["artifacts"] = {
        "response_dataset": {"path": str(response_path.resolve()), "sha256": sha256(response_path)},
        "QP_action": {"path": str(qp_path.resolve()), "sha256": sha256(qp_path)},
        "feedback_candidates": {
            "path": str(feedback_candidates_path.resolve()),
            "sha256": sha256(feedback_candidates_path),
        },
        "feedback_best": {
            "path": str(feedback_best_path.resolve()),
            "sha256": sha256(feedback_best_path),
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "stage_1": report["stage_1_local_controllability"],
        "stage_2": report["stage_2_structured_feedback"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
