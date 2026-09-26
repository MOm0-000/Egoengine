#!/usr/bin/env python3
"""Joint source-57 residual-sequence and translation-support CEM audit."""

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
from audit_taco_pour_binary_translation_last_off_reversal import (  # noqa: E402
    reproduce_baseline,
)
from audit_taco_pour_binary_translation_oracle import compact_source_state  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_source45_state_entry import state_record  # noqa: E402
from audit_taco_pour_source57_temporal_hold_gate import _assert_scores_exact  # noqa: E402
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    evaluate_candidates,
    load_contract as load_tail_contract,
    restore_semantic_choice,
)


SOURCE = 57
HORIZON = 3


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_action_feasibility_minimum_rho_v1":
        raise ValueError("unsupported minimum-rho contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("minimum-rho gate is not authorized")
    support = contract["support_parameter"]
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
        "action_frame_changed": False,
        "nontranslation_action_support_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    if (
        any(contract["runtime"].get(k) != v for k, v in required_runtime.items())
        or contract.get("paper_faithful") is not False
        or contract["source_state"] != {
            "name": "tail_source57",
            "source_endpoint": 57,
            "provenance": "selected_tail_semantic_oracle_path_through_source56",
            "exact_gate_A0_state_reproduction_required": True,
        }
        or support["applies_only_to_normalized_action_indices"] != [0, 1, 2]
        or support["lower"] != 1.0
        or support["upper"] != 3.0
        or support["constant_across_three_step_sequence"] is not True
        or support["actuator_ctrlrange_remains_hard_constraint"] is not True
        or support["all_other_33_dimensions_remain_in_current_state_feasible_support"] is not True
        or optimizer["algorithm"] != "constrained_deterministic_CEM"
        or optimizer["horizon_control_intervals"] != HORIZON
        or optimizer["population"] != 32
        or optimizer["elites"] != 8
        or optimizer["iterations"] != 8
        or optimizer["candidate_sequence_count"] != 512
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
        raise ValueError("minimum-rho definition changed")
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


def rho_from_latent(value: float, lower: float, upper: float) -> float:
    return lower + 0.5 * (float(value) + 1.0) * (upper - lower)


def rho_to_latent(value: float, lower: float, upper: float) -> float:
    return 2.0 * (float(value) - lower) / (upper - lower) - 1.0


def expanded_bounds(env, rho: float) -> tuple[np.ndarray, np.ndarray]:
    low_t, high_t = env.current_normalized_action_bounds()
    low = low_t[0].detach().cpu().numpy().astype(np.float64)
    high = high_t[0].detach().cpu().numpy().astype(np.float64)
    contract = env._state_feasible_action_contract
    if contract is None:
        raise RuntimeError("state-feasible action contract is not enabled")
    reference = env._reference_ctrls(env.time_indices, offset=1)
    reference, _ = env._snap_state_feasible_reference(reference)
    controlled = reference[0, contract["indices"]].detach().cpu().numpy().astype(np.float64)
    limited = contract["limited"].detach().cpu().numpy().astype(bool)
    safe_lower = contract["safe_lower_f32"].detach().cpu().numpy().astype(np.float64)
    safe_upper = contract["safe_upper_f32"].detach().cpu().numpy().astype(np.float64)
    for index in range(3):
        low[index] = -rho
        high[index] = rho
        if limited[index]:
            low[index] = max(low[index], (safe_lower[index] - controlled[index]) / 0.05)
            high[index] = min(high[index], (safe_upper[index] - controlled[index]) / 0.05)
        if low[index] > high[index]:
            raise RuntimeError("expanded translation support is empty")
    return low, high


def rollout(*, backend, snapshot: dict, action_latent: np.ndarray, rho: float,
            original_residual) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    backend.env.env_cfg.residual = replace(
        original_residual, residual_clip=0.05 * float(rho)
    )
    scores = []
    finite_rows = []
    terminated = []
    actions = []
    lows = []
    highs = []
    qpos_rows = []
    qvel_rows = []
    for offset in range(HORIZON):
        low, high = expanded_bounds(backend.env, rho)
        action = latent_to_action(action_latent[offset], low, high).astype(np.float32)
        backend.step(action[None], SOURCE + offset)
        score = float(np.asarray(backend.last_info["object_tracking_error"])[0])
        qpos = backend.env._mjwp.get_qpos(
            backend.env.ego_cfg, backend.env.env
        )[0].detach().cpu().numpy()
        qvel = backend.env._mjwp.get_qvel(
            backend.env.ego_cfg, backend.env.env
        )[0].detach().cpu().numpy()
        finite = bool(
            np.isfinite(score) and np.isfinite(qpos).all() and np.isfinite(qvel).all()
        )
        scores.append(score)
        finite_rows.append(finite)
        terminated.append(bool(np.asarray(backend.last_info["terminated"])[0]))
        actions.append(action)
        lows.append(low)
        highs.append(high)
        qpos_rows.append(qpos)
        qvel_rows.append(qvel)
    backend.env.env_cfg.residual = original_residual
    score_array = np.asarray(scores, np.float64)
    nonfinite = int((~np.asarray(finite_rows)).sum())
    violations = int((score_array >= 1.0).sum()) if nonfinite == 0 else HORIZON
    finite_scores = score_array[np.isfinite(score_array)]
    maximum = float(finite_scores.max()) if finite_scores.size else float("inf")
    total = float(finite_scores.sum()) if finite_scores.size else float("inf")
    feasible = nonfinite == 0 and violations == 0
    return {
        "rho": float(rho),
        "scores": score_array,
        "finite": np.asarray(finite_rows, np.bool_),
        "terminated": np.asarray(terminated, np.bool_),
        "actions": np.asarray(actions, np.float32),
        "low": np.asarray(lows, np.float32),
        "high": np.asarray(highs, np.float32),
        "qpos": np.asarray(qpos_rows, np.float32),
        "qvel": np.asarray(qvel_rows, np.float32),
        "nonfinite_endpoint_count": nonfinite,
        "tracking_boundary_violation_count": violations,
        "maximum_objective_score": maximum,
        "sum_objective_score": total,
        "feasible": feasible,
    }


def rank_key(row: dict) -> tuple:
    if row["feasible"]:
        return (
            0,
            row["rho"],
            row["maximum_objective_score"],
            row["sum_objective_score"],
        )
    return (
        1,
        row["nonfinite_endpoint_count"],
        row["tracking_boundary_violation_count"],
        row["maximum_objective_score"],
        row["sum_objective_score"],
        row["rho"],
    )


def bounds_sequence(*, backend, snapshot: dict, rho: float, original_residual) -> tuple[np.ndarray, np.ndarray]:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    backend.env.env_cfg.residual = replace(original_residual, residual_clip=0.05 * rho)
    lows = []
    highs = []
    for offset in range(HORIZON):
        low, high = expanded_bounds(backend.env, rho)
        lows.append(low)
        highs.append(high)
        backend.step(np.zeros((1, 36), np.float32), SOURCE + offset)
    backend.env.env_cfg.residual = original_residual
    return np.asarray(lows), np.asarray(highs)


def capture_source57(*, backend, policy, boundary, parent_paths,
                     tail_report, reference_qpos, reference_qvel, objective) -> dict:
    captures, _ = reproduce_baseline(
        backend=backend,
        policy=policy,
        boundary=boundary,
        reference_qpos=reference_qpos,
        reference_qvel=reference_qvel,
        objective=objective,
        expected_arrays=parent_paths["baseline_oracle_arrays"],
        expected_off_sources=json.loads(
            parent_paths["baseline_oracle_report"].read_text()
        )["oracle_result"]["OFF_sources"],
        capture_sources=(44,),
    )
    backend.restore(captures[44]["snapshot"])
    backend.verify_restored_snapshot(captures[44]["snapshot"])
    policy.rnn_states = clone_hidden(captures[44]["pre_hidden"])
    for source in range(44, 57):
        evaluated = evaluate_candidates(
            backend=backend,
            policy=policy,
            source=source,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        expected = (
            tail_report["tail_decisions"][source - 44]
            if source <= 55 else tail_report["source56_fork"]
        )
        _assert_scores_exact(evaluated, expected)
        expected_selected = (
            expected["selected"] if source <= 55
            else expected["selected_for_continuation"]
        )
        if evaluated["selected"]["name"] != expected_selected:
            raise RuntimeError(f"tail selected mode changed at source {source}")
        restore_semantic_choice(
            backend=backend,
            policy=policy,
            evaluated=evaluated,
            candidate=evaluated["selected"],
        )
    return {
        "snapshot": backend.snapshot(),
        "source_state": compact_source_state(
            state_record(backend.env), SOURCE,
            reference_qpos, reference_qvel, objective,
        ),
    }


def write_summary(path: Path, report: dict) -> None:
    decision = report["decision"]
    best = report["best_observed_candidate"]
    lines = [
        "# Gate A-rho: minimum translation-support diagnostic",
        "",
        f"- feasible candidates: {decision['feasible_candidate_count']}",
        f"- feasible sequence found: {decision['expanded_support_feasible_sequence_found']}",
        f"- best observed rho: {best['rho']}",
        f"- best scores: {best['scores']}",
        f"- next blocker: {decision['next_blocker']}",
        "",
        "rho was optimized jointly with the three-step 36-D sequence; no manual",
        "scale list, per-axis support search, policy training or chunk commit was used.",
        "The best observed rho is not claimed as a mathematical minimum.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_action_feasibility_minimum_rho_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_action_feasibility_minimum_rho_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_a0 = json.loads(paths["gate_A0_report"].read_text())
    if gate_a0["decision"]["source57_current_support_feasible_sequence_found"] is not False:
        raise ValueError("minimum-rho gate requires a negative current-support result")
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
    temporary = tempfile.TemporaryDirectory(prefix=".minimum_rho_", dir=ROOT / "runs")
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
        a0_arrays = np.load(paths["gate_A0_best_sequences"])
        tail_index = list(a0_arrays["benchmark"]).index("tail_source57")
        a0_action = np.asarray(a0_arrays["action"][tail_index], np.float64)
        low1, high1 = bounds_sequence(
            backend=backend,
            snapshot=source["snapshot"],
            rho=1.0,
            original_residual=original_residual,
        )
        a0_latent = action_to_latent(a0_action, low1, high1)
        zero_latent = action_to_latent(np.zeros_like(a0_action), low1, high1)

        optimizer = contract["optimizer"]
        support = contract["support_parameter"]
        rho_lower = float(support["lower"])
        rho_upper = float(support["upper"])
        dataset = {key: [] for key in (
            "restart_index", "iteration", "sample_index", "rho", "rho_latent",
            "action_latent", "action", "score", "finite", "terminated", "feasible",
        )}
        global_best = None
        feasible_count = 0
        restart_reports = []
        for restart_index, restart in enumerate(optimizer["restarts"]):
            rng = np.random.default_rng(int(restart["seed"]))
            mean_action = a0_latent.astype(np.float64).copy()
            mean_rho = rho_to_latent(
                float(restart["initial_rho"]), rho_lower, rho_upper
            )
            std_action = np.full_like(
                mean_action, float(optimizer["initial_action_latent_standard_deviation"])
            )
            std_rho = float(optimizer["initial_rho_latent_standard_deviation"])
            iteration_reports = []
            for iteration in range(int(optimizer["iterations"])):
                action_samples = np.clip(
                    rng.normal(
                        mean_action,
                        std_action,
                        size=(int(optimizer["population"]),) + mean_action.shape,
                    ),
                    -1.0,
                    1.0,
                )
                rho_samples = np.clip(
                    rng.normal(mean_rho, std_rho, size=int(optimizer["population"])),
                    -1.0,
                    1.0,
                )
                action_samples[0] = mean_action
                rho_samples[0] = mean_rho
                action_samples[1] = a0_latent
                rho_samples[1] = -1.0
                action_samples[2] = zero_latent
                rho_samples[2] = -1.0
                rows = []
                for sample_index in range(int(optimizer["population"])):
                    rho = rho_from_latent(
                        rho_samples[sample_index], rho_lower, rho_upper
                    )
                    outcome = rollout(
                        backend=backend,
                        snapshot=source["snapshot"],
                        action_latent=action_samples[sample_index],
                        rho=rho,
                        original_residual=original_residual,
                    )
                    row = {
                        **outcome,
                        "action_latent": action_samples[sample_index].astype(np.float32),
                        "rho_latent": float(rho_samples[sample_index]),
                        "restart_index": restart_index,
                        "iteration": iteration,
                        "sample_index": sample_index,
                    }
                    rows.append(row)
                    feasible_count += int(row["feasible"])
                    if global_best is None or rank_key(row) < rank_key(global_best):
                        global_best = row
                    for key in dataset:
                        if key == "restart_index": value = restart_index
                        elif key == "iteration": value = iteration
                        elif key == "sample_index": value = sample_index
                        elif key == "action_latent": value = row["action_latent"]
                        elif key == "rho_latent": value = row["rho_latent"]
                        elif key == "action": value = row["actions"]
                        elif key == "score": value = row["scores"]
                        else: value = row[key]
                        dataset[key].append(value)
                order = sorted(range(len(rows)), key=lambda index: rank_key(rows[index]))
                elites = [rows[index] for index in order[: int(optimizer["elites"])]]
                elite_actions = np.asarray([row["action_latent"] for row in elites])
                elite_rhos = np.asarray([row["rho_latent"] for row in elites])
                weight = float(optimizer["elite_update_new_weight"])
                minimum = float(optimizer["minimum_standard_deviation"])
                mean_action = (1.0 - weight) * mean_action + weight * elite_actions.mean(axis=0)
                std_action = np.maximum(
                    minimum,
                    (1.0 - weight) * std_action + weight * elite_actions.std(axis=0),
                )
                mean_rho = float((1.0 - weight) * mean_rho + weight * elite_rhos.mean())
                std_rho = float(max(
                    minimum,
                    (1.0 - weight) * std_rho + weight * elite_rhos.std(),
                ))
                best_iteration = rows[order[0]]
                iteration_reports.append({
                    "iteration": iteration,
                    "feasible_sequences": int(sum(row["feasible"] for row in rows)),
                    "best_rho": best_iteration["rho"],
                    "best_scores": best_iteration["scores"].tolist(),
                    "best_maximum_score": best_iteration["maximum_objective_score"],
                })
                print(
                    f"minimum-rho restart={restart['name']} iter={iteration} "
                    f"feasible={iteration_reports[-1]['feasible_sequences']} "
                    f"rho={best_iteration['rho']:.6f} "
                    f"best_max={best_iteration['maximum_objective_score']:.9f}",
                    flush=True,
                )
            restart_reports.append({
                "name": restart["name"],
                "seed": restart["seed"],
                "initial_rho": restart["initial_rho"],
                "iterations": iteration_reports,
            })
        if global_best is None:
            raise RuntimeError("minimum-rho CEM produced no candidates")
        repeats = [
            rollout(
                backend=backend,
                snapshot=source["snapshot"],
                action_latent=global_best["action_latent"],
                rho=global_best["rho"],
                original_residual=original_residual,
            )
            for _ in range(2)
        ]
        repeatable = all(
            np.array_equal(repeats[0][key], repeats[1][key])
            for key in ("scores", "actions", "qpos", "qvel", "finite", "terminated")
        )
        if not repeatable:
            raise RuntimeError("best minimum-rho candidate is not CPU repeatable")
        backend.env.env_cfg.residual = original_residual
        policy.writer.close()
    finally:
        temporary.cleanup()

    found = bool(feasible_count)
    if found:
        next_blocker = "gate_A_expanded_support_result_requires_action_interface_decision"
    else:
        next_blocker = "gate_A_coordinate_frame_comparison_required"
    best_serializable = {
        "rho": global_best["rho"],
        "scores": global_best["scores"].tolist(),
        "finite": global_best["finite"].tolist(),
        "terminated": global_best["terminated"].tolist(),
        "actions": global_best["actions"].tolist(),
        "state_feasible_low": global_best["low"].tolist(),
        "state_feasible_high": global_best["high"].tolist(),
        "tracking_boundary_violation_count": global_best[
            "tracking_boundary_violation_count"
        ],
        "maximum_objective_score": global_best["maximum_objective_score"],
        "sum_objective_score": global_best["sum_objective_score"],
        "feasible": global_best["feasible"],
        "bitwise_repeatable_on_CPU": repeatable,
    }
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate_A_rho",
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
        "optimizer": optimizer,
        "support_parameter": support,
        "restarts": restart_reports,
        "best_observed_candidate": best_serializable,
        "decision": {
            "feasible_candidate_count": feasible_count,
            "expanded_support_feasible_sequence_found": found,
            "smallest_observed_feasible_rho": (
                global_best["rho"] if found else None
            ),
            "reported_rho_is_mathematical_minimum": False,
            "manual_rho_or_axis_sweep_authorized": False,
            "automatic_upper_bound_expansion_authorized": False,
            "support_or_frame_change_authorized_for_training": False,
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
            "rho_jointly_optimized_not_manually_swept": True,
            "rho_applies_only_to_right_wrist_world_translation": True,
            "nontranslation_support_unchanged": True,
            "actuator_ctrlrange_remains_hard": True,
            "no_object_frame_search": True,
            "no_policy_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    candidates_path = args.output / contract["artifacts"]["candidate_dataset"]
    np.savez_compressed(
        candidates_path,
        restart_index=np.asarray(dataset["restart_index"], np.int16),
        iteration=np.asarray(dataset["iteration"], np.int16),
        sample_index=np.asarray(dataset["sample_index"], np.int16),
        rho=np.asarray(dataset["rho"], np.float32),
        rho_latent=np.asarray(dataset["rho_latent"], np.float32),
        action_latent=np.asarray(dataset["action_latent"], np.float32),
        action=np.asarray(dataset["action"], np.float32),
        score=np.asarray(dataset["score"], np.float32),
        finite=np.asarray(dataset["finite"], np.bool_),
        terminated=np.asarray(dataset["terminated"], np.bool_),
        feasible=np.asarray(dataset["feasible"], np.bool_),
    )
    best_path = args.output / contract["artifacts"]["best_sequence"]
    np.savez_compressed(
        best_path,
        rho=np.asarray([global_best["rho"]], np.float32),
        action_latent=np.asarray(global_best["action_latent"], np.float32),
        action=np.asarray(global_best["actions"], np.float32),
        score=np.asarray(global_best["scores"], np.float32),
        qpos=np.asarray(global_best["qpos"], np.float32),
        qvel=np.asarray(global_best["qvel"], np.float32),
        feasible=np.asarray([global_best["feasible"]], np.bool_),
    )
    report["artifacts"] = {
        "candidate_dataset": {
            "path": str(candidates_path.resolve()),
            "sha256": sha256(candidates_path),
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
        "best_observed_candidate": best_serializable,
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
