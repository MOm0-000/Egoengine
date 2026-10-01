#!/usr/bin/env python3
"""Short-horizon CEM feasibility audit for the frozen Pour action contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _actuator_metadata,
    _require_exact_trace,
)
from audit_taco_pour_binary_translation_last_off_reversal import (  # noqa: E402
    reproduce_baseline,
)
from audit_taco_pour_binary_translation_oracle import (  # noqa: E402
    compact_source_state,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_source45_prefix_source50_refinement import (  # noqa: E402
    require_prefix_arrays,
    run_policy_from_boundary,
)
from audit_taco_pour_source45_state_entry import (  # noqa: E402
    run_ppo_with_states,
    state_record,
)
from audit_taco_pour_source57_temporal_hold_gate import (  # noqa: E402
    _assert_scores_exact,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    evaluate_candidates,
    load_contract as load_tail_contract,
    restore_semantic_choice,
)


BENCHMARKS = (
    ("formal_source45", 45),
    ("prefix_source50", 50),
    ("tail_source57", 57),
)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_action_feasibility_cem_v1":
        raise ValueError("unsupported action-feasibility contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("action-feasibility gate is not authorized")
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
        "action_scale_changed": False,
        "action_support_changed": False,
        "PPO_hyperparameters_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    expected_states = [
        {"name": "formal_source45", "source_endpoint": 45,
         "provenance": "formal_single_pass_deterministic_PPO_path"},
        {"name": "prefix_source50", "source_endpoint": 50,
         "provenance": "source45_zero_right_wrist_translation_then_frozen_PPO"},
        {"name": "tail_source57", "source_endpoint": 57,
         "provenance": "selected_tail_semantic_oracle_path_through_source56"},
    ]
    action = contract["action_contract"]
    optimizer = contract["optimizer"]
    stopping = contract["stopping_rule"]
    if (
        any(contract["runtime"].get(k) != v for k, v in required_runtime.items())
        or contract.get("paper_faithful") is not False
        or contract["fixed_diagnostic_states"] != expected_states
        or action != {
            "action_dimensions": 36,
            "representation": "world_frame_additive_residual",
            "residual_scale": 0.05,
            "current_state_feasible_support_only": True,
            "per_transition_bounds_recomputed_from_frozen_reference_and_ctrlrange": True,
            "relaxed_support_allowed": False,
            "object_frame_comparison_allowed": False,
            "axis_or_scale_sweep_allowed": False,
        }
        or optimizer["algorithm"] != "deterministic_CEM"
        or optimizer["horizon_control_intervals"] != 3
        or optimizer["population"] != 24
        or optimizer["elites"] != 6
        or optimizer["iterations"] != 5
        or [row["name"] for row in optimizer["restarts"]]
        != ["frozen_actor_centered", "zero_residual_centered"]
        or optimizer["normalized_latent_interval"] != [-1.0, 1.0]
        or optimizer["initial_standard_deviation"] != 0.6
        or optimizer["elite_update_new_weight"] != 0.75
        or optimizer["minimum_standard_deviation"] != 0.05
        or optimizer["stop_on_first_feasible"] is not False
        or optimizer["candidate_sequence_count_per_state"] != 240
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
        raise ValueError("action-feasibility definition changed")
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


def action_to_latent(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    width = high - low
    if np.any(width <= 0.0):
        raise ValueError("state-feasible action support must have positive width")
    latent = 2.0 * (action - low) / width - 1.0
    return np.clip(latent, -1.0, 1.0)


def latent_to_action(latent: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + 0.5 * (latent + 1.0) * (high - low)


def rank_key(row: dict) -> tuple:
    return (
        row["nonfinite_endpoint_count"],
        row["tracking_boundary_violation_count"],
        row["maximum_objective_score"],
        row["sum_objective_score"],
        row["squared_normalized_action_norm"],
    )


def rollout_sequence(*, backend, snapshot: dict, source: int,
                     latent: np.ndarray, expected_low: np.ndarray,
                     expected_high: np.ndarray) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    scores = []
    finite_rows = []
    terminated = []
    actions = []
    qpos_rows = []
    qvel_rows = []
    for offset in range(latent.shape[0]):
        low_t, high_t = backend.env.current_normalized_action_bounds()
        low = low_t[0].detach().cpu().numpy()
        high = high_t[0].detach().cpu().numpy()
        if not np.array_equal(low, expected_low[offset]):
            raise RuntimeError("current low action bound changed across candidate states")
        if not np.array_equal(high, expected_high[offset]):
            raise RuntimeError("current high action bound changed across candidate states")
        action = latent_to_action(latent[offset], low, high).astype(np.float32)
        backend.step(action[None], source + offset)
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
        qpos_rows.append(qpos)
        qvel_rows.append(qvel)
    scores_array = np.asarray(scores, np.float64)
    nonfinite = int((~np.asarray(finite_rows, np.bool_)).sum())
    violations = int((scores_array >= 1.0).sum()) if nonfinite == 0 else len(scores)
    finite_scores = scores_array[np.isfinite(scores_array)]
    maximum = float(finite_scores.max()) if finite_scores.size else float("inf")
    total = float(finite_scores.sum()) if finite_scores.size else float("inf")
    action_array = np.asarray(actions, np.float32)
    feasible = nonfinite == 0 and violations == 0
    return {
        "scores": scores_array,
        "finite": np.asarray(finite_rows, np.bool_),
        "terminated": np.asarray(terminated, np.bool_),
        "actions": action_array,
        "qpos": np.asarray(qpos_rows, np.float32),
        "qvel": np.asarray(qvel_rows, np.float32),
        "nonfinite_endpoint_count": nonfinite,
        "tracking_boundary_violation_count": violations,
        "maximum_objective_score": maximum,
        "sum_objective_score": total,
        "squared_normalized_action_norm": float(np.square(action_array).sum()),
        "feasible": feasible,
    }


def nominal_sequences(*, backend, policy, benchmark: dict, horizon: int) -> dict:
    backend.restore(benchmark["snapshot"])
    backend.verify_restored_snapshot(benchmark["snapshot"])
    policy.rnn_states = clone_hidden(benchmark["hidden"])
    actor_actions = []
    low_rows = []
    high_rows = []
    for offset in range(horizon):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])[0]
        low = result["action_lows"][0].detach().cpu().numpy()
        high = result["action_highs"][0].detach().cpu().numpy()
        env_low, env_high = backend.env.current_normalized_action_bounds()
        if not np.array_equal(low, env_low[0].detach().cpu().numpy()):
            raise RuntimeError("actor and environment low bounds differ")
        if not np.array_equal(high, env_high[0].detach().cpu().numpy()):
            raise RuntimeError("actor and environment high bounds differ")
        actor_actions.append(action)
        low_rows.append(low)
        high_rows.append(high)
        backend.step(action[None], benchmark["source"] + offset)
    low_array = np.asarray(low_rows, np.float32)
    high_array = np.asarray(high_rows, np.float32)
    actor_array = np.asarray(actor_actions, np.float32)
    zero_array = np.zeros_like(actor_array)
    if np.any(zero_array < low_array) or np.any(zero_array > high_array):
        raise RuntimeError("zero residual leaves current state-feasible support")
    return {
        "actor_actions": actor_array,
        "zero_actions": zero_array,
        "low": low_array,
        "high": high_array,
        "actor_latent": action_to_latent(actor_array, low_array, high_array),
        "zero_latent": action_to_latent(zero_array, low_array, high_array),
    }


def optimize_benchmark(*, backend, policy, benchmark: dict,
                       optimizer: dict, dataset: dict[str, list]) -> dict:
    horizon = int(optimizer["horizon_control_intervals"])
    nominal = nominal_sequences(
        backend=backend, policy=policy, benchmark=benchmark, horizon=horizon
    )
    anchor_rows = {}
    for name, latent in (
        ("frozen_actor_open_loop", nominal["actor_latent"]),
        ("zero_residual", nominal["zero_latent"]),
    ):
        anchor_rows[name] = rollout_sequence(
            backend=backend,
            snapshot=benchmark["snapshot"],
            source=benchmark["source"],
            latent=latent,
            expected_low=nominal["low"],
            expected_high=nominal["high"],
        )

    global_best = None
    feasible_count = 0
    restart_reports = []
    for restart_index, restart in enumerate(optimizer["restarts"]):
        rng = np.random.default_rng(
            int(optimizer["benchmark_base_seeds"][benchmark["name"]])
            + int(restart["seed_offset"])
        )
        mean = (
            nominal["actor_latent"].astype(np.float64).copy()
            if restart["name"] == "frozen_actor_centered"
            else nominal["zero_latent"].astype(np.float64).copy()
        )
        std = np.full_like(mean, float(optimizer["initial_standard_deviation"]))
        iteration_reports = []
        for iteration in range(int(optimizer["iterations"])):
            samples = rng.normal(
                mean,
                std,
                size=(int(optimizer["population"]),) + mean.shape,
            )
            samples = np.clip(samples, -1.0, 1.0)
            samples[0] = mean
            samples[1] = nominal["actor_latent"]
            samples[2] = nominal["zero_latent"]
            rows = []
            for sample_index, latent in enumerate(samples):
                outcome = rollout_sequence(
                    backend=backend,
                    snapshot=benchmark["snapshot"],
                    source=benchmark["source"],
                    latent=latent,
                    expected_low=nominal["low"],
                    expected_high=nominal["high"],
                )
                row = {
                    **outcome,
                    "latent": latent.astype(np.float32),
                    "restart_index": restart_index,
                    "iteration": iteration,
                    "sample_index": sample_index,
                }
                rows.append(row)
                feasible_count += int(outcome["feasible"])
                if global_best is None or rank_key(row) < rank_key(global_best):
                    global_best = row
                dataset["benchmark"].append(benchmark["name"])
                dataset["restart_index"].append(restart_index)
                dataset["iteration"].append(iteration)
                dataset["sample_index"].append(sample_index)
                dataset["latent"].append(row["latent"])
                dataset["action"].append(row["actions"])
                dataset["score"].append(row["scores"])
                dataset["finite"].append(row["finite"])
                dataset["terminated"].append(row["terminated"])
                dataset["feasible"].append(row["feasible"])
            order = sorted(range(len(rows)), key=lambda index: rank_key(rows[index]))
            elite_latent = np.asarray([
                rows[index]["latent"]
                for index in order[: int(optimizer["elites"])]
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
                "best_scores": best_iteration["scores"].tolist(),
                "best_violation_count": best_iteration[
                    "tracking_boundary_violation_count"
                ],
                "best_maximum_score": best_iteration["maximum_objective_score"],
                "mean_standard_deviation": float(std.mean()),
            })
            print(
                f"{benchmark['name']} restart={restart['name']} iter={iteration} "
                f"feasible={iteration_reports[-1]['feasible_sequences']} "
                f"best_max={best_iteration['maximum_objective_score']:.9f}",
                flush=True,
            )
        restart_reports.append({
            "name": restart["name"],
            "seed": (
                int(optimizer["benchmark_base_seeds"][benchmark["name"]])
                + int(restart["seed_offset"])
            ),
            "iterations": iteration_reports,
        })

    if global_best is None:
        raise RuntimeError("CEM produced no candidate")
    repeats = [
        rollout_sequence(
            backend=backend,
            snapshot=benchmark["snapshot"],
            source=benchmark["source"],
            latent=global_best["latent"],
            expected_low=nominal["low"],
            expected_high=nominal["high"],
        )
        for _ in range(2)
    ]
    repeatable = all(
        np.array_equal(repeats[0][key], repeats[1][key])
        for key in ("scores", "actions", "qpos", "qvel", "finite", "terminated")
    )
    if not repeatable:
        raise RuntimeError(f"CPU best-sequence replay changed for {benchmark['name']}")
    return {
        "name": benchmark["name"],
        "source_endpoint": benchmark["source"],
        "provenance": benchmark["provenance"],
        "source_state": benchmark["source_state"],
        "horizon_control_intervals": horizon,
        "evaluated_CEM_sequences": (
            int(optimizer["population"])
            * int(optimizer["iterations"])
            * len(optimizer["restarts"])
        ),
        "feasible_CEM_sequence_count": feasible_count,
        "any_CEM_sequence_feasible": feasible_count > 0,
        "anchor_sequences": {
            name: {
                "scores": row["scores"].tolist(),
                "feasible": row["feasible"],
                "tracking_boundary_violation_count": row[
                    "tracking_boundary_violation_count"
                ],
            }
            for name, row in anchor_rows.items()
        },
        "state_feasible_low": nominal["low"].tolist(),
        "state_feasible_high": nominal["high"].tolist(),
        "restarts": restart_reports,
        "best_sequence": {
            "scores": global_best["scores"].tolist(),
            "finite": global_best["finite"].tolist(),
            "terminated": global_best["terminated"].tolist(),
            "actions": global_best["actions"].tolist(),
            "tracking_boundary_violation_count": global_best[
                "tracking_boundary_violation_count"
            ],
            "maximum_objective_score": global_best["maximum_objective_score"],
            "sum_objective_score": global_best["sum_objective_score"],
            "feasible": global_best["feasible"],
            "bitwise_repeatable_on_CPU": repeatable,
        },
        "_best_latent": global_best["latent"],
        "_best_qpos": global_best["qpos"],
        "_best_qvel": global_best["qvel"],
    }


def capture_benchmarks(*, backend, policy, boundary, formal: dict,
                       paths: dict[str, Path], reference_qpos,
                       reference_qvel, objective) -> list[dict]:
    ppo, _, capture45, _ = run_ppo_with_states(
        backend=backend, policy=policy, boundary=boundary
    )
    _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")
    source45_state = compact_source_state(
        state_record(backend.env), 49, reference_qpos, reference_qvel, objective
    )
    del source45_state

    prefix, _, _, capture50 = run_policy_from_boundary(
        backend=backend,
        policy=policy,
        boundary=boundary,
        prefix_suppression=True,
    )
    require_prefix_arrays(prefix, paths["source45_arrays"])
    if capture50 is None:
        raise RuntimeError("source50 prefix capture missing")

    _, tail_paths, _ = load_tail_contract(paths["tail_contract"])
    tail_report = json.loads(paths["tail_report"].read_text())
    captures, _ = reproduce_baseline(
        backend=backend,
        policy=policy,
        boundary=boundary,
        reference_qpos=reference_qpos,
        reference_qvel=reference_qvel,
        objective=objective,
        expected_arrays=tail_paths["baseline_oracle_arrays"],
        expected_off_sources=json.loads(
            tail_paths["baseline_oracle_report"].read_text()
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
    capture57 = {
        "environment_state": backend.snapshot(),
        "pre_forward_hidden": clone_hidden(policy.rnn_states),
    }

    raw = [
        {
            "name": "formal_source45",
            "source": 45,
            "provenance": "formal_single_pass_deterministic_PPO_path",
            "snapshot": capture45["environment_state"],
            "hidden": capture45["pre_forward_hidden"],
        },
        {
            "name": "prefix_source50",
            "source": 50,
            "provenance": "source45_zero_right_wrist_translation_then_frozen_PPO",
            "snapshot": capture50["environment_state"],
            "hidden": capture50["pre_forward_hidden"],
        },
        {
            "name": "tail_source57",
            "source": 57,
            "provenance": "selected_tail_semantic_oracle_path_through_source56",
            "snapshot": capture57["environment_state"],
            "hidden": capture57["pre_forward_hidden"],
        },
    ]
    result = []
    for row in raw:
        backend.restore(row["snapshot"])
        backend.verify_restored_snapshot(row["snapshot"])
        row["source_state"] = compact_source_state(
            state_record(backend.env), row["source"],
            reference_qpos, reference_qvel, objective,
        )
        result.append(row)
    return result


def write_summary(path: Path, report: dict) -> None:
    lines = ["# Gate A0: current-support action feasibility", ""]
    for row in report["benchmarks"]:
        lines.append(
            f"- {row['name']}: feasible={row['any_CEM_sequence_feasible']}, "
            f"count={row['feasible_CEM_sequence_count']}, "
            f"best_max={row['best_sequence']['maximum_objective_score']:.9f}, "
            f"scores={row['best_sequence']['scores']}"
        )
    lines.extend([
        "",
        f"- source57 decisive result: {report['decision']['source57_classification']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
        "This is a deterministic CPU CEM diagnostic under the unchanged 36-D",
        "world-frame residual support. It does not train a policy, relax support,",
        "change frame, accept a chunk, or prove mathematical infeasibility when",
        "a finite optimizer fails to find a solution.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_action_feasibility_cem_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_action_feasibility_cem_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    _, parent_paths, _ = load_tail_contract(paths["tail_contract"])

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
    distribution, _ = load_truncated_gaussian_profile(
        parent_paths["distribution_profile"]
    )
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
    formal = json.loads(paths["formal_report"].read_text())

    temporary = tempfile.TemporaryDirectory(prefix=".action_feasibility_", dir=ROOT / "runs")
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
        benchmarks = capture_benchmarks(
            backend=backend,
            policy=policy,
            boundary=boundary,
            formal=formal,
            paths=paths,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        dataset: dict[str, list] = {
            key: [] for key in (
                "benchmark", "restart_index", "iteration", "sample_index",
                "latent", "action", "score", "finite", "terminated", "feasible",
            )
        }
        results = []
        for benchmark in benchmarks:
            results.append(optimize_benchmark(
                backend=backend,
                policy=policy,
                benchmark=benchmark,
                optimizer=contract["optimizer"],
                dataset=dataset,
            ))
        policy.writer.close()
    finally:
        temporary.cleanup()

    source57 = next(row for row in results if row["name"] == "tail_source57")
    if source57["any_CEM_sequence_feasible"]:
        source57_classification = "current_action_representation_and_support_locally_sufficient"
        next_blocker = "gate_B_observation_sufficiency_contract_required"
        gate_b_allowed = True
    else:
        source57_classification = "current_support_feasibility_not_demonstrated_by_finite_CEM"
        next_blocker = "gate_A_minimum_rho_or_coordinate_frame_contract_required"
        gate_b_allowed = False

    serializable = []
    best_latent = []
    best_qpos = []
    best_qvel = []
    for row in results:
        best_latent.append(row.pop("_best_latent"))
        best_qpos.append(row.pop("_best_qpos"))
        best_qvel.append(row.pop("_best_qvel"))
        serializable.append(row)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate_A0",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "policy_optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "formal_single_pass_PPO_trace_exactly_reproduced": True,
            "source45_translation_prefix_exactly_reproduced": True,
            "tail_selected_sources44_through56_exactly_reproduced": True,
            "all_best_sequences_bitwise_repeatable_on_CPU": all(
                row["best_sequence"]["bitwise_repeatable_on_CPU"]
                for row in serializable
            ),
        },
        "actuator_names": list(actuator_names),
        "optimizer": contract["optimizer"],
        "benchmarks": serializable,
        "decision": {
            "source57_classification": source57_classification,
            "source57_current_support_feasible_sequence_found": source57[
                "any_CEM_sequence_feasible"
            ],
            "finite_optimizer_failure_is_mathematical_infeasibility_proof": False,
            "support_or_frame_change_authorized": False,
            "gate_B_observation_sufficiency_read_only_design_allowed": gate_b_allowed,
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
            "no_new_per_source_manual_counterfactuals": True,
            "current_world_frame_36d_support_only": True,
            "three_interval_horizon_only": True,
            "no_support_relaxation_or_frame_search": True,
            "no_policy_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    candidate_path = args.output / contract["artifacts"]["candidate_dataset"]
    np.savez_compressed(
        candidate_path,
        benchmark=np.asarray(dataset["benchmark"], dtype="U32"),
        restart_index=np.asarray(dataset["restart_index"], np.int16),
        iteration=np.asarray(dataset["iteration"], np.int16),
        sample_index=np.asarray(dataset["sample_index"], np.int16),
        latent=np.asarray(dataset["latent"], np.float32),
        action=np.asarray(dataset["action"], np.float32),
        score=np.asarray(dataset["score"], np.float32),
        finite=np.asarray(dataset["finite"], np.bool_),
        terminated=np.asarray(dataset["terminated"], np.bool_),
        feasible=np.asarray(dataset["feasible"], np.bool_),
    )
    best_path = args.output / contract["artifacts"]["best_sequences"]
    np.savez_compressed(
        best_path,
        benchmark=np.asarray([row["name"] for row in serializable], dtype="U32"),
        latent=np.asarray(best_latent, np.float32),
        action=np.asarray([
            row["best_sequence"]["actions"] for row in serializable
        ], np.float32),
        score=np.asarray([
            row["best_sequence"]["scores"] for row in serializable
        ], np.float32),
        qpos=np.asarray(best_qpos, np.float32),
        qvel=np.asarray(best_qvel, np.float32),
        feasible=np.asarray([
            row["best_sequence"]["feasible"] for row in serializable
        ], np.bool_),
    )
    report["artifacts"] = {
        "candidate_dataset": {
            "path": str(candidate_path.resolve()),
            "sha256": sha256(candidate_path),
        },
        "best_sequences": {
            "path": str(best_path.resolve()),
            "sha256": sha256(best_path),
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "benchmarks": [
            {
                "name": row["name"],
                "feasible": row["any_CEM_sequence_feasible"],
                "count": row["feasible_CEM_sequence_count"],
                "best_scores": row["best_sequence"]["scores"],
            }
            for row in serializable
        ],
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
