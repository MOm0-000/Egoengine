#!/usr/bin/env python3
"""Read-only semantic suppression oracle over the Pour tail sources 44--56."""

from __future__ import annotations

import argparse
from collections import Counter
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
from audit_taco_pour_binary_translation_last_off_reversal import (  # noqa: E402
    evaluate_source,
    reproduce_baseline,
    restore_choice,
)
from audit_taco_pour_binary_translation_oracle import (  # noqa: E402
    choose_branch,
    compact_outcome,
    compact_source_state,
    one_step,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_source45_state_entry import actor_record, state_record  # noqa: E402


TAIL_START = 44
TAIL_END = 55
FORK_SOURCE = 56
CANDIDATES = {
    "complete_PPO": (),
    "zero_right_wrist_translation": (0, 1, 2),
    "zero_right_wrist_rotation": (3, 4, 5),
    "zero_right_fingers": tuple(range(6, 18)),
    "zero_complete_right_wrist": tuple(range(0, 6)),
    "zero_entire_right_hand": tuple(range(0, 18)),
    "zero_full_36d_residual": tuple(range(0, 36)),
}


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_tail_semantic_suppression_oracle_v1":
        raise ValueError("unsupported tail semantic oracle contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("tail semantic oracle is not authorized")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_steps": 0,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "action_scale_changed": False,
        "action_bound_changed": False,
        "action_distribution_changed": False,
        "reference_timing_changed": False,
        "action_frame_changed": False,
        "PPO_hyperparameters_changed": False,
        "learned_gate_training_allowed": False,
        "actor_learning_rate_5e_minus_5_training_allowed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    candidates = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["candidates"]
    }
    prefix = contract["prefix"]
    tail = contract["tail_selection"]
    fork = contract["source56_fork"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or candidates != CANDIDATES
        or prefix != {
            "start_endpoint": 20,
            "last_untouched_source": 43,
            "exact_binary_oracle_arrays_required": True,
            "capture_exact_source44_complete_physics_state_and_pre_forward_RNN_hidden": True,
        }
        or tail["first_source"] != TAIL_START
        or tail["last_source"] != TAIL_END
        or tail["actor_forward_calls_per_source"] != 1
        or tail["same_pre_step_snapshot_and_post_forward_hidden_for_all_candidates"] is not True
        or tail["branch_horizon_control_intervals"] != 1
        or tail["contact_used"] is not False
        or tail["source_index_used"] is not False
        or tail["future_beyond_one_step_used"] is not False
        or tail["scale_sweep_allowed"] is not False
        or tail["additional_candidates_allowed"] is not False
        or fork["evaluate_same_seven_candidates"] is not True
        or fork["actor_forward_calls"] != 1
        or fork["contact_used_for_pass_fail"] is not False
        or fork["continuation_if_passed"]
        != "selected_source56_branch_then_original_binary_translation_oracle_to_endpoint60_or_failure"
        or contract["frozen_candidates"] != {
            "learned_gate": "blocked",
            "actor_learning_rate_5e_minus_5": "blocked",
            "reward_change": "blocked",
            "PPO_retraining": "blocked",
        }
    ):
        raise ValueError("tail semantic oracle definition changed")
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


def select_candidate(candidates: list[dict]) -> tuple[dict, str]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            bool(item[1]["outcome"]["terminated"]),
            float(item[1]["outcome"]["objective_score"]),
            len(item[1]["zero_normalized_action_indices"]),
            item[0],
        ),
    )
    selected = ranked[0][1]
    survivors = [row for row in candidates if not row["outcome"]["terminated"]]
    score_matches = [
        row for row in candidates
        if row["outcome"]["terminated"] == selected["outcome"]["terminated"]
        and row["outcome"]["objective_score"] == selected["outcome"]["objective_score"]
    ]
    if survivors and len(survivors) != len(candidates):
        reason = "prefer_nonterminated_candidate_then_lower_score"
    elif len(score_matches) > 1:
        minimum_zeroed = min(
            len(row["zero_normalized_action_indices"]) for row in score_matches
        )
        reason = (
            "exact_score_tie_prefer_fewer_zeroed_dimensions_then_declared_order"
            if len([row for row in score_matches if len(row["zero_normalized_action_indices"]) == minimum_zeroed]) > 1
            else "exact_score_tie_prefer_fewer_zeroed_dimensions"
        )
    else:
        reason = "lower_objective_score"
    return selected, reason


def evaluate_candidates(*, backend, policy, source: int, reference_qpos,
                        reference_qvel, objective) -> dict:
    pre_snapshot = backend.snapshot()
    pre_hidden = clone_hidden(policy.rnn_states)
    pre_state = state_record(backend.env)
    packed = policy.obs_to_tensors(backend.observation())
    result = policy.get_deterministic_action_values(packed)
    post_hidden = clone_hidden(result["rnn_states"])
    full_action = policy.preprocess_actions(result["deterministic_actions"])
    low = result["action_lows"][0].detach().cpu().numpy()
    high = result["action_highs"][0].detach().cpu().numpy()
    actor = actor_record(result, full_action[0])
    candidates = []
    for name, zero_indices in CANDIDATES.items():
        action = full_action.copy()
        if zero_indices:
            indices = np.asarray(zero_indices, np.int64)
            if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
                raise RuntimeError(f"zero leaves state-feasible support for {name} at {source}")
            action[0, indices] = 0.0
        backend.restore(pre_snapshot)
        backend.verify_restored_snapshot(pre_snapshot)
        policy.rnn_states = clone_hidden(post_hidden)
        raw = one_step(
            backend=backend, action=action, source=source,
            label=f"tail_semantic_source_{source}_{name}",
        )
        candidates.append({
            "name": name,
            "zero_normalized_action_indices": list(zero_indices),
            "normalized_action": action[0].copy(),
            "raw": raw,
            "outcome": compact_outcome(raw["row"], raw["state"], reference_qpos),
        })
    selected, reason = select_candidate(candidates)
    return {
        "source": source,
        "pre_snapshot": pre_snapshot,
        "pre_hidden": pre_hidden,
        "post_hidden": post_hidden,
        "source_state": compact_source_state(
            pre_state, source, reference_qpos, reference_qvel, objective
        ),
        "actor": {
            "raw_mu": actor["actor_mu"],
            "sigma": actor["actor_sigma"],
            "state_feasible_low": actor["state_feasible_low"],
            "state_feasible_high": actor["state_feasible_high"],
            "complete_deterministic_action": actor["deterministic_action"],
        },
        "candidates": candidates,
        "selected": selected,
        "selection_reason": reason,
    }


def restore_semantic_choice(*, backend, policy, evaluated: dict, candidate: dict) -> None:
    backend.restore(candidate["raw"]["snapshot"])
    backend.verify_restored_snapshot(candidate["raw"]["snapshot"])
    policy.rnn_states = clone_hidden(evaluated["post_hidden"])


def compact_candidate(candidate: dict) -> dict:
    return {
        "name": candidate["name"],
        "zero_normalized_action_indices": candidate["zero_normalized_action_indices"],
        "normalized_action": candidate["normalized_action"].tolist(),
        "outcome": candidate["outcome"],
    }


def continue_binary_oracle(*, backend, policy, selected_source56: dict,
                           source56_post_hidden, reference_qpos,
                           reference_qvel, objective) -> tuple[list[dict], list[dict]]:
    rows = [selected_source56["raw"]["row"]]
    decisions = []
    backend.restore(selected_source56["raw"]["snapshot"])
    backend.verify_restored_snapshot(selected_source56["raw"]["snapshot"])
    policy.rnn_states = clone_hidden(source56_post_hidden)
    for source in range(57, 60):
        evaluated = evaluate_source(
            backend=backend, policy=policy, source=source,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective,
        )
        selected, reason = choose_branch(
            evaluated["ON_raw"], evaluated["OFF_raw"]
        )
        chosen = restore_choice(
            backend=backend, policy=policy, evaluated=evaluated,
            selected=selected,
        )
        rows.append(chosen["row"])
        decisions.append({
            "source_endpoint": source,
            "selected": selected,
            "selection_reason": reason,
            "ON_score": evaluated["ON"]["objective_score"],
            "OFF_score": evaluated["OFF"]["objective_score"],
            "selected_outcome": compact_outcome(
                chosen["row"], chosen["state"], reference_qpos
            ),
        })
        if chosen["row"]["terminated"]:
            break
    return rows, decisions


def write_summary(path: Path, report: dict) -> None:
    result = report["result"]
    lines = [
        "# Tail semantic suppression oracle",
        "",
        f"- selected modes at sources 44--55: {result['selected_modes_44_to_55']}",
        f"- source-56 passing candidates: {result['source56_passing_candidates']}",
        f"- successful intervals: {result['successful_intervals']}/40",
        f"- first failure endpoint: {result['first_failure_endpoint']}",
        f"- next blocker: {result['next_blocker']}",
        "",
        "The selector uses only next-endpoint termination/score plus the predeclared least-intervention tie break.",
        "No contact signal, source-index rule, scale sweep, training or chunk commit is used.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_tail_semantic_suppression_oracle_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_tail_semantic_suppression_oracle_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)

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

    source56_report = json.loads(paths["source56_semantic_report"].read_text())
    baseline_report = json.loads(paths["baseline_oracle_report"].read_text())
    if (
        source56_report["decision"]["any_semantic_branch_passes_endpoint57"] is not False
        or baseline_report["oracle_result"]["successful_intervals"] != 36
    ):
        raise ValueError("tail semantic prerequisite evidence changed")
    expected_off_sources = baseline_report["oracle_result"]["OFF_sources"]
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution_spec, _ = load_truncated_gaussian_profile(paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
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
    boundary = load_gzip_torch(paths["boundary"])
    checkpoint = load_gzip_torch(paths["checkpoint"])

    temporary = tempfile.TemporaryDirectory(prefix=".tail_semantic_", dir=ROOT / "runs")
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
            distribution_spec=distribution_spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        captures, _ = reproduce_baseline(
            backend=backend, policy=policy, boundary=boundary,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective, expected_arrays=paths["baseline_oracle_arrays"],
            expected_off_sources=expected_off_sources,
            capture_sources=(TAIL_START,),
        )
        backend.restore(captures[TAIL_START]["snapshot"])
        backend.verify_restored_snapshot(captures[TAIL_START]["snapshot"])
        policy.rnn_states = clone_hidden(captures[TAIL_START]["pre_hidden"])

        tail_decisions = []
        selected_rows = []
        source56 = None
        for source in range(TAIL_START, FORK_SOURCE + 1):
            evaluated = evaluate_candidates(
                backend=backend, policy=policy, source=source,
                reference_qpos=reference_qpos, reference_qvel=reference_qvel,
                objective=objective,
            )
            if source == FORK_SOURCE:
                source56 = evaluated
                break
            selected = evaluated["selected"]
            restore_semantic_choice(
                backend=backend, policy=policy, evaluated=evaluated,
                candidate=selected,
            )
            selected_rows.append(selected["raw"]["row"])
            tail_decisions.append({
                "source_endpoint": source,
                "actor_forward_calls": 1,
                "source_state": evaluated["source_state"],
                "actor": evaluated["actor"],
                "candidates": [compact_candidate(row) for row in evaluated["candidates"]],
                "selected": selected["name"],
                "selection_reason": evaluated["selection_reason"],
                "contact_used_for_selection": False,
            })
            if selected["outcome"]["terminated"]:
                break
        policy.writer.close()
    finally:
        temporary.cleanup()

    if source56 is None:
        source56_candidates = []
        passing = []
        source56_selected = None
        continuation_rows = []
        continuation_decisions = []
    else:
        source56_candidates = [compact_candidate(row) for row in source56["candidates"]]
        passing = [
            row["name"] for row in source56["candidates"]
            if not row["outcome"]["terminated"]
            and row["outcome"]["objective_score"] < 1.0
        ]
        source56_selected = source56["selected"]
        if passing:
            continuation_rows, continuation_decisions = continue_binary_oracle(
                backend=backend, policy=policy,
                selected_source56=source56_selected,
                source56_post_hidden=source56["post_hidden"],
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                objective=objective,
            )
        else:
            continuation_rows = [source56_selected["raw"]["row"]]
            continuation_decisions = []

    all_selected_rows = selected_rows + continuation_rows
    failure = next((row for row in all_selected_rows if row["terminated"]), None)
    successful_intervals = 24 + sum(not row["terminated"] for row in all_selected_rows)
    next_blocker = (
        contract["decision_if_source56_passes"]
        if passing else contract["decision_if_source56_all_fail"]
    )
    selected_modes = [row["selected"] for row in tail_decisions]
    mode_counts = dict(Counter(selected_modes))

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["decision_arrays"]
    np.savez_compressed(
        arrays_path,
        tail_source_endpoint=np.asarray(
            [row["source_endpoint"] for row in tail_decisions], np.int64
        ),
        tail_selected_candidate=np.asarray(selected_modes, dtype="U40"),
        tail_candidate_score=np.asarray([
            [candidate["outcome"]["objective_score"] for candidate in row["candidates"]]
            for row in tail_decisions
        ], np.float32),
        tail_candidate_terminated=np.asarray([
            [candidate["outcome"]["terminated"] for candidate in row["candidates"]]
            for row in tail_decisions
        ], np.bool_),
        tail_selected_endpoint_qpos=np.asarray(
            [row["endpoint_qpos"] for row in selected_rows], np.float32
        ),
        tail_selected_endpoint_qvel=np.asarray(
            [row["endpoint_qvel"] for row in selected_rows], np.float32
        ),
        source56_candidate_score=np.asarray(
            [row["outcome"]["objective_score"] for row in source56_candidates],
            np.float32,
        ),
        source56_candidate_terminated=np.asarray(
            [row["outcome"]["terminated"] for row in source56_candidates],
            np.bool_,
        ),
    )
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "complete_binary_oracle_arrays_bitwise_reproduced": True,
            "source20_through_43_untouched": True,
            "source44_complete_snapshot_and_pre_forward_hidden_captured": True,
        },
        "selection_contract": {
            "candidate_order": list(CANDIDATES),
            "zero_indices": {key: list(value) for key, value in CANDIDATES.items()},
            "actuator_names": list(actuator_names),
            "actor_forward_calls_per_source": 1,
            "same_snapshot_and_post_forward_hidden_for_candidates": True,
            "ordered_rules": contract["tail_selection"]["ordered_rules"],
            "contact_used": False,
            "source_index_used": False,
            "future_beyond_one_step_used": False,
        },
        "tail_decisions": tail_decisions,
        "source56_fork": None if source56 is None else {
            "source_state": source56["source_state"],
            "actor": source56["actor"],
            "candidates": source56_candidates,
            "passing_candidates": passing,
            "selected_for_continuation": source56_selected["name"],
            "selection_reason": source56["selection_reason"],
            "binary_oracle_continuation": continuation_decisions,
        },
        "result": {
            "selected_modes_44_to_55": selected_modes,
            "selected_mode_counts": mode_counts,
            "source56_reached": source56 is not None,
            "source56_passing_candidates": passing,
            "source56_any_candidate_passes": bool(passing),
            "successful_intervals": successful_intervals,
            "forty_of_forty": successful_intervals == 40,
            "first_failure_endpoint": None if failure is None else int(failure["endpoint"]),
            "next_blocker": next_blocker,
            "semantic_suppression_family_exhausted": not bool(passing),
            "learned_gate_training_authorized": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
        },
        "decision_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "local_one_step_semantic_oracle_not_EgoEngine_RL": True,
            "does_not_search_active_corrective_actions": True,
            "does_not_search_scales_axes_or_frames": True,
            "contact_is_diagnostic_only": True,
            "no_training_or_chunk_commit": True,
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        **report["result"],
    }, indent=2))


if __name__ == "__main__":
    main()
