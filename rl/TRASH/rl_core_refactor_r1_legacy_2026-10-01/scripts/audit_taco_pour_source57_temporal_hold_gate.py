#!/usr/bin/env python3
"""Read-only source-57 temporal hold attribution for the Pour tail."""

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
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    evaluate_candidates,
    load_contract as load_tail_contract,
    restore_semantic_choice,
)


SOURCE = 57
CANDIDATES = (
    ("formal_source57_PPO", "keep", ()),
    ("translation_OFF", "zero", (0, 1, 2)),
    ("hold_source56_right_wrist_rotation", "copy", (3, 4, 5)),
    ("hold_source56_right_fingers", "copy", tuple(range(6, 18))),
    (
        "hold_source56_right_wrist_rotation_and_fingers",
        "copy",
        tuple(range(3, 18)),
    ),
    ("hold_source56_entire_right_hand", "copy", tuple(range(0, 18))),
)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source57_temporal_hold_gate_v1":
        raise ValueError("unsupported source57 temporal-hold contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source57 temporal-hold gate is not authorized")
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
        "actor_learning_rate_5e_minus_5_training_allowed": False,
        "learned_gate_training_allowed": False,
        "PPO_retraining_allowed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    declared = []
    for row in contract["candidates"]:
        operation = {
            "keep_source57_action": "keep",
            "zero_source57_indices": "zero",
            "copy_source56_action_indices": "copy",
        }[row["operation"]]
        declared.append((row["name"], operation, tuple(row.get("indices", ()))))
    source_state = contract["source_state"]
    semantics = contract["source56_residual_semantics"]
    primary = contract["primary_gate"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or tuple(declared) != CANDIDATES
        or source_state["source_endpoint"] != SOURCE
        or tuple(source_state["parent_sources_reproduced"]) != tuple(range(44, 57))
        or source_state["actor_forward_calls"] != 1
        or source_state["same_complete_physics_snapshot_for_all_candidates"] is not True
        or source_state["same_pre_forward_observation_and_RNN_hidden"] is not True
        or source_state["same_post_forward_RNN_hidden_for_all_physics_branches"] is not True
        or semantics["representation"] != "normalized_state_feasible_residual_action"
        or semantics["copied_values_must_be_inside_source57_state_feasible_bounds"] is not True
        or semantics["silent_clamp_allowed"] is not False
        or semantics["scale_sweep_allowed"] is not False
        or semantics["axis_sweep_allowed"] is not False
        or primary["endpoint58_does_not_terminate"] is not True
        or primary["endpoint58_objective_score_strictly_below"] != 1.0
        or primary["contact_used_for_pass_fail"] is not False
        or contract["frozen_decisions"] != {
            "actor_learning_rate_5e_minus_5": "blocked",
            "learned_gate": "blocked",
            "reward_change": "blocked",
            "PPO_retraining": "blocked",
            "chunk_commit": "blocked",
        }
    ):
        raise ValueError("source57 temporal-hold definition changed")
    paths = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _assert_scores_exact(produced: dict, expected: dict) -> None:
    expected_scores = {
        row["name"]: row["outcome"]["objective_score"]
        for row in expected["candidates"]
    }
    for row in produced["candidates"]:
        if row["outcome"]["objective_score"] != expected_scores[row["name"]]:
            raise RuntimeError(
                f"parent candidate changed at source {produced['source']}: {row['name']}"
            )


def _contact_summary(state: dict) -> dict:
    contact = state["right_hand_tool_contact"]
    live = contact["live_contacts"]
    return {
        "active_finger_roles": contact["active_finger_roles"],
        "live_contact_count": len(live),
        "sum_normal_force": float(sum(row["normal_force"] for row in live)),
        "live_contacts": live,
    }


def build_action_candidates(action57: np.ndarray, action56: np.ndarray,
                            low: np.ndarray, high: np.ndarray) -> list[dict]:
    result = []
    for name, operation, indices_tuple in CANDIDATES:
        action = action57.copy()
        indices = np.asarray(indices_tuple, np.int64)
        copied = np.asarray([], np.float64)
        if operation == "zero":
            copied = np.zeros(len(indices), np.float64)
            action[0, indices] = 0.0
        elif operation == "copy":
            copied = action56[indices]
            action[0, indices] = copied
        feasible = bool(
            not len(indices)
            or (
                np.all(action[0, indices] >= low[indices] - 1e-7)
                and np.all(action[0, indices] <= high[indices] + 1e-7)
            )
        )
        if feasible and len(indices):
            lower_margin = float(np.min(action[0, indices] - low[indices]))
            upper_margin = float(np.min(high[indices] - action[0, indices]))
        else:
            lower_margin = None
            upper_margin = None
        result.append({
            "name": name,
            "operation": operation,
            "indices": list(indices_tuple),
            "normalized_action": action,
            "copied_values": copied.tolist(),
            "inside_source57_state_feasible_bounds": feasible,
            "minimum_lower_bound_margin": lower_margin,
            "minimum_upper_bound_margin": upper_margin,
        })
    return result


def continue_selector(*, backend, policy, branch: dict, post_hidden,
                      reference_qpos, reference_qvel, objective) -> dict:
    backend.restore(branch["raw"]["snapshot"])
    backend.verify_restored_snapshot(branch["raw"]["snapshot"])
    policy.rnn_states = clone_hidden(post_hidden)
    decisions = []
    rows = [branch["raw"]["row"]]
    for source in (58, 59):
        evaluated = evaluate_source(
            backend=backend,
            policy=policy,
            source=source,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        selected, reason = choose_branch(evaluated["ON_raw"], evaluated["OFF_raw"])
        chosen = restore_choice(
            backend=backend, policy=policy, evaluated=evaluated, selected=selected
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
    failure = next((row for row in rows if row["terminated"]), None)
    successful_intervals = 37 + sum(not row["terminated"] for row in rows)
    return {
        "binary_selector_decisions": decisions,
        "successful_intervals": successful_intervals,
        "forty_of_forty": successful_intervals == 40,
        "first_failure_endpoint": None if failure is None else int(failure["endpoint"]),
    }


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Source-57 temporal hold gate",
        "",
        f"- passing candidates: {report['decision']['passing_candidates']}",
        f"- best successful intervals: {report['decision']['best_successful_intervals']}/40",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
        "All branches share one source-57 snapshot and one actor forward.",
        "No copied residual is silently clamped and contact is diagnostic only.",
        "No training or chunk commit is authorized.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/taco_pour_source57_temporal_hold_gate_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_source57_temporal_hold_gate_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    tail_report = json.loads(paths["tail_report"].read_text())
    transition_report = json.loads(paths["transition_report"].read_text())
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

    temporary = tempfile.TemporaryDirectory(prefix=".source57_hold_", dir=ROOT / "runs")
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

        source56_action = None
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
                raise RuntimeError(f"parent selected mode changed at source {source}")
            restore_semantic_choice(
                backend=backend,
                policy=policy,
                evaluated=evaluated,
                candidate=evaluated["selected"],
            )
            if source == 56:
                source56_action = np.asarray(
                    evaluated["actor"]["complete_deterministic_action"], np.float32
                )
        if source56_action is None:
            raise RuntimeError("source56 action was not reproduced")

        pre_snapshot = backend.snapshot()
        pre_state = transition_report["transition_audit"]["source57_state"]
        packed = policy.obs_to_tensors(backend.observation())
        actor_result = policy.get_deterministic_action_values(packed)
        post_hidden = clone_hidden(actor_result["rnn_states"])
        action57 = policy.preprocess_actions(actor_result["deterministic_actions"])
        low = actor_result["action_lows"][0].detach().cpu().numpy()
        high = actor_result["action_highs"][0].detach().cpu().numpy()
        if not np.array_equal(
            action57[0],
            np.asarray(
                transition_report["transition_audit"]["actor_source56_to_57"][
                    "source57_normalized_action"
                ],
                np.float32,
            ),
        ):
            raise RuntimeError("source57 deterministic action changed")

        candidates = build_action_candidates(action57, source56_action, low, high)
        branch_results = []
        for candidate in candidates:
            if not candidate["inside_source57_state_feasible_bounds"]:
                branch_results.append({
                    **{key: value for key, value in candidate.items() if key != "normalized_action"},
                    "normalized_action": candidate["normalized_action"][0].tolist(),
                    "executed": False,
                    "passes_endpoint58": False,
                    "continuation": None,
                })
                continue
            backend.restore(pre_snapshot)
            backend.verify_restored_snapshot(pre_snapshot)
            policy.rnn_states = clone_hidden(post_hidden)
            raw = one_step(
                backend=backend,
                action=candidate["normalized_action"],
                source=57,
                label=f"source57_temporal_hold_{candidate['name']}",
            )
            outcome = compact_outcome(raw["row"], raw["state"], reference_qpos)
            full_state = compact_source_state(
                raw["state"], 58, reference_qpos, reference_qvel, objective
            )
            passes = not outcome["terminated"] and outcome["objective_score"] < 1.0
            branch = {
                **{key: value for key, value in candidate.items() if key != "normalized_action"},
                "normalized_action": candidate["normalized_action"][0].tolist(),
                "executed": True,
                "raw": raw,
                "outcome": outcome,
                "endpoint58_contact": _contact_summary(full_state),
                "endpoint58_tool_freejoint_velocity": full_state["tool"]["freejoint_qvel"],
                "passes_endpoint58": passes,
                "continuation": None,
            }
            branch_results.append(branch)

        formal = next(row for row in branch_results if row["name"] == "formal_source57_PPO")
        off = next(row for row in branch_results if row["name"] == "translation_OFF")
        expected = tail_report["source56_fork"]["binary_oracle_continuation"][0]
        if (
            formal["outcome"]["objective_score"] != expected["ON_score"]
            or off["outcome"]["objective_score"] != expected["OFF_score"]
        ):
            raise RuntimeError("formal source57 anchors changed")

        passing = [row for row in branch_results if row["passes_endpoint58"]]
        for branch in passing:
            branch["continuation"] = continue_selector(
                backend=backend,
                policy=policy,
                branch=branch,
                post_hidden=post_hidden,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                objective=objective,
            )
        policy.writer.close()
    finally:
        temporary.cleanup()

    serializable = []
    for branch in branch_results:
        serializable.append({
            key: value for key, value in branch.items() if key not in {"raw"}
        })
    best = max(
        (
            row["continuation"]["successful_intervals"]
            for row in serializable if row["continuation"] is not None
        ),
        default=37,
    )
    passing_names = [row["name"] for row in serializable if row["passes_endpoint58"]]
    next_blocker = (
        "temporal_hold_mechanism_requires_state_explainable_persistence_candidate"
        if passing_names
        else "temporal_hold_insufficient_active_lag_correction_required"
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
            "tail_selected_sources44_through56_exactly_reproduced": True,
            "formal_source57_PPO_anchor_exactly_reproduced": True,
            "translation_OFF_anchor_exactly_reproduced": True,
            "source57_actor_forward_calls": 1,
            "same_source57_snapshot_and_post_forward_hidden_for_all_candidates": True,
        },
        "actuator_names": list(actuator_names),
        "source57_state": pre_state,
        "source56_normalized_action": source56_action.tolist(),
        "source57_normalized_action": action57[0].tolist(),
        "source57_state_feasible_low": low.tolist(),
        "source57_state_feasible_high": high.tolist(),
        "branches": serializable,
        "decision": {
            "passing_candidates": passing_names,
            "any_temporal_hold_passes_endpoint58": bool(passing_names),
            "best_successful_intervals": best,
            "forty_of_forty": best == 40,
            "source57_temporal_switch_is_sufficient": bool(passing_names),
            "actor_LR_5e_minus_5_unblocked": False,
            "learned_gate_training_authorized": False,
            "reward_change_authorized": False,
            "PPO_retraining_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": next_blocker,
        },
        "measurement_boundaries": {
            "copies_normalized_residual_not_command_or_joint_motion": True,
            "all_copied_actions_checked_against_source57_support": True,
            "no_silent_clamp": True,
            "contact_is_diagnostic_only": True,
            "no_source57_axis_or_scale_sweep": True,
            "no_training_or_chunk_commit": True,
        },
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["arrays"]
    np.savez_compressed(
        arrays_path,
        candidate=np.asarray([row["name"] for row in serializable], dtype="U64"),
        feasible=np.asarray([
            row["inside_source57_state_feasible_bounds"] for row in serializable
        ], np.bool_),
        passes_endpoint58=np.asarray([
            row["passes_endpoint58"] for row in serializable
        ], np.bool_),
        score=np.asarray([
            np.nan if not row["executed"] else row["outcome"]["objective_score"]
            for row in serializable
        ], np.float64),
        position_error_xyz=np.asarray([
            [np.nan, np.nan, np.nan] if not row["executed"] else
            row["outcome"]["tool_position_error_xyz_actual_minus_reference_m"]
            for row in serializable
        ], np.float64),
        contact_force=np.asarray([
            np.nan if not row["executed"] else
            row["endpoint58_contact"]["sum_normal_force"]
            for row in serializable
        ], np.float64),
        action=np.asarray([
            row["normalized_action"] for row in serializable
        ], np.float32),
    )
    report["arrays"] = {
        "path": str(arrays_path.resolve()),
        "sha256": sha256(arrays_path),
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        **report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
