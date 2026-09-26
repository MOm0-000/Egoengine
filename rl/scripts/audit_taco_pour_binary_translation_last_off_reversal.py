#!/usr/bin/env python3
"""Reverse each final translation-OFF oracle decision once, without training."""

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
from audit_taco_pour_binary_translation_oracle import (  # noqa: E402
    END,
    START,
    TRANSLATION_INDICES,
    choose_branch,
    compact_outcome,
    compact_source_state,
    one_step,
    on_off_physical_delta,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
    zero_hidden,
)
from audit_taco_pour_source45_state_entry import actor_record, state_record  # noqa: E402


FORCED_SOURCES = (52, 53, 54, 55)
ARRAY_KEYS = (
    "source_endpoint",
    "selected_off",
    "actor_mu_xyz",
    "on_score",
    "off_score",
    "on_terminated",
    "off_terminated",
    "on_action",
    "off_action",
    "selected_endpoint_qpos",
    "selected_endpoint_qvel",
)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_binary_translation_last_off_reversal_gate_v1":
        raise ValueError("unsupported last-OFF reversal contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("last-OFF reversal gate is not authorized")
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
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    branches = tuple(row["forced_ON_source"] for row in contract["branches"])
    semantics = contract["branch_semantics"]
    selection = contract["selection_rule_after_reversal"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or contract.get("classification")
        != "local_engineering_oracle_myopia_audit_not_EgoEngine_RL"
        or branches != FORCED_SOURCES
        or contract["baseline"]["successful_intervals"] != 36
        or contract["baseline"]["first_failure_endpoint"] != 57
        or semantics["exact_baseline_snapshot_and_pre_forward_hidden_at_forced_source"] is not True
        or semantics["action_at_forced_source"] != "complete_frozen_deterministic_PPO_action"
        or semantics["baseline_action_at_forced_source_must_be_OFF"] is not True
        or semantics["intervention_control_steps"] != 1
        or semantics["from_next_source"] != "resume_same_one_step_binary_translation_oracle"
        or semantics["additional_forced_decisions_allowed"] is not False
        or semantics["multiple_reversals_per_branch_allowed"] is not False
        or semantics["scale_sweep_allowed"] is not False
        or selection["contact_used"] is not False
        or selection["source_index_used"] is not False
        or selection["future_beyond_one_step_used"] is not False
        or contract["primary_passing_condition"]["contact_required"] is not False
        or contract["frozen_training_candidates"]["learned_gate"] != "blocked"
        or contract["frozen_training_candidates"]["actor_learning_rate_5e_minus_5"] != "blocked"
    ):
        raise ValueError("last-OFF reversal definition changed")
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


def evaluate_source(*, backend, policy, source: int, reference_qpos: np.ndarray,
                    reference_qvel: np.ndarray, objective) -> dict:
    """Evaluate ON/OFF from one state with exactly one shared actor forward."""
    pre_snapshot = backend.snapshot()
    pre_hidden = clone_hidden(policy.rnn_states)
    pre_state = state_record(backend.env)
    packed = policy.obs_to_tensors(backend.observation())
    result = policy.get_deterministic_action_values(packed)
    post_hidden = clone_hidden(result["rnn_states"])
    action_on = policy.preprocess_actions(result["deterministic_actions"])
    action_off = action_on.copy()
    indices = np.asarray(TRANSLATION_INDICES, np.int64)
    low = result["action_lows"][0].detach().cpu().numpy()
    high = result["action_highs"][0].detach().cpu().numpy()
    if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
        raise RuntimeError(f"translation OFF leaves feasible support at source {source}")
    action_off[0, indices] = 0.0

    policy.rnn_states = clone_hidden(post_hidden)
    on = one_step(
        backend=backend, action=action_on, source=source,
        label=f"last_off_source_{source}_ON",
    )
    backend.restore(pre_snapshot)
    backend.verify_restored_snapshot(pre_snapshot)
    policy.rnn_states = clone_hidden(post_hidden)
    off = one_step(
        backend=backend, action=action_off, source=source,
        label=f"last_off_source_{source}_OFF",
    )
    greedy, reason = choose_branch(on, off)
    actor = actor_record(result, action_on[0])
    return {
        "source": source,
        "pre_snapshot": pre_snapshot,
        "pre_hidden": pre_hidden,
        "post_hidden": post_hidden,
        "source_state": compact_source_state(
            pre_state, source, reference_qpos, reference_qvel, objective
        ),
        "actor": {
            "raw_mu_xyz": actor["actor_mu"][:3],
            "sigma_xyz": actor["actor_sigma"][:3],
            "state_feasible_low_xyz": actor["state_feasible_low"][:3],
            "state_feasible_high_xyz": actor["state_feasible_high"][:3],
            "complete_deterministic_action": actor["deterministic_action"],
        },
        "action_on": action_on,
        "action_off": action_off,
        "ON_raw": on,
        "OFF_raw": off,
        "ON": compact_outcome(on["row"], on["state"], reference_qpos),
        "OFF": compact_outcome(off["row"], off["state"], reference_qpos),
        "ON_minus_OFF_separate_physical_deltas": on_off_physical_delta(on, off),
        "greedy_selected": greedy,
        "greedy_reason": reason,
    }


def restore_choice(*, backend, policy, evaluated: dict, selected: str) -> dict:
    chosen = evaluated[f"{selected}_raw"]
    backend.restore(chosen["snapshot"])
    backend.verify_restored_snapshot(chosen["snapshot"])
    policy.rnn_states = clone_hidden(evaluated["post_hidden"])
    return chosen


def append_arrays(arrays: dict[str, list], evaluated: dict, selected: str) -> None:
    chosen = evaluated[f"{selected}_raw"]
    arrays["source_endpoint"].append(evaluated["source"])
    arrays["selected_off"].append(selected == "OFF")
    arrays["actor_mu_xyz"].append(evaluated["actor"]["raw_mu_xyz"])
    arrays["on_score"].append(evaluated["ON"]["objective_score"])
    arrays["off_score"].append(evaluated["OFF"]["objective_score"])
    arrays["on_terminated"].append(evaluated["ON"]["terminated"])
    arrays["off_terminated"].append(evaluated["OFF"]["terminated"])
    arrays["on_action"].append(evaluated["action_on"][0].copy())
    arrays["off_action"].append(evaluated["action_off"][0].copy())
    arrays["selected_endpoint_qpos"].append(chosen["row"]["endpoint_qpos"])
    arrays["selected_endpoint_qvel"].append(chosen["row"]["endpoint_qvel"])


def array_result(arrays: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=(
            np.bool_ if key in {"selected_off", "on_terminated", "off_terminated"}
            else np.int64 if key == "source_endpoint"
            else np.float32
        ))
        for key, value in arrays.items()
    }


def reproduce_baseline(*, backend, policy, boundary, reference_qpos,
                       reference_qvel, objective, expected_arrays: Path,
                       expected_off_sources: list[int]) -> tuple[dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    captures = {}
    decisions = []
    arrays = {key: [] for key in ARRAY_KEYS}
    for source in range(START, END):
        if source in FORCED_SOURCES:
            captures[source] = {
                "snapshot": backend.snapshot(),
                "pre_hidden": clone_hidden(policy.rnn_states),
            }
        evaluated = evaluate_source(
            backend=backend, policy=policy, source=source,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective,
        )
        selected = evaluated["greedy_selected"]
        chosen = restore_choice(
            backend=backend, policy=policy, evaluated=evaluated, selected=selected
        )
        append_arrays(arrays, evaluated, selected)
        decisions.append({
            "source_endpoint": source,
            "selected": selected,
            "selection_reason": evaluated["greedy_reason"],
            "selected_outcome_endpoint": int(chosen["row"]["endpoint"]),
            "selected_score": float(chosen["row"]["objective_score"][0]),
            "selected_terminated": bool(chosen["row"]["terminated"]),
        })
        if chosen["row"]["terminated"]:
            break
    actual = array_result(arrays)
    expected = np.load(expected_arrays)
    for key in ARRAY_KEYS:
        if key not in expected.files or not np.array_equal(actual[key], expected[key]):
            raise RuntimeError(f"baseline oracle changed: {key}")
    off_sources = [row["source_endpoint"] for row in decisions if row["selected"] == "OFF"]
    if off_sources != expected_off_sources or len(captures) != len(FORCED_SOURCES):
        raise RuntimeError("baseline OFF decisions or forced-source captures changed")
    if decisions[-1]["selected_outcome_endpoint"] != 57 or not decisions[-1]["selected_terminated"]:
        raise RuntimeError("baseline endpoint-57 failure changed")
    return captures, {"decisions": decisions, "arrays": actual}


def branch_decision_record(evaluated: dict, selected: str, forced: bool) -> dict:
    return {
        "source_endpoint": evaluated["source"],
        "actor_forward_calls": 1,
        "pre_forward_hidden_shared": True,
        "post_forward_hidden_shared": True,
        "source_state": evaluated["source_state"],
        "actor": evaluated["actor"],
        "ON": evaluated["ON"],
        "OFF": evaluated["OFF"],
        "ON_minus_OFF_separate_physical_deltas": evaluated[
            "ON_minus_OFF_separate_physical_deltas"
        ],
        "greedy_selected": evaluated["greedy_selected"],
        "selected": selected,
        "selection_overridden_once": forced,
        "selection_reason": (
            "predeclared_single_last_OFF_reversal"
            if forced else evaluated["greedy_reason"]
        ),
        "contact_used_for_selection": False,
    }


def separate_tool_pose_delta(on: dict, off: dict) -> dict:
    on_qpos = np.asarray(on["tool_freejoint_qpos"], np.float64)
    off_qpos = np.asarray(off["tool_freejoint_qpos"], np.float64)
    on_quaternion = on_qpos[3:7] / np.linalg.norm(on_qpos[3:7])
    off_quaternion = off_qpos[3:7] / np.linalg.norm(off_qpos[3:7])
    quaternion_dot = float(np.clip(abs(np.dot(on_quaternion, off_quaternion)), -1.0, 1.0))
    return {
        "tool_position_l2_m": float(np.linalg.norm(on_qpos[:3] - off_qpos[:3])),
        "tool_rotation_distance_rad": float(2.0 * np.arccos(quaternion_dot)),
        "mixed_position_quaternion_norm_reported": False,
    }


def run_reversal(*, backend, policy, capture: dict, forced_source: int,
                 reference_qpos, reference_qvel, objective) -> tuple[dict, dict]:
    backend.restore(capture["snapshot"])
    backend.verify_restored_snapshot(capture["snapshot"])
    policy.rnn_states = clone_hidden(capture["pre_hidden"])
    decisions = []
    arrays = {key: [] for key in ARRAY_KEYS}
    source56 = None
    failure_endpoint = None
    for source in range(forced_source, END):
        evaluated = evaluate_source(
            backend=backend, policy=policy, source=source,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective,
        )
        forced = source == forced_source
        if forced and evaluated["greedy_selected"] != "OFF":
            raise RuntimeError(f"source {source} is no longer a baseline OFF decision")
        selected = "ON" if forced else evaluated["greedy_selected"]
        chosen = restore_choice(
            backend=backend, policy=policy, evaluated=evaluated, selected=selected
        )
        append_arrays(arrays, evaluated, selected)
        record = branch_decision_record(evaluated, selected, forced)
        decisions.append(record)
        if source == 56:
            source56 = {
                "source_tracking": evaluated["source_state"]["tracking"],
                "source_contact": evaluated["source_state"]["right_hand_tool_contact"],
                "ON": evaluated["ON"],
                "OFF": evaluated["OFF"],
                "ON_minus_OFF_separate_physical_deltas": evaluated[
                    "ON_minus_OFF_separate_physical_deltas"
                ],
                "ON_minus_OFF_tool_pose_separate_deltas": separate_tool_pose_delta(
                    evaluated["ON"], evaluated["OFF"]
                ),
                "greedy_selected": evaluated["greedy_selected"],
                "selected": selected,
            }
        if chosen["row"]["terminated"]:
            failure_endpoint = int(chosen["row"]["endpoint"])
            break
    suffix_success = sum(
        not decision[decision["selected"]]["terminated"]
        for decision in decisions
    )
    successful_intervals = forced_source - START + suffix_success
    forced = decisions[0]
    endpoint57 = next(
        (
            decision[decision["selected"]]
            for decision in decisions
            if decision[decision["selected"]]["endpoint"] == 57
        ),
        None,
    )
    result = {
        "name": f"force_source{forced_source}_ON",
        "forced_ON_source": forced_source,
        "forced_source_baseline_greedy_selection": forced["greedy_selected"],
        "immediate_ON_score": forced["ON"]["objective_score"],
        "immediate_OFF_score": forced["OFF"]["objective_score"],
        "immediate_ON_minus_OFF_score_cost": (
            forced["ON"]["objective_score"] - forced["OFF"]["objective_score"]
        ),
        "successful_intervals": successful_intervals,
        "first_failure_endpoint": failure_endpoint,
        "survives_endpoint57": failure_endpoint is None or failure_endpoint > 57,
        "source56_diagnostic": source56,
        "endpoint57_selected_outcome": endpoint57,
        "downstream_decisions": decisions,
    }
    return result, array_result(arrays)


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Binary translation last-OFF reversal gate",
        "",
        f"- baseline reproduced bitwise: {report['regression_gate']['baseline_decision_arrays_bitwise_reproduced']}",
        f"- any branch survives endpoint 57: {report['decision']['any_branch_survives_endpoint57']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
    ]
    for row in report["branches"]:
        lines.extend([
            f"- {row['name']}: {row['successful_intervals']}/40, "
            f"failure={row['first_failure_endpoint']}, "
            f"survives57={row['survives_endpoint57']}, "
            f"immediate ON-OFF cost={row['immediate_ON_minus_OFF_score_cost']:.9g}",
        ])
    lines.extend([
        "",
        "Each branch reverses exactly one predeclared OFF decision, then resumes the same one-step oracle.",
        "This is a read-only local engineering oracle audit, not EgoEngine RL.",
        "No training, task acceptance, or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_binary_translation_last_off_reversal_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_binary_translation_last_off_reversal_gate_v1",
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

    baseline_report = json.loads(paths["baseline_oracle_report"].read_text())
    expected_off_sources = contract["baseline"]["expected_OFF_sources"]
    if (
        baseline_report["oracle_result"]["successful_intervals"] != 36
        or baseline_report["oracle_result"]["first_failure_endpoint"] != 57
        or baseline_report["oracle_result"]["OFF_sources"] != expected_off_sources
    ):
        raise ValueError("baseline oracle report changed")
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
    if tuple(actuator_names[:3]) != (
        "R_forearm_tx_position", "R_forearm_ty_position", "R_forearm_tz_position",
    ):
        raise RuntimeError("action indices 0:3 are not right-wrist translation")
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])
    checkpoint = load_gzip_torch(paths["checkpoint"])

    temporary = tempfile.TemporaryDirectory(prefix=".last_off_reversal_", dir=ROOT / "runs")
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
        captures, baseline = reproduce_baseline(
            backend=backend, policy=policy, boundary=boundary,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective, expected_arrays=paths["baseline_oracle_arrays"],
            expected_off_sources=expected_off_sources,
        )
        branch_results = []
        branch_arrays = {}
        for forced_source in FORCED_SOURCES:
            result, arrays = run_reversal(
                backend=backend, policy=policy, capture=captures[forced_source],
                forced_source=forced_source, reference_qpos=reference_qpos,
                reference_qvel=reference_qvel, objective=objective,
            )
            branch_results.append(result)
            for key, value in arrays.items():
                branch_arrays[f"force_source{forced_source}_ON__{key}"] = value
        policy.writer.close()
    finally:
        temporary.cleanup()

    passing = [row["name"] for row in branch_results if row["survives_endpoint57"]]
    next_blocker = (
        contract["decision_if_any_pass"] if passing else contract["decision_if_all_fail"]
    )
    source55 = next(row for row in branch_results if row["forced_ON_source"] == 55)
    source55_at_56 = source55["source56_diagnostic"]
    if source55_at_56 is None:
        raise RuntimeError("source-55 reversal no longer reaches source 56")
    source55_changes_tool_and_contact = bool(
        source55_at_56["ON_minus_OFF_tool_pose_separate_deltas"][
            "tool_position_l2_m"
        ] > 0.0
        and source55_at_56["ON"]["right_hand_tool_contact"][
            "has_any_live_right_hand_tool_contact"
        ]
    )
    baseline_source56_decision = baseline_report["decision_trace"][-1]
    baseline_source56_pose_delta = separate_tool_pose_delta(
        baseline_source56_decision["ON"], baseline_source56_decision["OFF"]
    )
    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **branch_arrays)
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
            "baseline_decision_arrays_bitwise_reproduced": True,
            "baseline_successful_intervals": len(baseline["decisions"]) - 1,
            "baseline_first_failure_endpoint": baseline["decisions"][-1][
                "selected_outcome_endpoint"
            ],
            "forced_source_complete_snapshots_captured": list(FORCED_SOURCES),
            "forced_source_pre_forward_hidden_captured": list(FORCED_SOURCES),
        },
        "intervention_contract": {
            "forced_sources": list(FORCED_SOURCES),
            "one_reversal_per_branch": True,
            "forced_action": "translation_ON_complete_frozen_PPO_action",
            "from_next_source": "same_one_step_binary_translation_oracle",
            "actor_forward_calls_per_source": 1,
            "same_pre_step_snapshot_and_post_forward_hidden_for_ON_OFF": True,
            "contact_used_for_selection": False,
            "future_beyond_one_step_used": False,
        },
        "branches": branch_results,
        "decision": {
            "passing_branches": passing,
            "any_branch_survives_endpoint57": bool(passing),
            "one_step_greedy_myopia_supported": bool(passing),
            "source55_reversal_changes_source56_tool_pose_and_contact": (
                source55_changes_tool_and_contact
            ),
            "source55_reversal_restores_endpoint57_feasibility": False,
            "baseline_source56_ON_OFF_tool_pose_separate_deltas": (
                baseline_source56_pose_delta
            ),
            "source55_reversal_source56_ON_OFF_tool_pose_separate_deltas": (
                source55_at_56["ON_minus_OFF_tool_pose_separate_deltas"]
            ),
            "legacy_mixed_tool_freejoint_qpos_L2_not_interpreted_as_metres": (
                source55_at_56["ON_minus_OFF_separate_physical_deltas"][
                    "tool_freejoint_qpos_l2"
                ]
            ),
            "source55_reversal_source56_ON_contact_fingers": source55_at_56[
                "ON"
            ]["right_hand_tool_contact"]["active_finger_roles"],
            "source55_reversal_source56_OFF_contact_fingers": source55_at_56[
                "OFF"
            ]["right_hand_tool_contact"]["active_finger_roles"],
            "next_blocker": next_blocker,
            "next_read_only_direction": (
                "source56_semantic_action_attribution_from_source55_reversal_state"
                if not passing and source55_changes_tool_and_contact
                else "horizon_aware_oracle_diagnostic"
                if passing
                else "source56_semantic_authority_attribution"
            ),
            "learned_gate_training_authorized": False,
            "actor_LR_5e_minus_5_unblocked": False,
            "chunk_acceptance_or_commit_authorized": False,
        },
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "read_only_local_engineering_oracle_audit": True,
            "one_step_reversal_does_not_prove_long_horizon_optimality": True,
            "contact_is_secondary_only": True,
            "no_action_subspace_or_scale_sweep": True,
            "no_training_or_chunk_commit": True,
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "passing_branches": passing,
        "next_blocker": next_blocker,
        "branches": [
            {
                "name": row["name"],
                "successful_intervals": row["successful_intervals"],
                "first_failure_endpoint": row["first_failure_endpoint"],
                "survives_endpoint57": row["survives_endpoint57"],
            }
            for row in branch_results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
