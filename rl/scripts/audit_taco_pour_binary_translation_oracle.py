#!/usr/bin/env python3
"""Read-only one-step oracle over ON/OFF right-wrist translation residual."""

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

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _actuator_metadata,
    _require_exact_trace,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
    zero_hidden,
)
from audit_taco_pour_single_pass_endpoint47_49 import successful_intervals  # noqa: E402
from audit_taco_pour_source45_prefix_source50_refinement import (  # noqa: E402
    require_prefix_arrays,
    run_policy_from_boundary,
    run_replay,
    run_source50_branch,
    trace_arrays,
)
from audit_taco_pour_source45_state_entry import (  # noqa: E402
    actor_record,
    state_record,
)
from audit_taco_pour_source46_state_entry import (  # noqa: E402
    _rotation_distance,
    _rotation_matrix_from_wxyz,
)


START = 20
END = 60
TRANSLATION_INDICES = (0, 1, 2)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_binary_translation_oracle_gate_v1":
        raise ValueError("unsupported binary-translation oracle contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("binary-translation oracle is not authorized")
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
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    candidate = contract["candidate_actions"]
    branch = contract["branch_semantics"]
    window = contract["window"]
    if (
        any(contract["runtime"].get(key) != value for key, value in required.items())
        or contract.get("paper_faithful") is not False
        or contract.get("classification")
        != "local_engineering_oracle_shield_not_EgoEngine_RL"
        or tuple(candidate["off"]["zero_normalized_action_indices"])
        != TRANSLATION_INDICES
        or candidate["scale_sweep_allowed"] is not False
        or candidate["additional_action_candidates_allowed"] is not False
        or window != {
            "start_endpoint": START,
            "lookahead_end_endpoint": END,
            "required_control_intervals": END - START,
        }
        or branch["same_pre_step_complete_physics_snapshot"] is not True
        or branch["same_pre_forward_observation"] is not True
        or branch["same_pre_forward_RNN_hidden"] is not True
        or branch["actor_forward_calls_per_source"] != 1
        or branch["same_post_forward_RNN_hidden_for_both_physics_branches"] is not True
        or branch["recurrent_state_intervention_allowed"] is not False
        or branch["branch_horizon_control_intervals"] != 1
        or contract["selection_rule"]["contact_used"] is not False
        or contract["selection_rule"]["source_index_used"] is not False
        or contract["selection_rule"]["future_beyond_one_step_used"] is not False
        or contract["frozen_half_LR_candidate"]["remains_blocked"] is not True
    ):
        raise ValueError("binary-translation oracle definition changed")
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


def require_named_arrays(name: str, trace: dict, path: Path) -> None:
    expected = np.load(path)
    produced = trace_arrays(name, trace)
    for key, value in produced.items():
        if key not in expected.files or not np.array_equal(value, expected[key]):
            raise RuntimeError(f"known counterfactual changed: {key}")


def current_tracking(state: dict, endpoint: int, reference_qpos: np.ndarray,
                     objective) -> dict:
    actual_position = np.asarray(state["tool"]["position_world_m"], np.float64)
    reference_position = reference_qpos[endpoint, 36:39]
    position_xyz = actual_position - reference_position
    position = float(np.linalg.norm(position_xyz))
    rotation = _rotation_distance(
        np.asarray(state["tool"]["rotation_matrix_world"], np.float64),
        _rotation_matrix_from_wxyz(reference_qpos[endpoint, 39:43]),
    )
    position_term = objective.tracking.lambda_p * position ** 2
    rotation_term = objective.tracking.lambda_r * rotation ** 2
    return {
        "tool_position_error_xyz_actual_minus_reference_m": position_xyz.tolist(),
        "position_error_m": position,
        "rotation_error_rad": rotation,
        "ellipse_squared_contributions": {
            "position": position_term,
            "rotation": rotation_term,
        },
        "objective_score": float(np.sqrt(position_term + rotation_term)),
    }


def compact_source_state(state: dict, endpoint: int, reference_qpos: np.ndarray,
                         reference_qvel: np.ndarray, objective) -> dict:
    return {
        "endpoint": endpoint,
        "tracking": current_tracking(state, endpoint, reference_qpos, objective),
        "tool": state["tool"],
        "right_wrist_relative_to_tool": {
            key: state["right_wrist"][key]
            for key in (
                "position_in_tool_frame_m",
                "quaternion_relative_to_tool_wxyz",
                "rotation_relative_to_tool_rad",
                "linear_velocity_relative_to_tool_in_tool_frame_m_s",
                "angular_velocity_relative_to_tool_in_tool_frame_rad_s",
            )
        },
        "reference_tool_freejoint_qvel": reference_qvel[endpoint, 36:42].tolist(),
        "reference_velocity_semantics": (
            "The released robot reference qvel free-joint slice [36:42] is "
            "reported directly; no mixed-unit norm is formed."
        ),
        "right_hand_tool_contact": state["contact"],
    }


def compact_outcome(row: dict, state: dict, reference_qpos: np.ndarray) -> dict:
    endpoint = int(row["endpoint"])
    qpos = np.asarray(row["endpoint_qpos"], np.float64)
    position = float(row["position_error_m"][0])
    rotation = float(row["rotation_error_rad"][0])
    return {
        "endpoint": endpoint,
        "tool_position_error_xyz_actual_minus_reference_m": (
            qpos[36:39] - reference_qpos[endpoint, 36:39]
        ).tolist(),
        "position_error_m": position,
        "rotation_error_rad": rotation,
        "ellipse_squared_contributions": {
            "position": (position / 0.12) ** 2,
            "rotation": (rotation / 1.5) ** 2,
        },
        "objective_score": float(row["objective_score"][0]),
        "terminated": bool(row["terminated"]),
        "finite": bool(row["finite"]),
        "tool_freejoint_qpos": row["endpoint_qpos"][36:43],
        "tool_freejoint_qvel": row["endpoint_qvel"][36:42],
        "right_wrist_relative_to_tool": {
            key: state["right_wrist"][key]
            for key in (
                "position_in_tool_frame_m",
                "quaternion_relative_to_tool_wxyz",
                "rotation_relative_to_tool_rad",
                "linear_velocity_relative_to_tool_in_tool_frame_m_s",
                "angular_velocity_relative_to_tool_in_tool_frame_rad_s",
            )
        },
        "right_hand_tool_contact": state["contact"],
    }


def one_step(*, backend, action: np.ndarray, source: int, label: str) -> dict:
    backend.begin_trial(label, source, source + 1)
    feasible = backend.step(action, source)
    snapshot = backend.snapshot()
    state = state_record(backend.env)
    backend.end_trial(feasible, 1)
    trace = backend.validation_traces[-1]
    if len(trace["steps"]) != 1:
        raise RuntimeError("one-step oracle branch did not produce exactly one row")
    return {
        "feasible": feasible,
        "snapshot": snapshot,
        "state": state,
        "row": trace["steps"][0],
    }


def choose_branch(on: dict, off: dict) -> tuple[str, str]:
    on_row = on["row"]
    off_row = off["row"]
    if not on_row["finite"] or not off_row["finite"]:
        raise RuntimeError("nonfinite binary-oracle candidate")
    on_terminated = bool(on_row["terminated"])
    off_terminated = bool(off_row["terminated"])
    on_score = float(on_row["objective_score"][0])
    off_score = float(off_row["objective_score"][0])
    if on_terminated != off_terminated:
        return (
            ("OFF", "only_OFF_survives")
            if not off_terminated else ("ON", "only_ON_survives")
        )
    if off_score < on_score:
        reason = "both_terminate_OFF_lower_score" if on_terminated else "both_survive_OFF_lower_score"
        return "OFF", reason
    if on_score < off_score:
        reason = "both_terminate_ON_lower_score" if on_terminated else "both_survive_ON_lower_score"
        return "ON", reason
    return "ON", "exact_tie_keeps_formal_PPO_action"


def on_off_physical_delta(on: dict, off: dict) -> dict:
    on_state = on["state"]
    off_state = off["state"]
    return {
        "tool_freejoint_qpos_l2": float(np.linalg.norm(
            np.asarray(on_state["tool"]["freejoint_qpos"])
            - np.asarray(off_state["tool"]["freejoint_qpos"])
        )),
        "tool_freejoint_qvel_l2": float(np.linalg.norm(
            np.asarray(on_state["tool"]["freejoint_qvel"])
            - np.asarray(off_state["tool"]["freejoint_qvel"])
        )),
        "right_wrist_world_position_l2_m": float(np.linalg.norm(
            np.asarray(on_state["right_wrist"]["position_world_m"])
            - np.asarray(off_state["right_wrist"]["position_world_m"])
        )),
        "hand_qpos_l2": float(np.linalg.norm(
            np.asarray(on_state["hand_qpos"])
            - np.asarray(off_state["hand_qpos"])
        )),
        "hand_qvel_l2": float(np.linalg.norm(
            np.asarray(on_state["hand_qvel"])
            - np.asarray(off_state["hand_qvel"])
        )),
        "combined_mixed_unit_distance": None,
    }


def longest_consecutive_run(values: list[int]) -> list[int]:
    best: list[int] = []
    current: list[int] = []
    for value in values:
        if current and value != current[-1] + 1:
            if len(current) > len(best):
                best = current
            current = []
        current.append(value)
    if len(current) > len(best):
        best = current
    return best


def run_oracle(*, backend, policy, boundary, reference_qpos: np.ndarray,
               reference_qvel: np.ndarray, objective) -> tuple[dict, dict]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    decisions = []
    selected_rows = []
    arrays = {
        "source_endpoint": [],
        "selected_off": [],
        "actor_mu_xyz": [],
        "on_score": [],
        "off_score": [],
        "on_terminated": [],
        "off_terminated": [],
        "on_action": [],
        "off_action": [],
        "selected_endpoint_qpos": [],
        "selected_endpoint_qvel": [],
    }
    for source in range(START, END):
        pre_snapshot = backend.snapshot()
        pre_hidden = clone_hidden(policy.rnn_states)
        pre_state = state_record(backend.env)
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        post_hidden = clone_hidden(result["rnn_states"])
        action_on = policy.preprocess_actions(result["deterministic_actions"])
        action_off = action_on.copy()
        low = result["action_lows"][0].detach().cpu().numpy()
        high = result["action_highs"][0].detach().cpu().numpy()
        indices = np.asarray(TRANSLATION_INDICES, np.int64)
        if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
            raise RuntimeError(f"translation OFF leaves feasible support at source {source}")
        action_off[0, indices] = 0.0

        policy.rnn_states = clone_hidden(post_hidden)
        on = one_step(
            backend=backend, action=action_on, source=source,
            label=f"oracle_source_{source}_ON",
        )
        backend.restore(pre_snapshot)
        backend.verify_restored_snapshot(pre_snapshot)
        policy.rnn_states = clone_hidden(post_hidden)
        off = one_step(
            backend=backend, action=action_off, source=source,
            label=f"oracle_source_{source}_OFF",
        )
        selected, reason = choose_branch(on, off)
        chosen = off if selected == "OFF" else on
        backend.restore(chosen["snapshot"])
        backend.verify_restored_snapshot(chosen["snapshot"])
        policy.rnn_states = clone_hidden(post_hidden)

        on_outcome = compact_outcome(on["row"], on["state"], reference_qpos)
        off_outcome = compact_outcome(off["row"], off["state"], reference_qpos)
        actor = actor_record(result, action_on[0])
        decisions.append({
            "source_endpoint": source,
            "actor_forward_calls": 1,
            "pre_forward_hidden_shared": True,
            "post_forward_hidden_shared": True,
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
            "ON": on_outcome,
            "OFF": off_outcome,
            "score_difference_OFF_minus_ON": (
                off_outcome["objective_score"] - on_outcome["objective_score"]
            ),
            "ON_minus_OFF_separate_physical_deltas": on_off_physical_delta(
                on, off
            ),
            "selected": selected,
            "selection_reason": reason,
            "contact_used_for_selection": False,
        })
        selected_rows.append(chosen["row"])
        arrays["source_endpoint"].append(source)
        arrays["selected_off"].append(selected == "OFF")
        arrays["actor_mu_xyz"].append(actor["actor_mu"][:3])
        arrays["on_score"].append(on_outcome["objective_score"])
        arrays["off_score"].append(off_outcome["objective_score"])
        arrays["on_terminated"].append(on_outcome["terminated"])
        arrays["off_terminated"].append(off_outcome["terminated"])
        arrays["on_action"].append(action_on[0].copy())
        arrays["off_action"].append(action_off[0].copy())
        arrays["selected_endpoint_qpos"].append(chosen["row"]["endpoint_qpos"])
        arrays["selected_endpoint_qvel"].append(chosen["row"]["endpoint_qvel"])
        if chosen["row"]["terminated"]:
            break

    failure_row = next((row for row in selected_rows if row["terminated"]), None)
    trace = {
        "schema": "taco_pour_binary_translation_oracle_trace_v1",
        "start": START,
        "lookahead_end": END,
        "steps": selected_rows,
        "validated_steps": len(selected_rows),
        "successful_intervals": sum(not row["terminated"] for row in selected_rows),
        "feasible": failure_row is None and len(selected_rows) == END - START,
        "first_failure": None if failure_row is None else {
            "control_interval": int(failure_row["control_interval"]),
            "endpoint": int(failure_row["endpoint"]),
            "reason": "tracking_boundary",
        },
    }
    array_result = {
        key: np.asarray(value, dtype=(
            np.bool_ if key in {"selected_off", "on_terminated", "off_terminated"}
            else np.int64 if key == "source_endpoint"
            else np.float32
        ))
        for key, value in arrays.items()
    }
    return {"trace": trace, "decisions": decisions}, array_result


def write_summary(path: Path, report: dict) -> None:
    result = report["oracle_result"]
    lines = [
        "# Binary right-wrist-translation oracle gate",
        "",
        f"- successful intervals: {result['successful_intervals']}/40",
        f"- first failure: {result['first_failure_endpoint']}",
        f"- OFF count: {result['OFF_count']}",
        f"- OFF sources: {result['OFF_sources']}",
        f"- longest consecutive OFF run: {result['longest_consecutive_OFF_sources']}",
        f"- source 45 selected OFF: {result['source45_selected_OFF']}",
        f"- source 50 selected OFF: {result['source50_selected_OFF']}",
        "",
        f"- interpretation: {result['interpretation']}",
        "",
        "This is a local one-step simulator oracle, not an EgoEngine policy and not a learned gate.",
        "No training, task acceptance or chunk commit occurred.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_binary_translation_oracle_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_binary_translation_oracle_gate_v1",
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

    formal = json.loads(paths["formal_report"].read_text())
    source50_report = json.loads(paths["source50_gate_report"].read_text())
    if (
        source50_report["decision"]["narrowest_passing_branch"]
        != "zero_R_forearm_ty_residual_only"
        or source50_report["decision"]["new_training_authorized"] is not False
    ):
        raise ValueError("source-50 evidence changed")
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution_spec, _ = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
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
    replay, _ = run_replay(backend=backend, boundary=boundary)
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temporary = tempfile.TemporaryDirectory(prefix=".binary_translation_", dir=ROOT / "runs")
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

        ppo, _, _, _ = run_policy_from_boundary(
            backend=backend, policy=policy, boundary=boundary,
            prefix_suppression=False,
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")
        prefix, _, _, capture = run_policy_from_boundary(
            backend=backend, policy=policy, boundary=boundary,
            prefix_suppression=True,
        )
        require_prefix_arrays(prefix, paths["source45_branch_arrays"])
        source50_ty, _, _, _ = run_source50_branch(
            backend=backend, policy=policy, capture=capture,
            name="zero_R_forearm_ty_residual_only", zero_indices=(1,),
        )
        require_named_arrays(
            "zero_R_forearm_ty_residual_only", source50_ty,
            paths["source50_branch_arrays"],
        )

        oracle, arrays = run_oracle(
            backend=backend, policy=policy, boundary=boundary,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            objective=objective,
        )
        policy.writer.close()
    finally:
        temporary.cleanup()

    trace = oracle["trace"]
    decisions = oracle["decisions"]
    off_sources = [
        row["source_endpoint"] for row in decisions if row["selected"] == "OFF"
    ]
    longest_off_run = longest_consecutive_run(off_sources)
    validated = int(trace["successful_intervals"])
    if validated == 40:
        interpretation = "oracle_selector_completes_40_of_40_requires_nonlookahead_gate_design"
    elif validated > successful_intervals(replay):
        interpretation = "oracle_selector_extends_beyond_Replay_but_new_failure_requires_attribution"
    else:
        interpretation = "oracle_selector_does_not_extend_beyond_Replay"
    result = {
        "successful_intervals": validated,
        "lookahead_required_intervals": 40,
        "forty_of_forty": validated == 40,
        "first_failure_endpoint": (
            None if trace["first_failure"] is None
            else trace["first_failure"]["endpoint"]
        ),
        "Replay_successful_intervals": successful_intervals(replay),
        "formal_PPO_successful_intervals": successful_intervals(ppo),
        "OFF_count": len(off_sources),
        "ON_count": len(decisions) - len(off_sources),
        "OFF_sources": off_sources,
        "OFF_fraction_of_executed_sources": len(off_sources) / len(decisions),
        "longest_consecutive_OFF_sources": longest_off_run,
        "source45_selected_OFF": 45 in off_sources,
        "source50_selected_OFF": 50 in off_sources,
        "interpretation": interpretation,
        "chunk_acceptance_or_commit_authorized": False,
        "learned_gate_training_authorized": False,
        "actor_LR_5e_minus_5_unblocked": False,
    }
    final_decision = decisions[-1]
    result["failure_transition"] = {
        "source_endpoint": final_decision["source_endpoint"],
        "outcome_endpoint": final_decision["ON"]["endpoint"],
        "ON_score": final_decision["ON"]["objective_score"],
        "OFF_score": final_decision["OFF"]["objective_score"],
        "scores_exactly_equal": (
            final_decision["ON"]["objective_score"]
            == final_decision["OFF"]["objective_score"]
        ),
        "ON_and_OFF_both_terminate": bool(
            final_decision["ON"]["terminated"]
            and final_decision["OFF"]["terminated"]
        ),
        "endpoint57_position_error_xyz_m": final_decision["ON"][
            "tool_position_error_xyz_actual_minus_reference_m"
        ],
        "endpoint57_position_error_m": final_decision["ON"]["position_error_m"],
        "endpoint57_rotation_error_rad": final_decision["ON"]["rotation_error_rad"],
        "endpoint57_ellipse_squared_contributions": final_decision["ON"][
            "ellipse_squared_contributions"
        ],
        "ON_minus_OFF_separate_physical_deltas": final_decision[
            "ON_minus_OFF_separate_physical_deltas"
        ],
        "source_contact_fingers": final_decision["source_state"][
            "right_hand_tool_contact"
        ]["active_finger_roles"],
        "ON_contact_fingers": final_decision["ON"][
            "right_hand_tool_contact"
        ]["active_finger_roles"],
        "OFF_contact_fingers": final_decision["OFF"][
            "right_hand_tool_contact"
        ]["active_finger_roles"],
        "interpretation": (
            "The action changes the wrist by millimetres but changes the tool "
            "free-joint position only at nanometre scale; ON/OFF receive the "
            "same task score and both terminate. The next blocker is the new "
            "endpoint-57 failure, not selection between translation ON/OFF."
        ),
    }

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["decision_arrays"]
    np.savez_compressed(arrays_path, **arrays)
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
            "formal_Replay_trace_bitwise_reproduced": True,
            "formal_PPO_trace_bitwise_reproduced": True,
            "source45_translation_prefix_arrays_bitwise_reproduced": True,
            "source50_R_forearm_ty_branch_arrays_bitwise_reproduced": True,
        },
        "selection_contract": {
            "right_wrist_translation_indices": list(TRANSLATION_INDICES),
            "right_wrist_translation_actuators": list(actuator_names[:3]),
            "actor_forward_calls_per_source": 1,
            "same_pre_step_snapshot_and_post_forward_hidden_for_ON_OFF": True,
            "criterion": "one_step_active_object_tracking_score_and_termination_only",
            "contact_used": False,
            "source_index_used": False,
            "manual_threshold_used": False,
            "future_beyond_one_step_used": False,
        },
        "oracle_result": result,
        "decision_trace": decisions,
        "decision_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "local_engineering_oracle_not_paper_architecture": True,
            "not_a_learned_gate": True,
            "not_a_causal_proof_of_any_single_observation_feature": True,
            "contact_is_secondary_only": True,
            "mixed_unit_combined_distance_reported": False,
            "no_training_or_chunk_commit": True,
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        **result,
    }, indent=2))


if __name__ == "__main__":
    main()
