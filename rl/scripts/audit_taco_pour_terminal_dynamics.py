#!/usr/bin/env python3
"""Read-only endpoint 54-60 dynamics audit for the frozen Pour actor."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_pour_objective_mapping_closed_loop import (
    IndependentThresholdMJWPVectorEnv,
    load_actor,
    load_boundary,
    trace_signature,
)
from run_mjwp_ppo import (
    MJWPVectorEnvConfig,
    _build_network_config,
    _build_ppo_config,
    _load_ego_config,
    _load_reference,
)
from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.mjwp_env import _object_pose_parts
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation
from video_to_spider.rl.replay_rl import (
    MJWPChunkBackend,
    _model_state_sha256,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
    load_truncated_gaussian_profile,
)


START_ENDPOINT = 20
END_ENDPOINT = 60
ANALYSIS_ENDPOINTS = tuple(range(54, 61))
PHASE_OFFSETS = (0, -1, -2)
POSITION_THRESHOLD_M = 0.12
ROTATION_THRESHOLD_RAD = 1.5
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HANDS = ("right", "left")
OBJECTS = ("tool", "target")
GROUPS = {
    "right_wrist_translation": tuple(range(0, 3)),
    "right_wrist_rotation": tuple(range(3, 6)),
    "right_fingers": tuple(range(6, 18)),
    "left_wrist_translation": tuple(range(18, 21)),
    "left_wrist_rotation": tuple(range(21, 24)),
    "left_fingers": tuple(range(24, 36)),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_terminal_dynamics_diagnostic_v1":
        raise ValueError("unsupported terminal-dynamics contract")
    if contract.get("status") != "frozen_read_only_no_promotion":
        raise ValueError("terminal-dynamics diagnostic is not frozen")
    execution = contract.get("execution", {})
    if (
        execution.get("device") != "cpu"
        or execution.get("training_allowed") is not False
        or execution.get("optimizer_updates") != 0
        or execution.get("local_action_perturbations_allowed") is not False
        or execution.get("start_endpoint") != START_ENDPOINT
        or execution.get("end_endpoint") != END_ENDPOINT
        or execution.get("analysis_endpoints") != list(ANALYSIS_ENDPOINTS)
        or execution.get("phase_reference_offsets") != list(PHASE_OFFSETS)
        or execution.get("runtime_reward_reference_offset_audit") != 1
    ):
        raise ValueError("terminal-dynamics execution contract changed")
    semantics = contract.get("continuation_semantics", {})
    if (
        semantics.get("axis_valid_prefix_end") != 57
        or semantics.get("axis_failure_endpoint") != 58
        or semantics.get("diagnostic_continuation_endpoints") != [59, 60]
    ):
        raise ValueError("formal and diagnostic continuation semantics changed")
    promotion = contract.get("promotion", {})
    if any(
        promotion.get(name) is not False
        for name in (
            "training_allowed",
            "chunk_commit_allowed",
            "formal_objective_change_allowed",
            "task_success_claim_allowed",
        )
    ):
        raise ValueError("terminal diagnostic must forbid training and promotion")
    inputs = {
        "boundary": args.boundary,
        "actor_manifest": args.actor_manifest,
        "actor_artifact": args.actor,
        "objective_mapping_report": args.objective_mapping_report,
        "config": args.config,
        "initialization_report": args.initialization_report,
        "protocol": args.protocol,
        "objective_profile": args.objective_profile,
        "observation_profile": args.observation_profile,
        "action_profile": args.action_profile,
        "action_distribution_profile": args.action_distribution_profile,
        "objective_mapping_implementation": (
            ROOT / "scripts/audit_taco_pour_objective_mapping_closed_loop.py"
        ),
        "implementation": Path(__file__).resolve(),
    }
    actual = {name: sha256(value) for name, value in inputs.items()}
    expected = contract.get("input_sha256", {})
    if actual != expected:
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise ValueError("hash-bound terminal diagnostic inputs changed: " + ", ".join(changed))
    return contract, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "input_sha256": actual,
    }


def rotation_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    trace = (left * right).sum(dim=(-2, -1))
    value = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return float(value.item())


def actuator_metadata(model: mujoco.MjModel) -> tuple[list[str], list[str], np.ndarray]:
    names, units, qpos_addresses = [], [], []
    for actuator in range(36):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        joint = int(model.actuator_trnid[actuator, 0])
        joint_type = int(model.jnt_type[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            unit = "m"
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            unit = "rad"
        else:
            raise ValueError("controlled hand actuator is not slide or hinge")
        names.append(name or f"actuator_{actuator}")
        units.append(unit)
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
    expected_units = ["m"] * 3 + ["rad"] * 15 + ["m"] * 3 + ["rad"] * 15
    if units != expected_units or len(set(qpos_addresses)) != 36:
        raise ValueError("the bimanual actuator grouping no longer matches the frozen contract")
    return names, units, np.asarray(qpos_addresses, dtype=np.int64)


def group_vector(values: np.ndarray) -> dict[str, list[float]]:
    return {name: values[list(indices)].tolist() for name, indices in GROUPS.items()}


def group_norms(values: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "l2": float(np.linalg.norm(values[list(indices)])),
            "max_abs": float(np.max(np.abs(values[list(indices)]), initial=0.0)),
        }
        for name, indices in GROUPS.items()
    }


def action_margin_record(
    *,
    mu: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    action: np.ndarray,
    names: list[str],
    units: list[str],
    scale: float,
) -> dict[str, Any]:
    lower_margin = action - low
    upper_margin = high - action
    nearest = np.minimum(lower_margin, upper_margin)
    at_lower = action == low
    at_upper = action == high
    global_lower_hit = at_lower & (low == -1.0)
    global_upper_hit = at_upper & (high == 1.0)
    actuator_lower_hit = at_lower & (low > -1.0)
    actuator_upper_hit = at_upper & (high < 1.0)
    return {
        "actor_mu": mu.tolist(),
        "state_feasible_low": low.tolist(),
        "state_feasible_high": high.tolist(),
        "deterministic_action": action.tolist(),
        "actuator_names": names,
        "actuator_units": units,
        "nearest_bound_margin_normalized": nearest.tolist(),
        "nearest_bound_margin_physical": (nearest * scale).tolist(),
        "clipped_mu_component_count": int(np.count_nonzero(action != mu)),
        "at_lower_bound_components": np.flatnonzero(at_lower).tolist(),
        "at_upper_bound_components": np.flatnonzero(at_upper).tolist(),
        "groups": {
            name: {
                "component_count": len(indices),
                "clipped_mu_count": int(np.count_nonzero((action != mu)[list(indices)])),
                "at_lower_bound_count": int(np.count_nonzero(at_lower[list(indices)])),
                "at_upper_bound_count": int(np.count_nonzero(at_upper[list(indices)])),
                "global_normalized_bound_hit_count": int(np.count_nonzero(
                    (global_lower_hit | global_upper_hit)[list(indices)]
                )),
                "actuator_ctrlrange_bound_hit_count": int(np.count_nonzero(
                    (actuator_lower_hit | actuator_upper_hit)[list(indices)]
                )),
                "minimum_margin_normalized": float(nearest[list(indices)].min()),
                "minimum_margin_physical": float(nearest[list(indices)].min() * scale),
                "mean_abs_deterministic_action": float(np.mean(np.abs(action[list(indices)]))),
            }
            for name, indices in GROUPS.items()
        },
    }


def phase_record(env, qpos: torch.Tensor, endpoint: int) -> dict[str, Any]:
    actual_position, actual_rotation = _object_pose_parts(qpos, int(env.ego_cfg.nq_obj))[0]
    actual_position = actual_position[0]
    actual_rotation = actual_rotation[0]
    def compare(offset: int) -> dict[str, Any]:
        reference_endpoint = endpoint + offset
        reference_position, reference_rotation = _object_pose_parts(
            env.qpos_ref[reference_endpoint : reference_endpoint + 1],
            int(env.ego_cfg.nq_obj),
        )[0]
        delta = actual_position - reference_position[0]
        position_error = float(torch.linalg.vector_norm(delta).item())
        orientation_error = rotation_distance(actual_rotation, reference_rotation[0])
        score = math.sqrt(
            (position_error / POSITION_THRESHOLD_M) ** 2
            + (orientation_error / ROTATION_THRESHOLD_RAD) ** 2
        )
        return {
            "reference_endpoint": reference_endpoint,
            "reference_position_m": reference_position[0].cpu().tolist(),
            "position_error_vector_actual_minus_reference_m": delta.cpu().tolist(),
            "position_error_m": position_error,
            "rotation_error_rad": orientation_error,
            "axis_intercept_score": score,
        }
    comparisons = {str(offset): compare(offset) for offset in PHASE_OFFSETS}
    return {
        "actual_tool_position_m": actual_position.cpu().tolist(),
        "comparisons": comparisons,
        # This is not another phase candidate. It audits the goal actually used
        # by the current runtime after time_indices has already been incremented.
        "runtime_scoring_target": compare(env.observation_contract.goal_reference_offset),
        "best_position_offset": min(
            PHASE_OFFSETS, key=lambda offset: comparisons[str(offset)]["position_error_m"]
        ),
        "best_rotation_offset": min(
            PHASE_OFFSETS, key=lambda offset: comparisons[str(offset)]["rotation_error_rad"]
        ),
        "best_axis_score_offset": min(
            PHASE_OFFSETS, key=lambda offset: comparisons[str(offset)]["axis_intercept_score"]
        ),
    }


def contact_record(flags: np.ndarray, previous: set[str]) -> tuple[dict, set[str]]:
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (2, 2, 5):
        raise ValueError("expected (right,left) x (tool,target) x five fingers")
    active = {
        f"{hand}:{obj}:{finger}"
        for hand_index, hand in enumerate(HANDS)
        for object_index, obj in enumerate(OBJECTS)
        for finger_index, finger in enumerate(FINGERS)
        if flags[hand_index, object_index, finger_index]
    }
    return {
        "semantics": "reward_facing_live_finger_object_contacts_not_all_scene_contacts",
        "active_roles": sorted(active),
        "added_since_previous_endpoint": sorted(active - previous),
        "removed_since_previous_endpoint": sorted(previous - active),
        "changed_since_previous_endpoint": active != previous,
        "by_hand_object": {
            f"{hand}:{obj}": [
                finger
                for finger_index, finger in enumerate(FINGERS)
                if flags[hand_index, object_index, finger_index]
            ]
            for hand_index, hand in enumerate(HANDS)
            for object_index, obj in enumerate(OBJECTS)
        },
    }, active


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase = {}
    for offset in PHASE_OFFSETS:
        items = [row["phase"]["comparisons"][str(offset)] for row in rows]
        phase[str(offset)] = {
            "mean_position_error_m": float(np.mean([item["position_error_m"] for item in items])),
            "mean_rotation_error_rad": float(np.mean([item["rotation_error_rad"] for item in items])),
            "mean_axis_intercept_score": float(np.mean([item["axis_intercept_score"] for item in items])),
            "best_position_count": sum(row["phase"]["best_position_offset"] == offset for row in rows),
            "best_rotation_count": sum(row["phase"]["best_rotation_offset"] == offset for row in rows),
            "best_axis_score_count": sum(row["phase"]["best_axis_score_offset"] == offset for row in rows),
        }
    action_groups = {}
    for group in GROUPS:
        values = [row["incoming_action"]["groups"][group] for row in rows]
        action_groups[group] = {
            "steps_with_any_bound_hit": sum(
                item["at_lower_bound_count"] + item["at_upper_bound_count"] > 0
                for item in values
            ),
            "total_bound_hit_components": sum(
                item["at_lower_bound_count"] + item["at_upper_bound_count"]
                for item in values
            ),
            "global_normalized_bound_hit_components": sum(
                item["global_normalized_bound_hit_count"] for item in values
            ),
            "actuator_ctrlrange_bound_hit_components": sum(
                item["actuator_ctrlrange_bound_hit_count"] for item in values
            ),
            "steps_with_global_normalized_bound_hit": sum(
                item["global_normalized_bound_hit_count"] > 0 for item in values
            ),
            "steps_with_actuator_ctrlrange_bound_hit": sum(
                item["actuator_ctrlrange_bound_hit_count"] > 0 for item in values
            ),
            "steps_with_mu_clipped": sum(item["clipped_mu_count"] > 0 for item in values),
            "minimum_margin_normalized": min(item["minimum_margin_normalized"] for item in values),
            "mean_abs_action_across_steps": float(
                np.mean([item["mean_abs_deterministic_action"] for item in values])
            ),
        }
    runtime_position_differences = [
        abs(
            row["runtime_reported_tracking_error"]["position_error_m"]
            - row["phase"]["runtime_scoring_target"]["position_error_m"]
        )
        for row in rows
    ]
    runtime_rotation_differences = [
        abs(
            row["runtime_reported_tracking_error"]["rotation_error_rad"]
            - row["phase"]["runtime_scoring_target"]["rotation_error_rad"]
        )
        for row in rows
    ]
    terminal_same_time = rows[-1]["phase"]["comparisons"]["0"]
    terminal_runtime = rows[-1]["phase"]["runtime_scoring_target"]
    return {
        "phase_comparison": phase,
        "incoming_action_bound_use": action_groups,
        "contact_transition_endpoints": [
            row["endpoint"] for row in rows if row["contact"]["changed_since_previous_endpoint"]
        ],
        "tool_position_error_growth_m": {
            "endpoint_54": rows[0]["phase"]["comparisons"]["0"]["position_error_m"],
            "endpoint_60": rows[-1]["phase"]["comparisons"]["0"]["position_error_m"],
            "increase": (
                rows[-1]["phase"]["comparisons"]["0"]["position_error_m"]
                - rows[0]["phase"]["comparisons"]["0"]["position_error_m"]
            ),
        },
        "runtime_reward_alignment": {
            "declared_reward_endpoint": "t_plus_1",
            "expected_reference_offset_from_outcome_endpoint": 0,
            "observed_reference_offset_from_outcome_endpoint": 1,
            "reported_error_matches_observed_plus_one_target": (
                max(runtime_position_differences, default=math.inf) < 1e-6
                and max(runtime_rotation_differences, default=math.inf) < 1e-6
            ),
            "maximum_position_error_recompute_difference_m": max(
                runtime_position_differences, default=math.inf
            ),
            "maximum_rotation_error_recompute_difference_rad": max(
                runtime_rotation_differences, default=math.inf
            ),
            "off_by_one_detected": True,
            "endpoint_60": {
                "runtime_target_reference_endpoint": terminal_runtime["reference_endpoint"],
                "runtime_position_error_m": terminal_runtime["position_error_m"],
                "same_time_reference_endpoint": terminal_same_time["reference_endpoint"],
                "same_time_position_error_m": terminal_same_time["position_error_m"],
                "same_time_rotation_error_rad": terminal_same_time["rotation_error_rad"],
                "same_time_independent_threshold_pass": (
                    terminal_same_time["position_error_m"] <= POSITION_THRESHOLD_M
                    and terminal_same_time["rotation_error_rad"] <= ROTATION_THRESHOLD_RAD
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--actor-manifest", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--objective-mapping-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--action-profile", type=Path, required=True)
    parser.add_argument("--action-distribution-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    contract, contract_artifact = load_contract(args.contract, args)
    boundary, boundary_artifact = load_boundary(args.boundary)
    actor_state, actor_artifact = load_actor(args.actor_manifest, args.actor)
    mapping_report = json.loads(args.objective_mapping_report.read_text())
    prior_branch = mapping_report["branches"]["independent_thresholds"]
    _, initialization = load_accepted_initialization(args.initialization_report, args.config)
    base_objective = load_runtime_objective(
        args.protocol,
        args.objective_profile,
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    objective = replace(
        base_objective,
        objective_id="taco_pour_terminal_dynamics_independent_threshold_continuation",
        tracking_metric_name="max_independent_threshold_ratio",
        provenance=(
            "Read-only diagnostic continuation under the previously declared "
            "independent thresholds; not an author-recovered objective."
        ),
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=False
    )
    residual, residual_artifact = load_residual_action_profile(args.action_profile)
    action_spec, distribution_artifact = load_truncated_gaussian_profile(
        args.action_distribution_profile
    )
    config = _load_ego_config(str(args.config), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = IndependentThresholdMJWPVectorEnv(
        config,
        reference,
        num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=False,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,),
            object_roles=OBJECTS,
            objective=objective,
            observation=observation,
            residual=residual,
        ),
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    names, units, qpos_addresses = actuator_metadata(env.env.model_cpu)
    ppo_config = replace(
        _build_ppo_config(
            num_envs=1,
            horizon_length=40,
            seq_length=4,
            max_epochs=8,
            learning_rate=1e-4,
            device="cpu",
            asymmetric_critic=None,
        ),
        clip_actions=False,
    )
    rows = []
    previous_contacts: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="egoengine_terminal_dynamics_") as temp_dir:
        agent = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp_dir),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=action_spec,
        )
        agent.model.load_state_dict(actor_state)
        if _model_state_sha256(agent.model.state_dict()) != actor_artifact["actor_state_sha256"]:
            raise ValueError("frozen actor changed during terminal diagnostic load")
        agent.set_eval()
        agent.rnn_states = [
            state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
        ]
        backend = MJWPChunkBackend(env)
        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        backend.begin_trial("independent_threshold_diagnostic_continuation", 20, 60)
        valid_steps = 0
        for source_endpoint in range(START_ENDPOINT, END_ENDPOINT):
            observation_tensor = agent.obs_to_tensors(backend.observation())
            values = agent.get_deterministic_action_values(observation_tensor)
            agent.rnn_states = values["rnn_states"]
            mu = values["mus"][0].detach().cpu().numpy()
            low = values["action_lows"][0].detach().cpu().numpy()
            high = values["action_highs"][0].detach().cpu().numpy()
            deterministic = values["deterministic_actions"][0].detach().cpu().numpy()
            action = agent.preprocess_actions(values["deterministic_actions"])
            accepted = backend.step(action, source_endpoint)
            outcome_endpoint = source_endpoint + 1
            info = backend.last_info
            contact, active = contact_record(info["contact_flags"][0], previous_contacts)
            previous_contacts = active
            if outcome_endpoint in ANALYSIS_ENDPOINTS:
                qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)
                qvel = env._mjwp.get_qvel(env.ego_cfg, env.env)
                phase = phase_record(env, qpos, outcome_endpoint)
                current_hand_qpos = qpos[0, qpos_addresses].detach().cpu().numpy()
                next_reference = env.ctrl_ref[outcome_endpoint + 1, :36].cpu().numpy()
                hand_to_next = current_hand_qpos - next_reference
                object_velocity = qvel[0, -12:].reshape(2, 6)[0, :3].detach().cpu().numpy()
                reference_velocity = env.qvel_ref[outcome_endpoint, -12:].reshape(2, 6)[0, :3].cpu().numpy()
                rows.append(
                    {
                        "endpoint": outcome_endpoint,
                        "axis_semantics": (
                            "formal_valid_prefix"
                            if outcome_endpoint <= 57
                            else "formal_failure_endpoint"
                            if outcome_endpoint == 58
                            else "diagnostic_continuation_only"
                        ),
                        "phase": phase,
                        "runtime_reported_tracking_error": {
                            "position_error_m": float(info["object_position_error"][0, 0]),
                            "rotation_error_rad": float(info["object_rotation_error"][0, 0]),
                        },
                        "incoming_action_source_endpoint": source_endpoint,
                        "incoming_action": action_margin_record(
                            mu=mu,
                            low=low,
                            high=high,
                            action=deterministic,
                            names=names,
                            units=units,
                            scale=action_spec.residual_scale,
                        ),
                        "hand_qpos_minus_next_reference_ctrl": {
                            "next_reference_endpoint": outcome_endpoint + 1,
                            "signed_full_36": hand_to_next.tolist(),
                            "by_group": group_vector(hand_to_next),
                            "group_norms": group_norms(hand_to_next),
                        },
                        "tool_linear_velocity": {
                            "actual_m_per_s": object_velocity.tolist(),
                            "reference_m_per_s": reference_velocity.tolist(),
                            "actual_minus_reference_m_per_s": (
                                object_velocity - reference_velocity
                            ).tolist(),
                        },
                        "contact": contact,
                    }
                )
            if not accepted:
                break
            valid_steps += 1
        backend.end_trial(valid_steps == 40, valid_steps)
        trace = backend.validation_traces[-1]
        agent.writer.close()

    rerun_signature = trace_signature(trace)
    prior_signature = prior_branch["trajectory_signature_sha256"]
    regression = {
        "expected_validated_steps": 39,
        "actual_validated_steps": trace["validated_steps"],
        "expected_failure_endpoint": 60,
        "actual_failure": trace["first_failure"],
        "expected_trajectory_signature_sha256": prior_signature,
        "actual_trajectory_signature_sha256": rerun_signature,
        "passed": (
            trace["validated_steps"] == 39
            and trace["first_failure"]["endpoint"] == 60
            and rerun_signature == prior_signature
        ),
    }
    if not regression["passed"]:
        raise RuntimeError("terminal diagnostic did not reproduce the frozen continuation")
    report_summary = summary(rows)
    if not report_summary["runtime_reward_alignment"][
        "reported_error_matches_observed_plus_one_target"
    ]:
        raise RuntimeError("runtime tracking error does not match its observed reference target")
    report = {
        "schema": "taco_pour_terminal_dynamics_report_v1",
        "status": "completed_read_only_no_promotion",
        "paper_faithful": False,
        "scope": {
            "training_executed": False,
            "optimizer_updates": 0,
            "local_action_perturbations_executed": False,
            "device": "cpu",
            "analysis_endpoints": list(ANALYSIS_ENDPOINTS),
            "phase_offsets": list(PHASE_OFFSETS),
            "endpoint_59_60_semantics": (
                "diagnostic continuation after the formal axis-intercept objective "
                "would have terminated at endpoint 58"
            ),
            "contact_semantics": "reward-facing finger-object flags; not all scene contacts",
        },
        "contract": contract_artifact,
        "boundary": boundary_artifact,
        "actor": actor_artifact,
        "objective_mapping_report": {
            "path": str(args.objective_mapping_report.resolve()),
            "sha256": sha256(args.objective_mapping_report),
        },
        "objective": objective.as_report(),
        "observation": observation.as_report(),
        "residual_action": residual_artifact,
        "action_distribution": distribution_artifact,
        "actuator_groups": {name: list(indices) for name, indices in GROUPS.items()},
        "trajectory_regression_gate": regression,
        "endpoint_rows": rows,
        "summary": report_summary,
        "promotion": {
            "chunk_committed": False,
            "formal_objective_changed": False,
            "full_RL_authorized": False,
            "task_success_claimed": False,
        },
    }
    args.output.mkdir(parents=True)
    output = args.output / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "trajectory_regression_gate": regression,
        "summary": report["summary"],
        "report": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
