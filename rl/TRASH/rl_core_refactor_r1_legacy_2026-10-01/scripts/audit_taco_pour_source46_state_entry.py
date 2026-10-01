#!/usr/bin/env python3
"""Read-only source-46 state-entry gate for the frozen Pour actor."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
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
from audit_corrected_policy_decision_attribution import _contact_state  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
    zero_hidden,
)
from audit_taco_pour_single_pass_endpoint47_49 import (  # noqa: E402
    step_map,
    successful_intervals,
)


SOURCE = 46
TOOL_QPOS = slice(36, 43)
TOOL_QVEL = slice(36, 42)
EXPECTED_BRANCHES = {
    "zero_R_forearm_ty_residual_only": (1,),
    "zero_right_wrist_translation": tuple(range(0, 3)),
    "zero_complete_right_wrist": tuple(range(0, 6)),
    "zero_entire_right_hand_residual": tuple(range(0, 18)),
    "zero_full_36d_residual": tuple(range(0, 36)),
}


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source46_state_entry_gate_v1":
        raise ValueError("unsupported source-46 state-entry contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source-46 state-entry gate is not authorized")
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
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("source-46 state-entry runtime is not fail-closed")
    actual = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["branches"]
    }
    if actual != EXPECTED_BRANCHES:
        raise ValueError("source-46 branch set changed")
    if (
        contract["source"]["endpoint"] != SOURCE
        or contract["intervention_control_steps"] != 1
        or contract["continuation"] != "frozen_deterministic_PPO"
        or contract["rollout_end_endpoint"] != 60
        or contract["individual_actuator_sweep_allowed"] is not False
        or contract["scale_sweep_allowed"] is not False
        or contract["frozen_half_LR_candidate"]["remains_blocked"] is not True
    ):
        raise ValueError("source-46 intervention definition changed")
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


def _rotation_matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left).T @ np.asarray(right)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _matrix_quaternion(matrix: np.ndarray) -> list[float]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, np.float64).reshape(9))
    return quaternion.tolist()


def _wp_numpy(value) -> np.ndarray:
    import warp as wp

    return wp.to_torch(value).detach().cpu().numpy().astype(np.float64)


def _tool_body_id(model) -> int:
    matches = [
        joint for joint in range(model.njnt)
        if int(model.jnt_qposadr[joint]) == TOOL_QPOS.start
    ]
    if len(matches) != 1 or int(model.jnt_type[matches[0]]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("could not resolve the tool free joint")
    return int(model.jnt_bodyid[matches[0]])


def physical_state(env) -> dict:
    """Record geometry and velocity without inventing a mesh positive gap."""
    qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy().astype(np.float64)
    qvel = env._mjwp.get_qvel(env.ego_cfg, env.env)[0].detach().cpu().numpy().astype(np.float64)
    data = env.env.data_wp
    model = env.env.model_cpu
    site_positions = _wp_numpy(data.site_xpos)[0]
    site_matrices = _wp_numpy(data.site_xmat)[0].reshape(model.nsite, 3, 3)
    cvel = _wp_numpy(data.cvel)[0]
    subtree_com = _wp_numpy(data.subtree_com)[0]

    palm_site = int(env.env_cfg.palm_site_ids[0])
    fingertip_sites = np.asarray(env.env_cfg.fingertip_site_ids[:5], np.int64)
    palm_body = int(model.site_bodyid[palm_site])
    tool_body = _tool_body_id(model)

    tool_position = qpos[36:39]
    tool_rotation = _rotation_matrix_from_wxyz(qpos[39:43])
    palm_position = site_positions[palm_site]
    palm_rotation = site_matrices[palm_site]
    fingertip_world = site_positions[fingertip_sites]

    def point_velocity(body: int, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        angular = cvel[body, :3]
        linear_at_subtree_com = cvel[body, 3:6]
        linear = linear_at_subtree_com + np.cross(
            angular, position - subtree_com[body]
        )
        return linear, angular

    palm_linear, palm_angular = point_velocity(palm_body, palm_position)
    tool_linear, tool_angular = point_velocity(tool_body, tool_position)
    world_to_tool = tool_rotation.T
    relative_position = world_to_tool @ (palm_position - tool_position)
    relative_rotation = world_to_tool @ palm_rotation
    relative_linear = world_to_tool @ (palm_linear - tool_linear)
    relative_angular = world_to_tool @ (palm_angular - tool_angular)
    fingertip_tool = (fingertip_world - tool_position) @ tool_rotation

    return {
        "endpoint": int(env.time_indices[0]),
        "hand_qpos": qpos[:36].tolist(),
        "hand_qvel": qvel[:36].tolist(),
        "tool": {
            "freejoint_qpos": qpos[TOOL_QPOS].tolist(),
            "freejoint_qvel": qvel[TOOL_QVEL].tolist(),
            "position_world_m": tool_position.tolist(),
            "quaternion_world_wxyz": qpos[39:43].tolist(),
            "rotation_matrix_world": tool_rotation.tolist(),
            "origin_linear_velocity_world_m_s": tool_linear.tolist(),
            "angular_velocity_world_rad_s": tool_angular.tolist(),
        },
        "right_wrist": {
            "palm_site_id": palm_site,
            "palm_body_id": palm_body,
            "position_world_m": palm_position.tolist(),
            "quaternion_world_wxyz": _matrix_quaternion(palm_rotation),
            "rotation_matrix_world": palm_rotation.tolist(),
            "linear_velocity_world_m_s": palm_linear.tolist(),
            "angular_velocity_world_rad_s": palm_angular.tolist(),
            "position_in_tool_frame_m": relative_position.tolist(),
            "quaternion_relative_to_tool_wxyz": _matrix_quaternion(relative_rotation),
            "rotation_relative_to_tool_rad": _rotation_distance(
                np.eye(3), relative_rotation
            ),
            "linear_velocity_relative_to_tool_in_tool_frame_m_s": relative_linear.tolist(),
            "angular_velocity_relative_to_tool_in_tool_frame_rad_s": relative_angular.tolist(),
        },
        "right_fingertips": {
            "site_ids": fingertip_sites.tolist(),
            "positions_world_m": fingertip_world.tolist(),
            "positions_in_tool_frame_m": fingertip_tool.tolist(),
        },
        "velocity_semantics": (
            "MJWP cvel is interpreted as [angular, linear] in world-aligned "
            "coordinates about subtree_com; site/origin linear velocity adds "
            "omega cross (point - subtree_com). Relative values are then "
            "expressed in the current tool frame."
        ),
        "contact": _contact_state(env),
    }


def outcome_state(
    state: dict, row: dict, reference_qpos: np.ndarray,
) -> dict:
    endpoint = int(row["endpoint"])
    if state["endpoint"] != endpoint:
        raise RuntimeError("physical state endpoint does not match validation trace")
    qpos = np.asarray(row["endpoint_qpos"], np.float64)
    position = float(row["position_error_m"][0])
    rotation = float(row["rotation_error_rad"][0])
    return {
        **state,
        "tracking": {
            "tool_position_error_xyz_actual_minus_reference_m": (
                qpos[36:39] - reference_qpos[endpoint, 36:39]
            ).tolist(),
            "position_error_m": position,
            "rotation_error_rad": rotation,
            "objective_score": float(row["objective_score"][0]),
            "ellipse_squared_contributions": {
                "position": (position / 0.12) ** 2,
                "rotation": (rotation / 1.5) ** 2,
            },
            "terminated": bool(row["terminated"]),
        },
    }


def state_distance(candidate: dict, reference: dict) -> dict:
    cand_tool = candidate["tool"]
    ref_tool = reference["tool"]
    cand_wrist = candidate["right_wrist"]
    ref_wrist = reference["right_wrist"]
    tips = np.asarray(candidate["right_fingertips"]["positions_in_tool_frame_m"])
    ref_tips = np.asarray(reference["right_fingertips"]["positions_in_tool_frame_m"])
    tip_delta = np.linalg.norm(tips - ref_tips, axis=1)
    return {
        "tool_position_l2_m": float(np.linalg.norm(
            np.asarray(cand_tool["position_world_m"])
            - np.asarray(ref_tool["position_world_m"])
        )),
        "tool_rotation_geodesic_rad": _rotation_distance(
            np.asarray(cand_tool["rotation_matrix_world"]),
            np.asarray(ref_tool["rotation_matrix_world"]),
        ),
        "tool_freejoint_qvel_l2": float(np.linalg.norm(
            np.asarray(cand_tool["freejoint_qvel"])
            - np.asarray(ref_tool["freejoint_qvel"])
        )),
        "right_wrist_position_world_l2_m": float(np.linalg.norm(
            np.asarray(cand_wrist["position_world_m"])
            - np.asarray(ref_wrist["position_world_m"])
        )),
        "right_wrist_rotation_world_geodesic_rad": _rotation_distance(
            np.asarray(cand_wrist["rotation_matrix_world"]),
            np.asarray(ref_wrist["rotation_matrix_world"]),
        ),
        "right_wrist_position_in_tool_frame_l2_m": float(np.linalg.norm(
            np.asarray(cand_wrist["position_in_tool_frame_m"])
            - np.asarray(ref_wrist["position_in_tool_frame_m"])
        )),
        "right_wrist_tool_relative_linear_velocity_l2_m_s": float(np.linalg.norm(
            np.asarray(cand_wrist[
                "linear_velocity_relative_to_tool_in_tool_frame_m_s"
            ]) - np.asarray(ref_wrist[
                "linear_velocity_relative_to_tool_in_tool_frame_m_s"
            ])
        )),
        "right_wrist_tool_relative_angular_velocity_l2_rad_s": float(np.linalg.norm(
            np.asarray(cand_wrist[
                "angular_velocity_relative_to_tool_in_tool_frame_rad_s"
            ]) - np.asarray(ref_wrist[
                "angular_velocity_relative_to_tool_in_tool_frame_rad_s"
            ])
        )),
        "right_fingertip_tool_frame_distance_m": {
            "per_finger": tip_delta.tolist(),
            "rms": float(np.sqrt(np.mean(tip_delta ** 2))),
            "maximum": float(tip_delta.max()),
        },
        "hand_qpos_l2": float(np.linalg.norm(
            np.asarray(candidate["hand_qpos"])
            - np.asarray(reference["hand_qpos"])
        )),
        "hand_qvel_l2": float(np.linalg.norm(
            np.asarray(candidate["hand_qvel"])
            - np.asarray(reference["hand_qvel"])
        )),
        "combined_mixed_unit_distance": None,
    }


def _has_live_right_tool_contact(state: dict) -> bool:
    contact = state["contact"]
    return bool(contact["active_right_tool_fingers"] and contact["live_mjwp_contacts"])


def run_replay_with_states(*, backend, boundary) -> tuple[dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("Replay", 20, 60)
    states = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if source in (SOURCE, 47, 48, 49):
            states[source] = physical_state(backend.env)
        feasible = backend.step(np.zeros((1, 36), dtype=np.float32), source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint in (47, 48, 49):
            states[endpoint] = physical_state(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], states


def run_ppo_with_states(
    *, backend, policy, boundary,
) -> tuple[dict, dict[int, dict], dict, dict[int, np.ndarray]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    backend.begin_trial("PPO", 20, 60)
    states = {}
    capture = None
    actions = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if source in (SOURCE, 47, 48, 49):
            states[source] = physical_state(backend.env)
        if source == SOURCE:
            capture = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": clone_hidden(policy.rnn_states),
            }
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        actions[source] = action[0].copy()
        if source == SOURCE:
            capture.update({
                "actor_mu": result["mus"][0].detach().cpu().numpy().copy(),
                "actor_sigma": result["sigmas"][0].detach().cpu().numpy().copy(),
                "action_low": result["action_lows"][0].detach().cpu().numpy().copy(),
                "action_high": result["action_highs"][0].detach().cpu().numpy().copy(),
                "normalized_action": action[0].copy(),
            })
        feasible = backend.step(action, source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint in (47, 48, 49):
            states[endpoint] = physical_state(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    if capture is None:
        raise RuntimeError("formal PPO did not reach source 46")
    return backend.validation_traces[-1], states, capture, actions


def run_branch(
    *, env, backend, policy, capture: dict, name: str,
    zero_indices: tuple[int, ...],
) -> tuple[dict, dict[int, dict], dict, dict[str, np.ndarray]]:
    backend.restore(capture["environment_state"])
    backend.verify_restored_snapshot(capture["environment_state"])
    policy.rnn_states = clone_hidden(capture["pre_forward_hidden"])
    states = {SOURCE: physical_state(env)}
    backend.begin_trial(name, SOURCE, 60)
    original_action = None
    executed_action = None
    attempted = 0
    feasible = True
    for current in range(SOURCE, 60):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if current == SOURCE:
            original_action = action[0].copy()
            low = result["action_lows"][0].detach().cpu().numpy()
            high = result["action_highs"][0].detach().cpu().numpy()
            indices = np.asarray(zero_indices, np.int64)
            if len(indices) and (
                np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7)
            ):
                raise RuntimeError(f"zero leaves state-feasible support for {name}")
            action[0, indices] = 0.0
            executed_action = action[0].copy()
        feasible = backend.step(action, current)
        attempted += 1
        endpoint = int(env.time_indices[0])
        if endpoint in (47, 48, 49):
            states[endpoint] = physical_state(env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    trace = backend.validation_traces[-1]
    arrays = {
        f"{name}_endpoint": np.asarray(
            [row["endpoint"] for row in trace["steps"]], np.int64
        ),
        f"{name}_score": np.asarray(
            [row["objective_score"][0] for row in trace["steps"]], np.float64
        ),
        f"{name}_qpos": np.asarray(
            [row["endpoint_qpos"] for row in trace["steps"]], np.float32
        ),
        f"{name}_qvel": np.asarray(
            [row["endpoint_qvel"] for row in trace["steps"]], np.float32
        ),
        f"{name}_normalized_action": np.asarray(
            [row["raw_residual_action"] for row in trace["steps"]], np.float32
        ),
    }
    return trace, states, {
        "original_action": original_action,
        "executed_action": executed_action,
    }, arrays


def summarize_branch(
    *, name: str, zero_indices: tuple[int, ...], trace: dict,
    states: dict[int, dict], actions: dict, formal_ppo: dict,
    formal_states: dict[str, dict[int, dict]], reference_qpos: np.ndarray,
    actuator_names: tuple[str, ...],
) -> dict:
    rows = step_map(trace)
    formal_rows = step_map(formal_ppo)
    outcomes = {
        str(endpoint): outcome_state(states[endpoint], rows[endpoint], reference_qpos)
        for endpoint in (47, 48, 49) if endpoint in rows and endpoint in states
    }
    endpoint_48 = outcomes.get("48")
    endpoint_49 = outcomes.get("49")
    live_48 = endpoint_48 is not None and _has_live_right_tool_contact(endpoint_48)
    score_49 = None if endpoint_49 is None else endpoint_49["tracking"]["objective_score"]
    formal_score_49 = float(formal_rows[49]["objective_score"][0])
    survived_49 = endpoint_49 is not None and not endpoint_49["tracking"]["terminated"]
    passes = bool(live_48 and survived_49 and score_49 < formal_score_49)
    failure = trace["first_failure"]
    original = np.asarray(actions["original_action"])
    executed = np.asarray(actions["executed_action"])
    return {
        "name": name,
        "source_endpoint": SOURCE,
        "zero_normalized_action_indices": list(zero_indices),
        "zeroed_actuators": [actuator_names[index] for index in zero_indices],
        "formal_normalized_action": original.tolist(),
        "executed_normalized_action": executed.tolist(),
        "untargeted_action_dimensions_bitwise_unchanged": bool(np.array_equal(
            np.delete(original, zero_indices), np.delete(executed, zero_indices)
        )),
        "source_46_state": states[SOURCE],
        "outcome_states": outcomes,
        "endpoint_47_distance_to_formal_Replay": state_distance(
            states[47], formal_states["Replay"][47]
        ),
        "endpoint_47_distance_to_formal_PPO": state_distance(
            states[47], formal_states["PPO"][47]
        ),
        "endpoint_48_has_live_right_hand_tool_contact": live_48,
        "endpoint_49_survives": survived_49,
        "endpoint_49_score": score_49,
        "endpoint_49_score_delta_vs_formal_PPO": (
            None if score_49 is None else score_49 - formal_score_49
        ),
        "successful_intervals_from_source": successful_intervals(trace),
        "first_failure_endpoint_or_60": (
            60 if failure is None else int(failure["endpoint"])
        ),
        "passes_predeclared_gate": passes,
        "formal_acceptance_or_chunk_commit_allowed": False,
    }


def decision(branches: list[dict], contract: dict) -> dict:
    passed = [row for row in branches if row["passes_predeclared_gate"]]
    passed_names = [row["name"] for row in passed]
    if not passed:
        direction = contract["decision_if_all_fail"]
    elif passed_names == ["zero_R_forearm_ty_residual_only"]:
        direction = contract["decision_if_gate_passes"][passed_names[0]]
    elif passed_names == ["zero_full_36d_residual"]:
        direction = contract["decision_if_gate_passes"][passed_names[0]]
    else:
        best = min(passed, key=lambda row: row["endpoint_49_score"])
        direction = contract["decision_if_gate_passes"][best["name"]]
    return {
        "passing_branches": passed_names,
        "gate_passed": bool(passed),
        "next_read_only_or_candidate_direction": direction,
        "new_training_authorized": False,
        "actor_LR_5e_minus_5_unblocked": False,
        "reward_change_authorized": False,
        "chunk_acceptance_or_commit_authorized": False,
    }


def write_summary(path: Path, report: dict) -> None:
    lines = ["# Source-46 state-entry gate", ""]
    for row in report["branches"]:
        lines.append(
            f"- {row['name']}: contact@48="
            f"{row['endpoint_48_has_live_right_hand_tool_contact']}, "
            f"score@49={row['endpoint_49_score']}, "
            f"failure={row['first_failure_endpoint_or_60']}, "
            f"pass={row['passes_predeclared_gate']}"
        )
    lines.extend([
        "",
        f"- gate passed: {report['decision']['gate_passed']}",
        "- next direction: "
        f"{report['decision']['next_read_only_or_candidate_direction']}",
        "",
        "No training, optimizer update, reward change, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_source46_state_entry_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_source46_state_entry_gate_v1",
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
    if actuator_names[1] != "R_forearm_ty_position":
        raise RuntimeError("action index 1 is not R_forearm_ty_position")
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])
    replay, replay_states = run_replay_with_states(
        backend=backend, boundary=boundary
    )
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temporary = tempfile.TemporaryDirectory(prefix=".source46_entry_", dir=ROOT / "runs")
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
        ppo, ppo_states, capture, _ = run_ppo_with_states(
            backend=backend, policy=policy, boundary=boundary
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")

        baseline, baseline_states, _, _ = run_branch(
            env=env, backend=backend, policy=policy, capture=capture,
            name="formal_source46_suffix", zero_indices=(),
        )
        expected_suffix = [
            row for row in ppo["steps"] if int(row["endpoint"]) > SOURCE
        ]
        if baseline["steps"] != expected_suffix:
            raise RuntimeError("restored formal source-46 suffix is not exact")
        for endpoint in (47, 48, 49):
            if baseline_states[endpoint] != ppo_states[endpoint]:
                raise RuntimeError(
                    f"restored source-46 physical record differs at endpoint {endpoint}"
                )

        branches = []
        arrays = {}
        formal_states = {"Replay": replay_states, "PPO": ppo_states}
        for name, indices in EXPECTED_BRANCHES.items():
            trace, states, action_detail, branch_arrays = run_branch(
                env=env, backend=backend, policy=policy, capture=capture,
                name=name, zero_indices=indices,
            )
            branches.append(summarize_branch(
                name=name,
                zero_indices=indices,
                trace=trace,
                states=states,
                actions=action_detail,
                formal_ppo=ppo,
                formal_states=formal_states,
                reference_qpos=reference_qpos,
                actuator_names=actuator_names,
            ))
            arrays.update(branch_arrays)
        policy.writer.close()
    finally:
        temporary.cleanup()

    replay_rows = step_map(replay)
    ppo_rows = step_map(ppo)
    formal_state_report = {
        mode: {
            str(endpoint): outcome_state(
                states[endpoint],
                (replay_rows if mode == "Replay" else ppo_rows)[endpoint],
                reference_qpos,
            )
            for endpoint in (46, 47, 48, 49)
        }
        for mode, states in (("Replay", replay_states), ("PPO", ppo_states))
    }
    gate_decision = decision(branches, contract)

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "regression_gate": {
            "formal_Replay_trace_bitwise_reproduced": True,
            "formal_PPO_trace_bitwise_reproduced": True,
            "restored_source46_suffix_bitwise_reproduced": True,
            "restored_source46_physical_records_exact": True,
            "formal_Replay_successful_intervals": successful_intervals(replay),
            "formal_PPO_successful_intervals": successful_intervals(ppo),
            "formal_PPO_endpoint49_score": float(
                step_map(ppo)[49]["objective_score"][0]
            ),
        },
        "actuator_contract": {
            "names": list(actuator_names),
            "R_forearm_ty_index": 1,
            "formal_source46_normalized_action": capture[
                "normalized_action"
            ].tolist(),
            "formal_source46_right_wrist_translation": capture[
                "normalized_action"
            ][:3].tolist(),
        },
        "formal_state_entry": formal_state_report,
        "branches": branches,
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "positive_mesh_surface_gap_computed": False,
            "mj_geomDistance_used": False,
            "no_contact_pose_surrogates": [
                "right_fingertips_in_tool_frame",
                "right_wrist_pose_relative_to_tool",
                "right_wrist_velocity_relative_to_tool",
            ],
            "endpoint47_state_distance_is_not_a_success_criterion": True,
            "mixed_unit_combined_distance_reported": False,
        },
        "decision": gate_decision,
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "passing_branches": gate_decision["passing_branches"],
        "next_direction": gate_decision[
            "next_read_only_or_candidate_direction"
        ],
        "branches": [
            {
                "name": row["name"],
                "contact48": row[
                    "endpoint_48_has_live_right_hand_tool_contact"
                ],
                "score49": row["endpoint_49_score"],
                "failure": row["first_failure_endpoint_or_60"],
                "pass": row["passes_predeclared_gate"],
            }
            for row in branches
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
