#!/usr/bin/env python3
"""Audit corrected Pour reference timing, recurrent context and action frame."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _load_gzip_torch,
    _require_exact_trace,
    _sha256,
)
from audit_corrected_input_bifurcation_attribution import (  # noqa: E402
    _bounds_and_mode,
    _forward_raw,
    _normalization_state,
    _run_replay_capture,
)
from audit_corrected_local_controllability import _clone_rnn  # noqa: E402


AUDIT_ENDPOINTS = tuple(range(40, 48))
CAPTURE_ENDPOINTS = tuple(range(39, 48))
REFERENCE_SLICE = slice(126, 216)
RIGHT_WRIST_Y = 1


def _load_contract(path: Path) -> tuple[dict, dict]:
    contract_path = path
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_reference_timing_action_frame_audit_v1":
        raise ValueError("unsupported timing/action-frame audit contract")
    if contract.get("status") != "authorized_read_only_semantic_audit":
        raise ValueError("timing/action-frame audit is not authorized")
    if contract.get("paper_faithful") is not False:
        raise ValueError("local semantic audit cannot be paper-faithful")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "optimizer_steps": 0,
        "actor_frozen": True,
        "normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "formal_residual_scale_changed": False,
        "formal_residual_bound_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("runtime contract is not fail-closed")
    for row in contract["inputs"].values():
        artifact = Path(row["path"])
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
    if contract["reference_forward_counterfactual"] != {
        "source_endpoints": list(AUDIT_ENDPOINTS),
        "coherent_reference_context_shifts": [-1, 0, 1],
        "fixed_physical_observation": "ppo_path_current_state",
        "fixed_hidden": "ppo_path_pre_forward_hidden",
        "shifted_fields": [
            "goal_object_anchors", "reference_ctrl", "reference_ctrl_preview"
        ],
        "execute_action": False,
        "candidate_metric": "sum_positive_right_wrist_y_mu_excess_above_state_high",
        "minimum_relative_improvement_for_closed_loop": 0.05,
        "tie_order": [-1, 1],
    }:
        raise ValueError("reference counterfactual differs from frozen definition")
    if contract["hidden_isolation"]["hidden_variants"] != [
        "ppo_path_pre_forward",
        "replay_path_pre_forward",
        "zero_hidden",
        "previous_source_ppo_pre_forward",
    ]:
        raise ValueError("hidden variants differ from frozen definition")
    return contract, {
        "path": str(contract_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64).copy()


def _zero_hidden(policy):
    return [
        state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()
    ]


def _run_ppo_capture(*, env, backend, policy, boundary, expected_trace: dict):
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = _zero_hidden(policy)
    backend.begin_trial("rl", 20, 60)
    records = {}
    validated = 0
    for source in range(20, 60):
        packed = policy.obs_to_tensors(backend.observation())
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, packed["obs"], hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        if source in CAPTURE_ENDPOINTS:
            records[source] = {
                "raw_observation": forward["raw_observation"],
                "pre_forward_hidden": hidden,
                "actor_mu": forward["actor_mu"],
                "actor_logstd": forward["actor_logstd"],
                "state_low": bounded["state_low"],
                "state_high": bounded["state_high"],
                "deterministic_action": bounded["deterministic_action"],
                "qpos": _numpy(env._mjwp.get_qpos(env.ego_cfg, env.env))[0],
            }
        feasible = backend.step(
            bounded["deterministic_action"][None].astype(np.float32), source
        )
        if not feasible:
            break
        validated += 1
    backend.end_trial(validated == 40, validated)
    trace = backend.validation_traces[-1]
    _require_exact_trace(trace, expected_trace, "formal PPO")
    if set(records) != set(CAPTURE_ENDPOINTS):
        raise RuntimeError("formal PPO did not capture all semantic-audit endpoints")
    return trace, records


def _replay_actor_records(policy, replay_capture: dict) -> dict:
    policy.rnn_states = _zero_hidden(policy)
    records = {}
    for source in range(20, max(CAPTURE_ENDPOINTS) + 1):
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(
            policy, replay_capture["observations"][source], hidden
        )
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        if source in CAPTURE_ENDPOINTS:
            records[source] = {
                "pre_forward_hidden": hidden,
                "actor_mu": forward["actor_mu"],
                "actor_logstd": forward["actor_logstd"],
            }
    return records


def _reference_context(env, backend, source: int) -> np.ndarray:
    saved = env.time_indices.copy()
    try:
        env.time_indices[:] = source
        packed = backend.observation()
        raw = packed["obs"] if isinstance(packed, dict) else packed
        return np.asarray(raw[0, REFERENCE_SLICE], dtype=np.float64).copy()
    finally:
        env.time_indices[:] = saved


def _timing_ledger(observation) -> dict:
    rows = {}
    for source in AUDIT_ENDPOINTS:
        rows[str(source)] = {
            "simulator_state_endpoint_before_action": source,
            "hand_object_and_contact_observation_endpoint": source,
            "goal_object_anchors_reference_endpoint": source + observation.goal_reference_offset,
            "actor_reference_ctrl_endpoint": source + observation.command_offset,
            "actor_reference_ctrl_preview_endpoint": source + observation.command_preview_offset,
            "base_command_reference_endpoint": source + 1,
            "residual_corrects_command_endpoint": source + 1,
            "action_interval": [source, source + 1],
            "simulator_state_endpoint_after_action": source + 1,
            "reward_reference_endpoint": source + 1,
            "returned_observation_goal_reference_endpoint": source + 2,
        }
    return rows


def _timing_forward(policy, records: dict, contexts: dict[int, np.ndarray]):
    arrays = {}
    rows = {}
    metrics = {}
    for shift in (-1, 0, 1):
        excess = 0.0
        shift_rows = {}
        mus = []
        for source in AUDIT_ENDPOINTS:
            raw = records[source]["raw_observation"].copy()
            raw[REFERENCE_SLICE] = contexts[source + shift]
            forward = _forward_raw(
                policy, raw[None].astype(np.float32),
                records[source]["pre_forward_hidden"],
            )
            mu_y = float(forward["actor_mu"][RIGHT_WRIST_Y])
            high_y = float(records[source]["state_high"][RIGHT_WRIST_Y])
            excess += max(mu_y - high_y, 0.0)
            shift_rows[str(source)] = {
                "actor_mu_y": mu_y,
                "state_high_y": high_y,
                "positive_excess": max(mu_y - high_y, 0.0),
            }
            mus.append(forward["actor_mu"])
        rows[str(shift)] = shift_rows
        metrics[str(shift)] = excess
        arrays[f"timing_shift_{shift}_actor_mu"] = np.stack(mus)
    base = metrics["0"]
    ordered = sorted((-1, 1), key=lambda shift: (metrics[str(shift)], (-1, 1).index(shift)))
    best = ordered[0]
    improvement = (base - metrics[str(best)]) / base if base > 0 else 0.0
    selected = best if improvement >= 0.05 else None
    return {
        "rows": rows,
        "sum_positive_mu_excess": metrics,
        "selected_nonzero_shift": selected,
        "selected_relative_improvement": improvement,
        "selection_threshold": 0.05,
    }, arrays


def _hidden_isolation(policy, ppo: dict, replay: dict):
    rows = {}
    arrays = {}
    zero = _zero_hidden(policy)
    for source in AUDIT_ENDPOINTS:
        raw = ppo[source]["raw_observation"]
        variants = {
            "ppo_path_pre_forward": ppo[source]["pre_forward_hidden"],
            "replay_path_pre_forward": replay[source]["pre_forward_hidden"],
            "zero_hidden": zero,
            "previous_source_ppo_pre_forward": ppo[source - 1][
                "pre_forward_hidden"
            ],
        }
        source_rows = {}
        for name, hidden in variants.items():
            forward = _forward_raw(
                policy, raw[None].astype(np.float32), hidden
            )
            mu_y = float(forward["actor_mu"][RIGHT_WRIST_Y])
            source_rows[name] = {
                "actor_mu_y": mu_y,
                "inside_state_support": bool(
                    ppo[source]["state_low"][RIGHT_WRIST_Y]
                    <= mu_y
                    <= ppo[source]["state_high"][RIGHT_WRIST_Y]
                ),
            }
            arrays[f"hidden_source_{source}_{name}_actor_mu"] = forward["actor_mu"]
        rows[str(source)] = source_rows
    return rows, arrays


def _site_pose(model, data, qpos: np.ndarray, site_id: int):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return (
        np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _angle_and_projection(direction: np.ndarray, desired: np.ndarray) -> dict:
    projection = float(np.dot(direction, desired))
    return {
        "direction_world": direction.tolist(),
        "signed_projection": projection,
        "angle_deg": float(np.degrees(np.arccos(np.clip(projection, -1.0, 1.0)))),
    }


def _transform_translations(
    action: np.ndarray,
    *,
    frame: str,
    current_rotations: tuple[np.ndarray, np.ndarray],
    reference_rotations: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    result = action.copy()
    rotations = current_rotations if frame == "current_palm_site" else reference_rotations
    for offset, rotation in zip((0, 18), rotations, strict=True):
        result[offset : offset + 3] = rotation @ action[offset : offset + 3]
    return result


def _base_translation_axis_gate(model, data, qpos, right_palm_site: int) -> dict:
    actuator_ids = [model.actuator(name).id for name in (
        "R_forearm_tx_position", "R_forearm_ty_position", "R_forearm_tz_position"
    )]
    qpos_indices = [
        int(model.jnt_qposadr[int(model.actuator_trnid[index, 0])])
        for index in actuator_ids
    ]
    epsilon = 1e-6
    columns = []
    for index in qpos_indices:
        plus = qpos.copy()
        minus = qpos.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        p_plus, _ = _site_pose(model, data, plus, right_palm_site)
        p_minus, _ = _site_pose(model, data, minus, right_palm_site)
        columns.append((p_plus - p_minus) / (2.0 * epsilon))
    jacobian = np.column_stack(columns)
    return {
        "actuator_names": [
            "R_forearm_tx_position", "R_forearm_ty_position", "R_forearm_tz_position"
        ],
        "palm_translation_jacobian_world": jacobian.tolist(),
        "maximum_absolute_difference_from_identity": float(
            np.abs(jacobian - np.eye(3)).max()
        ),
        "robot_base_translation_frame_equals_world": bool(
            np.allclose(jacobian, np.eye(3), rtol=0.0, atol=1e-8)
        ),
    }


def _frame_geometry(env, ppo: dict, reference_qpos: np.ndarray, contract: dict):
    model = env.env.model_cpu
    data = mujoco.MjData(model)
    palm_ids = (model.site("right_palm").id, model.site("left_palm").id)
    rows = {}
    arrays = {}
    transformed = {"current_palm_site": {}, "reference_palm_site": {}}
    for source in AUDIT_ENDPOINTS:
        actual = ppo[source]["qpos"]
        goal = reference_qpos[source + 1]
        current_poses = tuple(_site_pose(model, data, actual, site) for site in palm_ids)
        reference_poses = tuple(_site_pose(model, data, goal, site) for site in palm_ids)
        desired = goal[36:39] - actual[36:39]
        desired_norm = float(np.linalg.norm(desired))
        if desired_norm <= 0.0:
            raise RuntimeError("tool correction direction is undefined")
        desired_unit = desired / desired_norm
        directions = {
            "world_or_robot_base": np.array([0.0, 1.0, 0.0]),
            "current_palm_site": current_poses[0][1][:, 1],
            "reference_palm_site": reference_poses[0][1][:, 1],
        }
        rows[str(source)] = {
            "desired_tool_correction_world_m": desired.tolist(),
            "desired_tool_correction_norm_m": desired_norm,
            "positive_y_basis": {
                name: _angle_and_projection(direction, desired_unit)
                for name, direction in directions.items()
            },
        }
        action = ppo[source]["deterministic_action"]
        for frame in transformed:
            converted = _transform_translations(
                action,
                frame=frame,
                current_rotations=(current_poses[0][1], current_poses[1][1]),
                reference_rotations=(reference_poses[0][1], reference_poses[1][1]),
            )
            low = ppo[source]["state_low"]
            high = ppo[source]["state_high"]
            translation_indices = np.r_[0:3, 18:21]
            within = bool(np.all(
                (converted[translation_indices] >= low[translation_indices] - 1e-7)
                & (converted[translation_indices] <= high[translation_indices] + 1e-7)
            ))
            transformed[frame][str(source)] = {
                "right_translation_before": action[:3].tolist(),
                "right_translation_after": converted[:3].tolist(),
                "left_translation_before": action[18:21].tolist(),
                "left_translation_after": converted[18:21].tolist(),
                "maximum_absolute_transformed_translation": float(
                    np.abs(converted[translation_indices]).max()
                ),
                "inside_existing_normalized_support": within,
            }
            arrays[f"frame_source_{source}_{frame}_action"] = converted

    selection_sources = tuple(contract["frame_geometry"]["geometry_selection_sources"])
    projection = {
        frame: float(np.mean([
            rows[str(source)]["positive_y_basis"][frame]["signed_projection"]
            for source in selection_sources
        ]))
        for frame in ("world_or_robot_base", "current_palm_site", "reference_palm_site")
    }
    local_frame = max(
        ("current_palm_site", "reference_palm_site"), key=projection.get
    )
    projection_improvement = projection[local_frame] - projection["world_or_robot_base"]
    all_inside = all(
        row["inside_existing_normalized_support"]
        for row in transformed[local_frame].values()
    )
    selected = (
        local_frame
        if projection_improvement >= 0.1 and all_inside
        else None
    )
    axis_gate = _base_translation_axis_gate(
        model, data, ppo[40]["qpos"], palm_ids[0]
    )
    return {
        "rows": rows,
        "mean_signed_projection_sources_43_to_46": projection,
        "best_local_frame": local_frame,
        "best_projection_improvement_over_world": projection_improvement,
        "transformed_action_support": transformed,
        "all_best_frame_transformed_actions_inside_existing_support": all_inside,
        "selected_frame_for_closed_loop": selected,
        "base_translation_axis_gate": axis_gate,
    }, arrays


def _run_diagnostic_candidate(
    *, env, backend, policy, boundary, contexts, reference_qpos,
    timing_shift: int | None = None, frame: str | None = None,
):
    if (timing_shift is None) == (frame is None):
        raise ValueError("diagnostic candidate must change exactly one semantic")
    model = env.env.model_cpu
    data = mujoco.MjData(model)
    palm_ids = (model.site("right_palm").id, model.site("left_palm").id)
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = _zero_hidden(policy)
    name = (
        f"diagnostic_actor_reference_shift_{timing_shift:+d}"
        if timing_shift is not None
        else f"diagnostic_translation_frame_{frame}"
    )
    backend.begin_trial(name, 20, 60)
    rows = []
    validated = 0
    invalid_support = None
    for source in range(20, 60):
        packed = policy.obs_to_tensors(backend.observation())
        raw = np.asarray(packed["obs"].detach().cpu().numpy(), dtype=np.float32)
        if timing_shift is not None:
            raw = raw.copy()
            raw[0, REFERENCE_SLICE] = contexts[source + timing_shift]
        hidden = _clone_rnn(policy.rnn_states)
        forward = _forward_raw(policy, raw, hidden)
        bounded = _bounds_and_mode(env, forward)
        policy.rnn_states = _clone_rnn(forward["next_hidden"])
        action = bounded["deterministic_action"].copy()
        if frame is not None:
            actual = _numpy(env._mjwp.get_qpos(env.ego_cfg, env.env))[0]
            goal = reference_qpos[source + 1]
            current_rotations = tuple(
                _site_pose(model, data, actual, site)[1] for site in palm_ids
            )
            reference_rotations = tuple(
                _site_pose(model, data, goal, site)[1] for site in palm_ids
            )
            action = _transform_translations(
                action,
                frame=frame,
                current_rotations=current_rotations,
                reference_rotations=reference_rotations,
            )
            translation_indices = np.r_[0:3, 18:21]
            low = bounded["state_low"][translation_indices]
            high = bounded["state_high"][translation_indices]
            values = action[translation_indices]
            if np.any((values < low - 1e-7) | (values > high + 1e-7)):
                invalid_support = source
                break
        feasible = backend.step(action[None].astype(np.float32), source)
        rows.append({
            "source_endpoint": source,
            "outcome_endpoint": int(env.time_indices[0]),
            "actor_mu_y": float(forward["actor_mu"][RIGHT_WRIST_Y]),
            "deterministic_action_y": float(bounded["deterministic_action"][RIGHT_WRIST_Y]),
            "executed_world_action_y": float(action[RIGHT_WRIST_Y]),
            "tracking_score": float(backend.last_info["object_tracking_error"][0]),
            "position_error_m": float(backend.last_info["object_position_error"][0, 0]),
            "rotation_error_rad": float(backend.last_info["object_rotation_error"][0, 0]),
            "feasible": bool(feasible),
        })
        if not feasible:
            break
        validated += 1
    completed = invalid_support is None
    backend.end_trial(completed and validated == 40, validated)
    return {
        "candidate": name,
        "timing_shift": timing_shift,
        "translation_frame": frame,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "commit_allowed": False,
        "validated_intervals": validated,
        "first_failure": backend.validation_traces[-1]["first_failure"],
        "aborted_before_step_for_support_violation_at_source": invalid_support,
        "rows": rows,
    }


def _write_summary(path: Path, report: dict):
    lines = [
        "# Pour reference timing × recurrent context × action frame audit v1",
        "",
        "Read-only local semantic audit; no optimizer step or chunk commit occurred.",
        "",
        "## Timing forward counterfactual",
        "",
    ]
    timing = report["reference_forward_counterfactual"]
    for shift in (-1, 0, 1):
        lines.append(
            f"- shift {shift:+d}: summed positive μ_y excess = "
            f"{timing['sum_positive_mu_excess'][str(shift)]:.6f}."
        )
    lines.extend(["", "## Hidden isolation", ""])
    for source, rows in report["hidden_isolation"].items():
        values = ", ".join(
            f"{name}={row['actor_mu_y']:.4f}" for name, row in rows.items()
        )
        lines.append(f"- source {source}: {values}.")
    frame = report["frame_geometry"]
    lines.extend([
        "",
        "## Frame geometry",
        "",
        f"- Mean +y projection (43–46): {frame['mean_signed_projection_sources_43_to_46']}.",
        f"- Best local frame: `{frame['best_local_frame']}`; support-safe: {frame['all_best_frame_transformed_actions_inside_existing_support']}.",
        "",
        "## Diagnostic closed loop",
        "",
    ])
    if report["closed_loop_counterfactuals"]:
        for row in report["closed_loop_counterfactuals"]:
            lines.append(
                f"- `{row['candidate']}`: {row['validated_intervals']}/40, "
                f"first failure={row['first_failure']}."
            )
    else:
        lines.append("- No candidate satisfied its predeclared eligibility gate.")
    lines.extend([
        "",
        "## Limits",
        "",
        "- Reference shifts modify actor input only; base command and reward keep the corrected t→t+1 contract.",
        "- Frame projections are geometric diagnostics, not contact-dynamics gradients.",
        "- A frame candidate is not run if transforming the frozen action would leave the existing normalized support.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_reference_timing_action_frame_audit_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_reference_timing_action_frame_audit_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
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
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, _model_state_sha256
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    contract, contract_artifact = _load_contract(args.contract)
    paths = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    formal = json.loads(paths["formal_report"].read_text())
    traces = {row["mode"]: row for row in formal["chunks"][0]["validation_traces"]}
    objective = load_runtime_objective(
        paths["replay_rl_protocol_snapshot"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        paths["replay_rl_protocol_snapshot"], paths["observation_profile"],
        require_run_ready=True,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    spec, distribution = load_truncated_gaussian_profile(paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
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
    backend = MJWPChunkBackend(env)
    boundary, _, _ = _load_gzip_torch(paths["endpoint_20_boundary"], torch)
    checkpoint, _, checkpoint_raw = _load_gzip_torch(paths["checkpoint_artifact"], torch)
    actor_hash = formal["chunks"][0]["training_runs"][0]["actor_state_sha256"]
    if _model_state_sha256(checkpoint["model"]) != actor_hash:
        raise ValueError("checkpoint actor differs from formal corrected run")

    replay_trace, replay_capture = _run_replay_capture(
        backend=backend, boundary=boundary, expected_trace=traces["replay"]
    )
    with tempfile.TemporaryDirectory(
        prefix=".timing_action_frame_", dir=ROOT / "runs"
    ) as temporary:
        temp = Path(temporary)
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=temp / "policy_runtime",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        model_hash_before = _model_state_sha256(policy.model.state_dict())
        normalization_before = _normalization_state(policy)

        ppo_trace, ppo_records = _run_ppo_capture(
            env=env, backend=backend, policy=policy, boundary=boundary,
            expected_trace=traces["rl"],
        )
        replay_records = _replay_actor_records(policy, replay_capture)
        contexts = {
            source: _reference_context(env, backend, source)
            for source in range(19, 61)
        }
        timing, timing_arrays = _timing_forward(policy, ppo_records, contexts)
        hidden, hidden_arrays = _hidden_isolation(
            policy, ppo_records, replay_records
        )
        reference_qpos = reference[0].cpu().numpy().astype(np.float64)
        frame, frame_arrays = _frame_geometry(
            env, ppo_records, reference_qpos, contract
        )

        candidates = []
        if timing["selected_nonzero_shift"] is not None:
            candidates.append(_run_diagnostic_candidate(
                env=env, backend=backend, policy=policy, boundary=boundary,
                contexts=contexts, reference_qpos=reference_qpos,
                timing_shift=int(timing["selected_nonzero_shift"]),
            ))
        if frame["selected_frame_for_closed_loop"] is not None:
            candidates.append(_run_diagnostic_candidate(
                env=env, backend=backend, policy=policy, boundary=boundary,
                contexts=contexts, reference_qpos=reference_qpos,
                frame=str(frame["selected_frame_for_closed_loop"]),
            ))
        if len(candidates) > contract["closed_loop_counterfactual"]["maximum_candidates"]:
            raise RuntimeError("candidate count exceeds frozen contract")

        normalization_after = _normalization_state(policy)
        model_hash_after = _model_state_sha256(policy.model.state_dict())
        normalization_unchanged = all(
            np.array_equal(normalization_before[name], normalization_after[name])
            if isinstance(normalization_before[name], np.ndarray)
            else normalization_before[name] == normalization_after[name]
            for name in normalization_before
        )
        if model_hash_before != model_hash_after or not normalization_unchanged:
            raise RuntimeError("semantic audit changed actor/normalization state")

        arrays = {
            "source_endpoints": np.asarray(AUDIT_ENDPOINTS, dtype=np.int64),
            "ppo_raw_observation": np.stack([
                ppo_records[source]["raw_observation"] for source in AUDIT_ENDPOINTS
            ]),
            "ppo_actor_mu": np.stack([
                ppo_records[source]["actor_mu"] for source in AUDIT_ENDPOINTS
            ]),
            "ppo_deterministic_action": np.stack([
                ppo_records[source]["deterministic_action"] for source in AUDIT_ENDPOINTS
            ]),
            **timing_arrays,
            **hidden_arrays,
            **frame_arrays,
        }
        npz_path = temp / contract["artifacts"]["full_array_npz"]
        np.savez_compressed(npz_path, **arrays)
        report = {
            "schema": contract["schema"],
            "status": "completed_read_only_semantic_audit",
            "paper_faithful": False,
            "scope": {
                "training_performed": False,
                "optimizer_steps": 0,
                "actor_frozen": True,
                "normalization_frozen": True,
                "reward_objective_scale_and_bound_unchanged": True,
                "chunk_acceptance_allowed": False,
                "chunk_commit_allowed": False,
            },
            "contract": contract_artifact,
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "actor": {
                "state_sha256": actor_hash,
                "checkpoint_uncompressed_sha256": hashlib.sha256(
                    checkpoint_raw
                ).hexdigest(),
                "distribution": distribution,
            },
            "regression_gate": {
                "formal_replay_trace_exactly_reproduced": True,
                "formal_ppo_trace_exactly_reproduced": True,
                "replay_validated_intervals": replay_trace["validated_steps"],
                "ppo_validated_intervals": ppo_trace["validated_steps"],
                "actor_state_unchanged": model_hash_before == model_hash_after,
                "normalization_state_unchanged": normalization_unchanged,
            },
            "timing_ledger": _timing_ledger(observation),
            "reference_forward_counterfactual": timing,
            "hidden_isolation": hidden,
            "frame_geometry": frame,
            "closed_loop_counterfactuals": candidates,
            "full_arrays": {
                "path": str((args.output / npz_path.name).resolve()),
                "sha256": _sha256(npz_path),
                "keys": sorted(arrays),
            },
            "findings": {
                "reference_timing_candidate_selected": timing[
                    "selected_nonzero_shift"
                ],
                "frame_candidate_selected": frame[
                    "selected_frame_for_closed_loop"
                ],
                "zero_hidden_returns_mu_inside_support_sources": [
                    int(source) for source, rows in hidden.items()
                    if rows["zero_hidden"]["inside_state_support"]
                ],
                "replay_hidden_returns_mu_inside_support_sources": [
                    int(source) for source, rows in hidden.items()
                    if rows["replay_path_pre_forward"]["inside_state_support"]
                ],
                "base_translation_frame_is_world": frame[
                    "base_translation_axis_gate"
                ]["robot_base_translation_frame_equals_world"],
                "diagnostic_candidates_run": [row["candidate"] for row in candidates],
                "formal_success_or_promotion_claimed": False,
            },
            "interpretation_limits": [
                "Reference shifts alter actor reference context only; base command and reward retain corrected t-to-t+1 alignment.",
                "Hidden substitutions keep the current 236-D observation fixed but create diagnostic recurrent states.",
                "Frame projections are geometric comparisons and do not model contact transmission.",
                "A local-frame closed loop is rejected rather than clipped or rescaled if transformed actions leave existing support.",
                "All closed-loop candidates are diagnostic and cannot accept or commit a chunk.",
            ],
        }
        (temp / "contract_snapshot.yaml").write_bytes(args.contract.read_bytes())
        (temp / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        _write_summary(temp / "summary.md", report)
        policy.writer.close()
        policy_dir = temp / "policy_runtime"
        if policy_dir.exists():
            shutil.rmtree(policy_dir)
        temp.rename(args.output)

    print(json.dumps({
        "status": report["status"],
        "regression_gate": report["regression_gate"],
        "timing_metrics": timing["sum_positive_mu_excess"],
        "timing_selected": timing["selected_nonzero_shift"],
        "hidden_mu_y": hidden,
        "frame_projection": frame["mean_signed_projection_sources_43_to_46"],
        "frame_selected": frame["selected_frame_for_closed_loop"],
        "closed_loop": [
            {"candidate": row["candidate"], "validated": row["validated_intervals"]}
            for row in candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
