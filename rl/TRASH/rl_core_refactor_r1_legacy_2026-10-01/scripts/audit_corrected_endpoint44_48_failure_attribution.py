#!/usr/bin/env python3
"""Read-only attribution of the corrected PPO failure at endpoints 44--48.

The audit first reproduces the saved CPU Replay and PPO validation traces from
the exact endpoint-20 boundary and frozen actor.  It emits no result unless
both traces match the formal report exactly.  No optimizer step is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

ENDPOINTS = tuple(range(44, 49))
HAND_NAMES = ("right", "left")
OBJECT_NAMES = ("tool", "target")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ACTION_GROUPS = {
    "right_wrist_translation": slice(0, 3),
    "right_wrist_rotation": slice(3, 6),
    "right_fingers": slice(6, 18),
    "left_wrist_translation": slice(18, 21),
    "left_wrist_rotation": slice(21, 24),
    "left_fingers": slice(24, 36),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _group_rows(values: np.ndarray, names: tuple[str, ...]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for group, selection in ACTION_GROUPS.items():
        row = values[selection]
        result[group] = {
            "actuator_names": list(names[selection]),
            "values": row.tolist(),
            "mean_abs": float(np.abs(row).mean()),
            "max_abs": float(np.abs(row).max()),
        }
    return result


def _active_contacts(flags: np.ndarray) -> list[str]:
    flags = np.asarray(flags, dtype=bool)
    return [
        f"{HAND_NAMES[hand]}-{OBJECT_NAMES[obj]}-{FINGER_NAMES[finger]}"
        for hand in range(2)
        for obj in range(2)
        for finger in range(5)
        if flags[hand, obj, finger]
    ]


def _load_gzip_torch(path: Path, torch):
    artifact = path.read_bytes()
    raw = gzip.decompress(artifact)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False), artifact, raw


def _run_trial(backend, *, mode: str, policy=None) -> tuple[dict, dict[int, dict]]:
    backend.begin_trial(mode, 20, 60)
    extra = {}
    validated = 0
    for reference_step in range(20, 60):
        before = None
        if policy is None:
            action = np.zeros((1, 36), dtype=np.float32)
        else:
            observation = policy.obs_to_tensors(backend.observation())
            result = policy.get_deterministic_action_values(observation)
            policy.rnn_states = result["rnn_states"]
            action = policy.preprocess_actions(result["deterministic_actions"])
            before = {
                "actor_mu": result["mus"][0].detach().cpu().numpy(),
                "actor_sigma": result["sigmas"][0].detach().cpu().numpy(),
                "action_low": result["action_lows"][0].detach().cpu().numpy(),
                "action_high": result["action_highs"][0].detach().cpu().numpy(),
                "deterministic_normalized_action": (
                    result["deterministic_actions"][0].detach().cpu().numpy()
                ),
            }
        feasible = backend.step(action, reference_step)
        endpoint = int(backend.env.time_indices[0])
        extra[endpoint] = {
            "contact_flags": np.asarray(
                backend.last_info["contact_flags"][0], dtype=bool
            ),
            "active_contacts": _active_contacts(
                backend.last_info["contact_flags"][0]
            ),
        }
        if before is not None:
            extra[endpoint].update(before)
        if not feasible:
            break
        validated += 1
    feasible = validated == 40
    backend.end_trial(feasible, validated)
    return backend.validation_traces[-1], extra


def _require_exact_trace(actual: dict, expected: dict, label: str) -> None:
    if actual != expected:
        actual_steps = actual.get("steps", [])
        expected_steps = expected.get("steps", [])
        mismatch = []
        if len(actual_steps) != len(expected_steps):
            mismatch.append(
                f"step_count {len(actual_steps)} != {len(expected_steps)}"
            )
        for index, (left, right) in enumerate(
            zip(actual_steps, expected_steps, strict=False)
        ):
            fields = [name for name in left if left.get(name) != right.get(name)]
            if fields:
                mismatch.append(f"step {index}: {fields[:8]}")
                break
        raise RuntimeError(
            f"{label} does not exactly reproduce the formal CPU trace: "
            + "; ".join(mismatch[:4])
        )


def _actuator_metadata(model, control_indices: tuple[int, ...]):
    names = []
    qpos_indices = []
    units = []
    for actuator in control_indices:
        names.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            or f"actuator_{actuator}"
        )
        joint = int(model.actuator_trnid[actuator, 0])
        qpos_indices.append(int(model.jnt_qposadr[joint]))
        joint_type = int(model.jnt_type[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            units.append("m")
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            units.append("rad")
        else:
            raise ValueError("all controlled XHand joints must be scalar slide/hinge")
    if len(names) != 36 or len(set(qpos_indices)) != 36:
        raise ValueError("expected 36 distinct scalar XHand actuators")
    return tuple(names), np.asarray(qpos_indices, dtype=np.int64), tuple(units)


def _final_endpoint_rows(
    traces: dict[str, dict],
    extras: dict[str, dict[int, dict]],
    reference_qpos: np.ndarray,
    reference_ctrl: np.ndarray,
    actuator_names: tuple[str, ...],
    actuator_qpos_indices: np.ndarray,
    actuator_units: tuple[str, ...],
) -> dict:
    rows = {}
    for mode, trace in traces.items():
        steps = {int(step["endpoint"]): step for step in trace["steps"]}
        mode_rows = {}
        for endpoint in ENDPOINTS:
            step = steps[endpoint]
            qpos = np.asarray(step["endpoint_qpos"], dtype=np.float64)
            actual_tool = qpos[36:39]
            goal_tool = reference_qpos[endpoint, 36:39]
            vector = actual_tool - goal_tool
            position = float(step["position_error_m"][0])
            rotation = float(step["rotation_error_rad"][0])
            position_term = (position / 0.12) ** 2
            rotation_term = (rotation / 1.5) ** 2
            score = float(step["objective_score"][0])
            if not np.isclose(np.linalg.norm(vector), position, rtol=0.0, atol=2e-7):
                raise RuntimeError("tool position vector does not reproduce reported norm")
            if not np.isclose(
                np.sqrt(position_term + rotation_term), score,
                rtol=0.0, atol=2e-6,
            ):
                raise RuntimeError("ellipse terms do not reproduce objective score")

            current_reference = reference_ctrl[endpoint]
            next_reference = reference_ctrl[min(endpoint + 1, len(reference_ctrl) - 1)]
            commanded = np.asarray(step["commanded_ctrl"], dtype=np.float64)
            hand_qpos = qpos[actuator_qpos_indices]
            requested = np.asarray(step["requested_residual"], dtype=np.float64)
            effective = np.asarray(
                step["effective_residual_after_ctrlrange"], dtype=np.float64
            )
            lost = np.asarray(step["residual_lost_to_ctrlrange"], dtype=np.float64)
            if not np.array_equal(requested, effective + lost):
                raise RuntimeError("validation residual decomposition identity failed")

            row = {
                "position_error_m": position,
                "rotation_error_rad": rotation,
                "objective_score": score,
                "ellipse_squared_contributions": {
                    "position": position_term,
                    "rotation": rotation_term,
                    "position_fraction": position_term / (position_term + rotation_term),
                },
                "tool_position_error_vector_actual_minus_reference_m": vector.tolist(),
                "reward": {
                    "tracking": float(step["aggregate_tracking_reward"]),
                    "contact_bonus": float(step["aggregate_contact_bonus"]),
                    "lift": float(step["lift_reward"]),
                    "total": float(step["total_reward"]),
                },
                "contact": {
                    "flags_hand_object_finger": extras[mode][endpoint][
                        "contact_flags"
                    ].tolist(),
                    "active": extras[mode][endpoint]["active_contacts"],
                    "bonus_per_hand_object": step["contact_bonus_per_hand_object"],
                },
                "control": {
                    "actuator_names": list(actuator_names),
                    "actuator_units": list(actuator_units),
                    "commanded_ctrl": commanded.tolist(),
                    "requested_residual": requested.tolist(),
                    "effective_residual_after_ctrlrange": effective.tolist(),
                    "residual_lost_to_ctrlrange": lost.tolist(),
                    "lost_nonzero_count": int(np.count_nonzero(lost)),
                    "lost_above_1e_8_count": int(
                        np.count_nonzero(np.abs(lost) > 1e-8)
                    ),
                    "lost_max_abs": float(np.abs(lost).max()),
                    "hand_qpos_minus_same_endpoint_reference_ctrl": (
                        hand_qpos - current_reference
                    ).tolist(),
                    "hand_qpos_minus_next_endpoint_reference_ctrl": (
                        hand_qpos - next_reference
                    ).tolist(),
                    "grouped_requested_residual": _group_rows(
                        requested, actuator_names
                    ),
                    "grouped_qpos_to_next_reference_gap": _group_rows(
                        hand_qpos - next_reference, actuator_names
                    ),
                },
                "terminated": bool(step["terminated"]),
            }
            if mode == "rl":
                diagnostic = extras[mode][endpoint]
                mu = diagnostic["actor_mu"]
                sigma = diagnostic["actor_sigma"]
                low = diagnostic["action_low"]
                high = diagnostic["action_high"]
                action = diagnostic["deterministic_normalized_action"]
                lower_margin = action - low
                upper_margin = high - action
                margin = np.minimum(lower_margin, upper_margin)
                at_bound = (lower_margin <= 2e-6) | (upper_margin <= 2e-6)
                row["policy"] = {
                    "actor_mu": mu.tolist(),
                    "actor_sigma": sigma.tolist(),
                    "state_feasible_low": low.tolist(),
                    "state_feasible_high": high.tolist(),
                    "deterministic_normalized_action": action.tolist(),
                    "mu_outside_feasible_count": int(
                        np.count_nonzero((mu < low) | (mu > high))
                    ),
                    "action_at_feasible_bound_count": int(at_bound.sum()),
                    "minimum_action_to_bound_margin": float(margin.min()),
                    "grouped_normalized_action": _group_rows(
                        action, actuator_names
                    ),
                    "grouped_actor_mu": _group_rows(mu, actuator_names),
                    "grouped_action_bound_margin": _group_rows(
                        margin, actuator_names
                    ),
                }
            mode_rows[str(endpoint)] = row

        for endpoint in ENDPOINTS:
            key = str(endpoint)
            previous = steps[endpoint - 1]
            current = mode_rows[key]
            previous_vector = (
                np.asarray(previous["endpoint_qpos"], dtype=np.float64)[36:39]
                - reference_qpos[endpoint - 1, 36:39]
            )
            current["change_from_previous_endpoint"] = {
                "position_error_m": (
                    current["position_error_m"]
                    - float(previous["position_error_m"][0])
                ),
                "rotation_error_rad": (
                    current["rotation_error_rad"]
                    - float(previous["rotation_error_rad"][0])
                ),
                "objective_score": (
                    current["objective_score"]
                    - float(previous["objective_score"][0])
                ),
                "tool_position_error_vector_m": (
                    np.asarray(
                        current[
                            "tool_position_error_vector_actual_minus_reference_m"
                        ]
                    )
                    - previous_vector
                ).tolist(),
            }
        rows[mode] = mode_rows

    comparison = {}
    for endpoint in ENDPOINTS:
        key = str(endpoint)
        replay = rows["replay"][key]
        rl = rows["rl"][key]
        comparison[key] = {
            "ppo_minus_replay_position_error_m": (
                rl["position_error_m"] - replay["position_error_m"]
            ),
            "ppo_minus_replay_rotation_error_rad": (
                rl["rotation_error_rad"] - replay["rotation_error_rad"]
            ),
            "ppo_minus_replay_objective_score": (
                rl["objective_score"] - replay["objective_score"]
            ),
            "ppo_minus_replay_tool_position_error_vector_m": (
                np.asarray(
                    rl["tool_position_error_vector_actual_minus_reference_m"]
                )
                - np.asarray(
                    replay["tool_position_error_vector_actual_minus_reference_m"]
                )
            ).tolist(),
        }
    return {"per_mode": rows, "ppo_minus_replay": comparison}


def _training_endpoint_rows(
    directory: Path,
    model,
    control_indices: tuple[int, ...],
    spec,
    actuator_names: tuple[str, ...],
    final_rows: dict,
) -> tuple[dict, list[dict]]:
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        normalized_action_bounds_numpy,
    )

    arrays: dict[str, list[np.ndarray]] = {}
    epoch_ids = []
    artifacts = []
    files = sorted(directory.glob("epoch_*_visits.npz"))
    if len(files) != 8:
        raise ValueError("corrected v5 audit requires exactly eight epoch files")
    for epoch, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as source:
            for name in source.files:
                arrays.setdefault(name, []).append(source[name].copy())
            epoch_ids.append(np.full(len(source["outcome_endpoint"]), epoch, np.int32))
        artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    data = {name: np.concatenate(rows, axis=0) for name, rows in arrays.items()}
    data["epoch_id"] = np.concatenate(epoch_ids)
    if len(data["outcome_endpoint"]) != 1280:
        raise ValueError("corrected v5 trace must contain 1280 samples")

    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)[list(control_indices)]
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)[
        list(control_indices)
    ]
    low, high, bounds_audit = normalized_action_bounds_numpy(
        data["reference_ctrl"],
        ctrllimited=limited,
        ctrlrange=ranges,
        residual_scale=spec.residual_scale,
        reference_snap_tolerance=spec.reference_snap_tolerance,
    )
    mode = np.maximum(np.minimum(data["actor_mu"], high), low)
    mu_outside = (data["actor_mu"] < low) | (data["actor_mu"] > high)
    mode_at_bound = np.minimum(mode - low, high - mode) <= 2e-6
    sample_at_bound = np.minimum(
        data["sampled_action_clamped"] - low,
        high - data["sampled_action_clamped"],
    ) <= 2e-6
    if np.any(data["sampled_action_clamped"] < low - 2e-6) or np.any(
        data["sampled_action_clamped"] > high + 2e-6
    ):
        raise RuntimeError("training sample lies outside state-feasible bounds")

    rows = {}
    for endpoint in (47, 48):
        select = data["outcome_endpoint"] == endpoint
        cpu = final_rows["per_mode"]["rl"][str(endpoint)]
        metrics = {
            "position_error_m": _distribution(data["tool_position_error_m"][select]),
            "rotation_error_rad": _distribution(data["tool_rotation_error_rad"][select]),
            "objective_score": _distribution(data["tool_objective_score"][select]),
        }
        cpu_values = {
            "position_error_m": cpu["position_error_m"],
            "rotation_error_rad": cpu["rotation_error_rad"],
            "objective_score": cpu["objective_score"],
        }
        rows[str(endpoint)] = {
            "visit_count": int(select.sum()),
            "epochs_with_visit": sorted(set(map(int, data["epoch_id"][select]))),
            "tracking_termination_count": int(
                data["tracking_terminated"][select].sum()
            ),
            "tracking_termination_rate": float(
                data["tracking_terminated"][select].mean()
            ),
            "metrics": metrics,
            "final_cpu_ppo_metrics": cpu_values,
            "final_cpu_metric_within_training_min_max": {
                name: bool(
                    metrics[name]["min"] <= value <= metrics[name]["max"]
                )
                for name, value in cpu_values.items()
            },
            "actor_mu": _distribution(data["actor_mu"][select]),
            "actor_sigma": _distribution(data["actor_sigma"][select]),
            "mu_outside_feasible_count": int(mu_outside[select].sum()),
            "mu_outside_feasible_sample_count": int(
                mu_outside[select].any(axis=1).sum()
            ),
            "deterministic_mode_at_bound_count": int(mode_at_bound[select].sum()),
            "deterministic_mode_at_bound_sample_count": int(
                mode_at_bound[select].any(axis=1).sum()
            ),
            "sampled_action_at_bound_count": int(sample_at_bound[select].sum()),
            "sampled_action_at_bound_sample_count": int(
                sample_at_bound[select].any(axis=1).sum()
            ),
            "action_groups": {
                group: {
                    "mu": _distribution(data["actor_mu"][select, selection]),
                    "sigma": _distribution(data["actor_sigma"][select, selection]),
                    "low": _distribution(low[select, selection]),
                    "high": _distribution(high[select, selection]),
                    "mu_outside_count": int(mu_outside[select, selection].sum()),
                    "mode_at_bound_count": int(mode_at_bound[select, selection].sum()),
                    "sample_at_bound_count": int(sample_at_bound[select, selection].sum()),
                }
                for group, selection in ACTION_GROUPS.items()
            },
            "contact_flag_true_rates": {
                f"{HAND_NAMES[hand]}-{OBJECT_NAMES[obj]}-{FINGER_NAMES[finger]}": float(
                    data["contact_flags"][select, hand, obj, finger].mean()
                )
                for hand in range(2)
                for obj in range(2)
                for finger in range(5)
            },
            "residual_lost_nonzero_count": int(
                np.count_nonzero(data["residual_lost_to_ctrlrange"][select])
            ),
            "exact_final_cpu_state_coverage_claim": False,
            "coverage_limitation": (
                "v5 stores endpoint, errors, policy distribution, residuals and "
                "coarse contact flags, but not the complete observation, qpos/qvel, "
                "solver/contact buffers, or recurrent state. Matching endpoint or "
                "metric ranges cannot prove that the exact CPU failure state was seen."
            ),
        }
    return {
        "schema": "taco_ppo_training_visitation_v5",
        "sample_count": int(len(data["outcome_endpoint"])),
        "bounds_recomputation": bounds_audit,
        "endpoints": rows,
        "actuator_names": list(actuator_names),
    }, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    run = ROOT / "runs/taco_pour_corrected_first_ppo_v1"
    parser.add_argument("--formal-report", type=Path, default=run / "report.json")
    parser.add_argument(
        "--checkpoint-artifact", type=Path,
        default=run / "checkpoint_artifacts/last_ep_8_rew__6.6354837_.pth.gz",
    )
    parser.add_argument(
        "--boundary", type=Path, default=run / "input_contracts/resume_boundary.pt.gz"
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml",
    )
    parser.add_argument(
        "--protocol", type=Path, default=run / "input_contracts/replay_rl_protocol.yaml"
    )
    parser.add_argument(
        "--objective-profile", type=Path,
        default=run / "input_contracts/objective_profile.yaml",
    )
    parser.add_argument(
        "--observation-profile", type=Path,
        default=run / "input_contracts/observation_profile.yaml",
    )
    parser.add_argument(
        "--action-profile", type=Path,
        default=run / "input_contracts/action_profile.yaml",
    )
    parser.add_argument(
        "--distribution-profile", type=Path,
        default=run / "input_contracts/state_feasible_action_profile.yaml",
    )
    parser.add_argument(
        "--initialization-report", type=Path,
        default=ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json",
    )
    parser.add_argument(
        "--training-visitation", type=Path,
        default=run / "ppo_chunk_20/training_visitation",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/corrected_endpoint44_48_failure_attribution_v1",
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

    formal = json.loads(args.formal_report.read_text())
    if (
        _sha256(args.config)
        != formal["input_contract_snapshots"]["simulator_config"]["sha256"]
    ):
        raise ValueError("simulator config differs from the formal-run snapshot")
    expected_traces = {
        trace["mode"]: trace for trace in formal["chunks"][0]["validation_traces"]
    }
    if set(expected_traces) != {"replay", "rl"}:
        raise ValueError("formal report must contain exactly Replay and RL traces")

    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant="tool_only", require_run_ready=True,
    )
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True,
    )
    residual, residual_artifact = load_residual_action_profile(args.action_profile)
    spec, distribution_artifact = load_truncated_gaussian_profile(
        args.distribution_profile
    )
    if not np.isclose(residual.residual_scale, spec.residual_scale):
        raise ValueError("residual and distribution profiles disagree")
    _, initialization = load_accepted_initialization(
        args.initialization_report, args.config
    )
    config = _load_ego_config(str(args.config), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
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

    boundary, boundary_artifact, boundary_raw = _load_gzip_torch(
        args.boundary, torch
    )
    if (
        boundary.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or np.asarray(boundary["time_indices"]).tolist() != [20]
    ):
        raise ValueError("audit requires the corrected endpoint-20 boundary")

    checkpoint, checkpoint_artifact, checkpoint_raw = _load_gzip_torch(
        args.checkpoint_artifact, torch
    )
    expected_actor_hash = formal["chunks"][0]["training_runs"][0][
        "actor_state_sha256"
    ]
    if _model_state_sha256(checkpoint["model"]) != expected_actor_hash:
        raise ValueError("checkpoint actor does not match the formal report")

    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    replay_trace, replay_extra = _run_trial(backend, mode="replay")
    _require_exact_trace(replay_trace, expected_traces["replay"], "Replay")

    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    with tempfile.TemporaryDirectory(prefix="corrected_endpoint44_48_") as temp:
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
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        policy.rnn_states = [
            state.to("cpu").zero_()
            for state in policy.model.get_default_rnn_state()
        ]
        rl_trace, rl_extra = _run_trial(backend, mode="rl", policy=policy)
        policy.writer.close()
    _require_exact_trace(rl_trace, expected_traces["rl"], "PPO")

    control_indices = tuple(env.env_cfg.residual.hand_control_indices)
    actuator_names, qpos_indices, actuator_units = _actuator_metadata(
        env.env.model_cpu, control_indices
    )
    reference_qpos = reference[0].cpu().numpy().astype(np.float64)
    reference_ctrl = reference[2].cpu().numpy().astype(np.float64)[:, control_indices]
    final_rows = _final_endpoint_rows(
        {"replay": replay_trace, "rl": rl_trace},
        {"replay": replay_extra, "rl": rl_extra},
        reference_qpos,
        reference_ctrl,
        actuator_names,
        qpos_indices,
        actuator_units,
    )
    training, training_artifacts = _training_endpoint_rows(
        args.training_visitation,
        env.env.model_cpu,
        control_indices,
        spec,
        actuator_names,
        final_rows,
    )

    ppo_lost_max = max(
        final_rows["per_mode"]["rl"][str(endpoint)]["control"]["lost_max_abs"]
        for endpoint in ENDPOINTS
    )
    ppo_material_lost = sum(
        final_rows["per_mode"]["rl"][str(endpoint)]["control"][
            "lost_above_1e_8_count"
        ]
        for endpoint in ENDPOINTS
    )
    all_contact_bonus_zero = all(
        final_rows["per_mode"][mode][str(endpoint)]["reward"]["contact_bonus"]
        == 0.0
        for mode in ("replay", "rl")
        for endpoint in ENDPOINTS
    )
    final_rl = final_rows["per_mode"]["rl"]["48"]
    final_comparison = final_rows["ppo_minus_replay"]["48"]
    right_wrist_bound_hits = {}
    for endpoint in ENDPOINTS:
        policy_row = final_rows["per_mode"]["rl"][str(endpoint)]["policy"]
        action = np.asarray(policy_row["deterministic_normalized_action"])
        low = np.asarray(policy_row["state_feasible_low"])
        high = np.asarray(policy_row["state_feasible_high"])
        hit = (action - low <= 2e-6) | (high - action <= 2e-6)
        right_wrist_bound_hits[str(endpoint)] = [
            actuator_names[index] for index in range(6) if hit[index]
        ]
    endpoint_48_training = training["endpoints"]["48"]
    report = {
        "schema": "corrected_endpoint44_48_failure_attribution_v1",
        "status": "passed_read_only_attribution",
        "paper_faithful": False,
        "scope": {
            "endpoints": list(ENDPOINTS),
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint_resume_authorized": False,
            "objective_changed": False,
            "action_scale_changed": False,
            "curriculum_changed": False,
        },
        "inputs": {
            "formal_report": {
                "path": str(args.formal_report.resolve()),
                "sha256": _sha256(args.formal_report),
            },
            "checkpoint": {
                "path": str(args.checkpoint_artifact.resolve()),
                "artifact_sha256": hashlib.sha256(checkpoint_artifact).hexdigest(),
                "uncompressed_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
                "actor_state_sha256": expected_actor_hash,
            },
            "boundary": {
                "path": str(args.boundary.resolve()),
                "artifact_sha256": hashlib.sha256(boundary_artifact).hexdigest(),
                "uncompressed_sha256": hashlib.sha256(boundary_raw).hexdigest(),
                "endpoint": 20,
            },
            "simulator_config": {
                "path": str(args.config.resolve()),
                "sha256": _sha256(args.config),
                "matches_formal_snapshot_sha256": (
                    _sha256(args.config)
                    == formal["input_contract_snapshots"]["simulator_config"]["sha256"]
                ),
            },
            "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)},
            "objective_profile": {
                "path": str(args.objective_profile.resolve()),
                "sha256": _sha256(args.objective_profile),
            },
            "observation_profile": {
                "path": str(args.observation_profile.resolve()),
                "sha256": _sha256(args.observation_profile),
            },
            "action_profile": residual_artifact,
            "distribution_profile": distribution_artifact,
            "training_visitation_artifacts": training_artifacts,
        },
        "regression_gate": {
            "replay_trace_exactly_reproduced": True,
            "ppo_trace_exactly_reproduced": True,
            "replay_validated_intervals": replay_trace["validated_steps"],
            "replay_first_failure": replay_trace["first_failure"],
            "ppo_validated_intervals": rl_trace["validated_steps"],
            "ppo_first_failure": rl_trace["first_failure"],
            "restored_boundary_exactly_verified_count": backend.verified_restore_count,
        },
        "final_cpu_endpoint_attribution": final_rows,
        "training_v5_endpoint_47_48": training,
        "measured_checks": {
            "ppo_ctrlrange_loss_above_1e_8_is_zero_at_44_48": (
                ppo_material_lost == 0
            ),
            "ppo_maximum_roundoff_level_ctrlrange_loss_at_44_48": ppo_lost_max,
            "contact_bonus_is_zero_for_both_modes_at_44_48": all_contact_bonus_zero,
            "training_v5_does_not_store_exact_final_cpu_state": True,
        },
        "evidence_summary": {
            "failure_endpoint": 48,
            "endpoint_48_position_error_m": final_rl["position_error_m"],
            "endpoint_48_rotation_error_rad": final_rl["rotation_error_rad"],
            "endpoint_48_objective_score": final_rl["objective_score"],
            "endpoint_48_position_squared_contribution": final_rl[
                "ellipse_squared_contributions"
            ]["position"],
            "endpoint_48_rotation_squared_contribution": final_rl[
                "ellipse_squared_contributions"
            ]["rotation"],
            "endpoint_48_position_alone_exceeds_ellipse_boundary": (
                final_rl["ellipse_squared_contributions"]["position"] > 1.0
            ),
            "endpoint_48_ppo_minus_replay_position_error_m": final_comparison[
                "ppo_minus_replay_position_error_m"
            ],
            "endpoint_48_ppo_minus_replay_rotation_error_rad": final_comparison[
                "ppo_minus_replay_rotation_error_rad"
            ],
            "endpoint_48_ppo_minus_replay_position_vector_m": final_comparison[
                "ppo_minus_replay_tool_position_error_vector_m"
            ],
            "right_wrist_actions_at_state_feasible_bound": right_wrist_bound_hits,
            "right_tool_contacts": {
                mode: {
                    str(endpoint): [
                        name for name in final_rows["per_mode"][mode][str(endpoint)][
                            "contact"
                        ]["active"]
                        if name.startswith("right-tool-")
                    ]
                    for endpoint in ENDPOINTS
                }
                for mode in ("replay", "rl")
            },
            "endpoint_48_final_cpu_metric_within_training_min_max": (
                endpoint_48_training[
                    "final_cpu_metric_within_training_min_max"
                ]
            ),
            "endpoint_48_training_visit_count": endpoint_48_training["visit_count"],
            "endpoint_48_training_termination_count": endpoint_48_training[
                "tracking_termination_count"
            ],
        },
        "interpretation_limits": [
            "This report attributes one frozen corrected run; it does not train or select a new policy.",
            "Training v5 coverage at endpoints 47/48 is endpoint- and metric-level only, not exact-state coverage.",
            "Coarse contact flags do not encode contact geometry pairs, forces, or full solver state.",
            "The local normalized ellipse and state-feasible truncated Gaussian are unpublished engineering choices.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": report["status"],
        "output": str((args.output / "report.json").resolve()),
        "regression_gate": report["regression_gate"],
        "measured_checks": report["measured_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
