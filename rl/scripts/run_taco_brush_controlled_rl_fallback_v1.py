#!/usr/bin/env python3
"""Bounded Replay->RL fallback for the Brush sample with fail-closed CPU adjudication."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [
    str(RL_ROOT / "src"),
    str(RL_ROOT / "scripts"),
    str(RL_ROOT / "external" / "human2sim2robot"),
    str(RL_ROOT / "external" / "spider_compat"),
]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
    table_audit,
)
from egoengine_repro.evaluation.controlled_physics_replay import (  # noqa: E402
    control_substep_schedule,
    longest_true_run,
    proxy_native_geometry_relation,
    proxy_native_mismatch,
    quaternion_distance_rad,
    synchronized_forward_snapshot,
)
from egoengine_repro.retarget.collision_audit import collision_families  # noqa: E402
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from run_taco_brush_controlled_physics_replay_v1 import (  # noqa: E402
    CONTACT_GROUPS,
    empty_interval,
    native_audit,
    object_state,
    sample_contacts,
    warning_counts,
)
from video_to_spider.rl.action_contract import load_residual_action_profile  # noqa: E402
from video_to_spider.rl.core.env import (  # noqa: E402
    IndependentWorlds,
    load_ego_config,
    load_reference,
)
from video_to_spider.rl.core.policy import (  # noqa: E402
    PolicyBundle,
    burn_in_prefix,
    policy_step,
)
from video_to_spider.rl.core.ppo import PPOConfig, PPOTrainer  # noqa: E402
from video_to_spider.rl.core.rollout import FixedBoundaryCollector  # noqa: E402
from video_to_spider.rl.mjwp_env import MJWPVectorEnv, MJWPVectorEnvConfig  # noqa: E402
from video_to_spider.rl.objective_contract import load_runtime_objective  # noqa: E402
from video_to_spider.rl.observation_contract import load_runtime_observation  # noqa: E402
from run_taco_brush_issue14_mink_candidate_v1 import relevant_pairs  # noqa: E402
from spider.config import build_force_closure_geom_maps  # noqa: E402


SCHEMA = "taco_brush_controlled_rl_fallback_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_controlled_rl_fallback_v1.yaml"
ANCHORS = np.asarray(((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_world(paths: dict[str, Path], *, seed: int, asymmetric: bool) -> MJWPVectorEnv:
    config = load_ego_config(paths["simulator_config"], device="cpu")
    reference = load_reference(config.data_path, device="cpu")
    objective = load_runtime_objective(
        paths["protocol"], paths["objective"],
        tracking_variant="tool_and_target", require_run_ready=True,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation"], require_run_ready=True,
    )
    residual, _ = load_residual_action_profile(paths["action"])
    world = MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=asymmetric,
            max_episode_length=60,
            tracked_object_indices=None,
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
            residual=residual,
            reset_hand_noise_std=0.0,
            reset_object_pos_noise_std=0.0,
            reset_object_rot_noise_std=0.0,
        ),
        seed=seed,
    )
    world.set_exact_initial_state(
        qpos=reference[0][0].cpu().numpy(),
        qvel=np.zeros(int(config.nv), dtype=np.float32),
        ctrl=reference[2][0].cpu().numpy(),
    )
    world.enable_state_feasible_action_contract(reference_snap_tolerance=2.0e-7)
    return world


def prepare_boundary(
    world: MJWPVectorEnv, *, start: int = 20, end: int = 60
) -> tuple[list[np.ndarray], dict[str, Any]]:
    prefix: list[np.ndarray] = []
    zero = np.zeros((1, 36), dtype=np.float32)
    for source in range(start):
        observed = world.current_observation()
        actor_observation = observed["obs"] if isinstance(observed, dict) else observed
        actor_observation = np.asarray(actor_observation)
        if actor_observation.shape != (1, 236):
            raise RuntimeError("boundary prefix actor observation has the wrong shape")
        prefix.append(actor_observation.copy())
        _, _, done, info = world.step(zero, auto_reset=False)
        if bool(done[0]) or bool(info["terminated"][0]):
            raise RuntimeError(f"Replay prefix terminated unexpectedly at endpoint {source + 1}")
    if int(world.simulation_physics_steps) != int(
        control_substep_schedule(
            control_endpoints=start, control_hz=30.0,
            simulation_dt_s=float(world.ego_cfg.sim_dt),
        ).sum()
    ):
        raise RuntimeError("MJWP Replay prefix did not preserve the authored control clock")
    world.set_chunk_reset(start=start, end=end)
    return prefix, world.get_env_state()


def initialize_zero_mean_actor(policy: PolicyBundle) -> None:
    with torch.no_grad():
        policy.actor.a2c_network.mu.weight.zero_()
        policy.actor.a2c_network.mu.bias.zero_()


def mjwp_safety_observer(world: MJWPVectorEnv, limits: dict[str, Any], maxima: dict[str, float]):
    def observe(_substep: int) -> None:
        qpos = world._mjwp.get_qpos(world.ego_cfg, world.env)
        qvel = world._mjwp.get_qvel(world.ego_cfg, world.env)
        if not bool(torch.isfinite(qpos).all()) or not bool(torch.isfinite(qvel).all()):
            raise RuntimeError("TRAINING_NONFINITE_STATE")
        object_velocity = qvel[:, -12:].reshape(1, 2, 6)
        linear = float(torch.linalg.vector_norm(object_velocity[..., :3], dim=-1).max())
        angular = float(torch.linalg.vector_norm(object_velocity[..., 3:], dim=-1).max())
        maxima["object_linear_speed_m_s"] = max(maxima["object_linear_speed_m_s"], linear)
        maxima["object_angular_speed_rad_s"] = max(maxima["object_angular_speed_rad_s"], angular)
        contact = world.env.data_wp.contact
        count = int(world.env.data_wp.nacon.numpy()[0])
        if count:
            distances = wp.to_torch(contact.dist)[:count]
            penetration = float(torch.clamp_min(-distances, 0.0).max())
            maxima["proxy_penetration_m"] = max(maxima["proxy_penetration_m"], penetration)
            if penetration > float(limits["maximum_proxy_penetration_m"]):
                raise RuntimeError("TRAINING_PROXY_PENETRATION_LIMIT")
            addresses = wp.to_torch(contact.efc_address)[:count].long()
            forces = wp.to_torch(world.env.data_wp.efc.force)[0]
            valid = (addresses >= 0) & (addresses < forces.shape[0])
            if bool(valid.any()):
                sampled = torch.where(valid, forces[addresses.clamp(0, forces.shape[0] - 1)], 0.0)
                force = float(sampled.abs().max())
                maxima["constraint_force_component_n"] = max(
                    maxima["constraint_force_component_n"], force
                )
                if force > float(limits["maximum_contact_force_n"]):
                    raise RuntimeError("TRAINING_CONTACT_FORCE_LIMIT")
        if linear > float(limits["maximum_object_linear_speed_m_s"]):
            raise RuntimeError("TRAINING_OBJECT_LINEAR_SPEED_LIMIT")
        if angular > float(limits["maximum_object_angular_speed_rad_s"]):
            raise RuntimeError("TRAINING_OBJECT_ANGULAR_SPEED_LIMIT")
    return observe


def evaluate_mjwp(
    paths: dict[str, Path], policy: PolicyBundle, prefix: list[np.ndarray],
    boundary: dict[str, Any], *, seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    world = make_world(paths, seed=seed, asymmetric=False)
    # Move the fresh world to the declared boundary before restoring the exact snapshot shape/state.
    prepare_boundary(world)
    world.set_env_state(boundary)
    states = burn_in_prefix(policy.actor, prefix, worlds=1)
    rows: dict[str, list[Any]] = {name: [] for name in (
        "source", "outcome", "action", "mu", "sigma", "reward", "score",
        "position_error", "rotation_error", "terminated", "timeout", "contact_flags",
        "qpos", "qvel", "ctrl",
    )}
    for _ in range(40):
        observed = world.current_observation()
        assert isinstance(observed, np.ndarray)
        observation = torch.as_tensor(observed, dtype=torch.float32)
        low, high = world.current_normalized_action_bounds()
        action, output = policy_step(
            policy.actor, observation, states, low, high,
            stochastic=False, distribution=policy.distribution,
        )
        states = output.rnn_states
        _, reward, _, info = world.step(action, auto_reset=False)
        rows["source"].append(int(info["source_reference_endpoint"][0]))
        rows["outcome"].append(int(info["outcome_reference_endpoint"][0]))
        rows["action"].append(action[0].numpy().copy())
        rows["mu"].append(output.mu[0].numpy().copy())
        rows["sigma"].append(output.sigma[0].numpy().copy())
        rows["reward"].append(float(reward[0]))
        rows["score"].append(float(info["object_tracking_error"][0]))
        rows["position_error"].append(np.asarray(info["object_position_error"][0]).copy())
        rows["rotation_error"].append(np.asarray(info["object_rotation_error"][0]).copy())
        rows["terminated"].append(bool(info["terminated"][0]))
        rows["timeout"].append(bool(info["time_outs"][0]))
        rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
        rows["qpos"].append(world._mjwp.get_qpos(world.ego_cfg, world.env)[0].numpy().copy())
        rows["qvel"].append(world._mjwp.get_qvel(world.ego_cfg, world.env)[0].numpy().copy())
        rows["ctrl"].append(world._last_ctrl[0].numpy().copy())
        if rows["terminated"][-1]:
            break
    arrays = {name: np.asarray(value) for name, value in rows.items()}
    failed = np.flatnonzero(arrays["terminated"])
    valid = int(failed[0]) if len(failed) else len(arrays["outcome"])
    summary = {
        "valid_intervals": valid,
        "first_failure_endpoint": int(arrays["outcome"][failed[0]]) if len(failed) else None,
        "strict_40_of_40": bool(valid == 40 and len(arrays["outcome"]) == 40),
        "mean_action_rms": float(np.sqrt(np.mean(arrays["action"] ** 2))),
        "bound_fraction": float(np.mean(np.isclose(np.abs(arrays["action"]), 1.0, atol=1e-6))),
    }
    return summary, arrays


def rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def anchors(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(position) + ANCHORS @ rotation_matrix(quaternion).T


def cpu_contact_flags(
    model: mujoco.MjModel, data: mujoco.MjData, finger_map: np.ndarray,
    object_map: np.ndarray,
) -> np.ndarray:
    result = np.zeros((2, 2, 5), dtype=np.float32)
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        f0, f1 = int(finger_map[first]), int(finger_map[second])
        o0, o1 = int(object_map[first]), int(object_map[second])
        finger, obj = ((f0, o1) if f0 >= 0 and o1 >= 0 else (f1, o0))
        if not (0 <= finger < 10 and 0 <= obj < 2) or float(contact.dist) > 0.0:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        if float(wrench[0]) > 0.0:
            result[finger // 5, obj, finger % 5] = 1.0
    return result


def cpu_observation(
    model: mujoco.MjModel, data: mujoco.MjData, reference: dict[str, np.ndarray],
    source: int, finger_map: np.ndarray, object_map: np.ndarray,
) -> np.ndarray:
    tips = np.asarray([
        data.site(model.site(f"{side}_{finger}_tip").id).xpos.copy()
        for side in ("right", "left") for finger in FINGERS
    ])
    palms = np.asarray([
        data.site(model.site(f"{side}_palm").id).xpos.copy()
        for side in ("right", "left")
    ])
    current = []
    goal = []
    for offset in (36, 43):
        current.append(anchors(data.qpos[offset:offset + 3], data.qpos[offset + 3:offset + 7]))
        target = reference["qpos"][min(source + 1, len(reference["qpos"]) - 1)]
        goal.append(anchors(target[offset:offset + 3], target[offset + 3:offset + 7]))
    observation = np.concatenate([
        data.qpos[:36], data.qvel[:36], tips.reshape(-1), palms.reshape(-1),
        np.asarray(current).reshape(-1), np.asarray(goal).reshape(-1),
        reference["ctrl"][min(source + 1, len(reference["ctrl"]) - 1)],
        reference["ctrl"][min(source + 2, len(reference["ctrl"]) - 1)],
        cpu_contact_flags(model, data, finger_map, object_map).reshape(-1),
    ]).astype(np.float32)
    if observation.shape != (236,) or not np.isfinite(observation).all():
        raise RuntimeError("CPU actor observation contract failed")
    return observation[None]


def cpu_action_bounds(model: mujoco.MjModel, reference_ctrl: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    reference = np.asarray(reference_ctrl, dtype=np.float64)
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    low = np.full(model.nu, -1.0, dtype=np.float64)
    high = np.full(model.nu, 1.0, dtype=np.float64)
    violation = np.maximum(model.actuator_ctrlrange[:, 0] - reference, 0.0)
    violation = np.maximum(violation, np.maximum(reference - model.actuator_ctrlrange[:, 1], 0.0))
    if float(violation[limited].max(initial=0.0)) > 2.0e-7:
        raise RuntimeError("CPU base command violates ctrlrange beyond snap tolerance")
    feasible_low = np.where(reference < model.actuator_ctrlrange[:, 0], reference,
                            model.actuator_ctrlrange[:, 0])
    feasible_high = np.where(reference > model.actuator_ctrlrange[:, 1], reference,
                             model.actuator_ctrlrange[:, 1])
    low[limited] = np.maximum(low[limited], (feasible_low[limited] - reference[limited]) / 0.05)
    high[limited] = np.minimum(high[limited], (feasible_high[limited] - reference[limited]) / 0.05)
    if np.any(low > high) or np.any(low > 0.0) or np.any(high < 0.0):
        raise RuntimeError("CPU state-feasible action support is invalid")
    return torch.as_tensor(low[None], dtype=torch.float32), torch.as_tensor(high[None], dtype=torch.float32)


def cpu_validate(
    cfg: dict[str, Any], paths: dict[str, Path], policy: PolicyBundle, output: Path,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    with np.load(paths["reference"], allow_pickle=False) as source:
        reference = {name: np.asarray(source[name]).copy() for name in ("qpos", "qvel", "ctrl")}
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = 0.0
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    initial_warnings = warning_counts(data)
    if any(initial_warnings):
        raise RuntimeError(f"CPU initialization warning: {initial_warnings}")
    finger_map, _, object_map = build_force_closure_geom_maps(model, "bimanual")
    finger_map = np.asarray(finger_map)
    object_map = np.asarray(object_map)
    support = yaml.safe_load(paths["support_contract"].read_text())
    simulator = support["simulator"]
    plane = Plane(
        normal=np.asarray(simulator["normal"], dtype=np.float64),
        offset=float(simulator["offset_m"]), frame=str(simulator["frame"]),
    )
    object_specs = {
        "brush": {"joint": "right_object_joint", "contact": "right_hand_brush", "floor": "brush_floor"},
        "bowl": {"joint": "left_object_joint", "contact": "left_hand_bowl", "floor": "bowl_floor"},
    }
    initial_objects = {
        name: object_state(model, data, spec["joint"]) for name, spec in object_specs.items()
    }
    states = tuple(state[:, :1].zero_() for state in policy.actor.get_default_rnn_state())
    schedule = control_substep_schedule(
        control_endpoints=60, control_hz=float(cfg["sample"]["control_hz"]),
        simulation_dt_s=float(model.opt.timestep),
    )
    limits = cfg["hard_stop"]
    audit_endpoints = set(int(value) for value in cfg["native_geometry_audit"]["endpoints"])
    endpoint_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    action_rows: list[np.ndarray] = []
    hard_stop = None
    tracking_failure = None
    previous_objects = copy.deepcopy(initial_objects)
    physics_steps = 0
    endpoint_audit_data = mujoco.MjData(model)

    def append_endpoint(endpoint: int, interval: dict[str, dict[str, Any]]) -> None:
        row: dict[str, Any] = {"endpoint": endpoint, "time_s": float(data.time)}
        scores = []
        for name, spec in object_specs.items():
            state = object_state(model, data, spec["joint"])
            joint = model.joint(spec["joint"])
            qadr = int(joint.qposadr[0])
            ref_pos = reference["qpos"][endpoint, qadr:qadr + 3]
            ref_quat = reference["qpos"][endpoint, qadr + 3:qadr + 7]
            position_error = float(np.linalg.norm(state["position_m"] - ref_pos))
            rotation_error = quaternion_distance_rad(state["quaternion_wxyz"], ref_quat)
            score = math.sqrt(
                (position_error / 0.12) ** 2 + (rotation_error / 1.2) ** 2
            )
            scores.append(score)
            row.update({
                f"{name}_position_m": state["position_m"],
                f"{name}_position_error_m": position_error,
                f"{name}_rotation_error_rad": rotation_error,
                f"{name}_score": score,
                f"{name}_actual_lift_m": float(
                    state["position_m"][2] - initial_objects[name]["position_m"][2]
                ),
                f"{name}_hand_normal_impulse_ns": interval[spec["contact"]]["normal_impulse_ns"],
                f"{name}_floor_normal_impulse_ns": interval[spec["floor"]]["normal_impulse_ns"],
                f"{name}_hand_max_force_n": interval[spec["contact"]]["max_force_n"],
                f"{name}_floor_max_force_n": interval[spec["floor"]]["max_force_n"],
            })
        row["aggregate_score"] = float(np.mean(scores))
        endpoint_rows.append(row)
        if endpoint in audit_endpoints:
            synchronized_forward_snapshot(
                mujoco, model, data, endpoint_audit_data,
            )
            audited = native_audit(
                model, endpoint_audit_data, plane,
                [
                    {"hand": "right", "object": "brush", "role": "tool", "object_side": "right"},
                    {"hand": "left", "object": "bowl", "role": "target", "object_side": "left"},
                ],
                float(cfg["native_geometry_audit"]["tolerance_m"]),
            )
            native_rows.append({"endpoint": endpoint, **audited})

    append_endpoint(0, empty_interval())
    policy.actor.eval()
    for source, substeps in enumerate(schedule):
        observation = torch.as_tensor(
            cpu_observation(model, data, reference, source, finger_map, object_map),
            dtype=torch.float32,
        )
        base = reference["ctrl"][source + 1]
        low, high = cpu_action_bounds(model, base)
        action, actor_output = policy_step(
            policy.actor, observation, states, low, high,
            stochastic=False, distribution=policy.distribution,
        )
        states = actor_output.rnn_states
        if source < int(cfg["acceptance"]["replay_prefix_end_endpoint"]):
            action = torch.zeros_like(action)
        action_np = action[0].numpy().copy()
        action_rows.append(action_np)
        data.ctrl[:] = base + np.clip(0.05 * action_np, -0.05, 0.05)
        interval = empty_interval()
        for substep in range(int(substeps)):
            interval_start_time_s = float(data.time)
            mujoco.mj_step(model, data)
            physics_steps += 1
            new_contacts, maxima = sample_contacts(
                model, data, float(model.opt.timestep), interval,
                source=source, substep=substep,
                interval_start_time_s=interval_start_time_s,
            )
            contact_rows.extend(new_contacts)
            current = {
                name: object_state(model, data, spec["joint"])
                for name, spec in object_specs.items()
            }
            failures = []
            if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.ctrl, data.qacc)):
                failures.append("NONFINITE_STATE")
            warnings = warning_counts(data)
            if limits["fail_on_any_mujoco_warning"] and any(warnings):
                failures.append("MUJOCO_WARNING")
            if maxima["force_n"] > float(limits["maximum_contact_force_n"]):
                failures.append("CONTACT_FORCE_LIMIT")
            if maxima["proxy_penetration_m"] > float(limits["maximum_proxy_penetration_m"]):
                failures.append("PROXY_PENETRATION_LIMIT")
            for name, state in current.items():
                if np.linalg.norm(state["linear_velocity_m_s"]) > float(limits["maximum_object_linear_speed_m_s"]):
                    failures.append(f"{name.upper()}_LINEAR_SPEED_LIMIT")
                if np.linalg.norm(state["angular_velocity_rad_s"]) > float(limits["maximum_object_angular_speed_rad_s"]):
                    failures.append(f"{name.upper()}_ANGULAR_SPEED_LIMIT")
                if np.linalg.norm(state["position_m"] - previous_objects[name]["position_m"]) > float(limits["maximum_object_translation_per_substep_m"]):
                    failures.append(f"{name.upper()}_SUBSTEP_TRANSLATION_LIMIT")
                if quaternion_distance_rad(state["quaternion_wxyz"], previous_objects[name]["quaternion_wxyz"]) > float(limits["maximum_object_rotation_per_substep_rad"]):
                    failures.append(f"{name.upper()}_SUBSTEP_ROTATION_LIMIT")
            previous_objects = current
            if failures:
                hard_stop = {
                    "reasons": sorted(set(failures)), "source_endpoint": source,
                    "substep": substep, "physics_step": physics_steps,
                    "maximum_contact_force_n": maxima["force_n"],
                    "maximum_proxy_penetration_m": maxima["proxy_penetration_m"],
                }
                break
        if hard_stop is not None:
            break
        append_endpoint(source + 1, interval)
        if source + 1 > int(cfg["acceptance"]["replay_prefix_end_endpoint"]):
            current_row = endpoint_rows[-1]
            failed_objects = [
                name for name in object_specs if current_row[f"{name}_score"] > 1.0
            ]
            if failed_objects:
                tracking_failure = {
                    "endpoint": source + 1, "objects": failed_objects,
                    "scores": {name: current_row[f"{name}_score"] for name in object_specs},
                }
                break

    scalar_fields = [
        "endpoint", "time_s", "aggregate_score",
        *[f"{name}_{suffix}" for name in object_specs for suffix in (
            "position_error_m", "rotation_error_rad", "score", "actual_lift_m",
            "hand_normal_impulse_ns", "floor_normal_impulse_ns", "hand_max_force_n",
            "floor_max_force_n",
        )],
    ]
    with (output / "cpu_endpoint_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in endpoint_rows)
    with gzip.open(output / "cpu_contact_trace.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = [
            "source_endpoint", "outcome_endpoint", "substep",
            "contact_state_time_s", "interval_start_time_s", "interval_end_time_s",
            "contact_index",
            "group", "geom1", "geom2", "distance_m", "normal_force_n", "force_norm_n",
            "efc_address",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(contact_rows)
    np.savez_compressed(output / "cpu_actions.npz", action=np.asarray(action_rows))
    write_json(output / "cpu_native_geometry_trace.json", native_rows)
    if hard_stop is not None:
        write_json(output / "hard_stop.json", hard_stop)

    pickup = {}
    for name, spec in object_specs.items():
        mask = [
            row[f"{name}_actual_lift_m"] >= float(cfg["acceptance"]["object_lift_delta_m"])
            and row[f"{name}_hand_normal_impulse_ns"] > 0.0
            and row[f"{name}_floor_normal_impulse_ns"] == 0.0
            for row in endpoint_rows[1:]
        ]
        run = longest_true_run(mask)
        pickup[name] = {
            "maximum_actual_lift_m": max(row[f"{name}_actual_lift_m"] for row in endpoint_rows),
            "stable_run": run,
            "passed": run >= int(cfg["acceptance"]["stable_consecutive_control_intervals"]),
        }

    native_blockers = []
    force_mismatches = []
    geometry_mismatches = []
    mismatch_impulse = {"brush": 0.0, "bowl": 0.0}
    total_sampled_impulse = {"brush": 0.0, "bowl": 0.0}
    endpoint_by_id = {int(row["endpoint"]): row for row in endpoint_rows}
    for audited in native_rows:
        endpoint = int(audited["endpoint"])
        row = endpoint_by_id[endpoint]
        for name, spec in object_specs.items():
            interaction = audited["interactions"][name]
            minimum = interaction["native_minimum"]
            proxy = interaction["proxy_minimum"]
            impulse = float(row[f"{name}_hand_normal_impulse_ns"])
            total_sampled_impulse[name] += impulse
            if minimum is None or minimum.get("distance_m") is None:
                force_class = "UNKNOWN_NATIVE_DISTANCE"
                geometry_class = "UNKNOWN_GEOMETRY_DISTANCE"
            else:
                force_class = proxy_native_mismatch(
                    proxy_contact=impulse > 0.0,
                    native_distance_m=float(minimum["distance_m"]),
                    tolerance_m=float(cfg["native_geometry_audit"]["tolerance_m"]),
                )
                geometry_class = (
                    "UNKNOWN_GEOMETRY_DISTANCE" if proxy is None else
                    proxy_native_geometry_relation(
                        proxy_distance_m=float(proxy["distance_m"]),
                        native_distance_m=float(minimum["distance_m"]),
                        tolerance_m=float(cfg["native_geometry_audit"]["tolerance_m"]),
                    )
                )
            if not force_class.startswith("AGREEMENT_"):
                force_mismatches.append({
                    "endpoint": endpoint, "object": name,
                    "classification": force_class, "normal_impulse_ns": impulse,
                })
                mismatch_impulse[name] += impulse
            if geometry_class.startswith("PROXY_") or geometry_class.startswith("UNKNOWN_"):
                geometry_mismatches.append({
                    "endpoint": endpoint, "object": name,
                    "classification": geometry_class, "normal_impulse_ns": impulse,
                })
            if minimum is None or minimum.get("classification") in {
                "PENETRATION", "UNKNOWN_INTERSECTION_DEPTH", "UNKNOWN_CONTAINMENT_OPEN_MESH",
            }:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"hand_{name}",
                    "classification": None if minimum is None else minimum.get("classification"),
                })
        for entity, table_row in audited["table"].items():
            limit = (
                float(cfg["native_geometry_audit"]["dynamic_object_floor_penetration_limit_m"])
                if entity in {"brush", "bowl"}
                else float(cfg["native_geometry_audit"]["tolerance_m"])
            )
            if float(table_row["minimum_signed_distance_m"]) < -limit:
                native_blockers.append({
                    "endpoint": endpoint, "entity": f"{entity}_table",
                    "distance_m": table_row["minimum_signed_distance_m"], "limit_m": limit,
                })

    tracking_pass = bool(
        hard_stop is None and tracking_failure is None
        and endpoint_rows[-1]["endpoint"] == 60
        and len([row for row in endpoint_rows if 20 < row["endpoint"] <= 60]) == 40
    )
    strict_pass = bool(
        tracking_pass and all(row["passed"] for row in pickup.values())
        and not native_blockers and not force_mismatches and not geometry_mismatches
    )
    maximum_force = max((row["force_norm_n"] for row in contact_rows), default=0.0)
    maximum_penetration = max((max(0.0, -row["distance_m"]) for row in contact_rows), default=0.0)
    return {
        "strict_gate_passed": strict_pass,
        "classification": (
            "CONTROLLED_RL_HARD_STOP" if hard_stop is not None else
            "CONTROLLED_RL_STRICT_PASS_NO_PROMOTION" if strict_pass else
            "CONTROLLED_RL_FAILED_CLOSED"
        ),
        "completed_endpoint": int(endpoint_rows[-1]["endpoint"]),
        "physics_steps": physics_steps,
        "hard_stop": hard_stop,
        "tracking": {"passed_40_of_40": tracking_pass, "failure": tracking_failure},
        "pickup": pickup,
        "native_blockers": native_blockers,
        "proxy_native_force_exposure_mismatches": force_mismatches,
        "proxy_native_geometry_mismatches": geometry_mismatches,
        "proxy_error_dynamic_exposure": {
            name: {
                "sampled_normal_impulse_ns": total_sampled_impulse[name],
                "mismatch_normal_impulse_ns": mismatch_impulse[name],
                "mismatch_impulse_fraction": (
                    mismatch_impulse[name] / total_sampled_impulse[name]
                    if total_sampled_impulse[name] else None
                ),
                "causal_interpretation": "DYNAMIC_EXPOSURE_NOT_A_TRUE_GEOMETRY_COUNTERFACTUAL",
            } for name in object_specs
        },
        "maximum_contact_force_n": maximum_force,
        "maximum_proxy_penetration_m": maximum_penetration,
        "promotion_authorized": False,
        "chunk_commit_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected controlled RL schema")
    if any(cfg["authorization"].get(name) is not False for name in (
        "modify_reference", "modify_scene", "modify_collision_geometry", "parameter_sweep",
        "seed_sweep", "warm_start", "promotion", "chunk_commit",
    )):
        raise ValueError("controlled RL authorization is not fail-closed")
    paths = {
        name: Path(value).resolve(strict=(name != "output"))
        for name, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    replay = json.loads(Path(cfg["trigger"]["replay_result"]).read_text())
    if replay.get("classification") != cfg["trigger"]["required_replay_classification"]:
        raise RuntimeError("Replay did not meet the frozen RL trigger")
    if replay.get("hard_stop") is not None:
        raise RuntimeError("Replay hard-stopped; RL fallback is not allowed")
    output.mkdir(parents=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    manifest = {
        "schema": SCHEMA,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "config": artifact(config_path),
        "inputs": {name: artifact(path) for name, path in paths.items() if name != "output"},
        "trigger": {"replay_result": artifact(Path(cfg["trigger"]["replay_result"])),
                    "classification": replay["classification"], "hard_stop": replay["hard_stop"]},
        "training": cfg["training"], "acceptance": cfg["acceptance"],
        "hard_stop": cfg["hard_stop"], "authorization": cfg["authorization"],
    }
    write_json(output / "input_manifest.json", manifest)

    first_world = make_world(paths, seed=0, asymmetric=True)
    prefix, boundary = prepare_boundary(first_world)
    worlds = [first_world] + [make_world(paths, seed=index, asymmetric=True) for index in range(1, 4)]
    for world in worlds[1:]:
        prepare_boundary(world)
        world.set_env_state(boundary)
    training_maxima = {
        "proxy_penetration_m": 0.0, "constraint_force_component_n": 0.0,
        "object_linear_speed_m_s": 0.0, "object_angular_speed_rad_s": 0.0,
    }
    observers = [mjwp_safety_observer(world, cfg["hard_stop"], training_maxima) for world in worlds]
    environment = IndependentWorlds(worlds, substep_observers=observers)
    policy = PolicyBundle.create(worlds=4, with_critic=True)
    initialize_zero_mean_actor(policy)
    collector = FixedBoundaryCollector(
        environment=environment, policy=policy, boundary_state=boundary,
        observation_prefix=prefix, horizon=40, start_endpoint=20, end_endpoint=60,
    )
    trainer = PPOTrainer(policy=policy, collector=collector, config=PPOConfig())

    initial, initial_arrays = evaluate_mjwp(paths, policy, prefix, boundary, seed=0)
    if initial["mean_action_rms"] > 1.0e-7:
        raise RuntimeError("baseline-preserving actor is not zero residual at epoch 0")
    np.savez_compressed(output / "mjwp_epoch_0000_evaluation.npz", **initial_arrays)
    metrics = []
    try:
        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            report = trainer.run_epoch()
            metrics.append({
                "epoch": epoch,
                "samples": report["samples"],
                "actor_loss": report["actor"]["actor_loss"],
                "internal_value_loss": report["actor"]["internal_value_loss"],
                "gradient_norm_before_clip": report["actor"]["gradient_norm_before_clip"],
                "exact_kl": report["actor"]["exact_kl"],
                "post_ratio": report["actor"]["post_ratio"],
                "live_rollout_ratio_max_abs_error": report["actor"]["live_rollout_ratio_max_abs_error"],
                "normalization_version": report["normalization_version_after_commit"],
            })
            if environment.worlds[0].simulation_physics_steps * 4 > int(cfg["hard_stop"]["maximum_training_physics_steps"]):
                raise RuntimeError("TRAINING_PHYSICS_BUDGET_EXCEEDED")
    except Exception as error:
        write_json(output / "training_hard_stop.json", {
            "classification": "CONTROLLED_RL_TRAINING_HARD_STOP",
            "error": repr(error), "completed_epochs": len(metrics),
            "training_maxima": training_maxima,
        })
        raise
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in metrics:
            stream.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
    final, final_arrays = evaluate_mjwp(paths, policy, prefix, boundary, seed=0)
    np.savez_compressed(output / "mjwp_epoch_0008_evaluation.npz", **final_arrays)
    torch.save({
        "actor": policy.actor.state_dict(),
        "critic": policy.critic.model.state_dict() if policy.critic is not None else None,
        "actor_optimizer": policy.actor_optimizer.state_dict(),
        "normalization_version": policy.normalization_version,
        "boundary": boundary,
        "prefix": prefix,
    }, output / "diagnostic_checkpoint.pt")

    cpu = cpu_validate(cfg, paths, policy, output)
    result = {
        "schema": SCHEMA,
        "classification": cpu["classification"],
        "replay_trigger": replay["classification"],
        "training": {
            "epochs": len(metrics), "samples": len(metrics) * 160,
            "physics_steps_upper_accounting": len(metrics) * 160 * 17,
            "maxima": training_maxima,
            "epoch0_mjwp": initial, "epoch8_mjwp": final,
        },
        "cpu_acceptance": cpu,
        "strict_gate_passed": cpu["strict_gate_passed"],
        "promotion_authorized": False,
        "chunk_commit_allowed": False,
        "paper_faithful": False,
    }
    write_json(output / "results.json", result)
    write_json(output / "checkpoint_eligibility.json", {
        "schema": "taco_brush_controlled_rl_checkpoint_eligibility_v1",
        "checkpoint": "diagnostic_checkpoint.pt",
        "behavioral_audit_only": True,
        "eligible_for_warm_start": False,
        "eligible_for_algorithm_comparison": True,
        "eligible_for_promotion": False,
        "eligible_for_chunk_commit": False,
        "reason": result["classification"],
    })
    summary = [
        "# Brush controlled Replay → RL fallback v1", "",
        f"Classification: `{result['classification']}`", "",
        f"- Replay trigger: `{result['replay_trigger']}`",
        f"- bounded fresh RL: `{len(metrics)} epochs / {len(metrics) * 160} samples`",
        f"- MJWP epoch 0: `{initial['valid_intervals']}/40`",
        f"- MJWP epoch 8: `{final['valid_intervals']}/40`",
        f"- CPU completed endpoint: `{cpu['completed_endpoint']}`",
        f"- CPU tracking 40/40: `{cpu['tracking']['passed_40_of_40']}`",
        f"- brush stable pickup: `{cpu['pickup']['brush']['passed']}`",
        f"- bowl stable pickup: `{cpu['pickup']['bowl']['passed']}`",
        f"- maximum contact force: `{cpu['maximum_contact_force_n']:.6f} N`",
        f"- maximum proxy penetration: `{1000 * cpu['maximum_proxy_penetration_m']:.6f} mm`",
        f"- native blockers: `{len(cpu['native_blockers'])}`",
        f"- proxy/native force mismatches: `{len(cpu['proxy_native_force_exposure_mismatches'])}`",
        f"- proxy/native geometry mismatches: `{len(cpu['proxy_native_geometry_mismatches'])}`",
        "- promotion/chunk commit: `forbidden`", "",
        "A successful optimizer run is not acceptance. Only the independent CPU contact,",
        "tracking, native-geometry and two-object stable-pickup conjunction may pass.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
