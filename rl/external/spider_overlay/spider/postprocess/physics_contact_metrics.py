"""MuJoCo acceptance for executable manipulation, plus grasp diagnostics.

The acceptance rule is deliberately manipulation-mode agnostic.  An object may
be pushed with unilateral contact while it is supported by the environment.
Once environmental support is absent, the measured hand-object force must be
large enough to explain the object's free-space dynamics.  Thumb opposition
and a two-sided force-closure proxy remain reported as stability bonuses, but
they are not universal acceptance gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
    ).lower()


def _is_object_geom(name: str, side: str) -> bool:
    prefix = f"{side}_object"
    return name == prefix or name.startswith(f"{prefix}_")


def _finger_for_geom(name: str, side: str) -> str | None:
    if _is_object_geom(name, side):
        return None
    for finger in FINGER_ORDER:
        if f"{side}_{finger}" in name:
            return finger
    return None


def _is_hand_geom(name: str, side: str) -> bool:
    return (
        name.startswith(f"collision_hand_{side}_")
        or f"{side}_palm" in name
        or _finger_for_geom(name, side) is not None
    )


def _flatten_artifact(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    with np.load(path, allow_pickle=False) as artifact:
        qpos = np.asarray(artifact["qpos"], dtype=np.float64)
        qvel = (
            np.asarray(artifact["qvel"], dtype=np.float64)
            if "qvel" in artifact.files else None
        )
        ctrl = (
            np.asarray(artifact["ctrl"], dtype=np.float64)
            if "ctrl" in artifact.files else None
        )
        time = (
            np.asarray(artifact["time"], dtype=np.float64).reshape(-1)
            if "time" in artifact.files else None
        )
        frequency = (
            float(np.asarray(artifact["frequency"]).reshape(()))
            if "frequency" in artifact.files else None
        )
    return qpos, qvel, ctrl, time, frequency


def _state_times(
    count: int, time: np.ndarray | None, frequency: float | None, default_dt: float,
) -> np.ndarray:
    if time is not None and len(time) == count and np.all(np.isfinite(time)):
        values = time - time[0]
        if count < 2 or np.all(np.diff(values) > 0.0):
            return values
    dt = 1.0 / frequency if frequency is not None and frequency > 0.0 else default_dt
    return np.arange(count, dtype=np.float64) * dt


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    fallback = np.zeros_like(quaternion)
    fallback[..., 0] = 1.0
    return np.where(norm > 1e-12, quaternion / np.maximum(norm, 1e-12), fallback)


def _interpolate_pose7(
    poses: np.ndarray, source_time: np.ndarray, target_time: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(target_time), 7), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(
            target_time, source_time, poses[:, axis],
            left=poses[0, axis], right=poses[-1, axis],
        )
    quaternions = _normalize_quaternion(np.asarray(poses[:, 3:7], dtype=np.float64))
    for index, value in enumerate(target_time):
        upper = int(np.searchsorted(source_time, value, side="right"))
        if upper <= 0:
            result[index, 3:7] = quaternions[0]
            continue
        if upper >= len(source_time):
            result[index, 3:7] = quaternions[-1]
            continue
        lower = upper - 1
        span = source_time[upper] - source_time[lower]
        alpha = 0.0 if span <= 0.0 else (value - source_time[lower]) / span
        first = quaternions[lower]
        second = quaternions[upper]
        dot = float(np.dot(first, second))
        if dot < 0.0:
            second = -second
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            blended = first + alpha * (second - first)
        else:
            angle = float(np.arccos(dot))
            blended = (
                np.sin((1.0 - alpha) * angle) * first
                + np.sin(alpha * angle) * second
            ) / np.sin(angle)
        result[index, 3:7] = _normalize_quaternion(blended)
    return result


def _body_point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point: np.ndarray,
) -> np.ndarray:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0,
    )
    angular = velocity[:3]
    linear = velocity[3:]
    return linear + np.cross(angular, point - data.xipos[int(body_id)])


def _summary(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def analyze_physics_contacts(
    model_path: str | Path,
    trajectory_path: str | Path,
    embodiment_type: str,
    *,
    reference_trajectory_path: str | Path | None = None,
    ref_dt: float = 0.02,
    min_normal_force_n: float = 0.2,
    min_environment_support_force_n: float = 0.05,
    minimum_explanatory_force_fraction: float = 0.75,
    minimum_explained_free_space_fraction: float = 0.90,
    min_opposition_cosine: float = 0.2,
    max_penetration_m: float = 0.003,
    max_contact_slip_p95_m_s: float = 0.30,
    object_position_threshold_m: float = 0.10,
    object_rotation_threshold_rad: float = 0.50,
    minimum_force_closure_frames: int = 3,
) -> dict[str, object]:
    if embodiment_type not in {"right", "left", "bimanual"}:
        raise ValueError("embodiment_type must be right, left, or bimanual")
    if not 0.0 <= minimum_explanatory_force_fraction <= 1.0:
        raise ValueError("minimum_explanatory_force_fraction must be in [0, 1]")
    if not 0.0 <= minimum_explained_free_space_fraction <= 1.0:
        raise ValueError("minimum_explained_free_space_fraction must be in [0, 1]")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos, qvel, ctrl, artifact_time, frequency = _flatten_artifact(trajectory_path)
    if qpos.ndim < 2 or qpos.shape[-1] != model.nq:
        raise ValueError(
            f"trajectory qpos shape {qpos.shape} is incompatible with nq={model.nq}"
        )
    states = qpos.reshape(-1, qpos.shape[-1])
    state_time = _state_times(len(states), artifact_time, frequency, ref_dt)
    velocity_states = None
    if qvel is not None and qvel.shape[:-1] == qpos.shape[:-1] and qvel.shape[-1] == model.nv:
        velocity_states = qvel.reshape(-1, qvel.shape[-1])
    control_states = None
    if ctrl is not None and ctrl.shape[:-1] == qpos.shape[:-1] and ctrl.shape[-1] == model.nu:
        control_states = ctrl.reshape(-1, ctrl.shape[-1])

    sides = {
        "right": ("right",), "left": ("left",),
        "bimanual": ("right", "left"),
    }[embodiment_type]
    object_info: dict[str, dict[str, Any]] = {}
    for side in sides:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_joint",
        )
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object",
        )
        if body_id >= 0 and joint_id < 0 and int(model.body_jntnum[body_id]) == 1:
            # Older generated scenes left the free joint unnamed.
            joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or body_id < 0:
            raise RuntimeError(f"missing {side} object free joint/body")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"{side}_object_joint is not a free joint")
        object_info[side] = {
            "joint_id": int(joint_id),
            "body_id": int(body_id),
            "qpos_adr": int(model.jnt_qposadr[joint_id]),
            "dof_adr": int(model.jnt_dofadr[joint_id]),
            "mass_kg": float(model.body_subtreemass[body_id]),
        }

    per_side: dict[str, dict[str, Any]] = {
        side: {
            "contact_frame_count": 0,
            "environment_support_frame_count": 0,
            "opposed_contact_frame_count": 0,
            "force_closure_frame_count": 0,
            "max_simultaneous_fingers": 0,
            "max_hand_object_penetration_m": 0.0,
            "max_object_environment_penetration_m": 0.0,
            "most_opposed_contact_normal_dot": None,
            "per_finger_contact_frame_count": {finger: 0 for finger in FINGER_ORDER},
            "per_finger_max_normal_force_n": {finger: 0.0 for finger in FINGER_ORDER},
        }
        for side in sides
    }
    temporal = {
        side: {
            "support": [], "hand_force_n": [], "support_force_n": [],
            "penetration_m": [], "slip_m_s": [], "object_position": [],
            "object_velocity": [], "object_contact_force_world_n": [],
        }
        for side in sides
    }

    all_side_names = tuple(sides)
    for frame_index, state in enumerate(states):
        data.qpos[:] = state
        data.qvel[:] = velocity_states[frame_index] if velocity_states is not None else 0.0
        if model.nu:
            data.ctrl[:] = control_states[frame_index] if control_states is not None else 0.0
        mujoco.mj_forward(model, data)

        active = {side: set() for side in sides}
        normal_force = {
            side: {finger: 0.0 for finger in FINGER_ORDER} for side in sides
        }
        normal_vector = {
            side: {finger: np.zeros(3, dtype=np.float64) for finger in FINGER_ORDER}
            for side in sides
        }
        hand_force = {side: 0.0 for side in sides}
        support_force = {side: 0.0 for side in sides}
        hand_penetration = {side: 0.0 for side in sides}
        environment_penetration = {side: 0.0 for side in sides}
        slip_samples = {side: [] for side in sides}

        for contact_index, contact in enumerate(data.contact[: data.ncon]):
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            first, second = _geom_name(model, geom1), _geom_name(model, geom2)
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            force_n = max(float(wrench[0]), 0.0)
            contact_normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0]
            penetration = max(-float(contact.dist), 0.0)

            for side in sides:
                first_object = _is_object_geom(first, side)
                second_object = _is_object_geom(second, side)
                if first_object == second_object:
                    continue
                other_geom = geom2 if first_object else geom1
                other_name = second if first_object else first
                object_geom = geom1 if first_object else geom2
                object_body = int(model.geom_bodyid[object_geom])
                other_body = int(model.geom_bodyid[other_geom])
                other_is_hand = _is_hand_geom(other_name, side)
                other_is_any_hand = any(
                    _is_hand_geom(other_name, candidate_side)
                    for candidate_side in all_side_names
                )
                other_is_any_object = any(
                    _is_object_geom(other_name, candidate_side)
                    for candidate_side in all_side_names
                )
                other_is_fixed_environment = bool(
                    not other_is_any_hand and not other_is_any_object
                    and int(model.body_jntnum[other_body]) == 0
                )

                if other_is_hand:
                    hand_force[side] += force_n
                    hand_penetration[side] = max(hand_penetration[side], penetration)
                    finger = _finger_for_geom(other_name, side)
                    if finger is not None:
                        active[side].add(finger)
                        normal_force[side][finger] += force_n
                        # Point from the object towards the finger, matching the
                        # existing thumb-vs-other opposition diagnostic.
                        direction = contact_normal.copy()
                        if not first_object:
                            direction = -direction
                        normal_vector[side][finger] += force_n * direction
                    if force_n >= min_environment_support_force_n:
                        point = np.asarray(contact.pos, dtype=np.float64)
                        object_velocity = _body_point_velocity(
                            model, data, object_body, point,
                        )
                        hand_velocity = _body_point_velocity(
                            model, data, other_body, point,
                        )
                        relative = hand_velocity - object_velocity
                        tangential = relative - np.dot(relative, contact_normal) * contact_normal
                        slip_samples[side].append(float(np.linalg.norm(tangential)))
                elif other_is_fixed_environment:
                    support_force[side] += force_n
                    environment_penetration[side] = max(
                        environment_penetration[side], penetration,
                    )

        for side in sides:
            fingers = active[side]
            record = per_side[side]
            supported = support_force[side] >= min_environment_support_force_n
            if hand_force[side] >= min_environment_support_force_n:
                record["contact_frame_count"] += 1
            if supported:
                record["environment_support_frame_count"] += 1
            record["max_simultaneous_fingers"] = max(
                record["max_simultaneous_fingers"], len(fingers)
            )
            if "thumb" in fingers and len(fingers) >= 2:
                record["opposed_contact_frame_count"] += 1
            for finger in fingers:
                record["per_finger_contact_frame_count"][finger] += 1
                record["per_finger_max_normal_force_n"][finger] = max(
                    float(record["per_finger_max_normal_force_n"][finger]),
                    normal_force[side][finger],
                )
            record["max_hand_object_penetration_m"] = max(
                float(record["max_hand_object_penetration_m"]), hand_penetration[side],
            )
            record["max_object_environment_penetration_m"] = max(
                float(record["max_object_environment_penetration_m"]),
                environment_penetration[side],
            )

            thumb_force = normal_force[side]["thumb"]
            thumb_vector = normal_vector[side]["thumb"]
            thumb_norm = float(np.linalg.norm(thumb_vector))
            best_dot: float | None = None
            if thumb_force >= min_normal_force_n and thumb_norm > 1e-12:
                thumb_direction = thumb_vector / thumb_norm
                for finger in FINGER_ORDER[1:]:
                    other_vector = normal_vector[side][finger]
                    other_norm = float(np.linalg.norm(other_vector))
                    if normal_force[side][finger] < min_normal_force_n or other_norm <= 1e-12:
                        continue
                    dot = float(np.dot(thumb_direction, other_vector / other_norm))
                    best_dot = dot if best_dot is None else min(best_dot, dot)
            if best_dot is not None:
                previous = record["most_opposed_contact_normal_dot"]
                record["most_opposed_contact_normal_dot"] = (
                    best_dot if previous is None else min(float(previous), best_dot)
                )
                if best_dot <= -min_opposition_cosine and hand_penetration[side] <= max_penetration_m:
                    record["force_closure_frame_count"] += 1

            info = object_info[side]
            adr, dof = info["qpos_adr"], info["dof_adr"]
            temporal[side]["support"].append(supported)
            temporal[side]["hand_force_n"].append(hand_force[side])
            temporal[side]["support_force_n"].append(support_force[side])
            temporal[side]["penetration_m"].append(
                max(hand_penetration[side], environment_penetration[side])
            )
            temporal[side]["slip_m_s"].extend(slip_samples[side])
            temporal[side]["object_position"].append(state[adr : adr + 3].copy())
            temporal[side]["object_velocity"].append(
                velocity_states[frame_index, dof : dof + 3].copy()
                if velocity_states is not None else np.full(3, np.nan)
            )
            # Generalized constraint force on the object's translational free
            # joint includes normal and friction components in world axes.  On
            # unsupported frames this is the physical hand-object resultant,
            # so it can test both magnitude and direction of force transfer.
            temporal[side]["object_contact_force_world_n"].append(
                np.asarray(data.qfrc_constraint[dof : dof + 3], dtype=np.float64).copy()
            )

    reference_states = reference_time = None
    if reference_trajectory_path is not None:
        reference_qpos, _, _, raw_reference_time, reference_frequency = _flatten_artifact(
            reference_trajectory_path,
        )
        if reference_qpos.shape[-1] != model.nq:
            raise ValueError("reference trajectory nq does not match model")
        reference_states = reference_qpos.reshape(-1, reference_qpos.shape[-1])
        reference_time = _state_times(
            len(reference_states), raw_reference_time, reference_frequency, ref_dt,
        )

    tracking_per_side: dict[str, Any] = {}
    all_position_errors: list[np.ndarray] = []
    all_rotation_errors: list[np.ndarray] = []
    free_space_total = 0
    free_space_explained = 0
    all_slip: list[float] = []
    maximum_penetration = 0.0
    force_explanation_per_side: dict[str, Any] = {}
    for side in sides:
        info = object_info[side]
        positions = np.asarray(temporal[side]["object_position"], dtype=np.float64)
        velocities = np.asarray(temporal[side]["object_velocity"], dtype=np.float64)
        if not np.isfinite(velocities).all():
            if len(positions) >= 2:
                velocities = np.gradient(positions, state_time, axis=0, edge_order=1)
            else:
                velocities = np.zeros_like(positions)
        if len(velocities) >= 2:
            accelerations = np.gradient(velocities, state_time, axis=0, edge_order=1)
        else:
            accelerations = np.zeros_like(velocities)
        required_force_vector = info["mass_kg"] * (
            accelerations - np.asarray(model.opt.gravity)[None, :]
        )
        required_force = np.linalg.norm(required_force_vector, axis=1)
        required_force = np.maximum(required_force, min_normal_force_n)
        hand_force_values = np.asarray(temporal[side]["hand_force_n"], dtype=np.float64)
        contact_force_vectors = np.asarray(
            temporal[side]["object_contact_force_world_n"], dtype=np.float64,
        )
        support = np.asarray(temporal[side]["support"], dtype=bool)
        free_space = ~support
        required_direction = required_force_vector / np.maximum(
            np.linalg.norm(required_force_vector, axis=1, keepdims=True), 1e-12,
        )
        explanatory_force = np.sum(
            contact_force_vectors * required_direction, axis=1,
        )
        explained = (
            hand_force_values >= min_normal_force_n
        ) & (
            explanatory_force
            >= minimum_explanatory_force_fraction * required_force
        )
        explained_free = free_space & explained
        free_count = int(free_space.sum())
        explained_count = int(explained_free.sum())
        free_space_total += free_count
        free_space_explained += explained_count
        force_explanation_per_side[side] = {
            "environment_supported_frame_count": int(support.sum()),
            "free_space_frame_count": free_count,
            "force_explained_free_space_frame_count": explained_count,
            "force_explained_free_space_fraction": (
                float(explained_count / free_count) if free_count else 1.0
            ),
            "required_hand_force_n": _summary(required_force[free_space]),
            "measured_hand_normal_force_n": _summary(hand_force_values[free_space]),
            "measured_force_along_required_direction_n": _summary(
                explanatory_force[free_space]
            ),
            "object_mass_kg": float(info["mass_kg"]),
        }
        all_slip.extend(temporal[side]["slip_m_s"])
        maximum_penetration = max(
            maximum_penetration,
            max(temporal[side]["penetration_m"], default=0.0),
        )

        if reference_states is not None and reference_time is not None:
            adr = info["qpos_adr"]
            actual_pose = states[:, adr : adr + 7]
            reference_pose = _interpolate_pose7(
                reference_states[:, adr : adr + 7], reference_time, state_time,
            )
            position_error = np.linalg.norm(actual_pose[:, :3] - reference_pose[:, :3], axis=1)
            actual_quaternion = _normalize_quaternion(actual_pose[:, 3:7])
            reference_quaternion = _normalize_quaternion(reference_pose[:, 3:7])
            quaternion_dot = np.abs(np.sum(actual_quaternion * reference_quaternion, axis=1))
            rotation_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
            all_position_errors.append(position_error)
            all_rotation_errors.append(rotation_error)
            tracking_per_side[side] = {
                "position_error_m": _summary(position_error),
                "rotation_error_rad": _summary(rotation_error),
            }

    position_errors = np.concatenate(all_position_errors) if all_position_errors else np.zeros(0)
    rotation_errors = np.concatenate(all_rotation_errors) if all_rotation_errors else np.zeros(0)
    tracking_evaluated = bool(len(position_errors) and len(rotation_errors))
    tracking_passed = bool(
        tracking_evaluated
        and float(np.mean(position_errors)) < object_position_threshold_m
        and float(np.mean(rotation_errors)) < object_rotation_threshold_rad
    )
    slip_values = np.asarray(all_slip, dtype=np.float64)
    slip_p95 = float(np.percentile(slip_values, 95)) if len(slip_values) else None
    slip_passed = bool(slip_p95 is None or slip_p95 <= max_contact_slip_p95_m_s)
    penetration_passed = bool(maximum_penetration <= max_penetration_m)
    free_space_fraction = (
        float(free_space_explained / free_space_total) if free_space_total else 1.0
    )
    force_explanation_passed = bool(
        free_space_fraction >= minimum_explained_free_space_fraction
    )

    opposed_frames = sum(int(record["opposed_contact_frame_count"]) for record in per_side.values())
    force_closure_frames = sum(int(record["force_closure_frame_count"]) for record in per_side.values())
    acceptance_checks = {
        "object_trajectory_tracking": tracking_passed,
        "bounded_relevant_penetration": penetration_passed,
        "reasonable_contact_slip": slip_passed,
        "free_space_motion_has_force_explanation": force_explanation_passed,
    }
    return {
        "schema_version": "2.0",
        "trajectory_state_count": int(len(states)),
        "state_signals_used": {
            "qpos": True,
            "qvel": velocity_states is not None,
            "ctrl": control_states is not None,
            "reference_trajectory": reference_trajectory_path is not None,
        },
        "per_side": per_side,
        "object_tracking": {
            "evaluated": tracking_evaluated,
            "passed": tracking_passed,
            "position_error_m": _summary(position_errors),
            "rotation_error_rad": _summary(rotation_errors),
            "per_side": tracking_per_side,
            "thresholds": {
                "mean_position_m": float(object_position_threshold_m),
                "mean_rotation_rad": float(object_rotation_threshold_rad),
            },
        },
        "support_conditioned_force_explanation": {
            "passed": force_explanation_passed,
            "environment_supported_frame_count": int(len(states) * len(sides) - free_space_total),
            "free_space_frame_count": int(free_space_total),
            "force_explained_free_space_frame_count": int(free_space_explained),
            "force_explained_free_space_fraction": free_space_fraction,
            "per_side": force_explanation_per_side,
            "thresholds": {
                "min_environment_support_force_n": float(min_environment_support_force_n),
                "min_hand_normal_force_n": float(min_normal_force_n),
                "minimum_explanatory_force_fraction": float(minimum_explanatory_force_fraction),
                "minimum_explained_free_space_fraction": float(minimum_explained_free_space_fraction),
                "force_direction_test": "dot(qfrc_constraint_xyz, unit(m*(a-g)))",
            },
        },
        "contact_quality": {
            "maximum_relevant_penetration_m": float(maximum_penetration),
            "maximum_allowed_penetration_m": float(max_penetration_m),
            "penetration_passed": penetration_passed,
            "contact_slip_sample_count": int(len(slip_values)),
            "contact_slip_p95_m_s": slip_p95,
            "maximum_contact_slip_p95_m_s": float(max_contact_slip_p95_m_s),
            "slip_passed": slip_passed,
        },
        "physical_acceptance": {
            "accepted": bool(all(acceptance_checks.values())),
            "checks": acceptance_checks,
            "failed_checks": [name for name, passed in acceptance_checks.items() if not passed],
            "criterion": (
                "track the object without unreasonable penetration or slip; "
                "unilateral motion is allowed under environmental support, "
                "while unsupported motion requires sufficient measured hand force"
            ),
        },
        "opposed_contact_frame_count": opposed_frames,
        "minimum_opposed_contact_frames": 3,
        "grasp_opposition_success": bool(opposed_frames >= 3),
        "force_closure_frame_count": force_closure_frames,
        "minimum_force_closure_frames": int(minimum_force_closure_frames),
        "force_closure_success": bool(force_closure_frames >= minimum_force_closure_frames),
        "force_closure_thresholds": {
            "min_normal_force_n": float(min_normal_force_n),
            "min_opposition_cosine": float(min_opposition_cosine),
            "max_penetration_m": float(max_penetration_m),
        },
        "force_closure_policy": "diagnostic_stability_bonus_not_universal_gate",
        "force_closure_interpretation": (
            "two-sided frictional force-closure proxy; not a full 6D "
            "grasp-wrench-space proof"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--reference-trajectory-path", type=Path)
    parser.add_argument("--ref-dt", type=float, default=0.02)
    parser.add_argument("--embodiment-type", choices=["right", "left", "bimanual"], required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--min-normal-force-n", type=float, default=0.2)
    parser.add_argument("--min-environment-support-force-n", type=float, default=0.05)
    parser.add_argument("--minimum-explanatory-force-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-explained-free-space-fraction", type=float, default=0.90)
    parser.add_argument("--min-opposition-cosine", type=float, default=0.2)
    parser.add_argument("--max-penetration-m", type=float, default=0.003)
    parser.add_argument("--max-contact-slip-p95-m-s", type=float, default=0.30)
    parser.add_argument("--object-position-threshold-m", type=float, default=0.10)
    parser.add_argument("--object-rotation-threshold-rad", type=float, default=0.50)
    parser.add_argument("--minimum-force-closure-frames", type=int, default=3)
    args = parser.parse_args(argv)
    metrics = analyze_physics_contacts(
        args.model_path, args.trajectory_path, args.embodiment_type,
        reference_trajectory_path=args.reference_trajectory_path,
        ref_dt=args.ref_dt,
        min_normal_force_n=args.min_normal_force_n,
        min_environment_support_force_n=args.min_environment_support_force_n,
        minimum_explanatory_force_fraction=args.minimum_explanatory_force_fraction,
        minimum_explained_free_space_fraction=args.minimum_explained_free_space_fraction,
        min_opposition_cosine=args.min_opposition_cosine,
        max_penetration_m=args.max_penetration_m,
        max_contact_slip_p95_m_s=args.max_contact_slip_p95_m_s,
        object_position_threshold_m=args.object_position_threshold_m,
        object_rotation_threshold_rad=args.object_rotation_threshold_rad,
        minimum_force_closure_frames=args.minimum_force_closure_frames,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
