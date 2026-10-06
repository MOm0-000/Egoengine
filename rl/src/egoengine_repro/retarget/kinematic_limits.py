"""Shared low-level kinematic limits used by active retargeters."""

from __future__ import annotations

from typing import Any

import numpy as np


class FrameDisplacementLimit:
    """Limit aggregate per-frame hand displacement across all QP solves."""

    def __init__(
        self, model: Any, mujoco: Any, velocity_map: dict[str, float],
        constraint_type: Any,
    ) -> None:
        dofs: list[int] = []
        qpos_addresses: list[int] = []
        speeds: list[float] = []
        for joint in range(int(model.njnt)):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint
            ) or ""
            if name not in velocity_map:
                continue
            if int(model.jnt_type[joint]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(
                    f"frame displacement limit requires a 1-DoF joint: {name}"
                )
            dofs.append(int(model.jnt_dofadr[joint]))
            qpos_addresses.append(int(model.jnt_qposadr[joint]))
            speeds.append(float(velocity_map[name]))
        self.model = model
        self.constraint_type = constraint_type
        self.dofs = np.asarray(dofs, dtype=np.int64)
        self.qpos_addresses = np.asarray(qpos_addresses, dtype=np.int64)
        self.speeds = np.asarray(speeds, dtype=np.float64)
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None

    def set_previous(self, qpos: np.ndarray | None, dt: float | None = None) -> None:
        if qpos is None:
            self.lower = self.upper = None
            return
        if dt is None or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("frame displacement limit requires a positive frame dt")
        center = np.asarray(qpos, dtype=np.float64)[self.qpos_addresses]
        radius = self.speeds * float(dt)
        radius -= np.minimum(1.0e-6, 0.01 * radius)
        self.lower = center - radius
        self.upper = center + radius

    def compute_qp_inequalities(self, configuration: Any, _dt: float) -> Any:
        if self.lower is None or self.upper is None or not len(self.dofs):
            return self.constraint_type()
        projection = np.eye(self.model.nv, dtype=np.float64)[self.dofs]
        current = np.asarray(configuration.q, dtype=np.float64)[self.qpos_addresses]
        matrix = np.vstack([projection, -projection])
        bounds = np.concatenate([self.upper - current, current - self.lower])
        return self.constraint_type(matrix, bounds)

    def minimum_margin(self, configuration: Any) -> float:
        if self.lower is None or self.upper is None or not len(self.dofs):
            return float("inf")
        current = np.asarray(configuration.q, dtype=np.float64)[self.qpos_addresses]
        return float(np.minimum(current - self.lower, self.upper - current).min())


class StrictCollisionLimit:
    """Require bounded separation when an upstream MINK pair is invalid."""

    def __init__(
        self, inner: Any, mujoco: Any, *, minimum_distance: float,
        depenetration_step: float, gain: float = 0.85,
        deepest_invalid_only: bool = False,
    ) -> None:
        self.inner = inner
        self.mujoco = mujoco
        self.minimum_distance = float(minimum_distance)
        self.depenetration_step = float(depenetration_step)
        self.gain = float(gain)
        self.deepest_invalid_only = bool(deepest_invalid_only)
        self.geom_id_pairs = inner.geom_id_pairs
        self.active = True
        self.enabled = False

    def compute_qp_inequalities(self, configuration: Any, dt: float) -> Any:
        constraint = self.inner.compute_qp_inequalities(configuration, dt)
        if not self.active:
            return type(constraint)()
        if not self.enabled:
            return constraint
        if constraint.inactive or constraint.G is None or constraint.h is None:
            return constraint
        h = np.asarray(constraint.h, dtype=np.float64).copy()
        matrix = np.asarray(constraint.G, dtype=np.float64).copy()
        fromto = np.empty(6, dtype=np.float64)
        detection = float(self.inner.collision_detection_distance)
        invalid: list[tuple[float, int, np.ndarray]] = []
        for index, (geom_a, geom_b) in enumerate(self.geom_id_pairs):
            distance = float(self.mujoco.mj_geomDistance(
                configuration.model, configuration.data,
                geom_a, geom_b, detection, fromto,
            ))
            if abs(distance - detection) < 1e-12:
                continue
            residual = distance - self.minimum_distance
            if residual >= 0.0:
                h[index] = self.gain * residual
            else:
                row = matrix[index].copy()
                if self.deepest_invalid_only:
                    matrix[index] = 0.0
                    h[index] = np.inf
                else:
                    h[index] = 1e-5
                invalid.append((residual, index, row))
        if invalid:
            residual, index, row = min(invalid, key=lambda item: item[0])
            if self.deepest_invalid_only:
                matrix[index] = row
            h[index] = -min(
                self.depenetration_step,
                self.gain * (-residual),
            )
        return type(constraint)(matrix, h)


def joint_velocity_limits(
    model: Any, mujoco: Any, settings: Any,
) -> dict[str, float]:
    if hasattr(settings, "to_dict"):
        settings = settings.to_dict()
    try:
        category_limits = {
            key: float(settings[key])
            for key in ("base_translation", "base_rotation", "finger")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "velocity limits must define base_translation, base_rotation, and finger"
        ) from error
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in category_limits.values()
    ):
        raise ValueError("all joint velocity limits must be finite and positive")
    limits: dict[str, float] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if (
            name is None
            or "object" in name
            or int(model.jnt_type[joint_id])
            == int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            continue
        if any(token in name for token in ("_tx_", "_ty_", "_tz_")):
            limits[name] = category_limits["base_translation"]
        elif any(token in name for token in ("roll", "pitch", "yaw")):
            limits[name] = category_limits["base_rotation"]
        else:
            limits[name] = category_limits["finger"]
    return limits


def enable_planning_collision_masks(
    model: Any, hand_geom_ids: list[int], object_geom_ids: list[int],
) -> None:
    """Expose explicit-pair geoms to MINK's planning pair enumerator."""
    for geom_id in set(hand_geom_ids) | set(object_geom_ids):
        model.geom_contype[geom_id] |= 1
        model.geom_conaffinity[geom_id] |= 1


def explicit_collision_groups(
    model: Any, mujoco: Any, *, hand_geom_ids: set[int],
    object_geom_ids: set[int] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Return one MINK group per selected explicit MuJoCo collision pair."""
    groups: list[tuple[list[str], list[str]]] = []
    object_ids = None if object_geom_ids is None else set(object_geom_ids)
    for pair_id in range(int(model.npair)):
        geom_a = int(model.pair_geom1[pair_id])
        geom_b = int(model.pair_geom2[pair_id])
        if object_ids is None:
            selected = geom_a in hand_geom_ids and geom_b in hand_geom_ids
        else:
            selected = (
                (geom_a in hand_geom_ids and geom_b in object_ids)
                or (geom_b in hand_geom_ids and geom_a in object_ids)
            )
        if not selected:
            continue
        name_a = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_a
        )
        name_b = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_b
        )
        if name_a is None or name_b is None:
            raise ValueError("explicit collision pair contains an unnamed geom")
        groups.append(([name_a], [name_b]))
    return groups
