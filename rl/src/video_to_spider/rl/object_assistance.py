"""Deterministic, training-only Cartesian assistance for one rigid tool body.

This module deliberately contains no curriculum or reward logic.  It resolves
one MuJoCo free body and computes the finite world-frame wrench requested by
the virtual-object-assist experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class UnsupportedAssistanceBody(RuntimeError):
    """The scene cannot satisfy the frozen single-rigid-body contract."""


@dataclass(frozen=True)
class ToolAssistSpec:
    body_name: str = "right_object"
    position_frequency: float = 15.0
    rotation_frequency: float = 10.0
    damping_ratio: float = 1.0
    force_cap_gravity_multiple: float = 3.0
    torque_cap_acceleration: float = 100.0
    substeps: int = 10
    control_dt: float = 1.0 / 30.0

    def validate(self) -> None:
        scalars = (
            self.position_frequency,
            self.rotation_frequency,
            self.damping_ratio,
            self.force_cap_gravity_multiple,
            self.torque_cap_acceleration,
            self.control_dt,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scalars):
            raise ValueError("assistance constants must be finite and positive")
        if self.substeps != 10:
            raise ValueError("virtual-object assist is frozen to ten physics substeps")


@dataclass(frozen=True)
class ToolBodyInfo:
    body_name: str
    body_id: int
    joint_id: int
    qpos_address: int
    qvel_address: int
    mass: float
    principal_inertia: tuple[float, float, float]
    inertial_position: tuple[float, float, float]
    inertial_quaternion_wxyz: tuple[float, float, float, float]
    gravity_world: tuple[float, float, float]

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WrenchResult:
    wrench: np.ndarray
    raw_force: np.ndarray
    raw_torque: np.ndarray
    force_capped: bool
    torque_capped: bool


def _xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be one finite wxyz row")
    norm = np.linalg.norm(value)
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    value = value / norm
    return value[[1, 2, 3, 0]]


def rotation_from_wxyz(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(_xyzw(quaternion_wxyz))


def resolve_tool_body(
    model: mujoco.MjModel,
    *,
    spec: ToolAssistSpec,
    object_roles: tuple[str, ...],
    reference_qpos: np.ndarray,
    expected_tool_pose: tuple[np.ndarray, np.ndarray],
) -> ToolBodyInfo:
    """Resolve and validate the one supported free-body tool mapping."""
    spec.validate()
    if not object_roles or object_roles[0] != "tool":
        raise UnsupportedAssistanceBody("object_roles[0] is not the tracked tool")
    body_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
    )
    if body_id < 0:
        raise UnsupportedAssistanceBody(f"tool body {spec.body_name!r} is missing")
    if int(model.body_parentid[body_id]) != 0:
        raise UnsupportedAssistanceBody("tool free body is not a world child")
    if int(model.body_jntnum[body_id]) != 1:
        raise UnsupportedAssistanceBody("tool body does not have exactly one joint")
    joint_id = int(model.body_jntadr[body_id])
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise UnsupportedAssistanceBody("tool joint is not a MuJoCo free joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    if qpos_address < 0 or qpos_address + 7 > model.nq:
        raise UnsupportedAssistanceBody("tool qpos address is invalid")
    if qvel_address < 0 or qvel_address + 6 > model.nv:
        raise UnsupportedAssistanceBody("tool qvel address is invalid")
    mass = float(model.body_mass[body_id])
    inertia = np.asarray(model.body_inertia[body_id], dtype=np.float64)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    if mass <= 0.0 or np.any(inertia <= 0.0):
        raise UnsupportedAssistanceBody("tool mass/inertia must be positive")
    if not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= 0.0:
        raise UnsupportedAssistanceBody("nonzero finite gravity is required")
    row = np.asarray(reference_qpos, dtype=np.float64)
    pose_position = row[qpos_address : qpos_address + 3]
    pose_rotation = rotation_from_wxyz(row[qpos_address + 3 : qpos_address + 7]).as_matrix()
    expected_position, expected_rotation = expected_tool_pose
    if not np.allclose(pose_position, expected_position, rtol=0.0, atol=1.0e-7):
        raise UnsupportedAssistanceBody("resolved free joint does not match reference tool position")
    if not np.allclose(pose_rotation, expected_rotation, rtol=0.0, atol=1.0e-6):
        raise UnsupportedAssistanceBody("resolved free joint does not match reference tool rotation")
    return ToolBodyInfo(
        body_name=spec.body_name,
        body_id=body_id,
        joint_id=joint_id,
        qpos_address=qpos_address,
        qvel_address=qvel_address,
        mass=mass,
        principal_inertia=tuple(float(value) for value in inertia),
        inertial_position=tuple(float(value) for value in model.body_ipos[body_id]),
        inertial_quaternion_wxyz=tuple(
            float(value) for value in model.body_iquat[body_id]
        ),
        gravity_world=tuple(float(value) for value in gravity),
    )


def assistance_scale(epoch: int) -> float:
    if not isinstance(epoch, int) or not 1 <= epoch <= 250:
        raise ValueError("assistance epoch must be in [1, 250]")
    if epoch <= 50:
        return 1.0
    if epoch <= 199:
        return float(200 - epoch) / 150.0
    return 0.0


def interpolate_target(
    position0: np.ndarray,
    quaternion0_wxyz: np.ndarray,
    position1: np.ndarray,
    quaternion1_wxyz: np.ndarray,
    *,
    fraction: float,
    control_dt: float,
    inertial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return target COM position/rotation and their world velocities."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("substep interpolation fraction must be in [0, 1)")
    p0 = np.asarray(position0, dtype=np.float64)
    p1 = np.asarray(position1, dtype=np.float64)
    rotation0 = rotation_from_wxyz(quaternion0_wxyz)
    rotation1 = rotation_from_wxyz(quaternion1_wxyz)
    rotation_vector = (rotation1 * rotation0.inv()).as_rotvec()
    rotation = Rotation.from_rotvec(fraction * rotation_vector) * rotation0
    matrix = rotation.as_matrix()
    origin = p0 + fraction * (p1 - p0)
    linear_origin = (p1 - p0) / float(control_dt)
    angular_world = rotation_vector / float(control_dt)
    offset_world = matrix @ np.asarray(inertial_position, dtype=np.float64)
    com = origin + offset_world
    linear_com = linear_origin + np.cross(angular_world, offset_world)
    return com, matrix, linear_com, angular_world


def _norm_cap(vector: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum:
        return vector, False
    return vector * (maximum / norm), True


def compute_wrench(
    *,
    alpha: float,
    spec: ToolAssistSpec,
    body: ToolBodyInfo,
    target_com_position: np.ndarray,
    target_rotation_world: np.ndarray,
    target_com_velocity: np.ndarray,
    target_angular_velocity_world: np.ndarray,
    live_qpos: np.ndarray,
    live_qvel: np.ndarray,
) -> WrenchResult:
    """Compute the capped world-frame COM wrench for one live substep."""
    if alpha == 0.0:
        zeros = np.zeros(3, dtype=np.float64)
        return WrenchResult(np.zeros(6), zeros, zeros.copy(), False, False)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("assistance alpha must be finite and in (0, 1]")
    qpos = np.asarray(live_qpos, dtype=np.float64)
    qvel = np.asarray(live_qvel, dtype=np.float64)
    qadr, vadr = body.qpos_address, body.qvel_address
    origin = qpos[qadr : qadr + 3]
    rotation = rotation_from_wxyz(qpos[qadr + 3 : qadr + 7]).as_matrix()
    linear_origin = qvel[vadr : vadr + 3]
    angular_body = qvel[vadr + 3 : vadr + 6]
    angular_world = rotation @ angular_body
    offset_world = rotation @ np.asarray(body.inertial_position)
    com = origin + offset_world
    linear_com = linear_origin + np.cross(angular_world, offset_world)
    position_error = np.asarray(target_com_position) - com
    rotation_error = Rotation.from_matrix(
        np.asarray(target_rotation_world) @ rotation.T
    ).as_rotvec()
    gravity = np.asarray(body.gravity_world)
    force_raw = body.mass * (
        spec.position_frequency**2 * position_error
        + 2.0
        * spec.damping_ratio
        * spec.position_frequency
        * (np.asarray(target_com_velocity) - linear_com)
        - gravity
    )
    inertia_body = (
        rotation_from_wxyz(np.asarray(body.inertial_quaternion_wxyz)).as_matrix()
        @ np.diag(np.asarray(body.principal_inertia))
        @ rotation_from_wxyz(np.asarray(body.inertial_quaternion_wxyz)).as_matrix().T
    )
    inertia_world = rotation @ inertia_body @ rotation.T
    torque_raw = inertia_world @ (
        spec.rotation_frequency**2 * rotation_error
        + 2.0
        * spec.damping_ratio
        * spec.rotation_frequency
        * (np.asarray(target_angular_velocity_world) - angular_world)
    )
    force, force_capped = _norm_cap(
        force_raw, spec.force_cap_gravity_multiple * body.mass * np.linalg.norm(gravity)
    )
    torque, torque_capped = _norm_cap(
        torque_raw,
        spec.torque_cap_acceleration * max(body.principal_inertia),
    )
    wrench = alpha * np.concatenate((force, torque))
    if not np.isfinite(wrench).all():
        raise FloatingPointError("assistance wrench is non-finite")
    return WrenchResult(wrench, force_raw, torque_raw, force_capped, torque_capped)
