"""Pure contracts for the bounded Brush physics Replay diagnostic."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


REFERENCE_CTRL_OUTCOME_ENDPOINT = "reference_ctrl_outcome_endpoint"
REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION = (
    "reference_ctrl_linear_substep_interpolation"
)
SUPPORTED_COMMAND_TIMINGS = frozenset({
    REFERENCE_CTRL_OUTCOME_ENDPOINT,
    REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION,
})


@dataclass(frozen=True)
class DirectJointPositionActuator:
    """Compiled contract for one unit-gear joint position actuator."""

    actuator_name: str
    actuator_id: int
    joint_name: str
    joint_id: int
    qpos_address: int
    dof_address: int
    position_gain: float
    velocity_gain: float


@dataclass(frozen=True)
class FeedforwardForces:
    """Generalized-force terms applied in addition to position actuators."""

    gravity: np.ndarray
    reference_velocity: np.ndarray
    total: np.ndarray


def direct_joint_position_actuators(
    mujoco_api: Any,
    model: Any,
    actuator_names: Iterable[str],
) -> tuple[DirectJointPositionActuator, ...]:
    """Validate and describe direct, unit-gear position actuators."""
    names = tuple(str(name) for name in actuator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("controlled actuator names must be unique and nonempty")

    contracts = []
    dof_addresses = set()
    for name in names:
        actuator_id = int(model.actuator(name).id)
        if int(model.actuator_trntype[actuator_id]) != int(
            mujoco_api.mjtTrn.mjTRN_JOINT
        ):
            raise ValueError(f"{name} is not a direct joint actuator")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {
            int(mujoco_api.mjtJoint.mjJNT_HINGE),
            int(mujoco_api.mjtJoint.mjJNT_SLIDE),
        }:
            raise ValueError(f"{name} does not control a scalar hinge/slide joint")
        if int(model.actuator_dyntype[actuator_id]) != int(
            mujoco_api.mjtDyn.mjDYN_NONE
        ):
            raise ValueError(f"{name} has unsupported actuator dynamics")
        if int(model.actuator_gaintype[actuator_id]) != int(
            mujoco_api.mjtGain.mjGAIN_FIXED
        ) or int(model.actuator_biastype[actuator_id]) != int(
            mujoco_api.mjtBias.mjBIAS_AFFINE
        ):
            raise ValueError(f"{name} is not an affine position actuator")

        gear = np.asarray(model.actuator_gear[actuator_id], dtype=np.float64)
        if not np.allclose(gear, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
            raise ValueError(f"{name} must have direct unit transmission")
        position_gain = float(model.actuator_gainprm[actuator_id, 0])
        velocity_gain = float(-model.actuator_biasprm[actuator_id, 2])
        if not math.isfinite(position_gain) or position_gain <= 0.0:
            raise ValueError(f"{name} has an invalid position gain")
        if not math.isfinite(velocity_gain) or velocity_gain < 0.0:
            raise ValueError(f"{name} has an invalid velocity gain")
        if not np.isclose(
            model.actuator_biasprm[actuator_id, 1], -position_gain,
            rtol=0.0, atol=1.0e-12,
        ):
            raise ValueError(f"{name} does not implement position error feedback")

        dof_address = int(model.jnt_dofadr[joint_id])
        if dof_address in dof_addresses:
            raise ValueError("controlled actuators must map to distinct DOFs")
        dof_addresses.add(dof_address)
        contracts.append(DirectJointPositionActuator(
            actuator_name=name,
            actuator_id=actuator_id,
            joint_name=model.joint(joint_id).name,
            joint_id=joint_id,
            qpos_address=int(model.jnt_qposadr[joint_id]),
            dof_address=dof_address,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
        ))
    return tuple(contracts)


def reference_velocity_between_endpoints(
    mujoco_api: Any,
    model: Any,
    source_qpos: np.ndarray,
    outcome_qpos: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Differentiate adjacent endpoint poses over their actual step duration."""
    source = np.asarray(source_qpos, dtype=np.float64)
    outcome = np.asarray(outcome_qpos, dtype=np.float64)
    if source.shape != (int(model.nq),) or outcome.shape != (int(model.nq),):
        raise ValueError("reference endpoint qpos arrays have the wrong shape")
    if not np.isfinite(source).all() or not np.isfinite(outcome).all():
        raise ValueError("reference endpoint qpos arrays must be finite")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("reference interval duration must be positive and finite")
    velocity = np.empty(int(model.nv), dtype=np.float64)
    mujoco_api.mj_differentiatePos(
        model, velocity, float(duration_s), source, outcome,
    )
    return velocity


def gravity_bias_at_state(
    mujoco_api: Any,
    model: Any,
    live_data: Any,
    scratch_data: Any,
) -> np.ndarray:
    """Return gravity bias at live qpos using an independent zero-velocity buffer."""
    if live_data is scratch_data:
        raise ValueError("gravity calculation requires independent MjData")
    scratch_data.time = float(live_data.time)
    scratch_data.qpos[:] = live_data.qpos
    scratch_data.qvel[:] = 0.0
    for name in ("act", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
        source = getattr(live_data, name, None)
        destination = getattr(scratch_data, name, None)
        if source is not None and destination is not None:
            destination[...] = source
    scratch_data.qfrc_applied[:] = 0.0
    scratch_data.xfrc_applied[:] = 0.0
    mujoco_api.mj_forward(model, scratch_data)
    return np.asarray(scratch_data.qfrc_bias, dtype=np.float64).copy()


def controlled_feedforward_forces(
    *,
    nv: int,
    contracts: Iterable[DirectJointPositionActuator],
    gravity_bias: np.ndarray,
    reference_velocity: np.ndarray,
    gravity_feedforward: bool,
    reference_velocity_feedback: bool,
) -> FeedforwardForces:
    """Build sparse gravity and reference-velocity generalized forces."""
    gravity_bias = np.asarray(gravity_bias, dtype=np.float64)
    reference_velocity = np.asarray(reference_velocity, dtype=np.float64)
    if gravity_bias.shape != (nv,) or reference_velocity.shape != (nv,):
        raise ValueError("gravity and reference velocity must have shape (nv,)")
    if not np.isfinite(gravity_bias).all() or not np.isfinite(
        reference_velocity
    ).all():
        raise ValueError("feedforward inputs must be finite")

    gravity = np.zeros(nv, dtype=np.float64)
    velocity = np.zeros(nv, dtype=np.float64)
    for contract in contracts:
        dof = contract.dof_address
        if gravity_feedforward:
            gravity[dof] = gravity_bias[dof]
        if reference_velocity_feedback:
            velocity[dof] = contract.velocity_gain * reference_velocity[dof]
    return FeedforwardForces(
        gravity=gravity,
        reference_velocity=velocity,
        total=gravity + velocity,
    )


def synchronized_forward_snapshot(
    mujoco_api: Any,
    model: Any,
    source: Any,
    destination: Any,
) -> Any:
    """Copy a post-step state and refresh all position-derived quantities."""
    if source is destination:
        raise ValueError("synchronized snapshot requires independent MjData")
    destination.time = float(source.time)
    for name in (
        "qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied",
        "mocap_pos", "mocap_quat", "eq_active", "userdata",
    ):
        source_value = getattr(source, name, None)
        destination_value = getattr(destination, name, None)
        if source_value is not None and destination_value is not None:
            destination_value[...] = source_value
    mujoco_api.mj_forward(model, destination)
    return destination


def integrate_freejoint_pose(
    mujoco_api: Any,
    model: Any,
    pose: np.ndarray,
    velocity: np.ndarray,
    timestep_s: float,
) -> np.ndarray:
    """Advance a one-freejoint pose with MuJoCo's position integrator."""
    qpos = np.asarray(pose, dtype=np.float64).copy()
    qvel = np.asarray(velocity, dtype=np.float64)
    if int(model.nq) != 7 or int(model.nv) != 6:
        raise ValueError("freejoint reconstruction requires nq=7 and nv=6")
    if qpos.shape != (7,) or qvel.shape != (6,):
        raise ValueError("freejoint pose/velocity shapes must be (7,) and (6,)")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("freejoint pose and velocity must be finite")
    if not math.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("timestep must be positive and finite")
    mujoco_api.mj_integratePos(model, qpos, qvel, float(timestep_s))
    return qpos


def control_substep_schedule(
    *, control_endpoints: int, control_hz: float, simulation_dt_s: float
) -> np.ndarray:
    """Return the deterministic native-step count for each control interval.

    The authored MuJoCo timestep is intentionally preserved.  When it does not
    divide the 30 Hz reference period, endpoint times are rounded upward to the
    first available native simulator time.  The resulting bounded phase error
    is less than one native timestep and does not accumulate.
    """
    if control_endpoints <= 0:
        raise ValueError("control_endpoints must be positive")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if not math.isfinite(simulation_dt_s) or simulation_dt_s <= 0:
        raise ValueError("simulation_dt_s must be positive and finite")
    cumulative = np.ceil(
        np.arange(1, control_endpoints + 1, dtype=np.float64)
        / (control_hz * simulation_dt_s)
        - 1.0e-12
    ).astype(np.int64)
    counts = np.diff(np.r_[0, cumulative])
    if np.any(counts <= 0):
        raise ValueError("each control interval must contain a physics step")
    return counts


def reference_control_target(
    controls: np.ndarray,
    *,
    source_endpoint: int,
    outcome_endpoint: int,
    substep: int,
    substeps: int,
    command_timing: str,
) -> tuple[np.ndarray, float]:
    """Return one frozen or linearly interpolated actuator position target."""
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("controls must be a finite 2D array")
    if outcome_endpoint != source_endpoint + 1:
        raise ValueError("control target requires adjacent reference endpoints")
    if not 0 <= source_endpoint < outcome_endpoint < len(values):
        raise ValueError("control endpoint is out of range")
    if substeps <= 0 or not 0 <= substep < substeps:
        raise ValueError("substep must be in [0, substeps)")
    if command_timing not in SUPPORTED_COMMAND_TIMINGS:
        raise ValueError(f"unsupported command timing: {command_timing}")
    if command_timing == REFERENCE_CTRL_OUTCOME_ENDPOINT:
        return values[outcome_endpoint].copy(), 1.0
    alpha = float(substep + 1) / float(substeps)
    target = values[source_endpoint] + alpha * (
        values[outcome_endpoint] - values[source_endpoint]
    )
    return target, alpha


def quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    """Sign-invariant angular distance for scalar-first quaternions."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero quaternion is invalid")
    dot = abs(float((left / left_norm) @ (right / right_norm)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def longest_true_run(values: Iterable[bool]) -> int:
    """Length of the longest consecutive true run."""
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def proxy_native_mismatch(
    *, proxy_contact: bool, native_distance_m: float, tolerance_m: float
) -> str:
    """Classify proxy-force/native-surface agreement at one sampled endpoint."""
    if not math.isfinite(native_distance_m):
        return "UNKNOWN_NONFINITE_NATIVE_DISTANCE"
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if native_distance_m < -tolerance_m:
        native = "PENETRATION"
    elif native_distance_m > tolerance_m:
        native = "CLEARANCE"
    else:
        native = "CONTACT"
    if proxy_contact and native == "CLEARANCE":
        return "PROXY_FORCE_WHILE_NATIVE_CLEAR"
    if not proxy_contact and native == "PENETRATION":
        return "NATIVE_PENETRATION_WITHOUT_PROXY_FORCE"
    return f"AGREEMENT_{native}_{'FORCE' if proxy_contact else 'NO_FORCE'}"


def proxy_native_geometry_relation(
    *, proxy_distance_m: float, native_distance_m: float, tolerance_m: float
) -> str:
    """Compare endpoint proxy and native surface signs without force timing."""
    if not math.isfinite(proxy_distance_m) or not math.isfinite(native_distance_m):
        return "UNKNOWN_NONFINITE_GEOMETRY_DISTANCE"
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    def classify(value: float) -> str:
        if value < -tolerance_m:
            return "PENETRATION"
        if value > tolerance_m:
            return "CLEARANCE"
        return "CONTACT"

    proxy = classify(proxy_distance_m)
    native = classify(native_distance_m)
    if proxy == "PENETRATION" and native == "CLEARANCE":
        return "PROXY_OVERCOVERAGE_PENETRATION_WHILE_NATIVE_CLEAR"
    if proxy == "CLEARANCE" and native == "PENETRATION":
        return "PROXY_MISSED_NATIVE_PENETRATION"
    return f"GEOMETRY_{proxy}_VS_{native}"

