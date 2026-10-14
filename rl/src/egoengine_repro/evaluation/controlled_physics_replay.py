"""Pure contracts for the bounded Brush physics Replay diagnostic."""

from __future__ import annotations

import math
from collections.abc import Iterable
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

