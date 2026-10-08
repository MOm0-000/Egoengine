"""Pure contracts for the bounded Brush physics Replay diagnostic."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


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

