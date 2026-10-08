"""Pure helpers for the Brush Replay failure-attribution audit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def semantic_finger(name: str) -> str:
    lowered = name.lower()
    if "_mid_" in lowered or lowered.endswith("_mid"):
        return "middle"
    for value in ("thumb", "index", "middle", "ring", "pinky", "palm"):
        if value in lowered:
            return value
    return "other"


def vector_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("vectors must have shape (3,)")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-15:
        return None
    return float(np.clip(float(left @ right) / denominator, -1.0, 1.0))


def classify_native_measurement(measured: dict[str, Any], tolerance_m: float) -> str:
    """Apply the repository's fail-closed native material convention."""
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    depth = measured.get("reported_penetration_depth_m")
    containment = measured.get("containment_followup")
    unknown = bool(
        isinstance(containment, list)
        and any(str(row.get("status", "")).startswith("UNKNOWN") for row in containment)
    )
    intersection = bool(
        measured.get("surface_intersection", False)
        or measured.get("containment_penetration", False)
    )
    if intersection and depth is None:
        return "UNKNOWN_INTERSECTION_DEPTH"
    if intersection and float(depth) > tolerance_m:
        return "PENETRATION"
    if intersection:
        return "CONTACT_WITHIN_TOLERANCE"
    if unknown:
        return "UNKNOWN_CONTAINMENT_OPEN_MESH"
    distance = float(measured["distance_m"])
    if distance < -tolerance_m:
        return "PENETRATION"
    if distance > tolerance_m:
        return "CLEARANCE"
    return "CONTACT_WITHIN_TOLERANCE"

