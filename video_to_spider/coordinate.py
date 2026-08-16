#!/usr/bin/env python3
"""Frozen transform helpers for device-level coordinates.

Convention: T_A_B maps a point from frame B into frame A.
"""

from __future__ import annotations

import numpy as np


def invert_transform(T_A_B: np.ndarray) -> np.ndarray:
    T = np.asarray(T_A_B, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose_transform(T_A_B: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    return np.asarray(T_A_B, dtype=np.float64) @ np.asarray(T_B_C, dtype=np.float64)


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    T = np.asarray(T_A_B, dtype=np.float64)
    points = np.asarray(points_B, dtype=np.float64)
    homogeneous = np.concatenate(
        [points[..., :3], np.ones((*points.shape[:-1], 1), dtype=np.float64)],
        axis=-1,
    )
    return np.einsum("ij,...j->...i", T, homogeneous)[..., :3]
