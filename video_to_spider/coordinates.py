"""Coordinate and time conversion utilities using the frozen V1 conventions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .schemas import ArtifactValidationError, validate_transforms


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform)
    validate_transforms("transform", value)
    result = np.zeros_like(value)
    rotation_t = np.swapaxes(value[..., :3, :3], -1, -2)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -(rotation_t @ value[..., :3, 3, None])[..., 0]
    result[..., 3, 3] = 1
    return result


def compose_transforms(T_A_B: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    a, b = np.asarray(T_A_B), np.asarray(T_B_C)
    validate_transforms("T_A_B", a)
    validate_transforms("T_B_C", b)
    result = a @ b
    validate_transforms("T_A_C", result)
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value, xyz = np.asarray(transform), np.asarray(points)
    validate_transforms("transform", value)
    if xyz.shape[-1] != 3 or not np.all(np.isfinite(xyz)):
        raise ArtifactValidationError("points must be finite and end in dimension 3")
    return (value[..., :3, :3] @ xyz[..., None])[..., 0] + value[..., :3, 3]


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ArtifactValidationError("rotation matrix must end in (3, 3)")
    xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    wxyz[wxyz[:, 0] < 0] *= -1
    return wxyz.reshape(matrix.shape[:-2] + (4,))


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion)
    if value.shape[-1] != 4 or not np.all(np.isfinite(value)):
        raise ArtifactValidationError("quaternion must be finite and end in dimension 4")
    norm = np.linalg.norm(value, axis=-1)
    if np.any(norm < 1e-12):
        raise ArtifactValidationError("zero quaternion is invalid")
    normalized = value / norm[..., None]
    xyzw = normalized.reshape(-1, 4)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix().reshape(value.shape[:-1] + (3, 3))


def rotation_geodesic_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(np.asarray(a), -1, -2) @ np.asarray(b)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(relative.shape[:-2])


def resample_transforms(
    timestamps_s: np.ndarray, transforms: np.ndarray, target_timestamps_s: np.ndarray
) -> np.ndarray:
    source_t = np.asarray(timestamps_s, dtype=np.float64)
    target_t = np.asarray(target_timestamps_s, dtype=np.float64)
    values = np.asarray(transforms)
    validate_transforms("transforms", values)
    if values.shape != (source_t.size, 4, 4):
        raise ArtifactValidationError("transforms and timestamps length mismatch")
    if source_t.size < 2 or np.any(np.diff(source_t) <= 0) or np.any(np.diff(target_t) <= 0):
        raise ArtifactValidationError("source and target timestamps must be strictly increasing")
    if target_t[0] < source_t[0] - 1e-9 or target_t[-1] > source_t[-1] + 1e-9:
        raise ArtifactValidationError("target timestamps may not extrapolate")
    result = np.repeat(np.eye(4, dtype=values.dtype)[None], target_t.size, axis=0)
    for axis in range(3):
        result[:, axis, 3] = np.interp(target_t, source_t, values[:, axis, 3])
    rotations = Rotation.from_matrix(values[:, :3, :3])
    result[:, :3, :3] = Slerp(source_t, rotations)(target_t).as_matrix()
    return result

