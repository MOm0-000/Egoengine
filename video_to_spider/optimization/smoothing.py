"""Small deterministic smoothing primitives used by the V1 optimizer."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def smooth_second_difference(values: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    """Solve weighted data fidelity plus squared second differences."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != weights.size:
        raise ValueError("values and weights have different timelines")
    count = weights.size
    if count < 3 or strength <= 0:
        return values.copy()
    difference = np.zeros((count - 2, count), dtype=np.float64)
    for row in range(count - 2):
        difference[row, row : row + 3] = [1.0, -2.0, 1.0]
    normal = np.diag(np.maximum(weights, 1e-6)) + strength * (difference.T @ difference)
    flat = values.reshape(count, -1)
    # ``values`` may come from advanced indexing (for example the five WiLoR
    # fingertips) and therefore be non-contiguous. Reshaping an empty array
    # with matching non-contiguous strides can return a temporary copy, so
    # assignments through ``result.reshape`` would leave the result storage
    # uninitialized. Solve into an explicitly contiguous flat buffer instead.
    flat_result = np.empty(flat.shape, dtype=np.float64)
    for column in range(flat.shape[1]):
        flat_result[:, column] = spsolve(csc_matrix(normal), weights * flat[:, column])
    return flat_result.reshape(values.shape)


def smooth_rotations(rotations: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    """Smooth continuous quaternions and project them back to SO(3)."""
    from scipy.spatial.transform import Rotation

    rotations = np.asarray(rotations, dtype=np.float64)
    quaternions = Rotation.from_matrix(rotations).as_quat()  # xyzw
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    filtered = smooth_second_difference(quaternions, weights, strength)
    filtered /= np.maximum(np.linalg.norm(filtered, axis=1, keepdims=True), 1e-12)
    return Rotation.from_quat(filtered).as_matrix()
