from __future__ import annotations

import numpy as np


def _fit(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centroid, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, float(normal @ centroid)


def test_unconstrained_plane_fit_recovers_tilt_without_horizontal_prior() -> None:
    x, y = np.meshgrid(np.linspace(-0.3, 0.2, 11), np.linspace(-0.2, 0.4, 13))
    z = 0.7 + 0.2 * x - 0.15 * y
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    normal, offset = _fit(points)
    expected = np.array([-0.2, 0.15, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1e-12)
    assert np.isclose(offset, 0.7 / np.linalg.norm([-0.2, 0.15, 1.0]), atol=1e-12)
