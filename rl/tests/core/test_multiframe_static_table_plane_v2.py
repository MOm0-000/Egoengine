from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_multiframe_static_table_world_consistency_v2 import (  # noqa: E402
    fit_all_points_plane,
    validate_per_frame_polygons,
)


def test_unconstrained_plane_fit_recovers_tilt_without_horizontal_prior() -> None:
    x, y = np.meshgrid(np.linspace(-0.3, 0.2, 11), np.linspace(-0.2, 0.4, 13))
    z = 0.7 + 0.2 * x - 0.15 * y
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    result = fit_all_points_plane(points)
    normal, offset = result["normal"], result["offset_m"]
    expected = np.array([-0.2, 0.15, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1e-12)
    assert np.isclose(offset, 0.7 / np.linalg.norm([-0.2, 0.15, 1.0]), atol=1e-12)


def test_per_frame_selection_rejects_reused_polygon_sets() -> None:
    polygons = {
        "a": [[1, 1], [4, 1], [4, 4]],
        "b": [[6, 1], [9, 1], [9, 4]],
        "c": [[1, 6], [4, 6], [4, 9]],
    }
    with np.testing.assert_raises_regex(ValueError, "reused across frames"):
        validate_per_frame_polygons({"0": polygons, "1": polygons}, [0, 1], 20, 20)
