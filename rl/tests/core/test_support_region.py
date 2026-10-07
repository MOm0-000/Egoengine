from __future__ import annotations

import numpy as np
import pytest

from egoengine_repro.scene.support_region import inspect_lower_support_region


def _ring(*, wave_m: float = 0.0, sectors: int = 72):
    angles = np.arange(sectors) * 2.0 * np.pi / sectors
    radii = np.array([0.030, 0.035, 0.040])
    vertices = []
    for angle in angles:
        for radius in radii:
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            z = 0.2 + 0.1 * x - 0.05 * y + wave_m * np.sin(4.0 * angle)
            vertices.append([x, y, z])
    vertices = np.asarray(vertices)
    faces = []
    for sector in range(sectors):
        next_sector = (sector + 1) % sectors
        for radial in range(2):
            a = sector * 3 + radial
            b = next_sector * 3 + radial
            c = next_sector * 3 + radial + 1
            d = sector * 3 + radial + 1
            faces.extend([[a, b, c], [a, c, d]])
    return vertices, np.asarray(faces)


def test_tilted_flat_ring_yields_arbitrary_well_defined_plane() -> None:
    vertices, faces = _ring()
    result = inspect_lower_support_region(vertices, faces)
    expected = np.array([-0.1, 0.05, 1.0])
    expected /= np.linalg.norm(expected)
    assert result["classification"] == "SUPPORT_PLANE_WELL_DEFINED"
    assert result["plane"]["normal"] == pytest.approx(expected, abs=1e-12)
    assert result["planarity"]["absolute_distance_m"]["maximum"] < 1e-12
    assert result["coverage"]["in_plane_singular_value_aspect"] > 0.9


def test_spatially_distributed_but_wavy_ring_is_not_planar() -> None:
    vertices, faces = _ring(wave_m=0.003)
    result = inspect_lower_support_region(vertices, faces)
    assert result["decision_gates"]["region_sufficient"] is True
    assert result["decision_gates"]["approximately_planar"] is False
    assert result["classification"] == "SUPPORT_GEOMETRY_NOT_PLANAR"


def test_line_like_lower_envelope_is_insufficient() -> None:
    vertices, faces = _ring()
    vertices[:, 1] *= 0.005
    result = inspect_lower_support_region(vertices, faces)
    assert result["decision_gates"]["region_sufficient"] is False
    assert result["classification"] == "SUPPORT_REGION_INSUFFICIENT"
