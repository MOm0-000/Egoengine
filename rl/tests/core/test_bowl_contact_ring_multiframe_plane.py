from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_bowl_contact_ring_multiframe_plane_consistency_v1 import (  # noqa: E402
    angle_deg,
    fit_plane,
)


def test_rigidly_transformed_annulus_recovers_its_plane() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    ring = np.column_stack([0.03 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)])
    theta = np.deg2rad(17.0)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    translation = np.array([0.2, -0.1, 0.7])
    world = ring @ rotation.T + translation
    result = fit_plane(world)
    expected = rotation @ np.array([0.0, 0.0, 1.0])
    assert angle_deg(result["normal"], expected) < 1e-6
    assert np.isclose(result["offset_m"], expected @ translation, atol=1e-12)
    assert result["absolute_distance_m"]["maximum"] < 1e-12


def test_parallel_translated_rings_expose_plane_separation() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    first = np.column_stack([0.04 * np.cos(angle), 0.04 * np.sin(angle), np.full_like(angle, 0.5)])
    second = first + np.array([0.02, -0.03, 0.004])
    plane_first = fit_plane(first)
    plane_second = fit_plane(second)
    assert angle_deg(plane_first["normal"], plane_second["normal"]) < 1e-6
    distance_mm = (plane_first["normal"] @ plane_second["centroid_world_m"] - plane_first["offset_m"]) * 1000.0
    assert np.isclose(distance_mm, 4.0, atol=1e-9)
