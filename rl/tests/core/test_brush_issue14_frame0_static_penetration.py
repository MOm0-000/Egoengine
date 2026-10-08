from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    classify_distance,
    object_global_minimum,
    transform_world_horizontal_plane_to_sim,
)


def test_object_global_minimum_uses_every_frame_and_vertex() -> None:
    vertices = np.array([[0.0, 0.0, 0.2], [1.0, 0.0, -0.1]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, 2, 3] = -0.3
    poses[2, 2, 3] = -0.2
    result = object_global_minimum(vertices, poses, role="tool", object_id="x")
    assert result["frame"] == 1
    assert result["vertex_index"] == 1
    assert result["minimum_world_z_m"] == -0.4


def test_world_horizontal_plane_transform_preserves_equation() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.5, -0.25, 0.17]
    plane = transform_world_horizontal_plane_to_sim(transform, 0.55)
    assert np.array_equal(plane.normal, [0.0, 0.0, 1.0])
    assert np.isclose(plane.offset, 0.72, atol=1e-15)
    source = np.array([0.1, 0.2, 0.55, 1.0])
    sim = transform @ source
    assert np.isclose(plane.signed_distance(sim[:3]), 0.0, atol=1e-15)


def test_penetration_threshold_is_not_relaxed() -> None:
    tolerance = 5e-5
    assert classify_distance(-tolerance - 1e-12, tolerance) == "PENETRATION"
    assert classify_distance(-tolerance, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert classify_distance(tolerance, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert classify_distance(tolerance + 1e-12, tolerance) == "CLEARANCE"
