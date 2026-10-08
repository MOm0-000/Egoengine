from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import trimesh


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_collision_model_accuracy_v1 import (  # noqa: E402
    distance_class,
    union_signed_distance,
)


def test_union_signed_distance_reports_proxy_overcoverage() -> None:
    native_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    proxy = trimesh.creation.icosphere(subdivisions=2, radius=1.1)
    signed, winners = union_signed_distance(native_points, [proxy])
    assert np.all(signed > 0.09)
    assert np.array_equal(winners, np.zeros(2, dtype=int))


def test_union_signed_distance_reports_proxy_undercoverage() -> None:
    native_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    proxy = trimesh.creation.icosphere(subdivisions=2, radius=0.9)
    signed, _ = union_signed_distance(native_points, [proxy])
    assert np.all(signed < -0.09)


def test_distance_class_uses_frozen_native_tolerance() -> None:
    tolerance = 5e-5
    assert distance_class(-5.1e-5, tolerance) == "PENETRATION"
    assert distance_class(-5.0e-5, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert distance_class(5.0e-5, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert distance_class(5.1e-5, tolerance) == "CLEARANCE"
