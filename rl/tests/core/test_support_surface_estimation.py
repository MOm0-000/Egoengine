from __future__ import annotations

import numpy as np
import pytest

from egoengine_repro.scene.support_surface_estimation import (
    backproject_metric_depth,
    camera_points_to_world,
    evaluate_validation_evidence,
    fit_horizontal_support_plane,
    foreground_excluded_background_points,
)


INTRINSIC = np.array([[80.0, 0.0, 31.5], [0.0, 80.0, 23.5], [0.0, 0.0, 1.0]])


def _fit(points: np.ndarray):
    return fit_horizontal_support_plane(
        points, random_seed=0, maximum_tilt_from_world_z_deg=10.0,
        inlier_distance_m=0.002, maximum_iterations=128, minimum_candidate_points=64,
    )


def _plane_depth(height: float, shape=(48, 64)) -> np.ndarray:
    return np.full(shape, height, dtype=np.float32)


def test_synthetic_flat_table_recovers_height() -> None:
    camera, _ = backproject_metric_depth(_plane_depth(0.55), INTRINSIC, spatial_stride_px=2)
    estimate = _fit(camera_points_to_world(camera, np.eye(4)))
    assert estimate.plane.normal == pytest.approx([0, 0, 1], abs=1e-10)
    assert estimate.plane.offset == pytest.approx(0.55, abs=1e-7)


def test_tilted_table_is_not_forced_horizontal() -> None:
    depth = _plane_depth(0.55)
    camera, _ = backproject_metric_depth(depth, INTRINSIC, spatial_stride_px=2)
    camera[:, 2] = 0.03 * camera[:, 0] - 0.02 * camera[:, 1] + 0.55
    estimate = _fit(camera)
    expected = np.array([-0.03, 0.02, 1.0])
    expected /= np.linalg.norm(expected)
    assert estimate.plane.normal == pytest.approx(expected, abs=1e-7)
    assert estimate.plane.offset == pytest.approx(0.55 / np.linalg.norm([-0.03, 0.02, 1]), abs=1e-7)


def test_camera_motion_roundtrip_recovers_same_world_plane() -> None:
    world = np.column_stack([
        np.linspace(-0.2, 0.2, 100), np.linspace(0.15, -0.15, 100), np.full(100, 0.55),
    ])
    for translation in ([0, 0, 0], [0.1, -0.05, 0.02], [-0.07, 0.03, -0.01]):
        transform = np.eye(4)
        transform[:3, 3] = translation
        camera = world @ transform[:3, :3].T + transform[:3, 3]
        recovered = camera_points_to_world(camera, transform)
        assert recovered == pytest.approx(world, abs=1e-12)


def test_foreground_occlusion_is_excluded_from_fit() -> None:
    depth = _plane_depth(0.55)
    depth[18:30, 25:39] = 0.35
    foreground = np.zeros_like(depth, dtype=bool)
    foreground[18:30, 25:39] = True
    points, counts = foreground_excluded_background_points(
        depth, INTRINSIC, np.eye(4), foreground,
        dilation_px=2, interaction_roi_scale=1.5, spatial_stride_px=1,
    )
    estimate = _fit(points)
    assert counts["foreground_pixels_after_dilation"] > counts["foreground_pixels_before_dilation"]
    assert estimate.plane.offset == pytest.approx(0.55, abs=1e-7)


def test_missing_depth_does_not_require_hole_filling() -> None:
    depth = _plane_depth(0.55)
    rng = np.random.default_rng(0)
    depth[rng.random(depth.shape) < 0.7] = 0.0
    camera, _ = backproject_metric_depth(depth, INTRINSIC)
    estimate = _fit(camera)
    assert estimate.plane.offset == pytest.approx(0.55, abs=1e-7)


def test_invalid_inputs_fail_closed() -> None:
    depth = _plane_depth(0.55)
    bad_intrinsic = INTRINSIC.copy()
    bad_intrinsic[0, 0] = np.nan
    with pytest.raises(ValueError):
        backproject_metric_depth(depth, bad_intrinsic)
    with pytest.raises(ValueError):
        camera_points_to_world(np.zeros((3, 3)), np.eye(3))
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError):
        camera_points_to_world(np.zeros((3, 3)), reflected)
    with pytest.raises(ValueError):
        _fit(np.zeros((10, 3)))
    wall = np.column_stack([np.zeros(100), np.linspace(-1, 1, 100), np.linspace(0, 1, 100)])
    with pytest.raises(ValueError):
        _fit(wall)


def test_validation_keeps_large_deviation_and_does_not_issue_accuracy_certificate() -> None:
    from egoengine_repro.scene.support_surface import Plane

    deviations_m = np.array([0.0, 0.001, 0.002, 0.030])
    points = np.column_stack([np.zeros(4), np.zeros(4), 0.55 + deviations_m])
    result = evaluate_validation_evidence(
        Plane([0, 0, 1], 0.55, "world"), points,
        independent_table_region_available=False,
        absolute_position_reference_available=False,
    )
    metrics = result["background_relative_to_candidate"]
    assert metrics["point_count"] == 4
    assert metrics["absolute_residual_maximum_m"] == pytest.approx(0.030)
    assert result["candidate_residual_filter_applied"] is False
    assert result["background_is_not_asserted_to_be_all_table"] is True
    assert result["independent_table_validation"]["status"] == (
        "NOT_COMPLETED_NO_INDEPENDENT_TABLE_REGION"
    )
    assert result["absolute_position_precision"]["status"] == (
        "NOT_VERIFIED_NO_ABSOLUTE_REFERENCE"
    )
    assert result["absolute_position_precision"]["uncertainty_interval_m"] is None
