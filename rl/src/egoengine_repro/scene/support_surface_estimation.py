"""Generic metric-depth support-plane estimation primitives.

Planes follow :mod:`support_surface`: ``normal dot point = offset``.  This
module contains no task/sample identifiers and never reads object bottoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from egoengine_repro.scene.support_surface import Plane


def _intrinsic(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsic focal lengths must be positive")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9):
        raise ValueError("intrinsic homogeneous row is invalid")
    return matrix


def _rigid(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_to_camera must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("world_to_camera homogeneous row is invalid")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("world_to_camera rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("world_to_camera rotation is not proper")
    return transform


def backproject_metric_depth(
    depth_m: Any,
    intrinsic: Any,
    *,
    selector: Any | None = None,
    spatial_stride_px: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject selected positive metric depth pixels into camera coordinates."""
    depth = np.asarray(depth_m)
    matrix = _intrinsic(intrinsic)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise TypeError("depth_m must be a 2-D floating-point metric array")
    if spatial_stride_px <= 0:
        raise ValueError("spatial_stride_px must be positive")
    valid = np.isfinite(depth) & (depth > 0)
    if selector is not None:
        selected = np.asarray(selector, dtype=bool)
        if selected.shape != depth.shape:
            raise ValueError("selector shape differs from depth")
        valid &= selected
    lattice = np.zeros(depth.shape, dtype=bool)
    lattice[::spatial_stride_px, ::spatial_stride_px] = True
    valid &= lattice
    v, u = np.nonzero(valid)
    if not len(u):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)
    z = depth[v, u].astype(np.float64)
    x = (u.astype(np.float64) - matrix[0, 2]) / matrix[0, 0] * z
    y = (v.astype(np.float64) - matrix[1, 2]) / matrix[1, 1] * z
    return np.column_stack([x, y, z]), np.column_stack([u, v])


def camera_points_to_world(points_camera: Any, world_to_camera: Any) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    transform = _rigid(world_to_camera)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("camera points must be finite Nx3")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points - translation) @ rotation


def interaction_roi_mask(foreground_mask: Any, *, scale: float) -> np.ndarray:
    foreground = np.asarray(foreground_mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("foreground mask must be 2-D")
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("interaction ROI scale must be finite and at least one")
    rows, columns = np.nonzero(foreground)
    if not len(columns):
        raise ValueError("foreground projection is empty")
    center_x = (float(columns.min()) + float(columns.max())) / 2.0
    center_y = (float(rows.min()) + float(rows.max())) / 2.0
    width = (float(columns.max() - columns.min() + 1)) * scale
    height = (float(rows.max() - rows.min() + 1)) * scale
    x0 = max(0, int(np.floor(center_x - width / 2.0)))
    x1 = min(foreground.shape[1], int(np.ceil(center_x + width / 2.0)))
    y0 = max(0, int(np.floor(center_y - height / 2.0)))
    y1 = min(foreground.shape[0], int(np.ceil(center_y + height / 2.0)))
    result = np.zeros_like(foreground)
    result[y0:y1, x0:x1] = True
    return result


def foreground_excluded_background_points(
    depth_m: Any,
    intrinsic: Any,
    world_to_camera: Any,
    foreground_mask: Any,
    *,
    dilation_px: int,
    interaction_roi_scale: float,
    spatial_stride_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    foreground = np.asarray(foreground_mask, dtype=bool)
    depth = np.asarray(depth_m)
    if foreground.shape != depth.shape:
        raise ValueError("foreground mask shape differs from depth")
    if dilation_px < 0:
        raise ValueError("foreground dilation must be nonnegative")
    roi = interaction_roi_mask(foreground, scale=interaction_roi_scale)
    if dilation_px:
        size = 2 * dilation_px + 1
        foreground = cv2.dilate(
            foreground.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
        ) > 0
    valid = np.isfinite(depth) & (depth > 0)
    selector = valid & roi & ~foreground
    camera, _ = backproject_metric_depth(
        depth, intrinsic, selector=selector, spatial_stride_px=spatial_stride_px,
    )
    world = camera_points_to_world(camera, world_to_camera)
    return world, {
        "valid_depth_pixels": int(valid.sum()),
        "foreground_pixels_before_dilation": int(np.asarray(foreground_mask, dtype=bool).sum()),
        "foreground_pixels_after_dilation": int(foreground.sum()),
        "roi_pixels": int(roi.sum()),
        "background_valid_pixels_in_roi": int(selector.sum()),
        "sampled_candidate_points": int(len(world)),
    }


@dataclass(frozen=True)
class PlaneEstimate:
    plane: Plane
    candidate_count: int
    inlier_count: int
    inlier_fraction: float
    tilt_from_world_z_deg: float
    median_absolute_residual_m: float
    p95_absolute_residual_m: float
    xy_bounds_m: np.ndarray
    xy_footprint_area_m2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane.to_dict(),
            "candidate_count": self.candidate_count,
            "inlier_count": self.inlier_count,
            "inlier_fraction": self.inlier_fraction,
            "tilt_from_world_z_deg": self.tilt_from_world_z_deg,
            "median_absolute_residual_m": self.median_absolute_residual_m,
            "p95_absolute_residual_m": self.p95_absolute_residual_m,
            "xy_bounds_m": self.xy_bounds_m.tolist(),
            "xy_footprint_area_m2": self.xy_footprint_area_m2,
        }


def _plane_from_coefficients(coefficients: np.ndarray) -> Plane:
    a, b, c = np.asarray(coefficients, dtype=np.float64)
    raw_normal = np.array([-a, -b, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(raw_normal))
    return Plane(normal=raw_normal / norm, offset=float(c / norm), frame="world")


def _tilt_deg(normal: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))


def fit_horizontal_support_plane(
    points_world: Any,
    *,
    random_seed: int,
    maximum_tilt_from_world_z_deg: float,
    inlier_distance_m: float,
    maximum_iterations: int,
    minimum_candidate_points: int,
) -> PlaneEstimate:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("candidate points must be finite Nx3")
    if len(points) < max(3, minimum_candidate_points):
        raise ValueError("too few support-plane candidate points")
    if maximum_iterations <= 0 or inlier_distance_m <= 0:
        raise ValueError("RANSAC iterations and distance must be positive")
    if not 0 <= maximum_tilt_from_world_z_deg < 90:
        raise ValueError("maximum tilt must be in [0, 90)")
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    rng = np.random.default_rng(random_seed)
    best: tuple[int, float, np.ndarray] | None = None
    for _ in range(maximum_iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        local = design[indices]
        if np.linalg.cond(local) > 1e8:
            continue
        coefficients = np.linalg.solve(local, points[indices, 2])
        plane = _plane_from_coefficients(coefficients)
        if _tilt_deg(plane.normal) > maximum_tilt_from_world_z_deg:
            continue
        residual = np.abs(plane.signed_distance(points))
        inliers = residual <= inlier_distance_m
        score = (int(inliers.sum()), -float(np.median(residual[inliers])) if inliers.any() else -np.inf)
        if best is None or score[:2] > best[:2]:
            best = (score[0], score[1], inliers)
    if best is None or best[0] < 3:
        raise ValueError("no valid near-horizontal plane")
    inliers = best[2]
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[inliers], points[inliers, 2], rcond=None)
        plane = _plane_from_coefficients(coefficients)
        if _tilt_deg(plane.normal) > maximum_tilt_from_world_z_deg:
            raise ValueError("refined plane exceeds maximum tilt")
        updated = np.abs(plane.signed_distance(points)) <= inlier_distance_m
        if np.array_equal(updated, inliers):
            break
        inliers = updated
        if int(inliers.sum()) < 3:
            raise ValueError("refined plane lost all inliers")
    residual = np.abs(plane.signed_distance(points[inliers]))
    bounds = np.stack([points[inliers, :2].min(axis=0), points[inliers, :2].max(axis=0)])
    extent = bounds[1] - bounds[0]
    return PlaneEstimate(
        plane=plane, candidate_count=len(points), inlier_count=int(inliers.sum()),
        inlier_fraction=float(inliers.mean()), tilt_from_world_z_deg=_tilt_deg(plane.normal),
        median_absolute_residual_m=float(np.median(residual)),
        p95_absolute_residual_m=float(np.percentile(residual, 95)),
        xy_bounds_m=bounds, xy_footprint_area_m2=float(extent[0] * extent[1]),
    )


def evaluate_validation_evidence(
    plane: Plane,
    validation_points: Any,
    *,
    independent_table_region_available: bool,
    absolute_position_reference_available: bool,
) -> dict[str, Any]:
    """Describe validation evidence without selecting points by candidate residual.

    Background points can include non-table surfaces.  They are therefore
    reported only as deviations from the candidate plane.  This routine does
    not manufacture an independent table region or an absolute accuracy
    certificate when neither was supplied.
    """
    points = np.asarray(validation_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("validation points must be finite Nx3")
    if not len(points):
        raise ValueError("validation points cannot be empty")
    background = evaluate_plane_on_points(plane, points)
    return {
        "background_relative_to_candidate": background,
        "background_point_count": int(len(points)),
        "background_is_not_asserted_to_be_all_table": True,
        "candidate_residual_filter_applied": False,
        "independent_table_validation": {
            "status": (
                "AVAILABLE_BUT_NOT_IMPLEMENTED"
                if independent_table_region_available
                else "NOT_COMPLETED_NO_INDEPENDENT_TABLE_REGION"
            ),
        },
        "absolute_position_precision": {
            "status": (
                "AVAILABLE_BUT_NOT_IMPLEMENTED"
                if absolute_position_reference_available
                else "NOT_VERIFIED_NO_ABSOLUTE_REFERENCE"
            ),
            "uncertainty_interval_m": None,
        },
    }


def aggregate_support_planes(estimates: Iterable[PlaneEstimate]) -> Plane:
    values = list(estimates)
    if not values:
        raise ValueError("cannot aggregate zero support planes")
    normals = np.stack([value.plane.normal for value in values])
    if np.any(normals[:, 2] <= 0):
        raise ValueError("support-plane normals must face world +Z")
    normal = normals.mean(axis=0)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("support-plane normals cancel")
    normal /= norm
    offset = float(np.median([value.plane.offset for value in values]))
    return Plane(normal=normal, offset=offset, frame="world")


def evaluate_plane_on_points(plane: Plane, points_world: Any) -> dict[str, float | int]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("validation points must be finite Nx3")
    if not len(points):
        raise ValueError("validation points are empty")
    residual = plane.signed_distance(points)
    absolute = np.abs(residual)
    median = float(np.median(residual))
    return {
        "point_count": len(points),
        "signed_residual_median_m": median,
        "signed_residual_mad_m": float(np.median(np.abs(residual - median))),
        "signed_residual_p05_m": float(np.percentile(residual, 5)),
        "signed_residual_p50_m": median,
        "signed_residual_p95_m": float(np.percentile(residual, 95)),
        "absolute_residual_median_m": float(np.median(absolute)),
        "absolute_residual_p95_m": float(np.percentile(absolute, 95)),
        "absolute_residual_maximum_m": float(np.max(absolute)),
    }
