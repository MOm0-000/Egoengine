"""Geometry-only audit of a mesh's lower support region.

The gravity-up axis is used only to select the object's underside.  The
resulting plane is an unconstrained orthogonal fit; it is not forced to be
horizontal and it does not consume a table or support-surface contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CLASSIFICATIONS = {
    "SUPPORT_PLANE_WELL_DEFINED",
    "SUPPORT_REGION_INSUFFICIENT",
    "SUPPORT_GEOMETRY_NOT_PLANAR",
}


def _unit(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return vector / length


def _tangent_basis(up: np.ndarray) -> np.ndarray:
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = seed - up * float(seed @ up)
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    return np.stack([first, second])


def _edge_resolution(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if not len(lengths) or np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("mesh edges must have finite positive length")
    return {
        "unique_edge_count": int(len(edges)),
        "median_edge_length_m": float(np.median(lengths)),
        "p95_edge_length_m": float(np.percentile(lengths, 95)),
    }


def inspect_lower_support_region(
    vertices_world: Any,
    faces: Any,
    *,
    gravity_up: Any = (0.0, 0.0, 1.0),
    angular_sectors: int = 72,
    points_per_sector: int = 3,
) -> dict[str, Any]:
    """Fit and assess an arbitrary plane from a distributed lower envelope.

    Three low vertices per angular sector avoid equating one global minimum
    vertex with a support plane.  Quality gates are relative to mesh edge
    resolution rather than sensor/table accuracy:

    - at least 90% of angular sectors represented;
    - both in-plane spans at least ten median mesh edges and aspect >= 0.25;
    - orthogonal P95 <= 0.20 and max <= 0.25 median mesh edge.
    """
    vertices = np.asarray(vertices_world, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_world must be finite Nx3")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("faces must be nonempty Mx3")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("face index is outside vertices")
    if angular_sectors < 8 or points_per_sector < 1:
        raise ValueError("angular sampling contract is too small")

    up = _unit(gravity_up, "gravity_up")
    basis = _tangent_basis(up)
    projected = vertices @ basis.T
    center_2d = (projected.min(axis=0) + projected.max(axis=0)) / 2.0
    centered_2d = projected - center_2d
    angle = np.arctan2(centered_2d[:, 1], centered_2d[:, 0])
    height = vertices @ up
    selected: list[int] = []
    represented = 0
    width = 2.0 * np.pi / angular_sectors
    # Put the nominal rays at sector centres.  This avoids losing alternating
    # sectors when a regular mesh happens to place vertices exactly on bin
    # boundaries and floating-point roundoff sends both neighbours to one bin.
    sector_index = np.floor((angle + np.pi + 0.5 * width) / width).astype(np.int64)
    sector_index %= angular_sectors
    for sector in range(angular_sectors):
        indices = np.flatnonzero(sector_index == sector)
        if not len(indices):
            continue
        represented += 1
        count = min(points_per_sector, len(indices))
        selected.extend(indices[np.argsort(height[indices])[:count]].tolist())
    selected_indices = np.unique(np.asarray(selected, dtype=np.int64))
    if len(selected_indices) < 3:
        raise ValueError("lower envelope has fewer than three unique points")
    points = vertices[selected_indices]

    centroid = points.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(points - centroid, full_matrices=False)
    normal = axes[-1]
    if float(normal @ up) < 0:
        normal = -normal
    offset = float(normal @ centroid)
    signed = points @ normal - offset
    absolute = np.abs(signed)
    in_plane = (points - centroid) @ axes[:2].T
    spans = np.ptp(in_plane, axis=0)
    spans = np.sort(spans)[::-1]
    radial = np.linalg.norm(in_plane, axis=1)
    edges = _edge_resolution(vertices, triangles)
    median_edge = float(edges["median_edge_length_m"])
    angular_fraction = represented / angular_sectors
    aspect = float(singular_values[1] / singular_values[0]) if singular_values[0] > 0 else 0.0
    p95 = float(np.percentile(absolute, 95))
    maximum = float(absolute.max())
    sufficient = (
        angular_fraction >= 0.90
        and len(selected_indices) >= angular_sectors
        and spans[1] >= 10.0 * median_edge
        and aspect >= 0.25
    )
    planar = p95 <= 0.20 * median_edge and maximum <= 0.25 * median_edge
    if not sufficient:
        classification = "SUPPORT_REGION_INSUFFICIENT"
    elif not planar:
        classification = "SUPPORT_GEOMETRY_NOT_PLANAR"
    else:
        classification = "SUPPORT_PLANE_WELL_DEFINED"

    return {
        "classification": classification,
        "selection": {
            "method": "gravity_lower_envelope_by_angular_sector",
            "gravity_up": up.tolist(),
            "angular_sectors": angular_sectors,
            "represented_sectors": represented,
            "angular_coverage_fraction": float(angular_fraction),
            "points_per_sector": points_per_sector,
            "unique_support_point_count": int(len(selected_indices)),
            "support_vertex_indices": selected_indices.tolist(),
        },
        "plane": {
            "equation": "normal dot x = offset",
            "normal": normal.tolist(),
            "offset_m": offset,
            "fit_constraint": "UNCONSTRAINED_ORTHOGONAL_SVD",
        },
        "coverage": {
            "world_xyz_range_m": np.ptp(points, axis=0).tolist(),
            "in_plane_principal_range_m": spans.tolist(),
            "in_plane_singular_value_aspect": aspect,
            "radial_distance_m": {
                "minimum": float(radial.min()),
                "p05": float(np.percentile(radial, 5)),
                "median": float(np.median(radial)),
                "p95": float(np.percentile(radial, 95)),
                "maximum": float(radial.max()),
            },
        },
        "planarity": {
            "absolute_distance_m": {
                "median": float(np.median(absolute)),
                "p95": p95,
                "maximum": maximum,
            },
            "maximum_below_fitted_plane_m": float(max(0.0, -signed.min())),
            "maximum_above_fitted_plane_m": float(max(0.0, signed.max())),
        },
        "mesh_resolution": edges,
        "decision_gates": {
            "region_sufficient": bool(sufficient),
            "approximately_planar": bool(planar),
            "minimum_angular_coverage_fraction": 0.90,
            "minimum_minor_span_in_median_edges": 10.0,
            "minimum_in_plane_aspect": 0.25,
            "maximum_p95_distance_in_median_edges": 0.20,
            "maximum_distance_in_median_edges": 0.25,
            "observed_p95_distance_in_median_edges": p95 / median_edge,
            "observed_maximum_distance_in_median_edges": maximum / median_edge,
        },
    }
