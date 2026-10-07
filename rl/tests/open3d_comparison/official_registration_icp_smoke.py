#!/usr/bin/env python3
"""Standalone official Open3D ICP smoke; imports no project solver module."""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d


if o3d.__version__ != "0.20.0":
    raise RuntimeError(f"expected Open3D 0.20.0, found {o3d.__version__}")

vertices = np.asarray([
    [0.00, 0.00, 0.00],
    [0.11, 0.00, 0.00],
    [0.01, 0.07, 0.00],
    [0.02, 0.01, 0.13],
], dtype=np.float64)
faces = np.asarray([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
], dtype=np.int32)
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(faces),
)
o3d.utility.random.seed(20261007)
target = mesh.sample_points_uniformly(
    number_of_points=5000,
    use_triangle_normal=True,
)
transform = np.eye(4)
transform[:3, 3] = [0.004, -0.003, 0.005]
source_points = np.asarray(target.points)[::5] - transform[:3, 3]
source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
result = o3d.pipelines.registration.registration_icp(
    source,
    target,
    0.05,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=60,
    ),
)
print(json.dumps({
    "open3d_version": o3d.__version__,
    "registration_icp_calls": 1,
    "fitness": result.fitness,
    "inlier_rmse_m": result.inlier_rmse,
    "correspondence_count": len(result.correspondence_set),
    "translation_m": np.asarray(result.transformation)[:3, 3].tolist(),
}, sort_keys=True))
