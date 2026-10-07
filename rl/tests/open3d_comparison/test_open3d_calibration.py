from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
import trimesh

from egoengine_repro.evaluation.calibration_known_answer import (
    independent_apply,
    independent_inverse,
    independent_transform,
    transform_error,
)
from egoengine_repro.evaluation.open3d_calibration import (
    deterministic_sampled_surface,
    fit_open3d_surface_correction,
    grouped_point_to_plane_icp,
)
from egoengine_repro.evaluation.taco_calibration_residual import SurfaceObservation


def _mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [0.00, 0.00, 0.00], [0.11, 0.00, 0.00], [0.01, 0.07, 0.00],
        [0.02, 0.01, 0.13],
    ])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _cloud(points: np.ndarray, normals: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def test_requires_frozen_open3d_version():
    assert o3d.__version__ == "0.20.0"


@pytest.mark.parametrize(
    "method,estimator",
    [
        (
            "OPEN3D_POINT_TO_PLANE",
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        ),
        (
            "OPEN3D_ROBUST_POINT_TO_PLANE",
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=0.01)
            ),
        ),
    ],
)
def test_single_group_adapter_matches_official_registration_icp(method, estimator):
    sampled = deterministic_sampled_surface(_mesh().vertices, _mesh().faces)
    expected = independent_transform([0.004, -0.003, 0.005], [0.012, -0.008, 0.006])
    source = independent_apply(sampled.points_local, independent_inverse(expected))
    grouped = grouped_point_to_plane_icp(
        [source], [sampled.points_local], [sampled.normals_local], method=method,
        maximum_correspondence_distance_m=0.05, maximum_iterations=40,
        relative_fitness_tolerance=1e-6, relative_rmse_tolerance=1e-6,
        huber_delta_m=0.01, minimum_correspondences=6,
        maximum_translation_m=0.10, maximum_rotation_deg=10.0,
    )
    official = o3d.pipelines.registration.registration_icp(
        _cloud(source), _cloud(sampled.points_local, sampled.normals_local), 0.05,
        np.eye(4), estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=40,
        ),
    )
    assert np.allclose(np.asarray(grouped["transform"]), official.transformation, atol=1e-10)
    assert grouped["optimizer"]["cross_group_correspondence_count"] == 0


@pytest.mark.parametrize("model", ["WORLD_FIXED", "CAMERA_LOCAL"])
@pytest.mark.parametrize(
    "method", ["OPEN3D_POINT_TO_PLANE", "OPEN3D_ROBUST_POINT_TO_PLANE"]
)
def test_recovers_one_common_transform_across_frames(model, method):
    mesh = _mesh()
    sampled = deterministic_sampled_surface(mesh.vertices, mesh.faces)
    expected = independent_transform([0.004, -0.003, 0.005], [0.012, -0.008, 0.006])
    rows = []
    for index in range(4):
        object_to_world = independent_transform(
            [0.2 + 0.02 * index, -0.04 + 0.01 * index, 0.75],
            [0.02 * index, -0.015 * index, 0.01 * index],
        )
        world_to_camera = independent_transform(
            [-0.01 + 0.01 * index, 0.02 - 0.005 * index, 0.08],
            [-0.03 * index, 0.02 * index, -0.01 * index],
        )
        true_world = independent_apply(sampled.points_local, object_to_world)
        if model == "WORLD_FIXED":
            raw_world = independent_apply(true_world, independent_inverse(expected))
            observed = independent_apply(raw_world, world_to_camera)
        else:
            true_camera = independent_apply(true_world, world_to_camera)
            observed = independent_apply(true_camera, independent_inverse(expected))
        rows.append(SurfaceObservation(observed, world_to_camera, object_to_world, frame=index))
    fit = fit_open3d_surface_correction(
        rows, sampled, model, method=method, minimum_correspondences=12,
    )
    error = transform_error(np.asarray(fit["transform"]), expected)
    assert error["translation_error_mm"] < 1e-6
    assert error["rotation_error_deg"] < 1e-5
    assert all(
        row["cross_group_correspondence_count"] == 0
        for row in fit["optimizer"]["iteration_trace"]
    )


def test_no_correspondence_fails_closed_instead_of_returning_identity():
    sampled = deterministic_sampled_surface(_mesh().vertices, _mesh().faces)
    with pytest.raises(ValueError, match="too few grouped correspondences"):
        grouped_point_to_plane_icp(
            [sampled.points_local + 5.0], [sampled.points_local], [sampled.normals_local],
            method="OPEN3D_POINT_TO_PLANE", maximum_correspondence_distance_m=0.01,
            maximum_iterations=10, relative_fitness_tolerance=1e-6,
            relative_rmse_tolerance=1e-6, huber_delta_m=0.01,
            minimum_correspondences=6, maximum_translation_m=0.1,
            maximum_rotation_deg=10.0,
        )


def test_deterministic_sampling_is_independent_of_observations():
    first = deterministic_sampled_surface(_mesh().vertices, _mesh().faces)
    second = deterministic_sampled_surface(_mesh().vertices.copy(), _mesh().faces.copy())
    assert np.array_equal(first.points_local, second.points_local)
    assert np.array_equal(first.normals_local, second.normals_local)
    assert np.array_equal(first.sampled_face_indices, second.sampled_face_indices)
