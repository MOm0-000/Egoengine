from __future__ import annotations

import inspect

import numpy as np
import open3d as o3d
import pytest
import trimesh

from egoengine_repro.evaluation.calibration_known_answer import (
    deterministic_surface_points,
    independent_apply,
    independent_inverse,
    independent_transform,
    transform_error,
)
from egoengine_repro.evaluation.open3d_calibration import (
    fit_open3d_single_frame,
    official_uniform_sampled_surface,
)
from egoengine_repro.evaluation.taco_calibration_residual import (
    SurfaceObservation,
    TriangleSurface,
    fit_single_surface_correction,
    fit_surface_correction,
)


def _mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [0.00, 0.00, 0.00],
        [0.11, 0.00, 0.00],
        [0.01, 0.07, 0.00],
        [0.02, 0.01, 0.13],
    ])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _cloud(points: np.ndarray, normals: np.ndarray | None = None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def _observation(source: np.ndarray) -> SurfaceObservation:
    return SurfaceObservation(source, np.eye(4), np.eye(4), sequence="public", frame=0)


def _sorted_pairs(value) -> np.ndarray:
    pairs = np.asarray(value, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return pairs
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def test_requires_frozen_open3d_version():
    assert o3d.__version__ == "0.20.0"


def test_official_uniform_sampling_is_seeded_and_uses_triangle_normals():
    mesh = _mesh()
    first = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=4096, random_seed=314159,
    )
    second = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=4096, random_seed=314159,
    )
    third = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=4096, random_seed=314160,
    )
    assert np.array_equal(first.points_local, second.points_local)
    assert np.array_equal(first.normals_local, second.normals_local)
    assert first.sha256 == second.sha256
    assert first.sha256 != third.sha256
    face_normals = np.asarray(mesh.face_normals)
    for normal in first.normals_local[::127]:
        assert np.max(face_normals @ normal) > 1.0 - 1e-12


@pytest.mark.parametrize(
    "method,estimator",
    [
        (
            "OPEN3D_OFFICIAL_SINGLE_FRAME",
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        ),
        (
            "OPEN3D_HUBER_SINGLE_FRAME",
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=0.01)
            ),
        ),
    ],
)
def test_wrapper_equals_direct_official_registration_icp(method, estimator):
    mesh = _mesh()
    sampled = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=5000, random_seed=271828,
    )
    expected = independent_transform([0.004, -0.003, 0.005], [0.012, -0.008, 0.006])
    source = independent_apply(sampled.points_local[::5], independent_inverse(expected))
    wrapped = fit_open3d_single_frame(
        _observation(source), sampled, "CAMERA_LOCAL", method=method,
        maximum_correspondence_distance_m=0.05, maximum_iterations=60,
        relative_fitness_tolerance=1e-6, relative_rmse_tolerance=1e-6,
        huber_delta_m=0.01, minimum_correspondences=24,
        include_correspondence_set=True,
    )
    direct = o3d.pipelines.registration.registration_icp(
        _cloud(source),
        _cloud(sampled.points_local, sampled.normals_local),
        0.05,
        np.eye(4),
        estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=60,
        ),
    )
    # Open3D's parallel reduction can move the last few float64 bits between
    # otherwise identical calls.  This is an implementation-equivalence
    # tolerance, not an exam accuracy threshold.
    assert np.allclose(
        np.asarray(wrapped["transform"]), direct.transformation,
        rtol=0.0, atol=1e-14,
    )
    assert wrapped["registration_result"]["fitness"] == direct.fitness
    assert np.isclose(
        wrapped["registration_result"]["inlier_rmse_m"], direct.inlier_rmse,
        rtol=0.0, atol=1e-14,
    )
    assert np.array_equal(
        _sorted_pairs(wrapped["registration_result"]["correspondence_set"]),
        _sorted_pairs(direct.correspondence_set),
    )
    assert wrapped["optimizer"]["official_full_interface_call_count"] == 1


def test_zero_correspondence_is_not_reported_as_successful_identity():
    mesh = _mesh()
    sampled = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=1000, random_seed=42,
    )
    result = fit_open3d_single_frame(
        _observation(sampled.points_local[:100] + 5.0),
        sampled,
        "CAMERA_LOCAL",
        method="OPEN3D_OFFICIAL_SINGLE_FRAME",
        maximum_correspondence_distance_m=0.01,
        minimum_correspondences=24,
    )
    assert result["raw_solver_status"] == "INSUFFICIENT_CORRESPONDENCES"
    assert result["registration_result"]["correspondence_count"] == 0
    assert np.array_equal(np.asarray(result["transform"]), np.eye(4))


@pytest.mark.parametrize("scenario", ["complex_zero", "independent_sampling", "noise"])
def test_official_wrapper_matches_direct_on_public_stress_cases(scenario):
    rng = np.random.default_rng(20261007)
    mesh = trimesh.convex.convex_hull(
        rng.uniform(-0.08, 0.08, size=(36, 3))
        + rng.normal(0.0, [0.018, 0.011, 0.025], size=(36, 3))
    )
    sampled = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=8000, random_seed=8080,
    )
    if scenario == "complex_zero":
        source = sampled.points_local[::7].copy()
    else:
        o3d.utility.random.seed(9090)
        source_cloud = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        ).sample_points_uniformly(number_of_points=1200, use_triangle_normal=True)
        source = np.asarray(source_cloud.points).copy()
        transform = independent_transform(
            [0.003, -0.002, 0.004], [0.009, -0.006, 0.004],
        )
        source = independent_apply(source, independent_inverse(transform))
        if scenario == "noise":
            source += rng.normal(0.0, 0.0003, size=source.shape)
    wrapped = fit_open3d_single_frame(
        _observation(source), sampled, "CAMERA_LOCAL",
        method="OPEN3D_OFFICIAL_SINGLE_FRAME",
        maximum_correspondence_distance_m=0.05,
        maximum_iterations=60,
        minimum_correspondences=24,
        include_correspondence_set=True,
    )
    direct = o3d.pipelines.registration.registration_icp(
        _cloud(source),
        _cloud(sampled.points_local, sampled.normals_local),
        0.05,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60,
        ),
    )
    assert np.allclose(
        np.asarray(wrapped["transform"]), direct.transformation,
        rtol=0.0, atol=1e-13,
    )
    assert wrapped["registration_result"]["fitness"] == direct.fitness
    assert np.isclose(
        wrapped["registration_result"]["inlier_rmse_m"], direct.inlier_rmse,
        rtol=0.0, atol=1e-13,
    )
    assert np.array_equal(
        _sorted_pairs(wrapped["registration_result"]["correspondence_set"]),
        _sorted_pairs(direct.correspondence_set),
    )


def test_current_single_frame_entry_changes_only_observation_count_contract():
    mesh = _mesh()
    points = deterministic_surface_points(mesh, points_per_face=3)
    expected = independent_transform([0.002, -0.001, 0.003], [0.006, -0.004, 0.003])
    observation = _observation(independent_apply(points, independent_inverse(expected)))
    surface = TriangleSurface(mesh.vertices, mesh.faces)
    with pytest.raises(ValueError, match="at least two camera observations"):
        fit_surface_correction(
            [observation], surface, "CAMERA_LOCAL", minimum_points=12,
            maximum_function_evaluations=120,
        )
    result = fit_single_surface_correction(
        observation, surface, "CAMERA_LOCAL", minimum_points=12,
        maximum_function_evaluations=120,
    )
    error = transform_error(np.asarray(result["transform"]), expected)
    assert error["translation_error_mm"] < 0.1
    assert error["rotation_error_deg"] < 0.1
    assert result["fit_contract"]["loss"] == "soft_l1"
    assert result["fit_contract"]["finite_difference_relative_step"] == 0.001


def test_frame_order_and_neighbor_content_cannot_change_single_frame_answer():
    mesh = _mesh()
    sampled = official_uniform_sampled_surface(
        mesh.vertices, mesh.faces, number_of_points=4000, random_seed=7,
    )
    corrections = [
        independent_transform([0.002, 0.0, 0.0], [0.0, 0.004, 0.0]),
        independent_transform([-0.003, 0.001, 0.0], [0.002, 0.0, -0.003]),
        independent_transform([0.0, -0.002, 0.004], [-0.004, 0.001, 0.0]),
    ]
    observations = [
        _observation(independent_apply(sampled.points_local[::4], independent_inverse(row)))
        for row in corrections
    ]

    def solve(order):
        return {
            index: fit_open3d_single_frame(
                observations[index], sampled, "CAMERA_LOCAL",
                method="OPEN3D_OFFICIAL_SINGLE_FRAME", minimum_correspondences=24,
            )
            for index in order
        }

    forward = solve([0, 1, 2])
    reverse = solve([2, 1, 0])
    for index in range(3):
        assert np.allclose(
            np.asarray(forward[index]["transform"]),
            np.asarray(reverse[index]["transform"]),
            rtol=0.0, atol=1e-14,
        )


def test_active_source_contains_no_project_icp_loop_or_update_api():
    source = inspect.getsource(
        __import__(
            "egoengine_repro.evaluation.open3d_calibration",
            fromlist=["fit_open3d_single_frame"],
        )
    )
    assert "grouped_point_to_plane_icp" not in source
    assert "compute_transformation" not in source
    assert "KDTreeFlann" not in source
    assert "registration_icp(" in source
