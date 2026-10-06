from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from egoengine_repro.evaluation.taco_calibration_residual import (
    SurfaceObservation,
    TriangleSurface,
    apply_transform,
    fit_surface_correction,
    invert_transform,
    surface_residuals,
    transform_from_parameters,
)


def _mesh_and_surface():
    vertices = np.array([
        [0.00, 0.00, 0.00], [0.11, 0.00, 0.00], [0.01, 0.07, 0.00],
        [0.02, 0.01, 0.13],
    ])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    assert mesh.is_watertight
    return mesh, TriangleSurface(mesh.vertices, mesh.faces)


def _surface_points(mesh: trimesh.Trimesh) -> np.ndarray:
    # Fixed barycentric points avoid random sampling and include all faces.
    weights = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    return np.concatenate([
        weights @ mesh.vertices[face] for face in np.asarray(mesh.faces)
    ])


def _pose(translation, rotvec) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def _observations(model: str, correction: np.ndarray, camera_indices=(0, 1, 2, 3)):
    mesh, surface = _mesh_and_surface()
    local = _surface_points(mesh)
    rows = []
    for index in camera_indices:
        object_to_world = _pose(
            [0.25 + 0.01 * index, -0.04 + 0.005 * index, 0.7],
            [0.03 * index, -0.02 * index, 0.015 * index],
        )
        world_to_camera = _pose(
            [0.02 * index, -0.01 * index, 0.12],
            [-0.04 * index, 0.025 * index, -0.01 * index],
        )
        true_world = apply_transform(local, object_to_world)
        if model == "WORLD_FIXED":
            raw_world = apply_transform(true_world, invert_transform(correction))
            raw_camera = apply_transform(raw_world, world_to_camera)
        else:
            true_camera = apply_transform(true_world, world_to_camera)
            raw_camera = apply_transform(true_camera, invert_transform(correction))
        rows.append(SurfaceObservation(
            raw_camera, world_to_camera, object_to_world, sequence="synthetic", frame=index,
        ))
    return mesh, surface, rows


@pytest.mark.parametrize("model", ["WORLD_FIXED", "CAMERA_LOCAL"])
def test_recovers_synthetic_correction_across_unseen_camera_pose(model):
    expected = transform_from_parameters([0.006, -0.004, 0.008, 0.015, -0.012, 0.010])
    _, surface, train = _observations(model, expected, camera_indices=(0, 1, 2))
    _, _, validation = _observations(model, expected, camera_indices=(4, 5))
    fit = fit_surface_correction(
        train, surface, model, robust_scale_m=0.002,
        maximum_function_evaluations=120, minimum_points=24,
    )
    recovered = np.asarray(fit["transform"])
    before = np.median(np.abs(surface_residuals(validation, surface, np.eye(4), model)))
    after = np.median(np.abs(surface_residuals(validation, surface, recovered, model)))
    assert after < 2e-4
    assert after < before * 0.05


@pytest.mark.parametrize("model", ["WORLD_FIXED", "CAMERA_LOCAL"])
def test_identity_data_returns_near_identity(model):
    _, surface, rows = _observations(model, np.eye(4))
    fit = fit_surface_correction(
        rows, surface, model, robust_scale_m=0.002,
        maximum_function_evaluations=80, minimum_points=24,
    )
    assert fit["translation_norm_m"] < 2e-4
    assert fit["rotation_angle_deg"] < 0.2


def test_degenerate_points_fail_closed():
    mesh, surface = _mesh_and_surface()
    points = np.column_stack([np.linspace(0, 0.1, 20), np.zeros(20), np.ones(20)])
    rows = [
        SurfaceObservation(points, np.eye(4), np.eye(4), frame=index)
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="degenerate"):
        fit_surface_correction(rows, surface, "WORLD_FIXED", minimum_points=24)


def test_too_few_observations_fail_closed():
    mesh, surface = _mesh_and_surface()
    points = _surface_points(mesh)
    row = SurfaceObservation(points, np.eye(4), np.eye(4))
    with pytest.raises(ValueError, match="two camera observations"):
        fit_surface_correction([row], surface, "CAMERA_LOCAL", minimum_points=4)
