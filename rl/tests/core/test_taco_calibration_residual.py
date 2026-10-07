from __future__ import annotations

import numpy as np
import pytest
import trimesh
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from egoengine_repro.evaluation.taco_calibration_residual import (
    CalibrationCandidateRejected, SurfaceObservation, TriangleSurface,
    fit_multi_surface_correction, fit_surface_correction, surface_residuals,
)
import egoengine_repro.evaluation.taco_calibration_residual as residual_module
from egoengine_repro.evaluation.calibration_known_answer import (
    independent_apply, independent_inverse, independent_transform, transform_error,
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
    return independent_transform(translation, rotvec)


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
        true_world = independent_apply(local, object_to_world)
        if model == "WORLD_FIXED":
            raw_world = independent_apply(true_world, independent_inverse(correction))
            raw_camera = independent_apply(raw_world, world_to_camera)
        else:
            true_camera = independent_apply(true_world, world_to_camera)
            raw_camera = independent_apply(true_camera, independent_inverse(correction))
        rows.append(SurfaceObservation(
            raw_camera, world_to_camera, object_to_world, sequence="synthetic", frame=index,
        ))
    return mesh, surface, rows


@pytest.mark.parametrize("model", ["WORLD_FIXED", "CAMERA_LOCAL"])
def test_recovers_synthetic_correction_across_unseen_camera_pose(model):
    expected = independent_transform([0.006, -0.004, 0.008], [0.015, -0.012, 0.010])
    _, surface, train = _observations(model, expected, camera_indices=(0, 1, 2))
    _, _, validation = _observations(model, expected, camera_indices=(4, 5))
    fit = fit_surface_correction(
        train, surface, model, robust_scale_m=0.002,
        maximum_function_evaluations=120, minimum_points=24,
    )
    recovered = np.asarray(fit["transform"])
    direct = transform_error(recovered, expected)
    before = np.median(np.abs(surface_residuals(validation, surface, np.eye(4), model)))
    after = np.median(np.abs(surface_residuals(validation, surface, recovered, model)))
    assert after < 2e-4
    assert after < before * 0.05
    assert direct["translation_error_mm"] < 0.1
    assert direct["rotation_error_deg"] < 0.01


@pytest.mark.parametrize("model", ["WORLD_FIXED", "CAMERA_LOCAL"])
def test_explicit_float32_finite_difference_step_repairs_known_answer(model):
    expected = independent_transform([0.006, -0.004, 0.008], [0.015, -0.012, 0.010])
    _, surface, rows = _observations(model, expected, camera_indices=(0, 1, 2, 3, 4, 5))
    legacy = fit_surface_correction(
        rows, surface, model, robust_scale_m=0.002,
        maximum_function_evaluations=240, minimum_points=24,
        finite_difference_relative_step=None,
    )
    fixed = fit_surface_correction(
        rows, surface, model, robust_scale_m=0.002,
        maximum_function_evaluations=240, minimum_points=24,
        finite_difference_relative_step=0.001,
    )
    legacy_error = transform_error(np.asarray(legacy["transform"]), expected)
    fixed_error = transform_error(np.asarray(fixed["transform"]), expected)
    assert fixed_error["translation_error_mm"] < legacy_error["translation_error_mm"] * 0.01
    assert fixed_error["rotation_error_deg"] < legacy_error["rotation_error_deg"]
    assert fixed_error["rotation_error_deg"] < 0.01


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


class _LinearSurface:
    def signed_distance(self, points):
        return np.asarray(points, dtype=np.float64)[:, 2]


def _candidate_rows():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(32, 3))
    return [
        SurfaceObservation(points, np.eye(4), np.eye(4), frame=0),
        SurfaceObservation(points + [0.02, 0.01, -0.03], np.eye(4), np.eye(4), frame=1),
    ]


def _solver_result(parameters, *, success=True, status=1, message="ok"):
    return SimpleNamespace(
        x=np.asarray(parameters, dtype=np.float64), success=success, status=status,
        message=message, nfev=7, cost=1.25, optimality=0.5,
    )


@pytest.mark.parametrize(
    "solver_result, expected_status",
    [
        (_solver_result([0.01, 0, 0, 0, 0, 0], success=False, status=0, message="budget"),
         "SOLVER_NOT_CONVERGED"),
        # Each translation component is within the +/-0.10 m search box, but
        # the vector norm exceeds the separate 0.10 m acceptance sphere.
        (_solver_result([0.08, 0.08, 0, 0, 0, 0]),
         "CANDIDATE_ACCEPTANCE_BOUND_EXCEEDED"),
    ],
)
def test_rejected_solver_candidate_preserves_full_result_and_reason(
    monkeypatch, solver_result, expected_status,
) -> None:
    monkeypatch.setattr(residual_module, "least_squares", lambda *_args, **_kwargs: solver_result)
    with pytest.raises(CalibrationCandidateRejected) as captured:
        fit_multi_surface_correction(
            [(_candidate_rows(), _LinearSurface())], "WORLD_FIXED", minimum_points=16,
        )
    rejected = captured.value
    candidate = rejected.candidate
    assert rejected.reason_code == expected_status
    assert candidate["candidate_status"] == expected_status
    assert candidate["accepted_for_use"] is False
    assert candidate["parameters_translation_m_then_rotvec_rad"] == pytest.approx(solver_result.x)
    assert np.asarray(candidate["transform"]).shape == (4, 4)
    assert candidate["optimizer"]["success"] is bool(solver_result.success)
    assert candidate["optimizer"]["status"] == solver_result.status
    assert candidate["optimizer"]["message"] == solver_result.message
    assert candidate["fit_contract"]["component_search_bounds"]["translation_each_axis_m"] == [-0.1, 0.1]
    assert candidate["fit_contract"]["candidate_acceptance_bounds"]["translation_vector_norm_m"] == 0.1


def test_accepted_solver_candidate_is_the_only_formally_usable_result(monkeypatch) -> None:
    solver_result = _solver_result([0.01, -0.02, 0.03, 0.01, 0, 0])
    monkeypatch.setattr(residual_module, "least_squares", lambda *_args, **_kwargs: solver_result)
    candidate = fit_multi_surface_correction(
        [(_candidate_rows(), _LinearSurface())], "WORLD_FIXED", minimum_points=16,
    )
    assert candidate["candidate_status"] == "ACCEPTED_FOR_USE"
    assert candidate["accepted_for_use"] is True
