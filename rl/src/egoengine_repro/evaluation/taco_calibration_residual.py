"""Generic rigid calibration-residual fitting against a triangle surface.

The implementation is deliberately sample agnostic.  It knows camera-space
points, camera poses, object poses, a triangle mesh and two correction
locations: a fixed world transform or a fixed camera-local transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


CorrectionModel = Literal["WORLD_FIXED", "CAMERA_LOCAL"]


def _rigid(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} rotation is not proper")
    return matrix


def transform_from_parameters(parameters: Any) -> np.ndarray:
    """Return a row-vector-compatible SE(3) matrix from ``[t, rotvec]``."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("calibration parameters must be a finite length-6 vector")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def invert_transform(transform: Any) -> np.ndarray:
    matrix = _rigid(transform, "transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def apply_transform(points: Any, transform: Any) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = _rigid(transform, "transform")
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be finite Nx3")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class SurfaceObservation:
    points_camera: np.ndarray
    world_to_camera: np.ndarray
    object_to_world: np.ndarray
    sequence: str = ""
    frame: int = -1

    def __post_init__(self) -> None:
        points = np.asarray(self.points_camera, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_camera must be finite Nx3")
        if not len(points):
            raise ValueError("surface observation cannot be empty")
        object.__setattr__(self, "points_camera", points)
        object.__setattr__(self, "world_to_camera", _rigid(self.world_to_camera, "world_to_camera"))
        object.__setattr__(self, "object_to_world", _rigid(self.object_to_world, "object_to_world"))


class TriangleSurface:
    """Reusable deterministic signed-distance scene in object-local metres."""

    def __init__(self, vertices: Any, faces: Any) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("vertices must be finite Nx3")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError("faces must be Mx3 with at least four triangles")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index is outside the vertex array")
        import open3d as o3d

        legacy = __import__("trimesh").Trimesh(vertices=vertices, faces=faces, process=False)
        if not legacy.is_watertight or not legacy.is_winding_consistent:
            raise ValueError("calibration surface must be watertight and winding-consistent")
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(
            o3d.core.Tensor(vertices.astype(np.float32)),
            o3d.core.Tensor(faces.astype(np.uint32)),
        )
        self.vertices = vertices
        self.faces = faces
        self._o3d = o3d
        self._scene = scene

    def signed_distance(self, points_local: Any) -> np.ndarray:
        points = np.asarray(points_local, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("query points must be finite Nx3")
        # Open3D is negative inside; reports use positive-inside residuals.
        values = -self._scene.compute_signed_distance(
            self._o3d.core.Tensor(points.astype(np.float32)), nthreads=4, nsamples=11,
        ).numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("surface distance produced non-finite values")
        return values


def corrected_points_object_local(
    observation: SurfaceObservation,
    correction: Any,
    model: CorrectionModel,
) -> np.ndarray:
    correction = _rigid(correction, "correction")
    if model not in ("WORLD_FIXED", "CAMERA_LOCAL"):
        raise ValueError(f"unknown correction model: {model}")
    points_camera = observation.points_camera
    if model == "CAMERA_LOCAL":
        points_camera = apply_transform(points_camera, correction)
    camera_to_world = invert_transform(observation.world_to_camera)
    points_world = apply_transform(points_camera, camera_to_world)
    if model == "WORLD_FIXED":
        points_world = apply_transform(points_world, correction)
    world_to_object = invert_transform(observation.object_to_world)
    return apply_transform(points_world, world_to_object)


def surface_residuals(
    observations: Iterable[SurfaceObservation],
    surface: TriangleSurface,
    correction: Any,
    model: CorrectionModel,
) -> np.ndarray:
    rows = [
        surface.signed_distance(corrected_points_object_local(row, correction, model))
        for row in observations
    ]
    if not rows:
        raise ValueError("at least one observation is required")
    return np.concatenate(rows)


def residual_metrics(residuals: Any) -> dict[str, float | int]:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("residuals must be a nonempty finite vector")
    absolute = np.abs(values)
    return {
        "point_count": int(len(values)),
        "signed_median_m": float(np.median(values)),
        "absolute_median_m": float(np.median(absolute)),
        "absolute_p90_m": float(np.percentile(absolute, 90)),
        "absolute_p95_m": float(np.percentile(absolute, 95)),
    }


def _geometry_guard(observations: list[SurfaceObservation], minimum_points: int) -> None:
    if len(observations) < 2:
        raise ValueError("at least two camera observations are required")
    points = np.concatenate([row.points_camera for row in observations])
    if len(points) < minimum_points:
        raise ValueError("too few calibration surface points")
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[-1] <= max(singular[0], 1e-12) * 1e-5:
        raise ValueError("calibration point geometry is degenerate")


def fit_surface_correction(
    observations: Iterable[SurfaceObservation],
    surface: TriangleSurface,
    model: CorrectionModel,
    *,
    robust_loss: str = "soft_l1",
    robust_scale_m: float = 0.01,
    maximum_translation_m: float = 0.10,
    maximum_rotation_deg: float = 10.0,
    maximum_function_evaluations: int = 80,
    minimum_points: int = 64,
    finite_difference_relative_step: float | None = 1e-3,
) -> dict[str, Any]:
    """Fit one fixed SE(3) correction; safety bounds are not task thresholds."""
    return fit_multi_surface_correction(
        [(list(observations), surface)], model,
        robust_loss=robust_loss,
        robust_scale_m=robust_scale_m,
        maximum_translation_m=maximum_translation_m,
        maximum_rotation_deg=maximum_rotation_deg,
        maximum_function_evaluations=maximum_function_evaluations,
        minimum_points=minimum_points,
        finite_difference_relative_step=finite_difference_relative_step,
    )


def fit_multi_surface_correction(
    datasets: Iterable[tuple[Iterable[SurfaceObservation], TriangleSurface]],
    model: CorrectionModel,
    *,
    robust_loss: str = "soft_l1",
    robust_scale_m: float = 0.01,
    maximum_translation_m: float = 0.10,
    maximum_rotation_deg: float = 10.0,
    maximum_function_evaluations: int = 80,
    minimum_points: int = 64,
    finite_difference_relative_step: float | None = 1e-3,
) -> dict[str, Any]:
    """Fit one correction shared by multiple independently posed surfaces."""
    groups = [(list(rows), surface) for rows, surface in datasets]
    if not groups:
        raise ValueError("at least one calibration surface dataset is required")
    all_rows = [row for rows, _ in groups for row in rows]
    _geometry_guard(all_rows, minimum_points)
    if model not in ("WORLD_FIXED", "CAMERA_LOCAL"):
        raise ValueError(f"unknown correction model: {model}")
    if robust_scale_m <= 0 or maximum_translation_m <= 0 or maximum_rotation_deg <= 0:
        raise ValueError("fit scales and safety bounds must be positive")
    if finite_difference_relative_step is not None and (
        not np.isfinite(finite_difference_relative_step)
        or finite_difference_relative_step <= 0
    ):
        raise ValueError("finite_difference_relative_step must be positive or None")
    rotation_bound = np.deg2rad(maximum_rotation_deg)
    lower = np.array([-maximum_translation_m] * 3 + [-rotation_bound] * 3)
    upper = -lower

    def objective(parameters: np.ndarray) -> np.ndarray:
        correction = transform_from_parameters(parameters)
        return np.concatenate([
            surface_residuals(rows, surface, correction, model)
            for rows, surface in groups
        ])

    # ``TriangleSurface`` uses Open3D's float32 raycasting distance field.  The
    # SciPy default finite-difference perturbation is sized for float64 and can
    # therefore be rounded away inside the distance query.  Keep ``None`` as a
    # reproducible legacy path, but use an explicit step for the corrected
    # implementation.  This changes numerical differentiation only; the
    # residual, robust loss and optimizer family are unchanged.
    difference_options = (
        {} if finite_difference_relative_step is None
        else {"diff_step": finite_difference_relative_step}
    )
    result = least_squares(
        objective,
        np.zeros(6, dtype=np.float64),
        bounds=(lower, upper),
        loss=robust_loss,
        f_scale=robust_scale_m,
        x_scale=np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]),
        max_nfev=maximum_function_evaluations,
        method="trf",
        **difference_options,
    )
    correction = transform_from_parameters(result.x)
    rotation_deg = float(np.degrees(np.linalg.norm(result.x[3:])))
    translation_m = float(np.linalg.norm(result.x[:3]))
    inside_safety_bound = (
        translation_m <= maximum_translation_m + 1e-12
        and rotation_deg <= maximum_rotation_deg + 1e-9
    )
    if not result.success or not inside_safety_bound:
        raise ValueError(
            "calibration fit failed or reached the numerical safety boundary: "
            f"success={result.success}, translation={translation_m}, rotation_deg={rotation_deg}"
        )
    return {
        "model": model,
        "parameters_translation_m_then_rotvec_rad": result.x.tolist(),
        "transform": correction.tolist(),
        "translation_norm_m": translation_m,
        "rotation_angle_deg": rotation_deg,
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "fit_contract": {
            "loss": robust_loss,
            "robust_scale_m": robust_scale_m,
            "maximum_translation_m": maximum_translation_m,
            "maximum_rotation_deg": maximum_rotation_deg,
            "bound_provenance": "NUMERICAL_SAFETY_BOUND",
            "maximum_function_evaluations": maximum_function_evaluations,
            "minimum_points": minimum_points,
            "finite_difference_relative_step": finite_difference_relative_step,
            "distance_field_precision": "float32",
        },
        "fit_metrics": residual_metrics(objective(result.x)),
        "identity_metrics": residual_metrics(objective(np.zeros(6))),
    }
