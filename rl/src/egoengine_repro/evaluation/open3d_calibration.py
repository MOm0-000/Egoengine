"""Official Open3D 0.20 single-frame calibration wrappers.

The active Open3D path performs input conversion and exactly one call to
``open3d.pipelines.registration.registration_icp`` per frame and coordinate
expression. Correspondence search, rigid updates, iteration and convergence
are owned by Open3D; this module contains no project ICP loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .taco_calibration_residual import (
    CorrectionModel,
    SurfaceObservation,
    apply_transform,
    invert_transform,
)


Open3DSingleFrameMethod = Literal[
    "OPEN3D_OFFICIAL_SINGLE_FRAME",
    "OPEN3D_HUBER_SINGLE_FRAME",
]


def _open3d() -> Any:
    import open3d as o3d

    if o3d.__version__ != "0.20.0":
        raise RuntimeError(f"Open3D 0.20.0 is required, found {o3d.__version__}")
    return o3d


def _finite_points(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{label} must be finite Nx3")
    if not len(points):
        raise ValueError(f"{label} cannot be empty")
    return points


def _unit_normals(value: Any, point_count: int) -> np.ndarray:
    normals = np.asarray(value, dtype=np.float64)
    if normals.shape != (point_count, 3) or not np.isfinite(normals).all():
        raise ValueError("sampled normals must be finite and match sampled points")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("sampled normals must be nonzero")
    return normals / lengths[:, None]


def _surface_hash(points: np.ndarray, normals: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(normals, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OfficialSampledSurface:
    """Point-and-triangle-normal output of Open3D uniform mesh sampling."""

    points_local: np.ndarray
    normals_local: np.ndarray
    number_of_points: int
    random_seed: int
    sha256: str

    def __post_init__(self) -> None:
        points = _finite_points(self.points_local, "sampled points")
        normals = np.asarray(self.normals_local, dtype=np.float64)
        if normals.shape != (len(points), 3) or not np.isfinite(normals).all():
            raise ValueError("sampled normals must be finite and match sampled points")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1e-12) or not np.allclose(lengths, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("sampled normals must be unit length")
        if self.number_of_points != len(points) or self.number_of_points <= 0:
            raise ValueError("sampled point-count metadata is inconsistent")
        if self.random_seed < 0:
            raise ValueError("sampling seed must be nonnegative")
        expected = _surface_hash(points, normals)
        if self.sha256 != expected:
            raise ValueError("sampled surface hash mismatch")
        object.__setattr__(self, "points_local", points)
        object.__setattr__(self, "normals_local", normals)


def official_uniform_sampled_surface(
    vertices: Any,
    faces: Any,
    *,
    number_of_points: int = 50_000,
    random_seed: int = 20261007,
) -> OfficialSampledSurface:
    """Sample a public mesh with Open3D's official uniform sampler."""
    o3d = _open3d()
    vertices_array = _finite_points(vertices, "vertices")
    faces_array = np.asarray(faces, dtype=np.int32)
    if faces_array.ndim != 2 or faces_array.shape[1] != 3 or not len(faces_array):
        raise ValueError("faces must be nonempty Mx3")
    if faces_array.min() < 0 or faces_array.max() >= len(vertices_array):
        raise ValueError("face index is outside the vertex array")
    if number_of_points <= 0 or random_seed < 0:
        raise ValueError("sampling count and seed must be valid")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_array),
        o3d.utility.Vector3iVector(faces_array),
    )
    o3d.utility.random.seed(int(random_seed))
    sampled = mesh.sample_points_uniformly(
        number_of_points=int(number_of_points),
        use_triangle_normal=True,
    )
    points = np.asarray(sampled.points, dtype=np.float64).copy()
    normals = _unit_normals(np.asarray(sampled.normals, dtype=np.float64), len(points))
    return OfficialSampledSurface(
        points_local=points,
        normals_local=normals,
        number_of_points=len(points),
        random_seed=int(random_seed),
        sha256=_surface_hash(points, normals),
    )


def _cloud(o3d: Any, points: np.ndarray, normals: np.ndarray | None = None) -> Any:
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        result.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return result


def _rotation_angle_deg(transform: np.ndarray) -> float:
    rotation = np.array(transform[:3, :3], dtype=np.float64, copy=True, order="C")
    return float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))
    )


def fit_open3d_single_frame(
    observation: SurfaceObservation,
    sampled_surface: OfficialSampledSurface,
    model: CorrectionModel,
    *,
    method: Open3DSingleFrameMethod,
    maximum_correspondence_distance_m: float = 0.05,
    maximum_iterations: int = 60,
    relative_fitness_tolerance: float = 1e-6,
    relative_rmse_tolerance: float = 1e-6,
    huber_delta_m: float = 0.01,
    minimum_correspondences: int = 24,
    advisory_maximum_translation_m: float = 0.10,
    advisory_maximum_rotation_deg: float = 10.0,
    include_correspondence_set: bool = False,
) -> dict[str, Any]:
    """Run one complete official ``registration_icp`` call for one frame."""
    if not isinstance(observation, SurfaceObservation):
        raise TypeError("observation must be a SurfaceObservation")
    if model not in ("WORLD_FIXED", "CAMERA_LOCAL"):
        raise ValueError(f"unknown correction model: {model}")
    positive = (
        maximum_correspondence_distance_m,
        relative_fitness_tolerance,
        relative_rmse_tolerance,
        huber_delta_m,
        advisory_maximum_translation_m,
        advisory_maximum_rotation_deg,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("ICP parameters must be finite and positive")
    if maximum_iterations <= 0 or minimum_correspondences < 1:
        raise ValueError("iteration and correspondence budgets must be positive")

    o3d = _open3d()
    if model == "WORLD_FIXED":
        source = apply_transform(
            observation.points_camera,
            invert_transform(observation.world_to_camera),
        )
        target_transform = observation.object_to_world
    else:
        source = observation.points_camera
        target_transform = observation.world_to_camera @ observation.object_to_world
    source = _finite_points(source, "source points")
    target = apply_transform(sampled_surface.points_local, target_transform)
    target_normals = sampled_surface.normals_local @ target_transform[:3, :3].T
    target_normals = _unit_normals(target_normals, len(target))

    if method == "OPEN3D_OFFICIAL_SINGLE_FRAME":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        robust_kernel = None
    elif method == "OPEN3D_HUBER_SINGLE_FRAME":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.HuberLoss(k=float(huber_delta_m))
        )
        robust_kernel = {"name": "HuberLoss", "k_m": float(huber_delta_m)}
    else:
        raise ValueError(f"unknown Open3D method: {method}")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=float(relative_fitness_tolerance),
        relative_rmse=float(relative_rmse_tolerance),
        max_iteration=int(maximum_iterations),
    )
    started = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        _cloud(o3d, source),
        _cloud(o3d, target, target_normals),
        float(maximum_correspondence_distance_m),
        np.eye(4, dtype=np.float64),
        estimator,
        criteria,
    )
    elapsed = time.perf_counter() - started
    transform = np.asarray(result.transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Open3D returned a non-finite transformation")
    correspondence_set = np.asarray(result.correspondence_set, dtype=np.int64).reshape(-1, 2)
    correspondence_count = int(len(correspondence_set))
    raw_status = (
        "OUTPUT"
        if correspondence_count >= int(minimum_correspondences)
        else "INSUFFICIENT_CORRESPONDENCES"
    )
    translation_m = float(np.linalg.norm(transform[:3, 3]))
    rotation_deg = _rotation_angle_deg(transform)
    within_advisory_bounds = (
        translation_m <= advisory_maximum_translation_m + 1e-12
        and rotation_deg <= advisory_maximum_rotation_deg + 1e-9
    )
    registration_result = {
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
        "correspondence_count": correspondence_count,
        "converged": "NOT_EXPOSED_BY_OPEN3D",
        "iteration_count": "NOT_EXPOSED_BY_OPEN3D",
    }
    if include_correspondence_set:
        registration_result["correspondence_set"] = correspondence_set.tolist()
    return {
        "model": model,
        "transform": transform.tolist(),
        "translation_norm_m": translation_m,
        "rotation_angle_deg": rotation_deg,
        "raw_solver_status": raw_status,
        "raw_solver_reason": (
            ""
            if raw_status == "OUTPUT"
            else f"{correspondence_count} correspondences < {minimum_correspondences}"
        ),
        "registration_result": registration_result,
        "optimizer": {
            "backend": "open3d.pipelines.registration.registration_icp",
            "open3d_version": o3d.__version__,
            "method": method,
            "official_full_interface_call_count": 1,
            "elapsed_seconds": float(elapsed),
        },
        "fit_contract": {
            "initial_transform": "identity",
            "maximum_correspondence_distance_m": float(maximum_correspondence_distance_m),
            "relative_fitness_tolerance": float(relative_fitness_tolerance),
            "relative_rmse_tolerance": float(relative_rmse_tolerance),
            "maximum_iterations": int(maximum_iterations),
            "minimum_correspondences": int(minimum_correspondences),
            "robust_kernel": robust_kernel,
            "sampled_surface_sha256": sampled_surface.sha256,
            "sampled_surface_point_count": sampled_surface.number_of_points,
            "sampled_surface_seed": sampled_surface.random_seed,
        },
        "safety_advisory": {
            "within_bounds": bool(within_advisory_bounds),
            "maximum_translation_m": float(advisory_maximum_translation_m),
            "maximum_rotation_deg": float(advisory_maximum_rotation_deg),
            "affects_raw_output_status": False,
        },
    }
