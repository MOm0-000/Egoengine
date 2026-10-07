"""Open3D 0.20 point-to-plane calibration with grouped correspondences.

The only project-specific part is correspondence grouping: measurements from a
frame may query only the sampled public model surface for that same frame.  The
rigid update itself is computed by Open3D's official
``TransformationEstimationPointToPlane.compute_transformation`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .taco_calibration_residual import (
    CorrectionModel,
    SurfaceObservation,
    apply_transform,
    invert_transform,
)


Open3DMethod = Literal["OPEN3D_POINT_TO_PLANE", "OPEN3D_ROBUST_POINT_TO_PLANE"]


@dataclass(frozen=True)
class SampledSurface:
    """A deterministic point-and-normal representation of a public mesh."""

    points_local: np.ndarray
    normals_local: np.ndarray
    sampled_face_indices: np.ndarray
    points_per_face: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_local, dtype=np.float64)
        normals = np.asarray(self.normals_local, dtype=np.float64)
        indices = np.asarray(self.sampled_face_indices, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("sampled surface points must be finite Nx3")
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError("sampled surface normals must match points")
        lengths = np.linalg.norm(normals, axis=1)
        if not len(points) or not np.allclose(lengths, 1.0, atol=1e-8):
            raise ValueError("sampled surface normals must be unit length")
        if indices.ndim != 1 or len(indices) * int(self.points_per_face) != len(points):
            raise ValueError("sampled face metadata is inconsistent")
        object.__setattr__(self, "points_local", points)
        object.__setattr__(self, "normals_local", normals)
        object.__setattr__(self, "sampled_face_indices", indices)


def deterministic_sampled_surface(
    vertices: Any,
    faces: Any,
    *,
    maximum_faces: int = 4096,
    points_per_face: int = 3,
) -> SampledSurface:
    """Sample public triangles without randomness or access to observations."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be nonempty Mx3")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index is outside the vertex array")
    if maximum_faces <= 0 or points_per_face not in (1, 3):
        raise ValueError("invalid deterministic sampling contract")
    if len(faces) > maximum_faces:
        chosen = np.unique(
            np.rint(np.linspace(0, len(faces) - 1, maximum_faces)).astype(np.int64)
        )
    else:
        chosen = np.arange(len(faces), dtype=np.int64)
    triangles = vertices[faces[chosen]]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    face_normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(face_normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("sampled mesh contains a degenerate triangle")
    face_normals = face_normals / lengths[:, None]
    weights = (
        np.asarray([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
        if points_per_face == 1
        else np.asarray(
            [[0.60, 0.20, 0.20], [0.20, 0.60, 0.20], [0.20, 0.20, 0.60]],
            dtype=np.float64,
        )
    )
    points = np.concatenate([weights @ triangle for triangle in triangles])
    normals = np.repeat(face_normals, len(weights), axis=0)
    return SampledSurface(points, normals, chosen, points_per_face)


def _open3d() -> Any:
    import open3d as o3d

    if o3d.__version__ != "0.20.0":
        raise RuntimeError(f"Open3D 0.20.0 is required, found {o3d.__version__}")
    return o3d


def _pcd(o3d: Any, points: np.ndarray, normals: np.ndarray | None = None) -> Any:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return cloud


def _rotation_angle(transform: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(transform[:3, :3]).as_rotvec()))


def grouped_point_to_plane_icp(
    source_groups: Iterable[np.ndarray],
    target_groups: Iterable[np.ndarray],
    target_normal_groups: Iterable[np.ndarray],
    *,
    method: Open3DMethod,
    maximum_correspondence_distance_m: float,
    maximum_iterations: int,
    relative_fitness_tolerance: float,
    relative_rmse_tolerance: float,
    huber_delta_m: float,
    minimum_correspondences: int,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> dict[str, Any]:
    """Run grouped ICP while delegating every SE(3) update to Open3D."""
    o3d = _open3d()
    sources = [np.asarray(row, dtype=np.float64) for row in source_groups]
    targets = [np.asarray(row, dtype=np.float64) for row in target_groups]
    normals = [np.asarray(row, dtype=np.float64) for row in target_normal_groups]
    if not sources or not (len(sources) == len(targets) == len(normals)):
        raise ValueError("source, target and normal groups must be nonempty and aligned")
    for source, target, normal in zip(sources, targets, normals, strict=True):
        if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
            raise ValueError("source groups must contain finite Nx3 points")
        if target.ndim != 2 or target.shape[1] != 3 or not np.isfinite(target).all():
            raise ValueError("target groups must contain finite Nx3 points")
        if normal.shape != target.shape or not np.isfinite(normal).all():
            raise ValueError("target normal groups must match target points")
        if not len(source) or not len(target):
            raise ValueError("ICP groups cannot be empty")
        if not np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-8):
            raise ValueError("target normals must be unit length")
    positive = (
        maximum_correspondence_distance_m,
        relative_fitness_tolerance,
        relative_rmse_tolerance,
        huber_delta_m,
        maximum_translation_m,
        maximum_rotation_deg,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("ICP scales, tolerances and safety bounds must be positive")
    if maximum_iterations <= 0 or minimum_correspondences < 6:
        raise ValueError("invalid ICP iteration or correspondence budget")
    if method == "OPEN3D_POINT_TO_PLANE":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == "OPEN3D_ROBUST_POINT_TO_PLANE":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.HuberLoss(k=float(huber_delta_m))
        )
    else:
        raise ValueError(f"unknown Open3D method: {method}")

    target_clouds = [_pcd(o3d, target, normal) for target, normal in zip(targets, normals, strict=True)]
    trees = [o3d.geometry.KDTreeFlann(cloud) for cloud in target_clouds]
    target_offsets = np.cumsum([0] + [len(row) for row in targets[:-1]])
    target_all = np.concatenate(targets)
    normal_all = np.concatenate(normals)
    target_all_cloud = _pcd(o3d, target_all, normal_all)
    total_source = sum(len(row) for row in sources)
    distance_sq = float(maximum_correspondence_distance_m) ** 2

    def match(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, list[int]]:
        transformed_groups = [apply_transform(row, transform) for row in sources]
        transformed = np.concatenate(transformed_groups)
        pairs: list[list[int]] = []
        squared: list[float] = []
        per_group: list[int] = []
        source_offset = 0
        for group_index, (points, tree) in enumerate(zip(transformed_groups, trees, strict=True)):
            count = 0
            for local_source, point in enumerate(points):
                found, indices, distances = tree.search_knn_vector_3d(point, 1)
                if found and float(distances[0]) <= distance_sq:
                    pairs.append([
                        source_offset + local_source,
                        int(target_offsets[group_index]) + int(indices[0]),
                    ])
                    squared.append(float(distances[0]))
                    count += 1
            per_group.append(count)
            source_offset += len(points)
        correspondence = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        fitness = len(correspondence) / total_source
        rmse = float(np.sqrt(np.mean(squared))) if squared else float("inf")
        return transformed, correspondence, fitness, rmse, per_group

    started = time.perf_counter()
    transform = np.eye(4, dtype=np.float64)
    transformed, correspondence, fitness, rmse, per_group = match(transform)
    if len(correspondence) < minimum_correspondences:
        raise ValueError(
            f"too few grouped correspondences: {len(correspondence)} < {minimum_correspondences}"
        )
    iterations: list[dict[str, Any]] = []
    converged = False
    for iteration in range(maximum_iterations):
        source_cloud = _pcd(o3d, transformed)
        delta = np.asarray(
            estimator.compute_transformation(
                source_cloud,
                target_all_cloud,
                o3d.utility.Vector2iVector(correspondence),
            ),
            dtype=np.float64,
        )
        if delta.shape != (4, 4) or not np.isfinite(delta).all():
            raise ValueError("Open3D returned a non-finite transformation")
        transform = delta @ transform
        translation = float(np.linalg.norm(transform[:3, 3]))
        rotation_deg = float(np.degrees(_rotation_angle(transform)))
        if translation > maximum_translation_m + 1e-12 or rotation_deg > maximum_rotation_deg + 1e-9:
            raise ValueError(
                "Open3D calibration reached the numerical safety boundary: "
                f"translation={translation}, rotation_deg={rotation_deg}"
            )
        previous_fitness, previous_rmse = fitness, rmse
        transformed, correspondence, fitness, rmse, per_group = match(transform)
        if len(correspondence) < minimum_correspondences:
            raise ValueError(
                f"too few grouped correspondences after update: {len(correspondence)}"
            )
        relative_fitness = abs(fitness - previous_fitness) / max(abs(previous_fitness), 1e-12)
        relative_rmse = abs(rmse - previous_rmse) / max(abs(previous_rmse), 1e-12)
        iterations.append({
            "iteration": iteration + 1,
            "correspondence_count": int(len(correspondence)),
            "correspondences_per_group": per_group,
            "fitness": float(fitness),
            "inlier_rmse_m": float(rmse),
            "increment_translation_m": float(np.linalg.norm(delta[:3, 3])),
            "increment_rotation_deg": float(np.degrees(_rotation_angle(delta))),
            "relative_fitness_change": float(relative_fitness),
            "relative_rmse_change": float(relative_rmse),
            "cross_group_correspondence_count": 0,
        })
        if relative_fitness <= relative_fitness_tolerance and relative_rmse <= relative_rmse_tolerance:
            converged = True
            break
    elapsed = time.perf_counter() - started
    return {
        "transform": transform.tolist(),
        "translation_norm_m": float(np.linalg.norm(transform[:3, 3])),
        "rotation_angle_deg": float(np.degrees(_rotation_angle(transform))),
        "optimizer": {
            "backend": "Open3D TransformationEstimationPointToPlane.compute_transformation",
            "open3d_version": o3d.__version__,
            "method": method,
            "converged": converged,
            "iterations": len(iterations),
            "maximum_iterations": maximum_iterations,
            "correspondence_count": int(len(correspondence)),
            "correspondences_per_group": per_group,
            "fitness": float(fitness),
            "inlier_rmse_m": float(rmse),
            "elapsed_seconds": float(elapsed),
            "cross_group_correspondence_count": 0,
            "iteration_trace": iterations,
        },
        "fit_contract": {
            "adapter": "PROJECT_GROUPED_CORRESPONDENCE_OFFICIAL_OPEN3D_UPDATE",
            "maximum_correspondence_distance_m": maximum_correspondence_distance_m,
            "relative_fitness_tolerance": relative_fitness_tolerance,
            "relative_rmse_tolerance": relative_rmse_tolerance,
            "huber_delta_m": huber_delta_m if method == "OPEN3D_ROBUST_POINT_TO_PLANE" else None,
            "minimum_correspondences": minimum_correspondences,
            "maximum_translation_m": maximum_translation_m,
            "maximum_rotation_deg": maximum_rotation_deg,
            "bound_provenance": "NUMERICAL_SAFETY_BOUND",
        },
    }


def fit_open3d_surface_correction(
    observations: Iterable[SurfaceObservation],
    sampled_surface: SampledSurface,
    model: CorrectionModel,
    *,
    method: Open3DMethod,
    maximum_correspondence_distance_m: float = 0.05,
    maximum_iterations: int = 60,
    relative_fitness_tolerance: float = 1e-6,
    relative_rmse_tolerance: float = 1e-6,
    huber_delta_m: float = 0.01,
    minimum_correspondences: int = 24,
    maximum_translation_m: float = 0.10,
    maximum_rotation_deg: float = 10.0,
) -> dict[str, Any]:
    """Estimate one correction shared by all frames using official Open3D updates."""
    rows = list(observations)
    if len(rows) < 2:
        raise ValueError("at least two camera observations are required")
    points = np.concatenate([row.points_camera for row in rows])
    if len(points) < minimum_correspondences:
        raise ValueError("too few calibration surface points")
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[-1] <= max(singular[0], 1e-12) * 1e-5:
        raise ValueError("calibration point geometry is degenerate")
    if model not in ("WORLD_FIXED", "CAMERA_LOCAL"):
        raise ValueError(f"unknown correction model: {model}")

    source_groups, target_groups, normal_groups = [], [], []
    for row in rows:
        if model == "WORLD_FIXED":
            source = apply_transform(row.points_camera, invert_transform(row.world_to_camera))
            target_transform = row.object_to_world
        else:
            source = row.points_camera
            target_transform = row.world_to_camera @ row.object_to_world
        target = apply_transform(sampled_surface.points_local, target_transform)
        target_normals = sampled_surface.normals_local @ target_transform[:3, :3].T
        target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
        source_groups.append(source)
        target_groups.append(target)
        normal_groups.append(target_normals)
    result = grouped_point_to_plane_icp(
        source_groups,
        target_groups,
        normal_groups,
        method=method,
        maximum_correspondence_distance_m=maximum_correspondence_distance_m,
        maximum_iterations=maximum_iterations,
        relative_fitness_tolerance=relative_fitness_tolerance,
        relative_rmse_tolerance=relative_rmse_tolerance,
        huber_delta_m=huber_delta_m,
        minimum_correspondences=minimum_correspondences,
        maximum_translation_m=maximum_translation_m,
        maximum_rotation_deg=maximum_rotation_deg,
    )
    result["model"] = model
    result["sampled_surface"] = {
        "point_count": int(len(sampled_surface.points_local)),
        "sampled_face_count": int(len(sampled_surface.sampled_face_indices)),
        "points_per_face": sampled_surface.points_per_face,
    }
    return result
