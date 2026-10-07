"""Independent known-answer fixtures and scoring for calibration audits.

This module must not import transform helpers from ``taco_calibration_residual``.
It constructs truth with an independent column-vector convention, serializes
only whitelisted solver inputs, and keeps ownership labels in private truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
import trimesh


def independent_transform(translation: Any, rotvec: Any) -> np.ndarray:
    """Build SE(3) with Rodrigues' formula, independently of the solver."""
    translation = np.asarray(translation, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if translation.shape != (3,) or rotvec.shape != (3,):
        raise ValueError("translation and rotvec must be length three")
    if not np.isfinite(translation).all() or not np.isfinite(rotvec).all():
        raise ValueError("transform parameters must be finite")
    angle = float(np.linalg.norm(rotvec))
    if angle == 0.0:
        rotation = np.eye(3)
    else:
        axis = rotvec / angle
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def independent_inverse(transform: Any) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def independent_apply(points: Any, transform: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points must be Nx3 and transform must be 4x4")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_error(estimate: Any, truth: Any) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if estimate.shape != (4, 4) or truth.shape != (4, 4):
        raise ValueError("estimate and truth must be 4x4")
    relative = estimate[:3, :3] @ truth[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "translation_error_mm": float(1000.0 * np.linalg.norm(estimate[:3, 3] - truth[:3, 3])),
        "rotation_error_deg": float(np.degrees(np.arccos(cosine))),
    }


def checkpoint_error(estimate: Any, truth: Any, points: Any) -> dict[str, float]:
    first = independent_apply(points, estimate)
    second = independent_apply(points, truth)
    values = 1000.0 * np.linalg.norm(first - second, axis=1)
    return {
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "maximum_mm": float(np.max(values)),
    }


def artificial_asymmetric_mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [-0.080, -0.055, -0.045], [0.095, -0.048, -0.038],
        [0.060, 0.078, -0.050], [-0.070, 0.062, -0.030],
        [-0.050, -0.035, 0.082], [0.072, -0.020, 0.064],
        [0.028, 0.055, 0.105],
    ])
    return trimesh.convex.convex_hull(vertices)


def deterministic_surface_points(mesh: trimesh.Trimesh, points_per_face: int) -> np.ndarray:
    if points_per_face not in (1, 3):
        raise ValueError("points_per_face must be 1 or 3")
    weights = np.array([[1 / 3, 1 / 3, 1 / 3]]) if points_per_face == 1 else np.array([
        [0.60, 0.20, 0.20], [0.20, 0.60, 0.20], [0.20, 0.20, 0.60],
    ])
    faces = np.asarray(mesh.faces)
    # Keep public exercises bounded without selecting faces by fit quality.
    if len(faces) > 512:
        indices = np.unique(np.rint(np.linspace(0, len(faces) - 1, 512)).astype(np.int64))
        faces = faces[indices]
    return np.concatenate([weights @ np.asarray(mesh.vertices)[face] for face in faces])


def _pose(index: int) -> tuple[np.ndarray, np.ndarray]:
    object_to_world = independent_transform(
        [0.22 + 0.017 * index, -0.06 + 0.011 * index, 0.82 + 0.006 * index],
        [0.045 * index, -0.031 * index, 0.019 * index],
    )
    world_to_camera = independent_transform(
        [-0.03 + 0.014 * index, 0.02 - 0.009 * index, 0.09],
        [-0.038 * index, 0.027 * index, -0.016 * index],
    )
    return object_to_world, world_to_camera


def generate_direct_case(
    mesh: trimesh.Trimesh,
    *,
    correction: np.ndarray,
    generation_model: str,
    camera_views: int,
    points_per_face: int,
    rng: np.random.Generator,
    noise_std_m: float = 0.0,
    missing_fraction: float = 0.0,
    outlier_fraction: float = 0.0,
    per_view_corrections: list[np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local = deterministic_surface_points(mesh, points_per_face)
    points, offsets, cameras, objects = [], [0], [], []
    corrections = per_view_corrections or [correction] * camera_views
    if len(corrections) != camera_views:
        raise ValueError("per-view correction count mismatch")
    for index in range(camera_views):
        object_to_world, world_to_camera = _pose(index)
        true_world = independent_apply(local, object_to_world)
        current = corrections[index]
        if generation_model == "WORLD_FIXED":
            raw_world = independent_apply(true_world, independent_inverse(current))
            observed = independent_apply(raw_world, world_to_camera)
        elif generation_model == "CAMERA_LOCAL":
            true_camera = independent_apply(true_world, world_to_camera)
            observed = independent_apply(true_camera, independent_inverse(current))
        else:
            raise ValueError("unknown generation model")
        if noise_std_m:
            observed = observed + rng.normal(0.0, noise_std_m, observed.shape)
        if missing_fraction:
            keep = rng.random(len(observed)) >= missing_fraction
            observed = observed[keep]
        if outlier_fraction and len(observed):
            count = max(1, int(round(outlier_fraction * len(observed))))
            chosen = rng.choice(len(observed), size=min(count, len(observed)), replace=False)
            observed[chosen] += rng.uniform(-0.025, 0.025, size=(len(chosen), 3))
        points.append(observed)
        offsets.append(offsets[-1] + len(observed))
        cameras.append(world_to_camera)
        objects.append(object_to_world)
    public = {
        "case_kind": np.asarray("direct"),
        "vertices": np.asarray(mesh.vertices, dtype=np.float64),
        "faces": np.asarray(mesh.faces, dtype=np.int64),
        "points_camera": np.concatenate(points),
        "observation_offsets": np.asarray(offsets, dtype=np.int64),
        "world_to_camera": np.stack(cameras),
        "object_to_world": np.stack(objects),
    }
    truth = {
        "has_unique_global_answer": per_view_corrections is None,
        "generation_model": generation_model,
        "recovery_transform": correction.tolist() if per_view_corrections is None else None,
        "per_view_recovery_transforms": [row.tolist() for row in corrections],
    }
    return public, truth


def _rays(width: int, height: int, intrinsic: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    directions = np.stack([
        (u - intrinsic[0, 2]) / intrinsic[0, 0],
        (v - intrinsic[1, 2]) / intrinsic[1, 1],
        np.ones_like(u),
    ], axis=-1).astype(np.float32)
    origins = np.zeros_like(directions)
    return np.concatenate([origins, directions], axis=-1)


def render_camera_z(
    meshes_camera: Iterable[trimesh.Trimesh],
    *,
    width: int,
    height: int,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render first-hit camera Z; private owner ids are one-based."""
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    for mesh in meshes_camera:
        scene.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )
    result = scene.cast_rays(o3d.core.Tensor(_rays(width, height, intrinsic)), nthreads=4)
    depth = result["t_hit"].numpy().astype(np.float64)
    geometry = result["geometry_ids"].numpy().astype(np.int64)
    valid = np.isfinite(depth)
    depth[~valid] = 0.0
    owner = np.zeros((height, width), dtype=np.uint16)
    owner[valid] = geometry[valid].astype(np.uint16) + 1
    return depth, owner


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def generate_image_case(
    *,
    occluder: str,
    encoding: str,
    pose_error_m: float,
    width: int,
    height: int,
    intrinsic: np.ndarray,
    depth_scale: float,
    erosion_px: int,
    spatial_stride_px: int,
    visibility_uncertainty_margin_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_local = artificial_asymmetric_mesh()
    measured, nominal_target, nominal_occluders, owners = [], [], [], []
    cameras, objects = [], []
    for index in range(3):
        object_to_world = independent_transform(
            [0.0 + 0.012 * index, 0.0 - 0.006 * index, 0.92],
            [0.05 + 0.04 * index, -0.12 + 0.03 * index, 0.04 * index],
        )
        world_to_camera = np.eye(4)
        target_camera = target_local.copy()
        target_camera.vertices = independent_apply(target_local.vertices, object_to_world)
        actual_meshes = [target_camera]
        nominal_rows: list[trimesh.Trimesh] = []
        if occluder != "none":
            if occluder == "hand_partial":
                center, extent = (0.075, 0.0, 0.78), (0.12, 0.22, 0.055)
            elif occluder == "hand_large":
                center, extent = (0.025, 0.0, 0.77), (0.22, 0.25, 0.060)
            elif occluder == "hand_complete":
                center, extent = (0.0, 0.0, 0.74), (0.45, 0.40, 0.070)
            elif occluder == "other_front":
                center, extent = (-0.055, 0.015, 0.80), (0.14, 0.16, 0.060)
            elif occluder == "other_behind":
                center, extent = (0.0, 0.0, 1.12), (0.30, 0.30, 0.060)
            else:
                raise ValueError(f"unknown occluder: {occluder}")
            actual = _box(extent, center)
            nominal_center = (center[0] + pose_error_m, center[1], center[2])
            nominal = _box(extent, nominal_center)
            actual_meshes.append(actual)
            nominal_rows.append(nominal)
        full_depth, owner = render_camera_z(
            actual_meshes, width=width, height=height, intrinsic=intrinsic,
        )
        target_depth, _ = render_camera_z(
            [target_camera], width=width, height=height, intrinsic=intrinsic,
        )
        occ_depths = []
        for row in nominal_rows:
            values, _ = render_camera_z([row], width=width, height=height, intrinsic=intrinsic)
            occ_depths.append(values)
        if encoding == "uint16":
            encoded = np.rint(full_depth * depth_scale)
            if np.any(encoded < 0) or np.any(encoded > np.iinfo(np.uint16).max):
                raise ValueError("uint16 depth encoding overflow")
            full_depth = encoded.astype(np.uint16).astype(np.float64) / depth_scale
        elif encoding != "float32":
            raise ValueError("unknown depth encoding")
        measured.append(full_depth.astype(np.float32))
        nominal_target.append(target_depth.astype(np.float32))
        nominal_occluders.append(
            np.stack(occ_depths).astype(np.float32)
            if occ_depths else np.empty((0, height, width), dtype=np.float32)
        )
        owners.append(owner)
        cameras.append(world_to_camera)
        objects.append(object_to_world)
    maximum_occluders = max(row.shape[0] for row in nominal_occluders)
    padded = np.zeros((3, maximum_occluders, height, width), dtype=np.float32)
    for index, row in enumerate(nominal_occluders):
        padded[index, : row.shape[0]] = row
    public = {
        "case_kind": np.asarray("image"),
        "vertices": np.asarray(target_local.vertices, dtype=np.float64),
        "faces": np.asarray(target_local.faces, dtype=np.int64),
        "measured_depth_m": np.stack(measured),
        "target_nominal_depth_m": np.stack(nominal_target),
        "occluder_nominal_depths_m": padded,
        "intrinsic": intrinsic.astype(np.float64),
        "world_to_camera": np.stack(cameras),
        "object_to_world": np.stack(objects),
        "erosion_px": np.asarray(erosion_px, dtype=np.int64),
        "spatial_stride_px": np.asarray(spatial_stride_px, dtype=np.int64),
        "visibility_uncertainty_margin_m": np.asarray(
            visibility_uncertainty_margin_m, dtype=np.float64,
        ),
    }
    truth = {
        "has_unique_global_answer": True,
        "generation_model": "IDENTITY_BOTH_MODELS",
        "recovery_transform": np.eye(4).tolist(),
        "owner_maps": np.stack(owners),
        "depth_encoding": encoding,
        "occluder": occluder,
        "nominal_occluder_pose_error_m": pose_error_m,
    }
    return public, truth


PUBLIC_INPUT_KEYS = {
    "case_kind", "vertices", "faces", "points_camera", "observation_offsets",
    "world_to_camera", "object_to_world", "measured_depth_m", "target_nominal_depth_m",
    "occluder_nominal_depths_m", "intrinsic", "erosion_px", "spatial_stride_px",
    "visibility_uncertainty_margin_m",
}


def write_public_case(path: Path, values: dict[str, np.ndarray]) -> None:
    unknown = set(values) - PUBLIC_INPUT_KEYS
    if unknown:
        raise ValueError(f"public case contains forbidden fields: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def load_public_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        unknown = set(source.files) - PUBLIC_INPUT_KEYS
        if unknown:
            raise ValueError(f"public case contains forbidden fields: {sorted(unknown)}")
        return {key: source[key] for key in source.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
