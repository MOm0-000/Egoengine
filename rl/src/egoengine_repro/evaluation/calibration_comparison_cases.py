"""Randomized known-answer cases for the Open3D calibration comparison.

This generator is independent of every solver.  Formal instances are created
only from an externally supplied secret seed; private ownership and transforms
are never serialized into the public case files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from .calibration_known_answer import (
    deterministic_surface_points,
    independent_apply,
    independent_inverse,
    independent_transform,
    render_camera_z,
)


def randomized_asymmetric_mesh(rng: np.random.Generator) -> trimesh.Trimesh:
    extents = rng.uniform([0.10, 0.08, 0.09], [0.18, 0.15, 0.17])
    points = rng.uniform(-0.5, 0.5, size=(14, 3)) * extents
    points[:4] += np.asarray([
        [0.035, -0.010, 0.020], [-0.020, 0.028, -0.015],
        [0.012, -0.032, 0.030], [-0.028, -0.020, -0.026],
    ])
    return trimesh.convex.convex_hull(points)


def _nonzero_vector(rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * rng.uniform(low, high)


def random_correction(rng: np.random.Generator, kind: str) -> np.ndarray:
    translation = np.zeros(3)
    rotation = np.zeros(3)
    if kind in ("translation", "combined"):
        translation = _nonzero_vector(rng, 0.004, 0.024)
    if kind in ("rotation", "combined"):
        rotation = _nonzero_vector(rng, np.deg2rad(0.8), np.deg2rad(4.5))
    if kind != "identity" and not (np.any(translation) or np.any(rotation)):
        raise ValueError(f"unknown correction kind: {kind}")
    return independent_transform(translation, rotation)


def random_small_image_correction(rng: np.random.Generator) -> np.ndarray:
    """Nonzero correction kept inside the frozen 2 mm visibility uncertainty."""
    translation = _nonzero_vector(rng, 0.00035, 0.00085)
    rotation = _nonzero_vector(rng, np.deg2rad(0.08), np.deg2rad(0.25))
    return independent_transform(translation, rotation)


def _random_pose_path(
    rng: np.random.Generator, camera_views: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    object_base = rng.uniform([0.15, -0.10, 0.72], [0.30, 0.08, 0.90])
    camera_base = rng.uniform([-0.06, -0.04, 0.04], [0.06, 0.04, 0.14])
    object_step = rng.uniform([-0.012, -0.009, -0.004], [0.018, 0.012, 0.008])
    camera_step = rng.uniform([-0.012, -0.010, -0.004], [0.014, 0.010, 0.006])
    object_rot_step = rng.uniform(-0.025, 0.025, size=3)
    camera_rot_step = rng.uniform(-0.030, 0.030, size=3)
    objects, cameras = [], []
    for index in range(camera_views):
        objects.append(independent_transform(
            object_base + index * object_step,
            rng.uniform(-0.06, 0.06, size=3) + index * object_rot_step,
        ))
        cameras.append(independent_transform(
            camera_base + index * camera_step,
            rng.uniform(-0.04, 0.04, size=3) + index * camera_rot_step,
        ))
    return objects, cameras


def generate_random_direct_case(
    mesh: trimesh.Trimesh,
    *,
    correction: np.ndarray,
    generation_model: str,
    camera_views: int,
    points_per_face: int,
    quality: str,
    rng: np.random.Generator,
    inconsistent: bool = False,
    identifiable: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local = deterministic_surface_points(mesh, points_per_face)
    objects, cameras = _random_pose_path(rng, camera_views)
    corrections = [correction] * camera_views
    if inconsistent:
        corrections = [
            random_correction(rng, "combined") for _ in range(camera_views)
        ]
    points, offsets = [], [0]
    for index, (object_to_world, world_to_camera, current) in enumerate(
        zip(objects, cameras, corrections, strict=True)
    ):
        true_world = independent_apply(local, object_to_world)
        if generation_model == "WORLD_FIXED":
            raw_world = independent_apply(true_world, independent_inverse(current))
            observed = independent_apply(raw_world, world_to_camera)
        elif generation_model == "CAMERA_LOCAL":
            true_camera = independent_apply(true_world, world_to_camera)
            observed = independent_apply(true_camera, independent_inverse(current))
        else:
            raise ValueError(f"unknown generation model: {generation_model}")
        if quality in ("noise", "combined", "strong_combined"):
            sigma = 0.00045 if quality != "strong_combined" else 0.0009
            observed += rng.normal(0.0, sigma, observed.shape)
        if quality in ("random_missing", "combined", "strong_combined"):
            fraction = 0.22 if quality != "strong_combined" else 0.38
            observed = observed[rng.random(len(observed)) >= fraction]
        if quality == "local_missing" and len(observed):
            axis = index % 3
            threshold = np.quantile(observed[:, axis], 0.68)
            observed = observed[observed[:, axis] <= threshold]
        if quality in ("outliers", "combined", "strong_combined") and len(observed):
            fraction = 0.05 if quality != "strong_combined" else 0.10
            count = max(1, round(fraction * len(observed)))
            chosen = rng.choice(len(observed), size=min(count, len(observed)), replace=False)
            observed[chosen] += rng.uniform(-0.025, 0.025, size=(len(chosen), 3))
        points.append(observed)
        offsets.append(offsets[-1] + len(observed))
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
        "has_unique_global_answer": bool(identifiable and not inconsistent),
        "generation_model": generation_model,
        "recovery_transform": correction.tolist() if identifiable and not inconsistent else None,
        "per_view_recovery_transforms": [row.tolist() for row in corrections],
        "insufficiency_proof": (
            "different frames were generated from deliberately incompatible corrections"
            if inconsistent else
            "the public shape has deliberately ambiguous rigid symmetries"
            if not identifiable else ""
        ),
    }
    return public, truth


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _camera_mesh(mesh_local: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    result = mesh_local.copy()
    result.vertices = independent_apply(mesh_local.vertices, transform)
    return result


def generate_random_image_case(
    mesh: trimesh.Trimesh,
    *,
    correction: np.ndarray,
    generation_model: str,
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
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    measured, nominal_target, nominal_occluders, owners = [], [], [], []
    cameras, objects = [], []
    observationally_insufficient = occluder in ("hand_complete", "hand_large")
    for index in range(3):
        object_to_world = independent_transform(
            [rng.uniform(-0.015, 0.015) + 0.009 * index,
             rng.uniform(-0.012, 0.012) - 0.005 * index,
             0.92 + rng.uniform(-0.01, 0.01)],
            rng.uniform(-0.10, 0.10, size=3),
        )
        world_to_camera = independent_transform(
            rng.uniform([-0.008, -0.006, -0.004], [0.008, 0.006, 0.004]),
            rng.uniform(-0.015, 0.015, size=3),
        )
        nominal_target_camera = world_to_camera @ object_to_world
        if generation_model == "CAMERA_LOCAL":
            actual_target_camera = independent_inverse(correction) @ nominal_target_camera
        elif generation_model == "WORLD_FIXED":
            actual_target_camera = world_to_camera @ independent_inverse(correction) @ object_to_world
        else:
            raise ValueError(f"unknown generation model: {generation_model}")
        nominal_target_mesh = _camera_mesh(mesh, nominal_target_camera)
        actual_target_mesh = _camera_mesh(mesh, actual_target_camera)
        actual_meshes = [actual_target_mesh]
        nominal_rows: list[trimesh.Trimesh] = []
        if occluder != "none":
            if occluder == "hand_partial":
                center, extent = (0.070, 0.0, 0.78), (0.12, 0.22, 0.055)
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
            jitter = rng.uniform(-0.004, 0.004, size=3)
            actual_center = tuple(np.asarray(center) + jitter)
            nominal_center = tuple(np.asarray(actual_center) + [pose_error_m, 0.0, 0.0])
            actual_meshes.append(_box(extent, actual_center))
            nominal_rows.append(_box(extent, nominal_center))
        full_depth, owner = render_camera_z(
            actual_meshes, width=width, height=height, intrinsic=intrinsic,
        )
        target_depth, _ = render_camera_z(
            [nominal_target_mesh], width=width, height=height, intrinsic=intrinsic,
        )
        occ_depths = [
            render_camera_z([row], width=width, height=height, intrinsic=intrinsic)[0]
            for row in nominal_rows
        ]
        if encoding == "uint16":
            encoded = np.rint(full_depth * depth_scale)
            if np.any(encoded < 0) or np.any(encoded > np.iinfo(np.uint16).max):
                raise ValueError("uint16 depth encoding overflow")
            full_depth = encoded.astype(np.uint16).astype(np.float64) / depth_scale
        elif encoding != "float32":
            raise ValueError(f"unknown depth encoding: {encoding}")
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
        "vertices": np.asarray(mesh.vertices, dtype=np.float64),
        "faces": np.asarray(mesh.faces, dtype=np.int64),
        "measured_depth_m": np.stack(measured),
        "target_nominal_depth_m": np.stack(nominal_target),
        "occluder_nominal_depths_m": padded,
        "intrinsic": np.asarray(intrinsic, dtype=np.float64),
        "world_to_camera": np.stack(cameras),
        "object_to_world": np.stack(objects),
        "erosion_px": np.asarray(erosion_px, dtype=np.int64),
        "spatial_stride_px": np.asarray(spatial_stride_px, dtype=np.int64),
        "visibility_uncertainty_margin_m": np.asarray(
            visibility_uncertainty_margin_m, dtype=np.float64,
        ),
    }
    truth = {
        "has_unique_global_answer": not observationally_insufficient,
        "generation_model": generation_model,
        "recovery_transform": None if observationally_insufficient else correction.tolist(),
        "owner_maps": np.stack(owners),
        "depth_encoding": encoding,
        "occluder": occluder,
        "nominal_occluder_pose_error_m": pose_error_m,
        "insufficiency_proof": (
            "the target is deliberately fully or almost fully occluded"
            if observationally_insufficient else ""
        ),
    }
    return public, truth
