"""V1 robust sequence alignment for real FoundationPose and WiLoR artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION, validate_aligned_trajectory, validate_contact
from ..adapters.sam3d_objects import _projected_mesh_mask_depth
from .contact import infer_contact
from .smoothing import smooth_rotations, smooth_second_difference

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])
HAND_ORDER = ["left", "right"]
HAND_ROLES = {"invalid", "passive", "active"}


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = points @ K.T
    return projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)


def _mask_centroids(masks: np.ndarray) -> np.ndarray:
    centroids = np.zeros((len(masks), 2), dtype=np.float64)
    for index, mask in enumerate(masks):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            centroids[index] = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    return centroids


def hand_anchored_object_observation(
    K: np.ndarray, object_translation_raw: np.ndarray, mask_centroid: np.ndarray,
    joints_camera: np.ndarray, hand_valid: np.ndarray, hand_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Place the object on its image ray using nearby visual hand joints as metric depth anchors."""
    count = len(object_translation_raw)
    observations = object_translation_raw.copy().astype(np.float64)
    weights = np.full(count, 0.08, dtype=np.float64)  # raw depth remains a weak prior
    anchor_distance = np.full(count, np.inf, dtype=np.float64)
    anchor_depth = np.zeros(count, dtype=np.float64)
    for index in range(count):
        candidates = []
        for hand in range(joints_camera.shape[1]):
            if not hand_valid[index, hand]:
                continue
            projected = _project(K, joints_camera[index, hand])
            distance = np.linalg.norm(projected - mask_centroid[index], axis=1)
            # Use the closest wrist/finger joints; the object is expected at a comparable depth.
            for joint_at in np.argsort(distance)[:4]:
                candidates.append((float(distance[joint_at]), float(joints_camera[index, hand, joint_at, 2]),
                                   float(hand_confidence[index, hand])))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        nearby = [item for item in candidates[:4] if item[1] > 0.05]
        if not nearby:
            continue
        pixel_distance = float(np.median([item[0] for item in nearby]))
        z_hand = float(np.median([item[1] for item in nearby]))
        confidence = float(np.mean([item[2] for item in nearby]))
        ray = np.linalg.inv(K) @ np.array([mask_centroid[index, 0], mask_centroid[index, 1], 1.0])
        observations[index] = ray * z_hand
        anchor_distance[index] = pixel_distance
        anchor_depth[index] = z_hand
        proximity = math.exp(-pixel_distance / 120.0)
        weights[index] = 0.25 + 0.75 * proximity * confidence
    return observations, weights, {
        "anchor_pixel_distance_median": float(np.median(anchor_distance[np.isfinite(anchor_distance)]))
        if np.isfinite(anchor_distance).any() else None,
        "anchor_depth_median_m": float(np.median(anchor_depth[anchor_depth > 0])) if np.any(anchor_depth > 0) else None,
        "raw_foundationpose_weight": 0.08,
    }


def _acceleration_jitter(values: np.ndarray, timestamps: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    dt = float(np.median(np.diff(timestamps)))
    acceleration = np.diff(values, n=2, axis=0) / max(dt * dt, 1e-8)
    return float(np.median(np.linalg.norm(acceleration.reshape(len(acceleration), -1), axis=1)))


def _reprojection_residual(K: np.ndarray, translations: np.ndarray, centroids: np.ndarray) -> float:
    projected = _project(K, translations)
    return float(np.median(np.linalg.norm(projected - centroids, axis=1)))


def _build_T_sim_world(T_world_object: np.ndarray, mesh_m: trimesh.Trimesh) -> np.ndarray:
    # EgoDex world is +Y up; rotate it to MuJoCo +Z up while preserving handedness.
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    initial_rotation = rotation @ T_world_object[0, :3, :3]
    initial_center = rotation @ T_world_object[0, :3, 3]
    rotated_vertices = np.asarray(mesh_m.vertices) @ initial_rotation.T
    lowest_relative = float(rotated_vertices[:, 2].min())
    desired_center = np.array([0.0, -0.06, -lowest_relative + 0.002])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = desired_center - initial_center
    return transform


def xhand_wrist_frames_from_joints(
    joints_camera: np.ndarray, side: str, valid: np.ndarray | None = None,
) -> np.ndarray:
    """Build the xHand palm-site frame from observed MANO landmarks."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    joints = np.asarray(joints_camera, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"joints_camera must have shape (T, 21, 3), got {joints.shape}")
    measured = np.ones(len(joints), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if measured.shape != (len(joints),):
        raise ValueError(f"valid must have shape ({len(joints)},), got {measured.shape}")

    z_axis = joints[:, 9] - joints[:, 0]
    y_aux = joints[:, 5] - joints[:, 13]
    z_norm = np.linalg.norm(z_axis, axis=1)
    z_unit = z_axis / np.maximum(z_norm[:, None], 1e-12)
    x_axis = np.cross(y_aux, z_unit)
    x_norm = np.linalg.norm(x_axis, axis=1)
    x_unit = x_axis / np.maximum(x_norm[:, None], 1e-12)
    y_axis = np.cross(z_unit, x_unit)
    y_norm = np.linalg.norm(y_axis, axis=1)
    usable = measured & (z_norm >= 1e-8) & (x_norm >= 1e-8) & (y_norm >= 1e-8)
    if measured.any() and not usable.any():
        raise ValueError("cannot construct xHand wrist frame from any valid MANO frame")
    if side == "left":
        x_unit *= -1.0
        y_axis *= -1.0
    rotations = np.stack([x_unit, y_axis, z_unit], axis=-1)
    valid_indices = np.flatnonzero(usable)
    if valid_indices.size == 0:
        canonical = np.diag([-1.0, -1.0, 1.0]) if side == "left" else np.eye(3)
        return np.repeat(canonical[None], len(joints), axis=0)
    valid_rotations = rotations[usable]
    if np.any(np.linalg.det(valid_rotations) < 0.999):
        raise ValueError("constructed xHand wrist frame is not a proper rotation")
    if valid_indices.size == 1:
        return np.repeat(valid_rotations, len(joints), axis=0)

    from scipy.spatial.transform import Rotation, Slerp

    result = np.empty((len(joints), 3, 3), dtype=np.float64)
    result[: valid_indices[0]] = valid_rotations[0]
    result[valid_indices[-1] + 1 :] = valid_rotations[-1]
    interpolation_indices = np.arange(valid_indices[0], valid_indices[-1] + 1)
    result[interpolation_indices] = Slerp(
        valid_indices, Rotation.from_matrix(valid_rotations),
    )(interpolation_indices).as_matrix()
    return result


def classify_hand_roles(
    T_sim_object: np.ndarray, fingertips_sim: np.ndarray, valid_hand: np.ndarray,
    object_scale_m: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Separate hand visibility/quality from participation in manipulation."""
    object_centers = np.asarray(T_sim_object)[:, :3, 3]
    roles: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    active_distance_m = max(0.08, 2.0 * float(object_scale_m))
    for hand in range(fingertips_sim.shape[1]):
        valid = np.asarray(valid_hand[:, hand], dtype=bool)
        valid_rate = float(np.mean(valid))
        nearest = np.min(
            np.linalg.norm(fingertips_sim[:, hand] - object_centers[:, None], axis=-1), axis=1,
        )
        valid_nearest = nearest[valid]
        median_distance = float(np.median(valid_nearest)) if valid_nearest.size else None
        minimum_distance = float(np.min(valid_nearest)) if valid_nearest.size else None
        if valid_rate < 0.5:
            role = "invalid"
        elif minimum_distance is not None and minimum_distance <= active_distance_m:
            role = "active"
        else:
            role = "passive"
        roles.append(role)
        diagnostics.append({
            "side": HAND_ORDER[hand], "role": role, "valid_rate": valid_rate,
            "minimum_fingertip_object_center_distance_m": minimum_distance,
            "median_fingertip_object_center_distance_m": median_distance,
            "active_distance_threshold_m": active_distance_m,
        })
    return roles, diagnostics


def object_minimum_z(mesh_m: trimesh.Trimesh, transforms: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_m.vertices, dtype=np.float64)
    values = np.asarray(transforms, dtype=np.float64)
    return np.asarray([
        float((vertices @ transform[:3, :3].T + transform[:3, 3]).min(axis=0)[2])
        for transform in values
    ])


def enforce_simulation_floor(
    mesh_m: trimesh.Trimesh, T_sim_object: np.ndarray, T_sim_wrist: np.ndarray,
    fingertips_sim: np.ndarray, hand_roles: list[str], *, object_clearance_m: float = 0.002,
    hand_clearance_m: float = 0.060,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep active interaction groups and passive visible hands above the simulator floor."""
    if any(role not in HAND_ROLES for role in hand_roles):
        raise ValueError(f"unsupported hand role in {hand_roles}")
    objects = np.asarray(T_sim_object, dtype=np.float64).copy()
    wrists = np.asarray(T_sim_wrist, dtype=np.float64).copy()
    fingertips = np.asarray(fingertips_sim, dtype=np.float64).copy()
    active = [index for index, role in enumerate(hand_roles) if role == "active"]
    before_object_min = object_minimum_z(mesh_m, objects)
    interaction_shift = np.zeros(len(objects), dtype=np.float64)
    for frame in range(len(objects)):
        required = object_clearance_m - before_object_min[frame]
        if active:
            hand_min = min(
                float(wrists[frame, active, 2, 3].min()),
                float(fingertips[frame, active, :, 2].min()),
            )
            required = max(required, hand_clearance_m - hand_min)
        interaction_shift[frame] = max(0.0, required)
    objects[:, 2, 3] += interaction_shift
    for hand in active:
        wrists[:, hand, 2, 3] += interaction_shift
        fingertips[:, hand, :, 2] += interaction_shift[:, None]

    passive_shift: dict[str, float] = {}
    for hand, role in enumerate(hand_roles):
        if role != "passive":
            continue
        hand_min = min(float(wrists[:, hand, 2, 3].min()), float(fingertips[:, hand, :, 2].min()))
        shift = max(0.0, hand_clearance_m - hand_min)
        wrists[:, hand, 2, 3] += shift
        fingertips[:, hand, :, 2] += shift
        passive_shift[HAND_ORDER[hand]] = shift

    after_object_min = object_minimum_z(mesh_m, objects)
    return objects, wrists, fingertips, {
        "object_clearance_m": object_clearance_m,
        "hand_target_clearance_m": hand_clearance_m,
        "object_min_z_before_m": float(before_object_min.min()),
        "object_min_z_after_m": float(after_object_min.min()),
        "interaction_shift_max_m": float(interaction_shift.max()),
        "interaction_shift_nonzero_frame_count": int(np.count_nonzero(interaction_shift > 0)),
        "passive_hand_constant_shift_m": passive_shift,
        "wrist_min_z_after_m": {
            HAND_ORDER[hand]: float(wrists[:, hand, 2, 3].min()) for hand in range(len(HAND_ORDER))
        },
        "fingertip_min_z_after_m": {
            HAND_ORDER[hand]: float(fingertips[:, hand, :, 2].min()) for hand in range(len(HAND_ORDER))
        },
    }


def _transform_series(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk->tik", left, right)


def _sample_indices(valid: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if indices.size <= maximum:
        return indices
    positions = np.linspace(0, indices.size - 1, maximum).round().astype(int)
    return indices[np.unique(positions)]


def _optimize_global_scale(
    canonical_mesh: trimesh.Trimesh, initial_scale_m: float, K: np.ndarray,
    poses_camera_object: np.ndarray, masks: np.ndarray, valid: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Estimate one episode-level mesh scale from aligned silhouette areas."""
    ratios: list[float] = []
    sampled = _sample_indices(valid, 18)
    for index in sampled:
        rendered, _ = _projected_mesh_mask_depth(
            canonical_mesh, K, poses_camera_object[index], masks[index].shape, initial_scale_m,
        )
        rendered_area = int(rendered.sum())
        mask_area = int(masks[index].sum())
        if rendered_area > 0 and mask_area > 0:
            ratios.append(math.sqrt(mask_area / rendered_area))
    if not ratios:
        return initial_scale_m, {
            "initial_scale_to_m": initial_scale_m, "optimized_scale_to_m": initial_scale_m,
            "scale_ratio": 1.0, "sample_count": 0,
            "method": "fallback_initial_scale_no_valid_silhouette",
        }
    ratio = float(np.clip(np.median(ratios), 0.10, 3.0))
    optimized = float(initial_scale_m * ratio)
    return optimized, {
        "initial_scale_to_m": initial_scale_m, "optimized_scale_to_m": optimized,
        "scale_ratio": ratio, "sample_count": len(ratios),
        "method": "median_sqrt_observed_to_rendered_mask_area_ratio",
    }


def _render_comparison(
    canonical_mesh: trimesh.Trimesh, K: np.ndarray, masks: np.ndarray, valid: np.ndarray,
    raw_poses: np.ndarray, raw_scale_m: float, aligned_poses: np.ndarray,
    aligned_scale_m: float, depth_group: Any | None,
) -> dict[str, Any]:
    raw_iou: list[float] = []
    aligned_iou: list[float] = []
    raw_depth: list[float] = []
    aligned_depth: list[float] = []
    sampled = _sample_indices(valid, 30)
    for index in sampled:
        mask = masks[index].astype(bool)
        depth_m = valid_depth = None
        if depth_group is not None:
            depth_m = np.asarray(depth_group["depth_m"][index], dtype=np.float32)
            valid_depth = np.asarray(depth_group["valid"][index], dtype=bool)
        for pose, scale, ious, depths in (
            (raw_poses[index], raw_scale_m, raw_iou, raw_depth),
            (aligned_poses[index], aligned_scale_m, aligned_iou, aligned_depth),
        ):
            rendered, rendered_depth = _projected_mesh_mask_depth(
                canonical_mesh, K, pose, mask.shape, scale,
            )
            union = np.logical_or(rendered, mask).sum()
            ious.append(float(np.logical_and(rendered, mask).sum() / union) if union else 0.0)
            if depth_m is not None and valid_depth is not None:
                overlap = (
                    rendered & mask & valid_depth & np.isfinite(depth_m)
                    & np.isfinite(rendered_depth) & (depth_m > 0)
                )
                if overlap.any():
                    residual = float(np.median(np.abs(rendered_depth[overlap] - depth_m[overlap])))
                    depths.append(residual / max(float(np.median(depth_m[overlap])), 1e-3))
    result: dict[str, Any] = {
        "sample_count": int(sampled.size),
        "silhouette_iou_raw": float(np.mean(raw_iou)) if raw_iou else None,
        "silhouette_iou_aligned": float(np.mean(aligned_iou)) if aligned_iou else None,
    }
    if raw_iou and aligned_iou:
        result["silhouette_iou_change"] = result["silhouette_iou_aligned"] - result["silhouette_iou_raw"]
    if raw_depth and aligned_depth:
        result.update({
            "relative_depth_residual_raw": float(np.median(raw_depth)),
            "relative_depth_residual_aligned": float(np.median(aligned_depth)),
            "relative_depth_residual_change": float(np.median(aligned_depth) - np.median(raw_depth)),
        })
    else:
        result["relative_depth_residual"] = "not_observable_missing_or_nonoverlapping_metric_depth"
    return result


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def optimize_run(
    run_dir: str | Path, *, smoothing_strength: float = 18.0,
    hand_smoothing_strength: float = 5.0, overwrite: bool = False,
) -> tuple[Path, Path]:
    root = Path(run_dir).resolve()
    output_dir = root / "optimization"
    aligned_path = output_dir / "aligned_trajectory.npz"
    contact_path = output_dir / "contact.npz"
    if aligned_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {aligned_path}; pass --overwrite")
    with np.load(root / "object_tracking/foundationpose_raw.npz", allow_pickle=False) as artifact:
        object_raw = {key: np.asarray(artifact[key]) for key in artifact.files}
    with np.load(root / "hands/wilor_raw.npz", allow_pickle=False) as artifact:
        hands_raw = {key: np.asarray(artifact[key]) for key in artifact.files}
    with np.load(root / "segmentation/object_masks.npz", allow_pickle=False) as artifact:
        masks_raw = {key: np.asarray(artifact[key]) for key in artifact.files}
    if not (
        np.array_equal(object_raw["frame_indices"], hands_raw["frame_indices"])
        and np.array_equal(object_raw["frame_indices"], masks_raw["frame_indices"])
    ):
        raise ValueError("WP2/WP3/WP6 timelines differ; full synchronized artifacts are required")
    timestamps = object_raw["timestamps_s"].astype(np.float64)
    frame_indices = object_raw["frame_indices"].astype(np.int64)
    frame_rows = json.loads((root / "frames/frame_index.json").read_text(encoding="utf-8"))["frames"]
    calibration_lookup = {
        int(row["source_frame_index"]): int(row["frame_index"]) for row in frame_rows
    }
    calibration_indices = np.asarray([calibration_lookup[int(frame)] for frame in frame_indices])
    T_world_camera = np.load(root / "calibration/T_world_camera.npy")[calibration_indices]
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    selected = json.loads((root / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
    canonical_mesh = trimesh.load_mesh(root / selected["canonical_visual_mesh"], process=False)
    initial_scale_to_m = float(selected["scale_to_m"])

    T_camera_object_raw = object_raw["T_camera_object"].astype(np.float64)
    object_translation_raw = T_camera_object_raw[:, :3, 3]
    joints_camera = (
        hands_raw["joints_camera_rootrel"].astype(np.float64)
        + hands_raw["translation_camera"].astype(np.float64)[:, :, None, :]
    )
    centroids = _mask_centroids(masks_raw["masks"].astype(bool))
    observations, observation_weights, anchor_metrics = hand_anchored_object_observation(
        K, object_translation_raw, centroids, joints_camera, hands_raw["valid"], hands_raw["score"],
    )
    aligned_translation_camera = smooth_second_difference(observations, observation_weights, smoothing_strength)
    object_rotation_camera = smooth_rotations(
        T_camera_object_raw[:, :3, :3], object_raw["confidence"], smoothing_strength * 0.25
    )
    T_camera_object_aligned = np.repeat(np.eye(4)[None], len(frame_indices), axis=0)
    T_camera_object_aligned[:, :3, :3] = object_rotation_camera
    T_camera_object_aligned[:, :3, 3] = aligned_translation_camera
    scale_valid = (
        object_raw["valid"].astype(bool) & masks_raw["valid"].astype(bool)
        & (T_camera_object_aligned[:, 2, 3] > 0)
    )
    scale_to_m, scale_metrics = _optimize_global_scale(
        canonical_mesh, initial_scale_to_m, K, T_camera_object_aligned,
        masks_raw["masks"].astype(bool), scale_valid,
    )
    mesh_m = canonical_mesh.copy()
    mesh_m.apply_scale(scale_to_m)
    T_world_object = np.einsum("tij,tjk->tik", T_world_camera, T_camera_object_aligned)

    hand_valid = hands_raw["valid"].astype(bool)
    hand_confidence = hands_raw["score"].astype(np.float64)
    # Preserve the measured left-hand weakness as lower absolute confidence.
    hand_confidence[:, 0] *= 0.45
    # WiLoR translation locates the MANO model origin, not the anatomical wrist.
    # Use the same translated joint array as the fingertip targets so one hand
    # cannot contain mutually inconsistent wrist and finger positions.
    wrists_camera = joints_camera[:, :, 0]
    fingertips_camera = joints_camera[:, :, FINGERTIP_INDICES]
    wrists_camera_aligned = np.empty_like(wrists_camera)
    fingertips_camera_aligned = np.empty_like(fingertips_camera)
    wrist_rotation_camera = np.empty((len(frame_indices), 2, 3, 3), dtype=np.float64)
    for hand in range(2):
        weights = np.where(hand_valid[:, hand], np.maximum(hand_confidence[:, hand], 0.05), 1e-6)
        wrists_camera_aligned[:, hand] = smooth_second_difference(
            wrists_camera[:, hand], weights, hand_smoothing_strength
        )
        fingertips_camera_aligned[:, hand] = smooth_second_difference(
            fingertips_camera[:, hand], weights, hand_smoothing_strength
        )
        xhand_rotation = xhand_wrist_frames_from_joints(
            joints_camera[:, hand], HAND_ORDER[hand], hand_valid[:, hand],
        )
        wrist_rotation_camera[:, hand] = smooth_rotations(
            xhand_rotation, weights, hand_smoothing_strength,
        )
    wrist_world = np.repeat(np.eye(4)[None, None], len(frame_indices) * 2, axis=0).reshape(len(frame_indices), 2, 4, 4)
    fingertips_world = np.empty_like(fingertips_camera_aligned)
    for hand in range(2):
        wrist_world[:, hand, :3, :3] = np.einsum(
            "tij,tjk->tik", T_world_camera[:, :3, :3], wrist_rotation_camera[:, hand]
        )
        wrist_world[:, hand, :3, 3] = np.einsum(
            "tij,tj->ti", T_world_camera[:, :3, :3], wrists_camera_aligned[:, hand]
        ) + T_world_camera[:, :3, 3]
        fingertips_world[:, hand] = np.einsum(
            "tij,tfj->tfi", T_world_camera[:, :3, :3], fingertips_camera_aligned[:, hand]
        ) + T_world_camera[:, None, :3, 3]
    T_sim_world = _build_T_sim_world(T_world_object, mesh_m)
    T_sim_object = _transform_series(T_sim_world, T_world_object)
    T_sim_wrist = np.einsum("ij,thjk->thik", T_sim_world, wrist_world)
    fingertips_sim = np.einsum(
        "ij,thfj->thfi", T_sim_world,
        np.concatenate([fingertips_world, np.ones((*fingertips_world.shape[:-1], 1))], axis=-1),
    )[..., :3]
    hand_roles, hand_role_metrics = classify_hand_roles(
        T_sim_object, fingertips_sim, hand_valid, scale_to_m,
    )
    T_sim_object, T_sim_wrist, fingertips_sim, floor_metrics = enforce_simulation_floor(
        mesh_m, T_sim_object, T_sim_wrist, fingertips_sim, hand_roles,
    )
    shared_betas = np.stack([
        np.median(hands_raw["mano_betas"][hand_valid[:, hand], hand], axis=0)
        if hand_valid[:, hand].any() else np.zeros(10)
        for hand in range(2)
    ])
    aligned = {
        "frame_indices": frame_indices, "timestamps_s": timestamps,
        "T_sim_object": T_sim_object[:, None].astype(np.float32),
        "T_sim_wrist": T_sim_wrist.astype(np.float32),
        "fingertips_sim": fingertips_sim.astype(np.float32),
        "mano_pose": hands_raw["mano_hand_pose"].astype(np.float32),
        "mano_betas": shared_betas.astype(np.float32),
        "object_scale_to_m": np.array([scale_to_m], dtype=np.float32),
        "valid_object": object_raw["valid"][:, None].astype(bool),
        "valid_hand": hand_valid, "confidence_object": object_raw["confidence"][:, None].astype(np.float32),
        "confidence_hand": hand_confidence.astype(np.float32),
    }
    validate_aligned_trajectory(aligned)
    contact, contact_positions, contact_metrics = infer_contact(
        mesh_m, T_sim_object, fingertips_sim, timestamps, hand_valid
    )
    T_sim_object_raw = np.einsum("ij,tjk,tkl->til", T_sim_world, T_world_camera, T_camera_object_raw)
    raw_fingertip_homogeneous = np.concatenate([
        fingertips_camera, np.ones((*fingertips_camera.shape[:-1], 1))
    ], axis=-1)
    T_sim_camera = np.einsum("ij,tjk->tik", T_sim_world, T_world_camera)
    fingertips_sim_raw = np.einsum(
        "tij,thfj->thfi", T_sim_camera, raw_fingertip_homogeneous,
    )[..., :3]
    _, _, raw_contact_metrics = infer_contact(
        mesh_m, T_sim_object_raw, fingertips_sim_raw, timestamps, hand_valid
    )
    contact_artifact = {
        "frame_indices": frame_indices, "timestamps_s": timestamps,
        "contact": contact, "contact_pos_object_local": contact_positions,
    }
    validate_contact(contact_artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(aligned_path, **aligned)
    np.savez_compressed(contact_path, **contact_artifact)
    raw_reprojection = _reprojection_residual(K, object_translation_raw, centroids)
    aligned_reprojection = _reprojection_residual(K, aligned_translation_camera, centroids)
    depth_group = None
    depth_path = root / "depth/metric_depth.zarr"
    if depth_path.exists():
        import zarr
        depth_group = zarr.open(str(depth_path), mode="r")
    render_metrics = _render_comparison(
        canonical_mesh, K, masks_raw["masks"].astype(bool), scale_valid,
        T_camera_object_raw, initial_scale_to_m, T_camera_object_aligned,
        scale_to_m, depth_group,
    )
    raw_slip = raw_contact_metrics.get("contact_local_slip_p95_m_s")
    aligned_slip = contact_metrics.get("contact_local_slip_p95_m_s")
    metrics = {
        "schema_version": SCHEMA_VERSION, "T_sim_world": T_sim_world.tolist(),
        "hand_order": HAND_ORDER, "left_absolute_confidence_multiplier": 0.45,
        "hands": {
            "roles": dict(zip(HAND_ORDER, hand_roles)),
            "role_policy": "visible reliable hands are retained; activity only controls interaction constraints",
            "role_metrics": hand_role_metrics,
            "wrist_position_source": "translated MANO wrist joint 0",
            "wrist_orientation_source": "landmark-derived xHand palm frame",
        },
        "simulation_floor": floor_metrics,
        "anchor": anchor_metrics,
        "raw_vs_aligned": {
            "object_translation_acceleration_jitter_raw_m_s2": _acceleration_jitter(object_translation_raw, timestamps),
            "object_translation_acceleration_jitter_aligned_m_s2": _acceleration_jitter(aligned_translation_camera, timestamps),
            "object_mask_centroid_reprojection_raw_px": raw_reprojection,
            "object_mask_centroid_reprojection_aligned_px": aligned_reprojection,
            "object_mask_centroid_reprojection_change_px": aligned_reprojection - raw_reprojection,
            **render_metrics,
            "penetration": "not_observable_aligned_hand_surface_not_materialized",
            "contact_local_slip_p95_m_s_raw": raw_slip if raw_slip is not None else "not_observable_no_raw_contact",
            "contact_local_slip_p95_m_s_aligned": aligned_slip,
            "contact_local_slip_change_m_s": (
                float(aligned_slip - raw_slip)
                if raw_slip is not None and aligned_slip is not None else "not_observable_no_raw_contact"
            ),
        },
        "contact": {"raw": raw_contact_metrics, "aligned": contact_metrics},
        "optimization": {
            "translation_objective": "weighted hand-ray metric observation + second-difference regularization",
            "smoothing_strength": smoothing_strength, "hand_smoothing_strength": hand_smoothing_strength,
            "depth_policy": "FoundationPose/Depth Anything raw translation weight 0.08; no per-frame mesh scale",
            "global_object_scale": scale_metrics,
        },
    }
    metrics_path = output_dir / "optimization_metrics.json"
    metrics_path.write_text(json.dumps(_safe_json(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from ..visualization import render_optimization

    visualization_outputs: list[str] = []
    visualization_warnings: list[str] = []
    try:
        visualization_path = render_optimization(root, overwrite=True)
        visualization_outputs.append(str(visualization_path.relative_to(root)))
    except Exception as error:
        visualization_warnings.append(
            f"optimization visualization failed without invalidating trajectory artifacts: "
            f"{type(error).__name__}: {error}"
        )
    manifest = RunManifest.load(root / "manifest.json")
    cache_key = stage_cache_key(
        "sequence_optimization", {"smoothing_strength": smoothing_strength,
        "hand_smoothing_strength": hand_smoothing_strength},
        [root / "object_tracking/foundationpose_raw.npz", root / "object_tracking/selected_mesh.json",
         root / "hands/wilor_raw.npz", root / "segmentation/object_masks.npz"],
    )
    manifest.start_stage("sequence_optimization", cache_key=cache_key, command=sys.argv, environment="v2s-opt")
    manifest.finish_stage(
        "sequence_optimization", success=True,
        outputs=[
            str(aligned_path.relative_to(root)), str(contact_path.relative_to(root)),
            str(metrics_path.relative_to(root)), *visualization_outputs,
        ],
        quality_metrics=metrics["raw_vs_aligned"],
        warnings=["left absolute hand confidence is reduced from its measured GT diagnostic",
                  "penetration is an unsigned nearest-surface proxy in the current V1 optimizer",
                  *visualization_warnings],
    )
    return aligned_path, contact_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoothing-strength", type=float, default=18.0)
    parser.add_argument("--hand-smoothing-strength", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    aligned, contact = optimize_run(
        args.run_dir, smoothing_strength=args.smoothing_strength,
        hand_smoothing_strength=args.hand_smoothing_strength, overwrite=args.overwrite,
    )
    print(aligned)
    print(contact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
