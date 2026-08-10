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
MIN_OBJECT_DEPTH_RATIO = 0.75
MAX_OBJECT_DEPTH_RATIO = 1.33
MIN_GLOBAL_SCALE_RATIO = 0.75
MAX_GLOBAL_SCALE_RATIO = 1.50
MAX_HAND_CALIBRATION_MEDIAN_REPROJECTION_PX = 12.0
MAX_HAND_CALIBRATION_P95_REPROJECTION_PX = 25.0
MAX_HAND_EXPORT_MEDIAN_REPROJECTION_PX = 20.0
MAX_HAND_EXPORT_P95_REPROJECTION_PX = 40.0
CONTACT_CANDIDATE_DISTANCE_PX = 16.0
MIN_CONTACT_CANDIDATE_SAMPLES = 8
MIN_CONTACT_CANDIDATE_FRAMES = 4
MIN_CONTACT_CANDIDATE_RUN = 3
MIN_CONTACT_SIMILARITY_RATIO = 0.15
MAX_CONTACT_SIMILARITY_RATIO = 1.50
MIN_CONTACT_OPPOSED_FRAMES = 3
MIN_CONTACT_OPPOSED_RUN = 3
MAX_CONTACT_SLIP_P95_M_S = 0.30


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


def _interpolate_valid_vectors(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid samples without changing measured samples."""
    result = np.asarray(values, dtype=np.float64).copy()
    usable = np.asarray(valid, dtype=bool) & np.isfinite(result).all(axis=1)
    usable &= result[:, 2] > 0.05
    indices = np.flatnonzero(usable)
    if not indices.size:
        raise ValueError("no valid positive-depth object translations")
    timeline = np.arange(len(result))
    for axis in range(result.shape[1]):
        result[:, axis] = np.interp(timeline, indices, result[indices, axis])
    return result


def _metric_mask_depths(
    depth_group: Any | None, frame_indices: np.ndarray, masks: np.ndarray, mask_valid: np.ndarray,
) -> np.ndarray:
    """Read robust object depths without assuming the Zarr and run use identical offsets."""
    depths = np.full(len(frame_indices), np.nan, dtype=np.float64)
    if depth_group is None:
        return depths
    lookup = {
        int(frame): index for index, frame in enumerate(np.asarray(depth_group["frame_indices"]))
    }
    for index, frame_index in enumerate(frame_indices):
        depth_at = lookup.get(int(frame_index))
        if depth_at is None or not mask_valid[index]:
            continue
        mask = masks[index].astype(bool)
        ys, xs = np.where(mask)
        if not xs.size:
            continue
        y_slice = slice(int(ys.min()), int(ys.max()) + 1)
        x_slice = slice(int(xs.min()), int(xs.max()) + 1)
        local_mask = mask[y_slice, x_slice]
        depth = np.asarray(depth_group["depth_m"][depth_at, y_slice, x_slice], dtype=np.float32)
        valid = np.asarray(depth_group["valid"][depth_at, y_slice, x_slice], dtype=bool)
        usable = local_mask & valid & np.isfinite(depth) & (depth > 0)
        if usable.any():
            depths[index] = float(np.median(depth[usable]))
    return depths


def calibrate_hand_depth_scale(
    K: np.ndarray, joints_camera: np.ndarray, hand_valid: np.ndarray,
    hand_confidence: np.ndarray, object_translation: np.ndarray,
    mask_centroid: np.ndarray, object_valid: np.ndarray, *, max_pixel_distance: float = 160.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate WiLoR translation scale without breaking its 2D evidence.

    WiLoR root-relative hand geometry is metric, but its weak-perspective camera
    translation can have a different episode-level depth scale. Estimate that
    scale only where a reconstructed hand is visibly near the object. A root-only
    depth change preserves the root projection but can collapse the projected
    hand around it, so reject any candidate that moves the full hand too far in
    image space.
    """
    joints = np.asarray(joints_camera, dtype=np.float64)
    calibrated = joints.copy()
    ratios: list[list[float]] = [[] for _ in range(joints.shape[1])]
    distances: list[list[float]] = [[] for _ in range(joints.shape[1])]
    for index in range(len(joints)):
        if not object_valid[index] or object_translation[index, 2] <= 0.05:
            continue
        for hand in range(joints.shape[1]):
            if not hand_valid[index, hand] or hand_confidence[index, hand] <= 0.2:
                continue
            projected = _project(K, joints[index, hand])
            pixel_distances = np.linalg.norm(projected - mask_centroid[index], axis=1)
            closest = np.argsort(pixel_distances)[:4]
            pixel_distance = float(np.median(pixel_distances[closest]))
            hand_depth = float(np.median(joints[index, hand, closest, 2]))
            if pixel_distance <= max_pixel_distance and hand_depth > 0.05:
                ratios[hand].append(float(object_translation[index, 2] / hand_depth))
                distances[hand].append(pixel_distance)
    records: dict[str, dict[str, Any]] = {}
    for hand in range(joints.shape[1]):
        raw_ratio = float(np.median(ratios[hand])) if ratios[hand] else 1.0
        candidate_ratio = float(np.clip(raw_ratio, 0.25, 5.0)) if len(ratios[hand]) >= 3 else 1.0
        roots = joints[:, hand, 0]
        usable = np.isfinite(roots).all(axis=1) & (roots[:, 2] > 0.05)
        candidate = joints[:, hand].copy()
        target_roots = roots[usable] * candidate_ratio
        candidate[usable] += (target_roots - roots[usable])[:, None, :]
        projected_raw = _project(K, joints[:, hand])
        projected_candidate = _project(K, candidate)
        reprojection = np.linalg.norm(projected_candidate - projected_raw, axis=-1)
        reprojection_usable = hand_valid[:, hand, None] & np.isfinite(reprojection)
        reprojection_values = reprojection[reprojection_usable]
        median_reprojection = (
            float(np.median(reprojection_values)) if reprojection_values.size else None
        )
        p95_reprojection = (
            float(np.percentile(reprojection_values, 95)) if reprojection_values.size else None
        )
        accepted = bool(
            len(ratios[hand]) >= 3
            and median_reprojection is not None
            and p95_reprojection is not None
            and median_reprojection <= MAX_HAND_CALIBRATION_MEDIAN_REPROJECTION_PX
            and p95_reprojection <= MAX_HAND_CALIBRATION_P95_REPROJECTION_PX
        )
        applied_ratio = candidate_ratio if accepted else 1.0
        if accepted:
            calibrated[:, hand] = candidate
        records[HAND_ORDER[hand] if hand < len(HAND_ORDER) else str(hand)] = {
            "sample_count": len(ratios[hand]),
            "raw_scale_ratio": raw_ratio,
            "candidate_scale_ratio": candidate_ratio,
            "applied_scale_ratio": applied_ratio,
            "candidate_reprojection_median_px": median_reprojection,
            "candidate_reprojection_p95_px": p95_reprojection,
            "accepted": accepted,
            "rejection_reason": (
                None if accepted else
                "insufficient_samples" if len(ratios[hand]) < 3 else
                "full_hand_reprojection_exceeds_trust_region"
            ),
            "median_pixel_distance": (
                float(np.median(distances[hand])) if distances[hand] else None
            ),
        }
    return calibrated, {
        "source": "object_depth_over_nearby_wilor_joint_depth",
        "per_hand": records,
        "ratio_clip": [0.25, 5.0],
        "max_pixel_distance": max_pixel_distance,
        "full_hand_reprojection_trust_region_px": {
            "median": MAX_HAND_CALIBRATION_MEDIAN_REPROJECTION_PX,
            "p95": MAX_HAND_CALIBRATION_P95_REPROJECTION_PX,
        },
    }


def hand_anchored_object_observation(
    K: np.ndarray, object_translation_raw: np.ndarray, mask_centroid: np.ndarray,
    joints_camera: np.ndarray, hand_valid: np.ndarray, hand_confidence: np.ndarray,
    *, object_valid: np.ndarray | None = None, object_confidence: np.ndarray | None = None,
    mask_valid: np.ndarray | None = None, metric_mask_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Constrain image position while preserving the metric object depth source."""
    count = len(object_translation_raw)
    object_valid = (
        np.asarray(object_valid, dtype=bool) if object_valid is not None
        else np.asarray(object_translation_raw)[:, 2] > 0.05
    )
    object_confidence = (
        np.asarray(object_confidence, dtype=np.float64) if object_confidence is not None
        else np.ones(count, dtype=np.float64)
    )
    mask_valid = np.ones(count, dtype=bool) if mask_valid is None else np.asarray(mask_valid, dtype=bool)
    metric_mask_depth = (
        np.full(count, np.nan, dtype=np.float64) if metric_mask_depth is None
        else np.asarray(metric_mask_depth, dtype=np.float64)
    )
    baseline = _interpolate_valid_vectors(object_translation_raw, object_valid)
    observations = baseline.copy()
    weights = np.where(
        object_valid, 0.35 + 0.65 * np.clip(object_confidence, 0.0, 1.0), 0.08,
    )
    anchor_distance = np.full(count, np.inf, dtype=np.float64)
    anchor_depth = np.zeros(count, dtype=np.float64)
    metric_depth_accepted = 0
    metric_depth_rejected = 0
    hand_depth_accepted = 0
    hand_depth_rejected = 0
    for index in range(count):
        if not mask_valid[index]:
            continue
        object_depth = float(baseline[index, 2])
        metric_depth = float(metric_mask_depth[index])
        if np.isfinite(metric_depth) and metric_depth > 0.05:
            ratio = metric_depth / object_depth
            if MIN_OBJECT_DEPTH_RATIO <= ratio <= MAX_OBJECT_DEPTH_RATIO:
                object_depth = 0.75 * object_depth + 0.25 * metric_depth
                metric_depth_accepted += 1
            else:
                metric_depth_rejected += 1
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
        candidates.sort(key=lambda item: item[0])
        nearby = [item for item in candidates[:4] if item[1] > 0.05]
        if nearby:
            pixel_distance = float(np.median([item[0] for item in nearby]))
            z_hand = float(np.median([item[1] for item in nearby]))
            confidence = float(np.mean([item[2] for item in nearby]))
            depth_ratio = z_hand / object_depth
            if pixel_distance <= 160.0 and MIN_OBJECT_DEPTH_RATIO <= depth_ratio <= MAX_OBJECT_DEPTH_RATIO:
                blend = 0.15 * math.exp(-pixel_distance / 120.0) * np.clip(confidence, 0.0, 1.0)
                object_depth = (1.0 - blend) * object_depth + blend * z_hand
                hand_depth_accepted += 1
            elif pixel_distance <= 160.0:
                hand_depth_rejected += 1
            anchor_distance[index] = pixel_distance
            anchor_depth[index] = z_hand
        ray = np.linalg.inv(K) @ np.array([mask_centroid[index, 0], mask_centroid[index, 1], 1.0])
        observations[index] = ray * (object_depth / max(float(ray[2]), 1e-8))
    return observations, weights, {
        "anchor_pixel_distance_median": float(np.median(anchor_distance[np.isfinite(anchor_distance)]))
        if np.isfinite(anchor_distance).any() else None,
        "anchor_depth_median_m": float(np.median(anchor_depth[anchor_depth > 0])) if np.any(anchor_depth > 0) else None,
        "object_depth_median_m": float(np.median(baseline[:, 2])),
        "metric_mask_depth_median_m": (
            float(np.median(metric_mask_depth[np.isfinite(metric_mask_depth)]))
            if np.isfinite(metric_mask_depth).any() else None
        ),
        "metric_depth_accepted_frame_count": metric_depth_accepted,
        "metric_depth_rejected_frame_count": metric_depth_rejected,
        "hand_depth_accepted_frame_count": hand_depth_accepted,
        "hand_depth_rejected_frame_count": hand_depth_rejected,
        "object_depth_ratio_gate": [MIN_OBJECT_DEPTH_RATIO, MAX_OBJECT_DEPTH_RATIO],
        "depth_source_policy": "FoundationPose primary; compatible mask depth and calibrated hand depth are low-weight refinements",
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


def _build_T_sim_world(
    T_world_object: np.ndarray, mesh_m: trimesh.Trimesh, *,
    wrist_world: np.ndarray | None = None, fingertips_world: np.ndarray | None = None,
    active_hint: np.ndarray | None = None, object_clearance_m: float = 0.002,
    hand_clearance_m: float = 0.060,
) -> np.ndarray:
    """Choose one simulation frame that clears the floor for the whole interaction."""
    # EgoDex world is +Y up; rotate it to MuJoCo +Z up while preserving handedness.
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    initial_center = rotation @ T_world_object[0, :3, 3]
    vertices = np.asarray(mesh_m.vertices, dtype=np.float64)
    object_minimum = min(
        float((vertices @ (rotation @ pose[:3, :3]).T + rotation @ pose[:3, 3]).min(axis=0)[2])
        for pose in np.asarray(T_world_object, dtype=np.float64)
    )
    z_translation = object_clearance_m - object_minimum
    if wrist_world is not None and fingertips_world is not None and active_hint is not None:
        active = np.flatnonzero(np.asarray(active_hint, dtype=bool))
        if active.size:
            wrist_points = np.asarray(wrist_world, dtype=np.float64)[:, active, :3, 3]
            fingertip_points = np.asarray(fingertips_world, dtype=np.float64)[:, active]
            hand_minimum = min(
                float((wrist_points @ rotation.T)[..., 2].min()),
                float((fingertip_points @ rotation.T)[..., 2].min()),
            )
            z_translation = max(z_translation, hand_clearance_m - hand_minimum)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:2, 3] = np.array([0.0, -0.06]) - initial_center[:2]
    transform[2, 3] = z_translation
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
    object_scale_m: float, active_hint: np.ndarray | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Separate hand visibility/quality from participation in manipulation."""
    object_centers = np.asarray(T_sim_object)[:, :3, 3]
    hinted = (
        np.zeros(fingertips_sim.shape[1], dtype=bool)
        if active_hint is None else np.asarray(active_hint, dtype=bool)
    )
    if hinted.shape != (fingertips_sim.shape[1],):
        raise ValueError("active_hint does not match hand dimension")
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
        elif hinted[hand] or (minimum_distance is not None and minimum_distance <= active_distance_m):
            role = "active"
        else:
            role = "passive"
        roles.append(role)
        diagnostics.append({
            "side": HAND_ORDER[hand], "role": role, "valid_rate": valid_rate,
            "minimum_fingertip_object_center_distance_m": minimum_distance,
            "median_fingertip_object_center_distance_m": median_distance,
            "active_distance_threshold_m": active_distance_m,
            "persistent_2d_contact_hint": bool(hinted[hand]),
        })
    return roles, diagnostics


def object_minimum_z(mesh_m: trimesh.Trimesh, transforms: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh_m.vertices, dtype=np.float64)
    values = np.asarray(transforms, dtype=np.float64)
    return np.asarray([
        float((vertices @ transform[:3, :3].T + transform[:3, 3]).min(axis=0)[2])
        for transform in values
    ])


def reject_unphysical_passive_hands(
    T_sim_wrist: np.ndarray, fingertips_sim: np.ndarray, hand_roles: list[str], *,
    hand_clearance_m: float = 0.060, max_correction_m: float = 0.050,
) -> tuple[list[str], dict[str, Any]]:
    """Drop passive hands whose absolute translation cannot satisfy floor geometry."""
    roles = list(hand_roles)
    diagnostics: dict[str, Any] = {}
    for hand, role in enumerate(roles):
        if role != "passive":
            continue
        hand_min = min(
            float(T_sim_wrist[:, hand, 2, 3].min()),
            float(fingertips_sim[:, hand, :, 2].min()),
        )
        required = max(0.0, hand_clearance_m - hand_min)
        rejected = required > max_correction_m
        if rejected:
            roles[hand] = "invalid"
        diagnostics[HAND_ORDER[hand]] = {
            "required_floor_correction_m": required,
            "maximum_allowed_correction_m": max_correction_m,
            "rejected": rejected,
        }
    return roles, diagnostics


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
    """Estimate a conservative episode-level mesh scale from silhouette areas."""
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
    raw_ratio = float(np.median(ratios))
    ratio = float(np.clip(raw_ratio, MIN_GLOBAL_SCALE_RATIO, MAX_GLOBAL_SCALE_RATIO))
    optimized = float(initial_scale_m * ratio)
    return optimized, {
        "initial_scale_to_m": initial_scale_m, "optimized_scale_to_m": optimized,
        "raw_scale_ratio": raw_ratio, "scale_ratio": ratio, "sample_count": len(ratios),
        "ratio_clip": [MIN_GLOBAL_SCALE_RATIO, MAX_GLOBAL_SCALE_RATIO],
        "ratio_was_clipped": bool(not math.isclose(raw_ratio, ratio)),
        "method": "clipped_median_sqrt_observed_to_rendered_mask_area_ratio",
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _fingertip_mask_distances(
    K: np.ndarray, fingertips_camera: np.ndarray, masks: np.ndarray, *, downsample: int = 4,
) -> np.ndarray:
    """Approximate fingertip-to-mask distances without a full-resolution distance transform."""
    fingertips = np.asarray(fingertips_camera, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=bool)
    if fingertips.ndim != 4 or fingertips.shape[2:] != (5, 3):
        raise ValueError(f"fingertips_camera must have shape (T, H, 5, 3), got {fingertips.shape}")
    if len(mask_values) != len(fingertips):
        raise ValueError("fingertips and masks have different timelines")
    pixels = _project(K, fingertips) / float(downsample)
    distances = np.full(fingertips.shape[:3], np.inf, dtype=np.float64)
    for frame, mask in enumerate(mask_values):
        height, width = mask.shape
        target_size = (
            max(1, int(math.ceil(width / downsample))),
            max(1, int(math.ceil(height / downsample))),
        )
        reduced = cv2.resize(
            mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not reduced.any():
            continue
        outside = cv2.distanceTransform(
            (~reduced).astype(np.uint8), cv2.DIST_L2, 5,
        ) * float(downsample)
        rounded = np.rint(pixels[frame]).astype(np.int64)
        for hand in range(fingertips.shape[1]):
            for finger in range(5):
                x, y = rounded[hand, finger]
                if (
                    fingertips[frame, hand, finger, 2] > 0.05
                    and 0 <= x < target_size[0] and 0 <= y < target_size[1]
                ):
                    distances[frame, hand, finger] = float(outside[y, x])
    return distances


def _optimize_contact_similarity(
    canonical_mesh: trimesh.Trimesh, base_scale_m: float, K: np.ndarray,
    poses_camera_object: np.ndarray, fingertips_camera: np.ndarray,
    masks: np.ndarray, object_valid: np.ndarray, hand_valid: np.ndarray,
    hand_confidence: np.ndarray, *, metric_hand_depth_valid: np.ndarray | None = None,
    timestamps_s: np.ndarray | None = None,
    candidate_distance_px: float = CONTACT_CANDIDATE_DISTANCE_PX,
    enter_distance_m: float = 0.012,
    max_contact_slip_p95_m_s: float = MAX_CONTACT_SLIP_P95_M_S,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Resolve monocular object scale from persistent 2D hand-object contact evidence.

    Scaling the canonical mesh and camera-space object translation by the same
    episode-level ratio leaves its image projection unchanged. Search that one
    degree of freedom for a scale that places persistent 2D fingertip candidates
    on the reconstructed object surface while preserving the metric MANO hand.
    """
    poses = np.asarray(poses_camera_object, dtype=np.float64)
    fingertips = np.asarray(fingertips_camera, dtype=np.float64)
    valid_object = np.asarray(object_valid, dtype=bool)
    valid_hand_values = np.asarray(hand_valid, dtype=bool)
    confidence = np.asarray(hand_confidence, dtype=np.float64)
    timestamps = (
        np.arange(len(poses), dtype=np.float64) / 30.0
        if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64)
    )
    if timestamps.shape != (len(poses),):
        raise ValueError("timestamps_s does not match pose timeline")
    trusted_depth = (
        np.ones(fingertips.shape[1], dtype=bool)
        if metric_hand_depth_valid is None else np.asarray(metric_hand_depth_valid, dtype=bool)
    )
    if trusted_depth.shape != (fingertips.shape[1],):
        raise ValueError("metric_hand_depth_valid does not match hand dimension")
    mask_distances = _fingertip_mask_distances(K, fingertips, masks)
    candidate = (
        (mask_distances <= candidate_distance_px)
        & valid_object[:, None, None]
        & valid_hand_values[:, :, None]
        & (confidence[:, :, None] >= 0.5)
        & trusted_depth[None, :, None]
        & np.isfinite(fingertips).all(axis=-1)
        & (fingertips[..., 2] > 0.05)
        & (poses[:, None, None, 2, 3] > 0.05)
    )
    hand_records: dict[str, dict[str, Any]] = {}
    qualifying_hands = np.zeros(fingertips.shape[1], dtype=bool)
    for hand in range(fingertips.shape[1]):
        opposed_frame_candidate = (
            candidate[:, hand, 0] & candidate[:, hand, 1:].any(axis=1)
        )
        opposed_candidate = candidate[:, hand] & opposed_frame_candidate[:, None]
        frame_candidate = opposed_frame_candidate
        sample_count = int(opposed_candidate.sum())
        frame_count = int(frame_candidate.sum())
        longest_run = _longest_true_run(frame_candidate)
        qualifies = bool(
            sample_count >= MIN_CONTACT_CANDIDATE_SAMPLES
            and frame_count >= MIN_CONTACT_CANDIDATE_FRAMES
            and longest_run >= MIN_CONTACT_CANDIDATE_RUN
        )
        qualifying_hands[hand] = qualifies
        side = HAND_ORDER[hand] if hand < len(HAND_ORDER) else str(hand)
        hand_records[side] = {
            "sample_count": sample_count,
            "frame_count": frame_count,
            "longest_consecutive_frame_run": longest_run,
            "qualifies": qualifies,
            "minimum_mask_distance_px": (
                float(np.min(mask_distances[:, hand][np.isfinite(mask_distances[:, hand])]))
                if np.isfinite(mask_distances[:, hand]).any() else None
            ),
            "metric_hand_depth_valid": bool(trusted_depth[hand]),
        }
    opposed_frames = candidate[:, :, 0] & candidate[:, :, 1:].any(axis=2)
    selected = candidate & opposed_frames[:, :, None] & qualifying_hands[None, :, None]
    selected_indices = np.argwhere(selected)
    if not selected_indices.size:
        return 1.0, np.zeros_like(qualifying_hands), {
            "applied": False,
            "reason": (
                "unvalidated_metric_hand_depth"
                if not trusted_depth.any() else
                "insufficient_persistent_opposed_2d_contact_candidates"
            ),
            "candidate_distance_px": candidate_distance_px,
            "hands": hand_records,
            "candidate_sample_count": 0,
            "metric_hand_depth_valid": trusted_depth.tolist(),
        }

    center_depth = poses[selected_indices[:, 0], 2, 3]
    fingertip_depth = fingertips[
        selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2], 2,
    ]
    depth_ratios = fingertip_depth / center_depth
    usable_ratio = (
        np.isfinite(depth_ratios)
        & (depth_ratios >= MIN_CONTACT_SIMILARITY_RATIO)
        & (depth_ratios <= MAX_CONTACT_SIMILARITY_RATIO)
    )
    selected_indices = selected_indices[usable_ratio]
    depth_ratios = depth_ratios[usable_ratio]
    if len(depth_ratios) < MIN_CONTACT_CANDIDATE_SAMPLES:
        return 1.0, np.zeros_like(qualifying_hands), {
            "applied": False,
            "reason": "candidate_depth_ratios_outside_bounds",
            "candidate_distance_px": candidate_distance_px,
            "hands": hand_records,
            "candidate_sample_count": int(len(depth_ratios)),
        }

    p10, p90 = np.percentile(depth_ratios, [10, 90])
    lower = max(MIN_CONTACT_SIMILARITY_RATIO, float(p10) - 0.05)
    upper = min(MAX_CONTACT_SIMILARITY_RATIO, float(p90) + 0.05)
    if upper <= lower:
        lower = max(MIN_CONTACT_SIMILARITY_RATIO, float(np.median(depth_ratios)) - 0.05)
        upper = min(MAX_CONTACT_SIMILARITY_RATIO, float(np.median(depth_ratios)) + 0.05)
    ratios = np.linspace(lower, upper, 81)
    canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float64) * float(base_scale_m)
    frame_groups: dict[int, np.ndarray] = {}
    base_vertices_camera: dict[int, np.ndarray] = {}
    for frame in np.unique(selected_indices[:, 0]):
        frame_groups[int(frame)] = selected_indices[selected_indices[:, 0] == frame]
        pose = poses[frame]
        base_vertices_camera[int(frame)] = canonical_vertices @ pose[:3, :3].T + pose[:3, 3]

    minimum_fit_samples = max(8, int(math.ceil(0.25 * len(selected_indices))))
    inference_valid = valid_hand_values & qualifying_hands[None]
    exit_distance_m = max(0.020, float(enter_distance_m))
    best: tuple[bool, int, int, int, float, int, int, float] | None = None
    best_ratio = 1.0
    best_distances = np.full(len(selected_indices), np.inf, dtype=np.float64)
    best_contact = np.zeros_like(candidate, dtype=bool)
    best_contact_metrics: dict[str, Any] = {}
    for ratio in ratios:
        distances = np.empty(len(selected_indices), dtype=np.float64)
        offset = 0
        for frame, indices in frame_groups.items():
            points = fingertips[frame, indices[:, 1], indices[:, 2]]
            vertices = base_vertices_camera[frame] * ratio
            delta = points[:, None, :] - vertices[None, :, :]
            count = len(indices)
            distances[offset : offset + count] = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)
            offset += count
        ratio_mesh = canonical_mesh.copy()
        ratio_mesh.apply_scale(float(base_scale_m * ratio))
        ratio_poses = poses.copy()
        ratio_poses[:, :3, 3] *= ratio
        inferred_contact, _, inferred_metrics = infer_contact(
            ratio_mesh, ratio_poses, fingertips, timestamps, inference_valid,
            enter_distance_m=enter_distance_m, exit_distance_m=exit_distance_m,
        )
        inferred = inferred_contact.astype(bool)
        opposed_contact = inferred[:, :, 0] & inferred[:, :, 1:].any(axis=2)
        longest_opposed_run = max(
            (_longest_true_run(opposed_contact[:, hand]) for hand in range(inferred.shape[1])),
            default=0,
        )
        opposed_frame_count = int(opposed_contact.sum())
        contact_sample_count = int(inferred.sum())
        slip = inferred_metrics.get("contact_local_slip_p95_m_s")
        slip_value = float(slip) if slip is not None else math.inf
        enter_count = int(np.count_nonzero(distances <= enter_distance_m))
        exit_count = int(np.count_nonzero(distances <= 0.020))
        truncated_loss = float(np.mean(np.minimum(distances, 0.030)))
        supported = bool(
            contact_sample_count >= minimum_fit_samples
            and opposed_frame_count >= MIN_CONTACT_OPPOSED_FRAMES
            and longest_opposed_run >= MIN_CONTACT_OPPOSED_RUN
            and slip_value <= max_contact_slip_p95_m_s
        )
        score = (
            supported, int(longest_opposed_run), opposed_frame_count,
            contact_sample_count, -slip_value, enter_count, exit_count, -truncated_loss,
        )
        if best is None or score > best:
            best = score
            best_ratio = float(ratio)
            best_distances = distances
            best_contact = inferred
            best_contact_metrics = inferred_metrics

    candidate_enter_contact = best_distances <= enter_distance_m
    opposed_fit = best_contact[:, :, 0] & best_contact[:, :, 1:].any(axis=2)
    fitted_opposed_frames = np.flatnonzero(opposed_fit.any(axis=1))
    longest_opposed_run = max(
        (_longest_true_run(opposed_fit[:, hand]) for hand in range(best_contact.shape[1])),
        default=0,
    )
    contact_frames = np.flatnonzero(best_contact.any(axis=(1, 2)))
    contact_sample_count = int(best_contact.sum())
    contact_slip = best_contact_metrics.get("contact_local_slip_p95_m_s")
    fit_supported = bool(
        contact_sample_count >= minimum_fit_samples and len(contact_frames) >= 2
        and len(fitted_opposed_frames) >= MIN_CONTACT_OPPOSED_FRAMES
        and longest_opposed_run >= MIN_CONTACT_OPPOSED_RUN
        and contact_slip is not None
        and float(contact_slip) <= max_contact_slip_p95_m_s
    )
    activity_hint = np.zeros(fingertips.shape[1], dtype=bool)
    if fit_supported:
        for hand in range(best_contact.shape[1]):
            activity_hint[hand] = (
                _longest_true_run(opposed_fit[:, hand]) >= MIN_CONTACT_OPPOSED_RUN
            )
    return (best_ratio if fit_supported else 1.0), activity_hint, {
        "applied": fit_supported,
        "reason": None if fit_supported else "surface_fit_has_insufficient_contact_support",
        "method": "persistent_2d_candidates_plus_hysteretic_contact_grid_search",
        "candidate_distance_px": candidate_distance_px,
        "enter_distance_m": enter_distance_m,
        "exit_distance_m": exit_distance_m,
        "maximum_contact_local_slip_p95_m_s": max_contact_slip_p95_m_s,
        "ratio_bounds": [MIN_CONTACT_SIMILARITY_RATIO, MAX_CONTACT_SIMILARITY_RATIO],
        "search_interval": [lower, upper],
        "raw_depth_ratio_median": float(np.median(depth_ratios)),
        "raw_depth_ratio_p10": float(p10),
        "raw_depth_ratio_p90": float(p90),
        "optimized_similarity_ratio": best_ratio,
        "candidate_sample_count": int(len(selected_indices)),
        "surface_contact_sample_count": contact_sample_count,
        "surface_contact_frame_count": int(len(contact_frames)),
        "surface_opposed_frame_count": int(len(fitted_opposed_frames)),
        "surface_opposed_longest_run": int(longest_opposed_run),
        "surface_candidate_enter_sample_count": int(candidate_enter_contact.sum()),
        "contact_local_slip_p95_m_s": contact_slip,
        "minimum_required_surface_samples": minimum_fit_samples,
        "minimum_required_opposed_frames": MIN_CONTACT_OPPOSED_FRAMES,
        "minimum_required_opposed_run": MIN_CONTACT_OPPOSED_RUN,
        "surface_distance_median_m": float(np.median(best_distances)),
        "surface_distance_p25_m": float(np.percentile(best_distances, 25)),
        "hands": hand_records,
        "activity_hint": {
            HAND_ORDER[hand]: bool(activity_hint[hand]) for hand in range(len(activity_hint))
        },
        "absolute_metric_scale_status": "contact_calibrated_not_externally_validated",
        "metric_hand_depth_valid": trusted_depth.tolist(),
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
    # Full-resolution CPU triangle rasterization dominates optimizer runtime;
    # twelve evenly spaced frames retain episode-wide QC coverage.
    sampled = _sample_indices(valid, 12)
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


def _hand_reprojection_metrics(
    K: np.ndarray, raw_fingertips_camera: np.ndarray,
    exported_fingertips_camera: np.ndarray, valid_hand: np.ndarray,
    hand_order: list[str],
) -> dict[str, Any]:
    """Measure whether exported hand targets still match the WiLoR image evidence."""
    raw = np.asarray(raw_fingertips_camera, dtype=np.float64)
    exported = np.asarray(exported_fingertips_camera, dtype=np.float64)
    valid = np.asarray(valid_hand, dtype=bool)
    if raw.shape != exported.shape or raw.shape[:2] != valid.shape:
        raise ValueError("hand reprojection inputs have incompatible shapes")
    pixel_error = np.linalg.norm(_project(K, exported) - _project(K, raw), axis=-1)
    positive_depth = (raw[..., 2] > 0.05) & (exported[..., 2] > 0.05)
    finite = np.isfinite(pixel_error) & np.isfinite(raw).all(axis=-1) & np.isfinite(exported).all(axis=-1)
    per_hand: dict[str, dict[str, Any]] = {}
    all_errors: list[np.ndarray] = []
    for hand, side in enumerate(hand_order):
        usable = valid[:, hand, None] & positive_depth[:, hand] & finite[:, hand]
        values = pixel_error[:, hand][usable]
        median = float(np.median(values)) if values.size else None
        p95 = float(np.percentile(values, 95)) if values.size else None
        passed = bool(
            median is not None and p95 is not None
            and median <= MAX_HAND_EXPORT_MEDIAN_REPROJECTION_PX
            and p95 <= MAX_HAND_EXPORT_P95_REPROJECTION_PX
        )
        per_hand[side] = {
            "sample_count": int(values.size),
            "median_px": median,
            "p95_px": p95,
            "passed": passed,
        }
        if values.size:
            all_errors.append(values)
    combined = np.concatenate(all_errors) if all_errors else np.zeros(0, dtype=np.float64)
    return {
        "per_hand": per_hand,
        "median_px": float(np.median(combined)) if combined.size else None,
        "p95_px": float(np.percentile(combined, 95)) if combined.size else None,
        "threshold_px": {
            "median": MAX_HAND_EXPORT_MEDIAN_REPROJECTION_PX,
            "p95": MAX_HAND_EXPORT_P95_REPROJECTION_PX,
        },
        "passed": bool(per_hand and all(record["passed"] for record in per_hand.values())),
        "reference": "raw WiLoR fingertip projections",
    }


def _manipulation_contact_metrics(
    contact: np.ndarray, hand_roles: list[str], hand_order: list[str], *,
    require_contact: bool, contact_local_slip_p95_m_s: float | None = None,
    max_contact_slip_p95_m_s: float = MAX_CONTACT_SLIP_P95_M_S,
) -> dict[str, Any]:
    values = np.asarray(contact) >= 0.5
    if values.ndim != 3 or values.shape[1] != len(hand_roles) or len(hand_roles) != len(hand_order):
        raise ValueError("contact, hand roles, and hand order are inconsistent")
    per_hand: dict[str, dict[str, Any]] = {}
    active_contact_frames = 0
    active_contact_samples = 0
    longest_active_contact_run = 0
    active_opposed_frames = 0
    longest_active_opposed_run = 0
    for hand, (side, role) in enumerate(zip(hand_order, hand_roles)):
        frame_contact = values[:, hand].any(axis=1)
        opposed_contact = values[:, hand, 0] & values[:, hand, 1:].any(axis=1)
        frame_count = int(frame_contact.sum())
        sample_count = int(values[:, hand].sum())
        if role == "active":
            active_contact_frames += frame_count
            active_contact_samples += sample_count
            longest_active_contact_run = max(
                longest_active_contact_run, _longest_true_run(frame_contact),
            )
            active_opposed_frames += int(opposed_contact.sum())
            longest_active_opposed_run = max(
                longest_active_opposed_run, _longest_true_run(opposed_contact),
            )
        per_hand[side] = {
            "role": role,
            "contact_rate": float(values[:, hand].mean()),
            "contact_frame_count": frame_count,
            "contact_sample_count": sample_count,
            "longest_consecutive_contact_run": _longest_true_run(frame_contact),
        }
    active_hands = [side for side, role in zip(hand_order, hand_roles) if role == "active"]
    contact_present = bool(
        active_hands
        and active_contact_frames >= 2
        and active_contact_samples >= 4
        and longest_active_contact_run >= MIN_CONTACT_OPPOSED_RUN
        and active_opposed_frames >= MIN_CONTACT_OPPOSED_FRAMES
        and longest_active_opposed_run >= MIN_CONTACT_OPPOSED_RUN
        and (
            contact_local_slip_p95_m_s is None
            or contact_local_slip_p95_m_s <= max_contact_slip_p95_m_s
        )
    )
    return {
        "required": require_contact,
        "passed": bool(contact_present or not require_contact),
        "active_hands": active_hands,
        "active_contact_frame_count": active_contact_frames,
        "active_contact_sample_count": active_contact_samples,
        "active_opposed_frame_count": active_opposed_frames,
        "longest_active_contact_run": longest_active_contact_run,
        "longest_active_opposed_run": longest_active_opposed_run,
        "contact_local_slip_p95_m_s": contact_local_slip_p95_m_s,
        "minimum_active_contact_frames": 2,
        "minimum_active_contact_samples": 4,
        "minimum_active_opposed_frames": MIN_CONTACT_OPPOSED_FRAMES,
        "minimum_consecutive_active_contact_frames": MIN_CONTACT_OPPOSED_RUN,
        "maximum_contact_local_slip_p95_m_s": max_contact_slip_p95_m_s,
        "per_hand": per_hand,
    }


def _optimization_quality_control(
    object_translation_raw: np.ndarray, object_translation_aligned: np.ndarray,
    scale_metrics: dict[str, Any], render_metrics: dict[str, Any],
    hand_reprojection_metrics: dict[str, Any] | None = None,
    contact_similarity_metrics: dict[str, Any] | None = None,
    manipulation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usable = (
        np.isfinite(object_translation_raw).all(axis=1)
        & np.isfinite(object_translation_aligned).all(axis=1)
        & (object_translation_raw[:, 2] > 0.05)
        & (object_translation_aligned[:, 2] > 0.05)
    )
    depth_ratios = (
        object_translation_aligned[usable, 2] / object_translation_raw[usable, 2]
        if usable.any() else np.zeros(0, dtype=np.float64)
    )
    median_depth_ratio = float(np.median(depth_ratios)) if depth_ratios.size else None
    depth_scale_preserved = bool(
        median_depth_ratio is not None
        and MIN_OBJECT_DEPTH_RATIO <= median_depth_ratio <= MAX_OBJECT_DEPTH_RATIO
    )
    raw_residual = render_metrics.get("relative_depth_residual_raw")
    aligned_residual = render_metrics.get("relative_depth_residual_aligned")
    metric_depth_preserved = True
    if isinstance(raw_residual, (int, float)) and isinstance(aligned_residual, (int, float)):
        metric_depth_preserved = bool(aligned_residual <= max(float(raw_residual) * 3.0, float(raw_residual) + 0.10))
    contact_override = bool(contact_similarity_metrics and contact_similarity_metrics.get("applied"))
    scale_ratio = float(scale_metrics["scale_ratio"])
    if contact_override:
        similarity_ratio = float(contact_similarity_metrics["optimized_similarity_ratio"])
        scale_supported = bool(
            MIN_CONTACT_SIMILARITY_RATIO <= similarity_ratio <= MAX_CONTACT_SIMILARITY_RATIO
            and int(contact_similarity_metrics["surface_contact_sample_count"])
            >= int(contact_similarity_metrics["minimum_required_surface_samples"])
            and int(contact_similarity_metrics["surface_opposed_frame_count"])
            >= int(contact_similarity_metrics["minimum_required_opposed_frames"])
            and int(contact_similarity_metrics["surface_opposed_longest_run"])
            >= int(contact_similarity_metrics["minimum_required_opposed_run"])
            and any(contact_similarity_metrics["metric_hand_depth_valid"])
        )
        checks = {
            "contact_similarity_scale_supported": scale_supported,
        }
    else:
        scale_supported = bool(MIN_GLOBAL_SCALE_RATIO <= scale_ratio <= MAX_GLOBAL_SCALE_RATIO)
        checks = {
            "camera_depth_scale_preserved": depth_scale_preserved,
            "metric_depth_residual_preserved": metric_depth_preserved,
            "global_scale_within_bounds": scale_supported,
        }
    if hand_reprojection_metrics is not None:
        checks["hand_image_alignment_preserved"] = bool(hand_reprojection_metrics["passed"])
    if manipulation_metrics is not None:
        checks["manipulation_contact_present"] = bool(manipulation_metrics["passed"])
    return {
        "export_ready": bool(all(checks.values())),
        "checks": checks,
        "median_aligned_to_raw_camera_depth_ratio": median_depth_ratio,
        "camera_depth_ratio_bounds": [MIN_OBJECT_DEPTH_RATIO, MAX_OBJECT_DEPTH_RATIO],
        "metric_depth_residual_policy": "aligned <= max(3 * raw, raw + 0.10)",
        "metric_depth_preserved": metric_depth_preserved,
        "metric_depth_overridden_by_contact": contact_override,
        "scale_policy": (
            "persistent 2D contact plus 3D surface fit overrides monocular metric depth"
            if contact_override else "preserve FoundationPose/Depth Anything metric scale"
        ),
        "hand_reprojection": hand_reprojection_metrics,
        "manipulation": manipulation_metrics,
    }


def optimize_run(
    run_dir: str | Path, *, smoothing_strength: float = 18.0,
    hand_smoothing_strength: float = 5.0, require_contact: bool = True,
    allow_unvalidated_contact_scale: bool = False,
    contact_enter_distance_m: float = 0.012,
    max_contact_slip_p95_m_s: float = MAX_CONTACT_SLIP_P95_M_S,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    root = Path(run_dir).resolve()
    output_dir = root / "optimization"
    aligned_path = output_dir / "aligned_trajectory.npz"
    contact_path = output_dir / "contact.npz"
    if aligned_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {aligned_path}; pass --overwrite")
    if not 0.0 < contact_enter_distance_m <= 0.05:
        raise ValueError("contact_enter_distance_m must be in (0, 0.05]")
    if not 0.0 < max_contact_slip_p95_m_s <= 2.0:
        raise ValueError("max_contact_slip_p95_m_s must be in (0, 2.0]")
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
    hand_valid = hands_raw["valid"].astype(bool)
    hand_confidence = hands_raw["score"].astype(np.float64)
    object_valid = object_raw["valid"].astype(bool)
    mask_valid = masks_raw["valid"].astype(bool)
    joints_camera_raw = (
        hands_raw["joints_camera_rootrel"].astype(np.float64)
        + hands_raw["translation_camera"].astype(np.float64)[:, :, None, :]
    )
    centroids = _mask_centroids(masks_raw["masks"].astype(bool))
    depth_group = None
    depth_path = root / "depth/metric_depth.zarr"
    if depth_path.exists():
        import zarr
        depth_group = zarr.open(str(depth_path), mode="r")
    metric_mask_depth = _metric_mask_depths(
        depth_group, frame_indices, masks_raw["masks"].astype(bool), mask_valid,
    )
    joints_camera, hand_depth_calibration = calibrate_hand_depth_scale(
        K, joints_camera_raw, hand_valid, hand_confidence, object_translation_raw,
        centroids, object_valid & mask_valid,
    )
    observations, observation_weights, anchor_metrics = hand_anchored_object_observation(
        K, object_translation_raw, centroids, joints_camera, hands_raw["valid"], hands_raw["score"],
        object_valid=object_valid, object_confidence=object_raw["confidence"],
        mask_valid=mask_valid, metric_mask_depth=metric_mask_depth,
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
    silhouette_scale_to_m, silhouette_scale_metrics = _optimize_global_scale(
        canonical_mesh, initial_scale_to_m, K, T_camera_object_aligned,
        masks_raw["masks"].astype(bool), scale_valid,
    )

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
    metric_hand_depth_valid = np.asarray([
        bool(hand_depth_calibration["per_hand"][side]["accepted"])
        for side in HAND_ORDER
    ], dtype=bool)
    if allow_unvalidated_contact_scale:
        metric_hand_depth_valid = np.asarray(hand_valid.any(axis=0), dtype=bool)
    contact_similarity_ratio, active_hint, contact_similarity_metrics = _optimize_contact_similarity(
        canonical_mesh, silhouette_scale_to_m, K, T_camera_object_aligned,
        fingertips_camera_aligned, masks_raw["masks"].astype(bool),
        object_valid & mask_valid, hand_valid, hand_confidence,
        metric_hand_depth_valid=metric_hand_depth_valid,
        timestamps_s=timestamps,
        enter_distance_m=contact_enter_distance_m,
        max_contact_slip_p95_m_s=max_contact_slip_p95_m_s,
    )
    scale_to_m = float(silhouette_scale_to_m * contact_similarity_ratio)
    T_camera_object_aligned[:, :3, 3] *= contact_similarity_ratio
    scale_metrics = {
        "initial_scale_to_m": initial_scale_to_m,
        "silhouette_optimized_scale_to_m": silhouette_scale_to_m,
        "silhouette_raw_scale_ratio": silhouette_scale_metrics.get("raw_scale_ratio"),
        "silhouette_scale_ratio": silhouette_scale_metrics["scale_ratio"],
        "silhouette_ratio_clip": silhouette_scale_metrics.get("ratio_clip"),
        "silhouette_ratio_was_clipped": silhouette_scale_metrics.get("ratio_was_clipped", False),
        "silhouette_sample_count": silhouette_scale_metrics["sample_count"],
        "contact_similarity_ratio": contact_similarity_ratio,
        "optimized_scale_to_m": scale_to_m,
        "scale_ratio": float(scale_to_m / initial_scale_to_m),
        "scale_ratio_semantics": "final optimized scale divided by selected-mesh initial scale",
        "final_similarity_ratio_bounds": (
            [MIN_CONTACT_SIMILARITY_RATIO, MAX_CONTACT_SIMILARITY_RATIO]
            if contact_similarity_metrics["applied"]
            else [MIN_GLOBAL_SCALE_RATIO, MAX_GLOBAL_SCALE_RATIO]
        ),
        "method": (
            "silhouette_scale_then_contact_similarity"
            if contact_similarity_metrics["applied"] else silhouette_scale_metrics["method"]
        ),
    }
    mesh_m = canonical_mesh.copy()
    mesh_m.apply_scale(scale_to_m)
    T_world_object = np.einsum("tij,tjk->tik", T_world_camera, T_camera_object_aligned)
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
    T_sim_world = _build_T_sim_world(
        T_world_object, mesh_m, wrist_world=wrist_world,
        fingertips_world=fingertips_world, active_hint=active_hint,
    )
    T_sim_object = _transform_series(T_sim_world, T_world_object)
    T_sim_wrist = np.einsum("ij,thjk->thik", T_sim_world, wrist_world)
    fingertips_sim = np.einsum(
        "ij,thfj->thfi", T_sim_world,
        np.concatenate([fingertips_world, np.ones((*fingertips_world.shape[:-1], 1))], axis=-1),
    )[..., :3]
    hand_roles, hand_role_metrics = classify_hand_roles(
        T_sim_object, fingertips_sim, hand_valid, scale_to_m, active_hint,
    )
    hand_roles, passive_hand_qc = reject_unphysical_passive_hands(
        T_sim_wrist, fingertips_sim, hand_roles,
    )
    T_sim_object, T_sim_wrist, fingertips_sim, floor_metrics = enforce_simulation_floor(
        mesh_m, T_sim_object, T_sim_wrist, fingertips_sim, hand_roles,
    )
    T_sim_camera = np.einsum("ij,tjk->tik", T_sim_world, T_world_camera)
    T_camera_object_exported = np.einsum(
        "tij,tjk->tik", np.linalg.inv(T_sim_camera), T_sim_object,
    )
    retained_hands = np.asarray([
        index for index, role in enumerate(hand_roles) if role != "invalid"
    ], dtype=np.int64)
    if retained_hands.size == 0:
        raise ValueError("sequence optimization requires at least one reconstructed hand")
    artifact_hand_order = [HAND_ORDER[index] for index in retained_hands]
    shared_betas = np.stack([
        np.median(hands_raw["mano_betas"][hand_valid[:, hand], hand], axis=0)
        if hand_valid[:, hand].any() else np.zeros(10)
        for hand in range(2)
    ])[retained_hands]
    aligned = {
        "frame_indices": frame_indices, "timestamps_s": timestamps,
        "T_sim_object": T_sim_object[:, None].astype(np.float32),
        "T_sim_wrist": T_sim_wrist[:, retained_hands].astype(np.float32),
        "fingertips_sim": fingertips_sim[:, retained_hands].astype(np.float32),
        "mano_pose": hands_raw["mano_hand_pose"][:, retained_hands].astype(np.float32),
        "mano_betas": shared_betas.astype(np.float32),
        "object_scale_to_m": np.array([scale_to_m], dtype=np.float32),
        "valid_object": object_raw["valid"][:, None].astype(bool),
        "valid_hand": hand_valid[:, retained_hands],
        "confidence_object": object_raw["confidence"][:, None].astype(np.float32),
        "confidence_hand": hand_confidence[:, retained_hands].astype(np.float32),
    }
    validate_aligned_trajectory(aligned)
    contact, contact_positions, contact_metrics = infer_contact(
        mesh_m, T_sim_object, fingertips_sim[:, retained_hands], timestamps,
        hand_valid[:, retained_hands],
        enter_distance_m=contact_enter_distance_m,
        exit_distance_m=max(0.020, contact_enter_distance_m),
    )
    retained_roles = [hand_roles[index] for index in retained_hands]
    manipulation_metrics = _manipulation_contact_metrics(
        contact, retained_roles, artifact_hand_order, require_contact=require_contact,
        contact_local_slip_p95_m_s=contact_metrics.get("contact_local_slip_p95_m_s"),
        max_contact_slip_p95_m_s=max_contact_slip_p95_m_s,
    )
    T_sim_object_raw = np.einsum("ij,tjk,tkl->til", T_sim_world, T_world_camera, T_camera_object_raw)
    raw_fingertip_homogeneous = np.concatenate([
        fingertips_camera, np.ones((*fingertips_camera.shape[:-1], 1))
    ], axis=-1)
    fingertips_sim_raw = np.einsum(
        "tij,thfj->thfi", T_sim_camera, raw_fingertip_homogeneous,
    )[..., :3]
    _, _, raw_contact_metrics = infer_contact(
        mesh_m, T_sim_object_raw, fingertips_sim_raw[:, retained_hands], timestamps,
        hand_valid[:, retained_hands],
        enter_distance_m=contact_enter_distance_m,
        exit_distance_m=max(0.020, contact_enter_distance_m),
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
    exported_translation_camera = T_camera_object_exported[:, :3, 3]
    aligned_reprojection = _reprojection_residual(K, exported_translation_camera, centroids)
    render_metrics = _render_comparison(
        canonical_mesh, K, masks_raw["masks"].astype(bool), scale_valid,
        T_camera_object_raw, initial_scale_to_m, T_camera_object_exported,
        scale_to_m, depth_group,
    )
    exported_fingertip_homogeneous = np.concatenate([
        fingertips_sim[:, retained_hands],
        np.ones((*fingertips_sim[:, retained_hands].shape[:-1], 1)),
    ], axis=-1)
    exported_fingertips_camera = np.einsum(
        "tij,thfj->thfi", np.linalg.inv(T_sim_camera), exported_fingertip_homogeneous,
    )[..., :3]
    hand_reprojection_metrics = _hand_reprojection_metrics(
        K, joints_camera_raw[:, retained_hands][:, :, FINGERTIP_INDICES],
        exported_fingertips_camera, hand_valid[:, retained_hands], artifact_hand_order,
    )
    quality_control = _optimization_quality_control(
        object_translation_raw, exported_translation_camera, scale_metrics, render_metrics,
        hand_reprojection_metrics, contact_similarity_metrics, manipulation_metrics,
    )
    raw_slip = raw_contact_metrics.get("contact_local_slip_p95_m_s")
    aligned_slip = contact_metrics.get("contact_local_slip_p95_m_s")
    metrics = {
        "schema_version": SCHEMA_VERSION, "T_sim_world": T_sim_world.tolist(),
        "hand_order": HAND_ORDER, "left_absolute_confidence_multiplier": 0.45,
        "hands": {
            "roles": dict(zip(HAND_ORDER, hand_roles)),
            "artifact_hand_order": artifact_hand_order,
            "role_policy": "visible reliable hands are retained; activity only controls interaction constraints",
            "role_metrics": hand_role_metrics,
            "passive_hand_qc": passive_hand_qc,
            "wrist_position_source": "translated MANO wrist joint 0",
            "wrist_orientation_source": "landmark-derived xHand palm frame",
            "depth_calibration": hand_depth_calibration,
            "reprojection": hand_reprojection_metrics,
        },
        "simulation_floor": floor_metrics,
        "anchor": anchor_metrics,
        "raw_vs_aligned": {
            "object_translation_acceleration_jitter_raw_m_s2": _acceleration_jitter(object_translation_raw, timestamps),
            "object_translation_acceleration_jitter_aligned_m_s2": _acceleration_jitter(exported_translation_camera, timestamps),
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
            "translation_objective": "mask-ray smoothing followed by contact-aware global similarity calibration",
            "smoothing_strength": smoothing_strength, "hand_smoothing_strength": hand_smoothing_strength,
            "depth_policy": "monocular metric depth is a prior; persistent contact may override it with a documented global similarity",
            "global_object_scale": scale_metrics,
            "contact_similarity": contact_similarity_metrics,
        },
        "quality_control": quality_control,
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
        "hand_smoothing_strength": hand_smoothing_strength,
        "require_contact": require_contact,
        "allow_unvalidated_contact_scale": allow_unvalidated_contact_scale,
        "contact_enter_distance_m": contact_enter_distance_m,
        "max_contact_slip_p95_m_s": max_contact_slip_p95_m_s},
        [root / "object_tracking/foundationpose_raw.npz", root / "object_tracking/selected_mesh.json",
         root / "hands/wilor_raw.npz", root / "segmentation/object_masks.npz"],
    )
    manifest.start_stage("sequence_optimization", cache_key=cache_key, command=sys.argv, environment="v2s-opt")
    manifest.finish_stage(
        "sequence_optimization", success=bool(quality_control["export_ready"]),
        outputs=[
            str(aligned_path.relative_to(root)), str(contact_path.relative_to(root)),
            str(metrics_path.relative_to(root)), *visualization_outputs,
        ],
        quality_metrics=metrics["raw_vs_aligned"],
        warnings=["left absolute hand confidence is reduced from its measured GT diagnostic",
                  "penetration is an unsigned nearest-surface proxy in the current V1 optimizer",
                  *([] if quality_control["export_ready"] else ["optimization failed object, hand, or manipulation export QC"]),
                  *visualization_warnings],
    )
    if not quality_control["export_ready"]:
        raise RuntimeError("sequence_optimization_failed: aligned trajectory did not pass export QC")
    return aligned_path, contact_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoothing-strength", type=float, default=18.0)
    parser.add_argument("--hand-smoothing-strength", type=float, default=5.0)
    parser.add_argument("--allow-no-contact", action="store_true")
    parser.add_argument("--allow-unvalidated-contact-scale", action="store_true")
    parser.add_argument("--contact-enter-distance-m", type=float, default=0.012)
    parser.add_argument("--max-contact-slip-p95-m-s", type=float, default=MAX_CONTACT_SLIP_P95_M_S)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    aligned, contact = optimize_run(
        args.run_dir, smoothing_strength=args.smoothing_strength,
        hand_smoothing_strength=args.hand_smoothing_strength,
        require_contact=not args.allow_no_contact,
        allow_unvalidated_contact_scale=args.allow_unvalidated_contact_scale,
        contact_enter_distance_m=args.contact_enter_distance_m,
        max_contact_slip_p95_m_s=args.max_contact_slip_p95_m_s,
        overwrite=args.overwrite,
    )
    print(aligned)
    print(contact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
