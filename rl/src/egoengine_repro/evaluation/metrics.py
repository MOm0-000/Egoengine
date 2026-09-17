"""Metric implementations used by the 3.1 offline evaluator."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(finite.size), "mean": float(finite.mean()),
        "median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)),
    }


def rotation_geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(np.asarray(a), -1, -2) @ np.asarray(b)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(relative.shape[:-2])


def project_points(K: np.ndarray, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    homogeneous = np.einsum("...ij,...kj->...ki", np.asarray(K, dtype=np.float64), points)
    valid = np.isfinite(points).all(axis=-1) & (homogeneous[..., 2] > 1e-8)
    uv = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    uv[valid] = homogeneous[..., :2][valid] / homogeneous[..., 2:3][valid]
    return uv, valid


def mask_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, predicted_valid: np.ndarray | None = None,
    ground_truth_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"mask arrays must share shape (T,H,W), got {pred.shape} and {gt.shape}")
    valid_p = pred.reshape(len(pred), -1).any(axis=1) if predicted_valid is None else np.asarray(predicted_valid, bool)
    valid_g = gt.reshape(len(gt), -1).any(axis=1) if ground_truth_valid is None else np.asarray(ground_truth_valid, bool)
    valid = valid_p & valid_g
    intersection = np.logical_and(pred, gt).sum(axis=(1, 2))
    union = np.logical_or(pred, gt).sum(axis=(1, 2))
    iou = np.divide(intersection, union, out=np.full(len(pred), np.nan), where=union > 0)
    consecutive = []
    for frame in range(1, len(pred)):
        if not (valid_p[frame - 1] and valid_p[frame]):
            continue
        inter = np.logical_and(pred[frame - 1], pred[frame]).sum()
        joined = np.logical_or(pred[frame - 1], pred[frame]).sum()
        if joined:
            consecutive.append(inter / joined)
    return {
        "iou": summary(iou[valid]),
        "prediction_valid_rate": float(valid_p.mean()),
        "ground_truth_valid_rate": float(valid_g.mean()),
        "paired_valid_rate": float(valid.mean()),
        "temporal_consecutive_iou": summary(np.asarray(consecutive)),
    }


def depth_metrics(predicted: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"depth arrays must share shape, got {pred.shape} and {gt.shape}")
    paired = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if valid is not None:
        paired &= np.asarray(valid, dtype=bool)
    if not paired.any():
        return {"valid_ratio": 0.0, "abs_rel": summary(np.zeros(0)), "scale_ratio": summary(np.zeros(0))}
    return {
        "valid_ratio": float(paired.mean()),
        "abs_rel": summary(np.abs(pred[paired] - gt[paired]) / gt[paired]),
        "scale_ratio": summary(pred[paired] / gt[paired]),
        "absolute_error_m": summary(np.abs(pred[paired] - gt[paired])),
    }


def depth_warp_residual(
    depth: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, *, stride: int = 8,
) -> dict[str, Any]:
    values = np.asarray(depth, dtype=np.float64)
    transforms = np.asarray(T_world_camera, dtype=np.float64)
    if values.ndim != 3 or transforms.shape != (len(values), 4, 4):
        raise ValueError("depth warp expects depth (T,H,W) and T_world_camera (T,4,4)")
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(values), axis=0)
    if intrinsics.shape != (len(values), 3, 3):
        raise ValueError("intrinsics must have shape (3,3) or (T,3,3)")
    residuals: list[np.ndarray] = []
    height, width = values.shape[1:]
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    for frame in range(len(values) - 1):
        sampled = values[frame, yy, xx].ravel()
        usable = np.isfinite(sampled) & (sampled > 0)
        if not usable.any():
            continue
        rays = (np.linalg.inv(intrinsics[frame]) @ pixels[usable].T).T
        camera_points = rays * sampled[usable, None]
        world_points = camera_points @ transforms[frame, :3, :3].T + transforms[frame, :3, 3]
        R_next_world = transforms[frame + 1, :3, :3].T
        next_points = (world_points - transforms[frame + 1, :3, 3]) @ R_next_world.T
        projected = (intrinsics[frame + 1] @ next_points.T).T
        positive = projected[:, 2] > 1e-8
        uv = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)).astype(np.int64)
        inside = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if not inside.any():
            continue
        observed = values[frame + 1, uv[inside, 1], uv[inside, 0]]
        expected = next_points[inside, 2]
        finite = np.isfinite(observed) & (observed > 0) & (expected > 0)
        if finite.any():
            residuals.append(np.abs(observed[finite] - expected[finite]) / expected[finite])
    merged = np.concatenate(residuals) if residuals else np.zeros(0)
    return {"relative_residual": summary(merged), "stride": int(stride)}


def _sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("mesh has no finite surface area")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=count, p=area / area.sum())]
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def mesh_metrics(
    predicted: trimesh.Trimesh, ground_truth: trimesh.Trimesh, *, sample_count: int = 5000,
    fscore_thresholds_m: Iterable[float] = (0.005, 0.01),
) -> dict[str, Any]:
    pred_points = _sample_mesh(predicted, sample_count, seed=17)
    gt_points = _sample_mesh(ground_truth, sample_count, seed=23)
    pred_to_gt = cKDTree(gt_points).query(pred_points, workers=-1)[0]
    gt_to_pred = cKDTree(pred_points).query(gt_points, workers=-1)[0]
    fscores: dict[str, float] = {}
    completeness: dict[str, float] = {}
    for threshold in fscore_thresholds_m:
        precision = float(np.mean(pred_to_gt <= threshold))
        recall = float(np.mean(gt_to_pred <= threshold))
        fscores[f"{float(threshold):.6f}"] = 2 * precision * recall / max(precision + recall, 1e-12)
        completeness[f"{float(threshold):.6f}"] = recall
    pred_diag = float(np.linalg.norm(np.asarray(predicted.bounds[1]) - np.asarray(predicted.bounds[0])))
    gt_diag = float(np.linalg.norm(np.asarray(ground_truth.bounds[1]) - np.asarray(ground_truth.bounds[0])))
    return {
        "chamfer_l1_m": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "pred_to_gt_m": summary(pred_to_gt), "gt_to_pred_m": summary(gt_to_pred),
        "fscore": fscores, "completeness": completeness,
        "scale_diagonal_pred_m": pred_diag, "scale_diagonal_gt_m": gt_diag,
        "scale_relative_error": abs(pred_diag - gt_diag) / max(gt_diag, 1e-12),
        "sample_count": int(sample_count),
    }


def object_trajectory_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray,
    *, mesh_points: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    paired = np.asarray(valid, dtype=bool)
    if pred.shape != gt.shape or pred.shape[1:] != (4, 4) or paired.shape != (len(pred),):
        raise ValueError("object trajectory inputs have incompatible shapes")
    translation = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    rotation = rotation_geodesic(pred[:, :3, :3], gt[:, :3, :3])
    jump_t = np.linalg.norm(np.diff(pred[:, :3, 3], axis=0), axis=1)
    jump_r = rotation_geodesic(pred[:-1, :3, :3], pred[1:, :3, :3])
    metrics: dict[str, Any] = {
        "translation_error_m": summary(translation[paired]),
        "rotation_geodesic_rad": summary(rotation[paired]),
        "valid_rate": float(paired.mean()),
        "translation_jump_m": summary(jump_t[paired[:-1] & paired[1:]]),
        "rotation_jump_rad": summary(jump_r[paired[:-1] & paired[1:]]),
    }
    if mesh_points is not None:
        points = np.asarray(mesh_points, dtype=np.float64)
        add, adds = [], []
        for frame in np.flatnonzero(paired):
            pred_world = points @ pred[frame, :3, :3].T + pred[frame, :3, 3]
            gt_world = points @ gt[frame, :3, :3].T + gt[frame, :3, 3]
            add.append(np.linalg.norm(pred_world - gt_world, axis=1).mean())
            adds.append(cKDTree(gt_world).query(pred_world, workers=-1)[0].mean())
        metrics["add_m"] = summary(np.asarray(add))
        metrics["add_s_m"] = summary(np.asarray(adds))
    return metrics


def contact_metrics(predicted: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"contact arrays must share shape, got {pred.shape} and {gt.shape}")
    use = np.ones(pred.shape, dtype=bool) if valid is None else np.broadcast_to(np.asarray(valid, bool), pred.shape)
    tp = int(np.count_nonzero(use & pred & gt))
    fp = int(np.count_nonzero(use & pred & ~gt))
    fn = int(np.count_nonzero(use & ~pred & gt))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def contact_slip(
    fingertips: np.ndarray, object_poses: np.ndarray, contact: np.ndarray, timestamps_s: np.ndarray,
) -> dict[str, Any]:
    tips = np.asarray(fingertips, dtype=np.float64)
    poses = np.asarray(object_poses, dtype=np.float64)
    active = np.asarray(contact, dtype=bool)
    if tips.shape[:-1] != active.shape or poses.shape != (len(tips), 4, 4):
        raise ValueError("contact slip inputs have incompatible shapes")
    local = np.einsum(
        "tij,thfj->thfi", np.swapaxes(poses[:, :3, :3], 1, 2),
        tips - poses[:, None, None, :3, 3],
    )
    dt = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    speed = np.linalg.norm(np.diff(local, axis=0), axis=-1) / dt[:, None, None]
    persistent = active[:-1] & active[1:]
    return {"local_slip_speed_m_s": summary(speed[persistent])}
