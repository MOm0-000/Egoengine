#!/usr/bin/env python3
"""D2 object-centric temporal depth fusion using BundleSDF object poses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_PATH = RUNS_ROOT / "adt_d2_objectcentric_summary.json"
POSE_ROOT = RUNS_ROOT / "adt_bundlesdf"


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.replace("_rgbobject", "")


def _load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _poses(row: dict) -> np.ndarray | None:
    pose_dir = POSE_ROOT / _row_key(row) / "ob_in_cam"
    if not pose_dir.is_dir():
        return None
    files = sorted(pose_dir.glob("*.txt"))
    if not files:
        return None
    poses = []
    for path in files:
        matrix = np.loadtxt(path).reshape(4, 4)
        if np.isfinite(matrix).all():
            poses.append(matrix)
    if len(poses) != 30:
        return None
    return np.stack(poses, axis=0)


def _unproject(mask: np.ndarray, depth: np.ndarray, valid: np.ndarray, K: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask & valid & (depth > 0) & np.isfinite(depth))
    z = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    u = K[0, 0] * x / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * y / np.maximum(z, 1e-6) + K[1, 2]
    return u, v


def _render_canonical_to_frame(canonical: np.ndarray, cam_from_obj: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    camera_points = np.einsum("ij,nj->ni", cam_from_obj[:3, :3], canonical) + cam_from_obj[:3, 3]
    u, v = _project(camera_points, K)
    height, width = shape
    valid = (
        (u >= 0) & (u < width) & (v >= 0) & (v < height)
        & (camera_points[:, 2] > 0)
    )
    ui = np.rint(u[valid]).astype(np.int64)
    vi = np.rint(v[valid]).astype(np.int64)
    rendered = np.full(shape, 0.0, dtype=np.float32)
    # Farthest-first painting; nearest visible point survives.
    order = np.argsort(-camera_points[valid, 2])
    rendered[vi[order], ui[order]] = camera_points[valid, 2][order]
    return rendered


def _depth_metrics(pred_depth: np.ndarray, pred_valid: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    gt_roi = roi & gt_valid & np.isfinite(gt_depth) & (gt_depth > 0)
    if not gt_roi.any():
        return {"pixels": 0, "invalid_rate": 1.0, "abs_rel": None, "rmse_m": None, "delta1": None}
    pred_roi = gt_roi & pred_valid & np.isfinite(pred_depth) & (pred_depth > 0)
    invalid = float((gt_roi & ~pred_roi).sum() / gt_roi.sum())
    if not pred_roi.any():
        return {"pixels": int(gt_roi.sum()), "invalid_rate": invalid, "abs_rel": None, "rmse_m": None, "delta1": None}
    a = pred_depth[pred_roi]
    b = gt_depth[pred_roi]
    ratio = np.maximum(a / np.maximum(b, 1e-3), b / np.maximum(a, 1e-3))
    return {
        "pixels": int(gt_roi.sum()),
        "invalid_rate": invalid,
        "abs_rel": float(np.mean(np.abs(a - b) / np.maximum(b, 1e-3))),
        "rmse_m": float(np.sqrt(np.mean((a - b) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def main() -> int:
    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    results = []
    for row in rows:
        source = Path(row["source_run"])
        target = Path(row["target_run"])
        key = _row_key(row)
        poses = _poses(row)
        record: dict[str, Any] = {
            "prototype": row["prototype"],
            "window": row["window"],
            "source_run": str(source),
            "status": "evaluated",
            "poses_available": poses is not None,
        }
        if poses is None:
            record["status"] = "no_bundlesdf_poses"
            results.append(record)
            continue
        depth_group = zarr.open(str(source / "depth" / "metric_depth.zarr"), mode="r")
        mask_npz = _load_masks(target / "segmentation" / "object_masks.npz")
        masks = mask_npz["masks"]
        if masks.shape[1:3] != depth_group["depth_m"].shape[1:3]:
            masks = np.stack([
                cv2.resize((mask > 0).astype(np.uint8), (depth_group["depth_m"].shape[2], depth_group["depth_m"].shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
                for mask in masks
            ])
        else:
            masks = masks > 0
        K = np.load(source / "calibration" / "intrinsics.npy").reshape(3, 3)
        depth = np.asarray(depth_group["depth_m"], dtype=np.float32)
        valid = np.asarray(depth_group["valid"], dtype=bool)

        canonical_points = []
        for index in range(len(depth)):
            if not masks[index].any():
                continue
            points = _unproject(masks[index], depth[index], valid[index], K)
            if not len(points):
                continue
            pose = poses[index]
            obj_points = np.einsum("ij,nj->ni", pose[:3, :3], points) + pose[:3, 3]
            canonical_points.append(obj_points)
        if not canonical_points:
            record["status"] = "no_fusion_points"
            results.append(record)
            continue
        canonical = np.concatenate(canonical_points, axis=0)
        voxel = 0.005
        keys = np.rint(canonical / voxel).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        fused = np.zeros((len(_), 3), dtype=np.float64)
        np.add.at(fused, inverse, canonical)
        fused /= counts[:, None]
        canonical = fused * voxel

        fused_depth = np.zeros_like(depth)
        fused_valid = np.zeros_like(valid)
        for index in range(len(depth)):
            if not masks[index].any():
                fused_depth[index] = depth[index]
                fused_valid[index] = valid[index]
                continue
            cam_from_obj = np.linalg.inv(poses[index])
            rendered = _render_canonical_to_frame(canonical, cam_from_obj, K, depth.shape[1:3])
            update = masks[index] & (rendered > 0)
            fused_depth[index] = depth[index]
            fused_valid[index] = valid[index]
            fused_depth[index][update] = rendered[update]
            fused_valid[index][update] = True

        gt_group = zarr.open(str(source / "evaluation" / "adt_rgb_gt" / "adt_gt.zarr"), mode="r")
        gt_lookup = {int(frame): index for index, frame in enumerate(gt_group["frame_indices"])}
        frame_metrics = []
        for index, frame in enumerate(mask_npz["frame_indices"]):
            frame = int(frame)
            if not masks[index].any() or frame not in gt_lookup:
                continue
            gi = gt_lookup[frame]
            gt_depth = np.asarray(gt_group["depth_m"][gi], dtype=np.float32)
            gt_valid = np.asarray(gt_group["valid"][gi], dtype=bool)
            frame_metrics.append(_depth_metrics(fused_depth[index], fused_valid[index], gt_depth, gt_valid, masks[index]))
        if frame_metrics:
            record["metrics"] = {
                "mean_abs_rel": float(np.nanmean([m["abs_rel"] for m in frame_metrics if m["abs_rel"] is not None])),
                "mean_rmse_m": float(np.nanmean([m["rmse_m"] for m in frame_metrics if m["rmse_m"] is not None])),
                "mean_delta1": float(np.nanmean([m["delta1"] for m in frame_metrics if m["delta1"] is not None])),
                "mean_invalid_rate": float(np.mean([m["invalid_rate"] for m in frame_metrics])),
                "frames": len(frame_metrics),
            }
        else:
            record["metrics"] = {"mean_abs_rel": None, "mean_rmse_m": None, "mean_delta1": None, "mean_invalid_rate": 1.0, "frames": 0}
        record["fusion_points"] = len(canonical)
        results.append(record)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
