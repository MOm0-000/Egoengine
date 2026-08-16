#!/usr/bin/env python3
"""D1/D2 object-ROI depth refinement benchmark for ADT RGB object runs.

This is a benchmark driver, not a production pipeline change. It reads an
existing ``*_rgbobject`` run, applies two object-ROI-only depth refinements and
writes both refined depth artifacts plus a JSON metric file:

D0 -- original ``depth/metric_depth.zarr``
D1 -- mask erosion + invalid fill (TELEA) + edge-aware bilateral filtering
D2 -- D1 followed by a light temporal median in object image coordinates

Only object-region depth is compared against ADT RGB GT depth. GT is used only
as an evaluator, never inside the refined artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _lookup(frame_indices: np.ndarray) -> dict[int, int]:
    return {int(frame): index for index, frame in enumerate(np.asarray(frame_indices))}


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _refine_d1(depth_m: np.ndarray, valid: np.ndarray, mask: np.ndarray, *, erode: int) -> tuple[np.ndarray, np.ndarray]:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    roi = _erode_mask(mask, erode)
    source = np.where(valid & roi, depth_m, 0.0).astype(np.float32)
    fill_mask = (roi & ~valid).astype(np.uint8)
    filled = cv2.inpaint(source, fill_mask, 3, cv2.INPAINT_TELEA)
    filled = cv2.bilateralFilter(filled, 5, 0.06, 7)
    filled = np.clip(filled, 0.0, 8.0)
    out_depth = depth_m.copy()
    out_valid = valid.copy()
    update = roi & ~valid
    out_depth[update] = filled[update]
    out_valid[update] = True
    return out_depth, out_valid


def _refine_d2(depth_stack: np.ndarray, valid_stack: np.ndarray, mask_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temporal median over a small centered window, masked to the current object ROI."""
    count = depth_stack.shape[0]
    out_depth = depth_stack.copy()
    out_valid = valid_stack.copy()
    for index in range(count):
        lo = max(0, index - 2)
        hi = min(count, index + 3)
        window_depth = depth_stack[lo:hi]
        window_valid = valid_stack[lo:hi] & mask_stack[lo:hi]
        window = np.where(window_valid, window_depth, np.nan).astype(np.float32)
        median = np.nanmedian(window, axis=0)
        current_roi = mask_stack[index]
        update = current_roi & np.isfinite(median)
        out_depth[index][update] = median[update]
        out_valid[index][update] = True
    return out_depth, out_valid


def _depth_metrics(pred_depth: np.ndarray, pred_valid: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray, roi: np.ndarray) -> dict[str, float | int]:
    gt_roi = roi & gt_valid & np.isfinite(gt_depth) & (gt_depth > 0)
    if not gt_roi.any():
        return {"object_roi_pixels": 0, "invalid_rate": 1.0, "abs_rel": None, "rmse_m": None, "delta1": None}
    pred_roi = gt_roi & pred_valid & np.isfinite(pred_depth) & (pred_depth > 0)
    invalid_rate = float((gt_roi & ~pred_roi).sum() / gt_roi.sum())
    if not pred_roi.any():
        return {"object_roi_pixels": int(gt_roi.sum()), "invalid_rate": invalid_rate, "abs_rel": None, "rmse_m": None, "delta1": None}
    a = pred_depth[pred_roi]
    b = gt_depth[pred_roi]
    abs_rel = float(np.mean(np.abs(a - b) / np.maximum(b, 1e-3)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    ratio = np.maximum(a / np.maximum(b, 1e-3), b / np.maximum(a, 1e-3))
    delta1 = float(np.mean(ratio < 1.25))
    return {
        "object_roi_pixels": int(gt_roi.sum()),
        "invalid_rate": invalid_rate,
        "abs_rel": abs_rel,
        "rmse_m": rmse,
        "delta1": delta1,
    }


def _write_zarr(path: Path, depth_m: np.ndarray, valid: np.ndarray, frame_indices: np.ndarray, timestamps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("frame_indices", data=np.asarray(frame_indices, dtype=np.int64))
    group.create_dataset("timestamps_s", data=np.asarray(timestamps, dtype=np.float64))
    group.create_dataset("depth_m", data=depth_m.astype(np.float32), chunks=(1, min(256, depth_m.shape[1]), min(256, depth_m.shape[2])), dtype="f4")
    group.create_dataset("valid", data=valid.astype(bool), chunks=(1, min(256, valid.shape[1]), min(256, valid.shape[2])), dtype="bool")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--erode", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.base_run.resolve()
    masks = _load_masks(root / "segmentation/object_masks.npz")
    pred_group = zarr.open(str(root / "depth/metric_depth.zarr"), mode="r")
    gt_group = zarr.open(str(root / "evaluation/adt_rgb_gt/adt_gt.zarr"), mode="r")

    pred_lookup = _lookup(pred_group["frame_indices"])
    gt_lookup = _lookup(gt_group["frame_indices"])

    frame_indices = masks["frame_indices"]
    timestamps = masks["timestamps_s"]
    mask_stack = masks["masks"].astype(bool)
    valid_stack = masks["valid"].astype(bool)

    depth0 = np.asarray(pred_group["depth_m"], dtype=np.float32)
    valid0 = np.asarray(pred_group["valid"], dtype=bool)

    depth1 = np.empty_like(depth0)
    valid1 = np.empty_like(valid0)
    for index, frame in enumerate(frame_indices):
        frame = int(frame)
        di = pred_lookup[frame]
        if not bool(valid_stack[index]) or not mask_stack[index].any():
            depth1[index] = depth0[di]
            valid1[index] = valid0[di]
            continue
        depth1[index], valid1[index] = _refine_d1(depth0[di], valid0[di], mask_stack[index], erode=args.erode)

    depth2, valid2 = _refine_d2(depth1, valid1, mask_stack)

    # Evaluate all variants against GT over each predicted object ROI.
    metrics = {"base_run": str(root), "variants": {}}
    for variant, depth, valid in (("d0", depth0, valid0), ("d1", depth1, valid1), ("d2", depth2, valid2)):
        frame_metrics: list[dict[str, float | int | None]] = []
        for index, frame in enumerate(frame_indices):
            frame = int(frame)
            if not bool(valid_stack[index]) or not mask_stack[index].any():
                continue
            gi = gt_lookup.get(frame)
            if gi is None:
                continue
            gt_depth = np.asarray(gt_group["depth_m"][gi], dtype=np.float32)
            gt_valid = np.asarray(gt_group["valid"][gi]).astype(bool)
            frame_metrics.append(_depth_metrics(depth[index], valid[index], gt_depth, gt_valid, mask_stack[index]))
        if frame_metrics:
            agg = {
                "mean_abs_rel": float(np.nanmean([m["abs_rel"] for m in frame_metrics if m["abs_rel"] is not None])),
                "mean_rmse_m": float(np.nanmean([m["rmse_m"] for m in frame_metrics if m["rmse_m"] is not None])),
                "mean_delta1": float(np.nanmean([m["delta1"] for m in frame_metrics if m["delta1"] is not None])),
                "mean_invalid_rate": float(np.mean([m["invalid_rate"] for m in frame_metrics])),
                "frames": len(frame_metrics),
            }
        else:
            agg = {"mean_abs_rel": None, "mean_rmse_m": None, "mean_delta1": None, "mean_invalid_rate": 1.0, "frames": 0}
        metrics["variants"][variant] = agg

    out_dir = root / "depth_roi_refined"
    _write_zarr(out_dir / "d1.zarr", depth1, valid1, frame_indices, timestamps)
    _write_zarr(out_dir / "d2.zarr", depth2, valid2, frame_indices, timestamps)
    (out_dir / "metadata.json").write_text(json.dumps({
        "schema_version": "1.0",
        "purpose": "object-ROI depth refinement ablation D1/D2",
        "ground_truth_used": False,
        "erode_radius": args.erode,
        "d1": "mask erosion + TELEA invalid fill + bilateral filtering",
        "d2": "d1 + temporal median (window 5) within object mask",
    }, indent=2) + "\n", encoding="utf-8")
    metrics_path = out_dir / "depth_metrics.json"
    metrics_path.write_text(json.dumps(_safe(metrics), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_safe(metrics), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
