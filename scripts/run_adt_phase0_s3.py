#!/usr/bin/env python3
"""Phase 0 S3: stereo-aware candidate selection over existing S1/S2 masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_PATH = RUNS_ROOT / "adt_phase0_s3_summary.json"


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[-2:] == shape:
        return mask.astype(bool)
    return cv2.resize(
        (mask > 0).astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _metrics(
    masks: np.ndarray,
    valid_mask: np.ndarray,
    mask_frame_indices: np.ndarray,
    depth_group: Any,
    hand_masks: np.ndarray | None,
) -> dict[str, Any]:
    frame_indices = np.asarray(depth_group["frame_indices"], dtype=np.int64)
    depth_lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
    depth_support: list[float] = []
    disparity_support: list[float] = []
    hand_dist: list[float] = []
    centroids: list[np.ndarray] = []
    frames = 0
    for index in range(len(masks)):
        if not bool(valid_mask[index]):
            continue
        mask = _resize_mask(masks[index], depth_group["depth_m"].shape[1:3])
        if not mask.any():
            continue
        frame_index = int(mask_frame_indices[index])
        depth_at = depth_lookup.get(frame_index)
        if depth_at is None:
            continue
        depth = np.asarray(depth_group["depth_m"][depth_at], dtype=np.float32)
        valid = np.asarray(depth_group["valid"][depth_at], dtype=bool)
        if "disparity_px" in depth_group:
            disparity = np.asarray(depth_group["disparity_px"][depth_at], dtype=np.float32)
        else:
            disparity = np.zeros_like(depth, dtype=np.float32)
        roi = mask & valid & np.isfinite(depth) & (depth > 0)
        if roi.any():
            depth_support.append(float(roi.sum() / mask.sum()))
            disparity_support.append(float((disparity[roi] > 0).mean()))
        else:
            depth_support.append(0.0)
            disparity_support.append(0.0)
        if hand_masks is not None and hand_masks.shape[-2:] == mask.shape:
            hand = hand_masks[index].astype(bool)
            if hand.any() and mask.any():
                dist = cv2.distanceTransform((~hand).astype(np.uint8), cv2.DIST_L2, 3)
                hand_dist.append(float(np.median(dist[mask])))
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            centroids.append(np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]))
        frames += 1
    if len(centroids) >= 2:
        points = np.stack(centroids)
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        temporal_jump_p95 = float(np.percentile(jumps, 95))
    else:
        temporal_jump_p95 = None
    return {
        "frames": frames,
        "mean_depth_support": float(np.mean(depth_support)) if depth_support else 0.0,
        "mean_positive_disparity_support": float(np.mean(disparity_support)) if disparity_support else 0.0,
        "median_hand_distance_px": float(np.median(hand_dist)) if hand_dist else None,
        "temporal_centroid_jump_p95_px": temporal_jump_p95,
    }


def main() -> int:
    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    results = []
    for row in rows:
        source = Path(row["source_run"])
        s1 = _load_npz(source / "segmentation_s1" / "object_masks.npz")
        s2 = _load_npz(source / "segmentation_s2_multianchor_merged" / "object_masks.npz")
        if s1 is None:
            continue
        depth_group = zarr.open(str(source / "depth" / "metric_depth.zarr"), mode="r")
        hand_npz = _load_npz(source / "segmentation" / "hand_masks.npz")
        hand_masks = hand_npz.get("masks") if hand_npz else None
        s1_metrics = _metrics(s1["masks"], s1["valid"], s1["frame_indices"], depth_group, hand_masks)
        s2_metrics = _metrics(s2["masks"], s2["valid"], s2["frame_indices"], depth_group, hand_masks) if s2 is not None else None
        s1_rate = float((s1["valid"]).mean())
        s2_rate = float((s2["valid"]).mean()) if s2 is not None else 0.0
        selected = "s2" if s2 is not None and s2_rate > s1_rate else "s1"
        results.append({
            "prototype": row["prototype"],
            "window": row["window"],
            "source_run": str(source),
            "s1_valid_rate": s1_rate,
            "s1": s1_metrics,
            "s2_valid_rate": s2_rate if s2 is not None else None,
            "s2": s2_metrics,
            "selected": selected,
        })
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
