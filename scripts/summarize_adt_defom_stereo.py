#!/usr/bin/env python3
"""Convert DEFOM-Stereo disparity outputs to metric depth and evaluate against ADT GT."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
OUTPUT = RUNS_ROOT / "adt_defom_stereo_summary.json"


def _load_final_masks(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_adt_phase0_sam3_ablation as phase0
    object_run = phase0._object_run(
        next((row for row in json.loads(SUMMARY_PATH.read_text()) if Path(row["run_dir"]) == run_dir), None)
    ) if False else run_dir
    candidates = [
        run_dir.parent / (run_dir.name + "_phase_final") / "segmentation" / "object_masks.npz",
    ]
    for row in json.loads(SUMMARY_PATH.read_text()):
        if Path(row["run_dir"]) == run_dir:
            candidate_run = phase0._object_run(row)
            if candidate_run is not None:
                candidates.append(
                    candidate_run.parent / (candidate_run.name + "_phase_final") / "segmentation" / "object_masks.npz"
                )
            break
    candidates.append(run_dir / "segmentation" / "object_masks.npz")
    for path in candidates:
        if path.is_file():
            with np.load(path, allow_pickle=False) as artifact:
                return (
                    np.asarray(artifact["frame_indices"], dtype=np.int64),
                    np.asarray(artifact["masks"], dtype=bool),
                    np.asarray(artifact["valid"], dtype=bool),
                )
    return None


def _metrics(pred_depth: np.ndarray, pred_valid: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray, roi: np.ndarray | None = None) -> dict[str, Any]:
    gt_valid = gt_valid & np.isfinite(gt_depth) & (gt_depth > 0)
    if roi is None:
        roi = np.ones_like(gt_depth, dtype=bool)
    sel = roi & gt_valid
    if not sel.any():
        return {"pixels": 0, "invalid_rate": 1.0, "abs_rel": None, "rmse_m": None, "delta1": None}
    pred_sel = sel & pred_valid & np.isfinite(pred_depth) & (pred_depth > 0)
    invalid_rate = float((sel & ~pred_sel).sum() / sel.sum())
    if not pred_sel.any():
        return {"pixels": int(sel.sum()), "invalid_rate": invalid_rate, "abs_rel": None, "rmse_m": None, "delta1": None}
    a = pred_depth[pred_sel]
    b = gt_depth[pred_sel]
    abs_rel = float(np.mean(np.abs(a - b) / np.maximum(b, 1e-3)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    ratio = np.maximum(a / np.maximum(b, 1e-3), b / np.maximum(a, 1e-3))
    delta1 = float(np.mean(ratio < 1.25))
    return {"pixels": int(sel.sum()), "invalid_rate": invalid_rate, "abs_rel": abs_rel, "rmse_m": rmse, "delta1": delta1}


def main() -> int:
    rows = json.loads(SUMMARY_PATH.read_text())
    records = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        frame_meta = json.loads((run_dir / "frames" / "frame_index.json").read_text())["frames"]
        stereo = json.loads((run_dir / "calibration" / "stereo.json").read_text())
        K = np.load(run_dir / "calibration" / "intrinsics.npy")
        fx = float(K[0, 0])
        baseline = float(stereo["baseline_m"])
        gt_path = run_dir / "evaluation" / "adt_gt" / "adt_gt.zarr"
        if not gt_path.is_dir():
            print("skip no gt", run_dir.name)
            continue
        gt = zarr.open_group(str(gt_path), mode="r")
        gt_frames = np.asarray(gt["frame_indices"], dtype=np.int64)
        gt_lookup = {int(frame): i for i, frame in enumerate(gt_frames)}
        pred_depth_list, pred_valid_list, gt_depth_list, gt_valid_list, mask_list = [], [], [], [], []
        mask_data = _load_final_masks(run_dir)
        for item in frame_meta:
            local_idx = int(item["frame_index"])
            source_idx = int(item["source_frame_index"])
            npy = run_dir / "evaluation" / "defom_stereo" / f"{local_idx:06d}.png_defomstereo_vitl_sceneflow.npy"
            if not npy.is_file():
                continue
            disp = np.load(npy, allow_pickle=False).astype(np.float32)
            valid_disp = disp > 0
            depth = np.full(disp.shape, np.nan, dtype=np.float32)
            depth[valid_disp] = fx * baseline / disp[valid_disp]
            valid = valid_disp & np.isfinite(depth) & (depth > 0)
            pred_depth_list.append(depth)
            pred_valid_list.append(valid)
            if source_idx in gt_lookup:
                gi = gt_lookup[source_idx]
                gt_depth_list.append(np.asarray(gt["depth_m"][gi], dtype=np.float32))
                gt_valid_list.append(np.asarray(gt["valid"][gi], dtype=bool))
            else:
                gt_depth_list.append(np.full(disp.shape, np.nan, dtype=np.float32))
                gt_valid_list.append(np.zeros(disp.shape, dtype=bool))
            if mask_data is not None:
                mlookup = {int(f): i for i, f in enumerate(mask_data[0])}
                if source_idx in mlookup:
                    mi = mlookup[source_idx]
                    mask = (mask_data[1][mi] & mask_data[2][mi]).astype(np.uint8)
                    if mask.shape != disp.shape:
                        import cv2
                        mask = cv2.resize(mask, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_list.append(mask.astype(bool))
                else:
                    mask_list.append(np.zeros(disp.shape, dtype=bool))
            else:
                mask_list.append(np.zeros(disp.shape, dtype=bool))
        if not pred_depth_list:
            continue
        pred_depth = np.stack(pred_depth_list)
        pred_valid = np.stack(pred_valid_list)
        gt_depth = np.stack(gt_depth_list)
        gt_valid = np.stack(gt_valid_list)
        masks = np.stack(mask_list) if mask_list else np.zeros_like(pred_valid)
        full = _metrics(pred_depth, pred_valid, gt_depth, gt_valid)
        obj = _metrics(pred_depth, pred_valid, gt_depth, gt_valid, masks.astype(bool))
        records.append({
            "sequence": row["sequence"],
            "prototype": row["prototype"],
            "window": row["window"],
            "run_dir": str(run_dir),
            "full": full,
            "object": obj,
        })
        print(json.dumps(records[-1], ensure_ascii=False))
    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
