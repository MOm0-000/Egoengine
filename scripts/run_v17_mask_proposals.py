#!/usr/bin/env python3
"""Generate InterFormer RAW hand masks and P0-P3 center proposals."""

from __future__ import annotations

import csv
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data_all/zzx/egoengine/video_to_spider")
OUT = ROOT / "runs/hot3d_hand_diagnosis/auto_crop_refinement_v17"
V16 = ROOT / "runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
CLIPS = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
INTERFORMER_ROOT = ROOT / "third_party/InterFormer"

CFG = INTERFORMER_ROOT / "mmseg/configs/config_interformer.py"
CKPT = INTERFORMER_ROOT / "checkpoint.pth"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(INTERFORMER_ROOT))
    from mmseg.apis import init_model, inference_model

    model = init_model(str(CFG), str(CKPT), device="cuda:0")
    keys = set()
    for name in (
        "real_automatic_strong_predictions.csv",
        "real_automatic_weak_predictions.csv",
    ):
        for r in read_csv(V16 / name):
            keys.add((r["clip"], int(r["frame"]), r["camera"]))

    rows = []
    for clip, frame, view in sorted(keys):
        with tarfile.open(CLIPS / clip, "r") as tar:
            b = tar.extractfile(
                f"{frame:06d}.image_{'1201-1' if view == 'left' else '1201-2'}.jpg"
            ).read()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        img448 = cv2.resize(img, (448, 448))
        pred = inference_model(model, img448)
        seg = pred.pred_sem_seg.data[0].cpu().numpy()
        hand = (seg == 1) | (seg == 2)
        ys, xs = np.where(hand)
        if len(xs) < 8:
            continue
        dist = cv2.distanceTransform(hand.astype(np.uint8), cv2.DIST_L2, 5)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        p0 = np.array([(x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2])
        p1 = np.array([float(np.mean(xs)), float(np.mean(ys))])
        max_flat = np.argmax(dist)
        max_y, max_x = np.unravel_index(max_flat, dist.shape)
        p2 = np.array([float(max_x), float(max_y)])
        # PCA: use thickest 25% pixels as palm proxy.
        thr = np.percentile(dist, 75)
        sel = dist >= thr
        if sel.sum() >= 8:
            py, px = np.where(sel)
            p3 = np.array([float(np.mean(px)), float(np.mean(py))])
        else:
            p3 = p2.copy()
        # scale 448 -> raw
        sx, sy = 640 / 448, 480 / 448
        def to_raw(p):
            return [float(p[0] * sx), float(p[1] * sy)]

        # Save downsampled mask for audit.
        mask_path = OUT / "masks" / clip.replace(".tar", "") / f"{frame:06d}"
        mask_path.mkdir(parents=True, exist_ok=True)
        np.save(str(mask_path / f"{view}.npy"), hand.astype(np.uint8))

        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "P0_bbox_center": json.dumps(to_raw(p0)),
                "P1_mask_centroid": json.dumps(to_raw(p1)),
                "P2_mask_dt_center": json.dumps(to_raw(p2)),
                "P3_pca_palm_center": json.dumps(to_raw(p3)),
                "mask_area": int(len(xs)),
            }
        )

    out_path = OUT / "palm_center_candidates_v17.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
