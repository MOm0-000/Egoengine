#!/usr/bin/env python3
"""Compute M1-tile InterFormer mask center proposals P0-P3."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data_all/zzx/egoengine/video_to_spider")
OUT = ROOT / "runs/hot3d_hand_diagnosis/auto_crop_refinement_v17"
V13 = ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
V11_TILES = (
    ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11/tiles"
)
INTERFORMER_ROOT = ROOT / "third_party/InterFormer"

CFG = INTERFORMER_ROOT / "mmseg/configs/config_interformer.py"
CKPT = INTERFORMER_ROOT / "checkpoint.pth"

TILE_DIRS = [
    (0, 0.0, 0.0),
    (1, 35.0, 0.0),
    (2, -35.0, 0.0),
    (3, 0.0, 35.0),
    (4, 0.0, -35.0),
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("", None) else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(INTERFORMER_ROOT))
    from mmseg.apis import init_model, inference_model

    model = init_model(str(CFG), str(CKPT), device="cuda:0")
    # Find selected tile per v16 view using v13 M1 components.
    inter_m1 = {
        (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"])): r
        for r in read_csv(V13 / "interformer_candidates.csv")
        if r["protocol"] == "m1"
    }
    v16_keys = set()
    for name in (
        "real_automatic_strong_predictions.csv",
        "real_automatic_weak_predictions.csv",
    ):
        v16_keys |= {
            (r["clip"], int(r["frame"]), r["camera"])
            for r in read_csv(ROOT / "runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16" / name)
        }

    rows = []
    for clip, frame, view in sorted(v16_keys):
        best_tile = None
        best_area = -1
        for tile_id in range(5):
            r = inter_m1.get((clip, frame, view, tile_id))
            if r is None:
                continue
            comps = jload(r["hand_components"]) or []
            for c in comps:
                if c["area"] > best_area:
                    best_area = c["area"]
                    best_tile = tile_id
        if best_tile is None:
            continue
        path = (
            V11_TILES
            / clip.replace(".tar", "")
            / f"{frame:06d}"
            / view
            / f"tile_{best_tile}.jpg"
        )
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        img448 = cv2.resize(img, (448, 448))
        pred = inference_model(model, img448)
        seg = pred.pred_sem_seg.data[0].cpu().numpy()
        hand = (seg == 1) | (seg == 2)
        ys, xs = np.where(hand)
        if len(xs) < 8:
            continue
        dist = cv2.distanceTransform(hand.astype(np.uint8), cv2.DIST_L2, 5)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        p0 = np.array([(x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2])
        p1 = np.array([float(np.mean(xs)), float(np.mean(ys))])
        max_flat = np.argmax(dist)
        my, mx = np.unravel_index(max_flat, dist.shape)
        p2 = np.array([float(mx), float(my)])
        thr = np.percentile(dist, 75)
        sel = dist >= thr
        if sel.sum() >= 8:
            py, px = np.where(sel)
            p3 = np.array([float(np.mean(px)), float(np.mean(py))])
        else:
            p3 = p2.copy()
        # Scale 448 -> 256 tile.
        def to_tile(p):
            return [float(p[0] * 256 / 448), float(p[1] * 256 / 448)]
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "tile_id": best_tile,
                "P0_tile": json.dumps(to_tile(p0)),
                "P1_tile": json.dumps(to_tile(p1)),
                "P2_tile": json.dumps(to_tile(p2)),
                "P3_tile": json.dumps(to_tile(p3)),
                "mask_area": int(len(xs)),
            }
        )
    out_path = OUT / "palm_center_candidates_m1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
