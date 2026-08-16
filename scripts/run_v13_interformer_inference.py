#!/usr/bin/env python3
"""Run InterFormer hand/object segmentation on RAW and M1 tile inputs."""

from __future__ import annotations

import csv
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFORMER_ROOT = REPO_ROOT / "third_party/InterFormer"
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
V11_TILES = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11/tiles"
)
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
CFG = INTERFORMER_ROOT / "mmseg/configs/config_interformer.py"
CKPT = INTERFORMER_ROOT / "checkpoint.pth"
MODEL_SIZE = (448, 448)


def long_no_seed_rows():
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def components(mask):
    if not mask.any():
        return []
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        if len(xs) < 4:
            continue
        area = int(len(xs))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        out.append(
            {
                "area": area,
                "bbox": [x0, y0, x1 + 1, y1 + 1],
                "center": [float((x0 + x1 + 1) / 2), float((y0 + y1 + 1) / 2)],
            }
        )
    out.sort(key=lambda c: c["area"], reverse=True)
    return out


def scale_bbox(bbox, sx, sy):
    if bbox is None:
        return None
    return [
        float(bbox[0] * sx),
        float(bbox[1] * sy),
        float(bbox[2] * sx),
        float(bbox[3] * sy),
    ]


def process_image(model, inference_model, img, src_w, src_h):
    img448 = cv2.resize(img, MODEL_SIZE)
    pred = inference_model(model, img448)
    seg = pred.pred_sem_seg.data[0].cpu().numpy()
    left_mask = seg == 1
    right_mask = seg == 2
    hand_mask = left_mask | right_mask
    left_comp = components(left_mask)
    right_comp = components(right_mask)
    hand_comp = components(hand_mask)
    sx = src_w / MODEL_SIZE[0]
    sy = src_h / MODEL_SIZE[1]
    return {
        "left_mask_area": int(left_mask.sum()),
        "right_mask_area": int(right_mask.sum()),
        "hand_mask_area": int(hand_mask.sum()),
        "left_components": [
            {
                "area": c["area"],
                "bbox": scale_bbox(c["bbox"], sx, sy),
                "center": [c["center"][0] * sx, c["center"][1] * sy],
            }
            for c in left_comp
        ],
        "right_components": [
            {
                "area": c["area"],
                "bbox": scale_bbox(c["bbox"], sx, sy),
                "center": [c["center"][0] * sx, c["center"][1] * sy],
            }
            for c in right_comp
        ],
        "hand_components": [
            {
                "area": c["area"],
                "bbox": scale_bbox(c["bbox"], sx, sy),
                "center": [c["center"][0] * sx, c["center"][1] * sy],
            }
            for c in hand_comp
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(INTERFORMER_ROOT))
    from mmseg.apis import init_model, inference_model

    model = init_model(str(CFG), str(CKPT), device="cuda:0")
    rows = long_no_seed_rows()
    records = []

    for clip, frame in rows:
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                b = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                out = process_image(model, inference_model, img, w, h)
                records.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "protocol": "raw",
                        "tile_id": "",
                        "left_mask_area": out["left_mask_area"],
                        "right_mask_area": out["right_mask_area"],
                        "hand_mask_area": out["hand_mask_area"],
                        "left_components": json.dumps(out["left_components"]),
                        "right_components": json.dumps(out["right_components"]),
                        "hand_components": json.dumps(out["hand_components"]),
                    }
                )

        tile_dir = V11_TILES / clip.replace(".tar", "") / f"{frame:06d}"
        for view in ("left", "right"):
            for tile_id in range(5):
                path = tile_dir / view / f"tile_{tile_id}.jpg"
                if not path.is_file():
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                out = process_image(model, inference_model, img, w, h)
                records.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "protocol": "m1",
                        "tile_id": tile_id,
                        "left_mask_area": out["left_mask_area"],
                        "right_mask_area": out["right_mask_area"],
                        "hand_mask_area": out["hand_mask_area"],
                        "left_components": json.dumps(out["left_components"]),
                        "right_components": json.dumps(out["right_components"]),
                        "hand_components": json.dumps(out["hand_components"]),
                    }
                )

    out_path = OUT / "interformer_candidates.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
