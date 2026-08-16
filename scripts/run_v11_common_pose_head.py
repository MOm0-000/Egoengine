#!/usr/bin/env python3
"""Run the common WiLoR pose head on external initializer proposals.

This isolates detector vs pose regression: MediaPipe bboxes are fed to the
existing WiLoR pose head. It is intentionally a diagnostic step, not a new
end-to-end detector.

Run inside v2s-wilor:

    conda run -n v2s-wilor python scripts/run_v11_common_pose_head.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11"
)
TILE_DIR = OUT / "tiles"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
PROTOCOL_PATH = OUT / "tile_protocol.json"
MEDIAPIPE_CSV = OUT / "initializer_candidates_mediapipe_tile.csv"
POSITIVE_CSV = OUT / "tile_positive_matrix.csv"
MANIFEST_PATH = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def _load_wilor():
    import os
    from pathlib import Path

    root = REPO_ROOT / "third_party/WiLoR"
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to

    model_obj, model_cfg = load_wilor(
        checkpoint_path=str(root / "pretrained_models/wilor_final.ckpt"),
        cfg_path=str(root / "pretrained_models/model_config.yaml"),
    )
    os.chdir(previous_cwd)
    return model_obj, model_cfg, ViTDetDataset, recursive_to


def _camera_translation(pred_cam, box_center, box_size, K):
    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def infer_boxes(
    model,
    cfg,
    dataset_cls,
    recursive_to,
    device,
    image_bgr,
    boxes,
    right,
    K,
):
    dataset = dataset_cls(
        cfg,
        image_bgr,
        np.asarray(boxes, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        rescale_factor=2.0,
        fp16=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    pred_cam = out["pred_cam"].clone()
    pred_cam[:, 1] *= (2 * batch["right"] - 1)
    camera_t = _camera_translation(
        pred_cam, batch["box_center"].float(), batch["box_size"].float(), K
    ).detach().cpu().numpy()
    joints = (
        out["pred_keypoints_3d"].detach().cpu().numpy()
        + camera_t[:, None, :]
    )
    uv = np.zeros((joints.shape[0], joints.shape[1], 2), dtype=np.float64)
    for i in range(joints.shape[0]):
        uv[i] = (
            np.einsum(
                "ij,...j->...i",
                K,
                joints[i],
            )[:, :2]
            / joints[i][..., 2:3]
        )
    return uv


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def main() -> int:
    device = torch.device("cuda")
    model, cfg, dataset_cls, recursive_to = _load_wilor()
    model = model.to(device).eval()

    pos_by = {}
    for r in csv.DictReader(POSITIVE_CSV.open(newline="", encoding="utf-8")):
        if r["gt_positive"] == "True":
            pos_by[
                (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"]))
            ] = r

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected_side = {
        item["clip"]: 0 if item["selected_side"] == "left" else 1
        for item in manifest["items"]
    }

    candidates = [
        r
        for r in csv.DictReader(
            MEDIAPIPE_CSV.open(newline="", encoding="utf-8")
        )
        if r["proposal_valid"] == "True"
    ]

    rows = []
    for cand in candidates:
        key = (
            cand["clip"],
            int(cand["frame"]),
            cand["camera"],
            int(cand["tile_id"]),
        )
        pos = pos_by.get(key)
        if pos is None:
            continue
        gt_bbox = json.loads(pos["gt_bbox"])
        det_bbox = json.loads(cand["bbox_tile"])
        if _iou(gt_bbox, det_bbox) < 0.5:
            continue

        img_path = (
            TILE_DIR
            / cand["clip"].replace(".tar", "")
            / f"{int(cand['frame']):06d}"
            / cand["camera"]
            / f"tile_{cand['tile_id']}.jpg"
        )
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        K = np.asarray(
            [[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        right = selected_side[cand["clip"]]
        try:
            uv = infer_boxes(
                model,
                cfg,
                dataset_cls,
                recursive_to,
                device,
                image,
                [det_bbox],
                [right],
                K,
            )[0]
        except Exception as exc:
            print(
                f"skip {cand['clip']}/{cand['frame']}/{cand['camera']}/{cand['tile_id']}: {exc}",
                file=sys.stderr,
            )
            continue
        rows.append(
            {
                "clip": cand["clip"],
                "frame": int(cand["frame"]),
                "camera": cand["camera"],
                "tile_id": int(cand["tile_id"]),
                "proposal_source": "mediapipe",
                "pose_head": "wilor",
                "keypoints_2d_tile": json.dumps(uv.tolist()),
                "bbox_tile": cand["bbox_tile"],
            }
        )

    out_path = OUT / "common_pose_head_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
