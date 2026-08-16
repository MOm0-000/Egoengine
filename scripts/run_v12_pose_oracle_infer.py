#!/usr/bin/env python3
"""Run WiLoR / HaMeR with oracle crop and known GT-derived bbox."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CROP_DIR = OUT / "crops"
MATRIX_CSV = OUT / "oracle_crop_matrix.csv"
MANIFEST_PATH = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wilor", "hamer"), required=True)
    parser.add_argument("--protocol", choices=("O-A", "O-B"), default="O-A")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(model_name):
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.run_stereo_crop_inference import _load_model

    return _load_model(model_name)


def _camera_translation(pred_cam, box_center, box_size, K):
    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def infer_boxes(model, cfg, dataset_cls, recursive_to, device, image_bgr, boxes, right, K):
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
    joints_root = out["pred_keypoints_3d"].detach().cpu().numpy().copy()
    flip_x = (2.0 * np.asarray(right, dtype=np.float32) - 1.0)[:, None, None]
    joints_root[..., 0] *= flip_x[..., 0]
    joints = joints_root + camera_t[:, None, :]
    uv = np.zeros((joints.shape[0], joints.shape[1], 2), dtype=np.float64)
    for i in range(joints.shape[0]):
        uv[i] = (
            np.einsum("ij,...j->...i", K, joints[i])[:, :2]
            / joints[i][..., 2:3]
        )
    return joints, uv


def enlarge_bbox(bbox, ratio=1.3):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max(1.0, x1 - x0) * ratio
    h = max(1.0, y1 - y0) * ratio
    side = max(w, h)
    return [
        cx - side / 2.0,
        cy - side / 2.0,
        cx + side / 2.0,
        cy + side / 2.0,
    ]


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, cfg, detector, dataset_cls, recursive_to = load_model(args.model)
    model = model.to(device).eval()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected_side = {
        item["clip"]: 0 if item["selected_side"] == "left" else 1
        for item in manifest["items"]
    }
    rows = []
    for r in csv.DictReader(MATRIX_CSV.open(newline="", encoding="utf-8")):
        if args.limit and len(rows) >= args.limit:
            break
        key = f"{args.protocol}_ok"
        if r.get(key) != "True":
            continue
        clip = r["clip"]
        frame = int(r["frame"])
        view = r["camera"]
        img_path = (
            CROP_DIR
            / clip.replace(".tar", "")
            / f"{frame:06d}"
            / view
            / f"{args.protocol}.jpg"
        )
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        gt_bbox = json.loads(r[f"{args.protocol}_gt_bbox"])
        if gt_bbox is None:
            continue
        box = enlarge_bbox(gt_bbox, 1.3)
        f = float(r[f"{args.protocol}_f"])
        K = np.asarray(
            [[f, 0.0, 127.5], [0.0, f, 127.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        right = selected_side[clip]
        try:
            joints, uv = infer_boxes(
                model,
                cfg,
                dataset_cls,
                recursive_to,
                device,
                image,
                [box],
                [right],
                K,
            )
        except Exception as exc:
            print(
                f"skip {clip}/{frame}/{view}/{args.protocol}: {exc}",
                file=sys.stderr,
            )
            continue
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "camera": view,
                "protocol": args.protocol,
                "model": args.model,
                "bbox_used": json.dumps(box),
                "joints_cam": json.dumps(joints[0].tolist()),
                "joints_2d_crop": json.dumps(uv[0].tolist()),
                "f": f,
                "T_world_from_eye": r[f"{args.protocol}_T_world_from_eye"],
                "selected_side": r["selected_side"],
            }
        )

    out_path = OUT / f"pose_oracle_{args.model}_{args.protocol}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
