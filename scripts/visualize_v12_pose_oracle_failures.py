#!/usr/bin/env python3
"""Visualize v12 oracle-crop pose failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CROP_DIR = OUT / "crops"
MATRIX_CSV = OUT / "oracle_crop_matrix.csv"
POSE_2D = OUT / "pose_2d_matrix.csv"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("", None) else None


def draw_joints(img, joints, color, radius=3):
    for x, y in joints:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)
    return img


def main() -> int:
    pose_rows = read_csv(POSE_2D)
    matrix = {}
    for r in read_csv(MATRIX_CSV):
        matrix[(r["clip"], int(r["frame"]), r["camera"])] = r

    rows = sorted(
        [r for r in pose_rows if r["model"] in ("wilor", "hamer", "umetrack")],
        key=lambda r: float(r.get("median_epe_px", 0) or 0),
        reverse=True,
    )
    selected = rows[:20]
    panels = []
    for r in selected:
        clip = r["clip"]
        frame = int(r["frame"])
        view = r["camera"]
        protocol = r["protocol"]
        model = r["model"]
        img_path = CROP_DIR / clip.replace(".tar", "") / f"{frame:06d}" / view / f"{protocol}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        m = matrix.get((clip, frame, view))
        if m is not None:
            gt_uv = jload(m[f"{protocol}_gt_joints_uv"])
            gt_vis = jload(m[f"{protocol}_gt_joints_visible"])
            if gt_uv is not None and gt_vis is not None:
                for j, (uv, vis) in enumerate(zip(gt_uv, gt_vis)):
                    if vis:
                        draw_joints(img, [uv], (80, 220, 80), 3)
        pred_file = {
            "wilor": OUT / f"pose_oracle_wilor_{protocol}.csv",
            "hamer": OUT / f"pose_oracle_hamer_{protocol}.csv",
            "umetrack": OUT / "pose_oracle_umetrack.csv",
        }[model]
        preds = read_csv(pred_file)
        pred = next(
            (
                p
                for p in preds
                if p["clip"] == clip
                and int(p["frame"]) == frame
                and p["camera"] == view
                and p.get("protocol", protocol) == protocol
            ),
            None,
        )
        if pred is not None:
            if model == "umetrack":
                uv = jload(pred["joints_2d_crop"])
            else:
                uv = jload(pred["joints_2d_crop"])
            if uv is not None:
                draw_joints(img, uv, (50, 50, 255), 2)
        label = np.full((256, 32, 3), 255, dtype=np.uint8)
        cv2.putText(
            label,
            f"{model} {protocol}",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            label,
            f"{clip[-8:]} f{frame} {view}",
            (4, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.concatenate([label, img], axis=1))

    # Arrange 5 columns.
    while len(panels) % 5:
        panels.append(np.full_like(panels[0], 255))
    rows_img = []
    for i in range(0, len(panels), 5):
        rows_img.append(np.concatenate(panels[i : i + 5], axis=1))
    sheet = np.concatenate(rows_img, axis=0)
    out_path = OUT / "pose_oracle_failures.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"wrote {out_path} shape={sheet.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
