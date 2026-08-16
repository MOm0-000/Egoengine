#!/usr/bin/env python3
"""UmeTrack known-skeleton oracle on prepared perspective crops.

Run inside the `umetrack` environment.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UMETRACK_ROOT = REPO_ROOT / "third_party/UmeTrack"
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CROP_DIR = OUT / "crops"
MATRIX_CSV = OUT / "oracle_crop_matrix.csv"

sys.path.insert(0, str(UMETRACK_ROOT))

from lib.common.camera import PinholePlaneCameraModel
from lib.common.hand import LEFT_HAND_INDEX, RIGHT_HAND_INDEX
from lib.models.model_loader import load_pretrained_model
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import _load_json, load_hand_model_from_dict


MODEL_PATH = UMETRACK_ROOT / "pretrained_models/pretrained_weights.torch"
GENERIC_PATH = UMETRACK_ROOT / "dataset/generic_hand_model.json"
CROP_SIZE = 96


def long_no_seed_rows():
    rows = []
    gap = (
        REPO_ROOT
        / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
    )
    with gap.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_pretrained_model(str(MODEL_PATH))
    model.eval()
    tracker = HandTracker(model, HandTrackerOpts())
    generic = load_hand_model_from_dict(_load_json(str(GENERIC_PATH)))
    rows = long_no_seed_rows()

    records = []
    for clip, frame in rows:
        clip_dir = CROP_DIR / clip.replace(".tar", "") / f"{frame:06d}"
        for view in ("left", "right"):
            for protocol in ("O-A", "O-B"):
                matrix_row = None
                # Matrix has no index column; find row by clip/frame/camera.
                with MATRIX_CSV.open(newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if (
                            r["clip"] == clip
                            and int(r["frame"]) == frame
                            and r["camera"] == view
                        ):
                            matrix_row = r
                            break
                if matrix_row is None or matrix_row.get(f"{protocol}_ok") != "True":
                    continue
                img_path = clip_dir / view / f"{protocol}.jpg"
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
                f256 = float(matrix_row[f"{protocol}_f"])
                f96 = f256 * CROP_SIZE / 256.0
                T_m = np.asarray(
                    json.loads(matrix_row[f"{protocol}_T_world_from_eye"]),
                    dtype=np.float64,
                )
                T_mm = T_m.copy()
                T_mm[:3, 3] *= 1000.0
                crop_cam = PinholePlaneCameraModel(
                    width=CROP_SIZE,
                    height=CROP_SIZE,
                    f=(f96, f96),
                    c=((CROP_SIZE - 1) / 2.0, (CROP_SIZE - 1) / 2.0),
                    distort_coeffs=[],
                    camera_to_world_xf=T_mm,
                )
                input_frame = InputFrame(
                    views=[
                        ViewData(
                            image=image,
                            camera=crop_cam,
                            camera_angle=0.0,
                        )
                    ]
                )
                hand_idx = (
                    LEFT_HAND_INDEX
                    if matrix_row["selected_side"] == "left"
                    else RIGHT_HAND_INDEX
                )
                crop_cameras = {hand_idx: {0: crop_cam}}
                try:
                    with torch.no_grad():
                        res = tracker.track_frame(
                            input_frame, generic, crop_cameras
                        )
                except Exception as exc:
                    print(
                        f"skip {clip}/{frame}/{view}/{protocol}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if hand_idx not in res.hand_poses:
                    continue
                pose = res.hand_poses[hand_idx]
                landmarks = landmarks_from_hand_pose(generic, pose, hand_idx)
                eye = crop_cam.world_to_eye(landmarks.astype(np.float64))
                uv = crop_cam.eye_to_window(eye)
                records.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "protocol": protocol,
                        "model": "umetrack",
                        "selected_side": matrix_row["selected_side"],
                        "landmarks_world_mm": json.dumps(landmarks.tolist()),
                        "joints_2d_crop": json.dumps(uv.tolist()),
                        "f96": f96,
                        "T_world_from_eye_mm": json.dumps(T_mm.tolist()),
                    }
                )

    out_path = OUT / "pose_oracle_umetrack.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
