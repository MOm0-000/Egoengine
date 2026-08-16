#!/usr/bin/env python3
"""EgoForce oracle-pose benchmark: use GT hand bbox, predicted forearm path."""

from __future__ import annotations

import csv
import json
import sys
import tarfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
MANIFEST_PATH = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"

EGOFORCE_ROOT = REPO_ROOT / "third_party/EgoForce"
if str(EGOFORCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT))
if str(EGOFORCE_ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT / "demo"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings import config as cfg
from models import HALO
from models.limb_model import LimbModel
from demo.demo_hand_arm_loader import DemoHandArmLoader
from mmdet.apis import DetInferencer

from scripts.run_egoforce_v10_full import (
    make_camera,
    mmdet_forearm_for_side,
    run_one_candidate,
)


def long_no_seed_rows():
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def enlarge_bbox(bbox, ratio=1.25):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max(1.0, x1 - x0) * ratio
    h = max(1.0, y1 - y0) * ratio
    side = max(w, h)
    return np.asarray(
        [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0],
        dtype=np.float32,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg.POSE_3D.CHECKPOINT_PATH = str(
        EGOFORCE_ROOT / "_DATA/model_weights.pth"
    )
    cfg.MANO_PATH = str(EGOFORCE_ROOT / "_DATA/mano")
    cfg.DETECTION.HAND_PATH = str(EGOFORCE_ROOT / "_DATA/detector.torchscript")
    cfg.DETECTION.HAND_ARM_PATH = str(EGOFORCE_ROOT / "_DATA/epoch_460.pth")

    model = HALO(cfg)
    model.load_state_dict(
        torch.load(cfg.POSE_3D.CHECKPOINT_PATH, map_location=device),
        strict=True,
    )
    model = model.to(device).eval()
    limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)
    for side in ("left", "right"):
        setattr(limb_model, side, None)
    left_loader = DemoHandArmLoader(
        cfg, None, undistort_inp=True, return_complete_image=False, hand_type="left"
    )
    right_loader = DemoHandArmLoader(
        cfg, None, undistort_inp=True, return_complete_image=False, hand_type="right"
    )
    mmdet_inferencer = DetInferencer(
        str(EGOFORCE_ROOT / "demo/rtmdet_tiny_8xb32-300e_combined_cutmix.py"),
        weights=cfg.DETECTION.HAND_ARM_PATH,
        device=str(device),
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    records = []

    for clip, frame in long_no_seed_rows():
        selected_side = clip_entry[clip]["selected_side"]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            hands = json.load(tar.extractfile(f"{frame:06d}.hands.json"))
            images_bgr = {}
            images_rgb = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                b = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()
                images_bgr[view] = cv2.imdecode(
                    np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                images_rgb[view] = cv2.cvtColor(
                    images_bgr[view], cv2.COLOR_BGR2RGB
                )

        for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
            hand_side_data = hands.get(selected_side)
            if hand_side_data is None:
                continue
            boxes = hand_side_data.get("boxes_amodal", {})
            raw_bbox = boxes.get(stream_id)
            if raw_bbox is None:
                records.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "selected_side": selected_side,
                        "valid": False,
                        "bbox": "",
                        "pred_j2d_raw": "",
                        "pred_j3d_cam": "",
                    }
                )
                continue
            bbox = enlarge_bbox(np.asarray(raw_bbox, dtype=np.float32))
            camera_model = make_camera(cams[stream_id])
            loader = left_loader if selected_side == "left" else right_loader
            loader.camera_model = camera_model
            loader.limb_model = limb_model
            rgb = images_rgb[view]
            bgr = images_bgr[view]
            # Keypoint input is ignored by HALO's hand crop path; GT is only
            # used to define the bbox ("where to look").
            candidate = {
                "handedness": selected_side,
                "score": 1.0,
                "keypoints": np.zeros((21, 2), dtype=np.float32),
                "bbox": bbox,
                "cls": 0,
            }
            arm_candidate = mmdet_forearm_for_side(
                mmdet_inferencer,
                bgr,
                bbox,
                selected_side,
            )
            try:
                out = run_one_candidate(
                    model,
                    limb_model,
                    loader,
                    rgb,
                    candidate,
                    selected_side,
                    arm_candidate,
                    device,
                )
            except Exception as exc:
                print(
                    f"skip {clip}/{frame}/{view}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                continue
            records.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "camera": view,
                    "selected_side": selected_side,
                    "valid": bool(out["final_observation_valid"]),
                    "bbox": json.dumps(np.asarray(out["bbox"]).tolist()),
                    "pred_j2d_raw": json.dumps(out["pred_j2d_raw"].tolist()),
                    "pred_j3d_cam": json.dumps(out["pred_j3d_cam"].tolist()),
                    "hand_kpt_w": json.dumps(out["hand_kpt_w"].tolist()),
                }
            )

    out_path = OUT / "pose_oracle_egoforce.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
