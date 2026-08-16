#!/usr/bin/env python3
"""Run EgoForce's official YOLO hand detector on v11 M1 tiles.

Run inside the v2s-egoforce environment:

    conda run -n v2s-egoforce python scripts/run_v11_egoforce_tile_detector.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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

EGOFORCE_ROOT = REPO_ROOT / "third_party/EgoForce"
if str(EGOFORCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT))
if str(EGOFORCE_ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT / "demo"))

from settings import config as cfg
from ultralytics import YOLO


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    detector_path = str(cfg.DETECTION.HAND_PATH)
    print(f"loading EgoForce detector {detector_path}", flush=True)
    detector = YOLO(detector_path, task="pose")
    records: list[dict] = []
    rows = long_no_seed_rows()

    for clip, frame in rows:
        clip_dir = TILE_DIR / clip.replace(".tar", "") / f"{frame:06d}"
        for view in ("left", "right"):
            for tile in protocol["directions"]:
                tile_id = tile["tile_id"]
                img_path = clip_dir / view / f"tile_{tile_id}.jpg"
                if not img_path.is_file():
                    continue
                bgr = cv2.imread(str(img_path))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = detector.predict(
                    rgb,
                    verbose=False,
                    conf=0.25,
                    device="cuda",
                )[0]
                boxes = (
                    result.boxes.xyxy.cpu().numpy().astype(np.float32)
                    if result.boxes is not None and len(result.boxes) else None
                )
                scores = (
                    result.boxes.conf.cpu().numpy().astype(np.float32)
                    if result.boxes is not None and len(result.boxes) else None
                )
                classes = (
                    result.boxes.cls.cpu().numpy().astype(np.int64)
                    if result.boxes is not None and len(result.boxes) else None
                )
                keypoints = (
                    result.keypoints.xy.cpu().numpy().astype(np.float32)
                    if result.keypoints is not None
                    else None
                )
                if boxes is None or len(boxes) == 0:
                    records.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "tile_id": tile_id,
                            "model": "egoforce_detector",
                            "proposal_valid": False,
                            "confidence": "",
                            "predicted_handedness": "",
                            "bbox_tile": "",
                            "keypoints_tile": "",
                        }
                    )
                    continue
                for i in range(len(boxes)):
                    cls = int(classes[i])
                    handedness = "left" if cls in (0, 2) else "right"
                    kp_json = (
                        json.dumps(keypoints[i].tolist())
                        if keypoints is not None
                        else ""
                    )
                    records.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "tile_id": tile_id,
                            "model": "egoforce_detector",
                            "proposal_valid": True,
                            "confidence": float(scores[i]),
                            "predicted_handedness": handedness,
                            "bbox_tile": json.dumps(boxes[i].tolist()),
                            "keypoints_tile": kp_json,
                        }
                    )

    out_path = OUT / "initializer_candidates_egoforce_tile.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
