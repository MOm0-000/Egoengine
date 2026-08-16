#!/usr/bin/env python3
"""Visualize v13 full-frame finder outputs."""

from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
V11_TILES = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11/tiles"
)
INTER_CSV = OUT / "interformer_candidates.csv"
V12_MATRIX = (
    REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12/oracle_crop_matrix.csv"
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_bbox(img, bbox, color, thickness=2, label=""):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(
            img, label, (x0, max(12, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return img


def main() -> int:
    inter_raw = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(INTER_CSV)
        if r["protocol"] == "raw"
    }
    inter_m1 = {
        (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"])): r
        for r in read_csv(INTER_CSV)
        if r["protocol"] == "m1"
    }
    raw_gt = {}
    for r in read_csv(V12_MATRIX):
        if r["raw_bbox"]:
            raw_gt[(r["clip"], int(r["frame"]), r["camera"])] = json.loads(
                r["raw_bbox"]
            )

    selected = [
        ("clip-001852.tar", 1),
        ("clip-001852.tar", 5),
        ("clip-001852.tar", 6),
        ("clip-001852.tar", 7),
        ("clip-001852.tar", 15),
        ("clip-001852.tar", 23),
        ("clip-001891.tar", 3),
        ("clip-001891.tar", 6),
        ("clip-001891.tar", 7),
        ("clip-002316.tar", 0),
    ]
    panels = []
    for clip, frame in selected:
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for view, stream in (("left", "1201-1"), ("right", "1201-2")):
                b = tar.extractfile(f"{frame:06d}.image_{stream}.jpg").read()
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (384, 384))
                gt = raw_gt.get((clip, frame, view))
                if gt is not None:
                    sx = 384 / 640
                    sy = 384 / 480
                    draw_bbox(
                        img,
                        [gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy],
                        (80, 220, 80),
                        2,
                        "GT",
                    )
                r = inter_raw.get((clip, frame, view))
                if r:
                    for c in json.loads(r["hand_components"]):
                        draw_bbox(
                            img,
                            [
                                c["bbox"][0] * sx,
                                c["bbox"][1] * sy,
                                c["bbox"][2] * sx,
                                c["bbox"][3] * sy,
                            ],
                            (50, 50, 255),
                            1,
                            "Inter",
                        )
                label = np.full((384, 20, 3), 255, dtype=np.uint8)
                cv2.putText(
                    label, clip, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )
                panels.append(np.concatenate([label, img], axis=1))

    while len(panels) % 4:
        panels.append(np.full_like(panels[0], 255))
    rows = []
    for i in range(0, len(panels), 4):
        rows.append(np.concatenate(panels[i : i + 4], axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(OUT / "fullframe_examples.png"), sheet)
    cv2.imwrite(str(OUT / "sanity_10_examples.png"), sheet)

    # Failure examples: RAW views with no InterFormer center usable.
    failure_panels = []
    for clip, frame in selected:
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for view, stream in (("left", "1201-1"), ("right", "1201-2")):
                r = inter_raw.get((clip, frame, view))
                gt = raw_gt.get((clip, frame, view))
                if not r or not gt:
                    continue
                comps = json.loads(r["hand_components"])
                usable = any(
                    gt[0] - 20 <= c["center"][0] <= gt[2] + 20
                    and gt[1] - 20 <= c["center"][1] <= gt[3] + 20
                    for c in comps
                )
                if usable:
                    continue
                b = tar.extractfile(f"{frame:06d}.image_{stream}.jpg").read()
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (384, 384))
                sx, sy = 384 / 640, 384 / 480
                draw_bbox(
                    img,
                    [gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy],
                    (80, 220, 80), 2, "GT",
                )
                for c in comps:
                    draw_bbox(
                        img,
                        [
                            c["bbox"][0] * sx,
                            c["bbox"][1] * sy,
                            c["bbox"][2] * sx,
                            c["bbox"][3] * sy,
                        ],
                        (50, 50, 255), 1, "Inter",
                    )
                failure_panels.append(img)
    if failure_panels:
        while len(failure_panels) % 4:
            failure_panels.append(np.full_like(failure_panels[0], 0))
        rows = []
        for i in range(0, len(failure_panels), 4):
            rows.append(np.concatenate(failure_panels[i : i + 4], axis=1))
        cv2.imwrite(
            str(OUT / "failure_examples.png"), np.concatenate(rows, axis=0)
        )
    print("wrote visualizations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
