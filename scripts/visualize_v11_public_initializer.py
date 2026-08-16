#!/usr/bin/env python3
"""Create v11 sanity and public-initializer contact sheets."""

from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11"
)
TILE_DIR = OUT / "tiles"
CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
POSITIVE_CSV = OUT / "tile_positive_matrix.csv"
MODEL_FILES = {
    "wilor": OUT / "initializer_candidates_wilor_tile.csv",
    "egoforce": OUT / "initializer_candidates_egoforce_tile.csv",
    "mediapipe": OUT / "initializer_candidates_mediapipe_tile.csv",
}
MODEL_COLORS = {
    "wilor": (50, 50, 255),
    "egoforce": (255, 150, 50),
    "mediapipe": (255, 90, 200),
}


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def read_tile_gt():
    out = {}
    for r in csv.DictReader(POSITIVE_CSV.open(newline="", encoding="utf-8")):
        key = (r["clip"], int(r["frame"]), r["camera"], int(r["tile_id"]))
        bbox = json.loads(r["gt_bbox"]) if r["gt_bbox"] else None
        out[key] = bbox
    return out


def read_model_boxes():
    out = {}
    for model, path in MODEL_FILES.items():
        out[model] = {}
        for r in csv.DictReader(path.open(newline="", encoding="utf-8")):
            if r["proposal_valid"] != "True":
                continue
            key = (
                r["clip"],
                int(r["frame"]),
                r["camera"],
                int(r["tile_id"]),
            )
            out[model].setdefault(key, []).append(
                (json.loads(r["bbox_tile"]), float(r["confidence"]))
            )
    return out


def draw_boxes(img, gt, boxes_by_model):
    if gt is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in gt]
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 220, 80), 2)
        cv2.putText(
            img,
            "GT",
            (x0, max(8, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 220, 80),
            1,
            cv2.LINE_AA,
        )
    for model, color in MODEL_COLORS.items():
        for bbox, score in boxes_by_model.get(model, {}).get("__tile__", []):
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    return img


def annotate_tile(
    clip, frame, view, tile_id, gt, boxes_by_model
):
    path = (
        TILE_DIR
        / clip.replace(".tar", "")
        / f"{frame:06d}"
        / view
        / f"tile_{tile_id}.jpg"
    )
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (256, 256))
    # boxes_by_model currently keyed globally; this helper receives a slice.
    draw_boxes(img, gt, boxes_by_model)
    return img


def make_contact_sheet(selected, out_name, tile_size=256):
    gt_by_tile = read_tile_gt()
    boxes_by_model = read_model_boxes()
    # Reorganize boxes as model -> tile -> list for fast per-tile lookup.
    model_tile_boxes = {}
    for model, boxes in boxes_by_model.items():
        model_tile_boxes[model] = boxes

    rows = []
    for clip, frame in selected:
        row_left_raw = None
        row_right_raw = None
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                b = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (tile_size, tile_size))
                if view == "left":
                    row_left_raw = img
                else:
                    row_right_raw = img

        panels = [row_left_raw]
        for tile_id in range(5):
            tile_gt = gt_by_tile.get((clip, frame, "left", tile_id))
            boxes_for_tile = {
                model: {
                    "__tile__": model_tile_boxes[model].get(
                        (clip, frame, "left", tile_id), []
                    )
                }
                for model in MODEL_FILES
            }
            panels.append(
                annotate_tile(
                    clip, frame, "left", tile_id, tile_gt, boxes_for_tile
                )
            )
        panels.append(row_right_raw)
        for tile_id in range(5):
            tile_gt = gt_by_tile.get((clip, frame, "right", tile_id))
            boxes_for_tile = {
                model: {
                    "__tile__": model_tile_boxes[model].get(
                        (clip, frame, "right", tile_id), []
                    )
                }
                for model in MODEL_FILES
            }
            panels.append(
                annotate_tile(
                    clip, frame, "right", tile_id, tile_gt, boxes_for_tile
                )
            )

        # Add a label strip as the first panel.
        label = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        cv2.putText(
            label,
            clip,
            (5, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            label,
            f"f{frame}",
            (5, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        panels = [label] + panels
        rows.append(np.concatenate(panels, axis=1))

    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(OUT / out_name), sheet)
    print(f"wrote {OUT / out_name} shape={sheet.shape}", flush=True)


def main() -> int:
    rows = long_no_seed_rows()
    # Representative hard subset: edge + very-small cases plus a few known
    # WiLoR/MediaPipe successes/failures.
    selected = [
        (r[0], r[1])
        for r in rows
        if (r[0], r[1])
        in {
            ("clip-001852.tar", 1),
            ("clip-001852.tar", 5),
            ("clip-001852.tar", 6),
            ("clip-001852.tar", 7),
            ("clip-001852.tar", 8),
            ("clip-001852.tar", 15),
            ("clip-001852.tar", 23),
            ("clip-001891.tar", 3),
            ("clip-001891.tar", 6),
            ("clip-001891.tar", 7),
            ("clip-002316.tar", 0),
        }
    ]
    if not selected:
        selected = rows[:10]
    make_contact_sheet(selected, "sanity_check_examples.png")
    make_contact_sheet(selected[:4], "public_initializer_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
