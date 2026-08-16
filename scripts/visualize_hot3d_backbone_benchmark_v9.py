#!/usr/bin/env python3
"""Contact-sheet visuals for v9 hand-backbone benchmark."""

from __future__ import annotations

import ast
import csv
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_tracking_toolkit import camera
from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
EGOFORCE_SELECTED = OUT / "egoforce_selected_predictions.json"
V8_SEED_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v8_multitile_full/seed_availability_o4_m1_m2.csv"
)


def parse_key(key: str):
    clip, frame, view = ast.literal_eval(key)
    return str(clip), int(frame), str(view)


def draw_hand(img, uv, color, label=None):
    uv = np.asarray(uv, dtype=np.int32)
    for i in range(uv.shape[0]):
        cv2.circle(img, tuple(uv[i]), 3, color, -1, lineType=cv2.LINE_AA)
    bones = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    for a, b in bones:
        if a < uv.shape[0] and b < uv.shape[0]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(
            img,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def make_panel(raw_img, gt_uv, pred_uv, title):
    img = raw_img.copy()
    if gt_uv is not None:
        draw_hand(img, gt_uv, (0, 220, 0), "GT")
    if pred_uv is not None:
        draw_hand(img, pred_uv, (0, 0, 255), "EgoForce")
    h, w = img.shape[:2]
    scale = 360 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.putText(
        img,
        title,
        (6, img.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return img


def main() -> int:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    egoforce = json.loads(EGOFORCE_SELECTED.read_text(encoding="utf-8"))
    egoforce_by_key = {parse_key(k): v for k, v in egoforce.items()}
    v8_seed = {
        (r["clip"], int(r["frame"])): r
        for r in csv.DictReader(V8_SEED_CSV.open(newline="", encoding="utf-8"))
    }
    mano = MANOHandModel(str(MANO_DIR))
    gt_world_cache = {}
    for clip in sorted(clip_entry.keys()):
        entry = clip_entry[clip]
        run_dir = Path(entry["run_dir"]).resolve()
        frame_numbers, world = _build_gt_world(run_dir, mano)
        gt_world_cache[clip] = {
            int(f): world[i] for i, f in enumerate(frame_numbers)
        }

    cases = [
        ("clip-001891.tar", 3, "both_success"),
        ("clip-001891.tar", 6, "both_success"),
        ("clip-001852.tar", 6, "left_pred_exists_but_miss"),
        ("clip-001852.tar", 13, "left_pred_exists_but_miss"),
        ("clip-002316.tar", 0, "right_success_left_bad"),
    ]

    panels = []
    for clip, frame, label in cases:
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            images = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                raw = cv2.imdecode(
                    np.frombuffer(
                        tar.extractfile(
                            f"{frame:06d}.image_{stream_id}.jpg"
                        ).read(),
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )
                images[view] = raw
        gt_world = gt_world_cache[clip][frame]
        row = []
        for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
            raw_cam = camera.from_json(cams[stream_id])
            gt_uv = raw_cam.world_to_window(gt_world)
            pred = egoforce_by_key.get((clip, frame, view))
            pred_uv = np.asarray(pred["pred_j2d_raw"]) if pred else None
            b0 = v8_seed.get((clip, frame), {})
            title = f"{clip[-4:]}/{frame} {view} {label} B0={b0.get('at_least_one','?')}"
            row.append(make_panel(images[view], gt_uv, pred_uv, title))
        panels.append(np.hstack(row))

    canvas = np.vstack(panels)
    cv2.imwrite(str(OUT / "egoforce_examples.png"), canvas)
    cv2.imwrite(str(OUT / "model_comparison_examples.png"), canvas)
    print("wrote", OUT / "egoforce_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
