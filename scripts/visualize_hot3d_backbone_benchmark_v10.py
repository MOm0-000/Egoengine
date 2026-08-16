#!/usr/bin/env python3
"""Contact sheets for v10 EgoForce failures and Hybrid rescues."""

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
    _build_gt_world,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10"
V9_SEED = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v9/seed_recall_38.csv"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
E1_SELECTED = OUT / "egoforce_full_selected_predictions.json"
FAILURE_MATRIX = OUT / "egoforce_failure_stage_matrix.csv"
OVERLAP = OUT / "seed_overlap_38.csv"


def parse_key(key: str):
    clip, frame, view = ast.literal_eval(key)
    return str(clip), int(frame), str(view)


def draw_hand(img, uv, color):
    uv = np.asarray(uv, dtype=np.int32)
    for i in range(min(21, uv.shape[0])):
        cv2.circle(img, tuple(uv[i]), 3, color, -1, lineType=cv2.LINE_AA)
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
        (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for a, b in bones:
        if a < uv.shape[0] and b < uv.shape[0]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 1, lineType=cv2.LINE_AA)


def panel(clip, frame, view, gt_world, selected, title):
    with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
        cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
        stream = "1201-1" if view == "left" else "1201-2"
        raw = cv2.imdecode(
            np.frombuffer(
                tar.extractfile(f"{frame:06d}.image_{stream}.jpg").read(),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
    cam = camera.from_json(cams[stream])
    gt_uv = cam.world_to_window(gt_world)
    pred = selected.get((clip, frame, view))
    img = raw.copy()
    draw_hand(img, gt_uv, (0, 220, 0))
    if pred is not None:
        draw_hand(img, np.asarray(pred["pred_j2d_raw"]), (0, 0, 255))
    scale = 360 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    cv2.putText(img, title, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)
    return img


def main() -> int:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    selected = {parse_key(k): v for k, v in json.loads(E1_SELECTED.read_text(encoding="utf-8")).items()}
    failure_rows = list(csv.DictReader(FAILURE_MATRIX.open(newline="", encoding="utf-8")))
    overlap_rows = list(csv.DictReader(OVERLAP.open(newline="", encoding="utf-8")))

    mano = MANOHandModel(str(MANO_DIR))
    gt_cache = {}
    for clip in sorted(clip_entry.keys()):
        run_dir = Path(clip_entry[clip]["run_dir"]).resolve()
        fn, world = _build_gt_world(run_dir, mano)
        gt_cache[clip] = {int(f): world[i] for i, f in enumerate(fn)}

    # Failure examples: one view per important stage.
    stages = {
        "A_no_hand_proposal": None,
        "B_no_forearm_proposal": None,
        "F_ray_space_solve_poor": None,
        "success": None,
    }
    for r in failure_rows:
        stage = r["failure_stage"]
        if stage in stages and stages[stage] is None:
            stages[stage] = r
    failure_panels = []
    for stage, r in stages.items():
        if r is None:
            continue
        clip = r["clip"]; frame = int(r["frame"]); view = r["camera"]
        title = f"{clip[-4:]}/{frame}/{view} {stage}"
        failure_panels.append(
            panel(clip, frame, view, gt_cache[clip][frame], selected, title)
        )
    if failure_panels:
        cv2.imwrite(str(OUT / "model_failure_examples.png"), np.vstack(failure_panels))

    # Hybrid rescue examples: WiLoR-only, EgoForce-only, both.
    wanted = {"wiLor_only": None, "egoforce_only": None, "both_success": None}
    for r in overlap_rows:
        if r["wiLor_only"] == "True" and wanted["wiLor_only"] is None:
            wanted["wiLor_only"] = r
        if r["egoforce_only"] == "True" and wanted["egoforce_only"] is None:
            wanted["egoforce_only"] = r
        if r["both_success"] == "True" and wanted["both_success"] is None:
            wanted["both_success"] = r
    rescue_panels = []
    for label, r in wanted.items():
        if r is None:
            continue
        clip = r["clip"]; frame = int(r["frame"])
        row = [
            panel(clip, frame, view, gt_cache[clip][frame], selected, f"{label} {view}")
            for view in ("left", "right")
        ]
        rescue_panels.append(np.hstack(row))
    if rescue_panels:
        cv2.imwrite(str(OUT / "hybrid_rescue_examples.png"), np.vstack(rescue_panels))

    print("wrote v10 example images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
