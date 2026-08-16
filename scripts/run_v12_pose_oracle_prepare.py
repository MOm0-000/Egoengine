#!/usr/bin/env python3
"""Prepare v12 oracle pose crops: O-A official/UmeTrack crop and O-B GT-bbox local crop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any

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
from hand_tracking_toolkit.dataset import (
    HandSide,
    decode_hand_crop_params,
    warp_image,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from hand_tracking_toolkit.camera_distortion import OVR624Distortion

from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    _build_gt_world,
)


OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CROP_DIR = OUT / "crops"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
CROP_SIZE = 256
OB_FOV_DEG = 25.0


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_gt_cache(rows):
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    mano = MANOHandModel(str(MANO_DIR))
    cache = {}
    for clip in sorted({c for c, _ in rows}):
        run_dir = Path(clip_entry[clip]["run_dir"]).resolve()
        frames, world = _build_gt_world(run_dir, mano)
        cache[clip] = {
            int(f): world[i] for i, f in enumerate(frames)
        }
    return cache


def _normalize(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def local_camera_from_center(
    raw_cam,
    center_uv,
    fov_deg,
    size,
):
    # Unproject raw fisheye ray for the desired bbox center.
    q = (np.asarray(center_uv, dtype=np.float64) - np.asarray(raw_cam.c)) / np.asarray(
        raw_cam.f
    )
    if hasattr(raw_cam.distort, "inverse_evaluate"):
        q = raw_cam.distort.inverse_evaluate(q)
    eye_dir = raw_cam.unproject(q)
    eye_dir = _normalize(eye_dir)
    center = raw_cam.pos()
    z_axis = raw_cam.orient() @ eye_dir
    raw_y = raw_cam.orient()[:, 1]
    y_axis = _normalize(raw_y - z_axis * float(np.dot(raw_y, z_axis)))
    if float(np.linalg.norm(y_axis)) < 1e-6:
        raw_y = raw_cam.orient()[:, 0]
        y_axis = _normalize(raw_y - z_axis * float(np.dot(raw_y, z_axis)))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = center
    f = size / 2.0 / math.tan(math.radians(fov_deg) / 2.0)
    return camera.PinholePlaneCameraModel(
        width=size,
        height=size,
        f=(f, f),
        c=((size - 1) / 2.0, (size - 1) / 2.0),
        distort_coeffs=[],
        T_world_from_eye=T,
    )


def raw_bbox_center(raw_cam, gt_world):
    uv = np.asarray(raw_cam.world_to_window(gt_world), dtype=np.float64)
    eye = np.asarray(raw_cam.world_to_eye(gt_world), dtype=np.float64)
    req = np.asarray(REQUIRED, dtype=np.int64)
    inside = (
        np.isfinite(uv[req]).all(axis=-1)
        & (eye[req, 2] > 0)
        & raw_cam.w_visible(uv[req])
    )
    pts = uv[req][inside]
    if len(pts) == 0:
        return None, None
    center = np.mean(pts, axis=0)
    bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    return center, bbox


def prepare_crop(
    clip,
    frame,
    view,
    stream_id,
    raw_cam,
    raw_image,
    crop_cam,
):
    out_dir = CROP_DIR / clip.replace(".tar", "") / f"{frame:06d}" / view
    out_dir.mkdir(parents=True, exist_ok=True)
    warped = warp_image(raw_cam, crop_cam, raw_image)
    if warped.ndim == 2:
        warped = np.stack([warped, warped, warped], axis=-1)
    warped = warped.astype(np.uint8)
    return warped


def main() -> int:
    args = parse_args()
    rows = long_no_seed_rows()
    if args.limit:
        rows = rows[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    gt_cache = build_gt_cache(rows)
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {item["clip"]: item for item in manifest["items"]}
    protocol = {
        "O-A": {
            "name": "official_hot3d_umetrack_crop",
            "source": "hand_crops.json",
            "crop_size": CROP_SIZE,
        },
        "O-B": {
            "name": "gt_bbox_local_perspective_crop",
            "source": "GT joints projected to raw fisheye",
            "crop_size": CROP_SIZE,
            "fov_deg": OB_FOV_DEG,
        },
    }
    rows_out = []

    for clip, frame in rows:
        selected_side = clip_entry[clip]["selected_side"]
        gt_world = gt_cache[clip][frame]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            hand_crops = json.load(tar.extractfile(f"{frame:06d}.hand_crops.json"))
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                raw_cam = camera.from_json(cams[stream_id])
                raw_bgr = cv2.imdecode(
                    np.frombuffer(
                        tar.extractfile(
                            f"{frame:06d}.image_{stream_id}.jpg"
                        ).read(),
                        np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

                # O-A official crop.
                crop_cams = decode_hand_crop_params(hand_crops, CROP_SIZE)
                side_enum = (
                    HandSide.LEFT
                    if selected_side == "left"
                    else HandSide.RIGHT
                )
                oa_cam = crop_cams[side_enum].get(stream_id)
                oa_ok = oa_cam is not None
                if oa_ok:
                    oa_img = prepare_crop(
                        clip, frame, view, stream_id, raw_cam, raw_bgr, oa_cam
                    )
                    cv2.imwrite(
                        str(
                            CROP_DIR
                            / clip.replace(".tar", "")
                            / f"{frame:06d}"
                            / view
                            / "O-A.jpg"
                        ),
                        oa_img,
                    )
                    oa_uv = np.asarray(oa_cam.world_to_window(gt_world), dtype=np.float64)
                    oa_eye = np.asarray(oa_cam.world_to_eye(gt_world), dtype=np.float64)
                    oa_vis = (
                        np.isfinite(oa_uv).all(axis=-1)
                        & (oa_eye[:, 2] > 0)
                        & oa_cam.w_visible(oa_uv)
                    )
                    oa_bbox = [float(oa_uv[:, 0].min()), float(oa_uv[:, 1].min()), float(oa_uv[:, 0].max()), float(oa_uv[:, 1].max())]
                    oa_f = float(oa_cam.f[0])
                    oa_T = oa_cam.T_world_from_eye.tolist()
                else:
                    oa_uv = None
                    oa_vis = None
                    oa_bbox = None
                    oa_f = None
                    oa_T = None

                # O-B GT bbox local crop.
                center, raw_bbox = raw_bbox_center(raw_cam, gt_world)
                ob_ok = center is not None
                if ob_ok:
                    ob_cam = local_camera_from_center(
                        raw_cam, center, OB_FOV_DEG, CROP_SIZE
                    )
                    ob_img = prepare_crop(
                        clip, frame, view, stream_id, raw_cam, raw_bgr, ob_cam
                    )
                    cv2.imwrite(
                        str(
                            CROP_DIR
                            / clip.replace(".tar", "")
                            / f"{frame:06d}"
                            / view
                            / "O-B.jpg"
                        ),
                        ob_img,
                    )
                    ob_uv = np.asarray(ob_cam.world_to_window(gt_world), dtype=np.float64)
                    ob_eye = np.asarray(ob_cam.world_to_eye(gt_world), dtype=np.float64)
                    ob_vis = (
                        np.isfinite(ob_uv).all(axis=-1)
                        & (ob_eye[:, 2] > 0)
                        & ob_cam.w_visible(ob_uv)
                    )
                    ob_bbox = [float(ob_uv[:, 0].min()), float(ob_uv[:, 1].min()), float(ob_uv[:, 0].max()), float(ob_uv[:, 1].max())]
                    ob_f = float(ob_cam.f[0])
                    ob_T = ob_cam.T_world_from_eye.tolist()
                else:
                    ob_uv = None
                    ob_vis = None
                    ob_bbox = None
                    ob_f = None
                    ob_T = None

                rows_out.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "stream_id": stream_id,
                        "selected_side": selected_side,
                        "O-A_ok": oa_ok,
                        "O-B_ok": ob_ok,
                        "O-A_gt_bbox": json.dumps(oa_bbox) if oa_bbox else "",
                        "O-B_gt_bbox": json.dumps(ob_bbox) if ob_bbox else "",
                        "O-A_f": oa_f if oa_f is not None else "",
                        "O-B_f": ob_f if ob_f is not None else "",
                        "O-A_T_world_from_eye": json.dumps(oa_T) if oa_T else "",
                        "O-B_T_world_from_eye": json.dumps(ob_T) if ob_T else "",
                        "O-A_gt_joints_uv": json.dumps(oa_uv.tolist()) if oa_uv is not None else "",
                        "O-B_gt_joints_uv": json.dumps(ob_uv.tolist()) if ob_uv is not None else "",
                        "O-A_gt_joints_visible": json.dumps(oa_vis.tolist()) if oa_vis is not None else "",
                        "O-B_gt_joints_visible": json.dumps(ob_vis.tolist()) if ob_vis is not None else "",
                        "raw_bbox": json.dumps(raw_bbox) if raw_bbox else "",
                    }
                )

    with (OUT / "oracle_crop_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    (OUT / "oracle_crop_protocols.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(rows_out)} oracle crop rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
