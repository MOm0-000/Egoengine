#!/usr/bin/env python3
"""v11 Public Ego Hand Initializer Benchmark: M1 tile preparation + models.

The script is split into stages so WiLoR/MediaPipe can run in v2s-wilor and
EgoForce's official YOLO hand detector can run in v2s-egoforce.

Examples:
  conda run -n v2s-wilor python scripts/run_v11_public_initializer_benchmark.py \
      --stage generate
  conda run -n v2s-wilor python scripts/run_v11_public_initializer_benchmark.py \
      --stage detect-wilor
  conda run -n v2s-wilor python scripts/run_v11_public_initializer_benchmark.py \
      --stage detect-mediapipe
  conda run -n v2s-wilor python scripts/run_v11_public_initializer_benchmark.py \
      --stage summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from scripts.evaluate_hot3d_fov_ablation import (
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
    _transform_points,
)


OUT = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/public_hand_initializer_benchmark_v11"
)
TILE_DIR = OUT / "tiles"
SUBSET_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
TILE_PROTOCOL = {
    "name": "M1",
    "tiles": 5,
    "size": 256,
    "f": 200.0,
    "directions": [
        {"tile_id": 0, "yaw_deg": 0.0, "pitch_deg": 0.0},
        {"tile_id": 1, "yaw_deg": 35.0, "pitch_deg": 0.0},
        {"tile_id": 2, "yaw_deg": -35.0, "pitch_deg": 0.0},
        {"tile_id": 3, "yaw_deg": 0.0, "pitch_deg": 35.0},
        {"tile_id": 4, "yaw_deg": 0.0, "pitch_deg": -35.0},
    ],
}
POSITIVE_DEFINITION = {
    "required_joints": REQUIRED.tolist(),
    "min_required_visible_ratio": 0.5,
    "min_bbox_inside_ratio": 0.30,
    "min_bbox_diagonal_px": 12.0,
    "tile_protocol": TILE_PROTOCOL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "generate",
            "detect-wilor",
            "detect-mediapipe",
            "summarize",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def long_no_seed_rows() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append((row["clip"], int(row["frame"])))
    return rows


def clip_manifest_map() -> dict[str, dict[str, Any]]:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    return {item["clip"]: item for item in manifest["items"]}


def build_gt_cache(
    rows: list[tuple[str, int]],
) -> dict[str, dict[int, np.ndarray]]:
    clip_map = clip_manifest_map()
    mano_model = MANOHandModel(str(MANO_DIR))
    cache: dict[str, dict[int, np.ndarray]] = {}
    for clip in sorted({clip for clip, _ in rows}):
        run_dir = Path(clip_map[clip]["run_dir"]).resolve()
        frame_numbers, world = _build_gt_world(run_dir, mano_model)
        cache[clip] = {
            int(frame): world[i] for i, frame in enumerate(frame_numbers)
        }
    return cache


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def make_tile_camera(
    raw_cam: camera.CameraModel,
    yaw_deg: float,
    pitch_deg: float,
    size: int = 256,
    f: float = 200.0,
) -> camera.PinholePlaneCameraModel:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    direction_cam = np.asarray(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float64,
    )
    direction_cam = _normalize(direction_cam)
    center = raw_cam.pos()
    z_axis = raw_cam.orient() @ direction_cam
    world_point = center + z_axis

    z_axis = _normalize(np.asarray(world_point[:3], dtype=np.float64) - center)
    raw_y = raw_cam.orient()[:, 1]
    y_axis = _normalize(raw_y - z_axis * float(np.dot(raw_y, z_axis)))
    if float(np.linalg.norm(y_axis)) < 1e-6:
        raw_y = raw_cam.orient()[:, 0]
        y_axis = _normalize(raw_y - z_axis * float(np.dot(raw_y, z_axis)))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return camera.PinholePlaneCameraModel(
        width=size,
        height=size,
        f=[f, f],
        c=(size / 2.0, size / 2.0),
        distort_coeffs=[],
        T_world_from_eye=transform,
    )


def warp_raw_tile(
    raw_cam: camera.CameraModel,
    raw_image: np.ndarray,
    tile_cam: camera.PinholePlaneCameraModel,
) -> np.ndarray:
    src = np.asarray(raw_image)
    if src.ndim == 2:
        src = np.stack([src, src, src], axis=-1)
    warped = warp_image(
        src_camera=raw_cam,
        dst_camera=tile_cam,
        src_image=src,
        interpolation=cv2.INTER_LINEAR,
        depth_check=True,
    )
    if warped.ndim == 2:
        warped = np.stack([warped, warped, warped], axis=-1)
    return warped.astype(np.uint8)


def tile_positive_row(
    clip: str,
    frame: int,
    view: str,
    tile_id: int,
    yaw: float,
    pitch: float,
    tile_cam: camera.PinholePlaneCameraModel,
    gt_world: np.ndarray,
) -> dict[str, Any]:
    eye = tile_cam.world_to_eye(gt_world)
    uv = tile_cam.world_to_window(gt_world)
    eye = np.asarray(eye, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    size = tile_cam.width

    req = np.asarray(REQUIRED, dtype=np.int64)
    req_eye = eye[req]
    req_uv = uv[req]
    wrist_eye = eye[0]
    wrist_uv = uv[0]
    wrist_visible = bool(
        np.isfinite(wrist_uv).all()
        and wrist_eye[2] > 0
        and tile_cam.w_visible(wrist_uv)
    )
    all_joints_visible = (
        np.isfinite(uv).all(axis=-1)
        & (eye[:, 2] > 0)
        & tile_cam.w_visible(uv)
    ).tolist()
    req_inside = (
        np.isfinite(req_uv).all(axis=-1)
        & (req_eye[:, 2] > 0)
        & tile_cam.w_visible(req_uv)
    )
    joint_visible_ratio = float(req_inside.mean())

    visible_pts = req_uv[req_inside]
    if len(visible_pts) == 0:
        bbox = None
        bbox_diagonal = 0.0
        bbox_inside_ratio = 0.0
        distance_to_boundary = 0.0
    else:
        x0, y0 = float(np.min(visible_pts[:, 0])), float(np.min(visible_pts[:, 1]))
        x1, y1 = float(np.max(visible_pts[:, 0])), float(np.max(visible_pts[:, 1]))
        bbox = [x0, y0, x1, y1]
        bbox_w = max(0.0, x1 - x0)
        bbox_h = max(0.0, y1 - y0)
        bbox_diagonal = float(math.hypot(bbox_w, bbox_h))
        area = bbox_w * bbox_h
        clip_x0 = max(0.0, min(float(size), x0))
        clip_y0 = max(0.0, min(float(size), y0))
        clip_x1 = max(0.0, min(float(size), x1))
        clip_y1 = max(0.0, min(float(size), y1))
        clipped_area = max(0.0, clip_x1 - clip_x0) * max(
            0.0, clip_y1 - clip_y0
        )
        bbox_inside_ratio = float(clipped_area / max(area, 1e-9))
        distance_to_boundary = float(
            np.min(
                np.concatenate(
                    [
                        visible_pts[:, 0],
                        size - visible_pts[:, 0],
                        visible_pts[:, 1],
                        size - visible_pts[:, 1],
                    ]
                )
            )
        )

    positive = bool(
        len(visible_pts) > 0
        and joint_visible_ratio >= POSITIVE_DEFINITION["min_required_visible_ratio"]
        and bbox_inside_ratio >= POSITIVE_DEFINITION["min_bbox_inside_ratio"]
        and bbox_diagonal >= POSITIVE_DEFINITION["min_bbox_diagonal_px"]
    )
    return {
        "clip": clip,
        "frame": frame,
        "camera": view,
        "tile_id": tile_id,
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "gt_hand_in_tile": bool(len(visible_pts) > 0),
        "gt_joint_visible_ratio": joint_visible_ratio,
        "gt_bbox_inside_ratio": bbox_inside_ratio,
        "gt_bbox_diagonal": bbox_diagonal,
        "distance_to_tile_boundary": distance_to_boundary,
        "gt_positive": positive,
        "gt_bbox": json.dumps(bbox) if bbox is not None else "",
        "gt_wrist_uv": (
            json.dumps(wrist_uv.tolist())
            if wrist_visible
            else ""
        ),
        "gt_joints_uv": json.dumps(uv.tolist()),
        "gt_joints_visible": json.dumps(all_joints_visible),
    }


def raw_view_metrics(
    raw_cam: camera.CameraModel,
    gt_world: np.ndarray,
) -> dict[str, float]:
    eye = raw_cam.world_to_eye(gt_world)
    uv = raw_cam.world_to_window(gt_world)
    req_inside = (
        (np.asarray(eye)[REQUIRED, 2] > 0)
        & raw_cam.w_visible(np.asarray(uv)[REQUIRED])
    )
    visible_pts = np.asarray(uv)[REQUIRED][req_inside]
    if len(visible_pts) == 0:
        bbox_diag = float("nan")
        wrist_boundary = float("nan")
    else:
        wh = visible_pts.max(axis=0) - visible_pts.min(axis=0)
        bbox_diag = float(np.linalg.norm(wh))
        wrist = np.asarray(uv)[0]
        wrist_boundary = float(
            min(
                wrist[0],
                raw_cam.width - wrist[0],
                wrist[1],
                raw_cam.height - wrist[1],
            )
        )
    if not req_inside.any():
        failure_type = "invisible"
    elif not math.isnan(wrist_boundary) and wrist_boundary < 60:
        failure_type = "edge_hand"
    elif not math.isnan(bbox_diag) and bbox_diag < 60:
        failure_type = "very_small_hand"
    else:
        failure_type = "center_hand"
    return {
        "gt_visible_raw": bool(req_inside.any()),
        "gt_bbox_diag_px": bbox_diag,
        "gt_wrist_distance_to_image_boundary_px": wrist_boundary,
        "failure_type": failure_type,
    }


def generate_tiles(rows: list[tuple[str, int]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    gt_cache = build_gt_cache(rows)
    matrix_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    manifest = clip_manifest_map()

    for clip, frame in rows:
        entry = manifest[clip]
        gt_world = gt_cache[clip][frame]
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            images: dict[str, np.ndarray] = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                b = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()
                images[view] = cv2.imdecode(
                    np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR
                )

        for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
            raw_cam = camera.from_json(cams[stream_id])
            raw_img = images[view]
            metrics = raw_view_metrics(raw_cam, gt_world)
            metrics.update(
                {
                    "clip": clip,
                    "frame": frame,
                    "camera": view,
                    "selected_side": entry["selected_side"],
                }
            )
            view_rows.append(metrics)
            out_view = TILE_DIR / clip.replace(".tar", "") / f"{frame:06d}" / view
            out_view.mkdir(parents=True, exist_ok=True)
            for tile in TILE_PROTOCOL["directions"]:
                tile_id = tile["tile_id"]
                tile_cam = make_tile_camera(
                    raw_cam, tile["yaw_deg"], tile["pitch_deg"]
                )
                tile_img = warp_raw_tile(raw_cam, raw_img, tile_cam)
                out_path = out_view / f"tile_{tile_id}.jpg"
                cv2.imwrite(str(out_path), tile_img)
                matrix_rows.append(
                    tile_positive_row(
                        clip,
                        frame,
                        view,
                        tile_id,
                        tile["yaw_deg"],
                        tile["pitch_deg"],
                        tile_cam,
                        gt_world,
                    )
                )

    fieldnames = list(matrix_rows[0].keys())
    with (OUT / "tile_positive_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    with (OUT / "tile_view_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
        writer.writeheader()
        writer.writerows(view_rows)

    positive = [r for r in matrix_rows if r["gt_positive"]]
    print(
        f"wrote {len(matrix_rows)} tile rows, {len(positive)} GT-positive "
        f"tiles, {len(view_rows)} view rows",
        flush=True,
    )
    (OUT / "tile_positive_definition.json").write_text(
        json.dumps(POSITIVE_DEFINITION, indent=2), encoding="utf-8"
    )
    (OUT / "tile_protocol.json").write_text(
        json.dumps(TILE_PROTOCOL, indent=2), encoding="utf-8"
    )


def detect_wilor(rows: list[tuple[str, int]], device: str) -> None:
    from ultralytics import YOLO

    detector_path = REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"
    detector = YOLO(str(detector_path))
    if torch.cuda.is_available() and device == "cuda":
        detector.to("cuda")
    records: list[dict[str, Any]] = []
    for clip, frame in rows:
        clip_dir = TILE_DIR / clip.replace(".tar", "") / f"{frame:06d}"
        for view in ("left", "right"):
            for tile in TILE_PROTOCOL["directions"]:
                tile_id = tile["tile_id"]
                img_path = clip_dir / view / f"tile_{tile_id}.jpg"
                if not img_path.is_file():
                    continue
                bgr = cv2.imread(str(img_path))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = detector.predict(
                    rgb, conf=0.30, verbose=False, device=detector.device
                )[0]
                if result.boxes is None or len(result.boxes) == 0:
                    records.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "camera": view,
                            "tile_id": tile_id,
                            "model": "wilor_detector",
                            "proposal_valid": False,
                            "confidence": "",
                            "predicted_handedness": "",
                            "bbox_tile": "",
                            "keypoints_tile": "",
                        }
                    )
                    continue
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                classes = result.boxes.cls.cpu().numpy().astype(np.int64)
                keypoints = (
                    result.keypoints.xy.cpu().numpy().astype(np.float32)
                    if result.keypoints is not None
                    else None
                )
                for i in range(len(boxes)):
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
                            "model": "wilor_detector",
                            "proposal_valid": True,
                            "confidence": float(scores[i]),
                            "predicted_handedness": int(classes[i]),
                            "bbox_tile": json.dumps(boxes[i].tolist()),
                            "keypoints_tile": kp_json,
                        }
                    )
    out_path = OUT / "initializer_candidates_wilor_tile.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)


def detect_mediapipe(rows: list[tuple[str, int]]) -> None:
    import mediapipe as mp

    Hands = mp.solutions.hands.Hands
    records: list[dict[str, Any]] = []
    with Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for clip, frame in rows:
            clip_dir = TILE_DIR / clip.replace(".tar", "") / f"{frame:06d}"
            for view in ("left", "right"):
                for tile in TILE_PROTOCOL["directions"]:
                    tile_id = tile["tile_id"]
                    img_path = clip_dir / view / f"tile_{tile_id}.jpg"
                    if not img_path.is_file():
                        continue
                    bgr = cv2.imread(str(img_path))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)
                    if not result.multi_hand_landmarks:
                        records.append(
                            {
                                "clip": clip,
                                "frame": frame,
                                "camera": view,
                                "tile_id": tile_id,
                                "model": "mediapipe",
                                "proposal_valid": False,
                                "confidence": "",
                                "predicted_handedness": "",
                                "bbox_tile": "",
                                "keypoints_tile": "",
                            }
                        )
                        continue
                    h, w = rgb.shape[:2]
                    for hand_idx, landmarks in enumerate(
                        result.multi_hand_landmarks
                    ):
                        handedness = result.multi_handedness[hand_idx]
                        label = handedness.classification[0].label
                        score = float(handedness.classification[0].score)
                        pts = np.asarray(
                            [[lm.x * w, lm.y * h] for lm in landmarks.landmark],
                            dtype=np.float32,
                        )
                        x0, y0 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                        x1, y1 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                        bbox = [x0, y0, x1, y1]
                        records.append(
                            {
                                "clip": clip,
                                "frame": frame,
                                "camera": view,
                                "tile_id": tile_id,
                                "model": "mediapipe",
                                "proposal_valid": True,
                                "confidence": score,
                                "predicted_handedness": (
                                    1 if label == "Right" else 0
                                ),
                                "bbox_tile": json.dumps(bbox),
                                "keypoints_tile": json.dumps(pts.tolist()),
                            }
                        )
    out_path = OUT / "initializer_candidates_mediapipe_tile.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)


def summarize() -> None:
    """Post-process generated/detection CSVs into v11 metrics and summary.

    This lightweight stage imports the more complete summarize module only
    when present; keeping the execution boundary explicit.
    """
    from scripts.summarize_v11_public_initializer_benchmark import main

    raise SystemExit(main())


def main() -> int:
    args = parse_args()
    rows = long_no_seed_rows()
    if args.limit:
        rows = rows[: args.limit]
    print(f"v11 rows: {len(rows)}", flush=True)
    if args.stage == "generate":
        generate_tiles(rows)
    elif args.stage == "detect-wilor":
        detect_wilor(rows, args.device)
    elif args.stage == "detect-mediapipe":
        detect_mediapipe(rows)
    elif args.stage == "summarize":
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
