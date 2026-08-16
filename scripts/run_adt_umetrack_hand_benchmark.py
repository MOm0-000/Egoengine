#!/usr/bin/env python3
"""H3 UmeTrack ablation on the 10 fixed ADT stereo clips.

UmeTrack is a native multi-view hand tracker, but its released inference
scripts assume GT hand poses and Meta fisheye-style camera streams. This
adapter builds the required InputFrame/crop cameras from the existing HaMeR
left/right observations and the ADT rectified stereo calibration, then runs
the model's unknown-skeleton scale-calibration + tracking path. The output QC
uses the same fixed stereo/reprojection thresholds as H2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UMETRACK_ROOT = REPO_ROOT / "third_party" / "UmeTrack"
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
OUTPUT_SUMMARY = RUNS_ROOT / "adt_umetrack_hand_benchmark_summary.json"

sys.path.insert(0, str(UMETRACK_ROOT))

from lib.common.camera import PinholePlaneCameraModel
from lib.common.crop import gen_crop_parameters_from_points
from lib.common.hand import LEFT_HAND_INDEX, RIGHT_HAND_INDEX, HandModel, scaled_hand_model
from lib.models.model_loader import load_pretrained_model
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import _load_json, load_hand_model_from_dict

REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
MIN_REPROJECTION_ERROR_PX = 3.0
MIN_JOINT_DEPTH_M = 0.05
MAX_JOINT_DEPTH_M = 3.0
MIN_REQUIRED_JOINT_RATE = 0.70
MIN_REQUIRED_FRAME_RATE = 0.60
CROP_SIZE = (96, 96)


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def _load_hamer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _fps(run_dir: Path) -> float:
    path = run_dir / "frames" / "frame_index.json"
    if path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8")).get("fps", 30.0))
    return 30.0


def _make_camera(K: np.ndarray, camera_to_world_xf: np.ndarray, size: int = 512) -> PinholePlaneCameraModel:
    return PinholePlaneCameraModel(
        width=size,
        height=size,
        f=(float(K[0, 0]), float(K[1, 1])),
        c=(float(K[0, 2]), float(K[1, 2])),
        distort_coeffs=[],
        camera_to_world_xf=camera_to_world_xf.astype(np.float64),
    )


def _bbox_from_joints(uv: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float] | None:
    uv = np.asarray(uv, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    sel = valid & np.isfinite(uv).all(axis=-1)
    if not sel.any():
        return None
    pts = uv[sel]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max((x1 - x0) / 2.0, (y1 - y0) / 2.0, 20.0) * 1.35
    return cx - half, cy - half, cx + half, cy + half


def _bbox_world_points(camera: PinholePlaneCameraModel, bbox: tuple[float, float, float, float], depth_m: float) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    eye = np.empty((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners):
        x_ndc = (u - camera.c[0]) / camera.f[0]
        y_ndc = (v - camera.c[1]) / camera.f[1]
        eye[i] = [x_ndc * depth_m, y_ndc * depth_m, depth_m]
    return camera.eye_to_world(eye)


def _build_crop_cameras(
    cameras: list[PinholePlaneCameraModel],
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    hand_idx: int,
    depth_m: float,
) -> dict[int, PinholePlaneCameraModel]:
    crops: dict[int, PinholePlaneCameraModel] = {}
    observations = [left_uv, right_uv]
    valids = [left_valid, right_valid]
    for cam_idx, camera in enumerate(cameras):
        bbox = _bbox_from_joints(observations[cam_idx], valids[cam_idx])
        if bbox is None:
            continue
        pts_world = _bbox_world_points(camera, bbox, max(depth_m, MIN_JOINT_DEPTH_M))
        try:
            crop = gen_crop_parameters_from_points(
                camera,
                pts_world,
                CROP_SIZE,
                mirror_img_x=(hand_idx == RIGHT_HAND_INDEX),
                camera_angle=0.0,
                focal_multiplier=0.9,
            )
        except Exception:
            continue
        crops[cam_idx] = crop
    return crops


def _track_one_run(
    run_dir: Path,
    model,
    tracker: HandTracker,
    generic_hand_model: HandModel,
    calibration_frames: int,
) -> dict[str, Any]:
    root = run_dir
    left_hamer = _load_hamer(root / "hands_hamer" / "hamer_raw.npz")
    right_hamer = _load_hamer(root / "hands_right_hamer" / "hamer_raw.npz")
    if not np.array_equal(left_hamer["frame_indices"], right_hamer["frame_indices"]):
        raise ValueError("left/right HaMeR timelines differ")

    stereo = json.loads((root / "calibration" / "stereo.json").read_text(encoding="utf-8"))
    baseline = float(stereo["baseline_m"])
    K_left = np.load(root / "calibration" / "intrinsics.npy").astype(np.float64)
    K_right = np.load(root / "calibration" / "intrinsics_right.npy").astype(np.float64)
    left_abs = left_hamer["joints_camera_rootrel"] + left_hamer["translation_camera"][:, :, None]
    right_abs = right_hamer["joints_camera_rootrel"] + right_hamer["translation_camera"][:, :, None]
    left_uv = _project(K_left, left_abs)
    right_uv = _project(K_right, right_abs)

    left_world = np.eye(4, dtype=np.float64)
    right_world = np.eye(4, dtype=np.float64)
    right_world[0, 3] = baseline
    left_camera = _make_camera(K_left, left_world)
    right_camera = _make_camera(K_right, right_world)
    cameras = [left_camera, right_camera]

    timestamps = np.asarray(left_hamer["timestamps_s"], dtype=np.float64)
    frame_indices = np.asarray(left_hamer["frame_indices"], dtype=np.int64)
    T = len(frame_indices)
    valid = left_hamer["valid"] & right_hamer["valid"]

    predicted_scales: dict[int, list[float]] = {LEFT_HAND_INDEX: [], RIGHT_HAND_INDEX: []}
    for t in range(min(calibration_frames, T)):
        left_img = cv2.imread(str(root / "frames" / "rgb" / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(root / "frames" / "right" / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue
        input_frame = InputFrame(
            views=[
                ViewData(image=left_img, camera=left_camera, camera_angle=0.0),
                ViewData(image=right_img, camera=right_camera, camera_angle=0.0),
            ]
        )
        for hand_idx in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
            if not (left_hamer["valid"][t, hand_idx] and right_hamer["valid"][t, hand_idx]):
                continue
            depth = float(np.median(left_abs[t, hand_idx, :, 2])) if np.any(left_hamer["valid"][t, hand_idx]) else 0.5
            depth = float(np.clip(depth, 0.1, 2.0))
            crop_cameras = {
                hand_idx: _build_crop_cameras(
                    cameras,
                    left_uv[t, hand_idx],
                    right_uv[t, hand_idx],
                    left_hamer["valid"][t, hand_idx],
                    right_hamer["valid"][t, hand_idx],
                    hand_idx,
                    depth,
                )
            }
            if len(crop_cameras.get(hand_idx, {})) != 2:
                continue
            try:
                res = tracker.track_frame_and_calibrate_scale(input_frame, crop_cameras)
            except Exception:
                continue
            if hand_idx in res.predicted_scales:
                predicted_scales[hand_idx].append(float(res.predicted_scales[hand_idx]))

    scales: dict[int, float] = {}
    for hand_idx in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
        values = np.asarray(predicted_scales[hand_idx], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        scales[hand_idx] = float(np.median(values)) if values.size else 1.0
    mean_scale = float(np.mean(list(scales.values())))
    calibrated_model = scaled_hand_model(generic_hand_model, mean_scale)
    tracker.reset_history()

    pred_joints = np.full((T, 2, 21, 3), np.nan, dtype=np.float64)
    pred_valid = np.zeros((T, 2, 21), dtype=bool)
    left_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    right_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    vertical_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    num_views = np.zeros((T, 2), dtype=np.int64)

    for t in range(T):
        left_img = cv2.imread(str(root / "frames" / "rgb" / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(root / "frames" / "right" / f"{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue
        input_frame = InputFrame(
            views=[
                ViewData(image=left_img, camera=left_camera, camera_angle=0.0),
                ViewData(image=right_img, camera=right_camera, camera_angle=0.0),
            ]
        )
        crop_cameras: dict[int, dict[int, PinholePlaneCameraModel]] = {}
        for hand_idx in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
            if not (left_hamer["valid"][t, hand_idx] or right_hamer["valid"][t, hand_idx]):
                continue
            depth = float(np.median(left_abs[t, hand_idx, :, 2])) if np.any(left_hamer["valid"][t, hand_idx]) else 0.5
            depth = float(np.clip(depth, 0.1, 2.0))
            crop = _build_crop_cameras(
                cameras,
                left_uv[t, hand_idx],
                right_uv[t, hand_idx],
                left_hamer["valid"][t, hand_idx],
                right_hamer["valid"][t, hand_idx],
                hand_idx,
                depth,
            )
            if crop:
                crop_cameras[hand_idx] = crop
        if not crop_cameras:
            continue
        try:
            res = tracker.track_frame(input_frame, calibrated_model, crop_cameras)
        except Exception:
            continue
        for hand_idx, pose in res.hand_poses.items():
            landmarks_mm = landmarks_from_hand_pose(calibrated_model, pose, hand_idx)
            landmarks_m = landmarks_mm.astype(np.float64) / 1000.0
            pred_joints[t, hand_idx] = landmarks_m
            num_views[t, hand_idx] = int(res.num_views.get(hand_idx, 0))
            proj_left = _project(K_left, landmarks_m)
            proj_right = _project(K_right, landmarks_m - np.asarray([baseline, 0.0, 0.0]))
            obs_left = left_uv[t, hand_idx]
            obs_right = right_uv[t, hand_idx]
            obs_valid = (
                left_hamer["valid"][t, hand_idx, None]
                & right_hamer["valid"][t, hand_idx, None]
                & np.isfinite(obs_left).all(axis=-1)
                & np.isfinite(obs_right).all(axis=-1)
            )
            le = np.linalg.norm(proj_left - obs_left, axis=-1)
            re = np.linalg.norm(proj_right - obs_right, axis=-1)
            ve = np.abs(proj_left[..., 1] - proj_right[..., 1])
            left_error[t, hand_idx] = le
            right_error[t, hand_idx] = re
            vertical_error[t, hand_idx] = ve
            joint_ok = (
                obs_valid
                & np.isfinite(landmarks_m).all(axis=-1)
                & (landmarks_m[..., 2] >= MIN_JOINT_DEPTH_M)
                & (landmarks_m[..., 2] <= MAX_JOINT_DEPTH_M)
                & (le <= MIN_REPROJECTION_ERROR_PX)
                & (re <= MIN_REPROJECTION_ERROR_PX)
                & (ve <= MIN_REPROJECTION_ERROR_PX)
            )
            pred_valid[t, hand_idx] = joint_ok

    frame_valid = pred_valid[:, :, REQUIRED].all(axis=-1)
    joint_rate = pred_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (frame_rate >= MIN_REQUIRED_FRAME_RATE)

    output_dir = root / "hands_umetrack"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "umetrack_raw.npz",
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        side=np.asarray([[0, 1]] * T, dtype=np.int8),
        valid=pred_valid,
        joints_world_m=pred_joints,
        left_reprojection_error_px=left_error,
        right_reprojection_error_px=right_error,
        vertical_error_px=vertical_error,
        num_views=num_views,
    )
    metrics = {
        "accepted_any_hand": bool(accepted.any()),
        "accepted_left": bool(accepted[LEFT_HAND_INDEX]),
        "accepted_right": bool(accepted[RIGHT_HAND_INDEX]),
        "joint_valid_rate_left": float(joint_rate[LEFT_HAND_INDEX]),
        "joint_valid_rate_right": float(joint_rate[RIGHT_HAND_INDEX]),
        "required_frame_valid_rate_left": float(frame_rate[LEFT_HAND_INDEX]),
        "required_frame_valid_rate_right": float(frame_rate[RIGHT_HAND_INDEX]),
        "scale_left": scales[LEFT_HAND_INDEX],
        "scale_right": scales[RIGHT_HAND_INDEX],
        "calibrated_scale": mean_scale,
    }
    (output_dir / "umetrack_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"run_dir": str(root), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    model_path = UMETRACK_ROOT / "pretrained_models" / "pretrained_weights.torch"
    model = load_pretrained_model(str(model_path))
    model.eval()
    tracker = HandTracker(model, HandTrackerOpts())
    generic = load_hand_model_from_dict(_load_json(str(UMETRACK_ROOT / "dataset" / "generic_hand_model.json")))

    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if args.max_runs:
        rows = rows[: args.max_runs]
    records = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        print(f"[umetrack] {run_dir.name}", flush=True)
        try:
            record = _track_one_run(run_dir, model, tracker, generic, args.calibration_frames)
        except Exception as exc:
            record = {"run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"}
        record.update({"sequence": row["sequence"], "prototype": row["prototype"], "window": row["window"]})
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    OUTPUT_SUMMARY.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUTPUT_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
