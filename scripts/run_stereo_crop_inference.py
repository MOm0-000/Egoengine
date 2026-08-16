#!/usr/bin/env python3
"""Run O1-Oracle GT crop and O1-Stereo weak-view crop inference."""

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
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from scripts.evaluate_hot3d_fov_ablation import (
    ABLATION_RUNS,
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
    _load_model_artifacts,
    _model_observation,
    _pinhole_visibility,
)


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v1"
P1_MANIFEST = ABLATION_RUNS / "P1" / "manifest.json"
VALID_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/p1_valid_region_audit/frame_view_matrix.csv"
)
WIDTH = 640
HEIGHT = 480
DEPTH_RANGE = (0.15, 1.5)
DEPTH_STEPS = (0.15, 0.30, 0.50, 0.80, 1.20, 1.50)


def _invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def _load_model(model: str):
    previous_cwd = Path.cwd()
    if model == "wilor":
        root = REPO_ROOT / "third_party/WiLoR"
        sys.path.insert(0, str(root))
        os.chdir(root)
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        from wilor.utils import recursive_to

        model_obj, model_cfg = load_wilor(
            checkpoint_path=str(root / "pretrained_models/wilor_final.ckpt"),
            cfg_path=str(root / "pretrained_models/model_config.yaml"),
        )
        detector = YOLO(str(root / "pretrained_models/detector.pt"))
        os.chdir(previous_cwd)
        return model_obj, model_cfg, detector, ViTDetDataset, recursive_to
    root = REPO_ROOT / "third_party/hamer"
    sys.path.insert(0, str(root))
    os.chdir(root)
    import hamer
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.models import load_hamer
    from hamer.utils import recursive_to
    from ultralytics import YOLO

    hamer.configs.CACHE_DIR_HAMER = str(root / "_DATA")
    model_obj, model_cfg = load_hamer(
        str(root / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    )
    detector = YOLO(str(REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"))
    os.chdir(previous_cwd)
    return model_obj, model_cfg, detector, ViTDetDataset, recursive_to


def _camera_translation(pred_cam, box_center, box_size, K):
    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def _reflect_rotations(rotations: np.ndarray) -> np.ndarray:
    reflection = np.diag([-1.0, 1.0, 1.0])
    return reflection @ rotations @ reflection


def _crop_with_padding(image: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    height, width = image.shape[:2]
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(width, x1)
    y1c = min(height, y1)
    crop = image[y0c:y1c, x0c:x1c].copy()
    top = y0c - y0
    left = x0c - x0
    bottom = y1 - y1c
    right = x1 - x1c
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        crop = cv2.copyMakeBorder(
            crop,
            max(0, top),
            max(0, bottom),
            max(0, left),
            max(0, right),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return crop


def _infer_crop(
    model_obj,
    model_cfg,
    detector,
    dataset_cls,
    recursive_to,
    device,
    image: np.ndarray,
    crop_origin: tuple[int, int],
    K_orig: np.ndarray,
    selected_side_index: int,
) -> dict[str, Any] | None:
    detections = detector(image, conf=0.3, verbose=False)[0]
    boxes: list[list[float]] = []
    rights: list[int] = []
    scores: list[float] = []
    for detection in detections:
        data = detection.boxes.data.detach().cpu().reshape(-1).numpy()
        boxes.append(data[:4].tolist())
        scores.append(float(data[4]))
        rights.append(int(data[5]))
    if not boxes:
        return None
    box_array = np.asarray(boxes, dtype=np.float32)
    right_array = np.asarray(rights, dtype=np.float32)
    dataset = dataset_cls(
        model_cfg, image, box_array, right_array, rescale_factor=2.0, fp16=False
    )
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                dataset, batch_size=len(dataset), shuffle=False, num_workers=0
            )
        )
    )
    batch = recursive_to(batch, device)
    with torch.no_grad():
        prediction = model_obj(batch)
    pred_cam = prediction["pred_cam"].clone()
    pred_cam[:, 1] *= (2 * batch["right"] - 1)
    K_crop = K_orig.copy()
    K_crop[0, 2] -= crop_origin[0]
    K_crop[1, 2] -= crop_origin[1]
    camera_t = _camera_translation(
        pred_cam, batch["box_center"].float(), batch["box_size"].float(), K_crop
    ).detach().cpu().numpy()
    best_index = None
    best_score = -1.0
    for detection_index in range(len(boxes)):
        hand_index = 1 if rights[detection_index] else 0
        if hand_index != selected_side_index:
            continue
        if scores[detection_index] > best_score:
            best_score = scores[detection_index]
            best_index = detection_index
    if best_index is None:
        return None
    joint = (
        prediction["pred_keypoints_3d"][best_index].detach().cpu().numpy()
        + camera_t[best_index][None, :]
    )
    original_box = np.asarray(boxes[best_index], dtype=np.float64).copy()
    original_box[0] += crop_origin[0]
    original_box[1] += crop_origin[1]
    original_box[2] += crop_origin[0]
    original_box[3] += crop_origin[1]
    return {
        "joints_camera": joint,
        "score": float(best_score),
        "box_original": original_box.tolist(),
    }


def _bbox_from_uv(uv: np.ndarray, width: int, height: int):
    inside = (
        np.isfinite(uv).all(axis=-1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not inside.any():
        return None
    points = uv[inside]
    return (
        int(math.floor(np.min(points[:, 0]))),
        int(math.floor(np.min(points[:, 1]))),
        int(math.ceil(np.max(points[:, 0]))),
        int(math.ceil(np.max(points[:, 1]))),
    )


def _enlarged_bbox(bbox, ratio: float, width: int, height: int):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * ratio
    h = (y1 - y0) * ratio
    return int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))


def _unproject_point(K: np.ndarray, uv: np.ndarray, depth: float) -> np.ndarray:
    x = (float(uv[0]) - float(K[0, 2])) / float(K[0, 0]) * depth
    y = (float(uv[1]) - float(K[1, 2])) / float(K[1, 1]) * depth
    return np.asarray([x, y, depth], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wilor", "hamer"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_obj, model_cfg, detector, dataset_cls, recursive_to = _load_model(args.model)
    model_obj = model_obj.to(device).eval()
    detector = detector.to(device)
    mano_model = MANOHandModel(str(MANO_DIR))
    manifest = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))["items"]
    valid_rows = {
        (r["clip"], int(r["frame"]), r["camera_side"]): r
        for r in csv.DictReader(open(VALID_CSV, encoding="utf-8"))
    }

    oracle_rows: list[dict[str, Any]] = []
    stereo_rows: list[dict[str, Any]] = []
    for entry in manifest:
        run_dir = Path(entry["run_dir"]).resolve()
        clip = entry["clip"]
        side = entry["selected_side"]
        side_index = 0 if side == "left" else 1
        frame_numbers, world = _build_gt_world(run_dir, mano_model)
        pin_left, pin_right, uv_left, uv_right = _pinhole_visibility(run_dir, world)
        K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
        K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
        T_left = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
        T_right = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
        artifacts = _load_model_artifacts(run_dir, args.model)
        if "left" not in artifacts or "right" not in artifacts:
            continue
        left_obs = _model_observation(artifacts["left"], side_index, K_left)
        right_obs = _model_observation(artifacts["right"], side_index, K_right)
        left_p = left_obs[2]
        right_p = right_obs[2]
        left_scores = np.asarray(artifacts["left"]["score"][:, side_index], dtype=np.float64)
        right_scores = np.asarray(artifacts["right"]["score"][:, side_index], dtype=np.float64)
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for t, frame_number in enumerate(frame_numbers):
                if args.limit and t >= args.limit:
                    break
                for view_name, K, T, uv_gt, base_detected, base_score in (
                    ("left", K_left, T_left[t], uv_left[t], left_p[t], left_scores[t]),
                    ("right", K_right, T_right[t], uv_right[t], right_p[t], right_scores[t]),
                ):
                    key = (clip, int(frame_number), view_name)
                    valid_row = valid_rows.get(key)
                    if valid_row is None or valid_row["gt_visible_valid"] != "True":
                        continue
                    image_path = (
                        run_dir / "frames/rgb" / f"{t:06d}.png"
                        if view_name == "left"
                        else run_dir / "frames/right" / f"{t:06d}.png"
                    )
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    bbox = _bbox_from_uv(uv_gt, WIDTH, HEIGHT)
                    if bbox is None:
                        continue
                    # O1-Oracle.
                    crop_box = _enlarged_bbox(bbox, 1.5, WIDTH, HEIGHT)
                    crop = _crop_with_padding(
                        image,
                        crop_box[0],
                        crop_box[1],
                        crop_box[2],
                        crop_box[3],
                    )
                    oracle = _infer_crop(
                        model_obj,
                        model_cfg,
                        detector,
                        dataset_cls,
                        recursive_to,
                        device,
                        crop,
                        (crop_box[0], crop_box[1]),
                        K,
                        side_index,
                    )
                    oracle_detected = oracle is not None
                    oracle_metrics = None
                    if oracle is not None:
                        pred_uv = _project(K, oracle["joints_camera"])
                        errors = np.linalg.norm(pred_uv - uv_gt, axis=-1)
                        oracle_metrics = {
                            "median_2d_error_px": float(np.median(errors)),
                            "mean_2d_error_px": float(np.mean(errors)),
                            "pck5": float(np.mean(errors <= 5)),
                            "pck10": float(np.mean(errors <= 10)),
                            "pck20": float(np.mean(errors <= 20)),
                            "wrist_error_px": float(np.linalg.norm(pred_uv[0] - uv_gt[0])),
                            "fingertip_error_px": float(
                                np.mean(
                                    np.linalg.norm(
                                        pred_uv[FINGERTIPS] - uv_gt[FINGERTIPS],
                                        axis=-1,
                                    )
                                )
                            ),
                            "nme_bbox_diagonal": float(
                                np.mean(errors)
                                / max(math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]), 1.0)
                            ),
                        }
                    oracle_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "view": view_name,
                            "selected_hand": side,
                            "detected": oracle_detected,
                            "score": oracle["score"] if oracle else None,
                            "metrics": oracle_metrics,
                        }
                    )
                # O1-Stereo: per-frame strong->weak recovery.
                strong_view = None
                if left_p[t] and right_p[t]:
                    strong_view = "left" if left_scores[t] >= right_scores[t] else "right"
                elif left_p[t]:
                    strong_view = "left"
                elif right_p[t]:
                    strong_view = "right"
                if strong_view is None:
                    stereo_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "strong_view": None,
                            "weak_view": None,
                            "status": "baseline_both_miss",
                            "recovered": False,
                            "score": None,
                        }
                    )
                    continue
                weak_view = "right" if strong_view == "left" else "left"
                strong_artifact = artifacts["left"] if strong_view == "left" else artifacts["right"]
                strong_K = K_left if strong_view == "left" else K_right
                strong_T = T_left[t] if strong_view == "left" else T_right[t]
                strong_joints = (
                    np.asarray(strong_artifact["joints_camera_rootrel"][t, side_index], dtype=np.float64)
                    + np.asarray(strong_artifact["translation_camera"][t, side_index, None], dtype=np.float64)
                )
                strong_uv = _project(strong_K, strong_joints)
                strong_bbox = _bbox_from_uv(strong_uv, WIDTH, HEIGHT)
                if strong_bbox is None:
                    stereo_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "strong_view": strong_view,
                            "weak_view": weak_view,
                            "status": "strong_bbox_invalid",
                            "recovered": False,
                            "score": None,
                        }
                    )
                    continue
                weak_K = K_right if weak_view == "right" else K_left
                weak_T = T_right[t] if weak_view == "right" else T_left[t]
                weak_image_path = (
                    run_dir / "frames/right" / f"{t:06d}.png"
                    if weak_view == "right"
                    else run_dir / "frames/rgb" / f"{t:06d}.png"
                )
                weak_image = cv2.imread(str(weak_image_path))
                if weak_image is None:
                    continue
                strong_wrist_uv = strong_uv[0]
                margin = max(strong_bbox[2] - strong_bbox[0], strong_bbox[3] - strong_bbox[1]) * 0.75
                margin = max(30.0, margin)
                best_candidate = None
                candidate_rows = []
                for depth in DEPTH_STEPS:
                    strong_point = _unproject_point(strong_K, strong_wrist_uv, depth)
                    strong_point_h = np.append(strong_point, 1.0)
                    world_point = strong_T @ strong_point_h
                    weak_camera_point = _invert(weak_T) @ world_point
                    weak_uv = _project(weak_K, weak_camera_point[:3])
                    if not np.isfinite(weak_uv).all():
                        continue
                    x0 = int(round(weak_uv[0] - margin))
                    y0 = int(round(weak_uv[1] - margin))
                    x1 = int(round(weak_uv[0] + margin))
                    y1 = int(round(weak_uv[1] + margin))
                    crop = _crop_with_padding(weak_image, x0, y0, x1, y1)
                    result = _infer_crop(
                        model_obj,
                        model_cfg,
                        detector,
                        dataset_cls,
                        recursive_to,
                        device,
                        crop,
                        (x0, y0),
                        weak_K,
                        side_index,
                    )
                    if result is None:
                        candidate_rows.append(
                            {
                                "depth": depth,
                                "status": "no_detection",
                                "score": None,
                            }
                        )
                        continue
                    candidate_rows.append(
                        {
                            "depth": depth,
                            "status": "detected",
                            "score": result["score"],
                        }
                    )
                    if best_candidate is None or result["score"] > best_candidate["score"]:
                        best_candidate = {
                            "depth": depth,
                            "result": result,
                            "score": result["score"],
                        }
                recovered = best_candidate is not None
                recovered_metrics = None
                recovered_joints = None
                if recovered:
                    weak_joints = best_candidate["result"]["joints_camera"]
                    weak_pred_uv = _project(weak_K, weak_joints)
                    weak_gt_uv = uv_left[t] if weak_view == "left" else uv_right[t]
                    errors = np.linalg.norm(weak_pred_uv - weak_gt_uv, axis=-1)
                    recovered_metrics = {
                        "median_2d_error_px": float(np.median(errors)),
                        "mean_2d_error_px": float(np.mean(errors)),
                        "pck5": float(np.mean(errors <= 5)),
                        "pck10": float(np.mean(errors <= 10)),
                        "pck20": float(np.mean(errors <= 20)),
                        "wrist_error_px": float(np.linalg.norm(weak_pred_uv[0] - weak_gt_uv[0])),
                        "fingertip_error_px": float(
                            np.mean(
                                np.linalg.norm(
                                    weak_pred_uv[FINGERTIPS] - weak_gt_uv[FINGERTIPS],
                                    axis=-1,
                                )
                            )
                        ),
                        "nme_bbox_diagonal": float(
                            np.mean(errors)
                            / max(
                                math.hypot(
                                    strong_bbox[2] - strong_bbox[0],
                                    strong_bbox[3] - strong_bbox[1],
                                ),
                                1.0,
                            )
                        ),
                    }
                    recovered_joints = weak_joints.tolist()
                stereo_rows.append(
                    {
                        "clip": clip,
                        "frame": int(frame_number),
                        "strong_view": strong_view,
                        "weak_view": weak_view,
                        "status": "recovered" if recovered else "no_weak_detection",
                        "recovered": recovered,
                        "score": best_candidate["score"] if recovered else None,
                        "depth": best_candidate["depth"] if recovered else None,
                        "recovered_metrics": recovered_metrics,
                        "recovered_joints_camera": recovered_joints,
                        "candidates": candidate_rows,
                    }
                )

    oracle_path = OUTPUT_ROOT / f"oracle_crop_{args.model}.json"
    stereo_path = OUTPUT_ROOT / f"stereo_recovery_{args.model}.json"
    oracle_path.write_text(
        json.dumps({"model": args.model, "rows": oracle_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    stereo_path.write_text(
        json.dumps({"model": args.model, "rows": stereo_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(oracle_path)
    print(stereo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
