#!/usr/bin/env python3
"""O2: RAW-fisheye stereo-guided local perspective rectification."""

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
import imageio.v2 as imageio
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
from scripts.run_stereo_crop_inference import (
    _load_model,
    _infer_crop,
    _invert,
    _transform_points,
    _project,
    _crop_with_padding,
    _bbox_from_uv,
    _unproject_point,
    _enlarged_bbox,
)


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v2"
P1_MANIFEST = ABLATION_RUNS / "P1" / "manifest.json"
VALID_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/p1_valid_region_audit/frame_view_matrix.csv"
)
WIDTH = 640
HEIGHT = 480
DEPTH_STEPS = (0.30, 0.50, 0.80, 1.20)
LOCAL_PROTOCOLS = {
    "L1": {"width": 256, "height": 256, "f": 200.0},
    "L2": {"width": 384, "height": 384, "f": 200.0},
}


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _local_camera_for_point(
    raw_cam: camera.CameraModel,
    world_point: np.ndarray,
    protocol: dict[str, Any],
) -> camera.PinholePlaneCameraModel:
    center = raw_cam.pos()
    z_axis = _normalize(np.asarray(world_point[:3], dtype=np.float64) - center)
    raw_y = raw_cam.orient()[:, 1]
    y_axis = _normalize(
        raw_y - z_axis * float(np.dot(raw_y, z_axis))
    )
    if float(np.linalg.norm(y_axis)) < 1e-6:
        raw_y = raw_cam.orient()[:, 0]
        y_axis = _normalize(
            raw_y - z_axis * float(np.dot(raw_y, z_axis))
        )
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    width = int(protocol["width"])
    height = int(protocol["height"])
    focal = float(protocol["f"])
    return camera.PinholePlaneCameraModel(
        width=width,
        height=height,
        f=[focal, focal],
        c=(width / 2.0, height / 2.0),
        distort_coeffs=[],
        T_world_from_eye=transform,
    )


def _warp_raw_local(
    raw_cam: camera.CameraModel,
    raw_image: np.ndarray,
    local_cam: camera.PinholePlaneCameraModel,
) -> np.ndarray:
    src = np.asarray(raw_image)
    if src.ndim == 2:
        src = np.stack([src, src, src], axis=-1)
    warped = warp_image(
        src_camera=raw_cam,
        dst_camera=local_cam,
        src_image=src,
        interpolation=cv2.INTER_LINEAR,
        depth_check=True,
    )
    if warped.ndim == 2:
        warped = np.stack([warped, warped, warped], axis=-1)
    return warped.astype(np.uint8)


def _local_to_weak_p1(
    local_joints: np.ndarray,
    local_T: np.ndarray,
    weak_T: np.ndarray,
    weak_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world = np.stack(
        [_transform_points(local_T, p) for p in local_joints],
        axis=0,
    )
    weak_cam = np.stack(
        [_transform_points(_invert(weak_T), p) for p in world],
        axis=0,
    )
    uv = _project(weak_K, weak_cam)
    return weak_cam, uv


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
    recovery_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

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
                cams = json.load(tar.extractfile(f"{int(frame_number):06d}.cameras.json"))
                raw_left = camera.from_json(cams["1201-1"])
                raw_right = camera.from_json(cams["1201-2"])
                raw_img_left = imageio.imread(
                    tar.extractfile(f"{int(frame_number):06d}.image_1201-1.jpg")
                )
                raw_img_right = imageio.imread(
                    tar.extractfile(f"{int(frame_number):06d}.image_1201-2.jpg")
                )

                # O2 raw-local oracle on valid near-boundary frames.
                for view_name, raw_cam, raw_img, uv_gt in (
                    ("left", raw_left, raw_img_left, uv_left[t]),
                    ("right", raw_right, raw_img_right, uv_right[t]),
                ):
                    key = (clip, int(frame_number), view_name)
                    vr = valid_rows.get(key)
                    if vr is None or vr["gt_visible_valid"] != "True":
                        continue
                    dist = float(vr["distance_to_invalid_boundary_px"])
                    if dist >= 10.0:
                        continue
                    bbox = _bbox_from_uv(uv_gt, WIDTH, HEIGHT)
                    if bbox is None:
                        continue
                    # Use GT wrist world direction for local oracle.
                    gt_world_point = world[t, 0]
                    best_oracle = None
                    for protocol_name, protocol in LOCAL_PROTOCOLS.items():
                        local_cam = _local_camera_for_point(raw_cam, gt_world_point, protocol)
                        local_img = _warp_raw_local(raw_cam, raw_img, local_cam)
                        result = _infer_crop(
                            model_obj,
                            model_cfg,
                            detector,
                            dataset_cls,
                            recursive_to,
                            device,
                            local_img,
                            (0, 0),
                            np.asarray(
                                [
                                    [protocol["f"], 0, protocol["width"] / 2],
                                    [0, protocol["f"], protocol["height"] / 2],
                                    [0, 0, 1],
                                ],
                                dtype=np.float64,
                            ),
                            side_index,
                        )
                        if result is not None and (
                            best_oracle is None or result["score"] > best_oracle["score"]
                        ):
                            best_oracle = {
                                "protocol": protocol_name,
                                "result": result,
                                "score": result["score"],
                            }
                    if best_oracle is None:
                        oracle_rows.append(
                            {
                                "clip": clip,
                                "frame": int(frame_number),
                                "view": view_name,
                                "detected": False,
                                "score": None,
                                "metrics": None,
                            }
                        )
                        continue
                    local_T = _local_camera_for_point(
                        raw_cam, gt_world_point, LOCAL_PROTOCOLS[best_oracle["protocol"]]
                    ).T_world_from_eye
                    weak_T = T_left[t] if view_name == "left" else T_right[t]
                    weak_K = K_left if view_name == "left" else K_right
                    weak_cam, pred_uv = _local_to_weak_p1(
                        best_oracle["result"]["joints_camera"],
                        local_T,
                        weak_T,
                        weak_K,
                    )
                    errors = np.linalg.norm(pred_uv - uv_gt, axis=-1)
                    oracle_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "view": view_name,
                            "detected": True,
                            "score": best_oracle["score"],
                            "protocol": best_oracle["protocol"],
                            "metrics": {
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
                                    / max(
                                        math.hypot(
                                            bbox[2] - bbox[0],
                                            bbox[3] - bbox[1],
                                        ),
                                        1.0,
                                    )
                                ),
                            },
                        }
                    )

                # O2 stereo-guided recovery.
                strong_view = None
                if left_p[t] and right_p[t]:
                    strong_view = "left" if left_scores[t] >= right_scores[t] else "right"
                elif left_p[t]:
                    strong_view = "left"
                elif right_p[t]:
                    strong_view = "right"
                if strong_view is None:
                    recovery_rows.append(
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
                    recovery_rows.append(
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
                weak_raw_cam = raw_left if weak_view == "left" else raw_right
                weak_raw_img = raw_img_left if weak_view == "left" else raw_img_right
                weak_T = T_left[t] if weak_view == "left" else T_right[t]
                weak_K = K_left if weak_view == "left" else K_right
                best_candidate = None
                for depth in DEPTH_STEPS:
                    strong_point = _unproject_point(strong_K, strong_uv[0], depth)
                    strong_point_h = np.append(strong_point, 1.0)
                    world_point = strong_T @ strong_point_h
                    for protocol_name, protocol in LOCAL_PROTOCOLS.items():
                        local_cam = _local_camera_for_point(weak_raw_cam, world_point, protocol)
                        local_img = _warp_raw_local(weak_raw_cam, weak_raw_img, local_cam)
                        local_K = np.asarray(
                            [
                                [protocol["f"], 0, protocol["width"] / 2],
                                [0, protocol["f"], protocol["height"] / 2],
                                [0, 0, 1],
                            ],
                            dtype=np.float64,
                        )
                        result = _infer_crop(
                            model_obj,
                            model_cfg,
                            detector,
                            dataset_cls,
                            recursive_to,
                            device,
                            local_img,
                            (0, 0),
                            local_K,
                            side_index,
                        )
                        if result is None:
                            candidate_rows.append(
                                {
                                    "clip": clip,
                                    "frame": int(frame_number),
                                    "strong_view": strong_view,
                                    "weak_view": weak_view,
                                    "depth": depth,
                                    "protocol": protocol_name,
                                    "status": "no_detection",
                                    "score": None,
                                }
                            )
                            continue
                        weak_cam, pred_uv = _local_to_weak_p1(
                            result["joints_camera"],
                            local_cam.T_world_from_eye,
                            weak_T,
                            weak_K,
                        )
                        candidate_rows.append(
                            {
                                "clip": clip,
                                "frame": int(frame_number),
                                "strong_view": strong_view,
                                "weak_view": weak_view,
                                "depth": depth,
                                "protocol": protocol_name,
                                "status": "detected",
                                "score": result["score"],
                            }
                        )
                        if best_candidate is None or result["score"] > best_candidate["score"]:
                            best_candidate = {
                                "depth": depth,
                                "protocol": protocol_name,
                                "result": result,
                                "score": result["score"],
                                "local_T": local_cam.T_world_from_eye,
                            }
                recovered = best_candidate is not None
                recovered_metrics = None
                recovered_joints = None
                if recovered:
                    weak_cam, pred_uv = _local_to_weak_p1(
                        best_candidate["result"]["joints_camera"],
                        best_candidate["local_T"],
                        weak_T,
                        weak_K,
                    )
                    weak_gt_uv = uv_left[t] if weak_view == "left" else uv_right[t]
                    errors = np.linalg.norm(pred_uv - weak_gt_uv, axis=-1)
                    recovered_metrics = {
                        "median_2d_error_px": float(np.median(errors)),
                        "mean_2d_error_px": float(np.mean(errors)),
                        "pck5": float(np.mean(errors <= 5)),
                        "pck10": float(np.mean(errors <= 10)),
                        "pck20": float(np.mean(errors <= 20)),
                        "wrist_error_px": float(np.linalg.norm(pred_uv[0] - weak_gt_uv[0])),
                        "fingertip_error_px": float(
                            np.mean(
                                np.linalg.norm(
                                    pred_uv[FINGERTIPS] - weak_gt_uv[FINGERTIPS],
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
                    recovered_joints = weak_cam.tolist()
                recovery_rows.append(
                    {
                        "clip": clip,
                        "frame": int(frame_number),
                        "strong_view": strong_view,
                        "weak_view": weak_view,
                        "status": "recovered" if recovered else "no_weak_detection",
                        "recovered": recovered,
                        "score": best_candidate["score"] if recovered else None,
                        "depth": best_candidate["depth"] if recovered else None,
                        "protocol": best_candidate["protocol"] if recovered else None,
                        "recovered_metrics": recovered_metrics,
                        "recovered_joints_camera": recovered_joints,
                    }
                )

    oracle_path = OUTPUT_ROOT / f"raw_local_oracle_{args.model}.json"
    recovery_path = OUTPUT_ROOT / f"stereo_recovery_v2_{args.model}.json"
    candidate_path = OUTPUT_ROOT / f"candidate_scores_{args.model}.csv"
    oracle_path.write_text(
        json.dumps({"model": args.model, "rows": oracle_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    recovery_path.write_text(
        json.dumps({"model": args.model, "rows": recovery_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with candidate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip", "frame", "strong_view", "weak_view", "depth",
                "protocol", "status", "score",
            ]
        )
        for row in candidate_rows:
            writer.writerow(
                [
                    row["clip"], row["frame"], row["strong_view"], row["weak_view"],
                    row["depth"], row["protocol"], row["status"], row["score"],
                ]
            )
    print(oracle_path)
    print(recovery_path)
    print(candidate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
