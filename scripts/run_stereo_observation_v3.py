#!/usr/bin/env python3
"""O3: stereo ROI forced WiLoR regression after detector diagnostic."""

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
from scripts.run_stereo_crop_inference import _load_model, _invert, _project
from scripts.run_stereo_raw_observation_v2 import (
    _local_camera_for_point,
    _warp_raw_local,
    _local_to_weak_p1,
)


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v3"
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


def _unproject_point(K: np.ndarray, uv: np.ndarray, depth: float) -> np.ndarray:
    x = (float(uv[0]) - float(K[0, 2])) / float(K[0, 0]) * depth
    y = (float(uv[1]) - float(K[1, 2])) / float(K[1, 1]) * depth
    return np.asarray([x, y, depth], dtype=np.float64)


def _detector_proposals(detector, image, threshold=0.05):
    detections = detector(image, conf=threshold, verbose=False)[0]
    proposals = []
    for detection in detections:
        data = detection.boxes.data.detach().cpu().reshape(-1).numpy()
        proposals.append(
            {
                "box": data[:4].tolist(),
                "score": float(data[4]),
                "class": int(data[5]),
            }
        )
    return proposals


def _forced_regression(
    model_obj,
    model_cfg,
    dataset_cls,
    recursive_to,
    device,
    image,
    right_hand: bool,
    local_K,
):
    height, width = image.shape[:2]
    box_array = np.asarray([[0, 0, width - 1, height - 1]], dtype=np.float32)
    right_array = np.asarray([1.0 if right_hand else 0.0], dtype=np.float32)
    dataset = dataset_cls(
        model_cfg, image, box_array, right_array, rescale_factor=2.0, fp16=False
    )
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
        )
    )
    batch = recursive_to(batch, device)
    with torch.no_grad():
        prediction = model_obj(batch)
    pred_cam = prediction["pred_cam"].clone()
    pred_cam[:, 1] *= (2 * batch["right"] - 1)
    focal = float((local_K[0, 0] + local_K[1, 1]) * 0.5)
    bs = batch["box_size"] * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (batch["box_center"][:, 0] - float(local_K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (batch["box_center"][:, 1] - float(local_K[1, 2])) / bs + pred_cam[:, 2]
    camera_t = torch.stack([tx, ty, tz], dim=-1).detach().cpu().numpy()
    joints = (
        prediction["pred_keypoints_3d"][0].detach().cpu().numpy()
        + camera_t[0][None, :]
    )
    return joints


def _epipolar_line_distance(weak_uv, strong_uv, F):
    x1 = np.append(strong_uv, 1.0)
    line = F.T @ x1
    denom = math.hypot(line[0], line[1])
    if denom < 1e-12:
        return 1e9
    return abs(float(np.append(weak_uv, 1.0) @ line)) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_obj, model_cfg, detector, dataset_cls, recursive_to = _load_model("wilor")
    model_obj = model_obj.to(device).eval()
    detector = detector.to(device)
    mano_model = MANOHandModel(str(MANO_DIR))
    manifest = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))["items"]
    valid_rows = {
        (r["clip"], int(r["frame"]), r["camera_side"]): r
        for r in csv.DictReader(open(VALID_CSV, encoding="utf-8"))
    }
    o2_rows = json.loads(
        (
            REPO_ROOT
            / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v2/stereo_recovery_v2_wilor.json"
        ).read_text(encoding="utf-8")
    )["rows"]
    o2_by = {(r["clip"], str(r["frame"])): r for r in o2_rows}
    stage_rows = []
    forced_rows = []

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
        artifacts = _load_model_artifacts(run_dir, "wilor")
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
                # Only process valid-both frames where O2 did not succeed.
                key = (clip, int(frame_number))
                left_valid = valid_rows.get((clip, int(frame_number), "left"), {}).get("gt_visible_valid") == "True"
                right_valid = valid_rows.get((clip, int(frame_number), "right"), {}).get("gt_visible_valid") == "True"
                if not (left_valid and right_valid):
                    continue
                o2r = o2_by.get((clip, str(frame_number)))
                o2_success = bool(
                    (left_p[t] and right_p[t])
                    or (o2r and o2r.get("recovered"))
                )
                if o2_success:
                    continue
                strong_view = None
                if left_p[t] and right_p[t]:
                    strong_view = "left" if left_scores[t] >= right_scores[t] else "right"
                elif left_p[t]:
                    strong_view = "left"
                elif right_p[t]:
                    strong_view = "right"
                if strong_view is None:
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
                weak_raw_cam = camera.from_json(
                    json.load(tar.extractfile(f"{int(frame_number):06d}.cameras.json"))[
                        "1201-1" if weak_view == "left" else "1201-2"
                    ]
                )
                weak_raw_img = imageio.imread(
                    tar.extractfile(
                        f"{int(frame_number):06d}.image_{'1201-1' if weak_view == 'left' else '1201-2'}.jpg"
                    )
                )
                weak_T = T_left[t] if weak_view == "left" else T_right[t]
                weak_K = K_left if weak_view == "left" else K_right
                # For each depth/protocol, detector diagnostic + forced regression.
                best_forced = None
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
                        proposals = _detector_proposals(detector, local_img, 0.05)
                        best_det_score = max([p["score"] for p in proposals], default=None)
                        best_det_class = None
                        if proposals:
                            best = max(proposals, key=lambda p: p["score"])
                            best_det_class = best["class"]
                        stage_rows.append(
                            {
                                "clip": clip,
                                "frame": int(frame_number),
                                "camera": weak_view,
                                "selected_side": side,
                                "crop_protocol": protocol_name,
                                "crop_bbox": [0, 0, protocol["width"], protocol["height"]],
                                "crop_size": f"{protocol['width']}x{protocol['height']}",
                                "detector_proposal_count": len(proposals),
                                "best_detector_score": best_det_score,
                                "predicted_handedness": best_det_class,
                                "handedness_score": None,
                                "regression_valid": None,
                                "required_joints_valid": None,
                                "final_observation_valid": False,
                                "failure_stage": "O2_miss",
                            }
                        )
                        forced_joints = _forced_regression(
                            model_obj,
                            model_cfg,
                            dataset_cls,
                            recursive_to,
                            device,
                            local_img,
                            side_index == 1,
                            local_K,
                        )
                        weak_cam, pred_uv = _local_to_weak_p1(
                            forced_joints,
                            local_cam.T_world_from_eye,
                            weak_T,
                            weak_K,
                        )
                        # Score forced candidate by wrist distance to weak GT? No, use epipolar from strong wrist.
                        F = None
                        score = 1e9
                        if F is None:
                            # Approximate epipolar line using known P1 relative geometry.
                            R_rel = np.linalg.inv(weak_T)[:3, :3] @ strong_T[:3, :3]
                            t_rel = np.linalg.inv(weak_T)[:3, 3] - np.linalg.inv(strong_T)[:3, 3]
                            tx = np.asarray([[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]])
                            F = np.linalg.inv(weak_K).T @ tx @ R_rel @ np.linalg.inv(strong_K)
                        score = _epipolar_line_distance(pred_uv[0], strong_uv[0], F)
                        if best_forced is None or score < best_forced["score"]:
                            best_forced = {
                                "score": score,
                                "joints_camera": weak_cam,
                                "pred_uv": pred_uv,
                                "depth": depth,
                                "protocol": protocol_name,
                            }
                if best_forced is None:
                    forced_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "strong_view": strong_view,
                            "weak_view": weak_view,
                            "recovered": False,
                            "score": None,
                            "depth": None,
                            "protocol": None,
                            "metrics": None,
                        }
                    )
                    continue
                weak_gt_uv = uv_left[t] if weak_view == "left" else uv_right[t]
                pred_uv = best_forced["pred_uv"]
                errors = np.linalg.norm(pred_uv - weak_gt_uv, axis=-1)
                forced_rows.append(
                    {
                        "clip": clip,
                        "frame": int(frame_number),
                        "strong_view": strong_view,
                        "weak_view": weak_view,
                        "recovered": True,
                        "score": best_forced["score"],
                        "depth": best_forced["depth"],
                        "protocol": best_forced["protocol"],
                        "joints_camera": best_forced["joints_camera"].tolist(),
                        "metrics": {
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
                        },
                    }
                )

    stage_path = OUTPUT_ROOT / "failure_stage_matrix.csv"
    stage_fields = [
        "clip", "frame", "camera", "selected_side", "crop_protocol", "crop_bbox",
        "crop_size", "detector_proposal_count", "best_detector_score",
        "predicted_handedness", "handedness_score", "regression_valid",
        "required_joints_valid", "final_observation_valid", "failure_stage",
    ]
    with stage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=stage_fields)
        writer.writeheader()
        writer.writerows(stage_rows)
    forced_path = OUTPUT_ROOT / "forced_roi_matrix.csv"
    forced_fields = [
        "clip", "frame", "strong_view", "weak_view", "recovered", "score",
        "depth", "protocol", "joints_camera", "metrics",
    ]
    with forced_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=forced_fields)
        writer.writeheader()
        writer.writerows(forced_rows)
    summary = {
        "model": "wilor",
        "stage_rows": len(stage_rows),
        "forced_rows": len(forced_rows),
        "forced_recovered": sum(1 for r in forced_rows if r["recovered"]),
    }
    (OUTPUT_ROOT / "forced_roi_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(stage_path)
    print(forced_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
