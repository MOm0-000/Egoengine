#!/usr/bin/env python3
"""O4: weak-view handedness override + stereo physical-hand association."""

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
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from scripts.evaluate_hot3d_fov_ablation import (
    ABLATION_RUNS,
    CLIPS_ROOT,
    MANO_DIR,
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


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v4"
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


def _proposals(detector, image, threshold=0.05):
    detections = detector(image, conf=threshold, verbose=False)[0]
    out = []
    for detection in detections:
        data = detection.boxes.data.detach().cpu().reshape(-1).numpy()
        out.append(
            {
                "box": data[:4].tolist(),
                "score": float(data[4]),
                "class": int(data[5]),
            }
        )
    return out


def _regress_boxes(
    model_obj,
    model_cfg,
    dataset_cls,
    recursive_to,
    device,
    image,
    boxes,
    rights,
    local_K,
):
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
    focal = float((local_K[0, 0] + local_K[1, 1]) * 0.5)
    bs = batch["box_size"] * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (batch["box_center"][:, 0] - float(local_K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (batch["box_center"][:, 1] - float(local_K[1, 2])) / bs + pred_cam[:, 2]
    camera_t = torch.stack([tx, ty, tz], dim=-1).detach().cpu().numpy()
    joints = (
        prediction["pred_keypoints_3d"].detach().cpu().numpy()
        + camera_t[:, None, :]
    )
    return joints


def _epipolar_distance(weak_uv, strong_uv, F):
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
    audit_rows = []
    identity_rows = []
    ranking_rows = []

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
                cams = json.load(tar.extractfile(f"{int(frame_number):06d}.cameras.json"))
                weak_raw_cam = camera.from_json(
                    cams["1201-1" if weak_view == "left" else "1201-2"]
                )
                weak_raw_img = imageio.imread(
                    tar.extractfile(
                        f"{int(frame_number):06d}.image_{'1201-1' if weak_view == 'left' else '1201-2'}.jpg"
                    )
                )
                weak_T = T_left[t] if weak_view == "left" else T_right[t]
                weak_K = K_left if weak_view == "left" else K_right
                weak_gt_uv = uv_left[t] if weak_view == "left" else uv_right[t]
                best = None
                for depth in DEPTH_STEPS:
                    strong_point = _unproject_point(strong_K, strong_uv[0], depth)
                    strong_point_h = np.append(strong_point, 1.0)
                    world_point = strong_T @ strong_point_h
                    for protocol_name, protocol in LOCAL_PROTOCOLS.items():
                        local_cam = _local_camera_for_point(weak_raw_cam, world_point, protocol)
                        det_local = float(np.linalg.det(local_cam.T_world_from_eye[:3, :3]))
                        audit_rows.append(
                            {
                                "clip": clip,
                                "frame": int(frame_number),
                                "view": weak_view,
                                "protocol": protocol_name,
                                "det_R": det_local,
                                "is_reflection": bool(det_local < 0),
                            }
                        )
                        local_img = _warp_raw_local(weak_raw_cam, weak_raw_img, local_cam)
                        local_K = np.asarray(
                            [
                                [protocol["f"], 0, protocol["width"] / 2],
                                [0, protocol["f"], protocol["height"] / 2],
                                [0, 0, 1],
                            ],
                            dtype=np.float64,
                        )
                        proposals = _proposals(detector, local_img, 0.05)
                        if not proposals:
                            continue
                        boxes = [p["box"] for p in proposals]
                        rights = [p["class"] for p in proposals]
                        joints_all = _regress_boxes(
                            model_obj,
                            model_cfg,
                            dataset_cls,
                            recursive_to,
                            device,
                            local_img,
                            boxes,
                            rights,
                            local_K,
                        )
                        R_rel = np.linalg.inv(weak_T)[:3, :3] @ strong_T[:3, :3]
                        t_rel = np.linalg.inv(weak_T)[:3, 3] - np.linalg.inv(strong_T)[:3, 3]
                        tx = np.asarray(
                            [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]]
                        )
                        F = np.linalg.inv(weak_K).T @ tx @ R_rel @ np.linalg.inv(strong_K)
                        for idx, proposal in enumerate(proposals):
                            weak_cam, pred_uv = _local_to_weak_p1(
                                joints_all[idx],
                                local_cam.T_world_from_eye,
                                weak_T,
                                weak_K,
                            )
                            epi = _epipolar_distance(pred_uv[0], strong_uv[0], F)
                            ranking_rows.append(
                                {
                                    "clip": clip,
                                    "frame": int(frame_number),
                                    "depth": depth,
                                    "protocol": protocol_name,
                                    "proposal_score": proposal["score"],
                                    "predicted_handedness": proposal["class"],
                                    "strong_identity": side_index,
                                    "epipolar_error_px": epi,
                                }
                            )
                            score = epi
                            if best is None or score < best["score"]:
                                best = {
                                    "score": score,
                                    "joints_camera": weak_cam,
                                    "pred_uv": pred_uv,
                                    "weak_handedness": proposal["class"],
                                    "proposal_score": proposal["score"],
                                    "epipolar_error_px": epi,
                                    "depth": depth,
                                    "protocol": protocol_name,
                                }
                if best is None:
                    identity_rows.append(
                        {
                            "clip": clip,
                            "frame": int(frame_number),
                            "strong_view": strong_view,
                            "weak_view": weak_view,
                            "strong_predicted_handedness": side_index,
                            "weak_predicted_handedness": None,
                            "final_track_identity": side_index,
                            "recovered": False,
                            "score": None,
                            "metrics": None,
                        }
                    )
                    continue
                errors = np.linalg.norm(best["pred_uv"] - weak_gt_uv, axis=-1)
                identity_rows.append(
                    {
                        "clip": clip,
                        "frame": int(frame_number),
                        "strong_view": strong_view,
                        "weak_view": weak_view,
                        "strong_predicted_handedness": side_index,
                        "weak_predicted_handedness": best["weak_handedness"],
                        "final_track_identity": side_index,
                        "recovered": True,
                        "score": best["score"],
                        "depth": best["depth"],
                        "protocol": best["protocol"],
                        "joints_camera": best["joints_camera"].tolist(),
                        "metrics": {
                            "median_2d_error_px": float(np.median(errors)),
                            "mean_2d_error_px": float(np.mean(errors)),
                            "pck5": float(np.mean(errors <= 5)),
                            "pck10": float(np.mean(errors <= 10)),
                            "pck20": float(np.mean(errors <= 20)),
                            "wrist_error_px": float(np.linalg.norm(best["pred_uv"][0] - weak_gt_uv[0])),
                            "fingertip_error_px": float(
                                np.mean(
                                    np.linalg.norm(
                                        best["pred_uv"][FINGERTIPS] - weak_gt_uv[FINGERTIPS],
                                        axis=-1,
                                    )
                                )
                            ),
                        },
                    }
                )

    # Write outputs.
    audit_path = OUTPUT_ROOT / "handedness_mismatch_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "rows": audit_rows,
                "reflection_count": sum(1 for r in audit_rows if r["is_reflection"]),
                "total": len(audit_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    identity_fields = [
        "clip", "frame", "strong_view", "weak_view", "strong_predicted_handedness",
        "weak_predicted_handedness", "final_track_identity", "recovered", "score",
        "depth", "protocol", "joints_camera", "metrics",
    ]
    with (OUTPUT_ROOT / "identity_override_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=identity_fields)
        writer.writeheader()
        writer.writerows(identity_rows)
    ranking_fields = [
        "clip", "frame", "depth", "protocol", "proposal_score",
        "predicted_handedness", "strong_identity", "epipolar_error_px",
    ]
    with (OUTPUT_ROOT / "candidate_ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranking_fields)
        writer.writeheader()
        writer.writerows(ranking_rows)
    summary = {
        "model": "wilor",
        "identity_rows": len(identity_rows),
        "recovered": sum(1 for r in identity_rows if r["recovered"]),
    }
    (OUTPUT_ROOT / "identity_override_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(audit_path)
    print(OUTPUT_ROOT / "identity_override_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
