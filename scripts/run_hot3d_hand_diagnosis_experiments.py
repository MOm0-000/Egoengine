#!/usr/bin/env python3
"""Run revised HOT3D hand-observation ablation experiments 0-3.

The script follows docs/Hand_2D_Observation_瓶颈定位_协议修订版.md.
GT is used only to freeze the evaluation domain and to construct diagnostic
oracles. It is never copied into a production inference artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))

from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis"
MANIFEST_PATH = OUTPUT_ROOT / "subset_manifest.json"
MANO_DIR = REPO_ROOT / "third_party/POEM-v2/assets/mano_v1_2/models"

LEFT_STREAM_KEY = "1201-1"
RIGHT_STREAM_KEY = "1201-2"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SCORE_THRESHOLD = 0.2
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
PCK_PIXEL_THRESHOLDS = (5.0, 10.0, 20.0)
PCK_3D_THRESHOLDS_MM = (10.0, 20.0, 30.0)

CAMERA_PROTOCOL = {
    "camera_protocol": "same_size_pinhole_from_source_fisheye",
    "note": "benchmark-only; not production rectification",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


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


def _procrustes_rotation(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_centered = pred - pred.mean(axis=0)
    gt_centered = gt - gt.mean(axis=0)
    covariance = pred_centered.T @ gt_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _value_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _frame_joint_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if mask is None:
        mask = np.ones(pred.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or not np.isfinite(pred[mask]).all() or not np.isfinite(gt[mask]).all():
        empty = {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
        }
        return {
            "count_frames": 0,
            "mpjpe_mm": empty,
            "root_aligned_mpjpe_mm": empty,
            "pa_mpjpe_mm": empty,
            "wrist_error_mm": empty,
            "fingertip_error_mm": empty,
            "bone_length_error_mm": empty,
            "pck": {f"{int(t)}mm": math.nan for t in PCK_3D_THRESHOLDS_MM},
        }

    p = pred[mask]
    g = gt[mask]
    errors_mm = np.linalg.norm(p - g, axis=-1) * 1000.0
    p_root = p - p[:, 0:1, :]
    g_root = g - g[:, 0:1, :]
    root_errors_mm = np.linalg.norm(p_root - g_root, axis=-1) * 1000.0
    p_center = p - p.mean(axis=1, keepdims=True)
    g_center = g - g.mean(axis=1, keepdims=True)
    pa_errors_mm = np.empty_like(root_errors_mm)
    for frame_index in range(len(p)):
        rotation = _procrustes_rotation(p_center[frame_index], g_center[frame_index])
        aligned = p_center[frame_index] @ rotation.T
        pa_errors_mm[frame_index] = (
            np.linalg.norm(aligned - g_center[frame_index], axis=-1) * 1000.0
        )
    wrist_error_mm = np.linalg.norm(p[:, 0] - g[:, 0], axis=-1) * 1000.0
    fingertip_error_mm = (
        np.linalg.norm(p[:, FINGERTIPS] - g[:, FINGERTIPS], axis=-1) * 1000.0
    )
    pred_bones = np.linalg.norm(p[:, FINGERTIPS] - p[:, 0:1, :], axis=-1)
    gt_bones = np.linalg.norm(g[:, FINGERTIPS] - g[:, 0:1, :], axis=-1)
    bone_error_mm = np.abs(pred_bones - gt_bones) * 1000.0
    pck = {
        f"{int(threshold)}mm": float(
            np.mean(root_errors_mm.ravel() <= threshold)
        )
        for threshold in PCK_3D_THRESHOLDS_MM
    }
    return {
        "count_frames": int(mask.sum()),
        "mpjpe_mm": _value_summary(errors_mm),
        "root_aligned_mpjpe_mm": _value_summary(root_errors_mm),
        "pa_mpjpe_mm": _value_summary(pa_errors_mm),
        "wrist_error_mm": _value_summary(wrist_error_mm),
        "fingertip_error_mm": _value_summary(fingertip_error_mm),
        "bone_length_error_mm": _value_summary(bone_error_mm),
        "pck": pck,
    }


def _pixel_metrics(
    pred_uv: np.ndarray,
    gt_uv: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    pred_uv = np.asarray(pred_uv, dtype=np.float64)
    gt_uv = np.asarray(gt_uv, dtype=np.float64)
    if mask is None:
        mask = np.ones(pred_uv.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or not np.isfinite(pred_uv[mask]).all() or not np.isfinite(gt_uv[mask]).all():
        empty = {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
        }
        return {
            "count_frames": 0,
            "keypoint_error_px": empty,
            "wrist_error_px": empty,
            "fingertip_error_px": empty,
            "required_keypoint_error_px": empty,
            "pck": {f"{int(t)}px": math.nan for t in PCK_PIXEL_THRESHOLDS},
        }
    p = pred_uv[mask]
    g = gt_uv[mask]
    errors_px = np.linalg.norm(p - g, axis=-1)
    required_errors_px = np.linalg.norm(p[:, REQUIRED] - g[:, REQUIRED], axis=-1)
    wrist_error_px = np.linalg.norm(p[:, 0] - g[:, 0], axis=-1)
    fingertip_error_px = np.linalg.norm(
        p[:, FINGERTIPS] - g[:, FINGERTIPS], axis=-1
    )
    pck = {
        f"{int(threshold)}px": float(np.mean(errors_px.ravel() <= threshold))
        for threshold in PCK_PIXEL_THRESHOLDS
    }
    return {
        "count_frames": int(mask.sum()),
        "keypoint_error_px": _value_summary(errors_px),
        "wrist_error_px": _value_summary(wrist_error_px),
        "fingertip_error_px": _value_summary(fingertip_error_px),
        "required_keypoint_error_px": _value_summary(required_errors_px),
        "pck": pck,
    }


def _relative_camera(
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left: np.ndarray,
    T_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T_left = np.asarray(T_left, dtype=np.float64)
    T_right = np.asarray(T_right, dtype=np.float64)
    R_wl = T_left[:3, :3]
    t_wl = T_left[:3, 3]
    R_wr = T_right[:3, :3]
    t_wr = T_right[:3, 3]
    R_rel = R_wr.T @ R_wl
    t_rel = R_wr.T @ (t_wl - t_wr)
    P_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K_right @ np.hstack([R_rel, t_rel[:, None]])
    return P_left, P_right, R_rel, t_rel


def _triangulate_pair(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left: np.ndarray,
    T_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P_left, P_right, _, _ = _relative_camera(K_left, K_right, T_left, T_right)
    count = len(left_uv)
    points = np.zeros((count, 3), dtype=np.float64)
    reprojection = np.full(count, np.inf, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    for index in range(count):
        ul = np.asarray(left_uv[index], dtype=np.float64)
        ur = np.asarray(right_uv[index], dtype=np.float64)
        if not (np.isfinite(ul).all() and np.isfinite(ur).all()):
            continue
        matrix = np.vstack(
            [
                ul[0] * P_left[2] - P_left[0],
                ul[1] * P_left[2] - P_left[1],
                ur[0] * P_right[2] - P_right[0],
                ur[1] * P_right[2] - P_right[1],
            ]
        )
        _, _, vt = np.linalg.svd(matrix)
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-12:
            continue
        point = homogeneous[:3] / homogeneous[3]
        if point[2] <= 0:
            continue
        points[index] = point
        left_proj = P_left @ np.append(point, 1.0)
        right_proj = P_right @ np.append(point, 1.0)
        if abs(left_proj[2]) < 1e-12 or abs(right_proj[2]) < 1e-12:
            continue
        left_2d = left_proj[:2] / left_proj[2]
        right_2d = right_proj[:2] / right_proj[2]
        reprojection[index] = max(
            float(np.linalg.norm(left_2d - ul)),
            float(np.linalg.norm(right_2d - ur)),
        )
        valid[index] = True
    return points, valid, reprojection


def _fundamental_from_cameras(
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left: np.ndarray,
    T_right: np.ndarray,
) -> np.ndarray:
    _, _, R_rel, t_rel = _relative_camera(K_left, K_right, T_left, T_right)
    tx = np.asarray(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(K_right).T @ tx @ R_rel @ np.linalg.inv(K_left)


def _sampson_error(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    F: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    if not valid.any():
        return {"median_px": math.nan, "p95_px": math.nan}
    left = np.asarray(left_uv[valid], dtype=np.float64)
    right = np.asarray(right_uv[valid], dtype=np.float64)
    left_h = np.hstack([left, np.ones((len(left), 1))])
    right_h = np.hstack([right, np.ones((len(right), 1))])
    f_left = left_h @ F.T
    f_right = right_h @ F
    numerator = np.sum(right_h * f_left, axis=-1) ** 2
    denominator = (
        f_left[:, 0] ** 2
        + f_left[:, 1] ** 2
        + f_right[:, 0] ** 2
        + f_right[:, 1] ** 2
    )
    distances = np.sqrt(np.abs(numerator) / np.maximum(denominator, 1e-12))
    return {
        "median_px": float(np.median(distances)),
        "p95_px": float(np.percentile(distances, 95)),
    }


def _load_model_artifacts(run_dir: Path, model: str) -> dict[str, dict[str, np.ndarray]]:
    if model == "wilor":
        paths = {
            "left": run_dir / "hands/wilor_raw.npz",
            "right": run_dir / "hands_right/wilor_raw.npz",
        }
    else:
        paths = {
            "left": run_dir / "hands_hamer/hamer_raw.npz",
            "right": run_dir / "hands_right_hamer/hamer_raw.npz",
        }
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for view, path in paths.items():
        if path.is_file():
            artifacts[view] = _load_npz(path)
    return artifacts


def _model_observation(
    artifact: dict[str, np.ndarray],
    side_index: int,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(artifact["valid"][:, side_index], dtype=bool)
    scores = np.asarray(artifact["score"][:, side_index], dtype=np.float64)
    joints_camera = (
        np.asarray(artifact["joints_camera_rootrel"][:, side_index], dtype=np.float64)
        + np.asarray(artifact["translation_camera"][:, side_index, None], dtype=np.float64)
    )
    uv = _project(K, joints_camera)
    finite_z = np.isfinite(joints_camera).all(axis=-1) & (joints_camera[..., 2] > 0)
    p = valid & (scores >= SCORE_THRESHOLD) & finite_z.all(axis=-1)
    required_available = p & finite_z[:, REQUIRED].all(axis=-1)
    return uv, joints_camera, p, required_available, scores


def _gt_domain(
    gt_world: np.ndarray,
    frame_numbers: np.ndarray,
    boxes_payload: dict[str, Any],
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left_all: np.ndarray,
    T_right_all: np.ndarray,
) -> dict[str, np.ndarray]:
    gt_left_camera = np.stack(
        [_transform_points(_invert(T_left_all[t]), gt_world[t]) for t in range(len(gt_world))],
        axis=0,
    )
    gt_right_camera = np.stack(
        [_transform_points(_invert(T_right_all[t]), gt_world[t]) for t in range(len(gt_world))],
        axis=0,
    )
    gt_uv_left = _project(K_left, gt_left_camera)
    gt_uv_right = _project(K_right, gt_right_camera)

    visible_left = np.zeros(len(gt_world), dtype=bool)
    visible_right = np.zeros(len(gt_world), dtype=bool)
    for offset, frame_number in enumerate(frame_numbers):
        boxes = boxes_payload["boxes_amodal"].get(str(int(frame_number)), {})
        box_left = boxes.get(LEFT_STREAM_KEY) is not None
        box_right = boxes.get(RIGHT_STREAM_KEY) is not None
        wrist_left_z = gt_left_camera[offset, 0, 2] > 0
        wrist_right_z = gt_right_camera[offset, 0, 2] > 0
        required_in_left = (
            (gt_uv_left[offset, REQUIRED, 0] >= 0)
            & (gt_uv_left[offset, REQUIRED, 0] < IMAGE_WIDTH)
            & (gt_uv_left[offset, REQUIRED, 1] >= 0)
            & (gt_uv_left[offset, REQUIRED, 1] < IMAGE_HEIGHT)
        ).any()
        required_in_right = (
            (gt_uv_right[offset, REQUIRED, 0] >= 0)
            & (gt_uv_right[offset, REQUIRED, 0] < IMAGE_WIDTH)
            & (gt_uv_right[offset, REQUIRED, 1] >= 0)
            & (gt_uv_right[offset, REQUIRED, 1] < IMAGE_HEIGHT)
        ).any()
        visible_left[offset] = bool(box_left and wrist_left_z and required_in_left)
        visible_right[offset] = bool(box_right and wrist_right_z and required_in_right)
    return {
        "left": visible_left,
        "right": visible_right,
        "both": visible_left & visible_right,
        "gt_uv_left": gt_uv_left,
        "gt_uv_right": gt_uv_right,
        "gt_left_camera": gt_left_camera,
        "gt_right_camera": gt_right_camera,
    }


def _build_gt_world(
    run_dir: Path,
    mano_model: MANOHandModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = _load_npz(run_dir / "hot3d_gt/hand_pose.npz")
    frame_numbers = np.asarray(gt["frame_numbers"], dtype=np.int64)
    count = len(frame_numbers)
    betas = (
        torch.from_numpy(np.asarray(gt["mano_beta"], dtype=np.float32))
        .float()
        .expand(count, -1)
    )
    theta = torch.from_numpy(np.asarray(gt["mano_theta"], dtype=np.float32)).float()
    wrist = torch.from_numpy(np.asarray(gt["wrist_xform"], dtype=np.float32)).float()
    hand_side = torch.full((count,), int(gt["hand_side"][0]), dtype=torch.long)
    with torch.no_grad():
        _, gt_landmarks = mano_model(
            betas,
            theta,
            wrist,
            is_right_hand=hand_side.bool(),
        )
    gt_world = gt_landmarks.detach().cpu().numpy().astype(np.float64)
    return frame_numbers, gt_world, gt


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / max(denominator, 1))


def _experiment0(
    entry: dict[str, Any],
    run_dir: Path,
    frame_numbers: np.ndarray,
    domain: dict[str, np.ndarray],
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left_all: np.ndarray,
    T_right_all: np.ndarray,
) -> dict[str, Any]:
    points = np.zeros((len(frame_numbers), 21, 3), dtype=np.float64)
    joint_valid = np.zeros((len(frame_numbers), 21), dtype=bool)
    required_valid = np.zeros(len(frame_numbers), dtype=bool)
    for t in range(len(frame_numbers)):
        tri_points, tri_valid, _ = _triangulate_pair(
            domain["gt_uv_left"][t],
            domain["gt_uv_right"][t],
            K_left,
            K_right,
            T_left_all[t],
            T_right_all[t],
        )
        points[t] = tri_points
        joint_valid[t] = tri_valid
        required_valid[t] = tri_valid[REQUIRED].all()
    both_domain = domain["both"]
    oracle_mask = both_domain & required_valid
    record = {
        "clip": entry["clip"],
        "selected_side": entry["selected_side"],
        "frames": len(frame_numbers),
        "frame_numbers": np.asarray(frame_numbers, dtype=np.int64).tolist(),
        "gt_visible_left": domain["left"].astype(bool).tolist(),
        "gt_visible_right": domain["right"].astype(bool).tolist(),
        "gt_visible_both": domain["both"].astype(bool).tolist(),
        "gt_visible_left_count": int(domain["left"].sum()),
        "gt_visible_right_count": int(domain["right"].sum()),
        "gt_visible_both_count": int(domain["both"].sum()),
        "gt_2d_oracle_both_domain": {
            "candidate_frames": int(both_domain.sum()),
            "required_valid_frames": int(oracle_mask.sum()),
            "metrics_3d": _frame_joint_metrics(
                points, domain["gt_left_camera"], oracle_mask
            ),
        },
    }
    return record


def _experiment1(
    entry: dict[str, Any],
    run_dir: Path,
    domain: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    side_index = 0 if entry["selected_side"] == "left" else 1
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    for model in ("wilor", "hamer"):
        artifacts = _load_model_artifacts(run_dir, model)
        if "left" not in artifacts or "right" not in artifacts:
            records.append(
                {
                    "clip": entry["clip"],
                    "selected_side": entry["selected_side"],
                    "model": model,
                    "status": "artifacts_missing",
                }
            )
            continue
        left = _model_observation(artifacts["left"], side_index, K_left)
        right = _model_observation(artifacts["right"], side_index, K_right)
        left_uv, _, left_p, left_req, left_scores = left
        right_uv, _, right_p, right_req, right_scores = right

        left_det = left_p & domain["left"]
        right_det = right_p & domain["right"]
        both_det = left_p & right_p & domain["both"]

        left_acc = left_req & domain["left"]
        right_acc = right_req & domain["right"]
        both_acc = left_req & right_req & domain["both"]
        left_only = left_acc & ~right_req
        right_only = right_acc & ~left_req

        record: dict[str, Any] = {
            "clip": entry["clip"],
            "selected_side": entry["selected_side"],
            "model": model,
            "status": "evaluated",
            "score_threshold": SCORE_THRESHOLD,
            "coverage": {
                "left": {
                    "gt_visible_frames": int(domain["left"].sum()),
                    "detected_frames": int(left_det.sum()),
                    "detection_rate": _safe_rate(int(left_det.sum()), int(domain["left"].sum())),
                    "required_available_frames": int(left_acc.sum()),
                    "required_landmark_rate": _safe_rate(
                        int(left_acc.sum()), int(domain["left"].sum())
                    ),
                    "median_confidence": (
                        float(np.median(left_scores[left_det]))
                        if left_det.any()
                        else math.nan
                    ),
                },
                "right": {
                    "gt_visible_frames": int(domain["right"].sum()),
                    "detected_frames": int(right_det.sum()),
                    "detection_rate": _safe_rate(int(right_det.sum()), int(domain["right"].sum())),
                    "required_available_frames": int(right_acc.sum()),
                    "required_landmark_rate": _safe_rate(
                        int(right_acc.sum()), int(domain["right"].sum())
                    ),
                    "median_confidence": (
                        float(np.median(right_scores[right_det]))
                        if right_det.any()
                        else math.nan
                    ),
                },
                "both": {
                    "gt_visible_frames": int(domain["both"].sum()),
                    "detected_frames": int(both_det.sum()),
                    "detection_rate": _safe_rate(int(both_det.sum()), int(domain["both"].sum())),
                    "required_available_frames": int(both_acc.sum()),
                    "required_landmark_rate": _safe_rate(
                        int(both_acc.sum()), int(domain["both"].sum())
                    ),
                },
            },
            "localization": {
                "left_only_left_view": _pixel_metrics(
                    left_uv, domain["gt_uv_left"], left_only
                ),
                "right_only_right_view": _pixel_metrics(
                    right_uv, domain["gt_uv_right"], right_only
                ),
                "both_left_view": _pixel_metrics(
                    left_uv, domain["gt_uv_left"], both_acc
                ),
                "both_right_view": _pixel_metrics(
                    right_uv, domain["gt_uv_right"], both_acc
                ),
            },
        }
        records.append(record)
    return records


def _experiment2(
    entry: dict[str, Any],
    run_dir: Path,
    domain: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    side_index = 0 if entry["selected_side"] == "left" else 1
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    T_left_all = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
    T_right_all = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
    for model in ("wilor", "hamer"):
        artifacts = _load_model_artifacts(run_dir, model)
        if "left" not in artifacts or "right" not in artifacts:
            records.append(
                {
                    "clip": entry["clip"],
                    "selected_side": entry["selected_side"],
                    "model": model,
                    "status": "artifacts_missing",
                }
            )
            continue
        left = _model_observation(artifacts["left"], side_index, K_left)
        right = _model_observation(artifacts["right"], side_index, K_right)
        left_uv, _, _, left_req, _ = left
        right_uv, _, _, right_req, _ = right
        domain_mask = left_req & right_req & domain["both"]
        if not domain_mask.any():
            records.append(
                {
                    "clip": entry["clip"],
                    "selected_side": entry["selected_side"],
                    "model": model,
                    "status": "no_both_view_frames",
                    "c0": None,
                    "c1": None,
                }
            )
            continue

        # The current artifacts contain exactly one detection per side slot.
        # Therefore C0 and C1 use the same GT-side identity; C1 is not
        # additionally ambiguous in this dataset.
        c0 = _correspondence_metrics(
            left_uv,
            right_uv,
            domain["gt_left_camera"],
            K_left,
            K_right,
            T_left_all,
            T_right_all,
            domain_mask,
            correspondence_label="current_joint_index_identity",
        )
        c1 = dict(c0)
        c1["correspondence_label"] = "gt_side_identity_no_2d_coordinate_change"
        records.append(
            {
                "clip": entry["clip"],
                "selected_side": entry["selected_side"],
                "model": model,
                "status": "evaluated",
                "c0": c0,
                "c1": c1,
                "c1_constructibility_note": (
                    "Single detection per side slot; GT correspondence selects "
                    "the same side index and does not modify predicted 2D coordinates."
                ),
            }
        )
    return records


def _correspondence_metrics(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    gt_left_camera: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left_all: np.ndarray,
    T_right_all: np.ndarray,
    frame_mask: np.ndarray,
    correspondence_label: str,
) -> dict[str, Any]:
    count_frames = len(frame_mask)
    points = np.zeros((count_frames, 21, 3), dtype=np.float64)
    joint_valid = np.zeros((count_frames, 21), dtype=bool)
    reprojection = np.full((count_frames, 21), np.inf, dtype=np.float64)
    left_reproj = np.full((count_frames, 21), np.inf, dtype=np.float64)
    right_reproj = np.full((count_frames, 21), np.inf, dtype=np.float64)
    valid_frames = np.zeros(count_frames, dtype=bool)
    sampson_values = []
    positive_depth_joint_count = 0
    total_joint_count = 0
    for t in range(count_frames):
        if not frame_mask[t]:
            continue
        tri_points, tri_valid, tri_reproj = _triangulate_pair(
            left_uv[t],
            right_uv[t],
            K_left,
            K_right,
            T_left_all[t],
            T_right_all[t],
        )
        points[t] = tri_points
        joint_valid[t] = tri_valid
        reprojection[t] = tri_reproj
        positive_depth_joint_count += int(tri_valid.sum())
        total_joint_count += 21
        valid_frames[t] = bool(tri_valid[REQUIRED].all())

        F = _fundamental_from_cameras(K_left, K_right, T_left_all[t], T_right_all[t])
        joint_ok = tri_valid
        if joint_ok.any():
            left_h = np.hstack([left_uv[t, joint_ok], np.ones((int(joint_ok.sum()), 1))])
            right_h = np.hstack([right_uv[t, joint_ok], np.ones((int(joint_ok.sum()), 1))])
            f_left = left_h @ F.T
            f_right = right_h @ F
            numerator = np.sum(right_h * f_left, axis=-1) ** 2
            denominator = (
                f_left[:, 0] ** 2
                + f_left[:, 1] ** 2
                + f_right[:, 0] ** 2
                + f_right[:, 1] ** 2
            )
            sampson_values.extend(
                np.sqrt(np.abs(numerator) / np.maximum(denominator, 1e-12)).tolist()
            )
        # Per-view reprojection.
        P_left, P_right, _, _ = _relative_camera(
            K_left, K_right, T_left_all[t], T_right_all[t]
        )
        for j in range(21):
            if not tri_valid[j]:
                continue
            point_h = np.append(tri_points[j], 1.0)
            left_proj = P_left @ point_h
            right_proj = P_right @ point_h
            if abs(left_proj[2]) < 1e-12 or abs(right_proj[2]) < 1e-12:
                continue
            left_2d = left_proj[:2] / left_proj[2]
            right_2d = right_proj[:2] / right_proj[2]
            left_reproj[t, j] = float(np.linalg.norm(left_2d - left_uv[t, j]))
            right_reproj[t, j] = float(np.linalg.norm(right_2d - right_uv[t, j]))

    reproj_summary = _value_summary(reprojection[frame_mask])
    left_reproj_summary = _value_summary(left_reproj[frame_mask])
    right_reproj_summary = _value_summary(right_reproj[frame_mask])
    sampson_array = np.asarray(sampson_values, dtype=np.float64)
    return {
        "correspondence_label": correspondence_label,
        "candidate_both_frames": int(frame_mask.sum()),
        "triangulated_required_valid_frames": int(valid_frames.sum()),
        "positive_depth_joint_ratio": _safe_rate(positive_depth_joint_count, total_joint_count),
        "sampson_epipolar_distance_px": {
            "median": (
                float(np.median(sampson_array))
                if sampson_array.size
                else math.nan
            ),
            "p95": (
                float(np.percentile(sampson_array, 95))
                if sampson_array.size
                else math.nan
            ),
        },
        "left_to_right_reprojection_error_px": left_reproj_summary,
        "right_to_left_reprojection_error_px": right_reproj_summary,
        "triangulation_reprojection_error_px": reproj_summary,
        "metrics_3d": _frame_joint_metrics(
            points, gt_left_camera, valid_frames
        ),
    }


def _experiment3(
    entry: dict[str, Any],
    run_dir: Path,
    domain: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    side_index = 0 if entry["selected_side"] == "left" else 1
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    for model in ("wilor", "hamer"):
        artifacts = _load_model_artifacts(run_dir, model)
        if "left" not in artifacts:
            records.append(
                {
                    "clip": entry["clip"],
                    "selected_side": entry["selected_side"],
                    "model": model,
                    "status": "artifacts_missing",
                }
            )
            continue
        _, pred_camera, _, left_req, _ = _model_observation(
            artifacts["left"], side_index, K_left
        )
        mask = left_req & domain["left"]
        record: dict[str, Any] = {
            "clip": entry["clip"],
            "selected_side": entry["selected_side"],
            "model": model,
            "status": "evaluated",
            "reference_view": "left",
            "candidate_frames": int(mask.sum()),
            "variants": {},
        }
        if not mask.any():
            records.append(record)
            continue
        gt_camera = domain["gt_left_camera"]
        variants = _depth_scale_variants(pred_camera, gt_camera, mask)
        gt_selected = gt_camera[mask]
        for name, (variant_points, extra) in variants.items():
            metrics = _frame_joint_metrics(variant_points, gt_selected)
            metrics.update(extra)
            record["variants"][name] = metrics
        records.append(record)
    return records


def _depth_scale_variants(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    masked_pred = pred[mask]
    masked_gt = gt[mask]

    d0 = masked_pred.copy()
    pred_wrist = d0[:, 0:1, :]
    gt_wrist = masked_gt[:, 0:1, :]
    d1 = d0 - pred_wrist + gt_wrist

    pred_root = d0 - pred_wrist
    pred_bones = np.linalg.norm(pred_root[:, FINGERTIPS], axis=-1)
    gt_bones = np.linalg.norm(masked_gt[:, FINGERTIPS] - gt_wrist[:, 0:1, :], axis=-1)
    bone_ratio = np.divide(
        gt_bones,
        pred_bones,
        out=np.full_like(pred_bones, np.nan, dtype=np.float64),
        where=(pred_bones > 1e-9),
    )
    finite = np.isfinite(bone_ratio)
    scale_factors = np.ones(len(masked_pred), dtype=np.float64)
    for frame_index in range(len(masked_pred)):
        if finite[frame_index].any():
            scale_factors[frame_index] = float(
                np.median(bone_ratio[frame_index, finite[frame_index]])
            )
    scale_factors = np.clip(scale_factors, 0.1, 10.0)
    d2 = pred_wrist + scale_factors[:, None, None] * pred_root
    d3 = d2 - d2[:, 0:1, :] + gt_wrist

    root_depth_error = np.abs(pred_wrist[:, 0, 2] - gt_wrist[:, 0, 2]) * 1000.0
    hand_scale_error = np.abs(pred_bones - gt_bones) * 1000.0
    return {
        "D0": (
            d0,
            {
                "root_depth_error_mm": _value_summary(root_depth_error),
                "hand_scale_error_mm": _value_summary(hand_scale_error),
                "applied_scale_factor": None,
            },
        ),
        "D1": (
            d1,
            {
                "root_depth_error_mm": {
                    "count": int(mask.sum()),
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                },
                "hand_scale_error_mm": _value_summary(hand_scale_error),
                "applied_scale_factor": None,
            },
        ),
        "D2": (
            d2,
            {
                "root_depth_error_mm": _value_summary(root_depth_error),
                "hand_scale_error_mm": _value_summary(
                    np.abs(
                        np.linalg.norm(d2[:, FINGERTIPS] - d2[:, 0:1, :], axis=-1)
                        - gt_bones
                    )
                    * 1000.0
                ),
                "applied_scale_factor": {
                    "mean": float(np.mean(scale_factors)),
                    "median": float(np.median(scale_factors)),
                    "p95": float(np.percentile(scale_factors, 95)),
                },
            },
        ),
        "D3": (
            d3,
            {
                "root_depth_error_mm": {
                    "count": int(mask.sum()),
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                },
                "hand_scale_error_mm": _value_summary(
                    np.abs(
                        np.linalg.norm(d3[:, FINGERTIPS] - d3[:, 0:1, :], axis=-1)
                        - gt_bones
                    )
                    * 1000.0
                ),
                "applied_scale_factor": {
                    "mean": float(np.mean(scale_factors)),
                    "median": float(np.median(scale_factors)),
                    "p95": float(np.percentile(scale_factors, 95)),
                },
            },
        ),
    }


def _write_config(name: str, payload: dict[str, Any]) -> Path:
    path = OUTPUT_ROOT / f"{name}_config.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_exp1_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip",
                "side",
                "model",
                "status",
                "left_detection_rate",
                "right_detection_rate",
                "both_detection_rate",
                "left_required_rate",
                "right_required_rate",
                "left_2d_median_px",
                "left_2d_p95_px",
                "left_2d_pck5",
                "left_2d_pck10",
                "left_2d_pck20",
                "right_2d_median_px",
                "right_2d_p95_px",
                "right_2d_pck5",
                "right_2d_pck10",
                "right_2d_pck20",
            ]
        )
        for row in rows:
            left = row.get("localization", {}).get(
                "left_only_left_view", {}
            ) if row.get("localization") else {}
            right = row.get("localization", {}).get(
                "right_only_right_view", {}
            ) if row.get("localization") else {}
            writer.writerow(
                [
                    row.get("clip"),
                    row.get("selected_side"),
                    row.get("model"),
                    row.get("status"),
                    row.get("coverage", {}).get("left", {}).get("detection_rate"),
                    row.get("coverage", {}).get("right", {}).get("detection_rate"),
                    row.get("coverage", {}).get("both", {}).get("detection_rate"),
                    row.get("coverage", {}).get("left", {}).get("required_landmark_rate"),
                    row.get("coverage", {}).get("right", {}).get("required_landmark_rate"),
                    left.get("keypoint_error_px", {}).get("median"),
                    left.get("keypoint_error_px", {}).get("p95"),
                    left.get("pck", {}).get("5px"),
                    left.get("pck", {}).get("10px"),
                    left.get("pck", {}).get("20px"),
                    right.get("keypoint_error_px", {}).get("median"),
                    right.get("keypoint_error_px", {}).get("p95"),
                    right.get("pck", {}).get("5px"),
                    right.get("pck", {}).get("10px"),
                    right.get("pck", {}).get("20px"),
                ]
            )


def _write_exp2_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip",
                "side",
                "model",
                "status",
                "candidate_both_frames",
                "required_valid_frames",
                "positive_depth_ratio",
                "sampson_median_px",
                "sampson_p95_px",
                "reproj_median_px",
                "reproj_p95_px",
                "c0_mpjpe_mm",
                "c0_root_aligned_mm",
                "c0_pa_mpjpe_mm",
                "c1_mpjpe_mm",
                "c1_root_aligned_mm",
                "c1_pa_mpjpe_mm",
            ]
        )
        for row in rows:
            c0 = row.get("c0") or {}
            c1 = row.get("c1") or {}
            writer.writerow(
                [
                    row.get("clip"),
                    row.get("selected_side"),
                    row.get("model"),
                    row.get("status"),
                    c0.get("candidate_both_frames"),
                    c0.get("triangulated_required_valid_frames"),
                    c0.get("positive_depth_joint_ratio"),
                    c0.get("sampson_epipolar_distance_px", {}).get("median"),
                    c0.get("sampson_epipolar_distance_px", {}).get("p95"),
                    c0.get("triangulation_reprojection_error_px", {}).get("median"),
                    c0.get("triangulation_reprojection_error_px", {}).get("p95"),
                    c0.get("metrics_3d", {}).get("mpjpe_mm", {}).get("mean"),
                    c0.get("metrics_3d", {}).get("root_aligned_mpjpe_mm", {}).get("mean"),
                    c0.get("metrics_3d", {}).get("pa_mpjpe_mm", {}).get("mean"),
                    c1.get("metrics_3d", {}).get("mpjpe_mm", {}).get("mean"),
                    c1.get("metrics_3d", {}).get("root_aligned_mpjpe_mm", {}).get("mean"),
                    c1.get("metrics_3d", {}).get("pa_mpjpe_mm", {}).get("mean"),
                ]
            )


def _write_exp3_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip",
                "side",
                "model",
                "status",
                "candidate_frames",
                "variant",
                "mpjpe_mm",
                "root_aligned_mm",
                "pa_mpjpe_mm",
                "wrist_error_mm",
                "fingertip_error_mm",
                "root_depth_error_mm",
                "hand_scale_error_mm",
            ]
        )
        for row in rows:
            for variant, metrics in (row.get("variants") or {}).items():
                writer.writerow(
                    [
                        row.get("clip"),
                        row.get("selected_side"),
                        row.get("model"),
                        row.get("status"),
                        row.get("candidate_frames"),
                        variant,
                        metrics.get("mpjpe_mm", {}).get("mean"),
                        metrics.get("root_aligned_mpjpe_mm", {}).get("mean"),
                        metrics.get("pa_mpjpe_mm", {}).get("mean"),
                        metrics.get("wrist_error_mm", {}).get("mean"),
                        metrics.get("fingertip_error_mm", {}).get("mean"),
                        metrics.get("root_depth_error_mm", {}).get("mean"),
                        metrics.get("hand_scale_error_mm", {}).get("mean"),
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mano-dir", type=Path, default=MANO_DIR)
    parser.add_argument(
        "--exp",
        choices=("all", "0", "1", "2", "3"),
        default="all",
        help="Run one experiment or all experiments.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mano_model = MANOHandModel(str(args.mano_dir))

    exp0_rows: list[dict[str, Any]] = []
    exp1_rows: list[dict[str, Any]] = []
    exp2_rows: list[dict[str, Any]] = []
    exp3_rows: list[dict[str, Any]] = []

    for entry in manifest["items"]:
        run_dir = Path(entry["run_dir"]).resolve()
        print(f"evaluating {run_dir.name}", flush=True)
        frame_numbers, gt_world, _ = _build_gt_world(run_dir, mano_model)
        boxes_payload = json.loads(
            (run_dir / "hot3d_gt/hand_boxes_visibility.json").read_text(encoding="utf-8")
        )
        K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
        K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
        T_left_all = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
        T_right_all = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
        domain = _gt_domain(
            gt_world,
            frame_numbers,
            boxes_payload,
            K_left,
            K_right,
            T_left_all,
            T_right_all,
        )

        if args.exp in ("all", "0"):
            exp0_rows.append(
                _experiment0(
                    entry,
                    run_dir,
                    frame_numbers,
                    domain,
                    K_left,
                    K_right,
                    T_left_all,
                    T_right_all,
                )
            )
        if args.exp in ("all", "1"):
            exp1_rows.extend(_experiment1(entry, run_dir, domain))
        if args.exp in ("all", "2"):
            exp2_rows.extend(_experiment2(entry, run_dir, domain))
        if args.exp in ("all", "3"):
            exp3_rows.extend(_experiment3(entry, run_dir, domain))

    config_base = {
        "schema_version": "1.0",
        "generated_by": Path(__file__).name,
        "manifest": str(MANIFEST_PATH),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "score_threshold": SCORE_THRESHOLD,
        "required_joints": REQUIRED.astype(int).tolist(),
        "fingertip_joints": FINGERTIPS.astype(int).tolist(),
        **CAMERA_PROTOCOL,
    }

    if args.exp in ("all", "0"):
        config = {
            **config_base,
            "experiment": "0",
            "domain_definition": (
                "GT-visible per view = box exists for stream_id AND wrist z > 0 "
                "AND at least one required landmark projects inside the image."
            ),
        }
        _write_config("exp0", config)
        _write_json(
            OUTPUT_ROOT / "eval_domain.json",
            {
                "schema_version": "1.0",
                "config": config,
                "items": exp0_rows,
            },
        )

    if args.exp in ("all", "1"):
        config = {
            **config_base,
            "experiment": "1",
            "detection_definition": (
                "model valid for selected side AND score >= 0.2 AND all 21 joints finite z > 0"
            ),
            "required_available_definition": (
                "detection AND all required joints [0,4,8,12,16,20] finite z > 0"
            ),
        }
        _write_config("exp1", config)
        _write_json(
            OUTPUT_ROOT / "exp1_2d_localization_summary.json",
            {
                "schema_version": "1.0",
                "config": config,
                "items": exp1_rows,
            },
        )
        _write_exp1_matrix(exp1_rows, OUTPUT_ROOT / "exp1_2d_localization_matrix.csv")

    if args.exp in ("all", "2"):
        config = {
            **config_base,
            "experiment": "2",
            "domain": (
                "GT-visible both AND left required available AND right required available"
            ),
            "c0": "Pred 2D + current joint-index correspondence",
            "c1": (
                "Pred 2D + GT side identity only; predicted 2D coordinates unchanged"
            ),
        }
        _write_config("exp2", config)
        _write_json(
            OUTPUT_ROOT / "exp2_correspondence_summary.json",
            {
                "schema_version": "1.0",
                "config": config,
                "items": exp2_rows,
            },
        )
        _write_exp2_matrix(exp2_rows, OUTPUT_ROOT / "exp2_correspondence_matrix.csv")

    if args.exp in ("all", "3"):
        config = {
            **config_base,
            "experiment": "3",
            "reference_view": "left",
            "oracle_variants": {
                "D0": "Pred raw",
                "D1": "D0 translated so pred wrist equals GT wrist",
                "D2": "D0 globally scaled to GT wrist-to-fingertip bone scale",
                "D3": "D2 then D1",
            },
        }
        _write_config("exp3", config)
        _write_json(
            OUTPUT_ROOT / "exp3_depth_scale_summary.json",
            {
                "schema_version": "1.0",
                "config": config,
                "items": exp3_rows,
            },
        )
        _write_exp3_matrix(exp3_rows, OUTPUT_ROOT / "exp3_depth_scale_matrix.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
