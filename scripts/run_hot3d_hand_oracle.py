#!/usr/bin/env python3
"""Run HOT3D hand 2D/3D oracle diagnostics on the fixed subset.

The script is benchmark-only. GT is used exclusively for evaluation and
oracle inputs; it is never copied into a production inference artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hand_tracking_toolkit.hand_models.mano_hand_model import (
    MANOHandModel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis"
MANIFEST_PATH = OUTPUT_ROOT / "subset_manifest.json"
MANO_DIR = REPO_ROOT / "third_party/POEM-v2/assets/mano_v1_2/models"
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
PCK_THRESHOLDS_MM = (10.0, 20.0, 30.0)
LEFT_STREAM_KEY = "1201-1"
RIGHT_STREAM_KEY = "1201-2"


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


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))
    return float(math.degrees(math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


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


def _metrics_for_frame(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if mask is None:
        mask = np.ones(pred.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or not np.isfinite(pred[mask]).all() or not np.isfinite(gt[mask]).all():
        return {
            "count": 0,
            "mpjpe_mm": math.nan,
            "root_aligned_mpjpe_mm": math.nan,
            "pa_mpjpe_mm": math.nan,
            "wrist_error_mm": math.nan,
            "fingertip_error_mm": math.nan,
            "bone_length_error_mm": math.nan,
            "pck": {f"{int(t)}mm": math.nan for t in PCK_THRESHOLDS_MM},
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
        pa_errors_mm[frame_index] = np.linalg.norm(aligned - g_center[frame_index], axis=-1) * 1000.0
    wrist_error_mm = np.linalg.norm(p[:, 0] - g[:, 0], axis=-1) * 1000.0
    fingertip_error_mm = np.linalg.norm(p[:, FINGERTIPS] - g[:, FINGERTIPS], axis=-1) * 1000.0
    pred_bones = np.linalg.norm(p[:, FINGERTIPS] - p[:, 0:1, :], axis=-1)
    gt_bones = np.linalg.norm(g[:, FINGERTIPS] - g[:, 0:1, :], axis=-1)
    bone_error_mm = np.abs(pred_bones - gt_bones) * 1000.0
    pck = {
        f"{int(threshold)}mm": float(np.mean(root_errors_mm.ravel() <= threshold))
        for threshold in PCK_THRESHOLDS_MM
    }
    return {
        "count": int(mask.sum()),
        "mpjpe_mm": float(np.mean(errors_mm)),
        "root_aligned_mpjpe_mm": float(np.mean(root_errors_mm)),
        "pa_mpjpe_mm": float(np.mean(pa_errors_mm)),
        "wrist_error_mm": float(np.mean(wrist_error_mm)),
        "fingertip_error_mm": float(np.mean(fingertip_error_mm)),
        "bone_length_error_mm": float(np.mean(bone_error_mm)),
        "pck": pck,
    }


def _relative_camera(K_left: np.ndarray, K_right: np.ndarray, T_left: np.ndarray, T_right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return P_left, P_right, R_rel


def _triangulate_pair(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_left: np.ndarray,
    T_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P_left, P_right, _ = _relative_camera(K_left, K_right, T_left, T_right)
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
        left_proj = _project(K_left, point)
        right_proj = _project(K_right, point - np.asarray([0.0, 0.0, 0.0]))
        # Reprojection into right uses the relative pose path.
        P_right_proj = P_right @ np.append(point, 1.0)
        if abs(P_right_proj[2]) < 1e-12:
            continue
        right_proj = P_right_proj[:2] / P_right_proj[2]
        reprojection[index] = max(
            float(np.linalg.norm(left_proj - ul)),
            float(np.linalg.norm(right_proj - ur)),
        )
        valid[index] = True
    return points, valid, reprojection


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


def _predicted_observations(
    artifact: dict[str, np.ndarray],
    selected_side: int,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(artifact["valid"][:, selected_side], dtype=bool)
    joints_camera = (
        np.asarray(artifact["joints_camera_rootrel"][:, selected_side], dtype=np.float64)
        + np.asarray(artifact["translation_camera"][:, selected_side, None], dtype=np.float64)
    )
    uv = _project(K, joints_camera)
    scores = np.asarray(artifact["score"][:, selected_side], dtype=np.float32)
    finite = np.isfinite(joints_camera).all(axis=-1) & (joints_camera[..., 2] > 0)
    frame_valid = valid & finite.all(axis=-1)
    return uv, joints_camera, frame_valid, scores


def _fundamental_from_cameras(K_left: np.ndarray, K_right: np.ndarray, T_left: np.ndarray, T_right: np.ndarray) -> np.ndarray:
    _, _, R_rel = _relative_camera(K_left, K_right, T_left, T_right)
    t_rel = np.linalg.inv(T_right)[:3, :3].T @ (np.linalg.inv(T_left)[:3, 3] - np.linalg.inv(T_right)[:3, 3])
    # Recover the right-in-left translation from _relative_camera is not exposed;
    # recompute directly from the relative camera poses.
    R_wl = T_left[:3, :3]
    t_wl = T_left[:3, 3]
    R_wr = T_right[:3, :3]
    t_wr = T_right[:3, 3]
    t_rel = R_wr.T @ (t_wl - t_wr)
    tx = np.asarray(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(K_right).T @ tx @ R_rel @ np.linalg.inv(K_left)


def _epipolar_error(left_uv: np.ndarray, right_uv: np.ndarray, F: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    if not valid.any():
        return {"median_px": math.nan, "p95_px": math.nan}
    left = np.asarray(left_uv[valid], dtype=np.float64)
    right = np.asarray(right_uv[valid], dtype=np.float64)
    left_h = np.hstack([left, np.ones((len(left), 1))])
    right_h = np.hstack([right, np.ones((len(right), 1))])
    lines_right = (left_h @ F.T)
    denominator = np.linalg.norm(lines_right[:, :2], axis=-1)
    distances = np.abs(np.sum(right_h * lines_right, axis=-1)) / np.maximum(denominator, 1e-12)
    return {
        "median_px": float(np.median(distances)),
        "p95_px": float(np.percentile(distances, 95)),
    }


def evaluate_run(
    run_dir: Path,
    manifest_entry: dict[str, Any],
    mano_model: MANOHandModel,
) -> dict[str, Any]:
    side_index = 0 if manifest_entry["selected_side"] == "left" else 1
    gt = _load_npz(run_dir / "hot3d_gt/hand_pose.npz")
    frame_numbers = np.asarray(gt["frame_numbers"], dtype=np.int64)
    T_count = len(frame_numbers)
    betas = torch.from_numpy(np.asarray(gt["mano_beta"], dtype=np.float32)).float().expand(T_count, -1)
    theta = torch.from_numpy(np.asarray(gt["mano_theta"], dtype=np.float32)).float()
    wrist = torch.from_numpy(np.asarray(gt["wrist_xform"], dtype=np.float32)).float()
    hand_side = torch.full((T_count,), int(gt["hand_side"][0]), dtype=torch.long)
    with torch.no_grad():
        _, gt_landmarks = mano_model(
            betas,
            theta,
            wrist,
            is_right_hand=hand_side.bool(),
        )
    gt_world = gt_landmarks.detach().cpu().numpy().astype(np.float64)

    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    T_left_all = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
    T_right_all = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)

    gt_left_camera = np.stack(
        [
            _transform_points(_invert(T_left_all[t]), gt_world[t])
            for t in range(T_count)
        ],
        axis=0,
    )
    gt_right_camera = np.stack(
        [
            _transform_points(_invert(T_right_all[t]), gt_world[t])
            for t in range(T_count)
        ],
        axis=0,
    )
    gt_uv_left = _project(K_left, gt_left_camera)
    gt_uv_right = _project(K_right, gt_right_camera)
    boxes_payload = json.loads(
        (run_dir / "hot3d_gt/hand_boxes_visibility.json").read_text(encoding="utf-8")
    )
    gt_visible_left = np.zeros(T_count, dtype=bool)
    gt_visible_right = np.zeros(T_count, dtype=bool)
    for offset, frame_number in enumerate(frame_numbers):
        boxes = boxes_payload["boxes_amodal"].get(str(int(frame_number)), {})
        gt_visible_left[offset] = LEFT_STREAM_KEY in boxes
        gt_visible_right[offset] = RIGHT_STREAM_KEY in boxes

    oracle_2d_records = []
    oracle_2d_valid_frames = 0
    for t in range(T_count):
        points, joint_valid, reprojection = _triangulate_pair(
            gt_uv_left[t],
            gt_uv_right[t],
            K_left,
            K_right,
            T_left_all[t],
            T_right_all[t],
        )
        required_valid = joint_valid[REQUIRED].all()
        oracle_2d_records.append(
            {
                "frame_index": int(frame_numbers[t]),
                "joint_valid": joint_valid.astype(bool).tolist(),
                "required_valid": bool(required_valid),
                "reprojection_error_px": float(np.median(reprojection[joint_valid])) if joint_valid.any() else math.nan,
            }
        )
        if required_valid:
            oracle_2d_valid_frames += 1
    gt_frames_for_metric = np.asarray([item["required_valid"] for item in oracle_2d_records], dtype=bool)
    h_oracle_2d = {
        "valid_frames": int((gt_frames_for_metric & gt_visible_left & gt_visible_right).sum()),
        "total_frames": T_count,
        "metrics": _metrics_for_frame(
            np.stack(
                [
                    _triangulate_pair(
                        gt_uv_left[t],
                        gt_uv_right[t],
                        K_left,
                        K_right,
                        T_left_all[t],
                        T_right_all[t],
                    )[0]
                    for t in range(T_count)
                ],
                axis=0,
            ),
            gt_left_camera,
            gt_frames_for_metric & gt_visible_left & gt_visible_right,
        ),
    }

    results: dict[str, Any] = {
        "run_dir": str(run_dir),
        "clip": manifest_entry["clip"],
        "selected_side": manifest_entry["selected_side"],
        "selected_object_id": manifest_entry["selected_object_id"],
        "frame_count": T_count,
        "gt": {
            "mano_beta": np.asarray(gt["mano_beta"]).tolist(),
            "valid_frames": int(np.asarray(gt["valid"]).sum()),
        },
        "h_oracle_2d": h_oracle_2d,
        "models": {},
    }

    for model in ("wilor", "hamer"):
        artifacts = _load_model_artifacts(run_dir, model)
        model_result: dict[str, Any] = {
            "status": "artifacts_missing" if not artifacts else "evaluated",
            "views": {},
        }
        if artifacts:
            left_uv, left_camera, left_valid, left_scores = _predicted_observations(
                artifacts.get("left", {}), side_index, K_left
            ) if "left" in artifacts else (None, None, None, None)
            right_uv, right_camera, right_valid, right_scores = _predicted_observations(
                artifacts.get("right", {}), side_index, K_right
            ) if "right" in artifacts else (None, None, None, None)
            model_result["views"] = {
                "left": {
                    "gt_visible_frames": int(gt_visible_left.sum()),
                    "detected_frames": int((left_valid & gt_visible_left).sum()) if left_valid is not None else 0,
                    "detection_rate": float((left_valid & gt_visible_left).sum() / max(int(gt_visible_left.sum()), 1)) if left_valid is not None else math.nan,
                    "median_confidence": float(np.median(left_scores[left_valid & gt_visible_left])) if left_valid is not None and (left_valid & gt_visible_left).any() else math.nan,
                },
                "right": {
                    "gt_visible_frames": int(gt_visible_right.sum()),
                    "detected_frames": int((right_valid & gt_visible_right).sum()) if right_valid is not None else 0,
                    "detection_rate": float((right_valid & gt_visible_right).sum() / max(int(gt_visible_right.sum()), 1)) if right_valid is not None else math.nan,
                    "median_confidence": float(np.median(right_scores[right_valid & gt_visible_right])) if right_valid is not None and (right_valid & gt_visible_right).any() else math.nan,
                },
            }
            if left_valid is not None and right_valid is not None:
                gt_visible_both = gt_visible_left & gt_visible_right
                both = left_valid & right_valid & gt_visible_both
                model_result["views"]["both_detected_frames"] = int(both.sum())
                model_result["views"]["gt_visible_both_frames"] = int(gt_visible_both.sum())
                model_result["views"]["both_detection_rate"] = float(
                    both.sum() / max(int(gt_visible_both.sum()), 1)
                )
                if both.any():
                    F = _fundamental_from_cameras(K_left, K_right, T_left_all[0], T_right_all[0])
                    both_joint_count = int(both.sum()) * 21
                    model_result["views"]["epipolar_error"] = _epipolar_error(
                        left_uv[both].reshape(-1, 2),
                        right_uv[both].reshape(-1, 2),
                        F,
                        np.ones(both_joint_count, dtype=bool),
                    )
            # Monocular model-to-GT 3D metric in each view where the selected
            # hand is actually visible, then choose the view with more frames.
            view_metrics: list[tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]] = []
            if "left" in artifacts and left_camera is not None:
                view_metrics.append(
                    (
                        "left",
                        left_camera,
                        gt_left_camera,
                        artifacts["left"],
                        T_left_all,
                    )
                )
            if "right" in artifacts and right_camera is not None:
                view_metrics.append(
                    (
                        "right",
                        right_camera,
                        gt_right_camera,
                        artifacts["right"],
                        T_right_all,
                    )
                )
            best_metrics: dict[str, Any] | None = None
            best_count = -1
            for view_name, pred_camera, gt_camera, pred_artifact, T_view_all in view_metrics:
                view_valid = (
                    (left_valid if view_name == "left" else right_valid)
                    & (gt_visible_left if view_name == "left" else gt_visible_right)
                )
                metrics = _metrics_for_frame(pred_camera, gt_camera, view_valid)
                model_result[f"{view_name}_view_3d_metrics"] = metrics
                if metrics.get("count", 0) > best_count:
                    best_count = int(metrics.get("count", 0))
                    best_metrics = metrics
                orientation_errors = []
                for t in range(T_count):
                    if not view_valid[t]:
                        continue
                    pred_rot = np.asarray(
                        pred_artifact["mano_global_orient"][t, side_index],
                        dtype=np.float64,
                    )
                    gt_wrist_rot_world = _axis_angle_to_rotation(
                        np.asarray(gt["wrist_xform"][t, :3], dtype=np.float64)
                    )
                    gt_wrist_rot_camera = T_view_all[t][:3, :3].T @ gt_wrist_rot_world
                    orientation_errors.append(
                        _rotation_angle_deg(pred_rot @ gt_wrist_rot_camera.T)
                    )
                model_result[f"{view_name}_wrist_orientation_error_deg"] = {
                    "median": float(np.median(orientation_errors)) if orientation_errors else math.nan,
                    "p95": float(np.percentile(orientation_errors, 95)) if orientation_errors else math.nan,
                }
            model_result["best_view_3d_metrics"] = best_metrics
            # H-Oracle-3D: triangulate predicted 2D in both views.
            if left_valid is not None and right_valid is not None:
                h3_points = np.zeros((T_count, 21, 3), dtype=np.float64)
                h3_valid_frames = np.zeros(T_count, dtype=bool)
                gt_visible_both = gt_visible_left & gt_visible_right
                for t in range(T_count):
                    if not (left_valid[t] and right_valid[t] and gt_visible_both[t]):
                        continue
                    points, joint_valid, _ = _triangulate_pair(
                        left_uv[t],
                        right_uv[t],
                        K_left,
                        K_right,
                        T_left_all[t],
                        T_right_all[t],
                    )
                    h3_points[t] = points
                    h3_valid_frames[t] = joint_valid[REQUIRED].all()
                if h3_valid_frames.any():
                    model_result["h_oracle_3d"] = {
                        "valid_frames": int(h3_valid_frames.sum()),
                        "total_frames": T_count,
                        "metrics": _metrics_for_frame(
                            h3_points,
                            gt_left_camera,
                            h3_valid_frames,
                        ),
                    }
                else:
                    model_result["h_oracle_3d"] = {
                        "valid_frames": 0,
                        "total_frames": T_count,
                        "metrics": None,
                    }
            # Per-view 2D error against GT.
            if left_uv is not None:
                model_result["views"]["left"]["gt_reprojection_error_px"] = _reprojection_summary(
                    left_uv, gt_uv_left, left_valid & gt_visible_left
                )
            if right_uv is not None:
                model_result["views"]["right"]["gt_reprojection_error_px"] = _reprojection_summary(
                    right_uv, gt_uv_right, right_valid & gt_visible_right
                )
        results["models"][model] = model_result
    return results


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / angle
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _reprojection_summary(pred_uv: np.ndarray, gt_uv: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = np.asarray(valid, dtype=bool)
    if not mask.any():
        return {"median_px": math.nan, "p95_px": math.nan}
    error = np.linalg.norm(np.asarray(pred_uv[mask]) - np.asarray(gt_uv[mask]), axis=-1)
    return {
        "median_px": float(np.median(error)),
        "p95_px": float(np.percentile(error, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mano-dir", type=Path, default=MANO_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "hand_oracle_summary.json")
    parser.add_argument("--matrix", type=Path, default=OUTPUT_ROOT / "hand_oracle_matrix.csv")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mano_model = MANOHandModel(str(args.mano_dir))
    results: list[dict[str, Any]] = []
    for entry in manifest["items"]:
        run_dir = Path(entry["run_dir"]).resolve()
        print(f"evaluating {run_dir.name}")
        results.append(evaluate_run(run_dir, entry, mano_model))
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import csv

    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip",
                "side",
                "model",
                "left_detection_rate",
                "right_detection_rate",
                "both_detection_rate",
                "left_gt_reproj_median_px",
                "right_gt_reproj_median_px",
                "model_3d_mpjpe_mm",
                "model_3d_root_aligned_mm",
                "model_3d_pa_mpjpe_mm",
                "h_oracle_3d_mpjpe_mm",
                "h_oracle_3d_root_aligned_mm",
                "h_oracle_3d_pa_mpjpe_mm",
                "h_oracle_2d_mpjpe_mm",
                "h_oracle_2d_root_aligned_mm",
            ]
        )
        for row in results:
            for model in ("wilor", "hamer"):
                model_row = row["models"].get(model, {})
                views = model_row.get("views", {})
                left = views.get("left", {})
                right = views.get("right", {})
                left_3d = model_row.get("best_view_3d_metrics", {}) or {}
                h3 = model_row.get("h_oracle_3d", {}).get("metrics", {}) or {}
                writer.writerow(
                    [
                        row["clip"],
                        row["selected_side"],
                        model,
                        left.get("detection_rate"),
                        right.get("detection_rate"),
                        views.get("both_detection_rate"),
                        left.get("gt_reprojection_error_px", {}).get("median_px"),
                        right.get("gt_reprojection_error_px", {}).get("median_px"),
                        left_3d.get("mpjpe_mm"),
                        left_3d.get("root_aligned_mpjpe_mm"),
                        left_3d.get("pa_mpjpe_mm"),
                        h3.get("mpjpe_mm"),
                        h3.get("root_aligned_mpjpe_mm"),
                        h3.get("pa_mpjpe_mm"),
                        row["h_oracle_2d"]["metrics"].get("mpjpe_mm"),
                        row["h_oracle_2d"]["metrics"].get("root_aligned_mpjpe_mm"),
                    ]
                )
    print(f"summary -> {args.output}")
    print(f"matrix -> {args.matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
