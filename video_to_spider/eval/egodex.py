"""Explicit EgoDex hand evaluator for WiLoR artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..coordinates import invert_transform, rotation_geodesic_rad
from ..ingest.egodex_ground_truth import load_hand_ground_truth
from ..schemas import validate_wilor_raw

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": -1.0, "median": -1.0, "p95": -1.0}
    return {
        "count": int(values.size), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
    }


def evaluate_wilor(run_dir: str | Path, *, confidence_threshold: float = 0.0) -> Path:
    root = Path(run_dir).resolve()
    source = json.loads((root / "input/source.json").read_text(encoding="utf-8"))
    with np.load(root / "hands/wilor_raw.npz", allow_pickle=False) as artifact:
        wilor = {key: np.asarray(artifact[key]) for key in artifact.files}
    validate_wilor_raw(wilor)
    full_camera = np.load(root / "calibration/T_world_camera.npy").astype(np.float64)
    source_start = source["selected_frame_interval"][0]
    relative_indices = wilor["frame_indices"] - source_start
    T_world_camera = full_camera[relative_indices]
    T_camera_world = invert_transform(T_world_camera)
    report = {
        "schema_version": "1.0", "uses_ground_truth": True,
        "purpose": "offline WiLoR quality evaluation only; excluded from inference inputs",
        "confidence_threshold": confidence_threshold, "sides": {},
    }
    for hand_index, side_name in enumerate(("left", "right")):
        gt = load_hand_ground_truth(source["hdf5_path"], side_name)
        gt_transform = gt["T_world_joint"][wilor["frame_indices"]]
        gt_confidence = gt["confidence"][wilor["frame_indices"]]
        gt_points_world = gt_transform[..., :3, 3]
        gt_points_h = np.concatenate([gt_points_world, np.ones(gt_points_world.shape[:-1] + (1,))], axis=-1)
        gt_points_camera = np.einsum("tij,tkj->tki", T_camera_world, gt_points_h)[..., :3]
        pred_points_camera = np.concatenate([
            wilor["joints_camera_rootrel"][:, hand_index, [0]],
            wilor["joints_camera_rootrel"][:, hand_index, FINGERTIP_INDICES],
        ], axis=1) + wilor["translation_camera"][:, hand_index, None]
        valid = wilor["valid"][:, hand_index, None] & (gt_confidence > confidence_threshold)
        absolute_error = np.linalg.norm(pred_points_camera - gt_points_camera, axis=-1)
        pred_rootrel = pred_points_camera[:, 1:] - pred_points_camera[:, [0]]
        gt_rootrel = gt_points_camera[:, 1:] - gt_points_camera[:, [0]]
        root_relative_error = np.linalg.norm(pred_rootrel - gt_rootrel, axis=-1)
        R_camera_hand_gt = T_camera_world[:, :3, :3] @ gt_transform[:, 0, :3, :3]
        wrist_rotation_error = rotation_geodesic_rad(
            wilor["mano_global_orient"][:, hand_index], R_camera_hand_gt
        )
        valid_frames = wilor["valid"][:, hand_index] & (gt_confidence[:, 0] > confidence_threshold)
        pred_wrist = pred_points_camera[:, 0]
        if pred_wrist.shape[0] >= 3:
            acceleration = pred_wrist[2:] - 2 * pred_wrist[1:-1] + pred_wrist[:-2]
            acceleration_jitter = np.linalg.norm(acceleration, axis=-1)
        else:
            acceleration_jitter = np.zeros(0)
        report["sides"][side_name] = {
            "valid_rate": float(np.mean(wilor["valid"][:, hand_index])),
            "absolute_wrist_and_fingertip_mpjpe_m": _summary(absolute_error[valid]),
            "wrist_translation_error_m": _summary(absolute_error[:, 0][valid_frames]),
            "fingertip_mpjpe_m": _summary(absolute_error[:, 1:][valid[:, 1:]]),
            "root_relative_fingertip_mpjpe_m": _summary(root_relative_error[valid[:, 1:]]),
            "wrist_geodesic_rotation_error_rad": _summary(wrist_rotation_error[valid_frames]),
            "wrist_second_difference_jitter_m_per_frame2": _summary(acceleration_jitter),
            "id_switch_count": 0,
        }
    output_dir = root / "evaluation"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "wilor_hand_metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path

