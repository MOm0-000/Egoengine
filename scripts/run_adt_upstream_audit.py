#!/usr/bin/env python3
"""Aggregate ADT upstream audit metrics without rerunning upstream models.

This is a P2/P4/P7-style diagnostic driver.  It consumes artifacts already
produced by ``run_adt_depth_benchmark.py`` and optional downstream stages:

- stereo-depth metrics and stereo-depth gate;
- SAM3 object-mask mIoU against ADT GT segmentation;
- stereo-hand triangulation gate;
- FoundationPose object-centroid translation error against ADT GT;
- SAM3D mesh scale/shape diagnostics against ADT object models.

It does not fit any scale, alter any gate, or use GT to initialize models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))


ADT_ROOT = Path("/data_all/zzx/egoengine/adt_data")
OBJECT_LIBRARY = Path("/data_all/zzx/egoengine/adt_object_library")
SUMMARY_PATH = _REPO_ROOT / "runs" / "adt_depth_benchmark_summary.json"


def _quat_xyzw_matrix(x, y, z, w) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_matrix(w, x, y, z) -> np.ndarray:
    return _quat_xyzw_matrix(x, y, z, w)


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def _as_trimesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value.copy()
    geometries = [geometry for geometry in value.geometry.values() if len(geometry.faces)]
    return trimesh.util.concatenate(geometries)


def _sample_points(mesh: trimesh.Trimesh, count: int = 5000) -> np.ndarray:
    points, _face_indices = trimesh.sample.sample_surface(mesh, min(count, max(2, len(mesh.faces))))
    return np.asarray(points, dtype=np.float64)


def _percentile(values: np.ndarray, q: float) -> float:
    if not len(values):
        return math.nan
    return float(np.percentile(values, q))


def _find_gt_mesh(prototype: str) -> Path | None:
    normalized = prototype.replace("_", "-").replace(" ", "-").lower()
    exact = OBJECT_LIBRARY / prototype / "3d-asset.glb"
    if exact.is_file():
        return exact
    for directory in sorted(OBJECT_LIBRARY.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.lower() == normalized:
            mesh = directory / "3d-asset.glb"
            if mesh.is_file():
                return mesh
        if normalized in directory.name.lower():
            mesh = directory / "3d-asset.glb"
            if mesh.is_file():
                return mesh
    return None


def _read_gt_trajectories(seq_dir: Path, object_uid: str):
    camera_rows = list(csv.DictReader((seq_dir / "aria_trajectory.csv").open(encoding="utf-8")))
    camera_times = np.asarray([float(row["tracking_timestamp_us"]) * 1e-6 for row in camera_rows], dtype=np.float64)
    object_rows = [
        row
        for row in csv.DictReader((seq_dir / "scene_objects.csv").open(encoding="utf-8"))
        if row["object_uid"] == object_uid
    ]
    object_times = np.asarray([float(row["timestamp[ns]"]) * 1e-9 for row in object_rows], dtype=np.float64)
    if not len(object_rows) or not len(camera_rows):
        raise RuntimeError("missing ADT GT camera/object trajectory")

    def camera_at(timestamp: float) -> np.ndarray:
        index = int(np.argmin(np.abs(camera_times - timestamp)))
        row = camera_rows[index]
        world_device = _pose(
            _quat_xyzw_matrix(
                float(row["qx_world_device"]),
                float(row["qy_world_device"]),
                float(row["qz_world_device"]),
                float(row["qw_world_device"]),
            ),
            [
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ],
        )
        return world_device

    def object_at(timestamp: float) -> np.ndarray:
        index = int(np.argmin(np.abs(object_times - timestamp)))
        row = object_rows[index]
        return _pose(
            _quat_wxyz_matrix(
                float(row["q_wo_w"]),
                float(row["q_wo_x"]),
                float(row["q_wo_y"]),
                float(row["q_wo_z"]),
            ),
            [
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ],
        )

    return camera_at, object_at


def _gt_camera_in_rect(reference: Path, timestamp: float, camera_at) -> np.ndarray:
    prepared_meta = json.loads((reference / "adt_stereo_prepare.json").read_text(encoding="utf-8"))
    rectified_rotation = np.asarray(prepared_meta["rectification"]["device_to_rectified_rotation"], dtype=np.float64)
    device_to_rect = np.eye(4, dtype=np.float64)
    device_to_rect[:3, :3] = rectified_rotation
    return camera_at(timestamp) @ device_to_rect


def _segmentation_iou(pred_path: Path, gt_path: Path) -> dict[str, Any]:
    import zarr

    pred = np.load(pred_path, allow_pickle=False)
    pred_masks = np.asarray(pred["masks"]).astype(bool)
    pred_valid = np.asarray(pred["valid"]).astype(bool)
    pred_frames = np.asarray(pred["frame_indices"], dtype=np.int64)
    gt = zarr.open_group(str(gt_path), mode="r")
    gt_masks = np.asarray(gt["object_mask"]).astype(bool)
    gt_frames = np.asarray(gt["frame_indices"], dtype=np.int64)
    ious: list[float] = []
    valid_flags: list[bool] = []
    for pred_index, frame_index in enumerate(pred_frames):
        matches = np.flatnonzero(gt_frames == frame_index)
        if not matches.size:
            continue
        gt_index = int(matches[0])
        pred_mask = pred_masks[pred_index]
        gt_mask = gt_masks[gt_index]
        intersection = int(np.logical_and(pred_mask, gt_mask).sum())
        union = int(np.logical_or(pred_mask, gt_mask).sum())
        ious.append(float(intersection / union) if union else 0.0)
        valid_flags.append(bool(pred_valid[pred_index] and bool(gt_mask.any())))
    values = np.asarray(ious, dtype=np.float64)
    valid = np.asarray(valid_flags, dtype=bool)
    return {
        "frame_count": int(len(values)),
        "valid_rate": float(valid.mean()) if valid.size else 0.0,
        "mean_iou": float(values.mean()) if values.size else 0.0,
        "median_iou": _percentile(values, 50),
        "p05_iou": _percentile(values, 5),
        "p95_iou": _percentile(values, 95),
    }


def _mesh_diagnostics(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh) -> dict[str, Any]:
    sample_count = max(3, min(3000, len(pred_mesh.faces), len(gt_mesh.faces)))
    pred_points = _sample_points(pred_mesh, sample_count)
    gt_points = _sample_points(gt_mesh, sample_count)
    pred_center = pred_points.mean(axis=0)
    gt_center = gt_points.mean(axis=0)
    centered_pred = pred_points - pred_center
    centered_gt = gt_points - gt_center
    sim3_scale = float(np.sqrt((centered_gt**2).sum() / max((centered_pred**2).sum(), 1e-12)))
    scaled_pred = centered_pred * sim3_scale
    rotation, _scale = orthogonal_procrustes(scaled_pred, centered_gt)
    aligned = scaled_pred @ rotation.T + gt_center
    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(aligned)
    distance_pred_to_gt = tree_gt.query(aligned)[0]
    distance_gt_to_pred = tree_pred.query(gt_points)[0]
    chamfer = float((distance_pred_to_gt.mean() + distance_gt_to_pred.mean()) / 2.0)
    diameter = float(np.linalg.norm(gt_points.max(axis=0) - gt_points.min(axis=0)))
    pred_extents = np.asarray(pred_mesh.extents, dtype=np.float64)
    gt_extents = np.asarray(gt_mesh.extents, dtype=np.float64)
    volume_ratio = float(pred_mesh.volume / max(gt_mesh.volume, 1e-12))
    longest_ratio = float(pred_extents.max() / max(gt_extents.max(), 1e-12))
    return {
        "sim3_chamfer_m": chamfer,
        "sim3_chamfer_normalized": float(chamfer / max(diameter, 1e-12)),
        "sim3_scale": sim3_scale,
        "volume_ratio": volume_ratio,
        "longest_extent_ratio": longest_ratio,
        "pred_extents_m": pred_extents.tolist(),
        "gt_extents_m": gt_extents.tolist(),
    }


def _object_pose_error(run_dir: Path, reference: Path, object_uid: str, prototype: str) -> dict[str, Any] | None:
    raw_path = run_dir / "object_tracking/foundationpose_raw.npz"
    selected_path = run_dir / "object_tracking/selected_mesh.json"
    if not raw_path.is_file() or not selected_path.is_file():
        return None
    gt_mesh_path = _find_gt_mesh(prototype)
    if gt_mesh_path is None:
        return {"status": "missing_gt_mesh"}
    raw = np.load(raw_path, allow_pickle=False)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    pred_mesh = _as_trimesh(trimesh.load(run_dir / selected["canonical_visual_mesh"], process=False))
    pred_mesh.apply_scale(float(selected["scale_to_m"]))
    gt_mesh = _as_trimesh(trimesh.load(gt_mesh_path, process=False))
    camera_at, object_at = _read_gt_trajectories(reference.parent, object_uid)
    times = np.asarray(json.loads((reference / "calibration/timestamps.json").read_text(encoding="utf-8"))["timestamps_s"], dtype=np.float64)
    pred_centroid = np.asarray(pred_mesh.centroid, dtype=np.float64)
    gt_centroid = np.asarray(gt_mesh.centroid, dtype=np.float64)
    errors: list[float] = []
    for index, transform in enumerate(raw["T_camera_object"]):
        if not bool(raw["valid"][index]):
            continue
        world_camera = _gt_camera_in_rect(reference, float(times[index]), camera_at)
        camera_world = _invert(world_camera)
        world_object = object_at(float(times[index]))
        gt_camera_object = camera_world @ world_object
        pred_camera_centroid = transform[:3, :3] @ pred_centroid + transform[:3, 3]
        gt_camera_centroid = gt_camera_object[:3, :3] @ gt_centroid + gt_camera_object[:3, 3]
        errors.append(float(np.linalg.norm(pred_camera_centroid - gt_camera_centroid)))
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "status": "evaluated",
        "centroid_translation_error_m": {
            "count": int(errors.size),
            "median": _percentile(errors, 50),
            "mean": float(errors.mean()) if errors.size else math.nan,
            "p95": _percentile(errors, 95),
            "max": float(errors.max()) if errors.size else math.nan,
        },
        "mesh": _mesh_diagnostics(pred_mesh, gt_mesh),
    }


def _row_for(summary_row: dict[str, Any]) -> dict[str, Any]:
    sequence = summary_row["sequence"]
    prototype = summary_row["prototype"]
    object_uid = str(summary_row.get("object_uid", ""))
    run_dir = Path(summary_row["run_dir"])
    seq_dir = ADT_ROOT / sequence
    reference = next(seq_dir.glob(f"stereo_prepared_{prototype}_f*"), None)
    if reference is None:
        # Some run dirs have no prepared marker in a predictable slug; infer from run id.
        slug = run_dir.name.replace("adt_", "", 1)
        parts = slug.rsplit("_f", 1)
        if len(parts) == 2:
            reference = seq_dir / f"stereo_prepared_{parts[0]}_f{parts[1]}"
    row: dict[str, Any] = {
        "sequence": sequence,
        "prototype": prototype,
        "object_uid": object_uid,
        "run_dir": str(run_dir),
        "depth_metrics": summary_row.get("metrics", {}),
    }
    gate_path = run_dir / "depth/depth_gate.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        row["stereo_depth_gate_accepted"] = bool(gate.get("accepted", False))
        row["stereo_depth_gate_failed_checks"] = [
            name for name, passed in gate.get("checks", {}).items() if not passed
        ]
    else:
        row["stereo_depth_gate_accepted"] = None
        row["stereo_depth_gate_failed_checks"] = []

    segmentation_path = run_dir / "segmentation/object_masks.npz"
    gt_zarr_path = run_dir / "evaluation/adt_gt/adt_gt.zarr"
    if segmentation_path.is_file() and gt_zarr_path.is_dir():
        row["sam3_segmentation"] = _segmentation_iou(segmentation_path, gt_zarr_path)
    else:
        row["sam3_segmentation"] = None

    hand_path = run_dir / "hands/wilor_stereo_metrics.json"
    if hand_path.is_file():
        hand = json.loads(hand_path.read_text(encoding="utf-8"))
        row["stereo_hand"] = {
            "accepted_any_hand": bool(hand.get("accepted_any_hand", False)),
            "per_hand": {
                side: {
                    "accepted": bool(side_data.get("accepted", False)),
                    "joint_valid_rate": float(side_data.get("joint_valid_rate", -1.0)),
                    "required_landmarks_frame_valid_rate": float(
                        side_data.get("required_landmarks_frame_valid_rate", -1.0)
                    ),
                    "reprojection_error_p95_px": float(
                        side_data.get("reprojection_error_px", {}).get("p95", -1.0)
                    ),
                }
                for side, side_data in hand.get("per_hand", {}).items()
            },
        }
    else:
        row["stereo_hand"] = None

    if reference is not None and object_uid:
        row["foundationpose_gt"] = _object_pose_error(run_dir, reference, object_uid, prototype)
    else:
        row["foundationpose_gt"] = None
    return row


def _gate_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_accept = false_accept = true_reject = false_reject = 0
    for row in rows:
        gate = row.get("stereo_depth_gate_accepted")
        if gate is None:
            continue
        metrics = row.get("depth_metrics", {})
        object_abs_rel = metrics.get("object_abs_rel", {}).get("median", math.nan)
        invalid_ratio = metrics.get("object_invalid_ratio", {}).get("median", math.nan)
        scale = metrics.get("object_scale_ratio_median", {}).get("median", math.nan)
        good = bool(
            object_abs_rel <= 0.15
            and invalid_ratio <= 0.20
            and 0.90 <= scale <= 1.10
        )
        if gate and good:
            true_accept += 1
        elif gate and not good:
            false_accept += 1
        elif not gate and good:
            false_reject += 1
        elif not gate and not good:
            true_reject += 1
    total = true_accept + false_accept + true_reject + false_reject
    return {
        "true_accept": true_accept,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "false_reject": false_reject,
        "false_accept_rate": float(false_accept / total) if total else math.nan,
        "false_reject_rate": float(false_reject / total) if total else math.nan,
        "precision": float(true_accept / (true_accept + false_accept)) if true_accept + false_accept else math.nan,
        "recall": float(true_accept / (true_accept + false_reject)) if true_accept + false_reject else math.nan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "runs" / "adt_upstream_audit_summary.json")
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [_row_for(item) for item in summary]
    payload = {
        "schema_version": "1.0",
        "source_summary": str(args.summary),
        "rows": rows,
        "stereo_depth_gate_reliability": _gate_reliability(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
