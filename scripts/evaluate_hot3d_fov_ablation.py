#!/usr/bin/env python3
"""Evaluate P0/P1/P2 FOV rectification ablation.

The script reads original HOT3D tars for raw fisheye visibility and each
protocol's prepared run for virtual-pinhole visibility and monocular hand
artifacts. It writes all requested ablation artifacts under
runs/hot3d_hand_diagnosis/fov_rectification_ablation/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
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

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
P0_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
ABLATION_RUNS = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/fov_rectification_ablation/runs"
)
OUTPUT_ROOT = (
    REPO_ROOT / "runs/hot3d_hand_diagnosis/fov_rectification_ablation"
)
MANO_DIR = REPO_ROOT / "third_party/POEM-v2/assets/mano_v1_2/models"
LEFT_STREAM = "1201-1"
RIGHT_STREAM = "1201-2"
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
PCK_PIXEL_THRESHOLDS = (5.0, 10.0, 20.0)
SCORE_THRESHOLD = 0.2


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


def _build_gt_world(
    run_dir: Path,
    mano_model: MANOHandModel,
) -> tuple[np.ndarray, np.ndarray]:
    gt = _load_npz(run_dir / "hot3d_gt/hand_pose.npz")
    frame_numbers = np.asarray(gt["frame_numbers"], dtype=np.int64)
    count = len(frame_numbers)
    beta = (
        torch.from_numpy(np.asarray(gt["mano_beta"], dtype=np.float32))
        .float()
        .expand(count, -1)
    )
    theta = torch.from_numpy(np.asarray(gt["mano_theta"], dtype=np.float32)).float()
    wrist = torch.from_numpy(np.asarray(gt["wrist_xform"], dtype=np.float32)).float()
    hand_side = torch.full((count,), int(gt["hand_side"][0]), dtype=torch.long)
    with torch.no_grad():
        _, landmarks = mano_model(
            beta,
            theta,
            wrist,
            is_right_hand=hand_side.bool(),
        )
    return frame_numbers, landmarks.detach().cpu().numpy().astype(np.float64)


def _model_observation(
    artifact: dict[str, np.ndarray],
    side_index: int,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(artifact["valid"][:, side_index], dtype=bool)
    score = np.asarray(artifact["score"][:, side_index], dtype=np.float64)
    joints_camera = (
        np.asarray(artifact["joints_camera_rootrel"][:, side_index], dtype=np.float64)
        + np.asarray(artifact["translation_camera"][:, side_index, None], dtype=np.float64)
    )
    uv = _project(K, joints_camera)
    finite_z = np.isfinite(joints_camera).all(axis=-1) & (joints_camera[..., 2] > 0)
    p = valid & (score >= SCORE_THRESHOLD) & finite_z.all(axis=-1)
    required = p & finite_z[:, REQUIRED].all(axis=-1)
    return uv, joints_camera, p, required


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
    return {
        view: _load_npz(path)
        for view, path in paths.items()
        if path.is_file()
    }


def _pixel_summary(
    pred_uv: np.ndarray,
    gt_uv: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not mask.any():
        return {
            "count_frames": 0,
            "keypoint_error_px": {"mean": math.nan, "median": math.nan, "p95": math.nan},
            "wrist_error_px": {"mean": math.nan, "median": math.nan, "p95": math.nan},
            "fingertip_error_px": {"mean": math.nan, "median": math.nan, "p95": math.nan},
            "pck": {f"{int(t)}px": math.nan for t in PCK_PIXEL_THRESHOLDS},
        }
    errors = np.linalg.norm(
        np.asarray(pred_uv[mask], dtype=np.float64)
        - np.asarray(gt_uv[mask], dtype=np.float64),
        axis=-1,
    )
    wrist = np.linalg.norm(
        np.asarray(pred_uv[mask, 0], dtype=np.float64)
        - np.asarray(gt_uv[mask, 0], dtype=np.float64),
        axis=-1,
    )
    fingertip = np.linalg.norm(
        np.asarray(pred_uv[mask][:, FINGERTIPS], dtype=np.float64)
        - np.asarray(gt_uv[mask][:, FINGERTIPS], dtype=np.float64),
        axis=-1,
    )
    return {
        "count_frames": int(mask.sum()),
        "keypoint_error_px": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
        },
        "wrist_error_px": {
            "mean": float(np.mean(wrist)),
            "median": float(np.median(wrist)),
            "p95": float(np.percentile(wrist, 95)),
        },
        "fingertip_error_px": {
            "mean": float(np.mean(fingertip)),
            "median": float(np.median(fingertip)),
            "p95": float(np.percentile(fingertip, 95)),
        },
        "pck": {
            f"{int(t)}px": float(np.mean(errors <= t))
            for t in PCK_PIXEL_THRESHOLDS
        },
    }


def _raw_visibility_cached() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    return {}


def _raw_visibility(
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    clip: str,
    side: str,
    frame_numbers: np.ndarray,
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    key = (clip, side)
    if key in cache:
        return cache[key]
    visible_left = np.zeros(len(frame_numbers), dtype=bool)
    visible_right = np.zeros(len(frame_numbers), dtype=bool)
    with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
        for offset, frame_number in enumerate(frame_numbers):
            cams = json.load(tar.extractfile(f"{int(frame_number):06d}.cameras.json"))
            for stream, output in (
                (LEFT_STREAM, visible_left),
                (RIGHT_STREAM, visible_right),
            ):
                cam = camera.from_json(cams[stream])
                eye = cam.world_to_eye(world[offset])
                uv = cam.world_to_window(world[offset])
                output[offset] = bool(
                    (eye[REQUIRED, 2] > 0).any()
                    and cam.w_visible(uv[REQUIRED]).any()
                )
    cache[key] = (visible_left, visible_right)
    return visible_left, visible_right


def _pinhole_visibility(
    run_dir: Path,
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    T_left = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
    T_right = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
    left_cam = np.stack(
        [_transform_points(_invert(T_left[t]), world[t]) for t in range(len(world))],
        axis=0,
    )
    right_cam = np.stack(
        [_transform_points(_invert(T_right[t]), world[t]) for t in range(len(world))],
        axis=0,
    )
    uv_left = _project(K_left, left_cam)
    uv_right = _project(K_right, right_cam)
    width = int(K_left[0, 2] * 2)
    height = int(K_left[1, 2] * 2)
    left_visible = np.zeros(len(world), dtype=bool)
    right_visible = np.zeros(len(world), dtype=bool)
    for t in range(len(world)):
        left_visible[t] = bool(
            left_cam[t, 0, 2] > 0
            and (
                (uv_left[t, REQUIRED, 0] >= 0)
                & (uv_left[t, REQUIRED, 0] < width)
                & (uv_left[t, REQUIRED, 1] >= 0)
                & (uv_left[t, REQUIRED, 1] < height)
            ).any()
        )
        right_visible[t] = bool(
            right_cam[t, 0, 2] > 0
            and (
                (uv_right[t, REQUIRED, 0] >= 0)
                & (uv_right[t, REQUIRED, 0] < width)
                & (uv_right[t, REQUIRED, 1] >= 0)
                & (uv_right[t, REQUIRED, 1] < height)
            ).any()
        )
    return left_visible, right_visible, uv_left, uv_right


def _camera_audit_entry(
    protocol: str,
    run_dir: Path,
    clip: str,
) -> dict[str, Any]:
    K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
    T_left = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
    T_right = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
    width = int(round(K_left[0, 2] * 2))
    height = int(round(K_left[1, 2] * 2))
    fx, fy = float(K_left[0, 0]), float(K_left[1, 1])
    cx, cy = float(K_left[0, 2]), float(K_left[1, 2])
    hfov = float(2 * math.degrees(math.atan((width / 2.0) / fx)))
    vfov = float(2 * math.degrees(math.atan((height / 2.0) / fy)))
    same_intrinsics = bool(np.allclose(K_left, K_right))
    # Rectified if both rotation matrices are identical.
    rotation_equal = bool(
        np.allclose(
            np.asarray(T_left[:, :3, :3]),
            np.asarray(T_right[:, :3, :3]),
            atol=1e-6,
        )
    )
    baseline = float(
        np.linalg.norm(
            np.asarray(T_right[0, :3, 3]) - np.asarray(T_left[0, :3, 3])
        )
    )
    return {
        "protocol": protocol,
        "clip": clip,
        "run_dir": str(run_dir),
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "hfov_deg": hfov,
        "vfov_deg": vfov,
        "same_left_right_intrinsics": same_intrinsics,
        "rectified_common_orientation": rotation_equal,
        "baseline_m": baseline,
    }


def _protocol_items(protocol: str) -> list[dict[str, Any]]:
    if protocol == "P0":
        return json.loads(P0_MANIFEST.read_text(encoding="utf-8"))["items"]
    manifest = json.loads(
        (ABLATION_RUNS / protocol / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest["items"]


def _empty_agg() -> dict[str, Any]:
    return {
        "raw_left": 0,
        "raw_right": 0,
        "raw_both": 0,
        "pinhole_left": 0,
        "pinhole_right": 0,
        "pinhole_both": 0,
        "pinhole_both_retained": 0,
        "raw_left_to_pinhole_invisible": 0,
        "raw_right_to_pinhole_invisible": 0,
        "frames": 0,
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mano_model = MANOHandModel(str(MANO_DIR))
    raw_cache = _raw_visibility_cached()
    protocols = ["P0", "P1", "P2"]
    visibility_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    camera_audit: list[dict[str, Any]] = []

    for protocol in protocols:
        items = _protocol_items(protocol)
        for entry in items:
            run_dir = Path(entry["run_dir"]).resolve()
            clip = entry["clip"]
            side = entry["selected_side"]
            side_index = 0 if side == "left" else 1
            frame_numbers, world = _build_gt_world(run_dir, mano_model)
            raw_left, raw_right = _raw_visibility(
                raw_cache, clip, side, frame_numbers, world
            )
            pin_left, pin_right, uv_left, uv_right = _pinhole_visibility(run_dir, world)
            gt_uv = {"left": uv_left, "right": uv_right}
            raw_both = raw_left & raw_right
            pin_both = pin_left & pin_right
            if protocol == "P0":
                camera_audit.append(_camera_audit_entry(protocol, run_dir, clip))
            visibility_rows.append(
                {
                    "protocol": protocol,
                    "clip": clip,
                    "selected_hand": side,
                    "frames": int(gt_uv["left"].shape[0]),
                    "raw_left": int(raw_left.sum()),
                    "raw_right": int(raw_right.sum()),
                    "raw_both": int(raw_both.sum()),
                    "pinhole_left": int(pin_left.sum()),
                    "pinhole_right": int(pin_right.sum()),
                    "pinhole_both": int(pin_both.sum()),
                    "pinhole_both_retained": int((raw_both & pin_both).sum()),
                    "retention": (
                        float((raw_both & pin_both).sum() / raw_both.sum())
                        if raw_both.any()
                        else None
                    ),
                    "raw_left_to_pinhole_invisible": int(
                        (raw_left & ~pin_left).sum()
                    ),
                    "raw_right_to_pinhole_invisible": int(
                        (raw_right & ~pin_right).sum()
                    ),
                }
            )
            K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
            K_right = np.load(
                run_dir / "calibration/intrinsics_right.npy"
            ).astype(np.float64)
            for model in ("wilor", "hamer"):
                artifacts = _load_model_artifacts(run_dir, model)
                if "left" not in artifacts or "right" not in artifacts:
                    detection_rows.append(
                        {
                            "protocol": protocol,
                            "clip": clip,
                            "model": model,
                            "status": "artifacts_missing",
                        }
                    )
                    continue
                left_obs = _model_observation(artifacts["left"], side_index, K_left)
                right_obs = _model_observation(artifacts["right"], side_index, K_right)
                left_uv, _, left_p, left_req = left_obs
                right_uv, _, right_p, right_req = right_obs
                left_det = left_p & pin_left
                right_det = right_p & pin_right
                both_det = left_p & right_p & pin_both
                detection_rows.append(
                    {
                        "protocol": protocol,
                        "clip": clip,
                        "selected_hand": side,
                        "model": model,
                        "status": "evaluated",
                        "gt_left_visible": int(pin_left.sum()),
                        "gt_right_visible": int(pin_right.sum()),
                        "gt_both_visible": int(pin_both.sum()),
                        "left_detection_rate": (
                            float(left_det.sum() / pin_left.sum())
                            if pin_left.any()
                            else math.nan
                        ),
                        "right_detection_rate": (
                            float(right_det.sum() / pin_right.sum())
                            if pin_right.any()
                            else math.nan
                        ),
                        "both_detection_rate": (
                            float(both_det.sum() / pin_both.sum())
                            if pin_both.any()
                            else math.nan
                        ),
                        "left_required_rate": (
                            float((left_req & pin_left).sum() / pin_left.sum())
                            if pin_left.any()
                            else math.nan
                        ),
                        "right_required_rate": (
                            float((right_req & pin_right).sum() / pin_right.sum())
                            if pin_right.any()
                            else math.nan
                        ),
                    }
                )
                for view_name, pred_uv, req_mask, gt_domain in (
                    ("left", left_uv, left_req, pin_left),
                    ("right", right_uv, right_req, pin_right),
                ):
                    mask = req_mask & gt_domain
                    metric = _pixel_summary(
                        pred_uv, gt_uv[view_name], mask
                    )
                    metric.update(
                        {
                            "protocol": protocol,
                            "clip": clip,
                            "selected_hand": side,
                            "model": model,
                            "view": view_name,
                        }
                    )
                    metric_rows.append(metric)

    # Aggregate visibility summary.
    vis_summary: dict[str, Any] = {"rows": visibility_rows}
    protocol_vis: dict[str, dict[str, Any]] = {}
    for row in visibility_rows:
        agg = protocol_vis.setdefault(row["protocol"], _empty_agg())
        agg["raw_left"] += row["raw_left"]
        agg["raw_right"] += row["raw_right"]
        agg["raw_both"] += row["raw_both"]
        agg["pinhole_left"] += row["pinhole_left"]
        agg["pinhole_right"] += row["pinhole_right"]
        agg["pinhole_both"] += row["pinhole_both"]
        agg["pinhole_both_retained"] += row["pinhole_both_retained"]
        agg["raw_left_to_pinhole_invisible"] += row[
            "raw_left_to_pinhole_invisible"
        ]
        agg["raw_right_to_pinhole_invisible"] += row[
            "raw_right_to_pinhole_invisible"
        ]
        agg["frames"] += row["frames"]
    for protocol, agg in protocol_vis.items():
        agg["raw_both_to_pinhole_both_retention"] = (
            float(agg["pinhole_both_retained"] / agg["raw_both"])
            if agg["raw_both"]
            else None
        )
    vis_summary["protocol_totals"] = protocol_vis

    # Aggregate hand detection summary.
    det_summary: dict[str, Any] = {"rows": detection_rows}
    det_totals: dict[str, dict[str, Any]] = {}
    for row in detection_rows:
        if row.get("status") != "evaluated":
            continue
        key = (row["protocol"], row["model"])
        agg = det_totals.setdefault(
            f"{key[0]}_{key[1]}",
            {
                "protocol": key[0],
                "model": key[1],
                "gt_left_visible": 0,
                "gt_right_visible": 0,
                "gt_both_visible": 0,
                "left_detected": 0,
                "right_detected": 0,
                "both_detected": 0,
            },
        )
        agg["gt_left_visible"] += row["gt_left_visible"]
        agg["gt_right_visible"] += row["gt_right_visible"]
        agg["gt_both_visible"] += row["gt_both_visible"]
        if math.isfinite(row["left_detection_rate"]):
            agg["left_detected"] += row["left_detection_rate"] * row["gt_left_visible"]
        if math.isfinite(row["right_detection_rate"]):
            agg["right_detected"] += row["right_detection_rate"] * row["gt_right_visible"]
        if math.isfinite(row["both_detection_rate"]):
            agg["both_detected"] += row["both_detection_rate"] * row["gt_both_visible"]
    for key, agg in det_totals.items():
        agg["left_detection_rate"] = (
            agg["left_detected"] / agg["gt_left_visible"]
            if agg["gt_left_visible"]
            else math.nan
        )
        agg["right_detection_rate"] = (
            agg["right_detected"] / agg["gt_right_visible"]
            if agg["gt_right_visible"]
            else math.nan
        )
        agg["both_detection_rate"] = (
            agg["both_detected"] / agg["gt_both_visible"]
            if agg["gt_both_visible"]
            else math.nan
        )
    det_summary["protocol_totals"] = list(det_totals.values())

    # Aggregate 2D metrics by protocol and model.
    metric_summary: dict[str, Any] = {"rows": metric_rows}
    metric_totals: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        key = (row["protocol"], row["model"])
        agg = metric_totals.setdefault(
            f"{key[0]}_{key[1]}",
            {
                "protocol": key[0],
                "model": key[1],
                "count_frames": 0,
                "keypoint_median_values": [],
                "wrist_median_values": [],
                "fingertip_median_values": [],
                "pck5_values": [],
                "pck10_values": [],
                "pck20_values": [],
            },
        )
        if row["count_frames"] == 0:
            continue
        agg["count_frames"] += row["count_frames"]
        agg["keypoint_median_values"].append(row["keypoint_error_px"]["median"])
        agg["wrist_median_values"].append(row["wrist_error_px"]["median"])
        agg["fingertip_median_values"].append(row["fingertip_error_px"]["median"])
        agg["pck5_values"].append(row["pck"]["5px"])
        agg["pck10_values"].append(row["pck"]["10px"])
        agg["pck20_values"].append(row["pck"]["20px"])
    for key, agg in metric_totals.items():
        agg["keypoint_error_px_median_of_medians"] = float(
            np.nanmedian(agg["keypoint_median_values"])
        )
        agg["wrist_error_px_median_of_medians"] = float(
            np.nanmedian(agg["wrist_median_values"])
        )
        agg["fingertip_error_px_median_of_medians"] = float(
            np.nanmedian(agg["fingertip_median_values"])
        )
        agg["mean_pck5"] = float(np.nanmean(agg["pck5_values"]))
        agg["mean_pck10"] = float(np.nanmean(agg["pck10_values"]))
        agg["mean_pck20"] = float(np.nanmean(agg["pck20_values"]))
        for key_name in (
            "keypoint_median_values",
            "wrist_median_values",
            "fingertip_median_values",
            "pck5_values",
            "pck10_values",
            "pck20_values",
        ):
            agg.pop(key_name)
    metric_summary["protocol_totals"] = list(metric_totals.values())

    # Write JSON and CSV artifacts.
    config = {
        "schema_version": "1.0",
        "protocols": {
            "P0": "same_size_pinhole_from_source_fisheye",
            "P1": "wide_pinhole_640x480_f160_rectified",
            "P2": "ultrawide_pinhole_1600x1200_f80_rectified",
        },
        "raw_visibility_definition": (
            "official HOT3D fisheye CameraModel; required joints "
            "[0,4,8,12,16,20] at least one inside raw sensor with z>0"
        ),
        "pinhole_visibility_definition": (
            "protocol virtual pinhole; wrist z>0 and at least one required joint "
            "inside output image"
        ),
        "detection_definition": (
            "model valid AND score>=0.2 AND all 21 joints finite z>0"
        ),
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "visibility_summary.json").write_text(
        json.dumps(vis_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "hand_detection_summary.json").write_text(
        json.dumps(det_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "hand_2d_summary.json").write_text(
        json.dumps(metric_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    camera_protocols = {
        "schema_version": "1.0",
        "P0": _camera_audit_entry("P0", Path(_protocol_items("P0")[0]["run_dir"]), _protocol_items("P0")[0]["clip"]),
        "P1": _camera_audit_entry("P1", Path(_protocol_items("P1")[0]["run_dir"]), _protocol_items("P1")[0]["clip"]),
        "P2": _camera_audit_entry("P2", Path(_protocol_items("P2")[0]["run_dir"]), _protocol_items("P2")[0]["clip"]),
        "all_P0_entries": [
            _camera_audit_entry("P0", Path(item["run_dir"]), item["clip"])
            for item in _protocol_items("P0")
        ],
    }
    (OUTPUT_ROOT / "camera_protocols.json").write_text(
        json.dumps(camera_protocols, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (OUTPUT_ROOT / "visibility_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "protocol", "clip", "selected_hand", "frames",
                "raw_left", "raw_right", "raw_both",
                "pinhole_left", "pinhole_right", "pinhole_both",
                "retention",
            ]
        )
        for row in visibility_rows:
            writer.writerow(
                [
                    row["protocol"], row["clip"], row["selected_hand"], row["frames"],
                    row["raw_left"], row["raw_right"], row["raw_both"],
                    row["pinhole_left"], row["pinhole_right"], row["pinhole_both"],
                    row["retention"],
                ]
            )
    with (OUTPUT_ROOT / "hand_detection_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "protocol", "clip", "selected_hand", "model", "status",
                "gt_left_visible", "gt_right_visible", "gt_both_visible",
                "left_detection_rate", "right_detection_rate", "both_detection_rate",
            ]
        )
        for row in detection_rows:
            writer.writerow(
                [
                    row.get("protocol"), row.get("clip"), row.get("selected_hand"),
                    row.get("model"), row.get("status"),
                    row.get("gt_left_visible"), row.get("gt_right_visible"),
                    row.get("gt_both_visible"), row.get("left_detection_rate"),
                    row.get("right_detection_rate"), row.get("both_detection_rate"),
                ]
            )
    with (OUTPUT_ROOT / "hand_2d_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "protocol", "clip", "selected_hand", "model", "view",
                "count_frames", "keypoint_median_px", "wrist_median_px",
                "fingertip_median_px", "pck5", "pck10", "pck20",
            ]
        )
        for row in metric_rows:
            writer.writerow(
                [
                    row["protocol"], row["clip"], row["selected_hand"], row["model"],
                    row["view"], row["count_frames"],
                    row["keypoint_error_px"]["median"],
                    row["wrist_error_px"]["median"],
                    row["fingertip_error_px"]["median"],
                    row["pck"]["5px"], row["pck"]["10px"], row["pck"]["20px"],
                ]
            )

    print("ablation outputs ->", OUTPUT_ROOT)
    print(json.dumps(vis_summary["protocol_totals"], indent=2, ensure_ascii=False))
    print(json.dumps(det_summary["protocol_totals"], indent=2, ensure_ascii=False))
    print(json.dumps(metric_summary["protocol_totals"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
