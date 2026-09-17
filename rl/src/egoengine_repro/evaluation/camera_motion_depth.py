"""Evaluation-only audit for camera-motion calibrated depth artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ..artifacts import artifact_record
from .metrics import summary


EPISODES = (
    "taco_brush_brush_bowl_20230927_027",
    "taco_cut_spatula_plate_20230917_020",
    "taco_skim_spatula_plate_20230926_004",
    "taco_smear_eraser_box_20231103_071",
)


def _fit_depth_model(predicted: np.ndarray, target: np.ndarray, *, affine: bool) -> dict[str, Any]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    use = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    x, y = x[use], y[use]
    if x.size < 8:
        return {"status": "insufficient_samples", "sample_count": int(x.size)}
    design = np.column_stack([x, np.ones_like(x)]) if affine else x[:, None]
    keep = np.ones(x.size, dtype=bool)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
        residual = y - design @ coefficients
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        threshold = max(3.0 * 1.4826 * mad, 0.01)
        keep = np.abs(residual - center) <= threshold
    coefficients, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
    residual = y[keep] - design[keep] @ coefficients
    scale = float(coefficients[0])
    shift = float(coefficients[1]) if affine else 0.0
    return {
        "status": "fit" if scale > 0 and np.isfinite(scale) else "invalid_scale",
        "sample_count": int(x.size),
        "inlier_count": int(np.count_nonzero(keep)),
        "scale": scale,
        "shift_m": shift,
        "residual_m": summary(np.abs(residual)),
    }


def _read_zarr(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    group = zarr.open_group(str(path), mode="r")
    return {name: group[name] for name in names}


def _uniform_sample(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if count <= 0 or len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[indices]


def _sample_episode(
    raw_path: Path, calibrated_path: Path, oracle_path: Path, *, max_samples: int,
) -> dict[str, Any]:
    raw = _read_zarr(raw_path, ("depth_m", "valid"))
    calibrated = _read_zarr(calibrated_path, ("depth_m", "valid"))
    oracle = _read_zarr(oracle_path, ("depth_m", "valid", "object_geometry_mask"))
    frame_count = min(raw["depth_m"].shape[0], calibrated["depth_m"].shape[0], oracle["depth_m"].shape[0])
    raw_values: list[np.ndarray] = []
    calibrated_values: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    calibrated_targets: list[np.ndarray] = []
    per_frame_limit = (
        max(1, int(np.ceil(max_samples / max(frame_count, 1))))
        if max_samples > 0 else 0
    )
    valid_counts = {"raw": 0, "calibrated": 0, "target": 0, "paired_raw": 0, "paired_calibrated": 0}
    for frame in range(frame_count):
        target = np.asarray(oracle["depth_m"][frame], dtype=np.float64)
        object_mask = np.asarray(oracle["object_geometry_mask"][frame], dtype=bool)
        raw_depth = np.asarray(raw["depth_m"][frame], dtype=np.float64)
        calibrated_depth = np.asarray(calibrated["depth_m"][frame], dtype=np.float64)
        target_valid = object_mask & np.isfinite(target) & (target > 0)
        raw_valid = target_valid & np.asarray(raw["valid"][frame], dtype=bool) & np.isfinite(raw_depth) & (raw_depth > 0)
        calibrated_valid = target_valid & np.asarray(calibrated["valid"][frame], dtype=bool) & np.isfinite(calibrated_depth) & (calibrated_depth > 0)
        valid_counts["target"] += int(target_valid.sum())
        valid_counts["raw"] += int((np.asarray(raw["valid"][frame], dtype=bool) & (raw_depth > 0)).sum())
        valid_counts["calibrated"] += int((np.asarray(calibrated["valid"][frame], dtype=bool) & (calibrated_depth > 0)).sum())
        valid_counts["paired_raw"] += int(raw_valid.sum())
        valid_counts["paired_calibrated"] += int(calibrated_valid.sum())
        if raw_valid.any():
            indices = np.flatnonzero(raw_valid)
            indices = _uniform_sample(indices, per_frame_limit)
            raw_values.append(raw_depth.reshape(-1)[indices])
            raw_targets.append(target.reshape(-1)[indices])
        if calibrated_valid.any():
            indices = np.flatnonzero(calibrated_valid)
            indices = _uniform_sample(indices, per_frame_limit)
            calibrated_values.append(calibrated_depth.reshape(-1)[indices])
            calibrated_targets.append(target.reshape(-1)[indices])
    raw_flat = np.concatenate(raw_values) if raw_values else np.zeros(0)
    target_raw = np.concatenate(raw_targets) if raw_targets else np.zeros(0)
    cal_flat = np.concatenate(calibrated_values) if calibrated_values else np.zeros(0)
    target_cal = np.concatenate(calibrated_targets) if calibrated_targets else np.zeros(0)
    if max_samples > 0:
        raw_indices = _uniform_sample(np.arange(len(raw_flat)), max_samples)
        cal_indices = _uniform_sample(np.arange(len(cal_flat)), max_samples)
        raw_flat, target_raw = raw_flat[raw_indices], target_raw[raw_indices]
        cal_flat, target_cal = cal_flat[cal_indices], target_cal[cal_indices]

    def direct(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        error = pred - target
        return {
            "sample_count": int(pred.size),
            "absolute_error_m": summary(np.abs(error)),
            "signed_error_m": summary(error),
            "depth_ratio": summary(pred / target) if pred.size else summary(np.zeros(0)),
        }

    object_pixels = valid_counts["target"]
    return {
        "frame_count": int(frame_count),
        "object_valid_pixel_count": object_pixels,
        "valid_rate": {
            "raw_on_object": valid_counts["paired_raw"] / max(object_pixels, 1),
            "calibrated_on_object": valid_counts["paired_calibrated"] / max(object_pixels, 1),
        },
        "raw_depth": {
            "direct": direct(raw_flat, target_raw),
            "scale_only": _fit_depth_model(raw_flat, target_raw, affine=False),
            "affine": _fit_depth_model(raw_flat, target_raw, affine=True),
        },
        "calibrated_depth": {
            "direct": direct(cal_flat, target_cal),
            "scale_only": _fit_depth_model(cal_flat, target_cal, affine=False),
            "affine": _fit_depth_model(cal_flat, target_cal, affine=True),
        },
        "sampling": {
            "max_samples": int(max_samples),
            "strategy": "uniform_per_frame_then_uniform_global",
            "per_frame_limit": int(per_frame_limit),
        },
    }


def audit_camera_motion_depth(
    run_dir: str | Path, oracle_root: str | Path, output: str | Path, *, max_samples: int = 200_000,
) -> Path:
    run_root, oracle_root, destination = Path(run_dir).resolve(), Path(oracle_root).resolve(), Path(output).resolve()
    episodes: dict[str, Any] = {}
    for episode_id in EPISODES:
        metadata_path = run_root / episode_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_raw = Path(metadata["source_depth"])
        if not source_raw.is_absolute():
            source_raw = (metadata_path.parent / source_raw).resolve()
        oracle_episode = oracle_root / episode_id / "profiles/known_mesh_oracle_depth/run/rendered_gt_proxy"
        oracle_metadata = json.loads((oracle_episode / "metadata.json").read_text(encoding="utf-8"))
        if oracle_metadata.get("quality_label") != "rendered_gt_proxy" or oracle_metadata.get("uses_ground_truth") is not True:
            raise ValueError(f"oracle artifact is not an evaluation-only rendered_gt_proxy: {oracle_episode}")
        episodes[episode_id] = _sample_episode(
            source_raw, run_root / episode_id / "metric_depth.zarr",
            oracle_episode / "object_metric_depth.zarr", max_samples=max_samples,
        )
        episodes[episode_id]["artifacts"] = {
            "raw_depth": artifact_record(source_raw),
            "calibrated_depth": artifact_record(run_root / episode_id / "metric_depth.zarr"),
            "rendered_gt_proxy_metadata": artifact_record(oracle_episode / "metadata.json"),
            "rendered_gt_proxy_depth": artifact_record(oracle_episode / "object_metric_depth.zarr"),
        }
    calibrated_means = [
        float(item["calibrated_depth"]["direct"]["absolute_error_m"]["mean"])
        for item in episodes.values()
    ]
    calibrated_p95 = [
        float(item["calibrated_depth"]["direct"]["absolute_error_m"]["p95"])
        for item in episodes.values()
    ]
    improves_raw = [
        float(item["calibrated_depth"]["direct"]["absolute_error_m"]["mean"])
        < float(item["raw_depth"]["direct"]["absolute_error_m"]["mean"])
        for item in episodes.values()
    ]
    payload = {
        "schema_version": "1.0",
        "profile": "camera_motion_calibrated_depth_object_audit",
        "scope": "evaluation_only",
        "quality_label": "rendered_gt_proxy",
        "run_name": run_root.name,
        "episodes": episodes,
        "inference_policy": {
            "object_gt_used": False, "gt_hand_used": False,
            "manual_point_used": False, "oracle_depth_used": False,
            "gt_enters_inference": False,
        },
        "acceptance": {
            "episode_count": len(episodes),
            "strict_four_episode_pairing": len(episodes) == 4,
            "calibrated_object_depth_equal_episode_mean_m": float(
                np.mean(calibrated_means)
            ),
            "calibrated_object_depth_worst_episode_p95_m": float(
                np.max(calibrated_p95)
            ),
            "calibrated_object_depth_mean_below_0.10": bool(
                np.mean(calibrated_means) < 0.10
            ),
            "calibrated_object_depth_mean_below_0.08": bool(
                np.mean(calibrated_means) < 0.08
            ),
            "calibrated_object_depth_all_p95_below_0.50": bool(
                np.max(calibrated_p95) < 0.50
            ),
            "calibrated_object_depth_improves_raw_all_episodes": bool(
                all(improves_raw)
            ),
            "foundationpose_comparison_candidate": bool(
                len(episodes) == 4
                and np.mean(calibrated_means) < 0.10
                and np.max(calibrated_p95) < 0.50
                and all(improves_raw)
            ),
            "translation_gate_evaluated": False,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
