"""Project calibrated left-rectified stereo depth into the ADT RGB camera.

This is the P3 benchmark adapter from ``ADT_上游_Benchmark_评测与改进方案``.
It does not modify FoundationStereo, GT, or any existing run artifact.  It
produces an RGB-camera-frame metric-depth Zarr and compares it with the ADT
RGB GT depth stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..ingest import adt_stereo


RGB_CAMERA_LABEL = "camera-rgb"


def _image_array(image: Any) -> np.ndarray:
    return np.asarray(image.to_numpy_array())


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _stream_id_for_shape(provider: Any, shape: tuple[int, ...]) -> Any:
    for stream_id in provider.get_all_streams():
        if provider.get_sensor_data_type(stream_id).name != "IMAGE":
            continue
        image, _record = provider.get_image_data_by_index(stream_id, 0)
        array = _image_array(image)
        if array.shape == shape:
            return stream_id
    raise RuntimeError(f"no ADT GT stream with shape {shape}")


def _project_rectified_depth_to_rgb(
    *,
    depth_m: np.ndarray,
    valid: np.ndarray,
    K_rect: np.ndarray,
    T_rgb_rect: np.ndarray,
    rgb_calib: Any,
    rgb_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = depth_m.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    rays_rect = np.stack(
        [
            (u - K_rect[0, 2]) / K_rect[0, 0],
            (v - K_rect[1, 2]) / K_rect[1, 1],
            np.ones_like(u),
        ],
        axis=-1,
    )
    points_rect = rays_rect * depth_m[..., None]
    usable = valid & np.isfinite(depth_m) & (depth_m > 0)
    points = points_rect[usable].reshape(-1, 3)
    depths = depth_m[usable].reshape(-1)
    if not points.size:
        empty = np.full(rgb_shape, np.inf, dtype=np.float32)
        return empty, ~np.isfinite(empty), {"projected_point_count": 0, "visible_point_count": 0, "z_collision_count": 0}
    points_rgb = points @ T_rgb_rect[:3, :3].T + T_rgb_rect[:3, 3]
    projected_depth = np.full(rgb_shape, np.inf, dtype=np.float32)
    collision_count = 0
    for point, depth in zip(points_rgb, depths):
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        pixel = rgb_calib.project(point)
        if pixel is None or not np.all(np.isfinite(pixel)):
            continue
        x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        if not (0 <= x < rgb_shape[1] and 0 <= y < rgb_shape[0]):
            continue
        previous = projected_depth[y, x]
        if np.isfinite(previous) and previous < point[2]:
            collision_count += 1
        if point[2] < previous:
            projected_depth[y, x] = float(point[2])
    projected_valid = np.isfinite(projected_depth)
    return projected_depth, projected_valid, {
        "projected_point_count": int(points.size),
        "visible_point_count": int(projected_valid.sum()),
        "z_collision_count": collision_count,
    }


def _metric_for_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"count": 0, "abs_rel": float("nan"), "rmse_m": float("nan"), "delta_1": float("nan"), "scale_ratio": float("nan")}
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    good = np.isfinite(p) & (p > 0) & (g > 0)
    p, g = p[good], g[good]
    if not p.size:
        return {"count": 0, "abs_rel": float("nan"), "rmse_m": float("nan"), "delta_1": float("nan"), "scale_ratio": float("nan")}
    ratio = np.maximum(p / g, g / p)
    return {
        "count": int(p.size),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "scale_ratio": float(np.median(p / g)),
    }


def evaluate(
    *,
    run_dir: Path,
    sequence_dir: Path,
    prepared_dir: Path,
    object_uid: int,
    output_dir: Path,
) -> dict[str, Any]:
    import zarr

    run_dir = Path(run_dir).resolve()
    sequence_dir = Path(sequence_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    output_dir = Path(output_dir).resolve()
    prepared_meta = json.loads((prepared_dir / "adt_stereo_prepare.json").read_text(encoding="utf-8"))
    rectified_rotation = np.asarray(prepared_meta["rectification"]["device_to_rectified_rotation"], dtype=np.float64)
    K_rect = np.asarray(prepared_meta["rectification"]["K_rect_left"], dtype=np.float64)
    timestamps_s = np.asarray(
        json.loads((prepared_dir / "calibration/timestamps.json").read_text(encoding="utf-8"))["timestamps_s"],
        dtype=np.float64,
    )
    frame_indices = np.asarray(
        json.loads((prepared_dir / "calibration/frame_indices.json").read_text(encoding="utf-8"))["frame_indices"],
        dtype=np.int64,
    )
    pred_group = zarr.open_group(str(run_dir / "evaluation/foundationstereo/metric_depth.zarr"), mode="r")
    pred_depth = np.asarray(pred_group["depth_m"])
    pred_valid = np.asarray(pred_group["valid"])

    data_provider, _calibration, _mps, sensor_data, _sophus = adt_stereo._load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/video.vrs"))
    device_calib = video_provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib(RGB_CAMERA_LABEL)
    rgb_shape = tuple(int(value) for value in rgb_calib.get_image_size())
    T_device_rgb = adt_stereo._se3_to_matrix(rgb_calib.get_transform_device_camera())
    T_device_rect = np.eye(4, dtype=np.float64)
    T_device_rect[:3, :3] = rectified_rotation
    T_rgb_rect = np.linalg.inv(T_device_rgb) @ T_device_rect

    depth_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/depth_images.vrs"))
    segmentation_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/segmentations.vrs"))
    rgb_gt_depth_stream = _stream_id_for_shape(depth_provider, rgb_shape)
    rgb_gt_segmentation_stream = _stream_id_for_shape(segmentation_provider, rgb_shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    projected_group = zarr.open_group(str(output_dir / "depth_rgb.zarr"), mode="w")
    projected_depth_store = projected_group.create_dataset(
        "depth_m", shape=(len(frame_indices), *rgb_shape), chunks=(1, min(256, rgb_shape[0]), min(256, rgb_shape[1])), dtype="f4"
    )
    projected_valid_store = projected_group.create_dataset(
        "valid", shape=(len(frame_indices), *rgb_shape), chunks=(1, min(256, rgb_shape[0]), min(256, rgb_shape[1])), dtype="bool"
    )
    projected_group.create_dataset("frame_indices", data=frame_indices)
    projected_group.create_dataset("timestamps_s", data=timestamps_s)

    per_frame: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []
    object_metrics: list[dict[str, Any]] = []
    coverage_ratios: list[float] = []
    for index, timestamp_s in enumerate(timestamps_s):
        timestamp_ns = int(round(float(timestamp_s) * 1e9))
        gt_depth_image, _ = depth_provider.get_image_data_by_time_ns(
            rgb_gt_depth_stream, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
        )
        gt_segmentation_image, _ = segmentation_provider.get_image_data_by_time_ns(
            rgb_gt_segmentation_stream, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
        )
        gt_depth = _image_array(gt_depth_image).astype(np.float32) / 1000.0
        gt_segmentation = _image_array(gt_segmentation_image).astype(np.int64)
        object_mask = gt_segmentation == int(object_uid)
        projected_depth, projected_valid, audit = _project_rectified_depth_to_rgb(
            depth_m=np.asarray(pred_depth[index]),
            valid=np.asarray(pred_valid[index]),
            K_rect=K_rect,
            T_rgb_rect=T_rgb_rect,
            rgb_calib=rgb_calib,
            rgb_shape=rgb_shape,
        )
        projected_depth_store[index] = projected_depth.astype(np.float32)
        projected_valid_store[index] = projected_valid
        gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
        common = projected_valid & gt_valid
        coverage = float(common.sum() / max(int(gt_valid.sum()), 1))
        coverage_ratios.append(coverage)
        full = _metric_for_mask(projected_depth, gt_depth, common)
        obj = _metric_for_mask(projected_depth, gt_depth, object_mask & gt_valid & projected_valid)
        full_metrics.append(full)
        object_metrics.append(obj)
        per_frame.append(
            {
                "frame_index": int(frame_indices[index]),
                "timestamp_s": float(timestamp_s),
                "coverage_of_gt_valid": coverage,
                "projection": audit,
                "full": full,
                "object": obj,
            }
        )

    def aggregate(metrics: list[dict[str, Any]], name: str) -> dict[str, Any]:
        for metric_name in ("abs_rel", "rmse_m", "delta_1", "scale_ratio"):
            values = np.asarray([item[metric_name] for item in metrics], dtype=np.float64)
            values = values[np.isfinite(values)]
            result = {
                "median": float(np.median(values)) if values.size else float("nan"),
                "p95": float(np.percentile(values, 95)) if values.size else float("nan"),
            }
            if name == "object":
                result["count"] = int(sum(item.get("count", 0) for item in metrics))
            return result
        return {}

    payload = {
        "schema_version": "1.0",
        "purpose": "project left-rectified stereo depth into ADT RGB frame and compare with RGB GT depth",
        "run_dir": str(run_dir),
        "prepared_dir": str(prepared_dir),
        "rgb_shape": list(rgb_shape),
        "gt_stream_selection": {
            "depth": str(rgb_gt_depth_stream),
            "segmentation": str(rgb_gt_segmentation_stream),
        },
        "metrics": {
            "coverage": {
                "median": float(np.median(coverage_ratios)),
                "p95": float(np.percentile(coverage_ratios, 95)),
                "min": float(np.min(coverage_ratios)),
            },
            "full": {
                "abs_rel": _summary(full_metrics, "abs_rel"),
                "rmse_m": _summary(full_metrics, "rmse_m"),
                "delta_1": _summary(full_metrics, "delta_1"),
                "scale_ratio": _summary(full_metrics, "scale_ratio"),
            },
            "object": {
                "abs_rel": _summary(object_metrics, "abs_rel"),
                "rmse_m": _summary(object_metrics, "rmse_m"),
                "delta_1": _summary(object_metrics, "delta_1"),
                "scale_ratio": _summary(object_metrics, "scale_ratio"),
            },
        },
        "per_frame": per_frame,
    }
    (output_dir / "stereo_depth_to_rgb_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _summary(metrics: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([item[key] for item in metrics], dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "median": float(np.median(values)) if values.size else float("nan"),
        "p95": float(np.percentile(values, 95)) if values.size else float("nan"),
        "count": int(values.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--object-uid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        run_dir=args.run_dir,
        sequence_dir=args.sequence_dir,
        prepared_dir=args.prepared_dir,
        object_uid=args.object_uid,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
