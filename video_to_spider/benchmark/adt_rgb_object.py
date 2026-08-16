"""Prepare an ADT RGB pinhole object branch for P4 object Digital Twin.

This is a benchmark-only adapter. It does not modify the production stereo
ingest and does not alter FoundationStereo or GT. It converts the raw ADT
``camera-rgb`` fisheye stream into a fixed pinhole camera, projects the
existing left-rectified FoundationStereo metric depth into that same pinhole,
and creates a standard run directory that the existing SAM3 / SAM3D /
FoundationPose adapters can consume.

The pinhole frame shares the raw RGB camera coordinate frame; only the camera
model changes from ``FISHEYE624`` to a pinhole model. This is necessary
because the production perception adapters are written against OpenCV pinhole
intrinsics, while ADT RGB is fisheye.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_to_spider.ingest import adt_stereo
from video_to_spider.manifest import RunManifest

from .adt_stereo_to_rgb import (
    RGB_CAMERA_LABEL,
    _image_array,
    _pose,
    _stream_id_for_shape,
)


DEFAULT_PINHOLE_SIZE = 1024
DEFAULT_PINHOLE_FOCAL = 250.0


def _pinhole_intrinsics(size: int, focal: float) -> np.ndarray:
    return np.asarray(
        [
            [focal, 0.0, float(size) / 2.0],
            [0.0, focal, float(size) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_rgb_pinhole_maps(
    rgb_calib: Any, size: int, focal: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
    )
    K = _pinhole_intrinsics(size, focal)
    rays = np.stack(
        [
            (u - K[0, 2]) / K[0, 0],
            (v - K[1, 2]) / K[1, 1],
            np.ones_like(u),
        ],
        axis=-1,
    )
    map_x = np.full((size, size), -1.0, dtype=np.float32)
    map_y = np.full((size, size), -1.0, dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            ray = rays[y, x]
            pixel = rgb_calib.project(ray)
            if pixel is None:
                continue
            px, py = np.asarray(pixel, dtype=np.float64)
            if np.isfinite(px) and np.isfinite(py):
                map_x[y, x] = float(px)
                map_y[y, x] = float(py)
                valid[y, x] = True
    return map_x, map_y, valid


def _remap_rgb_frame(
    raw_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_rgb)
    if raw.ndim == 3 and raw.shape[-1] == 3:
        raw = raw[:, :, ::-1]  # RGB -> BGR for OpenCV I/O
    remapped = cv2.remap(
        raw,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    remapped[~valid] = 0
    return remapped


def _project_rectified_depth_to_pinhole(
    *,
    depth_m: np.ndarray,
    valid: np.ndarray,
    K_rect: np.ndarray,
    T_rgb_rect: np.ndarray,
    K_rgb: np.ndarray,
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
    if not points.size:
        empty = np.full(rgb_shape, np.inf, dtype=np.float32)
        return empty, ~np.isfinite(empty), {"projected_point_count": 0, "visible_point_count": 0, "z_collision_count": 0}
    points_rgb = points @ T_rgb_rect[:3, :3].T + T_rgb_rect[:3, 3]
    projected_depth = np.full(rgb_shape, np.inf, dtype=np.float32)
    collision_count = 0
    for point in points_rgb:
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        pixel = K_rgb @ point
        if abs(pixel[2]) < 1e-12:
            continue
        x = pixel[0] / pixel[2]
        y = pixel[1] / pixel[2]
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < rgb_shape[1] and 0 <= yi < rgb_shape[0]):
            continue
        previous = projected_depth[yi, xi]
        if np.isfinite(previous) and previous < point[2]:
            collision_count += 1
        if point[2] < previous:
            projected_depth[yi, xi] = float(point[2])
    projected_valid = np.isfinite(projected_depth)
    return projected_depth, projected_valid, {
        "projected_point_count": int(points.size),
        "visible_point_count": int(projected_valid.sum()),
        "z_collision_count": collision_count,
    }


def _quat_xyzw_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
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


def _rgb_world_cameras(sequence_dir: Path, rgb_calib: Any, timestamps_s: np.ndarray) -> np.ndarray:
    rows = list(csv.DictReader((sequence_dir / "aria_trajectory.csv").open(encoding="utf-8")))
    camera_times = np.asarray([float(row["tracking_timestamp_us"]) * 1e-6 for row in rows], dtype=np.float64)
    T_device_rgb = adt_stereo._se3_to_matrix(rgb_calib.get_transform_device_camera())
    result = np.empty((len(timestamps_s), 4, 4), dtype=np.float64)
    for index, timestamp in enumerate(timestamps_s):
        row = rows[int(np.argmin(np.abs(camera_times - timestamp)))]
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
        result[index] = world_device @ T_device_rgb
    return result


def prepare_object_run(
    *,
    sequence_dir: str | Path,
    prepared_dir: str | Path,
    source_run_dir: str | Path,
    output_run_dir: str | Path,
    object_keyword_candidates: list[str],
    pinhole_size: int = DEFAULT_PINHOLE_SIZE,
    pinhole_focal: float = DEFAULT_PINHOLE_FOCAL,
    overwrite: bool = False,
) -> Path:
    import zarr

    sequence_dir = Path(sequence_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    source_run_dir = Path(source_run_dir).resolve()
    output_run_dir = Path(output_run_dir).resolve()
    if output_run_dir.exists() and any(output_run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output run: {output_run_dir}")
    if output_run_dir.exists() and overwrite:
        for child in output_run_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
        # Do not recursively delete unknown directories from an overwrite path.

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

    data_provider, _calibration, _mps, sensor_data, _sophus = adt_stereo._load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/video.vrs"))
    device_calib = video_provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib(RGB_CAMERA_LABEL)
    rgb_stream = video_provider.get_stream_id_from_label(RGB_CAMERA_LABEL)
    rgb_shape = tuple(int(value) for value in rgb_calib.get_image_size())
    T_device_rgb = adt_stereo._se3_to_matrix(rgb_calib.get_transform_device_camera())
    T_device_rect = np.eye(4, dtype=np.float64)
    T_device_rect[:3, :3] = rectified_rotation
    T_rgb_rect = np.linalg.inv(T_device_rgb) @ T_device_rect

    pinhole_K = _pinhole_intrinsics(pinhole_size, pinhole_focal)
    map_x, map_y, rgb_valid = _build_rgb_pinhole_maps(rgb_calib, pinhole_size, pinhole_focal)

    frames_dir = output_run_dir / "frames" / "rgb"
    calibration_dir = output_run_dir / "calibration"
    depth_dir = output_run_dir / "depth"
    input_dir = output_run_dir / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, Any]] = []
    base_timestamp_s = float(timestamps_s[0])
    for index, timestamp_s in enumerate(timestamps_s):
        timestamp_ns = int(round(float(timestamp_s) * 1e9))
        image, _record = video_provider.get_image_data_by_time_ns(
            rgb_stream, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
        )
        raw_rgb = _image_array(image)
        bgr = _remap_rgb_frame(raw_rgb, map_x, map_y, rgb_valid)
        relative_name = f"{index:06d}.png"
        cv2.imwrite(str(frames_dir / relative_name), bgr)
        frame_rows.append(
            {
                "frame_index": index,
                "source_frame_index": int(frame_indices[index]),
                "timestamp_s": float(timestamp_s) - base_timestamp_s,
                "source_timestamp_s": float(timestamp_s),
                "rgb_path": f"frames/rgb/{relative_name}",
                "source_relative_name": relative_name,
            }
        )

    (output_run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (output_run_dir / "frames/frame_index.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "fps": float(prepared_meta["fps"]),
                "frames": frame_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    np.save(calibration_dir / "intrinsics.npy", pinhole_K)
    world_cameras = _rgb_world_cameras(sequence_dir, rgb_calib, timestamps_s)
    np.save(calibration_dir / "T_world_camera.npy", world_cameras)

    pred_group = zarr.open_group(str(source_run_dir / "evaluation/foundationstereo/metric_depth.zarr"), mode="r")
    pred_depth = np.asarray(pred_group["depth_m"])
    pred_valid = np.asarray(pred_group["valid"])
    depth_group = zarr.open_group(str(depth_dir / "metric_depth.zarr"), mode="w")
    depth_store = depth_group.create_dataset(
        "depth_m", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="f4"
    )
    valid_store = depth_group.create_dataset(
        "valid", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="bool"
    )
    depth_group.create_dataset("frame_indices", data=frame_indices)
    depth_group.create_dataset("timestamps_s", data=timestamps_s)
    projection_summary: list[dict[str, Any]] = []
    for index in range(len(timestamps_s)):
        projected_depth, projected_valid, audit = _project_rectified_depth_to_pinhole(
            depth_m=np.asarray(pred_depth[index]),
            valid=np.asarray(pred_valid[index]),
            K_rect=K_rect,
            T_rgb_rect=T_rgb_rect,
            K_rgb=pinhole_K,
            rgb_shape=(pinhole_size, pinhole_size),
        )
        depth_store[index] = projected_depth.astype(np.float32)
        valid_store[index] = projected_valid
        projection_summary.append({"frame_index": int(frame_indices[index]), "timestamp_s": float(timestamps_s[index]), **audit})

    metadata = {
        "schema_version": "1.0",
        "model": "adt_stereo_to_rgb_pinhole",
        "purpose": "FoundationStereo rectified-left depth projected into a fixed ADT RGB pinhole camera",
        "depth_path": "depth/metric_depth.zarr",
        "K_pinhole": pinhole_K.tolist(),
        "pinhole_size": pinhole_size,
        "pinhole_focal": float(pinhole_focal),
        "rgb_shape": list(rgb_shape),
        "T_device_rgb": T_device_rgb.tolist(),
        "frame_count": int(len(timestamps_s)),
        "fps": float(prepared_meta["fps"]),
        "projection_summary": projection_summary,
    }
    (depth_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    source_payload = {
        "schema_version": "1.0",
        "source_type": "adt_rgb_pinhole_object_benchmark",
        "task_directory": "adt",
        "episode_id": str(sequence_dir.name),
        "instruction_text": "pick up the object",
        "instruction_source": "benchmark_adapter",
        "object_keyword_candidates": list(object_keyword_candidates),
        "video": {
            "frame_count": int(len(timestamps_s)),
            "fps": float(prepared_meta["fps"]),
            "width": pinhole_size,
            "height": pinhole_size,
        },
        "selected_frame_interval": [0, int(len(timestamps_s))],
        "reference_camera": "adt_rgb_pinhole",
        "camera_pose_source": str(calibration_dir / "T_world_camera.npy"),
        "depth_route_policy": "FoundationStereo -> calibrated RGB pinhole projection",
    }
    (input_dir / "source.json").write_text(json.dumps(source_payload, indent=2) + "\n", encoding="utf-8")

    config = {
        "sequence_dir": str(sequence_dir),
        "prepared_dir": str(prepared_dir),
        "source_run_dir": str(source_run_dir),
        "pinhole_size": pinhole_size,
        "pinhole_focal": pinhole_focal,
    }
    RunManifest.create(
        output_run_dir / "manifest.json",
        run_id=output_run_dir.name,
        source_episode=f"adt-rgb-object:{sequence_dir.name}",
        config=config,
        frame_count=int(len(timestamps_s)),
        fps=float(prepared_meta["fps"]),
    )
    return output_run_dir


def extract_rgb_pinhole_gt(
    *,
    sequence_dir: str | Path,
    prepared_dir: str | Path,
    object_uid: int,
    output_dir: str | Path,
    pinhole_size: int = DEFAULT_PINHOLE_SIZE,
    pinhole_focal: float = DEFAULT_PINHOLE_FOCAL,
    overwrite: bool = False,
) -> Path:
    """Extract ADT RGB GT depth/segmentation into the benchmark pinhole frame."""
    import numcodecs
    import zarr

    sequence_dir = Path(sequence_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty GT output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps_s = np.asarray(
        json.loads((prepared_dir / "calibration/timestamps.json").read_text(encoding="utf-8"))["timestamps_s"],
        dtype=np.float64,
    )
    frame_indices = np.asarray(
        json.loads((prepared_dir / "calibration/frame_indices.json").read_text(encoding="utf-8"))["frame_indices"],
        dtype=np.int64,
    )
    data_provider, _calibration, _mps, sensor_data, _sophus = adt_stereo._load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/video.vrs"))
    depth_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/depth_images.vrs"))
    segmentation_provider = data_provider.create_vrs_data_provider(str(sequence_dir / "vrs_files/segmentations.vrs"))
    rgb_calib = video_provider.get_device_calibration().get_camera_calib(RGB_CAMERA_LABEL)
    rgb_shape = tuple(int(value) for value in rgb_calib.get_image_size())
    map_x, map_y, valid = _build_rgb_pinhole_maps(rgb_calib, pinhole_size, pinhole_focal)
    depth_stream = _stream_id_for_shape(depth_provider, rgb_shape)
    segmentation_stream = _stream_id_for_shape(segmentation_provider, rgb_shape)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    group = zarr.open_group(str(output_dir / "adt_gt.zarr"), mode="w")
    depth_store = group.create_dataset(
        "depth_m", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="f4", compressor=compressor
    )
    segmentation_store = group.create_dataset(
        "segmentation", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="i8", compressor=compressor
    )
    mask_store = group.create_dataset(
        "object_mask", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="bool", compressor=compressor
    )
    valid_store = group.create_dataset(
        "valid", shape=(len(timestamps_s), pinhole_size, pinhole_size), chunks=(1, min(256, pinhole_size), min(256, pinhole_size)), dtype="bool", compressor=compressor
    )
    group.create_dataset("frame_indices", data=frame_indices)
    group.create_dataset("timestamps_s", data=timestamps_s)

    for index, timestamp_s in enumerate(timestamps_s):
        timestamp_ns = int(round(float(timestamp_s) * 1e9))
        depth_image, _ = depth_provider.get_image_data_by_time_ns(
            depth_stream, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
        )
        segmentation_image, _ = segmentation_provider.get_image_data_by_time_ns(
            segmentation_stream, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
        )
        raw_depth = _image_array(depth_image).astype(np.float32) / 1000.0
        raw_segmentation = _image_array(segmentation_image).astype(np.int64)
        depth_remap = cv2.remap(
            raw_depth,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        segmentation_remap = cv2.remap(
            raw_segmentation.astype(np.float64),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.int64)
        valid_frame = valid & np.isfinite(depth_remap) & (depth_remap > 0)
        depth_store[index] = depth_remap.astype(np.float32)
        segmentation_store[index] = segmentation_remap
        mask_store[index] = (segmentation_remap == int(object_uid)) & valid_frame
        valid_store[index] = valid_frame

    group.attrs.update(
        {
            "schema_version": "1.0",
            "ground_truth": True,
            "object_uid": int(object_uid),
            "depth_units": "meter",
            "depth_semantics": "ADT RGB camera-Z GT depth remapped into the benchmark RGB pinhole frame",
            "segmentation_semantics": "ADT instance id remapped into the benchmark RGB pinhole frame",
        }
    )
    metadata = {
        "schema_version": "1.0",
        "frame_count": int(len(timestamps_s)),
        "object_uid": int(object_uid),
        "depth_units": "meter",
        "ground_truth": True,
        "pinhole_size": pinhole_size,
        "pinhole_focal": float(pinhole_focal),
        "outputs": ["adt_gt.zarr"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_dir / "metadata.json"


def evaluate_sam3_against_gt(run_dir: str | Path, gt_dir: str | Path) -> dict[str, Any]:
    import zarr

    run_dir = Path(run_dir).resolve()
    gt_dir = Path(gt_dir).resolve()
    masks_path = run_dir / "segmentation/object_masks.npz"
    gt_group = zarr.open_group(str(gt_dir / "adt_gt.zarr"), mode="r")
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)
    with np.load(masks_path, allow_pickle=False) as artifact:
        pred_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        pred_masks = np.asarray(artifact["masks"], dtype=bool)
        pred_valid = np.asarray(artifact["valid"], dtype=bool)
    gt_indices = np.asarray(gt_group["frame_indices"], dtype=np.int64)
    gt_masks = np.asarray(gt_group["object_mask"], dtype=bool)
    gt_valid = np.asarray(gt_group["valid"], dtype=bool)
    gt_lookup = {int(frame): index for index, frame in enumerate(gt_indices)}
    selected = []
    for index, frame in enumerate(pred_indices):
        if int(frame) not in gt_lookup:
            continue
        gt_index = gt_lookup[int(frame)]
        pred = pred_masks[index] & bool(pred_valid[index])
        gt = gt_masks[gt_index] & gt_valid[gt_index]
        union = np.logical_or(pred, gt).sum()
        iou = float(np.logical_and(pred, gt).sum() / union) if union else 0.0
        selected.append((index, gt_index, iou))
    ious = np.asarray([item[2] for item in selected], dtype=np.float64)
    return {
        "matched_frames": int(len(selected)),
        "valid_rate": float(np.mean([bool(pred_valid[item[0]]) for item in selected])) if selected else float("nan"),
        "mean_iou": float(ious.mean()) if ious.size else float("nan"),
        "median_iou": float(np.median(ious)) if ious.size else float("nan"),
        "p05_iou": float(np.percentile(ious, 5)) if ious.size else float("nan"),
        "p95_iou": float(np.percentile(ious, 95)) if ious.size else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--object-keyword", action="append", dest="object_keywords", required=True)
    parser.add_argument("--pinhole-size", type=int, default=DEFAULT_PINHOLE_SIZE)
    parser.add_argument("--pinhole-focal", type=float, default=DEFAULT_PINHOLE_FOCAL)
    parser.add_argument("--gt-output-dir", type=Path)
    parser.add_argument("--object-uid", type=int)
    parser.add_argument("--evaluate-sam3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = prepare_object_run(
        sequence_dir=args.sequence_dir,
        prepared_dir=args.prepared_dir,
        source_run_dir=args.source_run_dir,
        output_run_dir=args.output_run_dir,
        object_keyword_candidates=args.object_keywords,
        pinhole_size=args.pinhole_size,
        pinhole_focal=args.pinhole_focal,
        overwrite=args.overwrite,
    )
    print(run_dir)
    if args.gt_output_dir is not None and args.object_uid is not None:
        extract_rgb_pinhole_gt(
            sequence_dir=args.sequence_dir,
            prepared_dir=args.prepared_dir,
            object_uid=args.object_uid,
            output_dir=args.gt_output_dir,
            pinhole_size=args.pinhole_size,
            pinhole_focal=args.pinhole_focal,
            overwrite=args.overwrite,
        )
    if args.evaluate_sam3:
        if args.gt_output_dir is None or args.object_uid is None:
            parser.error("--evaluate-sam3 requires --gt-output-dir and --object-uid")
        print(json.dumps(evaluate_sam3_against_gt(run_dir, args.gt_output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
