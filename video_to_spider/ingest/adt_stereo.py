"""Prepare Aria Digital Twin left/right SLAM streams for ``ingest-stereo``.

This module intentionally keeps the Project Aria Tools dependency lazy and
optional.  It is a thin boundary between the raw ADT VRS and the existing
calibrated-rectified-stereo ingestion path: it reads the official factory
calibration embedded in the VRS, uses that calibration to create a shared
upright linear target camera model for the left and right SLAM streams, and
writes exactly the files expected by :mod:`video_to_spider.ingest.stereo`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_LEFT_CAMERA_LABEL = "camera-slam-left"
DEFAULT_RIGHT_CAMERA_LABEL = "camera-slam-right"
DEFAULT_RECTIFIED_WIDTH = 512
DEFAULT_RECTIFIED_HEIGHT = 512
DEFAULT_RECTIFIED_FOCAL_LENGTH = 150.0
DEFAULT_MAX_SYNC_NS = 5_000_000


@dataclass(frozen=True)
class PreparedAdtStereo:
    """Files produced by :func:`prepare_adt_stereo`.

    ``left_dir``, ``right_dir`` and the calibration files can be passed
    directly to ``video_to_spider.cli ingest-stereo`` without further
    processing.
    """

    output_dir: Path
    left_dir: Path
    right_dir: Path
    intrinsics_path: Path
    right_intrinsics_path: Path
    common_valid_mask_path: Path
    timestamps_path: Path
    frame_indices_path: Path
    camera_poses_path: Path | None
    baseline_m: float
    fps: float
    metadata_path: Path


def _load_projectaria() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return ``(data_provider, calibration, image, sensor_data, mps, sophus)``."""
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core import calibration
        from projectaria_tools.core import image
        from projectaria_tools.core import mps
        from projectaria_tools.core import sensor_data
        from projectaria_tools.core import sophus
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "ADT preparation requires the optional `projectaria-tools` package. "
            "Install it into the `v2s-core` environment with "
            "`python -m pip install projectaria-tools==1.0.0`."
        ) from exc
    return data_provider, calibration, image, sensor_data, mps, sophus


def _se3_to_matrix(se3: Any) -> np.ndarray:
    return np.asarray(se3.to_matrix(), dtype=np.float64).reshape(4, 4)


def _linear_intrinsics(width: int, height: int, focal_length: float) -> np.ndarray:
    if width <= 0 or height <= 0 or focal_length <= 0:
        raise ValueError("width, height and focal_length must be positive")
    return np.asarray(
        [
            [focal_length, 0.0, float(width) / 2.0],
            [0.0, focal_length, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _squeeze_mono_image(image: Any) -> np.ndarray:
    array = np.asarray(image.to_numpy_array())
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a monochrome image, got shape {array.shape}")
    return array


def _rectify(
    image: Any,
    *,
    src_calib: Any,
    dst_calib: Any,
    interpolation: Any,
) -> np.ndarray:
    array = _squeeze_mono_image(image)
    calibration = _load_projectaria()[1]
    rectified = calibration.distort_by_calibration(
        array,
        dst_calib,
        src_calib,
        interpolation,
    )
    return np.clip(np.asarray(rectified), 0, 255).astype(np.uint8)


def _compute_common_valid_mask(
    *,
    left_image: Any,
    right_image: Any,
    left_src_calib: Any,
    right_src_calib: Any,
    left_dst_calib: Any,
    right_dst_calib: Any,
    calibration: Any,
    interpolation: Any,
) -> np.ndarray:
    """Estimate the true common valid domain by rectifying all-white images."""
    left_src = _squeeze_mono_image(left_image)
    right_src = _squeeze_mono_image(right_image)
    left_white = np.full(left_src.shape, 255.0, dtype=np.float32)
    right_white = np.full(right_src.shape, 255.0, dtype=np.float32)
    left_valid = np.asarray(
        calibration.distort_by_calibration(left_white, left_dst_calib, left_src_calib, interpolation)
    ) > 1e-6
    right_valid = np.asarray(
        calibration.distort_by_calibration(right_white, right_dst_calib, right_src_calib, interpolation)
    ) > 1e-6
    if left_valid.shape != right_valid.shape:
        raise ValueError(
            "left and right rectified outputs have different shapes: "
            f"{left_valid.shape} vs {right_valid.shape}"
        )
    return np.logical_and(left_valid, right_valid).astype(np.uint8)


def _nearest_mps_pose_index(mps_timestamps_ns: np.ndarray, query_ns: int) -> int:
    if mps_timestamps_ns.size == 0:
        raise ValueError("MPS trajectory is empty")
    index = int(np.searchsorted(mps_timestamps_ns, query_ns, side="left"))
    candidates = [max(0, index - 1), min(mps_timestamps_ns.size - 1, index)]
    return int(min(candidates, key=lambda item: abs(int(mps_timestamps_ns[item]) - query_ns)))


def prepare_adt_stereo(
    *,
    video_vrs: str | Path,
    output_dir: str | Path,
    left_camera_label: str = DEFAULT_LEFT_CAMERA_LABEL,
    right_camera_label: str = DEFAULT_RIGHT_CAMERA_LABEL,
    rectified_width: int = DEFAULT_RECTIFIED_WIDTH,
    rectified_height: int = DEFAULT_RECTIFIED_HEIGHT,
    focal_length: float = DEFAULT_RECTIFIED_FOCAL_LENGTH,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_sync_ns: int = DEFAULT_MAX_SYNC_NS,
    closed_loop_trajectory_path: str | Path | None = None,
    static_camera: bool = False,
) -> PreparedAdtStereo:
    """Read, synchronize and rectify ADT left/right SLAM images.

    This function does not call ``ingest-stereo``.  The caller can inspect the
    generated files first, then pass them to the existing CLI.  Camera poses are
    only written when ``closed_loop_trajectory_path`` is supplied and
    ``static_camera`` is false.
    """

    data_provider, calibration, image, sensor_data, mps, sophus = _load_projectaria()
    vrs_path = Path(video_vrs).resolve()
    if not vrs_path.is_file():
        raise FileNotFoundError(vrs_path)
    if closed_loop_trajectory_path is not None and not Path(closed_loop_trajectory_path).is_file():
        raise FileNotFoundError(closed_loop_trajectory_path)
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("require 0 <= start_frame < end_frame")
    if not np.isfinite(focal_length) or focal_length <= 0:
        raise ValueError("focal_length must be positive")

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    device_calib = provider.get_device_calibration()
    if device_calib is None:
        raise RuntimeError("VRS does not contain device calibration")

    left_src_calib = device_calib.get_camera_calib(left_camera_label)
    right_src_calib = device_calib.get_camera_calib(right_camera_label)
    left_stream_id = provider.get_stream_id_from_label(left_camera_label)
    right_stream_id = provider.get_stream_id_from_label(right_camera_label)
    if provider.get_num_data(left_stream_id) <= 0:
        raise RuntimeError(f"no data for left stream {left_camera_label}")
    if provider.get_num_data(right_stream_id) <= 0:
        raise RuntimeError(f"no data for right stream {right_camera_label}")

    T_device_left = _se3_to_matrix(left_src_calib.get_transform_device_camera())
    T_device_right = _se3_to_matrix(right_src_calib.get_transform_device_camera())
    T_left_right = np.linalg.inv(T_device_left) @ T_device_right
    baseline_m = float(np.linalg.norm(T_left_right[:3, 3]))
    if not np.isfinite(baseline_m) or not 0.01 <= baseline_m <= 0.30:
        raise ValueError(
            f"invalid ADT left/right baseline {baseline_m:.6f} m; expected [0.01, 0.30] m"
        )

    target_left_device = np.eye(4, dtype=np.float64)
    target_right_device = np.eye(4, dtype=np.float64)
    target_right_device[0, 3] = baseline_m
    left_dst_calib = calibration.get_linear_camera_calibration(
        rectified_width,
        rectified_height,
        float(focal_length),
        left_camera_label,
        sophus.SE3.from_matrix(target_left_device),
    )
    right_dst_calib = calibration.get_linear_camera_calibration(
        rectified_width,
        rectified_height,
        float(focal_length),
        right_camera_label,
        sophus.SE3.from_matrix(target_right_device),
    )

    left_timestamps_ns = np.asarray(
        provider.get_timestamps_ns(left_stream_id, sensor_data.TimeDomain.DEVICE_TIME),
        dtype=np.int64,
    )
    if left_timestamps_ns.ndim != 1 or left_timestamps_ns.size == 0:
        raise RuntimeError("left SLAM stream has no device timestamps")
    if len(np.unique(left_timestamps_ns)) != left_timestamps_ns.size:
        raise RuntimeError("left SLAM timestamps are not unique")
    if np.any(np.diff(left_timestamps_ns) <= 0):
        raise RuntimeError("left SLAM timestamps are not strictly increasing")
    source_indices = np.arange(start_frame, end_frame if end_frame is not None else left_timestamps_ns.size)
    if source_indices.size == 0:
        raise ValueError("selected frame interval is empty")
    if int(source_indices[-1]) >= left_timestamps_ns.size:
        raise ValueError("end_frame exceeds available left SLAM frames")

    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    left_dir = output_root / "rectified" / "left"
    right_dir = output_root / "rectified" / "right"
    calibration_dir = output_root / "calibration"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    interpolation = image.InterpolationMethod.NEAREST_NEIGHBOR
    sync_errors_ns: list[int] = []
    absolute_timestamps_s: list[float] = []
    kept_source_indices: list[int] = []
    first_left = first_right = None
    common_valid: np.ndarray | None = None

    for target_index, source_index in enumerate(source_indices):
        source_index_int = int(source_index)
        left_data, left_record = provider.get_image_data_by_index(left_stream_id, source_index_int)
        if left_data is None:
            raise RuntimeError(f"failed to read left image at source index {source_index_int}")
        left_timestamp_ns = int(left_record.capture_timestamp_ns)
        right_data, right_record = provider.get_image_data_by_time_ns(
            right_stream_id,
            left_timestamp_ns,
            sensor_data.TimeDomain.DEVICE_TIME,
            sensor_data.TimeQueryOptions.CLOSEST,
        )
        if right_data is None:
            raise RuntimeError(f"no right image within query range for left timestamp {left_timestamp_ns}")
        right_timestamp_ns = int(right_record.capture_timestamp_ns)
        sync_error_ns = abs(right_timestamp_ns - left_timestamp_ns)
        if sync_error_ns > max_sync_ns:
            raise RuntimeError(
                f"left/right sync error {sync_error_ns} ns exceeds "
                f"allowed maximum {max_sync_ns} ns at source index {source_index_int}"
            )
        sync_errors_ns.append(sync_error_ns)

        left_rectified = _rectify(
            left_data,
            src_calib=left_src_calib,
            dst_calib=left_dst_calib,
            interpolation=interpolation,
        )
        right_rectified = _rectify(
            right_data,
            src_calib=right_src_calib,
            dst_calib=right_dst_calib,
            interpolation=interpolation,
        )
        left_path = left_dir / f"{target_index:06d}.png"
        right_path = right_dir / f"{target_index:06d}.png"
        if not cv2.imwrite(str(left_path), left_rectified):
            raise RuntimeError(f"failed to write {left_path}")
        if not cv2.imwrite(str(right_path), right_rectified):
            raise RuntimeError(f"failed to write {right_path}")

        if first_left is None:
            first_left = left_data
            first_right = right_data
        absolute_timestamps_s.append(float(left_timestamp_ns) / 1e9)
        kept_source_indices.append(source_index_int)

    if first_left is None or first_right is None:
        raise RuntimeError("no stereo frames were written")
    common_valid = _compute_common_valid_mask(
        left_image=first_left,
        right_image=first_right,
        left_src_calib=left_src_calib,
        right_src_calib=right_src_calib,
        left_dst_calib=left_dst_calib,
        right_dst_calib=right_dst_calib,
        calibration=calibration,
        interpolation=interpolation,
    )

    K_left = _linear_intrinsics(rectified_width, rectified_height, focal_length)
    K_right = K_left.copy()
    intrinsics_path = calibration_dir / "K_rect.npy"
    right_intrinsics_path = calibration_dir / "K_rect_right.npy"
    common_valid_mask_path = calibration_dir / "stereo_common_valid.npy"
    timestamps_path = calibration_dir / "timestamps.json"
    frame_indices_path = calibration_dir / "frame_indices.json"
    np.save(intrinsics_path, K_left.astype(np.float32))
    np.save(right_intrinsics_path, K_right.astype(np.float32))
    np.save(common_valid_mask_path, common_valid)
    _write_json(timestamps_path, {"timestamps_s": absolute_timestamps_s})
    _write_json(frame_indices_path, {"frame_indices": kept_source_indices})

    nominal_rate = float("nan")
    for method_name in ("get_nominal_rate_hz", "get_nominalRateHz"):
        method = getattr(provider, method_name, None)
        if method is None:
            continue
        try:
            nominal_rate = float(method(left_stream_id))
            break
        except Exception:
            continue
    if np.isfinite(nominal_rate) and nominal_rate > 0:
        fps = float(nominal_rate)
    elif len(absolute_timestamps_s) > 1:
        diffs = np.diff(np.asarray(absolute_timestamps_s, dtype=np.float64))
        fps = float(1.0 / max(float(np.median(diffs)), 1e-9))
    else:
        raise RuntimeError("cannot infer fps from a single frame; use a longer interval")

    camera_poses_path: Path | None = None
    if not static_camera and closed_loop_trajectory_path is not None:
        trajectory = mps.read_closed_loop_trajectory(str(Path(closed_loop_trajectory_path).resolve()))
        mps_timestamps_ns = np.asarray(
            [int(pose.tracking_timestamp.total_seconds() * 1e9) for pose in trajectory],
            dtype=np.int64,
        )
        if mps_timestamps_ns.size == 0:
            raise ValueError("closed-loop trajectory contains no poses")
        order = np.argsort(mps_timestamps_ns)
        mps_timestamps_ns = mps_timestamps_ns[order]
        ordered_trajectory = [trajectory[i] for i in order]
        T_world_left = []
        for source_index in kept_source_indices:
            timestamp_ns = int(left_timestamps_ns[source_index])
            nearest_index = _nearest_mps_pose_index(mps_timestamps_ns, timestamp_ns)
            pose = ordered_trajectory[nearest_index]
            if abs(int(mps_timestamps_ns[nearest_index]) - timestamp_ns) > max_sync_ns:
                raise RuntimeError(
                    "closed-loop trajectory has no pose within sync tolerance for "
                    f"left timestamp {timestamp_ns}"
                )
            T_world_device = _se3_to_matrix(pose.transform_world_device)
            T_world_left.append(T_world_device @ target_left_device)
        T_world_left_array = np.asarray(T_world_left, dtype=np.float64).reshape(len(T_world_left), 4, 4)
        camera_poses_path = calibration_dir / "T_world_camera.npy"
        np.save(camera_poses_path, T_world_left_array.astype(np.float32))
    elif not static_camera:
        raise ValueError(
            "moving ADT recordings require --closed-loop-trajectory; use --static-camera "
            "only for a rigid camera"
        )

    sync_errors_array = np.asarray(sync_errors_ns, dtype=np.float64)
    metadata = {
        "source_vrs": str(vrs_path),
        "left_camera_label": left_camera_label,
        "right_camera_label": right_camera_label,
        "rectification": {
            "method": "official Aria factory calibration to shared upright linear stereo",
            "rectified_width": int(rectified_width),
            "rectified_height": int(rectified_height),
            "focal_length_px": float(focal_length),
            "K_rect_left": K_left.tolist(),
            "K_rect_right": K_right.tolist(),
            "baseline_m": baseline_m,
        },
        "frame_count": len(kept_source_indices),
        "fps": float(fps),
        "selected_source_indices": kept_source_indices,
        "timestamp_sync": {
            "max_abs_ns": int(sync_errors_array.max()) if sync_errors_array.size else 0,
            "p95_abs_ns": float(np.percentile(sync_errors_array, 95)) if sync_errors_array.size else 0.0,
            "accepted": bool(sync_errors_array.size and sync_errors_array.max() <= max_sync_ns),
        },
        "common_valid_pixel_ratio": float(np.mean(common_valid)),
        "static_camera": bool(static_camera),
        "camera_poses_source": (
            str(Path(closed_loop_trajectory_path).resolve())
            if camera_poses_path is not None
            else "none_static_camera"
        ),
    }
    metadata_path = output_root / "adt_stereo_prepare.json"
    _write_json(metadata_path, metadata)

    return PreparedAdtStereo(
        output_dir=output_root,
        left_dir=left_dir,
        right_dir=right_dir,
        intrinsics_path=intrinsics_path,
        right_intrinsics_path=right_intrinsics_path,
        common_valid_mask_path=common_valid_mask_path,
        timestamps_path=timestamps_path,
        frame_indices_path=frame_indices_path,
        camera_poses_path=camera_poses_path,
        baseline_m=baseline_m,
        fps=fps,
        metadata_path=metadata_path,
    )
