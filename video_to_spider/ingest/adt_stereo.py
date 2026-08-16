"""Prepare Aria Digital Twin left/right SLAM streams for ``ingest-stereo``.

This module intentionally keeps the Project Aria Tools dependency lazy and
optional.  It is a thin boundary between the raw ADT VRS and the existing
calibrated-rectified-stereo ingestion path.

The ADT SLAM pair uses Project Aria's ``FISHEYE624`` camera model.  The
installed ``projectaria-tools==1.0.0`` bindings only expose ``sophus.SE3d`` and
do not provide a way to create a linear camera with a non-identity
``T_Device_Camera``.  This adapter therefore performs the rectification itself
with the same official fisheye ``project`` function:

1. Estimate the left/right baseline from factory ``T_Device_Camera``.
2. Build a shared upright rectified camera frame (Bouguet-style) whose X axis
   is the baseline direction and whose optical axis stays close to the left
   camera optical axis.
3. Ray-remap every rectified pixel back through each fisheye camera using the
   official projection, yielding synchronized, row-aligned pinhole images.

This avoids guessing baseline, treating fisheye images as pinhole images, and
avoids depending on newer projectaria-tools APIs that are incompatible with the
project's pinned numpy<2 environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..schemas import SCHEMA_VERSION


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


def _load_projectaria() -> tuple[Any, Any, Any, Any, Any]:
    """Return ``(data_provider, calibration, mps, sensor_data, sophus)``."""
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core import calibration
        from projectaria_tools.core import mps
        from projectaria_tools.core import sensor_data
        from projectaria_tools.core import sophus
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "ADT preparation requires the optional `projectaria-tools` package. "
            "Install it into the `v2s-core` environment with "
            "`python -m pip install projectaria-tools==1.0.0`."
        ) from exc
    return data_provider, calibration, mps, sensor_data, sophus


def _se3_to_matrix(se3: Any) -> np.ndarray:
    # projectaria-tools 1.0.0 exposes sophus.SE3d.matrix(); newer bindings expose
    # sophus.SE3.to_matrix().  Support both so the adapter is easy to upgrade.
    matrix = getattr(se3, "matrix", None)
    if matrix is None:
        matrix = getattr(se3, "to_matrix", None)
    if matrix is None:
        raise TypeError(f"unsupported SE3 object: {type(se3)!r}")
    return np.asarray(matrix(), dtype=np.float64).reshape(4, 4)


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


def _build_rectified_frame(device_left: np.ndarray, device_right: np.ndarray) -> np.ndarray:
    """Return a shared device-to-rectified-camera rotation.

    Columns of the returned matrix are the rectified camera's X/Y/Z axes
    expressed in the device frame.  X is the baseline direction; Z is kept as
    close as possible to the left camera's optical axis while maintaining a
    right-handed frame.
    """
    R_left = device_left[:3, :3]
    t_left = device_left[:3, 3]
    R_right = device_right[:3, :3]
    t_right = device_right[:3, 3]
    baseline = t_right - t_left
    baseline_norm = float(np.linalg.norm(baseline))
    if not np.isfinite(baseline_norm) or baseline_norm <= 1e-9:
        raise ValueError("degenerate left/right baseline")
    x_axis = baseline / baseline_norm
    left_z = R_left[:, 2]
    y_axis = np.cross(x_axis, left_z)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.cross(x_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    if np.dot(z_axis, left_z) < 0:
        z_axis = -z_axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _build_rectification_maps(
    src_calib: Any,
    device_from_camera: np.ndarray,
    rectified_frame: np.ndarray,
    rectified_width: int,
    rectified_height: int,
    focal_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``cv2.remap`` maps from rectified pixels to fisheye pixels.

    Returns ``(map_x, map_y, valid)``.  ``valid`` is True where the rectified
    pixel maps inside the source fisheye image.
    """
    if rectified_width <= 0 or rectified_height <= 0 or focal_length <= 0:
        raise ValueError("rectified image dimensions and focal_length must be positive")
    source_size = np.asarray(src_calib.get_image_size(), dtype=np.int64)
    if source_size.size < 2 or np.any(source_size <= 0):
        raise RuntimeError(f"invalid source image size: {source_size}")
    src_width, src_height = int(source_size[0]), int(source_size[1])
    camera_from_device = device_from_camera[:3, :3].T
    cx = float(rectified_width) / 2.0
    cy = float(rectified_height) / 2.0
    u, v = np.meshgrid(
        np.arange(rectified_width, dtype=np.float64),
        np.arange(rectified_height, dtype=np.float64),
    )
    x = (u - cx) / focal_length
    y = (v - cy) / focal_length
    rectified_rays = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).T
    device_rays = rectified_frame @ rectified_rays
    source_rays = (camera_from_device @ device_rays).T.reshape(rectified_height, rectified_width, 3)
    valid = source_rays[..., 2] > 1e-6
    map_x = np.full((rectified_height, rectified_width), -1.0, dtype=np.float32)
    map_y = np.full((rectified_height, rectified_width), -1.0, dtype=np.float32)
    points = source_rays[valid].astype(np.float64)
    projected = [src_calib.project(point) for point in points]
    valid_rows, valid_cols = np.where(valid)
    for row, col, source_pixel in zip(valid_rows, valid_cols, projected):
        if source_pixel is None or not np.all(np.isfinite(source_pixel)):
            continue
        px, py = float(source_pixel[0]), float(source_pixel[1])
        if 0.0 <= px < src_width and 0.0 <= py < src_height:
            map_x[row, col] = px
            map_y[row, col] = py
    valid = (map_x >= 0.0) & (map_y >= 0.0)
    return map_x, map_y, valid


def _rectify_image(
    image: Any,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    source = _squeeze_mono_image(image)
    rectified = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rectified.astype(np.uint8)


def _nearest_mps_pose_index(mps_timestamps_ns: np.ndarray, query_ns: int) -> int:
    if mps_timestamps_ns.size == 0:
        raise ValueError("MPS trajectory is empty")
    index = int(np.searchsorted(mps_timestamps_ns, query_ns, side="left"))
    candidates = [max(0, index - 1), min(mps_timestamps_ns.size - 1, index)]
    return int(min(candidates, key=lambda item: abs(int(mps_timestamps_ns[item]) - query_ns)))


def _rectification_self_consistency(
    *,
    left_calib: Any,
    right_calib: Any,
    left_device_from_camera: np.ndarray,
    right_device_from_camera: np.ndarray,
    rectified_frame: np.ndarray,
    rectified_width: int,
    rectified_height: int,
    focal_length: float,
    sample_count: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Calibration-only geometric audit of the rectified stereo pair.

    This is not a scene-based GT audit.  It randomly samples 3D points in front
    of the left rectified camera, projects them through both fisheye cameras,
    and then measures the rectified vertical reprojection error and horizontal
    disparity.  Because the two rectified cameras share the same orientation and
    their baseline is parallel to the rectified X axis, a correct implementation
    should produce vertical errors at floating-point round-off level and
    positive horizontal disparity for every visible point.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    left_camera_from_device = left_device_from_camera[:3, :3].T
    right_camera_from_device = right_device_from_camera[:3, :3].T
    t_left = left_device_from_camera[:3, 3]
    t_right = right_device_from_camera[:3, 3]
    cx = float(rectified_width) / 2.0
    cy = float(rectified_height) / 2.0
    rng = np.random.default_rng(seed)
    vertical_px: list[float] = []
    disparities_px: list[float] = []
    visible = 0
    for _ in range(sample_count):
        u = float(rng.uniform(0, rectified_width - 1))
        v = float(rng.uniform(0, rectified_height - 1))
        depth = float(rng.uniform(0.5, 8.0))
        x = (u - cx) / focal_length
        y = (v - cy) / focal_length
        left_rect_point = np.array([x * depth, y * depth, depth], dtype=np.float64)
        device_point = rectified_frame @ left_rect_point + t_left
        left_pixel = left_calib.project(left_camera_from_device @ (device_point - t_left))
        right_pixel = right_calib.project(right_camera_from_device @ (device_point - t_right))
        if left_pixel is None or right_pixel is None:
            continue
        if not (np.all(np.isfinite(left_pixel)) and np.all(np.isfinite(right_pixel))):
            continue
        # Recover rectified normalized coordinates from each source pixel, so
        # this audit exercises the same unproject/project round trip used by
        # the remap maps.
        left_ray = left_calib.unproject(np.asarray(left_pixel, dtype=np.float64))
        right_ray = right_calib.unproject(np.asarray(right_pixel, dtype=np.float64))
        if left_ray is None or right_ray is None:
            continue
        left_rect_ray = rectified_frame.T @ (left_device_from_camera[:3, :3] @ left_ray)
        right_rect_ray = rectified_frame.T @ (right_device_from_camera[:3, :3] @ right_ray)
        if left_rect_ray[2] <= 1e-12 or right_rect_ray[2] <= 1e-12:
            continue
        left_v = cy + focal_length * left_rect_ray[1] / left_rect_ray[2]
        right_v = cy + focal_length * right_rect_ray[1] / right_rect_ray[2]
        left_u = cx + focal_length * left_rect_ray[0] / left_rect_ray[2]
        right_u = cx + focal_length * right_rect_ray[0] / right_rect_ray[2]
        vertical_px.append(abs(float(left_v - right_v)))
        disparities_px.append(float(left_u - right_u))
        visible += 1
    vertical_array = np.asarray(vertical_px, dtype=np.float64)
    disparity_array = np.asarray(disparities_px, dtype=np.float64)
    vertical_median = float(np.median(vertical_array)) if vertical_array.size else float("inf")
    vertical_p95 = float(np.percentile(vertical_array, 95)) if vertical_array.size else float("inf")
    positive_ratio = float(np.mean(disparity_array > 0)) if disparity_array.size else 0.0
    median_disparity = float(np.median(disparity_array)) if disparity_array.size else float("nan")
    checks = {
        "sufficient_feature_correspondences": bool(visible >= 40),
        "vertical_median_within_limit": bool(vertical_median <= 1.0),
        "vertical_p95_within_limit": bool(vertical_p95 <= 1.5),
        "left_right_order_positive_disparity": bool(positive_ratio >= 0.60 and median_disparity > 0),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "calibration self-consistency with official fisheye project/unproject",
        "sampled_visible_points": int(visible),
        "vertical_disparity_abs_median_px": vertical_median,
        "vertical_disparity_abs_p95_px": vertical_p95,
        "vertical_disparity_abs_max_px": float(vertical_array.max()) if vertical_array.size else float("inf"),
        "horizontal_disparity_median_px": median_disparity,
        "positive_horizontal_disparity_ratio": positive_ratio,
        "checks": checks,
        "accepted": bool(all(checks.values())),
        "image_height": int(rectified_height),
        "image_width": int(rectified_width),
    }


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

    data_provider, _calibration, mps, sensor_data, _sophus = _load_projectaria()
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

    rectified_frame = _build_rectified_frame(T_device_left, T_device_right)
    left_map_x, left_map_y, left_valid = _build_rectification_maps(
        left_src_calib, T_device_left, rectified_frame,
        rectified_width, rectified_height, float(focal_length),
    )
    right_map_x, right_map_y, right_valid = _build_rectification_maps(
        right_src_calib, T_device_right, rectified_frame,
        rectified_width, rectified_height, float(focal_length),
    )
    if not left_valid.any() or not right_valid.any():
        raise RuntimeError("rectification produced no valid pixels")

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

    sync_errors_ns: list[int] = []
    absolute_timestamps_s: list[float] = []
    kept_source_indices: list[int] = []

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

        left_rectified = _rectify_image(left_data, left_map_x, left_map_y)
        right_rectified = _rectify_image(right_data, right_map_x, right_map_y)
        left_path = left_dir / f"{target_index:06d}.png"
        right_path = right_dir / f"{target_index:06d}.png"
        if not cv2.imwrite(str(left_path), left_rectified):
            raise RuntimeError(f"failed to write {left_path}")
        if not cv2.imwrite(str(right_path), right_rectified):
            raise RuntimeError(f"failed to write {right_path}")

        absolute_timestamps_s.append(float(left_timestamp_ns) / 1e9)
        kept_source_indices.append(source_index_int)

    common_valid = np.logical_and(left_valid, right_valid).astype(np.uint8)
    K_rect = _linear_intrinsics(rectified_width, rectified_height, float(focal_length))
    intrinsics_path = calibration_dir / "K_rect.npy"
    right_intrinsics_path = calibration_dir / "K_rect_right.npy"
    common_valid_mask_path = calibration_dir / "stereo_common_valid.npy"
    timestamps_path = calibration_dir / "timestamps.json"
    frame_indices_path = calibration_dir / "frame_indices.json"
    np.save(intrinsics_path, K_rect.astype(np.float32))
    np.save(right_intrinsics_path, K_rect.astype(np.float32))
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
            # The prepared reference camera is the rectified left camera.  Its
            # device-to-camera rotation is rectified_frame.
            T_device_rect_left = np.eye(4, dtype=np.float64)
            T_device_rect_left[:3, :3] = rectified_frame
            T_world_left.append(T_world_device @ T_device_rect_left)
        T_world_left_array = np.asarray(T_world_left, dtype=np.float64).reshape(len(T_world_left), 4, 4)
        camera_poses_path = calibration_dir / "T_world_camera.npy"
        np.save(camera_poses_path, T_world_left_array.astype(np.float32))
    elif not static_camera:
        raise ValueError(
            "moving ADT recordings require --closed-loop-trajectory; use --static-camera "
            "only for a rigid camera"
        )

    geometry = _rectification_self_consistency(
        left_calib=left_src_calib,
        right_calib=right_src_calib,
        left_device_from_camera=T_device_left,
        right_device_from_camera=T_device_right,
        rectified_frame=rectified_frame,
        rectified_width=rectified_width,
        rectified_height=rectified_height,
        focal_length=float(focal_length),
    )

    sync_errors_array = np.asarray(sync_errors_ns, dtype=np.float64)
    metadata = {
        "source_vrs": str(vrs_path),
        "left_camera_label": left_camera_label,
        "right_camera_label": right_camera_label,
        "rectification": {
            "method": "official Aria FISHEYE624 project/unproject to shared upright rectified frame",
            "rectified_width": int(rectified_width),
            "rectified_height": int(rectified_height),
            "focal_length_px": float(focal_length),
            "K_rect_left": K_rect.tolist(),
            "K_rect_right": K_rect.tolist(),
            "baseline_m": baseline_m,
            "device_to_rectified_rotation": rectified_frame.tolist(),
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
        "rectification_geometry_audit": geometry,
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
