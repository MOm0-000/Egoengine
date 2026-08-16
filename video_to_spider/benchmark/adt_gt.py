"""Read Aria Digital Twin ground-truth depth/segmentation for rectified stereo.

This module is intentionally independent from FoundationStereo.  It converts
ADT camera-Z depth and instance segmentation from the original SLAM fisheye
camera into the same shared rectified frame used by
``ingest/adt_stereo.py``.  No predicted depth or model output is involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..ingest import adt_stereo


DEFAULT_LEFT_CAMERA_LABEL = adt_stereo.DEFAULT_LEFT_CAMERA_LABEL
DEFAULT_RIGHT_CAMERA_LABEL = adt_stereo.DEFAULT_RIGHT_CAMERA_LABEL


def _load_projectaria():
    return adt_stereo._load_projectaria()


def _as_image_array(image: Any) -> np.ndarray:
    array = np.asarray(image.to_numpy_array())
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a monochrome image, got shape {array.shape}")
    return array


@dataclass(frozen=True)
class RectificationMaps:
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    left_valid: np.ndarray
    device_to_rectified_rotation: np.ndarray
    left_device_from_camera: np.ndarray
    left_source_rays: np.ndarray
    source_width: int
    source_height: int


def build_left_rectification_maps(
    video_vrs: str | Path,
    *,
    left_camera_label: str = DEFAULT_LEFT_CAMERA_LABEL,
    right_camera_label: str = DEFAULT_RIGHT_CAMERA_LABEL,
    rectified_width: int = adt_stereo.DEFAULT_RECTIFIED_WIDTH,
    rectified_height: int = adt_stereo.DEFAULT_RECTIFIED_HEIGHT,
    focal_length: float = adt_stereo.DEFAULT_RECTIFIED_FOCAL_LENGTH,
) -> RectificationMaps:
    """Build the exact left-camera remap maps used by ADT stereo P0."""
    data_provider, _calibration, _mps, _sensor_data, _sophus = _load_projectaria()
    video_vrs = Path(video_vrs).resolve()
    provider = data_provider.create_vrs_data_provider(str(video_vrs))
    device_calib = provider.get_device_calibration()
    if device_calib is None:
        raise RuntimeError("VRS does not contain device calibration")
    left_calib = device_calib.get_camera_calib(left_camera_label)
    right_calib = device_calib.get_camera_calib(right_camera_label)
    T_device_left = adt_stereo._se3_to_matrix(left_calib.get_transform_device_camera())
    T_device_right = adt_stereo._se3_to_matrix(right_calib.get_transform_device_camera())
    rectified_frame = adt_stereo._build_rectified_frame(T_device_left, T_device_right)
    map_x, map_y, valid = adt_stereo._build_rectification_maps(
        left_calib,
        T_device_left,
        rectified_frame,
        rectified_width,
        rectified_height,
        float(focal_length),
    )
    source_size = np.asarray(left_calib.get_image_size(), dtype=np.int64)
    cx = float(rectified_width) / 2.0
    cy = float(rectified_height) / 2.0
    u, v = np.meshgrid(
        np.arange(rectified_width, dtype=np.float64),
        np.arange(rectified_height, dtype=np.float64),
    )
    x = (u - cx) / float(focal_length)
    y = (v - cy) / float(focal_length)
    rectified_rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    device_rays = np.einsum("ij,hwj->hwi", rectified_frame, rectified_rays)
    left_camera_from_device = T_device_left[:3, :3].T
    left_source_rays = np.einsum(
        "ij,hwj->hwi", left_camera_from_device, device_rays
    )
    return RectificationMaps(
        left_map_x=map_x,
        left_map_y=map_y,
        left_valid=valid,
        device_to_rectified_rotation=rectified_frame,
        left_device_from_camera=T_device_left,
        left_source_rays=left_source_rays,
        source_width=int(source_size[0]),
        source_height=int(source_size[1]),
    )


def load_prepared_frame_indices(prepared_dir: str | Path) -> tuple[list[int], list[float]]:
    prepared_dir = Path(prepared_dir)
    indices = json.loads(
        (prepared_dir / "calibration/frame_indices.json").read_text(encoding="utf-8")
    )["frame_indices"]
    timestamps_s = json.loads(
        (prepared_dir / "calibration/timestamps.json").read_text(encoding="utf-8")
    )["timestamps_s"]
    if len(indices) != len(timestamps_s):
        raise ValueError("prepared frame index and timestamp counts differ")
    return [int(index) for index in indices], [float(t) for t in timestamps_s]


@dataclass
class ADTGTFrame:
    frame_offset: int
    source_frame_index: int
    timestamp_s: float
    depth_rect_m: np.ndarray
    segmentation_rect: np.ndarray
    object_mask_rect: np.ndarray
    valid_rect: np.ndarray


def _image_shape(provider: Any, stream_id: Any) -> tuple[int, int]:
    image, _record = provider.get_image_data_by_index(stream_id, 0)
    array = _as_image_array(image)
    return int(array.shape[0]), int(array.shape[1])


def stream_inventory(provider: Any) -> list[dict[str, Any]]:
    """Enumerate image streams with their ids, shapes and sensor types.

    ADT depth/segmentation VRS files do not carry Aria device-model labels, so
    stream selection must be recorded explicitly instead of guessed silently.
    """
    inventory: list[dict[str, Any]] = []
    for stream_id in provider.get_all_streams():
        sensor_type = provider.get_sensor_data_type(stream_id)
        if sensor_type.name != "IMAGE":
            continue
        height, width = _image_shape(provider, stream_id)
        inventory.append(
            {
                "stream_id": str(stream_id),
                "sensor_data_type": sensor_type.name,
                "image_shape": [height, width],
                "frame_count": int(provider.get_num_data(stream_id)),
            }
        )
    return inventory


def find_left_gt_stream_id(gt_provider: Any, video_provider: Any, left_camera_label: str) -> Any:
    """Return the GT stream corresponding to the left SLAM camera.

    ADT depth/segmentation VRS files do not carry Aria device-model labels for
    their image streams.  ADT documents the image-stream order as RGB, SLAM-left,
    SLAM-right; the first image stream whose pixel dimensions match the video
    left SLAM camera is therefore selected as the left GT stream.
    """
    video_stream_id = video_provider.get_stream_id_from_label(left_camera_label)
    video_shape = _image_shape(video_provider, video_stream_id)
    candidates = [
        stream_id
        for stream_id in gt_provider.get_all_streams()
        if gt_provider.get_sensor_data_type(stream_id).name == "IMAGE"
        and _image_shape(gt_provider, stream_id) == video_shape
    ]
    if len(candidates) < 2:
        raise RuntimeError(
            f"expected two ADT GT streams matching video shape {video_shape}, got {len(candidates)}"
        )
    return candidates[0]


def convert_camera_z_to_rectified_z(
    source_depths: np.ndarray,
    source_rays: np.ndarray,
    left_device_from_camera: np.ndarray,
    device_to_rectified_rotation: np.ndarray,
) -> np.ndarray:
    """Convert physical left-SLAM camera-Z depths into rectified-left Z.

    ``source_rays`` are directions in the physical left camera frame (last axis
    length 3).  The returned value is the Z coordinate of the back-projected 3D
    point expressed in the shared rectified left camera frame.
    """
    source_depths = np.asarray(source_depths, dtype=np.float64)
    source_rays = np.asarray(source_rays, dtype=np.float64)
    if source_rays.ndim != 2 or source_rays.shape[-1] != 3:
        raise ValueError(f"source_rays must be Nx3, got {source_rays.shape}")
    if source_depths.shape != source_rays.shape[:1]:
        raise ValueError(f"source_depths/rays shape mismatch: {source_depths.shape} vs {source_rays.shape}")
    ray_z = source_rays[:, 2]
    if np.any(ray_z <= 1e-9):
        raise ValueError("source rays must have positive Z component")
    source_points_camera = source_rays * (source_depths / ray_z)[:, None]
    R_left = np.asarray(left_device_from_camera[:3, :3], dtype=np.float64)
    t_left = np.asarray(left_device_from_camera[:3, 3], dtype=np.float64)
    rectified_rotation = np.asarray(device_to_rectified_rotation, dtype=np.float64)
    source_points_device = np.einsum("ij,nj->ni", R_left, source_points_camera) + t_left
    rectified_points = np.einsum("ij,nj->ni", rectified_rotation.T, source_points_device - t_left)
    return rectified_points[:, 2]


def extract_gt_frame(
    *,
    video_provider: Any,
    depth_provider: Any,
    segmentation_provider: Any,
    maps: RectificationMaps,
    object_uid: int,
    source_frame_index: int,
    frame_offset: int,
    timestamp_s: float,
    depth_stream_id: Any,
    segmentation_stream_id: Any,
    sensor_data: Any = None,
) -> ADTGTFrame:
    if sensor_data is None:
        _, _, _, sensor_data, _ = _load_projectaria()
    video_stream_id = video_provider.get_stream_id_from_label(DEFAULT_LEFT_CAMERA_LABEL)
    video_image, video_record = video_provider.get_image_data_by_index(
        video_stream_id, int(source_frame_index)
    )
    timestamp_ns = int(video_record.capture_timestamp_ns)
    depth_image, _ = depth_provider.get_image_data_by_time_ns(
        depth_stream_id, timestamp_ns, sensor_data.TimeDomain.DEVICE_TIME, sensor_data.TimeQueryOptions.BEFORE
    )
    segmentation_image, _ = segmentation_provider.get_image_data_by_time_ns(
        segmentation_stream_id,
        timestamp_ns,
        sensor_data.TimeDomain.DEVICE_TIME,
        sensor_data.TimeQueryOptions.BEFORE,
    )
    depth_source = _as_image_array(depth_image).astype(np.float32) / 1000.0
    segmentation_source = _as_image_array(segmentation_image).astype(np.int64)
    src_x = np.rint(maps.left_map_x).astype(np.int32)
    src_y = np.rint(maps.left_map_y).astype(np.int32)
    in_bounds = (
        (src_x >= 0)
        & (src_x < maps.source_width)
        & (src_y >= 0)
        & (src_y < maps.source_height)
    )
    valid = maps.left_valid & in_bounds & (maps.left_source_rays[..., 2] > 1e-6)
    depth_rect = np.zeros_like(maps.left_valid, dtype=np.float32)
    segmentation_rect = np.zeros_like(maps.left_valid, dtype=np.int64)
    segmentation_rect[valid] = segmentation_source[src_y[valid], src_x[valid]]

    source_depths = depth_source[src_y[valid], src_x[valid]]
    source_rays = maps.left_source_rays[valid]
    depth_rect[valid] = convert_camera_z_to_rectified_z(
        source_depths,
        source_rays,
        maps.left_device_from_camera,
        maps.device_to_rectified_rotation,
    )
    object_mask_rect = (segmentation_rect == int(object_uid)) & valid
    return ADTGTFrame(
        frame_offset=int(frame_offset),
        source_frame_index=int(source_frame_index),
        timestamp_s=float(timestamp_s),
        depth_rect_m=depth_rect,
        segmentation_rect=segmentation_rect,
        object_mask_rect=object_mask_rect,
        valid_rect=valid,
    )


def write_gt_zarr(
    frames: list[ADTGTFrame],
    output_dir: str | Path,
    *,
    object_uid: int,
    gt_stream_selection: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Store ADT GT depth and segmentation in the rectified reference frame."""
    import numcodecs
    import zarr

    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"GT output exists and overwrite is false: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("cannot write GT output for zero frames")
    height, width = frames[0].depth_rect_m.shape
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    group = zarr.open_group(str(output_dir / "adt_gt.zarr"), mode="w")
    shape = (len(frames), height, width)
    depths = group.create_dataset("depth_m", shape=shape, dtype="f4", chunks=(1, min(256, height), min(256, width)), compressor=compressor)
    segmentations = group.create_dataset("segmentation", shape=shape, dtype="i8", chunks=(1, min(256, height), min(256, width)), compressor=compressor)
    masks = group.create_dataset("object_mask", shape=shape, dtype="bool", chunks=(1, min(256, height), min(256, width)), compressor=compressor)
    valid = group.create_dataset("valid", shape=shape, dtype="bool", chunks=(1, min(256, height), min(256, width)), compressor=compressor)
    group.create_dataset(
        "frame_indices",
        data=np.asarray([f.source_frame_index for f in frames], dtype=np.int64),
    )
    group.create_dataset(
        "timestamps_s",
        data=np.asarray([f.timestamp_s for f in frames], dtype=np.float64),
    )
    for offset, frame in enumerate(frames):
        depths[offset] = frame.depth_rect_m
        segmentations[offset] = frame.segmentation_rect
        masks[offset] = frame.object_mask_rect
        valid[offset] = frame.valid_rect
    group.attrs.update(
        {
            "schema_version": "1.0",
            "ground_truth": True,
            "object_uid": int(object_uid),
            "depth_units": "meter",
            "depth_semantics": "rectified-left camera Z (meters), converted from physical left-SLAM camera-Z GT depth via source-ray back-projection",
            "segmentation_semantics": "ADT instance id sampled into rectified left reference",
        }
    )
    metadata = {
        "schema_version": "1.0",
        "frame_count": len(frames),
        "object_uid": int(object_uid),
        "depth_units": "meter",
        "ground_truth": True,
        "depth_semantics": "rectified-left camera Z (meters)",
        "outputs": ["adt_gt.zarr"],
    }
    if gt_stream_selection is not None:
        group.attrs["gt_stream_selection"] = gt_stream_selection
        metadata["gt_stream_selection"] = gt_stream_selection
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_dir / "metadata.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-vrs", type=Path, required=True)
    parser.add_argument("--depth-vrs", type=Path, required=True)
    parser.add_argument("--segmentation-vrs", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--object-uid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--end-offset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    data_provider, _calibration, _mps, sensor_data, _sophus = _load_projectaria()
    video_provider = data_provider.create_vrs_data_provider(str(args.video_vrs))
    depth_provider = data_provider.create_vrs_data_provider(str(args.depth_vrs))
    segmentation_provider = data_provider.create_vrs_data_provider(str(args.segmentation_vrs))
    maps = build_left_rectification_maps(args.video_vrs)
    depth_stream_id = find_left_gt_stream_id(depth_provider, video_provider, DEFAULT_LEFT_CAMERA_LABEL)
    segmentation_stream_id = find_left_gt_stream_id(segmentation_provider, video_provider, DEFAULT_LEFT_CAMERA_LABEL)
    video_left_stream_id = video_provider.get_stream_id_from_label(DEFAULT_LEFT_CAMERA_LABEL)
    video_left_shape = _image_shape(video_provider, video_left_stream_id)
    gt_stream_selection = {
        "video_left": {
            "label": DEFAULT_LEFT_CAMERA_LABEL,
            "stream_id": str(video_left_stream_id),
            "image_shape": list(video_left_shape),
        },
        "selection_method": "ADT documented image-stream order RGB, SLAM-left, SLAM-right; first GT stream whose shape matches video left SLAM is selected",
        "depth": {
            "selected_stream_id": str(depth_stream_id),
            "inventory": stream_inventory(depth_provider),
        },
        "segmentation": {
            "selected_stream_id": str(segmentation_stream_id),
            "inventory": stream_inventory(segmentation_provider),
        },
    }
    indices, timestamps = load_prepared_frame_indices(args.prepared_dir)
    start = int(args.start_offset)
    end = int(args.end_offset) if args.end_offset is not None else len(indices)
    if start < 0 or end <= start or end > len(indices):
        raise ValueError("invalid GT extraction frame interval")
    frames = []
    for offset in range(start, end):
        frame = extract_gt_frame(
            video_provider=video_provider,
            depth_provider=depth_provider,
            segmentation_provider=segmentation_provider,
            maps=maps,
            object_uid=args.object_uid,
            source_frame_index=indices[offset],
            frame_offset=offset,
            timestamp_s=timestamps[offset],
            depth_stream_id=depth_stream_id,
            segmentation_stream_id=segmentation_stream_id,
            sensor_data=sensor_data,
        )
        frames.append(frame)
    metadata_path = write_gt_zarr(
        frames,
        args.output_dir,
        object_uid=args.object_uid,
        gt_stream_selection=gt_stream_selection,
        overwrite=args.overwrite,
    )
    print(json.dumps({"gt_output": str(args.output_dir), "metadata": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
