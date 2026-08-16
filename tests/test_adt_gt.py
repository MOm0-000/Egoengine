from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_to_spider.benchmark import adt_gt


class _SensorType:
    def __init__(self, name: str) -> None:
        self.name = name


class _Image:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy_array(self) -> np.ndarray:
        return self.array


class _Record:
    capture_timestamp_ns = 0


class _GTProvider:
    """Fake ADT GT provider with RGB + SLAM-left + SLAM-right image streams."""

    def __init__(self, shapes: dict[str, tuple[int, int]]) -> None:
        self.shapes = shapes
        self.ids = list(shapes.keys())

    def get_all_streams(self) -> list[str]:
        return list(self.ids)

    def get_sensor_data_type(self, stream_id: str) -> _SensorType:
        return _SensorType("IMAGE" if stream_id in self.shapes else "NOT_IMAGE")

    def get_num_data(self, stream_id: str) -> int:
        return 5

    def get_image_data_by_index(self, stream_id: str, index: int) -> tuple[_Image, _Record]:
        return _Image(np.zeros(self.shapes[stream_id], dtype=np.uint8)), _Record()


class _VideoProvider:
    def __init__(self, left_shape: tuple[int, int]) -> None:
        self.left_shape = left_shape

    def get_stream_id_from_label(self, label: str) -> str:
        return "camera-slam-left"

    def get_image_data_by_index(self, stream_id: str, index: int) -> tuple[_Image, _Record]:
        return _Image(np.zeros(self.left_shape, dtype=np.uint8)), _Record()


def test_stream_inventory_lists_image_streams() -> None:
    provider = _GTProvider({"rgb": (1408, 1408), "left": (480, 640), "right": (480, 640)})
    inventory = adt_gt.stream_inventory(provider)
    assert {item["stream_id"] for item in inventory} == {"rgb", "left", "right"}
    assert all(item["sensor_data_type"] == "IMAGE" for item in inventory)


def test_find_left_gt_stream_id_selects_first_matching_slam_shape() -> None:
    gt = _GTProvider({"rgb": (1408, 1408), "left": (480, 640), "right": (480, 640)})
    video = _VideoProvider((480, 640))
    assert adt_gt.find_left_gt_stream_id(gt, video, "camera-slam-left") == "left"


def test_find_left_gt_stream_id_requires_two_slam_shaped_streams() -> None:
    gt = _GTProvider({"rgb": (1408, 1408), "left": (480, 640)})
    video = _VideoProvider((480, 640))
    with pytest.raises(RuntimeError, match="expected two ADT GT streams"):
        adt_gt.find_left_gt_stream_id(gt, video, "camera-slam-left")


def test_convert_camera_z_to_rectified_z_identity_preserves_z() -> None:
    source_rays = np.asarray([[0.5, -0.5, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    source_depths = np.asarray([2.0, 3.0], dtype=np.float64)
    result = adt_gt.convert_camera_z_to_rectified_z(
        source_depths,
        source_rays,
        left_device_from_camera=np.eye(4),
        device_to_rectified_rotation=np.eye(3),
    )
    np.testing.assert_allclose(result, source_depths)


def test_convert_camera_z_to_rectified_z_rotation_preserves_z_axis() -> None:
    angle = np.deg2rad(90.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source_rays = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64)
    source_depths = np.asarray([2.0], dtype=np.float64)
    result = adt_gt.convert_camera_z_to_rectified_z(
        source_depths,
        source_rays,
        left_device_from_camera=np.eye(4),
        device_to_rectified_rotation=rotation,
    )
    np.testing.assert_allclose(result, [2.0])


def _frame(depth: np.ndarray, uid: int) -> adt_gt.ADTGTFrame:
    return adt_gt.ADTGTFrame(
        frame_offset=0,
        source_frame_index=0,
        timestamp_s=0.0,
        depth_rect_m=depth.astype(np.float32),
        segmentation_rect=np.full(depth.shape, uid, dtype=np.int64),
        object_mask_rect=np.full(depth.shape, True, dtype=bool),
        valid_rect=np.full(depth.shape, True, dtype=bool),
    )


def test_write_gt_zarr_round_trip(tmp_path: Path) -> None:
    depth = np.full((2, 2), 1.25, dtype=np.float32)
    frames = [_frame(depth, 7)]
    selection = {"depth": {"selected_stream_id": "345-2"}}
    adt_gt.write_gt_zarr(
        frames,
        tmp_path / "gt",
        object_uid=7,
        gt_stream_selection=selection,
    )
    import zarr

    group = zarr.open_group(str(tmp_path / "gt" / "adt_gt.zarr"), mode="r")
    np.testing.assert_allclose(np.asarray(group["depth_m"]), depth[None])
    assert dict(group.attrs)["gt_stream_selection"] == selection
