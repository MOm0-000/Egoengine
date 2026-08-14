from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_to_spider.ingest import adt_stereo


class _Matrix:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=np.float64)

    def to_matrix(self) -> np.ndarray:
        return self.value


class _SE3:
    def __init__(self, value: np.ndarray) -> None:
        self._matrix = _Matrix(value)

    @classmethod
    def from_matrix(cls, value: np.ndarray) -> "_SE3":
        return cls(value)

    def to_matrix(self) -> np.ndarray:
        return self._matrix.to_matrix()


class _Image:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy_array(self) -> np.ndarray:
        return self.array


class _Record:
    def __init__(self, timestamp_ns: int) -> None:
        self.capture_timestamp_ns = timestamp_ns


class _SrcCalib:
    def __init__(self, transform: np.ndarray) -> None:
        self.transform = transform

    def get_transform_device_camera(self) -> _SE3:
        return _SE3(self.transform)


class _DeviceCalib:
    def __init__(self, transforms: dict[str, np.ndarray]) -> None:
        self.transforms = transforms

    def get_camera_calib(self, label: str) -> _SrcCalib:
        return _SrcCalib(self.transforms[label])


class _Provider:
    def __init__(self, count: int, timestamps_ns: list[int]) -> None:
        self.count = count
        self.timestamps_ns = timestamps_ns
        self.device_calib = _DeviceCalib(
            {
                "camera-slam-left": np.eye(4),
                "camera-slam-right": _translation_matrix(0.07, 0.0, 0.0),
            }
        )

    def get_device_calibration(self) -> _DeviceCalib:
        return self.device_calib

    def get_stream_id_from_label(self, label: str) -> int:
        return 1 if label == "camera-slam-left" else 2

    def get_num_data(self, stream_id: int) -> int:
        return self.count

    def get_timestamps_ns(self, stream_id: int, time_domain: object) -> list[int]:
        return self.timestamps_ns

    def get_image_data_by_index(self, stream_id: int, index: int) -> tuple[_Image, _Record]:
        value = 100 + index if stream_id == 1 else 120 + index
        image = _Image(np.full((480, 640), value, dtype=np.uint8))
        return image, _Record(self.timestamps_ns[index])

    def get_image_data_by_time_ns(
        self, stream_id: int, time_ns: int, time_domain: object, query_options: object
    ) -> tuple[_Image, _Record]:
        index = self.timestamps_ns.index(time_ns)
        return self.get_image_data_by_index(stream_id, index)

    def get_nominal_rate_hz(self, stream_id: int) -> float:
        return 30.0


class _Calibration:
    @staticmethod
    def get_linear_camera_calibration(width, height, focal_length, label, transform):
        return object()

    @staticmethod
    def distort_by_calibration(array, dst_calib, src_calib, interpolation):
        return np.full((512, 512), 255, dtype=np.float32)


class _ImageModule:
    class InterpolationMethod:
        NEAREST_NEIGHBOR = object()


class _SensorData:
    class TimeDomain:
        DEVICE_TIME = object()

    class TimeQueryOptions:
        CLOSEST = object()


class _Pose:
    def __init__(self, timestamp_ns: int) -> None:
        self.tracking_timestamp = _TrackingTimestamp(timestamp_ns)
        self.transform_world_device = _SE3(np.eye(4))


class _TrackingTimestamp:
    def __init__(self, timestamp_ns: int) -> None:
        self.ns = timestamp_ns

    def total_seconds(self) -> float:
        return self.ns / 1e9


class _Mps:
    @staticmethod
    def read_closed_loop_trajectory(path: str) -> list[_Pose]:
        return [_Pose(0), _Pose(int(1e9)), _Pose(int(2e9)), _Pose(int(3e9))]


class _DataProvider:
    @staticmethod
    def create_vrs_data_provider(path: str) -> _Provider:
        return _Provider(3, [0, int(1e9), int(2e9)])


class _Sophus:
    SE3 = _SE3


def _translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = [x, y, z]
    return value


@pytest.fixture
def fake_projectaria(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        adt_stereo,
        "_load_projectaria",
        lambda: (_DataProvider, _Calibration, _ImageModule, _SensorData, _Mps, _Sophus),
    )
    return tmp_path


def test_prepare_adt_stereo_writes_ingest_inputs(fake_projectaria: Path) -> None:
    (fake_projectaria / "recording.vrs").write_bytes(b"x")
    (fake_projectaria / "closed_loop_trajectory.csv").write_text("x", encoding="utf-8")
    output = fake_projectaria / "adt_prepared"
    result = adt_stereo.prepare_adt_stereo(
        video_vrs=fake_projectaria / "recording.vrs",
        output_dir=output,
        closed_loop_trajectory_path=fake_projectaria / "closed_loop_trajectory.csv",
    )

    assert result.baseline_m == pytest.approx(0.07)
    assert result.fps == pytest.approx(30.0)
    assert result.left_dir.is_dir()
    assert result.right_dir.is_dir()
    assert len(list(result.left_dir.glob("*.png"))) == 3
    assert len(list(result.right_dir.glob("*.png"))) == 3
    assert result.intrinsics_path.is_file()
    assert result.common_valid_mask_path.is_file()
    assert result.camera_poses_path is not None and result.camera_poses_path.is_file()

    metadata = adt_stereo.json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 3
    assert metadata["timestamp_sync"]["accepted"] is True
    assert metadata["common_valid_pixel_ratio"] == 1.0


def test_prepare_adt_stereo_rejects_moving_recording_without_poses(
    fake_projectaria: Path,
) -> None:
    (fake_projectaria / "recording.vrs").write_bytes(b"x")
    with pytest.raises(ValueError, match="closed-loop-trajectory"):
        adt_stereo.prepare_adt_stereo(
            video_vrs=fake_projectaria / "recording.vrs",
            output_dir=fake_projectaria / "adt_prepared_missing_pose",
        )
