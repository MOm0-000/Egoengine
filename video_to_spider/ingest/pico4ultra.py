#!/usr/bin/env python3
"""PICO 4 Ultra recording adapter -> EgoRecording."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from video_to_spider.ego_recording import (
    CameraModel,
    CameraPair,
    EgoRecording,
    HandState,
    TimestampRecord,
)


class Pico4UltraAdapter:
    """Minimal adapter skeleton for the standard PICO recording format."""

    def load(self, recording_dir: str | Path) -> EgoRecording:
        root = Path(recording_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        calib = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        recording = EgoRecording(
            source_device="pico4ultra",
            capture_software_version=str(manifest.get("capture_software_version", "0.0.0")),
            capture_sdk_version=str(manifest.get("capture_sdk_version", "unverified")),
            calibration_id=str(calib.get("calibration_id", "")),
            calibration_version=str(calib.get("calibration_version", "")),
        )
        left = self._camera_from_calib(calib["left"])
        right = self._camera_from_calib(calib["right"])
        recording.cameras = CameraPair(
            left=left,
            right=right,
            T_device_camera_left=np.asarray(calib["T_device_camera_left"], dtype=np.float64),
            T_device_camera_right=np.asarray(calib["T_device_camera_right"], dtype=np.float64),
        )
        recording.timestamps = self._load_timestamps(root / "timestamps.csv")
        recording.device_pose = np.asarray(
            manifest.get("device_pose", np.eye(4).tolist()), dtype=np.float64
        )
        recording.hand = self._load_hand(root / "hand_tracking.csv")
        return recording

    @staticmethod
    def _camera_from_calib(item: dict) -> CameraModel:
        return CameraModel(
            width=int(item["width"]),
            height=int(item["height"]),
            intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
            distortion=np.asarray(item["distortion"], dtype=np.float64),
            distortion_model=str(item["distortion_model"]),
        )

    @staticmethod
    def _load_timestamps(path: Path) -> list[TimestampRecord]:
        del path  # PICO device timestamps are unverified until G1.
        return []

    @staticmethod
    def _load_hand(path: Path) -> list[HandState]:
        del path
        return []
