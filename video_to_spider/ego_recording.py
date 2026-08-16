#!/usr/bin/env python3
"""Device-agnostic EgoRecording schema for PICO/Aria-compatible pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


SCHEMA_VERSION = "1.0"
COORDINATE_CONVENTION = "T_A_B transforms points from frame B into frame A"


@dataclass(frozen=True)
class TimestampRecord:
    timestamp_value: float
    clock_domain: str
    capture_timestamp: float
    arrival_timestamp: float
    sync_reference: str
    sync_status: Literal["unsynced", "interpolated", "hardware_sync", "manual"]
    time_offset_to_reference_s: float
    interpolation_method: str | None


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    intrinsics: np.ndarray
    distortion: np.ndarray
    distortion_model: str


@dataclass
class CameraPair:
    left: CameraModel
    right: CameraModel
    T_device_camera_left: np.ndarray
    T_device_camera_right: np.ndarray


@dataclass
class HandState:
    handedness: str
    native_joint_positions: np.ndarray
    native_joint_rotations: np.ndarray
    wrist_pose: np.ndarray
    confidence: np.ndarray
    validity: np.ndarray
    source: str


@dataclass
class EgoRecording:
    schema_version: str = SCHEMA_VERSION
    source_device: str = "pico4ultra"
    capture_software_version: str = "0.0.0"
    capture_sdk_version: str = "unverified"
    calibration_id: str = ""
    calibration_version: str = ""
    coordinate_convention: str = COORDINATE_CONVENTION
    timestamps: list[TimestampRecord] = field(default_factory=list)
    cameras: CameraPair | None = None
    device_pose: np.ndarray | None = None
    hand: list[HandState] = field(default_factory=list)
    optional: dict[str, Any] = field(default_factory=dict)


def new_empty_recording() -> EgoRecording:
    return EgoRecording()
