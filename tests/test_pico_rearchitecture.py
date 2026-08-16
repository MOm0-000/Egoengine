#!/usr/bin/env python3
"""Smoke tests for PICO-first rearchitecture contracts."""

from __future__ import annotations

import numpy as np

from video_to_spider.coordinate import compose_transform, invert_transform
from video_to_spider.ego_recording import CameraModel, CameraPair, EgoRecording
from video_to_spider.hand.pico_system_hand import (
    CanonicalHandState,
    PicoSystemHandProvider,
    pico26_to_canonical21,
)


def test_ego_recording_schema_smoke():
    rec = EgoRecording(source_device="pico4ultra")
    rec.cameras = CameraPair(
        left=CameraModel(640, 480, np.eye(3), np.zeros(5), "radial"),
        right=CameraModel(640, 480, np.eye(3), np.zeros(5), "radial"),
        T_device_camera_left=np.eye(4),
        T_device_camera_right=np.eye(4),
    )
    assert rec.schema_version == "1.0"
    assert rec.source_device == "pico4ultra"


def test_coordinate_convention_roundtrip():
    T_A_B = np.eye(4)
    T_A_B[:3, 3] = [0.1, 0.0, 0.0]
    T_B_A = invert_transform(T_A_B)
    np.testing.assert_allclose(compose_transform(T_A_B, T_B_A), np.eye(4), atol=1e-10)


def test_pico26_to_canonical21_shape():
    joints = np.zeros((26, 3), dtype=np.float64)
    canonical = pico26_to_canonical21(joints)
    assert canonical.shape == (21, 3)


def test_pico_hand_to_mink_shape():
    positions = np.zeros((26, 3), dtype=np.float64)
    rotations = np.broadcast_to(np.eye(3), (26, 3, 3)).copy()
    hand = type(
        "Hand",
        (),
        {
            "handedness": "right",
            "native_joint_positions": positions,
            "native_joint_rotations": rotations,
            "wrist_pose": np.eye(4),
            "confidence": np.ones(26),
            "validity": np.ones(26, dtype=bool),
            "source": "pico_system",
        },
    )()
    provider = PicoSystemHandProvider()
    mink_state = provider.to_mink_state(hand)
    assert isinstance(mink_state, CanonicalHandState)
    assert mink_state.fingertip_positions.shape == (5, 3)
    assert mink_state.fingertip_orientations.shape == (5, 3, 3)
