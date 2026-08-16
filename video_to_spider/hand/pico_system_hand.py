#!/usr/bin/env python3
"""PICO system hand provider: 26 joints -> MINK fingertip/wrist schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from video_to_spider.ego_recording import HandState


PICO26_JOINT_ORDER = tuple(range(26))
FINGERTIP_INDICES = {"thumb": 4, "index": 9, "middle": 14, "ring": 19, "pinky": 24}
WRIST_INDEX = 0


@dataclass
class CanonicalHandState:
    wrist_position: np.ndarray
    wrist_orientation: np.ndarray
    fingertip_positions: np.ndarray
    fingertip_orientations: np.ndarray
    confidence: np.ndarray
    validity: bool
    source: str


def pico26_to_canonical21(joints: np.ndarray) -> np.ndarray:
    """Lossy PICO 26-joint to canonical 21-joint mapping.

    This is only for benchmark/visualization and must not be used as the
    primary MINK interface.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (26, 3):
        raise ValueError(f"expected (26, 3), got {joints.shape}")
    # Keep wrist + the first 20 finger landmarks; exact joint semantics must
    # be calibrated on-device before production use.
    return joints[:21].copy()


class PicoSystemHandProvider:
    """Wrap a PICO 26-joint HandState into MINK-ready fingertip/wrist."""

    def __init__(self, *, validity_threshold: float = 0.5):
        self.validity_threshold = validity_threshold

    def to_mink_state(self, hand: HandState) -> CanonicalHandState:
        positions = np.asarray(hand.native_joint_positions, dtype=np.float64)
        rotations = np.asarray(hand.native_joint_rotations, dtype=np.float64)
        if positions.shape != (26, 3):
            raise ValueError(f"expected 26 positions, got {positions.shape}")
        if rotations.shape != (26, 3, 3):
            raise ValueError(f"expected 26 rotations, got {rotations.shape}")
        wrist = positions[WRIST_INDEX]
        wrist_rotation = rotations[WRIST_INDEX]
        fingertips = np.stack([positions[FINGERTIP_INDICES[k]] for k in sorted(FINGERTIP_INDICES)])
        fingertip_rotations = np.stack([rotations[FINGERTIP_INDICES[k]] for k in sorted(FINGERTIP_INDICES)])
        validity = bool(
            np.asarray(hand.validity, dtype=bool).all()
            and float(np.min(hand.confidence)) >= self.validity_threshold
        )
        return CanonicalHandState(
            wrist_position=wrist,
            wrist_orientation=wrist_rotation,
            fingertip_positions=fingertips,
            fingertip_orientations=fingertip_rotations,
            confidence=np.asarray(hand.confidence, dtype=np.float64),
            validity=validity,
            source=hand.source,
        )
