"""Proper-chirality semantic frames for official UmeTrack hand tracking.

The official toolkit reflects the right wrist transform before skinning.  Its
landmark positions are correct, but a private skinning rotation read without
the matching reflection is in a different chirality convention.  These pure
helpers apply that convention consistently and calibrate a single neutral
semantic frame (x=palm normal, z=DIP-to-tip) for UmeTrack -> xHand MINK.
"""

from __future__ import annotations

import numpy as np


TIP_LANDMARKS = np.arange(5, dtype=np.int64)
DISTAL_LANDMARKS = np.asarray([7, 10, 13, 16, 19], dtype=np.int64)
DISTAL_SKINNING_FRAMES = np.asarray([4, 7, 10, 13, 16], dtype=np.int64)
WRIST_LANDMARK = 5
INDEX_PROXIMAL_LANDMARK = 8
MIDDLE_PROXIMAL_LANDMARK = 11
PINKY_PROXIMAL_LANDMARK = 17
RIGHT_REFLECTION = np.diag([-1.0, 1.0, 1.0])


def geometric_frame(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    x_axis = np.asarray(normal, dtype=np.float64)
    x_axis -= float(np.dot(x_axis, z_axis)) * z_axis
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def semantic_palm_frame(landmarks: np.ndarray) -> np.ndarray:
    values = np.asarray(landmarks, dtype=np.float64)
    if values.shape != (20, 3):
        raise ValueError("UmeTrack landmarks must have shape (20, 3)")
    wrist = values[WRIST_LANDMARK]
    palm_normal = np.cross(
        values[INDEX_PROXIMAL_LANDMARK] - wrist,
        values[PINKY_PROXIMAL_LANDMARK] - wrist,
    )
    return geometric_frame(
        palm_normal, values[MIDDLE_PROXIMAL_LANDMARK] - wrist,
    )


def make_proper_distal_rotations(
    skinning_rotations: np.ndarray, *, is_right: bool,
) -> np.ndarray:
    """Recover proper SO(3) from toolkit-convention skinning rotations."""
    rotations = np.asarray(skinning_rotations, dtype=np.float64)
    if rotations.shape[-3:] != (17, 3, 3):
        raise ValueError("skinning_rotations must end in shape (17, 3, 3)")
    distal = rotations[..., DISTAL_SKINNING_FRAMES, :, :]
    if is_right:
        distal = distal @ RIGHT_REFLECTION
    determinant = np.linalg.det(distal)
    if np.min(determinant) < 0.9999 or np.max(determinant) > 1.0001:
        raise ValueError("UmeTrack distal rotations are not proper after chirality repair")
    return distal


def neutral_semantic_calibration(
    neutral_landmarks: np.ndarray, proper_neutral_distal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    landmarks = np.asarray(neutral_landmarks, dtype=np.float64)
    distal = np.asarray(proper_neutral_distal, dtype=np.float64)
    if landmarks.shape != (20, 3) or distal.shape != (5, 3, 3):
        raise ValueError("neutral UmeTrack inputs have invalid shapes")
    palm = semantic_palm_frame(landmarks)
    semantic = np.asarray([
        geometric_frame(palm[:, 0], landmarks[tip] - landmarks[dip])
        for tip, dip in zip(TIP_LANDMARKS, DISTAL_LANDMARKS)
    ])
    calibration = np.einsum("fji,fjk->fik", distal, semantic)
    neutral_vectors = np.einsum(
        "ji,fj->fi", palm,
        landmarks[TIP_LANDMARKS] - landmarks[WRIST_LANDMARK],
    )
    return calibration, neutral_vectors, palm


def apply_semantic_calibration(
    proper_distal: np.ndarray, calibration: np.ndarray,
) -> np.ndarray:
    distal = np.asarray(proper_distal, dtype=np.float64)
    value = np.einsum("...fij,fjk->...fik", distal, calibration)
    determinant = np.linalg.det(value)
    if np.min(determinant) < 0.9999 or np.max(determinant) > 1.0001:
        raise ValueError("calibrated UmeTrack frames are not proper SO(3)")
    return value


def axis_contract(
    frames: np.ndarray, landmarks: np.ndarray, *, limit_deg: float = 1.0,
) -> dict[str, object]:
    rotations = np.asarray(frames, dtype=np.float64)
    points = np.asarray(landmarks, dtype=np.float64)
    direction = points[..., TIP_LANDMARKS, :] - points[..., DISTAL_LANDMARKS, :]
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    error = np.degrees(np.arccos(np.clip(
        np.sum(rotations[..., :, :, 2] * direction, axis=-1), -1.0, 1.0,
    )))
    determinant = np.linalg.det(rotations)
    passed = bool(
        np.percentile(error, 95) <= limit_deg
        and determinant.min() >= 0.9999 and determinant.max() <= 1.0001
    )
    return {
        "passed": passed,
        "dip_to_tip_error_deg": {
            "mean": float(error.mean()),
            "p95": float(np.percentile(error, 95)),
            "max": float(error.max()),
        },
        "proper_so3_determinant": {
            "minimum": float(determinant.min()),
            "maximum": float(determinant.max()),
        },
        "limit_deg": float(limit_deg),
        "episode_specific_axis_fit": False,
    }
