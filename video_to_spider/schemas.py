"""Strict validators for file artifacts shared across isolated environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA_VERSION = "1.0"
UNITS = "meter-second-radian"


class ArtifactValidationError(ValueError):
    """Raised when an artifact violates the frozen V1 protocol."""


def _array(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ArtifactValidationError(f"missing required array: {key}")
    value = np.asarray(arrays[key])
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise ArtifactValidationError(f"{key} has unsupported dtype {value.dtype}")
    return value


def require_shape(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape):
        raise ArtifactValidationError(f"{name} expected {len(shape)} dims, got {value.shape}")
    for actual, expected in zip(value.shape, shape):
        if expected is not None and actual != expected:
            raise ArtifactValidationError(f"{name} expected shape {shape}, got {value.shape}")


def require_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ArtifactValidationError(f"{name} contains NaN or infinity")


def validate_timestamps(frame_indices: np.ndarray, timestamps_s: np.ndarray) -> int:
    require_shape("frame_indices", frame_indices, (None,))
    require_shape("timestamps_s", timestamps_s, (frame_indices.shape[0],))
    require_finite("timestamps_s", timestamps_s)
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ArtifactValidationError("frame_indices must be an integer array")
    if frame_indices.size > 1 and np.any(np.diff(frame_indices) <= 0):
        raise ArtifactValidationError("frame_indices must be strictly increasing")
    if timestamps_s.size > 1 and np.any(np.diff(timestamps_s) <= 0):
        raise ArtifactValidationError("timestamps_s must be strictly increasing")
    return frame_indices.shape[0]


def validate_rotation_matrices(name: str, rotations: np.ndarray, atol: float = 1e-4) -> None:
    if rotations.shape[-2:] != (3, 3):
        raise ArtifactValidationError(f"{name} must end in (3, 3), got {rotations.shape}")
    require_finite(name, rotations)
    flat = rotations.reshape(-1, 3, 3)
    identity = np.eye(3, dtype=flat.dtype)
    errors = np.max(np.abs(np.swapaxes(flat, 1, 2) @ flat - identity), axis=(1, 2))
    determinants = np.linalg.det(flat)
    if np.any(errors > atol):
        raise ArtifactValidationError(f"{name} contains non-orthogonal rotations")
    if np.any(determinants <= 0) or np.any(np.abs(determinants - 1.0) > atol * 4):
        raise ArtifactValidationError(f"{name} contains improper rotations")


def validate_transforms(name: str, transforms: np.ndarray, atol: float = 1e-4) -> None:
    if transforms.shape[-2:] != (4, 4):
        raise ArtifactValidationError(f"{name} must end in (4, 4), got {transforms.shape}")
    require_finite(name, transforms)
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=transforms.dtype)
    if not np.allclose(transforms[..., 3, :], expected, atol=atol, rtol=0):
        raise ArtifactValidationError(f"{name} has invalid homogeneous last row")
    validate_rotation_matrices(f"{name} rotation", transforms[..., :3, :3], atol=atol)


def validate_wilor_raw(arrays: Mapping[str, np.ndarray]) -> None:
    frames = _array(arrays, "frame_indices")
    timestamps = _array(arrays, "timestamps_s")
    t = validate_timestamps(frames, timestamps)
    valid = _array(arrays, "valid")
    require_shape("valid", valid, (t, None))
    h = valid.shape[1]
    shapes = {
        "side": (t, h), "score": (t, h), "mano_global_orient": (t, h, 3, 3),
        "mano_hand_pose": (t, h, 15, 3, 3), "mano_betas": (t, h, 10),
        "joints_camera_rootrel": (t, h, 21, 3),
        "vertices_camera_rootrel": (t, h, 778, 3), "translation_camera": (t, h, 3),
    }
    for key, shape in shapes.items():
        value = _array(arrays, key)
        require_shape(key, value, shape)
        require_finite(key, value)
    validate_rotation_matrices("mano_global_orient", _array(arrays, "mano_global_orient"))
    validate_rotation_matrices("mano_hand_pose", _array(arrays, "mano_hand_pose"))


def validate_foundationpose_raw(arrays: Mapping[str, np.ndarray]) -> None:
    frames = _array(arrays, "frame_indices")
    timestamps = _array(arrays, "timestamps_s")
    t = validate_timestamps(frames, timestamps)
    for key in ("valid", "confidence", "registration_frame", "depth_residual", "mask_iou"):
        value = _array(arrays, key)
        require_shape(key, value, (t,))
        require_finite(key, value)
    transforms = _array(arrays, "T_camera_object")
    require_shape("T_camera_object", transforms, (t, 4, 4))
    validate_transforms("T_camera_object", transforms)


def validate_aligned_trajectory(arrays: Mapping[str, np.ndarray]) -> None:
    frames = _array(arrays, "frame_indices")
    timestamps = _array(arrays, "timestamps_s")
    t = validate_timestamps(frames, timestamps)
    obj = _array(arrays, "T_sim_object")
    wrist = _array(arrays, "T_sim_wrist")
    require_shape("T_sim_object", obj, (t, None, 4, 4))
    require_shape("T_sim_wrist", wrist, (t, None, 4, 4))
    o, h = obj.shape[1], wrist.shape[1]
    validate_transforms("T_sim_object", obj)
    validate_transforms("T_sim_wrist", wrist)
    shapes = {
        "fingertips_sim": (t, h, 5, 3), "mano_pose": (t, h, 15, 3, 3),
        "mano_betas": (h, 10), "object_scale_to_m": (o,), "valid_object": (t, o),
        "valid_hand": (t, h), "confidence_object": (t, o), "confidence_hand": (t, h),
    }
    for key, shape in shapes.items():
        value = _array(arrays, key)
        require_shape(key, value, shape)
        require_finite(key, value)
    if np.any(_array(arrays, "object_scale_to_m") <= 0):
        raise ArtifactValidationError("object_scale_to_m must be positive")
    validate_rotation_matrices("mano_pose", _array(arrays, "mano_pose"))


def validate_contact(arrays: Mapping[str, np.ndarray]) -> None:
    frames = _array(arrays, "frame_indices")
    timestamps = _array(arrays, "timestamps_s")
    t = validate_timestamps(frames, timestamps)
    contact = _array(arrays, "contact")
    require_shape("contact", contact, (t, None, 5))
    positions = _array(arrays, "contact_pos_object_local")
    require_shape("contact_pos_object_local", positions, (contact.shape[1], 5, 3))
    require_finite("contact", contact)
    require_finite("contact_pos_object_local", positions)
    if np.any((contact < 0) | (contact > 1)):
        raise ArtifactValidationError("contact must be binary/probability values in [0, 1]")


VALIDATORS = {
    "wilor_raw": validate_wilor_raw,
    "foundationpose_raw": validate_foundationpose_raw,
    "aligned_trajectory": validate_aligned_trajectory,
    "contact": validate_contact,
}


def validate_npz(path: str | Path, artifact_type: str) -> None:
    if artifact_type not in VALIDATORS:
        raise ArtifactValidationError(f"unknown artifact type: {artifact_type}")
    with np.load(Path(path), allow_pickle=False) as arrays:
        VALIDATORS[artifact_type](arrays)


@dataclass(frozen=True)
class FrameIndex:
    source_frame_index: int
    timestamp_s: float
    rgb_path: str
