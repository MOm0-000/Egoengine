"""Frozen NPZ protocols for enhanced MINK retargeting."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _required(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"missing required array: {key}")
    return np.asarray(arrays[key])


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _transforms(name: str, value: np.ndarray) -> None:
    if value.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must end in (4,4), got {value.shape}")
    _finite(name, value)
    if not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name} has invalid homogeneous rows")
    rotation = value[..., :3, :3].reshape(-1, 3, 3)
    if not np.allclose(np.swapaxes(rotation, 1, 2) @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} contains invalid rotations")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} contains a reflection instead of a rotation")


def validate_human_reference(arrays: Mapping[str, np.ndarray]) -> None:
    frame_indices = _required(arrays, "frame_indices")
    timestamps = _required(arrays, "timestamps_s")
    hand_order = _required(arrays, "hand_order")
    wrist = _required(arrays, "T_sim_wrist_target")
    tips = _required(arrays, "T_sim_fingertip_target")
    valid_hand = _required(arrays, "valid_hand")
    valid_orientation = _required(arrays, "valid_fingertip_orientation")
    confidence_hand = _required(arrays, "confidence_hand")
    confidence_tip = _required(arrays, "confidence_fingertip")
    if frame_indices.ndim != 1 or timestamps.shape != frame_indices.shape:
        raise ValueError("frame_indices and timestamps_s must have shape (T,)")
    if len(frame_indices) > 1 and (np.diff(frame_indices) <= 0).any():
        raise ValueError("frame_indices must be strictly increasing")
    if len(timestamps) > 1 and (np.diff(timestamps) <= 0).any():
        raise ValueError("timestamps_s must be strictly increasing")
    t, h = len(frame_indices), len(hand_order)
    if wrist.shape != (t, h, 4, 4) or tips.shape != (t, h, 5, 4, 4):
        raise ValueError("hand target transforms do not match timeline/hand dimensions")
    for name, value, shape in (
        ("valid_hand", valid_hand, (t, h)),
        ("valid_fingertip_orientation", valid_orientation, (t, h, 5)),
        ("confidence_hand", confidence_hand, (t, h)),
        ("confidence_fingertip", confidence_tip, (t, h, 5)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} expected {shape}, got {value.shape}")
        _finite(name, value)
    if set(hand_order.tolist()) - {"left", "right"}:
        raise ValueError("hand_order may only contain left/right")
    _transforms("T_sim_wrist_target", wrist)
    _transforms("T_sim_fingertip_target", tips)
    if "T_sim_object_reference" in arrays:
        objects = np.asarray(arrays["T_sim_object_reference"])
        if objects.ndim != 4 or objects.shape[0] != t:
            raise ValueError("T_sim_object_reference must have shape (T,O,4,4)")
        _transforms("T_sim_object_reference", objects)
    if "mano_joint_positions_sim" in arrays:
        joints = np.asarray(arrays["mano_joint_positions_sim"])
        if joints.shape != (t, h, 21, 3):
            raise ValueError("mano_joint_positions_sim must have shape (T,H,21,3)")
        _finite("mano_joint_positions_sim", joints)
    if "fingertip_direction_sim" in arrays:
        directions = np.asarray(arrays["fingertip_direction_sim"])
        if directions.shape != (t, h, 5, 3):
            raise ValueError("fingertip_direction_sim must have shape (T,H,5,3)")
        _finite("fingertip_direction_sim", directions)
        valid = valid_orientation.astype(bool)
        if valid.any() and not np.allclose(
            np.linalg.norm(directions[valid], axis=-1), 1.0, atol=1e-5,
        ):
            raise ValueError("valid fingertip directions must be unit length")


def validate_robot_reference(arrays: Mapping[str, np.ndarray]) -> None:
    frames = _required(arrays, "frame_indices")
    timestamps = _required(arrays, "timestamps_s")
    qpos = _required(arrays, "qpos")
    qvel = _required(arrays, "qvel")
    wrist = _required(arrays, "T_sim_wrist")
    tips = _required(arrays, "T_sim_fingertip")
    t = len(frames)
    if timestamps.shape != (t,) or qpos.ndim != 2 or qpos.shape[0] != t:
        raise ValueError("robot reference timeline is inconsistent")
    if qvel.ndim != 2 or qvel.shape[0] != t:
        raise ValueError("qvel must have shape (T,nv)")
    if wrist.shape[:2] != (t, len(_required(arrays, "hand_order"))) or wrist.shape[-2:] != (4, 4):
        raise ValueError("T_sim_wrist has invalid shape")
    if tips.shape != (t, wrist.shape[1], 5, 4, 4):
        raise ValueError("T_sim_fingertip has invalid shape")
    for key in (
        "retargeting_loss", "fingertip_position_error_m", "fingertip_orientation_error_rad",
        "wrist_position_error_m", "wrist_orientation_error_rad", "joint_limit_min_margin",
        "joint_limit_violation", "velocity_limit_violation", "self_collision_min_distance_m",
        "self_collision_violation",
    ):
        value = _required(arrays, key)
        if value.shape != (t,):
            raise ValueError(f"{key} must have shape (T,)")
        _finite(key, value)
    _finite("qpos", qpos)
    _finite("qvel", qvel)
    if "ctrl" in arrays:
        ctrl = np.asarray(arrays["ctrl"])
        if ctrl.ndim != 2 or ctrl.shape[0] != t:
            raise ValueError("ctrl must have shape (T,nu)")
        _finite("ctrl", ctrl)
    for key, width in (("initial_qpos", qpos.shape[1]), ("initial_qvel", qvel.shape[1])):
        if key in arrays:
            value = np.asarray(arrays[key])
            if value.shape != (width,):
                raise ValueError(f"{key} must have shape ({width},)")
            _finite(key, value)
    _transforms("T_sim_wrist", wrist)
    _transforms("T_sim_fingertip", tips)
    if "T_sim_wrist_target" in arrays:
        wrist_target = np.asarray(arrays["T_sim_wrist_target"])
        if wrist_target.shape != wrist.shape:
            raise ValueError("T_sim_wrist_target must match T_sim_wrist")
        _transforms("T_sim_wrist_target", wrist_target)
    if "T_sim_wrist_reward_reference" in arrays:
        wrist_reward = np.asarray(arrays["T_sim_wrist_reward_reference"])
        if wrist_reward.shape != wrist.shape:
            raise ValueError("T_sim_wrist_reward_reference must match T_sim_wrist")
        _transforms("T_sim_wrist_reward_reference", wrist_reward)
