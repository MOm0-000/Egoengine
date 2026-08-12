"""Axis-calibrated MANO wrist and distal-link orientation forward kinematics.

WiLoR exports MANO root and local joint rotations, but the previous pipeline
discarded their kinematic structure and rebuilt a fingertip frame from three
landmarks. This module keeps the MANO rotational FK intact. The constant frame
calibrations below were generated once from the licensed MANO_RIGHT neutral
mesh (zero betas, identity local rotations): ``z`` follows the neutral
DIP-to-tip direction and ``x`` follows the neutral palm normal projected onto
the distal cross-section. They are model conventions, not episode fits.

Only the small coordinate-frame constants are embedded here. The licensed MANO
geometry is neither copied nor required by the sequence optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# MANO internal joint order used by smplx/WiLoR. Joint zero is the wrist;
# rotations 1..15 are stored in mano_hand_pose[:, 0..14].
MANO_PARENTS = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    dtype=np.int64,
)

# SPIDER finger order is thumb, index, middle, ring, little.
MANO_DISTAL_JOINT_INDICES = np.array([15, 3, 6, 12, 9], dtype=np.int64)

# Fixed calibration from MANO_RIGHT v1.2 neutral geometry. Columns are the
# anatomical x/y/z axes expressed in each neutral distal-link frame.
_RIGHT_DISTAL_AXIS_CALIBRATION = np.array(
    [
        [
            [0.0512711850, 0.6766356000, -0.7345308000],
            [-0.9971145400, -0.0065457895, -0.0756297100],
            [-0.0559818400, 0.7362889600, 0.6743476000],
        ],
        [
            [-0.0459068400, -0.0207980280, -0.9987292000],
            [-0.9986254600, -0.0243619800, 0.0464093950],
            [-0.0252962450, 0.9994868600, -0.0196510550],
        ],
        [
            [-0.1400863800, -0.1627401600, -0.9766736600],
            [-0.9888978000, -0.0263841400, 0.1462360300],
            [-0.0495671700, 0.9863161400, -0.1572373400],
        ],
        [
            [-0.1350811300, -0.2916102000, -0.9469512000],
            [-0.9883647000, -0.0277821050, 0.1495440900],
            [-0.0699168800, 0.9561337000, -0.2844644000],
        ],
        [
            [-0.0511170250, -0.5183905000, -0.8536148700],
            [-0.9965807000, -0.0290797690, 0.0773380550],
            [-0.0649142340, 0.8546495000, -0.5151315000],
        ],
    ],
    dtype=np.float64,
)

_RIGHT_WRIST_AXIS_CALIBRATION = np.array(
    [
        [0.0164692900, -0.0350526240, -0.9992497600],
        [-0.9995758000, -0.0245871380, -0.0156121740],
        [-0.0240214430, 0.9990830000, -0.0354426880],
    ],
    dtype=np.float64,
)

_REFLECT_X = np.diag([-1.0, 1.0, 1.0])
_PRESERVE_RIGHT_HANDED_FRAME = np.diag([1.0, -1.0, 1.0])


def _axis_calibration(side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "right":
        return _RIGHT_WRIST_AXIS_CALIBRATION, _RIGHT_DISTAL_AXIS_CALIBRATION
    if side == "left":
        # WiLoR mirrors left-hand rotations as M R M. Mirroring the anatomical
        # frame itself needs one additional y-axis sign flip to keep det(R)=+1.
        wrist = (
            _REFLECT_X @ _RIGHT_WRIST_AXIS_CALIBRATION
            @ _PRESERVE_RIGHT_HANDED_FRAME
        )
        distal = np.einsum(
            "ij,fjk,kl->fil",
            _REFLECT_X,
            _RIGHT_DISTAL_AXIS_CALIBRATION,
            _PRESERVE_RIGHT_HANDED_FRAME,
        )
        return wrist, distal
    raise ValueError("side must be 'left' or 'right'")


def mano_global_joint_rotations(
    global_orient: np.ndarray, hand_pose: np.ndarray,
) -> np.ndarray:
    """Compose MANO local rotations into all 16 global link rotations."""
    root = np.asarray(global_orient, dtype=np.float64)
    local = np.asarray(hand_pose, dtype=np.float64)
    if root.ndim != 3 or root.shape[1:] != (3, 3):
        raise ValueError("global_orient must have shape (T, 3, 3)")
    if local.shape != (len(root), 15, 3, 3):
        raise ValueError("hand_pose must have shape (T, 15, 3, 3)")
    result = np.empty((len(root), 16, 3, 3), dtype=np.float64)
    result[:, 0] = root
    for joint in range(1, 16):
        parent = int(MANO_PARENTS[joint])
        result[:, joint] = result[:, parent] @ local[:, joint - 1]
    return result


def mano_wrist_and_distal_frames(
    global_orient: np.ndarray, hand_pose: np.ndarray, side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated wrist ``(T,3,3)`` and distal ``(T,5,3,3)`` frames."""
    wrist_calibration, distal_calibration = _axis_calibration(side)
    global_rotations = mano_global_joint_rotations(global_orient, hand_pose)
    wrist = global_rotations[:, 0] @ wrist_calibration
    distal = global_rotations[:, MANO_DISTAL_JOINT_INDICES] @ distal_calibration
    return wrist, distal


def _rotation_angle(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    cosine = np.clip(
        (np.trace(values, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0,
    )
    return np.arccos(cosine)


def mano_orientation_diagnostics(
    wrist_fk: np.ndarray,
    distal_fk: np.ndarray,
    joints_camera: np.ndarray,
    wrist_landmark: np.ndarray,
    distal_landmark: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Cross-check FK frames against independent 21-landmark geometry."""
    wrists = np.asarray(wrist_fk, dtype=np.float64)
    distal = np.asarray(distal_fk, dtype=np.float64)
    joints = np.asarray(joints_camera, dtype=np.float64)
    valid_values = np.asarray(valid, dtype=bool)
    if wrists.shape != (len(joints), 3, 3):
        raise ValueError("wrist_fk has an incompatible shape")
    if distal.shape != (len(joints), 5, 3, 3):
        raise ValueError("distal_fk has an incompatible shape")
    if joints.shape[1:] != (21, 3) or valid_values.shape != (len(joints),):
        raise ValueError("joints_camera/valid have incompatible shapes")

    tip_indices = np.array([4, 8, 12, 16, 20])
    dip_indices = tip_indices - 1
    observed_direction = joints[:, tip_indices] - joints[:, dip_indices]
    norms = np.linalg.norm(observed_direction, axis=-1)
    observed_direction /= np.maximum(norms[..., None], 1e-12)
    direction_angle = np.arccos(np.clip(
        np.sum(distal[..., 2] * observed_direction, axis=-1), -1.0, 1.0,
    ))
    full_delta = np.swapaxes(distal, -1, -2) @ np.asarray(
        distal_landmark, dtype=np.float64,
    )
    full_angle = _rotation_angle(full_delta)
    wrist_delta = np.swapaxes(wrists, -1, -2) @ np.asarray(
        wrist_landmark, dtype=np.float64,
    )
    wrist_angle = _rotation_angle(wrist_delta)
    usable_direction = valid_values[:, None] & (norms >= 1e-8)
    usable_full = np.broadcast_to(valid_values[:, None], direction_angle.shape)

    def summary(values: np.ndarray, usable: np.ndarray) -> dict[str, Any]:
        selected = values[usable]
        return {
            "sample_count": int(selected.size),
            "median_rad": float(np.median(selected)) if selected.size else None,
            "p95_rad": float(np.percentile(selected, 95)) if selected.size else None,
            "max_rad": float(np.max(selected)) if selected.size else None,
        }

    direction_summary = summary(direction_angle, usable_direction)
    return {
        "source": "MANO rotational FK with fixed neutral distal-axis calibration",
        "calibration": "MANO_RIGHT_v1_2_zero_betas_identity_pose",
        "episode_specific_axis_fit": False,
        "distal_axis_vs_dip_to_tip": direction_summary,
        "full_so3_vs_landmark_proxy": summary(full_angle, usable_full),
        "wrist_so3_vs_landmark_proxy": summary(wrist_angle, valid_values),
        # Fifteen degrees is far above expected model agreement but remains
        # tolerant of noisy monocular landmarks.
        "distal_axis_consistency_limit_rad": float(np.deg2rad(15.0)),
        "passed": bool(
            direction_summary["p95_rad"] is not None
            and direction_summary["p95_rad"] <= np.deg2rad(15.0)
        ),
    }
