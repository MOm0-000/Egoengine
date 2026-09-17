"""Map complete MANO21 motion to robot-compatible XHand finger motion.

The mapping deliberately does not copy human joint angles.  Human and XHand
kinematic trees are different, so each four-bone MANO finger is represented by
three morphology-normalized task-space features: fingertip position in the
anatomical palm frame, distal-bone direction, and total polyline curvature.
The temporal change of those features is transferred around a physically
screened XHand grasp and solved with bounded per-digit inverse kinematics.

This module only produces hand joint targets.  It never reads or writes an
object joint and cannot servo an object.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contracts import XHAND_FINGER_ORDER

FINGER_NAMES = XHAND_FINGER_ORDER
MANO21_FINGER_INDICES = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
MANO21_MCP_INDICES = (5, 9, 13, 17)


@dataclass(frozen=True)
class Mano21Features:
    """Morphology-normalized shape features for all five human digits."""

    palm_center: np.ndarray
    normalized_tip: np.ndarray
    distal_direction: np.ndarray
    total_curvature_rad: np.ndarray
    chain_length_m: np.ndarray


@dataclass(frozen=True)
class FullHandPriorTrajectory:
    """Precomputed XHand targets and the task-space evidence behind them."""

    qpos: np.ndarray
    desired_tip_local_m: np.ndarray
    desired_distal_direction_local: np.ndarray
    solved_tip_local_m: np.ndarray
    solved_distal_direction_local: np.ndarray
    position_residual_m: np.ndarray
    direction_residual_rad: np.ndarray
    optimizer_success: np.ndarray
    optimizer_evaluations: np.ndarray
    human_features: Mano21Features


def mano21_features(
    joint_positions: np.ndarray,
    palm_rotations: np.ndarray,
) -> Mano21Features:
    """Extract complete, palm-local MANO21 shape descriptors.

    ``joint_positions`` has shape ``(T,21,3)``.  The palm center uses the
    wrist and all four non-thumb MCPs.  Every other MANO point is consumed by
    exactly one four-point digit polyline.  Positions are divided by each
    digit's full wrist/palm-to-tip chain length, making temporal deltas less
    sensitive to the human/robot bone-length mismatch.
    """
    joints = np.asarray(joint_positions, dtype=np.float64)
    rotations = np.asarray(palm_rotations, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError("MANO21 joint positions must have shape (T,21,3)")
    if rotations.shape != (len(joints), 3, 3):
        raise ValueError("palm rotations must have shape (T,3,3)")
    if not np.isfinite(joints).all() or not np.isfinite(rotations).all():
        raise ValueError("MANO21 features require finite positions and rotations")
    if not np.allclose(
        np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1.0e-4,
    ):
        raise ValueError("palm rotations are not orthonormal")

    palm_center = (
        joints[:, 0] + sum(joints[:, index] for index in MANO21_MCP_INDICES)
    ) / 5.0
    normalized_tip = np.zeros((len(joints), 5, 3), dtype=np.float64)
    distal_direction = np.zeros_like(normalized_tip)
    total_curvature = np.zeros((len(joints), 5), dtype=np.float64)
    chain_length = np.zeros((len(joints), 5), dtype=np.float64)

    for finger, indices in enumerate(MANO21_FINGER_INDICES):
        digit = joints[:, indices]
        polyline = np.concatenate([palm_center[:, None], digit], axis=1)
        segments = np.diff(polyline, axis=1)
        lengths = np.linalg.norm(segments, axis=-1)
        if (lengths <= 1.0e-8).any():
            raise ValueError(f"MANO21 {FINGER_NAMES[finger]} contains a zero-length bone")
        unit = segments / lengths[..., None]
        full_length = lengths.sum(axis=1)
        # For row-vector points, R.T @ v is einsum("tji,tj->ti", R, v).
        tip_local = np.einsum(
            "tji,tj->ti", rotations, digit[:, -1] - palm_center,
        )
        distal_local = np.einsum("tji,tj->ti", rotations, unit[:, -1])
        bend_cosine = np.einsum("tki,tki->tk", unit[:, :-1], unit[:, 1:])
        normalized_tip[:, finger] = tip_local / full_length[:, None]
        distal_direction[:, finger] = distal_local / np.linalg.norm(
            distal_local, axis=1, keepdims=True,
        )
        total_curvature[:, finger] = np.arccos(
            np.clip(bend_cosine, -1.0, 1.0),
        ).sum(axis=1)
        chain_length[:, finger] = full_length

    return Mano21Features(
        palm_center=palm_center,
        normalized_tip=normalized_tip,
        distal_direction=distal_direction,
        total_curvature_rad=total_curvature,
        chain_length_m=chain_length,
    )


def rotation_between_vectors(first: np.ndarray, second: np.ndarray) -> Rotation:
    """Return the minimum rotation mapping unit ``first`` onto ``second``."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("vector rotation requires two 3-vectors")
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    if min(first_norm, second_norm) <= 1.0e-10:
        raise ValueError("cannot rotate a zero-length vector")
    first, second = first / first_norm, second / second_norm
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    cross = np.cross(first, second)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > 1.0e-10:
        return Rotation.from_rotvec(cross / cross_norm * np.arctan2(cross_norm, dot))
    if dot > 0.0:
        return Rotation.identity()
    # Antiparallel vectors have infinitely many solutions.  Pick a stable axis
    # orthogonal to the least-aligned coordinate basis.
    basis = np.eye(3)[int(np.argmin(np.abs(first)))]
    axis = np.cross(first, basis)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * np.pi)


def _joint_bounds(model: mujoco.MjModel, addresses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint_by_address = {
        int(model.jnt_qposadr[joint]): joint for joint in range(model.njnt)
    }
    lower, upper = [], []
    for address in addresses.tolist():
        joint = joint_by_address.get(int(address))
        if joint is None or not bool(model.jnt_limited[joint]):
            raise ValueError(f"XHand digit qpos address {address} lacks a finite joint limit")
        lower.append(float(model.jnt_range[joint, 0]))
        upper.append(float(model.jnt_range[joint, 1]))
    return np.asarray(lower), np.asarray(upper)


class FullHandPriorMapper:
    """Solve a bounded XHand trajectory from complete MANO21 temporal shape."""

    _DIGIT_JOINT_NAMES = {
        "thumb": (
            "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
        ),
        "index": (
            "right_hand_index_bend_joint", "right_hand_index_joint1",
            "right_hand_index_joint2",
        ),
        "middle": ("right_hand_mid_joint1", "right_hand_mid_joint2"),
        "ring": ("right_hand_ring_joint1", "right_hand_ring_joint2"),
        "pinky": ("right_hand_pinky_joint1", "right_hand_pinky_joint2"),
    }
    _DIGIT_BODY_NAMES = {
        "thumb": (
            "right_hand_thumb_bend_link", "right_hand_thumb_rota_link1",
            "right_hand_thumb_rota_link2",
        ),
        "index": (
            "right_hand_index_bend_link", "right_hand_index_rota_link1",
            "right_hand_index_rota_link2",
        ),
        "middle": ("right_hand_mid_link1", "right_hand_mid_link2"),
        "ring": ("right_hand_ring_link1", "right_hand_ring_link2"),
        "pinky": ("right_hand_pinky_link1", "right_hand_pinky_link2"),
    }

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.palm_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_palm",
        )
        if self.palm_site < 0:
            raise ValueError("XHand model lacks right_palm")
        self.tip_sites: list[int] = []
        self.addresses: list[np.ndarray] = []
        self.body_ids: list[np.ndarray] = []
        self.bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for name in FINGER_NAMES:
            site = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"right_{name}_tip",
            )
            joints = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                for joint_name in self._DIGIT_JOINT_NAMES[name]
            ]
            bodies = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for body_name in self._DIGIT_BODY_NAMES[name]
            ]
            if site < 0 or min(joints) < 0 or min(bodies) < 0:
                raise ValueError(f"XHand model lacks complete {name} kinematics")
            addresses = np.asarray(
                [int(model.jnt_qposadr[joint]) for joint in joints], dtype=np.int64,
            )
            self.tip_sites.append(site)
            self.addresses.append(addresses)
            self.body_ids.append(np.asarray(bodies, dtype=np.int64))
            self.bounds.append(_joint_bounds(model, addresses))

    def _kinematics(
        self, qpos: np.ndarray, finger: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        palm_rotation = self.data.site_xmat[self.palm_site].reshape(3, 3)
        palm_position = self.data.site_xpos[self.palm_site]
        tip_position = self.data.site_xpos[self.tip_sites[finger]]
        origins = self.data.xpos[self.body_ids[finger]]
        polyline = np.concatenate([
            palm_position[None], origins, tip_position[None],
        ])
        chain_length = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
        tip_local = palm_rotation.T @ (tip_position - palm_position)
        distal_local = palm_rotation.T @ (tip_position - origins[-1])
        distal_norm = float(np.linalg.norm(distal_local))
        if chain_length <= 1.0e-8 or distal_norm <= 1.0e-8:
            raise ValueError(f"XHand {FINGER_NAMES[finger]} has degenerate kinematics")
        return tip_local, distal_local / distal_norm, chain_length

    @staticmethod
    def _flexion_components(finger: int, width: int) -> np.ndarray:
        # Thumb joint 0 and index joint 0 primarily provide opposition/spread;
        # their remaining joints, and all two-DoF digits, provide flexion.
        return np.arange(1, width) if finger < 2 else np.arange(width)

    def solve(
        self, *, baseline_qpos: np.ndarray, mano21: np.ndarray,
        palm_rotations: np.ndarray, anchor_frame: int, start_frame: int,
        end_frame: int, motion_scales: np.ndarray,
        max_normalized_tip_delta: float = 0.35,
    ) -> FullHandPriorTrajectory:
        """Solve every digit around a robot grasp baseline in both time directions."""
        baseline = np.asarray(baseline_qpos, dtype=np.float64)
        scales = np.asarray(motion_scales, dtype=np.float64)
        if baseline.shape != (self.model.nq,) or not np.isfinite(baseline).all():
            raise ValueError("full-hand prior baseline must be one finite model qpos")
        if scales.shape != (5,) or not np.isfinite(scales).all() or (scales < 0.0).any():
            raise ValueError("full-hand prior requires five finite non-negative motion scales")
        if not 0 <= start_frame <= anchor_frame <= end_frame < len(mano21):
            raise ValueError("full-hand prior frame interval does not contain its anchor")
        if not 0.0 < max_normalized_tip_delta <= 1.0:
            raise ValueError("normalized tip displacement cap must be in (0,1]")

        features = mano21_features(mano21, palm_rotations)
        qpos = np.repeat(baseline[None], len(mano21), axis=0)
        desired_tip = np.full((len(mano21), 5, 3), np.nan, dtype=np.float64)
        desired_direction = np.full_like(desired_tip, np.nan)
        solved_tip = np.full_like(desired_tip, np.nan)
        solved_direction = np.full_like(desired_tip, np.nan)
        position_residual = np.full((len(mano21), 5), np.nan, dtype=np.float64)
        direction_residual = np.full_like(position_residual, np.nan)
        success = np.zeros((len(mano21), 5), dtype=bool)
        evaluations = np.zeros((len(mano21), 5), dtype=np.int64)

        baseline_kinematics = [
            self._kinematics(baseline, finger) for finger in range(5)
        ]

        def solve_frame(frame: int, previous: list[np.ndarray]) -> list[np.ndarray]:
            result = baseline.copy()
            next_previous: list[np.ndarray] = []
            for finger, addresses in enumerate(self.addresses):
                base_tip, base_direction, robot_length = baseline_kinematics[finger]
                scale = float(scales[finger])
                normalized_delta = (
                    features.normalized_tip[frame, finger]
                    - features.normalized_tip[anchor_frame, finger]
                ) * scale
                delta_norm = float(np.linalg.norm(normalized_delta))
                if delta_norm > max_normalized_tip_delta:
                    normalized_delta *= max_normalized_tip_delta / delta_norm
                target_tip = base_tip + robot_length * normalized_delta
                human_rotation = rotation_between_vectors(
                    features.distal_direction[anchor_frame, finger],
                    features.distal_direction[frame, finger],
                )
                target_direction = Rotation.from_rotvec(
                    human_rotation.as_rotvec() * scale,
                ).apply(base_direction)
                target_direction /= np.linalg.norm(target_direction)
                flexion = self._flexion_components(finger, len(addresses))
                target_flexion_sum = float(
                    baseline[addresses][flexion].sum()
                    + scale * (
                        features.total_curvature_rad[frame, finger]
                        - features.total_curvature_rad[anchor_frame, finger]
                    )
                )
                lower, upper = self.bounds[finger]
                start = np.clip(previous[finger], lower, upper)

                def residual(value: np.ndarray) -> np.ndarray:
                    trial = baseline.copy()
                    trial[addresses] = value
                    tip, direction, _ = self._kinematics(trial, finger)
                    return np.concatenate([
                        tip - target_tip,
                        0.015 * (direction - target_direction),
                        np.asarray([0.004 * (value[flexion].sum() - target_flexion_sum)]),
                        0.001 * (value - baseline[addresses]),
                        0.002 * (value - previous[finger]),
                    ])

                solution = least_squares(
                    residual, start, bounds=(lower, upper), max_nfev=80,
                    ftol=1.0e-9, xtol=1.0e-9, gtol=1.0e-9,
                )
                result[addresses] = solution.x
                tip, direction, _ = self._kinematics(result, finger)
                desired_tip[frame, finger] = target_tip
                desired_direction[frame, finger] = target_direction
                solved_tip[frame, finger] = tip
                solved_direction[frame, finger] = direction
                position_residual[frame, finger] = np.linalg.norm(tip - target_tip)
                direction_residual[frame, finger] = np.arccos(np.clip(
                    float(np.dot(direction, target_direction)), -1.0, 1.0,
                ))
                success[frame, finger] = bool(solution.success)
                evaluations[frame, finger] = int(solution.nfev)
                next_previous.append(solution.x.copy())
            qpos[frame] = result
            return next_previous

        anchor_values = [baseline[addresses].copy() for addresses in self.addresses]
        forward = solve_frame(anchor_frame, anchor_values)
        for frame in range(anchor_frame + 1, end_frame + 1):
            forward = solve_frame(frame, forward)
        backward = [baseline[addresses].copy() for addresses in self.addresses]
        for frame in range(anchor_frame - 1, start_frame - 1, -1):
            backward = solve_frame(frame, backward)

        return FullHandPriorTrajectory(
            qpos=qpos,
            desired_tip_local_m=desired_tip,
            desired_distal_direction_local=desired_direction,
            solved_tip_local_m=solved_tip,
            solved_distal_direction_local=solved_direction,
            position_residual_m=position_residual,
            direction_residual_rad=direction_residual,
            optimizer_success=success,
            optimizer_evaluations=evaluations,
            human_features=features,
        )
