"""Human2Sim2Robot (H2S2R) RL building blocks adapted to xHand.

This module intentionally keeps all reward, observation, action and
domain-randomization functions in NumPy so the numerical contract is easy to
test and auditable.  It does not pull in PyTorch, IsaacGym or rl-games; the
network/training layers live separately and are instantiated by the eventual
training entry point.

The important adaptation relative to H2S2R is the policy paradigm: H2S2R
outputs ``[palm position, palm orientation, hand PCA]`` for a geometric-fabric
controller.  EgoEngine instead expects a residual on top of an upstream MINK /
reference control signal ``a = a_ref + delta_a``.  We therefore expose
``XHandResidualPolicySpec`` and :func:`apply_residual_action` rather than a
palm/PCA action decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def make_anchor_points(length: float = 0.2) -> np.ndarray:
    """Return the three default object-local anchor offsets from Eq. (2)."""
    if length < 0.0:
        raise ValueError("length must be non-negative")
    return np.asarray(
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        dtype=np.float64,
    )


def make_symmetry_anchor_points(
    length: float = 0.2, symmetry_axis: Iterable[bool] | None = None
) -> np.ndarray:
    """Build anchor points, optionally removing rotation-symmetry redundancies.

    For a cylinder/sphere-style symmetry, anchors orthogonal to the symmetry
    axis measure orientation noise but carry no task information.  Following
    H2S2R Appendix C, those anchors are removed.
    """
    anchors = make_anchor_points(length)
    if symmetry_axis is None:
        return anchors
    axis = np.asarray(list(symmetry_axis), dtype=bool)
    if axis.shape != (3,):
        raise ValueError("symmetry_axis must contain exactly three booleans")
    keep = np.all(anchors[:, ~axis] == 0.0, axis=1)
    if not keep.any():
        raise ValueError("symmetry_axis removes all anchor points")
    return anchors[keep]


def transform_anchor_points(pose: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Transform object-local anchor points by one or more SE(3) poses.

    ``pose`` is either ``(4, 4)`` or ``(..., 4, 4)`` and ``anchors`` is
    ``(N, 3)``.  The returned array has shape ``(..., N, 3)``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if pose.shape[-2:] != (4, 4):
        raise ValueError("pose must have trailing shape (4, 4)")
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must have shape (N, 3)")
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    transformed = rotation @ anchors.T  # (..., 3, N)
    transformed = transformed + translation[..., None]
    return np.moveaxis(transformed, -1, -2)


def relative_pose_distance(
    goal_pose: np.ndarray,
    current_pose: np.ndarray,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Return H2S2R Eq. (2) anchor distance ``d(T_goal, T_obj)``."""
    if anchors is None:
        anchors = make_anchor_points()
    goal_anchors = transform_anchor_points(goal_pose, anchors)
    current_anchors = transform_anchor_points(current_pose, anchors)
    return np.sum(np.linalg.norm(goal_anchors - current_anchors, axis=-1), axis=-1)


@dataclass(frozen=True)
class AnchorRewardConfig:
    alpha: float = 10.0
    length: float = 0.2
    reset_distance_max_m: float = 0.25


def object_anchor_reward(
    goal_pose: np.ndarray,
    current_pose: np.ndarray,
    config: AnchorRewardConfig | None = None,
    *,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Return H2S2R Eq. (1) object-tracking reward ``exp(-alpha * d)``."""
    config = config or AnchorRewardConfig()
    if config.alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if anchors is None:
        anchors = make_anchor_points(config.length)
    distance = relative_pose_distance(goal_pose, current_pose, anchors)
    return np.exp(-config.alpha * distance)


@dataclass(frozen=True)
class XHandResidualPolicySpec:
    """Residual-policy action contract for the xHand embodiment.

    H2S2R uses ``[palm pos, palm euler, hand PCA]``; EgoEngine's RL slot is a
    residual on the upstream reference.  For the 18-DoF xHand we therefore use
    ``delta_u in R^hand_dof`` (joint-level residual in the reference control
    space), optionally clipped and scaled.  Object-guidance actuators are never
    part of the residual policy output.
    """

    hand_dof: int = 18
    residual_clip: float = 0.05
    residual_scale: float = 1.0
    hand_control_indices: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(18))
    )

    def validate(self) -> None:
        if self.hand_dof < 1:
            raise ValueError("hand_dof must be positive")
        if len(self.hand_control_indices) != self.hand_dof:
            raise ValueError("hand_control_indices length must equal hand_dof")
        if self.residual_clip < 0.0 or self.residual_scale <= 0.0:
            raise ValueError("residual_clip must be non-negative and residual_scale positive")


def apply_residual_action(
    reference_ctrls: np.ndarray,
    residual_delta: np.ndarray,
    spec: XHandResidualPolicySpec | None = None,
) -> np.ndarray:
    """Apply ``a_ref + clip(scale * delta)`` to the xHand control dimensions."""
    spec = spec or XHandResidualPolicySpec()
    spec.validate()
    reference = np.asarray(reference_ctrls, dtype=np.float64)
    delta = np.asarray(residual_delta, dtype=np.float64)
    if reference.shape[-1] < spec.hand_dof:
        raise ValueError("reference control dimension is smaller than hand_dof")
    if delta.shape[-1] != spec.hand_dof:
        raise ValueError("residual_delta last dimension must equal hand_dof")
    out = reference.copy()
    residual = np.clip(
        spec.residual_scale * delta, -spec.residual_clip, spec.residual_clip
    )
    out[..., list(spec.hand_control_indices)] += residual
    return out


def build_xhand_observation(
    *,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    fingertip_positions: np.ndarray,
    palm_position: np.ndarray,
    object_anchor_positions: np.ndarray,
    goal_anchor_positions: np.ndarray,
) -> np.ndarray:
    """Flatten the H2S2R Eq. (3) observation, reordered for xHand proprioception."""
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    fingertip_positions = np.asarray(fingertip_positions, dtype=np.float64)
    palm_position = np.asarray(palm_position, dtype=np.float64)
    object_anchor_positions = np.asarray(object_anchor_positions, dtype=np.float64)
    goal_anchor_positions = np.asarray(goal_anchor_positions, dtype=np.float64)
    if joint_positions.shape[-1] != joint_velocities.shape[-1]:
        raise ValueError("joint position and velocity dimensions differ")
    if fingertip_positions.shape[-2:] != (5, 3):
        raise ValueError("fingertip_positions must end with (5, 3)")
    if palm_position.shape[-1] != 3:
        raise ValueError("palm_position must end with dimension 3")
    if object_anchor_positions.shape[-2:] != goal_anchor_positions.shape[-2:]:
        raise ValueError("object and goal anchor positions must have the same trailing shape")
    return np.concatenate(
        [
            joint_positions,
            joint_velocities,
            fingertip_positions.reshape(*fingertip_positions.shape[:-2], 15),
            palm_position,
            object_anchor_positions.reshape(*object_anchor_positions.shape[:-2], -1),
            goal_anchor_positions.reshape(*goal_anchor_positions.shape[:-2], -1),
        ],
        axis=-1,
    )


def build_privileged_state(
    *,
    object_linear_velocity: np.ndarray,
    object_angular_velocity: np.ndarray,
    joint_forces: np.ndarray,
    fingertip_contact_forces: np.ndarray,
) -> np.ndarray:
    """Flatten the privileged critic state (velocities, forces, contacts)."""
    object_linear_velocity = np.asarray(object_linear_velocity, dtype=np.float64)
    object_angular_velocity = np.asarray(object_angular_velocity, dtype=np.float64)
    joint_forces = np.asarray(joint_forces, dtype=np.float64)
    fingertip_contact_forces = np.asarray(fingertip_contact_forces, dtype=np.float64)
    if object_linear_velocity.shape[-1] != 3 or object_angular_velocity.shape[-1] != 3:
        raise ValueError("object velocity arrays must end with dimension 3")
    return np.concatenate(
        [
            object_linear_velocity,
            object_angular_velocity,
            joint_forces,
            fingertip_contact_forces.reshape(*fingertip_contact_forces.shape[:-2], -1),
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class H2S2RTrainingConfig:
    """PPO/LSTM configuration copied from H2S2R Section F.

    Network sizes and optimizer values are verbatim H2S2R defaults; the
    observation/action dimensions are deferred to the xHand adapters above.
    """

    num_envs: int = 4096
    horizon_length: int = 16
    minibatch_size: int = 8192
    mini_epochs: int = 4
    learning_rate: float = 5e-4
    gamma: float = 0.998
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    actor_hidden: tuple[int, ...] = (512, 512)
    lstm_hidden: int = 1024
    critic_hidden: tuple[int, ...] = (1024, 512)
    normalize_observations: bool = True
    normalize_value: bool = True
    normalize_advantages: bool = True
    control_dt: float = 1.0 / 15.0
    sim_substeps: int = 8

    def validate(self) -> None:
        if self.minibatch_size % self.num_envs != 0:
            raise ValueError("minibatch_size must be divisible by num_envs")
        if self.horizon_length < 1 or self.mini_epochs < 1:
            raise ValueError("horizon_length and mini_epochs must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.clip_param <= 0.0 or self.entropy_coef < 0.0:
            raise ValueError("clip_param must be positive and entropy_coef non-negative")


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """H2S2R Section F domain-randomization schedule."""

    resample_every_steps: int = 720
    obs_noise_std: float = 0.01
    action_noise_std: float = 0.01
    gravity_noise_std: float = 0.3
    scale_range: tuple[float, float] = (0.7, 1.3)
    random_force_prob: float = 0.05
    random_force_mass_multiplier: float = 50.0

    def validate(self) -> None:
        if self.resample_every_steps < 1:
            raise ValueError("resample_every_steps must be positive")
        if self.obs_noise_std < 0.0 or self.action_noise_std < 0.0:
            raise ValueError("noise standard deviations must be non-negative")
        if self.gravity_noise_std < 0.0:
            raise ValueError("gravity_noise_std must be non-negative")
        if self.scale_range[0] <= 0.0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range must be positive and ordered")
        if not 0.0 <= self.random_force_prob <= 1.0:
            raise ValueError("random_force_prob must be in [0, 1]")


def sample_domain_randomization(
    rng: np.random.Generator,
    config: DomainRandomizationConfig | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Draw one DR instance following H2S2R Section F."""
    config = config or DomainRandomizationConfig()
    config.validate()
    low, high = config.scale_range
    return {
        "object_scale": float(rng.uniform(low, high)),
        "object_mass_scale": float(rng.uniform(low, high)),
        "object_friction_scale": float(rng.uniform(low, high)),
        "table_scale": float(rng.uniform(low, high)),
        "table_mass_scale": float(rng.uniform(low, high)),
        "table_friction_scale": float(rng.uniform(low, high)),
        "robot_scale": float(rng.uniform(low, high)),
        "robot_mass_scale": float(rng.uniform(low, high)),
        "robot_damping_scale": float(rng.uniform(low, high)),
        "robot_stiffness_scale": float(rng.uniform(low, high)),
        "robot_friction_scale": float(rng.uniform(low, high)),
        "gravity_noise": rng.normal(0.0, config.gravity_noise_std, size=3),
        "apply_random_force": bool(rng.random() < config.random_force_prob),
        "random_force_mass_multiplier": float(config.random_force_mass_multiplier),
    }
