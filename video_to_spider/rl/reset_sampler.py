"""Pre-grasp reset-prior sampler for the xHand residual RL environment.

EgoEngine's RL slot should not force the robot to reproduce the human hand pose
during the whole manipulation segment.  H2S2R shows that the human pre-grasp
pose is best used as a *reset seed* for exploration: every training episode
starts near the frame just before the object starts moving, and the policy is
then free to find an xHand-specific grasp while tracking the object trajectory.

This module is deliberately NumPy-only so it stays testable in ``v2s-core`` and
can be shared by the training environment and the final inference runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreGraspSamplerConfig:
    """Parameters for detecting and sampling the pre-grasp window.

    ``object_velocity_threshold`` marks a frame as "object moving".
    ``initial_move_frames_required`` requires a short consecutive run so that a
    single noisy frame does not move the window.
    ``settle_steps_before_motion`` and ``window_steps`` define the sampled
    region ending at the detected first-motion frame.
    """

    object_velocity_threshold: float = 0.05
    initial_move_frames_required: int = 3
    settle_steps_before_motion: int = 10
    window_steps: int = 20
    no_motion_fallback_start: int = 0
    no_motion_fallback_end: int = 10

    def validate(self) -> None:
        if self.object_velocity_threshold < 0.0:
            raise ValueError("object_velocity_threshold must be non-negative")
        if self.initial_move_frames_required < 1:
            raise ValueError("initial_move_frames_required must be positive")
        if self.settle_steps_before_motion < 0:
            raise ValueError("settle_steps_before_motion must be non-negative")
        if self.window_steps < 1:
            raise ValueError("window_steps must be positive")
        if self.no_motion_fallback_start < 0 or self.no_motion_fallback_end < 0:
            raise ValueError("no_motion fallback indices must be non-negative")


class PreGraspResetSampler:
    """Sample per-env reference indices from the pre-grasp reset prior."""

    def __init__(
        self,
        config: PreGraspSamplerConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.cfg = config or PreGraspSamplerConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(seed)

    def detect_first_motion_index(self, object_linear_velocity: np.ndarray) -> int | None:
        """Return the first frame of a sustained object-velocity onset."""
        object_linear_velocity = np.asarray(object_linear_velocity, dtype=np.float64)
        if object_linear_velocity.ndim != 2 or object_linear_velocity.shape[1] != 3:
            raise ValueError("object_linear_velocity must have shape (T, 3)")
        speed = np.linalg.norm(object_linear_velocity, axis=-1)
        moving = speed > self.cfg.object_velocity_threshold
        run = 0
        for i in range(moving.shape[0]):
            run = run + 1 if moving[i] else 0
            if run >= self.cfg.initial_move_frames_required:
                return i - self.cfg.initial_move_frames_required + 1
        return None

    def sample_indices(
        self,
        object_linear_velocity: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """Return ``num_samples`` pre-grasp reference indices."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        object_linear_velocity = np.asarray(object_linear_velocity, dtype=np.float64)
        if object_linear_velocity.ndim != 2 or object_linear_velocity.shape[1] != 3:
            raise ValueError("object_linear_velocity must have shape (T, 3)")
        length = object_linear_velocity.shape[0]
        if length == 0:
            raise ValueError("reference trajectory is empty")

        first_motion = self.detect_first_motion_index(object_linear_velocity)
        if first_motion is None:
            low = min(self.cfg.no_motion_fallback_start, length - 1)
            high = min(self.cfg.no_motion_fallback_end, length - 1)
        else:
            high = min(first_motion, length - 1)
            low = max(0, high - self.cfg.window_steps + 1)
            low = max(low, first_motion - self.cfg.settle_steps_before_motion)
        if low > high:
            low, high = high, low
        return self.rng.integers(int(low), int(high) + 1, size=num_samples).astype(np.int64)


def sample_pre_grasp_indices(
    object_linear_velocity: np.ndarray,
    num_samples: int,
    *,
    config: PreGraspSamplerConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Functional convenience wrapper around :class:`PreGraspResetSampler`."""
    return PreGraspResetSampler(config, seed=seed).sample_indices(
        object_linear_velocity, num_samples
    )
