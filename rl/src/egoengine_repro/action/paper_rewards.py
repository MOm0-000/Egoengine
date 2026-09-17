"""EgoEngine Eq. (2), Appendix C.1-C.13, without inferred paper coefficients.

Tensor functions keep all leading batch/hand/object axes. Reducing two objects
to one score is an experimental protocol decision, not part of these equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def _nonnegative(**values):
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f"expected finite nonnegative coefficients: {values}")


def _matching(a, b, tail):
    if a.shape != b.shape or tuple(a.shape[-len(tail):]) != tail:
        raise ValueError(f"matching tensor shapes ending in {tail} are required")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite state/reference")


def so3_distance(rotation, reference):
    """C.2: acos((trace(R_ref.T R) - 1) / 2), in radians."""
    _matching(rotation, reference, (3, 3))
    trace = (rotation * reference).sum(dim=(-2, -1))
    return torch.acos(((trace - 1) / 2).clamp(-1, 1))


@dataclass(frozen=True)
class TrackingObjective:
    lambda_p: float
    lambda_r: float
    boundary: float

    def __post_init__(self):
        _nonnegative(lambda_p=self.lambda_p, lambda_r=self.lambda_r, boundary=self.boundary)
        if self.boundary == 0 or self.lambda_p + self.lambda_r == 0:
            raise ValueError("a positive boundary and at least one tracking weight are required")


@dataclass(frozen=True)
class TrackingScore:
    position_error: torch.Tensor
    rotation_error: torch.Tensor
    error: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor


def object_tracking(position, rotation, reference_position, reference_rotation, objective):
    """C.1-C.4/C.9. Raw C-e is negative outside the valid regime.

    The boundary-violating step is not a valid step in C.10-C.12 and must not
    contribute to accumulated evaluation reward. No exponential or auxiliary
    term is hidden in this score.
    """
    _matching(position, reference_position, (3,))
    if position.shape[:-1] != rotation.shape[:-2]:
        raise ValueError("position and rotation leading dimensions differ")
    ep = torch.linalg.vector_norm(position - reference_position, dim=-1)
    er = so3_distance(rotation, reference_rotation)
    error = torch.sqrt(objective.lambda_p * ep.square() + objective.lambda_r * er.square())
    return TrackingScore(ep, er, error, objective.boundary - error, error > objective.boundary)


def human_mimic(position, rotation, joints, reference_position, reference_rotation,
                reference_joints, *, beta_x, beta_r, beta_q):
    """C.5. Inputs are wrist poses and finger joints, not all scene qpos."""
    _nonnegative(beta_x=beta_x, beta_r=beta_r, beta_q=beta_q)
    _matching(position, reference_position, (3,))
    _matching(joints, reference_joints, (joints.shape[-1],))
    if position.shape[:-1] != joints.shape[:-1] or position.shape[:-1] != rotation.shape[:-2]:
        raise ValueError("wrist and finger leading dimensions differ")
    return -(beta_x * (position - reference_position).square().sum(-1)
             + beta_r * so3_distance(rotation, reference_rotation).square()
             + beta_q * (joints - reference_joints).square().sum(-1))


def action_smoothness(action, previous_action):
    """C.6: penalize the total executed command difference, not residual alone."""
    _matching(action, previous_action, (action.shape[-1],))
    return -(action - previous_action).square().sum(-1)


def opposition_contact(finger_contacts, *, coefficient):
    """C.7. Last axis is thumb/index/middle/ring/pinky for ONE hand-object pair.

    Contacts from different hands or different objects must not be pooled.
    Callers must supply actual physical contacts, not distance/GT proxies.
    """
    _nonnegative(coefficient=coefficient)
    if finger_contacts.dtype != torch.bool or finger_contacts.shape[-1] != 5:
        raise ValueError("expected a boolean (...,5) physical-contact tensor")
    return coefficient * (finger_contacts[..., 0] & finger_contacts[..., 1:].any(-1))


def lifting(height, initial_height, *, lambda_z):
    """C.8 has no positive-part clamp; lowering the object gives negative reward."""
    _nonnegative(lambda_z=lambda_z)
    return lambda_z * (height - initial_height)


def taco_optimization_reward(object_reward, contact_reward, lift_reward=0.0):
    """C.2 TACO profile: human mimic and smoothness are disabled."""
    return object_reward + contact_reward + lift_reward


@dataclass(frozen=True)
class TrajectoryEvaluation:
    horizon: int
    valid_steps: int
    normalized_reward: float
    simulation_steps: int

    def __post_init__(self):
        if (not isinstance(self.horizon, int) or self.horizon < 1
                or not isinstance(self.valid_steps, int) or not 0 <= self.valid_steps <= self.horizon
                or not isinstance(self.simulation_steps, int) or self.simulation_steps < 0
                or not math.isfinite(self.normalized_reward)
                or not 0 <= self.normalized_reward <= self.valid_steps / self.horizon + 1e-12):
            raise ValueError("invalid C.3 trajectory summary")


def evaluate_errors(errors, *, boundary, horizon, simulation_steps):
    """Accumulate only the valid prefix; early termination cannot later recover."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive control-step count")
    if not math.isfinite(boundary) or boundary <= 0:
        raise ValueError("boundary must be finite and positive")
    errors = np.asarray(errors, dtype=float)
    if (errors.ndim != 1 or len(errors) > horizon or not np.isfinite(errors).all()
            or (errors < 0).any()):
        raise ValueError("expected finite nonnegative errors within the reference horizon")
    invalid = np.flatnonzero(errors > boundary)
    valid = int(invalid[0]) if len(invalid) else len(errors)
    reward = float(np.sum(boundary - errors[:valid]))
    return TrajectoryEvaluation(horizon, valid, reward / (boundary * horizon), simulation_steps)


def aggregate_evaluations(trajectories):
    """C.10-C.13; Cost is undefined (None), not zero, if nothing succeeded."""
    if not trajectories:
        raise ValueError("at least one evaluated trajectory is required")
    successes = [r for r in trajectories if r.valid_steps == r.horizon]
    return dict(
        SR=len(successes) / len(trajectories),
        Step=float(np.mean([r.valid_steps / r.horizon for r in trajectories])),
        Reward=float(np.mean([r.normalized_reward for r in trajectories])),
        Cost=(sum(r.simulation_steps for r in successes) / sum(r.horizon for r in successes)
              if successes else None),
    )
