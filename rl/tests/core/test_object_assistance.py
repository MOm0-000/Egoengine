from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video_to_spider.rl.object_assistance import (
    ToolAssistSpec,
    ToolBodyInfo,
    assistance_scale,
    compute_wrench,
    interpolate_target,
    rotation_from_wxyz,
)
from video_to_spider.rl.core.policy import (
    ASSISTED_CRITIC_INPUT_DIM,
    ASSISTED_CRITIC_INPUT_SPEC,
    PolicyBundle,
)
from video_to_spider.rl.core.rollout import make_critic_input
from video_to_spider.rl.core.state_io import (
    ASSISTED_TRAINING_CHECKPOINT_SCHEMA,
    build_training_checkpoint,
    validate_training_checkpoint,
)


def body() -> ToolBodyInfo:
    return ToolBodyInfo(
        body_name="right_object",
        body_id=2,
        joint_id=1,
        qpos_address=0,
        qvel_address=0,
        mass=2.0,
        principal_inertia=(0.2, 0.3, 0.4),
        inertial_position=(0.1, 0.0, 0.0),
        inertial_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        gravity_world=(0.0, 0.0, -10.0),
    )


def test_schedule_edges_are_frozen():
    assert assistance_scale(1) == 1.0
    assert assistance_scale(50) == 1.0
    assert assistance_scale(51) == 149 / 150
    assert assistance_scale(100) == 2 / 3
    assert assistance_scale(150) == 1 / 3
    assert assistance_scale(199) == 1 / 150
    assert assistance_scale(200) == 0.0
    assert assistance_scale(250) == 0.0


def test_interpolation_uses_com_and_world_velocity():
    com, rotation, velocity, omega = interpolate_target(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0]),
        np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
        fraction=0.5,
        control_dt=0.03,
        inertial_position=np.array([0.1, 0.0, 0.0]),
    )
    np.testing.assert_allclose(rotation, rotation_from_wxyz(
        np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
    ).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(omega, [0.0, 0.0, np.pi / 0.06])
    assert com.shape == velocity.shape == (3,)
    assert not np.allclose(com, [0.015, 0.0, 0.0])


def test_pd_direction_damping_caps_and_alpha_scaling():
    qpos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    qvel = np.zeros(6)
    spec = ToolAssistSpec()
    base = compute_wrench(
        alpha=1.0,
        spec=spec,
        body=body(),
        target_com_position=np.array([0.2, 0.0, 0.1]),
        target_rotation_world=rotation_from_wxyz(
            np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0])
        ).as_matrix(),
        target_com_velocity=np.zeros(3),
        target_angular_velocity_world=np.zeros(3),
        live_qpos=qpos,
        live_qvel=qvel,
    )
    half = compute_wrench(
        alpha=0.5,
        spec=spec,
        body=body(),
        target_com_position=np.array([0.2, 0.0, 0.1]),
        target_rotation_world=rotation_from_wxyz(
            np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0])
        ).as_matrix(),
        target_com_velocity=np.zeros(3),
        target_angular_velocity_world=np.zeros(3),
        live_qpos=qpos,
        live_qvel=qvel,
    )
    np.testing.assert_allclose(half.wrench, 0.5 * base.wrench)
    assert base.wrench[0] > 0.0 and base.wrench[2] > 0.0
    assert base.wrench[4] > 0.0
    assert np.linalg.norm(base.wrench[:3]) <= 60.0 + 1e-10
    assert np.linalg.norm(base.wrench[3:]) <= 40.0 + 1e-10


def test_zero_alpha_bypasses_invalid_pose_math():
    result = compute_wrench(
        alpha=0.0,
        spec=ToolAssistSpec(),
        body=body(),
        target_com_position=np.full(3, np.nan),
        target_rotation_world=np.full((3, 3), np.nan),
        target_com_velocity=np.full(3, np.nan),
        target_angular_velocity_world=np.full(3, np.nan),
        live_qpos=np.full(7, np.nan),
        live_qvel=np.full(6, np.nan),
    )
    assert np.array_equal(result.wrench, np.zeros(6))


def test_assisted_critic_appends_distinct_alpha_feature():
    actor = torch.zeros(2, 236)
    extra = torch.zeros(2, 108)
    endpoints = torch.tensor([40, 60])
    alpha = torch.tensor([1.0, 0.25])
    result = make_critic_input(actor, extra, endpoints, assistance_alpha=alpha)
    assert result.shape == (2, ASSISTED_CRITIC_INPUT_DIM)
    assert result[:, -2].tolist() == [0.0, 0.5]
    assert torch.equal(result[:, -1], alpha)
    policy = PolicyBundle.create(
        worlds=1,
        critic_input_dim=ASSISTED_CRITIC_INPUT_DIM,
        critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
    )
    assert policy.critic.model.running_mean_std.running_mean.shape == (
        ASSISTED_CRITIC_INPUT_DIM,
    )


def test_assisted_spec_rejects_legacy_dimension_pair():
    with pytest.raises(ValueError, match="dimension/specification"):
        PolicyBundle.create(
            worlds=1,
            critic_input_dim=ASSISTED_CRITIC_INPUT_DIM,
            critic_input_spec="actor236_privileged108_phase1_v1",
        )


def test_assisted_checkpoint_rejects_legacy_345_critic(monkeypatch):
    import video_to_spider.rl.core.state_io as state_io

    monkeypatch.setattr(state_io, "validate_physics_snapshot", lambda _state: None)
    policy = PolicyBundle.create(
        worlds=1,
        critic_input_dim=ASSISTED_CRITIC_INPUT_DIM,
        critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
    )
    payload = build_training_checkpoint(
        policy=policy,
        boundary_state={},
        observation_prefix=tuple(torch.zeros(1, 236) for _ in range(20)),
        next_epoch=2,
        cost={},
        metadata={},
    )
    assert payload["schema"] == ASSISTED_TRAINING_CHECKPOINT_SCHEMA
    validate_training_checkpoint(
        payload,
        expected_critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
        expected_critic_input_dimension=ASSISTED_CRITIC_INPUT_DIM,
    )
    with pytest.raises(ValueError, match="schema"):
        validate_training_checkpoint(payload)
