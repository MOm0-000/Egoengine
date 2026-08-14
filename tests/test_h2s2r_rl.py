import numpy as np
import pytest

from video_to_spider.rl import (
    AnchorRewardConfig,
    DomainRandomizationConfig,
    H2S2RTrainingConfig,
    XHandResidualPolicySpec,
    apply_residual_action,
    build_privileged_state,
    build_xhand_observation,
    make_anchor_points,
    make_symmetry_anchor_points,
    object_anchor_reward,
    relative_pose_distance,
    sample_domain_randomization,
    PreGraspResetSampler,
    PreGraspSamplerConfig,
    sample_pre_grasp_indices,
)


def _pose(translation, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_make_anchor_points_default():
    anchors = make_anchor_points(0.2)
    assert anchors.shape == (3, 3)
    np.testing.assert_allclose(anchors, 0.2 * np.eye(3))


def test_make_symmetry_anchor_points_keeps_axis_anchor():
    anchors = make_symmetry_anchor_points(0.2, [False, False, True])
    np.testing.assert_allclose(anchors, np.asarray([[0.0, 0.0, 0.2]]))


def test_relative_pose_distance_identity_is_zero():
    identity = np.eye(4)
    assert relative_pose_distance(identity, identity) == 0.0


def test_relative_pose_distance_translation_matches_3_times_length():
    goal = _pose([0.0, 0.0, 0.0], np.eye(3))
    current = _pose([0.2, 0.0, 0.0], np.eye(3))
    # Each anchor moves by 0.2 m along x, so total distance is 0.6.
    np.testing.assert_allclose(relative_pose_distance(goal, current), 0.6)


def test_object_anchor_reward_exp_decay():
    identity = np.eye(4)
    current = _pose([0.1, 0.0, 0.0], np.eye(3))
    reward = object_anchor_reward(identity, current, AnchorRewardConfig(alpha=10.0))
    np.testing.assert_allclose(reward, np.exp(-10.0 * 0.3), rtol=1e-8)


def test_apply_residual_action_clips_and_scales():
    spec = XHandResidualPolicySpec(
        hand_dof=3,
        residual_clip=0.05,
        residual_scale=2.0,
        hand_control_indices=(0, 1, 2),
    )
    reference = np.asarray([[0.0, 0.0, 0.0]])
    delta = np.asarray([[0.10, -0.10, 0.01]])
    out = apply_residual_action(reference, delta, spec)
    np.testing.assert_allclose(out, [[0.05, -0.05, 0.02]])


def test_build_xhand_observation_shape():
    observation = build_xhand_observation(
        joint_positions=np.zeros((18,)),
        joint_velocities=np.zeros((18,)),
        fingertip_positions=np.zeros((5, 3)),
        palm_position=np.zeros((3,)),
        object_anchor_positions=np.zeros((3, 3)),
        goal_anchor_positions=np.zeros((3, 3)),
    )
    assert observation.shape == (18 + 18 + 15 + 3 + 9 + 9,)


def test_build_privileged_state_shape():
    privileged = build_privileged_state(
        object_linear_velocity=np.zeros(3),
        object_angular_velocity=np.zeros(3),
        joint_forces=np.zeros(18),
        fingertip_contact_forces=np.zeros((5, 3)),
    )
    assert privileged.shape == (3 + 3 + 18 + 15,)


def test_training_config_validation():
    H2S2RTrainingConfig().validate()
    with pytest.raises(ValueError):
        H2S2RTrainingConfig(minibatch_size=1, num_envs=2).validate()


def test_sample_domain_randomization_values():
    rng = np.random.default_rng(0)
    config = DomainRandomizationConfig()
    sample = sample_domain_randomization(rng, config)
    assert sample["object_scale"] >= config.scale_range[0]
    assert sample["object_scale"] <= config.scale_range[1]
    assert sample["gravity_noise"].shape == (3,)
    assert isinstance(sample["apply_random_force"], bool)


def test_pre_grasp_sampler_detects_sustained_motion():
    cfg = PreGraspSamplerConfig(
        object_velocity_threshold=0.05,
        initial_move_frames_required=3,
        settle_steps_before_motion=10,
        window_steps=20,
    )
    vel = np.zeros((30, 3))
    vel[10:15] = 0.1
    sampler = PreGraspResetSampler(cfg, seed=0)
    assert sampler.detect_first_motion_index(vel) == 10
    indices = sampler.sample_indices(vel, 100)
    assert indices.shape == (100,)
    assert indices.min() >= 0
    assert indices.max() <= 10


def test_pre_grasp_sampler_no_motion_fallback():
    cfg = PreGraspSamplerConfig(no_motion_fallback_start=0, no_motion_fallback_end=5)
    vel = np.zeros((30, 3))
    indices = sample_pre_grasp_indices(vel, 20, config=cfg, seed=1)
    assert indices.min() >= 0
    assert indices.max() <= 5


def test_pre_grasp_sampler_rejects_bad_input():
    sampler = PreGraspResetSampler()
    with pytest.raises(ValueError):
        sampler.sample_indices(np.zeros((3,)), 2)
    with pytest.raises(ValueError):
        sampler.sample_indices(np.zeros((5, 3)), 0)
