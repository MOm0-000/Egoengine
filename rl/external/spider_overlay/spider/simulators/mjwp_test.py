# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types

import torch


def test_imports():
    import spider
    from spider.simulators import mjwp
    assert isinstance(spider.ROOT, str)
    assert isinstance(mjwp, types.ModuleType)


def test_contact_tracking_reward_applies_configured_scale():
    from spider.simulators.mjwp import _contact_tracking_reward

    positions = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
    reference = torch.zeros((2, 3))
    active = torch.tensor([1.0, 0.0])

    reward = _contact_tracking_reward(positions, reference, active, scale=5.0)

    torch.testing.assert_close(reward, torch.tensor([-25.0]))


def test_contact_tracking_reward_adds_thumb_opposition_bonus():
    from spider.simulators.mjwp import _contact_tracking_reward

    positions = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0],
                               [0.0, 0.0, 9.0], [0.0, 0.0, 9.0],
                               [0.0, 0.0, 9.0]]])
    reference = torch.zeros((5, 3))
    active = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])

    reward = _contact_tracking_reward(
        positions, reference, active, scale=1.0, opposition_scale=3.0
    )

    torch.testing.assert_close(reward, torch.tensor([-24.5]))


def test_paper_contact_bonus_uses_live_thumb_and_other_contact_only():
    from spider.simulators.mjwp import _paper_contact_bonus_from_tensors

    inputs = _synthetic_force_closure_inputs()
    reward, score = _paper_contact_bonus_from_tensors(
        geom=inputs["geom"],
        dist=inputs["dist"],
        worldid=inputs["worldid"],
        efc_address=inputs["efc_address"],
        efc_force=inputs["efc_force"],
        geom_finger_map=inputs["geom_finger_map"],
        geom_object_group_map=inputs["geom_object_group_map"],
        num_fingers=5,
        pyramidal_cone=True,
        reward_scale=2.0,
    )

    # World 0 has thumb + index contact; world 1 has thumb only.  Upstream
    # contact targets are intentionally absent from this API.
    torch.testing.assert_close(reward, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(score, torch.tensor([1.0, 0.0]))


def test_paper_pose_reward_separates_object_and_human_terms():
    from spider.config import Config
    from spider.simulators.mjwp import _paper_pose_rewards

    config = Config(
        embodiment_type="right", nq_obj=7, nu=8,
        pos_rew_scale=1.0, rot_rew_scale=0.25,
        base_pos_rew_scale=0.2, base_rot_rew_scale=1.0,
        joint_rew_scale=0.0,
        object_pos_threshold=0.08, object_rot_threshold=2.5,
    )
    diff = torch.zeros((1, 14))
    diff[:, 0] = 0.1       # hand base error
    diff[:, -6] = 0.03     # object position error
    diff[:, -3] = 0.4      # object geodesic rotation error

    obj, human, error = _paper_pose_rewards(config, diff)

    expected_error = torch.tensor([(0.03**2 + 0.25 * 0.4**2) ** 0.5])
    constant = (0.08**2 + 0.25 * 2.5**2) ** 0.5
    torch.testing.assert_close(error, expected_error)
    torch.testing.assert_close(obj, torch.tensor([constant]) - expected_error)
    torch.testing.assert_close(human, torch.tensor([-0.2 * 0.1**2]))


def _synthetic_force_closure_inputs():
    # geom 1=thumb, 2=index, 3=paired object.  EFC addresses are relative to
    # each MJWarp world, matching the real batched contact layout.
    geom = torch.tensor([[1, 3], [3, 2], [1, 3], [0, 0]])
    dist = torch.tensor([-0.001, -0.001, -0.001, 0.0])
    frame = torch.eye(3).repeat(4, 1, 1)
    # contact 0 normal hand->object is +x.  Contact 1 arrives in reversed geom
    # order, so its object->hand +x normal becomes hand->object -x.
    worldid = torch.tensor([0, 0, 1, 1])
    efc_address = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    efc_force = torch.zeros((2, 12))
    efc_force[0, 0] = 1.0
    efc_force[0, 6] = 1.0
    efc_force[1, 0] = 1.0
    return {
        "geom": geom,
        "dist": dist,
        "frame": frame,
        "worldid": worldid,
        "efc_address": efc_address,
        "efc_force": efc_force,
        "geom_finger_map": torch.tensor([-1, 0, 1, -1]),
        "geom_hand_group_map": torch.tensor([-1, 0, 0, -1]),
        "geom_object_group_map": torch.tensor([-1, -1, -1, 0]),
        "contact_ref": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        "pyramidal_cone": True,
        "reward_scale": 10.0,
        "penetration_reward_scale": 5.0,
        "min_normal_force_n": 0.2,
        "min_opposition_cosine": 0.2,
        "max_penetration_m": 0.003,
    }


def test_force_closure_constraint_uses_real_force_and_oriented_normals():
    from spider.simulators.mjwp import _force_closure_constraint_from_tensors

    reward, score, penetration = _force_closure_constraint_from_tensors(
        **_synthetic_force_closure_inputs()
    )

    # World 0 has force-bearing opposed thumb/index contacts, including a
    # reversed geom order.  World 1 has only the thumb contact and must fail.
    torch.testing.assert_close(reward, torch.tensor([10.0, 0.0]))
    torch.testing.assert_close(score, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(penetration, torch.tensor([0.001, 0.001]))


def test_force_closure_constraint_penalizes_deep_hand_object_penetration():
    from spider.simulators.mjwp import _force_closure_constraint_from_tensors

    inputs = _synthetic_force_closure_inputs()
    inputs["dist"] = inputs["dist"].clone()
    inputs["dist"][0] = -0.006
    reward, score, penetration = _force_closure_constraint_from_tensors(**inputs)

    torch.testing.assert_close(score[0], torch.tensor(1.0))
    torch.testing.assert_close(penetration[0], torch.tensor(0.006))
    # +10 force-closure bonus and -5 excess-penetration penalty.
    torch.testing.assert_close(reward[0], torch.tensor(5.0))
