"""General, video-independent tests for contact-aware xHand preshaping."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from spider.preprocess.grasp_preshape import (
    _finger_priority,
    _mpc_contact_handoff,
    _set_xhand_wrist_pose,
    _trajectory_floor_lift_profile,
)


def _wrist_model() -> mujoco.MjModel:
    xml = """
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body>
          <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
          <joint name="R_forearm_tx_link_joint" type="slide" axis="1 0 0"
                 limited="true" range="-2 2"/>
          <body>
            <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
            <joint name="R_forearm_ty_link_joint" type="slide" axis="0 1 0"
                   limited="true" range="-2 2"/>
            <body>
              <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
              <joint name="R_forearm_tz_link_joint" type="slide" axis="0 0 1"
                     limited="true" range="-2 2"/>
              <body>
                <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                <joint name="R_forearm_roll_link_joint" type="hinge" axis="0 0 1"
                       limited="true" range="-6.2 6.2"/>
                <body>
                  <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                  <joint name="R_forearm_pitch_link_joint" type="hinge" axis="1 0 0"
                         limited="true" range="-6.2 6.2"/>
                  <body>
                    <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                    <joint name="R_forearm_yaw_link_joint" type="hinge" axis="0 -1 0"
                           limited="true" range="-6.2 6.2"/>
                    <site name="right_palm"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_finger_priority_uses_observation_without_dropping_other_fingers():
    """Observed contact changes ranking but never removes candidate fingers."""
    contact = np.zeros((20, 5), dtype=bool)
    contact[4:10, 3] = True  # ring
    contact[4:8, 2] = True  # middle

    order = _finger_priority(contact, grasp_start=4)

    assert order[:2] == ["ring", "middle"]
    assert set(order) == {"index", "middle", "ring", "pinky"}


def test_analytic_wrist_transport_matches_requested_pose():
    """Analytic xHand coordinates reproduce translation and wrist rotation."""
    model = _wrist_model()
    qpos = np.zeros(model.nq)
    target = np.eye(4)
    target[:3, 3] = [0.13, -0.07, 0.21]
    target[:3, :3] = Rotation.from_euler(
        "ZXY", [0.31, -0.24, -0.42]
    ).as_matrix()

    solved = _set_xhand_wrist_pose(model, qpos, target, qpos, "right")
    data = mujoco.MjData(model)
    data.qpos[:] = solved
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    np.testing.assert_allclose(data.site_xpos[site], target[:3, 3], atol=1e-9)
    np.testing.assert_allclose(
        data.site_xmat[site].reshape(3, 3), target[:3, :3], atol=1e-9
    )


def test_floor_lift_is_derived_from_geometry_and_object_motion():
    """Floor correction follows geometry rather than an episode constant."""
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.03">
          <geom name="collision_hand_right_thumb_0" type="sphere" size="0.01"/>
        </body>
        <body name="right_object" pos="0 0 0.10">
          <freejoint name="right_object_joint"/>
          <geom name="right_object_0" type="sphere" size="0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    grasp = model.qpos0.copy()
    trajectory = np.repeat(grasp[None], 2, axis=0)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint"
    )
    address = model.jnt_qposadr[object_joint]
    trajectory[1, address + 2] = 0.05

    lift = _trajectory_floor_lift_profile(model, grasp, trajectory, "right")

    np.testing.assert_allclose(lift, [0.0, 0.03], atol=1e-9)


def test_mpc_handoff_replaces_stale_contact_channels():
    """MPC receives the synthesized opposed fingers from the grasp frame."""
    sites = "".join(
        f'<site name="right_{finger}_tip" pos="{index * 0.01} 0 0"/>'
        for index, finger in enumerate(
            ("thumb", "index", "middle", "ring", "pinky")
        )
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body>{sites}</body></worldbody></mujoco>"
    )
    qpos = np.zeros((4, model.nq))

    contact, positions = _mpc_contact_handoff(
        model, qpos, 2, ("thumb", "ring"), "right"
    )

    np.testing.assert_array_equal(contact[:2], 0.0)
    np.testing.assert_array_equal(contact[2:, [0, 3]], 1.0)
    np.testing.assert_array_equal(contact[2:, [1, 2, 4]], 0.0)
    np.testing.assert_allclose(positions[0, :, 0], np.arange(5) * 0.01)
