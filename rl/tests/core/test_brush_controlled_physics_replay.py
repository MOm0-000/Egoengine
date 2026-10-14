import numpy as np
import pytest
import mujoco

from egoengine_repro.evaluation.controlled_physics_replay import (
    REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION,
    REFERENCE_CTRL_OUTCOME_ENDPOINT,
    control_substep_schedule,
    integrate_freejoint_pose,
    longest_true_run,
    proxy_native_geometry_relation,
    proxy_native_mismatch,
    quaternion_distance_rad,
    reference_control_target,
    synchronized_forward_snapshot,
)


def test_native_timestep_schedule_is_bounded_and_nonaccumulating():
    schedule = control_substep_schedule(
        control_endpoints=80, control_hz=30.0, simulation_dt_s=0.002
    )
    assert set(schedule) == {16, 17}
    assert int(schedule.sum()) == 1334
    actual = np.cumsum(schedule) * 0.002
    requested = np.arange(1, 81) / 30.0
    assert np.all(actual >= requested)
    assert np.all(actual - requested < 0.002 + 1e-15)


def test_outcome_endpoint_control_is_held_for_every_substep():
    controls = np.array([[0.0, 2.0], [10.0, 12.0]])
    for substep in range(4):
        target, alpha = reference_control_target(
            controls,
            source_endpoint=0,
            outcome_endpoint=1,
            substep=substep,
            substeps=4,
            command_timing=REFERENCE_CTRL_OUTCOME_ENDPOINT,
        )
        assert target == pytest.approx([10.0, 12.0])
        assert alpha == 1.0


def test_interpolated_control_reaches_outcome_on_last_substep():
    controls = np.array([[0.0, 2.0], [10.0, 12.0]])
    expected = ([2.5, 4.5], [5.0, 7.0], [7.5, 9.5], [10.0, 12.0])
    for substep, expected_target in enumerate(expected):
        target, alpha = reference_control_target(
            controls,
            source_endpoint=0,
            outcome_endpoint=1,
            substep=substep,
            substeps=4,
            command_timing=REFERENCE_CTRL_LINEAR_SUBSTEP_INTERPOLATION,
        )
        assert target == pytest.approx(expected_target)
        assert alpha == pytest.approx((substep + 1) / 4)


def test_control_target_rejects_unknown_timing():
    with pytest.raises(ValueError, match="unsupported command timing"):
        reference_control_target(
            np.array([[0.0], [1.0]]),
            source_endpoint=0,
            outcome_endpoint=1,
            substep=0,
            substeps=1,
            command_timing="unknown",
        )


def test_post_step_geometry_requires_an_independent_forward_snapshot():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body><freejoint/><geom type="sphere" size="0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    assert data.qpos[0] == pytest.approx(0.01)
    assert data.geom_xpos[0, 0] == pytest.approx(0.0)

    snapshot = synchronized_forward_snapshot(
        mujoco, model, data, mujoco.MjData(model),
    )
    assert snapshot.qpos[0] == pytest.approx(data.qpos[0])
    assert snapshot.qvel[0] == pytest.approx(data.qvel[0])
    assert snapshot.geom_xpos[0, 0] == pytest.approx(data.qpos[0])


def test_synchronized_snapshot_rejects_live_data_alias():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="independent MjData"):
        synchronized_forward_snapshot(mujoco, model, data, data)


def test_freejoint_pose_reconstruction_uses_mujoco_integration():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><freejoint/>
          <geom type="sphere" size="0.1"/>
        </body></worldbody></mujoco>
        """
    )
    pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    velocity = np.array([0.5, -1.0, 2.0, 0.0, 0.0, 1.0])
    actual = integrate_freejoint_pose(mujoco, model, pose, velocity, 0.002)
    expected = pose.copy()
    mujoco.mj_integratePos(model, expected, velocity, 0.002)
    assert actual == pytest.approx(expected)
    assert actual[:3] == pytest.approx([1.001, 1.998, 3.004])


def test_quaternion_distance_is_sign_invariant():
    q = np.array([0.5, 0.5, -0.5, 0.5])
    assert quaternion_distance_rad(q, -q) == pytest.approx(0.0)
    assert quaternion_distance_rad([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(np.pi)


def test_longest_true_run():
    assert longest_true_run([False, True, True, False, True, True, True]) == 3
    assert longest_true_run([]) == 0


@pytest.mark.parametrize(
    ("proxy", "distance", "expected"),
    [
        (True, 1e-3, "PROXY_FORCE_WHILE_NATIVE_CLEAR"),
        (False, -1e-3, "NATIVE_PENETRATION_WITHOUT_PROXY_FORCE"),
        (True, 0.0, "AGREEMENT_CONTACT_FORCE"),
        (False, 1e-3, "AGREEMENT_CLEARANCE_NO_FORCE"),
    ],
)
def test_proxy_native_mismatch(proxy, distance, expected):
    assert proxy_native_mismatch(
        proxy_contact=proxy, native_distance_m=distance, tolerance_m=5e-5
    ) == expected


@pytest.mark.parametrize(
    ("proxy", "native", "expected"),
    [
        (-1e-3, 1e-3, "PROXY_OVERCOVERAGE_PENETRATION_WHILE_NATIVE_CLEAR"),
        (1e-3, -1e-3, "PROXY_MISSED_NATIVE_PENETRATION"),
        (0.0, -1e-3, "GEOMETRY_CONTACT_VS_PENETRATION"),
        (1e-3, 2e-3, "GEOMETRY_CLEARANCE_VS_CLEARANCE"),
    ],
)
def test_proxy_native_geometry_relation(proxy, native, expected):
    assert proxy_native_geometry_relation(
        proxy_distance_m=proxy, native_distance_m=native, tolerance_m=5e-5
    ) == expected
