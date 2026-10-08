import numpy as np
import pytest

from egoengine_repro.evaluation.controlled_physics_replay import (
    control_substep_schedule,
    longest_true_run,
    proxy_native_mismatch,
    quaternion_distance_rad,
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
