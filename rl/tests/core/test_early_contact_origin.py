import numpy as np
import pytest
import mujoco
import inspect
import egoengine_repro.early_contact_origin as early_contact_origin

from egoengine_repro.early_contact_origin import (
    actuator_unit,
    command_reference_endpoint,
    compose_qpos,
    distance_semantics,
    factorization_vectors,
    geom_part,
    object_local,
    run_early_contact_origin,
)


def test_pose_combinations_change_only_declared_fields_and_do_not_alias():
    reference = np.arange(50, dtype=np.float64)
    actual = reference + 100
    before_reference = reference.copy()
    before_actual = actual.copy()
    expected = {
        "RR": np.r_[reference[:36], reference[36:]],
        "AR": np.r_[actual[:36], reference[36:]],
        "RA": np.r_[reference[:36], actual[36:]],
        "AA": np.r_[actual[:36], actual[36:]],
    }
    for mode, value in expected.items():
        result = compose_qpos(reference, actual, mode)
        assert np.array_equal(result, value)
        result[:] = -1
    assert np.array_equal(reference, before_reference)
    assert np.array_equal(actual, before_actual)


def test_marker_factorization_and_object_local_transform():
    human = np.array([1.0, 2.0, 3.0])
    reference = np.array([2.0, 4.0, 6.0])
    actual = np.array([4.0, 8.0, 12.0])
    result = factorization_vectors(human, reference, actual)
    assert np.array_equal(result["H_to_A"], result["H_to_R"] + result["R_to_A"])
    pose = np.array([0.5, -0.25, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(object_local(np.array([0.6, -0.05, 1.3]), pose), [0.1, 0.2, 0.3])


def test_control_endpoint_alignment_and_units():
    assert command_reference_endpoint(0) == 0
    assert command_reference_endpoint(16) == 16
    assert actuator_unit(int(mujoco.mjtJoint.mjJNT_SLIDE)) == "m"
    assert actuator_unit(int(mujoco.mjtJoint.mjJNT_HINGE)) == "rad"
    with pytest.raises(ValueError):
        command_reference_endpoint(-1)


def test_distance_sentinel_and_geom_roles_are_not_silently_dropped():
    assert distance_semantics(0.2, 0.2, np.zeros(6)) == ("censored_at_distmax", None)
    assert distance_semantics(-0.001, 0.2, np.ones(6)) == ("measured", -0.001)
    assert geom_part("right_hand_thumb_rota_link2") == ("right", "thumb")
    assert geom_part("left_hand_mid_link2") == ("left", "middle")
    assert geom_part("left_hand_mystery_link") == ("left", "unclassified")
    assert geom_part("right_object") is None


def test_entrypoint_contains_no_forbidden_runtime_calls():
    source = inspect.getsource(early_contact_origin)
    for forbidden in ("mj_step(", "mj_step1(", "mj_step2(", "mj_forward(", "solve_ik("):
        assert forbidden not in source
