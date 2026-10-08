import numpy as np

from egoengine_repro.evaluation.replay_failure_attribution import (
    classify_native_measurement,
    semantic_finger,
    vector_cosine,
)


def test_semantic_finger_uses_collision_names():
    assert semantic_finger("collision_hand_left_middle_0") == "middle"
    assert semantic_finger("right_hand_pinky_link2") == "pinky"
    assert semantic_finger("right_object_0") == "other"


def test_vector_cosine_handles_opposition_and_zero():
    assert vector_cosine(np.array([1, 0, 0]), np.array([-2, 0, 0])) == -1.0
    assert vector_cosine(np.zeros(3), np.ones(3)) is None


def test_classify_native_measurement_is_fail_closed():
    penetration = {
        "surface_intersection": True,
        "containment_penetration": False,
        "reported_penetration_depth_m": 1e-3,
        "distance_m": -1e-3,
        "containment_followup": [],
    }
    clearance = {
        "surface_intersection": False,
        "containment_penetration": False,
        "reported_penetration_depth_m": 0.0,
        "distance_m": 1e-3,
        "containment_followup": [],
    }
    assert classify_native_measurement(penetration, 5e-5) == "PENETRATION"
    assert classify_native_measurement(clearance, 5e-5) == "CLEARANCE"
