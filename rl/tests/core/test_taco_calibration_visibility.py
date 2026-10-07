from __future__ import annotations

import numpy as np
import pytest

from egoengine_repro.evaluation.calibration_known_answer import (
    checkpoint_error, independent_apply, independent_inverse, independent_transform,
    transform_error,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (
    eroded_target_mask, measured_target_selector, target_visibility_mask,
)


def test_independent_transform_has_hand_checkable_translation_and_axis_rotation():
    transform = independent_transform([1.0, 2.0, 3.0], [0.0, 0.0, np.pi / 2])
    value = independent_apply([[1.0, 0.0, 0.0]], transform)[0]
    assert np.allclose(value, [1.0, 3.0, 3.0], atol=1e-12)
    restored = independent_apply([value], independent_inverse(transform))[0]
    assert np.allclose(restored, [1.0, 0.0, 0.0], atol=1e-12)


def test_transform_score_compares_full_direction_not_only_magnitude():
    truth = independent_transform([0.01, 0.0, 0.0], [0.0, 0.1, 0.0])
    wrong = independent_transform([-0.01, 0.0, 0.0], [0.0, -0.1, 0.0])
    result = transform_error(wrong, truth)
    assert result["translation_error_mm"] == pytest.approx(20.0)
    assert result["rotation_error_deg"] > 10.0
    checkpoints = checkpoint_error(wrong, truth, np.array([[0.2, 0.1, -0.1], [-0.1, 0.0, 0.3]]))
    assert checkpoints["maximum_mm"] > checkpoints["median_mm"]


def test_occluder_in_front_is_excluded_and_target_in_front_is_retained():
    target = np.full((5, 5), 1.0)
    front = np.zeros((5, 5)); front[2, 2] = 0.8
    behind = np.zeros((5, 5)); behind[1, 1] = 1.2
    visible = target_visibility_mask(target, [front, behind], uncertainty_margin_m=0.002)
    assert not visible[2, 2]
    assert visible[1, 1]


def test_pytorch3d_negative_background_is_treated_as_absent():
    target = np.array([[1.0, -1.0], [1.0, 1.0]])
    occluder = np.array([[-1.0, -1.0], [0.8, -1.0]])
    visible = target_visibility_mask(target, [occluder], uncertainty_margin_m=0.002)
    assert visible.tolist() == [[True, False], [False, True]]


def test_near_tie_is_conservatively_excluded():
    target = np.full((2, 2), 1.0)
    occluder = np.full((2, 2), 1.001)
    assert not target_visibility_mask(target, [occluder], uncertainty_margin_m=0.002).any()


def test_measured_selector_cannot_pass_occluder_depth_inside_target_silhouette():
    target = np.full((7, 7), 1.0)
    hand = np.zeros((7, 7)); hand[2:5, 2:5] = 0.75
    measured = np.minimum(target, np.where(hand > 0, hand, np.inf))
    selected = measured_target_selector(
        measured, target, [hand], uncertainty_margin_m=0.002, erosion_px=0,
    )
    assert not selected[2:5, 2:5].any()
    assert selected[0, 0]


def test_complete_occlusion_returns_empty_instead_of_zero_contamination_claim():
    target = np.full((4, 4), 1.0)
    hand = np.full((4, 4), 0.7)
    selected = measured_target_selector(
        hand, target, [hand], uncertainty_margin_m=0.002, erosion_px=0,
    )
    assert not selected.any()


def test_mask_erosion_is_explicit_and_validated():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    eroded = eroded_target_mask(mask, erosion_px=1)
    assert eroded.sum() == 1
    with pytest.raises(ValueError):
        eroded_target_mask(mask, erosion_px=-1)
