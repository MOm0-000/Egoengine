"""Closed-form checks use synthetic coefficients, not claimed paper values."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egoengine_repro.action.paper_rewards import (
    TrackingObjective, action_smoothness, aggregate_evaluations, evaluate_errors,
    human_mimic, lifting, object_tracking, opposition_contact, so3_distance,
    taco_optimization_reward,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("angle", [0, .7, np.pi])
def test_c2_rotation_geodesic(angle):
    rotation = tensor(Rotation.from_rotvec([0, 0, angle]).as_matrix())
    assert so3_distance(rotation, torch.eye(3, dtype=torch.float64)).item() == pytest.approx(angle)


def test_c1_to_c4_weighted_root_not_exponential_and_strict_boundary():
    objective = TrackingObjective(lambda_p=4, lambda_r=9, boundary=1)
    identity = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)
    positions = tensor([[.5, 0, 0], [.500001, 0, 0], [0, 0, 0]])
    result = object_tracking(positions, identity, torch.zeros_like(positions), identity, objective)
    np.testing.assert_allclose(result.error, [1, 1.000002, 0])
    np.testing.assert_allclose(result.reward, [0, -.000002, 1], atol=1e-12)
    assert result.terminated.tolist() == [False, True, False]
    score = object_tracking(tensor([.3, .4, 0]), tensor(Rotation.from_rotvec([0, 0, .2]).as_matrix()),
                            tensor([0, 0, 0]), identity[0], objective)
    assert score.error.item() == pytest.approx(np.sqrt(4 * .5**2 + 9 * .2**2))


def test_c5_negative_weighted_squared_wrist_and_finger_errors():
    result = human_mimic(tensor([1, 2, 0]), tensor(Rotation.from_rotvec([.4, 0, 0]).as_matrix()),
                         tensor([1, 2]), tensor([0, 0, 0]), torch.eye(3, dtype=torch.float64),
                         tensor([0, 0]), beta_x=2, beta_r=3, beta_q=4)
    assert result.item() == pytest.approx(-(2 * 5 + 3 * .4**2 + 4 * 5))


def test_c6_total_action_difference():
    assert action_smoothness(tensor([3, 4]), tensor([0, 0])).item() == -25


def test_c7_no_cross_hand_or_cross_object_contact_pooling():
    contact = torch.zeros((2, 2, 5), dtype=torch.bool)
    contact[0, 0, 0] = True
    contact[1, 0, 1] = True
    contact[0, 1, 1] = True
    assert opposition_contact(contact, coefficient=2).sum() == 0
    contact[0, 0, 4] = True
    np.testing.assert_array_equal(opposition_contact(contact, coefficient=2), [[2, 0], [0, 0]])
    with pytest.raises(ValueError, match="boolean"):
        opposition_contact(contact.float(), coefficient=2)


def test_c8_lift_is_signed_and_taco_total_has_no_mimic_or_smoothness():
    np.testing.assert_allclose(lifting(tensor([.6, .8]), .7, lambda_z=3), [-.3, .3])
    np.testing.assert_allclose(taco_optimization_reward(tensor([.7, .8]), tensor([0, 2])), [.7, 2.8])


def test_c9_to_c13_valid_prefix_normalization_and_success_only_cost():
    failed = evaluate_errors([.2, .4, 1.1, 0], boundary=1, horizon=4, simulation_steps=1000)
    success = evaluate_errors([0, 0], boundary=1, horizon=2, simulation_steps=40)
    result = aggregate_evaluations([failed, success])
    assert failed.valid_steps == 2
    assert result == pytest.approx(dict(SR=.5, Step=.75, Reward=.675, Cost=20))


def test_failure_on_first_step_and_no_success_cost_is_undefined():
    failed = evaluate_errors([1.01], boundary=1, horizon=209, simulation_steps=40)
    assert aggregate_evaluations([failed]) == dict(SR=0, Step=0, Reward=0, Cost=None)


@pytest.mark.parametrize("kwargs", [dict(lambda_p=-1, lambda_r=1, boundary=1),
                                   dict(lambda_p=0, lambda_r=0, boundary=1),
                                   dict(lambda_p=1, lambda_r=1, boundary=0)])
def test_invalid_coefficients_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackingObjective(**kwargs)


def test_nonfinite_or_mismatched_states_fail_closed():
    identity = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite"):
        object_tracking(tensor([np.nan, 0, 0]), identity, tensor([0, 0, 0]), identity,
                        TrackingObjective(1, 1, 1))
    with pytest.raises(ValueError, match="leading"):
        object_tracking(torch.zeros(2, 3), identity, torch.zeros(2, 3), identity,
                        TrackingObjective(1, 1, 1))
    with pytest.raises(ValueError, match="horizon"):
        evaluate_errors([], boundary=1, horizon=0, simulation_steps=0)
