from types import SimpleNamespace

import numpy as np
import pytest

from video_to_spider.rl.core.startup_planner import (
    DOF,
    HORIZON,
    PhysicsBudget,
    expand_knot_noise,
    hand_relation_error,
    improve_plan,
    knot_sources,
    make_noise_schedule,
    make_round_candidates,
    pair_pose,
    project_plan,
    rotation_error_rad,
    step_cost_terms,
    trajectory_cost,
)


def support():
    return (
        np.full((HORIZON, DOF), -0.75, dtype=np.float32),
        np.full((HORIZON, DOF), 0.80, dtype=np.float32),
    )


def identity_qpos():
    value = np.zeros(50, dtype=np.float64)
    value[39] = 1.0
    value[46] = 1.0
    return value


def test_knot_schedule_and_control_index_interpolation():
    np.testing.assert_array_equal(knot_sources(0), [0, 5, 10, 15, 20, 25, 30, 35, 39])
    np.testing.assert_array_equal(knot_sources(15), [15, 20, 25, 30, 35, 39])
    nodes = np.zeros((len(knot_sources(15)), DOF))
    nodes[:, 0] = np.arange(len(nodes))
    expanded = expand_knot_noise(15, nodes)
    assert expanded.shape == (25, DOF)
    assert expanded[0, 0] == 0.0
    assert expanded[5, 0] == 1.0
    assert expanded[-1, 0] == 5.0


def test_nominal_bytes_are_preserved_and_zero_slot_is_projected_once():
    low, high = support()
    nominal = np.linspace(-0.5, 0.5, HORIZON * DOF, dtype=np.float32).reshape(HORIZON, DOF)
    noise = make_noise_schedule()[(5, 0)]
    rows, counts = make_round_candidates(nominal, low, high, source=5, noise=noise)
    assert rows.shape == (32, HORIZON, DOF)
    assert rows[0].tobytes() == nominal.tobytes()
    assert rows[1].tobytes() == np.zeros_like(nominal).tobytes()
    assert counts[0] == 0
    assert rows.dtype == np.float32
    assert np.all(rows >= low) and np.all(rows <= high)


def test_projection_counts_components_and_requires_float32_support():
    low, high = support()
    proposal = np.zeros((HORIZON, DOF), dtype=np.float64)
    proposal[3, 7] = 2.0
    projected, count = project_plan(proposal, low, high)
    assert count == 1
    assert projected[3, 7] == high[3, 7]
    with pytest.raises(ValueError):
        project_plan(proposal, low.astype(np.float64), high)


def test_incomplete_forecast_cannot_win_and_all_failed_is_explicit():
    low, high = support()
    nominal = np.zeros((HORIZON, DOF), dtype=np.float32)
    noise = np.zeros((30, len(knot_sources(0)), DOF), dtype=np.float64)

    def evaluator(sequence, slot, projected):
        del projected
        return SimpleNamespace(
            sequence=sequence.copy(), complete=slot == 3,
            cost=-1000.0 if slot == 2 else float(slot),
        )

    _, winner, _, _ = improve_plan(
        nominal, low, high, source=0, noise=noise, evaluator=evaluator,
    )
    assert winner.complete and winner.cost == 3.0

    def all_failed(sequence, slot, projected):
        del slot, projected
        return SimpleNamespace(sequence=sequence.copy(), complete=False, cost=-1e9)

    with pytest.raises(RuntimeError, match="NO_COMPLETE_FORECAST_AT_SOURCE_0"):
        improve_plan(nominal, low, high, source=0, noise=noise, evaluator=all_failed)


def test_pair_transform_direction_and_quaternion_sign_invariance():
    qpos = identity_qpos()
    qpos[36:39] = [2.0, 1.0, 0.0]
    qpos[43:46] = [1.0, 1.0, 0.0]
    relative_position, relative_rotation = pair_pose(qpos)
    np.testing.assert_allclose(relative_position, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(relative_rotation, np.eye(3), atol=1e-12)
    assert rotation_error_rad(qpos[39:43], -qpos[39:43]) == pytest.approx(0.0)


def test_hand_relation_is_invariant_to_common_rigid_transform():
    qpos = identity_qpos()
    reference = identity_qpos()
    qpos[36:39] = [0.2, -0.1, 0.4]
    qpos[43:46] = [-0.3, 0.2, 0.1]
    reference[36:39] = [0.18, -0.08, 0.4]
    reference[43:46] = [-0.32, 0.22, 0.1]
    tips = np.arange(30, dtype=np.float64).reshape(10, 3) * 0.003
    reference_tips = tips + np.linspace(-0.01, 0.01, 30).reshape(10, 3)
    before = hand_relation_error(qpos, tips, reference, reference_tips)

    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([0.7, -0.4, 0.2])
    transformed = qpos.copy(); transformed_ref = reference.copy()
    for row in (transformed, transformed_ref):
        for offset in (36, 43):
            row[offset:offset + 3] = rotation @ row[offset:offset + 3] + translation
            row[offset + 3:offset + 7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    tips2 = tips @ rotation.T + translation
    reference_tips2 = reference_tips @ rotation.T + translation
    after = hand_relation_error(transformed, tips2, transformed_ref, reference_tips2)
    assert after == pytest.approx(before, abs=1e-12)


def test_cost_reports_both_objects_and_terminal_term():
    qpos = identity_qpos(); reference = identity_qpos()
    qpos[36] = 0.01
    qpos[46:50] *= -1.0
    tips = np.zeros((10, 3))
    row = step_cost_terms(
        qpos=qpos, reference_qpos=reference, tips=tips,
        reference_tips=tips, action=np.zeros(DOF), ctrl=np.zeros(DOF),
        previous_ctrl=np.zeros(DOF),
    )
    assert row["tool_position_error_m"] == pytest.approx(0.01)
    assert row["target_position_error_m"] == 0.0
    assert row["target_rotation_error_rad"] == pytest.approx(0.0)
    total, parts = trajectory_cost([row, row])
    assert total == pytest.approx(row["L"] + row["D_obj"] + row["D_pair"])
    assert parts["mean_D_obj"] == row["D_obj"]


def test_budget_is_external_to_snapshot_and_reserves_before_attempt():
    budget = PhysicsBudget({"search": 100, "setup": 2}, all_in_limit=102)
    budget.reserve("search", 80)
    budget.charge("search", 50)
    budget.reserve("setup", 1)
    budget.charge("setup", 1)
    assert budget.total == 51
    with pytest.raises(RuntimeError):
        budget.reserve("search", 51)
