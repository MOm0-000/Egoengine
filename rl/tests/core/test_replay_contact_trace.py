import numpy as np
import pytest

from video_to_spider.rl.replay_contact_trace import (
    BudgetLedger,
    canonical_contact_group,
    classify_geom_role,
    clone_array,
    contact_frame_wrench_to_world,
    control_unit_labels,
    encode_desired_residual,
    endpoint_sample_or_none,
    quintic_blend_weights,
    reference_decode_elliptic,
    reference_decode_pyramidal,
    replay_command,
    startup_control_sequences,
    transition_indices,
    valid_contact_prefix,
)


def test_transition_and_zero_residual_command_use_t_plus_one():
    ctrl = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    assert transition_indices(0, 0)["command_reference_endpoint"] == 1
    assert transition_indices(15, 9)["global_substep"] == 160
    assert np.array_equal(replay_command(ctrl, 0), ctrl[1])
    assert np.array_equal(replay_command(ctrl, 3), ctrl[4])


def test_clone_array_has_no_alias():
    source = np.arange(5)
    cloned = clone_array(source)
    source[:] = -1
    assert np.array_equal(cloned, np.arange(5))
    assert cloned.flags.owndata


def test_contact_decode_rotation_sign_condim_and_invalid_address():
    source = np.array([3.0, 2.0, -1.0, 0.5, 0.4, 0.3])
    before = source.copy()
    elliptic = reference_decode_elliptic(source, 0, 3)
    assert np.array_equal(elliptic, [3.0, 2.0, -1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(reference_decode_elliptic(source, 0, 1), [3, 0, 0, 0, 0, 0])
    assert np.array_equal(reference_decode_elliptic(source, 0, 6), source)
    assert np.array_equal(source, before)
    pyramid = reference_decode_pyramidal(
        np.array([4.0, 1.0, 2.0, 3.0, 99.0]), 0, 3, np.array([0.5, 0.25])
    )
    np.testing.assert_allclose(pyramid, [10.0, 1.5, -0.25, 0, 0, 0])
    assert reference_decode_elliptic(np.ones(4), -1, 3) is None
    assert reference_decode_pyramidal(np.ones(4), -1, 3, np.ones(2)) is None
    with pytest.raises(ValueError):
        reference_decode_pyramidal(np.ones(3), 0, 3, np.ones(2))

    padded = np.array([[1, 2], [3, 4], [999, 999]])
    active = valid_contact_prefix(padded, 2)
    padded[:] = -1
    assert np.array_equal(active, [[1, 2], [3, 4]])

    # Contact axes are stored in world coordinates as rows.
    frame = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = contact_frame_wrench_to_world(np.array([3, 2, 0, 0, 0, 1]), frame)
    np.testing.assert_allclose(world, [-2, 3, 0, 0, 0, 1])
    np.testing.assert_allclose(-world, [2, -3, 0, 0, 0, -1])


def test_contact_group_is_symmetric_and_named_by_physical_roles():
    assert canonical_contact_group("right_hand:index", "tool") == "right_hand_tool"
    assert canonical_contact_group("tool", "right_hand:index") == "right_hand_tool"
    assert canonical_contact_group("floor", "target") == "floor_target"
    # body 2 descends from right-hand root 1; body 3 is the tool root.
    role = classify_geom_role(
        1,
        np.array([0, 2, 3]),
        np.array([0, 0, 1, 0]),
        ["world", "right_hand", "right_index_tip", "right_object"],
        right_root=1,
        left_root=99,
        tool_root=3,
        target_root=98,
        floor_geom=0,
    )
    assert role == "right_hand:index"


def test_budget_is_external_and_cannot_be_rolled_back_by_snapshot_state():
    ledger = BudgetLedger(limit_physics_steps=400, limit_control_intervals=40)
    ledger.charge(physics_steps=200, control_intervals=20)
    fake_snapshot = {"physics_steps": 0, "control_intervals": 0}
    assert fake_snapshot["physics_steps"] == 0
    assert ledger.physics_steps == 200
    ledger.charge(physics_steps=200, control_intervals=20)
    with pytest.raises(RuntimeError):
        ledger.charge(physics_steps=1, control_intervals=0)


def test_startup_quintic_controls_are_exact_and_inputs_are_immutable():
    initial = np.linspace(-0.02, 0.02, 36, dtype=np.float32)
    reference = np.zeros((21, 36), dtype=np.float32)
    for endpoint in range(21):
        reference[endpoint] = np.float32(endpoint / 1000.0)
    initial_before, reference_before = initial.copy(), reference.copy()
    controls = startup_control_sequences(initial, reference)

    np.testing.assert_array_equal(
        quintic_blend_weights(), [0.05792, 0.31744, 0.68256, 0.94208, 1.0]
    )
    assert controls["hold_1_desired_ctrl"].shape == (1, 36)
    np.testing.assert_array_equal(controls["hold_1_desired_ctrl"][0], initial)
    np.testing.assert_array_equal(controls["blend_5_desired_ctrl"][4:], reference[5:21])
    assert np.flatnonzero(np.any(controls["blend_5_residual_action"] != 0, axis=1)).tolist() == [0, 1, 2, 3]
    assert np.max(np.abs(controls["hold_1_residual_action"])) <= 1.0
    assert np.max(np.abs(controls["blend_5_residual_action"])) <= 1.0
    np.testing.assert_array_equal(initial, initial_before)
    np.testing.assert_array_equal(reference, reference_before)


def test_startup_residual_encoding_and_unit_partition_are_explicit():
    reference = np.zeros((2, 4), dtype=np.float32)
    desired = np.asarray([[0.05, -0.05, 0.025, -0.025], [0, 0, 0, 0]], dtype=np.float32)
    action = encode_desired_residual(desired, reference)
    np.testing.assert_array_equal(action[0], [1.0, -1.0, 0.5, -0.5])
    realized = reference + np.float32(0.05) * action
    np.testing.assert_allclose(realized, desired, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(control_unit_labels(np.asarray([2, 2, 3, 3])), ["m", "m", "rad", "rad"])
    with pytest.raises(ValueError):
        control_unit_labels(np.asarray([1, 3]))


def test_startup_hold_endpoint_after_one_is_explicit_na_and_budget_reservation_is_fail_closed():
    hold = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(endpoint_sample_or_none(hold, 1), hold[1])
    assert endpoint_sample_or_none(hold, 2) is None
    with pytest.raises(ValueError):
        endpoint_sample_or_none(hold, -1)

    ledger = BudgetLedger(limit_physics_steps=1000, limit_control_intervals=81)
    ledger.charge(physics_steps=1, control_intervals=0)
    ledger.require_capacity(physics_steps=611, control_intervals=61)
    with pytest.raises(RuntimeError):
        ledger.require_capacity(physics_steps=1000, control_intervals=1)
    assert ledger.physics_steps == 1
    assert ledger.control_intervals == 0
