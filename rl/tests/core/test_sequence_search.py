from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from video_to_spider.rl.core.sequence_search import (
    BudgetLedger,
    BudgetStopped,
    DOF,
    HORIZON,
    SequenceResult,
    colored_noise,
    deduplicate_results,
    evaluate_sequence,
    masked_refit,
    project_sequence,
    sample_population,
    search_window,
    sequence_key,
)


def snapshot() -> dict:
    values = {
        "time_indices": np.asarray([40], dtype=np.int32),
        "start_indices": np.asarray([0], dtype=np.int32),
        "episode_lengths": np.asarray([80], dtype=np.int32),
        "rng_state": {"state": 0},
        "last_action": np.zeros((1, DOF), dtype=np.float32),
        "last_ctrl": np.zeros((1, DOF), dtype=np.float32),
        "qpos": np.zeros((1, 50), dtype=np.float32),
        "qvel": np.zeros((1, 48), dtype=np.float32),
        "ctrl": np.zeros((1, DOF), dtype=np.float32),
        "contact.geom": np.zeros((1,), dtype=np.int32),
        "contact.worldid": np.zeros((1,), dtype=np.int32),
        "efc.force": np.zeros((1,), dtype=np.float32),
        "prev.qpos": np.zeros((1, 50), dtype=np.float32),
        "prev.qvel": np.zeros((1, 48), dtype=np.float32),
        "prev.ctrl": np.zeros((1, DOF), dtype=np.float32),
    }
    return {
        "snapshot_schema": "egoengine_mjwp_snapshot_v3_reward_aligned",
        "warp_state_keys": tuple(values),
        **values,
    }


class FakeWorld:
    def __init__(self, failure_row: int | None = None, *, raise_row: int | None = None):
        self.failure_row = failure_row
        self.raise_row = raise_row
        self.calls = 0
        self._mjwp = self
        self.ego_cfg = object()
        self.env = object()
        self._last_ctrl = torch.zeros((1, DOF), dtype=torch.float32)
        self.qpos = torch.zeros((1, 50), dtype=torch.float32)
        self.qvel = torch.zeros((1, 48), dtype=torch.float32)
        self.set_env_state(snapshot())

    def set_env_state(self, state):
        self.start_indices = np.asarray(state["start_indices"]).copy()
        self.time_indices = np.asarray(state["time_indices"]).copy()
        self.calls = 0

    def current_observation(self):
        return np.full((1, 236), self.calls, dtype=np.float32)

    def current_normalized_action_bounds(self):
        return torch.full((1, DOF), -1.0), torch.full((1, DOF), 1.0)

    def get_qpos(self, _config, _env):
        return self.qpos

    def get_qvel(self, _config, _env):
        return self.qvel

    def step(self, action, *, auto_reset):
        assert not auto_reset
        row = self.calls
        self.calls += 1
        if row == self.raise_row:
            raise RuntimeError("simulator failure")
        source = int(self.start_indices[0] + self.time_indices[0])
        self.time_indices += 1
        outcome = source + 1
        terminated = row == self.failure_row
        timeout = outcome == 80
        score = 1.01 if terminated else 0.5
        self._last_ctrl = torch.as_tensor(action, dtype=torch.float32)
        self.qpos[:, 0] = row + 1
        self.qvel[:, 0] = row + 2
        info = {
            "source_reference_endpoint": np.asarray([source]),
            "outcome_reference_endpoint": np.asarray([outcome]),
            "command_reference_endpoint": np.asarray([outcome]),
            "reward_reference_endpoint": np.asarray([outcome]),
            "next_observation_goal_reference_endpoint": np.asarray([outcome + 1]),
            "terminated": np.asarray([terminated]),
            "time_outs": np.asarray([timeout]),
            "object_tracking_error": np.asarray([score]),
            "object_position_error": np.asarray([[0.1]]),
            "object_rotation_error": np.asarray([[0.2]]),
            "aggregate_tracking_reward": np.asarray([1.0 - score]),
            "aggregate_contact_bonus": np.asarray([0.0]),
            "lift_reward": np.asarray([0.0]),
            "contact_flags": np.zeros((1, 2, 2, 5), dtype=bool),
        }
        done = np.asarray([terminated or timeout])
        return self.current_observation(), np.asarray([1.0 - score]), done, info


def support():
    return (
        np.full((HORIZON, DOF), -1.0, dtype=np.float32),
        np.full((HORIZON, DOF), 1.0, dtype=np.float32),
    )


def ledger(limit=200):
    return BudgetLedger(
        phase_limits={"preflight": limit, "search": limit, "validation": 80},
        all_in_limit=limit + 80, final_reserve=80,
    )


def run_fake(failure_row, *, candidate_id=0):
    low, high = support()
    return evaluate_sequence(
        FakeWorld(failure_row), np.zeros((HORIZON, DOF), dtype=np.float32),
        low, high, snapshot(), ledger(), phase="search",
        candidate_id=candidate_id, generation=0, slot=0,
    )


def test_colored_noise_is_seeded_finite_and_correlated_only_in_time():
    left = colored_noise(np.random.Generator(np.random.PCG64(7)), 4096)
    right = colored_noise(np.random.Generator(np.random.PCG64(7)), 4096)
    assert left.shape == (4096, HORIZON, DOF)
    assert left.tobytes() == right.tobytes()
    assert np.isfinite(left).all()
    time_corr = np.corrcoef(left[:, :-1, :].ravel(), left[:, 1:, :].ravel())[0, 1]
    dof_corr = np.corrcoef(left[:, :, :-1].ravel(), left[:, :, 1:].ravel())[0, 1]
    assert time_corr > 0.5
    assert abs(dof_corr) < 0.03
    with pytest.raises(ValueError):
        colored_noise(np.random.default_rng(0), 1, beta=float("nan"))


def test_projection_and_population_preserve_float32_support_and_identity():
    low, high = support()
    u0 = np.zeros((HORIZON, DOF), dtype=np.float32)
    assert project_sequence(u0, low, high).tobytes() == u0.tobytes()
    rows = sample_population(
        np.random.Generator(np.random.PCG64(0)), u0.astype(np.float64),
        np.full_like(u0, 0.5, dtype=np.float64), low, high, 8, beta=2.5,
    )
    assert rows.dtype == np.float32
    assert rows[0].tobytes() == u0.tobytes()
    assert np.all(rows >= low) and np.all(rows <= high)


@pytest.mark.parametrize(
    ("failure_row", "valid", "executed", "failure", "strict"),
    [(0, 0, 1, 41, False), (39, 39, 40, 80, False), (None, 40, 40, None, True)],
)
def test_sequence_termination_contract(failure_row, valid, executed, failure, strict):
    result = run_fake(failure_row)
    assert result.valid_prefix == valid
    assert result.executed_controls == executed
    assert result.first_failure_endpoint == failure
    assert result.strict_success is strict
    assert int(result.executed_mask.sum()) == executed
    assert np.isnan(result.tracking_score[executed:]).all()


def test_first_failure_stops_even_if_later_fake_scores_would_recover():
    result = run_fake(2)
    assert result.executed_controls == 3
    assert result.outcome_endpoint[:3].tolist() == [41, 42, 43]
    assert np.all(result.outcome_endpoint[3:] == -1)


def test_rank_is_prefix_then_failure_then_reward_then_earlier_id():
    a = run_fake(2, candidate_id=1)
    b = run_fake(3, candidate_id=9)
    assert b.rank_key() > a.rank_key()
    better_failure = replace(a, first_failure_score=1.001, candidate_id=2)
    assert better_failure.rank_key() > a.rank_key()
    better_reward = replace(a, prefix_tracking_reward=a.prefix_tracking_reward + 1, candidate_id=3)
    assert better_reward.rank_key() > a.rank_key()
    later = replace(a, candidate_id=4)
    assert a.rank_key() > later.rank_key()


def test_masked_refit_leaves_unobserved_tail_and_single_observation_unchanged():
    first = run_fake(1, candidate_id=1)
    second = replace(run_fake(0, candidate_id=2), sequence=np.full((40, 36), 0.5, np.float32))
    mean = np.zeros((40, 36), dtype=np.float64)
    std = np.ones((40, 36), dtype=np.float64)
    new_mean, new_std, counts = masked_refit(
        [first, second], mean, std, np.full((40, 36), 0.02), old_weight=0.1
    )
    assert counts[:3].tolist() == [2, 1, 0]
    assert np.any(new_mean[0] != mean[0])
    assert np.array_equal(new_mean[1:], mean[1:])
    assert np.array_equal(new_std[1:], std[1:])


def test_exact_duplicate_mean_is_reused_without_new_evaluation_or_best_replacement():
    initial = run_fake(4, candidate_id=-1)
    calls = []

    def evaluator(sequence, candidate_id, generation, slot):
        calls.append(candidate_id)
        return replace(initial, candidate_id=candidate_id, generation=generation, slot=slot)

    low, high = support()
    outcome = search_window(
        initial=initial, low=low, high=high, evaluator=evaluator,
        rng=np.random.Generator(np.random.PCG64(0)), slots_per_generation=[1],
    )
    assert calls == []
    assert outcome.duplicate_reuses == 1
    assert outcome.best.candidate_id == -1
    assert outcome.candidate_slots == 1


def test_sequence_dedup_uses_exact_float32_bytes():
    one = run_fake(2, candidate_id=1)
    duplicate = replace(one, candidate_id=2)
    distinct = replace(one, candidate_id=3, sequence=np.full((40, 36), 1e-7, np.float32))
    rows = deduplicate_results([one, duplicate, distinct])
    assert len(rows) == 2
    assert sequence_key(one.sequence) != sequence_key(distinct.sequence)


def test_missing_snapshot_field_and_support_mismatch_fail_closed():
    bad = snapshot()
    del bad["prev.ctrl"]
    low, high = support()
    with pytest.raises(ValueError):
        evaluate_sequence(
            FakeWorld(), np.zeros((40, 36), np.float32), low, high, bad,
            ledger(), phase="search", candidate_id=0, generation=0, slot=0,
        )
    wrong_low = low.copy(); wrong_low[0, 0] = -0.5
    with pytest.raises(RuntimeError, match="support mismatch"):
        evaluate_sequence(
            FakeWorld(), np.zeros((40, 36), np.float32), wrong_low, high,
            snapshot(), ledger(), phase="search", candidate_id=0, generation=0, slot=0,
        )


def test_budget_is_not_rewound_by_world_restore_and_charges_failed_step():
    costs = ledger(limit=80)
    low, high = support()
    with pytest.raises(RuntimeError, match="simulator failure"):
        evaluate_sequence(
            FakeWorld(raise_row=0), np.zeros((40, 36), np.float32), low, high,
            snapshot(), costs, phase="search", candidate_id=0, generation=0, slot=0,
        )
    assert costs.controls["search"] == 1
    assert costs.report()["physics_steps"]["search"] == 10
    costs.controls["search"] = 41
    with pytest.raises(BudgetStopped):
        costs.reserve_candidate("search")


def test_result_npz_fields_are_plain_numpy_and_masked():
    result = run_fake(2)
    arrays = result.arrays(prefix="best_")
    assert arrays["best_sequence"].shape == (40, 36)
    assert arrays["best_executed_mask"].dtype == np.bool_
    assert arrays["best_strict_success"].shape == ()
