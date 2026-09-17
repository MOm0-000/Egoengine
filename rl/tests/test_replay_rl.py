"""Two-chunk scheduling tests, not evidence of a trained TACO RL controller."""

import copy
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egoengine_repro.action.replay_rl import solve_chunk


class Backend:
    def __init__(self, failure=lambda _action, _index: False):
        self.state = dict(cursor=0, action=None, warmstart=0, samples=[])
        self.rng = random.Random(4)
        self.simulation_steps = 0
        self.failure = failure

    def snapshot(self):
        return copy.deepcopy((self.state, self.rng.getstate()))

    def restore(self, snapshot):
        self.state, rng = copy.deepcopy(snapshot)
        self.rng.setstate(rng)

    def step(self, action, reference_step):
        assert self.state["cursor"] == reference_step
        self.state.update(cursor=reference_step + 1, action=action, warmstart=action + reference_step)
        self.state["samples"].append(self.rng.random())
        self.simulation_steps += 1
        return not self.failure(action, reference_step)


def test_replay_checks_40_commits_only_20_including_rng_and_warmstart():
    backend = Backend()
    expected = Backend()
    for i in range(20):
        expected.step(0, i)
    result = solve_chunk(backend, lambda _env, _i: 0,
                         lambda *_args: pytest.fail("Replay passed; RL must not train"), start=0, total_steps=60)
    assert result.mode == "replay"
    assert result.committed_end == 20
    assert backend.snapshot() == expected.snapshot()
    assert backend.simulation_steps == 40


def test_failed_second_chunk_escalates_and_training_state_is_rolled_back():
    backend = Backend(failure=lambda action, index: action == 0 and index == 25)
    initial = backend.snapshot()

    def train(env, start, end):
        assert (start, end) == (0, 40)
        assert env.snapshot() == initial
        env.step(2, 0)
        return lambda _env, _i: 1

    result = solve_chunk(backend, lambda _env, _i: 0, train, start=0, total_steps=60)
    assert result.mode == "rl"
    assert [(t.mode, t.feasible, t.validated_steps) for t in result.trials] == [
        ("replay", False, 25), ("rl", True, 40)]
    assert backend.state["cursor"] == 20
    assert backend.state["action"] == 1
    assert backend.simulation_steps == 26 + 1 + 40


def test_each_boundary_returns_to_replay_after_rl():
    backend = Backend(failure=lambda action, index: action == 0 and index == 0)
    first = solve_chunk(backend, lambda _env, _i: 0, lambda *_args: lambda _env, _i: 1,
                        start=0, total_steps=70)
    assert first.mode == "rl"
    second = solve_chunk(backend, lambda _env, _i: 0,
                         lambda *_args: pytest.fail("must return to Replay"), start=20, total_steps=70)
    assert second.mode == "replay"
    assert backend.state["cursor"] == 40


def test_both_modes_fail_preserves_boundary_but_not_simulation_cost():
    backend = Backend(failure=lambda _action, index: index == 25)
    initial = backend.snapshot()
    result = solve_chunk(backend, lambda _env, _i: 0, lambda *_args: lambda _env, _i: 1,
                         start=0, total_steps=60)
    assert result.mode is None
    assert result.committed_end == 0
    assert backend.snapshot() == initial
    assert backend.simulation_steps == 52


@pytest.mark.parametrize("length", [1, 19, 20, 29, 39, 40])
def test_tail_is_truncated_never_padded_or_advanced_twice(length):
    backend = Backend()
    result = solve_chunk(backend, lambda _env, _i: 0, lambda *_args: None,
                         start=0, total_steps=length)
    assert result.lookahead_end == length
    assert backend.state["cursor"] == min(20, length)
    assert backend.simulation_steps == length


def test_exception_during_training_restores_state():
    backend = Backend(failure=lambda _action, _index: True)
    initial = backend.snapshot()

    def train(env, *_args):
        env.step(1, 0)
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        solve_chunk(backend, lambda _env, _i: 0, train, start=0, total_steps=60)
    assert backend.snapshot() == initial
    assert backend.simulation_steps == 2


def test_non_boolean_feasibility_cannot_silently_pass():
    backend = Backend()
    initial = backend.snapshot()
    backend.step = lambda *_args: float("nan")
    with pytest.raises(ValueError, match="explicit boolean"):
        solve_chunk(backend, lambda *_args: 0, lambda *_args: None, start=0, total_steps=40)
    assert backend.snapshot() == initial
