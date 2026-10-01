from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video_to_spider.rl.core.distribution import DistributionSpec
from video_to_spider.rl.core.policy import DistributionOutput, recurrent_evaluate
from video_to_spider.rl.core.ppo import PPOConfig, update_actor
from video_to_spider.rl.core.rollout import FixedBoundaryCollector


class _RawCriticModel:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, payload):
        assert payload["is_train"] is False
        assert payload["update_obs_stats"] is False
        self.calls += 1
        normalized = (payload["obs"] - 2.0) / 2.0
        return {"values": 10.0 + 3.0 * normalized}


def test_collector_critic_value_normalizes_once_and_returns_raw_units():
    model = _RawCriticModel()
    collector = SimpleNamespace(policy=SimpleNamespace(critic=SimpleNamespace(model=model)))
    result = FixedBoundaryCollector._critic_value(collector, torch.tensor([[6.0]]))
    assert model.calls == 1
    assert result.item() == 16.0


class _SentinelActor:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def norm_obs(self, observation, *, update_stats=False):
        assert update_stats is False
        return observation

    def a2c_network(self, payload):
        state = payload["rnn_states"][0]
        batch = payload["obs"].shape[0]
        sentinel = state[0, :, 0].reshape(batch, 1)
        raw = sentinel.repeat(1, 36)
        return raw, torch.zeros_like(raw), sentinel, [state + 1.0]


def test_recurrent_evaluator_resets_before_observation_not_after_action():
    worlds, horizon = 2, 4
    actor = _SentinelActor()
    observations = torch.zeros(worlds * horizon, 236)
    actions = torch.zeros(worlds * horizon, 36)
    bounds = torch.full_like(actions, -1.0), torch.full_like(actions, 1.0)
    # World 0 starts a new episode at t=1. World 1 remains continuous.
    episode_start = torch.tensor(
        [True, True, False, False, True, False, False, False]
    )
    block_state = (torch.full((1, worlds, 1), 9.0),)
    reset_state = (torch.zeros(1, worlds, 1),)
    result = recurrent_evaluate(
        actor,
        observations=observations,
        actions=actions,
        low=bounds[0],
        high=bounds[1],
        episode_start=episode_start,
        block_start_states=block_state,
        reset_states=reset_state,
        worlds=worlds,
        horizon=horizon,
        sequence=4,
    )
    assert result.raw_location[:, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0]


class _FakeEnvironment:
    def __init__(self, *, omit_key=None):
        self.worlds = (object(), object())
        self.offset = 0
        self.omit_key = omit_key

    def restore_all(self, _state):
        self.offset = 0

    def current_observation(self):
        return {"obs": np.zeros((2, 236), np.float32), "states": np.zeros((2, 108), np.float32)}

    def normalized_action_bounds(self):
        return torch.full((2, 36), -1.0), torch.full((2, 36), 1.0)

    def step(self, _actions, *, auto_reset):
        assert auto_reset
        sources = (
            np.array([40, 40]), np.array([40, 41]),
            np.array([41, 42]), np.array([42, 40]),
        )[self.offset]
        terminated = np.array([self.offset == 0, False])
        timeout = np.array([False, self.offset == 2])
        done = terminated | timeout
        outcome = sources + 1
        info = {
            "source_reference_endpoint": sources,
            "outcome_reference_endpoint": outcome,
            "command_reference_endpoint": outcome,
            "reward_reference_endpoint": outcome,
            "next_observation_goal_reference_endpoint": outcome + 1,
            "time_outs": timeout,
            "terminated": terminated,
        }
        if self.omit_key:
            del info[self.omit_key]
        self.offset += 1
        observation = {
            "obs": np.zeros((2, 236), np.float32),
            "states": np.zeros((2, 108), np.float32),
        }
        return observation, np.ones(2, np.float32), done, info


def _fake_policy_step(_actor, observation, states, low, high, **_kwargs):
    actions = torch.zeros(observation.shape[0], 36)
    output = DistributionOutput(
        raw_location=actions,
        mu=actions,
        sigma=torch.ones_like(actions),
        values=torch.zeros(observation.shape[0], 1),
        neglogp=torch.zeros(observation.shape[0]),
        entropy=torch.zeros(observation.shape[0]),
        rnn_states=tuple(state + 1.0 for state in states),
    )
    return actions, output


def test_collector_uses_real_endpoints_timeout_and_rebuilds_boundary_hidden(monkeypatch):
    import video_to_spider.rl.core.rollout as rollout

    version = {"value": 2.0}
    monkeypatch.setattr(
        rollout,
        "burn_in_prefix",
        lambda _actor, _prefix, worlds: (torch.full((1, worlds, 1), version["value"]),),
    )
    monkeypatch.setattr(rollout, "policy_step", _fake_policy_step)
    policy = SimpleNamespace(
        actor=object(), critic=None, distribution=DistributionSpec(), normalization_version=7
    )
    collector = FixedBoundaryCollector(
        environment=_FakeEnvironment(),
        policy=policy,
        boundary_state={},
        observation_prefix=tuple(torch.zeros(1, 236) for _ in range(20)),
        horizon=4,
    )
    batch = collector.collect()
    assert batch.source_endpoint.tolist() == [40, 40, 41, 42, 40, 41, 42, 40]
    assert batch.episode_start.tolist() == [True, True, False, False, True, False, False, True]
    assert batch.timeout.tolist() == [False, False, False, False, False, False, True, False]
    assert batch.episode_serial.tolist() == [0, 1, 1, 1, 0, 0, 0, 1]
    assert batch.normalization_version == 7
    assert collector.reset_states[0][0, 0, 0].item() == 2.0
    version["value"] = 5.0
    collector.collect()
    assert collector.reset_states[0][0, 0, 0].item() == 5.0


def test_collector_rejects_missing_required_timeout_key(monkeypatch):
    import video_to_spider.rl.core.rollout as rollout

    monkeypatch.setattr(
        rollout, "burn_in_prefix", lambda _actor, _prefix, worlds: (torch.zeros(1, worlds, 1),)
    )
    monkeypatch.setattr(rollout, "policy_step", _fake_policy_step)
    collector = FixedBoundaryCollector(
        environment=_FakeEnvironment(omit_key="time_outs"),
        policy=SimpleNamespace(
            actor=object(), critic=None, distribution=DistributionSpec(), normalization_version=0
        ),
        boundary_state={},
        observation_prefix=tuple(torch.zeros(1, 236) for _ in range(20)),
        horizon=4,
    )
    with pytest.raises(KeyError, match="time_outs"):
        collector.collect()


class _ScalarActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


def _fake_recurrent(actor, *, observations, **_kwargs):
    count = observations.shape[0]
    scalar = actor.weight
    zeros = scalar.expand(count, 36) * 0.0
    return DistributionOutput(
        raw_location=zeros,
        mu=zeros,
        sigma=torch.ones_like(zeros),
        values=scalar.expand(count, 1),
        neglogp=scalar.expand(count) * 0.0,
        entropy=scalar.expand(count) * 0.0,
        rnn_states=(torch.zeros(1, 4, 1),),
    )


def _actor_input(*, old_neglogp):
    count = 160
    return {
        "obs": torch.zeros(count, 236),
        "actions": torch.zeros(count, 36),
        "action_lows": torch.full((count, 36), -1.0),
        "action_highs": torch.full((count, 36), 1.0),
        "episode_start": torch.zeros(count, dtype=torch.bool),
        "rnn_states": (torch.zeros(1, 40, 1),),
        "old_logp_actions": torch.full((count,), old_neglogp),
        "advantages": torch.zeros(count),
        "returns": torch.ones(count, 1),
        "mu": torch.zeros(count, 36),
        "sigma": torch.ones(count, 36),
        "raw_location": torch.zeros(count, 36),
        "normalization_version": 0,
    }


def test_live_likelihood_gate_catches_collection_recompute_mismatch(monkeypatch):
    import video_to_spider.rl.core.ppo as ppo

    monkeypatch.setattr(ppo, "recurrent_evaluate", _fake_recurrent)
    actor = _ScalarActor()
    policy = SimpleNamespace(
        actor=actor,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        distribution=DistributionSpec(),
        normalization_version=0,
    )
    with pytest.raises(RuntimeError, match="live rollout/recomputation"):
        update_actor(
            policy, _actor_input(old_neglogp=1.0), (torch.zeros(1, 4, 1),), PPOConfig()
        )


