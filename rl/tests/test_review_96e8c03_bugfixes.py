"""Regression contracts for the confirmed bugs reviewed at commit 96e8c03."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]

from human2sim2robot.ppo.ppo_agent import PpoAgent
from human2sim2robot.ppo.utils.running_mean_std import RunningMeanStd
from video_to_spider.rl.mjwp_env import (
    MJWPVectorEnv,
    _MJWP_SNAPSHOT_SCHEMA,
    _SNAPSHOT_TORCH_FIELDS,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
)


class _CaptureDataset:
    def update_values_dict(self, values):
        self.values_dict = values


def _ppo_batch(values: torch.Tensor, returns: torch.Tensor) -> dict[str, torch.Tensor]:
    samples = values.shape[0]
    return {
        "obses": torch.zeros(samples, 3),
        "returns": returns,
        "dones": torch.zeros(samples, dtype=torch.uint8),
        "values": values,
        "actions": torch.zeros(samples, 2),
        "neglogpacs": torch.zeros(samples),
        "mus": torch.zeros(samples, 2),
        "sigmas": torch.ones(samples, 2),
    }


def test_value_targets_share_one_normalizer_version_and_raw_gae_is_unchanged() -> None:
    old_values = torch.zeros(160, 1)
    returns = torch.full((160, 1), 10.0)
    raw_advantages = (returns - old_values).sum(dim=1)
    rms = RunningMeanStd((1,))
    agent = SimpleNamespace(
        cfg=SimpleNamespace(
            normalize_value=True,
            freeze_critic=False,
            normalize_advantage=False,
        ),
        value_mean_std=rms,
        is_rnn=False,
        dataset=_CaptureDataset(),
        has_asymmetric_critic=False,
    )

    PpoAgent.prepare_dataset(agent, _ppo_batch(old_values, returns))

    # Both operands must equal a pure transform under the one committed state.
    expected_old = rms(old_values, update_stats=False)
    expected_returns = rms(returns, update_stats=False)
    torch.testing.assert_close(agent.dataset.values_dict["old_values"], expected_old)
    torch.testing.assert_close(agent.dataset.values_dict["returns"], expected_returns)
    torch.testing.assert_close(agent.dataset.values_dict["advantages"], raw_advantages)
    assert rms.count.item() == 1 + old_values.shape[0] + returns.shape[0]

    before = deepcopy(rms.state_dict())
    rms(old_values, update_stats=False)
    rms(returns, update_stats=False)
    for key, value in before.items():
        torch.testing.assert_close(rms.state_dict()[key], value, rtol=0, atol=0)


def _incomplete_snapshot_without_qpos() -> dict[str, object]:
    rng = np.random.default_rng(5)
    state: dict[str, object] = {
        "snapshot_schema": _MJWP_SNAPSHOT_SCHEMA,
        "mujoco_warp_version": version("mujoco-warp"),
        "warp_state_keys": ("qpos", "prev.qpos"),
        "rng_state": deepcopy(rng.bit_generator.state),
        "prev.qpos": torch.tensor([[39.0]], dtype=torch.float32),
    }
    numpy_values = {
        "time_indices": np.array([40], dtype=np.int32),
        "start_indices": np.array([0], dtype=np.int64),
        "episode_lengths": np.array([60], dtype=np.int32),
    }
    state.update(numpy_values)
    state.update({name: torch.zeros(1) for name in _SNAPSHOT_TORCH_FIELDS})
    return state


def test_snapshot_missing_declared_warp_field_fails_before_any_mutation() -> None:
    env = object.__new__(MJWPVectorEnv)
    env.time_indices = np.array([999], dtype=np.int32)
    env._warp_state_keys = lambda: ("qpos", "prev.qpos")
    state = _incomplete_snapshot_without_qpos()

    with pytest.raises(ValueError, match="missing=.*qpos"):
        env.set_env_state(state)

    np.testing.assert_array_equal(env.time_indices, np.array([999], dtype=np.int32))


class _IdentityScheduler:
    @staticmethod
    def update(lr, *_args):
        return lr, 1.0


class _FakeCritic:
    def __init__(self, *, frozen: bool) -> None:
        self.cfg = SimpleNamespace(
            freeze_critic=frozen,
            mini_epochs=2,
            normalize_input=False,
        )
        self.dataset = ({"sample": 0}, {"sample": 1})
        self.num_minibatches = 2
        self.batch_size = 4
        self.epoch_num = 0
        self.frame = 0
        self.lr = 1e-4
        self.scheduler = _IdentityScheduler()
        self.writer = None
        self.gradient_calls = 0

    def train(self):
        return self

    def calc_gradients(self, _sample):
        self.gradient_calls += 1
        return torch.tensor(1.0)

    def update_lr(self, lr):
        self.lr = lr


@pytest.mark.parametrize(
    ("outer_frozen", "critic_frozen", "expected_gradient_calls"),
    [
        (False, False, 4),
        (False, True, 0),
        (True, False, 0),
        (True, True, 0),
    ],
)
def test_asymmetric_critic_freezes_when_either_layer_requests_it(
    outer_frozen: bool,
    critic_frozen: bool,
    expected_gradient_calls: int,
) -> None:
    critic = _FakeCritic(frozen=critic_frozen)
    agent = SimpleNamespace(
        cfg=SimpleNamespace(freeze_critic=outer_frozen),
        asymmetric_critic_net=critic,
    )

    StateFeasibleTruncatedGaussianPpoAgent.train_asymmetric_critic(agent)

    assert critic.gradient_calls == expected_gradient_calls
