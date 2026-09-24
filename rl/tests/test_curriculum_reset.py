"""Contracts for paired physical and recurrent curriculum boundaries."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.curriculum_reset import (
    capture_physics_rnn_boundary,
    refresh_boundary_rnn_for_actor,
    restore_physics_rnn_boundaries,
)


class _Normalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.25], dtype=torch.float32))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.running_mean_std = _Normalizer()

    @staticmethod
    def get_default_rnn_state():
        return (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))


class _Agent:
    device = "cpu"
    has_asymmetric_critic = False

    def __init__(self):
        self.model = _Model()
        self.rnn_states = [
            torch.tensor([[[1.0, 2.0]]]),
            torch.tensor([[[3.0, 4.0]]]),
        ]
        self.dones = torch.zeros(1, dtype=torch.uint8)
        self.current_rewards = torch.tensor([[5.0]])
        self.current_shaped_rewards = torch.tensor([[6.0]])
        self.current_lengths = torch.tensor([7.0])
        self.obs = None

    @staticmethod
    def obs_to_tensors(obs):
        return {key: torch.as_tensor(value) for key, value in obs.items()}

    def set_eval(self):
        self.model.eval()

    def get_action_values(self, obs):
        delta = obs["obs"].sum(dim=1).reshape(1, 1, 1)
        return {
            "rnn_states": [state + delta for state in self.rnn_states]
        }


class _World:
    num_envs = 1

    def __init__(self, endpoint: int, value: float):
        self.state = {
            "snapshot_schema": "egoengine_mjwp_snapshot_v2",
            "time_indices": np.array([endpoint], dtype=np.int32),
            "last_terminated": torch.tensor([False]),
            "warp_state_keys": ("qpos",),
            "qpos": torch.tensor([[value]], dtype=torch.float32),
        }

    def get_env_state(self):
        return deepcopy(self.state)

    def set_env_state(self, state):
        self.state = deepcopy(state)

    def current_observation(self):
        value = float(self.state["qpos"][0, 0])
        return {
            "obs": np.array([[value, value + 1]], dtype=np.float32),
            "states": np.array([[value + 2]], dtype=np.float32),
        }


class _Independent:
    def __init__(self, worlds):
        self.worlds = tuple(worlds)
        self.num_envs = len(worlds)

    def current_observation(self):
        return {
            key: np.concatenate([
                world.current_observation()[key] for world in self.worlds
            ], axis=0)
            for key in ("obs", "states")
        }


def _capture(agent, endpoint, value):
    prefix = [
        {
            "obs": np.array([[value, value + 1]], dtype=np.float32),
            "states": np.array([[value + 2]], dtype=np.float32),
        }
        for _ in range(endpoint - 20)
    ]
    return capture_physics_rnn_boundary(
        agent,
        _World(endpoint, value),
        rollout_start_endpoint=20,
        observation_prefix=prefix,
        provenance={"natural_simulator_rollout": True, "gt_state_injected": False},
    )


def test_capture_binds_physics_memory_actor_and_normalization():
    agent = _Agent()
    boundary = _capture(agent, 46, 1.5)
    assert boundary["reference_endpoint"] == 46
    assert boundary["actor_hash_scope"] == (
        "full_model_state_including_input_normalization"
    )
    assert boundary["actor_normalization_state_keys"] == [
        "running_mean_std.mean"
    ]
    assert boundary["exploration_rng_restored"] is False
    torch.testing.assert_close(boundary["rnn_states"][0], agent.rnn_states[0])


def test_four_world_restore_batches_matching_physics_and_rnn_states():
    source_agent = _Agent()
    boundaries = [
        _capture(source_agent, endpoint, float(index + 1))
        for index, endpoint in enumerate((46, 47, 48, 50))
    ]
    target_agent = _Agent()
    target_agent.model.load_state_dict(source_agent.model.state_dict())
    target_agent.rnn_states = [
        torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)
    ]
    env = _Independent([_World(20, 0.0) for _ in range(4)])
    audit = restore_physics_rnn_boundaries(target_agent, env, boundaries)
    assert audit["reference_endpoints"] == [46, 47, 48, 50]
    assert audit["physics_bitwise_equal"] == [True] * 4
    assert audit["rnn_state_shapes"] == [[1, 4, 2], [1, 4, 2]]
    np.testing.assert_array_equal(
        target_agent.obs["obs"].numpy(),
        np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32),
    )


def test_restore_rejects_rnn_memory_after_actor_change():
    source_agent = _Agent()
    boundary = _capture(source_agent, 46, 1.0)
    target_agent = _Agent()
    target_agent.model.load_state_dict(source_agent.model.state_dict())
    with torch.no_grad():
        next(target_agent.model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="different actor/normalization"):
        restore_physics_rnn_boundaries(
            target_agent, _World(20, 0.0), [boundary]
        )


def test_observation_prefix_refresh_rebinds_memory_to_changed_actor():
    source_agent = _Agent()
    boundary = _capture(source_agent, 23, 1.0)
    target_agent = _Agent()
    target_agent.model.load_state_dict(source_agent.model.state_dict())
    with torch.no_grad():
        next(target_agent.model.parameters()).add_(0.01)
    refreshed = refresh_boundary_rnn_for_actor(target_agent, boundary)
    assert refreshed["actor_state_sha256"] != boundary["actor_state_sha256"]
    assert refreshed["provenance"]["rnn_refresh"]["prefix_observations"] == 3
    assert refreshed["provenance"]["rnn_refresh"]["physical_state_changed"] is False
    audit = restore_physics_rnn_boundaries(
        target_agent, _World(20, 0.0), [refreshed]
    )
    assert audit["physics_bitwise_equal"] == [True]


def test_capture_rejects_terminated_tail_state():
    agent = _Agent()
    world = _World(50, 1.0)
    world.state["last_terminated"][:] = True
    with pytest.raises(ValueError, match="terminated"):
        capture_physics_rnn_boundary(
            agent,
            world,
            rollout_start_endpoint=20,
            observation_prefix=[world.current_observation()] * 30,
            provenance={},
        )
