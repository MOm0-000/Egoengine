from types import SimpleNamespace

import pytest
import torch

from video_to_spider.rl.core.env import IndependentWorlds
from video_to_spider.rl.core.policy import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_INPUT_DIM,
    PRIVILEGED_EXTRA_DIM,
    make_actor,
    make_external_critic,
)
from video_to_spider.rl.core.rollout import make_critic_input
from video_to_spider.rl.core.runner import _critic_diagnostics


def test_critic_input_preserves_raw_actor_extra_and_physical_phase():
    actor = torch.arange(4 * ACTOR_OBSERVATION_DIM, dtype=torch.float32).reshape(4, -1)
    extra = -torch.arange(4 * PRIVILEGED_EXTRA_DIM, dtype=torch.float32).reshape(4, -1)
    endpoints = torch.tensor([40, 53, 60, 79], dtype=torch.int64)
    result = make_critic_input(actor, extra, endpoints)
    assert result.shape == (4, CRITIC_INPUT_DIM)
    assert torch.equal(result[:, :ACTOR_OBSERVATION_DIM], actor)
    assert torch.equal(
        result[:, ACTOR_OBSERVATION_DIM:-1], extra
    )
    assert torch.equal(result[:, -1], torch.tensor([0.0, 0.325, 0.5, 0.975]))


def test_actor_task_pose_changes_are_visible_to_critic_prefix():
    actor = torch.zeros(2, ACTOR_OBSERVATION_DIM)
    actor[1, 17] = 0.125
    actor[1, 211] = -0.75
    extra = torch.ones(2, PRIVILEGED_EXTRA_DIM)
    result = make_critic_input(actor, extra, torch.tensor([46, 46]))
    assert not torch.equal(result[0], result[1])
    assert torch.equal(result[:, ACTOR_OBSERVATION_DIM:-1], extra)
    assert torch.equal(result[:, -1], torch.tensor([0.15, 0.15]))


def test_critic_input_rejects_invalid_endpoint_instead_of_clamping():
    with pytest.raises(ValueError, match="outside"):
        make_critic_input(
            torch.zeros(1, ACTOR_OBSERVATION_DIM),
            torch.zeros(1, PRIVILEGED_EXTRA_DIM),
            torch.tensor([81]),
        )


def test_independent_world_phase_reads_each_physical_cursor_without_mutation():
    worlds = [
        SimpleNamespace(num_envs=1, start_indices=[0], time_indices=[40]),
        SimpleNamespace(num_envs=1, start_indices=[13], time_indices=[40]),
        SimpleNamespace(num_envs=1, start_indices=[20], time_indices=[40]),
        SimpleNamespace(num_envs=1, start_indices=[39], time_indices=[40]),
    ]
    environment = IndependentWorlds(worlds)
    assert environment.current_reference_endpoints().tolist() == [40, 53, 60, 79]
    worlds[1].start_indices = [0]
    worlds[1].time_indices = [40]
    assert environment.current_reference_endpoints().tolist() == [40, 40, 60, 79]


def test_new_model_critic_is_345d_while_actor_remains_236d():
    actor = make_actor(worlds=1)
    critic = make_external_critic(worlds=1)
    assert actor.running_mean_std.running_mean.shape == (ACTOR_OBSERVATION_DIM,)
    assert critic.model.running_mean_std.running_mean.shape == (CRITIC_INPUT_DIM,)
    assert critic.model.a2c_network.actor_mlp[0].weight.shape == (1024, CRITIC_INPUT_DIM)


def test_critic_diagnostics_supports_runtime_torch_nonzero_api():
    batch = SimpleNamespace(
        source_endpoint=torch.tensor([40, 41, 60, 61]),
        outcome_endpoint=torch.tensor([41, 42, 61, 62]),
        values=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        returns=torch.tensor([[1.0], [0.8], [0.6], [0.4]]),
        rewards=torch.tensor([[0.5], [0.4], [0.3], [0.2]]),
        world_index=torch.tensor([0, 0, 1, 1]),
        episode_serial=torch.tensor([0, 0, 0, 0]),
        done_after=torch.tensor([False, True, False, False]),
        terminated=torch.tensor([False, True, False, False]),
    )
    diagnostics = _critic_diagnostics(batch)
    assert diagnostics["visible_terminal_episodes"] == 1
    assert diagnostics["collector_right_truncated_episodes"] == 1
    assert diagnostics["visible_terminal_mc_return_fit"]["sources_40_59"]["samples"] == 2
    assert diagnostics["visible_terminal_mc_return_fit"]["sources_60_79"]["samples"] == 0
