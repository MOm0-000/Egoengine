import pytest
import torch

from video_to_spider.rl.core.rollout import (
    compute_gae,
    pack_block_start_states,
    valid_prefix,
    world_major,
)


def test_world_major_mapping_is_explicit():
    values = torch.arange(3 * 2).reshape(3, 2)
    assert torch.equal(world_major(values), torch.tensor([0, 2, 4, 1, 3, 5]))


def test_gae_uses_post_action_done_mask():
    rewards = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1)
    values = torch.zeros_like(rewards)
    done_after = torch.tensor([[False], [True], [False]])
    returns = compute_gae(
        rewards, values, done_after, torch.tensor([[10.0]]), gamma=1.0, tau=1.0
    )
    assert torch.equal(returns[:, 0, 0], torch.tensor([3.0, 2.0, 13.0]))


@pytest.mark.parametrize(
    ("done_after", "expected"),
    [
        ([True, False, False], [1.0, 15.0, 13.0]),
        ([False, True, False], [3.0, 2.0, 13.0]),
        ([False, False, True], [6.0, 5.0, 3.0]),
        ([True, True, False], [1.0, 2.0, 13.0]),
        ([False, False, False], [16.0, 15.0, 13.0]),
    ],
)
def test_gae_terminal_and_collector_truncation_cases(done_after, expected):
    rewards = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1)
    values = torch.zeros_like(rewards)
    returns = compute_gae(
        rewards,
        values,
        torch.tensor(done_after, dtype=torch.bool).reshape(3, 1),
        torch.tensor([[10.0]]),
        gamma=1.0,
        tau=1.0,
    )
    assert torch.equal(returns[:, 0, 0], torch.tensor(expected))


def test_gae_handles_multiple_worlds_independently():
    rewards = torch.ones(3, 2, 1)
    values = torch.zeros_like(rewards)
    done_after = torch.tensor([[False, False], [True, False], [False, True]])
    returns = compute_gae(
        rewards, values, done_after, torch.ones(2, 1), gamma=1.0, tau=1.0
    )
    assert torch.equal(returns[:, 0, 0], torch.tensor([2.0, 1.0, 2.0]))
    assert torch.equal(returns[:, 1, 0], torch.tensor([3.0, 2.0, 1.0]))


def test_block_start_states_are_packed_world_major_with_unique_sentinels():
    blocks, worlds = 3, 2
    states = [
        torch.tensor([100 * block + world for world in range(worlds)]).reshape(1, worlds, 1)
        for block in range(blocks)
    ]
    packed = pack_block_start_states(states, worlds=worlds)
    assert packed[0, :, 0].tolist() == [0, 100, 200, 1, 101, 201]
    indices = torch.tensor([world * blocks for world in range(worlds)])
    assert packed.index_select(1, indices)[0, :, 0].tolist() == [0, 1]


def test_valid_prefix_never_reconnects_after_failure():
    assert valid_prefix([False, True, False], [False, False, False]) == 1
    assert valid_prefix([False, False], [False, True]) == 2
