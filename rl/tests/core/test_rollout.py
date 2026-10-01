import torch

from video_to_spider.rl.core.rollout import compute_gae, valid_prefix, world_major


def test_world_major_mapping_is_explicit():
    values = torch.arange(3 * 2).reshape(3, 2)
    assert torch.equal(world_major(values), torch.tensor([0, 2, 4, 1, 3, 5]))


def test_gae_uses_done_mask_and_bootstraps_right_edge():
    rewards = torch.ones(3, 2, 1)
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[0, 0], [1, 0], [0, 0]], dtype=torch.bool)
    returns = compute_gae(rewards, values, dones, torch.ones(2, 1), gamma=1.0, tau=1.0)
    assert torch.equal(returns[:, 0, 0], torch.tensor([1.0, 3.0, 2.0]))
    assert torch.equal(returns[:, 1, 0], torch.tensor([4.0, 3.0, 2.0]))


def test_valid_prefix_never_reconnects_after_failure():
    assert valid_prefix([False, True, False], [False, False, False]) == 1
    assert valid_prefix([False, False], [False, True]) == 2
