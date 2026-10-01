import torch

from video_to_spider.rl.core.distribution import (
    deterministic,
    entropy,
    exact_kl,
    log_prob,
    sample,
    support_anchored_bounded_mean,
)


def test_zero_anchor_and_one_sided_derivative():
    raw = torch.zeros(2, requires_grad=True)
    low = torch.tensor([0.0, -1.0])
    high = torch.tensor([1.0, 0.0])
    bounded = support_anchored_bounded_mean(raw, low, high)
    assert torch.equal(bounded, torch.zeros(2))
    bounded.sum().backward()
    assert torch.equal(raw.grad, torch.tensor([1.0, 1.0]))


def test_distribution_paths_share_support_and_identity_kl():
    torch.manual_seed(3)
    raw = torch.randn(8, 36)
    low = torch.full_like(raw, -0.7)
    high = torch.full_like(raw, 0.9)
    mu = support_anchored_bounded_mean(raw, low, high)
    sigma = torch.full_like(raw, 0.25)
    action = sample(mu, sigma, low, high)
    assert bool(((action >= low) & (action <= high)).all())
    assert torch.isfinite(log_prob(action, mu, sigma, low, high)).all()
    assert torch.isfinite(entropy(mu, sigma, low, high)).all()
    assert torch.equal(deterministic(mu, low, high), mu)
    assert float(exact_kl(mu, sigma, mu, sigma, low, high).abs().max()) < 1.0e-12
