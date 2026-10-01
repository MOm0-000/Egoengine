import torch

from video_to_spider.rl.state_feasible_truncated_gaussian import (
    CanonicalOldPolicyGateSpec,
    _module_parameter_sha256,
    _tensor_tree_sha256,
    canonical_old_policy_identity_report,
)


def _report(
    *,
    rollout_neglogp: torch.Tensor | None = None,
    semantic_override: dict[str, bool] | None = None,
):
    canonical_neglogp = torch.tensor([0.25, -0.5], dtype=torch.float32)
    if rollout_neglogp is None:
        rollout_neglogp = canonical_neglogp.clone()
    canonical_mu = torch.tensor(
        [[0.2, -0.3], [1.1, -0.8]], dtype=torch.float32
    )
    # Deliberately exceed the retired 8-epsilon absolute-mu threshold.  This
    # difference is diagnostic only when the action likelihood is equivalent.
    rollout_mu = canonical_mu.clone()
    rollout_mu[1, 0] += 10.0 * torch.finfo(torch.float32).eps
    sigma = torch.ones_like(canonical_mu)
    semantic = {
        "actor_parameter_hash_equal": True,
        "actor_RMS_hash_equal": True,
        "normalization_version_equal": True,
        "observations_equal": True,
        "dones_equal": True,
        "actions_equal": True,
        "action_low_equal": True,
        "action_high_equal": True,
        "RNN_start_states_equal": True,
    }
    if semantic_override:
        semantic.update(semantic_override)
    external_ratio = torch.exp(rollout_neglogp - canonical_neglogp)
    canonical_ratio = torch.exp(
        canonical_neglogp.detach() - canonical_neglogp
    )
    return canonical_old_policy_identity_report(
        rollout_mu=rollout_mu,
        canonical_mu=canonical_mu,
        rollout_sigma=sigma,
        canonical_sigma=sigma,
        rollout_neglogp=rollout_neglogp,
        canonical_neglogp=canonical_neglogp,
        rollout_to_canonical_ratio=external_ratio,
        canonical_optimizer_ratio=canonical_ratio,
        semantic_identity=semantic,
        gate=CanonicalOldPolicyGateSpec.v1(),
    )


def test_mu_difference_is_diagnostic_but_likelihood_and_semantics_gate():
    report = _report()
    assert report["passed"] is True
    assert report["rollout_vs_canonical"]["max_abs_mu_difference"] == (
        10.0 * torch.finfo(torch.float32).eps
    )
    assert report["rollout_vs_canonical"]["mu_sigma_are_diagnostic_only"] is True
    assert report["canonical_optimizer_identity"]["ratio_bitwise_equal_one"] is True


def test_canonical_ratio_is_one_while_policy_gradient_remains_nonzero():
    parameter = torch.tensor([0.3, -0.2], dtype=torch.float32, requires_grad=True)
    advantage = torch.tensor([1.5, -0.4], dtype=torch.float32)
    neglogp = parameter.square() + 0.1 * parameter
    ratio = torch.exp(neglogp.detach() - neglogp)
    assert torch.equal(ratio, torch.ones_like(ratio))
    loss = -(advantage * ratio).mean()
    loss.backward()
    assert parameter.grad is not None
    assert bool(torch.isfinite(parameter.grad).all())
    assert float(parameter.grad.norm()) > 0.0


def test_parameter_rms_and_rnn_defects_fail_closed_before_canonicalization():
    actor = torch.nn.Linear(3, 2)
    before = _module_parameter_sha256(actor)
    with torch.no_grad():
        actor.weight[0, 0] += 1.0e-3
    assert _module_parameter_sha256(actor) != before

    rnn = (torch.zeros((1, 4, 8)), torch.ones((1, 4, 8)))
    mutated = tuple(value.clone() for value in rnn)
    mutated[0][0, 0, 0] = 1.0
    assert _tensor_tree_sha256(rnn) != _tensor_tree_sha256(mutated)

    assert _report(
        semantic_override={"actor_parameter_hash_equal": False}
    )["passed"] is False
    assert _report(
        semantic_override={"actor_RMS_hash_equal": False}
    )["passed"] is False
    assert _report(
        semantic_override={"RNN_start_states_equal": False}
    )["passed"] is False


def test_wrong_rollout_logprob_is_not_hidden_by_canonical_snapshot():
    wrong = torch.tensor([0.251, -0.499], dtype=torch.float32)
    report = _report(rollout_neglogp=wrong)
    assert report["rollout_vs_canonical"][
        "likelihood_equivalence_passed"
    ] is False
    assert report["semantic_identity_hard_fail"] is True
    assert report["passed"] is False
