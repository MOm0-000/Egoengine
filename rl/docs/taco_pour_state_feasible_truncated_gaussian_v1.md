# Pour state-feasible truncated-Gaussian candidate

This is a local engineering candidate, not an action distribution recovered
from EgoEngine. It fixes the already measured many-to-one action channel while
keeping the residual scale at `0.05`.

For each state and actuator, the environment derives the normalized interval

```text
low  = max(-1, (ctrl_low  - reference_ctrl) / 0.05)
high = min(+1, (ctrl_high - reference_ctrl) / 0.05)
```

and samples directly from `Normal(mu, sigma)` truncated to that interval. The
deterministic action is `clip(mu, low, high)`. The MuJoCo `ctrlrange` remains a
final safety backstop. The actor's published `mu/sigma`, reward, observation,
network, critic, and `0.05` residual scale are unchanged.

The PPO likelihood includes the truncated-distribution normalizer:

```text
log p(a) = log Normal(a; mu, sigma)
           - log(Phi((high-mu)/sigma) - Phi((low-mu)/sigma))
```

`low/high` are stored beside each rollout action. The ordinary `[-1,1]` clamp
is disabled for this candidate because the distribution already has support
inside both `[-1,1]` and the current actuator-feasible interval.

## Float32 boundary contract

The MJCF decimal limits are not always exactly representable in the runtime's
`float32` controls. A reference may be repaired only when its original excess
is at most `2e-7`. The repaired value is the nearest representable value on the
feasible side of the limit. Action bounds are then checked through the exact
runtime arithmetic `float32(reference + 0.05 * action)` and moved inward by the
minimum representable action step if needed. A larger original reference
violation fails closed; the code never widens `ctrlrange`.

## Gates

The offline gate used all 1,280 states from the frozen v4 distribution audit
and eight samples per state:

- 46,080 state-action intervals were nonempty;
- all 368,640 sampled components were inside their state bounds;
- the ordinary `[-1,1]` clamp changed zero components;
- theoretical actuator-range loss was exactly zero;
- the minimum observed normalization mass was `0.00731560`;
- tail log-probability, entropy, and `mu/sigma` gradients were finite;
- repeated likelihoods were bitwise equal and the same-policy ratio was
  bitwise one.

The real four-world integration gate then used the frozen starts
`[20,20,20,46]` and one 40-step rollout, for 160 samples. It performed no
optimizer update. The `160 x 36` action bounds were present in the rollout
buffer, every sampled action was within them, the ordinary clamp changed zero
components, and all 5,760 recorded `residual_lost_to_ctrlrange` values were
exactly zero.

That gate also exposed and fixed an independent recurrent-likelihood issue.
The curriculum restores a nonzero LSTM memory after an episode reset, whereas
the generic batched `dones` path assumes a zero reset. Candidate likelihood
recomputation now follows the original four-world temporal layout, restores
the same nonzero memory on done, and keeps the existing four-step truncation
boundaries. With that path, recomputed `mu`, `sigma`, log-probability, and PPO
ratio are all bitwise identical to rollout values. Actor, critic, their two
optimizers, and normalization state remain unchanged.

Evidence:

- `runs/taco_pour_truncated_gaussian_offline_gate_v1/report.json`;
- `runs/taco_pour_truncated_gaussian_integration_gate_v1/report.json`.

## Status

The engineering gates pass. The profile remains
`engineering_candidate_gate_only`: task-level PPO, optimizer updates, chunk
commit, and any claim of paper faithfulness remain disabled. A task experiment
requires a separate explicit authorization and promotion of the profile.
