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

The reusable profile remains `engineering_candidate_gate_only`; it cannot
silently enable optimizer training. A separate hash-bound contract authorized
exactly one no-commit experiment with four worlds, eight epochs, seed 0, starts
`[20,20,20,46]`, and the unchanged `0.05` residual scale.

That experiment is complete. Deterministic CPU validation produced:

```text
Replay:                    29 / 40, failed at endpoint 50
Truncated-Gaussian PPO:    37 / 40, failed at endpoint 58
```

The PPO failure score was `1.0136032`, so it remains a strict failure. Across
the 30 endpoints shared with Replay, PPO had a lower tracking score at 28.
During all 1,280 GPU training samples, the ordinary hard clamp changed zero
components and actuator `ctrlrange` removed zero residual components. In CPU
validation, 61 components had only float-level reference-snap residue, with a
maximum of `5.24521e-8`, below the frozen `2e-7` tolerance; no material actuator
clipping occurred.

This supports two limited conclusions: the candidate removed the measured
many-to-one action-channel defect, and the learned policy was useful relative
to the same-run Replay. It did **not** solve the 40-step window and does not
authorize full RL, another seed, another epoch count, or a scale/curriculum
sweep. The historical ordinary-Gaussian 3+1 run reached 32/40, but the apparent
`32 -> 37` cross-run difference is context only because GPU optimization is not
deterministic.

Experiment evidence:

- `configs/taco_pour_state_feasible_truncated_gaussian_experiment_v1.yaml`;
- `runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1/report.json`;
- `runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1/analysis.json`;
- `runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1/cpu_validated_actor.pt.gz`.

The portable actor artifact is `24.7 MB` and matches the actor used for CPU
validation (`actor_state_sha256=39534360...d6711`). It intentionally excludes
the optimizer and asymmetric critic; the full local training checkpoint is not
part of the compact versioned evidence set.
