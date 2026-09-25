# TACO Pour PPO policy-distribution attribution v1

This is a diagnostic-only result. It does not promote a chunk, select a new
action mapping, or provide a performance comparison. GPU training is
non-deterministic; the run exists only to attribute the already-established
training-time action loss between the policy centre and stochastic exploration.

## Frozen run

- Reset endpoints: `[20, 20, 20, 46]`
- Four independent GPU worlds, eight epochs, 1,280 samples, seed 0
- Objective, observation, reward, residual mapping and PPO are unchanged
- Incoming endpoint-20 state restored after the diagnostic
- No committed boundary or `optimized_trajectory.npz` was written

The v4 trace records all 36 dimensions of the sampled action before and after
the official `[-1,1]` clamp, the Gaussian actor `mu` and `sigma`, the reference
control target, and requested/effective/lost residuals. `sigma` is the standard
deviation `exp(logstd)`, not variance. These tensors come from the same official
PPO rollout forward; logging does not run the actor a second time.

The paired CPU transparency audit passed bitwise for initial and final actor,
critic, both optimizers, complete physics state, RNN state, and Python, NumPy
and Torch RNG state. Thus v4 is an observational change.

## State-dependent feasible interval

For the frozen `residual_scale = residual_clip = 0.05`, each actuator uses:

```text
low  = max(-1, (ctrl_low  - reference_ctrl) / 0.05)
high = min(+1, (ctrl_high - reference_ctrl) / 0.05)
```

Float32 reference serialization placed 1,824 values infinitesimally outside a
decimal actuator limit. The maximum discrepancy was `5.24521e-8 rad`. The audit
snaps only discrepancies within the predeclared `2e-7` residual tolerance when
deriving the interval and fails on anything larger. No value exceeded that
tolerance.

## Result

Across 30,720 finger state-component samples:

| Measurement | Result |
| --- | ---: |
| Raw actor `mu` outside the intersected feasible interval | 7,048 / 30,720 (22.94%) |
| Officially-clamped `mu` still outside actuator-feasible interval | 2,369 / 30,720 (7.71%) |
| Observed sample changed by official `[-1,1]` clamp | 12,557 / 30,720 (40.88%) |
| Observed sample lost to actuator `ctrlrange` | 2,147 / 30,720 (6.99%) |
| Fully blocked requested residual | 1,986 |
| `ctrlrange` loss while clamped `mu` was infeasible | 1,552 / 2,147 (72.29%) |
| `ctrlrange` loss while clamped `mu` was feasible | 595 / 2,147 (27.71%) |

The Gaussian calculation agrees with the sampled result. Averaged over finger
state-components, the expected mass outside the official `[-1,1]` interval is
`40.94%`; the expected mass that remains actuator-infeasible after that clamp is
`6.92%` (2,126.78 expected components versus 2,147 observed). Actor `sigma`
stayed close to one (`p05=0.99925`, median `1.00013`, `p95=1.00107`).

The tail distribution is more constrained than the anchor distribution. The
mean expected post-policy-clamp actuator-infeasible finger mass is `10.00%` in
the endpoint-46 world versus `5.90%` in the three anchor worlds. For outcome
endpoints 46--53 it is `9.44%`.

## Interpretation

Both mechanisms are real:

1. Most observed actuator loss (72.29%) occurs when even the officially-clamped
   policy centre is outside the current actuator-feasible interval.
2. A substantial minority (27.71%) occurs when the centre is feasible but the
   Gaussian sample leaves the feasible interval.
3. Independently, the official `[-1,1]` clamp changes roughly 41% of sampled
   finger components, so fixing only actuator `ctrlrange` would not make the
   whole action channel one-to-one.

This supports designing one explicit state-dependent feasible action
parameterization that handles both the bounded policy support and actuator
limits. It does not yet select that parameterization or authorize training.
These are distribution attributions, not proof that either mechanism alone
caused the endpoint tracking failure.

Evidence:

- `runs/taco_pour_training_trace_transparency_v4/report.json`
- `runs/taco_pour_policy_distribution_attribution_v1/report.json`
- `runs/taco_pour_policy_distribution_attribution_v1/policy_distribution_attribution_audit.json`
