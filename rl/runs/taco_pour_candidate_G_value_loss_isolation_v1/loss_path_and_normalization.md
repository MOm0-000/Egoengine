# Candidate G first-update loss and normalization path

- Rollout GAE/value targets come from the independent asymmetric critic.
- `prepare_dataset()` normalizes `old_values` and `returns` through the asymmetric critic's value RMS; it is called once before all branches.
- The actor optimizer nevertheless contains an internal value head sharing actor MLP/LSTM/layer-normalization features with the policy mean head.
- The exact actor loss is `L_policy + 0.5 * critic_coef * L_internal_value`; frozen `critic_coef=4`, so the weighted internal term is `2 * L_internal_value`.
- The value baseline in the clipped internal loss is the normalized rollout value from the external critic; prediction is the actor's internal value head in that same optimizer dataset coordinate.
- The canonical old policy is the differentiable first evaluation detached only on the denominator side. Ratio is numerically one before the update, but policy gradients remain live.
- Policy mean head/log-sigma have no direct value-loss path; internal value head has no direct policy-loss path; shared MLP/LSTM/layer norm receive both.
- Observation RMS is frozen across reconstruction, gradients, shadow steps, and all four CPU probes. It is deliberately not committed in this experiment.
- Gradient norms are not interpreted as Adam contribution percentages; the paired shadow updates are the causal one-step comparison.
