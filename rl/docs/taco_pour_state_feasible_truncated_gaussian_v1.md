# State-feasible truncated Gaussian v1

This is a local engineering action distribution, not a published EgoEngine
parameterization. It keeps the approved residual scale at `0.05` and changes
only the policy distribution support.

For each state and actuator:

```text
low  = max(-1, (ctrl_low  - snapped_reference_ctrl) / 0.05)
high = min(+1, (ctrl_high - snapped_reference_ctrl) / 0.05)
action ~ Normal(mu, sigma) truncated to [low, high]
```

The deterministic CPU action is `clip(mu, low, high)`. PPO stores `low/high` in
the rollout and evaluates the exact truncated-normal log probability. The
ordinary `[-1,1]` hard clamp is disabled for this profile; MuJoCo `ctrlrange`
remains a safety backstop and must normally lose zero residual.

The active integration evidence is
`runs/taco_pour_corrected_ppo_gate_v1/report.json`. It uses four worlds all
restored from the corrected endpoint-20 boundary, performs zero optimizer
updates and writes no committed chunk. Historical gates and the historical
3+1 experiment are under `TRASH/historical_reward_misaligned_2026-09-25/`.
