# Pour endpoint 54-60 read-only dynamics audit

This diagnostic used the frozen state-feasible truncated-Gaussian actor and the
same complete CPU endpoint-20 boundary. It performed no training, optimizer
update, action perturbation, chunk commit, or objective promotion.

Endpoints 54-57 belong to the valid prefix of the current axis-intercept run.
Endpoint 58 is its failing endpoint. Endpoints 59-60 are diagnostic continuation
only: they would not exist after formal axis termination, but were previously
proven bitwise identical under the corner-ellipse and independent-threshold
branches.

## Reward/reference indexing defect

The audit found a one-frame mismatch between the declared transition contract
and the runtime reward target.

Before transition `t -> t+1`, the actor correctly sees the known target
`reference[t+1]`. After physics, however, the environment increments its cursor
to `t+1`, builds the next observation, and applies the observation's `+1` goal
offset again. The tracking reward and termination therefore compare the
physical outcome at endpoint `t+1` against `reference[t+2]`.

Across endpoints 54-60, the reported runtime position and rotation errors match
this future reference exactly, with zero recomputation difference. At endpoint
60:

```text
runtime target ref[61]: position error 0.1228889 m
same-time target ref[60]: position error 0.1153932 m
same-time rotation error: 0.6747977 rad
```

Thus the previously reported `2.889 mm` independent-threshold excess is not a
same-time endpoint-60 control error. It is the result of comparing endpoint 60
to reference 61. Against reference 60, both independent thresholds pass.

The actor observation can continue to expose `reference[t+1]` before an action,
but reward computation must be separated from construction of the next
observation. Until that is fixed and the frozen actor is rerun, formal training
and acceptance are fail-closed.

## Remaining read-only observations

These observations describe the frozen, misaligned-runtime trajectory; they do
not establish a new algorithm choice.

- Position was closest to `ref(t-2)` at all seven endpoints, while orientation
  was closest to `ref(t)` at all seven. This supports a translational lag, not a
  uniform full-pose phase shift.
- Same-time position error grew from `0.103213 m` at endpoint 54 to
  `0.115393 m` at endpoint 60. The vertical velocity stayed below the reference
  velocity throughout the interval.
- The right-wrist `y` translation and pitch action hit the global normalized
  residual boundary on all seven incoming actions. These are `[-1,1]` residual
  authority limits, not the much wider wrist actuator `ctrlrange`.
- The right hand retained pinky-tool contact through endpoint 58. That contact
  disappeared at endpoint 59 and index-tool contact appeared at endpoint 60.
  Therefore the reward-facing tool-contact role did not change before the
  original endpoint-58 failure; later contact changes belong only to the
  diagnostic continuation.

The reward-indexing defect must be fixed before using these saturation and lag
observations to justify a new residual scale, another PPO run, or a curriculum
change.

Evidence:

- `configs/taco_pour_terminal_dynamics_diagnostic_v1.yaml`;
- `runs/taco_pour_terminal_dynamics_diagnostic_v1/report.json`;
- `scripts/audit_taco_pour_terminal_dynamics.py`.
