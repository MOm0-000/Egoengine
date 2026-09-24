# Pour PPO training-time ctrlrange diagnostic

This is one predeclared, non-promotable diagnostic run. It reuses the frozen
3+1 setup without changing PPO, reward, observation, residual mapping, seed, or
budget:

```text
world reset endpoints: [20, 20, 20, 46]
worlds / epochs:        4 / 8
samples per epoch:      160
total samples:          1280
seed:                   0
```

The contract is
`configs/taco_pour_ctrlrange_training_distribution_v1.yaml`. The runner always
restored the incoming endpoint-20 snapshot and wrote neither an optimized
trajectory nor a committed boundary. The CPU Replay/PPO validation rows are
retained only to make the run complete; because GPU training is
non-deterministic, their scores are not a performance comparison with the
earlier 3+1 experiment.

## Measured result

The v3 trace contains the full 36-D requested residual, the control-target
residual left after enabled actuator `ctrlrange`, and their signed difference.
The identity `requested = effective + lost` is exact in all 1,280 samples.

Only finger actuators lost authority. Across all `1,280 × 36 = 46,080`
component samples:

- 2,271 requested components were truncated by `ctrlrange`;
- 2,099 of those requests were fully blocked;
- 1,114/1,280 world-step samples had at least one truncated finger request;
- `sum(abs(lost)) / sum(abs(requested)) = 0.04780075`.

The effect is not confined to the off-policy tail world:

| Scope | Affected world-steps | Truncated components | Fully blocked | Absolute-loss ratio |
| --- | ---: | ---: | ---: | ---: |
| three endpoint-20 anchor worlds | 800/960 | 1,469 | 1,331 | 0.04118 |
| endpoint-46 tail world | 314/320 | 802 | 768 | 0.06713 |
| outcome endpoints 46--53 | 396/411 | 955 | 919 | 0.06172 |

Every epoch had 133--144 affected world-steps. In every one of the 40 rollout
time slots of every epoch, at least one world had a truncated finger request.
Thus this is a training-distribution property, not an event seen only in the
final deterministic CPU failure trajectory.

The loss also cannot be explained only by the formal `±0.05` residual limit:

```text
at ±0.05, then truncated by ctrlrange:      929
below ±0.05, but truncated by ctrlrange:  1,342
at ±0.05, then fully blocked:               839
below ±0.05, but fully blocked:           1,260
```

The largest one-sided losses occur at the two thumb distal rotations and pinky
coordinates. Exact per-actuator positive/negative counts are stored in the
audit rather than collapsed into a hand-wide rate.

## Interpretation boundary

This establishes that PPO trains through a many-to-one action channel: many
different requested finger residuals become the same clipped control target.
It supports discussing an action parameterization that directly represents the
actuator-feasible interval. It does **not** by itself prove that this caused the
endpoint-53 failure, select a new action mapping, select a new scale, or
authorize another training run.

Evidence:

- run: `runs/taco_pour_ctrlrange_training_distribution_v1/report.json`;
- raw v3 epochs: `runs/taco_pour_ctrlrange_training_distribution_v1/ppo_diagnostic_chunk_20/training_visitation/`;
- descriptive audit: `runs/taco_pour_ctrlrange_training_distribution_v1/ctrlrange_distribution_audit.json`.
