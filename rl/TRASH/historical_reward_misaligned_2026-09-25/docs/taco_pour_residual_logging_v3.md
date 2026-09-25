# Pour residual logging v3

This change fixes an audit-data contract; it does not change PPO, the residual
mapping, rewards, observations, physics, or the 40/40 CPU acceptance rule.

## Three control-target quantities

For every one of the 36 XHand actuator coordinates, v3 records:

```text
requested_residual
    = requested_ctrl - reference_ctrl

effective_residual_after_ctrlrange
    = ctrl_after_ctrlrange - reference_ctrl_after_ctrlrange

residual_lost_to_ctrlrange
    = requested_residual - effective_residual_after_ctrlrange
```

The clamp calculation uses the compiled MuJoCo model's `ctrllimited`,
`ctrlrange`, `mjDSBL_CLAMPCTRL` state, and the same reference/control targets
used for the actual step. Coordinates without an active control range have
zero loss.

`effective_residual_after_ctrlrange` is only the control-target correction
remaining after actuator-range clipping. It is **not** the motion physically
realized in `qpos`; servo response, contact, constraints, and dynamics still
act after this point.

The decomposition is now shared by PPO training visitation, deterministic CPU
validation traces, and `optimized_trajectory.npz`. Newly generated artifacts
do not use the ambiguous `applied_residual` name.

## Historical v2 evidence

The v2 files were not rewritten or deleted. In those immutable artifacts,
`right_wrist_applied_residual` and validation-step `applied_residual` are legacy
names whose actual meaning is the requested residual before actuator
`ctrlrange`. Readers that support both schemas treat them only as that legacy
requested quantity; v2 cannot provide an after-`ctrlrange` value by itself.

## Transparency and numerical regression

The v3 gate repeated the existing one-epoch, four-step CPU audit from the same
complete endpoint-20 boundary and seed, once with logging disabled and once
with logging enabled. Initial and final actor, critic, both optimizers, complete
physics state, RNN state, and Python/NumPy/Torch RNG states were bitwise equal.
The raw v3 NPZ contains all three residual arrays with shape `(4, 36)` and no
legacy ambiguous field.

The same implementation was then applied offline to the frozen 3+1 CPU trace.
It exactly reproduced the prior independent audit:

```text
wrist translation components lost to ctrlrange: 0
wrist rotation components lost to ctrlrange:    0
finger components truncated:                   62
finger components fully blocked:              57
steps with at least one affected finger:       28 / 33
```

Evidence is in
`runs/taco_pour_training_trace_transparency_v3/report.json` and its
`with_logging/training_visitation/` artifacts.

No new grouped scale was selected and no new task-level PPO experiment was
authorized or executed by this change.
