# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→121` / planned `0→121`
- CPU MuJoCo physics steps: `2017`
- Control target timing: `reference_ctrl_linear_substep_interpolation`
- Controller mode: `vertical_gravity_reference_velocity_feedforward_v1`
- Gravity feedforward: `True`
- Reference velocity feedback: `True`
- Maximum applied extra generalized force: `62.894282 N`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`


## Vertical wrist tracking

| actuator | endpoint 4 below reference (mm) | endpoint 5 below reference (mm) |
|---|---:|---:|
| R_forearm_tz_position | -1.238018 | -1.051070 |
| L_forearm_tz_position | -1.237529 | -1.050102 |

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 1.935 / 108.933 | 0 | FAIL |
| bowl | 8.432 / 37.358 | 0 | FAIL |

## Tracking gate

- Tracking failures: `61`
- First tracking failure: `{'endpoint': 33, 'object': 'brush', 'score': 1.0285615163028672, 'position_error_m': 0.1076426730467139, 'rotation_error_rad': 0.7549186168839969}`

## Collision/proxy evidence

- Native blocking observations: `21`
- Endpoint proxy/native geometry mismatch observations: `1`
- Preceding-interval proxy-force/native-clear exposure observations: `3`
- Maximum runtime contact force: `270.314265 N`
- Maximum runtime proxy penetration: `3.614338 mm`
- Substep native hand/table actual minimum: `-3.795511 mm`
- Substep native hand/table reference minimum: `-0.978099 mm`
- Reference-safe / actual-penetrating substeps: `271`
- Interval 4->5 both reference and actual safe: `True`
- brush: sampled hand-object impulse `2.212998435 N·s`; mismatch impulse `0.000000000 N·s` (`0.000%`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.265841738 N·s`; mismatch impulse `0.008230286 N·s` (`3.096%`); mismatch-interval object displacement sum `2.599858 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
