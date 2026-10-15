# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→8` / planned `0→8`
- CPU MuJoCo physics steps: `134`
- Control target timing: `reference_ctrl_linear_substep_interpolation`
- Controller mode: `vertical_gravity_reference_velocity_feedforward_v1`
- Gravity feedforward: `True`
- Reference velocity feedback: `True`
- Maximum applied extra generalized force: `11.930681 N`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`


## Vertical wrist tracking

| actuator | endpoint 4 below reference (mm) | endpoint 5 below reference (mm) |
|---|---:|---:|
| R_forearm_tz_position | -1.237896 | -1.051008 |
| L_forearm_tz_position | -1.237458 | -1.050080 |

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 0.000 / 0.089 | 0 | FAIL |
| bowl | 0.560 / 0.118 | 0 | FAIL |

## Tracking gate

- Tracking failures: `0`
- First tracking failure: `NONE`

## Collision/proxy evidence

- Native blocking observations: `3`
- Endpoint proxy/native geometry mismatch observations: `2`
- Preceding-interval proxy-force/native-clear exposure observations: `2`
- Maximum runtime contact force: `7.295884 N`
- Maximum runtime proxy penetration: `0.928633 mm`
- Substep native hand/table actual minimum: `-1.056881 mm`
- Substep native hand/table reference minimum: `-0.103153 mm`
- Reference-safe / actual-penetrating substeps: `13`
- Interval 4->5 both reference and actual safe: `True`
- brush: sampled hand-object impulse `0.000000000 N·s`; mismatch impulse `0.000000000 N·s` (`N/A`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.172699993 N·s`; mismatch impulse `0.172699993 N·s` (`100.000%`); mismatch-interval object displacement sum `0.722741 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
