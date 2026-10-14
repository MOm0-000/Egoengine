# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_HARD_STOP`
- Strict gate: `FAIL`
- Completed: endpoint `0→19` / planned `0→121`
- CPU MuJoCo physics steps: `317`
- Control target timing: `reference_ctrl_linear_substep_interpolation`
- Controller mode: `vertical_gravity_reference_velocity_feedforward_v1`
- Gravity feedforward: `True`
- Reference velocity feedback: `True`
- Maximum applied extra generalized force: `42.406007 N`
- Hard stop: `['CONTROLLER_EXTRA_FORCE_LIMIT']`
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
| brush | 0.000 / 0.132 | 0 | FAIL |
| bowl | 0.336 / 0.197 | 0 | FAIL |

## Tracking gate

- Tracking failures: `0`
- First tracking failure: `NONE`

## Collision/proxy evidence

- Native blocking observations: `4`
- Endpoint proxy/native geometry mismatch observations: `0`
- Preceding-interval proxy-force/native-clear exposure observations: `0`
- Maximum runtime contact force: `52.401552 N`
- Maximum runtime proxy penetration: `1.430484 mm`
- Substep native hand/table actual minimum: `-2.978772 mm`
- Substep native hand/table reference minimum: `-0.260213 mm`
- Reference-safe / actual-penetrating substeps: `87`
- Interval 4->5 both reference and actual safe: `True`
- brush: sampled hand-object impulse `0.000000000 N·s`; mismatch impulse `0.000000000 N·s` (`N/A`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.000000000 N·s`; mismatch impulse `0.000000000 N·s` (`N/A`); mismatch-interval object displacement sum `0.000000 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
