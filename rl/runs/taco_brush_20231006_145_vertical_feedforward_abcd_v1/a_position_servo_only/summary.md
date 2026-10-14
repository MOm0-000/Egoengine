# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→5` / planned `0→5`
- CPU MuJoCo physics steps: `84`
- Control target timing: `reference_ctrl_linear_substep_interpolation`
- Controller mode: `vertical_gravity_reference_velocity_feedforward_v1`
- Gravity feedforward: `False`
- Reference velocity feedback: `False`
- Maximum applied extra generalized force: `0.000000 N`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`


## Vertical wrist tracking

| actuator | endpoint 4 below reference (mm) | endpoint 5 below reference (mm) |
|---|---:|---:|
| R_forearm_tz_position | 4.133334 | 3.701192 |
| L_forearm_tz_position | 4.131937 | 3.700621 |

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 0.000 / 0.019 | 0 | FAIL |
| bowl | 0.000 / 0.081 | 0 | FAIL |

## Tracking gate

- Tracking failures: `0`
- First tracking failure: `NONE`

## Collision/proxy evidence

- Native blocking observations: `1`
- Endpoint proxy/native geometry mismatch observations: `0`
- Preceding-interval proxy-force/native-clear exposure observations: `0`
- Maximum runtime contact force: `1.533203 N`
- Maximum runtime proxy penetration: `0.218815 mm`
- Substep native hand/table actual minimum: `-1.370980 mm`
- Substep native hand/table reference minimum: `2.433538 mm`
- Reference-safe / actual-penetrating substeps: `7`
- Interval 4->5 both reference and actual safe: `False`
- brush: sampled hand-object impulse `0.000000000 N·s`; mismatch impulse `0.000000000 N·s` (`N/A`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.000000000 N·s`; mismatch impulse `0.000000000 N·s` (`N/A`); mismatch-interval object displacement sum `0.000000 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
