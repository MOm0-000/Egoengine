# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→121` / planned `0→121`
- CPU MuJoCo physics steps: `2017`
- Control target timing: `reference_ctrl_linear_substep_interpolation`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 4.418 / 108.933 | 0 | FAIL |
| bowl | 13.663 / 37.358 | 6 | PASS |

## Tracking gate

- Tracking failures: `59`
- First tracking failure: `{'endpoint': 35, 'object': 'brush', 'score': 1.0365011770641523, 'position_error_m': 0.11188263978587452, 'rotation_error_rad': 0.6792337653725173}`

## Collision/proxy evidence

- Native blocking observations: `36`
- Endpoint proxy/native geometry mismatch observations: `2`
- Preceding-interval proxy-force/native-clear exposure observations: `2`
- Maximum runtime contact force: `270.305639 N`
- Maximum runtime proxy penetration: `2.467667 mm`
- Substep native hand/table actual minimum: `-3.495160 mm`
- Substep native hand/table reference minimum: `-0.978099 mm`
- Reference-safe / actual-penetrating substeps: `704`
- Interval 4->5 both reference and actual safe: `False`
- brush: sampled hand-object impulse `1.874161543 N·s`; mismatch impulse `0.000000000 N·s` (`0.000%`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.258472769 N·s`; mismatch impulse `0.040954934 N·s` (`15.845%`); mismatch-interval object displacement sum `6.836183 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
