# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→121` / planned `0→121`
- CPU MuJoCo physics steps: `2017`
- Control target timing: `reference_ctrl_outcome_endpoint`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 4.087 / 108.933 | 0 | FAIL |
| bowl | 13.279 / 37.358 | 6 | PASS |

## Tracking gate

- Tracking failures: `59`
- First tracking failure: `{'endpoint': 35, 'object': 'brush', 'score': 1.042158716660128, 'position_error_m': 0.11191055945171452, 'rotation_error_rad': 0.6977409607796496}`

## Collision/proxy evidence

- Native blocking observations: `37`
- Endpoint proxy/native geometry mismatch observations: `3`
- Preceding-interval proxy-force/native-clear exposure observations: `4`
- Maximum runtime contact force: `278.828710 N`
- Maximum runtime proxy penetration: `2.416757 mm`
- brush: sampled hand-object impulse `1.867826892 N·s`; mismatch impulse `0.000000000 N·s` (`0.000%`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.234830660 N·s`; mismatch impulse `0.051847565 N·s` (`22.079%`); mismatch-interval object displacement sum `10.409981 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
