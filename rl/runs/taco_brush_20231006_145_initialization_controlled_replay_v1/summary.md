# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→121` / planned `0→121`
- CPU MuJoCo physics steps: `2017`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 4.087 / 108.933 | 0 | FAIL |
| bowl | 13.317 / 37.358 | 6 | PASS |

## Tracking gate

- Tracking failures: `59`
- First tracking failure: `{'endpoint': 35, 'object': 'brush', 'score': 1.0416508482269609, 'position_error_m': 0.11204202373056109, 'rotation_error_rad': 0.6927199976330201}`

## Collision/proxy evidence

- Native blocking observations: `36`
- Endpoint proxy/native geometry mismatch observations: `2`
- Preceding-interval proxy-force/native-clear exposure observations: `3`
- Maximum runtime contact force: `278.717235 N`
- Maximum runtime proxy penetration: `2.416869 mm`
- brush: sampled hand-object impulse `1.884016239 N·s`; mismatch impulse `0.000000000 N·s` (`0.000%`); mismatch-interval object displacement sum `0.000000 mm`.
- bowl: sampled hand-object impulse `0.240344691 N·s`; mismatch impulse `0.049104476 N·s` (`20.431%`); mismatch-interval object displacement sum `2.891381 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
