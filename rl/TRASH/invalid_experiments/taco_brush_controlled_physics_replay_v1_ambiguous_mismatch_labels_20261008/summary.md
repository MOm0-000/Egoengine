# Brush controlled physics Replay v1

- Classification: `CONTROLLED_REPLAY_FAILED_CLOSED`
- Strict gate: `FAIL`
- Completed: endpoint `0→80` / planned `0→80`
- CPU MuJoCo physics steps: `1334`
- Hard stop: `NONE`
- RL / MPC / MINK: `NOT RUN`
- Promotion / chunk commit: `FORBIDDEN`

## Physical pickup gate

| object | max actual/reference lift (mm) | longest lifted + hand-force + no-floor run | result |
|---|---:|---:|---|
| brush | 5.137 / 114.387 | 0 | FAIL |
| bowl | 0.000 / 57.905 | 0 | FAIL |

## Collision/proxy evidence

- Native blocking observations: `48`
- Proxy/native mismatch observations: `5`
- Maximum runtime contact force: `258.759897 N`
- Maximum runtime proxy penetration: `3.079610 mm`
- brush: sampled hand-object impulse `6.938908014 N·s`; mismatch impulse `1.660891597 N·s` (`23.936%`); mismatch-interval object displacement sum `7.640629 mm`.
- bowl: sampled hand-object impulse `0.048630860 N·s`; mismatch impulse `0.000000000 N·s` (`0.000%`); mismatch-interval object displacement sum `0.000000 mm`.

These impulse/displacement values measure exposure along the proxy-driven Replay. They are not a counterfactual simulation with exact native-mesh contact and therefore do not prove how the true-geometry trajectory would evolve.

A strict PASS is never inferred merely because the bounded Replay reached its final endpoint.
