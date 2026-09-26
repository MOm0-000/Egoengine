# Pour post-fix single-actor-pass audit v1

Read-only audit. No training, optimizer step, checkpoint resume or chunk commit occurred.

## Endpoint-40 comparison point

- position error: 0.063990995 m
- rotation error: 0.364829004 rad
- normalized ellipse: 0.586105824
- PPO is no worse than Replay at every endpoint in the 35--40 window
- unit-bound deterministic action components: 6
- final deterministic failure: endpoint 49
- successful intervals: 28/40

## Saved update dynamics

- reconstructed actor updates: 8
- largest exact-KL mean across updates: 8.50014503e-18
- largest ratio-outside-clip fraction: 0.000000
- largest post-optimizer fixed-probe exact-KL mean: 0.0397361809

## Frozen final-policy stochastic diagnostic

- stochastic rollouts: 32
- reached endpoint 40: 31
- passed full 40-step window: 0

No algorithm change is selected by this report; it supplies evidence for the next single-variable decision.
