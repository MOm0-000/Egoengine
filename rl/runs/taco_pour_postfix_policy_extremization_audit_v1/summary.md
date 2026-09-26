# Pour post-fix PPO policy-extremization audit v1

Read-only audit. No training, optimizer step, checkpoint resume or chunk commit occurred.

## Endpoint-40 failure

- position error: 0.094780728 m
- rotation error: 0.958448827 rad
- normalized ellipse: 1.015934944
- first worse PPO score in the 35--40 window: endpoint 35
- unit-bound deterministic action components: 7

## Saved update dynamics

- reconstructed actor updates: 32
- largest exact-KL mean across updates: 0.212340225
- largest ratio-outside-clip fraction: 0.837500

## Frozen final-policy stochastic diagnostic

- stochastic rollouts: 32
- reached endpoint 40: 29
- passed full 40-step window: 0

No algorithm change is selected by this report; it supplies evidence for the next single-variable decision.
