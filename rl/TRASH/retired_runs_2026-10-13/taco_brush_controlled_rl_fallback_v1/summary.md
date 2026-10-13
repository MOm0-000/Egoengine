# Brush controlled Replay → RL fallback v1

Classification: `CONTROLLED_RL_FAILED_CLOSED`

- Replay trigger: `CONTROLLED_REPLAY_FAILED_CLOSED`
- bounded fresh RL: `8 epochs / 1280 samples`
- MJWP epoch 0: `23/40`
- MJWP epoch 8: `36/40`
- CPU completed endpoint: `48`
- CPU tracking 40/40: `False`
- brush stable pickup: `False`
- bowl stable pickup: `False`
- maximum contact force: `257.499742 N`
- maximum proxy penetration: `6.144563 mm`
- native blockers: `8`
- proxy/native force mismatches: `4`
- proxy/native geometry mismatches: `4`
- promotion/chunk commit: `forbidden`

A successful optimizer run is not acceptance. Only the independent CPU contact,
tracking, native-geometry and two-object stable-pickup conjunction may pass.
