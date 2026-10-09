# 20231006_145 safe-entry and XHand initialization feasibility

- Classification: `XHAND_KINEMATIC_INITIALIZATION_PASS`
- Safe replacement: source frames `0..10`; exact join at source frame `11`
- Direct crop deletes `11` frames / `0.367 s`
- Object reference modified: `NO`
- Active support modified: `NO`
- Hand-object zero-margin proxy repair reused: `NO`
- Physics / Replay / MPC / RL: `NOT RUN`

| candidate | status | frames | source start | native init | hand-table min (mm) | tip mean/max (mm) |
|---|---|---:|---:|---|---:|---:|
| direct_crop | MINK_FAIL: QP failed at frame 90: QP solver daqp failed to find a solution. | - | - | - | - | - |
| safe_entry | MINK_PASS | 122 | 0 | PASS | -0.013260 | 16.636/42.527 |

Only candidates passing this kinematic gate may enter a separately recorded controlled Replay. Passing MINK is not physical pickup evidence.
