# Pour mixed-unit reference-action scale audit v1

This read-only audit compares the current componentwise residual limit with
the 197 consecutive control-target changes in the frozen 198-frame Pour robot
reference. It ran no simulator and no PPO, changed no input, and does not treat
reference command changes as realized robot motion or an optimal residual.

The formal 36-dimensional actuator order was checked from the compiled model.
Each hand has three wrist translations in metres, three wrist rotations in
radians, and twelve finger joints in radians. The current local mapping applies
the same numeric limit, `0.05`, to every component. Consequently its physical
meaning is `0.05 m` for translations and `0.05 rad` for rotations and fingers.

## Aggregate component statistics

All values below are absolute consecutive increments, flattened within the
stated group. `limit/P95` compares the current component limit with the P95
reference increment. P95 is only a descriptive typical-high comparator, not a
success threshold.

| Hand / group | Unit | P50 | P90 | P95 | P99 | Max | limit/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Right wrist translation | m | 0.001127 | 0.006713 | 0.009195 | 0.012064 | 0.023086 | 5.438 |
| Right wrist rotation | rad | 0.009245 | 0.049577 | 0.080858 | 0.133332 | 0.133332 | 0.618 |
| Right finger joints | rad | 0.006220 | 0.091347 | 0.140979 | 0.248026 | 0.266666 | 0.355 |
| Left wrist translation | m | 0.000954 | 0.004950 | 0.006761 | 0.009225 | 0.015834 | 7.395 |
| Left wrist rotation | rad | 0.008188 | 0.057177 | 0.084534 | 0.128923 | 0.133332 | 0.591 |
| Left finger joints | rad | 0.008282 | 0.079872 | 0.121396 | 0.234924 | 0.266666 | 0.412 |

The same numeric residual limit is therefore several times a typical-high
wrist translation step, but smaller than a typical-high wrist rotation or
aggregate finger step. The reference dynamics do not numerically balance the
mixed units.

## Per-coordinate P95 check

| Coordinate | Right P95 | Right limit/P95 | Left P95 | Left limit/P95 |
|---|---:|---:|---:|---:|
| wrist translation x (m) | 0.005874 | 8.512 | 0.007966 | 6.277 |
| wrist translation y (m) | 0.009037 | 5.533 | 0.005854 | 8.541 |
| wrist translation z (m) | 0.010300 | 4.854 | 0.006292 | 7.947 |
| wrist roll (rad) | 0.059667 | 0.838 | 0.090467 | 0.553 |
| wrist pitch (rad) | 0.093162 | 0.537 | 0.075666 | 0.661 |
| wrist yaw (rad) | 0.078046 | 0.641 | 0.062949 | 0.794 |
| thumb bend (rad) | 0.103849 | 0.481 | 0.067607 | 0.740 |
| thumb rotation 1 (rad) | 0.100947 | 0.495 | 0.041758 | 1.197 |
| thumb rotation 2 (rad) | 0.093370 | 0.536 | 0.007446 | 6.715 |
| index bend (rad) | 0.026206 | 1.908 | 0.011241 | 4.448 |
| index joint 1 (rad) | 0.167169 | 0.299 | 0.115137 | 0.434 |
| index joint 2 (rad) | 0.116385 | 0.430 | 0.111113 | 0.450 |
| middle joint 1 (rad) | 0.171048 | 0.292 | 0.101963 | 0.490 |
| middle joint 2 (rad) | 0.134423 | 0.372 | 0.154321 | 0.324 |
| ring joint 1 (rad) | 0.190041 | 0.263 | 0.144974 | 0.345 |
| ring joint 2 (rad) | 0.133903 | 0.373 | 0.166367 | 0.301 |
| pinky joint 1 (rad) | 0.266666 | 0.188 | 0.189271 | 0.264 |
| pinky joint 2 (rad) | 0.151727 | 0.330 | 0.168495 | 0.297 |

The two unusually slow coordinates (left thumb rotation 2 and both index-bend
coordinates) also show why a single quantile cannot prescribe a safe new
scale. This audit establishes the mixed-unit imbalance only. It does not select
translation, wrist-rotation, or finger scales and does not authorize another
training run.

The complete P50/P90/P95/P99/max table for every coordinate, input hashes, and
limitations are stored in
`runs/taco_pour_reference_action_scale_audit_v1/report.json`.
