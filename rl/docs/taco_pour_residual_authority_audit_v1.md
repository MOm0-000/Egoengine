# Pour residual-authority audit v1

This audit uses only the already recorded deterministic CPU validation trace
from the frozen 3+1 experiment. It reconstructs source states for transitions
20 to 53, but runs no policy inference, physics, or PPO.

Three quantities are deliberately kept separate:

1. the residual requested by the policy and added to the next reference target;
2. the current actuated `qpos` to next-reference-target gap;
3. the target offset that remains after applying each actuator's `ctrlrange`.

The state gap includes reference motion, servo lag, contact constraints, and
dynamics. It is not an optimal residual label.

## Requested residual use

`Saturated components` means the recorded request reached the current
componentwise `0.05` limit. `Steps with any` counts control intervals in which
at least one component in the group reached that limit.

| Window | Group | Saturated components | Fraction | Steps with any |
|---|---|---:|---:|---:|
| endpoints 21--53 | wrist translation | 36 / 198 | 18.18% | 25 / 33 |
| endpoints 21--53 | wrist rotation | 39 / 198 | 19.70% | 26 / 33 |
| endpoints 21--53 | fingers | 166 / 792 | 20.96% | 33 / 33 |
| last 10, endpoints 44--53 | wrist translation | 10 / 60 | 16.67% | 9 / 10 |
| last 10, endpoints 44--53 | wrist rotation | 18 / 60 | 30.00% | 10 / 10 |
| last 10, endpoints 44--53 | fingers | 53 / 240 | 22.08% | 10 / 10 |
| last 5, endpoints 49--53 | wrist translation | 5 / 30 | 16.67% | 4 / 5 |
| last 5, endpoints 49--53 | wrist rotation | 5 / 30 | 16.67% | 5 / 5 |
| last 5, endpoints 49--53 | fingers | 24 / 120 | 20.00% | 5 / 5 |

This is neither of the two simple cases posed before the audit. Angular and
finger requests do use the bound, but translation is not rarely bounded: the
right wrist alone reaches `0.05 m` in 28 of 99 full-window components. Nor are
all categories comfortably inside the limit.

## Current state to next reference target

The values are absolute per-coordinate gaps before each recorded transition.

| Window / group | Unit | P50 | P90 | P95 | P99 | Max | `0.05 / P95` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full wrist translation | m | 0.0324 | 0.0555 | 0.0610 | 0.0701 | 0.0773 | 0.820 |
| Full wrist rotation | rad | 0.0438 | 0.1157 | 0.1472 | 0.2553 | 0.2840 | 0.340 |
| Full fingers | rad | 0.1001 | 0.3664 | 0.4546 | 0.6773 | 0.9122 | 0.110 |
| Last-10 wrist translation | m | 0.0287 | 0.0450 | 0.0489 | 0.0530 | 0.0533 | 1.022 |
| Last-10 wrist rotation | rad | 0.0449 | 0.0923 | 0.1119 | 0.1160 | 0.1180 | 0.447 |
| Last-10 fingers | rad | 0.1407 | 0.5071 | 0.6532 | 0.7696 | 0.9122 | 0.077 |

These values confirm why reference frame-to-frame increments were insufficient
for calibration. On the executed trajectory, the actual state can lag the next
target by much more than one reference increment. They still do not prescribe
the optimal residual because a larger target offset can worsen contact.

## Actuator control-range headroom

The frozen reference itself stays inside every `ctrlrange`, apart from at most
`6.94e-16` numerical roundoff. Wrist translation and rotation have more than
`0.05` headroom in both directions at every audited endpoint.

Finger targets differ:

- 93 of 792 endpoint-coordinate samples have less than `0.05 rad` negative
  headroom;
- 7 of 792 have less than `0.05 rad` positive headroom;
- 62 requested finger residuals are truncated by `ctrlrange`, across 28 of 33
  steps;
- 57 of those requests are completely blocked because the reference target is
  already at the relevant bound;
- in the last ten steps, 22 of 240 finger requests are truncated, and every
  step has at least one truncation.

The affected controls are right thumb rotation 2, right index joint 1, right
ring joint 2, right pinky joint 2, left thumb rotation 2, and left pinky joint
1. This is distinct from policy saturation: several blocked requests are
smaller than `0.05`, but point beyond an actuator limit.

## Decision

The audit does not support a simple “increase angular scale” experiment:

- all three groups materially use the current limit;
- mixed units remain an unclean action contract, but are not established as
  the primary cause of endpoint-53 failure;
- increasing finger scale alone cannot restore directions already blocked by
  `ctrlrange` and could only create more ineffective commands;
- no new translation, rotation, or finger scale is selected, and no training
  run is authorized.

The complete per-coordinate, per-window, and per-step measurements are in
`runs/taco_pour_residual_authority_audit_v1/report.json`.
