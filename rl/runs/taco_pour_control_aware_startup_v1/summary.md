# TACO Pour control-aware startup v1

Status: `COMPLETE_NO_PROMOTION`.

This is one bounded, local predictive-sampling experiment over endpoints
0--40. It compares the accepted initial state A and the previously frozen
left-aligned state L under one shared planner. It does not run RL, recover an
author MPC configuration, certify a reset, or authorize a chunk commit.

## Frozen method

- Physical implementation: `dce752247b23137e4860a1dfbb9d67b5e6eb1e59`.
- Replans: sources 0, 5, 10 and 15; every candidate predicts to endpoint 40.
- Search: four rounds, 32 slots per round and start condition; 1,024 total
  candidate slots. The source-15 plan supplies the frozen 20--40 suffix.
- Candidate slot 0 is byte-preserved nominal control, slot 1 is zero Replay,
  and the remaining slots use the shared frozen noise schedule.
- Objective: both object poses, object-frame relation, weak hand--object
  reference relation and control use. This is a local engineering objective.
- MJPC source study pin: `ff572a21e7c2bf9fda62e1862a758da7e9a8719b`.
- REGRIND source study pin: `38347a9e30184620df04e19c63c7c72378cae103`.

## Result

| condition | complete to 40 | planner cost | ctrlrange loss |
|---|---:|---:|---:|
| A_REPLAY | yes | 101.536432 | 0 |
| A_PLAN | yes | 12.009621 | 0 |
| L_REPLAY | yes | 86.160059 | 0 |
| L_PLAN | yes | 28.712533 | 0 |

A_PLAN passed the predeclared conservative numerical no-tradeoff gate against
A_REPLAY. Its 0--40 RMS changes were all improvements: tool position
`-43.500 mm`, tool rotation `-0.015942 rad`, target position `-13.310 mm`,
target rotation `-0.030666 rad`, object-frame relative translation
`-63.286 mm`, and relative rotation `-0.039567 rad`. Endpoint 20 and endpoint
40 also improved on all six required measures.

L_PLAN contains useful improvements but did not pass the no-tradeoff gate. In
0--20 versus L_REPLAY, target position RMS worsened by `2.513 mm`, target
rotation RMS by `0.019000 rad`, and pair translation RMS by `1.405 mm`, while
tool tracking and pair rotation improved. This is recorded as a trade-off, not
hidden behind the lower planner cost.

Both selected 40-control sequences reproduced bitwise in newly constructed
CPU MuJoCo-Warp worlds, including saved state, control and declared contact
evidence. Every selected five-step prefix also matched its forecast bitwise.
Across the noisy search population, `93.75%` of proposed action components
required projection into state-feasible support. The final executed action
bound fractions were much lower (`4.653%` for A_PLAN and `4.028%` for L_PLAN),
and the final actuator ctrlrange loss was exactly zero. The high search
projection rate remains an explicit candidate-generator caveat.

## Visual review

The RGB/source panel, both fixed-camera videos, all declared keyframes, each
condition's own source-14-near contact event, the metric curves and all eight
planning-source plots were generated from saved arrays without new physics.
No new gross object interpenetration, object disappearance or parking artifact
was found at the declared views. A_PLAN visibly preserves the bowl--tray
relation better than A_REPLAY. L_PLAN remains a numerical trade-off despite
passing the limited artifact-oriented visual gate.

## Accounting and boundaries

- Search: 325,320 physics substeps.
- Replay baselines: 800; selected executions: 800; cold replays: 800;
  setup: 4; retest: 0.
- All-in: `327724 / 336000` physical substeps.
- Planner candidate-retention updates: 32; RL optimizer updates: 0;
  actor/critic forwards: 0.
- Active tests after completion: 101 passed.
- No new s20/s40 was promoted, no RL was authorized and no chunk was committed.

Complete immutable arrays, contacts, snapshots, plots, videos, parity reports
and hashes live at
`/data_all/zzx/3.2RL/runs/taco_pour_control_aware_startup_v1`.
