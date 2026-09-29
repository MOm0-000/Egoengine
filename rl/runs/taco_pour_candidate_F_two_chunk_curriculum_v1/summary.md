# Candidate F — Balanced Two-Chunk Lookahead Curriculum

Formal classification: **F3 — TAIL_TRANSITION_REMAINS_BOTTLENECK**.

Candidate F is a local engineering curriculum, not a recovered EgoEngine author setting. Only the rollout reset/start-state distribution changed.

## Fixed-milestone validation

| Seed | Step 0 | 100k / epoch 62 | 200k / epoch 125 | Strict 40/40 |
|---:|---:|---:|---:|:---:|
| 0 | 9/40 (fail@50) | 17/40 (fail@58) | 14/40 (fail@55) | no |
| 1 | 9/40 (fail@50) | 19/40 (fail@60) | 19/40 (fail@60) | no |
| 2 | 9/40 (fail@50) | 20/40 (fail@61) | 20/40 (fail@61) | no |

## Post-curriculum coverage (epochs 63–125)

| Seed | Cohort | Samples | k>=21 | k>=25 | k>=30 | k>=35 | source60 actions | endpoint61 terminations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | all_worlds | 10080 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5073 | 5073 |
| 0 | anchor_worlds | 5040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 33 | 33 |
| 0 | tail_worlds | 5040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5040 | 5040 |
| 1 | all_worlds | 10080 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5103 | 5103 |
| 1 | anchor_worlds | 5040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 63 | 63 |
| 1 | tail_worlds | 5040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5040 | 5040 |
| 2 | all_worlds | 10080 | 0.000099 | 0.000000 | 0.000000 | 0.000000 | 5127 | 5125 |
| 2 | anchor_worlds | 5040 | 0.000198 | 0.000000 | 0.000000 | 0.000000 | 87 | 85 |
| 2 | tail_worlds | 5040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5040 | 5040 |

## Decision

Post-curriculum all-world deep-lookahead fractions: `[0.0, 0.0, 9.92063492063492e-05]`; median: `0.000000`.

Across the three fixed tail cohorts, all `15120` source60 actions terminated for tracking at endpoint61 (`15120`/`15120`).

No fixed validation milestone achieved strict 40/40. No Candidate F chunk was committed. The endpoint20→40 chunk remains committed; endpoint40→60 remains uncommitted.

The frozen decision rules prohibit an automatic curriculum-ratio sweep or a 400k extension. Optimizer/LR review is authorized only by F2.

Verification: tail-boundary and no-training sampler gates passed; the sampler gate used zero optimizer steps. The final repository suite passed 814 tests and 57 subtests (19 warnings).
