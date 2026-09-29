# Endpoint40 lookahead coverage / optimizer adjudication v1

Decision: **LOOKAHEAD_VISITATION_BOTTLENECK**.

No simulator rollout or training was executed. Candidate E was not authorized.

## Deep-lookahead visitation

| Run | epochs 1–62 | 63–125 | 126–188 | 189–250 | 126–250 |
|---|---:|---:|---:|---:|---:|
| successful source20 control | 0.333871 | 0.432143 | — | — | — |
| source40_seed0 | 0.000000 | 0.000694 | 0.001587 | 0.002923 | 0.002250 |
| source40_seed1 | 0.000000 | 0.000298 | 0.001885 | 0.011391 | 0.006600 |
| source40_seed2 | 0.000101 | 0.001389 | 0.002183 | 0.002923 | 0.002550 |

Successful-control reference fraction (epochs 63–125): `0.432142857`.
Failed-run second-half median: `0.002550000`.
Median/control ratio: `0.005900826`.

The failure runs overwhelmingly sample k≤20 and almost never reach source offsets k≥21; k≥25 is zero in every failed segment. This meets the prefrozen visitation-bottleneck rule before optimizer evidence is used.

## Credit join

`credit_join_not_proven`: the two artifact schemas have no shared sample identifier or row-order contract, so advantage/value rows were not guessed into visitation rows.

## Frozen progress

- endpoint20→40 committed: true
- endpoint40→60 committed: false
- Candidate E executed: false
