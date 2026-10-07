# Single-frame calibration formal exam: seed 28

Status: `COMPLETED_ONCE_AND_STOPPED`

This was the single user-authorized formal run using disclosed seed `28`. The
executor knew the seed, but all three algorithms remained blind to both the
seed and private answers. Each isolated solver process received exactly one
frame and one matching model. No method was retried, tuned, or selected using
exam results.

## Integrity

- Source commit: `07c054ba49272e2e6c225e0a4daf7e8f2c633f43`
- Groups: 96
- Frames: 448
- Coordinate-model attempts per method: 896
- Total expected/observed rows: 2688 / 2688
- All predictions existed and were read-only before the only scoring pass.
- All methods received identical frame/model inputs and visibility selection.
- Single-frame mount isolation passed.
- No precision pass threshold was defined in advance, so the exam compares
  coverage, accuracy, tail robustness, and runtime rather than declaring a
  binary pass/fail winner.

## Primary results

Accuracy statistics use only outputs with unique truth and a comparable result.
Coverage is comparable outputs divided by 760 unique-truth rows.

| Method | Comparable / 760 | Coverage | Unique-truth failures | Translation error mm (median / P95 / max) | Rotation error deg (median / P95 / max) | End-to-end seconds |
|---|---:|---:|---:|---:|---:|---:|
| CURRENT_FIXED_SINGLE_FRAME | 655 | 86.18% | 105 | 0.000211 / 13.1123 / 58.3816 | 0.0000142 / 1.13142 / 3.62926 | 663.151 |
| OPEN3D_OFFICIAL_SINGLE_FRAME | 754 | 99.21% | 6 | 0.175397 / 18.4182 / 96.9395 | 0.0120935 / 1.63596 / 6.89045 | 20.465 |
| OPEN3D_HUBER_SINGLE_FRAME | 752 | 98.95% | 8 | 0.177487 / 12.3702 / 41.3722 | 0.0119552 / 1.34718 / 3.14443 | 20.599 |

## TACO-mesh subset

| Method | Comparable / 380 | Coverage | Translation error mm (median / P95 / max) | Rotation error deg (median / P95 / max) | End-to-end seconds |
|---|---:|---:|---:|---:|---:|
| CURRENT_FIXED_SINGLE_FRAME | 275 | 72.37% | 0.000206 / 5.30633 / 12.8953 | 0.0000133 / 0.394766 / 0.923388 | 633.187 |
| OPEN3D_OFFICIAL_SINGLE_FRAME | 380 | 100.00% | 0.226137 / 11.1426 / 29.6501 | 0.0155147 / 0.828783 / 2.27179 | 11.168 |
| OPEN3D_HUBER_SINGLE_FRAME | 380 | 100.00% | 0.225073 / 6.91894 / 34.2728 | 0.0150787 / 0.591241 / 2.16218 | 11.254 |

## Interpretation

- The current project solver is conditionally extremely accurate when it
  returns a comparable answer, but this accuracy is coupled to substantially
  lower coverage. Its conditional median must not be read without the 105
  unique-truth failures (and 169 numerical failures over all rows).
- Official Open3D has the highest overall coverage and is about 32.4 times
  faster end-to-end than the current solver.
- Open3D Huber gives the strongest overall tail robustness: the lowest P95 and
  maximum translation error and the lowest P95 and maximum rotation error,
  while sacrificing two comparable rows relative to official point-to-plane
  Open3D.
- On the TACO-mesh subset, both Open3D methods cover all 380 unique-truth rows;
  the current solver covers 275. Huber has materially better P95 tails than
  official Open3D there.
- Because no precision threshold was frozen, this report does not manufacture
  a binary winner. The operational choice depends on whether near-zero error on
  a subset is more valuable than coverage, speed, and bounded tail error.

The full generated inputs, private answers, and raw outputs remain local under
this directory and are excluded from Git. The committed evidence consists of
this report, the frozen score summary, the integrity report, and hashes.
