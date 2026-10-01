# Candidate G saved FULL-update contraction v1

Validity: `COMPLETED_BOUNDED_FULL_UPDATE_CONTRACTION`  
Classification: `NO_RESTORATION_ON_PREDECLARED_SCALES`

## Valid-prefix intervals N

| seed | BASE 0 | FULL 1 | 1/2 | 1/4 | 1/8 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 14 | 14 | 15 | 15 |
| 1 | 20 | 14 | 15 | 15 | 13 |
| 2 | 20 | 20 | 15 | 12 | 15 |

## Fixed-batch exact KL mean

| seed | BASE 0 | FULL 1 | 1/2 | 1/4 | 1/8 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.76566e-13 | 0.521691 | 0.131261 | 0.0328833 | 0.0082275 |
| 1 | 1.89971e-13 | 0.509575 | 0.128095 | 0.0320892 | 0.00802905 |
| 2 | 1.83388e-13 | 0.605544 | 0.151909 | 0.03793 | 0.00947519 |

Common preservation set C: `[]`  
Common extension set E: `[]`

The three BASE actor state hashes are identical
(`b67ae3b491c67a9c6151826579f72487b4a13d290d7d781c67c55635ceb6eafb`).
All six BASE/FULL fixed-batch anchors and all saved closed-loop array fields
regressed successfully. KL contracts smoothly with alpha, while closed-loop N
is non-monotonic and every new condition remains below 20.

## Cost

- fixed-batch forward rows: 2400 in 15 forwards
- deterministic CPU rollouts: 15
- control intervals / physics steps: 600 / 6000
- burn-in / closed-loop actor rows: 300 / 600
- training samples, backward calls, optimizer steps: 0

This is a local, non-paper-faithful parameter intervention. Candidate G remains
`G3_NO_PREDECLARED_SUSTAINED_EXTENSION`; endpoint40→60 remains uncommitted.
