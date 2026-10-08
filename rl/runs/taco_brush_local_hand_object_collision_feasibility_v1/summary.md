# Brush local hand-object collision feasibility

- Classification: `LOCAL_OBJECT_COLLISION_CONSTRAINT_HAS_GEOMETRIC_SIDE_EFFECTS`
- Full 209-frame retarget: `NOT RUN`
- Physics/training: `NOT RUN`

| frame | object | baseline native (mm) | candidate native (mm) | candidate proxy (mm) | tip mean before/after (mm) | caused >2mm gap while clearing penetration | frame result |
|---:|---|---:|---:|---:|---:|---|---|
| 40 | brush | -2.416511 | 1.161336 | -0.000558 | 20.575729 / 21.975384 | False | `NATIVE_PENETRATION_CLEARED_WITHOUT_GAP_OVER_2MM` |
| 40 | bowl | 18.882395 | 19.018150 | 19.554831 | 20.575729 / 21.975384 | False | `NATIVE_PENETRATION_CLEARED_WITHOUT_GAP_OVER_2MM` |
| 80 | brush | -3.253140 | 3.710459 | 4.815507 | 16.637755 / 27.080691 | True | `PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS` |
| 80 | bowl | -1.054436 | -1.201594 | -0.000900 | 16.637755 / 27.080691 | False | `PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS` |
| 120 | brush | -2.176761 | 3.948637 | 5.358495 | 14.066057 / 27.623637 | True | `PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS` |
| 120 | bowl | -1.036054 | -0.645314 | -0.000442 | 14.066057 / 27.623637 | False | `PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS` |

## Preserved constraints

| frame | tip max before/after (mm) | joint limits | frame displacement | self collision | table support | object pose max change |
|---:|---:|---|---|---|---|---:|
| 40 | 41.786894 / 47.790025 | True | True | True | True | 3.519e-14 |
| 80 | 31.307003 / 39.198929 | True | True | True | True | 1.987e-13 |
| 120 | 28.866852 / 49.361185 | True | True | True | True | 8.182e-14 |

## Decision

- Proxy/native mismatch frames: 80, 120.
- Newly introduced gaps over 2 mm while clearing a baseline penetration: frame 80 brush, frame 120 brush.
- The existing MINK collision proxies are therefore not accepted as a reliable native-mesh hand-object correction for these three frames.
- Candidate promotion and the full 209-frame retarget remain forbidden.

The 2 mm value is a predeclared diagnostic reference only; it did not alter the zero-clearance solve. No threshold was changed after observing the result.
