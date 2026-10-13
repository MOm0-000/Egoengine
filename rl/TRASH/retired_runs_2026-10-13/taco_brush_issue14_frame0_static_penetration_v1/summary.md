# Brush frame-0 static penetration audit (Issue #14 scheme A)

The candidate horizontal source-world table is `Z=0.547500948 m`, set by `tool 071` frame `140` vertex `47958`. It maps to simulator `offset=0.717359939 m`, which is `-2.640 mm` relative to the active support.

## Frame-0 native material distance to table

| entity | Issue14 candidate (mm) | status | active support (mm) | status |
|---|---:|---|---:|---|
| brush | 1.328751 | CLEARANCE | -1.311310 | PENETRATION |
| bowl | 2.640059 | CLEARANCE | -0.000002 | CONTACT_WITHIN_TOLERANCE |
| left_xhand | 2.640061 | CLEARANCE | 0.000000 | CONTACT_WITHIN_TOLERANCE |
| right_xhand | 2.640061 | CLEARANCE | -0.000000 | CONTACT_WITHIN_TOLERANCE |

## Native material and MuJoCo shell-pair checks

| group | native pairs | native minimum/status | native penetrating | unknown | shell minimum/status | shell penetrating |
|---|---:|---|---:|---:|---|---:|
| self_explicit | 174 | 3.124607 mm / CLEARANCE | 0 | 0 | 0.008446 mm / CONTACT_WITHIN_TOLERANCE | 0 |
| self_nonadjacent_shells | 256 | 1.115065 mm / CLEARANCE | 0 | 0 | -12.236825 mm / PENETRATION | 17 |
| hand_tool | 24 | 32.929351 mm / CLEARANCE | 0 | 0 | 33.350492 mm / CLEARANCE | 0 |
| hand_target | 24 | 24.910915 mm / CLEARANCE | 0 | 0 | 27.719401 mm / CLEARANCE | 0 |
| tool_target | 1 | 89.667530 mm / CLEARANCE | 0 | 0 | 50.000000 mm / CLEARANCE | 0 |

## Conclusion

- Issue #14 candidate table initialization penetration: `NO`.
- Active support initialization penetration: `YES`.
- Active-support deepest material penetration is the brush at `-1.311310 mm`, native point `[0.6605867176985594, 0.04929095004937277, 0.7186886901152322]` in simulator coordinates.
- Every non-table native mesh-pair check completed with zero material penetrations and zero unknown results. The broad non-adjacent MuJoCo shell diagnostic reports `17` proxy overlaps (minimum `-12.236825 mm`), while its native triangle surfaces remain separated by `1.115065 mm`; proxy overlap is therefore not relabelled as native material penetration.
- This is a static geometry audit only. The candidate support is not promoted, the active contract is unchanged, and no MINK, physics step, Replay, MPC, or RL ran.
