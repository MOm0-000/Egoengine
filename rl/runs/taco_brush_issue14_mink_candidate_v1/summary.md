# Issue #14 Brush fresh MINK candidate

- Classification: `MINK_COMPLETE_FIXED_FRAME_HAND_OBJECT_PENETRATION`
- Complete MINK trajectory: `209/209` frames
- Native hand-table gate: `PASS`
- Self-collision gate: `PASS`
- Frame-0 full native static geometry gate: `PASS`
- Fixed-frame native hand-object geometry gate: `FAIL`
- Fixed-frame native penetrating pair observations: `16`
- Active SupportSurfaceContract modified: `NO`

## Kinematic pickup-alignment screen

| object | maximum lift from frame 0 (mm) | elevated frames | near fraction while elevated | minimum relevant-hand proxy distance (mm) | screen |
|---|---:|---:|---:|---:|---|
| brush | 87.996059 | 85 | 100.000% | -21.067158 | PASS |
| bowl | 61.167479 | 65 | 100.000% | -17.697347 | PASS |

The collision-proxy distance is only a proximity diagnostic. The fixed-frame native-mesh audit is the geometry gate; any penetration there prevents this candidate from being described as a normal pickup.

This is a fresh kinematic MINK result. Object poses follow the official reference and MINK freezes object DOFs; therefore proximity synchronized with object lift is evidence of hand-object alignment, not proof that contact forces physically lift either object. Physics, Replay, MPC, and RL did not run.
