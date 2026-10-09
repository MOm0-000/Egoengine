# 20231006_145 Issue #14 initialization audit

- Verdict: **OBJECTS_SETTLE_BUT_OFFICIAL_HAND_INITIALIZATION_PENETRATES_TABLE**
- Candidate table: world `z=0.548673187534 m`, from `target 180` frame 20 vertex 3634.
- The candidate remains read-only and was not promoted to the active support contract.

## Frame-0 static geometry

| entity | signed table distance (mm) | status | vertices materially below |
|---|---:|---|---:|
| brush | 3.458 | CLEARANCE | 0 |
| bowl | 0.032 | CONTACT_WITHIN_TOLERANCE | 0 |
| official MANO right | -4.959 | PENETRATION | 42 |
| official MANO left | -0.950 | PENETRATION | 4 |

## Isolated natural settling

| object | velocity variant | stable | first contact (s) | final raw-mesh gap median (mm) | final linear P95 (mm/s) | final angular P95 (rad/s) | final XY drift (mm) | final rotation (deg) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| brush | official_frame0_to_1_velocity | True | 0.02800000000000001 | -0.220 | 0.000 | 0.000000 | 4.579 | 9.917 |
| brush | zero_initial_velocity | True | 0.02800000000000001 | -0.220 | 0.000 | 0.000000 | 4.660 | 9.917 |
| bowl | official_frame0_to_1_velocity | True | 0.004 | -0.216 | 1.214 | 0.044347 | 0.759 | 1.395 |
| bowl | zero_initial_velocity | True | 0.004 | -0.216 | 1.291 | 0.046804 | 0.699 | 1.398 |

## Interpretation

The object check and the hand check are separate. Natural settling may remove a positive object-table gap, but it cannot repair official MANO vertices that start below the same table.
This is a local engineering diagnostic, not a paper-faithful physics reconstruction: measured physical parameters for object 171/180 were not published.
No MINK, full-scene Replay, MPC, or RL was run.
