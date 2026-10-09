# 20231006_145 initialization feasibility: source geometry

- Classification: `DIRECT_SOURCE_START_AVAILABLE`
- Issue #14 table: world `z=0.548673187534 m`
- First single fully legal source frame: `11`
- First 3-frame legal start: `11`
- MINK / physics / Replay / MPC / RL: `NOT RUN`

## Frame 0

| entity | minimum table distance (mm) |
|---|---:|
| official MANO right | -4.959141 |
| official MANO left | -0.950013 |
| brush | 3.457721 |
| bowl | 0.031921 |

## Frame-0 MANO regions materially below the table

| side | region | minimum (mm) | vertices | near fingertip vertices |
|---|---|---:|---:|---:|
| right | thumb | -2.504860 | 9 | 5 |
| right | middle | -0.809406 | 5 | 5 |
| right | ring | -1.731252 | 6 | 5 |
| right | pinky | -4.959141 | 22 | 19 |
| left | pinky | -0.950013 | 4 | 4 |

## Direct-start contract

Starting at frame 11 would delete original frames `0..10` (0.367 s).
That cropped sequence must not be described as a complete reproduction.

This phase only measures released MANO/object geometry. It does not repair the source, create a robot trajectory, or authorize physics.
