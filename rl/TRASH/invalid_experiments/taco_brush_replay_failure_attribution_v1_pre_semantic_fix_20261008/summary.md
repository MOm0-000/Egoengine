# Brush Replay failure mechanism attribution v1

- Observer parity: `PASS`
- Classification: `REPLAY_FAILURE_MECHANISM_ATTRIBUTED`
- Commands / physics / geometry changed: `NO`
- RL / MPC / MINK: `NOT RUN`

## Brush, outcomes 15–41

| finger | normal impulse (N·s) | object impulse xyz (N·s) | max force (N) | endpoints |
|---|---:|---|---:|---|
| index | 11.353390 | [3.543343, 1.975347, -7.327491] | 41.229224 | 15–41 |
| thumb | 2.962360 | [-0.812084, 1.553467, -2.460948] | 11.178922 | 26–40 |

- Hand vertical impulse: `-9.788439 N·s`
- Floor vertical impulse: `11.367087 N·s`
- Gravity impulse: `-1.583615 N·s`

## Left-hand self collision, outcomes 44–45

- Peak proxy pair: `collision_hand_left_index_0` / `collision_hand_left_middle_0`
- Peak force: `258.759897 N`
- Proxy signed distance: `-2.161202 mm`
- Native visual-mesh result: `CLEARANCE`; distance `0.650116 mm`

## Bowl, outcomes 52–60

| finger | normal impulse (N·s) | object impulse xyz (N·s) | max force (N) | endpoints |
|---|---:|---|---:|---|
| index | 0.156785 | [0.148471, 0.053446, 0.021404] | 6.689237 | 52–60 |
| middle | 0.000198 | [1.1e-05, 1.4e-05, 0.000279] | 0.139989 | 52–52 |

- Hand vertical impulse: `0.021684 N·s`
- Floor vertical impulse: `0.311131 N·s`
- Gravity impulse: `-0.333395 N·s`

Native geometry is an observer only. Force directions come from the exact replayed MuJoCo contact frame; no native-mesh force counterfactual is claimed.
