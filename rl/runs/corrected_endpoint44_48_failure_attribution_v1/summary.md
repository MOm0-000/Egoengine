# Corrected endpoint 44--48 failure attribution

This is a read-only analysis of the frozen corrected PPO run. It performs no
optimizer step, does not resume training, and cannot commit a chunk.

## Regression gate

- Replay exactly reproduces the formal trace: 30/40, first failure endpoint 51.
- PPO exactly reproduces the formal trace: 27/40, first failure endpoint 48.
- Both runs start from the same hash-bound corrected endpoint-20 CPU snapshot.

## What changes at endpoints 44--48

PPO still improves the combined objective through endpoint 46 because its
rotation error is roughly 0.5 rad lower than Replay. Its position error is
already worse, however, and continues to grow:

| endpoint | Replay position (m) | PPO position (m) | Replay rotation (rad) | PPO rotation (rad) | Replay score | PPO score |
|---:|---:|---:|---:|---:|---:|---:|
| 44 | 0.07225 | 0.07961 | 0.79835 | 0.23741 | 0.80359 | 0.68206 |
| 45 | 0.07257 | 0.08693 | 0.83797 | 0.31930 | 0.82329 | 0.75505 |
| 46 | 0.07407 | 0.09535 | 0.84602 | 0.32665 | 0.83611 | 0.82389 |
| 47 | 0.07795 | 0.10844 | 0.86549 | 0.36756 | 0.86882 | 0.93629 |
| 48 | 0.08122 | 0.12373 | 0.90107 | 0.40680 | 0.90497 | 1.06612 |

At endpoint 48:

```text
PPO tool position error vector = (-39.70, +107.32, -47.07) mm
PPO minus Replay vector        = (-26.82,  +45.83,  +4.42) mm
position squared contribution  = 1.06307
rotation squared contribution  = 0.07355
```

The position term alone exceeds the unit ellipse. This failure is therefore
not caused by a large rotation error being combined with an otherwise passing
position error.

## Action and contact evidence

The PPO actor is not losing meaningful commands to actuator `ctrlrange`. The
largest reported discrepancy is `1.788e-9`, consistent with float roundoff;
no discrepancy exceeds `1e-8`.

The deterministic right-wrist action does reach the local residual support:

- wrist y: bound at endpoints 44--48;
- wrist z: bound at endpoints 45--48;
- wrist x: bound at endpoints 47--48;
- wrist pitch: bound at endpoints 46--48.

At endpoint 48 the actor means for wrist translation are approximately
`[-1.429, 2.138, 1.318]`, while the executable normalized action is
`[-1, 1, 1]`. This proves that the local residual authority is active in the
failure, but it does not prove that increasing it would improve contact-rich
dynamics.

PPO has right-index/tool contact at endpoints 44--46 and loses it at 47--48.
Replay also changes this contact (absent at 47, present again at 48), so the
coincidence is informative but does not isolate contact loss as the sole cause.
Contact bonus is zero for both modes throughout the window; PPO is not trading
a positive contact bonus for tracking. Lift reward is at most about 0.00066,
far too small to explain the score reversal.

## What training v5 can and cannot establish

- Endpoint 47: 25 samples across all 8 epochs, 2 terminations. The final CPU
  position/rotation/score each lies inside the training min--max range.
- Endpoint 48: 23 samples across all 8 epochs, 2 terminations. The final CPU
  rotation lies inside the training range, but position `0.12373 m` exceeds the
  training maximum `0.11106 m`, and score `1.06612` exceeds the training
  maximum `1.03709`.

Thus the same reference endpoints were visited, but the final endpoint-48
position failure is outside the recorded training metric range. The trace does
not contain complete observation, qpos/qvel, contact/solver buffers or RNN
state, so it must not be described as exact-state coverage.

## Classification

The corrected PPO learned a useful rotation correction but accumulated a
larger bowl-position error. The crossover at 47--48 coincides with multi-axis
right-wrist residual saturation and loss of the recorded right-index/tool
contact. Auxiliary-reward competition and material `ctrlrange` clipping are
not supported. This report does not choose a new scale, reward, curriculum or
training run; a separate contract is required for the next algorithm change.
