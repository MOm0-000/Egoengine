# Pour Replay→RL current status

Only reward-aligned evidence is active.

- State transition: source `t`, command `ref[t+1]`, physical outcome `t+1`.
- Reward and termination: physical state `t+1` versus object reference `ref[t+1]`.
- Next actor observation: goal `ref[t+2]`, preview command `ctrl[t+3]` after the step.
- Snapshot schema: `egoengine_mjwp_snapshot_v3_reward_aligned`.
- Training trace schema: `taco_ppo_training_visitation_v5`.

The corrected Replay chain was rebuilt from accepted endpoint 0. Replay passed the
first 40-step lookahead and committed a new CPU endpoint-20 boundary. From that
new boundary, Replay passed 30 intervals and first failed at outcome endpoint 51.
No PPO was run during this rebase.

The current local action contract is the state-feasible truncated Gaussian. The
zero-optimizer 4-world gate starts all worlds from the new endpoint-20 boundary,
uses exact truncated-normal likelihoods, changes no action through the official
`[-1,1]` clamp, and loses no requested residual to actuator `ctrlrange`.

The one authorized fresh PPO experiment from `[20,20,20,20]` has completed with
4 worlds, 8 epochs, 1280 samples and seed 0. Replay passed 30/40; PPO passed
27/40 and failed at endpoint 48. CPU MuJoCo-Warp committed nothing, so the
formal boundary remains endpoint 20. No further sweep is authorized.

The read-only endpoint 44--48 attribution reproduces both formal CPU traces
exactly. PPO keeps a much smaller rotation error than Replay, but its position
error grows from 79.61 mm at endpoint 44 to 123.73 mm at endpoint 48. At the
failure, the position term alone is 1.06307, already outside the unit ellipse;
the rotation term is only 0.07355. The largest PPO-versus-Replay position
difference is the bowl y error (+45.83 mm at endpoint 48).

This is not explained by the auxiliary rewards or actuator `ctrlrange`:
contact bonus is zero for both modes throughout 44--48, lift reward remains
below 0.00066, and the largest after-`ctrlrange` residual discrepancy is only
1.79e-9. It does coincide with residual-support saturation: right wrist y is
at its local action bound throughout 44--48, z from 45, and x from 47. The
right index/tool contact present for PPO at 44--46 is absent at 47--48.

Training v5 visited endpoints 47 and 48 in all eight epochs (25 and 23 samples,
respectively). The final CPU endpoint-48 position error and objective score are
nevertheless outside the complete training-sample ranges at that endpoint.
This is metric-level evidence only: v5 did not save complete observations,
physics/solver state, or RNN state, so it cannot prove or disprove visitation
of the exact final CPU state.

All old performance, tail-curriculum and objective-mapping results produced with
the off-by-one reward target are archived under
`TRASH/historical_reward_misaligned_2026-09-25/`. They are not active performance
evidence and their actors and boundaries may not be resumed.
