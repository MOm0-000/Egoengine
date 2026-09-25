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

All old performance, tail-curriculum and objective-mapping results produced with
the off-by-one reward target are archived under
`TRASH/historical_reward_misaligned_2026-09-25/`. They are not active performance
evidence and their actors and boundaries may not be resumed.
