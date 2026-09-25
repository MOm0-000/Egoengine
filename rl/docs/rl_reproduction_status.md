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

The subsequent frozen-state controllability audit rejects the simple
"positive wrist-y scale is too small" explanation. Removing the first
`+0.05 m` y residual at sources 44, 45 and 46 reduces endpoint-48 bowl-y error
by 13.24, 19.43 and 8.57 mm respectively; extending the positive command is
worse. Zeroing wrist-y residual from each source onward changes the final score
from 1.06612 to 0.78347, 0.82702 and 0.98475, all passing at endpoint 48.

At source 47, translation perturbations still move the wrist qpos by roughly
14--15 mm across the grid. Final bowl-y spans only 0.078 mm for wrist y and
0.090 mm for wrist z; wrist x has a larger but still weak and non-monotonic
0.625 mm span. No recorded right fingertip/tool contact is active in those
branches. The late failure is therefore not a wrist actuator that cannot move;
local control transmission to the bowl has become very weak. This makes
increasing y scale the wrong next intervention. The next algorithm decision
must address the earlier saturated y decision/action parameterization, with
contact retention treated as a coupled mechanism.

The follow-up policy-decision attribution shows that formal CPU validation is
already deterministic: it executes `clip(mu, state_low, state_high)` from the
truncated distribution, with no stochastic sample and no squash transform.
Right-wrist-y `mu` rises from 0.498 at source 40 to 0.947 at 42, crosses the
upper bound at source 43 (1.294), and continues to 2.138 at source 47. Thus the
harmful positive-y command is a deterministic policy-function decision, not
evaluation exploration noise.

Setting only one wrist-y action to zero at source 44, 45 or 46, then immediately
returning to the unmodified policy, gives endpoint-48 scores 0.95594, 0.91712
and 0.98475. All three pass that endpoint. A within-support 12.5 mm negative-y
probe still changes endpoint-48 bowl-y by 5.94 mm from source 46, but by only
0.05 mm from source 47. The formal right-index/tool contact is present at
source 46 and absent at 47. This local response is non-smooth, so it is not
called a Jacobian; it nevertheless identifies source 47 as too late for a
useful wrist-y correction.

Contact geom, penetration, normal and force evidence comes directly from the
MJWP contact buffers. CPU `mj_geomDistance` is not used for positive separation
because it is invalid for the active mesh--SDF pair. The next read-only decision
must inspect actor inputs, reference timing and action coordinate frame. Reward,
residual bounds and exploration variance remain frozen.

All old performance, tail-curriculum and objective-mapping results produced with
the off-by-one reward target are archived under
`TRASH/historical_reward_misaligned_2026-09-25/`. They are not active performance
evidence and their actors and boundaries may not be resumed.
