# Replay→RL implementation contract

## Scheduling

`src/egoengine_repro/action/replay_rl.py::solve_chunk_dual_backend` implements
the approved two-mode variant of EgoEngine Appendix C.1 with a local
reproducibility extension:

1. At boundary `t`, save the complete simulator state.
2. Start with zero-residual Replay.
3. Validate at most 40 control intervals (`t→t+40`).
4. Save the exact state after interval 20.
5. Replay and the trained policy are evaluated by CPU MJWP. If all lookahead
   steps pass, restore and commit the CPU endpoint-20 state. If Replay fails,
   copy the exact CPU boundary to GPU, train residual PPO, transfer the actor
   weights to CPU, restore the CPU boundary, and validate the deterministic
   mean policy.
6. If both modes fail, restore the incoming state and commit nothing.

The corresponding fail-closed contract is
`configs/taco_pour_gpu_train_cpu_validate_v1.yaml`. GPU simulation may optimize
the policy but cannot accept a window or provide a committed state. Actor
inference during acceptance also runs on CPU. Every successful CPU commit is
copied field-for-field back to GPU before the next chunk; the transfer is
verified against all snapshot fields.

Each formal output also stores byte-identical snapshots of the protocol,
dual-backend contract, objective profile, observation profile, and simulator
config used by that run. The report hashes those copies, so later status edits
to the live protocol cannot erase the run's exact inputs.

MPC is absent by user decision. Simulation work counters are not rolled back.
Packed contact state, RNG, reference cursor, previous action/control, lifting
origin, and physics state are part of each snapshot. Snapshot schema
`egoengine_mjwp_snapshot_v2` covers all 342 runtime arrays exposed by
MuJoCo-Warp 3.13 across current/previous Data, Contact, and Constraint storage.
The loader rejects legacy partial snapshots, a different MuJoCo-Warp version,
or a missing/extra runtime field before copying any state.

## Endpoint indexing

State endpoint `t` executes `ctrl[t+1] + residual` for ten 300 Hz physics steps,
then advances to endpoint `t+1` and computes the reward against object GT
endpoint `t+1`. The actor input is frozen separately in
`configs/taco_pour_observation_local_236d_v1.yaml`; its goal and primary command
also use `t+1`. This removes the former one-step goal mismatch.

## Fail-closed entry

`scripts/run_taco_replay_rl.py` refuses to allocate the formal environment
unless all of the following pass:

- the selected objective profile is named by `configs/replay_rl_protocol.yaml`;
- the selected observation profile is named and hash-bound by the protocol;
- the initialization report says `accepted_for_replay_rl=true`;
- reset state shape, hashes, qpos/qvel/ctrl provenance, first command, and
  release timing match the report;
- the formal XML contains no remaining object hold equality;
- scene XML, mesh assets, simulator settings, capacity, dependency versions,
  high-resolution SDF depths/node counts, and compiled-model signature reproduce
  the accepted `physics_contract_sha256`.

Scope-specific audit reports may retain `training_ready=false`; for example,
the reset-only report does not authorize training by itself. The integrated
protocol opens only the named local objective/observation combination. It does
not open a paper-faithful run, because the paper omits exact coefficients and
the exact observation encoding.

## Reward and diagnostics

For each tracked object, every validation row records translation error,
rotation error, weighted error, tracking reward, and termination. The
two-object extension uses mean tracking reward and terminates if either object
exceeds the boundary. It also records each hand-object contact bonus and the
three reconstructable reward parts:

```text
total = aggregate_tracking_reward + aggregate_contact_bonus + lift_reward
```

`tool_only` remains the primary single manipulated-object version;
`tool_and_target` is a disclosed local extension.

## Current formal runtime

- config: `runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml`;
- scene: `runs/taco_pour_floor_contact_v1/candidate.xml`;
- reference: `runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz`;
- reset: `runs/taco_pour_initialization_protocol_v2/candidate_a/report.json`;
- active local proxy objective:
  `configs/taco_pour_local_normalized_ellipse_v1.yaml`;
- previous sensitivity comparator:
  `configs/taco_pour_local_unpublished_v1.yaml`;
- observation: `configs/taco_pour_observation_local_236d_v1.yaml`;
- capacities: 256 contacts / 1024 constraints per world;
- required runtime: MuJoCo 3.13, mujoco-warp 3.13, Warp 1.15 for the high-SDF model.

The actual Pour scheduler has now selected PPO on failed Replay windows. Every
committed endpoint and residual action is exported to `optimized_trajectory.npz`.
A full-horizon result is accepted only after the saved action sequence is
replayed from the accepted reset; a chunk-wise success flag alone is no longer
sufficient. Current results and repeatability limitations are in
`docs/rl_reproduction_status.md`.

For bounded PPO diagnostics, `--stop-after-first-ppo` stops immediately after
the first PPO-selected chunk. This separates a short learning check from a
full-horizon run; it does not relax the 40-step validation gate.

## Deterministic validation contract

Complete snapshot restoration does not make GPU MJWP deterministic. In the
Pour endpoint-20 audit, every one of the 342 restored fields was bitwise equal,
but three executions in the same compiled CUDA graph diverged in the first
3.33 ms physics substep. The contact geom set was unchanged when order was
ignored, while contact arrays, constraint rows, forces, qvel, and qpos differed.
This agrees with the
[upstream MJWP documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/#frequently-asked-questions):
GPU executions can differ because of non-deterministic atomic ordering; CPU
execution is the supported deterministic path. Upstream GPU determinism remains
tracked in [issue #562](https://github.com/google-deepmind/mujoco_warp/issues/562).

The copied Spider adapter now executes `mjwarp.step` directly on CPU instead of
trying to capture a CUDA graph. In this scene, CPU MJWP reproduced all ten
physics substeps bitwise and then reproduced both action sequences across three
same-environment and three fresh-environment trials.

This evidence is now bound by the formal dual-backend contract. The runner
creates distinct GPU-training and CPU-validation environments, verifies both
against the same physics contract, records a separate version/source/runtime
hash for each, and verifies each complete CPU→GPU boundary transfer exactly.
After GPU training, the actor state is copied to a fresh CPU inference agent
and its tensor-content hash must remain unchanged. Only the CPU closed-loop
40-step trace can select a mode, and only the CPU endpoint-20 snapshot can be
committed.

The integration has been tested with a real one-epoch GPU PPO update, CPU actor
inference, and a CPU physics step. It has not been used to authorize additional
normalized-ellipse training: the frozen 8-epoch policy remains 38/40 and the
single 16-epoch policy remains 27/40 under deterministic CPU validation. They
are independent GPU training runs, not two checkpoints on one deterministic
learning curve.

The checkpoint/trace audit additionally verified that the input-normalization
buffers are inside `model.state_dict()` and survive strict CPU loading bitwise.
It also verified matching zero-state recurrent semantics at training start,
terminal reset, and CPU validation start. The audit found that direct clipping
of normalized `[-1,1]` output to `[-0.05,0.05]` rad saturated `95.94%` and
`95.63%` of the frozen 8- and 16-epoch action components.

One predeclared local action-scale experiment has now been completed. It used
`delta_a = clip(0.05 u, -0.05, +0.05)` and changed no other algorithm setting.
The runner restored the exact historical endpoint-20 artifact rather than
regenerating an approximately equal state. Saturation fell to `26.57%` and the
maximum adjacent scalar action jump fell to `0.03465 rad`, so continuous action
magnitudes were restored. Task tracking did not improve: deterministic CPU
validation was `29/40`, failed at endpoint 50, and repeated bitwise three times;
the old eight-epoch policy remains `38/40`. The scaled candidate is therefore
not promoted and no longer-budget or seed search was performed. Appendix C.2's
disabled TACO smoothness reward remains unchanged. The scale remains a local,
non-author-recovered candidate, and the result demonstrates that interface
repair alone is not a sufficient PPO improvement.

The subsequent endpoint 40--50 read-only attribution records the recurrent
actor's mean before action limiting. At the critical endpoints 46--50, the
scaled policy's right-wrist translation mean never reached the `[-1,1]` limit
(`max abs=0.9184`), while its applied direction was opposite the old policy in
aggregate (`signed-sum cosine=-0.594`). The new bowl y/z position errors grew
by `+34.99/-26.56 mm`; the old policy reduced y error by `38.96 mm` and kept z
nearly stable. This rejects a simple “same direction, not enough range” account
and blocks a scalar-scale sweep. Before one learning change is selected, the
training rollouts must be audited for coverage of these terminal states and
corrective directions.

Actuator names also show that the shared residual vector mixes wrist slide
targets measured in metres with wrist/finger angles measured in radians. The
single scalar scale across these units is now explicit contract debt; it was not
changed or promoted by this diagnostic.

A formal-runner smoke in `runs/taco_pour_dual_backend_runner_smoke_v1/` used no
PPO training: CPU Replay passed 40/40, the runner committed the CPU endpoint-20
snapshot, GPU simulation work remained zero, and the exact committed snapshot
was verified after both CPU→GPU transfers. Its status is deliberately
`chunk_budget_reached_not_full_task_success`, not task success.
