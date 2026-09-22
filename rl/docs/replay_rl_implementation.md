# Replay→RL implementation contract

## Scheduling

`src/egoengine_repro/action/replay_rl.py::solve_chunk` implements the approved
two-mode variant of EgoEngine Appendix C.1:

1. At boundary `t`, save the complete simulator state.
2. Start with zero-residual Replay.
3. Validate at most 40 control intervals (`t→t+40`).
4. Save the exact state after interval 20.
5. If all lookahead steps pass, restore the saved endpoint-20 state and commit
   it. If Replay fails, restore the identical incoming boundary, train residual
   PPO, restore the boundary again, and validate the deterministic mean policy.
6. If both modes fail, restore the incoming state and commit nothing.

MPC is absent by user decision. Simulation work counters are not rolled back.
Packed contact state, RNG, reference cursor, previous action/control, lifting
origin, and physics state are part of each snapshot.

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
