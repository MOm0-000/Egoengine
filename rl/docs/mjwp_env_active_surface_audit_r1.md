# MJWP environment active-surface audit R1

## Scope

This audit follows the sole active path
`scripts/run_rl.py -> core.runner -> core.env -> MJWPVectorEnv`. It narrows the
local adapter; it does not replace Spider/MuJoCo-Warp physics, reorder substeps,
change contact bookkeeping, or weaken complete snapshots.

## Direct active public surface

`core.env` and the bounded verifier use:

- construction through `MJWPVectorEnv` / `MJWPVectorEnvConfig`;
- `current_observation` and `step`;
- `enable_state_feasible_action_contract` and
  `current_normalized_action_bounds`;
- `set_chunk_reset`, `get_env_state`, and `set_env_state`;
- the immutable `num_envs` count.

The active call graph keeps four independent one-world instances. It does not
use a packed four-world contact buffer.

## Required implementation closure

The following remain local implementation details because the direct surface
depends on them:

- reference controls, residual application, tracking reward and termination;
- actor and privileged observations;
- contact map resolution and live contact features;
- exact reset of selected worlds;
- complete Warp/contact/constraint/previous-state snapshot validation and copy;
- per-substep capacity checks and the optional read-only substep observer.

These are physical/runtime complexity, not historical orchestration, and were
not rewritten.

## Conservatively retained reusable capability

The config still describes pre-grasp sampling, zero-valued noise/domain fields,
single or bimanual object layouts, and one or more worlds. The fixed R1 builder
uses only one-world CPU instances and zero noise. These branches were retained
because deleting them would touch reset or physics semantics; their mere width
is not evidence that they are historical-only.

## Retired historical-only surface

The following had no active caller outside the retired Human2Sim2Robot trainer
or historical audit runners and were removed from the active module:

- the Gym/H2S2R `get_env_info`, `get_number_of_agents`, and public `reset`
  compatibility API;
- `set_train_info` and the environment-owned PPO training trace hooks;
- the diagnostic-only reference-snapping helper and old action-audit accessor.

This also removes the active imports of Gym, `PpoTrainingTrace`, and residual
trace reconstruction. Audit records now remain a trainer/audit concern, as
required by the core boundary. The byte-preserved pre-narrowing implementation
is recoverable under
`TRASH/rl_core_refactor_r1_legacy_2026-10-01/src/video_to_spider/rl/mjwp_env.py`.

## Result

The file decreased from 1,465 to 1,245 lines. The reduction is deliberately
limited: complete snapshots, contacts, reward/observation construction,
independent-world isolation and the exact physics step sequence remain intact.
