# Active RL runtime

## Current primary TACO sample

New reproduction work is directed at
`(brush, brush, bowl)/20231006_145`. Its complete official package is stored
outside Git at
`/data_all/zzx/3.2RL/data/taco_v1/brush_brush_bowl_20231006_145`, with tracked
acquisition provenance in `runs/taco_brush_20231006_145_acquisition_v1/`.

The current seven-run evidence chain is listed in `runs/README.md`. The latest
classification is
`STATIC_INITIALIZATION_FEASIBLE_EARLY_PHYSICS_EXECUTION_INVALID_REPLAY_FAILED_CLOSED`:
the DexTrack-style safe entry is kinematically feasible, but controlled Replay
still fails on early native hand/table collision and missing brush pickup.
RL, MPC, promotion, and chunk commit remain forbidden by this evidence.

The former `20230927_027` Brush work and all completed Pour investigations are
historical. Their run directories now live under
`TRASH/retired_runs_2026-10-13/`; they must not be treated as inputs to the
current sample.

The Pour runtime below is retained only to reproduce and inspect immutable
historical evidence.  It is no longer the default target for new experiments,
and its task-specific contracts must not be relabeled or reused as Brush
contracts.

The sole development entrypoint is:

```bash
PYTHONPATH="/data_all/zzx/3.2RL/.env_mjwp313_overlay:$PWD/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" \
OMP_NUM_THREADS=4 \
/data_all/zzx/egoengine/spider/.venv/bin/python scripts/run_rl.py inspect
```

Bounded structural verification is explicitly selected with `verify --physics`.
The active entrypoint also exposes the most recent frozen training contract:

```bash
python scripts/run_rl.py verify --config configs/taco_pour_rl_task_informed_critic_v2.yaml --physics
python scripts/run_rl.py train --config configs/taco_pour_rl_task_informed_critic_v2.yaml
python scripts/run_rl.py evaluate --config configs/taco_pour_rl_task_informed_critic_v2.yaml --checkpoint PATH
```

It also contains the bounded actor-free startup-planning path:

```bash
python scripts/run_rl.py plan-startup --startup-phase preflight
python scripts/run_rl.py plan-startup --startup-phase execute
python scripts/run_rl.py plan-startup --startup-phase analyze
```

The only authorized `taco_pour_control_aware_startup_v1` execution is complete.
It used two frozen full-state starts, four replans per start and no network
forward or RL update. A_PLAN passed the bounded numerical and visual gates;
L_PLAN remained a trade-off. This result does not authorize rerunning the
search, promoting a reset, training RL or committing a chunk. See the archived
`TRASH/retired_runs_2026-10-13/taco_pour_control_aware_startup_v1/summary.md`.

The training command is fail-closed on a successful real two-epoch/cold-resume
verification report. Its fixed task is TACO `20230927_017`, `tool_only`, CPU,
source 40 through endpoint 80, four independent worlds, 40 intervals, BPTT 4,
seed 0 and at most 400,000 training physics steps. It cannot commit a chunk.
The completed v2 decision does not authorize running the command again.

The authorized task-informed critic v2 pilot is complete. Its external critic
receives raw actor observation 236 + privileged extras 108 + physical phase 1;
the environment still returns the unchanged 236-D observation and 108-D
privileged extra. It consumed the full formal budget and ended
`COMPLETED_NO_STRICT_WINDOW_SUCCESS`; the five fixed evaluations validated
`20/18/20/19/19` intervals. This is a bounded negative result, not
authorization to run another seed, extend the budget, select an intermediate
checkpoint, or commit a chunk. See the archived
`TRASH/retired_runs_2026-10-13/taco_pour_rl_task_informed_critic_v2/summary.md`.

The collector uses raw-reward critic values, post-action `done_after`,
pre-forward `episode_start`, real environment reference cursors/timeouts and a
world-major RNN block layout. Boundary hidden state is rebuilt from the
immutable source20--39 observation prefix at every epoch under the current
actor/RMS. Live rollout likelihood is checked against recomputation before the
canonical PPO denominator is formed.

Historical Candidate runners are evidence only and, after R1 verification,
live under `TRASH/rl_core_refactor_r1_legacy_2026-10-01/` rather than beside the
active entrypoint.

The default `pytest` surface is likewise limited to the fixture-free active
core and runtime contracts. Server-fixture parity is selected explicitly with
`pytest tests/integration_core`; Candidate/Gate-specific historical tests live
under `TRASH/rl_active_test_surface_r1_2026-10-01/`. See `tests/README.md` for
the exact commands.

`MJWPVectorEnv` is now an actor-free physics adapter rather than an H2S2R
trainer-compatible environment. Its retained physical closure and deliberately
removed historical hooks are listed in
`docs/mjwp_env_active_surface_audit_r1.md`.

`core/runner.py` remains the only CLI dispatcher. The earlier no-split decision
in `docs/runner_surface_audit_r1.md` describes the R1 verification-only surface;
the subsequently authorized training path is kept in the same core rather than
reintroducing a Candidate-specific trainer.

Only immutable checkpoints, batches, and trajectories consumed by the active
verifier or tests stay in `runs/`. Completed historical evidence lives in the
dated archive and is not an alternate runtime implementation.
