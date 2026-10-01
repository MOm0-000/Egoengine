# Active RL runtime

The sole development entrypoint is:

```bash
PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" \
OMP_NUM_THREADS=4 \
/data_all/zzx/egoengine/spider/.venv/bin/python scripts/run_rl.py inspect
```

Bounded structural verification is explicitly selected with `verify --physics`.
The active entrypoint now also exposes the one authorized training contract:

```bash
python scripts/run_rl.py verify --config configs/taco_pour_rl_train_v1.yaml --physics
python scripts/run_rl.py train --config configs/taco_pour_rl_train_v1.yaml
python scripts/run_rl.py evaluate --config configs/taco_pour_rl_train_v1.yaml --checkpoint PATH
```

The training command is fail-closed on a successful real two-epoch/cold-resume
verification report. Its fixed task is TACO `20230927_017`, `tool_only`, CPU,
source 40 through endpoint 80, four independent worlds, 40 intervals, BPTT 4,
seed 0 and at most 400,000 training physics steps. It cannot commit a chunk.

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

Immutable checkpoints, batches and trajectories stay in `runs/` because the
new verifier consumes them by SHA-256. They are evidence, not alternate
runtime implementations.
