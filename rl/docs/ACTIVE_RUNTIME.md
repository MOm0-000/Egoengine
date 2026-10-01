# Active RL runtime

The sole development entrypoint is:

```bash
PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" \
OMP_NUM_THREADS=4 \
/data_all/zzx/egoengine/spider/.venv/bin/python scripts/run_rl.py inspect
```

Bounded verification is explicitly selected with `verify --physics`. A
separate recorded one-epoch smoke has also exercised the complete
collect/GAE/critic/actor/RMS sequence. The R1 entrypoint contains no long-train
command and cannot commit a chunk. Its fixed
task is TACO `20230927_017`, `tool_only`, CPU, source 40 through endpoint 80,
four independent worlds, 40 intervals, BPTT 4.

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

`core/runner.py` remains a single bounded verifier after review: splitting its
asset inspection, parity execution and small report writer would lengthen the
call chain without removing duplication. See `docs/runner_surface_audit_r1.md`.

Immutable checkpoints, batches and trajectories stay in `runs/` because the
new verifier consumes them by SHA-256. They are evidence, not alternate
runtime implementations.
