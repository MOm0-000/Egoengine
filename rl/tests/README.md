# Test surfaces

The default command is intentionally the active, fixture-free surface:

```bash
pytest
```

It covers `tests/core/` plus the four runtime contracts named in
`pytest.ini`. It must stay green in a clean checkout and must not depend on
the server-only `/data_all/zzx/3.2RL` asset tree.

The bounded saved-batch parity test is an explicit integration surface:

```bash
OMP_NUM_THREADS=4 \
RL_CORE_ASSET_ROOT=/data_all/zzx/3.2RL \
python -m pytest tests/integration_core
```

It skips with a precise missing-fixture reason when the immutable batch and
checkpoint payloads are not mounted.

Tests tied only to retired Candidate/Gate runners and historical experiment
artifacts live under
`TRASH/rl_active_test_surface_r1_2026-10-01/tests/`. They are excluded from
default collection. To inspect that historical surface explicitly, use:

```bash
PYTHONPATH="$PWD/scripts:$PWD/diagnostics:$PWD/src:$PWD/external/mink/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" \
python -m pytest --override-ini='norecursedirs=' \
  TRASH/rl_active_test_surface_r1_2026-10-01/tests
```

That command may require the historical artifact mounts recorded by those
tests; it is evidence replay, not the active development gate.
