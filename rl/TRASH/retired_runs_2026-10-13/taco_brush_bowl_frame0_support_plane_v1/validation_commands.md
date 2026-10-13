# Validation commands

The audit used only the bowl mesh and frame-0 official object pose:

```bash
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python \
  rl/scripts/audit_taco_brush_bowl_frame0_support_plane_v1.py
```

The geometry-specific synthetic checks were run with:

```bash
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python -m pytest -q \
  rl/tests/core/test_support_region.py
```

The complete active core suite was run in the MuJoCo-Warp-capable environment:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q rl/tests/core
```

Result: `129 passed`.

No Depth, table estimator, calibration, MINK, physics simulation, Replay, MPC,
or RL command was run for this audit.
