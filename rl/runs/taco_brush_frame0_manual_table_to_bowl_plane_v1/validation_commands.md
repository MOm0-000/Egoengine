# Validation commands

The RGB-only polygons and zero-filter measurement contract were frozen in
commit `03be375abe4ca170c47f6799532f23aaebea3f20` before signed distances were
computed.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_frame0_manual_table_to_bowl_plane_v1.py
```

The audit decodes every native depth frame to exercise the existing exact
uint16 decoder, but measures frame 0 only. It runs no table fit, calibration,
MINK, physics, Replay, MPC, or RL.

Targeted tests:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  rl/tests/core/test_manual_table_to_fixed_plane.py \
  rl/tests/core/test_support_region.py
```

Full active-core tests:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q rl/tests/core
```

Result: `130 passed`. The NPZ contains `129036` points, the compressed CSV
contains exactly `129036` data rows, and direct recomputation of every signed
distance from its saved world point had maximum error `0.0 m`.
