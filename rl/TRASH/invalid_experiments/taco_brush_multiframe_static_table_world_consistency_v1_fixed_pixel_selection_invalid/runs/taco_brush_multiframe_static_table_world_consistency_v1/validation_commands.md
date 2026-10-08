# Validation commands

The 12 frames and RGB-only polygons were frozen in commit
`8501994193423f0464a2cb697f25aac1a2e1b165` before selected-frame Depth planes
were computed.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_multiframe_static_table_world_consistency_v1.py
```

The audit uses all positive raw-depth measurements in the polygons. There is
no RANSAC, robust loss, horizontal prior, distance rejection, truncation,
subsampling, bowl plane, or calibration.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q rl/tests/core
```

Result: `131 passed`.

The saved NPZ contains `533764` points across 12 frames. Recomputing every
orthogonal plane from those saved world points reproduced every normal and
offset exactly (`0.0` maximum error), and the rejected-point count is zero.
