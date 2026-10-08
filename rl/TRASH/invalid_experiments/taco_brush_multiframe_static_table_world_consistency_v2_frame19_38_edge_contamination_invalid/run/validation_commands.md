# Validation commands

The independently selected RGB regions and all full-resolution overlays were frozen in commit `0c0f03fb7acee543432d9d1c7a0b813f014fab04` before the selected-frame Depth planes were computed.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_multiframe_static_table_world_consistency_v2.py
```

The audit uses every positive raw-depth measurement in each frame's independently frozen polygons. It does not use RANSAC, robust loss, a horizontal prior, distance rejection, residual truncation, spatial subsampling, the bowl plane, or calibration.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  rl/tests/core --junitxml \
  rl/runs/taco_brush_multiframe_static_table_world_consistency_v2/pytest_core.xml
```

Result: `132 passed`.

The saved NPZ was also independently re-fitted with the same unconstrained all-point SVD calculation and checked against every stored normal and offset. No selected valid point was omitted.

Result: `921757` saved points; maximum normal and offset reproduction errors were both exactly `0.0`.
