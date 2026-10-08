# Validation commands

Selection preview (executed before any selected-frame Depth was read):

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_multiframe_static_table_world_consistency_v3.py \
  --selection-preview-dir \
  rl/reviews/taco_brush_multiframe_static_table_world_consistency_v3_rgb_selection
```

Frozen v3 audit:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_multiframe_static_table_world_consistency_v3.py
```

Active tests:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=rl/runs/taco_brush_multiframe_static_table_world_consistency_v3/pytest_active.xml
```

Result: `88 passed`.

Independent recomputation loaded every `frame_*_point_world_m` array from
`all_selected_frame_points.npz`, refit all twelve planes with
`fit_all_points_plane`, required exact stored normal/offset equality, and
recomputed the aggregate offset, pairwise-angle, and frame-0-centroid
separation spans. All assertions passed.
