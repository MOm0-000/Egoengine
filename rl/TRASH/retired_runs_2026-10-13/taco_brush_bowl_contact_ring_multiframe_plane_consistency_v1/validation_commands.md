# Validation commands

The selected frames, fixed 216-vertex annular contact region, and fit contract were frozen in commit `ccf969b9eb14ece142774bb9ea5ffc5de9793f62`. The predeclared early/late aggregation was added in `82892e9080518551abf33a4d7705a4f586a7f9f8`, before this final clean-worktree run.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_bowl_contact_ring_multiframe_plane_consistency_v1.py
```

The run reads only bowl mesh 146, its official pose array, and the previously validated fixed contact-ring indices. It does not read Depth, estimate a table, run calibration, or modify the active support contract.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  rl/tests/core --junitxml \
  rl/runs/taco_brush_bowl_contact_ring_multiframe_plane_consistency_v1/pytest_core.xml
```

Result: `134 passed`.

The saved NPZ was independently re-fitted after the run. All `12 × 216` points were present; maximum normal and offset reproduction errors were both exactly `0.0`.
