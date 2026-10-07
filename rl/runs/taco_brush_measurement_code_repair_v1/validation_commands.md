# Validation commands

All commands exited with status `0`.  The JUnit XML files contain the actual
collected test cases and timings; no test count is hard-coded in JSON.

## Core geometry, masking, candidate, and summary tests

```bash
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python -m pytest -q \
  --junitxml=rl/runs/taco_brush_measurement_code_repair_v1/targeted_tests.xml \
  rl/tests/core/test_brush_measurement_code_repair.py \
  rl/tests/core/test_support_surface_estimation.py \
  rl/tests/core/test_taco_calibration_visibility.py \
  rl/tests/core/test_taco_calibration_residual.py
```

Result: `34 passed in 5.78s`.

## Open3D 0.20 interface and scoring tests

```bash
PYTHONPATH="$PWD/rl/src:$PWD/rl/scripts" \
/data_all/zzx/calibration_open3d_v0200/env/bin/python -m pytest -q \
  --junitxml=rl/runs/taco_brush_measurement_code_repair_v1/open3d_tests.xml \
  rl/tests/open3d_comparison/test_open3d_single_frame.py \
  rl/tests/open3d_comparison/test_single_frame_scoring.py
```

Result: `23 passed in 0.70s`.

## Frozen-artifact re-audit

```bash
/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python \
  rl/scripts/audit_taco_brush_measurement_code_repair_v1.py
```

Result: 836 rows; frame 139 same-plane tilt
`28.62714056890027 deg`, offset `0.5297505384842496 m`; all frozen historical
input directory hashes unchanged.

The default pytest surface was not run because this task explicitly forbids
robot simulation, MINK, Replay, MPC and RL.  The selected suites cover only
pure geometry, reporting, synthetic calibration interfaces and frozen artifact
reads.
