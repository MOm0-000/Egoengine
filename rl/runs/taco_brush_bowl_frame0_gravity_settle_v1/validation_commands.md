# Validation record

Repository branch: `3.2RL`

Formal experiment command:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/run_taco_brush_bowl_frame0_gravity_settle_v1.py \
  --config rl/configs/taco_brush_bowl_frame0_gravity_settle_v1.yaml
```

Result: `STABLE_SEATING`. Both the recorded frame-0 velocity and zero-velocity
control passed every criterion frozen in the config before the run.

Focused pre-run regression command:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  rl/tests/core/test_brush_bowl_gravity_settle.py \
  rl/tests/core/test_brush_issue14_frame0_static_penetration.py
```

Result: `7 passed in 2.38s`.

Default active-suite command:

```bash
cd /data_all/zzx/Egoengine_upload_3_2/rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_bowl_frame0_gravity_settle_v1/pytest.xml
```

Result: `160 passed in 21.08s`.

The isolated model fidelity gate verified one dynamic body, zero actuators, 32
unchanged convex bowl collision meshes, 32 unchanged explicit bowl-floor pairs,
and unchanged mass, inertia, gravity, time step, solver, geometry properties,
and contact parameters. The frame-0 bowl pose transport was bitwise exact.
