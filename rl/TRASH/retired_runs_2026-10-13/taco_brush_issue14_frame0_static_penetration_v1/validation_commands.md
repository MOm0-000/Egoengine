# Validation record

Repository branch: `3.2RL`

Audit command:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_issue14_frame0_static_penetration_v1.py \
  --config rl/configs/taco_brush_issue14_frame0_static_penetration_v1.yaml
```

The audit completed successfully. It executed zero physics steps and did not run
MINK, Replay, MPC, or reinforcement learning.

Default active test command:

```bash
cd /data_all/zzx/Egoengine_upload_3_2/rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_issue14_frame0_static_penetration_v1/pytest.xml
```

Result: `157 passed in 13.06s`.

Focused regression command:

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  rl/tests/core/test_brush_issue14_frame0_static_penetration.py
```

Result: `4 passed in 2.13s`. This includes the unchanged 0.05 mm material
penetration threshold and a closed-mesh containment regression.
