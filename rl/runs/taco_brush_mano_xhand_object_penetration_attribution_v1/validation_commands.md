# Validation commands

Formal audit revision: `f02c90c`.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_mano_xhand_object_penetration_attribution_v1.py \
  --config rl/configs/taco_brush_mano_xhand_object_penetration_attribution_v1.yaml

cd rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_mano_xhand_object_penetration_attribution_v1/pytest.xml
```

The audit performed zero retarget calls, zero physics steps, and zero training
steps. It did not modify the table, object poses, MINK settings, or source and
robot trajectories.
