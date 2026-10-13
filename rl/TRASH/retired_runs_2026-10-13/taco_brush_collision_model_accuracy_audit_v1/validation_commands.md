# Validation commands

Formal code revision: `865686c`

## Fixed-state collision representation audit

```bash
cd /data_all/zzx/Egoengine_upload_3_2
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_collision_model_accuracy_v1.py \
  --config rl/configs/taco_brush_collision_model_accuracy_audit_v1.yaml
```

Result: exit code 0; classification
`NO_EXISTING_DROP_IN_PROXY_UNIFORMLY_FIXES_NATIVE_GEOMETRY`.

## Active test suite

```bash
cd /data_all/zzx/Egoengine_upload_3_2/rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_collision_model_accuracy_audit_v1/pytest.xml
```

Result: `172 passed in 13.51s`.

No MINK optimization, collision-geometry generation, CoACD rerun, physics,
Replay, MPC, RL, full retarget, active-contract mutation, or promotion was
performed.
