# Validation commands

Formal code revision: `c7ed7c6`

## Isolated three-frame audit

```bash
cd /data_all/zzx/Egoengine_upload_3_2
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/audit_taco_brush_local_hand_object_collision_feasibility_v1.py \
  --config rl/configs/taco_brush_local_hand_object_collision_feasibility_v1.yaml
```

Result: exit code 0; classification
`LOCAL_OBJECT_COLLISION_CONSTRAINT_HAS_GEOMETRIC_SIDE_EFFECTS`.

## Active test suite

```bash
cd /data_all/zzx/Egoengine_upload_3_2/rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_local_hand_object_collision_feasibility_v1/pytest.xml
```

Result: `169 passed in 13.18s`.

No full 209-frame retarget, physics simulation, Replay, MPC, RL, candidate
promotion, or active-contract mutation was performed.
