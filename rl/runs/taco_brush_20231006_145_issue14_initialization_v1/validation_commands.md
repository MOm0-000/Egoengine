# Validation

- Formal audit:
  `/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python /data_all/zzx/Egoengine_upload_3_2/rl/scripts/audit_taco_brush_20231006_145_issue14_initialization_v1.py --config /data_all/zzx/Egoengine_upload_3_2/rl/configs/taco_brush_20231006_145_issue14_initialization_v1.yaml`
- Active suite, from `/data_all/zzx/Egoengine_upload_3_2/rl`:
  `PYTHONPATH="$PWD/.env_mjwp313_overlay:$PWD/src:$PWD/external/human2sim2robot:$PWD/external/spider_compat" OMP_NUM_THREADS=4 /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q --junitxml=runs/taco_brush_20231006_145_issue14_initialization_v1/pytest.xml`

Result: `189 passed in 13.33s`.
