# Validation

- Source geometry, safe-entry MINK, natural seating, and bounded Replay were each generated from their versioned configs in this study.
- `PYTHONPATH="/data_all/zzx/3.2RL/.env_mjwp313_overlay:$PWD/rl/src:$PWD/rl/external/human2sim2robot:$PWD/rl/external/spider_compat" OMP_NUM_THREADS=4 /data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q rl/tests/core --junitxml=rl/runs/taco_brush_20231006_145_initialization_study_v1/pytest.xml`
- Result: `170 passed`.
