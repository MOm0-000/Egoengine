# Validation commands

Repository revision used for the formal candidate: `bb0bd2d`.

```bash
/data_all/zzx/egoengine/spider/.venv/bin/python \
  rl/scripts/run_taco_brush_issue14_mink_candidate_v1.py \
  --config rl/configs/taco_brush_issue14_mink_candidate_v1.yaml

cd rl
/data_all/zzx/egoengine/spider/.venv/bin/python -m pytest -q \
  --junitxml=runs/taco_brush_issue14_mink_candidate_v1/pytest.xml
```

The formal run did not execute physics, Replay, MPC, reinforcement learning,
promotion, or chunk commit.
