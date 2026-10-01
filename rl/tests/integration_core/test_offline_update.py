import json
import os
from pathlib import Path

import pytest
import torch

from video_to_spider.rl.core.runner import _offline_actor_parity


def test_three_saved_batches_reproduce_full_actor_exactly():
    # The immutable oracle was recorded with four Torch/OMP worker threads.
    # Pin the in-process Torch pool so an explicit integration invocation does
    # not silently compare a different reduction schedule.
    torch.set_num_threads(4)
    project_root = Path(__file__).resolve().parents[2]
    asset_root = Path(os.environ.get("RL_CORE_ASSET_ROOT", project_root))
    root = asset_root / "runs/taco_pour_candidate_G_value_loss_isolation_v1"
    missing = []
    for seed in range(3):
        report_path = root / f"seed_{seed}" / "seed_report.json"
        if not report_path.is_file():
            missing.append(str(report_path))
            continue
        report = json.loads(report_path.read_text())
        paths = (
            report["batch"]["path"],
            report["pre_update_state"]["path"],
            report["shadow_state_artifacts"]["FULL"]["path"],
        )
        missing.extend(path for path in paths if not Path(path).is_file())
    if missing:
        pytest.skip(
            "explicit RL-core integration fixtures are not mounted; "
            "set RL_CORE_ASSET_ROOT to the immutable asset root. Missing: "
            + ", ".join(missing[:3])
        )
    rows, passed = _offline_actor_parity(root)
    assert passed
    assert len(rows) == 3
    assert all(row["actor_optimizer_steps"] == 2 for row in rows)
