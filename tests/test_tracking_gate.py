import json
from pathlib import Path

import numpy as np
import pytest

from video_to_spider.adapters.tracking_gate import (
    evaluate_tracking_gate,
    require_tracking_gate,
    write_tracking_gate,
)


def _make_run(root: Path, **metric_overrides: float) -> None:
    output = root / "object_tracking"
    output.mkdir(parents=True)
    metrics = {
        "valid_rate": 0.95,
        "mean_mask_iou": 0.55,
        "median_relative_depth_residual": 0.05,
        "translation_jump_p95_m": 0.04,
        "rotation_jump_p95_rad": 0.4,
        "registration_count": 3,
        "tracking_score": 0.7,
    }
    metrics.update(metric_overrides)
    (output / "tracking_metrics.json").write_text(
        json.dumps({"schema_version": "1.0", "metrics": metrics})
    )
    np.savez_compressed(
        output / "foundationpose_raw.npz", T_camera_object=np.eye(4)[None]
    )
    (output / "selected_mesh.json").write_text(
        json.dumps({"mesh_ranking": "mesh_proposals/mesh_ranking_metric_refit.json"})
    )


def test_tracking_gate_accepts_continuous_full_trajectory(tmp_path: Path):
    _make_run(tmp_path)
    payload = evaluate_tracking_gate(tmp_path)
    assert payload["accepted"]
    assert all(payload["checks"].values())
    assert payload["per_video_tuning"] is False


def test_tracking_gate_rejects_implausible_rotation_jump(tmp_path: Path):
    _make_run(tmp_path, rotation_jump_p95_rad=2.2)
    path = write_tracking_gate(tmp_path)
    payload = json.loads(path.read_text())
    assert not payload["accepted"]
    assert not payload["checks"]["rotation_jump_p95_rad"]
    with pytest.raises(RuntimeError, match="tracking_gate_rejected: rotation_jump_p95_rad"):
        require_tracking_gate(tmp_path)


def test_tracking_gate_detects_stale_trajectory(tmp_path: Path):
    _make_run(tmp_path)
    write_tracking_gate(tmp_path)
    with (tmp_path / "object_tracking/foundationpose_raw.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="tracking_gate_stale: trajectory"):
        require_tracking_gate(tmp_path)


def test_new_metric_pipeline_cannot_skip_tracking_gate(tmp_path: Path):
    _make_run(tmp_path)
    with pytest.raises(RuntimeError, match="tracking_gate_missing"):
        require_tracking_gate(tmp_path)


def test_legacy_pipeline_can_read_artifact_without_gate(tmp_path: Path):
    output = tmp_path / "object_tracking"
    output.mkdir(parents=True)
    (output / "selected_mesh.json").write_text(
        json.dumps({"mesh_ranking": "mesh_proposals/mesh_ranking.json"})
    )
    assert require_tracking_gate(tmp_path) is None
