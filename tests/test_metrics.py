import json

import numpy as np

from video_to_spider.eval.metrics import build_run_report


def test_unified_report_marks_missing_stages(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema_version":"1.0"}')
    report_path = build_run_report(tmp_path)
    report = json.loads(report_path.read_text())
    assert report["stages"]["manifest"]["status"] == "available"
    assert report["stages"]["mesh_proposals"]["status"] == "missing"
    assert report["diagnostics"]["visualization"]["status"] == "missing"
    assert not report["completion"]["m4_complete"]
    assert report["ground_truth_policy"]["ground_truth_consumed_by_inference"] is False


def test_m4_requires_real_mjwp_trajectory_and_simulation_video(tmp_path):
    for relative in (
        "manifest.json", "segmentation/metadata.json", "hands/metadata.json",
        "evaluation/wilor_hand_metrics.json", "depth/metadata.json",
        "mesh_proposals/mesh_ranking.json", "object_tracking/selected_mesh.json",
        "object_tracking/tracking_metrics.json", "optimization/optimization_metrics.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    np.savez(tmp_path / "optimization/aligned_trajectory.npz", value=np.ones(1))
    np.savez(tmp_path / "optimization/contact.npz", value=np.ones(1))
    trajectory = tmp_path / "trajectory_mjwp.npz"
    np.savez(trajectory, value=np.ones(1))
    spider_report = tmp_path / "spider_run_report.json"
    payload = {
        "commands": [{"returncode": 0}], "mjwp_metrics": {"record_count": 1},
        "artifacts": {"trajectory_mjwp": str(trajectory), "mjwp_video": None},
    }
    spider_report.write_text(json.dumps(payload))
    report = json.loads(build_run_report(tmp_path, spider_report=spider_report).read_text())
    assert report["completion"]["simulation_video_complete"] is False
    assert report["completion"]["m4_complete"] is False

    video = tmp_path / "visualization_mjwp.mp4"
    video.write_bytes(b"simulation-video")
    payload["artifacts"]["mjwp_video"] = str(video)
    spider_report.write_text(json.dumps(payload))
    report = json.loads(build_run_report(tmp_path, spider_report=spider_report).read_text())
    assert report["completion"]["simulation_video_complete"] is True
    assert report["completion"]["m4_complete"] is True
