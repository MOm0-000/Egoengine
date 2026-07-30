"""Artifact-only unified run report; unavailable metrics are explicit, never zero-filled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import SCHEMA_VERSION


def _json_or_missing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return {"status": "available", "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "path": str(path), "error": str(error)}


def _npz_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with np.load(path, allow_pickle=False) as artifact:
            arrays = {
                key: {"shape": list(artifact[key].shape), "dtype": str(artifact[key].dtype)}
                for key in artifact.files
            }
        return {"status": "available", "path": str(path), "arrays": arrays}
    except Exception as error:
        return {"status": "invalid", "path": str(path), "error": str(error)}


def build_run_report(run_dir: str | Path, *, spider_report: str | Path | None = None) -> Path:
    root = Path(run_dir).resolve()
    stages = {
        "manifest": _json_or_missing(root / "manifest.json"),
        "segmentation": _json_or_missing(root / "segmentation/metadata.json"),
        "wilor": _json_or_missing(root / "hands/metadata.json"),
        "wilor_gt_evaluation": _json_or_missing(root / "evaluation/wilor_hand_metrics.json"),
        "metric_depth": _json_or_missing(root / "depth/metadata.json"),
        "mesh_proposals": _json_or_missing(root / "mesh_proposals/mesh_ranking.json"),
        "selected_mesh": _json_or_missing(root / "object_tracking/selected_mesh.json"),
        "object_tracking": _json_or_missing(root / "object_tracking/tracking_metrics.json"),
        "optimization": _json_or_missing(root / "optimization/optimization_metrics.json"),
        "aligned_trajectory": _npz_summary(root / "optimization/aligned_trajectory.npz"),
        "contact": _npz_summary(root / "optimization/contact.npz"),
    }
    if spider_report is None:
        candidates = sorted((root / "spider_export").glob("**/spider_run_report.json"))
        spider_path = candidates[-1] if candidates else root / "spider_export/spider_run_report.json"
    else:
        spider_path = Path(spider_report).resolve()
    stages["spider"] = _json_or_missing(spider_path)
    missing = [name for name, record in stages.items() if record["status"] == "missing"]
    invalid = [name for name, record in stages.items() if record["status"] == "invalid"]
    spider_data = stages["spider"].get("data", {})
    spider_artifacts = spider_data.get("artifacts", {})
    trajectory_mjwp = spider_artifacts.get("trajectory_mjwp")
    mjwp_video = spider_artifacts.get("mjwp_video")
    trajectory_complete = bool(
        trajectory_mjwp and Path(trajectory_mjwp).is_file()
        and Path(trajectory_mjwp).stat().st_size > 0
    )
    simulation_video_complete = bool(
        mjwp_video and Path(mjwp_video).is_file() and Path(mjwp_video).stat().st_size > 0
    )
    spider_complete = bool(
        stages["spider"]["status"] == "available"
        and spider_data.get("mjwp_metrics") is not None
        and trajectory_complete and simulation_video_complete
        and all(command.get("returncode") == 0 for command in spider_data.get("commands", []))
    )
    report = {
        "schema_version": SCHEMA_VERSION, "run_dir": str(root), "stages": stages,
        "diagnostics": {
            "visualization": _json_or_missing(root / "visualization/visualization_manifest.json"),
        },
        "completion": {
            "available_stage_count": sum(record["status"] == "available" for record in stages.values()),
            "total_stage_count": len(stages), "missing_stages": missing, "invalid_stages": invalid,
            "spider_chain_complete": spider_complete,
            "simulation_video_complete": simulation_video_complete,
            "m4_complete": spider_complete and all(
                stages[name]["status"] == "available"
                for name in ("mesh_proposals", "selected_mesh", "object_tracking", "optimization",
                             "aligned_trajectory", "contact")
            ),
        },
        "ground_truth_policy": {
            "wilor_gt_evaluation_is_diagnostic_only": True,
            "ground_truth_consumed_by_inference": False,
        },
    }
    output = root / "evaluation/unified_run_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spider-report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(build_run_report(args.run_dir, spider_report=args.spider_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
