"""Hard quality gate between object tracking and robot-motion generation.

The limits are deliberately video-independent and intentionally generous: a
90-degree or 10-centimetre frame-to-frame P95 jump is already too discontinuous
to be a trustworthy object reference at normal video frame rates.  A rejected
trajectory remains on disk for diagnosis, but must not reach sequence
optimization, MINK, Replay, or MPC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_LIMITS = {
    "min_valid_rate": 0.80,
    "min_mean_mask_iou": 0.30,
    "max_median_relative_depth_residual": 0.20,
    "max_translation_jump_p95_m": 0.10,
    "max_rotation_jump_p95_rad": math.pi / 2.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_tracking_gate(
    run_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    metrics_file = Path(
        metrics_path or root / "object_tracking/tracking_metrics.json"
    ).resolve()
    trajectory_file = root / "object_tracking/foundationpose_raw.npz"
    selected_mesh_file = root / "object_tracking/selected_mesh.json"
    for required in (metrics_file, trajectory_file, selected_mesh_file):
        if not required.exists():
            raise FileNotFoundError(required)

    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    thresholds = {**DEFAULT_LIMITS, **(limits or {})}
    checks = {
        "valid_rate": bool(metrics["valid_rate"] >= thresholds["min_valid_rate"]),
        "mean_mask_iou": bool(
            metrics["mean_mask_iou"] >= thresholds["min_mean_mask_iou"]
        ),
        "median_relative_depth_residual": bool(
            metrics["median_relative_depth_residual"]
            <= thresholds["max_median_relative_depth_residual"]
        ),
        "translation_jump_p95_m": bool(
            metrics["translation_jump_p95_m"]
            <= thresholds["max_translation_jump_p95_m"]
        ),
        "rotation_jump_p95_rad": bool(
            metrics["rotation_jump_p95_rad"]
            <= thresholds["max_rotation_jump_p95_rad"]
        ),
    }
    return {
        "schema_version": "1.0",
        "policy": (
            "fixed full-trajectory geometric gate; rejected trajectories must not reach "
            "sequence optimization, MINK, Replay, or MPC"
        ),
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "limits": thresholds,
        "metrics": metrics,
        "inputs": {
            "tracking_metrics": str(metrics_file),
            "tracking_metrics_sha256": _sha256(metrics_file),
            "trajectory": str(trajectory_file),
            "trajectory_sha256": _sha256(trajectory_file),
            "selected_mesh": str(selected_mesh_file),
            "selected_mesh_sha256": _sha256(selected_mesh_file),
        },
        "ground_truth_used": False,
        "per_video_tuning": False,
    }


def write_tracking_gate(
    run_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    limits: dict[str, float] | None = None,
) -> Path:
    root = Path(run_dir).resolve()
    destination = Path(
        output_path or root / "object_tracking/tracking_gate.json"
    ).resolve()
    payload = evaluate_tracking_gate(root, metrics_path=metrics_path, limits=limits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _is_new_metric_pipeline(root: Path) -> bool:
    metadata_path = root / "depth/metadata.json"
    if metadata_path.exists():
        model = str(
            json.loads(metadata_path.read_text(encoding="utf-8")).get("model", "")
        ).lower()
        if (
            "da3" in model
            or "depth anything 3" in model
            or "foundationstereo" in model
        ):
            return True
    selected_path = root / "object_tracking/selected_mesh.json"
    if selected_path.exists():
        ranking = str(
            json.loads(selected_path.read_text(encoding="utf-8")).get("mesh_ranking", "")
        )
        return "metric_refit" in ranking
    return False


def require_tracking_gate(run_dir: str | Path) -> dict[str, Any] | None:
    """Require a fresh accepted gate for the new metric pipeline.

    Historical DAv2 artifacts remain readable so old experiments and fixtures
    are not retroactively invalidated.
    """
    root = Path(run_dir).resolve()
    gate_path = root / "object_tracking/tracking_gate.json"
    if not gate_path.exists():
        if _is_new_metric_pipeline(root):
            raise RuntimeError(
                "tracking_gate_missing: the metric pipeline requires full-trajectory QC"
            )
        return None
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    recorded = payload.get("inputs", {})
    for role in ("tracking_metrics", "trajectory", "selected_mesh"):
        artifact_path = Path(recorded.get(role, ""))
        expected_hash = recorded.get(f"{role}_sha256")
        if (
            not artifact_path.is_file()
            or not expected_hash
            or _sha256(artifact_path) != expected_hash
        ):
            raise RuntimeError(f"tracking_gate_stale: {role} changed after gating")
    if not payload.get("accepted", False):
        failed = [name for name, passed in payload.get("checks", {}).items() if not passed]
        raise RuntimeError(f"tracking_gate_rejected: {', '.join(failed)}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = write_tracking_gate(
        args.run_dir, metrics_path=args.metrics_path, output_path=args.output_path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
