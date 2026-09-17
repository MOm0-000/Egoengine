"""Controlled TACO depth-route comparison with camera-motion calibration."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from video_to_spider.schemas import SCHEMA_VERSION

from ..artifacts import artifact_record


PROFILE_ORDER = (
    "auto_current_raw_depth",
    "auto_segmentation_v2_raw_depth",
    "auto_segmentation_v2_camera_motion_calibrated_depth",
    "auto_segmentation_v2_camera_mano_fused_depth",
    "auto_segmentation_v2_oracle_depth",
)


@dataclass(frozen=True)
class TacoDepthCalibrationSpec:
    path: Path
    data: dict[str, Any]


def bundled_taco_depth_calibration_path() -> Path:
    return Path(str(files("egoengine_repro").joinpath(
        "configs", "taco_depth_calibration_dev4.yaml",
    )))


def load_taco_depth_calibration_spec(path: str | Path | None = None) -> TacoDepthCalibrationSpec:
    candidate = Path(path) if path is not None else bundled_taco_depth_calibration_path()
    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO depth calibration spec must use schema version 1.0")
    if tuple(data.get("profiles", ())) != PROFILE_ORDER:
        raise ValueError("TACO depth calibration profile order changed")
    policy = data.get("inference_policy", {})
    if policy != {
        "object_gt_used": False, "hand_gt_used": False,
        "manual_point_used": False, "oracle_depth_used": False,
        "mano_fusion": "blocked_until_independent_qc",
    }:
        raise ValueError("depth calibration policy permits forbidden inference inputs")
    return TacoDepthCalibrationSpec(candidate.resolve(), data)


def _episode_runs(segmentation_manifest: Path) -> dict[str, tuple[Path, Path]]:
    payload = json.loads(segmentation_manifest.read_text(encoding="utf-8"))
    source_manifest = Path(payload["source_ablation_manifest"]["path"]).resolve()
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    raw_runs = {}
    for episode in source_payload["episodes"]:
        profile = next(item for item in episode["profiles"] if item["profile"] == "auto_current")
        raw_runs[str(episode["episode_id"])] = Path(profile["run_dir"]).resolve()
    result: dict[str, tuple[Path, Path]] = {}
    for episode in payload["episodes"]:
        profile = episode["profiles"][0]
        episode_id = str(episode["episode_id"])
        result[episode_id] = Path(profile["run_dir"]).resolve(), raw_runs[episode_id]
    return result


def prepare_taco_depth_calibration(
    spec: TacoDepthCalibrationSpec, segmentation_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    segmentation_manifest = Path(segmentation_manifest).resolve()
    source_runs = _episode_runs(segmentation_manifest)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    runtime = spec.data["runtime"]
    for episode_id, (source_run, raw_depth_run) in sorted(source_runs.items()):
        root = output / "episodes" / episode_id
        camera_run = source_run
        calibrated_dir = camera_run / "depth_camera_motion_calibrated"
        stages = [
            {
                "name": "auto_current_raw_depth",
                "profile": "auto_current_raw_depth",
                "source_run": str(raw_depth_run), "depth_path": str(raw_depth_run / "depth/metric_depth.zarr"),
                "status": "available" if (raw_depth_run / "depth/metric_depth.zarr").exists() else "missing",
                "uses_ground_truth_in_inference": False,
            },
            {
                "name": "auto_segmentation_v2_raw_depth",
                "profile": "auto_segmentation_v2_raw_depth",
                "source_run": str(raw_depth_run), "mask_run": str(source_run), "depth_path": str(raw_depth_run / "depth/metric_depth.zarr"),
                "status": "available" if (raw_depth_run / "depth/metric_depth.zarr").exists() else "missing",
                "uses_ground_truth_in_inference": False,
            },
            {
                "name": "camera_motion_calibrated_depth",
                "profile": "auto_segmentation_v2_camera_motion_calibrated_depth",
                "source_run": str(source_run), "depth_path": str(calibrated_dir / "metric_depth.zarr"),
                "output_dir": str(calibrated_dir),
                "status": "pending",
                "command": [
                    "conda", "run", "--no-capture-output", "-n", "v2s-core", "python",
                    "-m", "video_to_spider.adapters.camera_motion_depth", "--run-dir", str(source_run),
                    "--raw-depth-run-dir", str(raw_depth_run),
                    "--output-dir", str(calibrated_dir), "--segment-length", str(runtime["segment_length_frames"]),
                    "--max-corners", str(runtime["max_corners"]), "--overwrite",
                ],
                "uses_ground_truth_in_inference": False,
            },
            {
                "name": "camera_mano_fused_depth",
                "profile": "auto_segmentation_v2_camera_mano_fused_depth",
                "source_run": str(source_run), "depth_path": None,
                "status": "blocked_not_validated",
                "blocked_reason": "MANO reprojection/valid-rate/temporal QC is not independently validated",
                "uses_ground_truth_in_inference": False,
            },
            {
                "name": "oracle_depth_diagnostic_upper_bound",
                "profile": "auto_segmentation_v2_oracle_depth",
                "source_run": str(source_run), "depth_path": None,
                "status": "diagnostic_only",
                "blocked_reason": "rendered_gt_proxy depth is not allowed in final automatic inference",
                "uses_ground_truth_in_inference": True,
            },
        ]
        plan = {
            "schema_version": SCHEMA_VERSION, "profile_order": list(PROFILE_ORDER),
            "episode_id": episode_id, "source_segmentation_manifest": str(segmentation_manifest),
            "uses_ground_truth_in_inference": False,
            "inference_policy": spec.data["inference_policy"], "stages": stages,
        }
        plan_path = root / "execution_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        episodes.append({"episode_id": episode_id, "execution_plan": artifact_record(plan_path)})
    if len(episodes) != int(spec.data["acceptance"]["expected_episode_count"]):
        raise ValueError("depth calibration requires the four frozen TACO episodes")
    manifest_path = output / "taco_depth_calibration_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "name": spec.data["name"],
        "spec": artifact_record(spec.path), "source_segmentation_manifest": artifact_record(segmentation_manifest),
        "profile_order": list(PROFILE_ORDER), "episodes": episodes,
        "frozen_segmentation_profile": "auto_segmentation_v2",
        "uses_ground_truth_in_inference": False,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def execute_taco_depth_calibration(manifest_path: str | Path, *, gpu: int | None = None) -> Path:
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for episode in manifest["episodes"]:
        plan_path = Path(episode["execution_plan"]["path"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for stage in plan["stages"]:
            if stage["name"] != "camera_motion_calibrated_depth" or stage["status"] == "available":
                continue
            if Path(stage["output_dir"]) .joinpath("metadata.json").is_file():
                stage["status"] = "completed"
                stage["return_code"] = 0
                plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                continue
            result = subprocess.run(stage["command"], check=False, capture_output=True, text=True)
            stage["return_code"] = int(result.returncode)
            stage["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                stage["stderr_tail"] = result.stderr[-4000:]
            else:
                stage["stdout_tail"] = result.stdout[-2000:]
            plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    result_path = manifest_file.parent / "taco_depth_calibration_results.json"
    by_episode = {}
    for episode in manifest["episodes"]:
        plan = json.loads(Path(episode["execution_plan"]["path"]).read_text(encoding="utf-8"))
        camera = next(item for item in plan["stages"] if item["name"] == "camera_motion_calibrated_depth")
        metadata = Path(camera["output_dir"]) / "metadata.json"
        payload = {"status": camera["status"], "metadata": artifact_record(metadata) if metadata.is_file() else None}
        if metadata.is_file():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            global_calibration = data.get("global_calibration", {})
            calibrated = [item for item in data.get("segments", []) if item.get("status") == "observed"]
            payload.update({
                "global_calibration_status": global_calibration.get("status", "unknown"),
                "strict_multiview_calibrated": bool(global_calibration.get("strict_multiview_calibrated", False)),
                "validated_pairwise_calibrated": bool(global_calibration.get("validated_pairwise_calibrated", False)),
                "calibrated_segment_count": len(calibrated),
                "median_residual_m": float(global_calibration.get("median_residual_m", -1.0)),
                "p95_residual_m": float(global_calibration.get("p95_residual_m", -1.0)),
                "scale": global_calibration.get("scale"),
                "shift_m": global_calibration.get("shift_m"),
                "scale_only": global_calibration.get("scale_only"),
                "scale_only_median_residual_m": global_calibration.get("scale_only_median_residual_m"),
                "scale_only_p95_residual_m": global_calibration.get("scale_only_p95_residual_m"),
                "segment_shrinkage": {
                    "segment_count": len(global_calibration.get("segment_shrinkage", {}).get("segments", [])),
                    "median_residual_m": global_calibration.get("segment_shrinkage", {}).get("median_residual_m"),
                    "p95_residual_m": global_calibration.get("segment_shrinkage", {}).get("p95_residual_m"),
                },
                "plane_constrained_affine": global_calibration.get("plane_constrained_affine"),
                "model_selection": global_calibration.get("model_selection"),
                "regularized_segment_offsets": global_calibration.get("regularized_segment_offsets"),
                "failure_reasons": global_calibration.get("failure_reasons", []),
                "plane_constraint": global_calibration.get("plane_constraint"),
                "observability": {
                    key: data.get("observability", {}).get(key)
                    for key in (
                        "baseline_count", "baseline_min_m", "baseline_median_m",
                        "baseline_p95_m", "baseline_max_m", "window_count",
                        "metric_point_count", "plane_inlier_ratio",
                    )
                },
                "observability_report": data.get("observability_report"),
                "automatic_input_policy": data.get("automatic_input_policy"),
            })
        by_episode[episode["episode_id"]] = payload
    result_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "profile_order": list(PROFILE_ORDER),
        "frozen_segmentation_profile": "auto_segmentation_v2",
        "by_episode": by_episode,
        "acceptance": {
            "four_episode_plans": len(by_episode) == 4,
            "camera_motion_artifact_available_all_episodes": all(item["status"] == "completed" for item in by_episode.values()),
            "translation_gate_evaluated": False,
            "translation_mean_m_below_0.10": False,
            "translation_p95_m_below_0.50": False,
            "camera_motion_global_calibration_statuses": sorted({item.get("global_calibration_status", "unknown") for item in by_episode.values()}),
            "mano_fusion_validated": False,
            "oracle_depth_final_automatic_allowed": False,
        },
    }, indent=2) + "\n", encoding="utf-8")
    return result_path
