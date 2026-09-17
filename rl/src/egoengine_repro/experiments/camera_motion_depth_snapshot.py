"""Machine-readable aggregate snapshot for camera-motion depth runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from egoengine_repro.artifacts import artifact_record


EPISODES = (
    "taco_brush_brush_bowl_20230927_027",
    "taco_cut_spatula_plate_20230917_020",
    "taco_skim_spatula_plate_20230926_004",
    "taco_smear_eraser_box_20231103_071",
)


def _working_tree_diff_sha256() -> str:
    diff = subprocess.check_output(["git", "diff", "--binary"], text=False)
    return hashlib.sha256(diff).hexdigest()


def _episode_payload(run_dir: Path, episode_id: str) -> dict[str, object]:
    episode_dir = run_dir / episode_id
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    calibration = metadata.get("global_calibration", {})
    reprojection = metadata.get("reprojection_affine_candidate", {})
    reprojection_scale_only = metadata.get("reprojection_scale_only_candidate", {})
    visual_reprojection = metadata.get("visual_pose_reprojection_affine_candidate", {})
    visual_reprojection_scale_only = metadata.get(
        "visual_pose_reprojection_scale_only_candidate", {}
    )
    visual_global_reprojection = metadata.get(
        "visual_pose_global_reprojection_affine_candidate", {}
    )
    visual_global_scale_only = metadata.get(
        "visual_pose_global_reprojection_scale_only_candidate", {}
    )
    visual_global_observability = metadata.get("visual_pose_global_observability", {})
    visual_selected = metadata.get("visual_pose_selected_candidate", {})
    visual_model_selection = metadata.get("visual_pose_model_selection", {})
    strict_visual_arbitration = metadata.get(
        "strict_visual_temporal_arbitration", {}
    )
    output_applied = metadata.get("output_applied", {})
    temporal_offset_candidate = metadata.get(
        "temporal_offset_shrinkage_candidate", {}
    )
    observability = metadata.get("observability", {})
    def record_optional(path: Path) -> dict[str, object]:
        if not path.exists():
            return {"path": str(path.resolve()), "type": "missing"}
        return artifact_record(path)

    return {
        "status": calibration.get("status", "unknown"),
        "base_calibration_status": calibration.get("base_calibration_status"),
        "strict_multiview_calibrated": bool(calibration.get("strict_multiview_calibrated", False)),
        "validated_pairwise_calibrated": bool(calibration.get("validated_pairwise_calibrated", False)),
        "scale": calibration.get("scale"),
        "shift_m": calibration.get("shift_m"),
        "median_residual_m": calibration.get("median_residual_m"),
        "p95_residual_m": calibration.get("p95_residual_m"),
        "failure_reasons": calibration.get("failure_reasons", []),
        "strict_published_pose_status": calibration.get(
            "strict_published_pose_status", calibration.get("status", "unknown")
        ),
        "visual_pose_recovery_status": calibration.get(
            "visual_pose_recovery_status", "not_recorded"
        ),
        "visual_pose_output_qc": calibration.get("visual_pose_output_qc", {}),
        "reprojection_affine_candidate": {
            key: reprojection.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "median_reprojection_px",
                "p95_reprojection_px", "jacobian_condition_number",
            )
        },
        "reprojection_scale_only_candidate": {
            key: reprojection_scale_only.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "median_reprojection_px",
                "p95_reprojection_px", "jacobian_condition_number",
            )
        },
        "visual_pose_reprojection_affine_candidate": {
            key: visual_reprojection.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "attempted_window_count",
                "pose_observable_window_count",
                "supported_pose_observable_window_count",
                "supported_pose_observable_window_fraction",
                "total_window_coverage_fraction",
                "minimum_total_window_coverage_fraction",
                "median_reprojection_px", "p95_reprojection_px",
                "jacobian_condition_number", "positive_depth_rate",
            )
        },
        "visual_pose_reprojection_scale_only_candidate": {
            key: visual_reprojection_scale_only.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "attempted_window_count",
                "pose_observable_window_count",
                "supported_pose_observable_window_count",
                "supported_pose_observable_window_fraction",
                "total_window_coverage_fraction",
                "minimum_total_window_coverage_fraction",
                "median_reprojection_px", "p95_reprojection_px",
                "jacobian_condition_number", "positive_depth_rate",
            )
        },
        "visual_pose_global_reprojection_affine_candidate": {
            key: visual_global_reprojection.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "attempted_window_count",
                "pose_observable_window_count",
                "supported_pose_observable_window_count",
                "supported_pose_observable_window_fraction",
                "total_window_coverage_fraction",
                "minimum_total_window_coverage_fraction",
                "median_reprojection_px", "p95_reprojection_px",
                "jacobian_condition_number", "positive_depth_rate",
            )
        },
        "visual_pose_global_reprojection_scale_only_candidate": {
            key: visual_global_scale_only.get(key)
            for key in (
                "status", "scale", "shift_m", "observation_count",
                "inlier_count", "inlier_ratio", "selected_window_count",
                "window_count", "supported_window_count",
                "supported_window_fraction", "attempted_window_count",
                "pose_observable_window_count",
                "supported_pose_observable_window_count",
                "supported_pose_observable_window_fraction",
                "total_window_coverage_fraction",
                "minimum_total_window_coverage_fraction",
                "median_reprojection_px", "p95_reprojection_px",
                "jacobian_condition_number", "positive_depth_rate",
            )
        },
        "visual_pose_global_observability": {
            key: visual_global_observability.get(key)
            for key in (
                "attempted_window_count", "selected_target_window_count",
                "pose_observable_window_count", "supported_window_count",
                "total_window_count", "baseline_observable_window_count",
                "baseline_observable_fraction", "window_ids",
            )
            if key in visual_global_observability
        },
        "visual_pose_selected_candidate": {
            key: visual_selected.get(key)
            for key in (
                "status", "selected_model", "scale", "shift_m",
                "observation_count", "inlier_count", "inlier_ratio",
                "selected_window_count", "supported_window_count",
                "total_window_coverage_fraction", "median_reprojection_px",
                "p95_reprojection_px", "positive_depth_rate",
            )
        },
        "visual_pose_model_selection": visual_model_selection,
        "strict_visual_temporal_arbitration": strict_visual_arbitration,
        "output_applied": output_applied,
        "temporal_offset_shrinkage_candidate": temporal_offset_candidate,
        "baseline": {
            key: observability.get(key)
            for key in (
                "baseline_count", "baseline_min_m", "baseline_median_m",
                "baseline_p95_m", "baseline_max_m", "window_count",
                "metric_point_count", "plane_inlier_ratio",
            )
        },
        "artifacts": {
            "metadata": artifact_record(metadata_path),
            "observability_report": artifact_record(episode_dir / "observability_report.json"),
            "raw_metric_scatter": record_optional(episode_dir / "raw_metric_scatter.png"),
            "metric_depth": record_optional(episode_dir / "metric_depth.zarr"),
        },
    }


def _optional_artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path.resolve()), "type": "missing"}
    return artifact_record(path)


def build_snapshot(run_dir: Path, source_manifest: Path, output: Path) -> Path:
    episodes = {
        episode_id: _episode_payload(run_dir, episode_id)
        for episode_id in EPISODES
    }
    strict_count = sum(bool(item["strict_multiview_calibrated"]) for item in episodes.values())
    pairwise_count = sum(bool(item["validated_pairwise_calibrated"]) for item in episodes.values())
    visual_statuses = {
        "calibrated_visual_pose_recovered",
        "calibrated_visual_pose_override",
    }
    visual_count = sum(
        item["status"] in visual_statuses
        or (
            item["status"] == "calibrated_temporal_offset_shrinkage"
            and item["base_calibration_status"] in visual_statuses
        )
        for item in episodes.values()
    )
    automatic_calibrated_count = sum(
        item["strict_multiview_calibrated"]
        or item["status"] in visual_statuses
        or item["status"] == "calibrated_temporal_offset_shrinkage"
        for item in episodes.values()
    )
    payload = {
        "schema_version": "1.0",
        "profile": "camera_motion_calibrated_depth",
        "run_name": run_dir.name,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_diff_sha256": _working_tree_diff_sha256(),
        "frozen_segmentation_profile": "auto_segmentation_v2",
        "source_segmentation_manifest": artifact_record(source_manifest),
        "evaluation_only_object_audit": _optional_artifact(
            run_dir / "camera_motion_depth_object_audit.json"
        ),
        "episodes": episodes,
        "acceptance": {
            "four_episode_artifact_coverage": len(episodes) == 4 and all(
                all(item["type"] != "missing" for item in episode["artifacts"].values())
                for episode in episodes.values()
            ),
            "strict_multiview_calibrated_count": strict_count,
            "strict_multiview_four_of_four": strict_count == 4,
            "validated_pairwise_calibrated_count": pairwise_count,
            "visual_pose_recovered_calibrated_count": visual_count,
            "visual_pose_recovered_four_of_four": visual_count == 4,
            "strict_published_pose_count": strict_count,
            "strict_published_pose_four_of_four": strict_count == 4,
            "visual_pose_recovery_only_count": visual_count,
            "automatic_visual_pose_recovered_count": visual_count,
            "automatic_visual_pose_recovered_four_of_four": visual_count == 4,
            "automatic_calibrated_count": automatic_calibrated_count,
            "automatic_calibrated_four_of_four": automatic_calibrated_count == 4,
            "strict_and_visual_four_of_four": strict_count == 4 and visual_count == 4,
            "all_four_have_depth_evidence": all(
                item["status"] in {"calibrated_multiview", "calibrated_pairwise_validated"}
                for item in episodes.values()
            ),
            "all_four_have_automatic_visual_depth_evidence": all(
                item["status"] in {
                    "calibrated_multiview", "calibrated_pairwise_validated",
                    "calibrated_visual_pose_recovered",
                    "calibrated_visual_pose_override",
                    "calibrated_temporal_offset_shrinkage",
                }
                for item in episodes.values()
            ),
            "translation_gate_evaluated": False,
            "translation_mean_m_below_0.10": False,
            "translation_mean_m_below_0.08": False,
            "translation_p95_m_below_0.50": False,
            "foundationpose_comparison_started": False,
            "rotation_tuning_started": False,
        },
        "inference_policy": {
            "object_gt_used": False,
            "gt_hand_used": False,
            "manual_point_used": False,
            "oracle_depth_used": False,
            "auto_current_modified": False,
            "pairwise_low_parallax_fallback": "validated_diagnostic_route_only",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
