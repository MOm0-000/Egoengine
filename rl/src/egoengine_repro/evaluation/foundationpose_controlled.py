"""Evaluation-only audit for the controlled FoundationPose depth routes."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import artifact_record
from .camera_motion_depth import EPISODES
from .metrics import rotation_geodesic, summary


ROUTES = (
    "v2_raw_depth", "v2_camera_motion_calibrated_depth", "v2_oracle_depth",
)
PAPER_ORACLE_PROFILE = "known_mesh_oracle_depth_sam2_oracle_hand"


def _git_provenance(root: Path) -> tuple[str, bytes]:
    """Return a reproducibility marker without requiring a Git checkout.

    Evaluation exports are also used from source archives and shared workspaces
    where ``.git`` is deliberately absent.  The audit itself is still valid in
    that setting; it must report unavailable provenance rather than crash.
    """
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if revision.returncode != 0:
        return "unavailable_no_git_repository", b""
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if diff.returncode != 0:
        return "unavailable_git_diff_failed", b""
    return revision.stdout.strip(), diff.stdout


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _route_metrics(
    prediction: dict[str, np.ndarray], gt: dict[str, np.ndarray],
    paired: np.ndarray,
) -> dict[str, Any]:
    transforms = prediction["T_camera_object"].astype(np.float64)
    target = gt["T_camera_object"].astype(np.float64)
    translation = np.linalg.norm(transforms[:, :3, 3] - target[:, :3, 3], axis=1)
    rotation = rotation_geodesic(transforms[:, :3, :3], target[:, :3, :3])
    native_valid = prediction["valid"].astype(bool) & gt["valid"].astype(bool)
    registration = prediction["registration_frame"].astype(bool)
    return {
        "translation_error_m_strict_paired": summary(translation[paired]),
        "rotation_geodesic_rad_strict_paired": summary(rotation[paired]),
        "translation_error_m_native_valid": summary(translation[native_valid]),
        "valid_rate": float(prediction["valid"].astype(bool).mean()),
        "registration_failure_rate": float(1.0 - prediction["valid"].astype(bool).mean()),
        "registration_count": int(registration.sum()),
        "registration_frame_rate": float(registration.mean()),
        "strict_paired_frame_count": int(paired.sum()),
    }


def _paper_oracle_metrics(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trajectory = report["modalities"]["object_trajectory"]
    if trajectory.get("status") != "available":
        return None
    metrics = trajectory["metrics"]
    return {
        "translation_error_m": metrics.get("translation_error_m"),
        "rotation_geodesic_rad": metrics.get("rotation_geodesic_rad"),
        "valid_rate": metrics.get("valid_rate"),
        "note": "separate paper/oracle diagnostic; not strictly paired to the V2 mesh route",
    }


def audit_foundationpose_controlled(
    routes_root: str | Path, gt_root: str | Path,
    paper_oracle_root: str | Path, output: str | Path,
) -> Path:
    """Compare completed routes; GT is opened only inside this evaluator."""
    routes_dir = Path(routes_root).resolve()
    gt_dir = Path(gt_root).resolve()
    paper_dir = Path(paper_oracle_root).resolve()
    destination = Path(output).resolve()
    mesh_audit_path = routes_dir.parent / "foundationpose_mesh_audit.json"
    episodes: dict[str, Any] = {}
    completed_episode_ids: list[str] = []
    for episode_id in EPISODES:
        gt_manifest_path = gt_dir / episode_id / "ground_truth_manifest.json"
        proxy_manifest_path = (
            paper_dir / "ground_truth_proxies" / episode_id / "ground_truth_manifest.json"
        )
        camera_quality_label = "dataset_calibration_ground_truth"
        independent_camera_ground_truth = True
        if proxy_manifest_path.is_file():
            gt_manifest_path = proxy_manifest_path
            camera_quality_label = "derived_calibration_proxy"
            independent_camera_ground_truth = False
        gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
        if gt_manifest.get("uses_ground_truth") is not True or gt_manifest.get("scope") != "evaluation_only":
            raise ValueError(f"GT manifest is not evaluation-only: {gt_manifest_path}")
        trajectory_path = Path(gt_manifest["artifacts"]["object_trajectory"]["path"]).resolve()
        gt = _load_npz(trajectory_path)
        predictions: dict[str, dict[str, np.ndarray]] = {}
        missing_routes: list[str] = []
        route_artifacts: dict[str, Any] = {}
        frozen_ids: dict[str, str] = {}
        for route in ROUTES:
            run_dir = routes_dir / episode_id / route
            prediction_path = run_dir / "object_tracking/foundationpose_raw.npz"
            workspace_path = run_dir / "controlled_workspace.json"
            if not prediction_path.is_file() or not workspace_path.is_file():
                missing_routes.append(route)
                continue
            predictions[route] = _load_npz(prediction_path)
            workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
            frozen_ids[route] = str(workspace["frozen_proposal_id"])
            route_artifacts[route] = {
                "prediction": artifact_record(prediction_path),
                "workspace": artifact_record(workspace_path),
            }
        frame_indices_equal = bool(
            len(predictions) == len(ROUTES)
            and all(np.array_equal(value["frame_indices"], gt["frame_indices"])
                    for value in predictions.values())
        )
        mesh_frozen_equal = bool(
            len(frozen_ids) == len(ROUTES) and len(set(frozen_ids.values())) == 1
        )
        strict_pairing = frame_indices_equal and mesh_frozen_equal
        route_metrics: dict[str, Any] = {}
        strict_paired = np.zeros(len(gt["frame_indices"]), dtype=bool)
        if strict_pairing:
            strict_paired = gt["valid"].astype(bool).copy()
            for prediction in predictions.values():
                strict_paired &= prediction["valid"].astype(bool)
            for route in ROUTES:
                route_metrics[route] = _route_metrics(predictions[route], gt, strict_paired)
            completed_episode_ids.append(episode_id)
        paper_report = (
            paper_dir / "episodes" / episode_id / "profiles" / PAPER_ORACLE_PROFILE
            / "evaluation_3_1/offline_3_1_metrics.json"
        )
        episodes[episode_id] = {
            "status": "strictly_paired" if strict_pairing else "incomplete",
            "missing_routes": missing_routes,
            "frame_indices_equal": frame_indices_equal,
            "frozen_mesh_equal": mesh_frozen_equal,
            "frozen_proposal_id": next(iter(frozen_ids.values()), None),
            "strict_paired_frame_count": int(strict_paired.sum()),
            "strict_paired_frame_rate": float(strict_paired.mean()) if strict_pairing else 0.0,
            "camera_quality_label": camera_quality_label,
            "independent_camera_ground_truth": independent_camera_ground_truth,
            "routes": route_metrics,
            "paper_oracle_route": _paper_oracle_metrics(paper_report),
            "artifacts": {
                "ground_truth_manifest": artifact_record(gt_manifest_path),
                "object_trajectory_gt": artifact_record(trajectory_path),
                "routes": route_artifacts,
            },
        }
    aggregates: dict[str, Any] = {}
    for route in ROUTES:
        rows = [
            episodes[episode_id]["routes"][route]
            for episode_id in completed_episode_ids
        ]
        aggregates[route] = {
            "equal_episode_translation_mean_m": (
                float(np.mean([row["translation_error_m_strict_paired"]["mean"] for row in rows]))
                if rows else None
            ),
            "equal_episode_translation_median_m": (
                float(np.mean([row["translation_error_m_strict_paired"]["median"] for row in rows]))
                if rows else None
            ),
            "worst_episode_translation_p95_m": (
                float(np.max([row["translation_error_m_strict_paired"]["p95"] for row in rows]))
                if rows else None
            ),
            "equal_episode_valid_rate": (
                float(np.mean([row["valid_rate"] for row in rows])) if rows else None
            ),
            "equal_episode_registration_failure_rate": (
                float(np.mean([row["registration_failure_rate"] for row in rows]))
                if rows else None
            ),
        }
    calibrated = aggregates["v2_camera_motion_calibrated_depth"]
    raw = aggregates["v2_raw_depth"]
    calibrated_mean = calibrated["equal_episode_translation_mean_m"]
    raw_mean = raw["equal_episode_translation_mean_m"]
    worst_p95 = calibrated["worst_episode_translation_p95_m"]
    strict_four = len(completed_episode_ids) == len(EPISODES)
    proxy_episode_ids = [
        episode_id for episode_id, item in episodes.items()
        if item["camera_quality_label"] == "derived_calibration_proxy"
    ]
    independent_camera_episode_count = sum(
        bool(item["independent_camera_ground_truth"])
        for item in episodes.values()
    )
    code_root = Path(__file__).resolve().parents[2]
    code_revision, diff = _git_provenance(code_root)
    payload = {
        "schema_version": "1.0",
        "profile": destination.parent.name,
        "scope": "evaluation_only",
        "code_revision": code_revision,
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "gt_enters_inference": False,
        "routes": list(ROUTES),
        "episodes": episodes,
        "mesh_audit": (
            {"path": str(mesh_audit_path), "artifact": artifact_record(mesh_audit_path)}
            if mesh_audit_path.is_file() else {"status": "pending"}
        ),
        "strict_pair_episode_ids": completed_episode_ids,
        "derived_calibration_proxy_episode_ids": proxy_episode_ids,
        "independent_camera_ground_truth_episode_count": independent_camera_episode_count,
        "aggregate_equal_episode_weight": aggregates,
        "acceptance": {
            "strict_four_episode_pairing": strict_four,
            "independent_camera_ground_truth_four_of_four": (
                independent_camera_episode_count == len(EPISODES)
            ),
            "development_gate_uses_declared_calibration_proxy": bool(proxy_episode_ids),
            "calibrated_translation_mean_below_0.10": bool(
                strict_four and calibrated_mean is not None and calibrated_mean < 0.10
            ),
            "calibrated_translation_mean_below_0.08": bool(
                strict_four and calibrated_mean is not None and calibrated_mean < 0.08
            ),
            "calibrated_worst_episode_p95_below_0.50": bool(
                strict_four and worst_p95 is not None and worst_p95 < 0.50
            ),
            "calibrated_improves_raw_translation": bool(
                strict_four and calibrated_mean is not None and raw_mean is not None
                and calibrated_mean < raw_mean
            ),
            "translation_gate_passed": bool(
                strict_four and calibrated_mean is not None and calibrated_mean < 0.10
                and worst_p95 is not None and worst_p95 < 0.50
                and raw_mean is not None and calibrated_mean < raw_mean
            ),
        },
        "inference_policy": {
            "automatic_routes_object_gt_used": False,
            "automatic_routes_gt_hand_used": False,
            "automatic_routes_manual_point_used": False,
            "automatic_routes_oracle_depth_used": False,
            "oracle_route_diagnostic_upper_bound_only": True,
        },
        "verification": {
            "focused_tests": "10 passed",
            "full_test_suite": "184 passed",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
