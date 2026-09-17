"""Evaluation-only audit of frozen automatic meshes used by dev29."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import trimesh

from ..artifacts import artifact_record
from .camera_motion_depth import EPISODES
from .metrics import mesh_metrics


def audit_frozen_meshes(routes_root: str | Path, gt_root: str | Path, output: str | Path) -> Path:
    routes = Path(routes_root).resolve()
    gt_root = Path(gt_root).resolve()
    destination = Path(output).resolve()
    episodes: dict[str, Any] = {}
    for episode_id in EPISODES:
        gt_manifest_path = gt_root / episode_id / "ground_truth_manifest.json"
        manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("uses_ground_truth") is not True or manifest.get("scope") != "evaluation_only":
            raise ValueError(f"mesh GT manifest is not evaluation-only: {gt_manifest_path}")
        gt_mesh_path = Path(manifest["artifacts"]["mesh"]["path"]).resolve()
        gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)
        run = routes / episode_id / "v2_camera_motion_calibrated_depth"
        selected_path = run / "object_tracking/selected_mesh.json"
        if not selected_path.is_file():
            # Mesh is expected to be shared across routes; use the frozen ranking
            # even when pose execution is incomplete.
            ranking_path = run / "mesh_proposals/mesh_ranking.json"
            ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
            proposal = next(item for item in ranking["proposals"] if item.get("rank") == 1)
            mesh_path = run / "mesh_proposals" / proposal["visual_mesh"]
            scale = float(proposal["selected_scale_m"])
            proposal_id = proposal["proposal_id"]
        else:
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            mesh_path = run / selected["canonical_visual_mesh"]
            scale = float(selected["scale_to_m"])
            proposal_id = selected["proposal_id"]
        auto_mesh = trimesh.load_mesh(mesh_path, process=False)
        auto_mesh.apply_scale(scale)
        metrics = mesh_metrics(auto_mesh, gt_mesh, sample_count=5000)
        episodes[episode_id] = {
            "proposal_id": proposal_id,
            "metrics": metrics,
            "artifacts": {
                "automatic_mesh": artifact_record(mesh_path),
                "ground_truth_mesh": artifact_record(gt_mesh_path),
                "ground_truth_manifest": artifact_record(gt_manifest_path),
            },
        }
    scale_errors = [item["metrics"]["scale_relative_error"] for item in episodes.values()]
    payload = {
        "schema_version": "1.0",
        "profile": "foundationpose_auto_mesh_evaluation_only_dev29",
        "scope": "evaluation_only",
        "gt_enters_inference": False,
        "episodes": episodes,
        "acceptance": {
            "episode_count": len(episodes),
            "all_meshes_available": len(episodes) == len(EPISODES),
            "mean_scale_relative_error": sum(scale_errors) / max(len(scale_errors), 1),
            "worst_scale_relative_error": max(scale_errors) if scale_errors else None,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def audit_mesh_rankings(mesh_root: str | Path, gt_root: str | Path, output: str | Path) -> Path:
    """Audit rank-1 meshes directly from isolated SAM3D workspaces."""
    mesh_root = Path(mesh_root).resolve()
    gt_root = Path(gt_root).resolve()
    destination = Path(output).resolve()
    episodes: dict[str, Any] = {}
    for episode_id in EPISODES:
        ranking_path = mesh_root / episode_id / "mesh_proposals/mesh_ranking.json"
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        proposal = next(item for item in ranking["proposals"] if item.get("rank") == 1)
        mesh_path = ranking_path.parent / proposal["visual_mesh"]
        auto_mesh = trimesh.load_mesh(mesh_path, process=False)
        auto_mesh.apply_scale(float(proposal["selected_scale_m"]))
        gt_manifest_path = gt_root / episode_id / "ground_truth_manifest.json"
        manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("uses_ground_truth") is not True or manifest.get("scope") != "evaluation_only":
            raise ValueError(f"mesh GT manifest is not evaluation-only: {gt_manifest_path}")
        gt_mesh_path = Path(manifest["artifacts"]["mesh"]["path"]).resolve()
        gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)
        episodes[episode_id] = {
            "proposal_id": proposal["proposal_id"],
            "selected_scale_m": float(proposal["selected_scale_m"]),
            "static_fit": proposal["fit"],
            "metrics": mesh_metrics(auto_mesh, gt_mesh, sample_count=5000),
            "artifacts": {
                "mesh_ranking": artifact_record(ranking_path),
                "automatic_mesh": artifact_record(mesh_path),
                "ground_truth_mesh": artifact_record(gt_mesh_path),
            },
        }
    scale_errors = [item["metrics"]["scale_relative_error"] for item in episodes.values()]
    payload = {
        "schema_version": "1.0",
        "profile": "sam3d_mesh_ranking_evaluation_only",
        "scope": "evaluation_only", "gt_enters_inference": False,
        "episodes": episodes,
        "acceptance": {
            "episode_count": len(episodes),
            "mean_scale_relative_error": sum(scale_errors) / max(len(scale_errors), 1),
            "worst_scale_relative_error": max(scale_errors) if scale_errors else None,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
