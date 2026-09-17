"""Build isolated FoundationPose workspaces for a depth-only comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..artifacts import artifact_record, ensure_isolated_output


SHARED_LINKS = ("frames", "calibration", "input", "segmentation", "mesh_proposals")
AUTOMATIC_ROUTES = {"v2_raw_depth", "v2_camera_motion_calibrated_depth"}
ORACLE_ROUTE = "v2_oracle_depth"
VALID_ROUTES = AUTOMATIC_ROUTES | {ORACLE_ROUTE}


def _replace_with_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def _select_frozen_proposal(ranking: dict[str, Any]) -> dict[str, Any]:
    qualified = [item for item in ranking["proposals"] if item.get("qualified")]
    if not qualified:
        raise ValueError("auto mesh ranking has no qualified proposal")
    return min(qualified, key=lambda item: int(item.get("rank", 1_000_000)))


def prepare_foundationpose_controlled_workspace(
    auto_mesh_run: str | Path, depth_source: str | Path,
    output_dir: str | Path, *, route: str,
) -> Path:
    """Prepare one route while freezing mask, mesh proposal, and frame range.

    ``depth_source`` must contain ``metric_depth.zarr`` and ``metadata.json``.
    Oracle depth is accepted only under the explicitly diagnostic route.
    """
    if route not in VALID_ROUTES:
        raise ValueError(f"unsupported controlled FoundationPose route: {route}")
    mesh_run = Path(auto_mesh_run).resolve()
    depth = Path(depth_source).resolve()
    output = ensure_isolated_output(output_dir, mesh_run)
    if any(output.iterdir()):
        raise FileExistsError(f"controlled workspace is not empty: {output}")
    depth_zarr = depth / "metric_depth.zarr"
    depth_metadata = depth / "metadata.json"
    if not depth_zarr.is_dir() or not depth_metadata.is_file():
        raise FileNotFoundError(f"depth source is incomplete: {depth}")
    ranking_path = mesh_run / "mesh_proposals/mesh_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    selected = _select_frozen_proposal(ranking)
    proposal_id = str(selected["proposal_id"])
    visual_mesh = mesh_run / "mesh_proposals" / str(selected["visual_mesh"])
    if not visual_mesh.is_file():
        raise FileNotFoundError(visual_mesh)
    for name in SHARED_LINKS:
        source = mesh_run / name
        if not source.exists():
            raise FileNotFoundError(source)
        _replace_with_symlink(output / name, source)
    (output / "depth").mkdir()
    _replace_with_symlink(output / "depth/metric_depth.zarr", depth_zarr)
    _replace_with_symlink(output / "depth/metadata.json", depth_metadata)
    shutil.copy2(mesh_run / "manifest.json", output / "manifest.json")
    diagnostic_oracle = route == ORACLE_ROUTE
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "run_id": f"{manifest.get('run_id', mesh_run.name)}-{route}",
        "repro_profile": route,
        "controlled_foundationpose": {
            "auto_mesh_run": str(mesh_run), "depth_source": str(depth),
            "frozen_proposal_id": proposal_id,
            "same_mesh_across_v2_routes": True,
            "eligible_for_automatic_default": not diagnostic_oracle,
            "diagnostic_upper_bound_only": diagnostic_oracle,
        },
        "ground_truth_isolation": {
            "object_gt_used": diagnostic_oracle,
            "gt_hand_used": False, "manual_point_used": False,
            "oracle_depth_used": diagnostic_oracle,
        },
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    evidence = {
        "schema_version": "1.0",
        "profile": "foundationpose_depth_controlled_workspace",
        "route": route, "workspace": str(output),
        "auto_mesh_run": str(mesh_run), "depth_source": str(depth),
        "frozen_proposal_id": proposal_id,
        "foundationpose_arguments": {
            "max_candidates": 3, "screening_radius": 1,
            "register_iter": 2, "track_iter": 1, "max_input_side": 640,
            "frozen_proposal_id": proposal_id, "skip_visualizations": True,
        },
        "controlled_artifacts": {
            "object_masks": artifact_record(mesh_run / "segmentation/object_masks.npz"),
            "mesh_ranking": artifact_record(ranking_path),
            "frozen_visual_mesh": artifact_record(visual_mesh),
            "depth": artifact_record(depth_zarr),
            "depth_metadata": artifact_record(depth_metadata),
            "intrinsics": artifact_record(mesh_run / "calibration/intrinsics.npy"),
            "frame_index": artifact_record(mesh_run / "frames/frame_index.json"),
        },
        "inference_policy": manifest["ground_truth_isolation"],
        "eligible_for_automatic_default": not diagnostic_oracle,
        "diagnostic_upper_bound_only": diagnostic_oracle,
    }
    evidence_path = output / "controlled_workspace.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence_path

