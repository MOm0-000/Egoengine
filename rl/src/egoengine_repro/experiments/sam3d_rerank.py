"""Rerank frozen SAM3D proposals against automatic metric-depth inputs."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..artifacts import artifact_record, ensure_isolated_output
from video_to_spider.adapters.sam3d_objects import static_fit_metrics


READ_ONLY_LINKS = ("frames", "calibration", "input", "segmentation", "depth")


def _replace_with_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rerank_frozen_sam3d_proposals(
    source_mesh_run: str | Path, output_dir: str | Path, *,
    metric_depth_source: str | Path | None = None,
) -> Path:
    """Recompute static fit without rerunning SAM3D or reading ground truth."""
    source = Path(source_mesh_run).resolve()
    output = ensure_isolated_output(output_dir, source)
    if any(output.iterdir()):
        raise FileExistsError(f"SAM3D rerank workspace is not empty: {output}")

    ranking_source = source / "mesh_proposals/mesh_ranking.json"
    ranking = json.loads(ranking_source.read_text(encoding="utf-8"))
    depth_source = (
        source / "depth"
        if metric_depth_source is None else Path(metric_depth_source).resolve()
    )
    if not (depth_source / "metric_depth.zarr").is_dir():
        raise FileNotFoundError(depth_source / "metric_depth.zarr")
    for name in READ_ONLY_LINKS:
        source_path = depth_source if name == "depth" else source / name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _replace_with_symlink(output / name, source_path)
    shutil.copy2(source / "manifest.json", output / "manifest.json")

    proposal_output = output / "mesh_proposals"
    proposal_output.mkdir()
    for proposal in ranking["proposals"]:
        proposal_id = str(proposal["proposal_id"])
        source_dir = source / "mesh_proposals" / proposal_id
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        _replace_with_symlink(proposal_output / proposal_id, source_dir)

    with np.load(output / "segmentation/object_masks.npz", allow_pickle=False) as artifact:
        object_frames = np.asarray(artifact["frame_indices"], dtype=np.int64)
        object_masks = np.asarray(artifact["masks"]).astype(bool)
        object_valid = np.asarray(artifact["valid"]).astype(bool)
    import zarr

    depth_group = zarr.open(str(output / "depth/metric_depth.zarr"), mode="r")
    depth_lookup = {
        int(frame): index
        for index, frame in enumerate(np.asarray(depth_group["frame_indices"]))
    }
    K = np.load(output / "calibration/intrinsics.npy").astype(np.float64)
    proposals: list[dict[str, Any]] = []
    for original in ranking["proposals"]:
        proposal = deepcopy(original)
        frame_index = int(proposal["frame_index"])
        mask_index = int(proposal["mask_index"])
        if mask_index >= len(object_frames) or int(object_frames[mask_index]) != frame_index:
            raise ValueError(
                f"proposal {proposal['proposal_id']} mask/frame alignment is invalid"
            )
        if not object_valid[mask_index] or not object_masks[mask_index].any():
            raise ValueError(f"proposal {proposal['proposal_id']} uses an invalid mask")
        if frame_index not in depth_lookup:
            raise ValueError(f"depth is missing proposal frame {frame_index}")
        mesh_path = source / "mesh_proposals" / str(proposal["visual_mesh"])
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"proposal is not a single mesh: {mesh_path}")
        layout = proposal["model_layout"]
        depth_index = depth_lookup[frame_index]
        fit, transform, scale = static_fit_metrics(
            mesh, object_masks[mask_index],
            np.asarray(depth_group["depth_m"][depth_index]),
            np.asarray(depth_group["valid"][depth_index]).astype(bool), K,
            rotation_wxyz=layout["rotation_wxyz"],
            translation=layout["translation"], layout_scale=layout["scale"],
        )
        integrity_score = float(proposal["integrity"]["score"])
        proposal.update({
            "fit": fit, "selected_scale_m": float(scale),
            "T_camera_object_initial": transform.tolist(),
            "static_score": float(
                0.35 * integrity_score
                + 0.40 * float(fit["silhouette_iou"])
                + 0.25 * float(fit["depth_score"])
            ),
            "reranked_from_source": str(ranking_source),
        })
        proposals.append(proposal)

    ordered = sorted(
        proposals,
        key=lambda item: (
            not bool(item.get("qualified")), -float(item.get("static_score", -1.0)),
            str(item["proposal_id"]),
        ),
    )
    for rank, proposal in enumerate(ordered, start=1):
        proposal["rank"] = rank
    reranked = {
        "schema_version": "1.0",
        "stage": "sam3d_objects_metric_depth_rerank",
        "ranking_policy": (
            "0.35 mesh_integrity + 0.40 silhouette_iou + "
            "0.25 exp(-relative_depth_residual)"
        ),
        "foundationpose_used": False,
        "sam3d_generation_rerun": False,
        "ground_truth_used": False,
        "source_mesh_ranking": str(ranking_source),
        "metric_depth_source": str(depth_source),
        "keyframes": ranking.get("keyframes", []),
        "proposals": [_finite_json(item) for item in ordered],
        "qualified_count": sum(bool(item.get("qualified")) for item in ordered),
        "success": any(bool(item.get("qualified")) for item in ordered),
    }
    ranking_output = proposal_output / "mesh_ranking.json"
    ranking_output.write_text(
        json.dumps(reranked, indent=2) + "\n", encoding="utf-8",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    isolation = {
        "object_gt_used": False, "gt_hand_used": False,
        "manual_point_used": False, "oracle_depth_used": False,
    }
    manifest.update({
        "run_id": f"{manifest.get('run_id', source.name)}-metric-depth-rerank",
        "repro_profile": "auto_segmentation_v2_sam3d_metric_depth_rerank",
        "branch_source_run": str(source),
        "metric_depth_source_run": str(depth_source),
        "ground_truth_isolation": isolation,
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    evidence = {
        "schema_version": "1.0",
        "profile": "sam3d_metric_depth_rerank_workspace",
        "workspace": str(output), "source_mesh_run": str(source),
        "metric_depth_source": str(depth_source),
        "metric_depth_overridden": metric_depth_source is not None,
        "sam3d_generation_rerun": False,
        "inference_policy": isolation,
        "read_only_links": {
            name: str((output / name).resolve()) for name in READ_ONLY_LINKS
        },
        "frozen_inputs": {
            "source_mesh_ranking": artifact_record(ranking_source),
            "object_masks": artifact_record(output / "segmentation/object_masks.npz"),
            "metric_depth": artifact_record(output / "depth/metric_depth.zarr"),
            "intrinsics": artifact_record(output / "calibration/intrinsics.npy"),
        },
        "outputs": {"mesh_ranking": artifact_record(ranking_output)},
    }
    evidence_path = output / "sam3d_rerank_workspace.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence_path
