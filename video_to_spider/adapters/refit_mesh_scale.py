"""Refit existing canonical SAM3D proposals to accepted primary metric depth.

This isolates the effect of depth and static scale without rerunning stochastic
mesh generation.  It writes a new ranking and never mutates proposal meshes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..schemas import SCHEMA_VERSION
from .depth_gate import require_depth_gate
from .sam3d_objects import _finite_json, mesh_integrity, static_fit_metrics


def run(
    run_dir: str | Path,
    *,
    source_ranking: str | Path | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    import zarr

    root = Path(run_dir).resolve()
    gate = require_depth_gate(root)
    if gate is None:
        raise RuntimeError("metric_refit_requires_accepted_metric_depth_gate")
    source = (
        Path(source_ranking).resolve()
        if source_ranking else root / "mesh_proposals/mesh_ranking.json"
    )
    destination = (
        Path(output_path).resolve()
        if output_path else root / "mesh_proposals/mesh_ranking_metric_refit.json"
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists: {destination}; pass --overwrite")
    ranking = json.loads(source.read_text(encoding="utf-8"))
    with np.load(root / "segmentation/object_masks.npz", allow_pickle=False) as artifact:
        mask_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        masks = np.asarray(artifact["masks"], dtype=bool)
    mask_lookup = {int(frame): index for index, frame in enumerate(mask_indices)}
    depth = zarr.open_group(str(root / "depth/metric_depth.zarr"), mode="r")
    depth_indices = np.asarray(depth["frame_indices"], dtype=np.int64)
    depth_lookup = {int(frame): index for index, frame in enumerate(depth_indices)}
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    proposals: list[dict[str, Any]] = []
    for original in ranking.get("proposals", []):
        proposal = dict(original)
        frame_index = int(proposal["frame_index"])
        if frame_index not in mask_lookup or frame_index not in depth_lookup:
            proposal.update({
                "qualified": False,
                "static_score": -1.0,
                "refit_error": "keyframe_missing_from_mask_or_depth_timeline",
            })
            proposals.append(proposal)
            continue
        mesh_path = root / "mesh_proposals" / proposal["visual_mesh"]
        mesh = trimesh.load_mesh(mesh_path, process=False)
        model_layout = proposal["model_layout"]
        depth_at = depth_lookup[frame_index]
        fit, transform, selected_scale = static_fit_metrics(
            mesh,
            masks[mask_lookup[frame_index]],
            np.asarray(depth["depth_m"][depth_at]),
            np.asarray(depth["valid"][depth_at], dtype=bool),
            K,
            rotation_wxyz=model_layout["rotation_wxyz"],
            translation=model_layout["translation"],
            layout_scale=model_layout["scale"],
        )
        integrity = mesh_integrity(mesh)
        qualified = bool(integrity["qualified"] and fit["scale_fit_accepted"])
        static_score = (
            0.30 * integrity["score"]
            + 0.45 * fit["silhouette_iou"]
            + 0.25 * fit["depth_score"]
        )
        proposal.update({
            "qualified": qualified,
            "static_score": float(static_score),
            "integrity": integrity,
            "fit": fit,
            "selected_scale_m": float(selected_scale),
            "T_camera_object_initial": transform.tolist(),
            "metric_refit": True,
        })
        proposals.append(proposal)
    ordered = sorted(
        proposals,
        key=lambda item: (not item.get("qualified", False), -float(item.get("static_score", -1.0))),
    )
    for rank, proposal in enumerate(ordered, start=1):
        proposal["rank"] = rank
    qualified_count = sum(bool(proposal.get("qualified")) for proposal in ordered)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "sam3d_objects_metric_scale_refit",
        "source_ranking": str(source),
        "primary_depth": str(root / "depth/metric_depth.zarr"),
        "depth_gate": str(root / "depth/depth_gate.json"),
        "depth_gate_policy": gate["policy"],
        "foundationpose_used": False,
        "keyframes": ranking.get("keyframes", []),
        "ranking_policy": (
            "qualified = mesh_integrity and metric scale fit; score = 0.30 integrity "
            "+ 0.45 silhouette_iou + 0.25 depth_score"
        ),
        "proposals": [_finite_json(proposal) for proposal in ordered],
        "qualified_count": qualified_count,
        "success": bool(qualified_count),
        "failure_reason": None if qualified_count else "no_proposal_passed_metric_scale_fit",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not qualified_count:
        raise RuntimeError("mesh_scale_refit_failed: no qualified proposal")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-ranking", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        run(
            args.run_dir,
            source_ranking=args.source_ranking,
            output_path=args.output_path,
            overwrite=args.overwrite,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
