#!/usr/bin/env python3
"""O5 hybrid candidate pool over O0/O1/O2/O3 outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_PATH = RUNS_ROOT / "adt_o5_hybrid_summary.json"
BC_PATH = RUNS_ROOT / "adt_phase_bc_ablation_summary.json"
BUNDLE_ROOT = RUNS_ROOT / "adt_bundlesdf"
MODELFREE_ROOT = RUNS_ROOT / "adt_foundationpose_modelfree"


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.replace("_rgbobject", "")


def _load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _fit_mesh(mesh: trimesh.Trimesh, mask: np.ndarray, depth: np.ndarray, valid: np.ndarray, K: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    camera = vertices
    z = camera[:, 2]
    if np.count_nonzero(z > 1e-5) < 3:
        return {"iou": 0.0, "relative_depth": None, "mesh_points": len(vertices)}
    uvw = camera @ K.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    rendered = np.zeros(mask.shape, dtype=np.uint8)
    for face in np.asarray(mesh.faces):
        if np.any(z[face] <= 1e-5):
            continue
        poly = np.rint(uv[face]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= mask.shape[1] or poly[:, 1].max() < 0 or poly[:, 1].min() >= mask.shape[0]:
            continue
        import cv2

        cv2.fillConvexPoly(rendered, poly, 1)
    rendered = rendered.astype(bool)
    union = np.logical_or(rendered, mask).sum()
    iou = float(np.logical_and(rendered, mask).sum() / union) if union else 0.0
    overlap = rendered & mask & valid & (depth > 0)
    if overlap.any():
        mesh_z = float(np.median(z[z > 1e-5]))
        gt_z = float(np.median(depth[overlap]))
        rel = abs(mesh_z - gt_z) / max(gt_z, 1e-3)
    else:
        rel = None
    return {"iou": iou, "relative_depth": rel, "mesh_points": int(len(vertices))}


def _candidate_score(fit: dict[str, Any], accepted: bool = False) -> float:
    iou = float(fit.get("iou") or 0.0)
    rel = fit.get("relative_depth")
    depth_score = math.exp(-min(max(float(rel), 0.0), 2.0)) if rel is not None else 0.0
    score = 0.55 * iou + 0.45 * depth_score
    if accepted:
        score += 0.10
    return float(score)


def main() -> int:
    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    bc = {row["run"].replace("_rgbobject", "").replace("_phase_final", ""): row for row in json.loads(BC_PATH.read_text(encoding="utf-8")) if "run" in row}
    results = []
    for row in rows:
        source = Path(row["source_run"])
        target = Path(row["target_run"])
        key = _row_key(row)
        mask_npz = _load_masks(target / "segmentation" / "object_masks.npz")
        depth_group = zarr.open(str(source / "depth" / "metric_depth.zarr"), mode="r")
        masks = mask_npz["masks"]
        valid_frames = mask_npz["valid"]
        ref_index = int(np.flatnonzero(valid_frames)[0]) if valid_frames.any() else 0
        mask = masks[ref_index] > 0
        depth = np.asarray(depth_group["depth_m"][ref_index], dtype=np.float32)
        valid = np.asarray(depth_group["valid"][ref_index], dtype=bool)
        K = np.load(source / "calibration" / "intrinsics.npy").reshape(3, 3)
        candidates = []

        # O2 BundleSDF
        for mesh_name in ("textured_mesh.obj", "mesh_cleaned.obj"):
            mesh_path = BUNDLE_ROOT / key / mesh_name
            if mesh_path.is_file():
                mesh = trimesh.load(mesh_path, process=False)
                fit = _fit_mesh(mesh, mask, depth, valid, K)
                candidates.append({"source": "o2_bundlesdf", "mesh": str(mesh_path), "fit": fit, "score": _candidate_score(fit)})
                break
        # O3 FoundationPose model-free
        mesh_path = MODELFREE_ROOT / key / "model_free_mesh.obj"
        if mesh_path.is_file():
            mesh = trimesh.load(mesh_path, process=False)
            fit = _fit_mesh(mesh, mask, depth, valid, K)
            candidates.append({"source": "o3_model_free", "mesh": str(mesh_path), "fit": fit, "score": _candidate_score(fit)})

        # O0/O1 existing FP accepted candidates
        for bc_key, bc_row in bc.items():
            if bc_key not in key:
                continue
            if bc_row.get("fp_final_accepted"):
                candidates.append({"source": "o0_o1_fp", "accepted": True, "fit": {"iou": bc_row.get("fp_final_iou"), "relative_depth": bc_row.get("fp_final_rel_depth")}, "score": 1.0})
                break

        if candidates:
            best = max(candidates, key=lambda c: float(c["score"]))
        else:
            best = None
        results.append({
            "prototype": row["prototype"],
            "window": row["window"],
            "source_run": str(source),
            "candidates": candidates,
            "selected": best,
        })
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False)[:30000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
