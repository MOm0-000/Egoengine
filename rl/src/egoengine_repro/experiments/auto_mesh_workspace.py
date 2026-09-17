"""Prepare isolated SAM3D workspaces from frozen automatic inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..artifacts import artifact_record, ensure_isolated_output


READ_ONLY_INPUTS = ("frames", "calibration", "input", "segmentation")


def prepare_auto_mesh_workspace(
    source_v2_run: str | Path, raw_depth_run: str | Path,
    output_dir: str | Path, *, overwrite: bool = False,
) -> Path:
    source = Path(source_v2_run).resolve()
    raw = Path(raw_depth_run).resolve()
    output = ensure_isolated_output(output_dir, source)
    if any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"auto mesh workspace is not empty: {output}")
    for name in READ_ONLY_INPUTS:
        target = output / name
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise FileExistsError(target)
        target.symlink_to(source_path, target_is_directory=True)
    depth_source = raw / "depth"
    if not (depth_source / "metric_depth.zarr").is_dir():
        # Camera-motion calibrated runs store metric_depth.zarr at the run
        # root, while regular model runs keep it under depth/.
        depth_source = raw
    if not (depth_source / "metric_depth.zarr").is_dir():
        raise FileNotFoundError(depth_source / "metric_depth.zarr")
    depth_target = output / "depth"
    if depth_target.is_symlink():
        depth_target.unlink()
    elif depth_target.exists():
        raise FileExistsError(depth_target)
    depth_target.symlink_to(depth_source, target_is_directory=True)
    shutil.copy2(source / "manifest.json", output / "manifest.json")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "run_id": f"{manifest.get('run_id', source.name)}-auto-mesh",
        "repro_profile": "auto_segmentation_v2_sam3d_mesh",
        "branch_source_run": str(source),
        "raw_depth_source_run": str(raw),
        "branch_inputs_are_read_only_symlinks": True,
        "ground_truth_isolation": {
            "object_gt_used": False, "gt_hand_used": False,
            "manual_point_used": False, "oracle_depth_used": False,
        },
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "profile": "auto_segmentation_v2_sam3d_mesh_workspace",
        "source_v2_run": str(source), "raw_depth_run": str(raw),
        "workspace": str(output),
        "read_only_links": {
            name: str((output / name).resolve())
            for name in (*READ_ONLY_INPUTS, "depth")
        },
        "frozen_inputs": {
            "object_masks": artifact_record(source / "segmentation/object_masks.npz"),
            "hand_masks": artifact_record(source / "segmentation/hand_masks.npz"),
            "raw_depth": artifact_record(depth_source / "metric_depth.zarr"),
            "intrinsics": artifact_record(source / "calibration/intrinsics.npy"),
        },
        "inference_policy": manifest["ground_truth_isolation"],
    }
    evidence_path = output / "auto_mesh_workspace.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence_path
