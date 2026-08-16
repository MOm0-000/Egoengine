#!/usr/bin/env python3
"""Prepare isolated Phase-final clones that use the selected best object mask."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
OUTPUT_STATE = RUNS_ROOT / "adt_phase_final_runs_state.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0

FAILED_PROTOTYPES = {"BlackCeramicMug", "Flask", "StepStool"}
SYMLINK_DIRS = (
    "frames",
    "calibration",
    "depth",
    "input",
    "evaluation",
    "hands",
    "hands_right",
    "hands_hamer",
    "hands_right_hamer",
    "hands_hamer_ba",
    "visualization",
)


def _best_mask_path(run_dir: Path, prototype: str) -> Path:
    if prototype in FAILED_PROTOTYPES:
        merged = run_dir / "segmentation_s2_multianchor_merged" / "object_masks.npz"
        if merged.is_file():
            return merged
    s1 = run_dir / "segmentation_s1" / "object_masks.npz"
    if s1.is_file():
        return s1
    return run_dir / "segmentation" / "object_masks.npz"


def _clone_run(source: Path, target: Path, prototype: str) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in SYMLINK_DIRS:
        source_dir = source / name
        if source_dir.is_symlink() or source_dir.is_dir():
            (target / name).symlink_to(source_dir, target_is_directory=True)
    manifest = source / "manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, target / "manifest.json")
    segmentation_source = source / "segmentation"
    if segmentation_source.is_dir():
        shutil.copytree(segmentation_source, target / "segmentation", dirs_exist_ok=True)
    else:
        (target / "segmentation").mkdir(parents=True, exist_ok=True)
    mask_path = _best_mask_path(source, prototype)
    if not mask_path.is_file():
        raise FileNotFoundError(f"no object mask for {source}: {mask_path}")
    shutil.copy2(mask_path, target / "segmentation" / "object_masks.npz")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="phase_final")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    state: dict[str, Any] = {
        "schema_version": "1.0",
        "suffix": args.suffix,
        "rows": [],
    }
    for row in phase0._rows():
        source = phase0._object_run(row)
        if source is None:
            state["rows"].append({"prototype": row["prototype"], "window": row["window"], "error": "missing_source"})
            continue
        target = RUNS_ROOT / f"{source.name}_{args.suffix}"
        source_mask = _best_mask_path(source, row["prototype"])
        if target.exists() and not args.force:
            status = "existing"
        else:
            _clone_run(source, target, row["prototype"])
            status = "prepared"
        entry = {
            "prototype": row["prototype"],
            "window": row["window"],
            "source_run": str(source),
            "target_run": str(target),
            "selected_mask": str(source_mask),
            "status": status,
        }
        state["rows"].append(entry)
        OUTPUT_STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
