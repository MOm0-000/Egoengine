#!/usr/bin/env python3
"""Normalize an Aria Digital Twin sequence folder into the repo's standard layout.

The official Aria Dataset Explorer download places VRS files directly at the
sequence root::

    <uid>/video.vrs
    <uid>/depth_images.vrs
    <uid>/segmentations.vrs
    <uid>/instances.json
    <uid>/mps/slam/closed_loop_trajectory.csv

The pipeline expects the VRS files under ``vrs_files/`` and keeps the other
files at the sequence root.  This script converts the folder in place and is
idempotent, so the same command can be reused for every future sample.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


VRS_FILES = {
    "video": {
        "dst": "vrs_files/video.vrs",
        "candidates": ("video.vrs",),
        "extract": False,
    },
    "depth": {
        "dst": "vrs_files/depth_images.vrs",
        "candidates": ("depth_images.vrs",),
        "extract": False,
    },
    "segmentation": {
        "dst": "vrs_files/segmentations.vrs",
        "candidates": ("segmentations.vrs",),
        "extract": False,
    },
}


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_file(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    return None


def _find_vrs_candidate(root: Path, kind: str, uid: str) -> Path | None:
    """Also accepts Explorer/CDN filenames such as ``<uid>_main_recording.vrs``."""
    config = VRS_FILES[kind]
    direct = _pick_file(root, config["candidates"])
    if direct is not None:
        return direct
    suffix = {
        "video": "_main_recording.vrs",
        "depth": "_depth.vrs",
        "segmentation": "_segmentation.vrs",
    }[kind]
    matches = sorted(root.glob(f"*{suffix}"))
    if kind == "video":
        main_vrs = [p for p in matches if p.name.endswith("_main_recording.vrs")]
        if main_vrs:
            return main_vrs[0]
    return matches[0] if matches else None


def _install_vrs(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        if src.stat().st_size == dst.stat().st_size and _sha1(src) == _sha1(dst):
            # Duplicate already installed; keep the canonical copy and remove the source.
            src.unlink()
            return "skip-duplicate"
        raise RuntimeError(
            f"refusing to overwrite {dst} with a different file {src}; "
            "resolve manually or move the existing file aside"
        )
    shutil.move(str(src), str(dst))
    return "moved"


def normalize_sequence(seq_dir: str | Path) -> dict[str, str]:
    seq_dir = Path(seq_dir).resolve()
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    uid = seq_dir.name
    report: dict[str, str] = {"uid": uid, "directory": str(seq_dir)}

    for kind, config in VRS_FILES.items():
        src = _find_vrs_candidate(seq_dir, kind, uid)
        if src is None:
            report[kind] = "missing"
            continue
        report[kind] = _install_vrs(src, seq_dir / config["dst"])

    # closed_loop_trajectory.csv may already be under mps/slam or at the root.
    dst_traj = seq_dir / "mps/slam/closed_loop_trajectory.csv"
    if not dst_traj.is_file():
        src_traj = seq_dir / "closed_loop_trajectory.csv"
        if src_traj.is_file():
            dst_traj.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_traj), str(dst_traj))
            report["trajectory"] = "moved"
        else:
            report["trajectory"] = "missing"
    else:
        report["trajectory"] = "present"

    instances = seq_dir / "instances.json"
    report["instances"] = "present" if instances.is_file() else "missing"
    report["normalized"] = bool(
        (seq_dir / "vrs_files/video.vrs").is_file()
        and (seq_dir / "vrs_files/depth_images.vrs").is_file()
        and (seq_dir / "vrs_files/segmentations.vrs").is_file()
        and dst_traj.is_file()
        and instances.is_file()
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(normalize_sequence(args.sequence_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
