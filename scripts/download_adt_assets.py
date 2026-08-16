#!/usr/bin/env python3
"""Download selected ADT sequence assets from the fresh CDN links JSON.

The JSON provided by the Aria Dataset Explorer uses this schema::

    {"sequences": {"<uid>": {"<asset>": {
        "filename": str, "sha1sum": str, "file_size_bytes": int, "download_url": str}}},
     "sequence_config": {...}}

For the stereo-depth benchmark we only need a few assets, so this script avoids
downloading synthetic video and MPS point maps by default.  It verifies the
published sha1, downloads independent assets in parallel, and lays files out as::

    <output_root>/<uid>/vrs_files/video.vrs
    <output_root>/<uid>/vrs_files/depth_images.vrs
    <output_root>/<uid>/vrs_files/segmentations.vrs
    <output_root>/<uid>/{instances.json, scene_objects.csv, ...}
    <output_root>/<uid>/mps/slam/closed_loop_trajectory.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ASSETS = ("main_vrs", "main_groundtruth", "depth", "segmentation", "mps_slam_trajectories")


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha1: str | None, label: str) -> Path:
    """Download with resume and retries via ``wget -c``.

    The container routes traffic through a proxy that occasionally truncates
    large transfers, so a plain single-pass ``urllib`` read can silently end
    early and produce a sha1 mismatch.  ``wget -c`` resumes the partial file on
    the next attempt.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 11):
        cmd = [
            "wget", "-c", "--tries=5", "--timeout=120", "--no-check-certificate",
            "-O", str(dest), url,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(f"    [{label}] wget attempt {attempt} failed (rc={proc.returncode})", flush=True)
            time.sleep(2)
            continue
        if expected_sha1 and _sha1_file(dest) != expected_sha1.lower():
            print(f"    [{label}] partial/truncated after attempt {attempt}, resuming", flush=True)
            time.sleep(1)
            continue
        return dest
    raise RuntimeError(f"failed to download {label} after 10 attempts")


def _extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        archive.extractall(dest_dir)
    return [dest_dir / name for name in names if not name.endswith("/")]


def _copy_vrs(entry: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if entry.suffix.lower() != ".vrs":
        raise RuntimeError(f"expected a .vrs file, got {entry}")
    shutil.copyfile(entry, dest)


def _install_asset(asset: str, downloaded: Path, seq_dir: Path) -> None:
    vrs_dir = seq_dir / "vrs_files"
    vrs_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(downloaded):
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _extract_zip(downloaded, Path(tmp))
            if asset == "main_vrs":
                candidates = [p for p in extracted if p.suffix.lower() == ".vrs"]
                _copy_vrs(candidates[0], vrs_dir / "video.vrs")
            elif asset == "depth":
                candidates = [p for p in extracted if p.name == "depth_images.vrs"]
                if candidates:
                    _copy_vrs(candidates[0], vrs_dir / "depth_images.vrs")
            elif asset == "segmentation":
                candidates = [p for p in extracted if p.name == "segmentations.vrs"]
                if candidates:
                    _copy_vrs(candidates[0], vrs_dir / "segmentations.vrs")
            elif asset == "mps_slam_trajectories":
                for p in extracted:
                    if p.name in {"closed_loop_trajectory.csv", "open_loop_trajectory.csv"}:
                        target = seq_dir / "mps" / "slam" / p.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(p, target)
            else:
                for p in extracted:
                    if p.is_file():
                        shutil.copyfile(p, seq_dir / p.name)
    else:
        if asset == "main_vrs":
            _copy_vrs(downloaded, vrs_dir / "video.vrs")
        else:
            shutil.copyfile(downloaded, seq_dir / downloaded.name)


def download_sequence(sequence: dict, uid: str, output_root: Path, assets: tuple[str, ...]) -> None:
    seq_dir = output_root / uid
    vrs_dir = seq_dir / "vrs_files"
    seq_dir.mkdir(parents=True, exist_ok=True)
    vrs_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, dict]] = []
    for asset in assets:
        entry = sequence.get(asset)
        if not isinstance(entry, dict):
            print(f"  [skip] {asset}: not present", flush=True)
            continue
        if not entry.get("download_url"):
            print(f"  [skip] {asset}: no download_url", flush=True)
            continue
        if asset == "main_vrs":
            existing = vrs_dir / "video.vrs"
            if existing.is_file() and _sha1_file(existing) == entry.get("sha1sum", "").lower():
                print(f"  [skip] main_vrs: existing video.vrs already matches sha1", flush=True)
                continue
        jobs.append((asset, entry))

    cache_dir = output_root / ".adt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    futures = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as executor:
        for asset, entry in jobs:
            filename = entry.get("filename") or f"{asset}.bin"
            dest = cache_dir / filename
            futures[executor.submit(_download, entry["download_url"], dest, entry.get("sha1sum"), asset)] = (asset, dest)
        for future in as_completed(futures):
            asset, dest = futures[future]
            try:
                future.result()
                print(f"  [installing] {asset}", flush=True)
                _install_asset(asset, dest, seq_dir)
            except Exception as exc:
                print(f"  [FAIL] {asset}: {exc}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdn-json", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    args = parser.parse_args(argv)
    payload = json.loads(args.cdn_json.read_text(encoding="utf-8"))
    sequences = payload["sequences"]
    for uid in args.sequences:
        if uid not in sequences:
            print(f"[missing] {uid}", file=sys.stderr, flush=True)
            continue
        print(f"[start] {uid}", flush=True)
        download_sequence(sequences[uid], uid, args.output_root, tuple(args.assets))
        print(f"[done] {uid} -> {args.output_root / uid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
