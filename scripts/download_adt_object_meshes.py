#!/usr/bin/env python3
"""Download ADT object 3D models (meshes) from the DTC objects links JSON.

Expected JSON schema::

    {"releases": {"ADT": {"objects": {
        "<prototype>": {
            "3d-asset_glb": {"filename": str, "sha1sum": str, "download_url": str},
            "metadata": {...},
            "license": {...}}}}}}

Each prototype is laid out as a small object library directory::

    <output_root>/<prototype>/3d-asset.glb
    <output_root>/<prototype>/metadata.json
    <output_root>/<prototype>/license.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_ASSET_MAP = {
    "3d-asset_glb": "3d-asset.glb",
    "metadata": "metadata.json",
    "license": "license.txt",
}


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(entry: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and entry.get("sha1sum") and _sha1_file(dest) == entry["sha1sum"].lower():
        print(f"  [skip] {dest.name}: already matches sha1", flush=True)
        return
    for _ in range(10):
        proc = subprocess.run(
            ["wget", "-c", "--tries=5", "--timeout=120", "--no-check-certificate", "-O", str(dest), entry["download_url"]],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.returncode == 0:
            if not entry.get("sha1sum") or _sha1_file(dest) == entry["sha1sum"].lower():
                return
            print(f"  [retry] {dest.name}: sha1 mismatch", flush=True)
        time.sleep(1)
    raise RuntimeError(f"failed to download {dest.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtc-json", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.dtc_json.read_text(encoding="utf-8"))
    objects = payload["releases"]["ADT"]["objects"]
    jobs = []
    for prototype in args.objects:
        if prototype not in objects:
            print(f"[missing] {prototype}", file=sys.stderr, flush=True)
            continue
        jobs.append((prototype, objects[prototype]))
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as executor:
        futures = {}
        for prototype, entry in jobs:
            out_dir = args.output_root / prototype
            for asset, target_name in _ASSET_MAP.items():
                if asset not in entry:
                    continue
                dest = out_dir / target_name
                futures[executor.submit(_download_file, entry[asset], dest)] = (prototype, target_name)
        for future in as_completed(futures):
            prototype, target_name = futures[future]
            try:
                future.result()
                print(f"  [ok] {prototype}/{target_name}", flush=True)
            except Exception as exc:
                print(f"  [FAIL] {prototype}/{target_name}: {exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
