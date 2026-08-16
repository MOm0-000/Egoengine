#!/usr/bin/env python3
"""Parallel HTTP-range downloader for a single large file.

The container proxy supports ``Accept-Ranges: bytes``, so a single large asset
can be split into chunks and fetched concurrently.  Completed chunks are kept in
a sidecar directory so a re-run resumes without redownloading them.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _head_size(url: str) -> int:
    proc = subprocess.run(
        ["curl", "-sk", "-I", "-L", url], capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine content-length for {url}")


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_range(url: str, dest: Path, start: int, end: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl", "-sk", "-L", "--retry", "10", "--retry-all-errors",
        "-C", "-", "-r", f"{start}-{end}", url, "-o", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed rc={proc.returncode} for range {start}-{end}")
    tmp.replace(dest)


def download(url: str, dest: Path, size: int, chunks: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    workdir = dest.parent / f"{dest.name}.parts"
    workdir.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing >= size:
        return dest
    start_offset = existing
    bounds = []
    step = (size - start_offset + chunks - 1) // chunks
    for i in range(chunks):
        start = start_offset + i * step
        if start >= size:
            break
        end = min(start + step - 1, size - 1)
        bounds.append((start, end))
    futures = {}
    with ThreadPoolExecutor(max_workers=len(bounds)) as executor:
        for start, end in bounds:
            part = workdir / f"{start:012d}-{end:012d}.part"
            if part.is_file() and part.stat().st_size == end - start + 1:
                continue
            futures[executor.submit(_fetch_range, url, part, start, end)] = (start, end)
        for future in as_completed(futures):
            future.result()
    mode = "ab" if existing else "wb"
    with dest.open(mode) as out:
        for start, end in bounds:
            part = workdir / f"{start:012d}-{end:012d}.part"
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--sha1")
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args(argv)
    size = _head_size(args.url)
    print(f"downloading {size / 1e6:.1f} MB in {args.chunks} chunks", flush=True)
    dest = download(args.url, args.dest, size, args.chunks)
    if args.sha1:
        digest = _sha1_file(dest)
        if digest != args.sha1.lower():
            print(f"sha1 mismatch: {digest} != {args.sha1}", file=sys.stderr)
            return 1
        print("sha1 OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
