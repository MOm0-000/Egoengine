#!/usr/bin/env python3
"""Fetch fresh ADT download-link JSON from the Aria Dataset Explorer API.

The Aria Dataset Explorer serves per-sequence download links from::

    GET https://explorer.projectaria.com/data/adt/<sequence_uid>/download_links
    Header: x-api-key: <your explorer API key>

The public metadata list (all sequence uids) is available without a key at::

    GET https://explorer.projectaria.com/data/adt

This script only fetches and archives the raw JSON responses so they can be
transformed into the CDN JSON consumed by ``adt_benchmark_dataset_downloader``.
It intentionally does not download the multi-GB VRS files itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


EXPLORER_API = "https://explorer.projectaria.com/data/adt"


def fetch_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(url, headers={"x-api-key": api_key})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {payload[:200]!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("ARIA_EXPLORER_API_KEY"))
    parser.add_argument("--sequence", action="append", dest="sequences", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.api_key:
        print("--api-key or ARIA_EXPLORER_API_KEY is required", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sequence in args.sequences:
        url = f"{EXPLORER_API}/{sequence}/download_links"
        try:
            payload = fetch_json(url, args.api_key)
        except Exception as exc:
            print(f"[FAIL] {sequence}: {exc}", file=sys.stderr)
            continue
        target = args.output_dir / f"{sequence}.json"
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] {sequence} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
