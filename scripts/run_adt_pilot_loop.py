#!/usr/bin/env python3
"""Watch the 10 pilot ADT downloads and run the P1 depth benchmark per sequence.

The long-running download session and this watcher run concurrently.  As soon
as a sequence has all required VRS/GT/MPS files it is evaluated once, then its
result is recorded in ``runs/adt_depth_benchmark_summary.json`` and skipped on
the next pass.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_adt_depth_benchmark import _sequence_ready, run_sequence


ADT_ROOT = Path("/data_all/zzx/egoengine/adt_data")
RUNS_ROOT = _REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
PILOT_JSON = _REPO_ROOT / "docs" / "adt_pilot_sequences.json"
FS_REPOSITORY = Path(
    "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/third_party/FoundationStereo"
)
FS_CHECKPOINT = Path(
    "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/checkpoints/23-51-11/model_best_bp2.pth"
)
FS_CONFIG = Path(
    "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/foundationstereo/checkpoints/23-51-11/cfg.yaml"
)


def _existing_sequences() -> set[str]:
    if not SUMMARY_PATH.is_file():
        return set()
    try:
        payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else []
        return {row.get("sequence") for row in rows if isinstance(row, dict)}
    except Exception:
        return set()


def _load_summary_rows(pilot_uids: set[str]) -> list[dict]:
    if not SUMMARY_PATH.is_file():
        return []
    try:
        payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else []
        return [row for row in rows if isinstance(row, dict) and row.get("sequence") in pilot_uids]
    except Exception:
        return []


def main() -> int:
    pilots = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    pilot_uids = {entry["uid"] for entry in pilots}
    done = _existing_sequences() & pilot_uids
    while len(done) < len(pilots):
        for entry in pilots:
            uid = entry["uid"]
            if uid in done:
                continue
            if not _sequence_ready(ADT_ROOT / uid):
                print(f"[waiting] {uid}", flush=True)
                continue
            print(f"[ready] {uid} -> {entry['objs']}", flush=True)
            try:
                summary = run_sequence(
                    uid=uid,
                    prototypes=entry["objs"],
                    adt_root=ADT_ROOT,
                    runs_root=RUNS_ROOT,
                    frames=30,
                    stride=10,
                    min_pixels=40,
                    prefer_dynamic=True,
                    start_frame=None,
                    end_frame=None,
                    force=False,
                    gpu_index=6,
                    gpu_uuid="GPU-70b5cd84-9fee-40a5-9109-513527bc8fad",
                    fs_repository=FS_REPOSITORY,
                    fs_checkpoint=FS_CHECKPOINT,
                    fs_config=FS_CONFIG,
                )
            except Exception as exc:
                print(f"[FAIL] {uid}: {exc}", flush=True)
                time.sleep(30)
                continue
            existing = _load_summary_rows(pilot_uids)
            existing = [row for row in existing if row.get("sequence") != uid]
            existing.append(summary)
            SUMMARY_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            done.add(uid)
            print(f"[done] {uid}", flush=True)
        if len(done) < len(pilots):
            time.sleep(60)
    print(f"[all-done] {len(done)} sequences", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
