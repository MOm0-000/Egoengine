#!/usr/bin/env python3
"""Run MUSt3R geometry priors on the fixed ADT windows.

MUSt3R's repository ships a CLI that reconstructs a directory of images and
exports point clouds. This driver prepares one directory per ADT sample and
invokes that CLI in the isolated ``must3r`` environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_ROOT = RUNS_ROOT / "adt_must3r"
MUST3R_ROOT = REPO_ROOT / "third_party/must3r"
CHECKPOINT = MUST3R_ROOT / "checkpoints/MUSt3R_512.pth"
MUST3R_PY = Path("/home/zzx/miniconda3/envs/must3r/bin/python")


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.removesuffix("_rgbobject")


def _prepare_images(row: dict, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    source_rgb = Path(row["source_run"]) / "frames" / "rgb"
    for path in sorted(source_rgb.glob("*.png")):
        target = image_dir / path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path)


def _run_cli(row: dict, image_dir: Path, out_dir: Path, gpu: str) -> dict:
    command = [
        str(MUST3R_PY),
        str(MUST3R_ROOT / "get_reconstruction.py"),
        "--image_dir", str(image_dir),
        "--output", str(out_dir),
        "--weights", str(CHECKPOINT),
        "--image_size", "512",
        "--execution_mode", "linseq",
        "--num_mem_imgs", "30",
        "--max_bs", "1",
        "--render_once",
        "--file_type", "ply",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = os.pathsep.join(
        [str(MUST3R_ROOT), str(MUST3R_ROOT / "dust3r"), env.get("PYTHONPATH", "")]
    )
    log_path = out_dir / "must3r.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=MUST3R_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.write(f"\n[returncode] {proc.returncode}\n")
    result = {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "returncode": proc.returncode,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
        "out_dir": str(out_dir),
    }
    scene_files = sorted(out_dir.glob("scene_*.ply")) + sorted(out_dir.glob("scene.pkl"))
    result["scene_files"] = [str(path.relative_to(out_dir)) for path in scene_files]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--only", action="append", dest="only_keys", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    summary_path = OUTPUT_ROOT / "adt_must3r_summary.json"
    if summary_path.is_file():
        summary_by_key = {
            item["row_key"]: item
            for item in json.loads(summary_path.read_text(encoding="utf-8"))
        }
    else:
        summary_by_key = {}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        key = _row_key(row)
        if args.only_keys and key not in args.only_keys:
            continue
        if args.skip_existing and summary_by_key.get(key, {}).get("returncode") == 0:
            continue
        image_dir = OUTPUT_ROOT / "inputs" / key
        out_dir = OUTPUT_ROOT / key
        print(f"[prepare] {key}", flush=True)
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
        _prepare_images(row, image_dir)
        print(f"[run] {key}", flush=True)
        result = _run_cli(row, image_dir, out_dir, args.gpu)
        summary_by_key[key] = result
        summary_path.write_text(
            json.dumps(list(summary_by_key.values()), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[done] {key} rc={result['returncode']} {result.get('elapsed_s', 0):.1f}s", flush=True)
        count += 1
        if args.limit and count >= args.limit:
            break

    print(json.dumps(list(summary_by_key.values()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
