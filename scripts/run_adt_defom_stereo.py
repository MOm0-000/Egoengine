#!/usr/bin/env python3
"""Run DEFOM-Stereo on the 10 ADT rectified left/right frame pairs."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
DEMO = REPO_ROOT / "third_party" / "DEFOM-Stereo" / "demo.py"
CHECKPOINT = REPO_ROOT / "third_party" / "DEFOM-Stereo" / "checkpoints" / "defomstereo_vitl_sceneflow.pth"


def main() -> int:
    rows = json.loads((RUNS_ROOT / "adt_depth_benchmark_summary.json").read_text())
    for row in rows:
        run_dir = Path(row["run_dir"])
        left_dir = run_dir / "frames" / "rgb"
        right_dir = run_dir / "frames" / "right"
        out_dir = run_dir / "evaluation" / "defom_stereo"
        if not left_dir.is_dir() or not right_dir.is_dir():
            print(f"[skip] missing frames {run_dir.name}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[defom] {run_dir.name}", flush=True)
        started = time.time()
        cmd = [
            "conda", "run", "-n", "defomstereo", "python", str(DEMO),
            "--restore_ckpt", str(CHECKPOINT),
            "-l", str(left_dir / "*.png"),
            "-r", str(right_dir / "*.png"),
            "--output_directory", str(out_dir),
            "--save_numpy",
        ]
        proc = subprocess.run(
            cmd, cwd=str(DEMO.parent), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        print(proc.stdout[-500:], flush=True)
        print(f"[defom] rc={proc.returncode} elapsed={time.time()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
