#!/usr/bin/env python3
"""Prepare ADT samples as BundleSDF custom videos and run BundleSDF.

This is an ablation driver for O2: unknown-object tracking + reconstruction.
It uses the fixed 10 ADT rows from ``runs/adt_phase_final_runs_state.json``,
exports each row into BundleSDF's ``rgb/depth/masks/cam_K.txt`` layout, then
runs ``third_party/BundleSDF/run_custom.py`` in the isolated ``bundlesdf``
conda environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_ROOT = RUNS_ROOT / "adt_bundlesdf"
LOG_ROOT = OUTPUT_ROOT / "logs"
BUNDLESDF_ROOT = REPO_ROOT / "third_party/BundleSDF"
BUNDLESDF_PY = Path("/home/zzx/miniconda3/envs/bundlesdf/bin/python")


def _row_key(row: dict) -> str:
    source = Path(row["source_run"])
    return source.name.replace("_rgbobject", "")


def _depth_zarr(row: dict) -> Path:
    return Path(row["source_run"]) / "depth" / "metric_depth.zarr"


def _prepare_input(row: dict, custom_dir: Path) -> None:
    source_run = Path(row["source_run"])
    target_run = Path(row["target_run"])

    rgb_src = source_run / "frames" / "rgb"
    rgb_dst = custom_dir / "rgb"
    depth_dst = custom_dir / "depth"
    mask_dst = custom_dir / "masks"
    for child in (rgb_dst, depth_dst, mask_dst):
        child.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_src.glob("*.png"))
    if not rgb_files:
        raise RuntimeError(f"no RGB frames under {rgb_src}")

    # Depth zarr is written in the original pipeline frame order.
    depth_root = zarr.open(_depth_zarr(row), mode="r")
    depth_m = depth_root["depth_m"][:]
    valid = depth_root["valid"][:]
    frame_indices = depth_root["frame_indices"][:]

    mask_path = target_run / "segmentation" / "object_masks.npz"
    mask_npz = np.load(mask_path)
    masks = mask_npz["masks"]
    mask_frame_indices = mask_npz["frame_indices"]
    mask_by_index = {
        int(frame_idx): mask
        for frame_idx, mask in zip(mask_frame_indices, masks)
    }

    # BundleSDF reads RGB/depth/masks by matching file stems; our filenames are
    # already ordered zero-padded frame IDs. Depth is in metres, export as mm.
    for i, rgb_file in enumerate(rgb_files):
        shutil.copy2(rgb_file, rgb_dst / rgb_file.name)

        frame_idx = int(frame_indices[i]) if i < len(frame_indices) else int(rgb_file.stem)
        if frame_idx not in mask_by_index:
            mask = np.zeros_like(masks[0], dtype=np.uint8)
        else:
            mask = (mask_by_index[frame_idx] > 0).astype(np.uint8)

        if i < len(depth_m):
            depth = depth_m[i].astype(np.float32)
            frame_valid = valid[i] if i < len(valid) else np.ones_like(depth, dtype=bool)
            depth = np.where(frame_valid & np.isfinite(depth) & (depth > 0) & (depth <= 1.0), depth, 0.0)
        else:
            depth = np.zeros_like(depth_m[0], dtype=np.float32)

        H, W = depth.shape[:2]
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        depth_mm = np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dst / rgb_file.name), depth_mm)
        cv2.imwrite(str(mask_dst / rgb_file.name), mask)

    K = np.load(source_run / "calibration" / "intrinsics.npy").reshape(3, 3)
    np.savetxt(custom_dir / "cam_K.txt", K, fmt="%.12g")

    meta = {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "source_run": str(source_run),
        "target_run": str(target_run),
        "rgb_files": len(rgb_files),
        "depth_zarr": str(_depth_zarr(row)),
        "mask_path": str(mask_path),
        "intrinsics": K.tolist(),
    }
    (custom_dir / "adt_bundlesdf_input.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_bundlesdf(row: dict, custom_dir: Path, out_dir: Path, gpu: str) -> dict:
    command = [
        str(BUNDLESDF_PY),
        "run_custom.py",
        "--mode", "run_video",
        "--video_dir", str(custom_dir),
        "--out_folder", str(out_dir),
        "--use_segmenter", "0",
        "--use_gui", "0",
        "--stride", "1",
        "--debug_level", "1",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(BUNDLESDF_ROOT)
    log_path = LOG_ROOT / f"{_row_key(row)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=BUNDLESDF_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.write(f"\n[returncode] {proc.returncode}\n")
    elapsed = time.time() - started
    result = {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
        "out_dir": str(out_dir),
    }
    if proc.returncode != 0:
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-50:])
        except Exception:
            tail = ""
        result["tail"] = tail
    return result


def _collect_outputs(out_dir: Path) -> dict:
    mesh_files = sorted(out_dir.glob("**/*.obj"))
    pose_files = sorted((out_dir / "ob_in_cam").glob("*.txt")) if (out_dir / "ob_in_cam").is_dir() else []
    textured = out_dir / "textured_mesh.obj"
    return {
        "mesh_count": len(mesh_files),
        "pose_count": len(pose_files),
        "textured_mesh_exists": textured.is_file(),
        "mesh_files": [str(p.relative_to(out_dir)) for p in mesh_files[-5:]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="6", help="CUDA_VISIBLE_DEVICES value for BundleSDF")
    parser.add_argument("--only", action="append", dest="only_keys", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    rows = state["rows"]
    summary_path = OUTPUT_ROOT / "adt_bundlesdf_summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_by_key = {row["row_key"]: row for row in previous}
    else:
        summary_by_key = {}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        key = _row_key(row)
        if args.only_keys and key not in args.only_keys:
            continue
        out_dir = OUTPUT_ROOT / key
        if args.skip_existing and summary_by_key.get(key, {}).get("returncode") == 0:
            continue
        print(f"[prepare] {key}", flush=True)
        custom_dir = OUTPUT_ROOT / "inputs" / key
        if custom_dir.is_dir():
            # Recreating symlink/copy artifacts with a clean directory is safer
            # than relying on partially-written inputs.
            for child in custom_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _prepare_input(row, custom_dir)
        print(f"[run] {key}", flush=True)
        result = _run_bundlesdf(row, custom_dir, out_dir, args.gpu)
        result.update(_collect_outputs(out_dir))
        summary_by_key[key] = result
        summary = list(summary_by_key.values())
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[done] {key} rc={result['returncode']} {result.get('elapsed_s', 0):.1f}s "
              f"mesh={result.get('mesh_count')} poses={result.get('pose_count')}", flush=True)
        count += 1
        if args.limit and count >= args.limit:
            break

    print(json.dumps(list(summary_by_key.values()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
