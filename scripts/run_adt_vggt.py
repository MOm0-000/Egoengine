#!/usr/bin/env python3
"""Run VGGT geometry priors on the fixed ADT sample windows.

The script exports VGGT camera encodings, camera matrices, depth maps, and
world point maps for each fixed ADT run. VGGT outputs are relative, so they
are treated as geometry priors only; the metric scale remains anchored to the
calibrated stereo baseline and FoundationStereo depth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_ROOT = RUNS_ROOT / "adt_vggt"
VGGT_ROOT = REPO_ROOT / "third_party/vggt"
CHECKPOINT = VGGT_ROOT / "checkpoints/model.pt"


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.removesuffix("_rgbobject")


def _load_model(device: str):
    sys.path.insert(0, str(VGGT_ROOT))
    from vggt.models.vggt import VGGT

    model = VGGT()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "cpu":
                module._buffers[name] = buffer.to(device)
    return model


def _run_row(model, row: dict, dtype: torch.dtype) -> dict:
    source_run = Path(row["source_run"])
    image_paths = sorted((source_run / "frames" / "rgb").glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"no images for {row}")
    sys.path.insert(0, str(VGGT_ROOT))
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images, original_coords = load_and_preprocess_images_square(
        [str(path) for path in image_paths], target_size=518
    )
    images = images.to(device="cuda", dtype=dtype)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=dtype):
            predictions = model(images.unsqueeze(0))

    pose_enc = predictions["pose_enc"].detach().float().squeeze(0).cpu().numpy()
    depth = predictions["depth"].detach().float().squeeze(0).cpu().numpy()
    depth_conf = predictions["depth_conf"].detach().float().squeeze(0).cpu().numpy()
    world_points = predictions["world_points"].detach().float().squeeze(0).cpu().numpy()
    world_points_conf = predictions["world_points_conf"].detach().float().squeeze(0).cpu().numpy()

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].float(), images.shape[-2:]
    )
    extrinsic = extrinsic.detach().squeeze(0).cpu().numpy()
    intrinsic = intrinsic.detach().squeeze(0).cpu().numpy()

    out_dir = OUTPUT_ROOT / _row_key(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vggt_raw.npz"
    np.savez_compressed(
        out_path,
        frame_indices=np.arange(len(image_paths), dtype=np.int64),
        pose_enc=pose_enc,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth=depth,
        depth_conf=depth_conf,
        world_points=world_points,
        world_points_conf=world_points_conf,
    )
    return {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "returncode": 0,
        "frames": len(image_paths),
        "out_path": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--only", action="append", dest="only_keys", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"device={device} dtype={dtype} checkpoint={CHECKPOINT}")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing VGGT checkpoint: {CHECKPOINT}")

    model = _load_model(device)
    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    summary_path = OUTPUT_ROOT / "adt_vggt_summary.json"
    if summary_path.is_file():
        summary_by_key = {
            item["row_key"]: item
            for item in json.loads(summary_path.read_text(encoding="utf-8"))
        }
    else:
        summary_by_key = {}

    count = 0
    for row in rows:
        key = _row_key(row)
        if args.only_keys and key not in args.only_keys:
            continue
        if args.skip_existing and summary_by_key.get(key, {}).get("returncode") == 0:
            continue
        print(f"[run] {key}", flush=True)
        started = time.time()
        result = _run_row(model, row, dtype)
        result["elapsed_s"] = time.time() - started
        summary_by_key[key] = result
        summary_path.write_text(
            json.dumps(list(summary_by_key.values()), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[done] {key} {result['elapsed_s']:.1f}s", flush=True)
        count += 1
        if args.limit and count >= args.limit:
            break

    print(json.dumps(list(summary_by_key.values()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
