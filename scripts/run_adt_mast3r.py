#!/usr/bin/env python3
"""Run MASt3R sparse/dense geometry priors on the fixed ADT windows."""

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
OUTPUT_ROOT = RUNS_ROOT / "adt_mast3r"
MAST3R_ROOT = REPO_ROOT / "third_party/mast3r"
CHECKPOINT = MAST3R_ROOT / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.removesuffix("_rgbobject")


def _run_row(model, row: dict, device: str) -> dict:
    sys.path.insert(0, str(MAST3R_ROOT))
    sys.path.insert(0, str(MAST3R_ROOT / "dust3r"))
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs

    source_run = Path(row["source_run"])
    filelist = [str(path) for path in sorted((source_run / "frames" / "rgb").glob("*.png"))]
    imgs = load_images(filelist, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="swin-5", prefilter=None, symmetrize=True)

    out_dir = OUTPUT_ROOT / _row_key(row)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model,
        lr1=0.07,
        niter1=300,
        lr2=0.014,
        niter2=0,
        device=device,
        opt_depth=False,
        shared_intrinsics=True,
        matching_conf_thr=0.0,
    )

    cams2world = scene.get_im_poses().cpu().numpy()
    focals = scene.get_focals().cpu().numpy()
    def to_np(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=False)
    pts3d = [to_np(p) for p in pts3d]
    confs = [to_np(c) for c in confs]
    rgbs = [to_np(img["img"]) for img in imgs]

    # Store a compact confidence-filtered point cloud per frame plus cameras.
    sample_pts = []
    sample_colors = []
    sample_confs = []
    for pts, conf, rgb in zip(pts3d, confs, rgbs):
        pts_arr = np.asarray(pts)
        conf_arr = np.asarray(conf)
        if pts_arr.ndim == 3:
            flat_pts = pts_arr.reshape(-1, 3)
            flat_conf = conf_arr.reshape(-1)
        else:
            flat_pts = pts_arr.reshape(-1, 3)
            flat_conf = conf_arr.reshape(-1)
        valid = (flat_conf > 2.0) & np.isfinite(flat_pts).all(axis=-1)
        pts_valid = flat_pts[valid]
        if pts_valid.size == 0:
            continue
        rgb_valid = rgb.reshape(-1, 3)[valid.ravel()]
        conf_valid = flat_conf[valid]
        keep = np.linspace(0, len(pts_valid) - 1, min(20000, len(pts_valid))).astype(np.int64)
        sample_pts.append(pts_valid[keep])
        sample_colors.append(rgb_valid[keep])
        sample_confs.append(conf_valid[keep])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "mast3r_raw.npz",
        cams2world=cams2world,
        focals=focals,
        pts3d_samples=np.asarray(sample_pts, dtype=object),
        colors_samples=np.asarray(sample_colors, dtype=object),
        conf_samples=np.asarray(sample_confs, dtype=object),
    )
    return {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "returncode": 0,
        "frames": len(filelist),
        "cameras": int(len(cams2world)),
        "out_path": str(out_dir / "mast3r_raw.npz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--only", action="append", dest="only_keys", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    sys.path.insert(0, str(MAST3R_ROOT))
    sys.path.insert(0, str(MAST3R_ROOT / "dust3r"))
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(str(CHECKPOINT)).to(device).eval()
    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    summary_path = OUTPUT_ROOT / "adt_mast3r_summary.json"
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
        try:
            result = _run_row(model, row, device)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = {
                "row_key": key,
                "prototype": row["prototype"],
                "window": row["window"],
                "returncode": 1,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        result["elapsed_s"] = time.time() - started
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
