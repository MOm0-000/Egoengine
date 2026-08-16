#!/usr/bin/env python3
"""Run FoundationPose model-free neural-object-field reconstruction on ADT.

This is O3 in the ablation plan. It selects reference frames from the fixed ADT
window, initializes camera poses with VGGT geometry priors, trains the
FoundationPose/BundleSDF neural object field in the isolated ``bundlesdf``
environment, and exports a metric mesh. Texture generation is intentionally
skipped because the downstream geometry gate only needs metric geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import yaml
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
STATE_PATH = RUNS_ROOT / "adt_phase_final_runs_state.json"
OUTPUT_ROOT = RUNS_ROOT / "adt_foundationpose_modelfree"
FP_ROOT = REPO_ROOT / "third_party/FoundationPose"
FP_BUNDLESDF = FP_ROOT / "bundlesdf"


def _row_key(row: dict) -> str:
    return Path(row["source_run"]).name.removesuffix("_rgbobject")


def _vggt_key(row: dict) -> str:
    return _row_key(row)


def _load_vggt_poses(row: dict) -> np.ndarray:
    raw = np.load(RUNS_ROOT / "adt_vggt" / _vggt_key(row) / "vggt_raw.npz")
    extrinsic = raw["extrinsic"]
    poses = []
    for cam_from_world in extrinsic:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :4] = cam_from_world
        poses.append(pose)
    return np.stack(poses, axis=0)


def _relative_poses(extrinsics: np.ndarray, indices: list[int]) -> np.ndarray:
    selected = extrinsics[indices]
    base_inv = np.linalg.inv(selected[0])
    return np.stack([selected[i] @ base_inv for i in range(len(selected))], axis=0)


def _select_reference_indices(masks: np.ndarray, valid: np.ndarray, count: int) -> list[int]:
    object_valid = np.zeros(len(masks), dtype=bool)
    for i, (mask, depth_valid) in enumerate(zip(masks, valid)):
        if i >= len(depth_valid):
            continue
        mask_bool = mask > 0
        object_valid[i] = bool((mask_bool & depth_valid & (depth_valid > 0)).any())
    usable = np.flatnonzero(object_valid)
    if usable.size == 0:
        raise RuntimeError("no reference frame has valid object depth")
    if usable.size <= count:
        return usable.tolist()
    chosen = np.linspace(0, usable.size - 1, count).astype(np.int64)
    return [int(usable[i]) for i in np.unique(chosen)]


def _run_row(row: dict, cfg: dict, num_ref_views: int) -> dict:
    source_run = Path(row["source_run"])
    target_run = Path(row["target_run"])
    depth_root = zarr.open(source_run / "depth" / "metric_depth.zarr", mode="r")
    depth_all = depth_root["depth_m"][:]
    valid_all = depth_root["valid"][:]

    mask_npz = np.load(target_run / "segmentation" / "object_masks.npz")
    masks_all = mask_npz["masks"]
    if masks_all.shape[1:3] != depth_all.shape[1:3]:
        masks_all = np.stack([
            cv2.resize((mask > 0).astype(np.uint8), (depth_all.shape[2], depth_all.shape[1]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
            for mask in masks_all
        ])
    else:
        masks_all = masks_all > 0

    indices = _select_reference_indices(masks_all, valid_all, num_ref_views)
    rgb_files = sorted((source_run / "frames" / "rgb").glob("*.png"))
    rgbs = [imageio.imread(str(rgb_files[i]))[..., :3] for i in indices]
    depths = [np.where(valid_all[i] & (depth_all[i] > 0) & (depth_all[i] <= 1.0), depth_all[i], 0.0) for i in indices]
    masks = [masks_all[i].astype(np.uint8) for i in indices]
    K = np.load(source_run / "calibration" / "intrinsics.npy").reshape(3, 3)

    vggt_poses = _load_vggt_poses(row)
    cam_in_ob = _relative_poses(vggt_poses, indices)

    sys.path.insert(0, str(FP_ROOT))
    sys.path.insert(0, str(FP_BUNDLESDF))
    from Utils import glcam_in_cvcam  # noqa: E402
    from bundlesdf.nerf_helpers import (  # noqa: E402
        get_optimized_poses_in_real_world,
        mesh_to_real_world,
        preprocess_data,
    )
    from bundlesdf.nerf_runner import NerfRunner  # noqa: E402
    from bundlesdf.tool import compute_scene_bounds  # noqa: E402

    out_dir = OUTPUT_ROOT / _row_key(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["bounding_box"] = np.array(cfg["bounding_box"], dtype=np.float32).reshape(2, 3)
    cfg["n_step"] = 800
    cfg["N_rand"] = 1024
    cfg["N_samples"] = 64
    cfg["N_samples_around_depth"] = 64
    cfg["finest_res"] = 256
    cfg["num_levels"] = 14
    cfg["mesh_resolution"] = 0.005
    cfg["n_train_image"] = len(indices)
    cfg["i_print"] = 200
    cfg["i_img"] = 1000000
    cfg["i_mesh"] = 1000000
    cfg["i_nerf_normals"] = 1000000
    cfg["i_save_ray"] = 1000000
    cfg["i_pose"] = 1000000
    cfg["i_weights"] = 1000000
    cfg["save_dir"] = str(out_dir)

    glcam_in_obs = cam_in_ob @ glcam_in_cvcam
    sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(
        None,
        glcam_in_obs,
        K,
        use_mask=True,
        base_dir=str(out_dir),
        rgbs=rgbs,
        depths=depths,
        masks=masks,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_eps_min_samples"],
    )
    cfg["sc_factor"] = sc_factor
    cfg["translation"] = translation

    rgbs_, depths_, masks_, _, poses = preprocess_data(
        np.asarray(rgbs),
        np.asarray(depths),
        np.asarray(masks),
        normal_maps=None,
        poses=glcam_in_obs,
        sc_factor=cfg["sc_factor"],
        translation=cfg["translation"],
    )
    nerf = NerfRunner(
        cfg,
        rgbs_,
        depths_,
        masks_,
        normal_maps=None,
        poses=poses,
        K=K,
        occ_masks=None,
        build_octree_pcd=pcd_normalized,
    )
    nerf.train()
    mesh = nerf.extract_mesh(isolevel=0, voxel_size=cfg["mesh_resolution"])
    optimized_poses, offset = get_optimized_poses_in_real_world(
        poses, nerf.models["pose_array"], cfg["sc_factor"], cfg["translation"]
    )
    mesh = mesh_to_real_world(
        mesh,
        pose_offset=offset,
        translation=nerf.cfg["translation"],
        sc_factor=nerf.cfg["sc_factor"],
    )
    mesh_path = out_dir / "model_free_mesh.obj"
    mesh.export(mesh_path)
    np.save(out_dir / "reference_indices.npy", np.asarray(indices))
    np.save(out_dir / "optimized_cam_in_ob.npy", optimized_poses)
    return {
        "row_key": _row_key(row),
        "prototype": row["prototype"],
        "window": row["window"],
        "returncode": 0,
        "reference_indices": indices,
        "mesh_path": str(mesh_path),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--num-ref-views", type=int, default=8)
    parser.add_argument("--only", action="append", dest="only_keys", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    with open(FP_BUNDLESDF / "config_ycbv.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = json.loads(STATE_PATH.read_text(encoding="utf-8"))["rows"]
    summary_path = OUTPUT_ROOT / "adt_foundationpose_modelfree_summary.json"
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
            result = _run_row(row, cfg, args.num_ref_views)
        except Exception as exc:
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
