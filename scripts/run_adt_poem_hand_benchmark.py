#!/usr/bin/env python3
"""H4 POEM-v2 multi-view hand ablation on the 10 fixed ADT stereo clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POEM_ROOT = REPO_ROOT / "third_party" / "POEM-v2"
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
OUTPUT_SUMMARY = RUNS_ROOT / "adt_poem_hand_benchmark_summary.json"

REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
MIN_REPROJECTION_ERROR_PX = 3.0
MIN_JOINT_DEPTH_M = 0.05
MAX_JOINT_DEPTH_M = 3.0
MIN_REQUIRED_JOINT_RATE = 0.70
MIN_REQUIRED_FRAME_RATE = 0.60


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    projected = np.einsum("ij,...j->...i", K, points)
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[..., :2] / projected[..., 2:3]


def _load_hamer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _bbox_from_joints(uv: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float] | None:
    uv = np.asarray(uv, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    sel = valid & np.isfinite(uv).all(axis=-1)
    if not sel.any():
        return None
    pts = uv[sel]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max((x1 - x0) / 2.0, (y1 - y0) / 2.0, 20.0) * 1.35
    return float(cx - half), float(cy - half), float(cx + half), float(cy + half)


def _make_poem_batch(
    img_list: list[np.ndarray],
    bbox_list: list[tuple[float, float, float, float] | None],
    req_flip: bool,
    camera_name_list: list[str],
    cam_intr_map: dict[str, np.ndarray],
    cam_extr_map: dict[str, np.ndarray],
    img_size: tuple[int, int],
    output_size: tuple[int, int],
    device: torch.device,
):
    # Use the official POEM batch formatting code.
    return infer.format_batch(
        img_list=img_list,
        bbox_list=bbox_list,
        req_flip=req_flip,
        camera_name_list=camera_name_list,
        cam_intr_map=cam_intr_map,
        cam_extr_map=cam_extr_map,
        img_size=img_size,
        output_size=output_size,
        device=device,
    )


def _track_one_run(
    run_dir: Path,
    model,
    cfg,
    device: torch.device,
) -> dict[str, Any]:
    root = run_dir
    left_hamer = _load_hamer(root / "hands_hamer" / "hamer_raw.npz")
    right_hamer = _load_hamer(root / "hands_right_hamer" / "hamer_raw.npz")
    if not np.array_equal(left_hamer["frame_indices"], right_hamer["frame_indices"]):
        raise ValueError("left/right HaMeR timelines differ")

    stereo = json.loads((root / "calibration" / "stereo.json").read_text(encoding="utf-8"))
    baseline = float(stereo["baseline_m"])
    K_left = np.load(root / "calibration" / "intrinsics.npy").astype(np.float64)
    K_right = np.load(root / "calibration" / "intrinsics_right.npy").astype(np.float64)
    left_abs = left_hamer["joints_camera_rootrel"] + left_hamer["translation_camera"][:, :, None]
    right_abs = right_hamer["joints_camera_rootrel"] + right_hamer["translation_camera"][:, :, None]
    left_uv = _project(K_left, left_abs)
    right_uv = _project(K_right, right_abs)

    left_extr = np.eye(4, dtype=np.float64)
    right_extr = np.eye(4, dtype=np.float64)
    right_extr[0, 3] = -baseline
    cam_intr_map = {"left": K_left, "right": K_right}
    cam_extr_map = {"left": left_extr, "right": right_extr}
    camera_name_list = ["left", "right"]
    img_size = (512, 512)
    output_size = tuple(int(v) for v in cfg.DATA_PRESET.IMAGE_SIZE)

    timestamps = np.asarray(left_hamer["timestamps_s"], dtype=np.float64)
    frame_indices = np.asarray(left_hamer["frame_indices"], dtype=np.int64)
    T = len(frame_indices)

    pred_joints = np.full((T, 2, 21, 3), np.nan, dtype=np.float64)
    pred_valid = np.zeros((T, 2, 21), dtype=bool)
    left_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    right_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    vertical_error = np.full((T, 2, 21), np.nan, dtype=np.float64)
    num_views = np.zeros((T, 2), dtype=np.int64)

    for t in range(T):
        left_img = cv2.imread(str(root / "frames" / "rgb" / f"{t:06d}.png"), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(root / "frames" / "right" / f"{t:06d}.png"), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            continue
        if left_img.ndim == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        if right_img.ndim == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        img_list = [left_img, right_img]

        for hand_idx in (0, 1):
            left_bbox = _bbox_from_joints(left_uv[t, hand_idx], left_hamer["valid"][t, hand_idx])
            right_bbox = _bbox_from_joints(right_uv[t, hand_idx], right_hamer["valid"][t, hand_idx])
            if left_bbox is None or right_bbox is None:
                continue
            bbox_list = [left_bbox, right_bbox]
            try:
                batch = _make_poem_batch(
                    img_list=img_list,
                    bbox_list=bbox_list,
                    req_flip=(hand_idx == 0),
                    camera_name_list=camera_name_list,
                    cam_intr_map=cam_intr_map,
                    cam_extr_map=cam_extr_map,
                    img_size=img_size,
                    output_size=output_size,
                    device=device,
                )
                if batch is None:
                    continue
                with torch.no_grad():
                    pred = model(batch, 0, "inference", epoch_idx=0)
                payload = infer.extract_pred(pred, batch, req_flip=(hand_idx == 0), cam_extr_map=cam_extr_map)
            except Exception:
                continue
            joints = np.asarray(payload["joints"], dtype=np.float64)
            if joints.shape != (21, 3):
                continue
            pred_joints[t, hand_idx] = joints
            num_views[t, hand_idx] = 2
            proj_left = _project(K_left, joints)
            proj_right = _project(K_right, joints - np.asarray([baseline, 0.0, 0.0]))
            obs_left = left_uv[t, hand_idx]
            obs_right = right_uv[t, hand_idx]
            obs_valid = (
                left_hamer["valid"][t, hand_idx]
                & right_hamer["valid"][t, hand_idx]
                & np.isfinite(obs_left).all(axis=-1)
                & np.isfinite(obs_right).all(axis=-1)
            )
            le = np.linalg.norm(proj_left - obs_left, axis=-1)
            re = np.linalg.norm(proj_right - obs_right, axis=-1)
            ve = np.abs(proj_left[..., 1] - proj_right[..., 1])
            left_error[t, hand_idx] = le
            right_error[t, hand_idx] = re
            vertical_error[t, hand_idx] = ve
            joint_ok = (
                obs_valid
                & np.isfinite(joints).all(axis=-1)
                & (joints[..., 2] >= MIN_JOINT_DEPTH_M)
                & (joints[..., 2] <= MAX_JOINT_DEPTH_M)
                & (le <= MIN_REPROJECTION_ERROR_PX)
                & (re <= MIN_REPROJECTION_ERROR_PX)
                & (ve <= MIN_REPROJECTION_ERROR_PX)
            )
            pred_valid[t, hand_idx] = joint_ok

    frame_valid = pred_valid[:, :, REQUIRED].all(axis=-1)
    joint_rate = pred_valid.mean(axis=(0, 2))
    frame_rate = frame_valid.mean(axis=0)
    accepted = (joint_rate >= MIN_REQUIRED_JOINT_RATE) & (frame_rate >= MIN_REQUIRED_FRAME_RATE)

    output_dir = root / "hands_poem"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "poem_raw.npz",
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        side=np.asarray([[0, 1]] * T, dtype=np.int8),
        valid=pred_valid,
        joints_world_m=pred_joints,
        left_reprojection_error_px=left_error,
        right_reprojection_error_px=right_error,
        vertical_error_px=vertical_error,
        num_views=num_views,
    )
    metrics = {
        "accepted_any_hand": bool(accepted.any()),
        "accepted_left": bool(accepted[0]),
        "accepted_right": bool(accepted[1]),
        "joint_valid_rate_left": float(joint_rate[0]),
        "joint_valid_rate_right": float(joint_rate[1]),
        "required_frame_valid_rate_left": float(frame_rate[0]),
        "required_frame_valid_rate_right": float(frame_rate[1]),
    }
    (output_dir / "poem_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"run_dir": str(root), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    os.chdir(POEM_ROOT)
    sys.path.insert(0, str(POEM_ROOT))
    global infer
    import tool.infer_hand as infer

    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1

    import lib.models
    from lib.utils import builder
    from lib.utils.config import get_config

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    cfg = get_config(str(POEM_ROOT / "config" / "release" / "eval_single.yaml"))
    arg_ns = argparse.Namespace(log_freq=20, cfg="config/release/eval_single.yaml", model="medium")
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=None, log_freq=arg_ns.log_freq)
    checkpoint = torch.load(str(POEM_ROOT / "checkpoints" / "medium.pth.tar"), map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"[poem] checkpoint missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device)
    model.eval()

    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if args.max_runs:
        rows = rows[: args.max_runs]
    records = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        print(f"[poem] {run_dir.name}", flush=True)
        try:
            record = _track_one_run(run_dir, model, cfg, device)
        except Exception as exc:
            record = {"run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"}
        record.update({"sequence": row["sequence"], "prototype": row["prototype"], "window": row["window"]})
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    OUTPUT_SUMMARY.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUTPUT_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
