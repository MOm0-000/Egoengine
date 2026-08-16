#!/usr/bin/env python3
"""Run Stereo Anywhere on the 10 ADT rectified left/right frame pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
STERO_ROOT = REPO_ROOT / "third_party" / "stereoanywhere"
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
STEREO_CHECKPOINT = STERO_ROOT / "weights" / "stereoanywhere_sceneflow.pth"
MONO_CHECKPOINT = REPO_ROOT / "third_party" / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitl.pth"


class Args:
    def __init__(self, gpu: int) -> None:
        self.cuda = gpu >= 0
        self.iters = 32
        self.iscale = 1.0
        self.mixed_precision = True
        self.vit_encoder = "vitl"
        self.monomodel = "DAv2"
        self.maxdisp = 192
        self.n_downsample = 2
        self.n_additional_hourglass = 0
        self.volume_channels = 8
        self.vol_downsample = 0
        self.vol_n_masks = 8
        self.use_truncate_vol = False
        self.mirror_conf_th = 0.98
        self.mirror_attenuation = 0.9
        self.use_aggregate_stereo_vol = False
        self.use_aggregate_mono_vol = False
        self.normal_gain = 10
        self.lrc_th = 1.0
        self.loadstereomodel = str(STEREO_CHECKPOINT)
        self.loadmonomodel = str(MONO_CHECKPOINT)


class StereoAnywhereWrapper(nn.Module):
    """Lightweight local copy of the official demo wrapper."""

    def __init__(self, args, stereo_model, mono_model):
        super().__init__()
        self.args = args
        self.stereo_model = stereo_model
        self.mono_model = mono_model

    def forward(self, left_img, right_img, left_mono, right_mono):
        ht, wt = left_img.shape[-2], left_img.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        left_img = F.pad(left_img, _pad, mode="replicate")
        right_img = F.pad(right_img, _pad, mode="replicate")
        left_mono = F.pad(left_mono, _pad, mode="replicate")
        right_mono = F.pad(right_mono, _pad, mode="replicate")
        pred_disps, _ = self.stereo_model(
            left_img, right_img, left_mono, right_mono, test_mode=True, iters=self.args.iters
        )
        pred_disp = -pred_disps.squeeze(1)
        hd, wd = pred_disp.shape[-2:]
        c = [_pad[2], hd - _pad[3], _pad[0], wd - _pad[1]]
        return pred_disp[..., c[0] : c[1], c[2] : c[3]]


def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def run_one_pair(
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    wrapper: nn.Module,
    mono_model,
    args: Args,
    device: torch.device,
    dtype: torch.dtype,
    intrinsics: np.ndarray,
    baseline: float,
    prefix: str,
) -> None:
    left_img = load_image(str(left_path)).to(device)
    right_img = load_image(str(right_path)).to(device)

    with torch.no_grad():
        with autocast(str(device).split(":")[0], enabled=args.mixed_precision):
            mono_input = torch.cat([left_img, right_img], 0)
            mono_depths = mono_model.infer_image(mono_input, input_size_width=518, input_size_height=518)
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            left_mono = mono_depths[0].unsqueeze(0)
            right_mono = mono_depths[1].unsqueeze(0)
        with autocast(str(device).split(":")[0], enabled=args.mixed_precision):
            disparity = wrapper(left_img, right_img, left_mono, right_mono)

    disparity_np = disparity.squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()
    fx = float(intrinsics[0, 0])
    depth = np.zeros_like(disparity_np, dtype=np.float32)
    valid = disparity_np > 0
    depth[valid] = fx * baseline / disparity_np[valid]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_stereoanywhere_disparity.npy", disparity_np)
    np.save(out_dir / f"{prefix}_stereoanywhere_depth.npy", depth)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args_cli = parser.parse_args()

    sys.path.insert(0, str(STERO_ROOT))
    from models.stereoanywhere import StereoAnywhere
    from models.depth_anything_v2 import get_depth_anything_v2

    device = torch.device(f"cuda:{args_cli.gpu}" if args_cli.gpu >= 0 and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    args = Args(args_cli.gpu)

    print(f"[stereoanywhere] loading models on {device}", flush=True)
    stereo_model = nn.DataParallel(StereoAnywhere(args))
    if device.type == "cuda":
        stereo_model = stereo_model.cuda()
    pretrain_dict = torch.load(args.loadstereomodel, map_location=device)
    pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
    stereo_model.load_state_dict(pretrain_dict, strict=True)
    stereo_model = stereo_model.module.eval().to(dtype)

    mono_model = get_depth_anything_v2(args.loadmonomodel, encoder=args.vit_encoder)
    mono_model = mono_model.to(device).to(dtype)
    mono_model.eval()

    wrapper = StereoAnywhereWrapper(args, stereo_model, mono_model)
    wrapper = wrapper.cuda().eval() if device.type == "cuda" else wrapper.eval()

    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if args_cli.max_runs:
        rows = rows[: args_cli.max_runs]

    for row in rows:
        run_dir = Path(row["run_dir"])
        left_dir = run_dir / "frames" / "rgb"
        right_dir = run_dir / "frames" / "right"
        out_dir = run_dir / "evaluation" / "stereoanywhere"
        if not left_dir.is_dir() or not right_dir.is_dir():
            print(f"[skip] missing frames {run_dir.name}", flush=True)
            continue

        stereo = json.loads((run_dir / "calibration" / "stereo.json").read_text(encoding="utf-8"))
        K = np.load(run_dir / "calibration" / "intrinsics.npy").astype(np.float64)
        baseline = float(stereo["baseline_m"])
        frame_meta = json.loads((run_dir / "frames" / "frame_index.json").read_text(encoding="utf-8"))["frames"]
        if args_cli.limit_frames:
            frame_meta = frame_meta[: args_cli.limit_frames]

        print(f"[stereoanywhere] {run_dir.name} frames={len(frame_meta)}", flush=True)
        started = time.time()
        for item in frame_meta:
            local_idx = int(item["frame_index"])
            left = left_dir / f"{local_idx:06d}.png"
            right = right_dir / f"{local_idx:06d}.png"
            prefix = f"{local_idx:06d}"
            if args_cli.skip_existing and (out_dir / f"{prefix}_stereoanywhere_depth.npy").is_file():
                continue
            run_one_pair(left, right, out_dir, wrapper, mono_model, args, device, dtype, K, baseline, prefix)
        print(f"[stereoanywhere] rc=0 elapsed={time.time() - started:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
