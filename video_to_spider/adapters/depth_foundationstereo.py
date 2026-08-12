"""FoundationStereo integration for an ingested, calibrated rectified run.

This module is deliberately a thin adapter around an unmodified, pinned
official checkout.  It owns input/output validation and the physical stereo
conversion ``depth_m = fx_rect * baseline_m / disparity_px``; it does not
change or reimplement the FoundationStereo network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .metric_depth_artifact import MetricDepthArtifactWriter, load_frame_rows


OFFICIAL_REPOSITORY_URL = "https://github.com/NVlabs/FoundationStereo.git"
OFFICIAL_COMMIT = "6e8806816b533e4d13ddbb95ffa907b797060a62"
BEST_GENERAL_SHA256 = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git_revision_and_tracked_status(repository: Path) -> tuple[str, str]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, status


def _normalize_uuid(value: str) -> str:
    normalized = value.strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def verify_exclusive_cuda_device(
    *, physical_gpu_index: int, expected_gpu_uuid: str
) -> dict[str, Any]:
    """Refuse to launch if the requested physical GPU already has a process."""
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu_index):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected exactly {physical_gpu_index!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the adapter requires exactly one explicitly exposed CUDA device")
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    mapping: dict[int, tuple[str, int, int]] = {}
    for row in rows:
        index, uuid, memory, utilization = [item.strip() for item in row.split(",")]
        mapping[int(index)] = (uuid, int(memory), int(utilization))
    if physical_gpu_index not in mapping:
        raise RuntimeError(f"physical GPU {physical_gpu_index} is absent")
    physical_uuid, memory_mib, utilization = mapping[physical_gpu_index]
    if _normalize_uuid(physical_uuid) != _normalize_uuid(expected_gpu_uuid):
        raise RuntimeError(
            f"physical GPU {physical_gpu_index} UUID is {physical_uuid}, not {expected_gpu_uuid}"
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    occupants = [
        row for row in processes
        if row.strip()
        and _normalize_uuid(row.split(",", maxsplit=1)[0]) == _normalize_uuid(physical_uuid)
    ]
    if occupants or memory_mib > 256 or utilization != 0:
        raise RuntimeError(
            f"GPU {physical_gpu_index} is occupied; refusing to interfere: "
            f"memory={memory_mib} MiB, utilization={utilization}%, processes={len(occupants)}"
        )
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = str(getattr(properties, "uuid", ""))
    if _normalize_uuid(torch_uuid) != _normalize_uuid(expected_gpu_uuid):
        raise RuntimeError(
            f"torch logical cuda:0 UUID {torch_uuid} does not match {expected_gpu_uuid}"
        )
    return {
        "physical_index": int(physical_gpu_index),
        "uuid": physical_uuid,
        "torch_logical_device": "cuda:0",
        "torch_uuid": torch_uuid,
        "name": properties.name,
        "launch_memory_used_mib": memory_mib,
        "launch_utilization_percent": utilization,
        "launch_compute_process_count": 0,
        "exclusive_preflight_passed": True,
    }


def disparity_to_metric_depth(
    disparity_processed_px: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    focal_x_px: float,
    baseline_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore disparity to original pixel units and apply calibrated geometry."""
    disparity = np.asarray(disparity_processed_px, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"disparity must be HxW, got {disparity.shape}")
    if not np.isfinite(focal_x_px) or focal_x_px <= 0:
        raise ValueError("focal_x_px must be positive and finite")
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("baseline_m must be positive and finite")
    processed_height, processed_width = disparity.shape
    x_processed = np.arange(processed_width, dtype=np.float32)[None, :]
    valid_small = (
        np.isfinite(disparity) & (disparity > 0) & ((x_processed - disparity) >= 0)
    )
    if (processed_width, processed_height) == (original_width, original_height):
        disparity_original = disparity.copy()
        valid = valid_small.copy()
    else:
        numerator = cv2.resize(
            np.where(valid_small, disparity, 0),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        weight = cv2.resize(
            valid_small.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        disparity_original = (
            numerator / np.maximum(weight, np.float32(1e-6))
        ) * np.float32(original_width / processed_width)
        valid = cv2.resize(
            valid_small.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool) & (weight > 1e-6)
    x_original = np.arange(original_width, dtype=np.float32)[None, :]
    valid &= (
        np.isfinite(disparity_original)
        & (disparity_original > 0)
        & ((x_original - disparity_original) >= 0)
    )
    depth = np.full((original_height, original_width), np.nan, dtype=np.float32)
    depth[valid] = np.float32(focal_x_px * baseline_m) / disparity_original[valid]
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    disparity_original[~valid] = np.nan
    return depth, disparity_original, valid


def apply_common_valid_domain(
    depth_m: np.ndarray,
    disparity_px: np.ndarray,
    valid: np.ndarray,
    common_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect stereo predictions with the camera toolkit's observable domain."""
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    disparity = np.asarray(disparity_px, dtype=np.float32).copy()
    geometric = np.asarray(valid, dtype=bool).copy()
    common = np.asarray(common_valid, dtype=bool)
    if depth.shape != disparity.shape or depth.shape != geometric.shape:
        raise ValueError("depth, disparity, and valid must have one shared image shape")
    if common.shape != geometric.shape:
        raise ValueError(
            f"stereo common-valid mask shape {common.shape} differs from {geometric.shape}"
        )
    geometric &= common
    depth[~geometric] = np.nan
    disparity[~geometric] = np.nan
    return depth, disparity, geometric


def _load_pair(
    run_dir: Path, row: dict[str, Any], scale: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if "right_rgb_path" not in row:
        raise ValueError("run is not stereo: frame row lacks right_rgb_path")
    left = cv2.imread(str(run_dir / row["rgb_path"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(run_dir / row["right_rgb_path"]), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape != right.shape:
        raise RuntimeError(f"unreadable or unequal rectified pair at frame {row['frame_index']}")
    height, width = left.shape[:2]
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    if scale < 1:
        target = (max(32, round(width * scale)), max(32, round(height * scale)))
        left = cv2.resize(left, target, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, target, interpolation=cv2.INTER_AREA)
    return (
        cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
        (width, height),
    )


class OfficialFoundationStereoPredictor:
    """Call only the public model API from the pinned official checkout."""

    def __init__(
        self, *, repository: Path, checkpoint: Path, config: Path,
        valid_iters: int, low_memory: bool,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        sys.dont_write_bytecode = True
        sys.path.insert(0, str(repository))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        cfg = OmegaConf.load(config)
        cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = int(valid_iters)
        cfg["low_memory"] = bool(low_memory)
        self.torch = torch
        self.InputPadder = InputPadder
        self.valid_iters = int(valid_iters)
        self.low_memory = bool(low_memory)
        model = FoundationStereo(cfg)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        self.model = model.to(torch.device("cuda:0")).eval()
        self.checkpoint_training = {
            "global_step": int(payload["global_step"]),
            "epoch": int(payload["epoch"]),
        }

    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        left = torch.as_tensor(np.ascontiguousarray(left_rgb), device="cuda:0").float()
        right = torch.as_tensor(np.ascontiguousarray(right_rgb), device="cuda:0").float()
        left = left[None].permute(0, 3, 1, 2)
        right = right[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            disparity = self.model.forward(
                left, right, iters=self.valid_iters, test_mode=True,
                low_memory=self.low_memory,
            )
        return padder.unpad(disparity.float()).cpu().numpy()[0, 0].astype(np.float32)


def run(
    args: argparse.Namespace,
    *,
    predictor_factory: Callable[..., Any] = OfficialFoundationStereoPredictor,
) -> Path:
    import torch

    run_dir = args.run_dir.resolve()
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    for required in (
        repository / ".git", repository / "core/foundation_stereo.py",
        checkpoint, config, run_dir / "calibration/stereo.json",
        run_dir / "calibration/stereo_common_valid.npy",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    revision, tracked_status = _git_revision_and_tracked_status(repository)
    if revision != OFFICIAL_COMMIT:
        raise RuntimeError(f"FoundationStereo revision {revision} != pinned {OFFICIAL_COMMIT}")
    if tracked_status:
        raise RuntimeError(
            "official FoundationStereo tracked files differ from the pinned checkout; refusing to run"
        )
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != BEST_GENERAL_SHA256:
        raise RuntimeError(
            f"FoundationStereo checkpoint SHA256 {checkpoint_hash} != audited {BEST_GENERAL_SHA256}"
        )
    cuda = verify_exclusive_cuda_device(
        physical_gpu_index=args.physical_gpu_index,
        expected_gpu_uuid=args.gpu_uuid,
    )
    stereo = json.loads((run_dir / "calibration/stereo.json").read_text(encoding="utf-8"))
    if not stereo.get("accepted", False):
        raise RuntimeError("stereo rectification gate is not accepted")
    K = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    K_right_path = run_dir / "calibration/intrinsics_right.npy"
    if not K_right_path.is_file():
        raise FileNotFoundError(K_right_path)
    K_right = np.load(K_right_path).astype(np.float64)
    stereo_K = np.asarray(stereo["K_rect_left"], dtype=np.float64)
    stereo_K_right = np.asarray(stereo["K_rect_right"], dtype=np.float64)
    if K.shape != (3, 3) or not np.allclose(K, stereo_K, atol=1e-5, rtol=1e-6):
        raise RuntimeError("left rectified intrinsics disagree across calibration artifacts")
    if (
        K_right.shape != (3, 3)
        or not np.allclose(K_right, stereo_K_right, atol=1e-5, rtol=1e-6)
    ):
        raise RuntimeError("right rectified intrinsics disagree across calibration artifacts")
    if not np.allclose(K, K_right, atol=1e-5, rtol=1e-6):
        raise RuntimeError(
            "FoundationStereo raw-pixel disparity requires a shared rectified K; "
            "re-rectify both views to one common pinhole projection with the official camera toolkit"
        )
    baseline_m = float(stereo["baseline_m"])
    common_valid = np.asarray(
        np.load(run_dir / "calibration/stereo_common_valid.npy"), dtype=bool
    )
    rows, start, end = load_frame_rows(run_dir, args.start_frame, args.end_frame)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "depth"
    predictor = predictor_factory(
        repository=repository,
        checkpoint=checkpoint,
        config=config,
        valid_iters=args.valid_iters,
        low_memory=args.low_memory,
    )
    writer = MetricDepthArtifactWriter(
        run_dir=run_dir,
        rows=rows,
        start=start,
        end=end,
        output_dir=output_dir,
        zarr_name="metric_depth.zarr",
        video_name="metric_depth.mp4",
        overwrite=args.overwrite,
        max_visual_depth_m=args.max_visual_depth,
    )
    disparity_store = writer.group.create_dataset(
        "disparity_px",
        shape=(len(rows), writer.height, writer.width),
        dtype="f4",
        chunks=writer.depths.chunks,
        compressor=writer.depths.compressor,
    )
    frame_times: list[float] = []
    processed_shapes: set[tuple[int, int]] = set()
    try:
        for offset, row in enumerate(rows):
            left_rgb, right_rgb, original_wh = _load_pair(run_dir, row, args.scale)
            if original_wh != (writer.width, writer.height):
                raise RuntimeError("stereo input resolution changes within the run")
            started = time.perf_counter()
            disparity_small = predictor(left_rgb, right_rgb)
            torch.cuda.synchronize()
            frame_times.append(time.perf_counter() - started)
            depth, disparity, valid = disparity_to_metric_depth(
                disparity_small,
                original_width=writer.width,
                original_height=writer.height,
                focal_x_px=float(K[0, 0]),
                baseline_m=baseline_m,
            )
            depth, disparity, valid = apply_common_valid_domain(
                depth, disparity, valid, common_valid,
            )
            processed_shapes.add(tuple(disparity_small.shape))
            disparity_store[offset] = disparity
            writer.write(offset, depth, valid_mask=valid)
    except Exception:
        writer.abort()
        raise
    metadata = {
        "model": "FoundationStereo",
        "role": "primary_metric_depth",
        "model_variant": "23-51-11 (ViT-Large best-general)",
        "official_repository": OFFICIAL_REPOSITORY_URL,
        "repository_path": str(repository),
        "repository_commit": revision,
        "official_tracked_files_clean": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(config),
        "license": "NVIDIA Source Code License (research/non-commercial only)",
        "device": cuda,
        "ground_truth_used": False,
        "scale_or_shift_alignment_applied": False,
        "metric_conversion": (
            "raw depth_m = original_K_rect.fx * baseline_m / original-grid disparity_px"
        ),
        "input_geometry": "pre-undistorted synchronized rectified stereo; left reference",
        "K_rect": K.tolist(),
        "K_rect_right": K_right.tolist(),
        "shared_rectified_intrinsics_verified": True,
        "common_valid_mask": "calibration/stereo_common_valid.npy",
        "common_valid_pixel_ratio": float(common_valid.mean()),
        "baseline_m": baseline_m,
        "network_input_scale": float(args.scale),
        "processed_disparity_shapes": [list(shape) for shape in sorted(processed_shapes)],
        "valid_iters": int(args.valid_iters),
        "low_memory": bool(args.low_memory),
        "mean_inference_s_per_pair": float(np.mean(frame_times)),
        "median_inference_s_per_pair": float(np.median(frame_times)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_training": getattr(predictor, "checkpoint_training", None),
        "requires_cross_model_gate": False,
        "requires_calibrated_stereo_integrity_gate": True,
    }
    return writer.finish(metadata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--max-visual-depth", type=float, default=5.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    print(run(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
