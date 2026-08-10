"""Depth Anything V2 indoor metric adapter with original-pixel artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..video import FFmpegVideoWriter, VIDEO_ENCODING


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _load_optional_masks(run_dir: Path, rows: list[dict]) -> tuple[np.ndarray | None, np.ndarray | None]:
    outputs = []
    wanted = np.asarray([row["source_frame_index"] for row in rows])
    for name in ("object_masks.npz", "hand_masks.npz"):
        path = run_dir / "segmentation" / name
        if not path.exists():
            outputs.append(None)
            continue
        with np.load(path, allow_pickle=False) as artifact:
            # NpzFile.__getitem__ decompresses the complete member on every
            # access.  Load masks once before selecting rows; indexing
            # artifact["masks"] inside the comprehension made a 54-frame 1080p
            # run decompress the same ~112 MiB array 54 times per artifact.
            frame_indices = artifact["frame_indices"]
            masks = artifact["masks"]
            lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
            if not all(int(frame) in lookup for frame in wanted):
                outputs.append(None)
            else:
                selected = np.asarray([lookup[int(frame)] for frame in wanted], dtype=np.int64)
                outputs.append(masks[selected].astype(bool, copy=False))
    return outputs[0], outputs[1]


def _masked_stats(depth: np.ndarray, masks: np.ndarray | None) -> list[dict[str, float] | None]:
    if masks is None:
        return [None] * depth.shape[0]
    result = []
    for frame_depth, mask in zip(depth, masks):
        values = frame_depth[mask & np.isfinite(frame_depth) & (frame_depth > 0)]
        result.append(None if values.size == 0 else {
            "count": int(values.size), "median_m": float(np.median(values)),
            "p10_m": float(np.percentile(values, 10)), "p90_m": float(np.percentile(values, 90)),
        })
    return result


def _background_warp_residuals(
    depth: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray,
    object_masks: np.ndarray | None, hand_masks: np.ndarray | None, stride: int = 8,
) -> list[dict[str, float]]:
    height, width = depth.shape[1:]
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
    rays = (np.linalg.inv(K) @ pixels.T).T
    metrics = []
    for index in range(depth.shape[0] - 1):
        z = depth[index, ::stride, ::stride].reshape(-1)
        source_valid = np.isfinite(z) & (z > 0)
        if object_masks is not None:
            source_valid &= ~object_masks[index, ::stride, ::stride].reshape(-1)
        if hand_masks is not None:
            source_valid &= ~hand_masks[index, ::stride, ::stride].reshape(-1)
        points_camera = rays[source_valid] * z[source_valid, None]
        points_h = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1)
        T_next_world = np.linalg.inv(T_world_camera[index + 1])
        points_next = (T_next_world @ T_world_camera[index] @ points_h.T).T[:, :3]
        projected = (K @ points_next.T).T
        uv = projected[:, :2] / projected[:, 2:3]
        px = np.rint(uv[:, 0]).astype(int)
        py = np.rint(uv[:, 1]).astype(int)
        valid = (points_next[:, 2] > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        target = np.full(points_next.shape[0], np.nan, dtype=np.float32)
        target[valid] = depth[index + 1, py[valid], px[valid]]
        valid &= np.isfinite(target) & (target > 0)
        if object_masks is not None:
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] &= ~object_masks[index + 1, py[valid_indices], px[valid_indices]]
        if hand_masks is not None:
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] &= ~hand_masks[index + 1, py[valid_indices], px[valid_indices]]
        residual = np.abs(points_next[valid, 2] - target[valid])
        metrics.append({
            "source_frame_offset": index, "target_frame_offset": index + 1,
            "valid_samples": int(residual.size),
            "median_abs_m": float(np.median(residual)) if residual.size else -1.0,
            "p95_abs_m": float(np.percentile(residual, 95)) if residual.size else -1.0,
        })
    return metrics


def run(args: argparse.Namespace) -> Path:
    metric_root = Path(__file__).resolve().parents[2] / "third_party/Depth-Anything-V2/metric_depth"
    sys.path.insert(0, str(metric_root))
    import torch
    import zarr
    from numcodecs import Blosc
    from depth_anything_v2.dpt import DepthAnythingV2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("metric depth requested CUDA but no device is visible")
    run_dir = args.run_dir.resolve()
    frame_rows = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))["frames"]
    start, end = args.start_frame, args.end_frame if args.end_frame is not None else len(frame_rows)
    if start < 0 or end <= start or end > len(frame_rows):
        raise ValueError(f"invalid adapter interval [{start}, {end})")
    rows = frame_rows[start:end]
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "depth"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"depth output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size == 0:
        raise ValueError(f"refusing zero-byte checkpoint: {checkpoint}")
    if args.dry_run:
        metadata_path.write_text(json.dumps({
            "dry_run": True, "checkpoint": str(checkpoint), "encoder": args.encoder,
            "frame_count": len(rows), "max_depth_m": args.max_depth,
        }, indent=2) + "\n")
        return metadata_path
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], "max_depth": args.max_depth})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(torch.device(args.device)).eval()
    depths = []
    visual_frames = []
    for row in rows:
        image = cv2.imread(str(run_dir / row["rgb_path"]))
        if image is None:
            raise RuntimeError(f"cannot read {row['rgb_path']}")
        prediction = model.infer_image(image, args.input_size).astype(np.float32)
        if prediction.shape != image.shape[:2]:
            raise RuntimeError(f"depth did not return to original pixels: {prediction.shape} vs {image.shape[:2]}")
        depths.append(prediction)
        normalized = np.clip(prediction / args.max_depth, 0, 1)
        visual_frames.append(cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    del model, state
    if args.device == "cuda":
        torch.cuda.empty_cache()
    depth = np.stack(depths)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth)
    gy, gx = np.gradient(depth, axis=(1, 2))
    uncertainty = np.clip(np.sqrt(gx * gx + gy * gy) / np.maximum(depth, 1e-3), 0, 1).astype(np.float16)
    zarr_path = output_dir / "metric_depth.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunks = (1, min(270, depth.shape[1]), min(480, depth.shape[2]))
    root.create_dataset("depth_m", data=depth, chunks=chunks, compressor=compressor)
    root.create_dataset("valid", data=valid, chunks=chunks, compressor=compressor)
    root.create_dataset("uncertainty_proxy", data=uncertainty, chunks=chunks, compressor=compressor)
    root.create_dataset("frame_indices", data=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64))
    root.create_dataset("timestamps_s", data=np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64))
    root.attrs.update({"schema_version": "1.0", "units": "meter", "pixel_mapping": "original RGB resolution"})
    object_masks, hand_masks = _load_optional_masks(run_dir, rows)
    K = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(run_dir / "calibration/T_world_camera.npy")[start:end].astype(np.float64)
    warp_metrics = _background_warp_residuals(depth, K, T_world_camera, object_masks, hand_masks)
    source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
    overlay_path = output_dir / "metric_depth.mp4"
    height, width = visual_frames[0].shape[:2]
    writer = FFmpegVideoWriter(
        overlay_path, float(source["video"]["fps"]), (width, height), overwrite=True,
    )
    try:
        for frame in visual_frames:
            writer.write(frame)
    finally:
        writer.release()
    metadata = {
        "schema_version": "1.0", "model": "Depth Anything V2 metric Hypersim",
        "checkpoint": str(checkpoint), "encoder": args.encoder, "device": args.device,
        "video_encoding": VIDEO_ENCODING,
        "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "input_size": args.input_size, "max_depth_m": args.max_depth,
        "original_resolution": [int(depth.shape[2]), int(depth.shape[1])],
        "valid_ratio": float(np.mean(valid)), "invalid_ratio": float(1 - np.mean(valid)),
        "min_depth_m": float(np.min(depth[valid])), "median_depth_m": float(np.median(depth[valid])),
        "max_depth_observed_m": float(np.max(depth[valid])),
        "object_mask_stats": _masked_stats(depth, object_masks),
        "hand_mask_stats": _masked_stats(depth, hand_masks),
        "background_warp_residuals": warp_metrics,
        "outputs": ["metric_depth.zarr", "metric_depth.mp4"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    print(run(build_parser().parse_args()))
