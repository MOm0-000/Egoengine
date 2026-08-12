"""UniDepthV2 independent metric-depth cross-check adapter.

This artifact is reject-only: it is never blended into the DA3 primary depth.
Known pinhole intrinsics are passed directly to ``model.infer``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .metric_depth_artifact import MetricDepthArtifactWriter, load_frame_rows


def run(args: argparse.Namespace) -> Path:
    import torch

    run_dir = args.run_dir.resolve()
    model_root = args.model_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not model_root.exists():
        raise FileNotFoundError(model_root)
    for required in (checkpoint / "config.json", checkpoint / "model.safetensors"):
        if not required.exists() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing UniDepth checkpoint file: {required}")
    rows, start, end = load_frame_rows(run_dir, args.start_frame, args.end_frame)
    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_dir / "depth_crosscheck"
    )
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "dry_run_unidepth.json"
        path.write_text(
            json.dumps(
                {
                    "dry_run": True,
                    "model_root": str(model_root),
                    "checkpoint": str(checkpoint),
                    "frame_count": len(rows),
                    "resolution_level": args.resolution_level,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("UniDepth requested CUDA but no device is visible")
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from unidepth.models import UniDepthV2

    device = torch.device(args.device)
    load_started = time.perf_counter()
    model = UniDepthV2.from_pretrained(str(checkpoint)).to(device).eval()
    model.resolution_level = args.resolution_level
    if args.device == "cuda":
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started
    K = torch.from_numpy(np.load(run_dir / "calibration/intrinsics.npy")).float()
    writer = MetricDepthArtifactWriter(
        run_dir=run_dir,
        rows=rows,
        start=start,
        end=end,
        output_dir=output_dir,
        zarr_name="unidepth_depth.zarr",
        video_name="unidepth_depth.mp4",
        overwrite=args.overwrite,
        max_visual_depth_m=args.max_visual_depth,
    )
    frame_times: list[float] = []
    processed_shapes: set[tuple[int, int]] = set()
    try:
        for index, row in enumerate(rows):
            image_path = run_dir / row["rgb_path"]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot read {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
            started = time.perf_counter()
            prediction = model.infer(tensor, K)
            if args.device == "cuda":
                torch.cuda.synchronize()
            frame_times.append(time.perf_counter() - started)
            depth = prediction["depth"][0, 0].float().cpu().numpy()
            feature = prediction.get("depth_features")
            if isinstance(feature, torch.Tensor):
                processed_shapes.add(tuple(int(value) for value in feature.shape[-2:]))
            writer.write(index, depth)
    except Exception:
        writer.abort()
        raise
    metadata = {
        "model": "UniDepthV2 ViT-L/14",
        "role": "independent_metric_depth_crosscheck",
        "checkpoint": str(checkpoint),
        "model_root": str(model_root),
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "ground_truth_used": False,
        "camera_input": "known 3x3 pinhole K passed directly to model.infer",
        "resolution_level": args.resolution_level,
        "processed_feature_shapes": [list(shape) for shape in sorted(processed_shapes)],
        "model_load_s": float(model_load_s),
        "mean_inference_s_per_frame": float(np.mean(frame_times)),
        "median_inference_s_per_frame": float(np.median(frame_times)),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
        ),
        "may_rescale_primary": False,
    }
    return writer.finish(metadata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--max-visual-depth", type=float, default=5.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    print(run(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
