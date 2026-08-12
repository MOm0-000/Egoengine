"""DA3METRIC-LARGE primary metric-depth adapter.

The official monocular metric conversion is ``focal_px * network_output / 300``.
Known camera intrinsics are transported to the processed image before applying
that conversion.  No hand or object ground truth is consumed by inference.
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
    model_source = model_root / "src"
    if not model_source.exists():
        raise FileNotFoundError(model_source)
    for required in (checkpoint / "config.json", checkpoint / "model.safetensors"):
        if not required.exists() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing DA3 checkpoint file: {required}")
    rows, start, end = load_frame_rows(run_dir, args.start_frame, args.end_frame)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "depth"
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "dry_run_da3.json"
        path.write_text(
            json.dumps(
                {
                    "dry_run": True,
                    "model_root": str(model_root),
                    "checkpoint": str(checkpoint),
                    "frame_count": len(rows),
                    "process_res": args.process_res,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DA3 requested CUDA but no device is visible")
    if str(model_source) not in sys.path:
        sys.path.insert(0, str(model_source))
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    load_started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(checkpoint)).to(device).eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started
    K = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
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
    frame_times: list[float] = []
    processed_shapes: set[tuple[int, int]] = set()
    processed_focals: list[float] = []
    try:
        for index, row in enumerate(rows):
            image_path = run_dir / row["rgb_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read {image_path}")
            height, width = image.shape[:2]
            started = time.perf_counter()
            prediction = model.inference(
                image=[str(image_path)],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            raw = np.asarray(prediction.depth[0], dtype=np.float32)
            processed_height, processed_width = raw.shape
            focal_x = K[0, 0] * processed_width / width
            focal_y = K[1, 1] * processed_height / height
            processed_focal = 0.5 * (focal_x + focal_y)
            metric_small = raw * np.float32(processed_focal / 300.0)
            metric = cv2.resize(metric_small, (width, height), interpolation=cv2.INTER_LINEAR)
            if args.device == "cuda":
                torch.cuda.synchronize()
            frame_times.append(time.perf_counter() - started)
            processed_shapes.add((processed_height, processed_width))
            processed_focals.append(float(processed_focal))
            writer.write(index, metric)
    except Exception:
        writer.abort()
        raise
    metadata = {
        "model": "Depth Anything 3 Metric Large",
        "role": "primary_metric_depth",
        "checkpoint": str(checkpoint),
        "model_root": str(model_root),
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "ground_truth_used": False,
        "metric_conversion": "processed_focal_px * network_output / 300",
        "camera_input": (
            "known K rescaled to the actual processed pixels for the official metric conversion; "
            "the metric-only network forward does not consume T"
        ),
        "process_res": args.process_res,
        "process_res_method": "upper_bound_resize",
        "processed_shapes": [list(shape) for shape in sorted(processed_shapes)],
        "processed_focal_px_range": [
            float(np.min(processed_focals)),
            float(np.max(processed_focals)),
        ],
        "model_load_s": float(model_load_s),
        "mean_inference_s_per_frame": float(np.mean(frame_times)),
        "median_inference_s_per_frame": float(np.median(frame_times)),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
        ),
        "requires_cross_model_gate": True,
    }
    return writer.finish(metadata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--process-res", type=int, default=504)
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
