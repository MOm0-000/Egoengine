"""Shared original-pixel artifact writer for metric-depth model adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..video import FFmpegVideoWriter, VIDEO_ENCODING
from .depth_anything import _background_warp_residuals, _load_optional_masks


def _frame_mask_stats(
    depth: np.ndarray, valid: np.ndarray, mask: np.ndarray | None
) -> dict[str, float | int] | None:
    if mask is None:
        return None
    values = depth[mask & valid]
    if not values.size:
        return None
    return {
        "count": int(values.size),
        "median_m": float(np.median(values)),
        "p10_m": float(np.percentile(values, 10)),
        "p90_m": float(np.percentile(values, 90)),
    }


def load_frame_rows(
    run_dir: Path, start_frame: int, end_frame: int | None
) -> tuple[list[dict[str, Any]], int, int]:
    all_rows = json.loads(
        (run_dir / "frames/frame_index.json").read_text(encoding="utf-8")
    )["frames"]
    start = int(start_frame)
    end = int(end_frame) if end_frame is not None else len(all_rows)
    if start < 0 or end <= start or end > len(all_rows):
        raise ValueError(f"invalid adapter interval [{start}, {end})")
    return all_rows[start:end], start, end


class MetricDepthArtifactWriter:
    """Incrementally store full-resolution depth without retaining a clip in RAM."""

    def __init__(
        self,
        *,
        run_dir: Path,
        rows: list[dict[str, Any]],
        start: int,
        end: int,
        output_dir: Path,
        zarr_name: str,
        video_name: str,
        overwrite: bool,
        max_visual_depth_m: float,
    ) -> None:
        import zarr
        from numcodecs import Blosc

        self.run_dir = run_dir
        self.rows = rows
        self.output_dir = output_dir
        self.metadata_path = output_dir / "metadata.json"
        self.zarr_path = output_dir / zarr_name
        self.video_path = output_dir / video_name
        if (self.metadata_path.exists() or self.zarr_path.exists()) and not overwrite:
            raise FileExistsError(f"depth output exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        first = cv2.imread(str(run_dir / rows[0]["rgb_path"]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"cannot read {rows[0]['rgb_path']}")
        self.height, self.width = first.shape[:2]
        self.max_visual_depth_m = float(max_visual_depth_m)
        self.object_masks, self.hand_masks = _load_optional_masks(run_dir, rows)
        self.K = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
        self.T_world_camera = np.load(run_dir / "calibration/T_world_camera.npy")[start:end].astype(
            np.float64
        )
        source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
        self.video = FFmpegVideoWriter(
            self.video_path,
            float(source["video"]["fps"]),
            (self.width, self.height),
            overwrite=True,
        )
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1, min(270, self.height), min(480, self.width))
        self.group = zarr.open_group(str(self.zarr_path), mode="w")
        shape = (len(rows), self.height, self.width)
        self.depths = self.group.create_dataset(
            "depth_m", shape=shape, dtype="f4", chunks=chunks, compressor=compressor
        )
        self.valid = self.group.create_dataset(
            "valid", shape=shape, dtype="bool", chunks=chunks, compressor=compressor
        )
        self.uncertainty = self.group.create_dataset(
            "uncertainty_proxy", shape=shape, dtype="f2", chunks=chunks, compressor=compressor
        )
        self.group.create_dataset(
            "frame_indices",
            data=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64),
        )
        self.group.create_dataset(
            "timestamps_s", data=np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64)
        )
        self.group.attrs.update(
            {"schema_version": "1.0", "units": "meter", "pixel_mapping": "original RGB resolution"}
        )
        self.frame_medians: list[float] = []
        self.frame_minima: list[float] = []
        self.frame_maxima: list[float] = []
        self.valid_counts: list[int] = []
        self.object_stats: list[dict[str, float | int] | None] = []
        self.hand_stats: list[dict[str, float | int] | None] = []
        self.warp_metrics: list[dict[str, float]] = []
        self.previous_depth: np.ndarray | None = None
        self.previous_index: int | None = None

    def write(
        self,
        index: int,
        depth_m: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        uncertainty_proxy: np.ndarray | None = None,
    ) -> None:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (self.height, self.width):
            raise RuntimeError(
                f"depth did not return to original pixels: {depth.shape} "
                f"vs {(self.height, self.width)}"
            )
        physical = np.isfinite(depth) & (depth > 0)
        if valid_mask is None:
            valid = physical
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != depth.shape:
                raise RuntimeError(
                    f"valid mask shape differs from depth: {valid.shape} vs {depth.shape}"
                )
            if np.any(valid & ~physical):
                raise RuntimeError(
                    f"metric-depth frame {index} labels non-positive/non-finite depth valid"
                )
        if not valid.any():
            raise RuntimeError(f"metric-depth frame {index} has no positive finite pixels")
        if uncertainty_proxy is None:
            filled_depth = np.where(valid, depth, np.median(depth[valid]))
            gy, gx = np.gradient(filled_depth)
            uncertainty = np.clip(
                np.sqrt(gx * gx + gy * gy) / np.maximum(filled_depth, 1e-3), 0, 1
            ).astype(np.float16)
            uncertainty[~valid] = np.float16(1.0)
        else:
            uncertainty = np.asarray(uncertainty_proxy, dtype=np.float16)
            if uncertainty.shape != depth.shape or not np.isfinite(uncertainty).all():
                raise RuntimeError(
                    "uncertainty proxy must be finite and match the original-pixel depth shape"
                )
        self.depths[index] = depth
        self.valid[index] = valid
        self.uncertainty[index] = uncertainty
        values = depth[valid]
        self.frame_medians.append(float(np.median(values)))
        self.frame_minima.append(float(np.min(values)))
        self.frame_maxima.append(float(np.max(values)))
        self.valid_counts.append(int(valid.sum()))
        self.object_stats.append(
            _frame_mask_stats(
                depth, valid, None if self.object_masks is None else self.object_masks[index]
            )
        )
        self.hand_stats.append(
            _frame_mask_stats(
                depth, valid, None if self.hand_masks is None else self.hand_masks[index]
            )
        )

        normalized = np.where(
            valid, np.clip(depth / self.max_visual_depth_m, 0, 1), 0
        )
        visualization = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        self.video.write(visualization)

        if self.previous_depth is not None and self.previous_index is not None:
            object_pair = None
            hand_pair = None
            if self.object_masks is not None:
                object_pair = self.object_masks[[self.previous_index, index]]
            if self.hand_masks is not None:
                hand_pair = self.hand_masks[[self.previous_index, index]]
            pair_metrics = _background_warp_residuals(
                np.stack([self.previous_depth, depth]),
                self.K,
                self.T_world_camera[[self.previous_index, index]],
                object_pair,
                hand_pair,
            )
            if pair_metrics:
                pair = dict(pair_metrics[0])
                pair["source_frame_offset"] = self.previous_index
                pair["target_frame_offset"] = index
                self.warp_metrics.append(pair)
        self.previous_depth = depth
        self.previous_index = index

    def finish(self, metadata: dict[str, Any]) -> Path:
        self.video.release()
        valid_ratio = sum(self.valid_counts) / max(1, len(self.rows) * self.height * self.width)
        payload = {
            "schema_version": "1.0",
            **metadata,
            "video_encoding": VIDEO_ENCODING,
            "frame_count": len(self.rows),
            "original_resolution": [self.width, self.height],
            "valid_ratio": float(valid_ratio),
            "invalid_ratio": float(1.0 - valid_ratio),
            "min_depth_m": float(np.min(self.frame_minima)),
            "median_of_frame_medians_m": float(np.median(self.frame_medians)),
            "max_depth_observed_m": float(np.max(self.frame_maxima)),
            "object_mask_stats": self.object_stats,
            "hand_mask_stats": self.hand_stats,
            "background_warp_residuals": self.warp_metrics,
            "outputs": [str(self.zarr_path.name), str(self.video_path.name)],
        }
        self.group.attrs.update(
            {
                "model": str(metadata.get("model", "unknown")),
                "ground_truth_used": False,
            }
        )
        self.metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.metadata_path

    def abort(self) -> None:
        self.video.release()
