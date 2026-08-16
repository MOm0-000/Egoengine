"""Compare FoundationStereo metric depth with ADT GT camera-Z depth.

Both inputs are assumed to be in the same rectified left reference frame.  The
script reports full-frame, object-region and object-boundary metrics and
explicitly separates valid/invalid pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_zarr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    depth = np.asarray(group["depth_m"])
    valid = np.asarray(group["valid"])
    disparity = np.asarray(group["disparity_px"]) if "disparity_px" in group else None
    return depth, valid, disparity


def _load_gt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    depth = np.asarray(group["depth_m"])
    valid = np.asarray(group["valid"])
    mask = np.asarray(group["object_mask"])
    attrs = dict(group.attrs)
    return depth, valid, mask, attrs


def _safe_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {
            "count": 0,
            "abs_rel": float("nan"),
            "rmse_m": float("nan"),
            "delta_1": float("nan"),
            "scale_ratio_median": float("nan"),
        }
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    valid_ratio = _safe_mask(p)
    p = p[valid_ratio]
    g = g[valid_ratio]
    if not p.size:
        return {
            "count": 0,
            "abs_rel": float("nan"),
            "rmse_m": float("nan"),
            "delta_1": float("nan"),
            "scale_ratio_median": float("nan"),
        }
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta_1 = float(np.mean(ratio < 1.25))
    scale_ratio_median = float(np.median(p / g))
    return {
        "count": int(p.size),
        "abs_rel": abs_rel,
        "rmse_m": rmse,
        "delta_1": delta_1,
        "scale_ratio_median": scale_ratio_median,
    }


def _object_boundary_mask(mask: np.ndarray, band: int = 3) -> np.ndarray:
    import cv2

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return dilated & (~eroded)


def evaluate(
    *,
    pred_depth_zarr: Path,
    gt_zarr: Path,
    output_dir: Path,
    boundary_band: int = 3,
    fx_rect_px: float | None = None,
    baseline_m: float | None = None,
) -> dict[str, Any]:
    pred, pred_valid, pred_disparity = _load_zarr(pred_depth_zarr)
    gt, gt_valid, object_mask, gt_attrs = _load_gt(gt_zarr)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred_valid.shape != pred.shape or gt_valid.shape != gt.shape or object_mask.shape != gt.shape:
        raise ValueError("zarr mask shape mismatch")
    if pred_disparity is not None and pred_disparity.shape != pred.shape:
        raise ValueError(f"pred disparity shape mismatch: {pred_disparity.shape} vs {pred.shape}")
    # Use a common physically valid domain.
    common_valid = gt_valid & pred_valid
    full_mask = common_valid & _safe_mask(gt)
    object_mask = object_mask & common_valid & _safe_mask(gt)
    boundary_mask = _object_boundary_mask(object_mask, boundary_band) & common_valid & _safe_mask(gt)
    per_frame: list[dict[str, Any]] = []
    invalid_ratios: list[float] = []
    edge_abs_rel: list[float] = []
    for frame_index in range(pred.shape[0]):
        full = _compute_metrics(pred[frame_index], gt[frame_index], full_mask[frame_index])
        obj = _compute_metrics(pred[frame_index], gt[frame_index], object_mask[frame_index])
        edge = _compute_metrics(pred[frame_index], gt[frame_index], boundary_mask[frame_index])
        # invalid rate: GT object pixels where pred is invalid/non-positive.
        obj_gt_valid = object_mask[frame_index]
        invalid_ratio = 0.0
        if obj_gt_valid.any():
            invalid_ratio = float(1.0 - np.mean(_safe_mask(pred[frame_index][obj_gt_valid])))
        invalid_ratios.append(invalid_ratio)
        if np.isfinite(edge["abs_rel"]):
            edge_abs_rel.append(edge["abs_rel"])
        per_frame.append(
            {
                "frame_index": frame_index,
                "full": full,
                "object": obj,
                "boundary": edge,
                "object_invalid_ratio": invalid_ratio,
            }
        )

    def aggregate(region: str, metric: str) -> dict[str, float]:
        values = [frame[region][metric] for frame in per_frame]
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return {"median": float("nan"), "mean": float("nan"), "p95": float("nan")}
        return {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
        }

    def region_aggregates(region: str) -> dict[str, Any]:
        return {
            f"{region}_abs_rel": aggregate(region, "abs_rel"),
            f"{region}_rmse_m": aggregate(region, "rmse_m"),
            f"{region}_delta_1": aggregate(region, "delta_1"),
            f"{region}_scale_ratio_median": aggregate(region, "scale_ratio_median"),
        }

    disparity_audit: dict[str, Any] | None = None
    pred_internal_consistency: dict[str, Any] | None = None
    if fx_rect_px is not None and baseline_m is not None:
        if not np.isfinite(fx_rect_px) or fx_rect_px <= 0:
            raise ValueError("fx_rect_px must be positive and finite")
        if not np.isfinite(baseline_m) or baseline_m <= 0:
            raise ValueError("baseline_m must be positive and finite")
        if pred_disparity is not None:
            reconstructed = (float(fx_rect_px) * float(baseline_m)) / np.maximum(pred_disparity, 1e-6)
            both_valid = common_valid & _safe_mask(pred) & _safe_mask(reconstructed)
            if both_valid.any():
                rel = np.abs(pred[both_valid] - reconstructed[both_valid]) / np.maximum(pred[both_valid], 1e-6)
                pred_internal_consistency = {
                    "median_rel_err": float(np.median(rel)),
                    "p95_rel_err": float(np.percentile(rel, 95)),
                    "max_rel_err": float(np.max(rel)),
                }
        gt_disparity = (float(fx_rect_px) * float(baseline_m)) / np.maximum(gt, 1e-6)
        gt_disparity = np.where(common_valid & _safe_mask(gt), gt_disparity, np.nan)
        pred_disp = np.where(common_valid & _safe_mask(pred), pred_disparity, np.nan) if pred_disparity is not None else np.full_like(gt, np.nan)
        disparity_audit = {}
        for name, mask in (("full", full_mask), ("object", object_mask)):
            region_mask = mask & np.isfinite(gt_disparity) & np.isfinite(pred_disp) & (pred_disp > 0) & (gt_disparity > 0)
            if region_mask.any():
                ratio = pred_disp[region_mask] / gt_disparity[region_mask]
                disparity_audit[name] = {
                    "median_ratio": float(np.median(ratio)),
                    "median_pred_disparity_px": float(np.median(pred_disp[region_mask])),
                    "median_gt_disparity_px": float(np.median(gt_disparity[region_mask])),
                }
            else:
                disparity_audit[name] = {"median_ratio": float("nan"), "median_pred_disparity_px": float("nan"), "median_gt_disparity_px": float("nan")}

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "pred_depth_zarr": str(pred_depth_zarr),
        "gt_zarr": str(gt_zarr),
        "frame_count": int(pred.shape[0]),
        "gt_depth_semantic": gt_attrs.get("depth_semantics"),
        "gt_stream_selection": gt_attrs.get("gt_stream_selection"),
        "metrics": {
            **region_aggregates("full"),
            **region_aggregates("object"),
            "object_invalid_ratio": {
                "median": float(np.median(invalid_ratios)),
                "mean": float(np.mean(invalid_ratios)),
                "p95": float(np.percentile(invalid_ratios, 95)),
            },
            "boundary_abs_rel": {
                "median": float(np.median(edge_abs_rel)) if edge_abs_rel else float("nan"),
                "mean": float(np.mean(edge_abs_rel)) if edge_abs_rel else float("nan"),
                "p95": float(np.percentile(edge_abs_rel, 95)) if edge_abs_rel else float("nan"),
            },
        },
        "disparity_audit": disparity_audit,
        "pred_internal_consistency": pred_internal_consistency,
        "per_frame": per_frame,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stereo_depth_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _read_prepared_geometry(prepared_dir: Path) -> tuple[float, float]:
    """Read ``focal_length_px`` and ``baseline_m`` from ADT prepare metadata."""
    import json

    path = prepared_dir / "adt_stereo_prepare.json"
    if not path.is_file():
        raise ValueError(f"prepared metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rect = payload.get("rectification", {})
    focal = rect.get("focal_length_px")
    baseline = rect.get("baseline_m")
    if focal is None or baseline is None:
        raise ValueError("prepared metadata lacks rectification.focal_length_px/baseline_m")
    return float(focal), float(baseline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-depth-zarr", type=Path, required=True)
    parser.add_argument("--gt-zarr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-band", type=int, default=3)
    parser.add_argument("--fx-rect-px", type=float)
    parser.add_argument("--baseline-m", type=float)
    parser.add_argument("--prepared-dir", type=Path)
    args = parser.parse_args(argv)
    fx = args.fx_rect_px
    baseline = args.baseline_m
    if args.prepared_dir is not None:
        prepared_fx, prepared_baseline = _read_prepared_geometry(args.prepared_dir)
        fx = fx if fx is not None else prepared_fx
        baseline = baseline if baseline is not None else prepared_baseline
    result = evaluate(
        pred_depth_zarr=args.pred_depth_zarr,
        gt_zarr=args.gt_zarr,
        output_dir=args.output_dir,
        boundary_band=args.boundary_band,
        fx_rect_px=fx,
        baseline_m=baseline,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
