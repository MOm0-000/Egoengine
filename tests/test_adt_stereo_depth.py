from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from video_to_spider.eval import adt_stereo_depth


def _write_pred(path: Path, depth: np.ndarray, *, with_disparity: bool) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("depth_m", data=depth.astype(np.float32))
    group.create_dataset("valid", data=np.ones(depth.shape, dtype=bool))
    if with_disparity:
        group.create_dataset("disparity_px", data=np.full(depth.shape, 5.0, dtype=np.float32))


def _write_gt(path: Path, depth: np.ndarray, object_mask: np.ndarray) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("depth_m", data=depth.astype(np.float32))
    group.create_dataset("valid", data=np.ones(depth.shape, dtype=bool))
    group.create_dataset("object_mask", data=object_mask.astype(bool))
    group.attrs["depth_semantics"] = "rectified-left camera Z (meters)"
    group.attrs["gt_stream_selection"] = {"depth": {"selected_stream_id": "345-2"}}


def test_evaluate_empty_object_mask_does_not_crash(tmp_path: Path) -> None:
    depth = np.full((1, 8, 8), 2.0, dtype=np.float32)
    _write_pred(tmp_path / "pred.zarr", depth, with_disparity=True)
    _write_gt(tmp_path / "gt.zarr", depth, np.zeros((1, 8, 8), dtype=bool))
    result = adt_stereo_depth.evaluate(
        pred_depth_zarr=tmp_path / "pred.zarr",
        gt_zarr=tmp_path / "gt.zarr",
        output_dir=tmp_path / "out",
        fx_rect_px=150.0,
        baseline_m=0.14,
    )
    assert result["gt_depth_semantic"] == "rectified-left camera Z (meters)"
    assert result["gt_stream_selection"]["depth"]["selected_stream_id"] == "345-2"
    assert np.isnan(result["metrics"]["object_abs_rel"]["median"])
    assert result["metrics"]["object_invalid_ratio"]["median"] == 0.0
    assert result["disparity_audit"]["object"]["median_ratio"] is not None


def test_evaluate_nan_and_zero_depths_are_safe(tmp_path: Path) -> None:
    depth = np.full((1, 4, 4), np.nan, dtype=np.float32)
    depth[0, 0, 0] = 0.0
    depth[0, 1, 1] = 2.0
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 1, 1] = True
    _write_pred(tmp_path / "pred.zarr", depth, with_disparity=False)
    _write_gt(tmp_path / "gt.zarr", depth, mask)
    result = adt_stereo_depth.evaluate(
        pred_depth_zarr=tmp_path / "pred.zarr",
        gt_zarr=tmp_path / "gt.zarr",
        output_dir=tmp_path / "out",
    )
    assert result["disparity_audit"] is None
    assert result["pred_internal_consistency"] is None
