"""Occlusion-aware depth selection for rigid calibration anchors.

All depth maps use camera-forward metric Z.  The functions deliberately
operate only on nominal rendered geometry and measured depth; hidden renderer
ownership labels are never accepted as algorithm inputs.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def _depth_map(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{label} must be a 2-D depth map")
    # PyTorch3D uses -1 for rasterizer background, while measured TACO depth
    # uses zero.  Both are treated as absent by the explicit ``> 0`` tests.
    if np.any(np.isinf(result)):
        raise ValueError(f"{label} contains invalid depth")
    return result


def target_visibility_mask(
    target_depth_m: object,
    occluder_depths_m: Iterable[object],
    *,
    uncertainty_margin_m: float,
) -> np.ndarray:
    """Return pixels where the nominal target is clearly the first surface.

    A near-tie is excluded conservatively.  ``uncertainty_margin_m`` is a
    visibility-order tolerance, not an algorithm accuracy threshold.
    """
    target = _depth_map(target_depth_m, "target_depth_m")
    if not np.isfinite(uncertainty_margin_m) or uncertainty_margin_m < 0:
        raise ValueError("uncertainty_margin_m must be finite and non-negative")
    visible = np.isfinite(target) & (target > 0)
    for index, value in enumerate(occluder_depths_m):
        occluder = _depth_map(value, f"occluder_depths_m[{index}]")
        if occluder.shape != target.shape:
            raise ValueError("target and occluder depth maps must have equal shapes")
        occluder_present = np.isfinite(occluder) & (occluder > 0)
        target_clearly_front = target + uncertainty_margin_m < occluder
        visible &= ~occluder_present | target_clearly_front
    return visible


def eroded_target_mask(mask: object, *, erosion_px: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not isinstance(erosion_px, int) or erosion_px < 0:
        raise ValueError("erosion_px must be a non-negative integer")
    if erosion_px == 0:
        return values.copy()
    size = 2 * erosion_px + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.erode(values.astype(np.uint8), kernel) > 0


def measured_target_selector(
    measured_depth_m: object,
    target_depth_m: object,
    occluder_depths_m: Iterable[object],
    *,
    uncertainty_margin_m: float,
    erosion_px: int,
    forbidden_mask: object,
) -> np.ndarray:
    """Select measured target depth outside every explicitly forbidden pixel.

    ``forbidden_mask`` is mandatory so real-data callers cannot silently treat
    a missing hand/uncertainty mask as an all-clear image.  It is independent
    of nominal depth ordering: a projected hand pixel remains forbidden even
    when the hand model is nominally behind the target.
    """
    measured = _depth_map(measured_depth_m, "measured_depth_m")
    target = _depth_map(target_depth_m, "target_depth_m")
    if measured.shape != target.shape:
        raise ValueError("measured and target depth maps must have equal shapes")
    forbidden = np.asarray(forbidden_mask)
    if forbidden.dtype != np.bool_ or forbidden.shape != measured.shape:
        raise ValueError("forbidden_mask must be a boolean mask matching depth")
    visible = target_visibility_mask(
        target, occluder_depths_m, uncertainty_margin_m=uncertainty_margin_m,
    )
    interior = eroded_target_mask(visible, erosion_px=erosion_px)
    return interior & ~forbidden & np.isfinite(measured) & (measured > 0)
