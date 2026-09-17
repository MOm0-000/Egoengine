"""Opt-in perception adapters for paper-faithful reproduction."""

from typing import Any

__all__ = [
    "camera_coordinate_sign", "project_hand_keypoints",
    "render_taco_oracle_metric_depth", "run_sam2_paper_masks",
]


def __getattr__(name: str) -> Any:
    if name in {"project_hand_keypoints", "run_sam2_paper_masks"}:
        from . import sam2

        return getattr(sam2, name)
    if name in {"camera_coordinate_sign", "render_taco_oracle_metric_depth"}:
        from . import oracle_depth

        return getattr(oracle_depth, name)
    raise AttributeError(name)
