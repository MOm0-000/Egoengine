"""Scene geometry contracts shared by ingestion, retargeting and audits."""

from .support_surface import (
    Plane,
    SupportSurfaceContract,
    SupportSurfaceSpec,
    build_taco_scene_alignment,
    make_taco_support_contract,
    resolve_support_surface,
    taco_project_sample_support_contract,
)

__all__ = [
    "Plane",
    "SupportSurfaceContract",
    "SupportSurfaceSpec",
    "build_taco_scene_alignment",
    "make_taco_support_contract",
    "resolve_support_surface",
    "taco_project_sample_support_contract",
]
