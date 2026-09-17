"""Active experiment setup and TACO 3.1 provenance utilities."""

from .manifest import ExperimentManifest
from .taco_ablation import (
    TacoAblationSpec, execute_taco_ablation, load_taco_ablation,
    prepare_taco_ablation, summarize_taco_ablation,
)
from .taco_auto_segmentation import (
    TacoAutoSegmentationSpec, execute_taco_auto_segmentation_v2,
    load_taco_auto_segmentation_spec, prepare_taco_auto_segmentation_v2,
    summarize_taco_auto_segmentation_v2,
)
from .taco_depth_calibration import (
    PROFILE_ORDER as TACO_DEPTH_PROFILE_ORDER,
    TacoDepthCalibrationSpec, execute_taco_depth_calibration,
    load_taco_depth_calibration_spec, prepare_taco_depth_calibration,
)
from .taco_oracle_depth import (
    TacoOracleDepthSpec, execute_taco_oracle_depth_ablation,
    load_taco_oracle_depth_spec, prepare_taco_oracle_depth_ablation,
    summarize_taco_oracle_depth_ablation,
)

__all__ = [
    "ExperimentManifest",
    "TACO_DEPTH_PROFILE_ORDER",
    "TacoAblationSpec",
    "TacoAutoSegmentationSpec",
    "TacoDepthCalibrationSpec",
    "TacoOracleDepthSpec",
    "execute_taco_ablation",
    "execute_taco_auto_segmentation_v2",
    "execute_taco_depth_calibration",
    "execute_taco_oracle_depth_ablation",
    "load_taco_ablation",
    "load_taco_auto_segmentation_spec",
    "load_taco_depth_calibration_spec",
    "load_taco_oracle_depth_spec",
    "prepare_taco_ablation",
    "prepare_taco_auto_segmentation_v2",
    "prepare_taco_depth_calibration",
    "prepare_taco_oracle_depth_ablation",
    "summarize_taco_ablation",
    "summarize_taco_auto_segmentation_v2",
    "summarize_taco_oracle_depth_ablation",
]
