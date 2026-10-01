"""Residual-RL task primitives and the stable :mod:`video_to_spider.rl.core`.

The active PPO orchestration is the direct, non-inherited ``core`` package.
``mjwp_env.py`` remains the audited low-level MuJoCo-Warp adapter; it does not
own the active policy or optimizer lifecycle.
"""

from .h2s2r import (
    AnchorRewardConfig,
    DomainRandomizationConfig,
    H2S2RTrainingConfig,
    XHandResidualPolicySpec,
    apply_residual_action,
    build_privileged_state,
    build_xhand_observation,
    make_anchor_points,
    make_symmetry_anchor_points,
    object_anchor_reward,
    relative_pose_distance,
    sample_domain_randomization,
    transform_anchor_points,
)
from .reset_sampler import (
    PreGraspResetSampler,
    PreGraspSamplerConfig,
    sample_pre_grasp_indices,
)

__all__ = [
    "AnchorRewardConfig",
    "DomainRandomizationConfig",
    "H2S2RTrainingConfig",
    "XHandResidualPolicySpec",
    "apply_residual_action",
    "build_privileged_state",
    "build_xhand_observation",
    "make_anchor_points",
    "make_symmetry_anchor_points",
    "object_anchor_reward",
    "PreGraspResetSampler",
    "PreGraspSamplerConfig",
    "relative_pose_distance",
    "sample_domain_randomization",
    "sample_pre_grasp_indices",
    "transform_anchor_points",
]
