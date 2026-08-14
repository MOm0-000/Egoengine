"""Human2Sim2Robot-style residual RL primitives adapted to xHand.

The numerical pieces here are deliberately dependency-light (NumPy only) so
they stay testable in the ``v2s-core`` environment and can later be wired into
a full PyTorch/rl-games training loop without modifying the cloned SPIDER
checkout.
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
