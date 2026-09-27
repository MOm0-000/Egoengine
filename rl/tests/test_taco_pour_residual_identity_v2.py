"""Residual-identity and baseline-preserving-bound contract tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.mjwp_env import MJWPVectorEnv
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    load_truncated_gaussian_profile,
    normalized_action_bounds_numpy,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"
)
MODEL = ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml"
ACTION = ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml"
DISTRIBUTION = ROOT / "configs/taco_pour_baseline_preserving_truncated_gaussian_v2.yaml"


def _reference_and_limits():
    reference = np.load(REFERENCE)["ctrl"].astype(np.float32)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nu == reference.shape[1] == 36
    return (
        reference,
        np.asarray(model.actuator_ctrllimited, dtype=bool),
        np.asarray(model.actuator_ctrlrange, dtype=np.float64),
    )


def _violation(command, limited, ranges):
    value = np.asarray(command, dtype=np.float64)
    below = np.where(limited, np.maximum(ranges[:, 0] - value, 0.0), 0.0)
    above = np.where(limited, np.maximum(value - ranges[:, 1], 0.0), 0.0)
    return np.maximum(below, above)


def test_zero_residual_is_bitwise_formal_reference_for_every_row():
    reference, _, _ = _reference_and_limits()
    residual, _ = load_residual_action_profile(ACTION)
    residual = replace(
        residual, hand_dof=36, hand_control_indices=tuple(range(36))
    )
    env = object.__new__(MJWPVectorEnv)
    env.env_cfg = SimpleNamespace(residual=residual)
    result = env._apply_residual(
        torch.from_numpy(reference.copy()),
        torch.zeros((len(reference), 36), dtype=torch.float32),
    ).numpy()
    assert result.tobytes() == reference.tobytes()


def test_zero_belongs_to_every_formal_reference_support():
    reference, limited, ranges = _reference_and_limits()
    spec, report = load_truncated_gaussian_profile(DISTRIBUTION)
    assert report["profile_id"] == "taco_pour_baseline_preserving_truncated_gaussian_v2"
    assert report["distribution"]["execution_reference"] == (
        "formal_reference_ctrl_unmodified"
    )
    low, high, audit = normalized_action_bounds_numpy(
        reference,
        ctrllimited=limited,
        ctrlrange=ranges,
        residual_scale=spec.residual_scale,
        reference_snap_tolerance=spec.reference_snap_tolerance,
    )
    assert np.all(low <= 0.0)
    assert np.all(high >= 0.0)
    assert audit["execution_reference"] == "formal_reference_ctrl_unmodified"
    assert audit["snapped_component_count"] == 0


def test_low_high_and_zero_never_worsen_baseline_violation():
    reference, limited, ranges = _reference_and_limits()
    spec, _ = load_truncated_gaussian_profile(DISTRIBUTION)
    low, high, _ = normalized_action_bounds_numpy(
        reference,
        ctrllimited=limited,
        ctrlrange=ranges,
        residual_scale=spec.residual_scale,
        reference_snap_tolerance=spec.reference_snap_tolerance,
    )
    baseline = _violation(reference, limited, ranges)
    scale = np.float32(spec.residual_scale)
    for action in (low.astype(np.float32), high.astype(np.float32), np.zeros_like(reference)):
        raw = (reference + scale * action).astype(np.float32)
        observed = _violation(raw, limited, ranges)
        assert float(np.max(observed - baseline)) <= 2.0e-15
