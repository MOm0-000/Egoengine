from pathlib import Path

import numpy as np
import pytest
import yaml

from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.h2s2r import apply_residual_action


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "configs/taco_pour_residual_action_legacy_v1.yaml"
SCALED = ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml"


def test_scaled_profile_uses_full_normalized_range_without_expanding_safety_limit():
    spec, report = load_residual_action_profile(SCALED)
    assert spec.residual_scale == spec.residual_clip == 0.05
    assert report["mapping_kind"] == "linear_scaled_candidate"
    reference = np.zeros((1, 18), dtype=np.float64)
    normalized = np.array([[0.2, -0.7, 1.0] + [0.0] * 15])
    output = apply_residual_action(reference, normalized, spec)
    np.testing.assert_allclose(output[0, :3], [0.01, -0.035, 0.05])
    assert np.abs(output).max() <= 0.05


def test_legacy_profile_preserves_frozen_checkpoint_mapping():
    spec, report = load_residual_action_profile(LEGACY)
    assert spec.residual_scale == 1.0
    assert spec.residual_clip == 0.05
    assert report["mapping_kind"] == "direct_clip_legacy"
    reference = np.zeros((1, 18), dtype=np.float64)
    normalized = np.array([[0.2, -0.7, 1.0] + [0.0] * 15])
    output = apply_residual_action(reference, normalized, spec)
    np.testing.assert_allclose(output[0, :3], [0.05, -0.05, 0.05])


def test_scaled_profile_fails_closed_if_scale_no_longer_matches_clip(tmp_path):
    profile = yaml.safe_load(SCALED.read_text())
    profile["mapping"]["residual_scale"] = 0.04
    altered = tmp_path / "action.yaml"
    altered.write_text(yaml.safe_dump(profile))
    with pytest.raises(ValueError, match="map normalized"):
        load_residual_action_profile(altered)
