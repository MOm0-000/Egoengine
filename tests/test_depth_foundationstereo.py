from pathlib import Path

import numpy as np
import pytest

from video_to_spider.adapters.depth_foundationstereo import (
    apply_common_valid_domain,
    disparity_to_metric_depth,
    _git_revision_and_tracked_status,
)
from video_to_spider.optimization.sequence import _resolve_contact_similarity_mode


def test_disparity_conversion_preserves_metric_scale_after_resize():
    # Four processed pixels correspond to eight original pixels, so 1 px of
    # network-grid disparity must become 2 original-grid pixels.
    disparity_small = np.full((2, 4), 1.0, dtype=np.float32)
    depth, disparity, valid = disparity_to_metric_depth(
        disparity_small,
        original_width=8,
        original_height=4,
        focal_x_px=500.0,
        baseline_m=0.06,
    )

    assert valid[:, 2:].all()
    assert not valid[:, :2].any()
    np.testing.assert_allclose(disparity[valid], 2.0, atol=1e-6)
    np.testing.assert_allclose(depth[valid], 15.0, atol=1e-6)


def test_disparity_conversion_rejects_invalid_correspondence_geometry():
    disparity = np.array([[1.0, 1.0, np.nan, -2.0]], dtype=np.float32)
    depth, _, valid = disparity_to_metric_depth(
        disparity,
        original_width=4,
        original_height=1,
        focal_x_px=100.0,
        baseline_m=0.05,
    )

    np.testing.assert_array_equal(valid, [[False, True, False, False]])
    assert depth[0, 1] == pytest.approx(5.0)
    assert np.isnan(depth[~valid]).all()


def test_common_valid_domain_masks_depth_and_disparity():
    depth = np.ones((2, 3), dtype=np.float32)
    disparity = np.full((2, 3), 5.0, dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)
    common = np.array([[False, True, True], [False, True, False]])

    masked_depth, masked_disparity, masked_valid = apply_common_valid_domain(
        depth, disparity, valid, common,
    )

    np.testing.assert_array_equal(masked_valid, common)
    assert np.isnan(masked_depth[~common]).all()
    assert np.isnan(masked_disparity[~common]).all()
    np.testing.assert_allclose(masked_depth[common], 1.0)


def test_stereo_gate_forces_contact_validation_without_target_rewrite():
    stereo_gate = {
        "accepted": True,
        "gate_kind": "calibrated_stereo_native_metric",
    }

    assert _resolve_contact_similarity_mode("auto", stereo_gate) == "validate_only"
    with pytest.raises(ValueError, match="forbids contact-similarity refinement"):
        _resolve_contact_similarity_mode("refine", stereo_gate)
    assert _resolve_contact_similarity_mode("validate_only", stereo_gate) == "validate_only"
    assert _resolve_contact_similarity_mode("auto", None) == "refine"


def test_official_checkout_audit_ignores_only_untracked_runtime_files(tmp_path: Path):
    # The production adapter rejects modifications to official tracked source;
    # untracked model caches remain outside the source-integrity decision.
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "model.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"], check=True)

    _, status = _git_revision_and_tracked_status(tmp_path)
    assert status == ""
    (tmp_path / "runtime.cache").write_text("cache", encoding="utf-8")
    _, status = _git_revision_and_tracked_status(tmp_path)
    assert status == ""
    tracked.write_text("value = 2\n", encoding="utf-8")
    _, status = _git_revision_and_tracked_status(tmp_path)
    assert status != ""
