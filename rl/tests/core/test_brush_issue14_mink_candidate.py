from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from run_taco_brush_issue14_mink_candidate_v1 import (  # noqa: E402
    classify_candidate,
    pickup_screen,
)


def test_candidate_support_is_exact_and_unpromoted() -> None:
    contract = yaml.safe_load((
        RL_ROOT / "configs/taco_brush_issue14_candidate_support_v1.yaml"
    ).read_text(encoding="utf-8"))
    assert contract["promotion_authorized"] is False
    assert contract["simulator"]["offset_m"] == 0.7173599392270725


def test_pickup_screen_requires_lift_and_sustained_near_alignment() -> None:
    result = pickup_screen(
        np.array([0.0, 0.005, 0.011, 0.020]),
        np.array([0.010, 0.003, 0.0015, 0.0010]),
        lift_delta_m=0.01,
        near_distance_m=0.002,
        fraction_min=0.5,
    )
    assert result["screen_pass"] is True
    assert result["first_elevated_frame"] == 2
    assert result["elevated_frame_count"] == 2
    assert result["near_fraction_during_elevation"] == 1.0


def test_pickup_screen_does_not_confuse_object_lift_with_hand_alignment() -> None:
    result = pickup_screen(
        np.array([0.0, 0.015, 0.025]),
        np.array([0.010, 0.009, 0.008]),
        lift_delta_m=0.01,
        near_distance_m=0.002,
        fraction_min=0.5,
    )
    assert result["elevated_frame_count"] == 2
    assert result["near_frame_count_during_elevation"] == 0
    assert result["screen_pass"] is False
    assert result["interpretation"] == "KINEMATIC_ALIGNMENT_ONLY_NOT_PHYSICAL_LIFT_PROOF"


def test_native_mesh_penetration_blocks_normal_pickup_classification() -> None:
    assert classify_candidate(
        static_pass=True,
        selected_native_pass=False,
        pickup_alignment_pass=True,
    ) == "MINK_COMPLETE_FIXED_FRAME_HAND_OBJECT_PENETRATION"
