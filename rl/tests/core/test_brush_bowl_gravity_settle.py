from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from run_taco_brush_bowl_frame0_gravity_settle_v1 import (  # noqa: E402
    assess_variant,
    quaternion_distance_rad,
)


def test_quaternion_distance_is_sign_invariant() -> None:
    q = np.array([0.9238795325, 0.0, 0.0, 0.3826834324])
    assert quaternion_distance_rad(q, -q) < 1e-12
    assert np.isclose(
        quaternion_distance_rad(np.array([1.0, 0.0, 0.0, 0.0]), q),
        np.pi / 4.0,
        atol=1e-10,
    )


def _criteria() -> dict:
    config = yaml.safe_load((
        RL_ROOT / "configs/taco_brush_bowl_frame0_gravity_settle_v1.yaml"
    ).read_text(encoding="utf-8"))
    return config["frozen_stability_criteria"]


def _stable_rows() -> list[dict]:
    rows = []
    for time in np.arange(0.0, 2.0001, 0.1):
        contact = time >= 0.2
        rows.append({
            "time_s": float(time),
            "contact_count": int(contact),
            "linear_speed_m_s": 0.001 if contact else 0.01,
            "angular_speed_rad_s": 0.01 if contact else 0.1,
            "position_x_m": 0.5,
            "position_y_m": -0.04,
            "orientation_change_rad": 0.001,
            "collision_minimum_distance_m": -0.0001 if contact else 0.002,
            "linear_velocity_z_m_s": -0.001,
            "normal_force_sum_N": 1.0 if contact else 0.0,
            "native_visual_clearance_m": 0.0 if contact else 0.002,
            "horizontal_displacement_m": 0.0,
        })
    return rows


def test_frozen_stability_criteria_accept_stable_tail() -> None:
    result = assess_variant(_stable_rows(), _criteria(), [])
    assert result["stable"] is True
    assert all(result["checks"].values())


def test_frozen_stability_criteria_reject_obvious_rebound() -> None:
    rows = _stable_rows()
    rows[5]["linear_velocity_z_m_s"] = 0.051
    result = assess_variant(rows, _criteria(), [])
    assert result["metrics"]["obvious_rebound"] is True
    assert result["checks"]["no_obvious_rebound"] is False
    assert result["stable"] is False
