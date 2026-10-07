from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egoengine_repro.evaluation.taco_calibration_residual import invert_transform
from egoengine_repro.scene.support_surface import Plane


RL_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = RL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_taco_brush_single_frame_calibration_penetration_v1 as brush_audit


def _rigid(rotvec: list[float], translation: list[float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def test_basic_distance_and_plane_normalization_known_answers() -> None:
    plane = Plane([0, 0, 2], 1.10, "world")
    canonical = Plane([0, 0, 1], 0.55, "world")
    points = np.array([[0.0, 0.0, 0.545], [1.0, -2.0, 0.55]])
    assert plane.normal == pytest.approx(canonical.normal, abs=0.0)
    assert plane.offset == pytest.approx(canonical.offset, abs=0.0)
    assert plane.signed_distance(points) == pytest.approx([-0.005, 0.0], abs=1e-15)
    for normal, offset in (([0, 0, 0], 1.0), ([np.nan, 0, 1], 1.0), ([0, 0, 1], np.inf)):
        with pytest.raises(ValueError):
            Plane(normal, offset, "world")


def test_plane_and_point_rigid_transform_preserves_signed_distance() -> None:
    plane = Plane([0.2, -0.3, 0.9], 0.44, "source")
    points = np.array([[0.1, 0.2, 0.6], [-0.5, 0.2, 0.7], [0.0, 0.0, 0.0]])
    transform = _rigid([0.2, -0.1, 0.05], [0.3, -0.4, 0.2])
    moved_points = points @ transform[:3, :3].T + transform[:3, 3]
    moved_plane = plane.transform(transform, target_frame="target")
    # The right-hand side is independently evaluated from n'=Rn and
    # d'=d+n' dot t rather than by invoking Plane.transform a second time.
    expected_normal = transform[:3, :3] @ plane.normal
    expected_offset = plane.offset + expected_normal @ transform[:3, 3]
    expected = moved_points @ expected_normal - expected_offset
    assert moved_plane.normal == pytest.approx(expected_normal, abs=1e-14)
    assert moved_plane.offset == pytest.approx(expected_offset, abs=1e-14)
    assert expected == pytest.approx(plane.signed_distance(points), abs=1e-14)


def test_camera_local_expression_matches_world_conjugation() -> None:
    world_plane = Plane([0.1, 0.2, 0.97], 0.6, "world")
    world_to_camera = _rigid([0.15, -0.08, 0.04], [0.2, -0.1, 0.3])
    camera_correction = _rigid([-0.06, 0.12, 0.03], [0.01, -0.02, 0.04])

    direct_camera = world_plane.transform(world_to_camera, target_frame="camera")
    direct_camera = direct_camera.transform(camera_correction, target_frame="camera_corrected")
    direct_world = direct_camera.transform(invert_transform(world_to_camera), target_frame="world")
    conjugated = world_plane.transform(
        invert_transform(world_to_camera) @ camera_correction @ world_to_camera,
        target_frame="world",
    )
    assert direct_world.normal == pytest.approx(conjugated.normal, abs=1e-14)
    assert direct_world.offset == pytest.approx(conjugated.offset, abs=1e-14)


def test_transform_same_plane_keeps_tilt_beyond_old_ten_degree_prior(monkeypatch) -> None:
    def forbidden_refit(*_args, **_kwargs):
        raise AssertionError("corrected points must never trigger a replacement-plane fit")

    monkeypatch.setattr(brush_audit, "fit_horizontal_support_plane", forbidden_refit)
    original = Plane([0, 0, 1], 0.55, "world")
    correction = _rigid([0.0, np.deg2rad(25.0), 0.0], [0.0, 0.0, 0.0])
    transformed = brush_audit.transform_same_baseline_plane(
        original, np.eye(4), correction, "WORLD_FIXED",
    )
    tilt = np.degrees(np.arccos(np.clip(transformed.normal[2], -1.0, 1.0)))
    assert tilt == pytest.approx(25.0, abs=1e-12)


def test_historical_frame139_uses_exact_saved_plane_and_transform() -> None:
    plane_path = RL_ROOT / "runs/taco_multiframe_table_depth_estimation_v1/per_frame_table_plane_estimates.jsonl"
    correction_path = RL_ROOT / (
        "runs/taco_brush_open3d_single_frame_calibration_penetration_v1/"
        "per_frame_corrections.npz"
    )
    records = [json.loads(line) for line in plane_path.read_text(encoding="utf-8").splitlines()]
    source = next(row for row in records if row["sample"] == "brush_brush_bowl_20230927_027" and row["frame"] == 139)
    original = Plane(source["plane"]["normal"], source["plane"]["offset_m"], "world")
    corrections = np.load(correction_path)
    transformed = original.transform(corrections["world_fixed_transform"][139], target_frame="world")
    tilt = np.degrees(np.arccos(np.clip(transformed.normal[2], -1.0, 1.0)))
    assert tilt == pytest.approx(28.62714056890027, abs=1e-10)
    assert transformed.offset == pytest.approx(0.5297505384842496, abs=1e-10)
    # This is a frozen-artifact regression only, not a video replay.
    assert tilt != pytest.approx(6.7788303309, abs=1e-6)
    assert transformed.offset != pytest.approx(0.6802256081, abs=1e-6)


def _summary_row(
    frame: int,
    *,
    brush_before: float,
    brush_after: float,
    bowl_before: float,
    bowl_after: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame": frame,
        "effective_comparison": True,
        "fit_status": "OUTPUT",
        "fit_reason": "",
        "candidate_exists": True,
        "translation_norm_mm": 0.0,
        "rotation_angle_deg": 0.0,
        "before_plane_offset_m": 0.55,
        "before_plane_tilt_deg": 0.0,
        "after_plane_offset_m": 0.55,
        "after_plane_tilt_deg": 0.0,
    }
    values = {
        "brush": (brush_before, brush_after),
        "bowl": (bowl_before, bowl_after),
        "left_hand": (0.0, 0.0),
        "right_hand": (0.0, 0.0),
        "hand": (0.0, 0.0),
    }
    for entity, (before, after) in values.items():
        row[f"{entity}_before_mm"] = before
        row[f"{entity}_after_mm"] = after
        row[f"{entity}_before_penetrating"] = before < -0.05
        row[f"{entity}_after_penetrating"] = after < -0.05
    return row


def test_summary_reports_mixed_result_and_preserves_zero_values() -> None:
    rows = [
        _summary_row(0, brush_before=-2.0, brush_after=0.0, bowl_before=-1.0, bowl_after=-3.0),
        _summary_row(1, brush_before=-1.0, brush_after=0.0, bowl_before=0.0, bowl_after=-2.0),
    ]
    result = brush_audit.classify_model(rows, frame_count=3)
    assert result["classification"] == "MIXED_LOCAL_IMPROVEMENT_AND_DETERIORATION"
    assert result["unchecked_frames"] == 1
    assert result["entities"]["brush"]["after_minimum_mm"] == 0.0
    assert result["entities"]["brush"]["interpretation"] == "IMPROVED"
    assert result["entities"]["bowl"]["interpretation"] == "WORSENED"

