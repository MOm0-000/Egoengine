from __future__ import annotations

from pathlib import Path
import sys


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_local_hand_object_collision_feasibility_v1 import (  # noqa: E402
    classify_frame,
)


def _candidate(*, native_distance: float, proxy_distance: float = 0.0) -> dict:
    return {
        "joint_limits_pass": True,
        "frame_displacement_pass": True,
        "self_collision_pass": True,
        "native_support_pass": True,
        "object_proxy_minimum_distance_m": proxy_distance,
        "native_hand_object_by_object": {
            "brush": {"distance_m": native_distance, "status": "PENETRATION" if native_distance < -5e-5 else "CLEARANCE"},
        },
    }


def _baseline(*, native_distance: float = -0.001) -> dict:
    return {
        "native_hand_object_by_object": {
            "brush": {"distance_m": native_distance, "status": "PENETRATION" if native_distance < -5e-5 else "CLEARANCE"},
        },
    }


def test_classification_rejects_proxy_clear_native_penetration() -> None:
    assert classify_frame(
        _baseline(), _candidate(native_distance=-0.001), None, gap_reference_m=0.002,
    ) == "PROXY_CLEAR_BUT_NATIVE_PENETRATION_REMAINS"


def test_classification_records_large_native_gap_without_relaxing_solve() -> None:
    assert classify_frame(
        _baseline(), _candidate(native_distance=0.003), None, gap_reference_m=0.002,
    ) == "NATIVE_PENETRATION_CLEARED_WITH_GAP_OVER_2MM"


def test_classification_fails_closed_on_solver_error() -> None:
    assert classify_frame(
        _baseline(), _candidate(native_distance=0.0), "QP failed", gap_reference_m=0.002,
    ) == "LOCAL_MINK_SOLVE_FAILED"
