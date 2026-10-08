from __future__ import annotations

from pathlib import Path
import sys


RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_mano_xhand_object_penetration_attribution_v1 import (  # noqa: E402
    origin_classification,
    penetration_status,
)


def test_penetration_status_preserves_existing_tolerance() -> None:
    tolerance = 5e-5
    assert penetration_status(-tolerance - 1e-12, tolerance) == "PENETRATION"
    assert penetration_status(-tolerance, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert penetration_status(tolerance, tolerance) == "CONTACT_WITHIN_TOLERANCE"
    assert penetration_status(tolerance + 1e-12, tolerance) == "CLEARANCE"


def test_origin_classification_distinguishes_inherited_and_added() -> None:
    tolerance = 5e-5
    assert origin_classification(-0.001, -0.002, tolerance) == (
        "OFFICIAL_GEOMETRY_PENETRATION_RETAINED_AFTER_RETARGET"
    )
    assert origin_classification(0.001, -0.002, tolerance) == (
        "RETARGET_ADDED_PENETRATION"
    )
    assert origin_classification(-0.001, 0.002, tolerance) == (
        "RETARGET_REMOVED_OFFICIAL_PENETRATION"
    )
    assert origin_classification(0.001, 0.002, tolerance) == (
        "NO_MATERIAL_PENETRATION"
    )
