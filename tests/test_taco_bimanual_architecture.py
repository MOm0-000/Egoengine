from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl/src"))

from egoengine_repro.retarget.taco_bimanual_settings import (
    TacoBimanualRetargetSettings,
    load_taco_bimanual_settings,
)


CONFIG = ROOT / "rl/configs/taco_bimanual_mink_local_v1.yaml"
SOURCE = ROOT / "rl/src/egoengine_repro/retarget/taco_bimanual.py"


def test_active_taco_config_is_exactly_consumed() -> None:
    settings, audit = load_taco_bimanual_settings(CONFIG)
    assert isinstance(settings, TacoBimanualRetargetSettings)
    assert audit["unknown_key_count"] == 0
    assert audit["unused_key_count"] == 0
    assert audit["declared_key_count"] == audit["consumed_key_count"] == 11


@pytest.mark.parametrize("unknown", ["posture_cost", "wrist_position_cost"])
def test_active_taco_config_rejects_historical_pseudo_switches(unknown: str) -> None:
    document = yaml.safe_load(CONFIG.read_text())
    document["settings"][unknown] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        TacoBimanualRetargetSettings.from_mapping(document["settings"])


def test_taco_bimanual_has_no_generic_retarget_import_or_pour_guard() -> None:
    source = SOURCE.read_text()
    assert "from .mink import" not in source
    assert "retarget_with_mink" not in source
    assert "local_surface_guard" not in source
    assert "_palm_thumb_surface_guard_" not in source


def test_brush_retarget_has_no_action_optimization_config_dependency() -> None:
    source = SOURCE.read_text()
    config = CONFIG.read_text()
    for forbidden in (
        "reward", "chunks", "ppo_", "mpc_", "position_boundary",
    ):
        assert forbidden not in source
        assert forbidden not in config


def test_shared_kinematic_limits_are_single_source_of_truth() -> None:
    generic = (ROOT / "rl/src/egoengine_repro/retarget/mink.py").read_text()
    taco = SOURCE.read_text()
    for private in (
        "class _FrameDisplacementLimit", "class _StrictCollisionLimit",
        "def _joint_velocity_limits", "def _explicit_collision_groups",
    ):
        assert private not in generic
        assert private not in taco
    assert "from .kinematic_limits import" in generic
    assert "from .kinematic_limits import" in taco
