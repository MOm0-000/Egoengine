"""Strict settings contract for the active TACO bimanual MINK retargeter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA = "taco_bimanual_mink_local_v1"
PROVENANCE = "LOCAL_IMPLEMENTATION_VALUES_NOT_PAPER_PUBLISHED"
SELF_ONLY_FINAL_FEASIBILITY = "SELF_ONLY_FINAL_FEASIBILITY"
UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY = (
    "UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY"
)


def _finite(name: str, value: Any, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(name: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], scope: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{scope} keys mismatch; missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True)
class VelocityLimits:
    base_translation: float
    base_rotation: float
    finger: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VelocityLimits":
        expected = {"base_translation", "base_rotation", "finger"}
        _exact_keys(value, expected, "velocity_limits")
        return cls(**{
            key: _finite(f"velocity_limits.{key}", value[key], positive=True)
            for key in sorted(expected)
        })

    def to_dict(self) -> dict[str, float]:
        return {
            "base_translation": self.base_translation,
            "base_rotation": self.base_rotation,
            "finger": self.finger,
        }


@dataclass(frozen=True)
class NativeSupportSettings:
    normal: tuple[float, float, float]
    offset_m: float
    frame: str
    minimum_clearance_m: float
    activation_distance_m: float | None
    gain: float
    depenetration_step_m: float
    validation_tolerance_m: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("native support normal must be a finite 3-vector")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-9):
            raise ValueError("native support normal must be unit length")
        _finite("native_support.offset_m", self.offset_m)
        _finite("native_support.minimum_clearance_m", self.minimum_clearance_m)
        if self.activation_distance_m is not None:
            activation = _finite(
                "native_support.activation_distance_m", self.activation_distance_m
            )
            if activation < self.minimum_clearance_m:
                raise ValueError("native support activation must be >= clearance")
        gain = _finite("native_support.gain", self.gain, positive=True)
        if gain > 1.0:
            raise ValueError("native support gain must be <= 1")
        _finite(
            "native_support.depenetration_step_m", self.depenetration_step_m,
            positive=True,
        )
        _finite(
            "native_support.validation_tolerance_m", self.validation_tolerance_m,
            nonnegative=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": list(self.normal),
            "offset_m": self.offset_m,
            "frame": self.frame,
            "minimum_clearance_m": self.minimum_clearance_m,
            "activation_distance_m": self.activation_distance_m,
            "gain": self.gain,
            "depenetration_step_m": self.depenetration_step_m,
            "validation_tolerance_m": self.validation_tolerance_m,
        }


@dataclass(frozen=True)
class TacoBimanualRetargetSettings:
    wrist_orientation_cost: float
    fingertip_position_cost: float
    fingertip_orientation_cost: float
    max_iterations_per_frame: int
    max_feasibility_iterations: int
    solver_primal_tolerance: float
    solver_dual_tolerance: float
    planning_collision_buffer_m: float
    accepted_min_self_collision_distance_m: float
    collision_acceptance_numerical_slack_m: float
    velocity_limits: VelocityLimits
    final_feasibility_algorithm: str = SELF_ONLY_FINAL_FEASIBILITY
    native_support: NativeSupportSettings | None = None

    def __post_init__(self) -> None:
        for name in (
            "wrist_orientation_cost", "fingertip_position_cost",
            "fingertip_orientation_cost",
        ):
            _finite(name, getattr(self, name), nonnegative=True)
        _integer(
            "max_iterations_per_frame", self.max_iterations_per_frame,
            positive=True,
        )
        _integer(
            "max_feasibility_iterations", self.max_feasibility_iterations,
            positive=True,
        )
        _finite(
            "solver_primal_tolerance", self.solver_primal_tolerance,
            positive=True,
        )
        _finite(
            "solver_dual_tolerance", self.solver_dual_tolerance,
            positive=True,
        )
        _finite(
            "planning_collision_buffer_m", self.planning_collision_buffer_m,
            nonnegative=True,
        )
        accepted = _finite(
            "accepted_min_self_collision_distance_m",
            self.accepted_min_self_collision_distance_m,
        )
        if accepted > 0.0:
            raise ValueError("accepted self-collision distance must be nonpositive")
        slack = _finite(
            "collision_acceptance_numerical_slack_m",
            self.collision_acceptance_numerical_slack_m,
            nonnegative=True,
        )
        if slack > 1e-9:
            raise ValueError("collision acceptance slack must be <= 1e-9")
        allowed = {
            SELF_ONLY_FINAL_FEASIBILITY,
            UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
        }
        if self.final_feasibility_algorithm not in allowed:
            raise ValueError("unknown final feasibility algorithm")
        if (
            self.final_feasibility_algorithm
            == UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY
            and self.native_support is None
        ):
            raise ValueError("unified feasibility requires native support settings")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any],
    ) -> "TacoBimanualRetargetSettings":
        expected = {
            "wrist_orientation_cost", "fingertip_position_cost",
            "fingertip_orientation_cost", "max_iterations_per_frame",
            "max_feasibility_iterations", "solver_primal_tolerance",
            "solver_dual_tolerance", "planning_collision_buffer_m",
            "accepted_min_self_collision_distance_m",
            "collision_acceptance_numerical_slack_m", "velocity_limits",
        }
        _exact_keys(value, expected, "taco_bimanual settings")
        return cls(
            wrist_orientation_cost=_finite(
                "wrist_orientation_cost", value["wrist_orientation_cost"],
                nonnegative=True,
            ),
            fingertip_position_cost=_finite(
                "fingertip_position_cost", value["fingertip_position_cost"],
                nonnegative=True,
            ),
            fingertip_orientation_cost=_finite(
                "fingertip_orientation_cost", value["fingertip_orientation_cost"],
                nonnegative=True,
            ),
            max_iterations_per_frame=_integer(
                "max_iterations_per_frame", value["max_iterations_per_frame"],
                positive=True,
            ),
            max_feasibility_iterations=_integer(
                "max_feasibility_iterations", value["max_feasibility_iterations"],
                positive=True,
            ),
            solver_primal_tolerance=_finite(
                "solver_primal_tolerance", value["solver_primal_tolerance"],
                positive=True,
            ),
            solver_dual_tolerance=_finite(
                "solver_dual_tolerance", value["solver_dual_tolerance"],
                positive=True,
            ),
            planning_collision_buffer_m=_finite(
                "planning_collision_buffer_m",
                value["planning_collision_buffer_m"], nonnegative=True,
            ),
            accepted_min_self_collision_distance_m=_finite(
                "accepted_min_self_collision_distance_m",
                value["accepted_min_self_collision_distance_m"],
            ),
            collision_acceptance_numerical_slack_m=_finite(
                "collision_acceptance_numerical_slack_m",
                value["collision_acceptance_numerical_slack_m"],
                nonnegative=True,
            ),
            velocity_limits=VelocityLimits.from_mapping(value["velocity_limits"]),
        )

    def input_dict(self) -> dict[str, Any]:
        return {
            "wrist_orientation_cost": self.wrist_orientation_cost,
            "fingertip_position_cost": self.fingertip_position_cost,
            "fingertip_orientation_cost": self.fingertip_orientation_cost,
            "max_iterations_per_frame": self.max_iterations_per_frame,
            "max_feasibility_iterations": self.max_feasibility_iterations,
            "solver_primal_tolerance": self.solver_primal_tolerance,
            "solver_dual_tolerance": self.solver_dual_tolerance,
            "planning_collision_buffer_m": self.planning_collision_buffer_m,
            "accepted_min_self_collision_distance_m": (
                self.accepted_min_self_collision_distance_m
            ),
            "collision_acceptance_numerical_slack_m": (
                self.collision_acceptance_numerical_slack_m
            ),
            "velocity_limits": self.velocity_limits.to_dict(),
        }


def load_taco_bimanual_settings(
    path: Path,
) -> tuple[TacoBimanualRetargetSettings, dict[str, Any]]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, Mapping):
        raise TypeError("TACO bimanual settings document must be a mapping")
    _exact_keys(document, {"schema", "provenance", "settings"}, "settings document")
    if document["schema"] != SCHEMA:
        raise ValueError("unexpected TACO bimanual settings schema")
    if document["provenance"] != PROVENANCE:
        raise ValueError("unexpected TACO bimanual settings provenance")
    settings = TacoBimanualRetargetSettings.from_mapping(document["settings"])
    audit = {
        "schema": "taco_bimanual_config_consumption_v1",
        "settings_path": str(path.resolve()),
        "declared_key_count": len(document["settings"]),
        "consumed_key_count": len(settings.input_dict()),
        "unknown_key_count": 0,
        "unused_key_count": 0,
        "declared_keys": sorted(document["settings"]),
        "consumed_keys": sorted(settings.input_dict()),
        "status": "PASS",
    }
    return settings, audit
