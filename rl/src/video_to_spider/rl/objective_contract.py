"""Explicit runtime objective selection for Replay→RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from egoengine_repro.action.paper_rewards import TrackingObjective


@dataclass(frozen=True)
class RuntimeObjective:
    objective_id: str
    status: str
    paper_faithful: bool
    tracking: TrackingObjective
    contact_coefficient: float
    lift_coefficient: float
    lift_object_role: str
    contact_reduction: str
    aggregation: str
    tracking_metric_name: str
    independent_position_threshold_m: float | None
    independent_rotation_threshold_rad: float | None
    provenance: str
    protocol_path: str
    protocol_sha256: str
    profile_path: str | None
    profile_sha256: str | None

    def as_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["tracking"]["lambda_R"] = report["tracking"].pop("lambda_r")
        report["tracking"]["C"] = report["tracking"].pop("boundary")
        return report


def _read_yaml(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return path, document, hashlib.sha256(raw).hexdigest()


def _number(mapping: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"objective field {key} must be an explicit number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"objective field {key} has an invalid value: {value}")
    return value


def load_runtime_objective(
    protocol_path: str | Path,
    profile_path: str | Path | None,
    *,
    tracking_variant: str,
    require_run_ready: bool,
) -> RuntimeObjective:
    """Resolve one objective without deriving unpublished values.

    With no profile, the request is paper-faithful and every paper field must
    be present in the protocol. A local run requires a separate, explicit
    profile listed by the protocol. Formal task runs additionally require the
    profile-specific run gate to be open.
    """
    if tracking_variant not in {"tool_only", "tool_and_target"}:
        raise ValueError(f"unsupported tracking variant: {tracking_variant}")
    protocol_file, protocol, protocol_hash = _read_yaml(protocol_path)
    runtime = protocol.get("runtime_contract", {})
    if not isinstance(runtime, dict):
        raise ValueError("protocol runtime_contract must be a mapping")

    if profile_path is None:
        if tracking_variant != "tool_only":
            raise ValueError("tool_and_target is a local extension, not a paper-faithful objective")
        tracking = protocol.get("tracking", {})
        contact = protocol.get("contact", {})
        lift = protocol.get("lift", {})
        unresolved = [
            name for name, value in (
                ("lambda_p", tracking.get("lambda_p")),
                ("lambda_R", tracking.get("lambda_R")),
                ("C", tracking.get("C")),
                ("contact.coefficient", contact.get("coefficient")),
                ("contact.reduction", contact.get("reduction")),
                ("lift.coefficient", lift.get("coefficient")),
                ("lift.object_role", lift.get("object_role")),
            ) if value is None
        ]
        if unresolved:
            raise ValueError(
                "paper-faithful objective is unresolved: " + ", ".join(unresolved)
            )
        if require_run_ready and not runtime.get("paper_faithful_run_ready", False):
            raise ValueError("paper-faithful run is blocked by the protocol")
        profile_file = None
        profile_hash = None
        source = {
            "objective_id": "egoengine_paper_objective",
            "status": "resolved_paper",
            "paper_faithful": True,
            "provenance": "EgoEngine Appendix C",
            "tracking": {
                "lambda_p": tracking["lambda_p"],
                "lambda_R": tracking["lambda_R"],
                "C": tracking["C"],
            },
            "contact": {"coefficient": contact["coefficient"], "reduction": contact["reduction"]},
            "lift": {"coefficient": lift["coefficient"], "object_role": lift["object_role"]},
            "aggregation": {tracking_variant: "single_object"},
        }
    else:
        profile_file, source, profile_hash = _read_yaml(profile_path)
        objective_id = source.get("objective_id")
        approved = runtime.get("local_objective_profiles", {})
        if not isinstance(objective_id, str) or objective_id not in approved:
            raise ValueError("local objective profile is not listed by the protocol")
        gate = approved[objective_id]
        if not isinstance(gate, dict) or not gate.get("objective_ready", False):
            raise ValueError("local objective profile is not approved as a resolved objective")
        if require_run_ready and not gate.get("run_ready", False):
            blockers = ", ".join(gate.get("blocked_by", ()))
            raise ValueError(f"local objective is explicit but formal run remains blocked: {blockers}")
        if require_run_ready and gate.get("profile_sha256") != profile_hash:
            raise ValueError("local objective profile does not match the protocol-bound hash")

    objective_id = source.get("objective_id")
    if not isinstance(objective_id, str) or not objective_id:
        raise ValueError("objective_id must be a nonempty string")
    if source.get("status") not in {"resolved_paper", "resolved_local_unpublished"}:
        raise ValueError("objective status must explicitly declare how it was resolved")
    if not isinstance(source.get("paper_faithful"), bool):
        raise ValueError("paper_faithful must be explicit")
    provenance = source.get("provenance")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("objective provenance must be explicit")
    tracking = source.get("tracking", {})
    contact = source.get("contact", {})
    lift = source.get("lift", {})
    aggregation = source.get("aggregation", {})
    diagnostics = source.get("diagnostics", {})
    if not all(isinstance(value, dict) for value in (tracking, contact, lift, aggregation, diagnostics)):
        raise ValueError("tracking/contact/lift/aggregation/diagnostics must be mappings")
    tracking_metric_name = tracking.get("metric_name", "weighted_tracking_error")
    if not isinstance(tracking_metric_name, str) or not tracking_metric_name:
        raise ValueError("tracking metric_name must be a nonempty string")
    independent = diagnostics.get("independent_thresholds")
    if independent is None:
        independent_position_threshold_m = None
        independent_rotation_threshold_rad = None
    else:
        if not isinstance(independent, dict):
            raise ValueError("diagnostics independent_thresholds must be a mapping")
        independent_position_threshold_m = _number(independent, "position_m", positive=True)
        independent_rotation_threshold_rad = _number(independent, "rotation_rad", positive=True)
    reduction = contact.get("reduction")
    if not isinstance(reduction, str) or not reduction:
        raise ValueError("contact reduction must be explicit")
    lift_object_role = lift.get("object_role")
    if not isinstance(lift_object_role, str) or not lift_object_role:
        raise ValueError("lift object_role must be explicit")
    aggregation_value = aggregation.get(tracking_variant)
    expected = "single_object" if tracking_variant == "tool_only" else "mean_reward_any_termination"
    if aggregation_value != expected:
        raise ValueError(
            f"{tracking_variant} aggregation must be explicitly set to {expected}"
        )
    if require_run_ready:
        observation = runtime.get("observation", {})
        if not isinstance(observation, dict) or not observation.get("formal_run_ready", False):
            raise ValueError("formal run is blocked by the unresolved observation contract")
        if not protocol.get("training_ready", False):
            raise ValueError("formal run is blocked because protocol training_ready is false")
    return RuntimeObjective(
        objective_id=objective_id,
        status=source["status"],
        paper_faithful=source["paper_faithful"],
        tracking=TrackingObjective(
            lambda_p=_number(tracking, "lambda_p"),
            lambda_r=_number(tracking, "lambda_R"),
            boundary=_number(tracking, "C", positive=True),
        ),
        contact_coefficient=_number(contact, "coefficient"),
        lift_coefficient=_number(lift, "coefficient"),
        lift_object_role=lift_object_role,
        contact_reduction=reduction,
        aggregation=aggregation_value,
        tracking_metric_name=tracking_metric_name,
        independent_position_threshold_m=independent_position_threshold_m,
        independent_rotation_threshold_rad=independent_rotation_threshold_rad,
        provenance=provenance,
        protocol_path=str(protocol_file),
        protocol_sha256=protocol_hash,
        profile_path=str(profile_file) if profile_file else None,
        profile_sha256=profile_hash,
    )
