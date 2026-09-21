"""Explicit actor/critic observation contract for Replay-to-RL runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeObservationContract:
    observation_id: str
    status: str
    paper_faithful: bool
    actor_dim: int
    critic_dim: int
    goal_reference_offset: int
    command_offset: int
    command_preview_offset: int
    include_command_preview: bool
    include_contact_flags: bool
    component_names: tuple[str, ...]
    provenance: str
    profile_path: str
    profile_sha256: str

    def as_report(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return path, document, hashlib.sha256(raw).hexdigest()


def load_runtime_observation(
    protocol_path: str | Path,
    profile_path: str | Path | None,
    *,
    require_run_ready: bool,
) -> RuntimeObservationContract:
    """Load the selected local encoding and reject implicit observations."""
    if profile_path is None:
        raise ValueError(
            "the paper does not publish an exact actor observation; an explicit local profile is required"
        )
    _, protocol, _ = _read_yaml(protocol_path)
    profile_file, profile, profile_hash = _read_yaml(profile_path)
    runtime = protocol.get("runtime_contract", {})
    observation_gate = runtime.get("observation", {}) if isinstance(runtime, dict) else {}
    profiles = observation_gate.get("local_profiles", {}) if isinstance(observation_gate, dict) else {}
    observation_id = profile.get("observation_id")
    if not isinstance(observation_id, str) or observation_id not in profiles:
        raise ValueError("observation profile is not listed by the protocol")
    gate = profiles[observation_id]
    if not isinstance(gate, dict):
        raise ValueError("observation profile gate must be a mapping")
    if gate.get("profile_sha256") != profile_hash:
        raise ValueError("observation profile hash does not match the approved protocol entry")
    if require_run_ready and not gate.get("run_ready", False):
        raise ValueError("observation profile is explicit but its formal run gate is closed")

    if profile.get("status") != "resolved_local_unpublished" or profile.get("paper_faithful") is not False:
        raise ValueError("the active observation must be declared resolved_local_unpublished")
    provenance = profile.get("provenance")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("observation provenance must be explicit")
    transition = profile.get("transition", {})
    if transition != {
        "state_endpoint": "t",
        "action_interval": "t_to_t_plus_1",
        "reward_endpoint": "t_plus_1",
    }:
        raise ValueError("observation transition timestamps are unsupported")

    actor = profile.get("actor", {})
    critic = profile.get("critic", {})
    components = actor.get("components", []) if isinstance(actor, dict) else []
    expected = (
        ("hand_qpos", 36, "t"),
        ("hand_qvel", 36, "t"),
        ("fingertip_positions", 30, "t"),
        ("palm_positions", 6, "t"),
        ("current_object_anchors", 18, "t"),
        ("goal_object_anchors", 18, "t_plus_1"),
        ("reference_ctrl", 36, "t_plus_1"),
        ("reference_ctrl_preview", 36, "t_plus_2"),
        ("contact_flags", 20, "t"),
    )
    actual = tuple(
        (item.get("name"), item.get("dimension"), item.get("timestamp"))
        for item in components if isinstance(item, dict)
    )
    if actual != expected or actor.get("dimension") != sum(item[1] for item in expected):
        raise ValueError("actor observation components/order/dimensions differ from the approved 236-D contract")
    if actor.get("object_order") != ["tool", "target"]:
        raise ValueError("actor object order must be [tool, target]")
    if actor.get("anchor_offsets_m") != [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]:
        raise ValueError("actor anchor encoding differs from the approved contract")
    if actor.get("measured_future_state_used") is not False or actor.get("future_reference_is_known_offline") is not True:
        raise ValueError("observation causality declarations are invalid")
    if actor.get("observation_noise_std") != 0.0:
        raise ValueError("the accepted first run requires zero observation noise")

    critic_components = critic.get("components", []) if isinstance(critic, dict) else []
    if critic.get("asymmetric") is not True or critic.get("dimension") != 108:
        raise ValueError("critic must use the approved 108-D asymmetric contract")
    if sum(item.get("dimension", -1) for item in critic_components if isinstance(item, dict)) != 108:
        raise ValueError("critic component dimensions do not sum to 108")

    return RuntimeObservationContract(
        observation_id=observation_id,
        status=profile["status"],
        paper_faithful=False,
        actor_dim=236,
        critic_dim=108,
        goal_reference_offset=1,
        command_offset=1,
        command_preview_offset=2,
        include_command_preview=True,
        include_contact_flags=True,
        component_names=tuple(item[0] for item in expected),
        provenance=provenance,
        profile_path=str(profile_file),
        profile_sha256=profile_hash,
    )
