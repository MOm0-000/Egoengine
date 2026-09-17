"""Configuration loading and validation for reproduction profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_SECTIONS = {
    "profile", "inputs", "perception", "retarget", "reward", "chunks", "evaluation",
}


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay a small diagnostic profile on its audited base."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_config(previous, value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path, ancestry: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in ancestry:
        chain = " -> ".join(str(item) for item in (*ancestry, resolved))
        raise ValueError(f"config inheritance cycle: {chain}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must contain a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent:
        raise ValueError("config extends must be a nonempty relative path")
    parent_path = (resolved.parent / parent).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"extended config does not exist: {parent_path}")
    return _merge_config(_read_config(parent_path, (*ancestry, resolved)), raw)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class ReproConfig:
    """Validated immutable reproduction configuration."""

    path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def name(self) -> str:
        return str(self.data["profile"]["name"])

    @property
    def uses_ground_truth(self) -> bool:
        return bool(self.data["profile"].get("uses_ground_truth", False))


def bundled_config_path(name: str) -> Path:
    if name not in {
        "paper_faithful", "paper_faithful_taco",
        "auto_current", "auto_current_mano_direction",
    }:
        raise ValueError(f"unknown bundled config: {name}")
    return Path(str(files("egoengine_repro").joinpath("configs", f"{name}.yaml")))


def load_config(path_or_name: str | Path) -> ReproConfig:
    candidate = Path(path_or_name)
    path = candidate if candidate.exists() else bundled_config_path(str(path_or_name))
    data = _read_config(path)
    missing = REQUIRED_SECTIONS - set(data)
    if missing:
        raise ValueError(f"config missing sections: {sorted(missing)}")
    profile = data["profile"]
    if not isinstance(profile, dict) or not profile.get("name"):
        raise ValueError("profile.name is required")
    chunk_length = int(data["chunks"].get("length", 0))
    lookahead = int(data["chunks"].get("lookahead_chunks", 0))
    if chunk_length <= 0 or lookahead < 1:
        raise ValueError("chunks.length must be positive and lookahead_chunks >= 1")
    digest = hashlib.sha256(_canonical_json(data).encode("ascii")).hexdigest()
    return ReproConfig(path=path.resolve(), data=data, sha256=digest)
