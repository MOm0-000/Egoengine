"""Artifact, checkpoint, full-physics snapshot and RNG I/O for the core path."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


CORE_CHECKPOINT_SCHEMA = "egoengine_rl_core_checkpoint_v1"
LEGACY_G_CHECKPOINT_SCHEMA = "taco_pour_algorithmic_benchmark_checkpoint_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: str | Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    observed = sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(f"artifact hash mismatch for {resolved}: {observed}")
    return resolved


def load_torch_gzip(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rb") as source:
        value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("torch artifact must contain a dictionary")
    return value


def write_torch_gzip(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as destination:
            torch.save(value, destination)


def capture_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_states(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != required:
        raise ValueError("RNG state payload is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda_all"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def validate_physics_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail closed before handing a full snapshot to MJWP.

    The v3 payload declares every runtime/Warp field in ``warp_state_keys``.
    This guard intentionally rejects a declaration whose corresponding payload
    field is absent; partial restore is never accepted.
    """
    if snapshot.get("snapshot_schema") != "egoengine_mjwp_snapshot_v3_reward_aligned":
        raise ValueError("unsupported MJWP snapshot schema")
    keys = snapshot.get("warp_state_keys")
    if not isinstance(keys, (tuple, list)) or not keys:
        raise ValueError("snapshot has no declared Warp state keys")
    missing = [name for name in keys if name not in snapshot]
    if missing:
        raise ValueError(f"snapshot is missing declared Warp state keys: {missing[:8]}")
    required = {
        "time_indices", "start_indices", "episode_lengths", "rng_state",
        "last_action", "last_ctrl", "qpos", "qvel", "ctrl",
        "contact.geom", "contact.worldid", "efc.force",
        "prev.qpos", "prev.qvel", "prev.ctrl",
    }
    absent = sorted(required - set(snapshot))
    if absent:
        raise ValueError(f"snapshot is missing required fields: {absent}")


def load_boundary_context(path: str | Path) -> dict[str, Any]:
    context = load_torch_gzip(path)
    if context.get("schema") != "egoengine_physics_rnn_boundary_v1":
        raise ValueError("unsupported boundary-context schema")
    if int(context.get("reference_endpoint", -1)) != 40:
        raise ValueError("R1 requires the endpoint-40 boundary context")
    if len(context.get("observation_prefix", ())) != 20:
        raise ValueError("boundary context must contain source20--39 observations")
    validate_physics_snapshot(context["physics_state"])
    return context


def import_legacy_g_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read the one recognized historical schema without mutating it."""
    payload = load_torch_gzip(path)
    if payload.get("schema") != LEGACY_G_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported legacy checkpoint schema")
    if payload.get("candidate") not in {"D", "G"}:
        raise ValueError("legacy checkpoint is outside the active D/G lineage")
    required = {
        "actor", "critic", "actor_optimizer", "critic_optimizer",
        "observation_normalization_version", "rng_states",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"legacy checkpoint is incomplete: {missing}")
    return payload


def build_core_checkpoint(
    *,
    policy: Any,
    environment_state: dict[str, Any],
    rnn_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    cursor: dict[str, int],
    cost: dict[str, int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    validate_physics_snapshot(environment_state)
    return {
        "schema": CORE_CHECKPOINT_SCHEMA,
        "actor": deepcopy(policy.actor.state_dict()),
        "critic": deepcopy(policy.critic.state_dict()) if policy.critic is not None else None,
        "actor_optimizer": deepcopy(policy.actor_optimizer.state_dict()),
        "critic_optimizer": deepcopy(policy.critic.optimizer.state_dict())
        if policy.critic is not None
        else None,
        "observation_normalization_version": int(policy.normalization_version),
        "rnn_states": tuple(state.detach().cpu().clone() for state in rnn_states),
        "environment": deepcopy(environment_state),
        "cursor": {name: int(value) for name, value in cursor.items()},
        "cost": {name: int(value) for name, value in cost.items()},
        "rng_states": capture_rng_states(),
        "metadata": deepcopy(metadata),
        "training_enabled": False,
        "chunk_commit_enabled": False,
    }


def validate_core_checkpoint(payload: dict[str, Any]) -> None:
    if payload.get("schema") != CORE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported core checkpoint schema")
    validate_physics_snapshot(payload["environment"])
    for name in ("actor", "actor_optimizer", "rnn_states", "cursor", "cost", "rng_states"):
        if name not in payload:
            raise ValueError(f"core checkpoint is missing {name}")
    if payload.get("training_enabled") is not False or payload.get("chunk_commit_enabled") is not False:
        raise ValueError("R1 checkpoint must remain non-training and non-committing")


def manifest_entry(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
