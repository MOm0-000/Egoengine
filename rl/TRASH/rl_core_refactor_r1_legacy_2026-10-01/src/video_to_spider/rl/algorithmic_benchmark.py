"""Fail-closed infrastructure for the Pour algorithmic training benchmark.

This module deliberately contains no physics, reward, observation, or action-
support changes.  It only defines the two declared initialization contracts,
step-budget milestones, and a resumable checkpoint envelope.  The benchmark
runner must pass Candidate B's zero-step Replay-preservation gate before any
optimizer update is permitted.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA = "taco_pour_algorithmic_benchmark_checkpoint_v1"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def replay_preserving_initialization(
    agent: Any, *, sigma_multiplier: float = 0.25
) -> dict[str, Any]:
    """Make only the actor mean head zero and reduce initial fixed sigma.

    All other actor parameters, the critic, and both optimizers remain at their
    fresh formal initialization.  ``sigma`` is the learnable log standard
    deviation parameter used by the current actor implementation.
    """

    if not math.isclose(float(sigma_multiplier), 0.25, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Candidate B is frozen to initial sigma multiplier 0.25")
    network = agent.model.a2c_network
    mean = getattr(network, "mu", None)
    sigma = getattr(network, "sigma", None)
    if not isinstance(mean, torch.nn.Linear):
        raise TypeError("Candidate B requires one Linear actor mean head")
    if not isinstance(sigma, torch.nn.Parameter) or not network.fixed_sigma:
        raise TypeError("Candidate B requires the current learnable fixed log-sigma")

    before = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    with torch.no_grad():
        mean.weight.zero_()
        if mean.bias is None:
            raise ValueError("actor mean head has no bias")
        mean.bias.zero_()
        sigma.add_(math.log(sigma_multiplier))
    after = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    changed = [
        name for name in before
        if not torch.equal(before[name], after[name])
    ]
    allowed = {"mu.weight", "mu.bias", "sigma"}
    if set(changed) - allowed:
        raise RuntimeError(f"Candidate B changed undeclared actor tensors: {changed}")
    if not torch.count_nonzero(mean.weight).item() == 0:
        raise RuntimeError("Candidate B mean weight is not exactly zero")
    if not torch.count_nonzero(mean.bias).item() == 0:
        raise RuntimeError("Candidate B mean bias is not exactly zero")
    sigma_value = torch.exp(sigma.detach())
    if not torch.allclose(
        sigma_value,
        torch.full_like(sigma_value, 0.25),
        rtol=0.0,
        atol=2.0e-8,
    ):
        raise RuntimeError("Candidate B initial sigma is not exactly the declared scale")
    return {
        "changed_network_state_keys": changed,
        "mean_weight_sha256": _tensor_sha256(mean.weight),
        "mean_bias_sha256": _tensor_sha256(mean.bias),
        "mean_weight_max_abs": float(mean.weight.detach().abs().max().cpu()),
        "mean_bias_max_abs": float(mean.bias.detach().abs().max().cpu()),
        "initial_sigma_min": float(sigma_value.min().cpu()),
        "initial_sigma_max": float(sigma_value.max().cpu()),
        "sigma_multiplier": float(sigma_multiplier),
        "sigma_remains_learnable": bool(sigma.requires_grad),
    }


def validate_budget_plan(plan: Mapping[str, Mapping[str, int]]) -> list[dict[str, int | str]]:
    """Validate and normalize the three predeclared fixed-horizon milestones."""

    expected = {
        "ckpt_100k": (100_000, 62, 99_200, 9_920),
        "ckpt_500k": (500_000, 313, 500_800, 50_080),
        "ckpt_1m": (1_000_000, 625, 1_000_000, 100_000),
    }
    rows: list[dict[str, int | str]] = []
    for name, values in expected.items():
        row = plan.get(name)
        if row is None:
            raise ValueError(f"missing benchmark milestone {name}")
        observed = (
            int(row["target_physics_steps"]),
            int(row["declared_nearest_epoch"]),
            int(row["actual_physics_steps"]),
            int(row["actual_control_intervals"]),
        )
        if observed != values:
            raise ValueError(f"benchmark milestone {name} changed: {observed}")
        target, epoch, physics, controls = observed
        if physics != epoch * 1_600 or controls != epoch * 160 or physics != controls * 10:
            raise ValueError(f"benchmark milestone {name} has inconsistent counters")
        rows.append({
            "name": name,
            "target_physics_steps": target,
            "epoch": epoch,
            "actual_physics_steps": physics,
            "actual_control_intervals": controls,
        })
    return rows


def capture_rng_states() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def build_checkpoint_payload(
    agent: Any,
    *,
    candidate: str,
    seed: int,
    simulation_physics_steps: int,
    simulation_control_intervals: int,
    config_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Capture the complete declared resume state without writing a chunk."""

    if candidate not in {"A", "B"}:
        raise ValueError("benchmark checkpoint candidate must be A or B")
    critic = getattr(agent, "asymmetric_critic_net", None)
    if critic is None:
        raise ValueError("benchmark checkpoint requires the frozen asymmetric critic")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "candidate": candidate,
        "seed": int(seed),
        "actor": deepcopy(agent.model.state_dict()),
        "critic": deepcopy(critic.state_dict()),
        "actor_optimizer": deepcopy(agent.optimizer.state_dict()),
        "critic_optimizer": deepcopy(critic.optimizer.state_dict()),
        "observation_normalization_version": int(
            agent._observation_normalization_version
        ),
        "simulation_physics_steps": int(simulation_physics_steps),
        "simulation_control_intervals": int(simulation_control_intervals),
        "agent_epoch": int(agent.epoch_num),
        "agent_frame": int(agent.frame),
        "agent_rnn_states": [state.detach().cpu().clone() for state in agent.rnn_states],
        "agent_observation": deepcopy(agent.obs),
        "agent_dones": agent.dones.detach().cpu().clone(),
        "environment": deepcopy(agent.env.get_env_state()),
        "rng_states": capture_rng_states(),
        "config_hashes": dict(config_hashes),
        "chunk_commit_allowed": False,
    }


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema", "candidate", "seed", "actor", "critic", "actor_optimizer",
        "critic_optimizer", "observation_normalization_version",
        "simulation_physics_steps", "simulation_control_intervals", "agent_epoch",
        "agent_frame", "agent_rnn_states", "agent_observation", "agent_dones",
        "environment", "rng_states", "config_hashes", "chunk_commit_allowed",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"benchmark checkpoint is incomplete: {sorted(missing)}")
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["chunk_commit_allowed"] is not False:
        raise ValueError("benchmark checkpoint schema or no-commit contract changed")
    physics = int(payload["simulation_physics_steps"])
    controls = int(payload["simulation_control_intervals"])
    if physics != controls * 10:
        raise ValueError("benchmark checkpoint simulation counters disagree")
    if not payload["config_hashes"]:
        raise ValueError("benchmark checkpoint has no frozen config hashes")


def write_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_checkpoint_payload(payload)
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    raw = buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
    }
