"""Exact paired physics/recurrent boundaries for tail-focused PPO sampling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import torch

from video_to_spider.rl.replay_rl import _model_state_sha256, _snapshot_value_equal


SCHEMA = "egoengine_physics_rnn_boundary_v1"


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return deepcopy(value)


def _batch_observations(rows: Sequence[Any]) -> Any:
    if isinstance(rows[0], dict):
        return {
            key: np.concatenate([np.asarray(row[key]) for row in rows], axis=0)
            for key in rows[0]
        }
    return np.concatenate([np.asarray(row) for row in rows], axis=0)


def make_rollout_start_boundary(
    agent: Any,
    env: Any,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Bind one exact rollout-start physics state to zero recurrent memory."""
    if env.num_envs != 1:
        raise ValueError("rollout-start boundary requires exactly one physical world")
    physics = env.get_env_state()
    endpoint = int(np.asarray(physics["time_indices"])[0])
    if bool(np.asarray(physics["last_terminated"])[0]):
        raise ValueError("terminated physics states cannot become rollout starts")
    default = agent.model.get_default_rnn_state()
    if default is None:
        raise ValueError("the current policy is not recurrent")
    states = tuple(
        _cpu_clone(state[:, :1, :].to(agent.device).zero_()) for state in default
    )
    normalization_keys = sorted(
        name for name in agent.model.state_dict() if "running_mean_std" in name
    )
    if not normalization_keys:
        raise ValueError("actor input-normalization state is not present in the model hash")
    return {
        "schema": SCHEMA,
        "reference_endpoint": endpoint,
        "rollout_start_endpoint": endpoint,
        "actor_state_sha256": _model_state_sha256(agent.model.state_dict()),
        "actor_hash_scope": "full_model_state_including_input_normalization",
        "actor_normalization_state_keys": normalization_keys,
        "physics_state": _cpu_clone(physics),
        "rnn_states": states,
        "agent_runtime": {
            "dones": torch.zeros(1, dtype=torch.uint8),
            "current_rewards": torch.zeros(1, 1, dtype=torch.float32),
            "current_shaped_rewards": torch.zeros(1, 1, dtype=torch.float32),
            "current_lengths": torch.zeros(1, dtype=torch.float32),
        },
        "observation": _cpu_clone(env.current_observation()),
        "observation_prefix": (),
        "provenance": _cpu_clone(provenance),
        "exploration_rng_restored": False,
        "exploration_rng_reason": (
            "the rollout start fixes physics and recurrent memory but each training "
            "branch draws an independent stochastic action"
        ),
    }


def capture_physics_rnn_boundary(
    agent: Any,
    env: Any,
    *,
    rollout_start_endpoint: int,
    observation_prefix: Sequence[Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Capture one naturally reached world and the actor memory that belongs to it."""
    if env.num_envs != 1:
        raise ValueError("capture requires exactly one physical world")
    if agent.rnn_states is None:
        raise ValueError("the current policy is not recurrent")
    if any(state.ndim != 3 or state.shape[1] != 1 for state in agent.rnn_states):
        raise ValueError("capture requires one-world recurrent states")
    physics = env.get_env_state()
    endpoint = int(np.asarray(physics["time_indices"])[0])
    if endpoint < rollout_start_endpoint:
        raise ValueError("boundary cannot precede the rollout start")
    if len(observation_prefix) != endpoint - rollout_start_endpoint:
        raise ValueError(
            "observation prefix must contain one source observation per transition"
        )
    if bool(np.asarray(physics["last_terminated"])[0]):
        raise ValueError("terminated physics states cannot become curriculum boundaries")
    if agent.dones.shape != (1,) or bool(agent.dones[0]):
        raise ValueError("curriculum boundary must be a nonterminal recurrent state")

    normalization_keys = sorted(
        name for name in agent.model.state_dict() if "running_mean_std" in name
    )
    if not normalization_keys:
        raise ValueError("actor input-normalization state is not present in the model hash")
    return {
        "schema": SCHEMA,
        "reference_endpoint": endpoint,
        "rollout_start_endpoint": int(rollout_start_endpoint),
        "actor_state_sha256": _model_state_sha256(agent.model.state_dict()),
        "actor_hash_scope": "full_model_state_including_input_normalization",
        "actor_normalization_state_keys": normalization_keys,
        "physics_state": _cpu_clone(physics),
        "rnn_states": tuple(_cpu_clone(state) for state in agent.rnn_states),
        "agent_runtime": {
            "dones": _cpu_clone(agent.dones),
            "current_rewards": _cpu_clone(agent.current_rewards),
            "current_shaped_rewards": _cpu_clone(agent.current_shaped_rewards),
            "current_lengths": _cpu_clone(agent.current_lengths),
        },
        "observation": _cpu_clone(env.current_observation()),
        "observation_prefix": _cpu_clone(tuple(observation_prefix)),
        "provenance": _cpu_clone(provenance),
        "exploration_rng_restored": False,
        "exploration_rng_reason": (
            "the boundary preserves physical and policy-memory semantics; each new "
            "training branch must draw an independent stochastic action"
        ),
    }


def refresh_boundary_rnn_for_actor(agent: Any, boundary: dict[str, Any]) -> dict[str, Any]:
    """Recompute one boundary's memory from its natural observation history."""
    if boundary.get("schema") != SCHEMA:
        raise ValueError(f"paired boundary schema must be {SCHEMA}")
    expected_length = (
        int(boundary["reference_endpoint"])
        - int(boundary["rollout_start_endpoint"])
    )
    prefix = tuple(boundary.get("observation_prefix", ()))
    if len(prefix) != expected_length:
        raise ValueError("paired boundary lacks its complete natural observation prefix")
    default = agent.model.get_default_rnn_state()
    if default is None:
        raise ValueError("the current policy is not recurrent")
    model_hash_before = _model_state_sha256(agent.model.state_dict())
    previous = agent.rnn_states
    try:
        agent.set_eval()
        agent.rnn_states = [
            state[:, :1, :].to(agent.device).zero_() for state in default
        ]
        for observation in prefix:
            result = agent.get_action_values(agent.obs_to_tensors(observation))
            agent.rnn_states = result["rnn_states"]
        refreshed_states = tuple(_cpu_clone(state) for state in agent.rnn_states)
    finally:
        agent.rnn_states = previous

    model_hash_after = _model_state_sha256(agent.model.state_dict())
    if model_hash_after != model_hash_before:
        raise RuntimeError(
            "observation-prefix replay changed actor or input-normalization state"
        )

    refreshed = _cpu_clone(boundary)
    old_hash = boundary["actor_state_sha256"]
    new_hash = model_hash_after
    refreshed["actor_state_sha256"] = new_hash
    refreshed["rnn_states"] = refreshed_states
    refreshed["provenance"]["rnn_refresh"] = {
        "method": "replay_natural_observation_prefix_under_current_actor",
        "source_actor_state_sha256": old_hash,
        "refreshed_actor_state_sha256": new_hash,
        "prefix_observations": len(prefix),
        "physical_state_changed": False,
        "actor_and_input_normalization_unchanged_during_replay": True,
        "actor_state_sha256_before_replay": model_hash_before,
        "actor_state_sha256_after_replay": model_hash_after,
    }
    return refreshed


def restore_physics_rnn_boundaries(
    agent: Any,
    env: Any,
    boundaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Restore one paired boundary per world, rejecting stale actor memories."""
    boundaries = tuple(boundaries)
    if len(boundaries) != int(env.num_envs):
        raise ValueError("one paired boundary is required for every physical world")
    actor_hash = _model_state_sha256(agent.model.state_dict())
    endpoints = []
    for boundary in boundaries:
        if boundary.get("schema") != SCHEMA:
            raise ValueError(f"paired boundary schema must be {SCHEMA}")
        if boundary.get("actor_state_sha256") != actor_hash:
            raise ValueError(
                "paired RNN state belongs to a different actor/normalization state"
            )
        endpoint = int(boundary["reference_endpoint"])
        if int(np.asarray(boundary["physics_state"]["time_indices"])[0]) != endpoint:
            raise ValueError("paired boundary endpoint and physics cursor differ")
        if bool(np.asarray(boundary["physics_state"]["last_terminated"])[0]):
            raise ValueError("terminated paired boundary is forbidden")
        endpoints.append(endpoint)

    worlds = getattr(env, "worlds", None)
    if worlds is None:
        if len(boundaries) != 1:
            raise ValueError("a single MJWP instance accepts one paired boundary")
        env.set_env_state(boundaries[0]["physics_state"])
        restored_physics = (env.get_env_state(),)
    else:
        if len(worlds) != len(boundaries):
            raise ValueError("independent-world count differs from paired boundaries")
        for world, boundary in zip(worlds, boundaries, strict=True):
            world.set_env_state(boundary["physics_state"])
        restored_physics = tuple(world.get_env_state() for world in worlds)

    physics_equal = [
        _snapshot_value_equal(boundary["physics_state"], restored)
        for boundary, restored in zip(boundaries, restored_physics, strict=True)
    ]
    if not all(physics_equal):
        raise ValueError("paired physical state did not restore bitwise")

    state_count = len(boundaries[0]["rnn_states"])
    if any(len(boundary["rnn_states"]) != state_count for boundary in boundaries):
        raise ValueError("paired boundaries disagree on recurrent-state count")
    agent.rnn_states = [
        torch.cat(
            [boundary["rnn_states"][index] for boundary in boundaries], dim=1
        ).to(agent.device)
        for index in range(state_count)
    ]
    runtime = {}
    for key in ("dones", "current_rewards", "current_shaped_rewards", "current_lengths"):
        runtime[key] = torch.cat(
            [boundary["agent_runtime"][key] for boundary in boundaries], dim=0
        ).to(agent.device)
        setattr(agent, key, runtime[key])

    expected_observation = _batch_observations([
        boundary["observation"] for boundary in boundaries
    ])
    actual_observation = env.current_observation()
    if not _snapshot_value_equal(expected_observation, actual_observation):
        raise ValueError("observation rebuilt from restored physics differs from capture")
    agent.obs = agent.obs_to_tensors(actual_observation)

    if getattr(agent, "has_asymmetric_critic", False):
        critic = agent.asymmetric_critic_net
        if critic.is_rnn:
            raise ValueError("recurrent asymmetric critic state is not supported by this contract")

    return {
        "schema": SCHEMA,
        "worlds": len(boundaries),
        "reference_endpoints": endpoints,
        "actor_state_sha256": actor_hash,
        "physics_bitwise_equal": physics_equal,
        "observation_bitwise_equal": True,
        "rnn_state_shapes": [list(state.shape) for state in agent.rnn_states],
        "agent_runtime_shapes": {
            key: list(value.shape) for key, value in runtime.items()
        },
        "exploration_rng_restored": False,
    }
