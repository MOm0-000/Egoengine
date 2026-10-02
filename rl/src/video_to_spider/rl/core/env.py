"""Actor-free adapter around the audited CPU MuJoCo-Warp environment."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spider.config import Config, load_config_yaml, process_config

from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.mjwp_env import MJWPVectorEnv, MJWPVectorEnvConfig
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation
from video_to_spider.rl.object_assistance import ToolAssistSpec

from .state_io import validate_physics_snapshot


def load_ego_config(path: str | Path, *, device: str = "cpu") -> Config:
    raw = load_config_yaml(str(path))
    allowed = {field.name for field in fields(Config)}
    data = {key: value for key, value in raw.items() if key in allowed}
    for key in ("pair_margin_range", "xy_offset_range"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    if data.get("noise_scale") is None:
        data.pop("noise_scale", None)
    data["device"] = device
    return process_config(Config(**data))


def load_reference(
    path: str | Path, *, device: str = "cpu", expected_frequency: float = 30.0
) -> tuple[torch.Tensor, ...]:
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in ("qpos", "qvel", "ctrl") if name not in source]
        if missing:
            raise ValueError(f"reference is missing {missing}")
        qpos, qvel, ctrl = (
            np.asarray(source[name], dtype=np.float32) for name in ("qpos", "qvel", "ctrl")
        )
        frequency = float(np.asarray(source.get("frequency", np.nan)))
        contact = np.asarray(source.get("contact", np.zeros((len(qpos), 10))), dtype=np.float32)
        contact_pos = np.asarray(
            source.get("contact_pos", np.zeros((len(qpos), 10, 3))), dtype=np.float32
        )
    if not (qpos.ndim == qvel.ndim == ctrl.ndim == 2 and len(qpos) == len(qvel) == len(ctrl)):
        raise ValueError("reference arrays have incompatible shapes")
    if not np.isclose(frequency, expected_frequency):
        raise ValueError("reference control frequency changed")
    if contact.shape != (len(qpos), 10) or contact_pos.shape != (len(qpos), 10, 3):
        raise ValueError("reference contact arrays have incompatible shapes")
    return tuple(
        torch.as_tensor(value, device=device)
        for value in (qpos, qvel, ctrl, contact, contact_pos)
    )


def make_world(
    *,
    simulator_config: str | Path,
    protocol: str | Path,
    objective_profile: str | Path,
    observation_profile: str | Path,
    action_profile: str | Path,
    boundary: dict[str, Any],
    seed: int,
    asymmetric_critic: bool,
    object_assistance: ToolAssistSpec | None = None,
) -> MJWPVectorEnv:
    config = load_ego_config(simulator_config)
    reference = load_reference(config.data_path)
    objective = load_runtime_objective(
        Path(protocol), Path(objective_profile), tracking_variant="tool_only", require_run_ready=False
    )
    observation = load_runtime_observation(
        Path(protocol), Path(observation_profile), require_run_ready=False
    )
    residual, _ = load_residual_action_profile(Path(action_profile))
    world = MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=asymmetric_critic,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,),
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
            residual=residual,
            object_assistance=object_assistance,
        ),
        seed=seed,
    )
    validate_physics_snapshot(boundary)
    world.set_env_state(boundary)
    world.set_chunk_reset(start=40, end=80)
    world.enable_state_feasible_action_contract(reference_snap_tolerance=2.0e-7)
    return world


class IndependentWorlds:
    """Four actual single-world instances; no packed contact buffer."""

    def __init__(self, worlds: list[MJWPVectorEnv]) -> None:
        if not worlds or any(world.num_envs != 1 for world in worlds):
            raise ValueError("IndependentWorlds requires single-world environments")
        self.worlds = tuple(worlds)

    def current_observation(self) -> dict[str, np.ndarray] | np.ndarray:
        rows = [world.current_observation() for world in self.worlds]
        if isinstance(rows[0], dict):
            return {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}
        return np.concatenate(rows, axis=0)

    def normalized_action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = [world.current_normalized_action_bounds() for world in self.worlds]
        return torch.cat([pair[0] for pair in pairs]), torch.cat([pair[1] for pair in pairs])

    def current_reference_endpoints(self) -> torch.Tensor:
        """Read each world's current physical reference cursor without mutation."""
        endpoints = []
        for world in self.worlds:
            start = torch.as_tensor(world.start_indices, dtype=torch.int64).reshape(-1)
            time = torch.as_tensor(world.time_indices, dtype=torch.int64).reshape(-1)
            if start.numel() != 1 or time.numel() != 1:
                raise ValueError("independent world reference cursor is not scalar")
            endpoints.append(start + time)
        return torch.cat(endpoints)

    @property
    def assistance_alpha(self) -> float:
        values = {world.assistance_alpha for world in self.worlds}
        if len(values) != 1:
            raise RuntimeError("independent worlds do not share one assistance alpha")
        return values.pop()

    def set_assistance_alpha(self, alpha: float) -> None:
        for world in self.worlds:
            world.set_assistance_alpha(alpha)

    def assistance_manifests(self) -> list[dict[str, Any]]:
        return [world.assistance_manifest() for world in self.worlds]

    def states(self) -> list[dict[str, Any]]:
        return [world.get_env_state() for world in self.worlds]

    def restore_all(self, state: dict[str, Any]) -> None:
        validate_physics_snapshot(state)
        for world in self.worlds:
            world.set_env_state(state)

    def step(self, actions: torch.Tensor, *, auto_reset: bool = False):
        if actions.shape[0] != len(self.worlds):
            raise ValueError("one action row is required per world")
        rows = [
            world.step(actions[index : index + 1], auto_reset=auto_reset)
            for index, world in enumerate(self.worlds)
        ]
        observations = rows[0][0]
        if isinstance(observations, dict):
            observation = {
                key: np.concatenate([row[0][key] for row in rows], axis=0)
                for key in observations
            }
        else:
            observation = np.concatenate([row[0] for row in rows], axis=0)
        reward = np.concatenate([np.asarray(row[1]) for row in rows])
        done = np.concatenate([np.asarray(row[2]) for row in rows])
        info = {key: np.concatenate([np.asarray(row[3][key]) for row in rows], axis=0) for key in rows[0][3]}
        return observation, reward, done, info
