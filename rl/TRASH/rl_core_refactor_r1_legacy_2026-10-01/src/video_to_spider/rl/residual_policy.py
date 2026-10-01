"""Lightweight xHand residual-policy inference wrapper for H2S2R PPO.

The training entry point (:file:`scripts/run_mjwp_ppo.py`) writes a
``train_metadata.json`` sidecar next to the PPO checkpoints.  This module
reconstructs the exact actor/asymmetric-critic configuration from that sidecar,
loads a checkpoint through the official :class:`PpoAgent.restore` path, and
exposes a small callable used by :file:`run_mjwp_modeswitch.py`.

No file in the cloned H2S2R repository is modified.  The wrapper only performs
inference against the public ``PpoAgent`` API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import gym
    from gym import spaces
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    gym = None
    spaces = None

from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig
from human2sim2robot.ppo.utils.asymmetric_critic import AsymmetricCriticConfig
from human2sim2robot.ppo.utils.network import MlpConfig, NetworkConfig, RnnConfig
from human2sim2robot.ppo.utils.rewards_shaper import RewardsShaperParams


class _StatefulEnv:
    """Minimal env descriptor used only to instantiate and restore PpoAgent."""

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int) -> None:
        assert spaces is not None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def get_env_info(self) -> dict[str, Any]:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
            "agents": 1,
            "value_size": 1,
        }

    def get_env_state(self) -> None:
        return None

    def set_env_state(self, state: Any) -> None:
        return None


def _build_network_config(metadata: dict[str, Any]) -> NetworkConfig:
    net = metadata["network"]
    rnn = net.get("rnn")
    return NetworkConfig(
        mlp=MlpConfig(units=net["mlp"]["units"]),
        rnn=(
            RnnConfig(
                units=rnn["units"],
                layers=rnn["layers"],
                name=rnn["name"],
                layer_norm=rnn.get("layer_norm", False),
                before_mlp=rnn.get("before_mlp", False),
                concat_input=rnn.get("concat_input", False),
                concat_output=rnn.get("concat_output", False),
            )
            if rnn is not None
            else None
        ),
        separate_value_mlp=net.get("separate_value_mlp", False),
        asymmetric_critic=net.get("asymmetric_critic", False),
    )


def _build_asymmetric_critic_config(
    metadata: dict[str, Any], batch_size: int
) -> AsymmetricCriticConfig | None:
    cfg = metadata.get("asymmetric_critic")
    if cfg is None:
        return None
    net = cfg["network"]
    rnn = net.get("rnn")
    return AsymmetricCriticConfig(
        name=cfg["name"],
        network=NetworkConfig(
            mlp=MlpConfig(units=net["mlp"]["units"]),
            rnn=(
                RnnConfig(
                    units=rnn["units"],
                    layers=rnn["layers"],
                    name=rnn["name"],
                    layer_norm=rnn.get("layer_norm", False),
                    before_mlp=rnn.get("before_mlp", False),
                    concat_input=rnn.get("concat_input", False),
                    concat_output=rnn.get("concat_output", False),
                )
                if rnn is not None
                else None
            ),
            separate_value_mlp=net.get("separate_value_mlp", False),
            asymmetric_critic=net.get("asymmetric_critic", True),
        ),
        normalize_input=cfg["normalize_input"],
        learning_rate=cfg["learning_rate"],
        mini_epochs=cfg["mini_epochs"],
        truncate_grads=cfg.get("truncate_grads", False),
        minibatch_size=cfg.get("minibatch_size", batch_size),
    )


def _build_ppo_config(
    metadata: dict[str, Any],
    *,
    device: str,
    asymmetric_critic: AsymmetricCriticConfig | None,
) -> PpoConfig:
    num_envs = metadata["num_envs"]
    horizon_length = metadata["horizon_length"]
    seq_length = metadata["seq_length"]
    batch_size = num_envs * horizon_length
    return PpoConfig(
        num_actors=num_envs,
        learning_rate=metadata["learning_rate"],
        entropy_coef=0.0,
        horizon_length=horizon_length,
        normalize_advantage=True,
        normalize_input=True,
        grad_norm=1.0,
        critic_coef=4,
        gamma=0.998,
        tau=0.95,
        reward_shaper=RewardsShaperParams(scale_value=metadata.get("reward_shaper_scale", 1.0)),
        mini_epochs=4,
        e_clip=0.2,
        multi_gpu=False,
        device=device,
        weight_decay=0.0,
        asymmetric_critic=asymmetric_critic,
        truncate_grads=True,
        save_frequency=0,
        save_best_after=100,
        print_stats=False,
        max_epochs=-1,
        max_frames=-1,
        lr_schedule=None,
        schedule_type="legacy",
        kl_threshold=None,
        seq_length=seq_length,
        bptt_length=seq_length,
        zero_rnn_on_done=True,
        normalize_rms_advantage=False,
        normalize_value=True,
        games_to_track=100,
        minibatch_size_per_env=0,
        minibatch_size=batch_size,
        mixed_precision=False,
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        value_bootstrap=False,
        adv_rms_momentum=0.5,
        clip_actions=True,
        schedule_entropy=False,
        freeze_critic=False,
    )


@dataclass
class ResidualPolicy:
    """Deterministic (or sampled) xHand residual action predictor."""

    checkpoint_path: Path
    train_metadata_path: Path
    device: str = "cuda:0"
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.metadata = json.loads(Path(self.train_metadata_path).read_text(encoding="utf-8"))
        self.obs_dim = int(self.metadata["obs_dim"])
        self.action_dim = int(self.metadata["action_dim"])
        self.state_dim = int(self.metadata["state_dim"])
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        num_envs = int(self.metadata["num_envs"])
        horizon_length = int(self.metadata["horizon_length"])
        batch_size = num_envs * horizon_length
        asymmetric_critic = _build_asymmetric_critic_config(self.metadata, batch_size)
        ppo_config = _build_ppo_config(
            self.metadata,
            device=self.device,
            asymmetric_critic=asymmetric_critic,
        )
        network_config = _build_network_config(self.metadata)
        self._agent = PpoAgent(
            experiment_dir=Path(self.checkpoint_path).parent,
            ppo_config=ppo_config,
            network_config=network_config,
            env=_StatefulEnv(self.obs_dim, self.action_dim, self.state_dim),
        )
        # H2S2R's official ``load_checkpoint`` uses PyTorch 2.6's new
        # ``weights_only=True`` default, which rejects the NumPy arrays stored
        # in the env-state portion of PpoAgent checkpoints.  We load the trusted
        # local checkpoint directly with ``weights_only=False`` and feed the
        # same ``set_full_state_weights`` path used by ``PpoAgent.restore``.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self._agent.set_full_state_weights(checkpoint, set_epoch=False)
        self._agent.init_tensors()
        self._agent.is_tensor_obses = True
        if self._agent.rnn_states is not None:
            self._agent.rnn_states = [
                state[:, :1, :].contiguous() for state in self._agent.rnn_states
            ]
        self._agent.set_eval()

    def predict_delta(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs must have shape ({self.obs_dim},), got {obs.shape}")
        if state.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), got {state.shape}")
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        result = self._agent.get_action_values({"obs": obs_t, "states": state_t})
        self._agent.rnn_states = result["rnn_states"]
        selected = result["mus"] if self.deterministic else result["actions"]
        processed = self._agent.preprocess_actions(selected)
        return processed.detach().cpu().numpy().reshape(self.action_dim)


def load_residual_policy(
    checkpoint_path: str | Path,
    train_metadata_path: str | Path,
    *,
    device: str = "cuda:0",
    deterministic: bool = True,
) -> ResidualPolicy:
    """Load a trained residual policy from a checkpoint and its sidecar."""
    return ResidualPolicy(
        checkpoint_path=Path(checkpoint_path),
        train_metadata_path=Path(train_metadata_path),
        device=device,
        deterministic=deterministic,
    )
