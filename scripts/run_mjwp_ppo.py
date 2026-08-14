"""Minimal xHand PPO smoke trainer using the official H2S2R ``PpoAgent``.

This script does not reimplement PPO.  It wraps the cloned Human2Sim2Robot
trainer around ``video_to_spider.rl.mjwp_env.MJWPVectorEnv``.  The policy action
space is the EgoEngine residual ``delta_u`` on the 18-DoF xHand reference
control, not H2S2R's palm/PCA fabric action.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_V2S_ROOT = Path(os.environ.get("VIDEO_TO_SPIDER_ROOT", _SCRIPT_DIR.parent)).resolve()
_H2S2R_ROOT = Path(
    os.environ.get("H2S2R_ROOT", _V2S_ROOT.parent / "reference" / "human2sim2robot")
).resolve()

for _path in (str(_V2S_ROOT), str(_H2S2R_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import torch
from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig
from human2sim2robot.ppo.utils.asymmetric_critic import AsymmetricCriticConfig
from human2sim2robot.ppo.utils.network import MlpConfig, NetworkConfig, RnnConfig
from human2sim2robot.ppo.utils.rewards_shaper import RewardsShaperParams
from spider.config import Config, load_config_yaml, process_config
from spider.io import load_data

from video_to_spider.rl.mjwp_env import MJWPVectorEnv, MJWPVectorEnvConfig


def _load_ego_config(config_path: str, device: str) -> Config:
    raw = load_config_yaml(config_path)
    allowed = {field.name for field in fields(Config)}
    data = {key: value for key, value in raw.items() if key in allowed}
    for key in ("pair_margin_range", "xy_offset_range"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    if data.get("noise_scale") is None:
        data.pop("noise_scale", None)
    data["device"] = device
    config = process_config(Config(**data))
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=(
            "/data_all/zzx/egoengine/experiments/stereo_bakeoff_20260812/"
            "hot3d/compelling_spoon_single_hand_v2/contact_guidance_output/config_act.yaml"
        ),
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--horizon_length", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--max_episode_length", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--asymmetric-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the H2S2R asymmetric critic consuming privileged MJWP state.",
    )
    return parser.parse_args()


def _build_network_config(seq_length: int) -> NetworkConfig:
    return NetworkConfig(
        mlp=MlpConfig(units=[512, 512]),
        rnn=RnnConfig(
            units=1024,
            layers=1,
            name="lstm",
            layer_norm=True,
            before_mlp=False,
            concat_input=False,
            concat_output=True,
        ),
        separate_value_mlp=False,
        asymmetric_critic=False,
    )


def _build_asymmetric_critic_config(minibatch_size: int) -> AsymmetricCriticConfig:
    """Official H2S2R asymmetric-critic MLP, re-expressed as a dataclass.

    The actor keeps its own LSTM policy in ``_build_network_config``.  This
    critic consumes the privileged 39-dim MJWP state and is intentionally kept
    as the official ``[1024, 512]`` MLP, which is the lower-risk starting point
    for the xHand adaptation.
    """
    return AsymmetricCriticConfig(
        name="xhand_asymmetric_critic_mlp",
        network=NetworkConfig(
            mlp=MlpConfig(units=[1024, 512]),
            rnn=None,
            separate_value_mlp=False,
            asymmetric_critic=True,
        ),
        normalize_input=True,
        learning_rate=5e-5,
        mini_epochs=4,
        truncate_grads=True,
        minibatch_size=minibatch_size,
    )


def _build_ppo_config(
    *,
    num_envs: int,
    horizon_length: int,
    seq_length: int,
    max_epochs: int,
    learning_rate: float,
    device: str,
    asymmetric_critic: AsymmetricCriticConfig | None,
) -> PpoConfig:
    batch_size = num_envs * horizon_length
    minibatch_size = batch_size
    return PpoConfig(
        num_actors=num_envs,
        learning_rate=learning_rate,
        entropy_coef=0.0,
        horizon_length=horizon_length,
        normalize_advantage=True,
        normalize_input=True,
        grad_norm=1.0,
        critic_coef=4,
        gamma=0.998,
        tau=0.95,
        reward_shaper=RewardsShaperParams(scale_value=1.0),
        mini_epochs=4,
        e_clip=0.2,
        multi_gpu=False,
        device=device,
        weight_decay=0.0,
        asymmetric_critic=asymmetric_critic,
        truncate_grads=True,
        save_frequency=max_epochs + 1,
        save_best_after=max_epochs + 1,
        print_stats=True,
        max_epochs=max_epochs,
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
        minibatch_size=minibatch_size,
        mixed_precision=False,
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        value_bootstrap=False,
        adv_rms_momentum=0.5,
        clip_actions=True,
        schedule_entropy=False,
        freeze_critic=False,
    )


def main() -> None:
    args = _parse_args()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    ego_config = _load_ego_config(args.config_path, args.device)
    ref_data = load_data(ego_config, ego_config.data_path)
    env_cfg = MJWPVectorEnvConfig(
        max_episode_length=args.max_episode_length,
        asymmetric_critic=args.asymmetric_critic,
        reset_hand_noise_std=0.01,
        reset_object_pos_noise_std=0.005,
        reset_object_rot_noise_std=0.01,
    )
    env = MJWPVectorEnv(
        ego_config,
        ref_data,
        num_envs=args.num_envs,
        env_config=env_cfg,
        seed=args.seed,
    )

    experiment_dir = Path(args.output_dir) if args.output_dir else (
        Path(ego_config.output_dir)
        / "rl_smoke"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.num_envs * args.horizon_length
    asymmetric_critic = (
        _build_asymmetric_critic_config(batch_size) if args.asymmetric_critic else None
    )

    ppo_config = _build_ppo_config(
        num_envs=args.num_envs,
        horizon_length=args.horizon_length,
        seq_length=args.seq_length,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        asymmetric_critic=asymmetric_critic,
    )
    network_config = _build_network_config(args.seq_length)

    env_info = env.get_env_info()
    train_metadata = {
        "obs_dim": int(env_info["observation_space"].shape[0]),
        "action_dim": int(env_info["action_space"].shape[0]),
        "state_dim": int(env_info["state_space"].shape[0]),
        "num_envs": args.num_envs,
        "horizon_length": args.horizon_length,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "reward_shaper_scale": 1.0,
        "network": {
            "mlp": {"units": [512, 512]},
            "rnn": {
                "units": 1024,
                "layers": 1,
                "name": "lstm",
                "layer_norm": True,
                "before_mlp": False,
                "concat_input": False,
                "concat_output": True,
            },
            "separate_value_mlp": False,
            "asymmetric_critic": False,
        },
        "asymmetric_critic": (
            {
                "name": asymmetric_critic.name,
                "normalize_input": asymmetric_critic.normalize_input,
                "learning_rate": asymmetric_critic.learning_rate,
                "mini_epochs": asymmetric_critic.mini_epochs,
                "truncate_grads": asymmetric_critic.truncate_grads,
                "minibatch_size": asymmetric_critic.minibatch_size,
                "network": {
                    "mlp": {"units": list(asymmetric_critic.network.mlp.units)},
                    "rnn": None,
                    "separate_value_mlp": asymmetric_critic.network.separate_value_mlp,
                    "asymmetric_critic": asymmetric_critic.network.asymmetric_critic,
                },
            }
            if asymmetric_critic is not None
            else None
        ),
    }
    (experiment_dir / "train_metadata.json").write_text(
        json.dumps(train_metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"env info: {env_info}")
    print(f"experiment_dir: {experiment_dir}")
    print(
        "PPO smoke config: "
        f"num_envs={args.num_envs} horizon={args.horizon_length} "
        f"seq={args.seq_length} max_epochs={args.max_epochs}"
    )

    agent = PpoAgent(
        experiment_dir=experiment_dir,
        ppo_config=ppo_config,
        network_config=network_config,
        env=env,
    )
    last_reward, epoch = agent.train()
    print(f"finished smoke training: last_reward={last_reward}, epoch={epoch}")


if __name__ == "__main__":
    main()
