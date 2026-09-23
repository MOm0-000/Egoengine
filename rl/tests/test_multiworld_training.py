"""CPU-only contracts for independent-world snapshot handling."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.mjwp_env import IndependentMJWPTrainingEnv
from video_to_spider.rl.replay_rl import MJWPIndependentTrainingBackend


class _World:
    num_envs = 1

    def __init__(self, value: int):
        self.state = {
            "snapshot_schema": "egoengine_mjwp_snapshot_v2",
            "warp_state_keys": ("qpos",),
            "qpos": torch.tensor([[value]], dtype=torch.float32),
            "time_indices": np.array([20], dtype=np.int32),
        }

    def get_env_state(self):
        return deepcopy(self.state)

    def set_env_state(self, state):
        self.state = deepcopy(state)


def _environment(values=(1, 2, 3, 4)):
    env = object.__new__(IndependentMJWPTrainingEnv)
    env.worlds = tuple(_World(value) for value in values)
    env.num_envs = len(env.worlds)
    return env


def test_checkpoint_round_trip_preserves_each_independent_world():
    env = _environment()
    checkpoint = env.get_env_state()
    assert checkpoint["schema"] == "egoengine_independent_mjwp_worlds_v1"
    assert [int(state["qpos"][0, 0]) for state in checkpoint["worlds"]] == [1, 2, 3, 4]
    env.set_env_state({
        "snapshot_schema": "egoengine_mjwp_snapshot_v2",
        "warp_state_keys": ("qpos",),
        "qpos": torch.tensor([[9]], dtype=torch.float32),
        "time_indices": np.array([20], dtype=np.int32),
    })
    assert [int(world.state["qpos"][0, 0]) for world in env.worlds] == [9, 9, 9, 9]
    env.set_env_state(checkpoint)
    assert [int(world.state["qpos"][0, 0]) for world in env.worlds] == [1, 2, 3, 4]


def test_training_backend_replicates_and_verifies_complete_source_snapshot():
    env = _environment()
    backend = MJWPIndependentTrainingBackend(env)
    source = env.worlds[0].get_env_state()
    backend.restore(source)
    backend.verify_restored_snapshot(source)
    assert backend.verified_restore_count == 1
    assert backend.restore_audits == [{
        "worlds": 4,
        "snapshot_keys_per_world": 4,
        "warp_state_fields_per_world": 1,
        "all_worlds_bitwise_equal_to_source": True,
    }]
