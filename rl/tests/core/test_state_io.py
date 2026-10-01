from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from video_to_spider.rl.core.state_io import (
    build_core_checkpoint,
    load_torch_gzip,
    validate_core_checkpoint,
    validate_physics_snapshot,
    write_torch_gzip,
)


def _snapshot():
    return {
        "snapshot_schema": "egoengine_mjwp_snapshot_v3_reward_aligned",
        "warp_state_keys": ("qpos", "qvel", "ctrl", "contact.geom", "contact.worldid", "efc.force", "prev.qpos", "prev.qvel", "prev.ctrl"),
        "time_indices": [0], "start_indices": [0], "episode_lengths": [0], "rng_state": {},
        "last_action": 0, "last_ctrl": 0,
        "qpos": 0, "qvel": 0, "ctrl": 0, "contact.geom": 0,
        "contact.worldid": 0, "efc.force": 0, "prev.qpos": 0,
        "prev.qvel": 0, "prev.ctrl": 0,
    }


def test_snapshot_validation_fails_before_partial_restore():
    snapshot = _snapshot()
    validate_physics_snapshot(snapshot)
    broken = deepcopy(snapshot)
    del broken["contact.geom"]
    with pytest.raises(ValueError, match="missing declared"):
        validate_physics_snapshot(broken)


def test_core_checkpoint_cold_roundtrip(tmp_path):
    actor = torch.nn.Linear(2, 1)
    critic_model = torch.nn.Linear(2, 1)
    policy = SimpleNamespace(
        actor=actor,
        critic=SimpleNamespace(
            state_dict=critic_model.state_dict,
            optimizer=torch.optim.Adam(critic_model.parameters()),
        ),
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        normalization_version=7,
    )
    payload = build_core_checkpoint(
        policy=policy,
        environment_state=_snapshot(),
        rnn_states=(torch.arange(4, dtype=torch.float32).reshape(1, 1, 4),),
        cursor={"epoch": 2, "endpoint": 40},
        cost={"physics_steps": 123},
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "checkpoint.pt.gz"
    write_torch_gzip(path, payload)
    restored = load_torch_gzip(path)
    validate_core_checkpoint(restored)
    assert restored["cursor"] == payload["cursor"]
    assert restored["cost"] == payload["cost"]
    assert torch.equal(restored["rnn_states"][0], payload["rnn_states"][0])
    assert all(
        torch.equal(restored["actor"][name], value)
        for name, value in payload["actor"].items()
    )
