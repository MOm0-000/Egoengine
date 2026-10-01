from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from video_to_spider.rl.core.state_io import (
    TRAINING_CHECKPOINT_SCHEMA,
    build_core_checkpoint,
    build_training_checkpoint,
    load_torch_gzip,
    restore_training_checkpoint,
    validate_core_checkpoint,
    validate_physics_snapshot,
    validate_training_checkpoint,
    write_torch_gzip,
    write_torch_gzip_atomic,
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


def test_epoch_boundary_training_checkpoint_atomic_roundtrip(tmp_path):
    actor = torch.nn.Linear(2, 1)
    critic_module = torch.nn.Linear(2, 1)
    critic = SimpleNamespace(
        state_dict=critic_module.state_dict,
        load_state_dict=critic_module.load_state_dict,
        optimizer=torch.optim.Adam(critic_module.parameters(), lr=5e-5),
    )
    policy = SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-4),
        normalization_version=3,
    )
    payload = build_training_checkpoint(
        policy=policy,
        boundary_state=_snapshot(),
        observation_prefix=tuple(torch.full((1, 2), float(index)) for index in range(20)),
        next_epoch=7,
        cost={"training_physics_steps": 9600},
        metadata={"config_sha256": "abc", "loss_semantics": "auxiliary_mse"},
    )
    assert payload["schema"] == TRAINING_CHECKPOINT_SCHEMA
    path = tmp_path / "latest.pt.gz"
    write_torch_gzip_atomic(path, payload)
    assert not list(tmp_path.glob("*.tmp"))
    restored = load_torch_gzip(path)
    validate_training_checkpoint(restored)

    target_actor = torch.nn.Linear(2, 1)
    target_critic_module = torch.nn.Linear(2, 1)
    target = SimpleNamespace(
        actor=target_actor,
        critic=SimpleNamespace(
            load_state_dict=target_critic_module.load_state_dict,
            optimizer=torch.optim.Adam(target_critic_module.parameters(), lr=5e-5),
        ),
        actor_optimizer=torch.optim.Adam(target_actor.parameters(), lr=1e-4),
        normalization_version=0,
    )
    restore_training_checkpoint(target, restored)
    assert target.normalization_version == 3
    assert all(
        torch.equal(target.actor.state_dict()[name], value)
        for name, value in actor.state_dict().items()
    )
