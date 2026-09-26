from pathlib import Path
from dataclasses import replace
import copy
import random

import gym
import numpy as np
import torch

from video_to_spider.rl.credit_audit import (
    _termination_outcomes,
    _world_major,
    apply_xor_patch,
    model_state_sha256,
    write_xor_patch,
    CreditInstrumentedTruncatedGaussianPpoAgent,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
    TruncatedGaussianActionSpec,
)
from human2sim2robot.ppo.utils.running_mean_std import RunningMeanStd
from run_mjwp_ppo import (
    _build_asymmetric_critic_config,
    _build_network_config,
    _build_ppo_config,
)


def test_world_major_matches_ppo_flattening():
    time_major = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    actual = _world_major(time_major)
    expected = np.concatenate([time_major[:, world] for world in range(2)])
    assert np.array_equal(actual, expected)


def test_lossless_actor_xor_patch_round_trip(tmp_path: Path):
    before = {
        "float": torch.tensor([[1.0, -2.5], [0.0, 9.0]], dtype=torch.float32),
        "count": torch.tensor(17, dtype=torch.int64),
    }
    after = {
        "float": torch.nextafter(
            before["float"], torch.full_like(before["float"], float("inf"))
        ),
        "count": torch.tensor(18, dtype=torch.int64),
    }
    path = tmp_path / "patch.npz"
    report = write_xor_patch(path, before, after)
    restored = apply_xor_patch(before, path)
    assert report["before_model_state_sha256"] == model_state_sha256(before)
    assert report["after_model_state_sha256"] == model_state_sha256(after)
    assert model_state_sha256(restored) == model_state_sha256(after)
    for name in after:
        assert torch.equal(restored[name], after[name])


def test_future_termination_endpoint_is_per_world():
    endpoints = np.array([
        21, 22, 23, 24,
        21, 22, 23, 24,
    ], dtype=np.int32)
    dones = np.array([
        False, False, True, False,
        False, False, False, False,
    ])
    actual = _termination_outcomes(endpoints, dones, worlds=2, horizon=4)
    assert actual.tolist() == [23, 23, 23, -1, -1, -1, -1, -1]


def test_observation_normalization_is_pure_until_explicit_update():
    normalizer = RunningMeanStd((3,))
    normalizer.train()
    observations = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    before = copy.deepcopy(normalizer.state_dict())
    first = normalizer(observations, update_stats=False)
    second = normalizer(observations, update_stats=False)
    assert torch.equal(first, second)
    assert _equal(before, normalizer.state_dict())
    normalizer.update(observations)
    assert not _equal(before, normalizer.state_dict())


class _DeterministicAuditEnv:
    num_envs = 4

    def __init__(self):
        self.cursor = np.zeros(4, dtype=np.int32)
        self.state = np.zeros((4, 39), dtype=np.float32)

    def get_env_info(self):
        return {
            "observation_space": gym.spaces.Box(
                -np.inf, np.inf, shape=(236,), dtype=np.float32
            ),
            "action_space": gym.spaces.Box(-1.0, 1.0, shape=(36,), dtype=np.float32),
            "state_space": gym.spaces.Box(
                -np.inf, np.inf, shape=(39,), dtype=np.float32
            ),
            "agents": 1,
            "value_size": 1,
        }

    def get_number_of_agents(self):
        return 1

    def _obs(self):
        obs = np.zeros((4, 236), dtype=np.float32)
        obs[:, 0] = self.cursor
        obs[:, 1:40] = self.state
        return {"obs": obs, "states": self.state.copy()}

    def reset(self):
        self.cursor[:] = 0
        self.state[:] = 0
        return self._obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        source = self.cursor.copy()
        self.state[:, :36] += actions * 0.001
        self.cursor += 1
        reward = (1.0 - np.square(actions).mean(axis=1)).astype(np.float32)
        done = self.cursor >= 3
        outcome = self.cursor.copy()
        info = {
            "source_reference_endpoint": source,
            "outcome_reference_endpoint": outcome,
            "command_reference_endpoint": outcome,
            "reward_reference_endpoint": outcome,
            "next_observation_goal_reference_endpoint": outcome + 1,
            "object_position_error": np.zeros((4, 2), np.float32),
            "object_rotation_error": np.zeros((4, 2), np.float32),
            "object_tracking_error_per_object": np.zeros((4, 2), np.float32),
            "object_tracking_reward_per_object": np.ones((4, 2), np.float32),
            "aggregate_tracking_reward": np.ones(4, np.float32),
            "aggregate_contact_bonus": np.zeros(4, np.float32),
            "lift_reward": np.zeros(4, np.float32),
            "contact_flags": np.zeros((4, 2, 2, 5), bool),
            "terminated": done.copy(),
            "time_outs": np.zeros(4, bool),
        }
        if done.any():
            self.cursor[done] = 0
            self.state[done] = 0
        return self._obs(), reward, done, info

    def set_train_info(self, frame, agent):
        del frame, agent

    def enable_state_feasible_action_contract(self, *, reference_snap_tolerance):
        self.reference_snap_tolerance = reference_snap_tolerance

    def current_normalized_action_bounds(self):
        return torch.full((4, 36), -1.0), torch.full((4, 36), 1.0)

    def tail_curriculum_audit(self):
        return None

    def state_feasible_action_audit(self):
        return {"reference_snap_tolerance": self.reference_snap_tolerance}

    def get_env_state(self):
        return {"cursor": self.cursor.copy(), "state": self.state.copy()}


def _training_result(tmp_path: Path, instrumented: bool):
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    env = _DeterministicAuditEnv()
    config = replace(
        _build_ppo_config(
            num_envs=4,
            horizon_length=4,
            seq_length=4,
            max_epochs=1,
            learning_rate=1e-4,
            device="cpu",
            asymmetric_critic=_build_asymmetric_critic_config(16),
        ),
        clip_actions=False,
        print_stats=False,
    )
    spec = TruncatedGaussianActionSpec(
        profile_id="unit_test",
        residual_scale=0.05,
        reference_snap_tolerance=2e-7,
        minimum_normalization_mass=1e-12,
        optimizer_training_authorized=True,
    )
    cls = (
        CreditInstrumentedTruncatedGaussianPpoAgent
        if instrumented else StateFeasibleTruncatedGaussianPpoAgent
    )
    kwargs = {}
    if instrumented:
        kwargs["credit_audit_dir"] = tmp_path / "credit"
    agent = cls(
        experiment_dir=tmp_path / "experiment",
        ppo_config=config,
        network_config=_build_network_config(4),
        env=env,
        distribution_spec=spec,
        **kwargs,
    )
    agent.train()
    audit = agent.finalize_credit_audit() if instrumented else None
    result = {
        "actor": copy.deepcopy(agent.model.state_dict()),
        "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
        "critic": copy.deepcopy(agent.asymmetric_critic_net.state_dict()),
        "critic_optimizer": copy.deepcopy(agent.asymmetric_critic_net.optimizer.state_dict()),
        "env": env.get_env_state(),
        "rnn": [state.detach().clone() for state in agent.rnn_states],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
        "normalization": agent.observation_normalization_audit(),
        "audit": audit,
    }
    agent.writer.close()
    return result


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_credit_logger_is_training_transparent(tmp_path: Path):
    baseline = _training_result(tmp_path / "baseline", False)
    audited = _training_result(tmp_path / "audited", True)
    for name in (
        "actor", "optimizer", "critic", "critic_optimizer", "env", "rnn",
        "python_rng", "numpy_rng", "torch_rng",
    ):
        assert _equal(baseline[name], audited[name]), name
    assert audited["audit"]["actor_updates"] == 4
    normalization = audited["normalization"]
    assert normalization["current_version"] == 1
    assert normalization["epochs"][0]["version_used_for_rollout_and_updates"] == 0
    assert normalization["epochs"][0]["statistics_committed_after_updates"] is True
    assert normalization["epochs"][0]["before"] == normalization["epochs"][0][
        "frozen_before_commit"
    ]
    assert normalization["epochs"][0]["before"] != normalization["epochs"][0]["after"]
    identity = normalization["pre_optimizer_likelihood_identity_checks"]
    assert len(identity) == 1
    assert identity[0]["maximum_abs_error_from_one"] <= identity[0]["tolerance"]
    critic = normalization["pre_optimizer_critic_value_identity_checks"]
    assert len(critic) == 1
    assert critic[0]["maximum_abs_error"] <= critic[0]["tolerance"]
