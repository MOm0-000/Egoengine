"""Small real-backend checks for the TACO MJWP/PPO adapter."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "external/mink/src")]

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
if not torch.cuda.is_available():
    pytest.skip("MJWP uses a CUDA graph", allow_module_level=True)

from run_mjwp_ppo import _load_ego_config, _load_reference
from video_to_spider.rl.mjwp_env import (
    MJWPVectorEnv,
    MJWPVectorEnvConfig,
    _MJWP_SNAPSHOT_SCHEMA,
    _WP_CONTACT_FIELDS,
    _WP_EFC_FIELDS,
    _WP_STATE_FIELDS,
    _object_pose_parts,
    _transform_anchors_torch,
)
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation


CONFIG = ROOT / "configs/taco_pour_bimanual_ppo.yaml"
OBJECTIVE = load_runtime_objective(
    ROOT / "configs/replay_rl_protocol.yaml",
    ROOT / "configs/taco_pour_local_unpublished_v1.yaml",
    tracking_variant="tool_and_target",
    require_run_ready=False,
)
OBSERVATION = load_runtime_observation(
    ROOT / "configs/replay_rl_protocol.yaml",
    ROOT / "configs/taco_pour_observation_local_236d_v1.yaml",
    require_run_ready=False,
)


@pytest.fixture(scope="module")
def env():
    config = _load_ego_config(str(CONFIG), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30.0)
    return MJWPVectorEnv(
        config,
        reference,
        num_envs=2,
        env_config=MJWPVectorEnvConfig(
            max_episode_length=2,
            asymmetric_critic=True,
            objective=OBJECTIVE,
            observation=OBSERVATION,
        ),
        seed=11,
    )


def test_reference_rate_endpoint_and_control_substeps(env):
    assert env.ctrl_ref.shape[0] == 198
    assert env.ego_cfg.ctrl_steps == 10
    expected_indices = env.start_indices + 1
    np.testing.assert_allclose(
        env._reference_ctrls(np.array([0, 0]), offset=1).cpu().numpy(),
        env.ctrl_ref[expected_indices].cpu().numpy(),
    )
    env.reset()
    before = env._mjwp.get_qpos(env.ego_cfg, env.env).clone()
    env.step(np.zeros((2, 36), dtype=np.float32))
    after = env._mjwp.get_qpos(env.ego_cfg, env.env)
    assert not torch.equal(before, after)
    import warp as wp
    np.testing.assert_allclose(wp.to_torch(env.env.data_wp.time).cpu().numpy(), 1.0 / 30.0, atol=2e-6)


def test_bimanual_observation_and_contact_contract(env):
    observation = env.reset()
    assert observation["obs"].shape == (2, 236)
    assert observation["states"].shape == (2, 108)
    assert env.env_cfg.fingertip_site_ids == (1, 4, 7, 10, 13, 17, 20, 23, 26, 29)
    assert env.env_cfg.palm_site_ids == (0, 16)
    assert env.env_cfg.domain.obs_noise_std == 0.0
    assert env.env_cfg.domain.action_noise_std == 0.0
    assert env.env_cfg.reset_hand_noise_std == 0.0
    assert env.env_cfg.reset_object_pos_noise_std == 0.0
    assert env.env_cfg.reset_object_rot_noise_std == 0.0
    assert env.tracked_object_indices == (0, 1)
    assert env.lift_object_index == env.object_roles.index(OBJECTIVE.lift_object_role) == 0
    actor = observation["obs"]
    np.testing.assert_allclose(
        actor[:, 144:180],
        env._reference_ctrls(env.time_indices, offset=1)[:, :36].cpu().numpy(),
    )
    np.testing.assert_allclose(
        actor[:, 180:216],
        env._reference_ctrls(env.time_indices, offset=2)[:, :36].cpu().numpy(),
    )
    goal_indices = np.minimum(env.start_indices + env.time_indices + 1, len(env.qpos_ref) - 1)
    goal_objects = _object_pose_parts(env.qpos_ref[goal_indices], int(env.ego_cfg.nq_obj))
    expected_goal_anchors = torch.cat([
        _transform_anchors_torch(pose[0], pose[1], env.anchors) for pose in goal_objects
    ], dim=1).reshape(2, -1)
    np.testing.assert_allclose(actor[:, 126:144], expected_goal_anchors.cpu().numpy())
    _, _, _, info = env.step(np.zeros((2, 36), dtype=np.float32))
    assert info["contact_flags"].shape == (2, 2, 2, 5)
    assert info["object_position_error"].shape == (2, 2)
    assert info["object_rotation_error"].shape == (2, 2)
    assert info["object_tracking_error_per_object"].shape == (2, 2)
    assert info["object_tracking_reward_per_object"].shape == (2, 2)
    assert info["object_terminated"].shape == (2, 2)
    assert info["contact_bonus_per_hand_object"].shape == (2, 2, 2)
    np.testing.assert_allclose(
        info["reward"],
        info["aggregate_tracking_reward"] + info["aggregate_contact_bonus"] + info["lift_reward"],
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        info["object_tracking_error"], info["object_tracking_error_per_object"].mean(axis=1)
    )
    np.testing.assert_array_equal(info["terminated"], info["object_terminated"].any(axis=1))


def test_autoreset_returns_new_observation_and_keeps_terminal_info(env):
    env.reset()
    env.step(np.zeros((2, 36), dtype=np.float32))
    observation, _, done, info = env.step(np.zeros((2, 36), dtype=np.float32))
    assert done.tolist() == [True, True]
    assert env.time_indices.tolist() == [0, 0]
    reset_observation = env.reset()
    np.testing.assert_allclose(observation["obs"], reset_observation["obs"], atol=0.0, rtol=0.0)
    assert np.isfinite(info["object_tracking_error"]).all()
    assert np.isfinite(info["contact_score"]).all()
    assert info["time_outs"].tolist() == [True, True]


def test_partial_reset_preserves_other_world_dynamics(env):
    import warp as wp

    env.reset()
    env.step(np.zeros((2, 36), dtype=np.float32))
    before = wp.to_torch(env.env.data_wp.qacc).clone()
    env._reset_worlds(np.array([True, False]))
    after = wp.to_torch(env.env.data_wp.qacc)
    np.testing.assert_array_equal(after[1].cpu().numpy(), before[1].cpu().numpy())


def test_snapshot_restores_all_saved_python_state(env):
    env.reset()
    snapshot = env.get_env_state()
    env.step(np.zeros((2, 36), dtype=np.float32))
    env.set_env_state(snapshot)
    restored = env.get_env_state()
    for key, value in snapshot.items():
        if isinstance(value, dict):
            assert restored[key] == value
        elif hasattr(value, "numpy"):
            np.testing.assert_array_equal(restored[key].numpy(), value.numpy())
        else:
            np.testing.assert_array_equal(restored[key], value)


def test_snapshot_field_contract_covers_mujoco_warp_313():
    import mujoco_warp as mjwarp

    scalar_or_nested = {
        "contact", "efc", "nworld", "naconmax", "naccdmax", "njmax",
        "nvmax", "nvmax_pad", "njmax_pad", "njmax_nnz",
    }
    assert set(_WP_STATE_FIELDS) == set(mjwarp.Data.__annotations__) - scalar_or_nested
    assert set(_WP_CONTACT_FIELDS) == set(mjwarp.Contact.__annotations__)
    assert set(_WP_EFC_FIELDS) == set(mjwarp.Constraint.__annotations__)


def test_legacy_partial_snapshot_is_rejected(env):
    state = env.get_env_state()
    assert state["snapshot_schema"] == _MJWP_SNAPSHOT_SCHEMA
    del state["snapshot_schema"]
    with pytest.raises(ValueError, match="legacy partial snapshots"):
        env.set_env_state(state)


def test_unused_contact_buffer_entries_do_not_count(env):
    import warp as wp
    env.reset()
    env.step(np.zeros((2, 36), dtype=np.float32))
    state = env.get_env_state()
    try:
        env.env.data_wp.nacon.zero_()
        wp.to_torch(env.env.data_wp.contact.frame).fill_(float("nan"))
        wp.synchronize()
        flags, forces = env._live_contact_features()
        assert not flags.any()
        assert not forces.any()
    finally:
        env.set_env_state(state)


@pytest.fixture(scope="module")
def single_env():
    config = _load_ego_config(str(CONFIG), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30.0)
    return MJWPVectorEnv(config, reference, num_envs=1,
                        env_config=MJWPVectorEnvConfig(max_episode_length=197, asymmetric_critic=True,
                                                       reference_start_index=0, objective=OBJECTIVE,
                                                       observation=OBSERVATION), seed=11)


def test_chunk_reset_restores_actual_incoming_state_not_reference(single_env):
    env = single_env
    env._chunk_reset_state = None
    env.reset()
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action
    backend = MJWPChunkBackend(env)
    backend.step(replay_action(backend, 0), 0)
    env.set_chunk_reset(start=1, end=3)
    boundary = env.get_env_state()
    assert not torch.equal(boundary["qpos"][0], env.qpos_ref[1].cpu())
    work = env.simulation_control_intervals
    env.step(np.zeros((1, 36), np.float32))
    env.reset()
    restored = env.get_env_state()
    for key in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time", "nacon", "ncollision",
                "contact.geom", "contact.worldid", "initial_object_heights", "last_action"):
        torch.testing.assert_close(restored[key], boundary[key], rtol=0, atol=0)
    assert env.time_indices.tolist() == [1]
    assert env.simulation_control_intervals == work + 1
    env.step(np.zeros((1, 36), np.float32), auto_reset=False)
    _, _, done, info = env.step(np.zeros((1, 36), np.float32), auto_reset=False)
    assert done.tolist() == [True] and info["time_outs"].tolist() == [True]
    assert env.time_indices.tolist() == [3]
    env.set_env_state(boundary)


def test_real_replay_to_official_ppo_fallback_contract(single_env, tmp_path):
    """Inject only Replay rejection to exercise fallback, NOT Pour task success."""
    from egoengine_repro.action.replay_rl import solve_chunk
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action, train_chunk_ppo

    env = single_env
    env._chunk_reset_state = None
    env.reset()
    backend = MJWPChunkBackend(env)
    backend.step(replay_action(backend, 0), 0)
    incoming = backend.snapshot()
    start = int(env.time_indices[0])
    calls = []
    real_step = backend.step
    injected = [False]

    def reject_replay_once(action, index):
        result = real_step(action, index)
        if not injected[0]:
            injected[0] = True
            return False
        return result

    backend.step = reject_replay_once

    def train(current, first, end):
        calls.append((first, end))
        assert int(env.time_indices[0]) == first
        torch.testing.assert_close(env.get_env_state()["qpos"], incoming["qpos"], rtol=0, atol=0)
        return train_chunk_ppo(current, first, end, tmp_path / "ppo", epochs=1, horizon=4)

    work = env.simulation_control_intervals
    result = solve_chunk(backend, replay_action, train, start=start, total_steps=start + 4)
    assert calls == [(start, start + 4)]
    assert [trial.mode for trial in result.trials] == ["replay", "rl"]
    assert backend.validation_traces[0]["first_failure"]["reason"] == "backend_rejected"
    assert env.simulation_control_intervals >= work + 5  # includes rejected Replay + actual PPO rollout
    assert int(env.time_indices[0]) == result.committed_end
    if result.mode is None:
        torch.testing.assert_close(env.get_env_state()["qpos"], incoming["qpos"], rtol=0, atol=0)


def test_gpu_training_actor_transfers_to_cpu_inference(single_env, tmp_path):
    from video_to_spider.rl.replay_rl import (
        MJWPChunkBackend,
        train_chunk_ppo,
    )

    cpu_config = _load_ego_config(str(CONFIG), "cpu")
    cpu_reference = _load_reference(
        cpu_config.data_path, "cpu", expected_frequency=30.0
    )
    cpu_env = MJWPVectorEnv(
        cpu_config,
        cpu_reference,
        num_envs=1,
        env_config=MJWPVectorEnvConfig(
            max_episode_length=197,
            asymmetric_critic=False,
            reference_start_index=0,
            objective=OBJECTIVE,
            observation=OBSERVATION,
        ),
        seed=11,
    )
    cpu_backend = MJWPChunkBackend(cpu_env)
    training_backend = MJWPChunkBackend(single_env)
    cpu_env.reset()
    incoming = cpu_backend.snapshot()
    training_backend.restore(incoming)
    training_backend.verify_restored_snapshot(incoming)

    policy = train_chunk_ppo(
        training_backend,
        0,
        4,
        tmp_path / "dual_backend_ppo",
        epochs=1,
        horizon=4,
        validation_env=cpu_env,
    )
    try:
        assert policy.audit["training_device"] == "cuda:0"
        assert policy.audit["policy_inference_device"] == "cpu"
        assert policy.audit["actor_transfer_bitwise_equal"] is True
        assert policy.audit["actor_state_sha256"] == policy.audit["transferred_actor_state_sha256"]
        cpu_backend.restore(incoming)
        action = policy(cpu_backend, 0)
        assert action.shape == (1, 36)
        assert np.isfinite(action).all()
        cpu_backend.step(action, 0)
    finally:
        policy.close()


def test_real_two_chunk_rollout_commits_only_first_chunk():
    """Static in-memory control fixture, not the Pour demonstration or its SR."""
    from egoengine_repro.action.replay_rl import solve_chunk
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action
    config = _load_ego_config(str(CONFIG), "cuda:0")
    loaded = _load_reference(config.data_path, "cuda:0", expected_frequency=30.0)
    qpos = loaded[0][0:1].repeat(61, 1)
    qpos[:, [2, 20]] += .2  # Keep the hands away from both resting objects.
    reference = (qpos, torch.zeros((61, 48), device="cuda:0"), qpos[:, :36].clone(),
                 torch.zeros((61, 10), device="cuda:0"), torch.zeros((61, 10, 3), device="cuda:0"))
    env = MJWPVectorEnv(config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(max_episode_length=60, asymmetric_critic=True,
                                      reference_start_index=0, objective=OBJECTIVE,
                                      observation=OBSERVATION))
    backend = MJWPChunkBackend(env)
    snapshots = {}
    actual_snapshot = backend.snapshot

    def capture():
        state = actual_snapshot()
        snapshots[int(env.time_indices[0])] = state
        return state

    backend.snapshot = capture
    before_work = env.simulation_control_intervals
    result = solve_chunk(backend, replay_action,
        lambda *_: pytest.fail("static replay fixture should not train"), start=0, total_steps=60)
    assert result.mode == "replay"
    assert (result.committed_end, result.lookahead_end) == (20, 40)
    assert env.time_indices.tolist() == [20]
    assert env.simulation_control_intervals == before_work + 40
    assert env.simulation_physics_steps == 400
    assert len(backend.validation_traces) == 1
    trace = backend.validation_traces[0]
    assert trace["mode"] == "replay" and trace["feasible"]
    assert len(trace["steps"]) == 40 and trace["first_failure"] is None
    assert trace["steps"][0]["tracked_object_roles"] == ["tool", "target"]
    assert len(trace["steps"][0]["position_error_m"]) == 2
    assert len(trace["steps"][0]["endpoint_qpos"]) == 50
    assert len(trace["steps"][0]["endpoint_qvel"]) == 48
    assert len(trace["steps"][0]["commanded_ctrl"]) == 36
    assert len(trace["steps"][0]["raw_residual_action"]) == 36
    assert len(trace["steps"][0]["applied_residual"]) == 36
    assert trace["steps"][0]["objective_metric_name"] == "weighted_tracking_error"
    assert trace["steps"][0]["objective_score"] == trace["steps"][0]["tracking_error"]
    assert trace["steps"][0]["independent_threshold_pass"] is None
    assert np.asarray(trace["steps"][0]["contact_bonus_per_hand_object"]).shape == (2, 2)
    assert trace["steps"][0]["total_reward"] == pytest.approx(
        trace["steps"][0]["aggregate_tracking_reward"]
        + trace["steps"][0]["aggregate_contact_bonus"]
        + trace["steps"][0]["lift_reward"], abs=1e-6
    )
    restored = actual_snapshot()
    # Compare with the SAME rollout's captured boundary. Re-running GPU physics
    # is not bitwise deterministic and is precisely why the scheduler saves it.
    for name in ("qpos", "qvel", "qacc_warmstart", "time", "last_action", "last_ctrl",
                 "nacon", "ncollision", "initial_object_heights"):
        torch.testing.assert_close(restored[name], snapshots[20][name], rtol=0, atol=0)
