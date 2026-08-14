import numpy as np
import pytest

from video_to_spider.export.egoengine_mode_switch import (
    ModeSwitchConfig,
    ModeSwitchRunner,
    RolloutResult,
    chunk_windows,
    feasibility_constant,
    is_feasible,
    object_tracking_error,
    select_solver,
    SOLVER_MPC,
    SOLVER_REPLAY,
    SOLVER_RL,
)


def _rollout(pos: float, rot: float, feasible: bool, mode: str) -> RolloutResult:
    return RolloutResult(
        position_error_m=np.asarray([pos]),
        rotation_error_rad=np.asarray([rot]),
        feasible=feasible,
        mode=mode,
    )


def test_feasibility_constant_matches_paper_rectangle():
    config = ModeSwitchConfig(
        object_pos_threshold_m=0.1,
        object_rot_threshold_rad=0.3,
        lambda_pos=1.0,
        lambda_rot=1.0,
    )
    np.testing.assert_allclose(feasibility_constant(config), np.sqrt(0.1**2 + 0.3**2))


def test_object_tracking_error_combines_position_and_rotation():
    config = ModeSwitchConfig(lambda_pos=2.0, lambda_rot=3.0)
    error = object_tracking_error(np.asarray([0.1]), np.asarray([0.2]), config)
    np.testing.assert_allclose(error, np.sqrt(2.0 * 0.1**2 + 3.0 * 0.2**2))


def test_chunk_windows_include_lookahead_window():
    config = ModeSwitchConfig(chunk_steps=10, lookahead_chunks=1)
    windows = chunk_windows(25, config)
    assert [w.index for w in windows] == [0, 1, 2]
    assert [(w.start, w.current_stop, w.next_stop) for w in windows] == [
        (0, 10, 20),
        (10, 20, 25),
        (20, 25, 25),
    ]


def test_select_solver_escalates_cheapest_first():
    assert select_solver(True, True, True, rl_enabled=True) == SOLVER_REPLAY
    assert select_solver(False, True, True, rl_enabled=True) == SOLVER_MPC
    assert select_solver(False, False, True, rl_enabled=True) == SOLVER_RL
    assert select_solver(False, False, True, rl_enabled=False) == SOLVER_MPC


def test_runner_prefers_replay_then_mpc():
    config = ModeSwitchConfig(chunk_steps=2, lookahead_chunks=1)
    calls: list[tuple[str, int, int]] = []
    runner = ModeSwitchRunner(
        replay_solver=lambda s, e: (calls.append(("replay", s, e)) or _rollout(0.01, 0.01, True, SOLVER_REPLAY)),
        mpc_solver=lambda s, e: (calls.append(("mpc", s, e)) or _rollout(0.01, 0.01, True, SOLVER_MPC)),
        config=config,
    )
    report = runner.run(4)
    assert [d["mode"] for d in report["decisions"]] == [SOLVER_REPLAY, SOLVER_REPLAY]
    assert ("mpc", 0, 2) not in calls
    assert not report["any_chunk_infeasible"]


def test_runner_escalates_to_mpc_when_replay_fails():
    config = ModeSwitchConfig(chunk_steps=2, lookahead_chunks=0)
    runner = ModeSwitchRunner(
        replay_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_REPLAY),
        mpc_solver=lambda s, e: _rollout(0.02, 0.02, True, SOLVER_MPC),
        config=config,
    )
    report = runner.run(4)
    assert [d["mode"] for d in report["decisions"]] == [SOLVER_MPC, SOLVER_MPC]
    assert not report["any_chunk_infeasible"]


def test_runner_escalates_to_rl_and_flags_exhausted_fallback():
    config = ModeSwitchConfig(chunk_steps=4, rl_enabled=True)
    runner = ModeSwitchRunner(
        replay_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_REPLAY),
        mpc_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_MPC),
        rl_solver=lambda s, e: _rollout(0.01, 0.01, True, SOLVER_RL),
        config=config,
    )
    report = runner.run(4)
    assert report["decisions"][0]["mode"] == SOLVER_RL
    assert report["decisions"][0]["rl_feasible"] is True
    assert not report["any_chunk_infeasible"]


def test_runner_requires_rl_solver_when_enabled():
    config = ModeSwitchConfig(chunk_steps=4, rl_enabled=True)
    runner = ModeSwitchRunner(
        replay_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_REPLAY),
        mpc_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_MPC),
        config=config,
    )
    with pytest.raises(NotImplementedError):
        runner.run(4)


def test_runner_reports_infeasible_when_all_solvers_fail_without_rl():
    config = ModeSwitchConfig(chunk_steps=4, rl_enabled=False)
    runner = ModeSwitchRunner(
        replay_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_REPLAY),
        mpc_solver=lambda s, e: _rollout(5.0, 5.0, False, SOLVER_MPC),
        config=config,
    )
    report = runner.run(4)
    assert report["any_chunk_infeasible"] is True
    assert report["decisions"][0]["mode"] == SOLVER_MPC
    assert report["decisions"][0]["chosen_feasible"] is False
