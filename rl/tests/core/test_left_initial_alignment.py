import inspect
from pathlib import Path

import numpy as np

from egoengine_repro.retarget.initial_hand import (
    scalar_hand_coordinates,
    solve_left_reference_aligned_initial,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/audit_taco_pour_left_initial_alignment.py"


class _Joint:
    def __init__(self, name):
        self.name = name


class _ScalarHandsModel:
    nq = 50
    nv = 48
    njnt = 38

    def __init__(self):
        right = [f"R_root_{i}" for i in range(6)] + [f"right_hand_joint_{i}" for i in range(12)]
        left = [f"L_root_{i}" for i in range(6)] + [f"left_hand_joint_{i}" for i in range(12)]
        self._names = right + left + ["right_object_joint", "left_object_joint"]
        self.jnt_qposadr = np.asarray(list(range(36)) + [36, 43])
        self.jnt_dofadr = np.asarray(list(range(36)) + [36, 42])

    def joint(self, index):
        return _Joint(self._names[index])


def test_scalar_hand_mapping_is_name_resolved_and_disjoint():
    mapping = scalar_hand_coordinates(_ScalarHandsModel())
    np.testing.assert_array_equal(mapping["right"]["qpos"], np.arange(18))
    np.testing.assert_array_equal(mapping["left"]["qpos"], np.arange(18, 36))
    np.testing.assert_array_equal(mapping["right"]["dofs"], np.arange(18))
    np.testing.assert_array_equal(mapping["left"]["dofs"], np.arange(18, 36))


def test_static_solver_separates_seed_target_and_writes_only_left_coordinates():
    source = inspect.getsource(solve_left_reference_aligned_initial)
    assert "target_data.qpos[:] = reference_qpos" in source
    assert "candidate[left_qpos] +=" in source
    assert "candidate[locked_qpos]" in source
    assert "seed_is_target" in source
    assert "np.broadcast_to(seed_qpos[locked_qpos]" in source
    assert "world.step" not in source


def test_runner_uses_reference_zero_and_geometry_only_selection():
    source = RUNNER.read_text()
    assert 'reference["qpos"][0]' in source
    solve = source[source.index("def static_solve"):source.index("def _candidate_initial")]
    assert "t0_legality_gate" in solve
    assert "validation_model = _compile(paths)" in solve
    assert "world.step" not in solve
    assert "physics_outcomes_used_for_selection\": False" in solve


def test_candidate_snapshot_is_rebuilt_and_hold_is_not_replay_prefix():
    source = RUNNER.read_text()
    physics = source[source.index("def physics"):source.index("def _pose")]
    assert "ledger.require_capacity" in physics
    assert physics.index("ledger.require_capacity") < physics.index("trace.make_world(paths, candidate)")
    assert "candidate_s0 = world.get_env_state()" in physics
    assert 'world, candidate_s0, "LEFT_ALIGNED_REPLAY", zero' in physics
    assert 'hold[1]' not in physics
    assert "old_contact_or_solver_cache_reused\": False" in physics


def test_replay_command_is_zero_residual_from_reference_one_and_na_is_not_pass():
    source = RUNNER.read_text()
    controls = source[source.index("def _controls"):source.index("def physics")]
    assert 'plan["original_residual_action"]' in controls
    assert "np.zeros_like(zero)" in controls
    assert 'reference[1:21]' in controls
    analysis = source[source.index("def analyze"):source.index("def hashes")]
    assert '"availability": "N/A"' in analysis
    assert "endpoint >= len" in analysis


def test_static_recovery_never_reruns_mink_or_uses_physics_for_selection():
    source = RUNNER.read_text()
    recovery = source[source.index("def recover_selection"):source.index("def _candidate_initial")]
    assert "solver_trace.npz" in recovery
    assert "initial_left_aligned.npz" in recovery
    assert "mink.solve_ik" not in recovery
    assert "world.step" not in recovery
    assert '"mink_rerun": False' in recovery
    assert '"physics_outcomes_used_for_selection": False' in recovery
