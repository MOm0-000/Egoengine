"""The frozen reset protocol must fail closed when t0 native geometry is illegal."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_taco_replay_rl import load_accepted_initialization

PROTOCOL = ROOT / "configs/taco_pour_initialization_protocol_v1.yaml"
RUN = ROOT / "runs/taco_pour_initialization_protocol_v1"
PPO = ROOT / "configs/taco_pour_bimanual_ppo.yaml"


def test_protocol_freezes_reset_only_contract_and_inputs():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    assert protocol["protocol_name"] == "taco_pour_initialization_protocol_v1"
    assert protocol["scope"] == "reset_only" and not protocol["training_ready"]
    formal = protocol["formal_inputs"]
    for key in ("scene", "reference"):
        path = Path(formal[key])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == formal[f"{key}_sha256"]
    assert formal["reference_index"] == 0
    assert (formal["nconmax_per_env"], formal["njmax_per_env"]) == (128, 512)
    assert protocol["hand_solver"]["solver_primal_tolerance"] == 1e-6
    assert protocol["hand_solver"]["solver_dual_tolerance"] == 1e-6
    assert protocol["hand_solver"]["planning_collision_buffer_m"] == 2e-6
    assert protocol["hand_solver"]["accepted_min_self_collision_distance_m"] == -1e-6
    assert protocol["candidate_b"]["preroll_control_intervals"] == 10
    assert protocol["passive_release_validation"]["control_intervals"] == 5
    assert protocol["t0_legality_gate"]["ctrlrange_numerical_tolerance"] == 1e-12


def test_both_artifacts_have_the_complete_common_state_contract():
    reference_path = Path(yaml.safe_load(PROTOCOL.read_text())["formal_inputs"]["reference"])
    with np.load(reference_path, allow_pickle=False) as source:
        object_qpos = source["qpos"][0, 36:]
    for key in ("candidate_a", "candidate_b"):
        with np.load(RUN / key / "initial_state.npz", allow_pickle=False) as state:
            assert state["qpos"].shape == (50,)
            assert state["qvel"].shape == (48,)
            assert state["ctrl"].shape == (36,)
            np.testing.assert_array_equal(state["qpos"][36:], object_qpos)
            np.testing.assert_array_equal(state["qvel"], np.zeros(48))
            np.testing.assert_array_equal(state["ctrl"], state["qpos"][:36])
            assert state["state_contract_version"].item() == "egoengine_replay_rl_initial_state_v1"
            assert state["reference_index"].item() == 0
            assert state["first_command_reference_index"].item() == 1
            assert state["post_release_validation_steps"].item() == 5
            assert state["object_constraints_released"].item()


def test_candidate_a_is_rejected_only_after_runtime_checks_reach_native_gate():
    builder = json.loads((RUN / "candidate_a/builder_report.json").read_text())
    gate = builder["t0_legality_gate"]
    assert builder["construction_complete"]
    assert builder["solver"]["solver_primal_tolerance"] == 1e-6
    assert builder["solver"]["solver_dual_tolerance"] == 1e-6
    assert builder["solver"]["minimum_distance_m"] == 2e-6
    assert builder["solver"]["accepted_min_distance_m"] == -1e-6
    assert all(value for key, value in gate["checks"].items() if key != "native_geometry")
    assert not gate["checks"]["native_geometry"] and not gate["passed"]
    failures = gate["native_collision_audit"]["failures"]
    assert ["left_middle_link2_visual", "left_object_visual"] in failures["hand_object"]
    assert ["left_pinky_link2_visual", "left_object_visual"] in failures["hand_object"]
    assert ["left_hand_link_visual", "left_thumb_rota1_visual"] in failures["omitted_nonadjacent"]
    assert not failures["unclassified_omitted_nonadjacent"]
    first = builder["first_command_diagnostic"]
    assert first["reference_ctrl_1_within_ctrlrange"]
    assert 0 < first["reference_ctrl_1_max_raw_violation"] < 1e-12


def test_candidate_b_stops_before_hold_or_release_when_depenetration_fails():
    builder = json.loads((RUN / "candidate_b/builder_report.json").read_text())
    assert not builder["construction_complete"]
    assert builder["solver"]["retained_feasible_iteration"] is None
    assert builder["solver"]["termination_reason"].startswith("qp_failed")
    assert not builder["held_object_preroll"]["executed"]
    assert builder["held_object_preroll"]["physics_steps"] == 0
    assert not builder["t0_legality_gate"]["passed"]


def test_release_validation_and_runner_remain_fail_closed():
    for key in ("candidate_a", "candidate_b"):
        report_path = RUN / key / "report.json"
        report = json.loads(report_path.read_text())
        assert not report["accepted_for_replay_rl"] and not report["training_ready"]
        assert report["physics_contract"]["schema"] == "egoengine_replay_rl_physics_v1"
        assert not report["release_validation"]["passed"]
        assert report["release_validation"]["failure_reason"] == "t0_legality_gate_failed"
        assert report["release_validation"]["physics_steps"] == 0
        assert not report["release_validation"]["formal_environment_created"]
        assert not report["release_validation"]["backend_setup_state_discarded_before_release"]
        assert not report["release_validation"]["reference_cursor_advanced"]
        assert not report["release_validation"]["reward_or_objective_computed"]
        try:
            load_accepted_initialization(report_path, PPO)
        except ValueError as error:
            assert "has not passed" in str(error)
        else:
            raise AssertionError("a rejected reset entered the Replay->RL runner")
    comparison = json.loads((RUN / "comparison.json").read_text())
    assert comparison["retained_candidates"] == []
    assert comparison["selection_deferred_until_observation_contract"]
    assert not comparison["training_ready"]
