"""Independent checks for the isolated objective and reachability diagnostics."""

from pathlib import Path
import json
import sys

import mink
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_taco_retarget_objectives import (
    Experiment, WorldPositionTask, fixed_axis_residual, load_inputs, semantic_audit,
)


@pytest.fixture(scope="module")
def experiment():
    scene, settings, human, robot, _ = load_inputs()
    return Experiment(scene, settings, human, robot)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("finger", ["thumb", "index", "middle", "ring", "pinky"])
def test_world_position_jacobian_matches_finite_difference(experiment, side, finger):
    exp = experiment
    q = exp.robot["qpos"][0].copy()
    exp.config.update(q)
    task = WorldPositionTask(exp.model, f"{side}_{finger}_tip", [0., 0., 0.])
    jac = task.compute_jacobian(exp.config)
    numeric = np.zeros_like(jac)
    for j in range(36):
        plus, minus = q.copy(), q.copy()
        plus[j] += 1e-7
        minus[j] -= 1e-7
        exp.config.update(plus)
        eplus = task.compute_error(exp.config).copy()
        exp.config.update(minus)
        numeric[:, j] = (eplus - task.compute_error(exp.config)) / 2e-7
    exp.config.update(q)
    np.testing.assert_allclose(jac, numeric, atol=3e-9, rtol=0)


def test_zero_rotation_cost_is_not_a_pure_euclidean_position_loss(experiment):
    exp = experiment
    exp.config.update(exp.robot["qpos"][0])
    sid = exp.model.site("right_index_tip").id
    current_R = exp.config.data.site_xmat[sid].reshape(3, 3)
    target = np.eye(4)
    target[:3, 3] = exp.config.data.site_xpos[sid] + current_R @ [.01, 0., 0.]
    target[:3, :3] = current_R @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose = mink.FrameTask("right_index_tip", "site", position_cost=10., orientation_cost=0.)
    pose.set_target(mink.SE3.from_matrix(target))
    point = WorldPositionTask(exp.model, "right_index_tip", target[:3, 3])
    assert np.linalg.norm(point.compute_error(exp.config)) == pytest.approx(.01)
    assert np.linalg.norm(pose.compute_error(exp.config)[:3]) == pytest.approx(.01 * np.pi / (2 * np.sqrt(2)))
    before = point.compute_error(exp.config).copy()
    target[:3, :3] = current_R
    pose.set_target(mink.SE3.from_matrix(target))
    np.testing.assert_array_equal(point.compute_error(exp.config), before)
    assert np.linalg.norm(pose.compute_error(exp.config)[:3]) == pytest.approx(.01)


def test_fixed_axis_lower_bound_is_translation_invariant():
    targets = np.array([.1, .13, .135])
    fixed = np.array([0, .02, .04])
    residual = fixed_axis_residual(targets, fixed)
    np.testing.assert_allclose(fixed_axis_residual(targets + 1.23, fixed), residual, atol=1e-15)
    optimum = np.mean(targets - fixed)
    for shift in np.linspace(-.1, .1, 13):
        assert np.sum((targets - fixed - optimum - shift) ** 2) >= np.sum(residual ** 2) - 1e-15


def test_source_kinematic_parameters_and_coordinate_invariant(experiment):
    exp = experiment
    scene, _, _, _, _ = load_inputs()
    audit = semantic_audit(scene, exp.human, exp.robot)
    for side in ("right", "left"):
        checks = audit["site_provenance"][side]
        assert len(checks["urdf_joint_checks"]) == 12
        assert max(r["range_max_difference_rad"] for r in checks["urdf_joint_checks"]) < 1e-12
        assert max(r["axis_max_difference"] for r in checks["urdf_joint_checks"]) < 1e-12
        assert checks["sites"]["pinky"]["site_to_urdf_tip_distance_m"] == pytest.approx(.01600781059358)
        assert checks["sites"]["pinky"]["site_axis_vs_urdf_endpoint_axis_deg"] == pytest.approx(8.42696902534)
        assert audit["fixed_wrist_axis_diagnostic"][side]["maximum_random_joint_test_drift_m"] < 1e-10
        assert audit["independent_urdf_fk_vs_mujoco"][side]["max_body_position_difference_m"] < 1e-6
        assert audit["independent_urdf_fk_vs_mujoco"][side]["max_body_rotation_difference_rad"] < 1e-5


def test_first_frame_objective_removal_keeps_declared_constraints(experiment):
    exp = experiment
    original = exp.robot["qpos"].copy()
    baseline, _ = exp.solve(0, "pose", 160)
    position, q = exp.solve(0, "position_wrist", 160)
    assert baseline["converged"]
    assert position["constraint_gate_passed"]
    assert np.mean(position["position_error_m"]) < np.mean(baseline["position_error_m"]) / 2
    np.testing.assert_allclose(q[36:], original[0, 36:], atol=1e-12)
    np.testing.assert_array_equal(exp.robot["qpos"], original)
    probe = exp.relaxed_bound_probe(0, "position_wrist", q)
    assert all(not r["state_integrated"] for r in probe["probes"])
    np.testing.assert_array_equal(exp.config.q, q)


@pytest.mark.parametrize("row", [0, 100, 197])
@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("finger", ["thumb", "pinky"])
def test_actual_frame_task_jacobian_matches_finite_difference(experiment, row, side, finger):
    exp = experiment
    q = exp.robot["qpos"][row].copy()
    exp.config.update(q)
    hi = ("right", "left").index(side)
    fi = ("thumb", "index", "middle", "ring", "pinky").index(finger)
    task = mink.FrameTask(f"{side}_{finger}_tip", "site", position_cost=10., orientation_cost=1.)
    task.set_target(mink.SE3.from_matrix(exp.human["T_sim_fingertip_target"][row, hi, fi]))
    jac = task.compute_jacobian(exp.config)
    numeric = np.zeros_like(jac)
    for j in range(36):
        plus, minus = q.copy(), q.copy()
        plus[j] += 1e-7
        minus[j] -= 1e-7
        exp.config.update(plus)
        error = task.compute_error(exp.config).copy()
        exp.config.update(minus)
        numeric[:, j] = (error - task.compute_error(exp.config)) / 2e-7
    exp.config.update(q)
    np.testing.assert_allclose(jac, numeric, atol=7e-9, rtol=0)
    weighted_jac, weighted_error, _ = task.compute_qp_residual(exp.config)
    np.testing.assert_allclose(weighted_jac, task.cost[:, None] * jac, atol=1e-12)
    np.testing.assert_allclose(weighted_error, -task.cost * task.compute_error(exp.config), atol=1e-12)


def test_shared_axis_orientation_bound_holds_for_arbitrary_actual_common_axis():
    from audit_taco_finger_axis_compatibility import pairwise_orientation_rms_lower_bound

    rng = np.random.default_rng(331)
    target = Rotation.random(90, random_state=rng).as_matrix().reshape(30, 3, 3, 3)
    _, lower = pairwise_orientation_rms_lower_bound(target)
    for _ in range(5):
        base = Rotation.random(30, random_state=rng).as_matrix()
        bends = Rotation.from_euler("x", rng.uniform(-np.pi, np.pi, (90, 1))).as_matrix().reshape(30, 3, 3, 3)
        actual = base[:, None] @ bends
        errors = np.rad2deg(Rotation.from_matrix((actual.swapaxes(-1, -2) @ target).reshape(-1, 3, 3)).magnitude()).reshape(30, 3)
        assert np.all(np.sqrt(np.mean(errors ** 2, axis=1)) >= lower - 1e-10)


@pytest.mark.parametrize("mode", ["pose", "split_pose", "se3_no_tip_rotation", "position_wrist", "position_only"])
def test_saved_diagnostics_preserve_inputs_and_match_independent_site_measurement(experiment, mode):
    exp = experiment
    directory = ROOT / "runs/taco_pour_objective_audit_v2"
    report = json.loads((directory / f"{mode}.json").read_text())
    with np.load(directory / "diagnostic_endpoints_not_reference.npz", allow_pickle=False) as src:
        states = src[mode]
    assert states.shape == exp.robot["qpos"].shape
    with np.load(ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1/robot_reference.npz",
                 allow_pickle=False) as source:
        historical_objects = source["qpos"][:, 36:]
    np.testing.assert_allclose(states[:, 36:], historical_objects, atol=1e-12, rtol=0)
    data = mujoco.MjData(exp.model)
    measured = []
    for row, q in enumerate(states):
        data.qpos[:] = q
        mujoco.mj_forward(exp.model, data)
        errors = []
        for hi, side in enumerate(("right", "left")):
            ids = [exp.model.site(f"{side}_{f}_tip").id for f in ("thumb", "index", "middle", "ring", "pinky")]
            errors.append(np.linalg.norm(data.site_xpos[ids] - exp.human["T_sim_fingertip_target"][row, hi, :, :3, 3], axis=-1))
        measured.append(errors)
        np.testing.assert_allclose(errors, report["rows"][row]["position_error_m"], atol=1e-12, rtol=0)
    assert report["summary"]["valid_rows"] == 198
    np.testing.assert_allclose(np.mean(measured, axis=(0, 2)), report["summary"]["mean_position_m_by_hand"], atol=1e-12)
