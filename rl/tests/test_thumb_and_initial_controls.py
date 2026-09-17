"""Native thumb evidence and instantaneous controls must not become hidden resets."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_taco_initial_controls import contact_normal_velocity, forward_case, pose_matched_ctrl, run as run_controls
from audit_taco_initialization import visual_meshes
from audit_taco_thumb_assembly import kinematic_source_check, native_intersection
from diagnose_taco_thumb_clearance import bisect_clear_endpoint, run as run_thumb
from egoengine_repro.retarget.paper_audit import artifact

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
BASELINE = ROOT / "runs/taco_pour_bimanual_gt_v1"
PREVIOUS = ROOT / "runs/taco_pour_initial_hand_v3/initial_hand_candidate.npz"
LATEST = ROOT / "runs/taco_pour_initial_hand_v4"
SWEEP = ROOT / "runs/taco_pour_thumb_assembly_v1/report.json"


def read_report(path):
    return json.loads(path.read_text())


def test_bisection_retains_a_tested_clear_endpoint():
    calls = []

    def intersects(angle):
        calls.append(angle)
        return angle >= .7

    clear, blocked = bisect_clear_endpoint(intersects, 0, 1)
    assert clear < .7 <= blocked and blocked - clear <= 1e-6
    assert clear in calls


@pytest.mark.parametrize("clear,blocked,tolerance", [(1, 0, 1e-6), (0, 1, 0), (0, 1, np.nan)])
def test_bisection_rejects_invalid_brackets(clear, blocked, tolerance):
    with pytest.raises(ValueError, match="finite ordered"):
        bisect_clear_endpoint(lambda _: True, clear, blocked, tolerance)


def test_bisection_does_not_invent_a_clear_seed():
    with pytest.raises(ValueError, match="clear lower"):
        bisect_clear_endpoint(lambda _: True, 0, 1)


def test_source_kinematics_match_and_do_not_supply_a_collision_exemption():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for side in ("right", "left"):
        report = kinematic_source_check(model, SWEEP.parent / f"xhand_{side}.urdf", side)
        assert not report["source_has_pair_exemption"]
        for joint in report["joints"]:
            assert joint["origin_translation_error_m"] == 0
            assert joint["origin_rotation_error_rad"] < 2e-6
            assert joint["joint_axis_error"] == joint["limit_max_error_rad"] == 0


def test_full_native_intersection_is_pose_dependent_and_wrist_invariant():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    meshes, _ = visual_meshes(SCENE, model)
    data = mujoco.MjData(model)
    results = []
    for path in (BASELINE / "robot_reference.npz", PREVIOUS, LATEST / "initial_hand_candidate.npz"):
        with np.load(path, allow_pickle=False) as source:
            data.qpos[:] = source["qpos"][0]
        mujoco.mj_forward(model, data)
        results.append(native_intersection(model, data, meshes, "left"))
    assert results[0]["intersection_nonempty"] and results[1]["intersection_nonempty"]
    assert results[0]["intersection_volume_m3"] == pytest.approx(204.288967e-9, rel=1e-6)
    assert results[0]["intersection_volume_m3"] == results[1]["intersection_volume_m3"]
    assert not results[2]["intersection_nonempty"]
    model_mesh = meshes[model.geom("left_hand_link_visual").id]
    with pytest.raises(ValueError, match="closed"):
        native_intersection(model, data, {**meshes, model.geom("left_hand_link_visual").id: model_mesh.submesh([[0]], append=True)}, "left")


def test_latest_candidate_changes_one_parent_coordinate_and_no_reference_objects():
    report = read_report(LATEST / "report.json")
    with np.load(PREVIOUS, allow_pickle=False) as source:
        previous = source["qpos"][0]
    with np.load(LATEST / "initial_hand_candidate.npz", allow_pickle=False) as source:
        candidate = source["qpos"][0]
        assert source["qpos"].shape == (1, 50)
        assert bool(source["diagnostic_only"]) and not bool(source["accepted_as_reset"])
        assert "qvel" not in source and "ctrl" not in source
    assert np.flatnonzero(candidate != previous).tolist() == [25]
    np.testing.assert_array_equal(candidate[36:], previous[36:])
    assert report["candidate_declared_and_audited_thumb_feasible"]
    assert not report["solver"]["globally_minimum_correction_proven"]
    assert not report["solver"]["positive_geometric_clearance_certified"]
    independent = read_report(LATEST / "audit.json")
    assert independent["independent_state_check"]["declared_state_feasible"]
    assert independent["native_hand_object_aabb_pairs_checked"] == 52
    assert independent["native_hand_object_sampled_inside_points"] == 0
    assert independent["frozen_coordinate_reference"] == artifact(PREVIOUS)


def test_all_diagnostics_preserve_inputs_and_remain_unaccepted():
    for path in (SWEEP, LATEST / "report.json", LATEST / "audit.json", LATEST / "initial_control_audit.json"):
        report = read_report(path)
        assert not report["accepted_as_reset"] and report["simulation_steps_executed"] == 0
        for record in report["preserved_artifacts"]:
            assert artifact(Path(record["path"])) == record
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    scope = protocol["initialization"]["hand_pose_diagnostic_scope"]
    assert scope["status"] == "temporary_single_sample_test"
    assert not scope["part_of_formal_pipeline"] and not scope["automatic_use_for_new_samples"]
    assert scope["cross_sample_generalization"] == "not_established"
    assert not protocol["training_ready"]
    assert Path(protocol["inputs"]["historical_orientation_bug_baseline"]) == BASELINE
    assert Path(protocol["inputs"]["active_robot_reference"]) == (
        ROOT / "runs/taco_pour_bimanual_mano_fk_v1/robot_reference.npz")
    assert protocol["initialization"]["diagnostic_candidate"]["baseline"] == (
        "historical_orientation_bug_reference_not_current_mano_fk")


def test_new_runners_reject_existing_outputs():
    with pytest.raises(FileExistsError):
        run_thumb(SCENE, BASELINE, PREVIOUS, SWEEP, LATEST)
    with pytest.raises(FileExistsError):
        run_controls(SCENE, BASELINE / "robot_reference.npz", PREVIOUS, LATEST / "initial_control_audit.json")


def test_pose_matched_command_uses_transmission_and_rejects_nonposition_servo():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j" type="slide"/><geom size=".1"/></body></worldbody><actuator><position joint="j" kp="10" gear="2"/></actuator></mujoco>')
    np.testing.assert_allclose(pose_matched_ctrl(model, np.array([.3])), [.6])
    model.actuator_biasprm[0, 1] = 0
    with pytest.raises(ValueError, match="position servos"):
        pose_matched_ctrl(model, np.array([.3]))


def test_contact_normal_velocity_sign_means_separation():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    for velocity in (-.2, .2):
        data.qvel[0] = velocity
        mujoco.mj_forward(model, data)
        assert data.ncon == 1
        assert contact_normal_velocity(model, data, data.contact[0]) == pytest.approx(velocity)


def test_forward_audit_does_not_step_or_mutate_the_input(monkeypatch):
    def forbidden(*_):
        raise AssertionError("physical integration not authorized in this diagnostic")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(LATEST / "initial_hand_candidate.npz", allow_pickle=False) as source:
        qpos = source["qpos"][0]
    original = qpos.copy()
    ctrl = pose_matched_ctrl(model, qpos)
    report = forward_case(model, qpos, np.zeros(model.nv), ctrl)
    np.testing.assert_array_equal(qpos, original)
    assert report["time_s"] == 0 and report["simulation_steps_executed"] == 0
    assert all(abs(a["actuator_force"]) < 1e-10 for a in report["actuators"])
    assert not report["stability_or_task_success_claim"]


def test_reference_command_pulls_toward_the_preserved_reference_not_candidate():
    report = read_report(LATEST / "initial_control_audit.json")
    assert len(report["cases"]) == 4
    assert not report["qvel_selected"] and not report["ctrl_selected"]
    original = next(c for c in report["cases"] if c["velocity"] == "zero" and c["command"] == "original_reference")
    force = {a["actuator"]: a["generalized_force"] for a in original["actuators"]}
    assert force["R_forearm_tz_position"] == pytest.approx(-14.877252239242078)
    assert force["L_forearm_tz_position"] == pytest.approx(-30.24199903635727)
    assert force["left_thumb_rota1_position"] == pytest.approx(17.08206338747753)
    assert all(not c["stability_or_task_success_claim"] for c in report["cases"])
