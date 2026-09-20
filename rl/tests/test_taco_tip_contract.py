"""The diagnostic changes a measuring convention, not physics or released GT."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_taco_initialization import visual_meshes
from audit_taco_tip_semantics import point_geometry, unit
from diagnose_taco_tip_contract import build_contract, load_inputs, rotation_steps, write_scene, verify_scene, run
from egoengine_repro.retarget.paper_audit import verify_artifacts
from egoengine_repro.retarget.taco_bimanual import FINGERS, SIDES


@pytest.fixture(scope="module")
def inputs():
    scene, _, human, robot, _ = load_inputs()
    target, contract = build_contract(scene, human)
    return scene, human, robot, target, contract


def test_gt_positions_objects_wrists_and_rotation_steps_are_unchanged(inputs):
    _, human, _, target, contract = inputs
    for key in human:
        if key not in ("T_sim_fingertip_target", "fingertip_orientation_source"):
            np.testing.assert_array_equal(target[key], human[key])
    old, new = [r["T_sim_fingertip_target"] for r in (human, target)]
    np.testing.assert_array_equal(old[..., :3, 3], new[..., :3, 3])
    np.testing.assert_allclose(rotation_steps(old[..., :3, :3]), rotation_steps(new[..., :3, :3]), atol=1e-12)
    assert contract["not_author_calibration"] and contract["not_formal_initializer"]
    assert not contract["fitted_to_episode"]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("finger", FINGERS)
def test_surface_point_and_frame_definitions(inputs, side, finger):
    scene, human, _, target, contract = inputs
    model = mujoco.MjModel.from_xml_path(str(scene))
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    meshes, _ = visual_meshes(scene, model)
    name = f"{side}_{finger}_tip"
    record = contract["sites"][name]
    sid, bid = model.site(name).id, model.body(record["body"]).id
    mesh = next(mesh for gid, mesh in meshes.items() if model.geom_bodyid[gid] == bid)
    point = np.asarray(record["surface_site_local_m"])
    assert trimesh.proximity.closest_point(mesh, point[None])[1][0] < 1e-10
    nearest = trimesh.proximity.closest_point(mesh, np.asarray(record["urdf_endpoint_local_m"])[None])[0][0]
    np.testing.assert_allclose(nearest, point, atol=1e-12)
    old = np.asarray(record["inherited_anatomical_frame_site_local"])
    new = np.asarray(record["candidate_anatomical_frame_site_local"])
    for frame in (old, new):
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.)
        np.testing.assert_allclose(np.cross(frame[:, 2], frame[:, 0]), frame[:, 1], atol=1e-12)
    world_new = zero.site_xmat[sid].reshape(3, 3) @ new
    body_axis = zero.xmat[bid].reshape(3, 3) @ unit(record["urdf_endpoint_local_m"])
    np.testing.assert_allclose(world_new[:, 2], body_axis, atol=1e-12)
    hi, fi = SIDES.index(side), FINGERS.index(finger)
    # Independent comparison of physical anatomical frames, not raw site axes.
    human_frame = human["T_sim_fingertip_target"][:, hi, fi, :3, :3] @ old
    target_frame = target["T_sim_fingertip_target"][:, hi, fi, :3, :3] @ new
    np.testing.assert_allclose(human_frame, target_frame, atol=1e-12)


def test_scene_physics_and_contact_aliases_do_not_change(inputs, tmp_path):
    scene, _, robot, _, contract = inputs
    candidate = tmp_path / "scene.xml"
    checks = write_scene(scene, candidate, contract)
    assert checks["changed_site_count"] == 10
    assert checks["physical_XML_unchanged"] and checks["contact_and_trace_aliases_unchanged"]
    models = [mujoco.MjModel.from_xml_path(str(p)) for p in (scene, candidate)]
    for q in robot["qpos"][[0, 100, 197]]:
        states = []
        for model in models:
            data = mujoco.MjData(model)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            states.append(data)
        # MuJoCo 3.12 renamed the dense mass matrix field from qM to M.
        for attr in ("xpos", "xmat", "geom_xpos", "geom_xmat", "qfrc_bias"):
            np.testing.assert_array_equal(getattr(states[0], attr), getattr(states[1], attr))
        mass_field = "qM" if hasattr(states[0], "qM") else "M"
        np.testing.assert_array_equal(getattr(states[0], mass_field), getattr(states[1], mass_field))
        aliases = [i for i in range(models[0].nsite)
                   if models[0].site(i).name.startswith(("track_hand_", "trace_hand_"))]
        assert len(aliases) == 20
        np.testing.assert_array_equal(states[0].site_xpos[aliases], states[1].site_xpos[aliases])
    with pytest.raises(FileExistsError):
        write_scene(scene, candidate, contract)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    modified = ET.parse(candidate)
    modified.getroot().find(".//geom[@name='floor']").set("pos", "0 0 0.7")
    changed = tmp_path / "bad_scene.xml"
    modified.write(changed)
    with pytest.raises(ValueError, match="scene changed beyond"):
        verify_scene(scene, changed, contract)


def test_surface_audit_keeps_unsigned_distance_and_checks_provenance():
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    values = point_geometry(mesh, [.25, .5, 0.], [0, 1, 0])
    assert values["unsigned_surface_distance_m"] == pytest.approx(.5)
    assert values["axial_gap_to_farthest_native_vertex_m"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="undefined axis"):
        unit([0, 0, 0])
    report = json.loads((ROOT / "runs/taco_pour_tip_semantics_v1/report.json").read_text())
    verify_artifacts(report["inputs"] + report["code"])
    from egoengine_repro.evaluation.taco_surface import MANO_TIP_VERTICES
    for side in SIDES:
        for index, finger in enumerate(FINGERS):
            record = report["per_hand"][side]["fingers"][finger]
            assert record["human"]["episode_shape"]["vertex_id"] == MANO_TIP_VERTICES[side][index]
            assert record["robot"]["inherited_site"]["unsigned_surface_distance_m"] > .003
            assert record["robot"]["named_urdf_tip"]["unsigned_surface_distance_m"] > .001


def test_saved_candidate_matches_independent_body_local_point_measurement(inputs):
    scene, human, robot, target, contract = inputs
    directory = ROOT / "runs/taco_pour_tip_contract_v1"
    report = json.loads((directory / "comparison.json").read_text())
    saved_contract = json.loads((directory / "geometry_contract.json").read_text())
    verify_artifacts(saved_contract["inputs"])
    # Generation hashes remain historical, not silently rewritten after solver
    # maintenance. Rebuild the entire geometry contract with current code and
    # compare all values; the archived trajectory is independently measured below.
    verify_artifacts([row for row in saved_contract["code"]
                      if Path(row["path"]).name not in {"taco_bimanual.py", "mink.py"}])
    for key, value in contract.items():
        assert saved_contract[key] == value
    baseline_report = json.loads((ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1/retarget_report.json").read_text())
    candidate_report = json.loads((directory / "retarget_report.json").read_text())
    assert candidate_report["inherited_settings"] == baseline_report["inherited_settings"]
    assert not report["reference_overwritten"] and not report["physics_validated"]
    with np.load(directory / "robot_reference.npz", allow_pickle=False) as src:
        candidate = dict(src)
    with np.load(directory / "human_reference.npz", allow_pickle=False) as src:
        for key in target:
            np.testing.assert_array_equal(src[key], target[key])
    with np.load(ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1/robot_reference.npz",
                 allow_pickle=False) as src:
        robot = dict(src)
    np.testing.assert_allclose(candidate["qpos"][:, 36:], robot["qpos"][:, 36:], atol=1e-12, rtol=0)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for state_name, states in (("original", robot["qpos"]), ("candidate", candidate["qpos"])):
        for point_name, key in (("inherited", "inherited_site_local_m"), ("surface", "surface_site_local_m")):
            measured = []
            for row, q in enumerate(states):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                errors = np.empty((2, 5))
                for hi, side in enumerate(SIDES):
                    for fi, finger in enumerate(FINGERS):
                        record = contract["sites"][f"{side}_{finger}_tip"]
                        body = model.body(record["body"]).id
                        world = data.xmat[body].reshape(3, 3) @ record[key] + data.xpos[body]
                        errors[hi, fi] = np.linalg.norm(world - human["T_sim_fingertip_target"][row, hi, fi, :3, 3])
                measured.append(errors)
            comparison = report["comparison"][f"{point_name}_points_at_{state_name}_qpos"]
            np.testing.assert_allclose(measured, comparison["position_error_m"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.mean(measured, axis=(0, 2)), comparison["mean_position_m_by_hand"], atol=1e-12)
