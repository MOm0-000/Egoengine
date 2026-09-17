"""Pour owns its geometry/reference; kinematic feasibility is not a physics pass."""

import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "diagnostics")]
from build_taco_bimanual_scene import selected_objects, snapshot_robot_assets
from egoengine_repro.retarget.collision_audit import explicit_hand_pairs, hand_ids
from egoengine_repro.retarget.mink import _enable_planning_collision_masks, _explicit_collision_groups
from egoengine_repro.retarget.paper_audit import artifact
from egoengine_repro.retarget.taco_bimanual import differentiate, pose7
from render_exact_deximit_triptych import load_reference_model, map_sim_to_reference_style

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
RUN = ROOT / "runs/taco_pour_bimanual_gt_v1"
BUNDLE = ROOT / "data/taco_v1/pour_bowl_plate"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def test_selected_objects_require_explicit_ids_and_keep_legacy_defaults():
    assert selected_objects()["right"]["object_id"] == "071"
    with pytest.raises(ValueError):
        selected_objects(BUNDLE)
    with pytest.raises(ValueError):
        selected_objects(BUNDLE, "../022", "135")
    specs = selected_objects(BUNDLE, "022", "135")
    assert all(s["visual_scale"] == .01 and s["collision_subdir"] == "convex_m" for s in specs.values())
    assert specs["right"]["source"].name == "022_cm.obj"


def test_snapshot_resolves_source_link_but_never_links_or_overwrites_output(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "hand.stl").write_bytes(b"source robot mesh")
    assets = tmp_path / "local"
    (assets / "robots").mkdir(parents=True)
    (assets / "robots/xhand").symlink_to(source, target_is_directory=True)
    xml = '<mujoco><asset><mesh name="hand" file="robots/xhand/hand.stl"/></asset></mujoco>'
    root = ET.fromstring(xml)
    report = snapshot_robot_assets(root, assets, "pour")
    output = assets / root.find("asset/mesh").get("file")
    assert output.read_bytes() == (source / "hand.stl").read_bytes()
    assert not output.is_symlink() and output.resolve().is_relative_to(assets)
    assert report[0]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        snapshot_robot_assets(ET.fromstring(xml), assets, "pour")
    with pytest.raises(ValueError):
        snapshot_robot_assets(ET.fromstring(xml), assets, "../escape")


def test_pour_scene_has_passive_objects_and_complete_cross_object_pairs(model):
    assert (model.nq, model.nv, model.nu) == (50, 48, 36)
    assert model.geom("floor").pos[2] == pytest.approx(.72)
    assert not model.geom_contype.any() and not model.geom_conaffinity.any()
    objects = []
    for side in ("right", "left"):
        joint = model.joint(f"{side}_object_joint")
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert joint.id not in model.actuator_trnid[:, 0]
        objects.append([i for i in range(model.ngeom) if model.geom(i).name.startswith(f"{side}_object_")
                        and not model.geom(i).name.endswith("visual")])
    assert list(map(len, objects)) == [32, 32]
    pairs = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    assert all(frozenset(pair) in pairs for pair in product(*objects))
    assert all(frozenset(pair) in pairs for pair in product(hand_ids(model), sum(objects, [])))
    assert model.npair == 2822


def test_pour_assets_resolve_inside_project_with_native_metric_inertias(model):
    root = ET.parse(SCENE).getroot()
    assets = SCENE.parent / root.find("compiler").get("meshdir")
    for entry in root.findall("asset/mesh"):
        path = assets / entry.get("file")
        assert path.resolve().is_relative_to(ROOT)
        assert path.is_file() and not path.is_symlink()
    report = json.loads(SCENE.with_suffix(".build.json").read_text())
    assert report["object_roles"] == {"right_object": "tool_022", "left_object": "target_135"}
    assert len(report["independent_robot_asset_snapshot"]) == 26
    for item in report["independent_robot_asset_snapshot"]:
        assert hashlib.sha256(Path(item["output"]).read_bytes()).hexdigest() == item["sha256"]
    for side in ("right", "left"):
        visual = root.find(f"asset/mesh[@name='{side}_visual']")
        np.testing.assert_allclose(np.fromstring(visual.get("scale"), sep=" "), [.01] * 3)
        assert model.body(f"{side}_object").mass[0] == pytest.approx(report["object_inertials"][side]["mass_kg"])
        assert "not a measured" in report["object_inertials"][side]["density_provenance"]
        assert np.linalg.eigvalsh(report["object_inertials"][side]["inertia_kg_m2"]).min() > 0
    parts = [e for e in root.findall("asset/mesh") if "/convex_m/" in e.get("file", "")]
    assert len(parts) == 64 and all(e.get("scale") is None for e in parts)


def test_pour_ik_runtime_pair_equality():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    pairs = explicit_hand_pairs(model)
    assert len(pairs) == 174
    hands = set(hand_ids(model))
    _enable_planning_collision_masks(model, list(hands), [])
    limit = mink.CollisionAvoidanceLimit(model, _explicit_collision_groups(model, mujoco, hand_geom_ids=hands))
    assert set(pairs) == set(limit.geom_id_pairs)


def test_pour_reference_preserves_gt_and_all_source_frames(model):
    with np.load(RUN / "human_reference.npz", allow_pickle=False) as data:
        human = dict(data)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as data:
        robot = dict(data)
    assert robot["qpos"].shape == (198, 50)
    assert robot["qvel"].shape == (198, 48)
    assert robot["ctrl"].shape == (198, 36)
    np.testing.assert_array_equal(robot["frame_indices"], np.arange(198))
    np.testing.assert_array_equal(human["frame_indices"], robot["frame_indices"])
    sequence = "(pour in some, bowl, plate)/20230927_017"
    objects = np.stack([np.load(BUNDLE / "object_poses/Object_Poses" / sequence / name).astype(float)
                        for name in ("tool_022.npy", "target_135.npy")], axis=1)
    np.testing.assert_allclose(human["T_sim_world"] @ objects, human["T_sim_object_reference"], atol=1e-12)
    for index, side in enumerate(("right", "left")):
        address = int(model.joint(f"{side}_object_joint").qposadr[0])
        expected = np.stack([pose7(t) for t in human["T_sim_object_reference"][:, index]])
        # Saved references may have been serialized under an older MuJoCo/
        # NumPy stack; allow only the observed sub-micrometre round-trip noise.
        np.testing.assert_allclose(robot["qpos"][:, address:address + 7], expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(robot["qvel"], differentiate(model, robot["qpos"], 1 / 30), atol=1e-12)


def test_pour_kinematic_pass_does_not_claim_physical_success():
    report = json.loads((RUN / "retarget_report.json").read_text())
    assert report["kinematic_model_feasible"]
    assert report["scene_sha256"] == artifact(SCENE)["sha256"]
    assert report["joint_limit_violating_frames"] == report["frame_velocity_violating_intervals"] == 0
    assert not report["complete_intrahand_geometry_coverage"]
    assert not report["strict_gate_passed"] and not report["rl_validation_completed"]


def test_pour_initialization_audit_preserves_inputs_and_rejects_a_physics_claim():
    report = json.loads((RUN / "initialization_audit.json").read_text())
    assert report["frames"] == 198 and report["sequence"].endswith("20230927_017")
    assert report["topology"]["ik_runtime_pairs_equal"]
    assert report["topology"]["hand_shell_geometry_equal_pinned_source"]
    assert report["alignment"]["native_compiled_scale_check_max_error_m"] < 1e-7
    assert not report["state_projection_applied"]
    assert report["simulation_steps_executed"] == 0
    assert not report["strict_gate_passed"]
    for record in report["preserved_artifacts"]:
        assert artifact(Path(record["path"])) == record


def test_pour_uses_existing_exact_renderer_mapping_without_rendering(model):
    visual, _ = load_reference_model(SCENE)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as data:
        qpos = data["qpos"][::25]
    mapped, report = map_sim_to_reference_style(model, visual, qpos,
        sim_left_object_joint="left_object_joint", ref_left_object_joint="left_object_joint")
    assert len(report["hands"]) == len(report["objects"]) == 2
    assert all(item["sampled_position_error_m_max"] < 1e-12 for item in report["sampled_hand_errors"])
    np.testing.assert_allclose(mapped[:, -14:], qpos[:, -14:], atol=1e-12)


def test_collision_parts_keep_source_hashes_and_disclose_approximation():
    for obj in ("022", "135"):
        directory = ROOT / "models/taco_xhand/assets/objects" / obj / "convex_m"
        report = json.loads((directory / "provenance.json").read_text())
        assert report["source_sha256"] == artifact(Path(report["source"]))["sha256"]
        assert report["parts"] == 32 and report["parameters"]["seed"] == 0
        assert not report["requested_threshold_certified"]
        assert "not published" in report["parameter_provenance"]
        for record in report["artifacts"]:
            assert record["sha256"] == artifact(Path(record["path"]))["sha256"]


def test_initial_native_contacts_provide_evidence_without_open_mesh_claims():
    report = json.loads((RUN / "initial_native_contact_audit.json").read_text())
    assert report["source_row_zero_based"] == 0
    assert report["simulation_steps_executed"] == 0
    assert not report["state_projection_applied"] and not report["physics_validated"]
    assert len(report["records"]) == 3
    for pair in report["records"]:
        forward, reverse = pair["directions"]
        assert forward["inside_samples_50um"] > 0
        assert forward["sampled_penetration_max_m"] > 0
        assert reverse["status"] == "target_open_mesh_signed_containment_not_used"
        assert "inside_samples_50um" not in reverse
    for source in report["preserved_artifacts"]:
        assert artifact(Path(source["path"])) == source
