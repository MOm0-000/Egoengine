"""The candidate's limited self model must never masquerade as full feasibility."""

from itertools import product
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_taco_bimanual_scene import _replace_contacts, _set_object_inertials
from egoengine_repro.retarget.collision_audit import (
    audit_intrahand_trajectory, explicit_hand_pairs, source_topology_report,
)
from egoengine_repro.retarget.mink import _enable_planning_collision_masks, _explicit_collision_groups

TEMPLATE = ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"
SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_source_contacts.xml"


def test_source_coverage_omissions_are_reported():
    report = source_topology_report(TEMPLATE)
    assert report["source_intrahand_pairs"] == 30
    assert report["source_interhand_pairs"] == 96
    assert len(report["omitted_intrahand_pairs"]) == 102


def test_contact_rebuild_preserves_shells_and_all_cross_hand_pairs():
    root = ET.parse(TEMPLATE).getroot()
    geometry_keys = ("type", "size", "pos", "quat")
    before = {g.get("name"): tuple(g.get(k) for k in geometry_keys)
              for g in root.findall(".//geom") if g.get("name", "").startswith("collision_hand_")}
    report = _replace_contacts(root)
    after = {g.get("name"): tuple(g.get(k) for k in geometry_keys)
             for g in root.findall(".//geom") if g.get("name", "").startswith("collision_hand_")}
    assert before == after
    assert report["pair_counts"]["full_interhand"] == 144
    assert report["complete_intrahand_geometry_coverage"] is False
    sides = [[g for g in before if g.startswith(f"collision_hand_{s}_")] for s in ("right", "left")]
    pairs = {frozenset((p.get("geom1"), p.get("geom2"))) for p in root.findall("contact/pair")}
    assert all(frozenset(pair) in pairs for pair in product(*sides))
    assert all(g.get("contype") == g.get("conaffinity") == "0" for g in root.findall(".//geom"))


def test_runtime_and_mink_have_identical_self_pairs():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    pairs = explicit_hand_pairs(model)
    assert len(pairs) == 174
    hands = {g for pair in pairs for g in pair}
    assert not model.geom_contype.any()
    _enable_planning_collision_masks(model, list(hands), [])
    limit = mink.CollisionAvoidanceLimit(model, _explicit_collision_groups(
        model, mujoco, hand_geom_ids=hands))
    assert set(limit.geom_id_pairs) == set(pairs)
    # The file/physics model is not mutated by MINK's private enumeration masks.
    assert not mujoco.MjModel.from_xml_path(str(SCENE)).geom_contype.any()


def test_intrahand_audit_enumerates_declared_and_omitted_pairs():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(ROOT / "runs/taco_brush_bimanual_gt_v4/robot_reference.npz",
                 allow_pickle=False) as source:
        report = audit_intrahand_trajectory(model, source["qpos"])
    assert report["pair_count"] == 132
    assert report["counts"] == {
        "declared": 30,
        "omitted_assembly_adjacent": 20,
        "omitted_nonadjacent": 82,
    }
    assert report["by_classification"]["omitted_nonadjacent"]["pair_count"] == 82
    assert "penetrating_pair_count" in report["by_classification"]["omitted_nonadjacent"]
    assert len(report["pairs"]) == 132
    assert all("minimum_distance_m" in pair and "worst_frame" in pair
               and "penetrating_frames" in pair for pair in report["pairs"])


def test_object_inertia_uses_native_volume_not_sum_of_hulls():
    root = ET.parse(SCENE).getroot()
    report = _set_object_inertials(root)
    assert report["right"]["mass_kg"] == pytest.approx(0.17936513290406047)
    assert report["left"]["mass_kg"] == pytest.approx(0.11328417068338886)
    for side in ("right", "left"):
        inertia = np.asarray(report[side]["inertia_kg_m2"])
        assert np.linalg.eigvalsh(inertia).min() > 0
        assert "not a measured" in report[side]["density_provenance"]
        assert len(root.findall(f"worldbody/body[@name='{side}_object']/inertial")) == 1


def test_latest_candidate_passes_only_its_declared_kinematic_model():
    import json
    run = ROOT / "runs/taco_brush_bimanual_gt_v4"
    report = json.loads((run / "retarget_report.json").read_text())
    assert report["kinematic_model_feasible"]
    assert report["joint_limit_violating_frames"] == 0
    assert report["frame_velocity_violating_intervals"] == 0
    assert report["self_collision_pair_count"] == 174
    assert not report["complete_intrahand_geometry_coverage"]
    assert not report["strict_gate_passed"]
    assert not report["rl_validation_completed"]
    with np.load(run / "robot_reference.npz", allow_pickle=False) as data:
        assert data["qpos"].shape == (209, 50)
        assert data["qvel"].shape == (209, 48)
        assert data["ctrl"].shape == (209, 36)
        assert data["frame_velocity_max_ratio"].max() <= 1 + 1e-6


def test_failed_qp_exports_only_a_marked_failure_prefix(tmp_path, monkeypatch):
    import json
    from egoengine_repro.retarget.taco_bimanual import retarget
    run = ROOT / "runs/taco_brush_bimanual_gt_v4"
    report = json.loads((run / "retarget_report.json").read_text())

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected QP failure")

    monkeypatch.setattr(mink, "solve_ik", fail)
    with pytest.raises(RuntimeError, match="injected QP failure"):
        retarget(Path(report["scene"]), run / "human_reference.npz",
                 report["inherited_settings"], tmp_path)
    assert not (tmp_path / "robot_reference.npz").exists()
    failure = json.loads((tmp_path / "retarget_failure.json").read_text())
    assert failure["failed_frame"] == 0
    assert not failure["strict_gate_passed"]
    with np.load(tmp_path / "failed_kinematic_prefix.npz", allow_pickle=False) as data:
        assert data["qpos"].shape == (1, 50)


def test_inertial_correction_keeps_approved_renderer_mapping():
    from render_exact_deximit_triptych import load_reference_model, map_sim_to_reference_style
    scene = SCENE.with_name("scene_source_contacts_mass.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    visual_model, _ = load_reference_model(scene)
    with np.load(ROOT / "runs/taco_brush_bimanual_gt_v4/robot_reference.npz", allow_pickle=False) as data:
        qpos = data["qpos"][::25]
    mapped, report = map_sim_to_reference_style(model, visual_model, qpos,
        sim_left_object_joint="left_object_joint", ref_left_object_joint="left_object_joint")
    assert len(report["hands"]) == len(report["objects"]) == 2
    assert all(item["sampled_position_error_m_max"] < 1e-12 for item in report["sampled_hand_errors"])
    np.testing.assert_allclose(mapped[:, -14:], qpos[:, -14:], atol=1e-12)
