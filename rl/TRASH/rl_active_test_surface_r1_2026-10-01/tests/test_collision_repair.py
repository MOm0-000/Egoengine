"""A collision repair must preserve dynamics, geometry provenance and pair parity."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "external/mink/src")]
from audit_taco_collision_repair import union_contains, body_hulls, contact_discrepancy_masks
from audit_taco_collision_coverage import coverage_inventory
from audit_taco_initialization import visual_meshes
from build_taco_collision_repair import SCENE, OUTPUT, build, check_unchanged_dynamics
from egoengine_repro.retarget.collision_audit import explicit_hand_pairs, hand_ids
from egoengine_repro.retarget.mink import _enable_planning_collision_masks, _explicit_collision_groups


def test_convex_union_not_convex_hull_of_all_parts():
    a, b = trimesh.creation.box(), trimesh.creation.box()
    b.apply_translation([2, 0, 0])
    assert union_contains([a, b], np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 2, 0]])).tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="finite"):
        union_contains([a], np.array([[np.nan, 0, 0]]))


def test_exact_stl_welding_changes_no_triangle_or_fills_no_hole():
    pytest.importorskip("coacd")
    from prepare_taco_bowl_collision import weld_exact_vertices
    box = trimesh.creation.box()
    soup = trimesh.Trimesh(vertices=box.triangles.reshape(-1, 3),
                          faces=np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
    welded = weld_exact_vertices(soup)
    assert welded.is_volume and not soup.is_watertight
    np.testing.assert_array_equal(welded.triangles, soup.triangles)
    opened = trimesh.Trimesh(vertices=soup.vertices, faces=soup.faces[:-1], process=False)
    assert not weld_exact_vertices(opened).is_watertight


def test_partial_repair_preserves_dynamics_and_fills_two_body_inventory_gaps():
    before = mujoco.MjModel.from_xml_path(str(SCENE))
    after = mujoco.MjModel.from_xml_path(str(OUTPUT / "scene.xml"))
    assert check_unchanged_dynamics(before, after)["state_dimensions"] == [50, 48, 36]
    meshes, _ = visual_meshes(OUTPUT / "scene.xml", after)
    inventory = coverage_inventory(after, meshes)
    assert inventory["native_meshes_without_same_body_shell"] == []
    assert not inventory["coverage_complete"]
    assert len(hand_ids(after)) == 26
    after.body_mass[1] += .001
    with pytest.raises(ValueError, match="body_mass"):
        check_unchanged_dynamics(before, after)


def test_repaired_runtime_pairs_are_identical_in_mink():
    import mink
    model = mujoco.MjModel.from_xml_path(str(OUTPUT / "scene.xml"))
    pairs = explicit_hand_pairs(model)
    hands = set(hand_ids(model))
    _enable_planning_collision_masks(model, list(hands), [])
    limit = mink.CollisionAvoidanceLimit(model, _explicit_collision_groups(model, mujoco, hand_geom_ids=hands))
    assert set(limit.geom_id_pairs) == set(pairs)
    for side in ("right", "left"):
        required = tuple(sorted((model.geom(f"collision_hand_{side}_palm_hull").id,
                                 model.geom(f"collision_hand_{side}_thumb_hull").id)))
        assert required in pairs


def test_measured_candidate_is_never_overwritten():
    with pytest.raises(FileExistsError):
        build(SCENE, OUTPUT)


def test_body_hulls_exclude_unnamed_visual_geometries():
    model = mujoco.MjModel.from_xml_path(str(OUTPUT / "scene_budgeted_palms.xml"))
    assert len(body_hulls(model, "left_hand_index_rota_link2")) == 1
    assert len(body_hulls(model, "left_object")) == 86


def test_clipped_finger_parts_stay_in_source_hull_and_remain_closed():
    import json
    from egoengine_repro.retarget.paper_audit import verify_artifacts
    for label in ("left_index_distal", "left_middle_distal"):
        record = json.loads((OUTPUT / "assets" / f"{label}_clipped" / "provenance.json").read_text())
        verify_artifacts([record["postprocess_input"]])
        native = trimesh.load_mesh(record["source"], process=True)
        native.apply_scale(record["unit_scale"])
        hull = native.convex_hull
        assert record["parts"] == 16
        for entry in record["artifacts"]:
            mesh = trimesh.load_mesh(entry["path"], process=True)
            assert mesh.is_volume
            assert union_contains([hull], mesh.vertices, tolerance=1e-7).all()
        assert record["containment_halfspace_max_excess_m"] < 1e-7


def test_reporting_threshold_cannot_create_missed_contact_alarm():
    # First two distances reproduce the two resolved Pour discrepancies.
    masks = contact_discrepancy_masks(
        np.array([True, True, True, False]),
        np.array([-39.15e-6, -24.25e-6, 0, -.001]),
        np.array([True, True, False, True]))
    assert masks["native_crossing_not_over_50um"].tolist() == [True, True, True, False]
    assert masks["native_crossing_without_runtime_contact"].tolist() == [False, False, True, False]
    assert masks["shell_overlap_without_surface_crossing"].tolist() == [False, False, False, True]


def test_nonfinite_contact_distance_cannot_become_a_clear_report():
    with pytest.raises(ValueError, match="finite"):
        contact_discrepancy_masks(np.array([True]), np.array([np.nan]), np.array([True]))
