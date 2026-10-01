"""Residual-contact audits must not confuse surface crossing and solid overlap."""

from pathlib import Path
import json
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_taco_remaining_contacts import closed_component, native_pair_evidence, run, source_joint_check
from audit_taco_index_root_collision import classify_pose


def test_native_box_overlap_uses_mm_cubed_without_changing_inputs():
    a = trimesh.creation.box(extents=[.01] * 3)
    b = a.copy(); b.apply_translation([.005, 0, 0])
    original = b.vertices.copy()
    result = native_pair_evidence(a, b, np.zeros(3), np.array([0, 0, 1]))
    assert result["surface_crossing"]
    assert result["intersection_volume_mm3"] == pytest.approx(500, abs=.001)
    assert result["solid_sources"] == ["full_native", "full_native"]
    np.testing.assert_array_equal(b.vertices, original)


def test_contained_solid_can_overlap_without_triangle_crossing():
    a = trimesh.creation.box(extents=[.02] * 3)
    b = trimesh.creation.box(extents=[.01] * 3)
    result = native_pair_evidence(a, b)
    assert not result["surface_crossing"]
    assert result["intersection_volume_mm3"] == pytest.approx(1000, abs=.001)
    assert result["surface_containment_first_in_second_then_reverse"][1]["sampled_max_inside_m"] > .0049


def test_open_surface_not_silently_closed():
    box = trimesh.creation.box(extents=[.01] * 3)
    opened = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    mesh, tag = closed_component(opened)
    assert mesh is None and tag == "no_unique_closed_component"
    result = native_pair_evidence(opened, box)
    assert "intersection_volume_mm3" not in result
    assert result["solid_sources"][0] == "no_unique_closed_component"


def test_real_distal_components_unchanged_when_implicit_repair_disabled():
    root = Path(__file__).resolve().parents[1] / "models/taco_xhand/assets/robots/xhand_pour_20230927_017/assets"
    for name in ("left_hand_index_rota_link2.STL", "left_hand_mid_link2.STL"):
        mesh = trimesh.load_mesh(root / name, process=True)
        old = [m for m in mesh.split(only_watertight=False) if m.is_volume]
        new, tag = closed_component(mesh)
        assert tag == "auxiliary_closed_component" and len(old) == 1
        assert len(new.faces) == 7908
        np.testing.assert_array_equal(old[0].triangles, new.triangles)


def test_source_joint_check_reports_real_axis_mismatch():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody><body name="parent">
      <body name="child" pos="0 0 .01"><joint name="hinge" axis="0 0 1" range="0 1"/>
        <geom type="sphere" size=".01"/></body></body></worldbody></mujoco>''')
    root = ET.fromstring('''<robot><joint name="hinge"><parent link="parent"/><child link="child"/>
      <origin xyz="0 0 .01" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="1"/>
      </joint></robot>''')
    check = source_joint_check(model, 0, root)
    assert not check["matches"] and check["axis_error"] > 1
    root.find("joint/axis").set("xyz", "0 0 1")
    assert source_joint_check(model, 0, root)["matches"]


def test_existing_diagnostics_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(Path("not-read.xml"), tmp_path)


def test_actual_index_false_positives_are_pose_specific_not_pair_exemptions():
    root = Path(__file__).resolve().parents[1]
    for name in ("index_root_baseline.json", "index_cavities_screen.json"):
        report = json.loads((root / "runs/taco_pour_collision_repair" / name).read_text())
        for side in ("right", "left"):
            checked = [r for r in report["results"][side]["poses"] if "native" in r]
            assert len(checked) == 4  # neutral, actual initial, and both limits
            for record in checked:
                expected = ("native_overlap_confirmed" if abs(record["angle_rad"]) == .175
                            else "shell_false_positive_at_audited_pose")
                assert classify_pose(record) == expected
            for record in report["results"][side]["poses"]:
                if "native" not in record:
                    assert classify_pose(record) == "native_overlap_unresolved"


def test_surface_nonintersection_does_not_clear_contained_solid_or_open_mesh():
    outer = trimesh.creation.box(extents=[.02] * 3)
    inner = trimesh.creation.box(extents=[.01] * 3)
    contained = dict(shell_minimum_m=-.001, native=native_pair_evidence(outer, inner))
    assert classify_pose(contained) == "native_overlap_confirmed"
    inner.apply_translation([.1, 0, 0])
    separate = dict(shell_minimum_m=-.001, native=native_pair_evidence(outer, inner))
    assert classify_pose(separate) == "shell_false_positive_at_audited_pose"
    separate["native"]["solid_sources"][0] = "auxiliary_closed_component"
    assert classify_pose(separate) == "native_overlap_unresolved"


@pytest.mark.parametrize("volume", [np.nan, np.inf, -1])
def test_invalid_volume_cannot_clear_native_overlap(volume):
    with pytest.raises(ValueError, match="volume"):
        classify_pose(dict(shell_minimum_m=-.001, native=dict(intersection_volume_mm3=volume)))
