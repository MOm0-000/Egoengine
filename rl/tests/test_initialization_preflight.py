"""Native surface checks must not depend on the runtime collision pair table."""

from pathlib import Path
import json
import sys

import mujoco
import numpy as np
import pytest
import trimesh
import yaml

fcl = pytest.importorskip("fcl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_taco_initialization_preflight import object_overfill_samples, surface_intersects, triangle_object, run


def test_native_crossing_separation_and_transform_updates():
    box = trimesh.creation.box()
    a, b = triangle_object(box), triangle_object(box)
    b.setTransform(fcl.Transform(np.array([.75, 0, 0])))
    assert surface_intersects(a, b)
    b.setTransform(fcl.Transform(np.array([2., 0, 0])))
    assert not surface_intersects(a, b)  # Do not reuse stale CollisionResult.
    b.setTransform(fcl.Transform(np.array([.75, 0, 0])))
    assert surface_intersects(a, b)


def test_open_triangle_crossing_requires_no_hole_filling():
    a = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    b = trimesh.Trimesh(vertices=[[0, 0, -1], [0, 0, 1], [0, .5, 1]], faces=[[0, 1, 2]])
    assert not a.is_watertight and not b.is_watertight
    assert surface_intersects(triangle_object(a), triangle_object(b))


def test_fully_contained_solid_is_not_reported_as_surface_crossing():
    outside, inside = trimesh.creation.box(), trimesh.creation.box(extents=[.1, .1, .1])
    assert not surface_intersects(triangle_object(outside), triangle_object(inside))
    # This is why FCL-only results are not a no-penetration certificate.


def test_empty_mesh_and_overwrite_rejected(tmp_path):
    with pytest.raises(ValueError, match="finite nonempty"):
        triangle_object(trimesh.Trimesh())
    with pytest.raises(FileExistsError):
        run(Path("unused"), Path("unused"), tmp_path)


def test_surface_intersection_is_invariant_under_shared_rigid_transform():
    box = trimesh.creation.box()
    a, b = triangle_object(box), triangle_object(box)
    rot = trimesh.transformations.rotation_matrix(.71, [1., 2., 3.])[:3, :3]
    translation = np.array([.3, -.2, .72])
    for offset, expected in ((.75, True), (2., False)):
        a.setTransform(fcl.Transform(rot, translation))
        b.setTransform(fcl.Transform(rot, translation + rot @ [offset, 0, 0]))
        assert surface_intersects(a, b) is expected


def test_compiled_collision_overfill_uses_native_body_frame_and_metric_units(monkeypatch):
    box = trimesh.creation.box()
    vertices = " ".join(map(str, box.vertices.ravel()))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <asset><mesh name="outer" vertex="{vertices}" scale="1.02 1.02 1.02"/></asset>
      <worldbody>
        <body pos=".1 .2 .3" euler="23 41 9">
          <geom name="right_object_visual" type="box" size=".5 .5 .5"/>
          <geom name="right_object_0" type="mesh" mesh="outer"/>
        </body>
        <body pos="-.2 .3 .1" euler="31 10 -8">
          <geom name="left_object_visual" type="box" size=".5 .5 .5"/>
          <geom name="left_object_0" type="mesh" mesh="outer"/>
        </body>
      </worldbody></mujoco>''')
    meshes = {model.geom(f"{side}_object_visual").id: box for side in ("right", "left")}
    def forbidden(*_):
        raise AssertionError("shape-only queries must not run constraint dynamics at default qpos")
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    result = object_overfill_samples(model, meshes)
    for row in result.values():
        assert row["sampled_overfill_distance_m_percentiles"][-1] == pytest.approx(np.sqrt(3) * .01, abs=1e-7)
        assert not row["maximum_error_certified"] and not row["underfill_checked"]


def test_current_protocol_and_ppo_use_corrected_reference_without_promoting_reset():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    ppo = yaml.safe_load((ROOT / "configs/taco_pour_bimanual_ppo.yaml").read_text())
    assert ppo["data_path"] == protocol["inputs"]["active_robot_reference"]
    assert "mano_fk" in ppo["data_path"]
    assert not protocol["initialization"]["candidate_comparison"]["completed"]
    assert not protocol["training_ready"]
    report = json.loads(Path(protocol["audit_results"]["active_initialization_preflight"]).read_text())
    assert report["coordinates"]["coordinate_export_consistent"]
    assert not report["coordinates"]["original_sensor_registration_certified"]
    assert not report["collision_model_validated"] and not report["replay_rl_ready"]
    assert report["native_surfaces"]["frames"] == 198
    assert len(report["native_surfaces"]["pairs"]) == 378  # All 28 native surfaces, including index roots.
    assert not report["original_scene_or_reference_modified"]
