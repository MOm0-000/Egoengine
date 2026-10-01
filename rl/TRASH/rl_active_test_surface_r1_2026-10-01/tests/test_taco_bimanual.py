"""Regression checks for the isolated TACO double-hand data/model contract."""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "diagnostics"),
               str(ROOT / "external/mink/src")]

from egoengine_repro.retarget.taco_bimanual import differentiate, geometric_frame, pose7
from build_taco_bimanual_reference import _sim_alignment
from render_exact_deximit_triptych import load_reference_model, map_sim_to_reference_style

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_act.xml"
RUN = ROOT / "runs/taco_brush_bimanual_gt_v1"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def test_scene_two_passive_free_objects_and_friction(model):
    assert (model.nq, model.nv, model.nu) == (50, 48, 36)
    for side in ("right", "left"):
        joint = model.joint(f"{side}_object_joint")
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert joint.id not in model.actuator_trnid[:, 0]
    assert model.ncam == 1
    assert model.cam_bodyid[0] == 0
    assert model.cam_mode[0] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    assert model.geom("floor").pos[2] == pytest.approx(0.72)
    for i in range(model.ngeom):
        if model.geom_contype[i]:
            assert model.geom_condim[i] >= 3
            assert model.geom_friction[i, 0] > 0


def test_target_centimeters_converted_once():
    root = ET.parse(SCENE).getroot()
    visual = root.find("asset/mesh[@name='left_visual']")
    np.testing.assert_allclose(np.fromstring(visual.get("scale"), sep=" "), [0.01] * 3)
    parts = [mesh for mesh in root.findall("asset/mesh") if "/convex_m/" in mesh.get("file", "")]
    assert len(parts) == 32
    assert all(mesh.get("scale", "1 1 1") == "1 1 1" for mesh in parts)


def test_quaternion_alignment_does_not_use_position_as_rotation():
    gt = np.eye(4)
    gt[:3, 3] = [0.2, 0.3, 0.5]
    sim = np.eye(4)
    sim[:3, :3] = Rotation.from_euler("xyz", [.4, .5, -.6]).as_matrix()
    sim[:3, 3] = [.6, -.1, .8]
    alignment = _sim_alignment(np.r_[np.zeros(18), pose7(sim)], gt[None])
    np.testing.assert_allclose(alignment @ gt, sim, atol=1e-12)


def test_free_joint_velocity_respects_rotation_manifold(model):
    dt = 1 / 30
    qpos = np.tile(model.qpos0, (5, 1))
    for i in range(5):
        transform = np.eye(4)
        transform[:3, 3] = [.6 + i * dt * .1, 0, .8]
        transform[:3, :3] = Rotation.from_rotvec([0, 0, np.pi - .01 + i * dt * .2]).as_matrix()
        qpos[i, -7:] = pose7(transform)
    qvel = differentiate(model, qpos, dt)
    np.testing.assert_allclose(qvel[:, -6:], np.tile([.1, 0, 0, 0, 0, .2], (5, 1)), atol=1e-10)


def test_real_gt_preserves_relative_objects_and_all_209_frames():
    with np.load(RUN / "human_reference.npz", allow_pickle=False) as data:
        assert data["hand_order"].tolist() == ["right", "left"]
        assert data["object_roles"].tolist() == ["tool", "target"]
        np.testing.assert_array_equal(data["frame_indices"], np.arange(209))
        transformed = data["T_sim_object_reference"]
    base = ROOT / "data/taco_v1/dev4/object_poses/Object_Poses/(brush, brush, bowl)/20230927_027"
    tool = np.load(base / "tool_071.npy").astype(np.float64)
    target = np.load(base / "target_146.npy").astype(np.float64)
    np.testing.assert_allclose(np.linalg.inv(transformed[:, 0]) @ transformed[:, 1],
                               np.linalg.inv(tool) @ target, atol=1e-8)


def test_exact_renderer_maps_both_hands_and_both_objects(model):
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as data:
        qpos = data["qpos"][::25]
    ref, _ = load_reference_model(SCENE)
    mapped, report = map_sim_to_reference_style(model, ref, qpos,
        sim_left_object_joint="left_object_joint", ref_left_object_joint="left_object_joint")
    assert len(report["hands"]) == len(report["objects"]) == 2
    assert all(item["sampled_position_error_m_max"] < 1e-12 for item in report["sampled_hand_errors"])
    np.testing.assert_allclose(mapped[:, -14:], qpos[:, -14:], atol=1e-12)


def test_degenerate_geometric_orientation_rejected():
    with pytest.raises(ValueError):
        geometric_frame(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError):
        geometric_frame(np.ones(3), np.ones(3))
