"""MANO rotation transport and the isolated Pour correction experiment."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "external/mink/src")]

from audit_taco_pour_input_logic import mano_global_rotations, rotation_steps_degrees
from egoengine_repro.evaluation.taco_surface import _load_model_data, load_taco_mano_sequence
from egoengine_repro.retarget.taco_bimanual import MANO_DISTALS, mano_fingertip_frames, prepare

HANDS = ROOT / "data/taco_v1/pour_bowl_plate/hand_poses/Hand_Poses/(pour in some, bowl, plate)/20230927_017"
MODELS = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"
OLD = ROOT / "runs/taco_pour_bimanual_gt_v1"
NEW = ROOT / "runs/taco_pour_bimanual_mano_fk_v1"


@pytest.fixture(scope="module", params=["right", "left"])
def evidence(request):
    side = request.param
    pose, shape = HANDS / f"{side}_hand.pkl", HANDS / f"{side}_hand_shape.pkl"
    model = MODELS / f"MANO_{side.upper()}.pkl"
    joints = np.load(HANDS / "hand_joints.npy")[:, 1 if side == "right" else 0]
    frames, report = mano_fingertip_frames(pose, shape, model, side, joints)
    poses, _, _, _ = load_taco_mano_sequence(pose, shape)
    parents = np.asarray(_load_model_data(model)["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    independent = mano_global_rotations(poses, parents)[:, MANO_DISTALS]
    return side, frames, report, independent


def test_library_fk_matches_independent_parent_chain(evidence):
    _, frames, report, global_rotations = evidence
    calibration = np.array(report["neutral_distal_frames"])
    np.testing.assert_allclose(frames, global_rotations @ calibration, atol=1e-12, rtol=0)
    assert report["reconstructed_joint_max_error_m"] < 1e-6
    assert not report["calibration_fitted_to_episode"]
    np.testing.assert_allclose(frames.swapaxes(-1, -2) @ frames,
                               np.broadcast_to(np.eye(3), frames.shape), atol=1e-12)


def test_true_rotation_steps_preserved_without_palm_projection(evidence):
    _, frames, _, global_rotations = evidence
    np.testing.assert_allclose(rotation_steps_degrees(frames),
                               rotation_steps_degrees(global_rotations), atol=1e-11, rtol=0)
    assert rotation_steps_degrees(frames).max() < 20


def test_mismatched_released_gt_is_rejected():
    joints = np.load(HANDS / "hand_joints.npy")[:, 1].copy()
    joints[0, 8, 2] += .001
    with pytest.raises(ValueError, match="do not reproduce"):
        mano_fingertip_frames(HANDS / "right_hand.pkl", HANDS / "right_hand_shape.pkl",
                             MODELS / "MANO_RIGHT.pkl", "right", joints)


def test_prepare_never_overwrites_preserved_reference():
    with pytest.raises(FileExistsError):
        prepare(Path("unused"), Path("unused"), "unused", "unused", OLD, mano_model_dir=MODELS)


def test_saved_pour_changes_only_fingertip_rotations():
    with np.load(OLD / "human_reference.npz", allow_pickle=False) as a, np.load(NEW / "human_reference.npz", allow_pickle=False) as b:
        for key in a.files:
            if key != "T_sim_fingertip_target":
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        np.testing.assert_array_equal(a["T_sim_fingertip_target"][..., :3, 3], b["T_sim_fingertip_target"][..., :3, 3])
        np.testing.assert_array_equal(a["T_sim_fingertip_target"][..., 3, :], b["T_sim_fingertip_target"][..., 3, :])
        assert str(b["fingertip_orientation_source"]) == "MANO_rotational_FK_fixed_neutral_calibration_v1"
        steps = rotation_steps_degrees(b["T_sim_fingertip_target"][..., :3, :3])
        assert steps[102, 1, 2] == pytest.approx(3.93644175, abs=1e-7)
        assert steps.max() < 20
    with np.load(OLD / "robot_reference.npz", allow_pickle=False) as a, np.load(NEW / "robot_reference.npz", allow_pickle=False) as b:
        np.testing.assert_array_equal(a["qpos"][:, 36:], b["qpos"][:, 36:])
        np.testing.assert_array_equal(a["frame_indices"], b["frame_indices"])


def test_true_distal_rotation_stays_smooth_through_old_projection_singularity():
    calibration = Rotation.from_euler("xyz", [.2, -.5, .4]).as_matrix()
    true = Rotation.from_euler("y", [[89], [91]], degrees=True).as_matrix() @ calibration
    np.testing.assert_allclose(rotation_steps_degrees(true), [2], atol=1e-12)


def test_native_table_comparison_matches_compiled_mesh_vertices():
    from compare_taco_pour_mano_fix import SCENE, inspect

    report = inspect()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    for label, directory in (("old", OLD), ("corrected", NEW)):
        with np.load(directory / "robot_reference.npz", allow_pickle=False) as source:
            qpos = source["qpos"]
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            expected = report["results"][label]["native_table"][side]
            for name, value in ((f"{side}_hand_link_visual", expected["initial_palm_clearance_m"]),
                                (expected["initial_worst_visual"], expected["initial_whole_hand_min_clearance_m"])):
                geom = model.geom(name).id
                mesh = model.geom_dataid[geom]
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                assert vertices[:, 2].min() - .72 == pytest.approx(value, abs=2e-8)


def test_pure_error_costs_use_meters_and_radians_before_squaring():
    import mink

    task = mink.FrameTask("unused", "site", position_cost=10., orientation_cost=1.)
    translation = np.array([.01, 0, 0, 0, 0, 0])
    rotation = np.array([0, 0, 0, .1, 0, 0])
    assert np.sum((task.cost * translation) ** 2) == pytest.approx(np.sum((task.cost * rotation) ** 2))
