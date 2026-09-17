"""Independent checks for the input-logic investigation, not new IK policy."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "external/mink/src")]

from audit_taco_pour_input_logic import (
    mano_global_rotations, palm_height_decomposition, rotation_steps_degrees,
)
from egoengine_repro.retarget.taco_bimanual import geometric_frame


def test_current_palm_projection_can_flip_a_smooth_distal_frame():
    smooth = Rotation.from_euler("y", [[89], [91]], degrees=True).as_matrix()
    constructed = np.stack([geometric_frame([1, 0, 0], r[:, 2]) for r in smooth])
    np.testing.assert_allclose(rotation_steps_degrees(smooth), [2], atol=1e-10)
    assert rotation_steps_degrees(constructed)[0] > 179


def test_angular_steps_do_not_depend_on_constant_mano_robot_offsets():
    rotations = Rotation.from_euler("xyz", [[.1, .2, -.3], [.2, .25, -.32], [.3, .1, -.1]]).as_matrix()
    calibration = Rotation.from_euler("xyz", [.7, -.4, 1.1]).as_matrix()
    np.testing.assert_allclose(rotation_steps_degrees(rotations @ calibration),
                               rotation_steps_degrees(rotations), atol=1e-12)


def test_mano_fk_composes_noncommuting_parent_rotations():
    pose = np.zeros((1, 48))
    pose[0, :3] = [0, 0, np.pi / 2]
    pose[0, 3:6] = [np.pi / 2, 0, 0]
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    result = mano_global_rotations(pose, parents)
    expected = Rotation.from_rotvec(pose[0, :3]).as_matrix() @ Rotation.from_rotvec(pose[0, 3:6]).as_matrix()
    np.testing.assert_allclose(result[0, 3], expected, atol=1e-12)
    np.testing.assert_allclose(result[0, 4], result[0, 0], atol=1e-12)
    parents[1] = 1
    with pytest.raises(ValueError):
        mano_global_rotations(pose, parents)


def test_height_decomposition_separates_translation_and_rotation():
    vertices = np.array([[0, 0, 0], [0, 0, .04], [.02, 0, 0]])
    target = np.eye(4)
    target[2, 3] = .73
    actual = target.copy()
    actual[2, 3] -= .02
    actual[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    report = palm_height_decomposition(vertices, target, actual, .72)
    assert report["palm_at_input_wrist_clearance_m"] == pytest.approx(.01)
    assert report["wrist_vertical_displacement_m"] == pytest.approx(-.02)
    assert report["rotation_contribution_m"] == pytest.approx(-.01)
    assert report["saved_reference_palm_clearance_m"] == pytest.approx(-.02)


@pytest.mark.parametrize("bad", [np.full((2, 3, 3), np.nan), np.zeros((2, 3, 3))])
def test_invalid_rotations_are_not_silently_normalized(bad):
    with pytest.raises(ValueError):
        rotation_steps_degrees(bad)


def test_real_pour_left_middle_jump_is_in_saved_targets():
    with np.load(ROOT / "runs/taco_pour_bimanual_gt_v1/human_reference.npz", allow_pickle=False) as source:
        rotations = source["T_sim_fingertip_target"][102:104, 1, 2, :3, :3]
        points = source["joint_positions_sim"][102:104, 1]
    bone = points[:, 12] - points[:, 11]
    bone /= np.linalg.norm(bone, axis=1, keepdims=True)
    bone_step = np.rad2deg(np.arccos(np.clip(bone[0] @ bone[1], -1, 1)))
    assert rotation_steps_degrees(rotations)[0] == pytest.approx(160.865646, abs=1e-5)
    assert bone_step == pytest.approx(3.989703, abs=1e-5)
