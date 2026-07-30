from pathlib import Path
import json

import numpy as np
import pytest

from video_to_spider.export.spider import export_spider_dataset


def _write_artifacts(tmp_path: Path):
    t = 4
    timestamps = np.arange(t) / 30.0
    identity = np.repeat(np.eye(4)[None], t, axis=0)
    wrist = identity.copy()
    wrist[:, :3, 3] = [0.0, 0.0, 0.2]
    obj = identity.copy()
    obj[:, :3, 3] = [0.05, 0.0, 0.05]
    fingertips = np.zeros((t, 1, 5, 3))
    fingertips[..., 2] = 0.1
    aligned = tmp_path / "aligned.npz"
    np.savez(aligned, frame_indices=np.arange(t), timestamps_s=timestamps,
             T_sim_object=obj[:, None], T_sim_wrist=wrist[:, None], fingertips_sim=fingertips,
             mano_pose=np.repeat(np.eye(3)[None, None, None], t * 1 * 15, axis=0).reshape(t, 1, 15, 3, 3),
             mano_betas=np.zeros((1, 10)), object_scale_to_m=np.array([0.5]),
             valid_object=np.ones((t, 1), bool), valid_hand=np.ones((t, 1), bool),
             confidence_object=np.ones((t, 1)), confidence_hand=np.ones((t, 1)))
    contact = tmp_path / "contact.npz"
    np.savez(contact, frame_indices=np.arange(t), timestamps_s=timestamps,
             contact=np.zeros((t, 1, 5)), contact_pos_object_local=np.zeros((1, 5, 3)))
    mesh = tmp_path / "visual.obj"
    mesh.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n")
    return aligned, contact, mesh


def test_spider_export_shapes_and_inactive_identity(tmp_path: Path):
    aligned, contact, mesh = _write_artifacts(tmp_path)
    spider_package = tmp_path / "spider_package"
    for robot in ("mano", "xhand"):
        robot_dir = spider_package / "assets/robots" / robot
        robot_dir.mkdir(parents=True)
        (robot_dir / "right.xml").write_text("<mujoco/>")
    result = export_spider_dataset(
        aligned_path=aligned, contact_path=contact, visual_mesh_path=mesh,
        dataset_root=tmp_path / "dataset", task="synthetic", data_id=0, source_run_id="run",
        hand_sides=["right"], spider_package_root=spider_package, embodiment_type="right",
        hand_roles={"right": "passive"},
    )
    data = np.load(result["keypoints"])
    assert set(data.files) == {
        "qpos_obj_right", "qpos_obj_left", "qpos_wrist_right", "qpos_finger_right",
        "contact_right", "contact_pos_right", "qpos_wrist_left", "qpos_finger_left",
        "contact_left", "contact_pos_left",
    }
    assert data["qpos_finger_right"].shape[1:] == (5, 7)
    np.testing.assert_allclose(data["qpos_obj_left"][:, 3], 1)
    np.testing.assert_allclose(data["qpos_wrist_left"][:, 3], 1)
    exported_mesh = __import__("trimesh").load_mesh(result["object_dir"] / "visual.obj", process=False)
    assert np.isclose(exported_mesh.extents.max(), 0.05)
    task_info = json.loads(result["task_info"].read_text())
    assert task_info["hand_roles"] == {"right": "passive"}
    assert task_info["simulation_preflight"]["passed"]
    assert task_info["simulation_preflight"]["hand_target_clearance_m"] == 0.060


def test_spider_export_rejects_underground_object(tmp_path: Path):
    aligned, contact, mesh = _write_artifacts(tmp_path)
    with np.load(aligned) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    arrays["T_sim_object"][:, 0, 2, 3] = -0.01
    np.savez(aligned, **arrays)
    spider_package = tmp_path / "spider_package"
    for robot in ("mano", "xhand"):
        robot_dir = spider_package / "assets/robots" / robot
        robot_dir.mkdir(parents=True)
        (robot_dir / "right.xml").write_text("<mujoco/>")
    with pytest.raises(ValueError, match="object trajectory penetrates the floor"):
        export_spider_dataset(
            aligned_path=aligned, contact_path=contact, visual_mesh_path=mesh,
            dataset_root=tmp_path / "dataset", task="synthetic", data_id=0,
            source_run_id="run", hand_sides=["right"], spider_package_root=spider_package,
            embodiment_type="right", hand_roles={"right": "active"},
        )


def test_spider_export_requires_xhand_floor_clearance(tmp_path: Path):
    aligned, contact, mesh = _write_artifacts(tmp_path)
    with np.load(aligned) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    arrays["T_sim_wrist"][:, 0, 2, 3] = 0.015
    arrays["fingertips_sim"][:, 0, :, 2] = 0.015
    np.savez(aligned, **arrays)
    spider_package = tmp_path / "spider_package"
    for robot in ("mano", "xhand"):
        robot_dir = spider_package / "assets/robots" / robot
        robot_dir.mkdir(parents=True)
        (robot_dir / "right.xml").write_text("<mujoco/>")
    with pytest.raises(ValueError, match="lack xHand floor clearance"):
        export_spider_dataset(
            aligned_path=aligned, contact_path=contact, visual_mesh_path=mesh,
            dataset_root=tmp_path / "dataset", task="synthetic", data_id=0,
            source_run_id="run", hand_sides=["right"], spider_package_root=spider_package,
            embodiment_type="right", hand_roles={"right": "active"},
        )


def test_bimanual_export_requires_both_visible_hands(tmp_path: Path):
    aligned, contact, mesh = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="bimanual embodiment requires"):
        export_spider_dataset(
            aligned_path=aligned, contact_path=contact, visual_mesh_path=mesh,
            dataset_root=tmp_path / "dataset", task="synthetic", data_id=0,
            source_run_id="run", hand_sides=["right"], spider_package_root=tmp_path,
            embodiment_type="bimanual",
        )
