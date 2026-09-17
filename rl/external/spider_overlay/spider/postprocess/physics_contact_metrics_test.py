from pathlib import Path

import mujoco
import numpy as np

from spider.postprocess.physics_contact_metrics import analyze_physics_contacts


def test_analyze_physics_contacts_requires_thumb_opposition(tmp_path: Path):
    model = tmp_path / "scene.xml"
    model.write_text(
        "<mujoco><worldbody>"
        '<body name="right_object"><freejoint/>'
        '<geom name="right_object_geom" type="sphere" size="0.1" mass="0.1"/>'
        "</body>"
        '<body name="right_thumb"><geom name="right_thumb_geom" '
        'type="sphere" size="0.05" pos="0.14 0 0"/></body>'
        '<body name="right_index"><geom name="right_index_geom" '
        'type="sphere" size="0.05" pos="-0.14 0 0"/></body>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    qpos = np.zeros((4, 7), dtype=np.float64)
    qpos[:, 3] = 1.0
    trajectory = tmp_path / "trajectory.npz"
    np.savez(trajectory, qpos=qpos)

    metrics = analyze_physics_contacts(model, trajectory, "right")

    assert metrics["opposed_contact_frame_count"] == 4
    assert metrics["grasp_opposition_success"] is True

    force_metrics = analyze_physics_contacts(
        model,
        trajectory,
        "right",
        max_penetration_m=0.02,
    )
    assert force_metrics["force_closure_frame_count"] == 4
    assert force_metrics["force_closure_success"] is True
    assert (
        force_metrics["per_side"]["right"]["per_finger_max_normal_force_n"]["thumb"]
        >= 0.2
    )
    assert force_metrics["per_side"]["right"]["most_opposed_contact_normal_dot"] == -1.0


def test_supported_unilateral_push_passes_executability_without_opposition(tmp_path: Path):
    model = tmp_path / "supported_scene.xml"
    model.write_text(
        "<mujoco><option gravity='0 0 -9.81'/><worldbody>"
        '<geom name="floor" type="plane" size="1 1 .1"/>'
        '<body name="right_object" pos="0 0 .095"><freejoint name="right_object_joint"/>'
        '<geom name="right_object_geom" type="sphere" size="0.1" mass="0.1"/>'
        "</body>"
        '<body name="right_index"><geom name="right_index_geom" '
        'type="sphere" size="0.05" pos=".14 0 .095"/></body>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    parsed = mujoco.MjModel.from_xml_path(str(model))
    qpos = np.repeat(parsed.qpos0[None], 4, axis=0)
    trajectory = tmp_path / "supported.npz"
    reference = tmp_path / "supported_reference.npz"
    np.savez(trajectory, qpos=qpos, frequency=np.asarray(50.0))
    np.savez(reference, qpos=qpos, frequency=np.asarray(50.0))

    metrics = analyze_physics_contacts(
        model, trajectory, "right",
        reference_trajectory_path=reference,
        max_penetration_m=0.02,
    )

    assert metrics["grasp_opposition_success"] is False
    assert metrics["force_closure_success"] is False
    assert metrics["support_conditioned_force_explanation"][
        "environment_supported_frame_count"
    ] == 4
    assert metrics["physical_acceptance"]["accepted"] is True


def test_unsupported_motion_without_hand_force_fails_force_explanation(tmp_path: Path):
    model = tmp_path / "unsupported_scene.xml"
    model.write_text(
        "<mujoco><option gravity='0 0 -9.81'/><worldbody>"
        '<body name="right_object" pos="0 0 1"><freejoint name="right_object_joint"/>'
        '<geom name="right_object_geom" type="sphere" size="0.05" mass="0.1"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    parsed = mujoco.MjModel.from_xml_path(str(model))
    qpos = np.repeat(parsed.qpos0[None], 4, axis=0)
    trajectory = tmp_path / "unsupported.npz"
    reference = tmp_path / "unsupported_reference.npz"
    np.savez(trajectory, qpos=qpos, frequency=np.asarray(50.0))
    np.savez(reference, qpos=qpos, frequency=np.asarray(50.0))

    metrics = analyze_physics_contacts(
        model, trajectory, "right", reference_trajectory_path=reference,
    )

    assert metrics["object_tracking"]["passed"] is True
    assert metrics["support_conditioned_force_explanation"]["passed"] is False
    assert metrics["physical_acceptance"]["accepted"] is False
    assert metrics["physical_acceptance"]["failed_checks"] == [
        "free_space_motion_has_force_explanation"
    ]
