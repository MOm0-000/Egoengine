import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from video_to_spider.export.spider_runner import (
    _configure_contact_reward,
    _inspect_hand_floor_contacts,
    _normalize_kinematic_contact,
    _resolve_uv,
)


def test_inspect_hand_floor_contacts_counts_floor_pairs(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        "<mujoco><contact>"
        '<pair name="collision_hand_right_thumb_0_floor" solref="0.02 1"/>'
        '<pair name="collision_hand_right_thumb_0_right_object_0" solref="0.02 1"/>'
        "</contact></mujoco>"
    )

    report = _inspect_hand_floor_contacts([scene])

    pairs = {
        pair.get("name"): pair.get("solref")
        for pair in ET.parse(scene).getroot().findall("./contact/pair")
    }
    assert pairs["collision_hand_right_thumb_0_floor"] == "0.02 1"
    assert pairs["collision_hand_right_thumb_0_right_object_0"] == "0.02 1"
    assert report["pair_count_by_scene"] == {"scene.xml": 1}


def test_configure_contact_reward_resolves_fingertip_sites(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        "<mujoco><worldbody><body>"
        '<site name="unrelated"/>'
        '<site name="track_hand_right_thumb_tip"/>'
        '<site name="track_hand_right_index_tip"/>'
        '<site name="track_hand_right_middle_tip"/>'
        '<site name="track_hand_right_ring_tip"/>'
        '<site name="track_hand_right_pinky_tip"/>'
        "</body></worldbody></mujoco>"
    )
    task_info = tmp_path / "task_info.json"
    task_info.write_text('{"ref_dt": 0.02}')

    report = _configure_contact_reward(scene, task_info, "right")

    assert report["site_ids"] == [1, 2, 3, 4, 5]
    saved = __import__("json").loads(task_info.read_text())
    assert saved["contact_site_ids"] == [1, 2, 3, 4, 5]


def test_normalize_kinematic_contact_aligns_timeline_and_channels(tmp_path: Path):
    trajectory = tmp_path / "trajectory_kinematic.npz"
    contact = np.arange(6 * 10, dtype=np.float32).reshape(6, 10)
    contact_pos = np.repeat(contact[..., None], 3, axis=2)
    np.savez(
        trajectory, qpos=np.zeros((4, 25)), qvel=np.zeros((4, 24)),
        frequency=50.0, contact=contact, contact_pos=contact_pos,
    )

    report = _normalize_kinematic_contact(trajectory, expected_contacts=5)

    assert report["contact_shape"] == [4, 5]
    with np.load(trajectory) as artifact:
        np.testing.assert_allclose(artifact["contact"], contact[1:5, -5:])
        np.testing.assert_allclose(artifact["contact_pos"], contact_pos[1:5, -5:])


def test_resolve_uv_finds_project_local_tool(monkeypatch, tmp_path: Path):
    spider = tmp_path / "spider"
    uv = tmp_path / ".tools/uv/bin/uv"
    spider.mkdir()
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    monkeypatch.delenv("UV_EXECUTABLE", raising=False)
    monkeypatch.setattr("video_to_spider.export.spider_runner.shutil.which", lambda _: None)

    assert _resolve_uv(spider) == uv.resolve()


def test_resolve_uv_prefers_explicit_executable(monkeypatch, tmp_path: Path):
    spider = tmp_path / "spider"
    spider.mkdir()
    uv = tmp_path / "custom-uv"
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    monkeypatch.setenv("UV_EXECUTABLE", str(uv))

    assert _resolve_uv(spider) == uv.resolve()
