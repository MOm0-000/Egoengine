import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from video_to_spider.export.spider_runner import (
    _configure_contact_reward,
    _contact_position_channels,
    _inspect_hand_floor_contacts,
    _normalize_kinematic_contact,
    _replay_metrics,
    _resolve_uv,
    _run,
    clip_auxiliary_scale_for_spider,
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


def test_clip_auxiliary_scale_keeps_paper_request_visible():
    assert clip_auxiliary_scale_for_spider(2.0, 0.2) == pytest.approx(0.19)
    assert clip_auxiliary_scale_for_spider(0.01, 0.2) == pytest.approx(0.01)


def test_configure_contact_reward_resolves_fingertip_sites(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        "<mujoco><default><site size='0.01'/></default><worldbody><body>"
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
    contact = np.zeros((6, 10), dtype=np.float32)
    contact[2, 5] = 1.0
    contact[3, 5:7] = 1.0
    contact[4, 5:8] = 1.0
    contact_pos = np.repeat(contact[..., None], 3, axis=2)
    np.savez(
        trajectory, qpos=np.zeros((4, 25)), qvel=np.zeros((4, 24)),
        frequency=50.0, contact=contact, contact_pos=contact_pos,
    )

    report = _normalize_kinematic_contact(trajectory, expected_contacts=5)

    assert report["contact_shape"] == [4, 5]
    assert report["per_channel_contact_rate"] == [0.75, 0.5, 0.25, 0.0, 0.0]
    assert report["all_channels_identical"] is False
    with np.load(trajectory) as artifact:
        np.testing.assert_allclose(artifact["contact"], contact[1:5, -5:])
        np.testing.assert_allclose(artifact["contact_pos"], contact_pos[1:5, -5:])


def test_contact_position_channels_select_object_mocaps_in_finger_order(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    bodies = "".join(
        f'<body name="ref_{kind}_right_{finger}_tip" mocap="true"/>'
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for kind in ("object", "hand")
    )
    scene.write_text(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")

    channels, names = _contact_position_channels(scene, "right")

    assert channels == [0, 2, 4, 6, 8]
    assert names == [
        "ref_object_right_thumb_tip", "ref_object_right_index_tip",
        "ref_object_right_middle_tip", "ref_object_right_ring_tip",
        "ref_object_right_pinky_tip",
    ]


def _pose_trajectory(positions: np.ndarray) -> np.ndarray:
    qpos = np.zeros((len(positions), 12), dtype=np.float64)
    qpos[:, -7:-4] = positions
    qpos[:, -4] = 1.0
    return qpos


def test_replay_metrics_selects_replay_for_matching_physics(tmp_path: Path):
    positions = np.stack([
        np.linspace(0.0, 0.08, 60), np.zeros(60), np.zeros(60),
    ], axis=1)
    trajectory = tmp_path / "trajectory_kinematic.npz"
    rollout = tmp_path / "trajectory_ikrollout.npz"
    np.savez(trajectory, qpos=_pose_trajectory(positions))
    np.savez(rollout, qpos=_pose_trajectory(positions + 0.001))

    metrics = _replay_metrics(trajectory, rollout, "right")

    assert metrics["feasible"] is True
    assert metrics["decision"] == "Replay"
    assert len(metrics["windows"]) == 3


def test_replay_metrics_escalates_when_object_motion_is_not_transferred(tmp_path: Path):
    positions = np.stack([
        np.linspace(0.0, 0.10, 60), np.zeros(60), np.zeros(60),
    ], axis=1)
    trajectory = tmp_path / "trajectory_kinematic.npz"
    rollout = tmp_path / "trajectory_ikrollout.npz"
    np.savez(trajectory, qpos=_pose_trajectory(positions))
    np.savez(rollout, qpos=_pose_trajectory(np.zeros_like(positions)))

    metrics = _replay_metrics(trajectory, rollout, "right")

    assert metrics["feasible"] is False
    assert metrics["decision"] == "MPC"
    assert any(window["motion_transfer_ratio"] == 0.0 for window in metrics["windows"])


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


def test_run_can_audit_an_explicit_nonzero_fallback_code(tmp_path: Path):
    log = tmp_path / "command.log"

    record = _run(
        [sys.executable, "-c", "raise SystemExit(2)"],
        cwd=tmp_path,
        env={},
        log_path=log,
        allowed_returncodes=(0, 2),
    )

    assert record["returncode"] == 2
    assert "raise SystemExit(2)" in log.read_text()


def test_run_still_rejects_unexpected_failures(tmp_path: Path):
    with pytest.raises(RuntimeError, match=r"command failed \(1\)"):
        _run(
            [sys.executable, "-c", "raise SystemExit(1)"],
            cwd=tmp_path,
            env={},
            log_path=tmp_path / "command.log",
            allowed_returncodes=(0, 2),
        )
