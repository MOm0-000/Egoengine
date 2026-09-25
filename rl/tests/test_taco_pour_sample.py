"""Pour acquisition must preserve bytes, selection provenance and paper omissions."""

import json
from pathlib import Path
import stat
import sys
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from egoengine_repro.retarget.paper_audit import artifact
from prepare_taco_pour_sample import copy_member, select_candidate


def test_selection_uses_task_metadata_order_not_shortest_or_anticipated_success():
    base = dict(action="pour in some", tool="bowl", object="plate",
                all_modalities_complete="True", calib_status="good")
    rows = [dict(base, sequence_id="c", n_frames="30"),
            dict(base, sequence_id="a", n_frames="200"),
            dict(base, sequence_id="0", calib_status="bad"),
            dict(base, sequence_id="1", tool="kettle")]
    selected, candidates = select_candidate(rows)
    assert selected["sequence_id"] == "a"
    assert len(candidates) == 3


def test_member_copy_is_independent_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("data/sample.bin", b"original sample bytes")
    destination = tmp_path / "bundle/sample.bin"
    record = copy_member(archive, "data/sample.bin", destination)
    assert destination.read_bytes() == b"original sample bytes"
    assert not destination.is_symlink()
    assert record["output"] == artifact(destination)
    assert record["crc_verified_by_zip_reader"]
    with pytest.raises(FileExistsError):
        copy_member(archive, "data/sample.bin", destination)


def test_archive_symlink_is_not_extracted(tmp_path):
    archive = tmp_path / "source.zip"
    entry = ZipInfo("data/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as z:
        z.writestr(entry, "/unrelated/user/data")
    destination = tmp_path / "bundle/link"
    with pytest.raises(ValueError, match="regular file"):
        copy_member(archive, "data/link", destination)
    assert not destination.exists()


def test_real_pour_bundle_preserves_all_198_frames_and_source_bytes():
    bundle = ROOT / "data/taco_v1/pour_bowl_plate"
    manifest = json.loads((bundle / "acquisition_manifest.json").read_text())
    report = json.loads((bundle / "input_audit.json").read_text())
    assert manifest["files"] == 13
    retired = {record["path"] for record in manifest["retired_outputs"]}
    for record in manifest["records"]:
        output = Path(record["output"]["path"])
        assert output.is_relative_to(bundle)
        if str(output) in retired:
            assert not output.exists()
            continue
        assert not output.is_symlink()
        assert artifact(output) == record["output"]
    assert manifest["currently_retained_initial_files"] == 12
    assert set(report["counts"].values()) == {198}
    assert report["status"]["gt_structurally_usable"]
    assert report["status"]["released_camera_front_check_passed"]
    assert report["independent_hand_archives_agree"]
    assert not report["frames_trimmed"]
    assert not report["physics_validated"]
    assert report["objects"]["tool"]["object_id"] == "022"
    assert report["objects"]["target"]["object_id"] == "135"
    assert not report["missing_modalities"]
    original = bundle / "depth_original/taco_pour_bowl_plate_20230927_017.avi"
    original_manifest = json.loads(
        (bundle / "depth_original/original_depth_manifest.json").read_text()
    )
    assert artifact(original)["sha256"] == original_manifest["sha256"]
    assert report["videos"]["depth_original"]["pix_fmt"] == "gray16le"
    assert report["videos"]["depth_original"]["depth_scale"] == 4000.0
    joints = np.load(bundle / "hand_poses/Hand_Poses/(pour in some, bowl, plate)/20230927_017/hand_joints.npy")
    assert joints.shape == (198, 2, 21, 3)


def test_active_protocol_does_not_infer_pour_weights_or_reuse_brush_reset():
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    assert protocol["inputs"]["active_episode"] == "20230927_017"
    assert "20230927_017" in protocol["inputs"]["approved_episodes"]
    example = protocol["tracking"]["published_task_example"]
    assert example["position_threshold_m"] == .12
    assert example["rotation_threshold_rad"] == 1.5
    assert all(protocol["tracking"][key] is None for key in ("lambda_p", "lambda_R", "C"))
    initialization = json.loads(Path(protocol["audit_results"]["active_initialization"]).read_text())
    assert protocol["initialization"]["current_episode_robot_initial_state_audited"]
    assert protocol["scene_alignment"]["original_depth_status"] == "decoded_uint16_scale_4000_all_198_frames"
    assert protocol["scene_alignment"]["depth_based_realignment"].startswith("not_applied")
    assert protocol["initialization"]["known_initial_hand_table_shell_penetration_m"] == pytest.approx(
        -initialization["initial_distances"]["hand_floor"]["min_distance_m"])
    assert "known_initial_hand_table_penetration_m" not in protocol["initialization"]
    assert protocol["initialization"]["initial_palm_index_native_penetration"] is False
    # Historical source-reference penetration remains recorded even though a
    # separately hash-bound v2 reset has now passed its release gate.
    assert min(protocol["initialization"]["initial_native_all_hand_table_penetration_m"].values()) > .009
    assert protocol["initialization"]["initial_native_left_palm_thumb_intersection_mm3"] > 9
    assert protocol["audit_results"]["new_reset_applied"]
    assert not protocol["training_ready"]
    assert "corrected_local_controllability_diagnosed_new_action_parameterization_decision_required" in protocol["blocking_checks"]
    assert protocol["solver_modes"] == ["replay", "rl"]
