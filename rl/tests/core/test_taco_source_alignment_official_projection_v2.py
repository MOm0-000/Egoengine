from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_contract_is_zero_runtime_and_pins_the_official_entrypoint():
    cfg = yaml.safe_load(
        (ROOT / "configs/taco_source_alignment_official_projection_v2.yaml").read_text()
    )
    assert cfg["schema"] == "taco_source_alignment_official_projection_v2"
    assert cfg["expected_baseline"] == "777bbb841adafb62b2f55579e352824cc56abf31"
    assert cfg["sequence"]["focus_frames"] == [0, 14, 15, 16, 17, 20]
    assert cfg["sequence"]["timing_offsets"] == [-1, 0, 1]
    assert cfg["official"]["entrypoint"] == "dataset_utils/project_pose_to_egocentric_view.py"
    assert not any(cfg["authorization"].values())


def test_taco_wrapper_delegates_projection_math_to_the_official_code():
    source = (
        ROOT / "src/egoengine_repro/evaluation/taco_official_projection.py"
    ).read_text()
    assert "from pyt3d_wrapper import Pyt3DWrapper" in source
    assert "transform_points_screen" in source
    assert "np.linalg.inv" not in source
    assert "@ intrinsic" not in source
    assert "extrinsic[:3" not in source
    assert "camera = points" not in source


def test_duplicate_taco_projection_implementations_are_not_active():
    closure = (ROOT / "src/egoengine_repro/retarget/contract_closure.py").read_text()
    paper = (ROOT / "src/egoengine_repro/retarget/paper_audit.py").read_text()
    historical_closure = (
        ROOT / "scripts/audit_taco_pour_retarget_contract_closure_v2.py"
    ).read_text()
    assert "def project_world_points" not in closure
    assert "def project_world(" not in paper
    assert "project_world_points(" not in historical_closure
    for name in (
        "audit_taco_pour_depth_registration.py",
        "audit_taco_pour_table_calibration.py",
        "audit_taco_pour_raw_depth_table.py",
    ):
        assert not (ROOT / "scripts" / name).exists()


def test_audit_runner_has_no_physics_control_retarget_or_rl_path():
    source = (
        ROOT / "scripts/audit_taco_source_alignment_official_projection_v2.py"
    ).read_text()
    assert "mj_step(" not in source
    assert "mujoco" not in source.lower()
    assert "PPO" not in source
    for name in (
        "source_pins.json",
        "input_manifest.json",
        "projection_code_cleanup.json",
        "official_egocentric_projection_manifest.json",
        "object_depth_control.json",
        "hand_depth_consistency.json",
        "allocentric_mask_secondary_check.json",
        "source_alignment_decision.json",
    ):
        assert name in source


def test_frozen_evidence_fails_closed_at_the_rigid_depth_control():
    evidence = ROOT / "runs/taco_source_alignment_official_projection_v2"
    decision = json.loads((evidence / "source_alignment_decision.json").read_text())
    objects = json.loads((evidence / "object_depth_control.json").read_text())
    hands = json.loads((evidence / "hand_depth_consistency.json").read_text())
    cleanup = json.loads((evidence / "projection_code_cleanup.json").read_text())
    allocentric = json.loads(
        (evidence / "allocentric_mask_secondary_check.json").read_text()
    )
    assert decision["classification"] == "CAMERA_DEPTH_REGISTRATION_UNRESOLVED"
    assert not any(decision["runtime_counts"].values())
    assert objects["classification"] == "CAMERA_DEPTH_REGISTRATION_UNRESOLVED"
    assert objects["per_object_aggregate"]["tool"]["minimum_valid_pixel_fraction"] > 0.99
    assert objects["per_object_aggregate"]["target"]["minimum_valid_pixel_fraction"] < 0.14
    assert hands["classification"] == "PROHIBITED_BY_CAMERA_DEPTH_GATE"
    assert hands["executed"] is False
    assert cleanup["wrapper_contains_projection_mathematics"] is False
    assert cleanup["official_commit"] == "b385ac1e89f35214bfbbef0d63e6df0ac8dac02e"
    assert allocentric["role"] == "secondary_conflict_detection_only"
    assert allocentric["not_independent_ground_truth"] is True

    hashes = {
        relative: expected
        for expected, relative in (
            line.split(maxsplit=1)
            for line in (evidence / "server_artifacts.sha256").read_text().splitlines()
        )
    }
    for relative in (
        "source_pins.json",
        "object_depth_control.json",
        "source_alignment_decision.json",
        "summary.md",
    ):
        assert relative in hashes
