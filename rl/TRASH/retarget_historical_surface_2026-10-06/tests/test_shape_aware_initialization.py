from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from audit_left_ring_pinky_tray_geometry import (
    classify_material_pair,
    compile_model,
    current_geometry,
    geometry_regression,
    input_paths,
    load_contract,
    write_json,
)
from build_taco_pour_initialization_candidates import visual_meshes
from egoengine_repro.retarget.shape_aware import (
    bend_angle,
    fit_left_ring_pinky_shape,
    human_two_link_descriptor,
)


@pytest.fixture(scope="module")
def frozen_inputs():
    contract = load_contract()
    paths = input_paths(contract)
    model = compile_model(paths)
    with np.load(paths["accepted_initial_state"], allow_pickle=False) as archive:
        qpos = archive["qpos"].copy()
    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = {name: archive[name].copy() for name in archive.files}
    return contract, paths, model, qpos, human


def _synthetic_hand(bent: bool) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    for start in (13, 17):
        points[start] = (-0.02 if start == 13 else -0.04, 0.0, 0.0)
        points[start + 1] = points[start] + (0.0, 0.0, -0.03)
        offset = (0.0, -0.02, -0.02) if bent else (0.0, 0.0, -0.03)
        points[start + 3] = points[start + 1] + offset
    return points


def test_human_descriptor_distinguishes_bent_from_straight():
    wrist = np.eye(4)
    straight = human_two_link_descriptor(_synthetic_hand(False), wrist)
    bent = human_two_link_descriptor(_synthetic_hand(True), wrist)
    for finger in ("ring", "pinky"):
        assert straight[finger]["bend_angle_rad"] == pytest.approx(0.0)
        assert bent[finger]["bend_angle_rad"] > 0.5
        assert bend_angle(
            bent[finger]["proximal_direction"], bent[finger]["distal_direction"]
        ) == pytest.approx(bent[finger]["bend_angle_rad"])


def test_geometry_checker_regressions_and_unknown_fail_closed():
    report = geometry_regression(5e-5)
    assert report["passed"]
    assert report["cases"]["unknown_open_tray"]["certified"] is False
    assert report["cases"]["unknown_open_tray"]["classification"] == "UNKNOWN_NOT_CERTIFIED"


def test_reports_serialize_path_provenance(tmp_path):
    target = tmp_path / "report.json"
    write_json(target, {"mesh": Path("relative/mesh.obj")})
    assert '"relative/mesh.obj"' in target.read_text()


def test_current_frozen_endpoint0_classification_stable(frozen_inputs):
    contract, paths, model, qpos, _ = frozen_inputs
    meshes, _ = visual_meshes(paths["scene"], model)
    report = current_geometry(
        model,
        qpos,
        meshes,
        float(contract["geometry"]["material_reporting_threshold_m"]),
    )
    assert report["classification"] == "NEAR_CONTACT_NO_MATERIAL_PENETRATION"
    assert report["certified_no_material_penetration"]
    assert report["runtime_collision_geometry"]["minimum_distance_m"] < 5e-5


def test_robot_shape_fit_is_deterministic_and_freezes_other_coordinates(frozen_inputs):
    contract, _, model, qpos, human = frozen_inputs
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    descriptor = human_two_link_descriptor(
        human["joint_positions_sim"][0, hand], human["T_sim_wrist_target"][0, hand]
    )
    order = ("thumb", "index", "middle", "ring", "pinky")
    targets = {
        finger: human["T_sim_fingertip_target"][0, hand, order.index(finger), :3, 3]
        for finger in ("ring", "pinky")
    }

    def unconstrained(_qpos):
        return np.ones(1)

    first, first_report = fit_left_ring_pinky_shape(
        model, qpos, descriptor, targets, unconstrained, max_iterations=50
    )
    second, second_report = fit_left_ring_pinky_shape(
        model, qpos, descriptor, targets, unconstrained, max_iterations=50
    )
    assert np.array_equal(first, second)
    assert first_report["candidate_joint_values_rad"] == second_report["candidate_joint_values_rad"]
    movable = set(first_report["qpos_addresses"])
    locked = [index for index in range(model.nq) if index not in movable]
    assert np.array_equal(first[locked], qpos[locked])
    assert first_report["candidate_shape_objective"] < first_report["accepted_shape_objective"]


def test_contract_task_manifest_cannot_silently_enable_unused_posture_weight():
    contract = load_contract()
    assert contract["shape_fit"]["component_weights"] == [1.0, 1.0, 1.0]
    assert contract["shape_fit"]["residual_components"] == [
        "proximal_unit_direction_in_palm_frame",
        "distal_unit_direction_in_palm_frame",
        "fingertip_position_normalized_by_robot_finger_length",
    ]
    assert "posture_weight" not in contract["shape_fit"]


def test_completed_run_contract_when_artifacts_exist():
    run_root = os.environ.get("EGOENGINE_SHAPE_AWARE_RUN")
    if run_root is None:
        pytest.skip("set EGOENGINE_SHAPE_AWARE_RUN to audit a completed bounded run")
    run = Path(run_root)
    if not (run / "decision.json").is_file():
        pytest.skip("bounded integration run has not been executed")
    import json

    static = json.loads((run / "candidate_static_gate.json").read_text())
    decision = json.loads((run / "decision.json").read_text())
    visual = json.loads((run / "visual_review.json").read_text())
    if not static["passed"]:
        accounting = json.loads((run / "cost_accounting.json").read_text())
        assert decision["classification"] == "STATIC_GATE_FAILED_NO_PHYSICS"
        assert accounting["physics_steps"] == 0
        assert not (run / "cold_replay_parity.json").exists()
        assert not (run / "comparison.csv").exists()
        assert visual["endpoint"] == 0
        assert len(list((run / "static_review").glob("endpoint_*.png"))) == 6
        return
    cold = json.loads((run / "cold_replay_parity.json").read_text())
    control = json.loads((run / "control_validation.json").read_text())
    assert static["checks"]["object_qpos_byte_identical"]
    assert static["checks"]["right_hand_byte_identical"]
    assert control["executed_ctrl_rows_1_20_byte_identical_A_vs_shape"]
    assert cold["bitwise_equal"]
    assert visual["endpoints"] == list(range(21))
    for view in ("oblique", "top"):
        assert len(list((run / "frames" / view).glob("endpoint_*.png"))) == 21

