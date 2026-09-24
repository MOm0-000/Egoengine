"""Evidence checks for the no-training mixed-unit action-scale audit."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runs/taco_pour_reference_action_scale_audit_v1/report.json"
REFERENCE_PATH = ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text())


def test_reference_action_scale_audit_is_read_only_and_hash_bound():
    report = _report()
    assert report["schema"] == "taco_pour_reference_action_scale_audit_v1"
    assert report["status"] == "mixed_unit_imbalance_quantified_no_scale_candidate_selected"
    assert report["scope"] == {
        "training_executed": False,
        "physics_rollout_executed": False,
        "simulator_environments_created": 0,
        "reference_or_configuration_modified": False,
        "quantity_measured": "absolute consecutive reference control-target increments",
        "realized_robot_motion_measured": False,
        "author_recovered_residual_scale": False,
    }
    for record in report["preserved_inputs"] + report["audit_code"]:
        path = Path(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_all_36_coordinates_match_the_frozen_reference_statistics():
    report = _report()
    with np.load(REFERENCE_PATH, allow_pickle=False) as arrays:
        ctrl = np.asarray(arrays["ctrl"], dtype=np.float64)
    increments = np.abs(np.diff(ctrl, axis=0))
    assert ctrl.shape == (198, 36)
    assert report["reference"] == {
        "frames": 198,
        "transitions": 197,
        "frequency_hz": 30.0,
        "control_dimensions": 36,
    }
    rows = report["hands"]["right"]["coordinates"] + report["hands"]["left"]["coordinates"]
    assert [row["index"] for row in rows] == list(range(36))
    assert len({row["actuator"] for row in rows}) == 36
    for row in rows:
        values = increments[:, row["index"]]
        expected = dict(zip(
            ("p50", "p90", "p95", "p99"),
            np.quantile(values, (0.50, 0.90, 0.95, 0.99)),
        ))
        expected["max"] = values.max()
        for name, value in expected.items():
            assert row["absolute_increment"][name] == float(value)
            assert row["residual_limit_over_increment"][name] == float(0.05 / value)


def test_audit_quantifies_imbalance_without_selecting_a_new_scale():
    report = _report()
    mapping = report["current_local_action_mapping"]
    assert mapping["translation_component_limit_m"] == 0.05
    assert mapping["rotation_and_finger_component_limit_rad"] == 0.05
    assert mapping["same_numeric_scale_across_incompatible_units"] is True
    interpretation = report["interpretation"]
    assert interpretation["mixed_unit_scale_is_balanced_by_reference_dynamics"] is False
    assert interpretation["split_scale_candidate_selected"] is False
    assert interpretation["next_training_authorized"] is False
    for side in ("right", "left"):
        ratios = interpretation[f"{side}_p95_ratios"]
        assert ratios["wrist_translation_components"] > 5.0
        assert ratios["wrist_rotation_components"] < 1.0
        assert ratios["finger_joint_components"] < 0.5
