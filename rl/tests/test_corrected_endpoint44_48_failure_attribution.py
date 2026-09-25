import gzip
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/corrected_endpoint44_48_failure_attribution_v1/report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_failure_attribution_is_read_only_and_trace_bound():
    report = json.loads(REPORT.read_text())
    assert report["schema"] == "corrected_endpoint44_48_failure_attribution_v1"
    assert report["status"] == "passed_read_only_attribution"
    assert report["scope"]["training_performed"] is False
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["checkpoint_resume_authorized"] is False
    assert report["scope"]["endpoints"] == [44, 45, 46, 47, 48]

    gate = report["regression_gate"]
    assert gate["replay_trace_exactly_reproduced"]
    assert gate["ppo_trace_exactly_reproduced"]
    assert gate["replay_validated_intervals"] == 30
    assert gate["ppo_validated_intervals"] == 27
    assert gate["ppo_first_failure"]["endpoint"] == 48


def test_corrected_failure_is_position_dominated_without_material_ctrlrange_loss():
    report = json.loads(REPORT.read_text())
    evidence = report["evidence_summary"]
    assert evidence["endpoint_48_position_alone_exceeds_ellipse_boundary"]
    assert evidence["endpoint_48_position_squared_contribution"] > 1.0
    assert evidence["endpoint_48_rotation_squared_contribution"] < 0.1
    assert evidence["endpoint_48_ppo_minus_replay_position_error_m"] > 0.04
    assert evidence["endpoint_48_ppo_minus_replay_rotation_error_rad"] < -0.49

    checks = report["measured_checks"]
    assert checks["ppo_ctrlrange_loss_above_1e_8_is_zero_at_44_48"]
    assert checks["ppo_maximum_roundoff_level_ctrlrange_loss_at_44_48"] < 2e-9
    assert checks["contact_bonus_is_zero_for_both_modes_at_44_48"]


def test_training_coverage_claim_stays_at_metric_level():
    report = json.loads(REPORT.read_text())
    training = report["training_v5_endpoint_47_48"]
    assert training["sample_count"] == 1280
    endpoint_47 = training["endpoints"]["47"]
    endpoint_48 = training["endpoints"]["48"]
    assert endpoint_47["visit_count"] == 25
    assert endpoint_48["visit_count"] == 23
    assert endpoint_47["epochs_with_visit"] == list(range(1, 9))
    assert endpoint_48["epochs_with_visit"] == list(range(1, 9))
    assert endpoint_48["final_cpu_metric_within_training_min_max"] == {
        "objective_score": False,
        "position_error_m": False,
        "rotation_error_rad": True,
    }
    assert endpoint_48["exact_final_cpu_state_coverage_claim"] is False
    assert report["measured_checks"]["training_v5_does_not_store_exact_final_cpu_state"]


def test_attribution_inputs_and_protocol_status_are_unambiguous():
    report = json.loads(REPORT.read_text())
    for name in ("formal_report", "protocol", "objective_profile", "observation_profile"):
        row = report["inputs"][name]
        assert _sha256(Path(row["path"])) == row["sha256"]
    for name in ("checkpoint", "boundary"):
        row = report["inputs"][name]
        path = Path(row["path"])
        assert _sha256(path) == row["artifact_sha256"]
        assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == row[
            "uncompressed_sha256"
        ]

    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    corrected = protocol["evaluation"]["corrected_replay"]
    assert corrected["second_window_replay_failed"] is True
    assert corrected["ppo_fallback_subsequently_executed"] is True
    assert "ppo_started" not in corrected
    assert Path(
        protocol["audit_results"]["corrected_endpoint44_48_failure_attribution"]
    ) == REPORT
