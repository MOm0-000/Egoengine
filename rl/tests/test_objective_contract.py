from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.objective_contract import load_runtime_objective


PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
LOCAL = ROOT / "configs/taco_pour_local_unpublished_v1.yaml"
NORMALIZED = ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml"


def test_paper_objective_fails_closed_while_coefficients_are_unresolved():
    with pytest.raises(ValueError, match="paper-faithful objective is unresolved.*lambda_p.*lambda_R.*C.*contact.*lift"):
        load_runtime_objective(PROTOCOL, None, tracking_variant="tool_only", require_run_ready=True)


def test_local_profile_is_explicit_and_auditable_for_smoke_tests():
    objective = load_runtime_objective(
        PROTOCOL, LOCAL, tracking_variant="tool_and_target", require_run_ready=False
    )
    assert objective.objective_id == "taco_pour_local_unpublished_v1"
    assert not objective.paper_faithful
    assert objective.tracking.lambda_p == 1.0
    assert objective.tracking.lambda_r == 1.0
    assert objective.tracking.boundary == pytest.approx(1.5047923441623355)
    assert objective.contact_coefficient == 0.25
    assert objective.lift_coefficient == 0.1
    assert objective.lift_object_role == "tool"
    report = objective.as_report()
    assert report["protocol_sha256"]
    assert report["profile_sha256"]
    assert report["tracking"] == {"lambda_p": 1.0, "lambda_R": 1.0, "C": pytest.approx(1.5047923441623355)}


def test_explicit_local_objective_is_open_only_after_all_named_gates_pass():
    objective = load_runtime_objective(
        PROTOCOL, LOCAL, tracking_variant="tool_and_target", require_run_ready=True
    )
    assert objective.status == "resolved_local_unpublished"
    assert not objective.paper_faithful


def test_normalized_ellipse_profile_uses_paper_scales_as_local_axis_intercepts():
    objective = load_runtime_objective(
        PROTOCOL, NORMALIZED, tracking_variant="tool_only", require_run_ready=True
    )
    assert objective.objective_id == "taco_pour_local_normalized_ellipse_v1"
    assert not objective.paper_faithful
    assert objective.tracking_metric_name == "normalized_ellipse_score"
    assert objective.tracking.lambda_p == pytest.approx(1.0 / 0.12**2)
    assert objective.tracking.lambda_r == pytest.approx(1.0 / 1.5**2)
    assert objective.tracking.boundary == 1.0
    assert objective.independent_position_threshold_m == 0.12
    assert objective.independent_rotation_threshold_rad == 1.5
    assert "not author-recovered" in objective.provenance


def test_normalized_ellipse_axis_intercepts_and_stricter_interior_tradeoff():
    objective = load_runtime_objective(
        PROTOCOL, NORMALIZED, tracking_variant="tool_only", require_run_ready=False
    )
    tracking = objective.tracking
    assert (tracking.lambda_p * 0.12**2) ** 0.5 == pytest.approx(tracking.boundary)
    assert (tracking.lambda_r * 1.5**2) ** 0.5 == pytest.approx(tracking.boundary)
    combined = (tracking.lambda_p * 0.10**2 + tracking.lambda_r * 1.0**2) ** 0.5
    assert combined > tracking.boundary


def test_formal_local_objective_is_hash_bound(tmp_path):
    changed = tmp_path / "changed.yaml"
    changed.write_text(NORMALIZED.read_text().replace("C: 1.0", "C: 1.01"))
    with pytest.raises(ValueError, match="protocol-bound hash"):
        load_runtime_objective(
            PROTOCOL, changed, tracking_variant="tool_only", require_run_ready=True
        )


def test_aggregation_variant_is_part_of_the_profile_contract(tmp_path):
    profile = tmp_path / "changed.yaml"
    profile.write_text(LOCAL.read_text().replace("object_role: tool", "object_role: target"))
    target_lift = load_runtime_objective(
        PROTOCOL, profile, tracking_variant="tool_and_target", require_run_ready=False
    )
    assert target_lift.lift_object_role == "target"

    text = profile.read_text().replace(
        "tool_and_target: mean_reward_any_termination", "tool_and_target: single_object"
    )
    profile.write_text(text)
    with pytest.raises(ValueError, match="aggregation must be explicitly"):
        load_runtime_objective(
            PROTOCOL, profile, tracking_variant="tool_and_target", require_run_ready=False
        )
