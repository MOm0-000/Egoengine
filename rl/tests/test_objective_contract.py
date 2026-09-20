from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.objective_contract import load_runtime_objective


PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
LOCAL = ROOT / "configs/taco_pour_local_unpublished_v1.yaml"


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


def test_explicit_local_objective_does_not_bypass_formal_run_gates():
    with pytest.raises(ValueError, match="formal run remains blocked.*initialization.*observation"):
        load_runtime_objective(
            PROTOCOL, LOCAL, tracking_variant="tool_and_target", require_run_ready=True
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
