"""The left palm/thumb topology audit is read-only and reproducible."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/taco_pour_left_palm_thumb_boundary_topology_v1.yaml"
REPORT = ROOT / "runs/taco_pour_left_palm_thumb_boundary_topology_v1/report.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_topology_protocol_binds_all_inputs_and_is_read_only():
    protocol = yaml.safe_load(PROTOCOL.read_text())
    assert protocol["protocol_name"] == "taco_pour_left_palm_thumb_boundary_topology_v1"
    assert protocol["training_ready"] is False
    assert protocol["formal_scene_modified"] is False
    assert protocol["reference_modified"] is False
    for row in protocol["inputs"].values():
        assert _sha(row["path"]) == row["sha256"]
    contracts = protocol["contracts"]
    assert contracts["assume_one_interval_per_slice"] is False
    assert contracts["rejected_hybrid_is_evaluated_without_refitting"] is True
    assert contracts["add_hand_object_avoidance_to_mink"] is False
    assert not any(contracts[key] for key in (
        "run_retargeting", "run_capacity", "run_initialization", "run_training"
    ))


def test_native_and_hybrid_topology_is_simple_but_boundary_is_not_exact():
    report = json.loads(REPORT.read_text())
    assert report["status"] == "boundary_topology_mapped_guard_not_promoted"
    assert report["sampling"]["thumb_bend_slices"] == 257
    assert report["sampling"]["thumb_rota1_coarse_samples_per_slice"] == 513
    topology = report["topology"]
    assert topology["native_interval_count_histogram"] == {"0": 104, "1": 153}
    assert topology["hybrid_interval_count_histogram"] == {"0": 104, "1": 153}
    assert topology["native_multi_interval_slice_indices"] == []
    assert topology["native_is_zero_or_one_upper_tail_interval_on_every_slice"]
    assert topology["native_hybrid_topology_mismatch_slice_count"] == 0
    assert [row["sampled_bend_range_rad"]
            for row in topology["native_sampled_collision_bend_bands"]] == [
        [0.0, 0.900703125], [1.6512890625, 1.83]
    ]
    comparison = report["boundary_comparison"]
    assert comparison["comparable_slice_count"] == 153
    assert comparison["negative_fp_band_slices"] == 32
    assert comparison["positive_fn_band_slices"] == 121
    assert 0.00049 < comparison["maximum_absolute_error_rad"] < 0.00051


def test_transitions_keep_numerical_brackets_and_counterexamples_reproduce():
    report = json.loads(REPORT.read_text())
    widths = [
        transition["numerical_bracket_width_rad"]
        for row in report["slice_results"]
        for family in ("native", "hybrid")
        for transition in row[family]["transitions"]
    ]
    assert widths and max(widths) <= 1e-7
    assert all(row["reproduced"] for row in report["known_counterexamples"])
    assert {row["kind"] for row in report["known_counterexamples"]} == {
        "false_negative", "false_positive"
    }


def test_no_guard_was_promoted_and_gate_order_is_preserved():
    report = json.loads(REPORT.read_text())
    decision = report["decision"]
    assert decision["recommended_next_step"].startswith("boundary_curve_cegis")
    assert not any(decision[key] for key in (
        "guard_promoted", "formal_scene_modified", "reference_modified",
        "mink_retarget_run", "capacity_run", "initialization_run", "training_run",
    ))
    order = decision["promotion_order"]
    assert order.index("rerun_all_198_mink_endpoints_with_self_collision_only") < order.index(
        "rerun_full_horizon_0_197_external_and_floor_attribution_on_new_reference"
    )
    assert order[-1] == "freeze_observation_contract_then_open_first_replay_rl_window"
