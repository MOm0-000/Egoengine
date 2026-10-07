from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from calibration_single_frame_comparison_v1 import (  # noqa: E402
    METHODS,
    _pairwise_variation,
    _truth_for_model,
    _write_csv,
    score_cases,
)
from egoengine_repro.evaluation.calibration_known_answer import (  # noqa: E402
    independent_inverse,
    independent_transform,
    transform_error,
    write_json,
)
from egoengine_repro.evaluation.calibration_single_frame_cases import (  # noqa: E402
    TRUTH_UNIQUE,
    dual_coordinate_truth,
)


def test_dual_coordinate_truth_uses_full_conjugation():
    world_to_camera = independent_transform(
        [0.18, -0.09, 0.31], [0.21, -0.14, 0.08],
    )
    recovery = independent_transform(
        [0.012, -0.007, 0.004], [-0.06, 0.03, 0.04],
    )
    camera, world = dual_coordinate_truth(
        recovery, "WORLD_FIXED", world_to_camera,
    )
    expected = world_to_camera @ recovery @ independent_inverse(world_to_camera)
    assert np.allclose(world, recovery, rtol=0.0, atol=1e-15)
    assert np.allclose(camera, expected, rtol=0.0, atol=1e-15)


def test_truth_lookup_is_per_frame_and_relative_rotation_error_is_exact():
    first = independent_transform([0.001, 0.0, 0.0], [0.02, 0.0, 0.0])
    second = independent_transform([0.009, 0.0, 0.0], [0.0, -0.07, 0.0])
    frames = [
        {"injected_recovery_world": first, "injected_recovery_camera": first},
        {"injected_recovery_world": second, "injected_recovery_camera": second},
    ]
    assert not np.array_equal(
        _truth_for_model(frames[0], "WORLD_FIXED"),
        _truth_for_model(frames[1], "WORLD_FIXED"),
    )
    assert transform_error(second, second) == {
        "translation_error_mm": 0.0,
        "rotation_error_deg": 0.0,
    }
    relative = transform_error(second, first)
    assert relative["translation_error_mm"] > 0.0
    assert relative["rotation_error_deg"] > 0.0


def test_csv_preserves_numeric_and_string_zero(tmp_path: Path):
    path = tmp_path / "zeros.csv"
    _write_csv(
        path,
        [{"integer": 0, "float": 0.0, "text": "0.0", "missing": None}],
        ["integer", "float", "text", "missing"],
    )
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["integer"] == "0"
    assert row["float"] == "0.0"
    assert row["text"] == "0.0"
    assert row["missing"] == ""


def test_pairwise_variation_uses_relative_rigid_error():
    transforms = [
        independent_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        independent_transform([0.001, 0.0, 0.0], [0.0, 0.0, np.pi / 2]),
    ]
    result = _pairwise_variation(transforms)
    assert result["pair_count"] == 1
    assert result["translation_max_mm"] == pytest.approx(1.0)
    assert result["rotation_max_deg"] == pytest.approx(90.0)


def _score_fixture(root: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    input_dir = root / "input"
    private_dir = root / "private"
    prepared_dir = root / "prepared"
    input_dir.mkdir()
    private_dir.mkdir()
    (prepared_dir / "frames").mkdir(parents=True)
    write_json(input_dir / "manifest.json", {
        "group_count": 1,
        "frame_count": 2,
        "cases": [{
            "case_id": "case", "file": "unused.npz", "sha256": "unused",
            "frame_count": 2,
        }],
    })
    identity = np.eye(4)
    shifted = independent_transform([0.002, -0.001, 0.003], [0.01, 0.0, 0.0])
    frames = []
    records = []
    for frame_id, truth in enumerate((identity, shifted)):
        task_id = f"case_f{frame_id:03d}"
        frame_file = prepared_dir / "frames" / f"{task_id}.npz"
        np.savez_compressed(
            frame_file,
            points_camera=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            world_to_camera=identity,
            object_to_world=identity,
            case_kind=np.asarray("direct"),
        )
        from egoengine_repro.evaluation.calibration_known_answer import sha256
        records.append({
            "task_id": task_id, "case_id": "case", "frame_id": frame_id,
            "frame_file": f"frames/{task_id}.npz", "frame_sha256": sha256(frame_file),
            "model_file": "unused.npz", "model_sha256": "model-hash",
            "selected_index_sha256": "selection-hash", "selected_flat_indices": [],
        })
        frames.append({
            "frame_id": frame_id,
            "frame_truth_status": TRUTH_UNIQUE,
            "truth_status_reason": "public scoring fixture",
            "injected_recovery_world": truth.tolist(),
            "injected_recovery_camera": truth.tolist(),
            "check_points_local": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        })
    write_json(prepared_dir / "manifest.json", {
        "frame_count": 2, "records": records,
    })
    write_json(private_dir / "answer_manifest.json", {
        "cases": [{
            "case_id": "case", "case_kind": "direct", "condition": "fixture",
            "quality": "clean", "shape_source": "artificial",
            "generation_model": "WORLD_FIXED", "frames": frames,
        }],
    })
    method_dirs = {}
    for method in METHODS:
        method_dir = root / method.lower()
        method_dir.mkdir()
        predictions = []
        for frame_id, truth in enumerate((identity, shifted)):
            task_id = f"case_f{frame_id:03d}"
            models = []
            for coordinate_model in ("WORLD_FIXED", "CAMERA_LOCAL"):
                stopped = frame_id == 1
                models.append({
                    "coordinate_model": coordinate_model,
                    "transform_units": "metres_and_rotation_matrix",
                    "raw_solver_status": "STOPPED_INSUFFICIENT_CORRESPONDENCES" if stopped else "OUTPUT",
                    "raw_solver_reason": "fixture stop" if stopped else "",
                    "input_evidence_status": "TARGET_POINTS_PRESENT",
                    "fit": None if stopped else {
                        "transform": truth.tolist(),
                        "registration_result": {
                            "fitness": 0.0,
                            "inlier_rmse_m": 0.0,
                            "correspondence_count": 0,
                        },
                        "safety_advisory": {"within_bounds": True},
                    },
                    "solve_seconds": 0.0,
                    "peak_rss_delta_kib": 0,
                })
            record = records[frame_id]
            predictions.append({
                "task_id": task_id, "case_id": "case", "frame_id": frame_id,
                "frame_sha256": record["frame_sha256"],
                "model_sha256": "model-hash",
                "selected_index_sha256": "selection-hash",
                "preprocessing_seconds": 0.0,
                "model_sampling_seconds": 0.0,
                "method": method,
                "models": models,
            })
        write_json(method_dir / "predictions.json", {
            "method": method, "frame_count": 2, "predictions": predictions,
        })
        (method_dir / "predictions.json").chmod(0o444)
        method_dirs[method] = method_dir
    return input_dir, private_dir, prepared_dir, method_dirs


def test_scoring_keeps_stopped_frame_and_zero_metrics(tmp_path: Path):
    input_dir, private_dir, prepared_dir, methods = _score_fixture(tmp_path)
    output = tmp_path / "score"
    score_cases(input_dir, private_dir, prepared_dir, methods, output)
    with (output / "frame_comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    comparable = [row for row in rows if row["included_in_accuracy_summary"] == "True"]
    stopped = [row for row in rows if row["raw_solver_status"].startswith("STOPPED")]
    assert len(comparable) == 6
    assert len(stopped) == 6
    assert all(row["translation_error_mm"] == "0.0" for row in comparable)
    assert all(row["rotation_error_deg"] == "0.0" for row in comparable)
    assert all(row["raw_solver_reason"] == "fixture stop" for row in stopped)


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "units", "nonfinite", "nonrigid", "unsealed"],
)
def test_scoring_fails_closed_on_invalid_predictions(tmp_path: Path, fault: str):
    input_dir, private_dir, prepared_dir, methods = _score_fixture(tmp_path)
    path = methods[METHODS[0]] / "predictions.json"
    path.chmod(0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fault == "duplicate":
        payload["predictions"][1] = payload["predictions"][0]
    elif fault == "missing":
        payload["predictions"] = payload["predictions"][:1]
    elif fault == "units":
        payload["predictions"][0]["models"][0]["transform_units"] = "millimetres"
    elif fault == "nonfinite":
        payload["predictions"][0]["models"][0]["fit"]["transform"][0][0] = float("nan")
    elif fault == "nonrigid":
        payload["predictions"][0]["models"][0]["fit"]["transform"][0][0] = 2.0
    else:
        pass
    write_json(path, payload)
    if fault != "unsealed":
        path.chmod(0o444)
    with pytest.raises(ValueError):
        score_cases(
            input_dir, private_dir, prepared_dir, methods,
            tmp_path / f"score_{fault}",
        )
