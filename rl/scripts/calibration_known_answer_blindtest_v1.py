#!/usr/bin/env python3
"""Public practice and user-operated blind exam tools for calibration v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import sys
import tempfile
from typing import Any
import uuid

import cv2
import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.calibration_known_answer import (  # noqa: E402
    artificial_asymmetric_mesh, checkpoint_error, generate_direct_case,
    generate_image_case, independent_transform, load_public_case, sha256,
    transform_error, write_json, write_public_case,
)
from egoengine_repro.evaluation.taco_calibration_residual import (  # noqa: E402
    CalibrationCandidateRejected, SurfaceObservation, TriangleSurface,
    fit_surface_correction,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (  # noqa: E402
    eroded_target_mask, measured_target_selector,
)
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    backproject_metric_depth,
)


DEFAULT_CONFIG = RL_ROOT / "configs/calibration_known_answer_blindtest_v1.yaml"
MODELS = ("WORLD_FIXED", "CAMERA_LOCAL")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["schema"] != "calibration_known_answer_blindtest_v1":
        raise ValueError("unexpected config schema")
    if cfg["status"] != "PUBLIC_DEVELOPMENT_ONLY" or cfg["formal_exam_executed"] is not False:
        raise ValueError("this repository config must not claim a formal exam")
    forbidden = {
        "formal_secret_generation_by_codex", "real_taco_refit", "real_taco_source_mutation",
        "mink", "physics", "replay", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    }
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden runtime or formal-exam authorization is enabled")
    if set(cfg["public_development"]["quality_cases"][0]) != {
        "name", "noise_std_m", "missing_fraction", "outlier_fraction",
    }:
        raise ValueError("quality case contract changed")
    if cfg["fit"]["fixed_finite_difference_relative_step"] != 0.001:
        raise ValueError("fixed numerical differentiation contract changed")
    return cfg


def _load_shapes(cfg: dict[str, Any]) -> dict[str, trimesh.Trimesh]:
    artificial = artificial_asymmetric_mesh()
    taco_path = Path(cfg["shapes"]["taco"]["path"]).resolve(strict=True)
    taco = trimesh.load_mesh(taco_path, process=False)
    taco.apply_scale(float(cfg["shapes"]["taco"]["scale"]))
    result = {"artificial": artificial, "taco": taco}
    for name, mesh in result.items():
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError(f"{name} mesh is unsupported: surface is not closed and consistent")
    return result


def _case_id(index: int, public_development: bool) -> str:
    return f"public_{index:03d}" if public_development else uuid.uuid4().hex


def generate_cases(
    cfg: dict[str, Any], *, seed: int, public_dir: Path, private_dir: Path,
    public_development: bool,
) -> None:
    if public_dir.exists() or private_dir.exists():
        raise FileExistsError("generation output directories must not already exist")
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (public_dir / "cases").mkdir()
    (private_dir / "truth").mkdir()
    rng = np.random.default_rng(seed)
    shapes = _load_shapes(cfg)
    records, truths = [], []
    index = 0

    def save_case(public: dict[str, np.ndarray], truth: dict[str, Any], metadata: dict[str, Any]) -> None:
        nonlocal index
        index += 1
        identifier = _case_id(index, public_development)
        path = public_dir / "cases" / f"{identifier}.npz"
        owner = truth.pop("owner_maps", None)
        write_public_case(path, public)
        record = {
            "case_id": identifier, "file": f"cases/{identifier}.npz",
            "sha256": sha256(path),
            **(
                metadata if public_development
                else {
                    "case_kind": metadata["case_kind"],
                    "shape_source": metadata["shape_source"],
                    "condition": "UNDISCLOSED",
                    "quality": "UNDISCLOSED",
                }
            ),
        }
        records.append(record)
        if owner is not None:
            owner_path = private_dir / "truth" / f"{identifier}_owner.npy"
            np.save(owner_path, owner, allow_pickle=False)
            truth["owner_map_file"] = f"truth/{identifier}_owner.npy"
            truth["owner_map_sha256"] = sha256(owner_path)
        check_points = rng.uniform(-0.12, 0.12, size=(64, 3))
        check_path = private_dir / "truth" / f"{identifier}_check_points.npy"
        np.save(check_path, check_points, allow_pickle=False)
        truth["check_points_file"] = f"truth/{identifier}_check_points.npy"
        truth["check_points_sha256"] = sha256(check_path)
        truths.append({"case_id": identifier, **metadata, **truth})

    clean = cfg["public_development"]["quality_cases"][0]
    for shape_name, mesh in shapes.items():
        for generation_model in MODELS:
            for correction_case in cfg["public_development"]["correction_cases"]:
                correction = independent_transform(
                    correction_case["translation_m"], correction_case["rotvec_rad"],
                )
                public, truth = generate_direct_case(
                    mesh, correction=correction, generation_model=generation_model,
                    camera_views=int(cfg["public_development"]["camera_views"]),
                    points_per_face=int(cfg["public_development"]["points_per_face"]),
                    rng=rng, noise_std_m=clean["noise_std_m"],
                    missing_fraction=clean["missing_fraction"],
                    outlier_fraction=clean["outlier_fraction"],
                )
                save_case(public, truth, {
                    "case_kind": "direct", "shape_source": shape_name,
                    "condition": correction_case["name"], "quality": "clean",
                })

    combined = next(row for row in cfg["public_development"]["correction_cases"] if row["name"] == "combined")
    correction = independent_transform(combined["translation_m"], combined["rotvec_rad"])
    for generation_model in MODELS:
        for quality in cfg["public_development"]["quality_cases"][1:]:
            public, truth = generate_direct_case(
                shapes["artificial"], correction=correction, generation_model=generation_model,
                camera_views=int(cfg["public_development"]["camera_views"]),
                points_per_face=int(cfg["public_development"]["points_per_face"]), rng=rng,
                noise_std_m=float(quality["noise_std_m"]),
                missing_fraction=float(quality["missing_fraction"]),
                outlier_fraction=float(quality["outlier_fraction"]),
            )
            save_case(public, truth, {
                "case_kind": "direct", "shape_source": "artificial",
                "condition": "combined", "quality": quality["name"],
            })

    first = independent_transform([0.012, -0.006, 0.010], [0.020, -0.010, 0.015])
    second = independent_transform([-0.008, 0.011, -0.005], [-0.018, 0.024, -0.012])
    corrections = [first if i % 2 == 0 else second for i in range(int(cfg["public_development"]["camera_views"]))]
    public, truth = generate_direct_case(
        shapes["artificial"], correction=np.eye(4), generation_model="WORLD_FIXED",
        camera_views=len(corrections), points_per_face=3, rng=rng,
        per_view_corrections=corrections,
    )
    save_case(public, truth, {
        "case_kind": "direct", "shape_source": "artificial",
        "condition": "no_common_correction", "quality": "clean",
    })

    public, truth = generate_direct_case(
        shapes["artificial"], correction=np.eye(4), generation_model="WORLD_FIXED",
        camera_views=2, points_per_face=3, rng=rng,
    )
    count = public["observation_offsets"][1]
    line = np.column_stack([np.linspace(-0.05, 0.05, count), np.zeros(count), np.ones(count)])
    public["points_camera"] = np.concatenate([line, line])
    truth["has_unique_global_answer"] = False
    truth["recovery_transform"] = None
    truth["insufficiency_proof"] = "all observed points are collinear"
    save_case(public, truth, {
        "case_kind": "direct", "shape_source": "artificial",
        "condition": "degenerate_collinear", "quality": "clean",
    })

    image_cfg = cfg["image_cases"]
    intrinsic = np.array([
        [image_cfg["fx"], 0.0, (image_cfg["width"] - 1) / 2.0],
        [0.0, image_cfg["fy"], (image_cfg["height"] - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])
    for row in image_cfg["cases"]:
        public, truth = generate_image_case(
            occluder=row["occluder"], encoding=row["encoding"],
            pose_error_m=float(row["pose_error_m"]), width=int(image_cfg["width"]),
            height=int(image_cfg["height"]), intrinsic=intrinsic,
            depth_scale=float(image_cfg["depth_scale"]), erosion_px=int(image_cfg["erosion_px"]),
            spatial_stride_px=int(image_cfg["spatial_stride_px"]),
            visibility_uncertainty_margin_m=float(image_cfg["visibility_uncertainty_margin_m"]),
        )
        save_case(public, truth, {
            "case_kind": "image", "shape_source": "artificial",
            "condition": row["name"], "quality": row["encoding"],
        })

    if not public_development:
        records = [records[index] for index in rng.permutation(len(records))]
        truths = [truths[index] for index in rng.permutation(len(truths))]
    write_json(public_dir / "manifest.json", {
        "schema": "calibration_blindtest_public_input_v1",
        "contains_private_answer": False,
        "case_count": len(records), "cases": records,
        "field_whitelist": cfg["formal_exam"]["input_field_whitelist"],
    })
    write_json(private_dir / "answer_manifest.json", {
        "schema": "calibration_blindtest_private_answer_v1",
        "case_count": len(truths), "cases": truths,
        "seed_not_serialized": True,
    })
    if not public_development:
        for root in (public_dir, private_dir):
            for path in root.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)


def _observations(case: dict[str, np.ndarray], selector: str) -> tuple[list[SurfaceObservation], list[list[int]]]:
    kind = str(case["case_kind"].item())
    rows, selected = [], []
    if kind == "direct":
        offsets = case["observation_offsets"]
        for index in range(len(offsets) - 1):
            points = case["points_camera"][offsets[index]: offsets[index + 1]]
            rows.append(SurfaceObservation(
                points, case["world_to_camera"][index], case["object_to_world"][index], frame=index,
            ))
            selected.append([])
        return rows, selected
    if kind != "image":
        raise ValueError(f"unknown case kind: {kind}")
    for index, measured in enumerate(case["measured_depth_m"]):
        target = case["target_nominal_depth_m"][index]
        if selector == "legacy":
            mask = eroded_target_mask(target > 0, erosion_px=int(case["erosion_px"]))
            mask &= measured > 0
        elif selector == "occlusion_aware":
            occluders = case["occluder_nominal_depths_m"][index]
            forbidden = np.any(
                np.isfinite(occluders) & (occluders > 0), axis=0,
            )
            mask = measured_target_selector(
                measured, target, occluders,
                uncertainty_margin_m=float(case["visibility_uncertainty_margin_m"]),
                erosion_px=int(case["erosion_px"]),
                forbidden_mask=forbidden,
            )
        else:
            raise ValueError(f"unknown selector: {selector}")
        points, pixels = backproject_metric_depth(
            measured, case["intrinsic"], selector=mask,
            spatial_stride_px=int(case["spatial_stride_px"]),
        )
        rows.append(SurfaceObservation(
            points, case["world_to_camera"][index], case["object_to_world"][index], frame=index,
        ))
        selected.append((pixels[:, 1] * measured.shape[1] + pixels[:, 0]).astype(int).tolist())
    return rows, selected


def solve_cases(cfg: dict[str, Any], input_dir: Path, output_dir: Path, *, version: str) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("solver output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    selector = "legacy" if version == "baseline" else "occlusion_aware"
    diff_step = cfg["fit"][f"{version}_finite_difference_relative_step"]
    predictions = []
    for record in manifest["cases"]:
        path = input_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"public case hash mismatch: {record['case_id']}")
        case = load_public_case(path)
        model_rows = []
        try:
            observations, selected = _observations(case, selector)
            surface = TriangleSurface(case["vertices"], case["faces"])
            for model in MODELS:
                try:
                    fit = fit_surface_correction(
                        observations, surface, model,
                        robust_loss=cfg["fit"]["robust_loss"],
                        robust_scale_m=float(cfg["fit"]["robust_scale_m"]),
                        maximum_translation_m=float(cfg["fit"]["maximum_translation_m"]),
                        maximum_rotation_deg=float(cfg["fit"]["maximum_rotation_deg"]),
                        maximum_function_evaluations=int(cfg["fit"]["maximum_function_evaluations"]),
                        minimum_points=int(cfg["fit"]["minimum_points"]),
                        finite_difference_relative_step=diff_step,
                    )
                    model_rows.append({"model": model, "status": "OUTPUT", "fit": fit})
                except CalibrationCandidateRejected as error:
                    model_rows.append({
                        "model": model, "status": "REJECTED_CANDIDATE",
                        "reason": error.reason_code, "candidate": error.candidate,
                        "accepted_for_use": False,
                    })
                except (ValueError, RuntimeError) as error:
                    model_rows.append({"model": model, "status": "STOPPED", "reason": str(error)})
        except (ValueError, RuntimeError) as error:
            selected = []
            model_rows = [{"model": model, "status": "STOPPED", "reason": str(error)} for model in MODELS]
        predictions.append({
            "case_id": record["case_id"], "selector": selector,
            "finite_difference_relative_step": diff_step,
            "selected_flat_indices": selected, "models": model_rows,
        })
    write_json(output_dir / "predictions.json", {
        "schema": "calibration_blindtest_predictions_v1", "version": version,
        "case_count": len(predictions), "predictions": predictions,
    })


COMPARISON_FIELDS = [
    "case_id", "case_kind", "condition", "quality", "shape_source", "version", "model",
    "answer_status", "true_translation_mm", "estimated_translation_mm", "translation_error_mm",
    "true_rotvec_rad", "estimated_rotvec_rad", "rotation_error_deg", "checkpoint_mean_mm",
    "checkpoint_median_mm", "checkpoint_p95_mm", "checkpoint_maximum_mm", "reason",
]


def _rotvec(matrix: np.ndarray) -> list[float]:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0]
    vector = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    axis = vector / (2.0 * np.sin(angle))
    return (axis * angle).tolist()


def score_cases(
    input_dir: Path, private_dir: Path, baseline_dir: Path, fixed_dir: Path, output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError("score output directory must not already exist")
    output_dir.mkdir(parents=True)
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    truth = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
    truths = {row["case_id"]: row for row in truth["cases"]}
    public_rows = {row["case_id"]: row for row in public["cases"]}
    comparisons, visibility = {"baseline": [], "fixed": []}, []
    for version, directory in (("baseline", baseline_dir), ("fixed", fixed_dir)):
        result = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
        for prediction in result["predictions"]:
            identifier = prediction["case_id"]
            answer = truths[identifier]
            metadata = answer
            if "owner_map_file" in answer:
                owner_path = private_dir / answer["owner_map_file"]
                if sha256(owner_path) != answer["owner_map_sha256"]:
                    raise ValueError("private owner map hash mismatch")
                owners = np.load(owner_path, allow_pickle=False)
                public_case = load_public_case(input_dir / public_rows[identifier]["file"])
                stride = int(public_case["spatial_stride_px"])
                chosen_target = chosen_other = eligible_target = 0
                for frame, indices in enumerate(prediction["selected_flat_indices"]):
                    selected_owner = owners[frame].reshape(-1)[np.asarray(indices, dtype=np.int64)]
                    chosen_target += int(np.sum(selected_owner == 1))
                    chosen_other += int(np.sum(selected_owner > 1))
                    lattice = np.zeros(owners[frame].shape, dtype=bool)
                    lattice[::stride, ::stride] = True
                    eligible_target += int(np.sum((owners[frame] == 1) & lattice))
                total = chosen_target + chosen_other
                visibility.append({
                    "case_id": identifier, "condition": metadata["condition"], "version": version,
                    "selected_total": total, "selected_true_target": chosen_target,
                    "selected_occluder_or_other": chosen_other,
                    "contamination_fraction": None if total == 0 else chosen_other / total,
                    "eligible_true_target": eligible_target,
                    "retained_true_target_fraction": None if eligible_target == 0 else chosen_target / eligible_target,
                    "deleted_true_target": eligible_target - chosen_target,
                    "empty_selection": total == 0,
                })
            for model_row in prediction["models"]:
                row = {
                    "case_id": identifier, "case_kind": metadata["case_kind"],
                    "condition": metadata["condition"], "quality": metadata["quality"],
                    "shape_source": metadata["shape_source"], "version": version,
                    "model": model_row["model"], "answer_status": model_row["status"],
                    "true_translation_mm": "", "estimated_translation_mm": "",
                    "translation_error_mm": "", "true_rotvec_rad": "",
                    "estimated_rotvec_rad": "", "rotation_error_deg": "",
                    "checkpoint_mean_mm": "", "checkpoint_median_mm": "",
                    "checkpoint_p95_mm": "", "checkpoint_maximum_mm": "",
                    "reason": model_row.get("reason", ""),
                }
                expected_for_model = answer["generation_model"] in (model_row["model"], "IDENTITY_BOTH_MODELS")
                if not answer["has_unique_global_answer"]:
                    row["answer_status"] = (
                        "OUTPUT_WITHOUT_IDENTIFIABILITY_CERTIFICATE"
                        if model_row["status"] == "OUTPUT" else "STOPPED_INFORMATION_INSUFFICIENT"
                    )
                    row["reason"] = answer.get("insufficiency_proof", "no common correction exists")
                elif model_row["status"] == "OUTPUT" and expected_for_model:
                    expected = np.asarray(answer["recovery_transform"])
                    estimate = np.asarray(model_row["fit"]["transform"])
                    errors = transform_error(estimate, expected)
                    points_path = private_dir / answer["check_points_file"]
                    if sha256(points_path) != answer["check_points_sha256"]:
                        raise ValueError("private checkpoint hash mismatch")
                    check = checkpoint_error(estimate, expected, np.load(points_path, allow_pickle=False))
                    row.update({
                        "true_translation_mm": json.dumps((1000 * expected[:3, 3]).tolist()),
                        "estimated_translation_mm": json.dumps((1000 * estimate[:3, 3]).tolist()),
                        "translation_error_mm": errors["translation_error_mm"],
                        "true_rotvec_rad": json.dumps(_rotvec(expected[:3, :3])),
                        "estimated_rotvec_rad": json.dumps(_rotvec(estimate[:3, :3])),
                        "rotation_error_deg": errors["rotation_error_deg"],
                        "checkpoint_mean_mm": check["mean_mm"],
                        "checkpoint_median_mm": check["median_mm"],
                        "checkpoint_p95_mm": check["p95_mm"],
                        "checkpoint_maximum_mm": check["maximum_mm"],
                    })
                elif model_row["status"] == "OUTPUT":
                    row["answer_status"] = "OUTPUT_ALTERNATIVE_MODEL_NO_UNIQUE_TRUTH_COMPARISON"
                    row["estimated_translation_mm"] = json.dumps(
                        (1000 * np.asarray(model_row["fit"]["transform"])[:3, 3]).tolist()
                    )
                comparisons[version].append(row)

    for version, rows in comparisons.items():
        with (output_dir / f"{version}_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    combined_rows = comparisons["baseline"] + comparisons["fixed"]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(combined_rows)
    visibility_fields = list(visibility[0]) if visibility else []
    with (output_dir / "occlusion_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=visibility_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(visibility)
    shutil.copy2(output_dir / "occlusion_comparison.csv", output_dir / "visibility_comparison.csv")
    raw = output_dir / "raw_predictions"
    raw.mkdir()
    shutil.copy2(baseline_dir / "predictions.json", raw / "baseline_predictions.json")
    shutil.copy2(fixed_dir / "predictions.json", raw / "fixed_predictions.json")
    comparable = [row for row in combined_rows if row["translation_error_mm"]]
    (output_dir / "comparison.md").write_text(
        "# 校准已知答案逐题评分\n\n"
        f"- 总算法行数：`{len(combined_rows)}`\n"
        f"- 有唯一答案且产生可比输出：`{len(comparable)}`\n"
        "- 未设置精度通过线；完整平移、旋转和检查点误差见 `comparison.csv`。\n"
        "- 停止、失败、替代模型和无公共修正题均保留，没有从汇总中静默删除。\n",
        encoding="utf-8",
    )
    input_hash = sha256(input_dir / "manifest.json")
    answer_hash = sha256(private_dir / "answer_manifest.json")
    baseline_hash = sha256(baseline_dir / "predictions.json")
    fixed_hash = sha256(fixed_dir / "predictions.json")
    write_json(output_dir / "exam_integrity_report.json", {
        "schema": "calibration_blindtest_exam_integrity_v1",
        "input_manifest_sha256": input_hash,
        "private_answer_manifest_sha256": answer_hash,
        "baseline_predictions_sha256": baseline_hash,
        "fixed_predictions_sha256": fixed_hash,
        "predictions_existed_before_scoring": True,
        "answer_not_part_of_solver_input_contract": True,
        "precision_pass_threshold_defined": False,
    })
    write_json(output_dir / "detailed_score.json", {
        "schema": "calibration_known_answer_public_score_v1",
        "precision_pass_threshold_defined": False,
        "comparisons": comparisons, "visibility": visibility,
    })


def isolation_demo(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="calibration_isolation_") as root_value:
        root = Path(root_value)
        public, private, writable = root / "public", root / "private", root / "output"
        public.mkdir(); private.mkdir(); writable.mkdir()
        (public / "question.txt").write_text("public\n", encoding="utf-8")
        (private / "fake_secret.txt").write_text("must-not-be-visible\n", encoding="utf-8")
        probe = (
            "import json,pathlib,socket; "
            "secret=pathlib.Path('/private/fake_secret.txt').exists(); "
            "public=pathlib.Path('/input/question.txt').read_text().strip(); "
            "network=True; "
            "\ntry:\n socket.create_connection(('1.1.1.1',53),timeout=.2)\n"
            "except OSError:\n network=False\n"
            "pathlib.Path('/output/report.json').write_text(json.dumps({'secret_visible':secret,'public':public,'network_available':network}))"
        )
        command = [
            "bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
            "--ro-bind", str(public), "/input", "--bind", str(writable), "/output",
            "--tmpfs", "/tmp", "/usr/bin/python3", "-c", probe,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        report_path = writable / "report.json"
        probe_report = json.loads(report_path.read_text()) if report_path.exists() else None
        report = {
            "schema": "calibration_blindtest_isolation_demo_v1",
            "mechanism": "bubblewrap user/mount/network namespaces",
            "command_exit_code": result.returncode,
            "fake_secret_host_path_not_mounted": True,
            "probe": probe_report,
            "stderr": result.stderr,
            "formal_exam_isolation_proven": False,
            "interpretation": "local fake-secret demo only; user must provide independent formal environment",
        }
        write_json(output, report)
        return report


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def develop(cfg_path: Path) -> None:
    cfg = _load_config(cfg_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="calibration_public_practice_") as root_value:
        root = Path(root_value)
        public, private = root / "input", root / "private"
        baseline, fixed, scored = root / "baseline", root / "fixed", root / "score"
        generate_cases(
            cfg, seed=int(cfg["public_development"]["seed"]), public_dir=public,
            private_dir=private, public_development=True,
        )
        solve_cases(cfg, public, baseline, version="baseline")
        solve_cases(cfg, public, fixed, version="fixed")
        score_cases(public, private, baseline, fixed, scored)
        shutil.copy2(public / "manifest.json", output / "public_case_manifest.json")
        shutil.copy2(scored / "baseline_comparison.csv", output / "baseline_comparison.csv")
        shutil.copy2(scored / "fixed_comparison.csv", output / "fixed_comparison.csv")
        shutil.copy2(scored / "occlusion_comparison.csv", output / "occlusion_comparison.csv")
        shutil.copy2(scored / "detailed_score.json", output / "detailed_score.json")

    isolation = isolation_demo(output / "isolation_demo_report.json")
    baseline_rows = _csv_rows(output / "baseline_comparison.csv")
    fixed_rows = _csv_rows(output / "fixed_comparison.csv")
    visibility_rows = _csv_rows(output / "occlusion_comparison.csv")
    comparable_baseline = [r for r in baseline_rows if r["translation_error_mm"]]
    comparable_fixed = [r for r in fixed_rows if r["translation_error_mm"]]
    numeric = lambda rows, key: np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    visibility_by_version = {
        version: [row for row in visibility_rows if row["version"] == version]
        for version in ("baseline", "fixed")
    }
    summary_metrics = {
        "baseline": {
            "comparable_outputs": len(comparable_baseline),
            "translation_error_mm_median": float(np.median(numeric(comparable_baseline, "translation_error_mm"))),
            "translation_error_mm_max": float(np.max(numeric(comparable_baseline, "translation_error_mm"))),
            "rotation_error_deg_median": float(np.median(numeric(comparable_baseline, "rotation_error_deg"))),
            "rotation_error_deg_max": float(np.max(numeric(comparable_baseline, "rotation_error_deg"))),
            "selected_occluder_points": sum(int(r["selected_occluder_or_other"]) for r in visibility_by_version["baseline"]),
        },
        "fixed": {
            "comparable_outputs": len(comparable_fixed),
            "translation_error_mm_median": float(np.median(numeric(comparable_fixed, "translation_error_mm"))),
            "translation_error_mm_max": float(np.max(numeric(comparable_fixed, "translation_error_mm"))),
            "rotation_error_deg_median": float(np.median(numeric(comparable_fixed, "rotation_error_deg"))),
            "rotation_error_deg_max": float(np.max(numeric(comparable_fixed, "rotation_error_deg"))),
            "selected_occluder_points": sum(int(r["selected_occluder_or_other"]) for r in visibility_by_version["fixed"]),
        },
    }
    write_json(output / "source_manifest.json", {
        "schema": "calibration_known_answer_source_manifest_v1",
        "repository_head_at_development": _git_head(),
        "baseline_commit": cfg["baseline_commit"],
        "config": _artifact(cfg_path),
        "taco_shape": _artifact(Path(cfg["shapes"]["taco"]["path"])),
        "dependencies": {
            "python": sys.version, "numpy": np.__version__,
            "scipy": __import__("scipy").__version__, "open3d": __import__("open3d").__version__,
            "trimesh": trimesh.__version__, "opencv": cv2.__version__,
        },
    })
    write_json(output / "freeze_manifest.json", {
        "schema": "calibration_known_answer_freeze_manifest_v1",
        "formal_exam_executed": False,
        "baseline": {"git_commit": cfg["baseline_commit"], "finite_difference_relative_step": None, "selector": "legacy_target_only"},
        "fixed": {
            "git_commit": _git_head(),
            "finite_difference_relative_step": cfg["fit"]["fixed_finite_difference_relative_step"],
            "selector": "nominal_first_surface_occlusion_aware",
        },
        "files": {
            str(path.relative_to(REPO_ROOT)): _artifact(path)
            for path in (
                cfg_path,
                RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py",
                RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_visibility.py",
                RL_ROOT / "src/egoengine_repro/evaluation/calibration_known_answer.py",
                Path(__file__),
            )
        },
    })
    (output / "bugfix_log.md").write_text(
        "# 已证实并修复的程序错误\n\n"
        "1. **数值差分精度错配**：Open3D signed distance 为 float32，旧 SciPy 默认差分可被舍入吞掉。"
        "保留 `diff_step=null` 的原版入口；修复版固定 `diff_step=0.001`，不改变目标、loss 或求解器。\n"
        "2. **目标单独轮廓忽略遮挡**：旧选择器在目标投影内直接读取 measured depth。"
        "修复版同时使用名义目标、双手和其他物体的深度顺序，近似平局保守排除。\n\n"
        f"公开练习汇总：`{json.dumps(summary_metrics, ensure_ascii=False)}`。\n",
        encoding="utf-8",
    )
    insufficient = [r for r in fixed_rows if r["answer_status"] == "OUTPUT_WITHOUT_IDENTIFIABILITY_CERTIFICATE"]
    (output / "method_limits.md").write_text(
        "# 当前方法限制\n\n"
        "- 点到表面目标不是点对应目标；表面分数下降不等于恢复真值。逐题矩阵误差见 CSV。\n"
        "- 当前求解器没有一般的六自由度可辨识性证书；无公共修正题仍可能输出折中值。\n"
        f"- 公开练习中无唯一答案却输出的行数：`{len(insufficient)}`。这属于方法限制，未增加新目标或模型选择器。\n"
        "- 名义手/物体姿态有误时，可见性过滤可能漏删或错删；相关题单列，不读取真值 ownership。\n"
        "- 合成测试不能证明真实 TACO 约 2 cm 偏差已经修复；真实结论仍为 `CALIBRATION_RESIDUAL_UNRESOLVED`。\n",
        encoding="utf-8",
    )
    instructions = f"""# 用户主持的正式保密考试

正式考试**尚未执行**。在 Codex 无权访问的独立账号或机器上，固定仓库提交后执行：

```bash
PY=/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
TOOL=rl/scripts/calibration_known_answer_blindtest_v1.py

# 1. 检查隔离（只使用假秘密）
$PY $TOOL isolation-demo --output /secure/audit/isolation_demo.json

# 2. 用户在私有环境生成题面和答案；seed 文件不得挂给 solver
$PY $TOOL generate --secret-seed-file /secure/private/seed.txt \\
  --public-dir /secure/public/input --private-dir /secure/private/answer

# 3. 分别运行两个冻结版本；run-bwrap 只挂载只读题面、冻结代码和单独输出目录
$PY $TOOL run-bwrap --input-dir /secure/public/input --output-dir /secure/output/baseline --version baseline
$PY $TOOL run-bwrap --input-dir /secure/public/input --output-dir /secure/output/fixed --version fixed

# 4. 两份原始预测写完并改为只读后，用户才运行评分
$PY $TOOL score --input-dir /secure/public/input --private-dir /secure/private/answer \\
  --baseline-dir /secure/output/baseline --fixed-dir /secure/output/fixed \\
  --output-dir /secure/score

# 5. 校验公开输入、私有答案和原始输出中的哈希
$PY $TOOL verify --input-dir /secure/public/input --private-dir /secure/private/answer
```

不要把 `/secure/private`、用户主目录、开发工作区或网络挂入 solver。成绩公开前不要向 Codex提供题目、日志或中间分数。
"""
    (output / "user_exam_instructions.md").write_text(instructions, encoding="utf-8")
    summary = f"""# 校准算法已知答案测试与遮挡修复 v1

**状态：开发与公开练习完成；正式保密考试未执行。**

- 本轮直接比较了正确 4×4 恢复变换与算法输出；没有用表面距离冒充真值误差。
- 已证实并最小修复两个实现错误：float32 距离场与默认有限差分尺度错配；目标单独轮廓未排除名义手/其他物体遮挡。
- 原版公开可比题：translation median/max = `{summary_metrics['baseline']['translation_error_mm_median']:.6f}/{summary_metrics['baseline']['translation_error_mm_max']:.6f} mm`；rotation median/max = `{summary_metrics['baseline']['rotation_error_deg_median']:.6f}/{summary_metrics['baseline']['rotation_error_deg_max']:.6f} deg`。
- 修复版：translation median/max = `{summary_metrics['fixed']['translation_error_mm_median']:.6f}/{summary_metrics['fixed']['translation_error_mm_max']:.6f} mm`；rotation median/max = `{summary_metrics['fixed']['rotation_error_deg_median']:.6f}/{summary_metrics['fixed']['rotation_error_deg_max']:.6f} deg`。
- 遮挡题误选 occluder 点总数：原版 `{summary_metrics['baseline']['selected_occluder_points']}`，修复版 `{summary_metrics['fixed']['selected_occluder_points']}`。完整逐题误选、保留与错删见 `occlusion_comparison.csv`；空选择不会被记成零污染。
- 人工形状与现有 TACO mesh 结果分别保留在 CSV，不合并伪装成单一精度。
- 无公共修正/退化输入暴露了当前方法缺少一般可辨识性证书；未更换目标、增加多起点或扩展模型。
- bubblewrap 假秘密演示：exit `{isolation['command_exit_code']}`，secret visible `{None if isolation['probe'] is None else isolation['probe']['secret_visible']}`，network available `{None if isolation['probe'] is None else isolation['probe']['network_available']}`。这不等于正式环境已经隔离。
- 真实 TACO 数据没有重拟合或修改，上一轮 `CALIBRATION_RESIDUAL_UNRESOLVED` 结论保持不变。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    rows = []
    for path in sorted(output.iterdir()):
        if path.name != "server_artifacts.sha256" and path.is_file():
            rows.append(f"{sha256(path)}  {path.name}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify(input_dir: Path, private_dir: Path | None) -> None:
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    for row in public["cases"]:
        if sha256(input_dir / row["file"]) != row["sha256"]:
            raise ValueError(f"public case hash mismatch: {row['case_id']}")
    if private_dir is not None:
        truth = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
        for row in truth["cases"]:
            for field in ("owner_map", "check_points"):
                file_key, hash_key = f"{field}_file", f"{field}_sha256"
                if file_key in row and sha256(private_dir / row[file_key]) != row[hash_key]:
                    raise ValueError(f"private {field} hash mismatch: {row['case_id']}")
    print(json.dumps({"public_input_verified": True, "private_answer_verified": private_dir is not None}))


def bwrap_command(input_dir: Path, output_dir: Path, version: str) -> list[str]:
    python_executable = Path(sys.executable).absolute()
    python_root = python_executable.parents[1]
    return [
        "bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--ro-bind", str(python_root), str(python_root),
        "--ro-bind", str(REPO_ROOT), "/workspace", "--ro-bind", str(input_dir.resolve()), "/input",
        "--bind", str(output_dir.resolve()), "/output", "--tmpfs", "/tmp", "--chdir", "/workspace",
        str(python_executable), "/workspace/rl/scripts/calibration_known_answer_blindtest_v1.py",
        "solve", "--input-dir", "/input", "--output-dir", "/output", "--version", version,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("develop")
    generate = commands.add_parser("generate")
    generate.add_argument("--secret-seed-file", type=Path, required=True)
    generate.add_argument("--public-dir", type=Path, required=True)
    generate.add_argument("--private-dir", type=Path, required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--input-dir", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    solve.add_argument("--version", choices=("baseline", "fixed"), required=True)
    score = commands.add_parser("score")
    score.add_argument("--input-dir", type=Path, required=True)
    score.add_argument("--private-dir", type=Path, required=True)
    score.add_argument("--baseline-dir", type=Path, required=True)
    score.add_argument("--fixed-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    demo = commands.add_parser("isolation-demo")
    demo.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--input-dir", type=Path, required=True)
    check.add_argument("--private-dir", type=Path)
    printed = commands.add_parser("print-bwrap")
    printed.add_argument("--input-dir", type=Path, required=True)
    printed.add_argument("--output-dir", type=Path, required=True)
    printed.add_argument("--version", choices=("baseline", "fixed"), required=True)
    isolated = commands.add_parser("run-bwrap")
    isolated.add_argument("--input-dir", type=Path, required=True)
    isolated.add_argument("--output-dir", type=Path, required=True)
    isolated.add_argument("--version", choices=("baseline", "fixed"), required=True)
    args = parser.parse_args()
    cfg = _load_config(args.config.resolve(strict=True))
    if args.command == "develop":
        develop(args.config.resolve(strict=True))
    elif args.command == "generate":
        seed_text = args.secret_seed_file.resolve(strict=True).read_text(encoding="utf-8").strip()
        seed = int(seed_text)
        generate_cases(cfg, seed=seed, public_dir=args.public_dir, private_dir=args.private_dir, public_development=False)
    elif args.command == "solve":
        solve_cases(cfg, args.input_dir.resolve(strict=True), args.output_dir, version=args.version)
    elif args.command == "score":
        score_cases(
            args.input_dir.resolve(strict=True), args.private_dir.resolve(strict=True),
            args.baseline_dir.resolve(strict=True), args.fixed_dir.resolve(strict=True), args.output_dir,
        )
    elif args.command == "isolation-demo":
        isolation_demo(args.output)
    elif args.command == "verify":
        verify(args.input_dir.resolve(strict=True), None if args.private_dir is None else args.private_dir.resolve(strict=True))
    elif args.command == "print-bwrap":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        print(shlex.join(bwrap_command(args.input_dir, args.output_dir, args.version)))
    elif args.command == "run-bwrap":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        subprocess.run(bwrap_command(args.input_dir, args.output_dir, args.version), check=True)
        prediction = args.output_dir / "predictions.json"
        prediction.chmod(0o444)
        print(json.dumps({"isolated_solver_completed": True, "predictions_sha256": sha256(prediction)}))


if __name__ == "__main__":
    main()
