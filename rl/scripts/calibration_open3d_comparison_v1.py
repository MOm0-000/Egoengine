#!/usr/bin/env python3
"""Three-way known-answer comparison for frozen current and Open3D solvers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

import cv2
import numpy as np
import open3d as o3d
import scipy
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.calibration_comparison_cases import (  # noqa: E402
    generate_random_direct_case,
    generate_random_image_case,
    random_correction,
    random_small_image_correction,
    randomized_asymmetric_mesh,
)
from egoengine_repro.evaluation.calibration_known_answer import (  # noqa: E402
    checkpoint_error,
    independent_transform,
    load_public_case,
    sha256,
    transform_error,
    write_json,
    write_public_case,
)
from egoengine_repro.evaluation.open3d_calibration import (  # noqa: E402
    SampledSurface,
    deterministic_sampled_surface,
    fit_open3d_surface_correction,
)
from egoengine_repro.evaluation.taco_calibration_residual import (  # noqa: E402
    SurfaceObservation,
    TriangleSurface,
    fit_surface_correction,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (  # noqa: E402
    measured_target_selector,
)
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    backproject_metric_depth,
)


DEFAULT_CONFIG = RL_ROOT / "configs/calibration_open3d_comparison_v1.yaml"
METHODS = (
    "CURRENT_FIXED", "OPEN3D_POINT_TO_PLANE", "OPEN3D_ROBUST_POINT_TO_PLANE",
)
MODELS = ("WORLD_FIXED", "CAMERA_LOCAL")
PREPARED_KEYS = {
    "vertices", "faces", "points_camera", "observation_offsets", "world_to_camera",
    "object_to_world", "sample_points_local", "sample_normals_local",
    "sample_face_indices", "sample_points_per_face",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["schema"] != "calibration_open3d_comparison_v1":
        raise ValueError("unexpected comparison config schema")
    if cfg["status"] != "PUBLIC_DEVELOPMENT_ONLY" or cfg["formal_exam_executed"] is not False:
        raise ValueError("repository config must not claim that the formal exam ran")
    forbidden = ("formal_secret_generation_by_codex", "real_taco_refit", "real_taco_source_mutation",
                 "physics", "replay", "reinforcement_learning", "chunk_commit")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden authorization is enabled")
    if tuple(cfg["methods"]) != METHODS:
        raise ValueError("the three frozen comparison methods changed")
    residual = RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py"
    known = RL_ROOT / "src/egoengine_repro/evaluation/calibration_known_answer.py"
    if sha256(residual) != cfg["current_fixed"]["residual_module_sha256"]:
        raise ValueError("CURRENT_FIXED residual implementation changed")
    if sha256(known) != cfg["current_fixed"]["known_answer_module_sha256"]:
        raise ValueError("CURRENT_FIXED known-answer implementation changed")
    wheel = Path(cfg["open3d"]["wheel"])
    if sha256(wheel) != cfg["open3d"]["wheel_sha256"]:
        raise ValueError("Open3D wheel hash mismatch")
    return cfg


def _load_taco_shape(cfg: dict[str, Any]) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(Path(cfg["shapes"]["taco"]["path"]).resolve(strict=True), process=False)
    mesh.apply_scale(float(cfg["shapes"]["taco"]["scale"]))
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("TACO shape is not a supported closed surface")
    return mesh


def _case_id(index: int, public_development: bool) -> str:
    return f"public_{index:03d}" if public_development else uuid.uuid4().hex


def generate_cases(
    cfg: dict[str, Any], *, seed: int, public_dir: Path, private_dir: Path,
    public_development: bool,
) -> None:
    if public_dir.exists() or private_dir.exists():
        raise FileExistsError("generation output directories must not exist")
    public_dir.mkdir(parents=True); (public_dir / "cases").mkdir()
    private_dir.mkdir(parents=True); (private_dir / "truth").mkdir()
    rng = np.random.default_rng(seed)
    budget = cfg["public_development"] if public_development else cfg["formal_exam"]
    taco = _load_taco_shape(cfg)
    records: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []

    def save_case(public: dict[str, np.ndarray], truth: dict[str, Any], metadata: dict[str, Any]) -> None:
        identifier = _case_id(len(records) + 1, public_development)
        path = public_dir / "cases" / f"{identifier}.npz"
        owner = truth.pop("owner_maps", None)
        write_public_case(path, public)
        records.append({
            "case_id": identifier,
            "file": f"cases/{identifier}.npz",
            "sha256": sha256(path),
            **(metadata if public_development else {
                "case_kind": metadata["case_kind"],
                "shape_source": metadata["shape_source"],
                "condition": "UNDISCLOSED",
                "quality": "UNDISCLOSED",
            }),
        })
        if owner is not None:
            owner_path = private_dir / "truth" / f"{identifier}_owner.npy"
            np.save(owner_path, owner, allow_pickle=False)
            truth["owner_map_file"] = f"truth/{identifier}_owner.npy"
            truth["owner_map_sha256"] = sha256(owner_path)
        check_path = private_dir / "truth" / f"{identifier}_check_points.npy"
        np.save(check_path, rng.uniform(-0.12, 0.12, size=(64, 3)), allow_pickle=False)
        truth["check_points_file"] = f"truth/{identifier}_check_points.npy"
        truth["check_points_sha256"] = sha256(check_path)
        truths.append({"case_id": identifier, **metadata, **truth})

    corrections = ("identity", "translation", "rotation", "combined")
    for index in range(int(budget["direct_clean"])):
        shape_source = "taco" if index % 2 else "artificial"
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        model = MODELS[(index // 2) % 2]
        correction_kind = corrections[(index // 4) % len(corrections)]
        public, truth = generate_random_direct_case(
            mesh, correction=random_correction(rng, correction_kind), generation_model=model,
            camera_views=5, points_per_face=3, quality="clean", rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": shape_source,
            "condition": correction_kind, "quality": "clean",
        })

    qualities = ("noise", "random_missing", "local_missing", "outliers", "combined", "strong_combined")
    for index in range(int(budget["direct_degraded"])):
        shape_source = "taco" if index % 2 else "artificial"
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        model = MODELS[(index // 2) % 2]
        quality = qualities[index % len(qualities)]
        public, truth = generate_random_direct_case(
            mesh, correction=random_correction(rng, "combined"), generation_model=model,
            camera_views=5, points_per_face=3, quality=quality, rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": shape_source,
            "condition": "combined", "quality": quality,
        })

    image_cfg = cfg["image"]
    intrinsic = np.asarray([
        [image_cfg["fx"], 0.0, (image_cfg["width"] - 1) / 2],
        [0.0, image_cfg["fy"], (image_cfg["height"] - 1) / 2],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    occluders = (
        "none", "hand_partial", "other_front", "other_behind",
        "hand_complete", "hand_large", "hand_partial", "other_front",
    )
    for index in range(int(budget["image"])):
        model = MODELS[index % 2]
        occluder = occluders[index % len(occluders)]
        pose_error = 0.015 if index % 8 == 6 else 0.0
        correction_regime = "small_nonzero" if index % 8 < 4 else "stress_nonzero"
        image_correction = (
            random_small_image_correction(rng)
            if correction_regime == "small_nonzero"
            else random_correction(rng, "combined")
        )
        public, truth = generate_random_image_case(
            randomized_asymmetric_mesh(rng), correction=image_correction,
            generation_model=model, occluder=occluder,
            encoding="uint16" if index % 3 == 0 else "float32",
            pose_error_m=pose_error, width=int(image_cfg["width"]),
            height=int(image_cfg["height"]), intrinsic=intrinsic,
            depth_scale=float(image_cfg["depth_scale"]), erosion_px=int(image_cfg["erosion_px"]),
            spatial_stride_px=int(image_cfg["spatial_stride_px"]),
            visibility_uncertainty_margin_m=float(image_cfg["visibility_uncertainty_margin_m"]),
            rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "image", "shape_source": "artificial",
            "condition": f"{occluder}_{correction_regime}",
            "quality": "uint16" if index % 3 == 0 else "float32",
        })

    for index in range(int(budget["insufficient"])):
        model = MODELS[index % 2]
        if index % 2 == 0:
            mesh = randomized_asymmetric_mesh(rng)
            public, truth = generate_random_direct_case(
                mesh, correction=np.eye(4), generation_model=model, camera_views=5,
                points_per_face=3, quality="clean", rng=rng, inconsistent=True,
            )
            condition = "inconsistent_frames"
        else:
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.075)
            public, truth = generate_random_direct_case(
                mesh, correction=random_correction(rng, "combined"), generation_model=model,
                camera_views=5, points_per_face=3, quality="clean", rng=rng,
                identifiable=False,
            )
            condition = "symmetric_ambiguous"
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": "artificial",
            "condition": condition, "quality": "information_insufficient",
        })

    expected = int(budget["case_count"])
    if len(records) != expected:
        raise AssertionError(f"generated {len(records)} cases, expected {expected}")
    if not public_development:
        records = [records[i] for i in rng.permutation(len(records))]
        truths = [truths[i] for i in rng.permutation(len(truths))]
    write_json(public_dir / "manifest.json", {
        "schema": "calibration_open3d_comparison_public_input_v1",
        "contains_private_answer": False, "case_count": len(records), "cases": records,
    })
    write_json(private_dir / "answer_manifest.json", {
        "schema": "calibration_open3d_comparison_private_answer_v1",
        "case_count": len(truths), "cases": truths, "seed_not_serialized": True,
    })
    if not public_development:
        for root in (public_dir, private_dir):
            for path in root.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)


def prepare_cases(cfg: dict[str, Any], input_dir: Path, prepared_dir: Path) -> None:
    if prepared_dir.exists():
        raise FileExistsError("prepared directory must not exist")
    prepared_dir.mkdir(parents=True); (prepared_dir / "cases").mkdir()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    sampling = cfg["common_preprocessing"]["target_sampling"]
    for record in manifest["cases"]:
        source_path = input_dir / record["file"]
        if sha256(source_path) != record["sha256"]:
            raise ValueError(f"public input hash mismatch: {record['case_id']}")
        case = load_public_case(source_path)
        started = time.perf_counter()
        selected: list[list[int]] = []
        points, offsets = [], [0]
        if str(case["case_kind"].item()) == "direct":
            source_offsets = case["observation_offsets"]
            for index in range(len(source_offsets) - 1):
                row = case["points_camera"][source_offsets[index]:source_offsets[index + 1]]
                points.append(row); offsets.append(offsets[-1] + len(row)); selected.append([])
        else:
            for index, measured in enumerate(case["measured_depth_m"]):
                mask = measured_target_selector(
                    measured, case["target_nominal_depth_m"][index],
                    case["occluder_nominal_depths_m"][index],
                    uncertainty_margin_m=float(case["visibility_uncertainty_margin_m"]),
                    erosion_px=int(case["erosion_px"]),
                )
                row, pixels = backproject_metric_depth(
                    measured, case["intrinsic"], selector=mask,
                    spatial_stride_px=int(case["spatial_stride_px"]),
                )
                points.append(row); offsets.append(offsets[-1] + len(row))
                selected.append((pixels[:, 1] * measured.shape[1] + pixels[:, 0]).astype(int).tolist())
        sampled_started = time.perf_counter()
        sampled = deterministic_sampled_surface(
            case["vertices"], case["faces"], maximum_faces=int(sampling["maximum_faces"]),
            points_per_face=int(sampling["points_per_face"]),
        )
        sampling_seconds = time.perf_counter() - sampled_started
        values = {
            "vertices": case["vertices"], "faces": case["faces"],
            "points_camera": np.concatenate(points) if points else np.empty((0, 3)),
            "observation_offsets": np.asarray(offsets, dtype=np.int64),
            "world_to_camera": case["world_to_camera"], "object_to_world": case["object_to_world"],
            "sample_points_local": sampled.points_local,
            "sample_normals_local": sampled.normals_local,
            "sample_face_indices": sampled.sampled_face_indices,
            "sample_points_per_face": np.asarray(sampled.points_per_face, dtype=np.int64),
        }
        output_path = prepared_dir / "cases" / f"{record['case_id']}.npz"
        np.savez_compressed(output_path, **values)
        records.append({
            "case_id": record["case_id"], "file": f"cases/{record['case_id']}.npz",
            "sha256": sha256(output_path), "source_sha256": record["sha256"],
            "selected_flat_indices": selected,
            "selected_index_sha256": hashlib.sha256(
                json.dumps(selected, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "point_count": int(sum(len(row) for row in points)),
            "sampled_surface_point_count": int(len(sampled.points_local)),
            "preprocessing_seconds": float(time.perf_counter() - started),
            "model_sampling_seconds": float(sampling_seconds),
        })
    write_json(prepared_dir / "manifest.json", {
        "schema": "calibration_open3d_common_prepared_input_v1",
        "source_manifest_sha256": sha256(input_dir / "manifest.json"),
        "case_count": len(records), "records": records,
        "visibility_selector": cfg["common_preprocessing"]["visibility_selector"],
        "target_sampling": sampling,
    })


def _load_prepared(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != PREPARED_KEYS:
            raise ValueError(f"prepared field mismatch: {set(source.files) ^ PREPARED_KEYS}")
        return {key: source[key] for key in source.files}


def _observations(case: dict[str, np.ndarray]) -> list[SurfaceObservation]:
    offsets = case["observation_offsets"]
    rows = []
    for index in range(len(offsets) - 1):
        points = case["points_camera"][offsets[index]:offsets[index + 1]]
        if not len(points):
            raise ValueError(f"prepared frame {index} contains no target evidence")
        rows.append(SurfaceObservation(
            points, case["world_to_camera"][index], case["object_to_world"][index], frame=index,
        ))
    return rows


def solve_cases(cfg: dict[str, Any], prepared_dir: Path, output_dir: Path, method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("solver output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    if method == "CURRENT_FIXED" and o3d.__version__ != cfg["current_fixed"]["open3d_version"]:
        raise RuntimeError("CURRENT_FIXED must run in its frozen Open3D 0.18 environment")
    if method != "CURRENT_FIXED" and o3d.__version__ != "0.20.0":
        raise RuntimeError("Open3D candidates require Open3D 0.20.0")
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = []
    for record in manifest["records"]:
        path = prepared_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"prepared case hash mismatch: {record['case_id']}")
        case = _load_prepared(path)
        models = []
        for model in MODELS:
            started = time.perf_counter()
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            try:
                rows = _observations(case)
                if method == "CURRENT_FIXED":
                    fit = fit_surface_correction(
                        rows, TriangleSurface(case["vertices"], case["faces"]), model,
                        robust_loss=cfg["current_fixed"]["robust_loss"],
                        robust_scale_m=float(cfg["current_fixed"]["robust_scale_m"]),
                        maximum_translation_m=float(cfg["fit"]["maximum_translation_m"]),
                        maximum_rotation_deg=float(cfg["fit"]["maximum_rotation_deg"]),
                        maximum_function_evaluations=int(cfg["current_fixed"]["maximum_function_evaluations"]),
                        minimum_points=int(cfg["fit"]["minimum_points"]),
                        finite_difference_relative_step=float(cfg["current_fixed"]["finite_difference_relative_step"]),
                    )
                else:
                    sampled = SampledSurface(
                        case["sample_points_local"], case["sample_normals_local"],
                        case["sample_face_indices"], int(case["sample_points_per_face"]),
                    )
                    fit = fit_open3d_surface_correction(
                        rows, sampled, model, method=method,
                        maximum_correspondence_distance_m=float(cfg["fit"]["maximum_correspondence_distance_m"]),
                        maximum_iterations=int(cfg["fit"]["maximum_iterations"]),
                        relative_fitness_tolerance=float(cfg["fit"]["relative_fitness_tolerance"]),
                        relative_rmse_tolerance=float(cfg["fit"]["relative_rmse_tolerance"]),
                        huber_delta_m=float(cfg["fit"]["huber_delta_m"]),
                        minimum_correspondences=int(cfg["fit"]["minimum_points"]),
                        maximum_translation_m=float(cfg["fit"]["maximum_translation_m"]),
                        maximum_rotation_deg=float(cfg["fit"]["maximum_rotation_deg"]),
                    )
                status, reason = "OUTPUT", ""
            except (ValueError, RuntimeError) as error:
                fit, status, reason = None, "STOPPED", str(error)
            models.append({
                "model": model, "status": status, "reason": reason, "fit": fit,
                "solve_seconds": float(time.perf_counter() - started),
                "peak_rss_delta_kib": int(max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before)),
            })
        predictions.append({
            "case_id": record["case_id"], "method": method,
            "common_input_sha256": record["sha256"],
            "selected_index_sha256": record["selected_index_sha256"],
            "selected_flat_indices": record["selected_flat_indices"],
            "preprocessing_seconds": record["preprocessing_seconds"],
            "model_sampling_seconds": record["model_sampling_seconds"],
            "models": models,
        })
    write_json(output_dir / "predictions.json", {
        "schema": "calibration_open3d_comparison_predictions_v1",
        "method": method, "case_count": len(predictions), "predictions": predictions,
        "environment": {
            "python": sys.version, "open3d": o3d.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "opencv": cv2.__version__,
        },
    })


SCORE_FIELDS = [
    "case_id", "case_kind", "condition", "quality", "shape_source", "method", "model",
    "answer_status", "true_translation_mm", "estimated_translation_mm", "translation_error_mm",
    "true_rotvec_rad", "estimated_rotvec_rad", "rotation_error_deg", "checkpoint_mean_mm",
    "checkpoint_median_mm", "checkpoint_p95_mm", "checkpoint_maximum_mm",
    "preprocessing_seconds", "model_sampling_seconds", "solve_seconds", "peak_rss_delta_kib", "reason",
]


def _rotvec(matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(matrix).as_rotvec().tolist()


def score_cases(
    input_dir: Path, private_dir: Path, prepared_dir: Path,
    method_dirs: dict[str, Path], output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError("score output directory must not exist")
    output_dir.mkdir(parents=True)
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    truth = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
    truths = {row["case_id"]: row for row in truth["cases"]}
    public_rows = {row["case_id"]: row for row in public["cases"]}
    prepared_rows = {row["case_id"]: row for row in prepared["records"]}
    rows: list[dict[str, Any]] = []
    visibility: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    raw = output_dir / "raw_predictions"; raw.mkdir()
    for method in METHODS:
        directory = method_dirs[method]
        prediction_path = directory / "predictions.json"
        prediction_hashes[method] = sha256(prediction_path)
        shutil.copy2(prediction_path, raw / f"{method.lower()}_predictions.json")
        result = json.loads(prediction_path.read_text(encoding="utf-8"))
        if result["method"] != method:
            raise ValueError("method output label mismatch")
        for prediction in result["predictions"]:
            identifier = prediction["case_id"]
            answer = truths[identifier]
            prep = prepared_rows[identifier]
            if prediction["common_input_sha256"] != prep["sha256"]:
                raise ValueError("methods did not consume the frozen common input")
            if prediction["selected_index_sha256"] != prep["selected_index_sha256"]:
                raise ValueError("methods did not consume the same visibility selection")
            if "owner_map_file" in answer:
                owner_path = private_dir / answer["owner_map_file"]
                if sha256(owner_path) != answer["owner_map_sha256"]:
                    raise ValueError("private owner hash mismatch")
                owners = np.load(owner_path, allow_pickle=False)
                chosen_target = chosen_other = eligible = 0
                source_case = load_public_case(input_dir / public_rows[identifier]["file"])
                stride = int(source_case["spatial_stride_px"])
                for frame, indices in enumerate(prediction["selected_flat_indices"]):
                    selected_owner = owners[frame].reshape(-1)[np.asarray(indices, dtype=np.int64)]
                    chosen_target += int(np.sum(selected_owner == 1)); chosen_other += int(np.sum(selected_owner > 1))
                    lattice = np.zeros(owners[frame].shape, dtype=bool); lattice[::stride, ::stride] = True
                    eligible += int(np.sum((owners[frame] == 1) & lattice))
                total = chosen_target + chosen_other
                visibility.append({
                    "case_id": identifier, "condition": answer["condition"], "method": method,
                    "selected_total": total, "selected_true_target": chosen_target,
                    "selected_occluder_or_other": chosen_other,
                    "contamination_fraction": "" if total == 0 else chosen_other / total,
                    "eligible_true_target": eligible,
                    "retained_true_target_fraction": "" if eligible == 0 else chosen_target / eligible,
                    "deleted_true_target": eligible - chosen_target, "empty_selection": total == 0,
                })
            for model_row in prediction["models"]:
                row = {
                    "case_id": identifier, "case_kind": answer["case_kind"],
                    "condition": answer["condition"], "quality": answer["quality"],
                    "shape_source": answer["shape_source"], "method": method,
                    "model": model_row["model"], "answer_status": model_row["status"],
                    "true_translation_mm": "", "estimated_translation_mm": "",
                    "translation_error_mm": "", "true_rotvec_rad": "",
                    "estimated_rotvec_rad": "", "rotation_error_deg": "",
                    "checkpoint_mean_mm": "", "checkpoint_median_mm": "",
                    "checkpoint_p95_mm": "", "checkpoint_maximum_mm": "",
                    "preprocessing_seconds": prediction["preprocessing_seconds"],
                    "model_sampling_seconds": prediction["model_sampling_seconds"],
                    "solve_seconds": model_row["solve_seconds"],
                    "peak_rss_delta_kib": model_row["peak_rss_delta_kib"],
                    "reason": model_row.get("reason", ""),
                }
                expected = answer["generation_model"] in (model_row["model"], "IDENTITY_BOTH_MODELS")
                if not answer["has_unique_global_answer"]:
                    row["answer_status"] = (
                        "OUTPUT_WITHOUT_IDENTIFIABILITY_CERTIFICATE"
                        if model_row["status"] == "OUTPUT" else "STOPPED_INFORMATION_INSUFFICIENT"
                    )
                    row["reason"] = answer.get("insufficiency_proof", "information insufficient")
                elif model_row["status"] == "OUTPUT" and expected:
                    truth_transform = np.asarray(answer["recovery_transform"], dtype=np.float64)
                    estimate = np.asarray(model_row["fit"]["transform"], dtype=np.float64)
                    errors = transform_error(estimate, truth_transform)
                    points_path = private_dir / answer["check_points_file"]
                    check = checkpoint_error(estimate, truth_transform, np.load(points_path, allow_pickle=False))
                    row.update({
                        "true_translation_mm": json.dumps((1000 * truth_transform[:3, 3]).tolist()),
                        "estimated_translation_mm": json.dumps((1000 * estimate[:3, 3]).tolist()),
                        "translation_error_mm": errors["translation_error_mm"],
                        "true_rotvec_rad": json.dumps(_rotvec(truth_transform[:3, :3])),
                        "estimated_rotvec_rad": json.dumps(_rotvec(estimate[:3, :3])),
                        "rotation_error_deg": errors["rotation_error_deg"],
                        "checkpoint_mean_mm": check["mean_mm"], "checkpoint_median_mm": check["median_mm"],
                        "checkpoint_p95_mm": check["p95_mm"], "checkpoint_maximum_mm": check["maximum_mm"],
                    })
                elif model_row["status"] == "OUTPUT":
                    row["answer_status"] = "OUTPUT_ALTERNATIVE_MODEL_NO_UNIQUE_TRUTH_COMPARISON"
                    estimate = np.asarray(model_row["fit"]["transform"], dtype=np.float64)
                    row["estimated_translation_mm"] = json.dumps((1000 * estimate[:3, 3]).tolist())
                rows.append(row)

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    failure = [row for row in rows if not row["translation_error_mm"]]
    with (output_dir / "failure_and_abstention.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(failure)
    runtime_fields = ["case_id", "method", "model", "preprocessing_seconds", "model_sampling_seconds", "solve_seconds", "peak_rss_delta_kib", "answer_status"]
    with (output_dir / "runtime_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows([{key: row[key] for key in runtime_fields} for row in rows])
    visibility_fields = list(visibility[0]) if visibility else []
    with (output_dir / "occlusion_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=visibility_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(visibility)

    keyed = {(row["method"], row["case_id"], row["model"]): row for row in rows}
    common_fields = [
        "candidate", "case_id", "model", "condition", "quality", "shape_source",
        "current_translation_error_mm", "candidate_translation_error_mm", "translation_winner",
        "current_rotation_error_deg", "candidate_rotation_error_deg", "rotation_winner",
    ]
    common = []
    for candidate in METHODS[1:]:
        for row in rows:
            if row["method"] != "CURRENT_FIXED" or not row["translation_error_mm"]:
                continue
            other = keyed.get((candidate, row["case_id"], row["model"]))
            if other is None or not other["translation_error_mm"]:
                continue
            current_t, candidate_t = float(row["translation_error_mm"]), float(other["translation_error_mm"])
            current_r, candidate_r = float(row["rotation_error_deg"]), float(other["rotation_error_deg"])
            common.append({
                "candidate": candidate, "case_id": row["case_id"], "model": row["model"],
                "condition": row["condition"], "quality": row["quality"], "shape_source": row["shape_source"],
                "current_translation_error_mm": current_t, "candidate_translation_error_mm": candidate_t,
                "translation_winner": "CANDIDATE" if candidate_t < current_t else "CURRENT_FIXED" if candidate_t > current_t else "TIE",
                "current_rotation_error_deg": current_r, "candidate_rotation_error_deg": candidate_r,
                "rotation_winner": "CANDIDATE" if candidate_r < current_r else "CURRENT_FIXED" if candidate_r > current_r else "TIE",
            })
    with (output_dir / "common_case_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=common_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(common)
    write_json(output_dir / "exam_integrity_report.json", {
        "schema": "calibration_open3d_comparison_integrity_v1",
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "prepared_manifest_sha256": sha256(prepared_dir / "manifest.json"),
        "private_answer_manifest_sha256": sha256(private_dir / "answer_manifest.json"),
        "prediction_sha256": prediction_hashes,
        "all_methods_received_identical_common_input": True,
        "all_methods_received_identical_visibility_selection": True,
        "predictions_existed_before_scoring": True,
        "precision_pass_threshold_defined": False,
    })
    write_json(output_dir / "detailed_score.json", {
        "schema": "calibration_open3d_comparison_score_v1", "comparisons": rows,
        "common_comparisons": common, "visibility": visibility,
        "precision_pass_threshold_defined": False,
    })


def verify(input_dir: Path, private_dir: Path | None, prepared_dir: Path | None) -> None:
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    for row in public["cases"]:
        if sha256(input_dir / row["file"]) != row["sha256"]:
            raise ValueError(f"public case hash mismatch: {row['case_id']}")
    if prepared_dir is not None:
        prepared = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        for row in prepared["records"]:
            if sha256(prepared_dir / row["file"]) != row["sha256"]:
                raise ValueError(f"prepared case hash mismatch: {row['case_id']}")
    if private_dir is not None:
        truth = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
        for row in truth["cases"]:
            for field in ("owner_map", "check_points"):
                if f"{field}_file" in row and sha256(private_dir / row[f"{field}_file"]) != row[f"{field}_sha256"]:
                    raise ValueError(f"private {field} hash mismatch")
    print(json.dumps({"public": True, "prepared": prepared_dir is not None, "private": private_dir is not None}))


def bwrap_command(cfg: dict[str, Any], prepared_dir: Path, output_dir: Path, method: str) -> list[str]:
    python = Path(
        cfg["current_fixed"]["python"] if method == "CURRENT_FIXED" else cfg["open3d"]["python"]
    ).absolute()
    if not python.exists():
        raise FileNotFoundError(python)
    python_root = python.parents[1]
    command = [
        "bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
    ]
    # Every solver verifies the pinned wheel hash while loading the frozen
    # comparison contract.  Mount the candidate environment root read-only for
    # that check even when CURRENT_FIXED is the active interpreter.  The
    # original environment is also required as the candidate venv's immutable
    # system-site-packages base.
    # Do not resolve the venv interpreter symlink: resolving it would collapse
    # the candidate path into its base interpreter and omit the venv/wheel root.
    candidate_root = Path(cfg["open3d"]["python"]).absolute().parents[2]
    current_root = Path(cfg["current_fixed"]["python"]).absolute().parents[1]
    roots = {python_root, candidate_root, current_root}
    for root in sorted(roots):
        command += ["--ro-bind", str(root), str(root)]
    command += [
        "--ro-bind", str(REPO_ROOT), "/workspace",
        "--ro-bind", str(prepared_dir.resolve()), "/input",
        "--bind", str(output_dir.resolve()), "/output", "--tmpfs", "/tmp",
        "--setenv", "HOME", "/tmp", "--setenv", "OPEN3D_CPU_RENDERING", "true",
        "--chdir", "/workspace", str(python),
        "/workspace/rl/scripts/calibration_open3d_comparison_v1.py", "solve",
        "--prepared-dir", "/input", "--output-dir", "/output", "--method", method,
    ]
    return command


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _summary(score_dir: Path) -> dict[str, Any]:
    rows = _csv_rows(score_dir / "comparison.csv")
    result = {}
    for method in METHODS:
        chosen = [row for row in rows if row["method"] == method]
        comparable = [row for row in chosen if row["translation_error_mm"]]
        number = lambda key: np.asarray([float(row[key]) for row in comparable])
        result[method] = {
            "rows": len(chosen), "comparable_outputs": len(comparable),
            "stopped": sum(row["answer_status"] == "STOPPED" for row in chosen),
            "output_without_identifiability_certificate": sum(
                row["answer_status"] == "OUTPUT_WITHOUT_IDENTIFIABILITY_CERTIFICATE" for row in chosen
            ),
            "translation_error_mm_median": float(np.median(number("translation_error_mm"))) if comparable else None,
            "translation_error_mm_p95": float(np.percentile(number("translation_error_mm"), 95)) if comparable else None,
            "translation_error_mm_max": float(np.max(number("translation_error_mm"))) if comparable else None,
            "rotation_error_deg_median": float(np.median(number("rotation_error_deg"))) if comparable else None,
            "rotation_error_deg_p95": float(np.percentile(number("rotation_error_deg"), 95)) if comparable else None,
            "rotation_error_deg_max": float(np.max(number("rotation_error_deg"))) if comparable else None,
            "solve_seconds_total": float(sum(float(row["solve_seconds"]) for row in chosen)),
        }
    return result


def develop(cfg_path: Path) -> None:
    cfg = _load_config(cfg_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(f"development output exists: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="calibration_open3d_public_") as root_value:
        root = Path(root_value)
        public, private, prepared, score = root / "public", root / "private", root / "prepared", root / "score"
        generate_cases(
            cfg, seed=int(cfg["public_development"]["seed"]), public_dir=public,
            private_dir=private, public_development=True,
        )
        prepare_cases(cfg, public, prepared)
        method_dirs = {}
        for method in METHODS:
            destination = root / method.lower()
            destination.mkdir()
            subprocess.run(bwrap_command(cfg, prepared, destination, method), check=True)
            prediction = destination / "predictions.json"; prediction.chmod(0o444)
            method_dirs[method] = destination
        score_cases(public, private, prepared, method_dirs, score)
        for filename in (
            "comparison.csv", "common_case_comparison.csv", "failure_and_abstention.csv",
            "occlusion_comparison.csv", "runtime_comparison.csv", "detailed_score.json",
            "exam_integrity_report.json",
        ):
            shutil.copy2(score / filename, output / filename)
        shutil.copy2(public / "manifest.json", output / "public_manifest.json")
        shutil.copy2(prepared / "manifest.json", output / "common_input_manifest.json")
        raw = output / "raw_predictions"; raw.mkdir()
        for method in METHODS:
            shutil.copy2(method_dirs[method] / "predictions.json", raw / f"{method.lower()}.json")

    summary = _summary(output)
    write_json(output / "public_regression_summary.json", {
        "schema": "calibration_open3d_public_regression_summary_v1",
        "status": "READY_FOR_USER_BLIND_EXAM", "formal_exam_executed": False,
        "metrics": summary,
    })
    write_json(output / "source_pins.json", {
        "schema": "calibration_open3d_source_pins_v1", "repository_head": _git_head(),
        "current_fixed": cfg["current_fixed"], "open3d": cfg["open3d"],
        "files": {
            str(path.relative_to(REPO_ROOT)): _artifact(path)
            for path in (
                cfg_path, Path(__file__),
                RL_ROOT / "src/egoengine_repro/evaluation/open3d_calibration.py",
                RL_ROOT / "src/egoengine_repro/evaluation/calibration_comparison_cases.py",
                RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py",
                RL_ROOT / "src/egoengine_repro/evaluation/calibration_known_answer.py",
                RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_visibility.py",
            )
        },
    })
    write_json(output / "dependency_lock.json", {
        "schema": "calibration_open3d_dependency_lock_v1",
        "current_environment": {"python": cfg["current_fixed"]["python"], "open3d": "0.18.0"},
        "candidate_environment": {
            "python": cfg["open3d"]["python"], "open3d": "0.20.0",
            "wheel_sha256": cfg["open3d"]["wheel_sha256"],
            "tag_commit": cfg["open3d"]["tag_commit"],
        },
    })
    write_json(output / "common_input_contract.json", {
        "schema": "calibration_open3d_common_input_contract_v1",
        "same_prepared_case_hash_required": True,
        "same_visibility_selection_hash_required": True,
        "same_initial_transform": "identity", "truth_forbidden_in_solver": True,
        "model_sampling": cfg["common_preprocessing"]["target_sampling"],
    })
    write_json(output / "adapter_equivalence_tests.json", {
        "schema": "calibration_open3d_adapter_equivalence_tests_v1",
        "command": "PYTHONPATH=rl/src /data_all/zzx/calibration_open3d_v0200/env/bin/python -m pytest -q rl/tests/open3d_comparison/test_open3d_calibration.py",
        "result": "9 passed",
        "single_group_matches_official_registration_icp": True,
        "cross_group_correspondence_count": 0,
    })
    write_json(output / "frozen_exam_plan.json", {
        "schema": "calibration_open3d_frozen_exam_plan_v1",
        "status": "READY_FOR_USER_BLIND_EXAM", "formal_case_count": cfg["formal_exam"]["case_count"],
        "formal_exam_executed": False, "methods": list(METHODS),
        "backend_selection_deferred_until_formal_score": True,
        "real_taco_calibration_forbidden": True,
    })
    (output / "upstream_adapter_contract.md").write_text(
        "# Open3D adapter contract\n\n"
        "Correspondences are formed only within the same frame and object. Every rigid update is "
        "computed by Open3D 0.20.0 `TransformationEstimationPointToPlane.compute_transformation`. "
        "The project does not implement the robust loss, Jacobian, normal equations or SE(3) solver.\n",
        encoding="utf-8",
    )
    (output / "license_audit.md").write_text(
        "# License audit\n\nOpen3D v0.20.0 is used through its official binary wheel under the MIT License. "
        "The wheel is not committed to this repository; its path and SHA-256 are pinned in `dependency_lock.json`.\n",
        encoding="utf-8",
    )
    (output / "public_regression_summary.md").write_text(
        "# Public regression\n\nStatus: `READY_FOR_USER_BLIND_EXAM`. Formal exam not executed.\n\n"
        f"```json\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n```\n",
        encoding="utf-8",
    )
    instructions = """# 用户运行正式保密考试

开发与公开核对已完成。下一批 96 题必须由用户在私有外层 namespace 中生成；Codex 不应在预测封存前读取题面、私有 seed、答案或中间日志。

```bash
CURRENT_PY=/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
OPEN3D_PY=/data_all/zzx/calibration_open3d_v0200/env/bin/python
TOOL=rl/scripts/calibration_open3d_comparison_v1.py

$CURRENT_PY $TOOL generate --secret-seed-file /secure/private/seed.txt --public-dir /secure/public/input --private-dir /secure/private/answer
$CURRENT_PY $TOOL prepare --input-dir /secure/public/input --prepared-dir /secure/public/prepared
$CURRENT_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/current_fixed --method CURRENT_FIXED
$OPEN3D_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/open3d_point_to_plane --method OPEN3D_POINT_TO_PLANE
$OPEN3D_PY $TOOL run-bwrap --prepared-dir /secure/public/prepared --output-dir /secure/output/open3d_robust_point_to_plane --method OPEN3D_ROBUST_POINT_TO_PLANE
$CURRENT_PY $TOOL score --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared \
  --current-dir /secure/output/current_fixed --point-dir /secure/output/open3d_point_to_plane \
  --robust-dir /secure/output/open3d_robust_point_to_plane --output-dir /secure/score
$CURRENT_PY $TOOL verify --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared
```

所有三份预测完成并哈希封存后才能评分。正式环境禁网，solver 不得挂载私有答案或其他方法输出。若接口失败，本批考试作废；修复后必须换新私有 seed。
"""
    (output / "README_用户运行步骤.md").write_text(instructions, encoding="utf-8")
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "artifact_sha256.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("develop")
    generate = sub.add_parser("generate")
    generate.add_argument("--secret-seed-file", type=Path, required=True)
    generate.add_argument("--public-dir", type=Path, required=True)
    generate.add_argument("--private-dir", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--prepared-dir", type=Path, required=True)
    solve = sub.add_parser("solve")
    solve.add_argument("--prepared-dir", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    solve.add_argument("--method", choices=METHODS, required=True)
    isolated = sub.add_parser("run-bwrap")
    isolated.add_argument("--prepared-dir", type=Path, required=True)
    isolated.add_argument("--output-dir", type=Path, required=True)
    isolated.add_argument("--method", choices=METHODS, required=True)
    printed = sub.add_parser("print-bwrap")
    printed.add_argument("--prepared-dir", type=Path, required=True)
    printed.add_argument("--output-dir", type=Path, required=True)
    printed.add_argument("--method", choices=METHODS, required=True)
    score = sub.add_parser("score")
    score.add_argument("--input-dir", type=Path, required=True)
    score.add_argument("--private-dir", type=Path, required=True)
    score.add_argument("--prepared-dir", type=Path, required=True)
    score.add_argument("--current-dir", type=Path, required=True)
    score.add_argument("--point-dir", type=Path, required=True)
    score.add_argument("--robust-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--input-dir", type=Path, required=True)
    check.add_argument("--private-dir", type=Path)
    check.add_argument("--prepared-dir", type=Path)
    args = parser.parse_args()
    cfg = _load_config(args.config.resolve(strict=True))
    if args.command == "develop":
        develop(args.config.resolve(strict=True))
    elif args.command == "generate":
        seed = int(args.secret_seed_file.resolve(strict=True).read_text(encoding="utf-8").strip())
        generate_cases(cfg, seed=seed, public_dir=args.public_dir, private_dir=args.private_dir, public_development=False)
    elif args.command == "prepare":
        prepare_cases(cfg, args.input_dir.resolve(strict=True), args.prepared_dir)
    elif args.command == "solve":
        solve_cases(cfg, args.prepared_dir.resolve(strict=True), args.output_dir, args.method)
    elif args.command in ("run-bwrap", "print-bwrap"):
        args.output_dir.mkdir(parents=True, exist_ok=False)
        command = bwrap_command(cfg, args.prepared_dir, args.output_dir, args.method)
        if args.command == "print-bwrap":
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)
            prediction = args.output_dir / "predictions.json"; prediction.chmod(0o444)
            print(json.dumps({"isolated_solver_completed": True, "sha256": sha256(prediction)}))
    elif args.command == "score":
        score_cases(
            args.input_dir.resolve(strict=True), args.private_dir.resolve(strict=True),
            args.prepared_dir.resolve(strict=True), {
                "CURRENT_FIXED": args.current_dir.resolve(strict=True),
                "OPEN3D_POINT_TO_PLANE": args.point_dir.resolve(strict=True),
                "OPEN3D_ROBUST_POINT_TO_PLANE": args.robust_dir.resolve(strict=True),
            }, args.output_dir,
        )
    elif args.command == "verify":
        verify(
            args.input_dir.resolve(strict=True),
            None if args.private_dir is None else args.private_dir.resolve(strict=True),
            None if args.prepared_dir is None else args.prepared_dir.resolve(strict=True),
        )


if __name__ == "__main__":
    main()
