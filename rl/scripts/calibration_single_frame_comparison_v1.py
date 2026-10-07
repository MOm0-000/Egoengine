#!/usr/bin/env python3
"""Single-frame known-answer comparison using official Open3D ICP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.calibration_known_answer import (  # noqa: E402
    checkpoint_error,
    deterministic_surface_points,
    independent_apply,
    load_public_case,
    sha256,
    transform_error,
    write_json,
    write_public_case,
)
from egoengine_repro.evaluation.calibration_single_frame_cases import (  # noqa: E402
    TRUTH_MULTIPLE,
    TRUTH_NO_TARGET,
    TRUTH_UNIQUE,
    generate_direct_sequence,
    generate_image_sequence,
    randomized_asymmetric_mesh,
)
from egoengine_repro.evaluation.open3d_calibration import (  # noqa: E402
    OfficialSampledSurface,
    fit_open3d_single_frame,
    official_uniform_sampled_surface,
)
from egoengine_repro.evaluation.taco_calibration_residual import (  # noqa: E402
    CalibrationCandidateRejected, SurfaceObservation,
    TriangleSurface,
    fit_single_surface_correction,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (  # noqa: E402
    measured_target_selector,
)
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    backproject_metric_depth,
)


DEFAULT_CONFIG = RL_ROOT / "configs/calibration_single_frame_comparison_v1.yaml"
METHODS = (
    "CURRENT_FIXED_SINGLE_FRAME",
    "OPEN3D_OFFICIAL_SINGLE_FRAME",
    "OPEN3D_HUBER_SINGLE_FRAME",
)
MODELS = ("WORLD_FIXED", "CAMERA_LOCAL")
FRAME_KEYS = {
    "points_camera", "world_to_camera", "object_to_world", "case_kind",
}
MODEL_KEYS = {
    "vertices", "faces", "sample_points_local", "sample_normals_local",
    "sample_point_count", "sample_seed", "sample_sha256",
}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["schema"] != "calibration_single_frame_comparison_v1":
        raise ValueError("unexpected single-frame config schema")
    if cfg["status"] != "PUBLIC_DEVELOPMENT_ONLY" or cfg["formal_exam_executed"] is not False:
        raise ValueError("repository config must not claim that the formal exam ran")
    forbidden = (
        "formal_secret_generation_by_codex", "real_taco_refit",
        "real_taco_source_mutation", "physics", "replay",
        "reinforcement_learning", "chunk_commit",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden authorization is enabled")
    if tuple(cfg["methods"]) != METHODS:
        raise ValueError("the three frozen methods changed")
    current = cfg["current_fixed_single_frame"]
    residual = RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py"
    known = RL_ROOT / "src/egoengine_repro/evaluation/calibration_known_answer.py"
    if sha256(residual) != current["residual_module_sha256"]:
        raise ValueError("single-frame current residual implementation changed")
    if sha256(known) != current["known_answer_module_sha256"]:
        raise ValueError("known-answer implementation changed")
    wheel = Path(cfg["open3d"]["wheel"])
    if sha256(wheel) != cfg["open3d"]["wheel_sha256"]:
        raise ValueError("Open3D wheel hash mismatch")
    return cfg


def _load_taco_shape(cfg: dict[str, Any]) -> trimesh.Trimesh:
    contract = cfg["shapes"]["taco"]
    mesh = trimesh.load_mesh(Path(contract["path"]).resolve(strict=True), process=False)
    mesh.apply_scale(float(contract["scale"]))
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("TACO shape is not a supported closed surface")
    return mesh


def _case_id(index: int, public_development: bool) -> str:
    return f"public_sf_{index:03d}" if public_development else uuid.uuid4().hex


def _budget_count(budget: dict[str, Any]) -> int:
    return sum(int(budget[key]["groups"]) for key in (
        "direct_clean", "direct_degraded", "image", "varying_recovery", "insufficient",
    ))


def generate_cases(
    cfg: dict[str, Any],
    *,
    seed: int,
    public_dir: Path,
    private_dir: Path,
    public_development: bool,
) -> None:
    if public_dir.exists() or private_dir.exists():
        raise FileExistsError("generation output directories must not exist")
    public_dir.mkdir(parents=True)
    (public_dir / "cases").mkdir()
    private_dir.mkdir(parents=True)
    (private_dir / "truth").mkdir()
    rng = np.random.default_rng(seed)
    budget = cfg["public_development"] if public_development else cfg["formal_exam"]
    if _budget_count(budget) != int(budget["group_count"]):
        raise ValueError("group budget does not add up")
    taco = _load_taco_shape(cfg)
    records: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []

    def save_case(
        public: dict[str, np.ndarray],
        truth: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        identifier = _case_id(len(records) + 1, public_development)
        path = public_dir / "cases" / f"{identifier}.npz"
        owner = truth.pop("owner_maps", None)
        write_public_case(path, public)
        frame_count = len(truth["frames"])
        disclosed = metadata if public_development else {
            "case_kind": metadata["case_kind"],
            "shape_source": metadata["shape_source"],
            "condition": "UNDISCLOSED",
            "quality": "UNDISCLOSED",
            "generation_model": "UNDISCLOSED",
        }
        records.append({
            "case_id": identifier,
            "file": f"cases/{identifier}.npz",
            "sha256": sha256(path),
            "frame_count": frame_count,
            **disclosed,
        })
        if owner is not None:
            owner_path = private_dir / "truth" / f"{identifier}_owner.npy"
            np.save(owner_path, owner, allow_pickle=False)
            truth["owner_map_file"] = f"truth/{identifier}_owner.npy"
            truth["owner_map_sha256"] = sha256(owner_path)
        truths.append({"case_id": identifier, **metadata, **truth})

    clean = budget["direct_clean"]
    kinds = ("identity", "translation", "rotation", "combined")
    for index in range(int(clean["groups"])):
        combination = index % 16
        shape_source = "taco" if combination % 2 else "artificial"
        generation_model = MODELS[(combination // 2) % 2]
        kind = kinds[(combination // 4) % 4]
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        public, truth = generate_direct_sequence(
            mesh,
            generation_model=generation_model,
            frame_count=int(clean["frames_per_group"]),
            correction_kind=kind,
            quality="clean",
            rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": shape_source,
            "condition": kind, "quality": "clean",
            "generation_model": generation_model,
        })

    degraded = budget["direct_degraded"]
    qualities = (
        "noise", "random_missing", "local_missing", "outliers",
        "combined", "strong_combined",
    )
    for index in range(int(degraded["groups"])):
        quality = qualities[index % 6]
        shape_source = "taco" if (index // 6) % 2 else "artificial"
        generation_model = MODELS[(index // 12) % 2]
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        public, truth = generate_direct_sequence(
            mesh,
            generation_model=generation_model,
            frame_count=int(degraded["frames_per_group"]),
            correction_kind="combined",
            quality=quality,
            rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": shape_source,
            "condition": "combined", "quality": quality,
            "generation_model": generation_model,
        })

    image = budget["image"]
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
    for index in range(int(image["groups"])):
        shape_source = "taco" if index % 2 else "artificial"
        generation_model = MODELS[(index // 2) % 2]
        occluder = occluders[index % len(occluders)]
        pose_error = 0.015 if index % 8 == 6 else 0.0
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        public, truth = generate_image_sequence(
            mesh,
            generation_model=generation_model,
            frame_count=int(image["frames_per_group"]),
            occluder=occluder,
            encoding="uint16" if index % 3 == 0 else "float32",
            pose_error_m=pose_error,
            width=int(image_cfg["width"]),
            height=int(image_cfg["height"]),
            intrinsic=intrinsic,
            depth_scale=float(image_cfg["depth_scale"]),
            erosion_px=int(image_cfg["erosion_px"]),
            spatial_stride_px=int(image_cfg["spatial_stride_px"]),
            visibility_uncertainty_margin_m=float(image_cfg["visibility_uncertainty_margin_m"]),
            rng=rng,
        )
        save_case(public, truth, {
            "case_kind": "image", "shape_source": shape_source,
            "condition": f"{occluder}_nonzero", "quality": truth["depth_encoding"],
            "generation_model": generation_model,
        })

    varying = budget["varying_recovery"]
    for index in range(int(varying["groups"])):
        shape_source = "taco" if index % 2 else "artificial"
        generation_model = MODELS[(index // 2) % 2]
        mesh = taco.copy() if shape_source == "taco" else randomized_asymmetric_mesh(rng)
        public, truth = generate_direct_sequence(
            mesh,
            generation_model=generation_model,
            frame_count=int(varying["frames_per_group"]),
            correction_kind="combined",
            quality="clean",
            rng=rng,
            varying_recovery=True,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": shape_source,
            "condition": "per_frame_varying", "quality": "clean",
            "generation_model": generation_model,
        })

    insufficient = budget["insufficient"]
    for index in range(int(insufficient["groups"])):
        generation_model = MODELS[(index // 2) % 2]
        empty = index % 2 == 1
        mesh = (
            randomized_asymmetric_mesh(rng)
            if empty else trimesh.creation.icosphere(subdivisions=2, radius=0.075)
        )
        public, truth = generate_direct_sequence(
            mesh,
            generation_model=generation_model,
            frame_count=int(insufficient["frames_per_group"]),
            correction_kind="combined",
            quality="clean",
            rng=rng,
            truth_status=TRUTH_NO_TARGET if empty else TRUTH_MULTIPLE,
            empty_frames=empty,
        )
        save_case(public, truth, {
            "case_kind": "direct", "shape_source": "artificial",
            "condition": "no_target" if empty else "symmetric_ambiguous",
            "quality": "information_insufficient",
            "generation_model": generation_model,
        })

    expected_groups = int(budget["group_count"])
    expected_frames = sum(row["frame_count"] for row in records)
    if len(records) != expected_groups:
        raise AssertionError(f"generated {len(records)} groups, expected {expected_groups}")
    if not public_development and expected_frames != int(budget["expected_frame_count"]):
        raise AssertionError(
            f"generated {expected_frames} frames, expected {budget['expected_frame_count']}"
        )
    if not public_development:
        order = rng.permutation(len(records))
        records = [records[index] for index in order]
        truth_by_id = {row["case_id"]: row for row in truths}
        truths = [truth_by_id[row["case_id"]] for row in records]
    write_json(public_dir / "manifest.json", {
        "schema": "calibration_single_frame_public_input_v1",
        "contains_private_answer": False,
        "group_count": len(records),
        "frame_count": expected_frames,
        "cases": records,
    })
    write_json(private_dir / "answer_manifest.json", {
        "schema": "calibration_single_frame_private_answer_v1",
        "group_count": len(truths),
        "frame_count": expected_frames,
        "cases": truths,
        "seed_not_serialized": True,
    })
    if not public_development:
        for root in (public_dir, private_dir):
            for path in root.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)


def _selected_hash(selected: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_cases(cfg: dict[str, Any], input_dir: Path, prepared_dir: Path) -> None:
    if prepared_dir.exists():
        raise FileExistsError("prepared directory must not exist")
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "models").mkdir()
    (prepared_dir / "frames").mkdir()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    sampler = cfg["open3d"]
    for case_record in manifest["cases"]:
        case_path = input_dir / case_record["file"]
        if sha256(case_path) != case_record["sha256"]:
            raise ValueError(f"public input hash mismatch: {case_record['case_id']}")
        case = load_public_case(case_path)
        sample_started = time.perf_counter()
        sampled = official_uniform_sampled_surface(
            case["vertices"],
            case["faces"],
            number_of_points=int(sampler["reference_point_count"]),
            random_seed=int(sampler["reference_sampling_seed"]),
        )
        sample_seconds = time.perf_counter() - sample_started
        model_path = prepared_dir / "models" / f"{case_record['case_id']}.npz"
        np.savez_compressed(
            model_path,
            vertices=case["vertices"],
            faces=case["faces"],
            sample_points_local=sampled.points_local,
            sample_normals_local=sampled.normals_local,
            sample_point_count=np.asarray(sampled.number_of_points, dtype=np.int64),
            sample_seed=np.asarray(sampled.random_seed, dtype=np.int64),
            sample_sha256=np.asarray(sampled.sha256),
        )
        model_hash = sha256(model_path)
        model_records.append({
            "case_id": case_record["case_id"],
            "file": f"models/{case_record['case_id']}.npz",
            "sha256": model_hash,
            "sample_sha256": sampled.sha256,
            "sample_point_count": sampled.number_of_points,
            "sampling_seconds": sample_seconds,
        })
        points: list[np.ndarray] = []
        selected_rows: list[list[int]] = []
        preprocessing_rows: list[float] = []
        if str(case["case_kind"].item()) == "direct":
            offsets = case["observation_offsets"]
            for frame_id in range(len(offsets) - 1):
                preprocessing_started = time.perf_counter()
                points.append(case["points_camera"][offsets[frame_id]:offsets[frame_id + 1]])
                selected_rows.append([])
                preprocessing_rows.append(time.perf_counter() - preprocessing_started)
        else:
            for frame_id, measured in enumerate(case["measured_depth_m"]):
                preprocessing_started = time.perf_counter()
                occluders = case["occluder_nominal_depths_m"][frame_id]
                forbidden = np.any(
                    np.isfinite(occluders) & (occluders > 0), axis=0,
                )
                mask = measured_target_selector(
                    measured,
                    case["target_nominal_depth_m"][frame_id],
                    occluders,
                    uncertainty_margin_m=float(case["visibility_uncertainty_margin_m"]),
                    erosion_px=int(case["erosion_px"]),
                    forbidden_mask=forbidden,
                )
                row, pixels = backproject_metric_depth(
                    measured,
                    case["intrinsic"],
                    selector=mask,
                    spatial_stride_px=int(case["spatial_stride_px"]),
                )
                points.append(row)
                selected_rows.append(
                    (pixels[:, 1] * measured.shape[1] + pixels[:, 0]).astype(int).tolist()
                )
                preprocessing_rows.append(time.perf_counter() - preprocessing_started)
        if len(points) != int(case_record["frame_count"]):
            raise ValueError("prepared frame count differs from public manifest")
        amortized_sampling = sample_seconds / max(len(points), 1)
        for frame_id, (point_row, selected, preprocessing_seconds) in enumerate(
            zip(points, selected_rows, preprocessing_rows, strict=True)
        ):
            started = time.perf_counter()
            task_id = f"{case_record['case_id']}_f{frame_id:03d}"
            frame_path = prepared_dir / "frames" / f"{task_id}.npz"
            np.savez_compressed(
                frame_path,
                points_camera=np.asarray(point_row, dtype=np.float64),
                world_to_camera=case["world_to_camera"][frame_id],
                object_to_world=case["object_to_world"][frame_id],
                case_kind=case["case_kind"],
            )
            records.append({
                "task_id": task_id,
                "case_id": case_record["case_id"],
                "frame_id": frame_id,
                "frame_file": f"frames/{task_id}.npz",
                "frame_sha256": sha256(frame_path),
                "model_file": f"models/{case_record['case_id']}.npz",
                "model_sha256": model_hash,
                "selected_flat_indices": selected,
                "selected_index_sha256": _selected_hash(selected),
                "point_count": int(len(point_row)),
                "preprocessing_seconds": float(
                    preprocessing_seconds + time.perf_counter() - started
                ),
                "open3d_reference_sampling_seconds_amortized": float(amortized_sampling),
            })
    write_json(prepared_dir / "manifest.json", {
        "schema": "calibration_single_frame_prepared_input_v1",
        "source_manifest_sha256": sha256(input_dir / "manifest.json"),
        "group_count": len(model_records),
        "frame_count": len(records),
        "records": records,
        "models": model_records,
        "single_frame_task_files": True,
        "visibility_selector": cfg["common_preprocessing"]["visibility_selector"],
        "reference_sampling": {
            "api": cfg["open3d"]["sampler_api"],
            "number_of_points": cfg["open3d"]["reference_point_count"],
            "use_triangle_normal": cfg["open3d"]["use_triangle_normal"],
            "seed": cfg["open3d"]["reference_sampling_seed"],
        },
    })


def _load_npz(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != expected:
            raise ValueError(f"prepared field mismatch: {set(source.files) ^ expected}")
        return {key: source[key] for key in source.files}


def solve_frame(
    cfg: dict[str, Any],
    frame_file: Path,
    model_file: Path,
    output_file: Path,
    method: str,
    metadata: dict[str, str | int | float],
) -> None:
    if output_file.exists():
        raise FileExistsError(output_file)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    current = method == "CURRENT_FIXED_SINGLE_FRAME"
    expected_version = cfg["current_fixed_single_frame"]["open3d_version"] if current else "0.20.0"
    if o3d.__version__ != expected_version:
        raise RuntimeError(f"{method} expected Open3D {expected_version}, found {o3d.__version__}")
    frame = _load_npz(frame_file, FRAME_KEYS)
    model_data = _load_npz(model_file, MODEL_KEYS)
    points = np.asarray(frame["points_camera"], dtype=np.float64)
    input_status = "TARGET_POINTS_PRESENT" if len(points) else "NO_TARGET_POINTS"
    surface = None
    sampled = None
    if current:
        surface = TriangleSurface(model_data["vertices"], model_data["faces"])
    else:
        sampled = OfficialSampledSurface(
            points_local=model_data["sample_points_local"],
            normals_local=model_data["sample_normals_local"],
            number_of_points=int(model_data["sample_point_count"]),
            random_seed=int(model_data["sample_seed"]),
            sha256=str(model_data["sample_sha256"].item()),
        )
    models = []
    for coordinate_model in MODELS:
        started = time.perf_counter()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        fit = None
        application_status = "UNAVAILABLE"
        if not len(points):
            raw_status = "STOPPED_NO_TARGET_EVIDENCE"
            raw_reason = "common preprocessing produced zero target points"
        else:
            try:
                observation = SurfaceObservation(
                    points,
                    frame["world_to_camera"],
                    frame["object_to_world"],
                    sequence=str(metadata["case_id"]),
                    frame=int(metadata["frame_id"]),
                )
                if current:
                    contract = cfg["current_fixed_single_frame"]
                    fit = fit_single_surface_correction(
                        observation,
                        surface,
                        coordinate_model,
                        robust_loss=contract["robust_loss"],
                        robust_scale_m=float(contract["robust_scale_m"]),
                        maximum_translation_m=float(cfg["fit"]["advisory_maximum_translation_m"]),
                        maximum_rotation_deg=float(cfg["fit"]["advisory_maximum_rotation_deg"]),
                        maximum_function_evaluations=int(contract["maximum_function_evaluations"]),
                        minimum_points=int(cfg["fit"]["minimum_points"]),
                        finite_difference_relative_step=float(contract["finite_difference_relative_step"]),
                    )
                    raw_status, raw_reason = "OUTPUT", ""
                    application_status = "AUTHORIZED"
                else:
                    fit = fit_open3d_single_frame(
                        observation,
                        sampled,
                        coordinate_model,
                        method=method,
                        maximum_correspondence_distance_m=float(cfg["fit"]["maximum_correspondence_distance_m"]),
                        maximum_iterations=int(cfg["fit"]["maximum_iterations"]),
                        relative_fitness_tolerance=float(cfg["fit"]["relative_fitness_tolerance"]),
                        relative_rmse_tolerance=float(cfg["fit"]["relative_rmse_tolerance"]),
                        huber_delta_m=float(cfg["fit"]["huber_delta_m"]),
                        minimum_correspondences=int(cfg["fit"]["minimum_points"]),
                        advisory_maximum_translation_m=float(cfg["fit"]["advisory_maximum_translation_m"]),
                        advisory_maximum_rotation_deg=float(cfg["fit"]["advisory_maximum_rotation_deg"]),
                    )
                    raw_status = (
                        "OUTPUT" if fit["raw_solver_status"] == "OUTPUT"
                        else "STOPPED_INSUFFICIENT_CORRESPONDENCES"
                    )
                    raw_reason = fit["raw_solver_reason"]
                    application_status = (
                        "AUTHORIZED" if fit["accepted_for_use"]
                        else "DIAGNOSTIC_ONLY_REJECTED_CANDIDATE"
                    )
            except CalibrationCandidateRejected as error:
                fit = error.candidate
                raw_status = "REJECTED_CANDIDATE"
                raw_reason = error.reason_code
                application_status = "DIAGNOSTIC_ONLY_REJECTED_CANDIDATE"
            except (ValueError, RuntimeError) as error:
                fit = None
                raw_status = "NUMERICAL_FAILURE"
                raw_reason = str(error)
        models.append({
            "coordinate_model": coordinate_model,
            "transform_units": "metres_and_rotation_matrix",
            "raw_solver_status": raw_status,
            "raw_solver_reason": raw_reason,
            "input_evidence_status": input_status,
            "formal_application_status": application_status,
            "fit": fit,
            "solve_seconds": float(time.perf_counter() - started),
            "peak_rss_delta_kib": int(
                max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before)
            ),
        })
    write_json(output_file, {
        "schema": "calibration_single_frame_task_prediction_v1",
        "method": method,
        **metadata,
        "single_frame_input_only": True,
        "solver_environment": {
            "python": sys.version,
            "open3d": o3d.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "models": models,
    })


def _solver_roots(cfg: dict[str, Any], method: str) -> tuple[Path, set[Path]]:
    python = Path(
        cfg["current_fixed_single_frame"]["python"]
        if method == "CURRENT_FIXED_SINGLE_FRAME" else cfg["open3d"]["python"]
    ).absolute()
    if not python.exists():
        raise FileNotFoundError(python)
    roots = {
        python.parents[1],
        Path(cfg["open3d"]["python"]).absolute().parents[2],
        Path(cfg["current_fixed_single_frame"]["python"]).absolute().parents[1],
    }
    return python, roots


def _bwrap_frame_command(
    cfg: dict[str, Any],
    record: dict[str, Any],
    frame_file: Path,
    model_file: Path,
    task_output: Path,
    method: str,
) -> list[str]:
    python, roots = _solver_roots(cfg, method)
    command = [
        "bwrap", "--unshare-all", "--die-with-parent",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
        "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev",
    ]
    for root in sorted(roots):
        command += ["--ro-bind", str(root), str(root)]
    command += [
        "--ro-bind", str(REPO_ROOT), "/workspace",
        "--ro-bind", str(frame_file.resolve()), "/input/frame.npz",
        "--ro-bind", str(model_file.resolve()), "/input/model.npz",
        "--bind", str(task_output.resolve()), "/output",
        "--tmpfs", "/tmp", "--setenv", "HOME", "/tmp",
        "--setenv", "OPEN3D_CPU_RENDERING", "true",
        "--setenv", "OMP_NUM_THREADS", str(cfg["open3d"]["maximum_threads"]),
        "--chdir", "/workspace", str(python),
        "/workspace/rl/scripts/calibration_single_frame_comparison_v1.py",
        "solve-frame", "--frame-file", "/input/frame.npz",
        "--model-file", "/input/model.npz",
        "--output-file", "/output/prediction.json",
        "--method", method,
        "--task-id", record["task_id"],
        "--case-id", record["case_id"],
        "--frame-id", str(record["frame_id"]),
        "--frame-sha256", record["frame_sha256"],
        "--model-sha256", record["model_sha256"],
        "--selected-index-sha256", record["selected_index_sha256"],
        "--point-count", str(record["point_count"]),
        "--preprocessing-seconds", str(record["preprocessing_seconds"]),
        "--model-sampling-seconds", str(
            record["open3d_reference_sampling_seconds_amortized"]
        ),
    ]
    return command


def run_isolated(
    cfg: dict[str, Any], prepared_dir: Path, output_dir: Path, method: str,
) -> None:
    if output_dir.exists():
        raise FileExistsError("method output directory must not exist")
    output_dir.mkdir(parents=True)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = []
    with tempfile.TemporaryDirectory(prefix=f"single_frame_{method.lower()}_") as temp_value:
        temp = Path(temp_value)
        for index, record in enumerate(manifest["records"]):
            frame_file = prepared_dir / record["frame_file"]
            model_file = prepared_dir / record["model_file"]
            if sha256(frame_file) != record["frame_sha256"]:
                raise ValueError("prepared frame hash mismatch")
            if sha256(model_file) != record["model_sha256"]:
                raise ValueError("prepared model hash mismatch")
            task_output = temp / f"task_{index:05d}"
            task_output.mkdir()
            subprocess.run(
                _bwrap_frame_command(
                    cfg, record, frame_file, model_file, task_output, method,
                ),
                check=True,
                stdout=subprocess.DEVNULL,
            )
            path = task_output / "prediction.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["task_id"] != record["task_id"] or result["method"] != method:
                raise ValueError("single-frame solver output identity mismatch")
            predictions.append(result)
    write_json(output_dir / "predictions.json", {
        "schema": "calibration_single_frame_predictions_v1",
        "method": method,
        "frame_count": len(predictions),
        "single_frame_mount_isolation": True,
        "predictions": predictions,
        "environment": {
            "python": sys.version,
            "orchestrator_open3d": o3d.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
        },
    })
    (output_dir / "predictions.json").chmod(0o444)


def _rotvec(matrix: np.ndarray) -> list[float]:
    return Rotation.from_matrix(matrix[:3, :3]).as_rotvec().tolist()


def _truth_for_model(frame_truth: dict[str, Any], model: str) -> np.ndarray:
    key = "injected_recovery_world" if model == "WORLD_FIXED" else "injected_recovery_camera"
    return _scoring_rigid(frame_truth[key], "private frame truth")


def _scoring_rigid(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} is not a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} rotation is not proper")
    return matrix


FRAME_FIELDS = [
    "case_id", "frame_id", "task_id", "case_kind", "condition", "quality",
    "shape_source", "generation_model", "method", "coordinate_model",
    "raw_solver_status", "raw_solver_reason", "input_evidence_status",
    "frame_truth_status", "truth_status_reason", "score_category",
    "included_in_accuracy_summary", "true_transform", "estimated_transform",
    "true_translation_mm", "estimated_translation_mm", "translation_error_mm",
    "true_rotvec_rad", "estimated_rotvec_rad", "rotation_error_deg",
    "checkpoint_mean_mm", "checkpoint_median_mm", "checkpoint_p95_mm",
    "checkpoint_maximum_mm", "fitness", "inlier_rmse_m", "correspondence_count",
    "safety_advisory_within_bounds", "preprocessing_seconds",
    "open3d_reference_sampling_seconds_amortized", "solve_seconds",
    "peak_rss_delta_kib",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _pairwise_variation(transforms: list[np.ndarray]) -> dict[str, Any]:
    if len(transforms) < 2:
        return {
            "pair_count": 0, "translation_median_mm": "", "translation_max_mm": "",
            "rotation_median_deg": "", "rotation_max_deg": "",
        }
    translations, rotations = [], []
    for left in range(len(transforms)):
        for right in range(left + 1, len(transforms)):
            error = transform_error(transforms[left], transforms[right])
            translations.append(float(error["translation_error_mm"]))
            rotations.append(float(error["rotation_error_deg"]))
    return {
        "pair_count": len(translations),
        "translation_median_mm": float(np.median(translations)),
        "translation_max_mm": float(np.max(translations)),
        "rotation_median_deg": float(np.median(rotations)),
        "rotation_max_deg": float(np.max(rotations)),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        comparable = [row for row in method_rows if row["included_in_accuracy_summary"]]
        task_rows = list({row["task_id"]: row for row in method_rows}.values())
        common_preprocessing_seconds = float(sum(
            float(row["preprocessing_seconds"]) for row in task_rows
        ))
        open3d_sampling_seconds = float(sum(
            float(row["open3d_reference_sampling_seconds_amortized"])
            for row in task_rows
            if row["open3d_reference_sampling_seconds_amortized"] != ""
        ))
        solve_seconds = float(sum(float(row["solve_seconds"]) for row in method_rows))
        def metric(name: str, operation) -> float | None:
            values = [float(row[name]) for row in comparable]
            return float(operation(values)) if values else None
        result[method] = {
            "attempt_rows": len(method_rows),
            "unique_truth_rows": sum(row["frame_truth_status"] == TRUTH_UNIQUE for row in method_rows),
            "comparable_outputs": len(comparable),
            "stopped_or_failed": sum(row["raw_solver_status"] != "OUTPUT" for row in method_rows),
            "no_target_evidence": sum(
                row["input_evidence_status"] == "NO_TARGET_POINTS" for row in method_rows
            ),
            "numerical_failures": sum(
                row["raw_solver_status"] == "NUMERICAL_FAILURE" for row in method_rows
            ),
            "outputs_without_unique_truth": sum(
                row["raw_solver_status"] == "OUTPUT"
                and row["frame_truth_status"] != TRUTH_UNIQUE
                for row in method_rows
            ),
            "translation_error_mm_median": metric("translation_error_mm", np.median),
            "translation_error_mm_p95": metric("translation_error_mm", lambda x: np.percentile(x, 95)),
            "translation_error_mm_max": metric("translation_error_mm", np.max),
            "rotation_error_deg_median": metric("rotation_error_deg", np.median),
            "rotation_error_deg_p95": metric("rotation_error_deg", lambda x: np.percentile(x, 95)),
            "rotation_error_deg_max": metric("rotation_error_deg", np.max),
            "common_preprocessing_seconds_total": common_preprocessing_seconds,
            "open3d_reference_sampling_seconds_total": open3d_sampling_seconds,
            "solve_seconds_total": solve_seconds,
            "accounted_end_to_end_seconds_total": (
                common_preprocessing_seconds + open3d_sampling_seconds + solve_seconds
            ),
        }
    return result


def _summary_breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "coordinate_model", "shape_source", "condition", "quality",
        "case_kind", "generation_model", "frame_truth_status",
    ):
        result[field] = {
            str(value): _summary([row for row in rows if row[field] == value])
            for value in sorted({row[field] for row in rows}, key=str)
        }
    return result


def score_cases(
    input_dir: Path,
    private_dir: Path,
    prepared_dir: Path,
    method_dirs: dict[str, Path],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError("score output directory must not exist")
    output_dir.mkdir(parents=True)
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    private = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
    public_by_id = {row["case_id"]: row for row in public["cases"]}
    truth_by_id = {row["case_id"]: row for row in private["cases"]}
    prepared_by_task = {row["task_id"]: row for row in prepared["records"]}
    rows: list[dict[str, Any]] = []
    internal_transforms: dict[tuple[str, str, int, str], tuple[np.ndarray, np.ndarray] | None] = {}
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir()
    prediction_hashes: dict[str, str] = {}
    for method in METHODS:
        path = method_dirs[method] / "predictions.json"
        if path.stat().st_mode & 0o222:
            raise ValueError("prediction file must be sealed read-only before scoring")
        prediction_hashes[method] = sha256(path)
        shutil.copy2(path, raw_dir / f"{method.lower()}.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["method"] != method or result["frame_count"] != prepared["frame_count"]:
            raise ValueError("method prediction header mismatch")
        seen: set[str] = set()
        for prediction in result["predictions"]:
            task_id = prediction["task_id"]
            if task_id in seen or task_id not in prepared_by_task:
                raise ValueError("duplicate or unknown task prediction")
            seen.add(task_id)
            prep = prepared_by_task[task_id]
            if (
                prediction["case_id"] != prep["case_id"]
                or int(prediction["frame_id"]) != int(prep["frame_id"])
            ):
                raise ValueError("solver task identity metadata mismatch")
            if prediction["frame_sha256"] != prep["frame_sha256"]:
                raise ValueError("solver did not consume the frozen frame input")
            if prediction["model_sha256"] != prep["model_sha256"]:
                raise ValueError("solver did not consume the frozen model input")
            if prediction["selected_index_sha256"] != prep["selected_index_sha256"]:
                raise ValueError("solver did not consume the common visibility selection")
            case_id = prediction["case_id"]
            frame_id = int(prediction["frame_id"])
            answer = truth_by_id[case_id]
            frame_truth = answer["frames"][frame_id]
            frame_data = _load_npz(prepared_dir / prep["frame_file"], FRAME_KEYS)
            coordinate_models = [row["coordinate_model"] for row in prediction["models"]]
            if len(coordinate_models) != len(MODELS) or set(coordinate_models) != set(MODELS):
                raise ValueError("solver coordinate-model outputs are missing or duplicated")
            for model_result in prediction["models"]:
                model = model_result["coordinate_model"]
                if model_result.get("transform_units") != "metres_and_rotation_matrix":
                    raise ValueError("solver transform unit contract mismatch")
                truth_transform = _truth_for_model(frame_truth, model)
                fit = model_result.get("fit")
                estimate = (
                    _scoring_rigid(fit["transform"], "solver transform")
                    if fit is not None and "transform" in fit else None
                )
                if model_result["raw_solver_status"] == "OUTPUT" and estimate is None:
                    raise ValueError("solver declared OUTPUT without a rigid transform")
                unique = frame_truth["frame_truth_status"] == TRUTH_UNIQUE
                output = model_result["raw_solver_status"] == "OUTPUT" and estimate is not None
                included = bool(unique and output)
                if included:
                    score_category = "COMPARABLE_OUTPUT"
                elif output:
                    score_category = "OUTPUT_WITHOUT_UNIQUE_TRUTH"
                elif frame_truth["frame_truth_status"] == TRUTH_NO_TARGET:
                    score_category = "NO_TARGET_EVIDENCE"
                else:
                    score_category = "STOPPED_OR_FAILED"
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "frame_id": frame_id,
                    "task_id": task_id,
                    "case_kind": answer["case_kind"],
                    "condition": answer["condition"],
                    "quality": answer["quality"],
                    "shape_source": answer["shape_source"],
                    "generation_model": answer["generation_model"],
                    "method": method,
                    "coordinate_model": model,
                    "raw_solver_status": model_result["raw_solver_status"],
                    "raw_solver_reason": model_result["raw_solver_reason"],
                    "input_evidence_status": model_result["input_evidence_status"],
                    "frame_truth_status": frame_truth["frame_truth_status"],
                    "truth_status_reason": frame_truth["truth_status_reason"],
                    "score_category": score_category,
                    "included_in_accuracy_summary": included,
                    "true_transform": json.dumps(truth_transform.tolist()),
                    "estimated_transform": "" if estimate is None else json.dumps(estimate.tolist()),
                    "true_translation_mm": json.dumps((1000 * truth_transform[:3, 3]).tolist()),
                    "estimated_translation_mm": "" if estimate is None else json.dumps((1000 * estimate[:3, 3]).tolist()),
                    "translation_error_mm": "",
                    "true_rotvec_rad": json.dumps(_rotvec(truth_transform)),
                    "estimated_rotvec_rad": "" if estimate is None else json.dumps(_rotvec(estimate)),
                    "rotation_error_deg": "",
                    "checkpoint_mean_mm": "",
                    "checkpoint_median_mm": "",
                    "checkpoint_p95_mm": "",
                    "checkpoint_maximum_mm": "",
                    "fitness": "",
                    "inlier_rmse_m": "",
                    "correspondence_count": "",
                    "safety_advisory_within_bounds": "",
                    "preprocessing_seconds": prediction["preprocessing_seconds"],
                    "open3d_reference_sampling_seconds_amortized": (
                        prediction["model_sampling_seconds"]
                        if method != "CURRENT_FIXED_SINGLE_FRAME" else ""
                    ),
                    "solve_seconds": model_result["solve_seconds"],
                    "peak_rss_delta_kib": model_result["peak_rss_delta_kib"],
                }
                if estimate is not None:
                    error = transform_error(estimate, truth_transform)
                    points_local = np.asarray(frame_truth["check_points_local"], dtype=np.float64)
                    points_world = independent_apply(points_local, frame_data["object_to_world"])
                    check_points = (
                        points_world if model == "WORLD_FIXED"
                        else independent_apply(points_world, frame_data["world_to_camera"])
                    )
                    check = checkpoint_error(estimate, truth_transform, check_points)
                    row.update({
                        "translation_error_mm": error["translation_error_mm"],
                        "rotation_error_deg": error["rotation_error_deg"],
                        "checkpoint_mean_mm": check["mean_mm"],
                        "checkpoint_median_mm": check["median_mm"],
                        "checkpoint_p95_mm": check["p95_mm"],
                        "checkpoint_maximum_mm": check["maximum_mm"],
                    })
                    registration = fit.get("registration_result", {})
                    row["fitness"] = registration.get("fitness", "")
                    row["inlier_rmse_m"] = registration.get("inlier_rmse_m", "")
                    row["correspondence_count"] = registration.get("correspondence_count", "")
                    row["safety_advisory_within_bounds"] = fit.get(
                        "safety_advisory", {},
                    ).get("within_bounds", "")
                rows.append(row)
                internal_transforms[(method, case_id, frame_id, model)] = (
                    (truth_transform, estimate) if estimate is not None else None
                )
        if len(seen) != prepared["frame_count"]:
            raise ValueError("method prediction is missing single-frame tasks")

    _write_csv(output_dir / "frame_comparison.csv", rows, FRAME_FIELDS)
    failures = [row for row in rows if not row["included_in_accuracy_summary"]]
    _write_csv(output_dir / "failures_and_unavailable.csv", failures, FRAME_FIELDS)

    keyed = {
        (row["method"], row["task_id"], row["coordinate_model"]): row
        for row in rows
    }
    common = []
    for candidate in METHODS[1:]:
        for row in rows:
            if row["method"] != METHODS[0] or not row["included_in_accuracy_summary"]:
                continue
            other = keyed[(candidate, row["task_id"], row["coordinate_model"])]
            if not other["included_in_accuracy_summary"]:
                continue
            current_t, candidate_t = float(row["translation_error_mm"]), float(other["translation_error_mm"])
            current_r, candidate_r = float(row["rotation_error_deg"]), float(other["rotation_error_deg"])
            common.append({
                "candidate": candidate,
                "case_id": row["case_id"],
                "frame_id": row["frame_id"],
                "task_id": row["task_id"],
                "coordinate_model": row["coordinate_model"],
                "condition": row["condition"],
                "quality": row["quality"],
                "shape_source": row["shape_source"],
                "current_translation_error_mm": current_t,
                "candidate_translation_error_mm": candidate_t,
                "translation_winner": "CANDIDATE" if candidate_t < current_t else "CURRENT_FIXED" if candidate_t > current_t else "TIE",
                "current_rotation_error_deg": current_r,
                "candidate_rotation_error_deg": candidate_r,
                "rotation_winner": "CANDIDATE" if candidate_r < current_r else "CURRENT_FIXED" if candidate_r > current_r else "TIE",
            })
    common_fields = list(common[0]) if common else ["candidate"]
    _write_csv(output_dir / "common_frame_comparison.csv", common, common_fields)

    runtime_fields = [
        "case_id", "frame_id", "task_id", "method", "coordinate_model",
        "preprocessing_seconds", "open3d_reference_sampling_seconds_amortized",
        "solve_seconds", "peak_rss_delta_kib", "raw_solver_status",
    ]
    _write_csv(output_dir / "runtime_comparison.csv", rows, runtime_fields)

    visibility = []
    for answer in private["cases"]:
        if "owner_map_file" not in answer:
            continue
        owner_path = private_dir / answer["owner_map_file"]
        if sha256(owner_path) != answer["owner_map_sha256"]:
            raise ValueError("private owner-map hash mismatch")
        owners = np.load(owner_path, allow_pickle=False)
        public_case = load_public_case(input_dir / public_by_id[answer["case_id"]]["file"])
        stride = int(public_case["spatial_stride_px"])
        for frame_id, owner in enumerate(owners):
            prep = prepared_by_task[f"{answer['case_id']}_f{frame_id:03d}"]
            indices = np.asarray(prep["selected_flat_indices"], dtype=np.int64)
            selected = owner.reshape(-1)[indices]
            lattice = np.zeros(owner.shape, dtype=bool)
            lattice[::stride, ::stride] = True
            eligible = int(np.sum((owner == 1) & lattice))
            target = int(np.sum(selected == 1))
            other = int(np.sum(selected > 1))
            visibility.append({
                "case_id": answer["case_id"], "frame_id": frame_id,
                "condition": answer["condition"], "selected_total": len(indices),
                "selected_true_target": target,
                "selected_occluder_or_other": other,
                "contamination_fraction": "" if not len(indices) else other / len(indices),
                "eligible_true_target": eligible,
                "retained_true_target_fraction": "" if not eligible else target / eligible,
                "deleted_true_target": eligible - target,
                "empty_selection": len(indices) == 0,
            })
    visibility_fields = list(visibility[0]) if visibility else ["case_id"]
    _write_csv(output_dir / "frame_visibility.csv", visibility, visibility_fields)

    variation = []
    for method in METHODS:
        for case_id, answer in truth_by_id.items():
            total = len(answer["frames"])
            for model in MODELS:
                truths, estimates = [], []
                for frame_id in range(total):
                    pair = internal_transforms[(method, case_id, frame_id, model)]
                    truths.append(_truth_for_model(answer["frames"][frame_id], model))
                    if pair is not None:
                        estimates.append(pair[1])
                variation.append({
                    "case_id": case_id, "method": method, "coordinate_model": model,
                    "successful_frames": len(estimates), "total_frames": total,
                    **{f"truth_{key}": value for key, value in _pairwise_variation(truths).items()},
                    **{f"output_{key}": value for key, value in _pairwise_variation(estimates).items()},
                })
    variation_fields = list(variation[0])
    _write_csv(output_dir / "sequence_variation.csv", variation, variation_fields)

    summary = _summary(rows)
    write_json(output_dir / "summary.json", {
        "schema": "calibration_single_frame_score_summary_v1",
        "group_count": public["group_count"],
        "frame_count": public["frame_count"],
        "attempt_rows_expected": public["frame_count"] * len(METHODS) * len(MODELS),
        "attempt_rows_observed": len(rows),
        "precision_pass_threshold_defined": False,
        "metrics": summary,
        "breakdowns": _summary_breakdowns(rows),
    })
    write_json(output_dir / "exam_integrity_report.json", {
        "schema": "calibration_single_frame_integrity_v1",
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "prepared_manifest_sha256": sha256(prepared_dir / "manifest.json"),
        "private_answer_manifest_sha256": sha256(private_dir / "answer_manifest.json"),
        "prediction_sha256": prediction_hashes,
        "all_methods_received_identical_frame_and_model_inputs": True,
        "all_methods_received_identical_visibility_selection": True,
        "single_frame_mount_isolation": True,
        "predictions_existed_before_scoring": True,
        "precision_pass_threshold_defined": False,
    })


def verify(input_dir: Path, private_dir: Path | None, prepared_dir: Path | None) -> None:
    public = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    for row in public["cases"]:
        if sha256(input_dir / row["file"]) != row["sha256"]:
            raise ValueError("public case hash mismatch")
    if prepared_dir is not None:
        prepared = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        for row in prepared["records"]:
            if sha256(prepared_dir / row["frame_file"]) != row["frame_sha256"]:
                raise ValueError("prepared frame hash mismatch")
            if sha256(prepared_dir / row["model_file"]) != row["model_sha256"]:
                raise ValueError("prepared model hash mismatch")
    if private_dir is not None:
        private = json.loads((private_dir / "answer_manifest.json").read_text(encoding="utf-8"))
        for row in private["cases"]:
            if "owner_map_file" in row and sha256(private_dir / row["owner_map_file"]) != row["owner_map_sha256"]:
                raise ValueError("private owner-map hash mismatch")
    print(json.dumps({
        "public": True,
        "prepared": prepared_dir is not None,
        "private": private_dir is not None,
    }))


def _representation_audit(prepared_dir: Path) -> dict[str, Any]:
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for model_record in manifest["models"][:4]:
        model = _load_npz(prepared_dir / model_record["file"], MODEL_KEYS)
        mesh = trimesh.Trimesh(
            vertices=model["vertices"], faces=model["faces"], process=False,
        )
        independent = deterministic_surface_points(mesh, points_per_face=3)
        sampled = model["sample_points_local"]
        distances, indices = cKDTree(sampled).query(independent, k=1)
        normals = model["sample_normals_local"][indices]
        point_to_plane = np.abs(np.sum((independent - sampled[indices]) * normals, axis=1))
        rows.append({
            "case_id": model_record["case_id"],
            "independent_point_count": len(independent),
            "reference_point_count": len(sampled),
            "sample_sha256": str(model["sample_sha256"].item()),
            "nearest_distance_median_mm": float(1000 * np.median(distances)),
            "nearest_distance_p95_mm": float(1000 * np.percentile(distances, 95)),
            "nearest_distance_max_mm": float(1000 * np.max(distances)),
            "point_to_plane_median_mm": float(1000 * np.median(point_to_plane)),
            "point_to_plane_p95_mm": float(1000 * np.percentile(point_to_plane, 95)),
        })
    return {
        "schema": "calibration_single_frame_representation_audit_v1",
        "official_sampler": "TriangleMesh.sample_points_uniformly",
        "use_triangle_normal": True,
        "question_samples_are_independent": True,
        "rows": rows,
    }


def _run_evidence(command: list[str], *, environment: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "exit_code": int(result.returncode),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output": result.stdout,
        "output_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }


def _development_validation(output: Path) -> None:
    tests = output / "tests"
    tests.mkdir()
    smoke_script = RL_ROOT / "tests/open3d_comparison/official_registration_icp_smoke.py"
    smoke = _run_evidence([sys.executable, str(smoke_script)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(RL_ROOT / "src")
    pytest_run = _run_evidence([
        sys.executable, "-m", "pytest", "-q",
        str(RL_ROOT / "tests/open3d_comparison/test_open3d_single_frame.py"),
        str(RL_ROOT / "tests/open3d_comparison/test_single_frame_scoring.py"),
    ], environment=environment)
    source = (RL_ROOT / "src/egoengine_repro/evaluation/open3d_calibration.py").read_text(
        encoding="utf-8",
    )
    static = {
        "registration_icp_call_expressions": source.count("registration_icp("),
        "contains_grouped_point_to_plane_icp": "grouped_point_to_plane_icp" in source,
        "contains_compute_transformation": "compute_transformation" in source,
        "contains_kdtreeflann": "KDTreeFlann" in source,
        "contains_project_iteration_loop": "for iteration" in source or "while iteration" in source,
    }
    write_json(tests / "standalone_official_smoke.json", smoke)
    write_json(tests / "wrapper_and_scoring_pytest.json", pytest_run)
    write_json(tests / "static_active_source_audit.json", static)
    write_json(output / "official_api_check.json", {
        "schema": "calibration_single_frame_official_api_check_v1",
        "open3d_version": o3d.__version__,
        "standalone_official_smoke": smoke,
        "wrapper_and_scoring_pytest": pytest_run,
        "static_active_source_audit": static,
        "wrapper_equivalence_covered_for_plain_and_huber": True,
        "wrapper_stress_cases": [
            "complex_mesh_zero_correction", "independent_sampling", "noisy_points",
            "zero_correspondences",
        ],
        "implementation_tolerance": {
            "rtol": 0.0,
            "simple_and_huber_atol": 1e-14,
            "stress_case_atol": 1e-13,
        },
        "implementation_tolerance_is_exam_threshold": False,
    })
    if smoke["exit_code"] or pytest_run["exit_code"]:
        raise RuntimeError("development validation command failed")
    if static != {
        "registration_icp_call_expressions": 1,
        "contains_grouped_point_to_plane_icp": False,
        "contains_compute_transformation": False,
        "contains_kdtreeflann": False,
        "contains_project_iteration_loop": False,
    }:
        raise RuntimeError(f"active Open3D source audit failed: {static}")


def _formal_blueprint_dry_run(cfg: dict[str, Any], output: Path) -> None:
    """Exercise only the public generator shape with a disclosed non-secret seed."""
    with tempfile.TemporaryDirectory(prefix="single_frame_blueprint_dry_run_") as temp_value:
        root = Path(temp_value)
        public_dir = root / "public"
        private_dir = root / "private"
        generate_cases(
            cfg,
            seed=202610079,
            public_dir=public_dir,
            private_dir=private_dir,
            public_development=False,
        )
        public = json.loads((public_dir / "manifest.json").read_text(encoding="utf-8"))
        private = json.loads(
            (private_dir / "answer_manifest.json").read_text(encoding="utf-8")
        )
        public_ids = [row["case_id"] for row in public["cases"]]
        truth_fields_complete = all(
            all(
                "injected_recovery_camera" in frame
                and "injected_recovery_world" in frame
                and "frame_truth_status" in frame
                and "truth_status_reason" in frame
                for frame in case["frames"]
            )
            for case in private["cases"]
        )
        report = {
            "schema": "calibration_single_frame_formal_blueprint_dry_run_v1",
            "status": "NON_SECRET_GENERATOR_DRY_RUN_ONLY",
            "formal_exam_executed": False,
            "predictions_executed": False,
            "scoring_executed": False,
            "disclosed_dry_run_seed": 202610079,
            "group_count": public["group_count"],
            "frame_count": public["frame_count"],
            "attempt_rows_expected": public["frame_count"] * len(METHODS) * len(MODELS),
            "private_group_count": private["group_count"],
            "private_frame_count": private["frame_count"],
            "public_contains_private_answer": public["contains_private_answer"],
            "case_ids_are_not_public_development_ids": all(
                not value.startswith("public_sf_") for value in public_ids
            ),
            "case_ids_unique": len(public_ids) == len(set(public_ids)),
            "dual_coordinate_truth_fields_complete": truth_fields_complete,
        }
        if report != {
            **report,
            "group_count": 96,
            "frame_count": 448,
            "attempt_rows_expected": 2688,
            "private_group_count": 96,
            "private_frame_count": 448,
            "public_contains_private_answer": False,
            "case_ids_are_not_public_development_ids": True,
            "case_ids_unique": True,
            "dual_coordinate_truth_fields_complete": True,
        }:
            raise RuntimeError(f"formal blueprint dry run failed: {report}")
        write_json(output / "formal_generator_dry_run.json", report)


def develop(cfg_path: Path) -> None:
    cfg = _load_config(cfg_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(f"development output exists: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="calibration_single_frame_public_") as temp_value:
        root = Path(temp_value)
        public = root / "public"
        private = root / "private"
        prepared = root / "prepared"
        score = root / "score"
        generate_cases(
            cfg,
            seed=int(cfg["public_development"]["seed"]),
            public_dir=public,
            private_dir=private,
            public_development=True,
        )
        prepare_cases(cfg, public, prepared)
        method_dirs = {}
        for method in METHODS:
            destination = root / method.lower()
            run_isolated(cfg, prepared, destination, method)
            method_dirs[method] = destination
        score_cases(public, private, prepared, method_dirs, score)
        shutil.copytree(score, output / "public_scores")
        shutil.copy2(public / "manifest.json", output / "public_manifest.json")
        shutil.copy2(prepared / "manifest.json", output / "prepared_manifest.json")
        write_json(output / "representation_audit.json", _representation_audit(prepared))
    summary = json.loads((output / "public_scores/summary.json").read_text(encoding="utf-8"))
    _development_validation(output)
    _formal_blueprint_dry_run(cfg, output)
    write_json(output / "source_and_environment.json", {
        "schema": "calibration_single_frame_source_environment_v1",
        "repository_head_before_freeze": _git_head(),
        "current_fixed_single_frame": cfg["current_fixed_single_frame"],
        "open3d": cfg["open3d"],
        "files": {
            str(path.relative_to(REPO_ROOT)): _artifact(path)
            for path in (
                cfg_path,
                Path(__file__),
                RL_ROOT / "src/egoengine_repro/evaluation/open3d_calibration.py",
                RL_ROOT / "src/egoengine_repro/evaluation/calibration_single_frame_cases.py",
                RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py",
                RL_ROOT / "src/egoengine_repro/evaluation/calibration_known_answer.py",
            )
        },
    })
    shutil.copy2(cfg_path, output / "frozen_config.yaml")
    write_json(output / "frozen_exam_plan.json", {
        "schema": "calibration_single_frame_frozen_exam_plan_v1",
        "status": "READY_FOR_USER_BLIND_EXAM",
        "formal_exam_executed": False,
        "formal_group_count": cfg["formal_exam"]["group_count"],
        "formal_frame_count": cfg["formal_exam"]["expected_frame_count"],
        "attempt_rows_expected": cfg["formal_exam"]["expected_frame_count"] * 6,
        "real_taco_calibration_forbidden": True,
    })
    table_rows = []
    for method in METHODS:
        row = summary["metrics"][method]
        table_rows.append(
            f"| `{method}` | {row['comparable_outputs']}/{row['attempt_rows']} | "
            f"{row['stopped_or_failed']} | {row['translation_error_mm_median']:.6g} | "
            f"{row['rotation_error_deg_median']:.6g} |"
        )
    (output / "summary.md").write_text(
        "# 单帧校准公开回归\n\n"
        "状态：`READY_FOR_USER_BLIND_EXAM`。正式新96组/448帧考试尚未执行。\n\n"
        "公开开发集只有22组/46帧，用于接口和评分回归，不用于宣称最终算法优劣。"
        "本轮没有定义毫米/角度及格线。\n\n"
        "| 方法 | 可比较输出/尝试行 | 停止或失败 | 平移误差中位数 mm | 旋转误差中位数 deg |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(table_rows)
        + "\n\n完整统计与分位数见 `public_scores/summary.json`；逐帧原始结果、失败原因、"
        "共同帧对照和运行时间均已保留。\n\n"
        f"```json\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n```\n",
        encoding="utf-8",
    )
    (output / "changes.md").write_text(
        "# Active calibration changes\n\n"
        "| Change | Status | Reason |\n|---|---|---|\n"
        "| Remove project grouped ICP loop and deterministic face-index sampler | removed from active code | Single-frame Open3D must call official `registration_icp` once |\n"
        "| Add `fit_single_surface_correction` | active | Allows one observation while preserving the frozen SciPy residual and solver settings |\n"
        "| Keep historical run evidence | retained under `rl/runs/calibration_open3d_comparison_v1` | Historical facts remain immutable |\n"
        "| Retire old comparison runner/config/tests | recoverable from commit `4e9ad4a8cfb593e6584b332f151939778416510e` | Prevent normal tooling from loading deleted grouped code |\n",
        encoding="utf-8",
    )
    instructions = """# 用户运行新单帧保密考试

开发和公开验证完成后，由用户在开发 Codex不可读的私有环境生成新的96组、448帧题目。不要复用此前任何公开题或seed。

```bash
CURRENT_PY=/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python
OPEN3D_PY=/data_all/zzx/calibration_open3d_v0200/env/bin/python
TOOL=rl/scripts/calibration_single_frame_comparison_v1.py

$CURRENT_PY $TOOL generate --secret-seed-file /secure/private/seed.txt --public-dir /secure/public/input --private-dir /secure/private/answer
$OPEN3D_PY $TOOL prepare --input-dir /secure/public/input --prepared-dir /secure/public/prepared
$OPEN3D_PY $TOOL verify --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared
$OPEN3D_PY $TOOL run-isolated --prepared-dir /secure/public/prepared --output-dir /secure/output/current --method CURRENT_FIXED_SINGLE_FRAME
$OPEN3D_PY $TOOL run-isolated --prepared-dir /secure/public/prepared --output-dir /secure/output/open3d --method OPEN3D_OFFICIAL_SINGLE_FRAME
$OPEN3D_PY $TOOL run-isolated --prepared-dir /secure/public/prepared --output-dir /secure/output/huber --method OPEN3D_HUBER_SINGLE_FRAME
$CURRENT_PY $TOOL score --input-dir /secure/public/input --private-dir /secure/private/answer --prepared-dir /secure/public/prepared --current-dir /secure/output/current --open3d-dir /secure/output/open3d --huber-dir /secure/output/huber --output-dir /secure/score
```

三份预测全部设为只读并封存哈希以后才允许执行 `score`。发生基础设施错误时本批作废并换全新seed；正常求解停止是成绩，不得重试取最好。
"""
    (output / "用户运行步骤.md").write_text(instructions, encoding="utf-8")
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
    solve = sub.add_parser("solve-frame")
    solve.add_argument("--frame-file", type=Path, required=True)
    solve.add_argument("--model-file", type=Path, required=True)
    solve.add_argument("--output-file", type=Path, required=True)
    solve.add_argument("--method", choices=METHODS, required=True)
    solve.add_argument("--task-id", required=True)
    solve.add_argument("--case-id", required=True)
    solve.add_argument("--frame-id", type=int, required=True)
    solve.add_argument("--frame-sha256", required=True)
    solve.add_argument("--model-sha256", required=True)
    solve.add_argument("--selected-index-sha256", required=True)
    solve.add_argument("--point-count", type=int, required=True)
    solve.add_argument("--preprocessing-seconds", type=float, required=True)
    solve.add_argument("--model-sampling-seconds", type=float, required=True)
    isolated = sub.add_parser("run-isolated")
    isolated.add_argument("--prepared-dir", type=Path, required=True)
    isolated.add_argument("--output-dir", type=Path, required=True)
    isolated.add_argument("--method", choices=METHODS, required=True)
    score = sub.add_parser("score")
    score.add_argument("--input-dir", type=Path, required=True)
    score.add_argument("--private-dir", type=Path, required=True)
    score.add_argument("--prepared-dir", type=Path, required=True)
    score.add_argument("--current-dir", type=Path, required=True)
    score.add_argument("--open3d-dir", type=Path, required=True)
    score.add_argument("--huber-dir", type=Path, required=True)
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
        generate_cases(
            cfg, seed=seed, public_dir=args.public_dir,
            private_dir=args.private_dir, public_development=False,
        )
    elif args.command == "prepare":
        prepare_cases(cfg, args.input_dir.resolve(strict=True), args.prepared_dir)
    elif args.command == "solve-frame":
        if sha256(args.frame_file) != args.frame_sha256:
            raise ValueError("mounted frame hash mismatch")
        if sha256(args.model_file) != args.model_sha256:
            raise ValueError("mounted model hash mismatch")
        solve_frame(
            cfg, args.frame_file, args.model_file, args.output_file, args.method,
            {
                "task_id": args.task_id, "case_id": args.case_id,
                "frame_id": args.frame_id, "frame_sha256": args.frame_sha256,
                "model_sha256": args.model_sha256,
                "selected_index_sha256": args.selected_index_sha256,
                "point_count": args.point_count,
                "preprocessing_seconds": args.preprocessing_seconds,
                "model_sampling_seconds": args.model_sampling_seconds,
            },
        )
    elif args.command == "run-isolated":
        run_isolated(cfg, args.prepared_dir.resolve(strict=True), args.output_dir, args.method)
    elif args.command == "score":
        score_cases(
            args.input_dir.resolve(strict=True),
            args.private_dir.resolve(strict=True),
            args.prepared_dir.resolve(strict=True),
            {
                METHODS[0]: args.current_dir.resolve(strict=True),
                METHODS[1]: args.open3d_dir.resolve(strict=True),
                METHODS[2]: args.huber_dir.resolve(strict=True),
            },
            args.output_dir,
        )
    elif args.command == "verify":
        verify(
            args.input_dir.resolve(strict=True),
            None if args.private_dir is None else args.private_dir.resolve(strict=True),
            None if args.prepared_dir is None else args.prepared_dir.resolve(strict=True),
        )


if __name__ == "__main__":
    main()
