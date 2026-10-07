#!/usr/bin/env python3
"""Apply the frozen single-frame calibration solver to Brush depth, then audit table penetration.

This is a static, temporary-point-cloud experiment.  It never rewrites source
depth, camera parameters, object poses, hand trajectories, or the active
support-surface contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.taco_calibration_residual import (  # noqa: E402
    SurfaceObservation, TriangleSurface, apply_transform,
    fit_single_surface_correction,
)
from egoengine_repro.evaluation.open3d_calibration import (  # noqa: E402
    fit_open3d_single_frame, official_uniform_sampled_surface,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (  # noqa: E402
    measured_target_selector, target_visibility_mask,
)
from egoengine_repro.evaluation.taco_depth import (  # noqa: E402
    DepthVideoSpec, ffprobe_video, iter_depth_frames, raw_depth_to_metres,
    stream_frame_count,
)
from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector, load_official_hand_sequence,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    backproject_metric_depth, camera_points_to_world,
    fit_horizontal_support_plane, foreground_excluded_background_points,
)


DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_single_frame_calibration_penetration_v1.yaml"
VISIBILITY_UNCERTAINTY_M = 8.0 / 4000.0
MODELS = ("WORLD_FIXED", "CAMERA_LOCAL")
ENTITIES = ("brush", "bowl", "left_hand", "right_hand", "hand")


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label}: missing={sorted(expected-actual)} unknown={sorted(actual-expected)}")


def load_contracts(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_keys = {
        "schema", "status", "minimum_baseline", "paper_faithful", "authorization",
        "sample_key", "upstream", "evaluation", "output",
    }
    if cfg.get("schema") == "taco_brush_open3d_single_frame_calibration_penetration_v1":
        root_keys.add("solver")
    exact_keys(cfg, root_keys, "root")
    exact_keys(cfg["authorization"], {
        "temporary_depth_point_correction", "static_penetration_audit",
        "raw_depth_modification", "camera_parameter_modification",
        "object_pose_modification", "hand_trajectory_modification",
        "active_support_contract_modification", "mink", "physics", "replay", "mpc",
        "reinforcement_learning",
    }, "authorization")
    exact_keys(cfg["upstream"], {
        "current_single_frame_config", "taco_calibration_config",
        "table_estimator_config", "active_support_contract",
    }, "upstream")
    exact_keys(cfg["evaluation"], {
        "coordinate_models", "penetration_tolerance_m", "require_same_effective_frames",
        "choose_best_coordinate_model",
    }, "evaluation")
    if cfg["schema"] not in {
        "taco_brush_single_frame_calibration_penetration_v1",
        "taco_brush_open3d_single_frame_calibration_penetration_v1",
    }:
        raise ValueError("unexpected schema")
    if cfg["status"] != "TEMPORARY_SINGLE_FRAME_CALIBRATION_STATIC_AUDIT":
        raise ValueError("unexpected status")
    if cfg["paper_faithful"] is not False:
        raise ValueError("local single-frame correction must not be labelled paper-faithful")
    allowed = {"temporary_depth_point_correction", "static_penetration_audit"}
    if not all(cfg["authorization"][key] for key in allowed):
        raise ValueError("temporary correction and static audit must be authorized")
    if any(value for key, value in cfg["authorization"].items() if key not in allowed):
        raise ValueError("source mutation and runtime execution are forbidden")
    evaluation = cfg["evaluation"]
    if evaluation != {
        "coordinate_models": ["WORLD_FIXED", "CAMERA_LOCAL"],
        "penetration_tolerance_m": 0.00005,
        "require_same_effective_frames": True,
        "choose_best_coordinate_model": False,
    }:
        raise ValueError("frozen evaluation contract changed")
    if "solver" in cfg:
        exact_keys(cfg["solver"], {"method"}, "solver")
        if cfg["solver"]["method"] != "OPEN3D_OFFICIAL_SINGLE_FRAME":
            raise ValueError("this audit authorizes only official Open3D point-to-plane")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], git_head(REPO_ROOT)],
        cwd=REPO_ROOT, check=True,
    )
    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["upstream"].items()}
    current = yaml.safe_load(paths["current_single_frame_config"].read_text(encoding="utf-8"))
    taco = yaml.safe_load(paths["taco_calibration_config"].read_text(encoding="utf-8"))
    table = yaml.safe_load(paths["table_estimator_config"].read_text(encoding="utf-8"))
    if current["schema"] != "calibration_single_frame_comparison_v1":
        raise ValueError("unexpected current single-frame configuration")
    if taco["schema"] != "taco_depth_mocap_calibration_residual_adjudication_v1":
        raise ValueError("unexpected TACO calibration configuration")
    if table["schema"] != "taco_multiframe_table_depth_estimation_v1":
        raise ValueError("unexpected table estimator configuration")
    return cfg, current, taco, table


def resolve_sample(cfg: dict[str, Any], taco: dict[str, Any]) -> dict[str, Any]:
    source = next(row for row in taco["samples"] if row["key"] == cfg["sample_key"])
    sample = dict(source)
    path_keys = {
        "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand",
        "left_shape", "right_shape", "object_pose_dir", "object_model_dir",
    }
    for key in path_keys:
        sample[key] = Path(sample[key]).resolve(strict=True)
    for role in ("tool", "target"):
        identifier = sample[f"{role}_id"]
        sample[f"{role}_pose"] = (
            sample["object_pose_dir"] / f"{role}_{identifier}.npy"
        ).resolve(strict=True)
        sample[f"{role}_mesh"] = (
            sample["object_model_dir"] / f"{identifier}_cm.obj"
        ).resolve(strict=True)
    return sample


def posed_mesh(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray) -> trimesh.Trimesh:
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    return trimesh.Trimesh(vertices=world, faces=faces, process=False)


def load_geometry(taco: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    chumpy = str(Path(taco["official"]["chumpy_dependency"]).resolve(strict=True))
    if chumpy not in sys.path:
        sys.path.insert(0, chumpy)
    dataset_utils = Path(taco["official"]["checkout"]) / "dataset_utils"
    hands: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        vertices, _, faces, _ = load_official_hand_sequence(
            dataset_utils=dataset_utils,
            pose_path=sample[f"{side}_hand"],
            shape_path=sample[f"{side}_shape"],
            side=side,
            device=taco["official"]["device"],
        )
        if vertices.shape[0] != sample["expected_frames"]:
            raise ValueError(f"{side} hand frame count mismatch")
        hands[side] = {"vertices": np.asarray(vertices), "faces": np.asarray(faces)}
    objects: dict[str, dict[str, np.ndarray]] = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(sample[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        poses = np.load(sample[f"{role}_pose"], mmap_mode="r")
        if poses.shape != (sample["expected_frames"], 4, 4):
            raise ValueError(f"{role} pose frame count mismatch")
        objects[role] = {
            "vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces),
            "poses": poses,
        }
    return {"hands": hands, "objects": objects, "dataset_utils": dataset_utils}


def correction_points_world(
    points_camera: np.ndarray,
    world_to_camera: np.ndarray,
    correction: np.ndarray,
    model: str,
) -> np.ndarray:
    values = points_camera
    if model == "CAMERA_LOCAL":
        values = apply_transform(values, correction)
    world = camera_points_to_world(values, world_to_camera)
    if model == "WORLD_FIXED":
        world = apply_transform(world, correction)
    return world


def fit_plane(points: np.ndarray, table: dict[str, Any]) -> Any:
    contract = table["plane_fit"]
    return fit_horizontal_support_plane(
        points,
        random_seed=int(contract["random_seed"]),
        maximum_tilt_from_world_z_deg=float(contract["maximum_tilt_from_world_z_deg"]),
        inlier_distance_m=float(contract["inlier_distance_m"]),
        maximum_iterations=int(contract["maximum_iterations"]),
        minimum_candidate_points=int(contract["minimum_candidate_points"]),
    )


def entity_distances(
    plane: Plane, geometry: dict[str, Any], frame: int,
) -> dict[str, float]:
    objects = geometry["objects"]
    brush = plane.minimum_signed_distance(objects["tool"]["vertices"], objects["tool"]["poses"][frame])
    bowl = plane.minimum_signed_distance(objects["target"]["vertices"], objects["target"]["poses"][frame])
    left = float(plane.signed_distance(geometry["hands"]["left"]["vertices"][frame]).min())
    right = float(plane.signed_distance(geometry["hands"]["right"]["vertices"][frame]).min())
    return {
        "brush": brush, "bowl": bowl, "left_hand": left, "right_hand": right,
        "hand": min(left, right),
    }


def fit_kwargs(current: dict[str, Any]) -> dict[str, Any]:
    local = current["current_fixed_single_frame"]
    common = current["fit"]
    return {
        "robust_loss": local["robust_loss"],
        "robust_scale_m": float(local["robust_scale_m"]),
        "maximum_translation_m": float(common["advisory_maximum_translation_m"]),
        "maximum_rotation_deg": float(common["advisory_maximum_rotation_deg"]),
        "maximum_function_evaluations": int(local["maximum_function_evaluations"]),
        "minimum_points": int(common["minimum_points"]),
        "finite_difference_relative_step": float(local["finite_difference_relative_step"]),
    }


def open3d_fit_kwargs(current: dict[str, Any]) -> dict[str, Any]:
    fit = current["fit"]
    return {
        "maximum_correspondence_distance_m": float(fit["maximum_correspondence_distance_m"]),
        "maximum_iterations": int(fit["maximum_iterations"]),
        "relative_fitness_tolerance": float(fit["relative_fitness_tolerance"]),
        "relative_rmse_tolerance": float(fit["relative_rmse_tolerance"]),
        "huber_delta_m": float(fit["huber_delta_m"]),
        "minimum_correspondences": int(fit["minimum_points"]),
        "advisory_maximum_translation_m": float(fit["advisory_maximum_translation_m"]),
        "advisory_maximum_rotation_deg": float(fit["advisory_maximum_rotation_deg"]),
    }


def adapt_open3d_fit(value: dict[str, Any]) -> dict[str, Any]:
    if value["raw_solver_status"] != "OUTPUT":
        raise ValueError(value["raw_solver_reason"])
    transform = np.asarray(value["transform"], dtype=np.float64)
    parameters = np.concatenate([
        transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
    ])
    result = dict(value)
    result["parameters_translation_m_then_rotvec_rad"] = parameters.tolist()
    result["fit_metrics"] = dict(value["registration_result"])
    result["identity_metrics"] = {"status": "NOT_COMPUTED_BY_OPEN3D_CONTRACT"}
    return result


def _failure_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def plane_fields(prefix: str, estimate: Any | None) -> dict[str, Any]:
    if estimate is None:
        return {
            f"{prefix}_plane_offset_m": None,
            f"{prefix}_plane_tilt_deg": None,
            f"{prefix}_plane_inlier_count": None,
            f"{prefix}_plane_inlier_fraction": None,
        }
    return {
        f"{prefix}_plane_offset_m": estimate.plane.offset,
        f"{prefix}_plane_tilt_deg": estimate.tilt_from_world_z_deg,
        f"{prefix}_plane_inlier_count": estimate.inlier_count,
        f"{prefix}_plane_inlier_fraction": estimate.inlier_fraction,
    }


def process(
    cfg: dict[str, Any], current: dict[str, Any], taco: dict[str, Any],
    table: dict[str, Any], sample: dict[str, Any], output: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    geometry = load_geometry(taco, sample)
    intrinsic = np.loadtxt(sample["intrinsic"])
    extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
    if extrinsic.shape != (sample["expected_frames"], 4, 4):
        raise ValueError("extrinsic frame count mismatch")
    stream = ffprobe_video(sample["depth_video"])
    count = stream_frame_count(stream)
    if count != sample["expected_frames"]:
        raise ValueError("Depth stream violates INDEX_ALIGNED frame count")
    spec = DepthVideoSpec(
        width=int(taco["depth"]["width"]), height=int(taco["depth"]["height"]),
        frame_count=count,
    )
    projector = TacoOfficialProjector(
        dataset_utils=geometry["dataset_utils"],
        image_size=(int(taco["depth"]["width"]), int(taco["depth"]["height"])),
        intrinsic=intrinsic, extrinsic=extrinsic[0], device=taco["official"]["device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(rasterizer.raster_settings, max_faces_per_bin=400000)
    tool = geometry["objects"]["tool"]
    target = geometry["objects"]["target"]
    open3d_method = cfg.get("solver", {}).get("method")
    if open3d_method is None:
        surface = TriangleSurface(tool["vertices"], tool["faces"])
        sampled_surface = None
        fits = fit_kwargs(current)
    else:
        surface = None
        sampled_surface = official_uniform_sampled_surface(
            tool["vertices"], tool["faces"],
            number_of_points=int(current["open3d"]["reference_point_count"]),
            random_seed=int(current["open3d"]["reference_sampling_seed"]),
        )
        fits = open3d_fit_kwargs(current)
    rows: list[dict[str, Any]] = []
    corrections: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    erosion = int(taco["anchors"]["interior_erosion_px"])
    calibration_stride = int(taco["anchors"]["spatial_stride_px"])
    threshold = float(cfg["evaluation"]["penetration_tolerance_m"])

    for frame, raw in enumerate(iter_depth_frames(sample["depth_video"], spec)):
        projector.set_camera(intrinsic, extrinsic[frame])
        tool_mesh = posed_mesh(tool["vertices"], tool["faces"], tool["poses"][frame])
        target_mesh = posed_mesh(target["vertices"], target["faces"], target["poses"][frame])
        hand_meshes = {
            side: trimesh.Trimesh(
                vertices=geometry["hands"][side]["vertices"][frame],
                faces=geometry["hands"][side]["faces"], process=False,
            )
            for side in ("left", "right")
        }
        nominal_tool_depth = projector.render_depth([tool_mesh])
        occluder_depths = [
            projector.render_depth([hand_meshes["left"]]),
            projector.render_depth([hand_meshes["right"]]),
            projector.render_depth([target_mesh]),
        ]
        visible = target_visibility_mask(
            nominal_tool_depth, occluder_depths,
            uncertainty_margin_m=VISIBILITY_UNCERTAINTY_M,
        )
        depth = raw_depth_to_metres(raw, scale=float(taco["depth"]["scale"]))
        selected = measured_target_selector(
            depth, nominal_tool_depth, occluder_depths,
            uncertainty_margin_m=VISIBILITY_UNCERTAINTY_M,
            erosion_px=erosion,
        )
        interior = cv2.erode(
            visible.astype(np.uint8), np.ones((2 * erosion + 1,) * 2, dtype=np.uint8),
        ) > 0
        points_tool, _ = backproject_metric_depth(
            depth, intrinsic, selector=selected, spatial_stride_px=calibration_stride,
        )
        foreground = projector.render_mask([
            hand_meshes["left"], hand_meshes["right"], tool_mesh, target_mesh,
        ])
        baseline_estimate = None
        baseline_reason = ""
        table_camera = np.empty((0, 3), dtype=np.float64)
        try:
            table_world, table_counts = foreground_excluded_background_points(
                depth, intrinsic, extrinsic[frame], foreground,
                dilation_px=int(table["foreground"]["dilation_px"]),
                interaction_roi_scale=float(table["roi"]["interaction_roi_scale"]),
                spatial_stride_px=int(table["sampling"]["spatial_stride_px"]),
            )
            table_camera = apply_transform(table_world, extrinsic[frame])
            baseline_estimate = fit_plane(table_world, table)
        except ValueError as error:
            table_counts = {}
            baseline_reason = _failure_text(error)

        for model in MODELS:
            fit = None
            fit_status = "OUTPUT"
            fit_reason = ""
            try:
                observation = SurfaceObservation(
                    points_camera=points_tool,
                    world_to_camera=extrinsic[frame],
                    object_to_world=tool["poses"][frame],
                    sequence=sample["key"], frame=frame,
                )
                if open3d_method is None:
                    fit = fit_single_surface_correction(
                        observation, surface, model, **fits,
                    )
                else:
                    fit = adapt_open3d_fit(fit_open3d_single_frame(
                        observation, sampled_surface, model,
                        method=open3d_method, **fits,
                    ))
            except (ValueError, RuntimeError) as error:
                fit_status = "FAILED"
                fit_reason = _failure_text(error)

            corrected_estimate = None
            corrected_reason = ""
            if fit is not None and baseline_estimate is not None:
                try:
                    corrected_world = correction_points_world(
                        table_camera, extrinsic[frame], np.asarray(fit["transform"]), model,
                    )
                    corrected_estimate = fit_plane(corrected_world, table)
                except ValueError as error:
                    corrected_reason = _failure_text(error)
            elif fit is None:
                corrected_reason = "correction unavailable"
            else:
                corrected_reason = "uncorrected table estimate unavailable"

            effective = baseline_estimate is not None and corrected_estimate is not None
            baseline_distances = (
                entity_distances(baseline_estimate.plane, geometry, frame)
                if effective else {key: None for key in ENTITIES}
            )
            corrected_distances = (
                entity_distances(corrected_estimate.plane, geometry, frame)
                if effective else {key: None for key in ENTITIES}
            )
            parameters = None if fit is None else np.asarray(
                fit["parameters_translation_m_then_rotvec_rad"], dtype=np.float64,
            )
            row: dict[str, Any] = {
                "frame": frame,
                "coordinate_model": model,
                "fit_status": fit_status,
                "fit_reason": fit_reason,
                "tool_projected_pixels": int(np.sum(nominal_tool_depth > 0)),
                "tool_visible_pixels": int(visible.sum()),
                "tool_selected_depth_pixels": int(selected.sum()),
                "tool_sampled_point_count": int(len(points_tool)),
                "tool_valid_depth_fraction": float(selected.sum() / max(int(interior.sum()), 1)),
                "baseline_table_status": "OUTPUT" if baseline_estimate is not None else "FAILED",
                "baseline_table_reason": baseline_reason,
                "corrected_table_status": "OUTPUT" if corrected_estimate is not None else "FAILED",
                "corrected_table_reason": corrected_reason,
                "effective_comparison": effective,
                "table_sampled_candidate_points": table_counts.get("sampled_candidate_points"),
                "translation_x_mm": None if parameters is None else parameters[0] * 1000.0,
                "translation_y_mm": None if parameters is None else parameters[1] * 1000.0,
                "translation_z_mm": None if parameters is None else parameters[2] * 1000.0,
                "translation_norm_mm": None if fit is None else fit["translation_norm_m"] * 1000.0,
                "rotation_rotvec_x_deg": None if parameters is None else np.degrees(parameters[3]),
                "rotation_rotvec_y_deg": None if parameters is None else np.degrees(parameters[4]),
                "rotation_rotvec_z_deg": None if parameters is None else np.degrees(parameters[5]),
                "rotation_angle_deg": None if fit is None else fit["rotation_angle_deg"],
                **plane_fields("before", baseline_estimate if effective else None),
                **plane_fields("after", corrected_estimate if effective else None),
            }
            for entity in ENTITIES:
                before = baseline_distances[entity]
                after = corrected_distances[entity]
                row[f"{entity}_before_mm"] = None if before is None else before * 1000.0
                row[f"{entity}_after_mm"] = None if after is None else after * 1000.0
                row[f"{entity}_before_penetrating"] = (
                    None if before is None else before < -threshold
                )
                row[f"{entity}_after_penetrating"] = (
                    None if after is None else after < -threshold
                )
            rows.append(row)
            corrections[model].append({
                "frame": frame, "status": fit_status, "reason": fit_reason,
                "transform": None if fit is None else fit["transform"],
                "parameters": None if fit is None else fit["parameters_translation_m_then_rotvec_rad"],
                "fit_metrics": None if fit is None else fit["fit_metrics"],
                "identity_metrics": None if fit is None else fit["identity_metrics"],
            })
        if (frame + 1) % 10 == 0 or frame + 1 == count:
            print(f"{sample['key']}: {frame + 1}/{count}", flush=True)

    del projector
    return rows, corrections


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(data.min()), "p05": float(np.percentile(data, 5)),
        "median": float(np.median(data)), "p95": float(np.percentile(data, 95)),
        "maximum": float(data.max()), "range": float(np.ptp(data)),
        "mad": float(np.median(np.abs(data - np.median(data)))),
    }


def table_variation(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    offsets = [row[f"{prefix}_plane_offset_m"] * 1000.0 for row in rows]
    tilts = [row[f"{prefix}_plane_tilt_deg"] for row in rows]
    return {
        "offset_mm": distribution(offsets),
        "tilt_deg": distribution(tilts),
    }


def classify_model(rows: list[dict[str, Any]], frame_count: int) -> dict[str, Any]:
    effective = [row for row in rows if row["effective_comparison"]]
    unavailable = frame_count - len(effective)
    entity_summary = {}
    for entity in ENTITIES:
        before_values = [row[f"{entity}_before_mm"] for row in effective]
        after_values = [row[f"{entity}_after_mm"] for row in effective]
        before_count = sum(bool(row[f"{entity}_before_penetrating"]) for row in effective)
        after_count = sum(bool(row[f"{entity}_after_penetrating"]) for row in effective)
        entity_summary[entity] = {
            "before_penetrating_frames": before_count,
            "after_penetrating_frames": after_count,
            "before_minimum_mm": None if not before_values else float(min(before_values)),
            "after_minimum_mm": None if not after_values else float(min(after_values)),
        }
    any_before = sum(any(bool(row[f"{entity}_before_penetrating"]) for entity in ("brush", "bowl", "hand")) for row in effective)
    any_after = sum(any(bool(row[f"{entity}_after_penetrating"]) for entity in ("brush", "bowl", "hand")) for row in effective)
    deepest = None
    if effective:
        candidates = [
            (row[f"{entity}_after_mm"], int(row["frame"]), entity)
            for row in effective for entity in ("brush", "bowl", "left_hand", "right_hand")
        ]
        value, frame, entity = min(candidates)
        deepest = {"frame": frame, "entity": entity, "minimum_distance_mm": float(value)}
    if not effective:
        classification = "UNABLE_TO_DETERMINE"
    elif unavailable == 0 and any_after == 0:
        classification = "PENETRATION_ELIMINATED"
    elif any_after > 0 and (
        any_after < any_before
        or min(row["brush_after_mm"] for row in effective) > min(row["brush_before_mm"] for row in effective)
        or min(row["bowl_after_mm"] for row in effective) > min(row["bowl_before_mm"] for row in effective)
        or min(row["hand_after_mm"] for row in effective) > min(row["hand_before_mm"] for row in effective)
    ):
        classification = "PENETRATION_REDUCED_BUT_REMAINS"
    elif unavailable > 0 and any_after == 0:
        classification = "UNABLE_TO_DETERMINE_FULL_SEQUENCE"
    else:
        classification = "PENETRATION_REMAINS"
    frame0 = next((row for row in rows if row["frame"] == 0), None)
    successes = [row for row in rows if row["fit_status"] == "OUTPUT"]
    failures: dict[str, int] = {}
    for row in rows:
        if row["fit_status"] != "OUTPUT":
            failures[row["fit_reason"]] = failures.get(row["fit_reason"], 0) + 1
    return {
        "classification": classification,
        "frame_count": frame_count,
        "correction_output_frames": len(successes),
        "correction_failed_frames": frame_count - len(successes),
        "effective_comparison_frames": len(effective),
        "unchecked_frames": unavailable,
        "any_entity_before_penetrating_frames": any_before,
        "any_entity_after_penetrating_frames": any_after,
        "entities": entity_summary,
        "deepest_after": deepest,
        "frame0": frame0,
        "correction_translation_norm_mm": distribution([row["translation_norm_mm"] for row in successes]),
        "correction_rotation_angle_deg": distribution([row["rotation_angle_deg"] for row in successes]),
        "table_variation_same_effective_frames": {
            "before": table_variation(effective, "before") if effective else None,
            "after": table_variation(effective, "after") if effective else None,
        },
        "fit_failure_reasons": failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_corrections(path: Path, corrections: dict[str, list[dict[str, Any]]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for model in MODELS:
        records = corrections[model]
        transforms = np.full((len(records), 4, 4), np.nan, dtype=np.float64)
        parameters = np.full((len(records), 6), np.nan, dtype=np.float64)
        success = np.zeros(len(records), dtype=bool)
        reasons = []
        for index, row in enumerate(records):
            success[index] = row["status"] == "OUTPUT"
            reasons.append(row["reason"])
            if row["transform"] is not None:
                transforms[index] = np.asarray(row["transform"])
                parameters[index] = np.asarray(row["parameters"])
        prefix = model.lower()
        arrays[f"{prefix}_transform"] = transforms
        arrays[f"{prefix}_parameters_translation_m_then_rotvec_rad"] = parameters
        arrays[f"{prefix}_success"] = success
        arrays[f"{prefix}_failure_reason"] = np.asarray(reasons, dtype="U512")
    arrays["frame"] = np.arange(len(corrections[MODELS[0]]), dtype=np.int64)
    np.savez_compressed(path, **arrays)


def summary_markdown(summary: dict[str, Any]) -> str:
    solver_name = summary.get("solver", {}).get(
        "method", "CURRENT_FIXED_SINGLE_FRAME",
    )
    lines = [
        "# Brush 单帧校准临时修正与桌面穿透检查",
        "",
        f"**最终结论：`{summary['overall_classification']}`。**",
        "",
        f"本轮逐帧独立使用 `{solver_name}`；未修改原始 Depth、相机参数、物体姿态、手轨迹或 active `SupportSurfaceContract`。世界坐标与相机坐标结果均完整保留，未按穿透结果挑选模型。桌面基线来自同一帧未修正原始深度，未使用旧的 bowl-bottom support 定义。",
        "",
        f"- 穿透门槛：`-{summary['penetration_tolerance_mm']:.3f} mm`（更负才计为穿透）",
        f"- 样本帧数：`{summary['frame_count']}`",
        f"- MINK / physics / Replay / MPC / RL：全部 `0`",
        "",
        "| 坐标模型 | 修正输出帧 | 同帧可比较 | 未检查 | 修正前任一实体穿透帧 | 修正后任一实体穿透帧 | 最深修正后穿透 | 分类 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        row = summary["models"][model]
        deepest = row["deepest_after"]
        deepest_text = "n/a" if deepest is None else f"{deepest['minimum_distance_mm']:.3f} mm @ frame {deepest['frame']} ({deepest['entity']})"
        lines.append(
            f"| {model} | {row['correction_output_frames']} | {row['effective_comparison_frames']} | "
            f"{row['unchecked_frames']} | {row['any_entity_before_penetrating_frames']} | "
            f"{row['any_entity_after_penetrating_frames']} | {deepest_text} | {row['classification']} |"
        )
    lines += ["", "## 各实体穿透帧数（相同有效帧）", ""]
    for model in MODELS:
        lines += [
            f"### {model}", "",
            "| 实体 | 修正前穿透帧 | 修正后穿透帧 | 修正前最小距离 mm | 修正后最小距离 mm |",
            "|---|---:|---:|---:|---:|",
        ]
        for entity in ("brush", "bowl", "left_hand", "right_hand", "hand"):
            row = summary["models"][model]["entities"][entity]
            lines.append(
                f"| {entity} | {row['before_penetrating_frames']} | {row['after_penetrating_frames']} | "
                f"{row['before_minimum_mm']:.3f} | {row['after_minimum_mm']:.3f} |"
            )
        frame0 = summary["models"][model]["frame0"]
        lines += ["", "第 0 帧："]
        if frame0 is None or not frame0["effective_comparison"]:
            reason = "missing row" if frame0 is None else (frame0["fit_reason"] or frame0["corrected_table_reason"] or frame0["baseline_table_reason"])
            lines.append(f"- 未能检查：`{reason}`")
        else:
            lines.append(
                "- " + ", ".join(
                    f"{entity} {frame0[f'{entity}_before_mm']:.3f}→{frame0[f'{entity}_after_mm']:.3f} mm"
                    for entity in ("brush", "bowl", "left_hand", "right_hand")
                )
            )
        variation = summary["models"][model]["table_variation_same_effective_frames"]
        if variation["before"] is not None:
            before = variation["before"]["offset_mm"]
            after = variation["after"]["offset_mm"]
            lines += [
                "", "桌面跨帧 offset 波动（相同有效帧）：",
                f"- 修正前 range `{before['range']:.3f} mm`，MAD `{before['mad']:.3f} mm`。",
                f"- 修正后 range `{after['range']:.3f} mm`，MAD `{after['mad']:.3f} mm`。",
            ]
    lines += [
        "", "## 解释边界", "",
        "- 部分帧改善不能视为全程通过。",
        "- 穿透减轻只说明该临时点云解释改变了估计桌面，不证明标定正确。",
        "- correction objective 只使用可见刷子表面深度；桌面、碗底和刷子底部距离均未参与拟合。碗只作事后验证。",
        "- 本轮没有把任何 correction 或桌面写入 active runtime。",
    ]
    return "\n".join(lines) + "\n"


def write_hashes(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.name}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg, current, taco, table = load_contracts(config_path)
    sample = resolve_sample(cfg, taco)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    active_path = Path(cfg["upstream"]["active_support_contract"]).resolve(strict=True)
    active_before = sha256(active_path)
    depth_before = sha256(sample["depth_video"])
    rows, corrections = process(cfg, current, taco, table, sample, output)
    if not any(row["fit_status"] == "OUTPUT" for row in rows):
        all_failed = True
    else:
        all_failed = False
    write_csv(output / "per_frame_results.csv", rows)
    write_corrections(output / "per_frame_corrections.npz", corrections)
    models = {
        model: classify_model([row for row in rows if row["coordinate_model"] == model], sample["expected_frames"])
        for model in MODELS
    }
    classifications = [models[model]["classification"] for model in MODELS]
    if all_failed:
        overall = "UNABLE_TO_DETERMINE_ALL_CORRECTIONS_FAILED"
    elif all(value == "PENETRATION_ELIMINATED" for value in classifications):
        overall = "PENETRATION_ELIMINATED"
    elif any(value == "PENETRATION_REDUCED_BUT_REMAINS" for value in classifications):
        overall = "PENETRATION_REDUCED_BUT_REMAINS"
    elif any(value == "PENETRATION_REMAINS" for value in classifications):
        overall = "PENETRATION_REMAINS"
    else:
        overall = "UNABLE_TO_DETERMINE_FULL_SEQUENCE"
    summary = {
        "schema": (
            "taco_brush_open3d_single_frame_calibration_penetration_summary_v1"
            if "solver" in cfg
            else "taco_brush_single_frame_calibration_penetration_summary_v1"
        ),
        "classification_options": [
            "PENETRATION_ELIMINATED", "PENETRATION_REDUCED_BUT_REMAINS",
            "PENETRATION_REMAINS", "UNABLE_TO_DETERMINE_FULL_SEQUENCE",
        ],
        "overall_classification": overall,
        "sample": sample["key"],
        "frame_count": sample["expected_frames"],
        "penetration_tolerance_mm": float(cfg["evaluation"]["penetration_tolerance_m"]) * 1000.0,
        "coordinate_model_selection_performed": False,
        "all_corrections_failed": all_failed,
        "models": models,
        "immutability": {
            "depth_sha256_before": depth_before,
            "depth_sha256_after": sha256(sample["depth_video"]),
            "depth_unchanged": depth_before == sha256(sample["depth_video"]),
            "active_support_contract_sha256_before": active_before,
            "active_support_contract_sha256_after": sha256(active_path),
            "active_support_contract_unchanged": active_before == sha256(active_path),
            "camera_parameters_modified": False,
            "object_poses_modified": False,
            "hand_trajectories_modified": False,
        },
        "runtime_counts": {"mink": 0, "physics": 0, "replay": 0, "mpc": 0, "rl": 0},
        "solver": cfg.get("solver", {"method": "CURRENT_FIXED_SINGLE_FRAME"}),
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    paths = {
        "config": config_path,
        "runner": Path(__file__),
        "current_single_frame_config": Path(cfg["upstream"]["current_single_frame_config"]),
        "taco_calibration_config": Path(cfg["upstream"]["taco_calibration_config"]),
        "table_estimator_config": Path(cfg["upstream"]["table_estimator_config"]),
        "active_support_contract": active_path,
        "calibration_module": RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py",
        "visibility_module": RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_visibility.py",
        "table_module": RL_ROOT / "src/egoengine_repro/scene/support_surface_estimation.py",
        "depth_video": sample["depth_video"],
        "intrinsic": sample["intrinsic"], "extrinsic": sample["extrinsic"],
        "tool_pose": sample["tool_pose"], "target_pose": sample["target_pose"],
        "tool_mesh": sample["tool_mesh"], "target_mesh": sample["target_mesh"],
        "left_hand": sample["left_hand"], "right_hand": sample["right_hand"],
        "left_shape": sample["left_shape"], "right_shape": sample["right_shape"],
    }
    if "solver" in cfg:
        paths["open3d_calibration_module"] = (
            RL_ROOT / "src/egoengine_repro/evaluation/open3d_calibration.py"
        )
        paths["open3d_wheel"] = Path(current["open3d"]["wheel"])
    write_json(output / "source_pins.json", {
        "schema": (
            "taco_brush_open3d_single_frame_calibration_penetration_source_pins_v1"
            if "solver" in cfg
            else "taco_brush_single_frame_calibration_penetration_source_pins_v1"
        ),
        "repository_head_before_audit": git_head(REPO_ROOT),
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    })
    write_json(output / "config_consumption_audit.json", {
        "schema": (
            "taco_brush_open3d_single_frame_calibration_penetration_config_consumption_v1"
            if "solver" in cfg
            else "taco_brush_single_frame_calibration_penetration_config_consumption_v1"
        ),
        "exact_key_validation": True,
        "single_run": True,
        "parameter_sweep_count": 0,
        "coordinate_models_both_reported": True,
        "coordinate_model_selection_performed": False,
        "solver": cfg.get("solver", {"method": "CURRENT_FIXED_SINGLE_FRAME"}),
        "fit_parameters": (
            open3d_fit_kwargs(current) if "solver" in cfg else fit_kwargs(current)
        ),
        "visibility_parameters": {
            "uncertainty_margin_m": VISIBILITY_UNCERTAINTY_M,
            "interior_erosion_px": taco["anchors"]["interior_erosion_px"],
            "spatial_stride_px": taco["anchors"]["spatial_stride_px"],
            "occluders": ["left_hand", "right_hand", "target_bowl"],
        },
        "table_parameters": {
            "foreground": table["foreground"], "roi": table["roi"],
            "sampling": table["sampling"], "plane_fit": table["plane_fit"],
        },
        "forbidden_fit_inputs": ["table", "bowl bottom", "brush bottom"],
    })
    write_hashes(output)
    print(json.dumps({
        "overall_classification": overall,
        "models": {model: {
            key: models[model][key] for key in (
                "classification", "correction_output_frames", "effective_comparison_frames",
                "unchecked_frames", "any_entity_before_penetrating_frames",
                "any_entity_after_penetrating_frames", "deepest_after",
            )
        } for model in MODELS},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
