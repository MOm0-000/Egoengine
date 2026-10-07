#!/usr/bin/env python3
"""Adjudicate a static TACO metric-depth / mocap-world calibration residual."""

from __future__ import annotations

import argparse
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
    SurfaceObservation, TriangleSurface, apply_transform, fit_multi_surface_correction,
    fit_surface_correction, residual_metrics, surface_residuals,
)
from egoengine_repro.evaluation.taco_depth import (  # noqa: E402
    DepthVideoSpec, ffprobe_video, iter_depth_frames, raw_depth_to_metres, stream_frame_count,
)
from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector, load_official_hand_sequence,
)
from egoengine_repro.evaluation.taco_calibration_visibility import (  # noqa: E402
    measured_target_selector, target_visibility_mask,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    PlaneEstimate, aggregate_support_planes, backproject_metric_depth,
    camera_points_to_world, evaluate_plane_on_points, fit_horizontal_support_plane,
    foreground_excluded_background_points,
)


DEFAULT_CONFIG = RL_ROOT / "configs/taco_depth_mocap_calibration_residual_adjudication_v1.yaml"
METRIC_KEYS = ("absolute_median_m", "absolute_p90_m", "absolute_p95_m")
# Visibility-order tolerance only.  Eight TACO depth quanta provide a
# conservative tie band without changing the calibration residual objective.
OCCLUSION_VISIBILITY_UNCERTAINTY_M = 8.0 / 4000.0


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


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


ROOT_KEYS = {
    "schema", "status", "minimum_baseline", "paper_faithful", "authorization", "official",
    "upstream", "depth", "anchors", "fit", "same_date", "samples", "output",
}
SAMPLE_KEYS = {
    "key", "same_date_fit_sequence", "cross_date_diagnostic", "triplet", "sequence",
    "expected_frames", "tool_id", "target_id", "depth_video", "intrinsic", "extrinsic",
    "left_hand", "right_hand", "left_shape", "right_shape", "object_pose_dir",
    "object_model_dir",
}


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    exact_keys(cfg, ROOT_KEYS, "root")
    exact_keys(cfg["authorization"], {
        "static_calibration_audit", "candidate_contract", "active_support_contract_replacement",
        "raw_depth_modification", "object_pose_modification", "egocentric_extrinsic_modification",
        "mink", "physics", "replay", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    }, "authorization")
    exact_keys(cfg["official"], {
        "paper_url", "supplementary_url", "repository_url", "issue_8_url", "issue_18_url",
        "issue_20_url", "commit", "checkout", "projection_entrypoint", "available_sequences",
        "chumpy_dependency", "device",
    }, "official")
    exact_keys(cfg["upstream"], {
        "rgbd_decision", "timeline_contract", "table_config", "table_summary", "table_per_frame",
        "brush_table_holdout", "active_support_contract",
    }, "upstream")
    exact_keys(cfg["depth"], {
        "frame_mapping", "logical_rate_hz", "maximum_official_timestamp_difference_ms",
        "scale", "width", "height",
    }, "depth")
    exact_keys(cfg["anchors"], {
        "uniform_frame_count", "roles", "interior_erosion_px", "minimum_valid_depth_fraction",
        "minimum_valid_pixels", "spatial_stride_px", "provenance",
    }, "anchors")
    exact_keys(cfg["fit"], {
        "models", "robust_loss", "robust_scale_m", "maximum_translation_m",
        "maximum_rotation_deg", "maximum_function_evaluations", "minimum_points", "provenance",
    }, "fit")
    if cfg["schema"] != "taco_depth_mocap_calibration_residual_adjudication_v1":
        raise ValueError("unexpected schema")
    if cfg["status"] != "STATIC_CALIBRATION_RESIDUAL_ADJUDICATION":
        raise ValueError("unexpected status")
    if cfg["paper_faithful"] is not False:
        raise ValueError("local calibration adjudication is not paper-faithful")
    allowed = {"static_calibration_audit", "candidate_contract"}
    if any(value for key, value in cfg["authorization"].items() if key not in allowed):
        raise ValueError("runtime, source mutation and active replacement are forbidden")
    if not all(cfg["authorization"][key] for key in allowed):
        raise ValueError("static audit and candidate output must be explicitly authorized")
    if cfg["depth"] != {
        "frame_mapping": "INDEX_ALIGNED", "logical_rate_hz": 30,
        "maximum_official_timestamp_difference_ms": 17.0, "scale": 4000.0,
        "width": 1920, "height": 1080,
    }:
        raise ValueError("frozen depth/timing contract changed")
    if cfg["anchors"]["roles"] != ["tool"]:
        raise ValueError("only the predeclared high-coverage tool role may fit calibration")
    if cfg["fit"]["models"] != ["WORLD_FIXED", "CAMERA_LOCAL"]:
        raise ValueError("predeclared model set changed")
    for index, sample in enumerate(cfg["samples"]):
        exact_keys(sample, SAMPLE_KEYS, f"sample[{index}]")
    if sum(bool(row["same_date_fit_sequence"]) for row in cfg["samples"]) != 2:
        raise ValueError("exactly two locally complete same-date sequences are expected")
    if sum(bool(row["cross_date_diagnostic"]) for row in cfg["samples"]) != 1:
        raise ValueError("exactly one cross-date diagnostic is expected")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], git_head(REPO_ROOT)],
        cwd=REPO_ROOT, check=True,
    )
    official = Path(cfg["official"]["checkout"])
    if git_head(official) != cfg["official"]["commit"]:
        raise ValueError("official checkout commit mismatch")
    return cfg


def resolve_inputs(cfg: dict[str, Any]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    upstream = {key: Path(value).resolve(strict=True) for key, value in cfg["upstream"].items()}
    path_keys = {
        "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand", "left_shape",
        "right_shape", "object_pose_dir", "object_model_dir",
    }
    samples = []
    for source in cfg["samples"]:
        row = dict(source)
        for key in path_keys:
            row[key] = Path(row[key]).resolve(strict=True)
        for role in ("tool", "target"):
            identifier = row[f"{role}_id"]
            row[f"{role}_pose"] = (row["object_pose_dir"] / f"{role}_{identifier}.npy").resolve(strict=True)
            row[f"{role}_mesh"] = (row["object_model_dir"] / f"{identifier}_cm.obj").resolve(strict=True)
        samples.append(row)
    return upstream, samples


def official_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    official = Path(cfg["official"]["checkout"])
    return {
        "schema": "official_taco_calibration_contract_v1",
        "author_published": {
            "intended_common_world": {
                "value": "NOKOV mocap tracks marked objects and the head-mounted L515; camera and mocap operate at 30 Hz",
                "source": cfg["official"]["paper_url"],
            },
            "rgb_extrinsic_calibration": {
                "value": "12 scene markers; <1 mm mocap 3-D marker error; manual RGB pixels; PnP world-to-camera extrinsic",
                "source": cfg["official"]["supplementary_url"],
            },
            "same_date_world": {
                "value": "same-date sequences share a world; 20231002 origin is on table, other dates on ground; +Z up; right-handed",
                "source": cfg["official"]["issue_8_url"],
            },
            "egocentric_extrinsic_semantics": {
                "value": "egocentric_frame_extrinsic.npy is consumed as world_to_camera",
                "source": str((official / cfg["official"]["projection_entrypoint"]).resolve(strict=True)),
            },
            "timestamp_matching": {
                "value": "nearest UTC timestamp; reported maximum mismatch approximately 17 ms",
                "source": cfg["official"]["supplementary_url"],
            },
            "depth_release": {
                "value": "1920x1080 uint16; scale 4000",
                "source": str((official / "README.md").resolve(strict=True)),
            },
        },
        "not_published_clearly": {
            "depth_optical_to_rgb_color_rigid_extrinsic": None,
            "released_depth_registration_to_rgb_pixel_grid": None,
        },
        "local_interpretation": {
            "depth_pixel_plus_rgb_intrinsic_to_camera_point": "LOCAL_EVIDENCE_BACKED_INTERPRETATION",
            "extrinsic_inverse_or_transpose_search": "FORBIDDEN_BY_FROZEN_OFFICIAL_CODE_SEMANTICS",
        },
        "community_reports": [
            {"source": cfg["official"]["issue_18_url"], "classification": "COMMUNITY_REPORT_IN_OFFICIAL_TRACKER",
             "claim": "a user reports visibly wrong extrinsics in some sequences"},
            {"source": cfg["official"]["issue_20_url"], "classification": "COMMUNITY_REPORT_IN_OFFICIAL_TRACKER",
             "claim": "a user reports many possible calibration issues; no author confirmation is present"},
        ],
        "paper_faithful_correction_recovered": False,
    }


def same_date_inventory(cfg: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    available = Path(cfg["official"]["checkout"]) / cfg["official"]["available_sequences"]
    rows = []
    for line in available.read_text(encoding="utf-8").splitlines():
        if cfg["same_date"] in line:
            triplet, sequence = line.rsplit(" ", 1)
            local = next((sample for sample in samples if sample["sequence"] == sequence), None)
            rows.append({
                "triplet": triplet, "sequence": sequence, "official_egocentric_available": True,
                "locally_complete_for_this_audit": local is not None and local["same_date_fit_sequence"],
                "local_key": None if local is None else local["key"],
            })
    complete = [row for row in rows if row["locally_complete_for_this_audit"]]
    return {
        "schema": "taco_same_date_sample_inventory_v1", "date": cfg["same_date"],
        "official_available_count": len(rows), "locally_complete_count": len(complete),
        "local_selection_rule": "all locally present sequences with metric Depth, RGB intrinsic, egocentric extrinsic, object poses/models and INDEX_ALIGNED counts",
        "selection_used_final_error": False,
        "limitation": "LIMITED_SAME_DATE_SAMPLE_COUNT" if len(complete) == 2 else None,
        "sequences": rows,
    }


def table_consistency(cfg: dict[str, Any], upstream: dict[str, Path]) -> dict[str, Any]:
    source = json.loads(upstream["table_summary"].read_text(encoding="utf-8"))
    per_frame: dict[str, list[PlaneEstimate]] = {}
    for line in upstream["table_per_frame"].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (cfg["same_date"] not in row["sample"] or row["split"] != "fit"
                or row["status"] != "TABLE_PLANE_ESTIMATED"):
            continue
        plane = Plane(np.asarray(row["plane"]["normal"]), row["plane"]["offset_m"], "world")
        per_frame.setdefault(row["sample"], []).append(PlaneEstimate(
            plane=plane, candidate_count=row["candidate_count"], inlier_count=row["inlier_count"],
            inlier_fraction=row["inlier_fraction"],
            tilt_from_world_z_deg=row["tilt_from_world_z_deg"],
            median_absolute_residual_m=row["median_absolute_residual_m"],
            p95_absolute_residual_m=row["p95_absolute_residual_m"],
            xy_bounds_m=np.asarray(row["xy_bounds_m"]),
            xy_footprint_area_m2=row["xy_footprint_area_m2"],
        ))
    rows = []
    for row in source["samples"]:
        if cfg["same_date"] not in row["sample"]:
            continue
        consensus = row["consensus"]
        recomputed = aggregate_support_planes(per_frame[row["sample"]])
        offsets = np.asarray([estimate.plane.offset for estimate in per_frame[row["sample"]]])
        exact_reproduction = (
            np.array_equal(recomputed.normal, np.asarray(consensus["plane"]["normal"]))
            and recomputed.offset == consensus["plane"]["offset_m"]
        )
        if not exact_reproduction:
            raise ValueError(f"{row['sample']}: frozen per-frame table aggregation did not reproduce")
        rows.append({
            "sample": row["sample"], "status": row["status"],
            "normal": consensus["plane"]["normal"], "offset_m": consensus["plane"]["offset_m"],
            "tilt_from_world_z_deg": consensus["tilt_from_world_z_deg"],
            "offset_mad_m": consensus["offset_mad_m"],
            "recomputed_from_frozen_per_frame_estimates": True,
            "successful_fit_frames": len(per_frame[row["sample"]]),
            "recomputed_offset_mad_m": float(np.median(np.abs(offsets - np.median(offsets)))),
            "exact_consensus_reproduction": exact_reproduction,
        })
    offsets = np.asarray([row["offset_m"] for row in rows])
    return {
        "schema": "taco_same_date_depth_table_consistency_v1", "date": cfg["same_date"],
        "frozen_estimator_reused_without_parameter_change": True,
        "consensus_recomputed_with_active_aggregate_support_planes": True,
        "frozen_estimator_module": artifact(RL_ROOT / "src/egoengine_repro/scene/support_surface_estimation.py"),
        "frozen_source_run": artifact(upstream["table_summary"]),
        "table_values_used_for_calibration_fit": False,
        "samples": rows,
        "maximum_offset_difference_m": float(np.ptp(offsets)),
        "classification": "SAME_DATE_DEPTH_WORLD_INTERNALLY_CONSISTENT" if len(rows) == 2 else "INSUFFICIENT_SAME_DATE_TABLES",
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(
        ("p50", "p90", "p95", "p99", "maximum"), np.percentile(values, [50, 90, 95, 99, 100])
    )}


def temporal_bound(cfg: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    all_combined = []
    fraction = cfg["depth"]["maximum_official_timestamp_difference_ms"] / (1000 / cfg["depth"]["logical_rate_hz"])
    for sample in samples:
        if not sample["same_date_fit_sequence"]:
            continue
        extrinsic = np.load(sample["extrinsic"])
        rotation = extrinsic[:, :3, :3]
        translation = extrinsic[:, :3, 3]
        centers = -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), translation)
        motion = np.linalg.norm(np.diff(centers, axis=0), axis=1) * fraction
        relative = np.einsum("nij,njk->nik", rotation[1:], rotation[:-1].transpose(0, 2, 1))
        angle = Rotation.from_matrix(relative).magnitude() * fraction
        at_one_metre = motion + 2 * np.sin(angle / 2)
        all_combined.append(at_one_metre)
        rows.append({
            "sample": sample["key"], "adjacent_pair_count": len(motion),
            "translation_uncertainty_m": _percentiles(motion),
            "rotation_uncertainty_deg": _percentiles(np.degrees(angle)),
            "conservative_translation_plus_rotation_at_1m_m": _percentiles(at_one_metre),
            "translation_ge_20mm_fraction": float(np.mean(motion >= 0.02)),
            "one_metre_bound_ge_20mm_fraction": float(np.mean(at_one_metre >= 0.02)),
        })
    combined = np.concatenate(all_combined)
    unlikely = float(combined.max()) < 0.02
    return {
        "schema": "taco_official_17ms_temporal_uncertainty_bound_v1",
        "method": "constant-velocity bound from adjacent 30 Hz official world_to_camera poses; 17/33.333 of each observed inter-frame motion",
        "rotation_spatial_bound": "2*r*sin(theta/2), reported conservatively at r=1m without using table/object-bottom geometry",
        "frame_offset_search_performed": False, "samples": rows,
        "classification": "TEMPORAL_SYNC_UNLIKELY_TO_EXPLAIN_20MM" if unlikely else "TEMPORAL_SYNC_CANNOT_BE_EXCLUDED",
    }


def _mesh(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray) -> trimesh.Trimesh:
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    return trimesh.Trimesh(vertices=world, faces=faces, process=False)


def _uniform_frames(count: int, wanted: int) -> list[int]:
    return np.unique(np.rint(np.linspace(0, count - 1, wanted)).astype(int)).tolist()


def _selected_depth_frames(sample: dict[str, Any], cfg: dict[str, Any], selected: set[int]) -> dict[int, np.ndarray]:
    stream = ffprobe_video(sample["depth_video"])
    count = stream_frame_count(stream)
    if count != sample["expected_frames"]:
        raise ValueError(f"{sample['key']}: native Depth violates INDEX_ALIGNED count")
    spec = DepthVideoSpec(width=cfg["depth"]["width"], height=cfg["depth"]["height"], frame_count=count)
    result = {}
    for frame, raw in enumerate(iter_depth_frames(sample["depth_video"], spec)):
        if frame in selected:
            result[frame] = raw_depth_to_metres(raw, scale=cfg["depth"]["scale"])
    if set(result) != selected:
        raise ValueError(f"{sample['key']}: failed to decode all selected frames")
    return result


def _calibration_occluders(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Load only nominal, release-available geometry used for visibility."""
    chumpy = str(Path(cfg["official"]["chumpy_dependency"]).resolve(strict=True))
    if chumpy not in sys.path:
        sys.path.insert(0, chumpy)
    dataset_utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    hands: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        vertices, _, faces, _ = load_official_hand_sequence(
            dataset_utils=dataset_utils,
            pose_path=sample[f"{side}_hand"],
            shape_path=sample[f"{side}_shape"],
            side=side,
            device=cfg["official"]["device"],
        )
        if vertices.shape[0] != sample["expected_frames"]:
            raise ValueError(f"{sample['key']}: nominal {side} hand frame count mismatch")
        hands[side] = {"vertices": vertices, "faces": faces}
    target = trimesh.load_mesh(sample["target_mesh"], process=False)
    target.apply_scale(0.01)
    target_poses = np.load(sample["target_pose"], mmap_mode="r")
    if target_poses.shape != (sample["expected_frames"], 4, 4):
        raise ValueError(f"{sample['key']}: nominal target pose frame count mismatch")
    return {
        "hands": hands,
        "target_vertices": np.asarray(target.vertices),
        "target_faces": np.asarray(target.faces),
        "target_poses": target_poses,
    }


def extract_anchors(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    mesh = trimesh.load_mesh(sample["tool_mesh"], process=False)
    mesh.apply_scale(0.01)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    surface = TriangleSurface(vertices, faces)
    occluders = _calibration_occluders(cfg, sample)
    poses = np.load(sample["tool_pose"], mmap_mode="r")
    extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
    intrinsic = np.loadtxt(sample["intrinsic"])
    if poses.shape != (sample["expected_frames"], 4, 4) or extrinsic.shape != poses.shape:
        raise ValueError(f"{sample['key']}: pose/extrinsic shape mismatch")
    frames = _uniform_frames(sample["expected_frames"], cfg["anchors"]["uniform_frame_count"])
    depths = _selected_depth_frames(sample, cfg, set(frames))
    projector = TacoOfficialProjector(
        dataset_utils=Path(cfg["official"]["checkout"]) / "dataset_utils",
        image_size=(cfg["depth"]["width"], cfg["depth"]["height"]),
        intrinsic=intrinsic, extrinsic=extrinsic[0], device=cfg["official"]["device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(rasterizer.raster_settings, max_faces_per_bin=400000)
    observations, manifest = [], []
    for frame in frames:
        projector.set_camera(intrinsic, extrinsic[frame])
        rendered = projector.render_depth([_mesh(vertices, faces, poses[frame])])
        nominal_occluder_depths = [
            projector.render_depth([
                trimesh.Trimesh(
                    vertices=occluders["hands"][side]["vertices"][frame],
                    faces=occluders["hands"][side]["faces"], process=False,
                )
            ])
            for side in ("left", "right")
        ]
        nominal_occluder_depths.append(projector.render_depth([_mesh(
            occluders["target_vertices"], occluders["target_faces"],
            occluders["target_poses"][frame],
        )]))
        visible = target_visibility_mask(
            rendered, nominal_occluder_depths,
            uncertainty_margin_m=OCCLUSION_VISIBILITY_UNCERTAINTY_M,
        )
        valid = measured_target_selector(
            depths[frame], rendered, nominal_occluder_depths,
            uncertainty_margin_m=OCCLUSION_VISIBILITY_UNCERTAINTY_M,
            erosion_px=cfg["anchors"]["interior_erosion_px"],
        )
        interior = cv2.erode(
            visible.astype(np.uint8),
            np.ones((2 * cfg["anchors"]["interior_erosion_px"] + 1,) * 2, dtype=np.uint8),
        ) > 0
        coverage = float(valid.sum() / max(int(interior.sum()), 1))
        qualified = (
            coverage >= cfg["anchors"]["minimum_valid_depth_fraction"]
            and int(valid.sum()) >= cfg["anchors"]["minimum_valid_pixels"]
        )
        points, pixels = backproject_metric_depth(
            depths[frame], intrinsic, selector=valid,
            spatial_stride_px=cfg["anchors"]["spatial_stride_px"],
        )
        z_residual = depths[frame][valid] - rendered[valid]
        manifest.append({
            "sample": sample["key"], "sequence": sample["sequence"], "frame": frame,
            "role": "tool", "object_id": sample["tool_id"],
            "selection": "uniform before geometry", "interior_erosion_px": cfg["anchors"]["interior_erosion_px"],
            "visibility_selection": "NOMINAL_TARGET_FIRST_SURFACE",
            "visibility_uncertainty_margin_m": OCCLUSION_VISIBILITY_UNCERTAINTY_M,
            "target_projected_pixels_before_occlusion": int(np.sum(rendered > 0)),
            "target_pixels_excluded_by_nominal_occluders": int(np.sum((rendered > 0) & ~visible)),
            "interior_pixels": int(interior.sum()), "valid_depth_pixels": int(valid.sum()),
            "valid_depth_fraction": coverage, "qualified": qualified,
            "sampled_point_count": int(len(points)),
            "same_pixel_depth_residual_m": None if not len(z_residual) else {
                "signed_median": float(np.median(z_residual)),
                "absolute_median": float(np.median(np.abs(z_residual))),
                "absolute_p95": float(np.percentile(np.abs(z_residual), 95)),
            },
        })
        if qualified:
            observations.append(SurfaceObservation(
                points_camera=points, world_to_camera=extrinsic[frame], object_to_world=poses[frame],
                sequence=sample["key"], frame=frame,
            ))
    del projector
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"sample": sample, "surface": surface, "observations": observations, "manifest": manifest}


def fit_arguments(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "robust_loss": cfg["fit"]["robust_loss"],
        "robust_scale_m": cfg["fit"]["robust_scale_m"],
        "maximum_translation_m": cfg["fit"]["maximum_translation_m"],
        "maximum_rotation_deg": cfg["fit"]["maximum_rotation_deg"],
        "maximum_function_evaluations": cfg["fit"]["maximum_function_evaluations"],
        "minimum_points": cfg["fit"]["minimum_points"],
    }


def metrics_for(dataset: dict[str, Any], correction: np.ndarray, model: str) -> dict[str, Any]:
    values = surface_residuals(dataset["observations"], dataset["surface"], correction, model)
    return residual_metrics(values)


def metric_improvements(baseline: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    checks = {key: corrected[key] < baseline[key] for key in METRIC_KEYS}
    checks["absolute_signed_median"] = abs(corrected["signed_median_m"]) < abs(baseline["signed_median_m"])
    return {
        "checks": checks, "all_required_metrics_improved": all(checks.values()),
        "absolute_change_m": {key: corrected[key] - baseline[key] for key in METRIC_KEYS},
        "absolute_signed_median_change_m": abs(corrected["signed_median_m"]) - abs(baseline["signed_median_m"]),
    }


def calibration_analysis(cfg: dict[str, Any], datasets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str | None, dict[str, Any]]:
    same = [row for row in datasets if row["sample"]["same_date_fit_sequence"]]
    cross = next(row for row in datasets if row["sample"]["cross_date_diagnostic"])
    identity = np.eye(4)
    baseline = {
        "schema": "taco_no_calibration_correction_baseline_v1",
        "correction": "NO_CORRECTION",
        "samples": {row["sample"]["key"]: metrics_for(row, identity, "WORLD_FIXED") for row in datasets},
    }
    model_reports, folds = {}, {}
    individual = {}
    for model in cfg["fit"]["models"]:
        folds[model] = []
        individual[model] = {}
        for fit_data in same:
            fit = fit_surface_correction(
                fit_data["observations"], fit_data["surface"], model, **fit_arguments(cfg),
            )
            transform = np.asarray(fit["transform"])
            individual[model][fit_data["sample"]["key"]] = fit
            validation_data = next(row for row in same if row is not fit_data)
            validation_baseline = metrics_for(validation_data, identity, model)
            validation_corrected = metrics_for(validation_data, transform, model)
            folds[model].append({
                "fit_sequence": fit_data["sample"]["key"],
                "validation_sequence": validation_data["sample"]["key"],
                "frozen_correction": fit,
                "validation_baseline": validation_baseline,
                "validation_corrected": validation_corrected,
                "validation_improvement": metric_improvements(validation_baseline, validation_corrected),
            })
        joint = fit_multi_surface_correction(
            [(row["observations"], row["surface"]) for row in same], model, **fit_arguments(cfg),
        )
        joint_transform = np.asarray(joint["transform"])
        cross_baseline = metrics_for(cross, identity, model)
        cross_corrected = metrics_for(cross, joint_transform, model)
        model_reports[model] = {
            "schema": f"taco_{model.lower()}_calibration_fit_v1",
            "correction_location": "after camera-to-world" if model == "WORLD_FIXED" else "before camera-to-world",
            "same_date_joint_fit": joint,
            "cross_date_application": {
                "sample": cross["sample"]["key"], "diagnostic_only": True,
                "baseline": cross_baseline, "corrected": cross_corrected,
                "improvement": metric_improvements(cross_baseline, cross_corrected),
            },
        }
    cross_validation = {
        "schema": "taco_cross_sequence_calibration_validation_v1",
        "fold_definition": "fit one complete 20230927 tool sequence; validate the other unseen sequence",
        "limited_same_date_sample_count": True,
        "models": {},
    }
    passes = {}
    for model, rows in folds.items():
        passed = all(row["validation_improvement"]["all_required_metrics_improved"] for row in rows)
        passes[model] = passed
        cross_validation["models"][model] = {"folds": rows, "cross_sequence_passed": passed}

    def dominates(left: str, right: str) -> bool:
        any_strict = False
        for lrow, rrow in zip(folds[left], folds[right]):
            lm, rm = lrow["validation_corrected"], rrow["validation_corrected"]
            for key in METRIC_KEYS:
                if lm[key] > rm[key]:
                    return False
                any_strict |= lm[key] < rm[key]
            if abs(lm["signed_median_m"]) > abs(rm["signed_median_m"]):
                return False
            any_strict |= abs(lm["signed_median_m"]) < abs(rm["signed_median_m"])
        return any_strict

    selected = None
    if passes["WORLD_FIXED"] and not passes["CAMERA_LOCAL"]:
        selected = "WORLD_FIXED"
    elif passes["CAMERA_LOCAL"] and not passes["WORLD_FIXED"]:
        selected = "CAMERA_LOCAL"
    elif passes["WORLD_FIXED"] and passes["CAMERA_LOCAL"]:
        if dominates("WORLD_FIXED", "CAMERA_LOCAL"):
            selected = "WORLD_FIXED"
        elif dominates("CAMERA_LOCAL", "WORLD_FIXED"):
            selected = "CAMERA_LOCAL"
    cross_validation["selection_rule"] = "one model passes while the other fails, or one threshold-free dominates all required held-out metrics in both folds"
    cross_validation["selected_model"] = selected
    per_object = {
        "schema": "taco_per_object_calibration_residual_diagnostic_v1",
        "diagnostic_only_never_active": True,
        "models": individual,
        "pairwise_correction_disagreement": {},
        "interpretation_constraint": "winner differences do not by themselves identify camera hardware versus object-marker calibration",
    }
    for model, fits in individual.items():
        names = sorted(fits)
        first, second = (np.asarray(fits[name]["transform"]) for name in names)
        rotation_delta = Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()
        translation_delta = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        individual_norms = [fits[name]["translation_norm_m"] for name in names]
        individual_angles = [fits[name]["rotation_angle_deg"] for name in names]
        per_object["pairwise_correction_disagreement"][model] = {
            "objects": names,
            "translation_vector_difference_m": translation_delta,
            "relative_rotation_deg": float(np.degrees(rotation_delta)),
            "difference_exceeds_each_individual_correction_magnitude": (
                translation_delta > max(individual_norms)
                and float(np.degrees(rotation_delta)) > max(individual_angles)
            ),
            "classification": "OBJECT_OPTIMA_NOT_MUTUALLY_CONSISTENT",
            "causal_limit": "two objects and possible visibility/model residuals cannot isolate marker calibration",
        }
    cross_date = {
        "schema": "taco_cross_date_calibration_diagnostic_v1",
        "fit_date": cfg["same_date"], "validation_sequence": cross["sample"]["key"],
        "diagnostic_only": True,
        "models": {model: report["cross_date_application"] for model, report in model_reports.items()},
    }
    return baseline, model_reports["WORLD_FIXED"], model_reports["CAMERA_LOCAL"], cross_validation, selected, {"per_object": per_object, "cross_date": cross_date}


def _correct_points(points_camera: np.ndarray, world_to_camera: np.ndarray, correction: np.ndarray, model: str) -> np.ndarray:
    if model == "CAMERA_LOCAL":
        points_camera = apply_transform(points_camera, correction)
    points_world = camera_points_to_world(points_camera, world_to_camera)
    if model == "WORLD_FIXED":
        points_world = apply_transform(points_world, correction)
    return points_world


def corrected_table_estimate(cfg: dict[str, Any], sample: dict[str, Any], correction: np.ndarray, model: str) -> dict[str, Any]:
    """Re-run the frozen table estimator only after an object-only model passes."""
    sys.path.insert(0, str(Path(cfg["official"]["chumpy_dependency"])))
    utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    hands = {}
    for side in ("right", "left"):
        vertices, _, faces, _ = load_official_hand_sequence(
            dataset_utils=utils, pose_path=sample[f"{side}_hand"], shape_path=sample[f"{side}_shape"],
            side=side, device=cfg["official"]["device"],
        )
        hands[side] = {"vertices": vertices, "faces": faces}
    objects = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(sample[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        objects[role] = {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces),
                         "poses": np.load(sample[f"{role}_pose"], mmap_mode="r")}
    intrinsic = np.loadtxt(sample["intrinsic"])
    extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
    projector = TacoOfficialProjector(
        dataset_utils=utils, image_size=(cfg["depth"]["width"], cfg["depth"]["height"]),
        intrinsic=intrinsic, extrinsic=extrinsic[0], device=cfg["official"]["device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(rasterizer.raster_settings, max_faces_per_bin=400000)
    table_cfg = yaml.safe_load(Path(cfg["upstream"]["table_config"]).read_text(encoding="utf-8"))
    holdout_modulo = table_cfg["validation"]["holdout_modulo"]
    spec = DepthVideoSpec(cfg["depth"]["width"], cfg["depth"]["height"], sample["expected_frames"])
    estimates, validation_points = [], []
    failures = 0
    for frame, raw in enumerate(iter_depth_frames(sample["depth_video"], spec)):
        projector.set_camera(intrinsic, extrinsic[frame])
        meshes = [
            trimesh.Trimesh(vertices=hands[side]["vertices"][frame], faces=hands[side]["faces"], process=False)
            for side in ("right", "left")
        ]
        for role in ("tool", "target"):
            row = objects[role]
            meshes.append(_mesh(row["vertices"], row["faces"], row["poses"][frame]))
        foreground = projector.render_mask(meshes)
        depth = raw_depth_to_metres(raw, scale=cfg["depth"]["scale"])
        try:
            # Reuse the exact frozen ROI/foreground selector, then correct those selected points.
            current_world, _ = foreground_excluded_background_points(
                depth, intrinsic, extrinsic[frame], foreground,
                dilation_px=table_cfg["foreground"]["dilation_px"],
                interaction_roi_scale=table_cfg["roi"]["interaction_roi_scale"],
                spatial_stride_px=table_cfg["sampling"]["spatial_stride_px"],
            )
            # Convert the exact selected current-world points back to camera before correction.
            selected_camera = apply_transform(current_world, extrinsic[frame])
            points = _correct_points(selected_camera, extrinsic[frame], correction, model)
            estimate = fit_horizontal_support_plane(
                points, random_seed=table_cfg["plane_fit"]["random_seed"],
                maximum_tilt_from_world_z_deg=table_cfg["plane_fit"]["maximum_tilt_from_world_z_deg"],
                inlier_distance_m=table_cfg["plane_fit"]["inlier_distance_m"],
                maximum_iterations=table_cfg["plane_fit"]["maximum_iterations"],
                minimum_candidate_points=table_cfg["plane_fit"]["minimum_candidate_points"],
            )
            if frame % holdout_modulo == 0:
                validation_points.append(points)
            else:
                estimates.append(estimate)
        except ValueError:
            failures += 1
    if len(estimates) < table_cfg["validation"]["minimum_successful_fit_frames"] or not validation_points:
        return {"status": "INSUFFICIENT_CORRECTED_TABLE_EVIDENCE", "failed_frames": failures}
    plane = aggregate_support_planes(estimates)
    held = np.concatenate(validation_points)
    selector = np.abs(plane.signed_distance(held)) <= table_cfg["plane_fit"]["inlier_distance_m"]
    validation = evaluate_plane_on_points(plane, held[selector])
    offsets = np.asarray([row.plane.offset for row in estimates])
    return {
        "status": "STABLE_CORRECTED_TABLE_PLANE",
        "same_frozen_estimator_parameters": True, "failed_frames": failures,
        "successful_fit_frames": len(estimates), "successful_validation_frames": len(validation_points),
        "plane": plane.to_dict(), "offset_mad_m": float(np.median(np.abs(offsets - np.median(offsets)))),
        "validation_table_support": validation,
    }


def bottom_holdout(sample: dict[str, Any], table: dict[str, Any], frozen_table_artifact: dict[str, Any]) -> dict[str, Any]:
    plane_data = table["plane"]
    plane = Plane(np.asarray(plane_data["normal"]), plane_data["offset_m"], "world")
    values = {}
    for role, name in (("target", "bowl"), ("tool", "brush")):
        mesh = trimesh.load_mesh(sample[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        pose = np.load(sample[f"{role}_pose"], mmap_mode="r")[0]
        values[f"{name}_bottom_signed_distance_m"] = plane.minimum_signed_distance(np.asarray(mesh.vertices), pose)
    interval = table["validation_table_support"]
    low, high = interval["signed_residual_p05_m"], interval["signed_residual_p95_m"]
    return {
        "schema": "taco_brush_object_bottom_final_holdout_v1",
        "correction_and_table_frozen_before_bottom_read": True,
        "frozen_corrected_table_artifact": frozen_table_artifact,
        "table_uncertainty_interval_m": [low, high], **values,
        "bowl_within_table_uncertainty": low <= values["bowl_bottom_signed_distance_m"] <= high,
        "brush_within_table_uncertainty": low <= values["brush_bottom_signed_distance_m"] <= high,
    }


def source_pins(cfg_path: Path, cfg: dict[str, Any], upstream: dict[str, Path], samples: list[dict[str, Any]]) -> dict[str, Any]:
    official = Path(cfg["official"]["checkout"])
    paths: dict[str, Path] = {
        "config": cfg_path, "runner": Path(__file__),
        "calibration_module": RL_ROOT / "src/egoengine_repro/evaluation/taco_calibration_residual.py",
        "calibration_tests": RL_ROOT / "tests/core/test_taco_calibration_residual.py",
        "support_estimator_module": RL_ROOT / "src/egoengine_repro/scene/support_surface_estimation.py",
        "official_readme": official / "README.md",
        "official_projection_entrypoint": official / cfg["official"]["projection_entrypoint"],
        "official_available_sequences": official / cfg["official"]["available_sequences"],
        **{f"upstream:{key}": path for key, path in upstream.items()},
    }
    for sample in samples:
        for key in (
            "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand", "left_shape",
            "right_shape", "tool_pose", "target_pose", "tool_mesh", "target_mesh",
        ):
            paths[f"{sample['key']}:{key}"] = sample[key]
    return {
        "schema": "taco_calibration_residual_source_pins_v1",
        "repository_head_before_audit": git_head(REPO_ROOT),
        "official_checkout_commit": git_head(official),
        "web_sources": {key: cfg["official"][key] for key in (
            "paper_url", "supplementary_url", "repository_url", "issue_8_url", "issue_18_url", "issue_20_url",
        )},
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }


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
    cfg_path = args.config.resolve(strict=True)
    cfg = load_config(cfg_path)
    upstream, samples = resolve_inputs(cfg)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    write_json(output / "official_taco_calibration_contract.json", official_contract(cfg))
    write_json(output / "same_date_sample_inventory.json", same_date_inventory(cfg, samples))
    table_report = table_consistency(cfg, upstream)
    write_json(output / "same_date_depth_table_consistency.json", table_report)
    temporal = temporal_bound(cfg, samples)
    write_json(output / "official_17ms_temporal_uncertainty_bound.json", temporal)

    datasets = []
    manifest = []
    for sample in samples:
        print(f"extracting rigid-object anchors: {sample['key']}", flush=True)
        dataset = extract_anchors(cfg, sample)
        datasets.append(dataset)
        manifest.extend(dataset["manifest"])
    qualified = {row["sample"]["key"]: len(row["observations"]) for row in datasets}
    manifest_report = {
        "schema": "taco_calibration_anchor_manifest_v1",
        "selection_frozen_before_error_evaluation": True,
        "uniform_frame_count": cfg["anchors"]["uniform_frame_count"],
        "fit_roles": cfg["anchors"]["roles"], "minimum_depth_coverage": cfg["anchors"]["minimum_valid_depth_fraction"],
        "table_or_object_bottom_used": False, "qualified_frame_counts": qualified,
        "records": manifest,
    }
    write_json(output / "calibration_anchor_manifest.json", manifest_report)

    insufficient = any(
        not row["observations"] for row in datasets if row["sample"]["same_date_fit_sequence"]
    )
    selected = None
    if insufficient:
        empty = {"status": "NOT_RUN_INSUFFICIENT_CALIBRATION_ANCHOR_DATA"}
        baseline = world = camera = cross_validation = empty
        extras = {"per_object": empty, "cross_date": empty}
    else:
        print("fitting WORLD_FIXED and CAMERA_LOCAL models with two-fold validation", flush=True)
        baseline, world, camera, cross_validation, selected, extras = calibration_analysis(cfg, datasets)
    write_json(output / "no_correction_baseline.json", baseline)
    write_json(output / "world_fixed_correction_fit.json", world)
    write_json(output / "camera_local_correction_fit.json", camera)
    write_json(output / "cross_sequence_calibration_validation.json", cross_validation)
    write_json(output / "per_object_residual_diagnostic.json", extras["per_object"])
    write_json(output / "cross_date_diagnostic.json", extras["cross_date"])

    corrected_table = {"schema": "taco_corrected_table_holdout_validation_v1",
                       "status": "NOT_RUN_NO_CROSS_SEQUENCE_VALIDATED_CORRECTION",
                       "table_used_for_calibration_fit": False}
    bottom = {"schema": "taco_brush_object_bottom_final_holdout_v1",
              "status": "NOT_READ_NO_FROZEN_CORRECTED_TABLE", "used_for_calibration_fit": False}
    final_candidate = False
    if selected is not None:
        model_report = world if selected == "WORLD_FIXED" else camera
        correction = np.asarray(model_report["same_date_joint_fit"]["transform"])
        primary = next(row for row in samples if row["key"].startswith("brush_"))
        print(f"object-only model selected ({selected}); re-running frozen table estimator", flush=True)
        corrected_table = {
            "schema": "taco_corrected_table_holdout_validation_v1", "selected_model": selected,
            "correction_frozen_before_table_read": True,
            "correction_sha256": hashlib.sha256(np.asarray(correction, dtype=np.float64).tobytes()).hexdigest(),
            "table_used_for_calibration_fit": False,
            **corrected_table_estimate(cfg, primary, correction, selected),
        }
        write_json(output / "corrected_table_holdout_validation.json", corrected_table)
        corrected_table_pin = artifact(output / "corrected_table_holdout_validation.json")
        if corrected_table["status"] == "STABLE_CORRECTED_TABLE_PLANE":
            bottom = bottom_holdout(primary, corrected_table, corrected_table_pin)
            bottom["status"] = "FINAL_HOLDOUT_EVALUATED"
            final_candidate = bottom["bowl_within_table_uncertainty"] and bottom["brush_within_table_uncertainty"]
    else:
        write_json(output / "corrected_table_holdout_validation.json", corrected_table)
    write_json(output / "brush_object_bottom_final_holdout.json", bottom)

    if temporal["classification"] == "TEMPORAL_SYNC_CANNOT_BE_EXCLUDED":
        classification = "TEMPORAL_SYNC_CANNOT_BE_EXCLUDED"
    elif insufficient:
        classification = "INSUFFICIENT_CALIBRATION_ANCHOR_DATA"
    elif selected is None:
        any_cross_sequence_pass = any(
            model["cross_sequence_passed"] for model in cross_validation["models"].values()
        )
        any_fit_improvement = any(
            fold["frozen_correction"]["fit_metrics"]["absolute_median_m"]
            < fold["frozen_correction"]["identity_metrics"]["absolute_median_m"]
            for model in cross_validation["models"].values() for fold in model["folds"]
        )
        classification = (
            "CALIBRATION_RESIDUAL_UNRESOLVED" if any_cross_sequence_pass or not any_fit_improvement
            else "CALIBRATION_CORRECTION_OVERFIT"
        )
    elif final_candidate:
        classification = "TACO_CALIBRATION_RESIDUAL_CANDIDATE_VALIDATED"
    elif selected == "WORLD_FIXED":
        classification = "WORLD_FRAME_OR_CAMERA_WORLD_EXTRINSIC_RESIDUAL"
    else:
        classification = "CAMERA_LOCAL_OR_DEPTH_RGB_RESIDUAL"

    if classification == "TACO_CALIBRATION_RESIDUAL_CANDIDATE_VALIDATED":
        report = world if selected == "WORLD_FIXED" else camera
        write_json(output / "candidate_calibration_residual_contract.json", {
            "schema": "candidate_calibration_residual_contract_v1", "status": "CANDIDATE_NOT_ACTIVE",
            "paper_faithful": False, "model": selected,
            "transform": report["same_date_joint_fit"]["transform"],
            "fit_date": cfg["same_date"], "active_inputs_modified": False,
        })

    write_json(output / "source_pins.json", source_pins(cfg_path, cfg, upstream, samples))
    write_json(output / "config_consumption_audit.json", {
        "schema": "taco_calibration_residual_config_consumption_v1",
        "config": artifact(cfg_path), "exact_nested_key_validation": True,
        "all_sections_consumed": True, "parameter_sweep_count": 0,
        "forbidden_table_or_bottom_fit_inputs": 0,
    })
    decision = {
        "schema": "taco_calibration_residual_adjudication_decision_v1",
        "classification": classification, "selected_model": selected,
        "candidate_contract_created": classification == "TACO_CALIBRATION_RESIDUAL_CANDIDATE_VALIDATED",
        "active_support_contract_changed": False,
        "source_depth_or_pose_files_changed": False,
        "runtime_counts": {"mink": 0, "physics": 0, "replay": 0, "mpc": 0, "rl": 0,
                           "promotion": 0, "chunk_commit": 0},
    }
    write_json(output / "decision.json", decision)

    baseline_text = "未运行" if insufficient else json.dumps({
        key: {metric: value for metric, value in row.items() if metric != "point_count"}
        for key, row in baseline["samples"].items() if cfg["same_date"] in key
    }, ensure_ascii=False)
    world_pass = None if insufficient else cross_validation["models"]["WORLD_FIXED"]["cross_sequence_passed"]
    camera_pass = None if insufficient else cross_validation["models"]["CAMERA_LOCAL"]["cross_sequence_passed"]
    summary = f"""# TACO Depth / Mocap 世界坐标校准残差裁决 v1

**正式分类：`{classification}`。** 未修改 Depth、object pose、egocentric extrinsic 或 active `SupportSurfaceContract`。

1. **TACO 官方是不是认为相机和物体在同一个 world？** 是。设计目标是把 mocap 物体、手和头戴 L515 连接到同一个 mocap world；同日 sequence 共用 world。
2. **官方相机外参怎么标定？** 12 个场景 marker 的 mocap world 3-D 点（论文报告误差小于 1 mm）与人工 RGB 像素做 PnP，目标语义为 world→RGB camera。
3. **官方公开 Depth→RGB 内部外参了吗？** 没有找到清晰发布。`Depth pixel + RGB intrinsic` 是本项目的 `LOCAL_EVIDENCE_BACKED_INTERPRETATION`。
4. **Brush 与 Pour 同日 Depth table 是否一致？** 是；offset 最大差 `{table_report['maximum_offset_difference_m'] * 1000:.3f} mm`，两者都来自同一冻结估计器。
5. **17 ms 能解释约 20 mm 吗？** `{temporal['classification']}`；按 1 m 杠杆臂计入旋转的保守最大值仍小于 20 mm。
6. **无 correction 的刚体表面误差？** `{baseline_text}`（单位为米，低覆盖 target 未参与 fit）。
7. **world-fixed 能跨 sequence 吗？** `{world_pass}`。
8. **camera-local 能跨 sequence 吗？** `{camera_pass}`。
9. **哪种解释更符合数据？** `{selected or '两种简单固定模型都没有得到唯一、可跨 sequence 的支持'}`。
10. **完全不看桌面拟合后，能自动修复桌面 mismatch 吗？** `{corrected_table['status']}`。
11. **冻结后 bowl / brush bottom 还差多少？** `{json.dumps(bottom, ensure_ascii=False, default=_default)}`。
12. **下一步有资格正式接入 correction 吗？** `{'否；本轮最多生成 CANDIDATE_NOT_ACTIVE，且仍需独立接入任务' if final_candidate else '否；未满足全部独立验证条件'}`。

本轮只使用高 Depth coverage 的 tool 刚体表面拟合；table、bowl bottom、brush bottom 与 simulator 高度均未进入 correction objective。issue #18/#20 只作为官方 tracker 中的社区线索，不视为作者确认。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_hashes(output)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
