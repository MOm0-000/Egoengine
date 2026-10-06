#!/usr/bin/env python3
"""Estimate TACO support planes from multiframe measured background depth."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.evaluation.taco_depth import (  # noqa: E402
    DepthVideoSpec, ffprobe_video, iter_depth_frames, raw_depth_to_metres, stream_frame_count,
)
from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector, load_official_hand_sequence,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from egoengine_repro.scene.support_surface_estimation import (  # noqa: E402
    PlaneEstimate, aggregate_support_planes, evaluate_plane_on_points,
    fit_horizontal_support_plane, foreground_excluded_background_points,
)


DEFAULT_CONFIG = RL_ROOT / "configs/taco_multiframe_table_depth_estimation_v1.yaml"


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


SAMPLE_KEYS = {
    "key", "primary", "triplet", "sequence", "expected_frames", "tool_id", "target_id",
    "root", "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand",
    "left_shape", "right_shape", "object_pose_dir", "object_model_dir",
}


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    exact_keys(cfg, {
        "schema", "status", "minimum_baseline", "paper_faithful", "authorization",
        "upstream", "official", "depth", "foreground", "roi", "sampling", "plane_fit",
        "validation", "samples", "output",
    }, "root")
    exact_keys(cfg["authorization"], {
        "support_plane_estimation", "active_support_contract_replacement", "mink", "physics",
        "replay", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    }, "authorization")
    exact_keys(cfg["upstream"], {
        "handoff", "decision", "timeline_contract", "depth_module", "support_module",
        "active_support_contract",
    }, "upstream")
    exact_keys(cfg["official"], {"commit", "checkout", "chumpy_dependency", "device"}, "official")
    exact_keys(cfg["depth"], {"frame_mapping", "logical_rate_hz", "scale", "width", "height"}, "depth")
    exact_keys(cfg["foreground"], {"dilation_px", "provenance"}, "foreground")
    exact_keys(cfg["roi"], {"interaction_roi_scale", "provenance"}, "roi")
    exact_keys(cfg["sampling"], {"spatial_stride_px", "provenance"}, "sampling")
    exact_keys(cfg["plane_fit"], {
        "random_seed", "maximum_tilt_from_world_z_deg", "inlier_distance_m",
        "maximum_iterations", "minimum_candidate_points", "maximum_consensus_offset_mad_m",
        "minimum_samples_with_stable_plane", "provenance",
    }, "plane_fit")
    exact_keys(cfg["validation"], {"holdout_modulo", "minimum_successful_fit_frames", "provenance"}, "validation")
    if cfg["schema"] != "taco_multiframe_table_depth_estimation_v1":
        raise ValueError("unexpected schema")
    if cfg["status"] != "ZERO_RUNTIME_MULTIFRAME_METRIC_DEPTH_TABLE_ESTIMATION":
        raise ValueError("unexpected runtime status")
    if cfg["paper_faithful"] is not False:
        raise ValueError("local estimator must not be labelled paper-faithful")
    if cfg["authorization"]["support_plane_estimation"] is not True:
        raise ValueError("this task must explicitly authorize only support-plane estimation")
    if any(value for key, value in cfg["authorization"].items() if key != "support_plane_estimation"):
        raise ValueError("runtime or active support replacement is forbidden")
    if len(cfg["samples"]) != 4 or sum(bool(row["primary"]) for row in cfg["samples"]) != 1:
        raise ValueError("exactly four samples and one primary sample are required")
    for index, sample in enumerate(cfg["samples"]):
        exact_keys(sample, SAMPLE_KEYS, f"sample[{index}]")
    if cfg["depth"] != {
        "frame_mapping": "INDEX_ALIGNED", "logical_rate_hz": 30, "scale": 4000.0,
        "width": 1920, "height": 1080,
    }:
        raise ValueError("upstream depth contract drift")
    if cfg["foreground"]["dilation_px"] != 5 or cfg["roi"]["interaction_roi_scale"] != 1.5:
        raise ValueError("frozen foreground/ROI contract changed")
    if cfg["sampling"]["spatial_stride_px"] != 8:
        raise ValueError("frozen spatial sampling changed")
    plane = cfg["plane_fit"]
    if (plane["random_seed"], plane["maximum_tilt_from_world_z_deg"], plane["inlier_distance_m"]) != (0, 10.0, 0.01):
        raise ValueError("frozen plane-fit core values changed")
    if cfg["validation"]["holdout_modulo"] != 5:
        raise ValueError("frozen holdout split changed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], git_head(REPO_ROOT)],
        cwd=REPO_ROOT, check=True,
    )
    official = Path(cfg["official"]["checkout"])
    if git_head(official) != cfg["official"]["commit"]:
        raise ValueError("official checkout pin mismatch")
    return cfg


def resolve_inputs(cfg: dict[str, Any]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    upstream = {key: Path(value).resolve(strict=True) for key, value in cfg["upstream"].items()}
    path_keys = {
        "root", "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand",
        "left_shape", "right_shape", "object_pose_dir", "object_model_dir",
    }
    samples = []
    for source in cfg["samples"]:
        sample = dict(source)
        for key in path_keys:
            sample[key] = Path(sample[key]).resolve(strict=True)
        for role in ("tool", "target"):
            object_id = sample[f"{role}_id"]
            sample[f"{role}_pose"] = (sample["object_pose_dir"] / f"{role}_{object_id}.npy").resolve(strict=True)
            sample[f"{role}_mesh"] = (sample["object_model_dir"] / f"{object_id}_cm.obj").resolve(strict=True)
        samples.append(sample)
    return upstream, samples


def input_contract(cfg: dict[str, Any], upstream: dict[str, Path]) -> tuple[dict[str, Any], bool]:
    decision = json.loads(upstream["decision"].read_text(encoding="utf-8"))
    timeline = json.loads(upstream["timeline_contract"].read_text(encoding="utf-8"))
    active = yaml.safe_load(upstream["active_support_contract"].read_text(encoding="utf-8"))
    checks = {
        "upstream_registration_resolved": decision["classification"] == "RGBD_REGISTRATION_RESOLVED_DEPTH_OBSERVABILITY_LIMITED",
        "upstream_authorizes_separate_estimator": decision["support_plane_estimator_may_resume_in_next_separate_task"] is True,
        "frame_mapping_index_aligned": timeline["frame_mapping"] == "INDEX_ALIGNED",
        "logical_rate_30_hz": timeline["logical_rate_hz"] == 30,
        "depth_scale_4000": timeline["depth_scale"] == 4000,
        "container_pts_not_logical_time": timeline["container_pts_are_logical_time"] is False,
        "active_old_support_is_target_bottom": (
            active["source"]["method"] == "support_entity_extreme_surface"
            and active["source"]["entity_role"] == "target"
            and active["source"]["entity_id"] == "146"
            and active["source"]["extremum"] == "minimum"
        ),
        "active_simulator_plane_still_0p72": active["simulator"]["offset_m"] == 0.72,
    }
    return {
        "schema": "taco_table_depth_input_contract_v1",
        "checks": checks, "passed": all(checks.values()),
        "depth": {"mapping": "INDEX_ALIGNED", "logical_rate_hz": 30, "scale": 4000,
                  "invalid": "raw == 0; discarded without interpolation"},
        "camera": {"extrinsic": "world_to_camera", "backprojection": "inverse rigid transform"},
        "world": {"up": [0, 0, 1]},
        "active_old_support": active,
        "estimator_forbidden_inputs": [
            "target/bowl bottom", "tool/brush bottom", "historical brush-floor residual",
            "active sample support offset", "simulator z=0.72", "table colour",
        ],
    }, all(checks.values())


def active_support_audit(upstream: dict[str, Path]) -> dict[str, Any]:
    active = yaml.safe_load(upstream["active_support_contract"].read_text(encoding="utf-8"))
    return {
        "schema": "taco_active_support_source_audit_v1",
        "active_contract": artifact(upstream["active_support_contract"]),
        "source_method": active["source"]["method"],
        "source_entity_role": active["source"]["entity_role"],
        "source_entity_id": active["source"]["entity_id"],
        "classification": "SAMPLE_SPECIFIC_TARGET_BOTTOM_SOURCE",
        "changed_by_this_task": False,
    }


def load_projection_geometry(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(cfg["official"]["chumpy_dependency"])))
    utils = Path(cfg["official"]["checkout"]) / "dataset_utils"
    hands = {}
    for side in ("right", "left"):
        vertices, _, faces, _ = load_official_hand_sequence(
            dataset_utils=utils, pose_path=sample[f"{side}_hand"],
            shape_path=sample[f"{side}_shape"], side=side, device=cfg["official"]["device"],
        )
        if vertices.shape[0] != sample["expected_frames"]:
            raise ValueError(f"{sample['key']}: official hand frame count mismatch")
        hands[side] = {"vertices": vertices, "faces": faces}
    objects = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(sample[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        poses = np.load(sample[f"{role}_pose"], mmap_mode="r")
        if poses.shape != (sample["expected_frames"], 4, 4):
            raise ValueError(f"{sample['key']}: object pose frame count mismatch")
        objects[role] = {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces), "poses": poses}
    return {"hands": hands, "objects": objects, "dataset_utils": utils}


def mesh_at(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray | None = None) -> trimesh.Trimesh:
    values = vertices if pose is None else vertices @ pose[:3, :3].T + pose[:3, 3]
    return trimesh.Trimesh(vertices=values, faces=faces, process=False)


def estimate_sample(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    geometry = load_projection_geometry(cfg, sample)
    intrinsic = np.loadtxt(sample["intrinsic"])
    extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
    if extrinsic.shape != (sample["expected_frames"], 4, 4):
        raise ValueError(f"{sample['key']}: extrinsic frame count mismatch")
    projector = TacoOfficialProjector(
        dataset_utils=geometry["dataset_utils"],
        image_size=(int(cfg["depth"]["width"]), int(cfg["depth"]["height"])),
        intrinsic=intrinsic, extrinsic=extrinsic[0], device=cfg["official"]["device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(rasterizer.raster_settings, max_faces_per_bin=400000)
    stream = ffprobe_video(sample["depth_video"])
    count = stream_frame_count(stream)
    if count != sample["expected_frames"]:
        raise ValueError(f"{sample['key']}: depth count differs from INDEX_ALIGNED contract")
    spec = DepthVideoSpec(width=int(cfg["depth"]["width"]), height=int(cfg["depth"]["height"]), frame_count=count)
    rows: list[dict[str, Any]] = []
    fit_estimates: list[PlaneEstimate] = []
    all_estimates: list[PlaneEstimate] = []
    validation_points: list[np.ndarray] = []
    fit_candidate_total = validation_candidate_total = 0
    holdout_modulo = int(cfg["validation"]["holdout_modulo"])
    for frame, raw in enumerate(iter_depth_frames(sample["depth_video"], spec)):
        projector.set_camera(intrinsic, extrinsic[frame])
        meshes = [
            mesh_at(geometry["hands"][side]["vertices"][frame], geometry["hands"][side]["faces"])
            for side in ("right", "left")
        ]
        for role in ("tool", "target"):
            obj = geometry["objects"][role]
            meshes.append(mesh_at(obj["vertices"], obj["faces"], obj["poses"][frame]))
        foreground = projector.render_mask(meshes)
        depth = raw_depth_to_metres(raw, scale=float(cfg["depth"]["scale"]))
        split = "validation" if frame % holdout_modulo == 0 else "fit"
        try:
            points, counts = foreground_excluded_background_points(
                depth, intrinsic, extrinsic[frame], foreground,
                dilation_px=int(cfg["foreground"]["dilation_px"]),
                interaction_roi_scale=float(cfg["roi"]["interaction_roi_scale"]),
                spatial_stride_px=int(cfg["sampling"]["spatial_stride_px"]),
            )
            estimate = fit_horizontal_support_plane(
                points, random_seed=int(cfg["plane_fit"]["random_seed"]),
                maximum_tilt_from_world_z_deg=float(cfg["plane_fit"]["maximum_tilt_from_world_z_deg"]),
                inlier_distance_m=float(cfg["plane_fit"]["inlier_distance_m"]),
                maximum_iterations=int(cfg["plane_fit"]["maximum_iterations"]),
                minimum_candidate_points=int(cfg["plane_fit"]["minimum_candidate_points"]),
            )
            row = {"sample": sample["key"], "frame": frame, "split": split,
                   "status": "TABLE_PLANE_ESTIMATED", "mask_and_point_counts": counts,
                   **estimate.to_dict()}
            all_estimates.append(estimate)
            if split == "fit":
                fit_estimates.append(estimate)
                fit_candidate_total += len(points)
            else:
                validation_points.append(points)
                validation_candidate_total += len(points)
        except ValueError as error:
            row = {"sample": sample["key"], "frame": frame, "split": split,
                   "status": "NO_TABLE_PLANE_EVIDENCE", "reason": str(error)}
        rows.append(row)
        if (frame + 1) % 50 == 0 or frame + 1 == count:
            print(f"{sample['key']}: {frame + 1}/{count}", flush=True)
    minimum = int(cfg["validation"]["minimum_successful_fit_frames"])
    if len(fit_estimates) < minimum:
        return {
            "sample": sample["key"], "status": "INSUFFICIENT_TABLE_DEPTH_EVIDENCE",
            "rows": rows, "successful_fit_frames": len(fit_estimates),
            "successful_validation_frames": len(validation_points),
            "fit_candidate_points": fit_candidate_total,
            "validation_candidate_points": validation_candidate_total,
        }
    consensus = aggregate_support_planes(fit_estimates)
    offsets = np.asarray([estimate.plane.offset for estimate in fit_estimates])
    offset_median = float(np.median(offsets))
    offset_mad = float(np.median(np.abs(offsets - offset_median)))
    normals = np.stack([estimate.plane.normal for estimate in fit_estimates])
    angular = np.degrees(np.arccos(np.clip(normals @ consensus.normal, -1.0, 1.0)))
    stable = offset_mad <= float(cfg["plane_fit"]["maximum_consensus_offset_mad_m"])
    if not validation_points:
        status = "INSUFFICIENT_TABLE_DEPTH_EVIDENCE"
        validation = None
    else:
        held = np.concatenate(validation_points, axis=0)
        all_background = evaluate_plane_on_points(consensus, held)
        residual = consensus.signed_distance(held)
        table_selector = np.abs(residual) <= float(cfg["plane_fit"]["inlier_distance_m"])
        if int(table_selector.sum()) < int(cfg["plane_fit"]["minimum_candidate_points"]):
            status = "INSUFFICIENT_TABLE_DEPTH_EVIDENCE"
            validation = {"all_background": all_background, "table_support": None,
                          "table_support_point_count": int(table_selector.sum())}
        else:
            table_support = evaluate_plane_on_points(consensus, held[table_selector])
            validation = {
                "all_background": all_background, "table_support": table_support,
                "table_support_point_count": int(table_selector.sum()),
                "table_support_fraction": float(table_selector.mean()),
                "selection": "absolute consensus-plane distance <= frozen RANSAC inlier distance",
            }
            status = "STABLE_TABLE_PLANE" if stable else "TABLE_PLANE_ESTIMATOR_UNSTABLE"
    return {
        "sample": sample["key"], "status": status, "rows": rows,
        "successful_fit_frames": len(fit_estimates),
        "successful_validation_frames": len(validation_points),
        "failed_frame_count": sum(row["status"] != "TABLE_PLANE_ESTIMATED" for row in rows),
        "fit_candidate_points": fit_candidate_total,
        "validation_candidate_points": validation_candidate_total,
        "consensus": {
            "plane": consensus.to_dict(),
            "tilt_from_world_z_deg": float(np.degrees(np.arccos(np.clip(consensus.normal[2], -1, 1)))),
            "offset_median_m": offset_median, "offset_mad_m": offset_mad,
            "offset_p05_m": float(np.percentile(offsets, 5)),
            "offset_p50_m": offset_median, "offset_p95_m": float(np.percentile(offsets, 95)),
            "normal_angular_deviation_deg": {
                "median": float(np.median(angular)), "p95": float(np.percentile(angular, 95)),
                "maximum": float(np.max(angular)),
            },
            "per_frame_residual_median_m": {
                "median": float(np.median([value.median_absolute_residual_m for value in fit_estimates])),
                "p95": float(np.percentile([value.median_absolute_residual_m for value in fit_estimates], 95)),
            },
        },
        "holdout_validation": validation,
    }


def transformed_minimum(mesh_path: Path, pose_path: Path, plane: Plane) -> float:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh.apply_scale(0.01)
    pose = np.load(pose_path, mmap_mode="r")[0]
    return plane.minimum_signed_distance(np.asarray(mesh.vertices), pose)


def brush_geometry_holdout(
    primary: dict[str, Any], result: dict[str, Any], consensus_artifact: dict[str, Any],
) -> dict[str, Any]:
    plane_data = result["consensus"]["plane"]
    plane = Plane(normal=np.asarray(plane_data["normal"]), offset=plane_data["offset_m"], frame="world")
    uncertainty = result["holdout_validation"]["table_support"]
    bowl = transformed_minimum(primary["target_mesh"], primary["target_pose"], plane)
    brush = transformed_minimum(primary["tool_mesh"], primary["tool_pose"], plane)
    low, high = uncertainty["signed_residual_p05_m"], uncertainty["signed_residual_p95_m"]
    bowl_status = "WITHIN_ESTIMATOR_UNCERTAINTY" if low <= bowl <= high else "OUTSIDE_ESTIMATOR_UNCERTAINTY"
    return {
        "schema": "taco_brush_table_geometry_holdout_v1",
        "estimator_frozen_before_object_bottom_read": True,
        "frozen_consensus_artifact": consensus_artifact,
        "uncertainty_source": "validation-frame measured table-support depth signed residual p05..p95",
        "uncertainty_interval_m": [low, high],
        "bowl_bottom_signed_distance_m": bowl, "bowl_status": bowl_status,
        "brush_bottom_signed_distance_m": brush,
        "historical_minus_1p311mm_used_by_estimator": False,
        "interpretation": "Object bottoms are post-freeze validators only and cannot alter the estimate.",
    }


def source_pins(
    cfg_path: Path, cfg: dict[str, Any], upstream: dict[str, Path], samples: list[dict[str, Any]],
) -> dict[str, Any]:
    sources: dict[str, Path] = {
        "config": cfg_path, "runner": Path(__file__),
        "estimator_module": RL_ROOT / "src/egoengine_repro/scene/support_surface_estimation.py",
        "estimator_tests": RL_ROOT / "tests/core/test_support_surface_estimation.py",
        "official_projection_wrapper": RL_ROOT / "src/egoengine_repro/evaluation/taco_official_projection.py",
        **{f"upstream:{key}": path for key, path in upstream.items()},
    }
    for sample in samples:
        for key in (
            "depth_video", "intrinsic", "extrinsic", "left_hand", "right_hand", "left_shape",
            "right_shape", "tool_pose", "target_pose", "tool_mesh", "target_mesh",
        ):
            sources[f"{sample['key']}:{key}"] = sample[key]
    return {
        "schema": "taco_multiframe_table_depth_source_pins_v1",
        "repository_head_before_audit": git_head(REPO_ROOT),
        "official_taco_commit": git_head(Path(cfg["official"]["checkout"])),
        "artifacts": {key: artifact(path) for key, path in sources.items()},
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
    contract, contract_ok = input_contract(cfg, upstream)
    write_json(output / "table_depth_input_contract.json", contract)
    write_json(output / "active_support_source_audit.json", active_support_audit(upstream))
    if not contract_ok:
        decision = {"schema": "taco_multiframe_table_depth_decision_v1",
                    "classification": "TABLE_DEPTH_INPUT_CONTRACT_DRIFT", "stop_phase": "A"}
        write_json(output / "decision.json", decision)
        raise RuntimeError(decision["classification"])

    results = []
    with (output / "per_frame_table_plane_estimates.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            result = estimate_sample(cfg, sample)
            results.append(result)
            for row in result.pop("rows"):
                stream.write(json.dumps(row, sort_keys=True, default=_default) + "\n")

    primary_sample = next(sample for sample in samples if sample["primary"])
    primary = next(result for result in results if result["sample"] == primary_sample["key"])
    if "consensus" in primary:
        consensus_report = {
            "schema": "taco_table_plane_consensus_v1", "sample": primary["sample"],
            "status": primary["status"], "successful_frame_count": primary["successful_fit_frames"],
            "failed_frame_count": primary["failed_frame_count"], **primary["consensus"],
        }
        write_json(output / "table_plane_consensus.json", consensus_report)
        consensus_pin = artifact(output / "table_plane_consensus.json")
        write_json(output / "table_plane_depth_holdout_validation.json", {
            "schema": "taco_table_plane_depth_holdout_validation_v1", "sample": primary["sample"],
            "holdout_rule": "frame_index % 5 == 0", **(primary["holdout_validation"] or {}),
        })
    else:
        consensus_pin = None

    if primary["status"] == "STABLE_TABLE_PLANE" and consensus_pin is not None:
        geometry_holdout = brush_geometry_holdout(primary_sample, primary, consensus_pin)
        write_json(output / "brush_table_holdout_validation.json", geometry_holdout)
    else:
        geometry_holdout = None

    multi = {
        "schema": "taco_multi_sample_table_estimation_summary_v1",
        "same_algorithm_and_parameters_for_every_sample": True,
        "samples": [{key: value for key, value in result.items() if key != "holdout_validation"}
                    for result in results],
        "status_counts": {status: sum(result["status"] == status for result in results) for status in (
            "STABLE_TABLE_PLANE", "INSUFFICIENT_TABLE_DEPTH_EVIDENCE", "TABLE_PLANE_ESTIMATOR_UNSTABLE",
        )},
    }
    write_json(output / "multi_sample_table_estimation_summary.json", multi)

    stable_count = sum(result["status"] == "STABLE_TABLE_PLANE" for result in results)
    any_unstable = any(result["status"] == "TABLE_PLANE_ESTIMATOR_UNSTABLE" for result in results)
    if primary["status"] == "INSUFFICIENT_TABLE_DEPTH_EVIDENCE":
        classification = "INSUFFICIENT_TABLE_DEPTH_EVIDENCE"
    elif primary["status"] == "TABLE_PLANE_ESTIMATOR_UNSTABLE" or any_unstable:
        classification = "TABLE_PLANE_ESTIMATOR_UNSTABLE"
    elif geometry_holdout is None or geometry_holdout["bowl_status"] != "WITHIN_ESTIMATOR_UNCERTAINTY":
        classification = "BRUSH_TABLE_HOLDOUT_MISMATCH"
    elif stable_count < int(cfg["plane_fit"]["minimum_samples_with_stable_plane"]):
        classification = "INSUFFICIENT_TABLE_DEPTH_EVIDENCE"
    else:
        classification = "MULTIFRAME_TABLE_PLANE_CANDIDATE_VALIDATED"

    if classification == "MULTIFRAME_TABLE_PLANE_CANDIDATE_VALIDATED":
        write_json(output / "candidate_support_surface_contract.json", {
            "schema": "candidate_support_surface_contract_v1",
            "method": "MULTIFRAME_RGBD_TABLE_PLANE_ESTIMATE",
            "provenance": "LOCAL_EVIDENCE_BACKED_ESTIMATOR",
            "status": "CANDIDATE_NOT_ACTIVE",
            "source_frame": "world", "plane": primary["consensus"]["plane"],
            "frozen_consensus_artifact": consensus_pin,
            "active_contract_replaced": False,
        })

    write_json(output / "source_pins.json", source_pins(cfg_path, cfg, upstream, samples))
    write_json(output / "config_consumption_audit.json", {
        "schema": "taco_multiframe_table_depth_config_consumption_v1",
        "config": artifact(cfg_path), "exact_nested_key_validation": True,
        "all_config_sections_consumed": True, "parameter_sweep_count": 0,
    })
    decision = {
        "schema": "taco_multiframe_table_depth_decision_v1", "classification": classification,
        "primary_sample_status": primary["status"], "stable_sample_count": stable_count,
        "bowl_holdout_status": None if geometry_holdout is None else geometry_holdout["bowl_status"],
        "candidate_contract_created": classification == "MULTIFRAME_TABLE_PLANE_CANDIDATE_VALIDATED",
        "active_support_contract_changed": False,
        "runtime_counts": {"mink": 0, "physics": 0, "replay": 0, "mpc": 0, "rl": 0,
                           "promotion": 0, "chunk_commit": 0, "active_support_replacement": 0},
    }
    write_json(output / "decision.json", decision)

    if "consensus" in primary:
        plane = primary["consensus"]["plane"]
        normal_text = json.dumps(plane["normal"])
        offset_text = f"{plane['offset_m']:.9f} m"
    else:
        normal_text = offset_text = "无可靠结果"
    bowl_text = "未读取（桌面估计未稳定）" if geometry_holdout is None else f"{geometry_holdout['bowl_bottom_signed_distance_m'] * 1000:.3f} mm"
    brush_text = "未读取（桌面估计未稳定）" if geometry_holdout is None else f"{geometry_holdout['brush_bottom_signed_distance_m'] * 1000:.3f} mm"
    total_frames = sum(sample["expected_frames"] for sample in samples)
    total_candidates = sum(result.get("fit_candidate_points", 0) + result.get("validation_candidate_points", 0) for result in results)
    summary = f"""# TACO 多帧桌面深度估计与验证 v1

**正式分类：`{classification}`。** active `SupportSurfaceContract` 没有修改。

1. **有没有用 bowl bottom 估桌面？** 没有。估计器只读取真实 metric Depth、相机、官方前景投影和 world +Z；consensus 文件写入并 SHA256 冻结后才读取 bowl/brush mesh。
2. **使用多少帧？** 四个样本共处理 `{total_frames}` 帧 native Depth；Brush 的 fit/holdout 划分严格由 `frame % 5` 决定。
3. **去前景后候选点：** 空间 stride=8 后四样本累计 `{total_candidates}` 个真实背景深度点；没有插值或单目补洞。
4. **逐帧稳定性：** Brush 状态为 `{primary['status']}`；fit offset MAD 为 `{primary.get('consensus', {}).get('offset_mad_m', 'N/A')}` m。
5. **最终桌面：** normal=`{normal_text}`，world-plane offset/高度=`{offset_text}`。
6. **真实 Depth holdout：** `{json.dumps(primary.get('holdout_validation'), default=_default, ensure_ascii=False)}`
7. **冻结后 bowl bottom 距离：** `{bowl_text}`；状态 `{None if geometry_holdout is None else geometry_holdout['bowl_status']}`。
8. **brush bottom 距离：** `{brush_text}`。
9. **旧 -1.311 mm：** 没有作为真值或输入延续；应由上面的新独立测量取代。
10. **其他样本：** `{json.dumps({result['sample']: result['status'] for result in results}, ensure_ascii=False)}`，全部使用同一算法和参数。
11. **能否下一任务替换 active contract？** `{'已生成候选，但仍需下一项独立 integration/promotion 任务' if classification == 'MULTIFRAME_TABLE_PLANE_CANDIDATE_VALIDATED' else '不能；当前分类未通过候选验证'}`。

本轮 MINK、physics、Replay、MPC、RL、promotion、chunk commit 和 active support replacement 均为 0。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    write_hashes(output)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
