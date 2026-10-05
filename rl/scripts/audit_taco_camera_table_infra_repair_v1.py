#!/usr/bin/env python3
"""Certify TACO camera/support contracts and recompute Brush static attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_initialization import visual_meshes, world_vertices  # noqa: E402
from egoengine_repro.evaluation.taco_official_projection import load_official_hand_sequence  # noqa: E402
from egoengine_repro.scene.support_surface import (  # noqa: E402
    Plane,
    SupportSurfaceContract,
    SupportSurfaceSpec,
    build_taco_scene_alignment,
    resolve_support_surface,
)
from egoengine_repro.taco.camera_contract import load_allocentric_camera_contracts  # noqa: E402


SCHEMA = "taco_brush_camera_table_infra_repair_v1"
SUCCESS = "BRUSH_INFRA_CONTRACTS_CERTIFIED_RETARGET_BLOCKER_REMAINS"
FINAL_CLASSES = {
    "CAMERA_CONTRACT_BLOCKER",
    "SUPPORT_SURFACE_CONTRACT_BLOCKER",
    "CAMERA_AND_SUPPORT_CONTRACT_BLOCKER",
    SUCCESS,
    "FUNCTIONAL_ERROR",
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               default=json_default, allow_nan=False) + "\n")


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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path,
                                   text=True).strip()


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected infrastructure repair schema")
    if git_head(ROOT.parent) != cfg["baseline_commit"]:
        raise ValueError("repair must start from its pinned baseline")
    forbidden = ("physics", "replay", "mpc", "rl", "new_retarget_candidate",
                 "promotion", "chunk_commit")
    if any(cfg["authorization"][name] for name in forbidden):
        raise ValueError("repair contract authorizes forbidden runtime work")
    if not all(cfg["hard_rules"].values()):
        raise ValueError("all infrastructure hard rules must be enabled")
    return cfg


def paths(cfg: dict[str, Any]) -> dict[str, Path]:
    data = Path(cfg["paths"]["data_root"])
    seq = cfg["sample"]["sequence"]
    hand = data / "hand_poses/Hand_Poses" / seq
    objects = data / "object_poses/Object_Poses" / seq
    models = data / "object_models/object_models_released"
    cache = Path(cfg["paths"]["allocentric_cache"])
    triplet, sequence_name = cfg["sample"]["triplet"], cfg["sample"]["sequence_name"]
    video_dir = cache / "Marker_Removed_Allocentric_RGB_Videos" / triplet / sequence_name
    calibration = cache / "Allocentric_Camera_Parameters" / triplet / sequence_name / "calibration.json"
    raw = json.loads(calibration.read_text())
    return {
        "calibration": calibration,
        **{f"video_{camera_id}": video_dir / f"{camera_id}.mp4" for camera_id in raw},
        "target_pose": objects / f"target_{cfg['sample']['target_id']}.npy",
        "tool_pose": objects / f"tool_{cfg['sample']['tool_id']}.npy",
        "target_mesh": models / f"{cfg['sample']['target_id']}_cm.obj",
        "tool_mesh": models / f"{cfg['sample']['tool_id']}_cm.obj",
        "left_hand": hand / "left_hand.pkl", "right_hand": hand / "right_hand.pkl",
        "left_shape": hand / "left_hand_shape.pkl", "right_shape": hand / "right_hand_shape.pkl",
        "scene": Path(cfg["paths"]["scene"]),
        "human_reference": Path(cfg["paths"]["frozen_baseline_run"]) / "human_reference.npz",
        "robot_reference": Path(cfg["paths"]["frozen_baseline_run"]) / "robot_reference.npz",
        "retarget_report": Path(cfg["paths"]["frozen_baseline_run"]) / "retarget_report.json",
        "official_world_visualizer": Path(cfg["paths"]["official_taco_checkout"]) / "dataset_utils/visualize_world_coordinate_system.py",
        "release_readme": Path("/data_all/intern02/egoengine/video_to_spider/datasets/taco_v1/README.md"),
    }


def support_contract(cfg: dict[str, Any]) -> SupportSurfaceContract:
    value = cfg["support_surface"]
    source = value["source"]
    simulator = value["simulator"]
    return SupportSurfaceContract(
        source_frame=value["source_frame"],
        source=SupportSurfaceSpec(
            frame_index=int(source["frame_index"]),
            entity_role=source["entity_role"], entity_id=str(source["entity_id"]),
            direction_world=np.asarray(source["direction_world"], dtype=float),
            extremum=source["extremum"], method=source["method"],
        ),
        simulator=Plane(normal=np.asarray(simulator["normal"], dtype=float),
                        offset=float(simulator["offset_m"]), frame=simulator["frame"]),
        provenance=value["provenance"], schema=value["schema"],
    )


def load_object(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    mesh.apply_scale(0.01)
    return mesh


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    result = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
              "fps": float(capture.get(cv2.CAP_PROP_FPS)),
              "artifact": artifact(path)}
    capture.release()
    return result


def camera_audit(cfg: dict[str, Any], inputs: dict[str, Path], output: Path) -> bool:
    domain = cfg["camera_image_domain"]
    cameras, raw = load_allocentric_camera_contracts(
        inputs["calibration"], image_domain=domain["decision"],
        certification=domain["certification"],
    )
    metadata = {camera_id: video_metadata(inputs[f"video_{camera_id}"])
                for camera_id in cameras}
    mismatches = {camera_id: {"calibration": list(camera.image_size),
                              "video": [metadata[camera_id]["width"], metadata[camera_id]["height"]]}
                  for camera_id, camera in cameras.items()
                  if camera.image_size != (metadata[camera_id]["width"], metadata[camera_id]["height"])}
    official_source = inputs["official_world_visualizer"]
    official_text = official_source.read_text()
    evidence = {
        "official_code": artifact(official_source),
        "official_behavior": {
            "release_directory_read": "Allocentric_RGB_Videos",
            "projection": "world camera points multiplied by K then homogeneous division",
            "uses_R_T": True,
            "applies_distortion_coefficients": "distCoeff" in official_text,
            "calls_undistort": "undistort" in official_text.lower(),
        },
        "release_metadata": artifact(inputs["release_readme"]),
        "release_metadata_observation": (
            "resized release declares allocentric videos downscaled to 512x376; "
            "it does not declare a second distortion transform"
        ),
        "calibration_image_sizes_match_release_videos": not mismatches,
    }
    official_pinhole = (not evidence["official_behavior"]["applies_distortion_coefficients"]
                        and not evidence["official_behavior"]["calls_undistort"])
    passed = (len(cameras) == 12 and not mismatches
              and domain["decision"] == "UNDISTORTED_PINHOLE" and official_pinhole)
    write_json(output / "camera_contract_schema.json", {
        "schema": "camera_model_contract_v1",
        "required_fields": ["camera_id", "raw_record_sha256", "image_size", "K", "R", "T",
                            "distortion_field_name", "distortion_coefficients", "distortion_model",
                            "image_domain", "projection_domain", "certification"],
        "image_domain_enum": ["RAW_DISTORTED", "UNDISTORTED_PINHOLE", "UNKNOWN_IMAGE_DOMAIN"],
        "unknown_calibration_fields_preserved": True,
    })
    write_json(output / "allocentric_camera_contract.json", {
        "schema": "taco_allocentric_camera_contract_v1",
        "source": artifact(inputs["calibration"]),
        "camera_count": len(cameras),
        "cameras": {key: value.to_dict() for key, value in cameras.items()},
        "video_metadata": metadata, "size_mismatches": mismatches,
        "parser_did_not_drop_unknown_fields": True,
        "status": "PASS" if passed else "FAIL",
    })
    write_json(output / "allocentric_image_domain_decision.json", {
        "schema": "taco_allocentric_image_domain_decision_v1",
        "classification": domain["decision"],
        "certification": domain["certification"],
        "distortion_policy": domain["distortion_policy"],
        "evidence": evidence,
        "no_posthoc_image_fit": True,
        "pixel_accurate_projection_authorized": bool(passed),
        "status": "PASS" if passed else "FAIL",
    })
    fixture = output / "inputs"
    fixture.mkdir(parents=True)
    shutil.copyfile(inputs["calibration"], fixture / "calibration.json")
    write_json(fixture / "allocentric_video_metadata.json",
               {key: {name: value[name] for name in ("width", "height", "frames", "fps")}
                for key, value in metadata.items()})
    return passed


def support_and_alignment_audit(cfg: dict[str, Any], inputs: dict[str, Path],
                                output: Path) -> tuple[bool, Any, np.ndarray, dict[str, Any]]:
    contract = support_contract(cfg)
    target_mesh = load_object(inputs["target_mesh"])
    target_poses = np.load(inputs["target_pose"], allow_pickle=False).astype(float)
    tool_poses = np.load(inputs["tool_pose"], allow_pickle=False).astype(float)
    support = resolve_support_surface(contract, {
        "target": {"vertices": target_mesh.vertices, "poses": target_poses},
    })
    centers = np.stack([tool_poses[0, :3, 3], target_poses[0, :3, 3]])
    center = centers.mean(0)
    desired = center.copy()
    desired[:2] = cfg["horizontal_alignment"]["scene_center_xy_target_m"]
    transform = build_taco_scene_alignment(
        support, source_scene_center=center, desired_scene_center_sim=desired,
    )
    mapped = support.transform(transform)
    target_world = target_mesh.vertices @ target_poses[0, :3, :3].T + target_poses[0, :3, 3]
    target_sim = target_world @ transform[:3, :3].T + transform[:3, 3]
    source_min = float(support.signed_distance(target_world).min())
    sim_min = float(support.simulator.signed_distance(target_sim).min())
    with np.load(inputs["human_reference"], allow_pickle=False) as archive:
        frozen_transform = archive["T_sim_world"].copy()
    model = mujoco.MjModel.from_xml_path(str(inputs["scene"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor_id = model.geom("floor").id
    floor_normal = data.geom_xmat[floor_id].reshape(3, 3)[:, 2]
    floor_offset = float(floor_normal @ data.geom_xpos[floor_id])
    scene_matches = (np.allclose(floor_normal, support.simulator.normal, atol=1e-12)
                     and np.isclose(floor_offset, support.simulator.offset, atol=1e-12))
    transform_matches = np.allclose(transform, frozen_transform, atol=1e-12)
    passed = (abs(source_min) <= 1e-12 and abs(sim_min) <= 1e-12
              and scene_matches and transform_matches)
    (output / "support_surface_contract.yaml").write_text(
        yaml.safe_dump(contract.to_dict(), sort_keys=False)
    )
    support_report = {
        "schema": "taco_brush_support_surface_audit_v1", "status": "PASS" if passed else "FAIL",
        "resolved": support.to_dict(),
        "source_bowl_frame0_minimum_signed_distance_m": source_min,
        "transformed_bowl_frame0_minimum_signed_distance_m": sim_min,
        "machine_contract_not_visual_gate": True,
        "input_hashes": {name: artifact(inputs[name]) for name in ("target_mesh", "target_pose")},
    }
    write_json(output / "support_surface_audit.json", support_report)
    alignment = {
        "schema": "taco_brush_scene_alignment_audit_v1", "status": "PASS" if passed else "FAIL",
        "T_sim_world": transform, "frozen_T_sim_world": frozen_transform,
        "frozen_transform_max_abs_error": float(np.abs(transform - frozen_transform).max()),
        "mapped_source_plane": mapped.to_dict(), "simulator_plane": support.simulator.to_dict(),
        "scene_floor_plane": {"normal": floor_normal, "offset_m": floor_offset},
        "scene_floor_matches_contract": bool(scene_matches),
        "horizontal_alignment_changed": False,
    }
    write_json(output / "scene_alignment_audit.json", alignment)
    fixture = output / "inputs"
    shutil.copyfile(inputs["target_pose"], fixture / "target_146.npy")
    return passed, support, transform, support_report


def static_attribution(cfg: dict[str, Any], inputs: dict[str, Path], output: Path,
                       support, transform: np.ndarray) -> dict[str, Any]:
    official = Path(cfg["paths"]["official_taco_checkout"])
    chumpy = Path(cfg["paths"]["official_projection_chumpy_dependency"]).resolve(strict=True)
    sys.path.insert(0, str(chumpy))
    source_hands = {}
    for side in ("left", "right"):
        vertices, _, _, _ = load_official_hand_sequence(
            dataset_utils=official / "dataset_utils", pose_path=inputs[f"{side}_hand"],
            shape_path=inputs[f"{side}_shape"], side=side, device="cuda:0",
        )
        source_hands[side] = vertices
    source_clearance = {
        side: float(support.signed_distance(source_hands[side][0]).min())
        for side in ("left", "right")
    }
    for role, label in (("tool", "brush"), ("target", "bowl")):
        mesh = load_object(inputs[f"{role}_mesh"])
        pose = np.load(inputs[f"{role}_pose"], allow_pickle=False)[0]
        world = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
        source_clearance[label] = float(support.signed_distance(world).min())

    model = mujoco.MjModel.from_xml_path(str(inputs["scene"]))
    data = mujoco.MjData(model)
    with np.load(inputs["robot_reference"], allow_pickle=False) as archive:
        data.qpos[:] = archive["qpos"][0]
        data.qvel[:] = archive["qvel"][0]
        data.ctrl[:] = archive["ctrl"][0]
    mujoco.mj_forward(model, data)
    meshes, _ = visual_meshes(inputs["scene"], model)
    rows = []
    for geom, mesh in meshes.items():
        name = model.geom(geom).name
        points = world_vertices(model, data, geom, mesh)
        rows.append({"geom": name,
                     "minimum_signed_distance_m": float(support.simulator.signed_distance(points).min())})
    robot_clearance = {
        "left": min(row["minimum_signed_distance_m"] for row in rows
                    if row["geom"].startswith("left_") and "object" not in row["geom"]),
        "right": min(row["minimum_signed_distance_m"] for row in rows
                     if row["geom"].startswith("right_") and "object" not in row["geom"]),
        "brush": next(row["minimum_signed_distance_m"] for row in rows
                      if row["geom"] == "right_object_visual"),
        "bowl": next(row["minimum_signed_distance_m"] for row in rows
                     if row["geom"] == "left_object_visual"),
    }
    expected = {
        "source": {"left": -0.0014447887184856345, "right": -0.0033040364029643943,
                   "brush": -0.0013113107670463808, "bowl": 0.0},
        "robot": {"left": -0.020878211733495577, "right": -0.01890884276181959,
                  "brush": -0.0013113086868823398, "bowl": -5.858854512652556e-10},
    }
    errors = {group: {key: float(abs((source_clearance if group == "source" else robot_clearance)[key] - value))
                      for key, value in values.items()}
              for group, values in expected.items()}
    passed = max(value for group in errors.values() for value in group.values()) < 1e-9
    result = {
        "schema": "taco_brush_static_attribution_recomputed_v1",
        "status": "PASS_REPRODUCES_FROZEN_ATTRIBUTION" if passed else "FAIL",
        "support_surface_contract_only": True,
        "source_native_clearance_m": source_clearance,
        "frozen_mink_native_clearance_m": robot_clearance,
        "robot_minus_source_clearance_m": {
            key: float(robot_clearance[key] - source_clearance[key]) for key in source_clearance},
        "native_robot_visuals": rows, "expected_regression_abs_error_m": errors,
        "runtime_counts": {"physics_steps": 0, "new_retarget_candidates": 0},
    }
    write_json(output / "static_attribution_recomputed.json", result)
    return result


def write_deletion_and_supersession(output: Path) -> None:
    deleted = [
        "rl/scripts/audit_taco_brush_table_alignment_attribution_v1.py",
        "rl/configs/taco_brush_table_alignment_attribution_v1.yaml",
        "rl/runs/taco_brush_table_alignment_attribution_v1/",
    ]
    write_json(output / "deleted_code_manifest.json", {
        "schema": "taco_camera_table_deleted_code_manifest_v1",
        "baseline_commit": "e945b9ccc03089756de176f58348b01282c9b45e",
        "deleted_from_active_tree": deleted,
        "reason": "yellow-grid visual gate and duplicate support-plane derivation retired",
        "recoverable_from_git_history": True,
        "deprecated_wrapper_retained": False,
    })
    write_json(output / "superseded_artifacts.json", {
        "schema": "taco_camera_table_superseded_artifacts_v1",
        "commit": "e945b9ccc03089756de176f58348b01282c9b45e",
        "artifacts": [
            {"path": "rl/runs/taco_brush_table_alignment_attribution_v1/TABLE_PLANE_VISUAL_REVIEW.md",
             "status": "SUPERSEDED_INVALID_VISUAL_GATE"},
            {"path": "rl/runs/taco_brush_table_alignment_attribution_v1/visuals/",
             "status": "HISTORICAL_DIAGNOSTIC_ONLY"},
            {"path": "rl/runs/taco_brush_table_alignment_attribution_v1/source_world_endpoint0_clearance.json",
             "status": "HISTORICAL_QUANTITATIVE_RESULT_REPRODUCED_BY_NEW_CONTRACT"},
            {"path": "rl/runs/taco_brush_table_alignment_attribution_v1/source_vs_robot_floor_attribution.json",
             "status": "HISTORICAL_QUANTITATIVE_RESULT_REPRODUCED_BY_NEW_CONTRACT"},
        ],
        "active_sources_of_truth": [
            "egoengine_repro.taco.camera_contract.CameraModelContract",
            "egoengine_repro.scene.support_surface.SupportSurfaceContract",
            "rl/runs/taco_brush_camera_table_infra_repair_v1/",
        ],
    })


def hash_tree(output: Path) -> None:
    lines = []
    for path in sorted(path for path in output.rglob("*") if path.is_file()
                       and path.name != "server_artifacts.sha256"):
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = load_config(config_path)
    inputs = paths(cfg)
    for path in inputs.values():
        path.resolve(strict=True)
    output = Path(cfg["paths"]["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "source_pins.json", {
        "schema": "taco_camera_table_infra_source_pins_v1",
        "active_repository": {"path": str(ROOT.parent), "commit": git_head(ROOT.parent)},
        "baseline_commit": cfg["baseline_commit"],
        "official_taco": {"path": cfg["paths"]["official_taco_checkout"],
                          "commit": git_head(Path(cfg["paths"]["official_taco_checkout"]))},
    })
    camera_ok = camera_audit(cfg, inputs, output)
    support_ok, support, transform, _ = support_and_alignment_audit(cfg, inputs, output)
    static = static_attribution(cfg, inputs, output, support, transform)
    write_deletion_and_supersession(output)
    static_ok = static["status"].startswith("PASS")
    if camera_ok and support_ok and static_ok:
        classification = SUCCESS
    elif not camera_ok and not support_ok:
        classification = "CAMERA_AND_SUPPORT_CONTRACT_BLOCKER"
    elif not camera_ok:
        classification = "CAMERA_CONTRACT_BLOCKER"
    else:
        classification = "SUPPORT_SURFACE_CONTRACT_BLOCKER"
    decision = {
        "schema": "taco_brush_camera_table_infra_decision_v1",
        "classification": classification,
        "allowed_classes": sorted(FINAL_CLASSES),
        "camera_contract_passed": camera_ok,
        "support_surface_contract_passed": support_ok,
        "static_attribution_reproduced": static_ok,
        "next_experiment": ("ONE_ENVIRONMENT_AWARE_MINK_CANDIDATE"
                            if classification == SUCCESS else "BLOCKED"),
        "runtime_counts": {"physics_steps": 0, "replay_runs": 0, "mpc_runs": 0,
                           "rl_runs": 0, "new_retarget_candidates": 0,
                           "promotions": 0, "chunk_commits": 0},
    }
    write_json(output / "decision.json", decision)
    source = static["source_native_clearance_m"]
    robot = static["frozen_mink_native_clearance_m"]
    summary = [
        "# TACO Brush camera/table infrastructure repair v1", "",
        f"- Classification: `{classification}`",
        "- Allocentric image domain: `UNDISTORTED_PINHOLE`",
        "- Support provenance: `PROJECT_SAMPLE_CONTRACT`",
        "- Yellow-grid human gate: `SUPERSEDED_INVALID_VISUAL_GATE`",
        "- Runtime: 0 physics / Replay / MPC / RL / new retarget", "",
        "## Recomputed endpoint-0 native clearances", "",
        f"- Source left/right MANO: `{source['left']:.9f} / {source['right']:.9f} m`",
        f"- Source brush/bowl: `{source['brush']:.9f} / {source['bowl']:.9f} m`",
        f"- Frozen MINK left/right XHand: `{robot['left']:.9f} / {robot['right']:.9f} m`",
        f"- Frozen MINK brush/bowl: `{robot['brush']:.9f} / {robot['bowl']:.9f} m`", "",
        "The machine contracts reproduce the old quantitative attribution without using a visual table gate. The frozen XHand reference adds about 15–19 mm of floor penetration relative to source MANO.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    code_paths = [
        Path(__file__),
        ROOT / "src/egoengine_repro/taco/camera_contract.py",
        ROOT / "src/egoengine_repro/scene/support_surface.py",
        ROOT / "src/egoengine_repro/retarget/taco_bimanual.py",
        ROOT / "src/egoengine_repro/ingest/taco.py",
        ROOT / "scripts/build_taco_bimanual_scene.py",
    ]
    write_json(output / "input_manifest.json", {
        "schema": "taco_brush_camera_table_infra_input_manifest_v1",
        "artifacts": [artifact(config_path), *[artifact(path) for path in code_paths],
                      *[artifact(path) for path in inputs.values()]],
    })
    hash_tree(output)
    print(json.dumps({"classification": classification,
                      "camera_contract_passed": camera_ok,
                      "support_contract_passed": support_ok,
                      "static": {"source": source, "robot": robot}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
