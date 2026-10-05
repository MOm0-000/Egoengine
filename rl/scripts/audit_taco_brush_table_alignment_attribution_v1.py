#!/usr/bin/env python3
"""Zero-physics attribution of Brush endpoint-0 floor penetration.

This audit freezes the existing Brush source/reference artifacts.  It never
steps MuJoCo, changes alignment, or constructs a new retarget candidate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_initialization import visual_meshes, world_vertices  # noqa: E402
from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector,
    load_official_hand_sequence,
    official_overlay,
)


SCHEMA = "taco_brush_table_alignment_attribution_v1"
FINAL_CLASSES = {
    "TABLE_PLANE_ASSUMPTION_BLOCKER",
    "RETARGET_ENVIRONMENT_COLLISION_BLOCKER",
    "SOURCE_TABLE_GEOMETRY_CONFLICT",
    "MIXED_TABLE_AND_RETARGET_BLOCKER",
    "TABLE_ATTRIBUTION_UNRESOLVED",
}
TIP_INDICES = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


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
        raise ValueError("unexpected attribution schema")
    if git_head(ROOT.parent) != cfg["baseline_commit"]:
        raise ValueError("audit must start from the pinned Brush baseline commit")
    forbidden = ("new_retarget_candidate", "physics", "replay", "mpc", "rl",
                 "promotion", "chunk_commit")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("attribution config authorizes forbidden runtime work")
    if not all(cfg["hard_rule"].values()):
        raise ValueError("all fail-closed hard rules must be enabled")
    return cfg


def sequence_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["paths"]["data_root"])
    seq = cfg["sample"]["sequence"]
    hands = root / "hand_poses/Hand_Poses" / seq
    objects = root / "object_poses/Object_Poses" / seq
    camera = root / "camera/Egocentric_Camera_Parameters" / seq
    models = root / "object_models/object_models_released"
    cache = Path(cfg["paths"]["allocentric_cache"])
    alloc_seq = cache / "Marker_Removed_Allocentric_RGB_Videos" / cfg["sample"]["triplet"] / cfg["sample"]["sequence_name"]
    return {
        "rgb": root / "rgb" / f"{cfg['sample']['episode']}.mp4",
        "hand_joints": hands / "hand_joints.npy",
        "left_hand": hands / "left_hand.pkl",
        "right_hand": hands / "right_hand.pkl",
        "left_shape": hands / "left_hand_shape.pkl",
        "right_shape": hands / "right_hand_shape.pkl",
        "tool_pose": objects / f"tool_{cfg['sample']['tool']['id']}.npy",
        "target_pose": objects / f"target_{cfg['sample']['target']['id']}.npy",
        "tool_mesh": models / f"{cfg['sample']['tool']['id']}_cm.obj",
        "target_mesh": models / f"{cfg['sample']['target']['id']}_cm.obj",
        "intrinsic": camera / "egocentric_intrinsic.txt",
        "extrinsic": camera / "egocentric_frame_extrinsic.npy",
        "scene": Path(cfg["paths"]["scene"]),
        "human_reference": Path(cfg["paths"]["baseline_run"]) / "human_reference.npz",
        "robot_reference": Path(cfg["paths"]["baseline_run"]) / "robot_reference.npz",
        "retarget_report": Path(cfg["paths"]["baseline_run"]) / "retarget_report.json",
        "allocentric_calibration": cache / "Allocentric_Camera_Parameters" / cfg["sample"]["triplet"] / cfg["sample"]["sequence_name"] / "calibration.json",
        **{f"allocentric_{camera_id}": alloc_seq / f"{camera_id}.mp4"
           for camera_id in cfg["visualization"]["allocentric_cameras"]},
    }


def transform_vertices(mesh: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray:
    return mesh.vertices @ pose[:3, :3].T + pose[:3, 3]


def video_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode {path} frame {frame}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_source(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    official = Path(cfg["paths"]["official_taco_checkout"])
    chumpy = Path(cfg["paths"]["official_projection_chumpy_dependency"]).resolve(strict=True)
    sys.path.insert(0, str(chumpy))
    hands = {}
    for side in ("left", "right"):
        vertices, joints, faces, _ = load_official_hand_sequence(
            dataset_utils=official / "dataset_utils",
            pose_path=paths[f"{side}_hand"], shape_path=paths[f"{side}_shape"],
            side=side, device=cfg["visualization"]["official_device"],
        )
        hands[side] = {"vertices": vertices, "joints": joints, "faces": faces}
    objects = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(paths[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        objects[role] = {
            "mesh": mesh,
            "poses": np.load(paths[f"{role}_pose"], allow_pickle=False).astype(float),
        }
    return {"hands": hands, "objects": objects,
            "landmarks": np.load(paths["hand_joints"], allow_pickle=False).astype(float)}


def alignment_registry(cfg: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    objects = source["objects"]
    centers = np.stack([objects["tool"]["poses"][0, :3, 3],
                        objects["target"]["poses"][0, :3, 3]])
    bowl_world = transform_vertices(objects["target"]["mesh"], objects["target"]["poses"][0])
    transform = np.eye(4)
    transform[:2, 3] = np.asarray(cfg["current_local_alignment_assumptions"]["scene_center_xy_target"]) - centers[:, :2].mean(0)
    transform[2, 3] = cfg["current_local_alignment_assumptions"]["sim_table_height_m"] - bowl_world[:, 2].min()
    return {
        "schema": "taco_brush_alignment_assumption_registry_v1",
        "computed_T_sim_world": transform,
        "assumptions": [
            {"name": "robot_base_scene_offset_magnitude_m", "value": 0.6,
             "status": "AUTHOR_PUBLISHED"},
            {"name": "sim_table_height_m", "value": 0.72,
             "status": "AUTHOR_PUBLISHED"},
            {"name": "scene_center_xy_target", "value": [0.6, 0.0],
             "formula": "[0.6,0] - endpoint0 mean object centers",
             "status": "LOCAL_ALIGNMENT_CONVENTION"},
            {"name": "source_table_proxy",
             "value": "endpoint0 target bowl native minimum z",
             "status": "LOCAL_ALIGNMENT_CONVENTION"},
            {"name": "world_to_sim_rotation", "value": "identity",
             "status": "LOCAL_ALIGNMENT_CONVENTION"},
        ],
        "no_fit_no_sweep": True,
    }


def endpoint0_clearance(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hands, objects = source["hands"], source["objects"]
    bowl_world = transform_vertices(objects["target"]["mesh"], objects["target"]["poses"][0])
    table_z = float(bowl_world[:, 2].min())
    rows: list[dict[str, Any]] = []

    def add(entity: str, side: str, kind: str, z: float) -> None:
        rows.append({"entity": entity, "side": side, "kind": kind,
                     "minimum_z_m": float(z), "table_z_m": table_z,
                     "clearance_m": float(z - table_z)})

    for side in ("left", "right"):
        add(f"{side}_MANO_mesh", side, "native_mesh", hands[side]["vertices"][0, :, 2].min())
        joints = source["landmarks"][0, 0 if side == "left" else 1]
        add(f"{side}_21_joints", side, "joint_set", joints[:, 2].min())
        add(f"{side}_wrist", side, "joint", joints[0, 2])
        for finger, index in TIP_INDICES.items():
            add(f"{side}_{finger}_tip", side, "fingertip", joints[index, 2])
    for role in ("tool", "target"):
        world = transform_vertices(objects[role]["mesh"], objects[role]["poses"][0])
        add("brush" if role == "tool" else "bowl", "none", "native_mesh", world[:, 2].min())
    report = {
        "schema": "taco_brush_source_world_endpoint0_clearance_v1",
        "coordinate_frame": "TACO source world",
        "table_proxy": "endpoint0 target bowl native minimum z",
        "table_z_m": table_z,
        "rows": rows,
        "common_translation_observation": (
            "Clearances are invariant under the current common world-to-sim translation; "
            "negative source clearances cannot be removed while keeping bowl minimum z on the floor."
        ),
    }
    return report, rows


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def support_diagnostics(cfg: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    fps = float(cfg["sample"]["fps"])
    result: dict[str, Any] = {
        "schema": "taco_brush_support_state_diagnostics_v1",
        "manual_bowl_support_state": "PENDING_HUMAN_REVIEW",
        "allowed_manual_values": ["RESTING_ON_TABLE", "HELD_ABOVE_TABLE", "AMBIGUOUS"],
        "static_motion_is_not_proof_of_table_support": True,
        "objects": {},
    }
    for role, label in (("target", "bowl"), ("tool", "brush")):
        item = source["objects"][role]
        poses = item["poses"]
        center = poses[:, :3, 3]
        translational = np.r_[0.0, np.linalg.norm(np.diff(center, axis=0), axis=1) * fps]
        relative = poses[:-1, :3, :3].swapaxes(1, 2) @ poses[1:, :3, :3]
        angular = np.r_[0.0, Rotation.from_matrix(relative).magnitude() * fps]
        min_z = np.asarray([transform_vertices(item["mesh"], pose)[:, 2].min() for pose in poses])
        windows = {}
        for name, stop in (("frames_0_20", 20), ("frames_0_40", 40), ("full_209", 208)):
            sl = slice(0, stop + 1)
            windows[name] = {
                "inclusive_frames": [0, stop],
                "translation_speed_m_s": stats(translational[sl]),
                "angular_speed_rad_s": stats(angular[sl]),
                "minimum_z_m": stats(min_z[sl]),
                "center_displacement_from_frame0_m": float(np.linalg.norm(center[stop] - center[0])),
                "minimum_z_change_from_frame0_m": float(min_z[stop] - min_z[0]),
            }
        result["objects"][label] = {
            "center_m": center, "translation_speed_m_s": translational,
            "angular_speed_rad_s": angular, "native_minimum_z_m": min_z,
            "windows": windows,
        }
    return result


def floor_attribution(cfg: dict[str, Any], paths: dict[str, Path], source: dict[str, Any],
                      registry: dict[str, Any]) -> dict[str, Any]:
    transform = np.asarray(registry["computed_T_sim_world"], dtype=float)
    floor = float(cfg["current_local_alignment_assumptions"]["sim_table_height_m"])
    source_rows = {}
    for side in ("left", "right"):
        values = source["hands"][side]["vertices"][0] @ transform[:3, :3].T + transform[:3, 3]
        source_rows[f"{side}_hand"] = float(values[:, 2].min() - floor)
    for role, label in (("tool", "brush"), ("target", "bowl")):
        values = transform_vertices(source["objects"][role]["mesh"], source["objects"][role]["poses"][0])
        values = values @ transform[:3, :3].T + transform[:3, 3]
        source_rows[label] = float(values[:, 2].min() - floor)

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    data = mujoco.MjData(model)
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        qpos = archive["qpos"][0].copy()
        qvel = archive["qvel"][0].copy()
        ctrl = archive["ctrl"][0].copy()
    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        target_tips = archive["T_sim_fingertip_target"][0].copy()
        target_wrists = archive["T_sim_wrist_target"][0].copy()
        archived_transform = archive["T_sim_world"].copy()
    if not np.array_equal(archived_transform, transform):
        raise ValueError("computed and archived T_sim_world differ")
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    meshes, _ = visual_meshes(paths["scene"], model)
    robot_visuals = []
    for geom, mesh in meshes.items():
        name = model.geom(geom).name
        values = world_vertices(model, data, geom, mesh)
        robot_visuals.append({"geom": name, "body": model.body(model.geom_bodyid[geom]).name,
                              "minimum_z_m": float(values[:, 2].min()),
                              "floor_clearance_m": float(values[:, 2].min() - floor)})
    robot_rows = {
        "right_hand": min(row["floor_clearance_m"] for row in robot_visuals if row["geom"].startswith("right_") and "object" not in row["geom"]),
        "left_hand": min(row["floor_clearance_m"] for row in robot_visuals if row["geom"].startswith("left_") and "object" not in row["geom"]),
        "brush": next(row["floor_clearance_m"] for row in robot_visuals if row["geom"] == "right_object_visual"),
        "bowl": next(row["floor_clearance_m"] for row in robot_visuals if row["geom"] == "left_object_visual"),
    }
    per_finger = {}
    for side in ("right", "left"):
        per_finger[side] = {}
        for finger in TIP_INDICES:
            selected = [row for row in robot_visuals
                        if row["geom"].startswith(f"{side}_{finger}")]
            per_finger[side][finger] = {
                "native_visuals": [row["geom"] for row in selected],
                "minimum_z_m": min(row["minimum_z_m"] for row in selected),
                "floor_clearance_m": min(row["floor_clearance_m"] for row in selected),
            }
    target_actual = {}
    for side_index, side in enumerate(("right", "left")):
        target_actual[side] = {
            "wrist": {"target_z_m": float(target_wrists[side_index, 2, 3]),
                       "actual_z_m": float(data.site(f"{side}_palm").xpos[2])},
            "fingertips": {},
        }
        for finger_index, finger in enumerate(TIP_INDICES):
            target_actual[side]["fingertips"][finger] = {
                "target_z_m": float(target_tips[side_index, finger_index, 2, 3]),
                "actual_z_m": float(data.site(f"{side}_{finger}_tip").xpos[2]),
            }
    return {
        "schema": "taco_brush_source_vs_robot_floor_attribution_v1",
        "physics_steps": 0,
        "shared_T_sim_world": transform,
        "floor_z_m": floor,
        "source_native_clearance_m": source_rows,
        "robot_native_clearance_m": robot_rows,
        "robot_minus_source_clearance_m": {key: float(robot_rows[key] - source_rows[key]) for key in source_rows},
        "robot_native_visuals": robot_visuals,
        "robot_per_finger_native_minimum": per_finger,
        "target_vs_actual_z": target_actual,
    }


def horizontal_diagnostic(paths: dict[str, Path], source: dict[str, Any]) -> dict[str, Any]:
    extrinsics = np.load(paths["extrinsic"], allow_pickle=False).astype(float)
    camera_centers = np.asarray([-e[:3, :3].T @ e[:3, 3] for e in extrinsics])
    tool = source["objects"]["tool"]["poses"][:, :3, 3]
    target = source["objects"]["target"]["poses"][:, :3, 3]
    scene_centers = (tool + target) / 2.0
    relative = camera_centers - scene_centers
    mean_relative = relative.mean(0)

    def horizontal_angle(vector: np.ndarray) -> dict[str, Any]:
        xy = np.asarray(vector[:2], dtype=float)
        norm = float(np.linalg.norm(xy))
        if norm == 0:
            return {"unit_xy": None, "unsigned_angle_to_plus_x_deg": None,
                    "signed_angle_from_plus_x_deg": None}
        unit = xy / norm
        return {"unit_xy": unit, "unsigned_angle_to_plus_x_deg": float(np.degrees(np.arccos(np.clip(unit[0], -1, 1)))),
                "signed_angle_from_plus_x_deg": float(np.degrees(np.arctan2(unit[1], unit[0])))}

    return {
        "schema": "taco_brush_horizontal_alignment_diagnostic_v1",
        "source_plus_z": [0.0, 0.0, 1.0],
        "simulator_axes": {"plus_x": [1.0, 0.0, 0.0], "plus_y": [0.0, 1.0, 0.0], "plus_z": [0.0, 0.0, 1.0]},
        "current_offset_direction": {"unit": [1.0, 0.0, 0.0], "magnitude_m": 0.6},
        "mean_ego_camera_center_relative_to_contemporaneous_object_pair_center_m": mean_relative,
        "scene_to_camera_horizontal": horizontal_angle(mean_relative),
        "camera_to_scene_horizontal": horizontal_angle(-mean_relative),
        "diagnostic_only": True,
        "alternate_transform_generated": False,
        "note": "Horizontal yaw cannot repair vertical floor penetration.",
    }


def frame_meshes(source: dict[str, Any], frame: int) -> list[trimesh.Trimesh]:
    values = [trimesh.Trimesh(source["hands"][side]["vertices"][frame],
                              source["hands"][side]["faces"], process=False)
              for side in ("right", "left")]
    for role in ("tool", "target"):
        mesh = source["objects"][role]["mesh"].copy()
        mesh.apply_transform(source["objects"][role]["poses"][frame])
        values.append(mesh)
    return values


def table_grid(source: dict[str, Any], table_z: float, margin: float,
               spacing: float) -> list[np.ndarray]:
    clouds = [source["hands"][side]["vertices"][0] for side in ("left", "right")]
    clouds.extend(transform_vertices(source["objects"][role]["mesh"],
                                     source["objects"][role]["poses"][0])
                  for role in ("tool", "target"))
    points = np.concatenate(clouds, axis=0)
    lo = points[:, :2].min(0) - margin
    hi = points[:, :2].max(0) + margin
    xs = np.arange(math.floor(lo[0] / spacing) * spacing,
                   math.ceil(hi[0] / spacing) * spacing + spacing / 2, spacing)
    ys = np.arange(math.floor(lo[1] / spacing) * spacing,
                   math.ceil(hi[1] / spacing) * spacing + spacing / 2, spacing)
    return ([np.array([[x, ys[0], table_z], [x, ys[-1], table_z]]) for x in xs]
            + [np.array([[xs[0], y, table_z], [xs[-1], y, table_z]]) for y in ys])


def configure_projector(cfg: dict[str, Any], size: tuple[int, int], intrinsic: np.ndarray,
                        extrinsic: np.ndarray) -> TacoOfficialProjector:
    projector = TacoOfficialProjector(
        dataset_utils=Path(cfg["paths"]["official_taco_checkout"]) / "dataset_utils",
        image_size=size, intrinsic=intrinsic, extrinsic=extrinsic,
        device=cfg["visualization"]["official_device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(rasterizer.raster_settings,
                                         max_faces_per_bin=400000)
    return projector


def draw_grid_and_joints(projector: TacoOfficialProjector, rgb: np.ndarray,
                         lines: list[np.ndarray], joints: list[np.ndarray]) -> np.ndarray:
    canvas = rgb.copy()
    grid_layer = canvas.copy()
    height, width = canvas.shape[:2]
    for line in lines:
        pixels, depth = projector.project_points(line)
        if np.isfinite(pixels).all() and np.isfinite(depth).all() and (depth > 0).all():
            p0, p1 = tuple(np.rint(pixels[0]).astype(int)), tuple(np.rint(pixels[1]).astype(int))
            cv2.line(grid_layer, p0, p1, (255, 255, 0), 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(grid_layer, 0.42, canvas, 0.58, 0)
    for values, color in zip(joints, ((0, 255, 0), (255, 0, 0))):
        pixels, depth = projector.project_points(values)
        for point, z in zip(pixels, depth):
            x, y = np.rint(point).astype(int)
            if z > 0 and 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
    return canvas


def visual_evidence(cfg: dict[str, Any], paths: dict[str, Path], source: dict[str, Any],
                    table_z: float, output: Path) -> dict[str, Any]:
    utils = Path(cfg["paths"]["official_taco_checkout"]) / "dataset_utils"
    visuals = output / "visuals"
    ego_dir = visuals / "egocentric"
    allo_dir = visuals / "allocentric_frame0"
    ego_dir.mkdir(parents=True)
    allo_dir.mkdir(parents=True)
    lines = table_grid(source, table_z, cfg["visualization"]["grid_margin_m"],
                       cfg["visualization"]["grid_spacing_m"])
    intrinsic = np.loadtxt(paths["intrinsic"]).astype(float)
    extrinsics = np.load(paths["extrinsic"], allow_pickle=False).astype(float)
    raw0 = video_frame(paths["rgb"], 0)
    projector = configure_projector(cfg, (raw0.shape[1], raw0.shape[0]), intrinsic, extrinsics[0])
    ego_rows = []
    for frame in cfg["visualization"]["support_keyframes"]:
        raw = video_frame(paths["rgb"], frame)
        projector.set_camera(intrinsic, extrinsics[frame])
        render = np.clip(projector.render_rgb(frame_meshes(source, frame)) * 255, 0, 255).astype(np.uint8)
        overlay = official_overlay(dataset_utils=utils, rgb=raw, render=render)
        annotated = draw_grid_and_joints(projector, overlay, lines,
                                         [source["hands"]["right"]["joints"][frame],
                                          source["hands"]["left"]["joints"][frame]])
        panel = np.concatenate([raw, annotated], axis=1)
        cv2.putText(panel, "RAW RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        cv2.putText(panel, "OFFICIAL PROJECTION + FIXED TABLE GRID", (raw.shape[1] + 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        path = ego_dir / f"frame_{frame:03d}_raw_and_table_overlay.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        ego_rows.append({"frame": frame, "path": str(path)})

    calibration = json.loads(paths["allocentric_calibration"].read_text())
    allo_rows = []
    thumbnails = []
    for camera_id in cfg["visualization"]["allocentric_cameras"]:
        raw = video_frame(paths[f"allocentric_{camera_id}"], 0)
        params = calibration[camera_id]
        expected_size = tuple(params["imgSize"])
        if (raw.shape[1], raw.shape[0]) != expected_size:
            raise ValueError(f"allocentric {camera_id} video/calibration size mismatch")
        k = np.asarray(params["K"], dtype=float).reshape(3, 3)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.asarray(params["R"], dtype=float).reshape(3, 3)
        extrinsic[:3, 3] = np.asarray(params["T"], dtype=float)
        allo_projector = configure_projector(cfg, expected_size, k, extrinsic)
        render = np.clip(allo_projector.render_rgb(frame_meshes(source, 0)) * 255, 0, 255).astype(np.uint8)
        overlay = official_overlay(dataset_utils=utils, rgb=raw, render=render)
        annotated = draw_grid_and_joints(allo_projector, overlay, lines,
                                         [source["hands"]["right"]["joints"][0],
                                          source["hands"]["left"]["joints"][0]])
        panel = np.concatenate([raw, annotated], axis=1)
        cv2.putText(panel, f"{camera_id} RAW", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .58,
                    (255, 255, 255), 2)
        cv2.putText(panel, "OFFICIAL PROJECTION + TABLE", (raw.shape[1] + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
        path = allo_dir / f"camera_{camera_id}_frame_000.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        allo_rows.append({"camera": camera_id, "path": str(path),
                          "calibration_img_size": list(expected_size),
                          "distortion_coefficients_present": True,
                          "projection": "pinned official TACO PyTorch3D wrapper"})
        thumb = cv2.resize(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), (768, 282))
        thumbnails.append(thumb)
    contact_sheet = allo_dir / "allocentric_12view_frame0_contact_sheet.jpg"
    sheet_rows = [np.concatenate(thumbnails[i:i + 3], axis=1) for i in range(0, 12, 3)]
    cv2.imwrite(str(contact_sheet), np.concatenate(sheet_rows, axis=0))
    return {
        "schema": "taco_brush_table_alignment_visual_manifest_v1",
        "official_projection_implementation": str(utils / "pyt3d_wrapper.py"),
        "grid": {"normal_source_world": [0, 0, 1], "z_m": table_z,
                 "margin_m": cfg["visualization"]["grid_margin_m"],
                 "spacing_m": cfg["visualization"]["grid_spacing_m"],
                 "image_fitted": False},
        "egocentric": ego_rows,
        "allocentric": allo_rows,
        "allocentric_contact_sheet": artifact(contact_sheet),
        "manual_review_required": True,
    }


def source_pins(cfg: dict[str, Any]) -> dict[str, Any]:
    official = Path(cfg["paths"]["official_taco_checkout"])
    if git_head(official) != cfg["pins"]["official_taco"]:
        raise ValueError("official TACO pin mismatch")
    return {
        "schema": "taco_brush_table_alignment_source_pins_v1",
        "active_repository": {"path": str(ROOT.parent), "commit": git_head(ROOT.parent)},
        "baseline_commit": cfg["baseline_commit"],
        "official_taco": {"path": str(official), "commit": git_head(official)},
        "mink": cfg["pins"]["mink"], "spider": cfg["pins"]["spider"],
    }


def write_review(output: Path, visual: dict[str, Any]) -> None:
    sheet = Path(visual["allocentric_contact_sheet"]["path"]).relative_to(output)
    lines = [
        "# Table-plane visual review", "",
        "This is a required human semantic gate. The fixed grid was not fitted to pixels.", "",
    ]
    for row in visual["egocentric"]:
        ego = Path(row["path"]).relative_to(output)
        lines.extend([f"## Egocentric frame {row['frame']}", "",
                      f"![egocentric frame {row['frame']}]({ego})", ""])
    lines.extend([
        "## Allocentric frame 0", "",
        f"![12 allocentric views]({sheet})", "",
        "## Required answers", "",
        "- `table_plane_alignment`: `PENDING`",
        "  - allowed: `CLEARLY_ALIGNED_WITH_VISIBLE_TABLE`",
        "  - allowed: `CLEARLY_MISALIGNED_WITH_VISIBLE_TABLE`",
        "  - allowed: `TABLE_NOT_VISUALLY_IDENTIFIABLE`",
        "- `bowl_support_state`: `PENDING`",
        "  - allowed: `RESTING_ON_TABLE`",
        "  - allowed: `HELD_ABOVE_TABLE`",
        "  - allowed: `AMBIGUOUS`", "",
        "No z offset or alternate alignment may be inferred automatically from this review.",
    ])
    (output / "TABLE_PLANE_VISUAL_REVIEW.md").write_text("\n".join(lines) + "\n")


def write_summary(output: Path, clearance: dict[str, Any], attribution: dict[str, Any],
                  decision: dict[str, Any]) -> None:
    source = {row["entity"]: row["clearance_m"] for row in clearance["rows"]}
    robot = attribution["robot_native_clearance_m"]
    lines = [
        "# TACO Brush table-alignment attribution v1", "",
        f"- Classification: `{decision['classification']}`",
        "- Human table-plane/support review: **pending**",
        "- Physics / Replay / MPC / RL / new retarget: `0 / 0 / 0 / 0 / 0`", "",
        "## Endpoint-0 clearances", "",
        f"- Source left MANO: `{source['left_MANO_mesh']:.9f} m`",
        f"- Source right MANO: `{source['right_MANO_mesh']:.9f} m`",
        f"- Source brush: `{source['brush']:.9f} m`",
        f"- Source bowl: `{source['bowl']:.9f} m` (plane definition)",
        f"- MINK left XHand: `{robot['left_hand']:.9f} m`",
        f"- MINK right XHand: `{robot['right_hand']:.9f} m`",
        f"- MINK brush: `{robot['brush']:.9f} m`", "",
        "The quantitative attribution is complete, but the contract forbids choosing a non-unresolved final class until a human classifies the projected table plane and bowl support state.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def hash_tree(output: Path) -> None:
    lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()
                       and p.name != "server_artifacts.sha256"):
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = load_config(config_path)
    output = Path(cfg["paths"]["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = sequence_paths(cfg)
    for path in paths.values():
        path.resolve(strict=True)
    source = load_source(cfg, paths)
    registry = alignment_registry(cfg, source)
    write_json(output / "source_pins.json", source_pins(cfg))
    write_json(output / "alignment_assumption_registry.json", registry)
    clearance, rows = endpoint0_clearance(source)
    write_json(output / "source_world_endpoint0_clearance.json", clearance)
    with (output / "source_world_endpoint0_clearance.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    support = support_diagnostics(cfg, source)
    write_json(output / "support_state_diagnostics.json", support)
    attribution = floor_attribution(cfg, paths, source, registry)
    write_json(output / "source_vs_robot_floor_attribution.json", attribution)
    write_json(output / "horizontal_alignment_diagnostic.json", horizontal_diagnostic(paths, source))
    visual = visual_evidence(cfg, paths, source, clearance["table_z_m"], output)
    write_json(output / "visual_manifest.json", visual)
    write_review(output, visual)
    input_paths = [config_path, Path(__file__), *paths.values()]
    write_json(output / "input_manifest.json", {
        "schema": "taco_brush_table_alignment_input_manifest_v1",
        "artifacts": [artifact(path) for path in input_paths],
    })
    decision = {
        "schema": "taco_brush_table_alignment_decision_v1",
        "classification": "TABLE_ATTRIBUTION_UNRESOLVED",
        "allowed_final_classes": sorted(FINAL_CLASSES),
        "reason": "Required manual table-plane alignment and bowl support-state review is pending.",
        "runtime_counts": {"physics_steps": 0, "replay_runs": 0, "mpc_runs": 0,
                           "rl_runs": 0, "new_retarget_candidates": 0,
                           "promotions": 0, "chunk_commits": 0},
        "no_transform_adjustment": True,
    }
    write_json(output / "decision.json", decision)
    write_summary(output, clearance, attribution, decision)
    visual_lines = ["# Visual evidence", "",
                    f"- [required review](TABLE_PLANE_VISUAL_REVIEW.md)",
                    f"- [12-view allocentric contact sheet]({Path(visual['allocentric_contact_sheet']['path']).relative_to(output)})"]
    visual_lines += [f"- [egocentric frame {row['frame']}]({Path(row['path']).relative_to(output)})"
                     for row in visual["egocentric"]]
    visual_lines += [f"- [allocentric {row['camera']}]({Path(row['path']).relative_to(output)})"
                     for row in visual["allocentric"]]
    (output / "VISUAL_INDEX.md").write_text("\n".join(visual_lines) + "\n")
    hash_tree(output)
    print(json.dumps({"classification": decision["classification"],
                      "output": str(output),
                      "source_clearance_m": {row["entity"]: row["clearance_m"] for row in rows},
                      "robot_clearance_m": attribution["robot_native_clearance_m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
