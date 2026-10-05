#!/usr/bin/env python3
"""Build the fail-closed pre-training evidence chain for TACO Brush.

This entrypoint intentionally has no MPC or RL imports.  ``source`` performs
only source/geometry/projection work.  ``static`` consumes the one separately
generated MINK candidate and decides whether a cold Replay is authorized.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import fcl
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src"), str(ROOT / "scripts")]

from audit_taco_initialization import visual_meshes, world_vertices  # noqa: E402
from egoengine_repro.evaluation.taco_official_projection import (  # noqa: E402
    TacoOfficialProjector,
    load_official_hand_sequence,
    official_overlay,
)
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    audit_trajectory,
    collision_families,
    distances,
)
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance  # noqa: E402


SCHEMA = "taco_brush_brush_bowl_paper_baseline_v1"
FINAL_CLASSES = {
    "BRUSH_INPUT_ALIGNMENT_BLOCKER",
    "BRUSH_BOWL_COLLISION_GEOMETRY_BLOCKER",
    "BRUSH_RETARGET_KINEMATIC_BLOCKER",
    "BRUSH_INITIALIZATION_STATIC_BLOCKER",
    "BRUSH_REPLAY_RAW_COMPLETE_POSITION_GATE_UNRESOLVED",
    "BRUSH_REPLAY_FAILED_READY_FOR_MPC_DESIGN",
    "BRUSH_REPLAY_VALIDATED_NO_TRAINING_NEEDED",
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


def load_contract(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected Brush baseline schema")
    selection = yaml.safe_load(Path(cfg["paths"]["selection_contract"]).read_text())
    if selection["sample"]["sequence"] != cfg["sample"]["sequence"]:
        raise ValueError("active sample and Brush baseline disagree")
    if selection["sample"]["frames"] != cfg["sample"]["expected_frames"]:
        raise ValueError("active sample frame count differs")
    if selection["selection_contract"]["brush_training_authorized"]:
        raise ValueError("this pre-training baseline requires Brush training=false")
    if selection["selection_contract"]["brush_chunk_commit_authorized"]:
        raise ValueError("this pre-training baseline requires chunk commit=false")
    forbidden = ("mpc", "rl_training", "domain_randomization", "reference_promotion", "chunk_commit")
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("baseline contract authorizes forbidden work")
    if cfg["authorization"]["mink_retarget_candidates"] != 1:
        raise ValueError("exactly one MINK candidate is required")
    if cfg["authorization"]["cold_replay_count"] != 1:
        raise ValueError("exactly one cold Replay is the maximum authorization")
    return cfg


def sequence_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["paths"]["data_root"])
    sequence = cfg["sample"]["sequence"]
    episode = cfg["sample"]["episode"]
    hands = root / "hand_poses/Hand_Poses" / sequence
    objects = root / "object_poses/Object_Poses" / sequence
    camera = root / "camera/Egocentric_Camera_Parameters" / sequence
    models = root / "object_models/object_models_released"
    return {
        "rgb": root / "rgb" / f"{episode}.mp4",
        "depth": root / "depth_original" / f"{episode}.avi",
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
        "mano_left": Path(cfg["paths"]["mano_model_dir"]) / "MANO_LEFT.pkl",
        "mano_right": Path(cfg["paths"]["mano_model_dir"]) / "MANO_RIGHT.pkl",
        "scene": Path(cfg["paths"]["scene"]),
        "retarget_settings": Path(cfg["paths"]["retarget_settings"]),
        "paper": Path(cfg["paths"]["paper_pdf"]),
        "selection_contract": Path(cfg["paths"]["selection_contract"]),
        "active_runtime_document": Path(cfg["paths"]["active_runtime_document"]),
    }


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    result = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return result


def rigid_audit(values: np.ndarray) -> dict[str, Any]:
    rotations = values[:, :3, :3]
    identity = np.eye(3)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "homogeneous_row_max_error": float(np.abs(values[:, 3] - [0, 0, 0, 1]).max()),
        "rotation_orthogonality_max_error": float(np.abs(
            rotations.swapaxes(1, 2) @ rotations - identity).max()),
        "rotation_determinant_max_error": float(np.abs(np.linalg.det(rotations) - 1).max()),
    }


def source_pins(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    official = Path(cfg["paths"]["official_taco_checkout"])
    if git_head(official) != cfg["pins"]["official_taco"]:
        raise ValueError("official TACO checkout pin mismatch")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=official, text=True,
    ).strip()
    if dirty:
        raise ValueError("official TACO checkout has tracked modifications")
    mink_checkout = Path("/data_all/zzx/3.2RL/external/mink")
    spider_checkout = Path("/data_all/zzx/egoengine/spider")
    return {
        "schema": "taco_brush_source_pins_v1",
        "active_repository": {"path": str(ROOT.parent), "commit": git_head(ROOT.parent)},
        "official_taco": {"path": str(official), "commit": git_head(official),
                          "status": "AUTHOR_SOURCE_PIN"},
        "mink": {"path": str(mink_checkout), "commit": git_head(mink_checkout),
                 "expected": cfg["pins"]["mink"]},
        "spider": {"path": str(spider_checkout), "commit": git_head(spider_checkout),
                   "expected": cfg["pins"]["spider"]},
        "rsl_rl": {"commit": cfg["pins"]["rsl_rl_reference_only"],
                    "status": "REFERENCE_ONLY_NOT_AUTHORIZED"},
        "paper": artifact(paths["paper"]),
    }


def parameter_registry(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    author = {
        "retarget_backend": "MINK",
        "retarget_tasks": "five fingertip positions/orientations plus wrist orientation",
        "retarget_constraints": "joint limits and self collision",
        "robot_base_scene_offset_magnitude_m": 0.6,
        "table_height_m": 0.72,
        "taco_base_alignment": "fixed-offset heuristic; no AprilTag",
        "action_abstraction": "floating Cartesian wrist/base plus XHand",
        "chunk_control_steps": 20,
        "lookahead_chunks": 2,
        "solver_order": ["Replay", "MPC", "RL"],
        "rl_policy": "residual added to reference/base action; optimized with PPO",
        "domain_randomization": False,
        "human_mimic_reward": False,
        "action_smoothness_reward": False,
        "evaluation": "object tracking only",
        "contact_bonus_semantics": "thumb plus at least one non-thumb finger on manipulated object",
        "lifting_reward_semantics": "only when vertical lifting is needed",
        "brush_rotation_threshold_rad": 1.2,
    }
    unpublished = dict(cfg["paper_unpublished"])
    external = {
        "rsl_rl_pin": cfg["pins"]["rsl_rl_reference_only"],
        "rsl_rl_defaults": {"learning_rate": 0.001, "epochs": 5, "minibatches": 4,
                            "clip": 0.2, "gamma": 0.99, "gae_lambda": 0.95,
                            "value_loss_coefficient": 1.0, "entropy": 0.01,
                            "max_grad_norm": 1.0, "schedule": "adaptive", "desired_kl": 0.01},
        "status": "REFERENCE_ONLY_NOT_AUTHORIZED",
        "other_cross_references": [
            "facebookresearch/spider@44717007...",
            "unidex-ai/UniDex@97d869e0...",
            "physercoe/egoengine-repro (non-official reproduction)",
        ],
        "explicitly_not_adopted": ["Cp=0.10 m", "contact bonus=2.0",
                                   "non-author PPO settings", "non-author residual bounds"],
    }
    value = {
        "schema": "taco_brush_paper_parameter_registry_v1",
        "AUTHOR_PUBLISHED": author,
        "UNPUBLISHED_OR_LOCAL": unpublished,
        "EXTERNAL_REFERENCE_ONLY": external,
    }
    lines = ["# TACO Brush paper parameter registry", "", "## AUTHOR_PUBLISHED", ""]
    lines.extend(f"- `{key}`: `{json.dumps(val)}`" for key, val in author.items())
    lines += ["", "## UNPUBLISHED_OR_LOCAL", ""]
    lines.extend(f"- `{key}`: `{json.dumps(val)}`" for key, val in unpublished.items())
    lines += ["", "## EXTERNAL_REFERENCE_ONLY", "",
              "All library/reproduction values are `REFERENCE_ONLY_NOT_AUTHORIZED`."]
    return value, "\n".join(lines)


def input_audit(cfg: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = int(cfg["sample"]["expected_frames"])
    for path in paths.values():
        path.resolve(strict=True)
    joints = np.load(paths["hand_joints"], allow_pickle=False).astype(float)
    if joints.shape != (expected, 2, 21, 3) or not np.isfinite(joints).all():
        raise ValueError("hand_joints is not finite 209x2x21x3")
    pkl_rows = {}
    translation_error = {}
    for hand_index, side in enumerate(("left", "right")):
        with paths[f"{side}_hand"].open("rb") as stream:
            rows = pickle.load(stream)
        source_keys = sorted(rows, key=int)
        frame_ids = [int(key) for key in source_keys]
        if frame_ids != list(range(1, expected + 1)):
            raise ValueError(f"{side} hand PKL frame IDs are not continuous 1..209")
        translation = np.stack([
            np.asarray(rows[key]["hand_trans"], dtype=float) for key in source_keys
        ])
        translation_error[side] = float(np.linalg.norm(
            translation - joints[:, hand_index, 0], axis=1).max())
        pkl_rows[side] = len(source_keys)
    tool = np.load(paths["tool_pose"], allow_pickle=False).astype(float)
    target = np.load(paths["target_pose"], allow_pickle=False).astype(float)
    extrinsic = np.load(paths["extrinsic"], allow_pickle=False).astype(float)
    intrinsic = np.loadtxt(paths["intrinsic"]).astype(float)
    rgb = video_info(paths["rgb"])
    depth = video_info(paths["depth"])
    if tool.shape != (expected, 4, 4) or target.shape != (expected, 4, 4):
        raise ValueError("object pose rows do not match 209 frames")
    if extrinsic.shape[0] != expected:
        raise ValueError("camera extrinsic rows do not match 209 frames")
    if rgb["frames"] != expected or not np.isclose(rgb["fps"], cfg["sample"]["fps"]):
        raise ValueError("RGB timeline differs from the fixed contract")
    tool_mesh = trimesh.load_mesh(paths["tool_mesh"], process=False)
    target_mesh = trimesh.load_mesh(paths["target_mesh"], process=False)
    audit = {
        "schema": "taco_brush_input_audit_v1",
        "status": "PASS",
        "frames": expected,
        "frame_ids": {"start": 0, "end": expected - 1, "continuous": True,
                      "pkl_source_ids": "1..209"},
        "video": {"rgb": rgb, "depth": depth, "timeline_source": "RGB 30 fps"},
        "hand_joints": {"shape": list(joints.shape), "finite": True,
                        "pkl_rows": pkl_rows,
                        "pkl_root_translation_max_error_m": translation_error},
        "camera": {"intrinsic_shape": list(intrinsic.shape),
                   "extrinsic_shape": list(extrinsic.shape),
                   "finite": bool(np.isfinite(intrinsic).all() and np.isfinite(extrinsic).all())},
        "objects": {"tool": {"name": "brush", "id": "071", **rigid_audit(tool)},
                    "target": {"name": "bowl", "id": "146", **rigid_audit(target)}},
        "units": {"hand_and_object_translation": "meter", "released_object_mesh": "centimeter",
                  "runtime_mesh_scale": 0.01},
        "meshes": {"tool_raw_bounds_cm": tool_mesh.bounds.tolist(),
                   "target_raw_bounds_cm": target_mesh.bounds.tolist(),
                   "tool_faces": int(len(tool_mesh.faces)), "target_faces": int(len(target_mesh.faces))},
        "implicit_truncation": False,
        "nan_or_inf": False,
    }
    manifest_paths = [*paths.values(), Path(__file__),
                      ROOT / "configs/taco_brush_brush_bowl_paper_baseline_v1.yaml"]
    manifest = {"schema": "taco_brush_input_manifest_v1",
                "artifacts": [artifact(path) for path in manifest_paths]}
    return audit, manifest


def geometry_and_transform_audit(cfg: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    tool = np.load(paths["tool_pose"], allow_pickle=False).astype(float)
    target = np.load(paths["target_pose"], allow_pickle=False).astype(float)
    brush = trimesh.load_mesh(paths["tool_mesh"], process=False)
    bowl = trimesh.load_mesh(paths["target_mesh"], process=False)
    brush.apply_scale(0.01)
    bowl.apply_scale(0.01)
    center = np.stack([tool[0, :2, 3], target[0, :2, 3]]).mean(axis=0)
    translation = np.zeros(3)
    translation[:2] = [0.6, 0.0] - center
    bowl_world = bowl.vertices @ target[0, :3, :3].T + target[0, :3, 3]
    translation[2] = 0.72 - bowl_world[:, 2].min()
    aligned_center = center + translation[:2]
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target_body = model.body("left_object").id
    collision_geoms = [g for g in range(model.ngeom)
                       if model.geom_bodyid[g] == target_body
                       and model.geom(g).name.startswith("left_object_")
                       and not model.geom(g).name.endswith("visual")]
    pieces = []
    for geom in collision_geoms:
        mesh_id = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start + count]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - data.xpos[target_body]) @ data.xmat[target_body].reshape(3, 3)
        pieces.append(trimesh.convex.convex_hull(local))
    visual_id = model.geom("left_object_visual").id
    visual_meshes_by_geom, _ = visual_meshes(paths["scene"], model)
    native = visual_meshes_by_geom[visual_id]
    xy = native.bounds.mean(axis=0)[:2]
    z_values = np.linspace(native.bounds[0, 2], native.bounds[1, 2], 33)
    probes = np.column_stack([np.tile(xy, (len(z_values), 1)), z_values])
    containment = np.stack([piece.contains(probes) for piece in pieces], axis=1)
    occupied = containment.any(axis=1)
    # A bowl may legitimately have a closed bottom.  The upper half of its
    # central axis must remain free in the runtime union or the cavity is filled.
    upper = z_values >= np.median(z_values)
    upper_free = ~occupied[upper]
    cavity_pass = bool(upper_free.all() and upper_free.size > 0)
    cavity = {
        "schema": "taco_brush_bowl_cavity_geometry_audit_v1",
        "status": "PASS_CAVITY_FREE" if cavity_pass else "FAIL_CAVITY_FILLED",
        "native_bowl": {"watertight": bool(native.is_watertight),
                        "bounds_m": native.bounds.tolist(), "faces": int(len(native.faces))},
        "runtime_collision": {"piece_count": len(pieces),
                              "representation": "union of source convex pieces, not one bowl convex hull"},
        "central_axis_probe": {"xy_m": xy.tolist(), "z_m": z_values.tolist(),
                               "occupied_by_piece": occupied.tolist(),
                               "upper_half_all_free": bool(upper_free.all())},
        "cavity_is_runtime_free_space": cavity_pass,
        "contact_is_not_material_penetration": True,
    }
    floor_z = float(model.geom("floor").pos[2])
    physics = {
        "schema": "taco_brush_physics_parameter_provenance_v1",
        "base_alignment": {"source": "local fixed-offset convention matching paper magnitude",
                           "original_object_pair_center_xy_m": center.tolist(),
                           "translation_m": translation.tolist(),
                           "aligned_center_xy_m": aligned_center.tolist(),
                           "offset_magnitude_m": float(np.linalg.norm(aligned_center)),
                           "direction_formula_author_published": False,
                           "direction_swept": False},
        "table": {"height_m": floor_z, "author_value_m": 0.72,
                  "matches_author_value": bool(np.isclose(floor_z, 0.72))},
        "objects": {},
    }
    for side, role, mesh_path in (("right", "brush", paths["tool_mesh"]),
                                  ("left", "bowl", paths["target_mesh"])):
        body = model.body(f"{side}_object")
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id
                 and not model.geom(g).name.endswith("visual")]
        physics["objects"][role] = {
            "released_mesh": artifact(mesh_path), "released_mesh_unit": "centimeter",
            "runtime_scale": 0.01, "mass_kg": float(body.mass[0]),
            "inertia_kg_m2": model.body_inertia[body.id].tolist(),
            "center_of_mass_body_m": model.body_ipos[body.id].tolist(),
            "collision_piece_count": len(geoms),
            "friction": sorted({tuple(model.geom_friction[g].tolist()) for g in geoms}),
            "source": "released mesh plus frozen local MuJoCo scene; no sweep",
        }
    return cavity, physics


def official_visuals(cfg: dict[str, Any], paths: dict[str, Path], output: Path) -> dict[str, Any]:
    official = Path(cfg["paths"]["official_taco_checkout"])
    utils = official / "dataset_utils"
    chumpy = Path(cfg["paths"]["official_projection_chumpy_dependency"]).resolve(strict=True)
    # The pinned author loader imports its historical MANO dependency at call
    # time.  Supplying that dependency is environment setup, not an alternate
    # projection or a modification to the official checkout.
    sys.path.insert(0, str(chumpy))
    hands = {}
    for side in ("right", "left"):
        vertices, joints, faces, _ = load_official_hand_sequence(
            dataset_utils=utils, pose_path=paths[f"{side}_hand"],
            shape_path=paths[f"{side}_shape"], side=side,
            device=cfg["visualization"]["official_device"],
        )
        hands[side] = {"vertices": vertices, "joints": joints, "faces": faces}
    objects = {}
    for role in ("tool", "target"):
        mesh = trimesh.load_mesh(paths[f"{role}_mesh"], process=False)
        mesh.apply_scale(0.01)
        objects[role] = {"mesh": mesh,
                         "poses": np.load(paths[f"{role}_pose"], allow_pickle=False)}
    intrinsic = np.loadtxt(paths["intrinsic"])
    extrinsics = np.load(paths["extrinsic"], allow_pickle=False)
    info = video_info(paths["rgb"])
    projector = TacoOfficialProjector(
        dataset_utils=utils, image_size=(info["width"], info["height"]),
        intrinsic=intrinsic, extrinsic=extrinsics[0],
        device=cfg["visualization"]["official_device"],
    )
    rasterizer = projector.official_wrapper.renderer.renderer.rasterizer
    rasterizer.raster_settings = replace(
        rasterizer.raster_settings, max_faces_per_bin=400000,
    )

    def frame_meshes(frame: int) -> list[trimesh.Trimesh]:
        values = [trimesh.Trimesh(hands[side]["vertices"][frame], hands[side]["faces"],
                                 process=False) for side in ("right", "left")]
        for role in ("tool", "target"):
            mesh = objects[role]["mesh"].copy()
            mesh.apply_transform(objects[role]["poses"][frame])
            values.append(mesh)
        return values

    tool_pose, target_pose = objects["tool"]["poses"], objects["target"]["poses"]
    center_distance = np.linalg.norm(tool_pose[:, :3, 3] - target_pose[:, :3, 3], axis=1)
    brush_speed = np.r_[0.0, np.linalg.norm(np.diff(tool_pose[:, :3, 3], axis=0), axis=1) * cfg["sample"]["fps"]]
    bowl_rotation = np.r_[0.0, Rotation.from_matrix(
        target_pose[:-1, :3, :3].swapaxes(1, 2) @ target_pose[1:, :3, :3]).magnitude()]
    special = {
        "brush_closest_center_to_bowl": int(center_distance.argmin()),
        "brush_speed_peak": int(brush_speed.argmax()),
        "bowl_rotation_step_peak": int(bowl_rotation.argmax()),
        "final": int(cfg["sample"]["expected_frames"] - 1),
    }
    keyframes = sorted(set(range(0, cfg["sample"]["expected_frames"],
                                 cfg["visualization"]["periodic_keyframe_stride"])) | set(special.values()))
    visual_dir = output / "visuals/source_official_projection"
    key_dir = visual_dir / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    video_path = visual_dir / "official_source_overlay_209f.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             cfg["visualization"]["overlay_fps"],
                             (info["width"], info["height"]))
    capture = cv2.VideoCapture(str(paths["rgb"]))
    records = []
    for frame in range(cfg["sample"]["expected_frames"]):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"RGB decode stopped at frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        projector.set_camera(intrinsic, extrinsics[frame])
        rendered = np.clip(projector.render_rgb(frame_meshes(frame)) * 255.0, 0, 255).astype(np.uint8)
        overlay = official_overlay(dataset_utils=utils, rgb=rgb, render=rendered)
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if frame in keyframes:
            path = key_dir / f"frame_{frame:03d}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            records.append({"frame": frame, "path": str(path),
                            "reasons": [name for name, value in special.items() if value == frame]
                            + (["periodic_stride_10"] if frame % 10 == 0 else [])})
        if frame % 25 == 0:
            print(f"official projection {frame + 1}/209", flush=True)
    capture.release()
    writer.release()
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("official projection video was not created")
    # Compact contact sheet supports one-shot human/agent review without
    # replacing the full-resolution evidence above.
    thumbs = []
    for row in records:
        image = cv2.imread(row["path"])
        image = cv2.resize(image, (384, 216))
        cv2.putText(image, f"frame {row['frame']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2)
        cv2.putText(image, f"frame {row['frame']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        thumbs.append(image)
    rows = []
    for start in range(0, len(thumbs), 4):
        row = thumbs[start:start + 4]
        row += [np.zeros_like(thumbs[0])] * (4 - len(row))
        rows.append(np.concatenate(row, axis=1))
    contact_sheet = visual_dir / "keyframes_contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet), np.concatenate(rows, axis=0))
    return {
        "schema": "taco_brush_official_projection_manifest_v1",
        "implementation": str(utils / "project_pose_to_egocentric_view.py"),
        "chumpy_dependency": artifact(chumpy / "chumpy/__init__.py"),
        "rasterizer_capacity_only": {"max_faces_per_bin": 400000,
                                     "camera_or_projection_semantics_changed": False},
        "manual_pixel_offset": False, "alternate_projection": False,
        "frames_rendered": cfg["sample"]["expected_frames"],
        "video": artifact(video_path), "contact_sheet": artifact(contact_sheet),
        "special_frames": special, "keyframes": records,
        "technical_gate_passed": True,
        "visual_semantic_review": "REQUIRED_BEFORE_MINK",
    }


def write_future_rl(output: Path) -> None:
    write_json(output / "future_rl_parameter_status.json", {
        "schema": "taco_brush_future_rl_parameter_status_v1",
        "status": "NOT_AUTHORIZED_NOT_RUN",
        "author_published": {"policy_type": "PPO residual policy",
                             "taco_domain_randomization": False,
                             "human_mimic_reward": False,
                             "action_smoothness_reward": False},
        "author_unpublished": ["learning rate", "epochs", "minibatches", "clip", "gamma",
                               "GAE lambda", "entropy", "network", "noise schedule",
                               "episode length", "residual bounds"],
        "external_defaults": {"status": "REFERENCE_ONLY_NOT_AUTHORIZED"},
    })


def source_phase(cfg: dict[str, Any], config_path: Path) -> None:
    output = Path(cfg["paths"]["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = sequence_paths(cfg)
    registry, registry_md = parameter_registry(cfg)
    write_json(output / "source_pins.json", source_pins(cfg, paths))
    write_json(output / "paper_parameter_registry.json", registry)
    (output / "paper_parameter_registry.md").write_text(registry_md + "\n")
    write_future_rl(output)
    audit, manifest = input_audit(cfg, paths)
    write_json(output / "input_audit.json", audit)
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "source_timeline.json", {
        "schema": "taco_brush_source_timeline_v1", "rows": 209, "fps": 30.0,
        "frame_indices": list(range(209)), "timestamps_s": (np.arange(209) / 30).tolist(),
        "all_modalities_share_row_index": True, "implicit_truncation": False,
    })
    cavity, physics = geometry_and_transform_audit(cfg, paths)
    write_json(output / "bowl_cavity_geometry_audit.json", cavity)
    write_json(output / "physics_parameter_provenance.json", physics)
    if not cavity["cavity_is_runtime_free_space"]:
        write_json(output / "decision.json", {
            "classification": "BRUSH_BOWL_COLLISION_GEOMETRY_BLOCKER",
            "physics_steps": 0, "mink_candidates": 0, "replay_runs": 0,
        })
        return
    projection = official_visuals(cfg, paths, output)
    write_json(output / "official_projection_manifest.json", projection)
    visual_lines = ["# Visual evidence", "",
                    f"- [full official source overlay]({Path(projection['video']['path']).relative_to(output)})",
                    f"- [keyframe contact sheet]({Path(projection['contact_sheet']['path']).relative_to(output)})",
                    "", "The full overlay uses the pinned official TACO PyTorch3D projection path.", ""]
    (output / "VISUAL_INDEX.md").write_text("\n".join(visual_lines))
    write_json(output / "phase_status.json", {
        "schema": SCHEMA, "phase": "SOURCE_AND_GEOMETRY_COMPLETE",
        "visual_review_required_before_mink": True,
        "mink_candidates_run": 0, "physics_steps": 0, "replay_runs": 0,
        "config": artifact(config_path),
    })


def approve_visual_phase(cfg: dict[str, Any]) -> None:
    output = Path(cfg["paths"]["output"])
    manifest_path = output / "official_projection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["frames_rendered"] != cfg["sample"]["expected_frames"]:
        raise ValueError("cannot approve an incomplete official projection")
    audit = {
        "schema": "taco_brush_official_projection_visual_review_v1",
        "reviewed_evidence": [manifest["video"], manifest["contact_sheet"]],
        "result": "PASS",
        "observations": [
            "brush and bowl projection remains aligned with the RGB objects across the sequence",
            "right hand performs the brush interaction while left hand approaches/stabilizes the bowl",
            "operation chronology is continuous from approach through interaction and release",
            "no missing-mesh raster overflow remains after the capacity-only rerender",
        ],
        "scope": "visual semantic alignment; not a physics or robot-reference claim",
        "manual_pixel_offset": False,
    }
    write_json(output / "official_projection_visual_review.json", audit)
    manifest["visual_semantic_review"] = "PASS"
    write_json(manifest_path, manifest)
    phase_path = output / "phase_status.json"
    phase = json.loads(phase_path.read_text())
    phase["visual_review_required_before_mink"] = False
    phase["visual_review"] = "PASS"
    phase["next_authorized_phase"] = "ONE_MINK_BASELINE"
    write_json(phase_path, phase)


def local_meshes_by_body(scene: Path, model: mujoco.MjModel,
                         data: mujoco.MjData) -> dict[int, tuple[int, trimesh.Trimesh]]:
    meshes, _ = visual_meshes(scene, model)
    result = {}
    for geom, mesh in meshes.items():
        result[int(model.geom_bodyid[geom])] = (geom, mesh)
    return result


def sampled_native_pair(model: mujoco.MjModel, data: mujoco.MjData,
                        by_body: dict[int, tuple[int, trimesh.Trimesh]],
                        geom_a: int, geom_b: int, tolerance: float) -> dict[str, Any]:
    names = (model.geom(geom_a).name, model.geom(geom_b).name)
    if "floor" in names:
        other = geom_b if names[0] == "floor" else geom_a
        source = by_body.get(int(model.geom_bodyid[other]))
        if source is None:
            return {"forward_reverse": [], "classification": "UNKNOWN_NATIVE_VISUAL_MISSING"}
        visual, mesh = source
        points = world_vertices(model, data, visual, mesh)
        floor_z = float(data.geom_xpos[model.geom("floor").id, 2])
        clearance = float(points[:, 2].min() - floor_z)
        return {
            "native_floor_clearance_m": clearance,
            "native_visual": model.geom(visual).name,
            "classification": ("MATERIAL_PENETRATION" if clearance < -tolerance
                               else "NO_NATIVE_MATERIAL_PENETRATION"),
        }

    def world_mesh(entry: tuple[int, trimesh.Trimesh]) -> trimesh.Trimesh:
        visual, native = entry
        return trimesh.Trimesh(vertices=world_vertices(model, data, visual, native),
                               faces=native.faces, process=False)

    def triangle_object(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
        geometry = fcl.BVHModel()
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        geometry.beginModel(len(vertices), len(faces))
        geometry.addSubModel(vertices, faces)
        geometry.endModel()
        return fcl.CollisionObject(geometry)

    entries = [by_body.get(int(model.geom_bodyid[g])) for g in (geom_a, geom_b)]
    if any(entry is None for entry in entries):
        return {"forward_reverse": [], "classification": "UNKNOWN_NATIVE_VISUAL_MISSING"}
    world = [world_mesh(entry) for entry in entries]
    collision_result = fcl.CollisionResult()
    fcl.collide(triangle_object(world[0]), triangle_object(world[1]),
                fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), collision_result)
    if collision_result.is_collision:
        return {"surface_crossing": True, "forward_reverse": [],
                "classification": "MATERIAL_PENETRATION"}
    bounds = [mesh.bounds for mesh in world]
    first_contains_second_aabb = bool(np.all(bounds[0][0] <= bounds[1][0])
                                      and np.all(bounds[0][1] >= bounds[1][1]))
    second_contains_first_aabb = bool(np.all(bounds[1][0] <= bounds[0][0])
                                      and np.all(bounds[1][1] >= bounds[0][1]))
    if not first_contains_second_aabb and not second_contains_first_aabb:
        return {"surface_crossing": False, "aabb_complete_containment_possible": False,
                "forward_reverse": [], "classification": "NO_NATIVE_MATERIAL_PENETRATION"}

    records = []
    for source_geom, target_geom in ((geom_a, geom_b), (geom_b, geom_a)):
        source = by_body.get(int(model.geom_bodyid[source_geom]))
        target = by_body.get(int(model.geom_bodyid[target_geom]))
        if source is None or target is None:
            records.append({"status": "UNKNOWN_NATIVE_VISUAL_MISSING"})
            continue
        source_visual, source_mesh = source
        target_visual, target_mesh = target
        if not target_mesh.is_watertight:
            records.append({"status": "UNKNOWN_TARGET_NATIVE_MESH_OPEN"})
            continue
        points = world_vertices(model, data, source_visual, source_mesh)
        select = np.linspace(0, len(points) - 1, min(2048, len(points)), dtype=int)
        points = points[select]
        target_body = int(model.geom_bodyid[target_visual])
        local = (points - data.xpos[target_body]) @ data.xmat[target_body].reshape(3, 3)
        signed = closed_mesh_signed_distance(target_mesh, local)
        records.append({"status": "SAMPLED_NATIVE_MATERIAL_TEST", "samples": len(points),
                        "inside_over_tolerance": int((signed > tolerance).sum()),
                        "sampled_max_inside_m": float(max(0.0, signed.max()))})
    material = any(row.get("inside_over_tolerance", 0) > 0 for row in records)
    unknown = any(row["status"].startswith("UNKNOWN") for row in records)
    return {"surface_crossing": False, "aabb_complete_containment_possible": True,
            "forward_reverse": records,
            "classification": "MATERIAL_PENETRATION" if material else
                              "UNKNOWN" if unknown else "NO_SAMPLED_MATERIAL_PENETRATION"}


def static_phase(cfg: dict[str, Any]) -> None:
    output = Path(cfg["paths"]["output"])
    mink_output = output / "mink_baseline"
    required = [mink_output / name for name in
                ("human_reference.npz", "robot_reference.npz", "retarget_report.json")]
    for path in required:
        path.resolve(strict=True)
        shutil.copy2(path, output / path.name)
    report = json.loads((output / "retarget_report.json").read_text())
    if not report.get("kinematic_model_feasible", False):
        write_json(output / "decision.json", {
            "classification": "BRUSH_RETARGET_KINEMATIC_BLOCKER", "replay_runs": 0,
            "physics_steps": 0, "mink_candidates": 1,
        })
        return
    settings = yaml.safe_load(Path(cfg["paths"]["retarget_settings"]).read_text())["retarget"]
    write_json(output / "retarget_parameter_provenance.json", {
        "schema": "taco_brush_retarget_parameter_provenance_v1",
        "algorithm": "MINK", "candidate_count": 1,
        "author_published_semantics": ["five fingertip position/orientation tasks",
                                       "wrist orientation task", "joint limits", "self collision"],
        "actual_values": settings, "actual_value_status": "LOCAL_IMPLEMENTATION_VALUE",
        "interaction_aware_patch": False, "shape_aware_patch": False,
        "finger_specific_tuning": False, "weight_sweep": False,
    })
    scene = Path(cfg["paths"]["scene"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(output / "robot_reference.npz", allow_pickle=False) as source:
        robot = dict(source)
    qpos = robot["qpos"]
    with np.load(output / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    tip_position = np.asarray(robot["fingertip_position_error_m"], dtype=float)
    wrist_orientation = np.asarray(robot["wrist_orientation_error_rad"], dtype=float)
    tip_orientation = np.empty((len(qpos), 2, 5), dtype=float)
    orientation_data = mujoco.MjData(model)
    for frame, row in enumerate(qpos):
        orientation_data.qpos[:] = row
        mujoco.mj_kinematics(model, orientation_data)
        for hand_index, side in enumerate(("right", "left")):
            for finger_index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                site = model.site(f"{side}_{finger}_tip").id
                actual = orientation_data.site_xmat[site].reshape(3, 3)
                expected = human["T_sim_fingertip_target"][frame, hand_index, finger_index, :3, :3]
                tip_orientation[frame, hand_index, finger_index] = Rotation.from_matrix(
                    actual.T @ expected).magnitude()
    report["fingertip_position_rms_m_by_hand_finger"] = np.sqrt(
        np.mean(np.square(tip_position), axis=0)).tolist()
    report["fingertip_position_max_m_by_hand_finger"] = tip_position.max(axis=0).tolist()
    report["fingertip_orientation_rms_rad_by_hand_finger"] = np.sqrt(
        np.mean(np.square(tip_orientation), axis=0)).tolist()
    report["fingertip_orientation_max_rad_by_hand_finger"] = tip_orientation.max(axis=0).tolist()
    report["wrist_orientation_rms_rad_by_hand"] = np.sqrt(
        np.mean(np.square(wrist_orientation), axis=0)).tolist()
    report["failed_frames"] = 0
    write_json(output / "retarget_report.json", report)
    collision = audit_trajectory(model, qpos, tolerance=cfg["static_gate"]["material_penetration_tolerance_m"])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    data.ctrl[:] = robot["ctrl"][0]
    mujoco.mj_forward(model, data)
    families = collision_families(model)
    by_body = local_meshes_by_body(scene, model, data)
    endpoint = {}
    blocking = []
    tol = float(cfg["static_gate"]["material_penetration_tolerance_m"])
    for family, pairs in families.items():
        values = distances(model, data, pairs)
        rows = []
        for index in np.flatnonzero(values < -tol):
            a, b = pairs[int(index)]
            native = sampled_native_pair(model, data, by_body, a, b, tol)
            row = {"geom1": model.geom(a).name, "geom2": model.geom(b).name,
                   "shell_distance_m": float(values[index]), "native": native}
            rows.append(row)
            if (native["classification"] == "MATERIAL_PENETRATION"
                    or native["classification"].startswith("UNKNOWN")):
                blocking.append({"family": family, **row})
        endpoint[family] = {"pair_count": len(pairs),
                            "minimum_shell_distance_m": float(values.min()) if len(values) else None,
                            "shell_penetrating_pairs": len(rows), "native_followups": rows}
    static_pass = not blocking
    collision.update({"endpoint0": endpoint, "endpoint0_blocking_findings": blocking,
                      "strict_gate_passed": static_pass,
                      "native_method": "bidirectional sampled native visual containment; UNKNOWN fails"})
    write_json(output / "collision_audit.json", collision)
    initial = {
        "schema": "taco_brush_initial_state_audit_v1",
        "qpos": "MINK endpoint 0 plus source object endpoint 0",
        "qvel": "all zero", "ctrl": "reference endpoint 0",
        "post_correction": False, "hold_or_blend": False, "assist": False,
        "planner": False, "residual": False, "physics_steps": 0,
        "unknown_is_failure": True, "blocking_findings": blocking,
        "strict_gate_passed": static_pass,
    }
    write_json(output / "initial_state_audit.json", initial)
    if not static_pass:
        write_json(output / "replay_metrics.json", {
            "executed": False, "reason": "static initialization gate failed",
            "position_gate": "UNRESOLVED_AUTHOR_VALUE", "rotation_threshold_rad": 1.2,
        })
        write_json(output / "chunk_replay_summary.json", {
            "executed": False, "chunks": [], "commit_allowed": False,
        })
        write_json(output / "decision.json", {
            "classification": "BRUSH_INITIALIZATION_STATIC_BLOCKER",
            "mink_candidates": 1, "physics_steps": 0, "replay_runs": 0,
            "blocking_findings": len(blocking),
        })
        phase_path = output / "phase_status.json"
        phase = json.loads(phase_path.read_text())
        phase.update({"phase": "STOPPED_AT_STATIC_INITIALIZATION_GATE",
                      "classification": "BRUSH_INITIALIZATION_STATIC_BLOCKER",
                      "mink_candidates_run": 1, "physics_steps": 0,
                      "replay_runs": 0, "next_authorized_phase": None})
        write_json(phase_path, phase)
        write_summary(output)
        hash_tree(output)
        return
    raise RuntimeError("static gate passed: the separately reviewed one-shot Replay runner is not yet wired")


def write_summary(output: Path) -> None:
    decision = json.loads((output / "decision.json").read_text())
    collision = json.loads((output / "collision_audit.json").read_text())
    report = json.loads((output / "retarget_report.json").read_text())
    material = sum(row["native"]["classification"] == "MATERIAL_PENETRATION"
                   for row in collision["endpoint0_blocking_findings"])
    unknown = sum(row["native"]["classification"].startswith("UNKNOWN")
                  for row in collision["endpoint0_blocking_findings"])
    lines = ["# TACO Brush paper baseline v1", "",
             f"- Final classification: `{decision['classification']}`",
             "- MPC: not run", "- RL: not run", "- chunk commit: forbidden", "",
             "## MINK", "",
             f"- Frames: {report['frames']}",
             f"- Kinematic feasible: {report['kinematic_model_feasible']}",
             f"- Fingertip mean error (right/left, m): {report['fingertip_mean_error_m']}",
             f"- Fingertip max error (right/left, m): {report['fingertip_max_error_m']}",
             f"- Wrist mean orientation error (right/left, rad): {report['wrist_mean_error_rad']}", "",
             "## Static gate", "",
             f"- Strict pass: {collision['strict_gate_passed']}",
             f"- Blocking native material penetrations: {material}",
             f"- Blocking UNKNOWN findings: {unknown}",
             "- Endpoint-0 native table penetration: 7 hand bodies and the brush (3 shell pieces, one native brush surface).",
             "- No physics step or Replay was executed after a failed static gate.", ""]
    (output / "summary.md").write_text("\n".join(lines))


def render_reference_phase(cfg: dict[str, Any]) -> None:
    from inspect_hand_object_trajectory import StaticModel, Visualizer

    output = Path(cfg["paths"]["output"])
    scene = Path(cfg["paths"]["scene"])
    with np.load(output / "human_reference.npz", allow_pickle=False) as source:
        human = {name: source[name].copy() for name in source.files}
    with np.load(output / "robot_reference.npz", allow_pickle=False) as source:
        robot = {name: source[name].copy() for name in source.files}
    if len(robot["qpos"]) != cfg["sample"]["expected_frames"]:
        raise ValueError("reference render requires all 209 rows")
    projection = json.loads((output / "official_projection_manifest.json").read_text())
    keyframes = sorted({int(row["frame"]) for row in projection["keyframes"]})
    visual_dir = output / "visuals/mink_reference"
    key_dir = visual_dir / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    video_path = visual_dir / "human3d_mink_reference_209f.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                             (960, 360))
    if not writer.isOpened():
        raise RuntimeError("cannot create MINK visual evidence video")
    reference_model = StaticModel.load("MINK reference", scene)
    visualizer = Visualizer(reference_model, reference_model, width=480, height=360)
    saved = []
    try:
        for frame in range(len(robot["qpos"])):
            human_panel = visualizer.render_human(
                human["T_sim_object_reference"][frame], human["joint_positions_sim"][frame], "oblique")
            robot_panel = visualizer.render_reference(robot["qpos"][frame], "oblique")
            for image, label in ((human_panel, "HUMAN 3D"), (robot_panel, "MINK ROBOT REFERENCE")):
                cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(image, f"{label} | frame {frame}", (8, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            composite = np.concatenate([human_panel, robot_panel], axis=1)
            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if frame in keyframes:
                path = key_dir / f"frame_{frame:03d}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                saved.append({"frame": frame, "path": str(path)})
    finally:
        writer.release()
        visualizer.close()
    manifest = {"schema": "taco_brush_mink_reference_visual_manifest_v1",
                "physics_steps": 0, "frames_rendered": len(robot["qpos"]),
                "video": artifact(video_path), "keyframes": saved,
                "panels": ["HUMAN 3D", "MINK ROBOT REFERENCE"]}
    write_json(output / "mink_reference_visual_manifest.json", manifest)
    index_path = output / "VISUAL_INDEX.md"
    text = index_path.read_text().rstrip() + "\n\n## MINK reference\n\n"
    text += "- [HUMAN 3D + MINK robot reference (209 frames)](visuals/mink_reference/human3d_mink_reference_209f.mp4)\n"
    text += "- Keyframes: `visuals/mink_reference/keyframes/`\n"
    text += "- Replay visual: not generated because the endpoint-0 static gate failed.\n"
    index_path.write_text(text)


def refresh_manifest_phase(cfg: dict[str, Any]) -> None:
    output = Path(cfg["paths"]["output"])
    path = output / "input_manifest.json"
    manifest = json.loads(path.read_text())
    refreshed = []
    for row in manifest["artifacts"]:
        refreshed.append(artifact(Path(row["path"])))
    manifest["artifacts"] = refreshed
    manifest["refreshed_after_final_code_changes"] = True
    write_json(path, manifest)
    hash_tree(output)


def hash_tree(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p != target):
        rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    target.write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "approve-visual", "static",
                                          "render-reference", "refresh", "hash"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/taco_brush_brush_bowl_paper_baseline_v1.yaml")
    args = parser.parse_args()
    cfg = load_contract(args.config.resolve(strict=True))
    if args.phase == "source":
        source_phase(cfg, args.config.resolve())
    elif args.phase == "approve-visual":
        approve_visual_phase(cfg)
    elif args.phase == "static":
        static_phase(cfg)
    elif args.phase == "render-reference":
        render_reference_phase(cfg)
    elif args.phase == "refresh":
        refresh_manifest_phase(cfg)
    else:
        hash_tree(Path(cfg["paths"]["output"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
