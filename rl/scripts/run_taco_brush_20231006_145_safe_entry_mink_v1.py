#!/usr/bin/env python3
"""Build a safe MANO entry and compare two fresh environment-constrained MINK runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_pair_measurement,
    triangle_object,
)
from build_taco_bimanual_scene import build as build_scene  # noqa: E402
from run_taco_brush_issue14_mink_candidate_v1 import (  # noqa: E402
    frame0_static_audit,
    selected_frame_native_hand_object_audit,
)
from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    MANO21_SOURCE,
    MANO_TIP_VERTICES,
    _load_model_data,
    reconstruct_taco_mano,
)
from egoengine_repro.retarget.schema import validate_human_reference  # noqa: E402
from egoengine_repro.retarget.taco_bimanual import retarget  # noqa: E402
from egoengine_repro.retarget.taco_bimanual_settings import (  # noqa: E402
    UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    NativeSupportSettings,
    load_taco_bimanual_settings,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


SCHEMA = "taco_brush_20231006_145_safe_entry_mink_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_20231006_145_safe_entry_mink_v1.yaml"
SIDES = ("right", "left")
TIPS = (4, 8, 12, 16, 20)
DIPS = (3, 7, 11, 15, 19)
MANO_DISTALS = (15, 3, 6, 12, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fcl_entry(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {"geom": name, "vertices_sim_m": vertices, "faces": faces,
            "mesh": mesh, "fcl": triangle_object(vertices, faces)}


def pair_distance(first: trimesh.Trimesh, second: trimesh.Trimesh, tolerance: float,
                  first_name: str, second_name: str) -> dict[str, Any]:
    value = native_pair_measurement(
        fcl_entry(first_name, first), fcl_entry(second_name, second), tolerance,
    )
    return {"distance_m": float(value["distance_m"]),
            "material_penetration": bool(float(value["distance_m"]) < -tolerance)}


def geometric_frame(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    z = np.asarray(direction, dtype=np.float64).copy()
    z /= np.linalg.norm(z)
    x = np.asarray(normal, dtype=np.float64).copy()
    x -= np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def fingertip_frames(pose_path: Path, shape_path: Path, model_path: Path,
                     side: str, expected_joints: np.ndarray) -> np.ndarray:
    import torch
    import smplx
    from smplx.lbs import batch_rigid_transform
    from smplx.utils import Struct

    with pose_path.open("rb") as stream:
        rows = pickle.load(stream)
    keys = sorted(rows, key=int)
    poses = np.stack([np.asarray(rows[key]["hand_pose"], dtype=np.float64) for key in keys])
    layer = smplx.MANO(
        "unused", data_struct=Struct(**_load_model_data(model_path)),
        is_rhand=side == "right", use_pca=False, flat_hand_mean=True,
        create_transl=False,
    )
    with torch.no_grad():
        neutral = layer(global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45),
                        betas=torch.zeros(1, 10))
    points = torch.cat([neutral.joints, neutral.vertices[:, MANO_TIP_VERTICES[side]]], 1)
    points = points[0, MANO21_SOURCE].numpy().astype(np.float64)
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    if side == "left":
        normal = -normal
    calibration = np.stack([
        geometric_frame(normal, points[tip] - points[dip]) for tip, dip in zip(TIPS, DIPS)
    ])
    rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 16, 3, 3)
    with torch.no_grad():
        _, transforms = batch_rigid_transform(
            torch.from_numpy(rotations),
            torch.zeros(len(poses), 16, 3, dtype=torch.float64), layer.parents,
        )
    frames = transforms.numpy()[:, MANO_DISTALS, :3, :3] @ calibration
    observed = expected_joints[:, TIPS] - expected_joints[:, DIPS]
    observed /= np.linalg.norm(observed, axis=-1, keepdims=True)
    if float(np.max(np.abs(np.linalg.norm(frames[..., 2], axis=-1) - 1.0))) > 1e-8:
        raise ValueError("invalid fingertip frames")
    return frames


def world_mesh(mesh: trimesh.Trimesh, pose: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(pose)
    return result


def prepare_scene(cfg: dict[str, Any], output: Path) -> Path:
    assets = output / "scene_assets"
    assets.mkdir()
    (assets / "robots").symlink_to(
        Path(cfg["paths"]["source_robot_assets"]).resolve(strict=True),
        target_is_directory=True,
    )
    data_root = Path(cfg["paths"]["data_root"])
    collision_root = Path(cfg["paths"]["collision_source"])
    objects = {}
    for side, key in (("right", "tool"), ("left", "target")):
        spec = cfg["sample"][key]
        destination = assets / "objects" / spec["id"]
        destination.mkdir(parents=True)
        source = (
            data_root / "object_models_released" / f"{spec['id']}_cm.obj"
        ).resolve(strict=True)
        (destination / "visual.obj").symlink_to(source)
        parts = sorted((collision_root / spec["id"]).glob("*.obj"), key=lambda path: int(path.stem))
        if len(parts) != 32:
            raise ValueError(f"expected 32 frozen collision parts for {spec['id']}")
        (destination / "convex_m").symlink_to(
            (collision_root / spec["id"]).resolve(strict=True),
            target_is_directory=True,
        )
        objects[side] = {
            "object_id": spec["id"], "visual_scale": 0.01,
            "collision_subdir": "convex_m", "source": source,
        }
    scene = output / "generated_scene.xml"
    build_scene(
        Path(cfg["paths"]["source_template"]), scene, assets, objects=objects,
        model_name="taco_brush_20231006_145_initialization_feasibility",
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("generated scene contract mismatch")
    return scene


def build_source_geometry(cfg: dict[str, Any]) -> tuple[dict, dict, np.ndarray]:
    data_root = Path(cfg["paths"]["data_root"])
    sequence = cfg["sample"]["sequence"]
    hands = {}
    for side in SIDES:
        vertices, joints, faces, _, keys = reconstruct_taco_mano(
            data_root / "Hand_Poses" / sequence / f"{side}_hand.pkl",
            data_root / "Hand_Poses" / sequence / f"{side}_hand_shape.pkl",
            data_root / "mano_v1_2/models" / f"MANO_{side.upper()}.pkl",
            side=side,
        )
        if len(keys) != int(cfg["sample"]["frames"]):
            raise ValueError("MANO timeline mismatch")
        hands[side] = {"vertices": np.asarray(vertices, dtype=np.float64),
                       "joints": np.asarray(joints, dtype=np.float64),
                       "faces": np.asarray(faces, dtype=np.int32)}
    objects = {}
    poses = []
    for key in ("tool", "target"):
        spec = cfg["sample"][key]
        mesh = trimesh.load_mesh(
            data_root / "object_models_released" / f"{spec['id']}_cm.obj", process=False,
        )
        mesh.apply_scale(0.01)
        pose = np.asarray(np.load(
            data_root / "Object_Poses" / sequence / f"{spec['role']}_{spec['id']}.npy",
            allow_pickle=False,
        ), dtype=np.float64)
        objects[spec["name"]] = {"mesh": mesh, "poses": pose}
        poses.append(pose)
    return hands, objects, np.stack(poses, axis=1)


def safe_entry_geometry(cfg: dict[str, Any], hands: dict, objects: dict) -> tuple[dict, dict]:
    join = int(cfg["entry_contract"]["join_source_frame"])
    retreat = float(cfg["entry_contract"]["initial_vertical_retreat_m"])
    tolerance = float(cfg["entry_contract"]["material_penetration_tolerance_m"])
    table_z = float(cfg["entry_contract"]["issue14_world_table_z_m"])
    modified = {}
    for side, hand in hands.items():
        vertices = hand["vertices"].copy()
        joints = hand["joints"].copy()
        for frame in range(join):
            u = frame / join
            smooth = u * u * (3.0 - 2.0 * u)
            offset = np.array([0.0, 0.0, retreat * (1.0 - smooth)])
            vertices[frame] = hand["vertices"][join] + offset
            joints[frame] = hand["joints"][join] + offset
        modified[side] = {**hand, "vertices": vertices, "joints": joints}

    rows = []
    all_legal = True
    for frame in range(join + 1):
        hmesh = {
            side: trimesh.Trimesh(vertices=hand["vertices"][frame], faces=hand["faces"], process=False)
            for side, hand in modified.items()
        }
        omesh = {
            name: world_mesh(entry["mesh"], entry["poses"][frame])
            for name, entry in objects.items()
        }
        table = {side: float(np.min(mesh.vertices[:, 2] - table_z)) for side, mesh in hmesh.items()}
        pair_values = {}
        for side in SIDES:
            for name in ("brush", "bowl"):
                pair_values[f"{side}_{name}"] = pair_distance(
                    hmesh[side], omesh[name], tolerance, side, name,
                )
        hands_pair = pair_distance(hmesh["right"], hmesh["left"], tolerance, "right", "left")
        legal = bool(
            min(table.values()) >= -tolerance
            and not hands_pair["material_penetration"]
            and not any(value["material_penetration"] for value in pair_values.values())
        )
        all_legal &= legal
        rows.append({
            "frame": frame, "right_table_m": table["right"], "left_table_m": table["left"],
            "right_brush_m": pair_values["right_brush"]["distance_m"],
            "right_bowl_m": pair_values["right_bowl"]["distance_m"],
            "left_brush_m": pair_values["left_brush"]["distance_m"],
            "left_bowl_m": pair_values["left_bowl"]["distance_m"],
            "hand_hand_m": hands_pair["distance_m"], "legal": legal,
        })
    exact_join = all(
        np.array_equal(modified[side][key][join], hands[side][key][join])
        for side in SIDES for key in ("vertices", "joints")
    )
    return modified, {"all_entry_frames_legal": all_legal, "exact_source_join": exact_join, "rows": rows}


def reference_arrays(cfg: dict[str, Any], model: mujoco.MjModel, hands: dict,
                     object_poses: np.ndarray, transform: np.ndarray,
                     source_indices: np.ndarray, fingertip_rotation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(source_indices)
    joints_world = np.stack([hands[side]["joints"][source_indices] for side in SIDES], axis=1)
    joints = joints_world @ transform[:3, :3].T + transform[:3, 3]
    objects = np.einsum("ij,tojk->toik", transform, object_poses[source_indices])
    wrists = np.tile(np.eye(4), (count, 2, 1, 1))
    tips = np.tile(np.eye(4), (count, 2, 5, 1, 1))
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    for hand_index, side in enumerate(SIDES):
        palm = model.site(f"{side}_palm").id
        root = model.body(f"{side}_hand_link").id
        middle = model.body(f"{side}_hand_mid_link1").id
        robot_frame = geometric_frame(
            zero.site_xmat[palm].reshape(3, 3)[:, 0], zero.xpos[middle] - zero.xpos[root],
        )
        wrist_offset = robot_frame.T @ zero.xmat[root].reshape(3, 3)
        local_tip = []
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            site = model.site(f"{side}_{finger}_tip").id
            distal = zero.site_xpos[site] - zero.xpos[model.site_bodyid[site]]
            local_tip.append(
                zero.site_xmat[site].reshape(3, 3).T @ geometric_frame(robot_frame[:, 0], distal)
            )
        for local_frame, source_frame in enumerate(source_indices):
            points = joints[local_frame, hand_index]
            normal = np.cross(points[5] - points[0], points[17] - points[0])
            if side == "left":
                normal = -normal
            human_frame = geometric_frame(normal, points[9] - points[0])
            wrists[local_frame, hand_index, :3, :3] = human_frame @ wrist_offset
            wrists[local_frame, hand_index, :3, 3] = points[0]
            for finger_index, tip_index in enumerate(TIPS):
                rotation = fingertip_rotation[side][source_frame]
                tips[local_frame, hand_index, finger_index, :3, :3] = (
                    transform[:3, :3] @ rotation[finger_index] @ local_tip[finger_index].T
                )
                tips[local_frame, hand_index, finger_index, :3, 3] = points[tip_index]
    arrays = {
        "frame_indices": source_indices.astype(np.int64),
        "timestamps_s": source_indices.astype(np.float64) / float(cfg["sample"]["fps"]),
        "hand_order": np.asarray(SIDES), "object_roles": np.asarray(["tool", "target"]),
        "T_sim_world": transform, "joint_positions_sim": joints,
        "T_sim_wrist_target": wrists, "T_sim_fingertip_target": tips,
        "T_sim_object_reference": objects,
        "valid_hand": np.ones((count, 2), dtype=bool),
        "valid_fingertip_orientation": np.ones((count, 2, 5), dtype=bool),
        "confidence_hand": np.ones((count, 2), dtype=np.float64),
        "confidence_fingertip": np.ones((count, 2, 5), dtype=np.float64),
        "fingertip_orientation_source": np.asarray("MANO_rotational_FK_fixed_neutral_calibration_v1"),
    }
    validate_human_reference(arrays)
    return arrays


def native_table_minimum(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    data = mujoco.MjData(model)
    floor_z = float(model.geom("floor").pos[2])
    minimum = np.inf
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if not (name.startswith(("right_", "left_")) and name.endswith("_visual")):
                continue
            if "object" in name:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            local = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
            world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            minimum = min(minimum, float(np.min(world[:, 2] - floor_z)))
    return float(minimum)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["schema"] != SCHEMA:
        raise ValueError("schema mismatch")
    if cfg["authorization"]["run_fresh_mink_candidates"] != 2:
        raise ValueError("exactly two fresh kinematic candidates are required")
    for key in ("physics", "replay", "mpc", "reinforcement_learning",
                "modify_original_data", "modify_active_support_contract",
                "reuse_old_brush_hand_object_zero_margin_proxy_fix", "promotion"):
        if cfg["authorization"][key]:
            raise ValueError(f"forbidden authorization: {key}")
    source_gate = json.loads(Path(cfg["paths"]["source_geometry_audit"]).read_text(encoding="utf-8"))
    join = int(cfg["entry_contract"]["join_source_frame"])
    if source_gate["direct_start"]["first_stable_legal_frame"] != join:
        raise ValueError("source geometry gate no longer supports the frozen join frame")
    output = Path(cfg["paths"]["output"])
    output.mkdir(parents=True, exist_ok=False)
    hands, objects, object_poses = build_source_geometry(cfg)
    safe_hands, entry_audit = safe_entry_geometry(cfg, hands, objects)
    write_json(output / "safe_entry_native_geometry.json", entry_audit)
    if not entry_audit["all_entry_frames_legal"] or not entry_audit["exact_source_join"]:
        result = {
            "schema": SCHEMA, "classification": "SAFE_ENTRY_KINEMATIC_GEOMETRY_FAIL",
            "mink_candidates_run": 0, "physics": False, "replay": False,
        }
        write_json(output / "results.json", result)
        (output / "summary.md").write_text(
            "# Safe-entry / MINK feasibility\n\nSafe-entry native geometry failed closed; MINK did not run.\n",
            encoding="utf-8",
        )
        return 0

    scene = prepare_scene(cfg, output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    table_world = float(cfg["entry_contract"]["issue14_world_table_z_m"])
    table_sim = float(cfg["entry_contract"]["simulator_table_z_m"])
    center = object_poses[0, :, :3, 3].mean(axis=0)
    transform = np.eye(4)
    transform[:3, 3] = np.array([0.6 - center[0], -center[1], table_sim - table_world])
    fingertip_rotation = {}
    data_root = Path(cfg["paths"]["data_root"])
    sequence = cfg["sample"]["sequence"]
    for side in SIDES:
        fingertip_rotation[side] = fingertip_frames(
            data_root / "Hand_Poses" / sequence / f"{side}_hand.pkl",
            data_root / "Hand_Poses" / sequence / f"{side}_hand_shape.pkl",
            data_root / "mano_v1_2/models" / f"MANO_{side.upper()}.pkl",
            side, hands[side]["joints"],
        )
        # The replacement prefix holds source-frame-11 articulation/orientation.
        fingertip_rotation[side][:join] = fingertip_rotation[side][join]

    references = {
        "direct_crop": reference_arrays(
            cfg, model, hands, object_poses, transform,
            np.arange(join, int(cfg["sample"]["frames"])), fingertip_rotation,
        ),
        "safe_entry": reference_arrays(
            cfg, model, safe_hands, object_poses, transform,
            np.arange(int(cfg["sample"]["frames"])), fingertip_rotation,
        ),
    }
    settings, consumption = load_taco_bimanual_settings(Path(cfg["mink_contract"]["settings"]))
    if consumption["unknown_key_count"] or consumption["unused_key_count"]:
        raise ValueError("MINK settings are not exactly consumed")
    support_cfg = cfg["mink_contract"]["native_support"]
    settings = replace(
        settings,
        final_feasibility_algorithm=UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
        native_support=NativeSupportSettings(
            normal=(0.0, 0.0, 1.0), offset_m=table_sim, frame="simulator",
            minimum_clearance_m=float(support_cfg["minimum_clearance_m"]),
            activation_distance_m=support_cfg["activation_distance_m"],
            gain=float(support_cfg["gain"]),
            depenetration_step_m=float(support_cfg["depenetration_step_m"]),
            validation_tolerance_m=float(support_cfg["validation_tolerance_m"]),
        ),
    )
    candidates = {}
    for name, arrays in references.items():
        candidate = output / name
        candidate.mkdir()
        human_path = candidate / "human_reference.npz"
        np.savez_compressed(human_path, **arrays)
        failure = None
        try:
            report = retarget(scene, human_path, settings, candidate)
        except RuntimeError as error:
            failure = str(error)
            report = None
        if failure is not None:
            candidates[name] = {"status": "MINK_FAIL", "failure": failure,
                                "human_reference": artifact(human_path)}
            continue
        with np.load(candidate / "robot_reference.npz", allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            tip_error = np.asarray(archive["fingertip_position_error_m"], dtype=np.float64)
        floor_min = native_table_minimum(model, qpos)
        initial_local_frames = list(range(join + 1)) if name == "safe_entry" else [0]
        plane = Plane(normal=np.asarray([0.0, 0.0, 1.0]), offset=table_sim, frame="simulator")
        initialization_rows = [
            {"local_frame": frame, "source_frame": int(arrays["frame_indices"][frame]),
             **frame0_static_audit(
                 model, qpos[frame], plane,
                 float(support_cfg["validation_tolerance_m"]),
             )}
            for frame in initial_local_frames
        ]
        initialization_native_pass = all(row["pass"] for row in initialization_rows)
        write_json(candidate / "initialization_native_geometry.json", {
            "local_frames": initial_local_frames,
            "source_frames": [int(arrays["frame_indices"][frame]) for frame in initial_local_frames],
            "all_pass": initialization_native_pass,
            "rows": initialization_rows,
        })
        requested_source_frames = [18, 30, 60, 90, 121]
        source_to_local = {int(value): index for index, value in enumerate(arrays["frame_indices"])}
        interaction_local_frames = [
            source_to_local[value] for value in requested_source_frames if value in source_to_local
        ]
        interaction_native = selected_frame_native_hand_object_audit(
            model, qpos, interaction_local_frames,
            float(support_cfg["validation_tolerance_m"]),
        )
        for row in interaction_native["rows"]:
            row["source_frame"] = int(arrays["frame_indices"][row["frame"]])
        write_json(candidate / "interaction_native_hand_object_audit.json", interaction_native)
        candidate_pass = bool(
            floor_min >= -float(support_cfg["validation_tolerance_m"])
            and initialization_native_pass
        )
        candidates[name] = {
            "status": "MINK_PASS" if candidate_pass else "MINK_NATIVE_INITIALIZATION_FAIL",
            "frames": len(qpos), "source_frame_start": int(arrays["frame_indices"][0]),
            "native_hand_table_minimum_m": floor_min,
            "initialization_native_geometry_pass": initialization_native_pass,
            "initialization_native_geometry": artifact(candidate / "initialization_native_geometry.json"),
            "interaction_native_hand_object_observations": {
                "source_frames": requested_source_frames,
                "penetrating_pair_observation_count": interaction_native["penetrating_pair_observation_count"],
                "unknown_pair_observation_count": interaction_native["unknown_pair_observation_count"],
                "artifact": artifact(candidate / "interaction_native_hand_object_audit.json"),
                "interpretation": "POST_HOC_INTERACTION_GEOMETRY;_NOT_A_ZERO_MARGIN_MINK_CONSTRAINT",
            },
            "fingertip_error_mean_m": float(np.mean(tip_error)),
            "fingertip_error_max_m": float(np.max(tip_error)),
            "retarget_report": report,
            "human_reference": artifact(human_path),
            "robot_reference": artifact(candidate / "robot_reference.npz"),
        }

    passed = [name for name, value in candidates.items() if value["status"] == "MINK_PASS"]
    classification = (
        "XHAND_KINEMATIC_INITIALIZATION_PASS"
        if passed else "XHAND_KINEMATIC_INITIALIZATION_FAIL"
    )
    result = {
        "schema": SCHEMA, "classification": classification,
        "source_direct_start": {"frame": join, "deleted_frames": list(range(join)),
                                "deleted_duration_s": join / float(cfg["sample"]["fps"])},
        "safe_entry": {
            "replaced_frames": list(range(join)),
            "join_frame": join,
            "exact_source_join": entry_audit["exact_source_join"],
            "all_entry_frames_native_geometry_legal": entry_audit["all_entry_frames_legal"],
            "initial_vertical_retreat_m": float(cfg["entry_contract"]["initial_vertical_retreat_m"]),
            "objects_modified": False,
        },
        "T_sim_world": transform.tolist(), "generated_scene": artifact(scene),
        "mink_candidates": candidates,
        "kinematically_eligible_candidates": passed,
        "replay_authorization": "NOT_DECIDED_BY_THIS_KINEMATIC_STAGE",
        "forbidden_runtime_counts": {"physics": 0, "replay": 0, "mpc": 0, "rl": 0},
        "provenance": {
            "source_geometry_gate": artifact(Path(cfg["paths"]["source_geometry_audit"])),
            "config": artifact(config_path),
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            ).strip(),
        },
    }
    write_json(output / "results.json", result)
    lines = [
        "# 20231006_145 safe-entry and XHand initialization feasibility", "",
        f"- Classification: `{classification}`",
        f"- Safe replacement: source frames `0..{join - 1}`; exact join at source frame `{join}`",
        f"- Direct crop deletes `{join}` frames / `{join / float(cfg['sample']['fps']):.3f} s`",
        "- Object reference modified: `NO`",
        "- Active support modified: `NO`",
        "- Hand-object zero-margin proxy repair reused: `NO`",
        "- Physics / Replay / MPC / RL: `NOT RUN`", "",
        "| candidate | status | frames | source start | native init | hand-table min (mm) | tip mean/max (mm) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for name, value in candidates.items():
        if value["status"] == "MINK_FAIL":
            lines.append(f"| {name} | {value['status']}: {value['failure']} | - | - | - | - | - |")
        else:
            lines.append(
                f"| {name} | {value['status']} | {value['frames']} | {value['source_frame_start']} | "
                f"{'PASS' if value['initialization_native_geometry_pass'] else 'FAIL'} | "
                f"{value['native_hand_table_minimum_m'] * 1000:.6f} | "
                f"{value['fingertip_error_mean_m'] * 1000:.3f}/{value['fingertip_error_max_m'] * 1000:.3f} |"
            )
    lines += ["", "Only candidates passing this kinematic gate may enter a separately recorded controlled Replay. Passing MINK is not physical pickup evidence.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classification": classification, "candidates": candidates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
