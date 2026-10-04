#!/usr/bin/env python3
"""Bounded whole-left-hand interaction-aware retargeting pilot for Pour 0..20.

The default ``static`` phase performs no environment construction and no
physics stepping.  It audits the old chain, generates exactly one sequential
candidate, and evaluates the frozen static contract.  Rendering is also
kinematic-only.  Physics is fail-closed behind a separately finalized strict
static/visual pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data_all/zzx/3.2RL")
CONFIG = ROOT / "configs/taco_pour_left_interaction_retarget_v1.yaml"
OUTPUT = ASSET_ROOT / "runs/taco_pour_left_interaction_retarget_v1"
BASELINE = "8c09f1e0d5ef718a78c76b283950cf2a033c567e"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LEFT = slice(18, 36)
RIGHT = slice(0, 18)
OBJECT = slice(36, 50)

sys.path.insert(0, str(ROOT / "scripts"))

from audit_taco_initialization import visual_meshes
from build_taco_pour_initialization_candidates import native_geometry_gate
from egoengine_repro.retarget.collision_audit import collision_families, distances
from egoengine_repro.retarget.interaction_aware import (
    FrameTarget,
    bone_error,
    deterministic_surface_samples,
    human_bone_directions,
    human_semantic_points,
    interaction_error,
    interaction_topology,
    pose_local_points,
    robot_bone_directions,
    robot_semantic_points,
    semantic_keypoint_map,
    solve_frame_two_stage,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from video_to_spider.rl.physics_contract import compile_mujoco_model


SOURCE_PINS = {
    "egoengine": {
        "kind": "project_supplied_paper",
        "paper": "EgoEngine From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations",
    },
    "toporetarget": {
        "kind": "paper_and_project_page_only_no_verified_method_repository",
        "arxiv": "2606.16272v2",
        "project": "https://tsinghua-mars-lab.github.io/toporetarget-web/",
    },
    "regrind": {
        "repository": "https://github.com/yunhaif/regrind",
        "commit": "38347a9e30184620df04e19c63c7c72378cae103",
    },
    "key_objectives": {
        "repository": "https://github.com/Mingrui-Yu/retargeting",
        "commit": "3846d3fa207165bb0d498145aac8b885a28ea923",
    },
    "maniptrans": {
        "repository": "https://github.com/ManipTrans/ManipTrans",
        "commit": "a3d08cfe3c3a5868a7f057533bcaf759c5af4705",
    },
    "spider": {
        "repository": "https://github.com/facebookresearch/spider",
        "commit": "44717007de41cbef7565dff7ff9f4453557a2d3d",
    },
    "unidex": {
        "repository": "https://github.com/unidex-ai/UniDex",
        "commit": "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d",
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean_descendant() -> None:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("interaction-retarget experiment requires a clean implementation worktree")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=ROOT, check=True)


def load_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONFIG.read_text())
    if contract.get("schema") != "taco_pour_left_interaction_retarget_v1":
        raise ValueError("unexpected interaction-retarget contract")
    if contract["solver"]["candidate_count"] != 1:
        raise ValueError("contract must authorize exactly one candidate")
    return contract


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def input_paths(contract: dict[str, Any]) -> dict[str, Path]:
    result = {name: resolve_path(value) for name, value in contract["inputs"].items()}
    result.update(
        interaction_contract=CONFIG,
        helper_source=ROOT / "src/egoengine_repro/retarget/interaction_aware.py",
        runner_source=Path(__file__),
    )
    return result


def compile_model(paths: dict[str, Path]) -> mujoco.MjModel:
    config = yaml.safe_load(paths["simulator_config"].read_text())
    return compile_mujoco_model(paths["scene"], config.get("sdf_octree_depths", {}))


def _pose7_rotation_translation(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return rotation, pose[:3]


def _object_points_from_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, local: np.ndarray
) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    body = int(model.body("left_object").id)
    return pose_local_points(local, data.xmat[body], data.xpos[body])


def build_targets(
    model: mujoco.MjModel,
    human: dict[str, np.ndarray],
    robot: dict[str, np.ndarray],
    local_samples: np.ndarray,
    kappa: float,
) -> list[FrameTarget]:
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    role = int(np.flatnonzero(human["object_roles"] == "target")[0])
    data = mujoco.MjData(model)
    targets: list[FrameTarget] = []
    for endpoint in range(21):
        human_points = human_semantic_points(human["joint_positions_sim"][endpoint, hand])
        human_bones = human_bone_directions(
            human["joint_positions_sim"][endpoint, hand],
            human["T_sim_wrist_target"][endpoint, hand],
        )
        source_transform = human["T_sim_object_reference"][endpoint, role]
        source_object = pose_local_points(
            local_samples, source_transform[:3, :3], source_transform[:3, 3]
        )
        robot_object = _object_points_from_qpos(model, data, robot["qpos"][endpoint], local_samples)
        source_vertices = np.concatenate([human_points, source_object])
        _, neighbors = interaction_topology(source_vertices)
        targets.append(
            FrameTarget(human_points, human_bones, robot_object, source_vertices, neighbors)
        )
    return targets


def _left_pairs(model: mujoco.MjModel) -> dict[str, list[tuple[int, int]]]:
    groups = collision_families(model)

    def is_left(geom: int) -> bool:
        return model.geom(geom).name.startswith("collision_hand_left_")

    result = {
        "self": [pair for pair in groups["self_explicit"] if is_left(pair[0]) or is_left(pair[1])],
        "tool": [pair for pair in groups["hand_tool"] if is_left(pair[0])],
        "target": [pair for pair in groups["hand_target"] if is_left(pair[0])],
        "floor": [pair for pair in groups["hand_floor"] if is_left(pair[0])],
        "named_guards": [],
    }
    for pair_id, pair in enumerate(zip(model.pair_geom1, model.pair_geom2)):
        name = model.pair(pair_id).name or ""
        if "index_root" in name or name.startswith("semantic_self_left_palm_thumb_surface_"):
            result["named_guards"].append((int(pair[0]), int(pair[1])))
    return result


def feasibility_callback(
    model: mujoco.MjModel,
    pair_groups: dict[str, list[tuple[int, int]]],
    minimum: float,
):
    data = mujoco.MjData(model)
    pairs = sum(pair_groups.values(), [])
    cache: dict[bytes, np.ndarray] = {}

    def margin(qpos: np.ndarray) -> np.ndarray:
        key = np.asarray(qpos, dtype=np.float64).tobytes()
        if key not in cache:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cache[key] = distances(model, data, pairs) - float(minimum)
        return cache[key].copy()

    return margin


def velocity_bounds(
    model: mujoco.MjModel,
    previous: np.ndarray,
    endpoint: int,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    addresses = np.arange(18, 36)
    joints = np.asarray([model.joint(i).id for i in [
        "L_forearm_tx_link_joint", "L_forearm_ty_link_joint", "L_forearm_tz_link_joint",
        "L_forearm_roll_link_joint", "L_forearm_pitch_link_joint", "L_forearm_yaw_link_joint",
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
        "left_hand_mid_joint1", "left_hand_mid_joint2", "left_hand_ring_joint1",
        "left_hand_ring_joint2", "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    ]])
    lower = model.jnt_range[joints, 0].astype(np.float64)
    upper = model.jnt_range[joints, 1].astype(np.float64)
    if endpoint > 0:
        limits = contract["solver"]["velocity_limits"]
        speed = np.r_[np.full(3, limits["base_translation"]), np.full(3, limits["base_rotation"]), np.full(12, limits["finger"])]
        step = speed * float(contract["solver"]["ref_dt_s"])
        lower = np.maximum(lower, previous[addresses] - step)
        upper = np.minimum(upper, previous[addresses] + step)
    if np.any(lower > upper):
        raise ValueError("empty joint/velocity bounds")
    return lower, upper


def _tip_and_orientation_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    human: dict[str, np.ndarray],
) -> dict[str, Any]:
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    data = mujoco.MjData(model)
    tip_position = np.empty((21, 5))
    tip_orientation = np.empty((21, 5))
    wrist_orientation = np.empty(21)
    for endpoint in range(21):
        data.qpos[:] = qpos[endpoint]
        mujoco.mj_kinematics(model, data)
        for index, finger in enumerate(FINGERS):
            site = int(model.site(f"left_{finger}_tip").id)
            target = human["T_sim_fingertip_target"][endpoint, hand, index]
            tip_position[endpoint, index] = np.linalg.norm(data.site_xpos[site] - target[:3, 3])
            tip_orientation[endpoint, index] = Rotation.from_matrix(
                target[:3, :3].T @ data.site_xmat[site].reshape(3, 3)
            ).magnitude()
        palm = int(model.body("left_hand_link").id)
        target_wrist = human["T_sim_wrist_target"][endpoint, hand]
        wrist_orientation[endpoint] = Rotation.from_matrix(
            target_wrist[:3, :3].T @ data.xmat[palm].reshape(3, 3)
        ).magnitude()
    return {
        "five_tip_position_error_m": tip_position,
        "five_tip_orientation_error_rad": tip_orientation,
        "wrist_orientation_error_rad": wrist_orientation,
        "five_tip_rms_m": float(np.sqrt(np.mean(tip_position * tip_position))),
        "per_finger": {
            finger: {
                "rms_m": float(np.sqrt(np.mean(tip_position[:, index] ** 2))),
                "max_m": float(tip_position[:, index].max()),
                "orientation_rms_rad": float(np.sqrt(np.mean(tip_orientation[:, index] ** 2))),
            }
            for index, finger in enumerate(FINGERS)
        },
    }


def trajectory_objectives(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    targets: list[FrameTarget],
    human: dict[str, np.ndarray],
    local_samples: np.ndarray,
    kappa: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    total = np.empty(21)
    im = np.empty(21)
    per_finger = {finger: np.empty(21) for finger in FINGERS}
    angles: list[dict[str, dict[str, float]]] = []
    palm_object = []
    for endpoint, target in enumerate(targets):
        directions = robot_bone_directions(model, data, qpos[endpoint])
        total[endpoint], contribution, angle = bone_error(directions, target.human_bones)
        for finger in FINGERS:
            per_finger[finger][endpoint] = contribution[finger]
        angles.append(angle)
        hand_points = robot_semantic_points(model, data, qpos[endpoint])
        object_points = _object_points_from_qpos(model, data, qpos[endpoint], local_samples)
        vertices = np.concatenate([hand_points, object_points])
        im[endpoint] = interaction_error(vertices, target.source_vertices, target.neighbors, kappa)
        body = int(model.body("left_object").id)
        object_frame = (hand_points - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        palm_object.append({
            "palm_xyz_m": object_frame[0].tolist(),
            "ring_xyz_m": object_frame[10:13].tolist(),
            "pinky_xyz_m": object_frame[13:16].tolist(),
        })
    tips = _tip_and_orientation_metrics(model, qpos, human)
    return {
        "E_bone": total,
        "E_bone_per_finger": per_finger,
        "direction_angle_error_deg": angles,
        "E_IM": im,
        "E_bone_rms_0_20": float(np.sqrt(np.mean(total))),
        "E_IM_rms_0_20": float(np.sqrt(np.mean(im))),
        "windows": {
            f"0_{end}": {
                "E_bone_rms": float(np.sqrt(np.mean(total[: end + 1]))),
                "E_IM_rms": float(np.sqrt(np.mean(im[: end + 1]))),
                "E_IM_sum": float(im[: end + 1].sum()),
            }
            for end in (5, 10, 20)
        },
        "per_finger_rms": {
            finger: float(np.sqrt(np.mean(value))) for finger, value in per_finger.items()
        },
        "tips": tips,
        "object_frame_diagnostics": palm_object,
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "E_bone_by_endpoint": metrics["E_bone"],
        "E_IM_by_endpoint": metrics["E_IM"],
        "E_bone_rms_0_20": metrics["E_bone_rms_0_20"],
        "E_IM_rms_0_20": metrics["E_IM_rms_0_20"],
        "windows": metrics["windows"],
        "per_finger_bone_rms": metrics["per_finger_rms"],
        "direction_angle_error_deg": metrics["direction_angle_error_deg"],
        "five_tip_rms_m": metrics["tips"]["five_tip_rms_m"],
        "fingertips": metrics["tips"]["per_finger"],
        "wrist_orientation_error_rms_rad": float(
            np.sqrt(np.mean(np.asarray(metrics["tips"]["wrist_orientation_error_rad"]) ** 2))
        ),
        "object_frame_diagnostics": metrics["object_frame_diagnostics"],
    }


def _geometry_fast_summary(
    model: mujoco.MjModel, qpos: np.ndarray, threshold: float
) -> dict[str, Any]:
    families = collision_families(model)
    data = mujoco.MjData(model)
    selected = ("self_explicit", "hand_tool", "hand_target", "hand_floor", "tool_target", "tool_floor", "target_floor")
    values = {name: np.empty(21) for name in selected}
    for endpoint in range(21):
        data.qpos[:] = qpos[endpoint]
        mujoco.mj_forward(model, data)
        for name in selected:
            values[name][endpoint] = distances(model, data, families[name]).min()
    return {
        "threshold_m": float(threshold),
        "minimum_distance_m": {name: float(value.min()) for name, value in values.items()},
        "penetrating_endpoints": {name: np.flatnonzero(value < -threshold).tolist() for name, value in values.items()},
    }


def kinematic_margin_report(
    model: mujoco.MjModel, qpos: np.ndarray, contract: dict[str, Any]
) -> dict[str, Any]:
    joint_names = [
        "L_forearm_tx_link_joint", "L_forearm_ty_link_joint", "L_forearm_tz_link_joint",
        "L_forearm_roll_link_joint", "L_forearm_pitch_link_joint", "L_forearm_yaw_link_joint",
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1",
        "left_hand_thumb_rota_joint2", "left_hand_index_bend_joint",
        "left_hand_index_joint1", "left_hand_index_joint2", "left_hand_mid_joint1",
        "left_hand_mid_joint2", "left_hand_ring_joint1", "left_hand_ring_joint2",
        "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    ]
    joint_ids = np.asarray([model.joint(name).id for name in joint_names], dtype=np.int64)
    values = np.asarray(qpos[:, LEFT], dtype=np.float64)
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    margins = np.minimum(values - lower, upper - values)
    limits = contract["solver"]["velocity_limits"]
    speed_limit = np.r_[
        np.full(3, limits["base_translation"]),
        np.full(3, limits["base_rotation"]),
        np.full(12, limits["finger"]),
    ]
    velocity = np.diff(values, axis=0) / float(contract["solver"]["ref_dt_s"])
    remaining = speed_limit[None] - np.abs(velocity)
    return {
        "per_joint": {
            name: {
                "position_min_margin": float(margins[:, index].min()),
                "velocity_limit": float(speed_limit[index]),
                "velocity_min_remaining": float(remaining[:, index].min()),
                "velocity_max_abs": float(np.abs(velocity[:, index]).max()),
            }
            for index, name in enumerate(joint_names)
        },
        "position_min_margin": float(margins.min()),
        "velocity_min_remaining": float(remaining.min()),
        "position_pass": bool(np.all(margins >= -1e-12)),
        "velocity_pass": bool(np.all(remaining >= -1e-9)),
    }


def static_legality(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    meshes: dict[int, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(contract["penetration"]["material_reporting_threshold_m"])
    minimum = float(contract["solver"]["declared_pair_min_distance_m"])
    families = collision_families(model)
    pair_groups = _left_pairs(model)
    data = mujoco.MjData(model)
    limited = np.flatnonzero(model.jnt_limited)
    addresses = model.jnt_qposadr[limited]
    ctrl_limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    ctrl_ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    rows = []
    all_pass = True
    for endpoint in range(21):
        variants = {}
        for precision, state in (
            ("float64", qpos[endpoint]),
            ("runtime_float32", qpos[endpoint].astype(np.float32).astype(np.float64)),
        ):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            pair_distance = {
                name: distances(model, data, pairs) for name, pairs in pair_groups.items()
            }
            joint_margin = np.minimum(
                state[addresses] - model.jnt_range[limited, 0],
                model.jnt_range[limited, 1] - state[addresses],
            )
            native = native_geometry_gate(model, state, meshes, threshold)
            checks = {
                "finite": bool(np.isfinite(state).all()),
                "joint_limits": bool(joint_margin.min() >= -1e-12),
                "actuator_ctrlrange": bool(np.all(
                    (~ctrl_limited)
                    | ((state[: model.nu] >= ctrl_ranges[:, 0] - 1e-12)
                       & (state[: model.nu] <= ctrl_ranges[:, 1] + 1e-12))
                )),
                "declared_left_pairs": bool(all(value.min() >= minimum for value in pair_distance.values())),
                "native_material_legality": bool(native["passed"]),
            }
            variants[precision] = {
                "checks": checks,
                "passed": bool(all(checks.values())),
                "joint_limit_min_margin": float(joint_margin.min()),
                "minimum_distances_m": {name: float(value.min()) for name, value in pair_distance.items()},
                "native_failures": native["failures"],
                "native_table_clearance_min_m": float(min(native["table_clearance_m"].values())),
            }
            all_pass &= variants[precision]["passed"]
        rows.append({"endpoint": endpoint, **variants})
    delta = np.diff(qpos[:, LEFT], axis=0)
    limits = contract["solver"]["velocity_limits"]
    speed = np.r_[np.full(3, limits["base_translation"]), np.full(3, limits["base_rotation"]), np.full(12, limits["finger"])]
    ratio = np.abs(delta) / (float(contract["solver"]["ref_dt_s"]) * speed)
    velocity_pass = bool((ratio <= 1.0 + 1e-9).all())
    return {
        "passed": bool(all_pass and velocity_pass),
        "float_variants": rows,
        "velocity_limits_pass": velocity_pass,
        "velocity_max_ratio": float(ratio.max()),
        "velocity_worst_interval": int(np.unravel_index(np.argmax(ratio), ratio.shape)[0]),
        "unknown_is_pass": False,
    }


def preflight_artifacts(paths: dict[str, Path]) -> list[dict[str, Any]]:
    records = [artifact(path) for path in paths.values()]
    records.extend(scene_mesh_artifacts(paths["scene"]))
    unique = {record["path"]: record for record in records}
    return [unique[key] for key in sorted(unique)]


def save_keypoint_debug(
    output: Path,
    model: mujoco.MjModel,
    human: dict[str, np.ndarray],
    old: np.ndarray,
    candidate: np.ndarray,
) -> None:
    hand = int(np.flatnonzero(human["hand_order"] == "left")[0])
    data = mujoco.MjData(model)
    labels = ["wrist", *[f"{finger}:{part}" for finger in FINGERS for part in ("prox", "second", "tip")]]
    directory = output / "keypoint_debug"
    directory.mkdir(parents=True, exist_ok=True)
    for endpoint in (0, 15, 16, 20):
        rows = [
            ("HUMAN", human_semantic_points(human["joint_positions_sim"][endpoint, hand])),
            ("OLD_MINK", robot_semantic_points(model, data, old[endpoint])),
            ("NEW", robot_semantic_points(model, data, candidate[endpoint])),
        ]
        fig = plt.figure(figsize=(15, 5))
        for index, (name, points) in enumerate(rows, 1):
            axis = fig.add_subplot(1, 3, index, projection="3d")
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=20)
            for label, point in zip(labels, points):
                axis.text(*point, label, fontsize=6)
            axis.set_title(f"{name} endpoint {endpoint}")
            axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_zlabel("z")
        fig.tight_layout()
        fig.savefig(directory / f"endpoint_{endpoint:03d}.png", dpi=160)
        plt.close(fig)


def static(output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require_clean_descendant()
    contract = load_contract()
    paths = input_paths(contract)
    preserved = preflight_artifacts(paths)
    verify_artifacts(preserved)
    output.mkdir(parents=True)
    write_json(output / "source_pins.json", {"schema": "taco_pour_left_interaction_source_pins_v1", "pins": SOURCE_PINS})
    write_json(output / "input_manifest.json", {
        "schema": "taco_pour_left_interaction_input_manifest_v1",
        "preflight_baseline_commit": BASELINE,
        "implementation_commit": head(),
        "artifacts": preserved,
        "physics_steps": 0,
        "policy_constructions": 0,
    })
    human = arrays(paths["human_reference"])
    robot = arrays(paths["robot_reference"])
    replay = arrays(paths["archived_replay_endpoints"])
    if robot["qpos"].shape[0] < 21 or replay["A_qpos"].shape != (21, 50):
        raise ValueError("0..20 reference/replay evidence unavailable")
    if not np.array_equal(robot["frame_indices"][:21], human["frame_indices"][:21]):
        raise ValueError("human/robot frame IDs differ")
    if not np.array_equal(robot["timestamps_s"][:21], human["timestamps_s"][:21]):
        raise ValueError("human/robot timestamps differ")
    model = compile_model(paths)
    mapping = semantic_keypoint_map(model)
    write_json(output / "semantic_keypoint_map.json", {
        "schema": "taco_pour_left_interaction_semantic_keypoints_v1",
        "mapping": mapping,
        "selection_frozen_before_optimization": True,
        "virtual_dip_joints": False,
    })
    meshes, mesh_paths = visual_meshes(paths["scene"], model)
    tray_geom = int(model.geom(contract["interaction_mesh"]["object_visual_geom"]).id)
    if tray_geom not in meshes:
        raise ValueError("target visual mesh is unavailable")
    tray = meshes[tray_geom]
    mesh_path_by_geom = dict(zip(meshes, mesh_paths, strict=True))
    local_samples, face_index, barycentric = deterministic_surface_samples(
        tray.vertices, tray.faces,
        int(contract["interaction_mesh"]["object_surface_samples"]),
        int(contract["interaction_mesh"]["sample_seed"]),
    )
    np.savez_compressed(
        output / "object_surface_samples.npz",
        points_tray_local_m=local_samples,
        face_index=face_index,
        barycentric=barycentric,
        mesh_sha256=np.asarray(sha256(mesh_path_by_geom[tray_geom])),
        rng_seed=np.asarray(contract["interaction_mesh"]["sample_seed"]),
    )
    targets = build_targets(
        model, human, robot, local_samples, float(contract["interaction_mesh"]["kappa"])
    )
    old_qpos = robot["qpos"][:21].astype(np.float64)
    accepted_qpos = replay["A_qpos"].astype(np.float64)
    current_old = trajectory_objectives(model, old_qpos, targets, human, local_samples, float(contract["interaction_mesh"]["kappa"]))
    current_a = trajectory_objectives(model, accepted_qpos, targets, human, local_samples, float(contract["interaction_mesh"]["kappa"]))
    current = {
        "schema": "taco_pour_left_interaction_current_reference_audit_v1",
        "human": {"E_bone": 0.0, "E_IM": 0.0, "semantic_keypoints": 16},
        "old_mink_reference": _compact_metrics(current_old),
        "accepted_a_replay": _compact_metrics(current_a),
        "old_runtime_geometry": _geometry_fast_summary(model, old_qpos, float(contract["penetration"]["material_reporting_threshold_m"])),
        "accepted_runtime_geometry": _geometry_fast_summary(model, accepted_qpos, float(contract["penetration"]["material_reporting_threshold_m"])),
        "old_joint_velocity_margins": kinematic_margin_report(model, old_qpos, contract),
        "accepted_joint_velocity_margins": kinematic_margin_report(model, accepted_qpos, contract),
        "old_float64_runtime_float32_legality": static_legality(model, old_qpos, meshes, contract),
        "accepted_float64_runtime_float32_legality": static_legality(model, accepted_qpos, meshes, contract),
    }
    write_json(output / "current_reference_audit.json", current)
    write_json(output / "human_to_old_reference.json", {
        "schema": "taco_pour_human_to_old_reference_attribution_v1",
        "endpoint0_E_bone": float(current_old["E_bone"][0]),
        "endpoint0_ring_E_bone": float(current_old["E_bone_per_finger"]["ring"][0]),
        "endpoint0_pinky_E_bone": float(current_old["E_bone_per_finger"]["pinky"][0]),
        "interpretation": "articulation mismatch already present in the old MINK reference",
    })
    write_json(output / "old_reference_to_accepted_a.json", {
        "schema": "taco_pour_old_reference_to_accepted_a_attribution_v1",
        "endpoint0_E_bone_delta": float(current_a["E_bone"][0] - current_old["E_bone"][0]),
        "endpoint0_ring_E_bone_delta": float(current_a["E_bone_per_finger"]["ring"][0] - current_old["E_bone_per_finger"]["ring"][0]),
        "endpoint0_pinky_E_bone_delta": float(current_a["E_bone_per_finger"]["pinky"][0] - current_old["E_bone_per_finger"]["pinky"][0]),
        "endpoint0_left_qpos_delta": (accepted_qpos[0, LEFT] - old_qpos[0, LEFT]).tolist(),
        "interpretation": "additional initialization-A wrist/finger change after old MINK retargeting",
    })

    pair_groups = _left_pairs(model)
    candidate = old_qpos.copy()
    solver_rows = []
    previous = old_qpos[0].copy()
    for endpoint in range(21):
        lower, upper = velocity_bounds(model, previous, endpoint, contract)
        feasibility = feasibility_callback(
            model, pair_groups, float(contract["solver"]["declared_pair_min_distance_m"])
        )
        candidate[endpoint], report = solve_frame_two_stage(
            model,
            old_qpos[endpoint],
            previous,
            targets[endpoint],
            feasibility,
            lower=lower,
            upper=upper,
            lambda_warm=float(contract["warm_start"]["lambda_bone"]),
            lambda_smooth=float(contract["warm_start"]["lambda_smooth"]),
            lambda_im=float(contract["refinement"]["lambda_interaction_mesh"]),
            lambda_bone=float(contract["refinement"]["lambda_bone"]),
            lambda_temporal=float(contract["refinement"]["lambda_temporal"]),
            lambda_base_translation=float(contract["refinement"]["lambda_base_translation_delta"]),
            lambda_base_rotation=float(contract["refinement"]["lambda_base_rotation_delta"]),
            kappa=float(contract["interaction_mesh"]["kappa"]),
            max_iterations=int(contract["solver"]["max_iterations_per_stage"]),
            ftol=float(contract["solver"]["ftol"]),
        )
        report["endpoint"] = endpoint
        solver_rows.append(report)
        previous = candidate[endpoint].copy()

    ctrl = robot["ctrl"][:21].copy()
    ctrl[:, LEFT] = candidate[:, LEFT]
    qvel = robot["qvel"][:21].copy()
    qvel[:, LEFT] = 0.0
    qvel[1:, LEFT] = np.diff(candidate[:, LEFT], axis=0) / float(contract["solver"]["ref_dt_s"])
    np.savez_compressed(
        output / "candidate_reference.npz",
        qpos=candidate,
        qvel=qvel,
        ctrl=ctrl,
        frequency=robot["frequency"],
        frame_indices=robot["frame_indices"][:21],
        timestamps_s=robot["timestamps_s"][:21],
        hand_order=robot["hand_order"],
        object_roles=robot["object_roles"],
    )
    candidate_metrics = trajectory_objectives(
        model, candidate, targets, human, local_samples, float(contract["interaction_mesh"]["kappa"])
    )
    legality = static_legality(model, candidate, meshes, contract)
    tol = float(contract["static_gate"]["strict_numeric_tolerance"])
    checks = {
        "right_hand_qpos_bitwise_unchanged": bool(np.array_equal(candidate[:, RIGHT], old_qpos[:, RIGHT])),
        "right_hand_qvel_bitwise_unchanged": bool(np.array_equal(qvel[:, RIGHT], robot["qvel"][:21, RIGHT])),
        "right_hand_ctrl_bitwise_unchanged": bool(np.array_equal(ctrl[:, RIGHT], robot["ctrl"][:21, RIGHT])),
        "object_qpos_bitwise_unchanged": bool(np.array_equal(candidate[:, OBJECT], old_qpos[:, OBJECT])),
        "object_qvel_bitwise_unchanged": bool(np.array_equal(qvel[:, 36:48], robot["qvel"][:21, 36:48])),
        "left_ctrl_equals_candidate_qpos": bool(np.array_equal(ctrl[:, LEFT], candidate[:, LEFT])),
        "frame_ids_unchanged": bool(np.array_equal(robot["frame_indices"][:21], human["frame_indices"][:21])),
        "timestamps_unchanged": bool(np.array_equal(robot["timestamps_s"][:21], human["timestamps_s"][:21])),
        "all_solver_stages_success": bool(all(row["warm"]["success"] and row["refine"]["success"] for row in solver_rows)),
        "legality_float64_and_runtime_float32": bool(legality["passed"]),
        "E_IM_rms_strictly_improved": bool(candidate_metrics["E_IM_rms_0_20"] < current_old["E_IM_rms_0_20"] - tol),
        "E_bone_rms_strictly_improved": bool(candidate_metrics["E_bone_rms_0_20"] < current_old["E_bone_rms_0_20"] - tol),
        "ring_E_bone_strictly_improved": bool(candidate_metrics["per_finger_rms"]["ring"] < current_old["per_finger_rms"]["ring"] - tol),
        "pinky_E_bone_strictly_improved": bool(candidate_metrics["per_finger_rms"]["pinky"] < current_old["per_finger_rms"]["pinky"] - tol),
        "endpoint0_ring_articulation_improved": bool(candidate_metrics["E_bone_per_finger"]["ring"][0] < current_old["E_bone_per_finger"]["ring"][0] - tol),
        "endpoint0_pinky_articulation_improved": bool(candidate_metrics["E_bone_per_finger"]["pinky"][0] < current_old["E_bone_per_finger"]["pinky"][0] - tol),
    }
    five_tip_nonworse = bool(
        candidate_metrics["tips"]["five_tip_rms_m"]
        <= current_old["tips"]["five_tip_rms_m"]
        + float(contract["static_gate"]["fingertip_nonworsening_tolerance_m"])
    )
    machine_pass = bool(all(checks.values()))
    if not machine_pass:
        classification = "STATIC_REFERENCE_GATE_FAILED_NO_PHYSICS"
    elif not five_tip_nonworse:
        classification = "REFERENCE_QUALITY_TRADEOFF_NO_PHYSICS"
    else:
        classification = "PENDING_VISUAL_REVIEW_NO_PHYSICS"
    gate = {
        "schema": "taco_pour_left_interaction_static_gate_v1",
        "classification": classification,
        "machine_checks": checks,
        "machine_pass": machine_pass,
        "five_tip_nonworsening": five_tip_nonworse,
        "visual_review_pass": None,
        "physics_authorized": False,
        "old_metrics": _compact_metrics(current_old),
        "candidate_metrics": _compact_metrics(candidate_metrics),
        "legality": legality,
        "joint_velocity_margins": kinematic_margin_report(model, candidate, contract),
        "solver": solver_rows,
    }
    write_json(output / "candidate_static_gate.json", gate)
    write_json(output / "decision.json", {
        "classification": classification,
        "physics_authorized": False,
        "authorization": contract["authorization"],
    })
    write_json(output / "cost_accounting.json", {
        "candidate_count": 1,
        "solver_frames": 21,
        "solver_stage_calls": 42,
        "objective_evaluations": int(sum(row["calls"]["warm_objective"] + row["calls"]["refine_objective"] for row in solver_rows)),
        "constraint_evaluations": int(sum(row["calls"]["constraint"] for row in solver_rows)),
        "physics_steps": 0,
        "planner_calls": 0,
        "policy_constructions": 0,
        "render_updates": 0,
    })
    write_json(output / "visual_review.json", {
        "status": "PENDING",
        "manifest_hash": None,
        "review_scope": list(range(21)),
        "views": contract["visuals"]["views"],
        "physics_authorized": False,
    })
    save_keypoint_debug(output, model, human, old_qpos, candidate)
    summary = f"""# TACO Pour left interaction-aware retargeting v1\n\n- Exactly one 0..20 candidate was generated; no parameter sweep or multi-start was used.\n- Machine static classification: `{classification}`.\n- Old/new E_IM RMS: `{current_old['E_IM_rms_0_20']:.9g}` -> `{candidate_metrics['E_IM_rms_0_20']:.9g}`.\n- Old/new E_bone RMS: `{current_old['E_bone_rms_0_20']:.9g}` -> `{candidate_metrics['E_bone_rms_0_20']:.9g}`.\n- Old/new five-tip RMS: `{current_old['tips']['five_tip_rms_m']:.9g}` -> `{candidate_metrics['tips']['five_tip_rms_m']:.9g} m`.\n- Physics steps: `0`; visual review remains pending.\n- No reset/reference promotion, 0..40 extension, planner, RL, or chunk commit is authorized.\n"""
    (output / "summary.md").write_text(summary)
    return gate


def _load_visual_module():
    path = ROOT / "scripts/inspect_hand_object_trajectory.py"
    spec = importlib.util.spec_from_file_location("interaction_retarget_visual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render(output: Path) -> dict[str, Any]:
    if not (output / "candidate_static_gate.json").is_file():
        raise RuntimeError("static phase must complete before rendering")
    contract = load_contract()
    paths = input_paths(contract)
    human = arrays(paths["human_reference"])
    robot = arrays(paths["robot_reference"])
    candidate = arrays(output / "candidate_reference.npz")
    vismod = _load_visual_module()
    rgb_path = ASSET_ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4"
    rgb, rgb_info = vismod.read_rgb_frames(rgb_path, 21)
    reference_model = vismod.StaticModel.load("old", paths["scene"])
    actual_model = vismod.StaticModel.load("new", paths["scene"])
    vis = vismod.Visualizer(reference_model, actual_model)
    directory = output / "visuals"
    for child in ("oblique", "top", "closeups"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        for endpoint in range(21):
            for view in ("oblique", "top"):
                panels = [
                    vismod.annotate_panel(
                        vismod.letterbox(rgb[endpoint], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT),
                        ["REAL RGB (independent camera)", f"source frame {endpoint}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_human(
                            human["T_sim_object_reference"][endpoint],
                            human["joint_positions_sim"][endpoint], view,
                        ),
                        ["HUMAN 3-D SOURCE", f"endpoint {endpoint} | {view}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_reference(robot["qpos"][endpoint], view),
                        ["OLD MINK REFERENCE", f"endpoint {endpoint} | {view}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_actual(candidate["qpos"][endpoint], view),
                        ["NEW INTERACTION REFERENCE", f"endpoint {endpoint} | {view}"],
                    ),
                ]
                image = np.concatenate(panels, axis=1)
                relative = f"visuals/{view}/endpoint_{endpoint:03d}.png"
                vismod.save_rgb(output / relative, image)
                manifest.append({"endpoint": endpoint, "view": view, "path": relative, "sha256": sha256(output / relative)})
                if endpoint in contract["visuals"]["closeup_endpoints"]:
                    close = cv2.resize(image[:, vismod.PANEL_WIDTH:], (1440, 540), interpolation=cv2.INTER_CUBIC)
                    close_relative = f"visuals/closeups/endpoint_{endpoint:03d}_{view}.png"
                    vismod.save_rgb(output / close_relative, close)
                    manifest.append({"endpoint": endpoint, "view": f"{view}_closeup", "path": close_relative, "sha256": sha256(output / close_relative)})
    finally:
        vis.close()
    manifest_path = output / "visual_manifest.json"
    write_json(manifest_path, {
        "schema": "taco_pour_left_interaction_visual_manifest_v1",
        "rgb": rgb_info,
        "camera_registration": "RGB time-aligned but not pixel-registered; 3-D panels use fixed simulation cameras",
        "images": manifest,
        "render_updates": int(vis.render_updates),
    })
    manifest_hash = sha256(manifest_path)
    review = json.loads((output / "visual_review.json").read_text())
    review.update(manifest_hash=manifest_hash, status="PENDING_BOUND_TO_IMAGE_MANIFEST")
    write_json(output / "visual_review.json", review)
    accounting = json.loads((output / "cost_accounting.json").read_text())
    accounting["render_updates"] = int(vis.render_updates)
    write_json(output / "cost_accounting.json", accounting)
    index = ["# Visual evidence index", "", f"Manifest SHA-256: `{manifest_hash}`", ""]
    for endpoint in range(21):
        index.append(
            f"- endpoint {endpoint}: [oblique](visuals/oblique/endpoint_{endpoint:03d}.png) · "
            f"[top](visuals/top/endpoint_{endpoint:03d}.png)"
        )
    (output / "VISUAL_INDEX.md").write_text("\n".join(index) + "\n")
    return {"manifest_sha256": manifest_hash, "images": len(manifest), "render_updates": vis.render_updates}


def finalize_static(output: Path, visual_pass: bool, note: str) -> dict[str, Any]:
    gate = json.loads((output / "candidate_static_gate.json").read_text())
    review = json.loads((output / "visual_review.json").read_text())
    if review.get("status") != "PENDING_BOUND_TO_IMAGE_MANIFEST":
        raise RuntimeError("visual evidence is not pending against a bound manifest")
    review.update(
        status="COMPLETE_PASS" if visual_pass else "COMPLETE_FAIL",
        passed=bool(visual_pass), reviewer_note=note,
    )
    write_json(output / "visual_review.json", review)
    gate["visual_review_pass"] = bool(visual_pass)
    if gate["machine_pass"] and gate["five_tip_nonworsening"] and visual_pass:
        classification = "STRICT_REFERENCE_QUALITY_PASS"
        physics_authorized = True
    elif gate["machine_pass"] and not gate["five_tip_nonworsening"]:
        classification = "REFERENCE_QUALITY_TRADEOFF_NO_PHYSICS"
        physics_authorized = False
    else:
        classification = "STATIC_REFERENCE_GATE_FAILED_NO_PHYSICS"
        physics_authorized = False
    gate.update(classification=classification, physics_authorized=physics_authorized)
    write_json(output / "candidate_static_gate.json", gate)
    contract = load_contract()
    write_json(output / "decision.json", {
        "classification": classification,
        "physics_authorized": physics_authorized,
        "authorization": contract["authorization"],
    })
    lines = (output / "summary.md").read_text().rstrip().splitlines()
    lines.extend(["", f"- Completed visual review: `{'PASS' if visual_pass else 'FAIL'}`.", f"- Final static classification: `{classification}`."])
    if not physics_authorized:
        lines.append("- Static contract forbids physics; the run stops with zero physics steps.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "server_artifacts.sha256")
    (output / "server_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    return {"classification": classification, "physics_authorized": physics_authorized}


def _load_trace_module():
    path = ROOT / "scripts/trace_replay_0_20.py"
    spec = importlib.util.spec_from_file_location("interaction_retarget_trace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_candidate_world(trace, candidate_path: Path, initial: dict[str, np.ndarray]):
    import torch

    runtime = trace.input_paths()
    config = trace.load_ego_config(runtime["simulator_config"], device="cpu")
    reference = trace.load_reference(candidate_path, device="cpu")
    objective = trace.load_runtime_objective(
        runtime["protocol"], runtime["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = trace.load_runtime_observation(
        runtime["protocol"], runtime["observation_profile"], require_run_ready=False,
    )
    residual, _ = trace.load_residual_action_profile(runtime["action_profile"])
    world = trace.MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
        env_config=trace.MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=False,
            max_episode_length=20,
            tracked_object_indices=(0,),
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
            residual=residual,
            object_assistance=None,
        ),
        seed=0,
    )
    tensors = [
        torch.as_tensor(initial[name][None], dtype=torch.float32)
        for name in ("qpos", "qvel", "ctrl")
    ]
    world._write_state(*tensors, np.asarray([True]))
    world._last_ctrl = tensors[2].clone()
    # Match the audited complete-snapshot convention for an unused MJWP
    # collision-sensor buffer allocated through wp.empty.
    for data in (world.env.data_wp, world.env.data_wp_prev):
        target = data.contact.geomcollisionid
        sentinel = torch.full(tuple(target.shape), -1, dtype=torch.int32)
        trace.wp.copy(target, trace.wp.from_torch(sentinel, dtype=target.dtype))
    trace.wp.synchronize()
    world._check_capacity()
    return world


def _object_quality_rows(
    qpos: np.ndarray, qvel: np.ndarray, reference: np.ndarray
) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {
        name: [] for name in (
            "tool_position_m", "tool_rotation_rad", "target_position_m",
            "target_rotation_rad", "target_frame_pair_translation_m",
            "target_frame_pair_rotation_rad", "world_pair_translation_m",
            "tool_linear_speed_m_s", "tool_angular_speed_rad_s",
            "target_linear_speed_m_s", "target_angular_speed_rad_s",
        )
    }
    for state, velocity, target in zip(qpos, qvel, reference, strict=True):
        tool_error = state[36:39] - target[36:39]
        target_error = state[43:46] - target[43:46]
        rows["tool_position_m"].append(float(np.linalg.norm(tool_error)))
        rows["target_position_m"].append(float(np.linalg.norm(target_error)))
        rows["tool_rotation_rad"].append(float(
            (Rotation.from_quat(target[[40, 41, 42, 39]]).inv()
             * Rotation.from_quat(state[[40, 41, 42, 39]])).magnitude()
        ))
        rows["target_rotation_rad"].append(float(
            (Rotation.from_quat(target[[47, 48, 49, 46]]).inv()
             * Rotation.from_quat(state[[47, 48, 49, 46]])).magnitude()
        ))
        world_pair = (state[36:39] - state[43:46]) - (target[36:39] - target[43:46])
        rows["world_pair_translation_m"].append(float(np.linalg.norm(world_pair)))
        tool = np.eye(4); target_object = np.eye(4)
        tool_ref = np.eye(4); target_ref = np.eye(4)
        tool[:3, 3], target_object[:3, 3] = state[36:39], state[43:46]
        tool_ref[:3, 3], target_ref[:3, 3] = target[36:39], target[43:46]
        tool[:3, :3] = Rotation.from_quat(state[[40, 41, 42, 39]]).as_matrix()
        target_object[:3, :3] = Rotation.from_quat(state[[47, 48, 49, 46]]).as_matrix()
        tool_ref[:3, :3] = Rotation.from_quat(target[[40, 41, 42, 39]]).as_matrix()
        target_ref[:3, :3] = Rotation.from_quat(target[[47, 48, 49, 46]]).as_matrix()
        relative = np.linalg.inv(target_object) @ tool
        relative_ref = np.linalg.inv(target_ref) @ tool_ref
        rows["target_frame_pair_translation_m"].append(
            float(np.linalg.norm(relative[:3, 3] - relative_ref[:3, 3]))
        )
        rows["target_frame_pair_rotation_rad"].append(float(
            Rotation.from_matrix(relative_ref[:3, :3].T @ relative[:3, :3]).magnitude()
        ))
        rows["tool_linear_speed_m_s"].append(float(np.linalg.norm(velocity[36:39])))
        rows["tool_angular_speed_rad_s"].append(float(np.linalg.norm(velocity[39:42])))
        rows["target_linear_speed_m_s"].append(float(np.linalg.norm(velocity[42:45])))
        rows["target_angular_speed_rad_s"].append(float(np.linalg.norm(velocity[45:48])))
    return {name: np.asarray(value, dtype=np.float64) for name, value in rows.items()}


def _quality_summary(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "rms_0_20": float(np.sqrt(np.mean(value * value))),
            "endpoint20": float(value[-1]),
        }
        for name, value in values.items()
    }


def physics(output: Path) -> dict[str, Any]:
    gate = json.loads((output / "candidate_static_gate.json").read_text())
    if gate.get("classification") != "STRICT_REFERENCE_QUALITY_PASS" or not gate.get("physics_authorized"):
        raise RuntimeError("STRICT_REFERENCE_QUALITY_PASS is required before any physics")
    require_clean_descendant()
    trace = _load_trace_module()
    contract = load_contract()
    paths = input_paths(contract)
    candidate_path = output / "candidate_reference.npz"
    candidate = arrays(candidate_path)
    initial = {
        "qpos": candidate["qpos"][0].astype(np.float32),
        "qvel": np.zeros(48, dtype=np.float32),
        "ctrl": candidate["ctrl"][0].astype(np.float32),
    }
    if not np.array_equal(initial["qpos"][36:], candidate["qpos"][0, 36:].astype(np.float32)):
        raise RuntimeError("candidate endpoint-0 object state changed")
    if not np.array_equal(initial["ctrl"], initial["qpos"][:36]):
        raise RuntimeError("candidate endpoint-0 control is not its hand qpos")

    ledger = trace.BudgetLedger(limit_physics_steps=650, limit_control_intervals=40)
    ledger.require_capacity(physics_steps=402, control_intervals=40)
    world = _make_candidate_world(trace, candidate_path, initial)
    ledger.charge(physics_steps=1, control_intervals=0)
    s0 = world.get_env_state()
    trace.torch_gzip_write(output / "endpoint0_complete_snapshot.pt.gz", s0)
    actions = np.zeros((20, 36), dtype=np.float32)
    expected = candidate["ctrl"][1:21].astype(np.float32)
    selected = trace._run_startup_condition(
        world, s0, "NEW_INTERACTION_REPLAY", actions, expected, ledger,
    )
    trace._save_condition(output, "NEW_INTERACTION_REPLAY", *selected)

    cold_world = _make_candidate_world(trace, candidate_path, initial)
    ledger.charge(physics_steps=1, control_intervals=0)
    cold = trace._run_startup_condition(
        cold_world, s0, "NEW_INTERACTION_COLD", actions, expected, ledger,
    )
    trace._save_condition(output, "NEW_INTERACTION_COLD", *cold)
    selected_rows, selected_contacts, selected_efc = trace._observer_arrays(selected[2])
    cold_rows, cold_contacts, cold_efc = trace._observer_arrays(cold[2])
    parity = {
        "endpoints": trace._compare_array_maps(selected[0], cold[0]),
        "substeps": trace._compare_array_maps(selected_rows, cold_rows),
        "contacts": trace._compare_array_maps(selected_contacts, cold_contacts),
        "efc_force": trace._compare_array_maps(selected_efc, cold_efc),
        "terminal_snapshot": trace.compare_snapshots(
            selected[1][max(selected[1])], cold[1][max(cold[1])]
        ),
    }
    parity["bitwise_equal"] = bool(
        parity["endpoints"]["all_equal"]
        and parity["substeps"]["all_equal"]
        and parity["contacts"]["all_equal"]
        and parity["efc_force"]["all_equal"]
        and parity["terminal_snapshot"]["all_common_equal"]
        and not parity["terminal_snapshot"]["only_current"]
        and not parity["terminal_snapshot"]["only_historical"]
    )
    write_json(output / "cold_replay_parity.json", parity)

    replay = arrays(paths["archived_replay_endpoints"])
    reference = arrays(paths["robot_reference"])["qpos"][:21].astype(np.float64)
    completed = len(selected[0]["qpos"]) == 21 and not selected[3]["terminated"]
    quality: dict[str, Any] = {
        "selected_completed_0_20": completed,
        "cold_bitwise_equal": parity["bitwise_equal"],
        "comparison_tolerance": 1e-9,
    }
    physical_quality_pass = False
    if completed:
        new_values = _object_quality_rows(
            selected[0]["qpos"], selected[0]["qvel"], reference
        )
        old_values = _object_quality_rows(replay["A_qpos"], replay["A_qvel"], reference)
        new_summary = _quality_summary(new_values)
        old_summary = _quality_summary(old_values)
        comparisons = {}
        for name in old_summary:
            comparisons[name] = {
                metric: {
                    "old": old_summary[name][metric],
                    "new": new_summary[name][metric],
                    "delta": new_summary[name][metric] - old_summary[name][metric],
                    "nonworsening": new_summary[name][metric] <= old_summary[name][metric] + 1e-9,
                    "clear_improvement": new_summary[name][metric] < old_summary[name][metric] - 1e-9,
                }
                for metric in ("rms_0_20", "endpoint20")
            }
        no_tradeoff = all(
            row[metric]["nonworsening"]
            for row in comparisons.values() for metric in ("rms_0_20", "endpoint20")
        )
        clear = any(
            row[metric]["clear_improvement"]
            for row in comparisons.values() for metric in ("rms_0_20", "endpoint20")
        )
        samples = arrays(output / "object_surface_samples.npz")["points_tray_local_m"]
        human = arrays(paths["human_reference"])
        model = compile_model(paths)
        targets = build_targets(
            model, human, arrays(paths["robot_reference"]), samples,
            float(contract["interaction_mesh"]["kappa"]),
        )
        executed_metrics = trajectory_objectives(
            model, selected[0]["qpos"].astype(np.float64), targets, human,
            samples, float(contract["interaction_mesh"]["kappa"]),
        )
        meshes, _ = visual_meshes(paths["scene"], model)
        threshold = float(contract["penetration"]["material_reporting_threshold_m"])
        native_rows = [
            native_geometry_gate(model, state.astype(np.float64), meshes, threshold)
            for state in selected[0]["qpos"]
        ]
        material_legal = bool(all(row["passed"] for row in native_rows))
        old_metrics = gate["old_metrics"]
        shape_preserved = bool(
            np.sqrt(np.mean(executed_metrics["E_bone_per_finger"]["ring"][:6]))
            < old_metrics["per_finger_bone_rms"]["ring"]
            and np.sqrt(np.mean(executed_metrics["E_bone_per_finger"]["pinky"][:6]))
            < old_metrics["per_finger_bone_rms"]["pinky"]
        )
        quality.update(
            old=old_summary,
            new=new_summary,
            comparisons=comparisons,
            no_object_or_pair_tradeoff=no_tradeoff,
            at_least_one_clear_improvement=clear,
            early_ring_pinky_shape_preserved=shape_preserved,
            executed_shape={
                "E_bone_rms_0_20": executed_metrics["E_bone_rms_0_20"],
                "ring_E_bone_rms_0_5": float(np.sqrt(np.mean(
                    executed_metrics["E_bone_per_finger"]["ring"][:6]
                ))),
                "pinky_E_bone_rms_0_5": float(np.sqrt(np.mean(
                    executed_metrics["E_bone_per_finger"]["pinky"][:6]
                ))),
            },
            executed_native_material_legality_pass=material_legal,
            executed_native_material_failures=[
                {"endpoint": endpoint, "failures": row["failures"]}
                for endpoint, row in enumerate(native_rows) if not row["passed"]
            ],
        )
        physical_quality_pass = bool(
            parity["bitwise_equal"] and no_tradeoff and clear and shape_preserved
            and material_legal
        )
    quality["passed"] = physical_quality_pass
    write_json(output / "physical_quality_gate.json", quality)
    if physical_quality_pass:
        classification = "INTERACTION_AWARE_LEFT_REFERENCE_0_20_VALIDATED"
    elif completed and parity["bitwise_equal"]:
        classification = "KINEMATIC_REFERENCE_PASS_PHYSICS_QUALITY_TRADEOFF"
    else:
        classification = "PHYSICS_EXECUTION_FAILED"
    accounting = json.loads((output / "cost_accounting.json").read_text())
    accounting.update(
        physics_steps=int(ledger.physics_steps),
        control_intervals=int(ledger.control_intervals),
        setup_physics_steps=2,
        hard_physics_ceiling=650,
    )
    write_json(output / "cost_accounting.json", accounting)
    write_json(output / "decision.json", {
        "classification": classification,
        "physics_authorized": True,
        "physical_quality_pass": physical_quality_pass,
        "authorization": contract["authorization"],
    })
    with (output / "summary.md").open("a") as stream:
        stream.write(
            f"\n- Phase-D classification: `{classification}`; cold replay bitwise: "
            f"`{parity['bitwise_equal']}`; physics steps: `{ledger.physics_steps}`.\n"
        )
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "server_artifacts.sha256"
    )
    (output / "server_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    return {"classification": classification, "physics_steps": ledger.physics_steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("static", "render", "finalize-static", "physics"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--visual-pass", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()
    if args.phase == "static":
        result = static(args.output)
    elif args.phase == "render":
        result = render(args.output)
    elif args.phase == "finalize-static":
        if args.visual_pass is None or not args.visual_note:
            raise ValueError("finalize-static requires explicit --visual-pass/--no-visual-pass and --visual-note")
        result = finalize_static(args.output, args.visual_pass, args.visual_note)
    else:
        result = physics(args.output)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
