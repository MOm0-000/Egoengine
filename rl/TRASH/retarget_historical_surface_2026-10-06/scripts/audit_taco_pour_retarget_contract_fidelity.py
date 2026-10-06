#!/usr/bin/env python3
"""Zero-physics fidelity audit for the Pour retarget representation contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data_all/zzx/3.2RL")
CONFIG = ROOT / "configs/taco_pour_retarget_contract_fidelity_audit_v1.yaml"
OUTPUT = ASSET_ROOT / "runs/taco_pour_retarget_contract_fidelity_audit_v1"
BASELINE = "805d2799a138aaa84bc322ca2b3c43e3d5ab157b"

sys.path.insert(0, str(ROOT / "scripts"))

from audit_taco_initialization import visual_meshes
from build_taco_pour_initialization_candidates import native_geometry_gate
from egoengine_repro.retarget.collision_audit import collision_families, distances
from egoengine_repro.retarget.contract_fidelity import (
    FINGERS,
    HUMAN_CHAINS,
    ROBOT_ACTIVE_JOINTS,
    ROBOT_CHAINS,
    Probe,
    angle,
    deterministic_probe_arrays,
    direct_directions,
    finite_difference_jacobian,
    finger_probes,
    human_chain,
    joint_chain_singular_values,
    palm_local,
    robot_landmarks,
    rotation_angle,
    semantic_landmark_candidates,
    wrist_z_probes,
)
from egoengine_repro.retarget.interaction_aware import (
    deterministic_surface_samples,
    interaction_error,
    interaction_topology,
    pose_local_points,
    robot_semantic_points,
)
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from video_to_spider.rl.physics_contract import compile_mujoco_model


SOURCE_PINS = {
    "egoengine": {
        "kind": "project_supplied_paper",
        "paper": "EgoEngine From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations",
        "local_path": str(ASSET_ROOT / "paper/EgoEngine From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations.pdf"),
    },
    "toporetarget": {
        "kind": "paper_and_project_page_only_no_verified_method_repository",
        "arxiv": "2606.16272v2",
        "arxiv_url": "https://arxiv.org/abs/2606.16272v2",
        "project": "https://tsinghua-mars-lab.github.io/toporetarget-web/",
    },
    "regrind": {
        "repository": "https://github.com/yunhaif/regrind",
        "commit": "38347a9e30184620df04e19c63c7c72378cae103",
        "local_checkout": "/data_all/zzx/egoengine_research_sources/regrind",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def contract() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text())
    if value.get("schema") != "taco_pour_retarget_contract_fidelity_audit_v1":
        raise ValueError("unexpected contract")
    if any(value["authorization"].values()):
        raise ValueError("fidelity audit must authorize no mutation/runtime work")
    return value


def paths(value: dict[str, Any]) -> dict[str, Path]:
    result = {key: Path(raw) for key, raw in value["inputs"].items()}
    result.update(
        contract=CONFIG,
        runner=Path(__file__),
        helper=ROOT / "src/egoengine_repro/retarget/contract_fidelity.py",
        interaction_helper=ROOT / "src/egoengine_repro/retarget/interaction_aware.py",
        prior_interaction_manifest=ASSET_ROOT / "runs/taco_pour_left_interaction_retarget_v1/input_manifest.json",
        prior_shape_manifest=ASSET_ROOT / "runs/taco_pour_shape_aware_initialization_v1/input_manifest.json",
    )
    return result


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean_baseline() -> None:
    observed = head()
    if observed != BASELINE:
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, observed], cwd=ROOT, check=True)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("formal audit requires a clean implementation worktree")


def compile_model(input_paths: dict[str, Path]) -> mujoco.MjModel:
    config = yaml.safe_load(input_paths["simulator_config"].read_text())
    return compile_mujoco_model(input_paths["scene"], config.get("sdf_octree_depths", {}))


def input_records(input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    records = [artifact(path) for path in input_paths.values()]
    records.extend(scene_mesh_artifacts(input_paths["scene"]))
    unique = {row["path"]: row for row in records}
    return [unique[key] for key in sorted(unique)]


def trajectory_inputs(input_paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    human = arrays(input_paths["human_reference"])
    old = arrays(input_paths["old_mink_reference"])
    replay = arrays(input_paths["accepted_a_replay"])
    shape = arrays(input_paths["shape_aware_candidate"])
    interaction = arrays(input_paths["interaction_v1_candidate"])
    trajectories = {
        "OLD_MINK_REFERENCE": {i: old["qpos"][i].astype(np.float64) for i in range(21)},
        "ACCEPTED_A_REPLAY": {i: replay["A_qpos"][i].astype(np.float64) for i in range(21)},
        "SHAPE_AWARE_NEGATIVE": {0: shape["qpos"].astype(np.float64)},
        "INTERACTION_V1_NEGATIVE": {i: interaction["qpos"][i].astype(np.float64) for i in range(21)},
    }
    if not np.array_equal(old["frame_indices"][:21], human["frame_indices"][:21]):
        raise ValueError("human and old-reference frame IDs differ")
    if not np.array_equal(old["timestamps_s"][:21], human["timestamps_s"][:21]):
        raise ValueError("human and old-reference timestamps differ")
    return human, trajectories


def model_rotations(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    palm = int(model.body("left_hand_link").id)
    return data.xpos[palm].copy(), data.xmat[palm].reshape(3, 3).copy()


def native_local_distance(mesh: trimesh.Trimesh, point: np.ndarray) -> tuple[np.ndarray, float, float | None]:
    closest, distance, _ = trimesh.proximity.closest_point_naive(mesh, np.asarray(point)[None])
    signed = None
    if mesh.is_volume:
        signed = float(closed_mesh_signed_distance(mesh, np.asarray(point)[None])[0])
    return closest[0], float(distance[0]), signed


def state_component_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tray_mesh: trimesh.Trimesh,
    surface_points: np.ndarray,
    human: dict[str, np.ndarray],
    qpos: np.ndarray,
    endpoint: int,
    trajectory: str,
) -> list[dict[str, Any]]:
    left = int(np.flatnonzero(human["hand_order"] == "left")[0])
    target = int(np.flatnonzero(human["object_roles"] == "target")[0])
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    landmarks = robot_landmarks(model, data, qpos, mapping="joint_anchor")
    palm_id = int(model.body("left_hand_link").id)
    palm_position = data.xpos[palm_id].copy()
    palm_rotation = data.xmat[palm_id].reshape(3, 3).copy()
    human_wrist = human["T_sim_wrist_target"][endpoint, left]
    source_joints = human["joint_positions_sim"][endpoint, left]
    source_object = human["T_sim_object_reference"][endpoint, target]
    target_body = int(model.body("left_object").id)
    object_position = data.xpos[target_body].copy()
    object_rotation = data.xmat[target_body].reshape(3, 3).copy()
    rows: list[dict[str, Any]] = []

    wrist_error = rotation_angle(palm_rotation, human_wrist[:3, :3])
    palm_source_local = (human_wrist[:3, 3] - source_object[:3, 3]) @ source_object[:3, :3]
    palm_robot_local = (palm_position - object_position) @ object_rotation
    rows.append(dict(
        trajectory=trajectory, endpoint=endpoint, finger="ALL",
        component="wrist_orientation_error_rad", value=wrist_error, units="rad",
    ))
    rows.append(dict(
        trajectory=trajectory, endpoint=endpoint, finger="ALL",
        component="wrist_tray_position_error_m",
        value=float(np.linalg.norm(palm_robot_local - palm_source_local)), units="m",
    ))

    global_bone = 0.0
    for finger_index, finger in enumerate(FINGERS):
        robot_chain = landmarks[finger]
        source_chain = human_chain(source_joints, finger)
        robot_local = palm_local(robot_chain, palm_position, palm_rotation)
        source_local = palm_local(source_chain, human_wrist[:3, 3], human_wrist[:3, :3])
        robot_prox, robot_distal = direct_directions(robot_local)
        source_prox, source_distal = direct_directions(source_local)
        tip_site = int(model.site(ROBOT_CHAINS[finger][-1]).id)
        robot_tip_position = data.site_xpos[tip_site]
        robot_tip_rotation = data.site_xmat[tip_site].reshape(3, 3)
        human_tip = human["T_sim_fingertip_target"][endpoint, left, finger_index]
        metrics = {
            "fingertip_position_error_m": float(np.linalg.norm(robot_tip_position - human_tip[:3, 3])),
            "fingertip_orientation_error_rad": rotation_angle(robot_tip_rotation, human_tip[:3, :3]),
            "wrist_to_tip_vector_error_m": float(np.linalg.norm(
                (robot_tip_position - palm_position) @ palm_rotation
                - (human_tip[:3, 3] - human_wrist[:3, 3]) @ human_wrist[:3, :3]
            )),
            "direct_proximal_orientation_error_rad": angle(robot_prox, source_prox),
            "direct_distal_orientation_error_rad": angle(robot_distal, source_distal),
        }
        thumb_robot = landmarks["thumb"][-1]
        thumb_source = human_chain(source_joints, "thumb")[-1]
        metrics["thumb_to_tip_vector_error_m"] = float(np.linalg.norm(
            (robot_tip_position - thumb_robot) @ palm_rotation
            - (source_chain[-1] - thumb_source) @ human_wrist[:3, :3]
        ))
        old_style_residual = (robot_prox - robot_distal) - (source_prox - source_distal)
        bone = float(np.dot(old_style_residual, old_style_residual))
        metrics["global_E_bone_diagnostic"] = bone
        global_bone += bone
        for component, value in metrics.items():
            rows.append(dict(
                trajectory=trajectory, endpoint=endpoint, finger=finger,
                component=component, value=value,
                units="rad" if component.endswith("_rad") else ("m" if component.endswith("_m") else "squared_direction"),
            ))

        if endpoint >= 14 and finger in ("ring", "pinky"):
            source_midpoints = ((source_chain[0] + source_chain[1]) * 0.5,
                                (source_chain[2] + source_chain[3]) * 0.5)
            robot_midpoints = ((robot_chain[0] + robot_chain[1]) * 0.5,
                               (robot_chain[1] + robot_chain[2]) * 0.5)
            for segment, source_mid, robot_mid, source_direction, robot_direction in zip(
                ("proximal", "distal"), source_midpoints, robot_midpoints,
                (source_prox, source_distal), (robot_prox, robot_distal), strict=True,
            ):
                source_mid_local = (source_mid - source_object[:3, 3]) @ source_object[:3, :3]
                robot_mid_local = (robot_mid - object_position) @ object_rotation
                closest, unsigned, signed = native_local_distance(tray_mesh, robot_mid_local)
                source_direction_tray = source_direction @ (human_wrist[:3, :3].T @ source_object[:3, :3])
                robot_direction_tray = robot_direction @ (palm_rotation.T @ object_rotation)
                common = dict(trajectory=trajectory, endpoint=endpoint, finger=finger, segment=segment)
                rows.extend([
                    {**common, "component": "near_interaction_position_error_m",
                     "value": float(np.linalg.norm(robot_mid_local - source_mid_local)), "units": "m"},
                    {**common, "component": "near_interaction_orientation_error_rad",
                     "value": angle(robot_direction_tray, source_direction_tray), "units": "rad"},
                    {**common, "component": "native_surface_unsigned_distance_m",
                     "value": unsigned, "units": "m"},
                    {**common, "component": "native_surface_signed_distance_m",
                     "value": "" if signed is None else signed, "units": "m",
                     "signed_distance_available": signed is not None,
                     "closest_point_tray_x": closest[0], "closest_point_tray_y": closest[1],
                     "closest_point_tray_z": closest[2],
                     "segment_midpoint_tray_x": robot_mid_local[0],
                     "segment_midpoint_tray_y": robot_mid_local[1],
                     "segment_midpoint_tray_z": robot_mid_local[2]},
                ])

    rows.append(dict(
        trajectory=trajectory, endpoint=endpoint, finger="ALL",
        component="global_E_bone_diagnostic", value=global_bone, units="squared_direction",
    ))

    legacy_points = robot_semantic_points(model, data, qpos)
    source_reduced = [source_joints[0]]
    for finger in FINGERS:
        chain = human_chain(source_joints, finger)
        source_reduced.extend([chain[0], chain[1], chain[3]])
    source_reduced = np.asarray(source_reduced)
    robot_object_points = pose_local_points(surface_points, object_rotation, object_position)
    source_object_points = pose_local_points(surface_points, source_object[:3, :3], source_object[:3, 3])
    source_vertices = np.concatenate([source_reduced, source_object_points])
    _, neighbors = interaction_topology(source_vertices)
    robot_vertices = np.concatenate([legacy_points, robot_object_points])
    rows.append(dict(
        trajectory=trajectory, endpoint=endpoint, finger="ALL",
        component="global_E_IM_diagnostic",
        value=interaction_error(robot_vertices, source_vertices, neighbors, 30.0),
        units="m_squared",
    ))
    return rows


def add_trajectory_regularizers(
    rows: list[dict[str, Any]], trajectories: dict[str, dict[int, np.ndarray]], old: dict[int, np.ndarray]
) -> None:
    for name, states in trajectories.items():
        ordered = sorted(states)
        previous = None
        for endpoint in ordered:
            qpos = states[endpoint]
            nominal = float(np.linalg.norm(qpos[18:36] - old[endpoint][18:36]))
            temporal = "" if previous is None or endpoint != previous[0] + 1 else float(
                np.linalg.norm(qpos[18:36] - previous[1][18:36])
            )
            rows.append(dict(trajectory=name, endpoint=endpoint, finger="ALL",
                             component="joint_nominal_deviation_rad_l2", value=nominal, units="rad_l2"))
            rows.append(dict(trajectory=name, endpoint=endpoint, finger="ALL",
                             component="temporal_joint_change_rad_l2", value=temporal, units="rad_l2"))
            previous = (endpoint, qpos)


def semantic_audit(
    output: Path, model: mujoco.MjModel, human: dict[str, np.ndarray], old: dict[int, np.ndarray], cfg: dict[str, Any]
) -> dict[str, Any]:
    mapping = semantic_landmark_candidates(model)
    epsilon = float(cfg["landmarks"]["finite_difference_rad"])
    records = []
    arrays_out: dict[str, np.ndarray] = {}
    own_threshold = float(cfg["landmarks"]["own_chain_min_sensitivity_m_per_rad"])
    unrelated_threshold = float(cfg["landmarks"]["unrelated_max_sensitivity_m_per_rad"])
    singular_threshold = float(cfg["landmarks"]["minimum_jacobian_singular_value_m_per_rad"])
    checks = []
    for endpoint in cfg["frames"]["semantic_debug"]:
        legacy = robot_landmarks(model, mujoco.MjData(model), old[endpoint], mapping="legacy_body_origin")
        anchors = robot_landmarks(model, mujoco.MjData(model), old[endpoint], mapping="joint_anchor")
        left = int(np.flatnonzero(human["hand_order"] == "left")[0])
        source_joints = human["joint_positions_sim"][endpoint, left]
        for finger in FINGERS:
            source_chain = human_chain(source_joints, finger)
            source_proximal, source_distal = direct_directions(source_chain)
            arrays_out[f"e{endpoint:02d}__human_{finger}__full_chain"] = source_chain
            arrays_out[f"e{endpoint:02d}__human_{finger}__direct_proximal"] = source_proximal
            arrays_out[f"e{endpoint:02d}__human_{finger}__direct_distal"] = source_distal
            singular = joint_chain_singular_values(model, old[endpoint], finger, epsilon)
            records.append({
                "endpoint": endpoint, "finger": finger,
                "human_full_chain_indices": list(HUMAN_CHAINS[finger]),
                "human_full_chain_segment_lengths_m": np.linalg.norm(
                    np.diff(source_chain, axis=0), axis=1
                ).tolist(),
                "human_direct_proximal_direction": source_proximal.tolist(),
                "human_direct_distal_direction": source_distal.tolist(),
                "legacy_vs_joint_anchor_max_abs_m": float(
                    np.max(np.abs(legacy[finger] - anchors[finger]))
                ),
                "joint_chain_singular_values_m_per_rad": singular.tolist(),
                "minimum_singular_value_m_per_rad": float(singular.min()),
            })
            checks.append(float(singular.min()) >= singular_threshold)
            for joint_name in ROBOT_ACTIVE_JOINTS[finger]:
                jac = finite_difference_jacobian(model, old[endpoint], joint_name, epsilon)
                key = f"e{endpoint:02d}__{joint_name}"
                arrays_out[key + "__wrist"] = jac["wrist"]
                for target_finger in FINGERS:
                    arrays_out[key + "__" + target_finger] = jac[target_finger]
                own_norm = float(np.linalg.norm(jac[finger]))
                unrelated = max(float(np.linalg.norm(jac[other])) for other in FINGERS if other != finger)
                records.append({
                    "endpoint": endpoint, "finger": finger, "joint": joint_name,
                    "effective_step_rad": float(jac["effective_step_rad"]),
                    "requested_center_rad": float(jac["requested_center_rad"]),
                    "evaluation_center_rad": float(jac["evaluation_center_rad"]),
                    "evaluation_center_shift_rad": float(jac["evaluation_center_shift_rad"]),
                    "own_chain_sensitivity_m_per_rad": own_norm,
                    "unrelated_chain_max_sensitivity_m_per_rad": unrelated,
                    "own_chain_meaningful": own_norm >= own_threshold,
                    "unrelated_chains_stationary": unrelated <= unrelated_threshold,
                })
                checks.extend([own_norm >= own_threshold, unrelated <= unrelated_threshold])
    np.savez_compressed(output / "semantic_landmark_jacobians.npz", **arrays_out)
    result = {
        "schema": "taco_pour_semantic_landmark_candidates_v1",
        "preferred_mapping": "joint_anchor",
        "selection_frozen_before_probe_scores": True,
        "fixed_link_local_points_added": [],
        "fake_joints_or_dofs_added": False,
        "mapping": mapping,
        "audit_records": records,
        "thresholds": cfg["landmarks"],
        "certified": bool(all(checks)),
    }
    write_json(output / "semantic_landmark_candidates.json", result)
    lines = [
        "# Semantic landmark audit", "",
        "- Preferred map: MuJoCo joint anchors (`data.xanchor`) plus physical fingertip sites.",
        "- Human ring/pinky diagnostics retain `[MCP, PIP, DIP, TIP]`; DIP is not discarded.",
        f"- Certified: `{result['certified']}`.",
        "- Body-origin landmarks are retained only as a numerical comparison.", "",
    ]
    for row in records:
        if "joint" in row:
            lines.append(
                f"- e{row['endpoint']:02d} {row['joint']}: own `{row['own_chain_sensitivity_m_per_rad']:.6g}` m/rad; "
                f"unrelated max `{row['unrelated_chain_max_sensitivity_m_per_rad']:.3g}` m/rad."
            )
    (output / "semantic_landmark_audit.md").write_text("\n".join(lines) + "\n")
    return result


def stop_after_semantic_blocker(output: Path, cfg: dict[str, Any]) -> None:
    """Close a Phase-A-only run without manufacturing downstream evidence."""
    write_json(output / "cost_accounting.json", {
        "physics_steps": 0, "control_intervals": 0, "planner_calls": 0,
        "actor_critic_forwards": 0, "optimizer_generated_candidates": 0,
        "rl_updates": 0, "chunk_commits": 0,
        "static_fk_states": 0, "deterministic_probe_count": 0, "render_updates": 0,
    })
    write_json(output / "decision.json", {
        "classification": "SEMANTIC_LANDMARK_BLOCKER",
        "blockers": ["SEMANTIC_LANDMARK_BLOCKER"],
        "semantic_landmarks_certified": False,
        "objective_components_certified": False,
        "collision_contract_certified": False,
        "downstream_phases_executed": False,
        "authorization": cfg["authorization"],
        "retarget_v2_contract_written": False,
    })
    (output / "summary.md").write_text(
        "# TACO Pour retarget contract fidelity audit v1\n\n"
        "- Final classification: `SEMANTIC_LANDMARK_BLOCKER`.\n"
        "- Phase A did not certify a defensible semantic landmark map.\n"
        "- Phases B--F were not executed, as required by the frozen contract.\n"
        "- No objective or collision conclusion was inferred from an uncertified representation.\n"
        "- Physics/control/planner/actor/critic/RL/promotion/chunk counts are all zero.\n"
    )
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "server_artifacts.sha256"
    )
    (output / "server_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )


def semantic_figures(
    output: Path, model: mujoco.MjModel, human: dict[str, np.ndarray], old: dict[int, np.ndarray], cfg: dict[str, Any]
) -> None:
    left = int(np.flatnonzero(human["hand_order"] == "left")[0])
    directory = output / "semantic_debug"
    directory.mkdir(parents=True, exist_ok=True)
    for endpoint in cfg["frames"]["semantic_figures"]:
        legacy = robot_landmarks(model, mujoco.MjData(model), old[endpoint], mapping="legacy_body_origin")
        anchors = robot_landmarks(model, mujoco.MjData(model), old[endpoint], mapping="joint_anchor")
        source = human["joint_positions_sim"][endpoint, left]
        fig = plt.figure(figsize=(15, 5))
        for panel, (title, value) in enumerate((("HUMAN FULL MANO21", source), ("LEGACY BODY ORIGINS", legacy), ("JOINT ANCHORS + TIP", anchors)), 1):
            axis = fig.add_subplot(1, 3, panel, projection="3d")
            if isinstance(value, dict):
                for finger in FINGERS:
                    points = value[finger]
                    axis.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", label=finger)
            else:
                for finger in FINGERS:
                    points = human_chain(value, finger)
                    axis.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", label=finger)
            axis.set_title(f"{title} | endpoint {endpoint}")
            axis.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(directory / f"endpoint_{endpoint:03d}.png", dpi=160)
        plt.close(fig)


def collision_row(
    model: mujoco.MjModel, meshes: dict[int, trimesh.Trimesh], qpos: np.ndarray,
    label: str, endpoint: int, cfg: dict[str, Any],
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    groups = collision_families(model)
    is_left = lambda geom: model.geom(geom).name.startswith("collision_hand_left_")
    target_pairs = [pair for pair in groups["hand_target"] if is_left(pair[0])]
    self_pairs = [pair for pair in groups["self_explicit"] if is_left(pair[0]) or is_left(pair[1])]
    target_distance = distances(model, data, target_pairs)
    self_distance = distances(model, data, self_pairs)
    limit = float(cfg["collision"]["runtime_minimum_distance_m"])
    numerical = float(cfg["collision"]["numerical_tolerance_m"])
    proxy_interference = bool(
        (target_distance < limit - numerical).any() or (self_distance < limit - numerical).any()
    )
    native = native_geometry_gate(
        model, qpos, meshes, float(cfg["collision"]["native_material_reporting_threshold_m"])
    )
    failures = native["failures"]
    relevant = {
        "left_hand_table": [name for name in failures["hand_table"] if name.startswith("left_")],
        "left_hand_object": [pair for pair in failures["hand_object"] if pair and pair[0].startswith("left_")],
        "left_omitted_nonadjacent": [pair for pair in failures["omitted_nonadjacent"] if any(name.startswith("left_") for name in pair)],
        "left_unclassified_omitted_nonadjacent": [pair for pair in failures["unclassified_omitted_nonadjacent"] if any(name.startswith("left_") for name in pair)],
    }
    native_unknown = bool(relevant["left_unclassified_omitted_nonadjacent"])
    native_interference = bool(
        relevant["left_hand_table"] or relevant["left_hand_object"] or relevant["left_omitted_nonadjacent"]
    )
    limited = model.actuator_ctrllimited.astype(bool)
    ranges = model.actuator_ctrlrange
    control = qpos[:model.nu]
    ctrl_legal = bool(np.all((~limited) | ((control >= ranges[:, 0] - numerical) & (control <= ranges[:, 1] + numerical))))
    joint_legal = True
    margins = []
    for joint in range(model.njnt):
        if not model.jnt_limited[joint]:
            continue
        address = int(model.jnt_qposadr[joint])
        margin = min(qpos[address] - model.jnt_range[joint, 0], model.jnt_range[joint, 1] - qpos[address])
        margins.append(float(margin))
        joint_legal &= margin >= -numerical
    if native_unknown:
        classification = "UNKNOWN"
    elif proxy_interference and native_interference:
        classification = "PROXY_AND_NATIVE_AGREE_INTERFERENCE"
    elif not proxy_interference and not native_interference:
        classification = "PROXY_AND_NATIVE_AGREE_LEGAL"
    elif proxy_interference:
        classification = "PROXY_STRICTER_THAN_NATIVE"
    else:
        classification = "NATIVE_STRICTER_THAN_PROXY"
    return {
        "state": label, "endpoint": endpoint,
        "runtime_target_min_distance_m": float(target_distance.min()),
        "runtime_self_min_distance_m": float(self_distance.min()),
        "runtime_proxy_interference": proxy_interference,
        "native_material_interference": native_interference,
        "native_unknown": native_unknown,
        "joint_legal": joint_legal,
        "joint_min_margin": min(margins) if margins else "",
        "control_legal": ctrl_legal,
        "classification": classification,
        "candidate_relevant_findings": json.dumps(relevant, sort_keys=True),
        "scene_monitor_findings": json.dumps({
            "right_or_other_hand_table": [name for name in failures["hand_table"] if not name.startswith("left_")],
            "object_table": failures["object_table"],
            "right_or_other_hand_object": [pair for pair in failures["hand_object"] if not (pair and pair[0].startswith("left_"))],
        }, sort_keys=True),
    }


def prepare(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require_clean_baseline()
    cfg = contract()
    input_paths = paths(cfg)
    records = input_records(input_paths)
    verify_artifacts(records)
    regrind_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_PINS["regrind"]["local_checkout"], text=True
    ).strip()
    if regrind_head != SOURCE_PINS["regrind"]["commit"]:
        raise RuntimeError("local REGRIND checkout differs from frozen pin")
    output.mkdir(parents=True)
    write_json(output / "source_pins.json", {
        "schema": "taco_pour_retarget_contract_fidelity_source_pins_v1",
        "pins": SOURCE_PINS,
        "source_lessons_used": {
            "egoengine": "fingertip pose and wrist orientation are preserved as separate MINK terms",
            "toporetarget": "E_bone and global E_IM remain diagnostics; no contact claim without labels",
            "regrind": "joint-anchor landmarks and nominal/trust-region semantics are preserved conceptually",
            "key_objectives": "factorized anchor, vector, orientation, nominal, and temporal terms",
            "maniptrans": "basic hand fidelity remains separate from interaction correction",
            "spider": "no physics before a credible kinematic contract",
            "unidex": "source boundary only; no author parameters inferred",
        },
    })
    write_json(output / "input_manifest.json", {
        "schema": "taco_pour_retarget_contract_fidelity_input_manifest_v1",
        "expected_baseline": BASELINE,
        "implementation_commit": head(),
        "artifacts": records,
        "budget": {
            "physics_steps": 0, "control_intervals": 0, "planner_calls": 0,
            "actor_critic_forwards": 0, "optimizer_generated_candidates": 0,
            "rl_updates": 0, "chunk_commits": 0,
        },
    })
    human, trajectories = trajectory_inputs(input_paths)
    old = trajectories["OLD_MINK_REFERENCE"]
    accepted = trajectories["ACCEPTED_A_REPLAY"]
    model = compile_model(input_paths)
    meshes, mesh_paths = visual_meshes(input_paths["scene"], model)
    mesh_path_by_geom = dict(zip(meshes, mesh_paths, strict=True))
    tray_geom = int(model.geom("left_object_visual").id)
    tray_mesh = meshes[tray_geom]
    surface_points, face_indices, barycentric = deterministic_surface_samples(
        tray_mesh.vertices, tray_mesh.faces, 50, 0
    )
    np.savez_compressed(
        output / "native_tray_surface_samples.npz",
        points_tray_local_m=surface_points, face_index=face_indices,
        barycentric=barycentric, mesh_sha256=np.asarray(sha256(mesh_path_by_geom[tray_geom])),
    )

    semantic = semantic_audit(output, model, human, old, cfg)
    semantic_figures(output, model, human, old, cfg)
    if not semantic["certified"]:
        stop_after_semantic_blocker(output, cfg)
        return

    component_rows: list[dict[str, Any]] = []
    data = mujoco.MjData(model)
    for trajectory, states in trajectories.items():
        for endpoint, qpos in sorted(states.items()):
            component_rows.extend(state_component_rows(
                model, data, tray_mesh, surface_points, human, qpos, endpoint, trajectory
            ))
    add_trajectory_regularizers(component_rows, trajectories, old)
    write_csv(output / "objective_components.csv", component_rows)

    probes = finger_probes(
        model, np.stack([old[i] for i in range(21)]),
        list(cfg["frames"]["counterfactual_finger_probes"]),
        list(cfg["finger_probe_fraction_of_joint_range"]),
    )
    probes.extend(wrist_z_probes(
        np.stack([old[i] for i in range(21)]),
        np.stack([accepted[i] for i in range(21)]),
        list(cfg["wrist_z_interpolation_fractions"]),
    ))
    probes.append(Probe(
        "historical_shape_aware_e00", "historical_negative", 0,
        trajectories["SHAPE_AWARE_NEGATIVE"][0].copy(), {"source": "SHAPE_AWARE_NEGATIVE"},
    ))
    for endpoint in cfg["frames"]["historical_interaction_visuals"]:
        probes.append(Probe(
            f"historical_interaction_v1_e{endpoint:02d}", "historical_negative", endpoint,
            trajectories["INTERACTION_V1_NEGATIVE"][endpoint].copy(),
            {"source": "INTERACTION_V1_NEGATIVE"},
        ))
    blind_ids = [f"P{index:03d}" for index in range(len(probes))]
    np.savez_compressed(output / "probe_states.npz", **deterministic_probe_arrays(probes))
    manifest = []
    for blind_id, probe in zip(blind_ids, probes, strict=True):
        manifest.append({
            "blind_id": blind_id, "probe_id": probe.probe_id, "family": probe.family,
            "endpoint": probe.endpoint, "metadata": probe.metadata,
            "optimizer_generated": False, "promotion_eligible": False,
        })
    write_json(output / "probe_manifest.json", {
        "schema": "taco_pour_retarget_contract_fidelity_probe_manifest_v1",
        "blinding": "visual filenames and panels expose only blind_id and endpoint",
        "probes": manifest,
    })
    probe_rows: list[dict[str, Any]] = []
    for blind_id, probe in zip(blind_ids, probes, strict=True):
        rows = state_component_rows(
            model, data, tray_mesh, surface_points, human, probe.qpos, probe.endpoint, blind_id
        )
        for row in rows:
            row.update(blind_id=blind_id, probe_id=probe.probe_id, family=probe.family)
        probe_rows.extend(rows)
    write_csv(output / "probe_metrics.csv", probe_rows)

    collision_rows = []
    for trajectory in ("OLD_MINK_REFERENCE", "ACCEPTED_A_REPLAY", "INTERACTION_V1_NEGATIVE"):
        for endpoint, qpos in sorted(trajectories[trajectory].items()):
            collision_rows.append(collision_row(model, meshes, qpos, trajectory, endpoint, cfg))
    collision_rows.append(collision_row(
        model, meshes, trajectories["SHAPE_AWARE_NEGATIVE"][0], "SHAPE_AWARE_NEGATIVE", 0, cfg
    ))
    for probe in probes:
        if probe.family == "wrist_z":
            collision_rows.append(collision_row(model, meshes, probe.qpos, probe.probe_id, 0, cfg))
    write_csv(output / "collision_contract_audit.csv", collision_rows)

    write_json(output / "cost_accounting.json", {
        "physics_steps": 0, "control_intervals": 0, "planner_calls": 0,
        "actor_critic_forwards": 0, "optimizer_generated_candidates": 0,
        "rl_updates": 0, "chunk_commits": 0,
        "static_fk_states": sum(len(states) for states in trajectories.values()) + len(probes),
        "objective_component_rows": len(component_rows),
        "deterministic_probe_count": len(probes),
        "render_updates": 0,
    })
    write_json(output / "decision.json", {
        "classification": "PENDING_BLINDED_VISUAL_REVIEW",
        "authorization": cfg["authorization"],
        "semantic_landmarks_certified": semantic["certified"],
    })
    (output / "summary.md").write_text(
        "# TACO Pour retarget contract fidelity audit v1\n\n"
        "- Zero-physics representation/objective/collision certification audit.\n"
        "- No optimizer-generated candidate exists.\n"
        "- Semantic, factorized metric, deterministic probe, and collision evidence are prepared.\n"
        "- Blinded visual review remains pending; objective values must not be revealed to the reviewer yet.\n"
        "- Physics/control/planner/RL/promotion/chunk counts are all zero.\n"
    )


def _load_visual_module():
    path = ROOT / "scripts/inspect_hand_object_trajectory.py"
    spec = importlib.util.spec_from_file_location("fidelity_visual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render(output: Path) -> None:
    if not (output / "probe_manifest.json").is_file():
        raise RuntimeError("prepare stage missing")
    cfg = contract()
    input_paths = paths(cfg)
    human = arrays(input_paths["human_reference"])
    old = arrays(input_paths["old_mink_reference"])
    probes = arrays(output / "probe_states.npz")
    manifest = json.loads((output / "probe_manifest.json").read_text())["probes"]
    vismod = _load_visual_module()
    rgb, rgb_info = vismod.read_rgb_frames(input_paths["rgb_video"], 21)
    reference_model = vismod.StaticModel.load("old", input_paths["scene"])
    actual_model = vismod.StaticModel.load("probe", input_paths["scene"])
    vis = vismod.Visualizer(reference_model, actual_model)
    visual_root = output / "visuals"
    for name in ("top", "oblique", "closeups"):
        (visual_root / name).mkdir(parents=True, exist_ok=True)
    image_rows = []
    try:
        for index, row in enumerate(manifest):
            endpoint = int(row["endpoint"])
            blind_id = row["blind_id"]
            qpos = probes["qpos"][index]
            for view in cfg["visual_review"]["views"]:
                panels = [
                    vismod.annotate_panel(
                        vismod.letterbox(rgb[endpoint], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT),
                        ["REAL RGB (independent camera)", f"source frame {endpoint}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_human(human["T_sim_object_reference"][endpoint],
                                         human["joint_positions_sim"][endpoint], view),
                        ["HUMAN 3-D SOURCE", f"endpoint {endpoint} | {view}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_reference(old["qpos"][endpoint], view),
                        ["OLD MINK REFERENCE", f"endpoint {endpoint} | {view}"],
                    ),
                    vismod.annotate_panel(
                        vis.render_actual(qpos, view),
                        [f"BLINDED PROBE {blind_id}", f"endpoint {endpoint} | {view}"],
                    ),
                ]
                image = np.concatenate(panels, axis=1)
                relative = f"visuals/{view}/{blind_id}.png"
                vismod.save_rgb(output / relative, image)
                close = cv2.resize(
                    image[:, vismod.PANEL_WIDTH:], (1440, 540), interpolation=cv2.INTER_CUBIC
                )
                close_relative = f"visuals/closeups/{blind_id}_{view}.png"
                vismod.save_rgb(output / close_relative, close)
                image_rows.extend([
                    {"blind_id": blind_id, "endpoint": endpoint, "view": view,
                     "path": relative, "sha256": sha256(output / relative)},
                    {"blind_id": blind_id, "endpoint": endpoint, "view": view + "_closeup",
                     "path": close_relative, "sha256": sha256(output / close_relative)},
                ])
    finally:
        vis.close()
    visual_manifest = output / "visual_manifest.json"
    write_json(visual_manifest, {
        "schema": "taco_pour_retarget_contract_fidelity_visual_manifest_v1",
        "probe_metadata_hidden": True,
        "objective_values_hidden": True,
        "rgb": rgb_info,
        "camera_registration": "RGB is time-aligned, not pixel-registered; 3-D panels use fixed simulation cameras",
        "images": image_rows,
        "render_updates": int(vis.render_updates),
    })
    manifest_hash = sha256(visual_manifest)
    lines = ["# Blinded visual index", "", f"Visual manifest SHA-256: `{manifest_hash}`", ""]
    for row in manifest:
        blind_id = row["blind_id"]
        lines.append(
            f"- {blind_id} endpoint {row['endpoint']}: "
            f"[top](visuals/top/{blind_id}.png) · [oblique](visuals/oblique/{blind_id}.png) · "
            f"[top close-up](visuals/closeups/{blind_id}_top.png) · "
            f"[oblique close-up](visuals/closeups/{blind_id}_oblique.png)"
        )
    (output / "VISUAL_INDEX.md").write_text("\n".join(lines) + "\n")
    accounting = json.loads((output / "cost_accounting.json").read_text())
    accounting["render_updates"] = int(vis.render_updates)
    write_json(output / "cost_accounting.json", accounting)
    write_json(output / "visual_review_status.json", {
        "status": "PENDING_BLINDED_LABELS",
        "visual_manifest_sha256": manifest_hash,
        "objective_metrics_must_remain_unread": True,
    })


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str, str], float]:
    result: dict[tuple[str, int, str, str], list[float]] = {}
    for row in rows:
        value = row.get("value", "")
        if value == "":
            continue
        key = (row["trajectory"], int(row["endpoint"]), row["finger"], row["component"])
        result.setdefault(key, []).append(float(value))
    return {key: float(np.sqrt(np.mean(np.square(values)))) for key, values in result.items()}


def objective_fidelity(
    output: Path, labels: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe_rows = read_csv(output / "probe_metrics.csv")
    objective_rows = read_csv(output / "objective_components.csv")
    probe = metric_lookup(probe_rows)
    old = metric_lookup([row for row in objective_rows if row["trajectory"] == "OLD_MINK_REFERENCE"])
    manifest = {row["blind_id"]: row for row in json.loads((output / "probe_manifest.json").read_text())["probes"]}
    definitions = {
        "ring_shape": [
            ("ring", "direct_proximal_orientation_error_rad"),
            ("ring", "direct_distal_orientation_error_rad"),
            ("ring", "fingertip_position_error_m"),
            ("ring", "fingertip_orientation_error_rad"),
        ],
        "pinky_shape": [
            ("pinky", "direct_proximal_orientation_error_rad"),
            ("pinky", "direct_distal_orientation_error_rad"),
            ("pinky", "fingertip_position_error_m"),
            ("pinky", "fingertip_orientation_error_rad"),
        ],
        "wrist_palm_height_phase": [("ALL", "wrist_tray_position_error_m")],
        "tray_rim_spatial_relation": [
            ("ring", "near_interaction_position_error_m"),
            ("ring", "near_interaction_orientation_error_rad"),
            ("pinky", "near_interaction_position_error_m"),
            ("pinky", "near_interaction_orientation_error_rad"),
        ],
        "collateral_thumb_index_middle": [
            (finger, "fingertip_position_error_m") for finger in ("thumb", "index", "middle")
        ] + [(finger, "fingertip_orientation_error_rad") for finger in ("thumb", "index", "middle")],
    }
    matrix: list[dict[str, Any]] = []
    for blind_id, aspect_labels in labels["labels"].items():
        endpoint = int(manifest[blind_id]["endpoint"])
        for aspect, visual_label in aspect_labels.items():
            if visual_label not in ("PREFERRED_OVER_OLD", "WORSE_THAN_OLD"):
                continue
            expected = "LOWER" if visual_label == "PREFERRED_OVER_OLD" else "HIGHER"
            for finger, component in definitions[aspect]:
                probe_key = (blind_id, endpoint, finger, component)
                old_key = ("OLD_MINK_REFERENCE", endpoint, finger, component)
                if probe_key not in probe or old_key not in old:
                    continue
                delta = probe[probe_key] - old[old_key]
                measured = "LOWER" if delta < -1e-12 else ("HIGHER" if delta > 1e-12 else "EQUAL")
                matrix.append({
                    "blind_id": blind_id, "probe_id": manifest[blind_id]["probe_id"],
                    "endpoint": endpoint, "aspect": aspect, "finger": finger,
                    "objective_component": component, "visual_label": visual_label,
                    "expected_direction": expected, "old_value": old[old_key],
                    "probe_value": probe[probe_key], "delta": delta,
                    "measured_direction": measured, "pass": measured == expected,
                })
    by_component: dict[str, list[bool]] = {}
    for row in matrix:
        by_component.setdefault(row["objective_component"], []).append(bool(row["pass"]))
    certification = {
        name: {
            "applicable_unambiguous_comparisons": len(values),
            "passed_comparisons": int(sum(values)),
            "certified_for_v2": bool(values and all(values)),
        }
        for name, values in sorted(by_component.items())
    }
    required = {
        "fingertip_position_error_m", "fingertip_orientation_error_rad",
        "wrist_tray_position_error_m", "direct_proximal_orientation_error_rad",
        "direct_distal_orientation_error_rad", "near_interaction_position_error_m",
        "near_interaction_orientation_error_rad",
    }
    missing = sorted(required - certification.keys())
    failed = sorted(name for name in required if name in certification and not certification[name]["certified_for_v2"])
    report = {
        "component_certification": certification,
        "required_components": sorted(required),
        "required_components_without_unambiguous_evidence": missing,
        "required_components_failed": failed,
        "certified": not missing and not failed,
        "global_E_bone_role": "DIAGNOSTIC_ONLY",
        "global_E_IM_role": "DIAGNOSTIC_ONLY",
        "scalar_weight_fit_performed": False,
    }
    return matrix, report


def finalize(output: Path) -> None:
    cfg = contract()
    input_paths = paths(cfg)
    records = json.loads((output / "input_manifest.json").read_text())["artifacts"]
    verify_artifacts(records)
    labels_path = output / "blinded_visual_labels.json"
    if not labels_path.is_file():
        raise RuntimeError("sealed blinded labels are missing")
    labels = json.loads(labels_path.read_text())
    if labels.get("sealed") is not True:
        raise RuntimeError("blinded labels must be explicitly sealed before metric reveal")
    visual_hash = sha256(output / "visual_manifest.json")
    if labels.get("visual_manifest_sha256") != visual_hash:
        raise RuntimeError("labels are not bound to the visual manifest")
    manifest = json.loads((output / "probe_manifest.json").read_text())["probes"]
    expected_ids = {row["blind_id"] for row in manifest}
    if set(labels.get("labels", {})) != expected_ids:
        raise RuntimeError("every blinded probe must have exactly one label record")
    aspects = set(cfg["visual_review"]["aspects"])
    allowed = set(cfg["visual_review"]["labels"])
    for row in labels["labels"].values():
        if set(row) != aspects or not set(row.values()) <= allowed:
            raise RuntimeError("visual label schema differs from frozen contract")

    matrix, objective = objective_fidelity(output, labels)
    write_csv(output / "objective_fidelity_matrix.csv", matrix)
    collision_rows = read_csv(output / "collision_contract_audit.csv")
    mismatch = [row for row in collision_rows if row["classification"] in {
        "PROXY_STRICTER_THAN_NATIVE", "NATIVE_STRICTER_THAN_PROXY", "UNKNOWN"
    }]
    proxy_strict_lower = [row for row in mismatch if row["classification"] == "PROXY_STRICTER_THAN_NATIVE"
                          and (row["state"].startswith("wrist_z_") or row["state"] == "OLD_MINK_REFERENCE")]
    collision = {
        "certified": not mismatch,
        "mismatch_count": len(mismatch),
        "proxy_stricter_lower_wrist_count": len(proxy_strict_lower),
        "classifications": {name: sum(row["classification"] == name for row in collision_rows)
                            for name in ("PROXY_AND_NATIVE_AGREE_LEGAL", "PROXY_AND_NATIVE_AGREE_INTERFERENCE",
                                         "PROXY_STRICTER_THAN_NATIVE", "NATIVE_STRICTER_THAN_PROXY", "UNKNOWN")},
    }
    semantic = json.loads((output / "semantic_landmark_candidates.json").read_text())
    blockers = []
    if not semantic["certified"]:
        blockers.append("SEMANTIC_LANDMARK_BLOCKER")
    if not objective["certified"]:
        blockers.append("OBJECTIVE_FIDELITY_BLOCKER")
    if not collision["certified"]:
        blockers.append("COLLISION_PROXY_SEMANTICS_BLOCKER")
    if not blockers:
        classification = "OBJECTIVE_AND_GEOMETRY_CONTRACT_CERTIFIED"
    elif len(blockers) == 1:
        classification = blockers[0]
    else:
        classification = "MULTIPLE_UPSTREAM_BLOCKERS"

    report_lines = [
        "# Objective fidelity report", "",
        f"- Factorized objective contract certified: `{objective['certified']}`.",
        f"- Collision representation contract certified: `{collision['certified']}`.",
        "- No scalar objective was fitted to visual labels.",
        "- `global_E_bone` and `global_E_IM` remain diagnostic-only.", "",
        "## Component certification", "",
    ]
    for name, row in objective["component_certification"].items():
        report_lines.append(
            f"- `{name}`: {row['passed_comparisons']}/{row['applicable_unambiguous_comparisons']} "
            f"comparisons; certified `{row['certified_for_v2']}`."
        )
    report_lines.extend([
        "",
        "## Missing or failed required evidence",
        "",
        f"- No unambiguous visual evidence: `{objective['required_components_without_unambiguous_evidence']}`.",
        f"- Failed at least one applicable comparison: `{objective['required_components_failed']}`.",
    ])
    report_lines.extend(["", "## Collision representation", "",
                         f"- Mismatches/unknowns: `{collision['mismatch_count']}`.",
                         f"- Proxy-stricter lower-wrist findings: `{collision['proxy_stricter_lower_wrist_count']}`."])
    (output / "objective_fidelity_report.md").write_text("\n".join(report_lines) + "\n")
    write_json(output / "decision.json", {
        "classification": classification,
        "blockers": blockers,
        "semantic_landmarks_certified": semantic["certified"],
        "objective_components_certified": objective["certified"],
        "collision_contract_certified": collision["certified"],
        "collision_summary": collision,
        "authorization": cfg["authorization"],
        "retarget_v2_contract_written": classification == "OBJECTIVE_AND_GEOMETRY_CONTRACT_CERTIFIED",
    })
    if classification == "OBJECTIVE_AND_GEOMETRY_CONTRACT_CERTIFIED":
        proposed = {
            "schema": "taco_pour_retarget_v2_contract_proposal",
            "status": "PROPOSED_NOT_RUN",
            "baseline": "OLD_MINK_REFERENCE",
            "preserved_terms": ["fingertip_position", "fingertip_orientation", "wrist_orientation"],
            "added_certified_terms": [name for name, row in objective["component_certification"].items()
                                      if row["certified_for_v2"]],
            "formulation": "trust_region_residual_around_old_mink",
            "solver": "DEFERRED",
            "physics_authorized": False,
        }
        (output / "retarget_v2_contract.yaml").write_text(yaml.safe_dump(proposed, sort_keys=False))
    lines = [
        "# TACO Pour retarget contract fidelity audit v1", "",
        f"- Final classification: `{classification}`.",
        f"- Semantic landmark contract certified: `{semantic['certified']}`.",
        f"- Objective fidelity contract certified: `{objective['certified']}`.",
        f"- Collision proxy/native contract certified: `{collision['certified']}`.",
        f"- Blinded labels SHA-256: `{sha256(labels_path)}`.",
        "- No scalar weights were fitted and no retargeted candidate was generated.",
        "- Physics/control/planner/actor/critic/RL/promotion/chunk counts remain zero.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    status = json.loads((output / "visual_review_status.json").read_text())
    status.update(
        status="SEALED_AND_REVEALED",
        blinded_labels_sha256=sha256(labels_path),
        objective_metrics_must_remain_unread=False,
    )
    write_json(output / "visual_review_status.json", status)
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "server_artifacts.sha256")
    (output / "server_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "render", "finalize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "render":
        render(args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
