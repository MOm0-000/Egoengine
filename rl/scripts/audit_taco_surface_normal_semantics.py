"""Read-only MANO/XHand fingertip surface-normal semantics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src"), str(ROOT / "scripts")]

from audit_taco_initialization import visual_meshes  # noqa: E402
from audit_taco_orientation_prerequisites import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_CANDIDATE,
    DEFAULT_HANDS,
    DEFAULT_HUMAN,
    DEFAULT_MANO,
    DEFAULT_SCENE,
    DEFAULT_VENDOR,
    FINGERS,
    SIDES,
    _current_anatomical_calibrations,
    _unit,
    artifact,
)
from egoengine_repro.evaluation.taco_surface import (  # noqa: E402
    MANO_TIP_VERTICES,
    reconstruct_taco_mano,
)


DEFAULT_OUTPUT = ROOT / "runs/taco_pour_surface_normal_audit_v1"


def _axis_angles(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(actual * target, axis=-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def _distribution(values: np.ndarray) -> dict:
    return {
        "mean_deg": float(np.mean(values)),
        "median_deg": float(np.median(values)),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(np.max(values)),
    }


def _normal_stability(normals: np.ndarray, frames: np.ndarray) -> dict:
    local = np.einsum("tji,tj->ti", frames, normals)
    mean = _unit(local.mean(axis=0))
    angles = _axis_angles(local, mean[None])
    return {
        "mean_local_direction": mean.tolist(),
        "mean_deviation_deg": float(angles.mean()),
        "p95_deviation_deg": float(np.percentile(angles, 95)),
        "max_deviation_deg": float(angles.max()),
    }


def _load_robot_surface_normals(model: mujoco.MjModel, scene: Path,
                                vendor: Path) -> dict[str, np.ndarray]:
    """Find one native distal mesh face normal near each URDF tip endpoint."""
    meshes, _ = visual_meshes(scene, model)
    result = {}
    for side in SIDES:
        urdf = ET.parse(vendor / f"xhand_{side}.urdf").getroot()
        for finger in FINGERS:
            site_id = model.site(f"{side}_{finger}_tip").id
            body_id = int(model.site_bodyid[site_id])
            candidates = [mesh for gid, mesh in meshes.items()
                          if int(model.geom_bodyid[gid]) == body_id]
            if len(candidates) != 1:
                raise ValueError(f"expected one native distal mesh for {side}_{finger}")
            fixed = [
                joint for joint in urdf.findall("joint")
                if joint.get("type") == "fixed"
                and joint.find("parent").get("link") == model.body(body_id).name
                and joint.find("child").get("link").endswith("tip")
            ]
            if len(fixed) != 1:
                raise ValueError(f"expected one URDF endpoint for {side}_{finger}")
            endpoint = np.fromstring(fixed[0].find("origin").get("xyz"), sep=" ")
            mesh = candidates[0]
            _, distance, face = trimesh.proximity.closest_point(mesh, endpoint[None])
            result[f"{side}_{finger}"] = {
                "body_id": body_id,
                "endpoint_local_m": endpoint.tolist(),
                "nearest_surface_distance_m": float(distance[0]),
                "normal_local": _unit(mesh.face_normals[int(face[0])]).tolist(),
                "mesh_watertight": bool(mesh.is_watertight),
            }
    return result


def _human_surface_normals(hands: Path, mano_models: Path) -> tuple[dict, dict]:
    normals = {}
    source = {}
    for side in SIDES:
        pose = hands / f"{side}_hand.pkl"
        shape = hands / f"{side}_hand_shape.pkl"
        model_path = mano_models / f"MANO_{side.upper()}.pkl"
        vertices, joints, faces, _, _ = reconstruct_taco_mano(
            pose, shape, model_path, side=side,
        )
        tip_vertices = MANO_TIP_VERTICES[side]
        values = np.empty((len(vertices), 5, 3), dtype=float)
        for frame, vertex in enumerate(vertices):
            mesh = trimesh.Trimesh(vertices=vertex, faces=faces, process=False)
            values[frame] = mesh.vertex_normals[tip_vertices]
        normals[side] = values
        source[side] = {
            "tip_vertex_indices": [int(v) for v in tip_vertices],
            "vertex_normals_are_mesh_vertex_normals": True,
            "reconstructed_joint_rows": int(len(joints)),
            "faces": int(len(faces)),
        }
    return normals, source


def _trajectory_normals(model: mujoco.MjModel, qpos: np.ndarray,
                        human: dict, calibrations: np.ndarray,
                        robot_surface: dict, human_surface: dict,
                        hand: int, finger: int) -> dict:
    side = SIDES[hand]
    key = f"{side}_{FINGERS[finger]}"
    data = mujoco.MjData(model)
    site_id = model.site(f"{side}_{FINGERS[finger]}_tip").id
    body_id = int(model.site_bodyid[site_id])
    target_frame = (
        human["T_sim_fingertip_target"][:, hand, finger, :3, :3]
        @ calibrations[hand, finger]
    )
    human_normal = human_surface[side]
    sim_rotation = human["T_sim_world"][:3, :3]
    human_normal = np.einsum("ij,tj->ti", sim_rotation, human_normal[:, finger])
    robot_normal = np.empty_like(human_normal)
    robot_frame = np.empty_like(target_frame)
    for row, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_rotation = data.xmat[body_id].reshape(3, 3)
        robot_normal[row] = body_rotation @ np.asarray(robot_surface[key]["normal_local"])
        robot_frame[row] = data.site_xmat[site_id].reshape(3, 3) @ calibrations[hand, finger]
    normal_error = _axis_angles(robot_normal, human_normal)
    human_axis = {
        name: _axis_angles(human_normal, target_frame[:, axis])
        for axis, name in enumerate(("x", "y", "z"))
    }
    robot_axis = {
        name: _axis_angles(robot_normal, robot_frame[:, axis])
        for axis, name in enumerate(("x", "y", "z"))
    }
    return {
        "human_tip_surface_normal_vs_target_anatomical_axes_deg": {
            key: _distribution(value) for key, value in human_axis.items()
        },
        "robot_native_surface_normal_vs_actual_anatomical_axes_deg": {
            key: _distribution(value) for key, value in robot_axis.items()
        },
        "human_surface_normal_stability_in_target_frame": _normal_stability(
            human_normal, target_frame,
        ),
        "robot_surface_normal_stability_in_actual_frame": _normal_stability(
            robot_normal, robot_frame,
        ),
        "human_vs_robot_surface_normal_error_deg": _distribution(normal_error),
        "normal_convention_is_diagnostic_only": True,
    }


def run(output: Path, human_path: Path, baseline_path: Path,
        candidate_path: Path, scene: Path, hands: Path, mano_models: Path,
        vendor: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with np.load(human_path, allow_pickle=False) as source:
        human = {key: np.asarray(source[key]) for key in source.files}
    with np.load(baseline_path, allow_pickle=False) as source:
        baseline = {key: np.asarray(source[key]) for key in source.files}
    with np.load(candidate_path, allow_pickle=False) as source:
        candidate = {key: np.asarray(source[key]) for key in source.files}
    if len(human["frame_indices"]) != len(baseline["qpos"]):
        raise ValueError("human and baseline frame counts differ")
    if not np.array_equal(baseline["frame_indices"], candidate["frame_indices"]):
        raise ValueError("baseline and candidate frame rows differ")
    model = mujoco.MjModel.from_xml_path(str(scene))
    calibrations = _current_anatomical_calibrations(model)
    robot_surface = _load_robot_surface_normals(model, scene, vendor)
    human_surface, human_source = _human_surface_normals(hands, mano_models)
    metrics = {"baseline_current_mink": {}, "candidate_spider_style": {}}
    for label, robot in (("baseline_current_mink", baseline),
                         ("candidate_spider_style", candidate)):
        for hand, side in enumerate(SIDES):
            metrics[label][side] = {}
            for finger, _ in enumerate(FINGERS):
                metrics[label][side][FINGERS[finger]] = _trajectory_normals(
                    model, robot["qpos"], human, calibrations, robot_surface,
                    human_surface, hand, finger,
                )
    inputs = [human_path, baseline_path, candidate_path, scene]
    inputs += [hands / f"{side}_hand.pkl" for side in SIDES]
    inputs += [hands / f"{side}_hand_shape.pkl" for side in SIDES]
    inputs += [mano_models / f"MANO_{side.upper()}.pkl" for side in SIDES]
    inputs += [vendor / f"xhand_{side}.urdf" for side in SIDES]
    report = {
        "status": "surface_normal_semantics_audit_not_a_mapping_change",
        "inputs": [artifact(path) for path in inputs],
        "code": [artifact(Path(__file__)), artifact(ROOT / "src/egoengine_repro/retarget/taco_bimanual.py")],
        "human_surface_source": human_source,
        "robot_surface_source": robot_surface,
        "metrics": metrics,
        "limitations": [
            "MANO vertex normals and nearest XHand mesh face normals are deterministic surface conventions, not proof of the author's pad/nail frame.",
            "The human tip is a skinned surface vertex, so its normal need not be rigidly attached to a distal joint.",
            "Open native meshes can have local normal-orientation ambiguity; nearest-face normals are not a contact certificate.",
            "No target, qpos, site, scene, cost, or physical state was modified.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--mano-models", type=Path, default=DEFAULT_MANO)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    args = parser.parse_args()
    report = run(
        args.output, args.human, args.baseline, args.candidate, args.scene,
        args.hands, args.mano_models, args.vendor,
    )
    for label, hands in report["metrics"].items():
        print(label)
        for side, fingers in hands.items():
            print(side, {
                finger: values["human_vs_robot_surface_normal_error_deg"]["mean_deg"]
                for finger, values in fingers.items()
            })


if __name__ == "__main__":
    main()
