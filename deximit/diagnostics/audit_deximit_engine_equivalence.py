#!/usr/bin/env python3
"""Audit an isolated MuJoCo scene against a candidate-free SAPIEN probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_error(actual: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - target) / denominator)


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def joint_child_links(urdf: Path) -> dict[str, str]:
    root = ET.parse(urdf).getroot()
    result: dict[str, str] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and joint.get("name"):
            result[str(joint.get("name"))] = str(child.get("link"))
    return result


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        pose[3:], scalar_first=True,
    ).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def collision_geometry_errors(
    model: mujoco.MjModel, data: mujoco.MjData, urdf: Path,
    sapien_link_names: list[str], sapien_link_pose: np.ndarray,
) -> list[dict[str, object]]:
    """Compare actual compiled collision vertices in the same world frame."""
    link_index = {name: index for index, name in enumerate(sapien_link_names)}
    mesh_geoms: dict[str, list[int]] = {}
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        mesh_geoms.setdefault(str(mesh_name), []).append(geom_id)
    rows: list[dict[str, object]] = []
    for link in ET.parse(urdf).getroot().findall("link"):
        collision = link.find("collision")
        mesh_element = None if collision is None else collision.find("geometry/mesh")
        if collision is None or mesh_element is None:
            continue
        link_name = str(link.get("name"))
        mesh_path = (urdf.parent / str(mesh_element.get("filename"))).resolve(strict=True)
        mesh_key = mesh_path.stem.lower()
        candidates = list(dict.fromkeys(
            mesh_geoms.get(mesh_key, []) + mesh_geoms.get(link_name, [])
        ))
        if len(candidates) != 1 or link_name not in link_index:
            raise ValueError(
                f"cannot map URDF collision {link_name!r} to one MuJoCo geom",
            )
        geom_id = candidates[0]
        mesh_id = int(model.geom_dataid[geom_id])
        address = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        mujoco_vertices = np.asarray(
            model.mesh_vert[address : address + count], dtype=np.float64,
        )
        mujoco_world = (
            mujoco_vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
            + data.geom_xpos[geom_id]
        )
        source_mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise ValueError(f"URDF collision {mesh_path} is not one mesh")
        scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
        vertices = np.asarray(source_mesh.vertices, dtype=np.float64) * scale
        origin = collision.find("origin")
        xyz = np.fromstring(
            "0 0 0" if origin is None else origin.get("xyz", "0 0 0"), sep=" ",
        )
        rpy = np.fromstring(
            "0 0 0" if origin is None else origin.get("rpy", "0 0 0"), sep=" ",
        )
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        local[:3, 3] = xyz
        world = pose_matrix(sapien_link_pose[link_index[link_name]]) @ local
        sapien_world = vertices @ world[:3, :3].T + world[:3, 3]
        maximum = float(max(
            cKDTree(mujoco_world).query(sapien_world)[0].max(),
            cKDTree(sapien_world).query(mujoco_world)[0].max(),
        ))
        rows.append({
            "link": link_name,
            "sapien_source_vertex_count": len(vertices),
            "mujoco_compiled_vertex_count": len(mujoco_vertices),
            "symmetric_nearest_vertex_error_m_max": maximum,
        })
    return rows


def cooked_collision_geometry_errors(
    model: mujoco.MjModel, data: mujoco.MjData, cooked_report: Path,
    sapien_link_names: list[str], sapien_link_pose: np.ndarray,
) -> list[dict[str, object]]:
    """Compare the PhysX-cooked hulls that actually enter both solvers."""
    report = json.loads(cooked_report.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "deximit_sapien_cooked_collisions_v1_diagnostic_only"
        or report.get("diagnostic_only") is not True
        or report.get("formal_renderer_3_3_eligible") is not False
        or report.get("grasp_candidate_executed") is not False
        or not isinstance(report.get("shapes"), list)
    ):
        raise ValueError("SAPIEN cooked-collision report is not isolated evidence")
    link_index = {name: index for index, name in enumerate(sapien_link_names)}
    geom_by_mesh: dict[str, int] = {}
    for geom in range(model.ngeom):
        mesh = int(model.geom_dataid[geom])
        if mesh < 0:
            continue
        mesh_name = str(mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh,
        ))
        if mesh_name.startswith("physx_cooked_"):
            if mesh_name in geom_by_mesh:
                raise ValueError(f"cooked mesh {mesh_name!r} has multiple geoms")
            geom_by_mesh[mesh_name] = geom

    rows: list[dict[str, object]] = []
    for shape in report["shapes"]:
        owner = str(shape.get("owner", ""))
        if owner == "right_object":
            continue
        if owner not in link_index:
            raise ValueError(f"cooked collision owner {owner!r} lacks a SAPIEN link")
        mesh_name = f"physx_cooked_{owner}"
        if mesh_name not in geom_by_mesh:
            raise ValueError(f"exact scene lacks cooked collision {mesh_name!r}")
        geom = geom_by_mesh[mesh_name]
        mesh = int(model.geom_dataid[geom])
        address = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        mujoco_vertices = np.asarray(
            model.mesh_vert[address : address + count], dtype=np.float64,
        )
        mujoco_world = (
            mujoco_vertices @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )

        source_path = Path(str(shape.get("mesh", ""))).resolve(strict=True)
        if shape.get("mesh_sha256") != sha256(source_path):
            raise ValueError(f"cooked collision {owner!r} hash changed")
        source_mesh = trimesh.load(source_path, force="mesh", process=False)
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise ValueError(f"cooked collision {source_path} is not one mesh")
        scale = np.asarray(shape.get("scale"), dtype=np.float64)
        local_pose = np.asarray(shape.get("local_pose_wxyz"), dtype=np.float64)
        if scale.shape != (3,) or local_pose.shape != (7,):
            raise ValueError(f"cooked collision {owner!r} transform is malformed")
        source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64) * scale
        source_world_pose = (
            pose_matrix(sapien_link_pose[link_index[owner]])
            @ pose_matrix(local_pose)
        )
        source_world = (
            source_vertices @ source_world_pose[:3, :3].T
            + source_world_pose[:3, 3]
        )
        maximum = float(max(
            cKDTree(mujoco_world).query(source_world)[0].max(),
            cKDTree(source_world).query(mujoco_world)[0].max(),
        ))
        rows.append({
            "link": owner,
            "sapien_cooked_vertex_count": len(source_vertices),
            "mujoco_collision_vertex_count": len(mujoco_vertices),
            "symmetric_nearest_vertex_error_m_max": maximum,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--sapien-urdf", type=Path, required=True)
    parser.add_argument("--sapien-cooked-collisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-step-substeps", type=int, default=1)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    probe_path = args.sapien_probe.expanduser().resolve(strict=True)
    urdf = args.sapien_urdf.expanduser().resolve(strict=True)
    cooked_report = (
        args.sapien_cooked_collisions.expanduser().resolve(strict=True)
        if args.sapien_cooked_collisions is not None else None
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite equivalence audit {output}")
    if args.source_step_substeps <= 0:
        raise ValueError("source-step-substeps must be positive")

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    data = mujoco.MjData(model)
    probe = np.load(probe_path, allow_pickle=False)
    sapien_joint_names = [str(value) for value in probe["joint_names"]]
    sapien_link_names = [str(value) for value in probe["link_names"]]
    configurations = np.asarray(probe["configurations"], dtype=np.float64)
    force_response = np.asarray(probe["force_response"], dtype=np.float64)
    velocity_response = np.asarray(probe["velocity_response"], dtype=np.float64)
    position_response = np.asarray(probe["position_response"], dtype=np.float64)
    source_velocity_qpos = (
        np.asarray(probe["velocity_qpos_response"], dtype=np.float64)
        if "velocity_qpos_response" in probe else None
    )
    source_position_qpos = (
        np.asarray(probe["position_qpos_response"], dtype=np.float64)
        if "position_qpos_response" in probe else None
    )
    velocity_amplitude = np.broadcast_to(
        np.asarray(probe["velocity_probe_amplitude"], dtype=np.float64), (18,),
    )
    position_amplitude = np.broadcast_to(
        np.asarray(probe["position_probe_amplitude"], dtype=np.float64), (18,),
    )
    position_sign = np.asarray(probe["position_probe_sign"], dtype=np.float64)
    link_poses = np.asarray(probe["link_pose_world_wxyz"], dtype=np.float64)
    if configurations.shape != (7, 18) or force_response.shape != (7, 18, 18):
        raise ValueError("unexpected SAPIEN drive-probe dimensions")

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_ids: list[int] = []
    for name in sapien_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo scene lacks SAPIEN joint {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{name}",
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo scene lacks drive for SAPIEN joint {name!r}")
        actuator_ids.append(actuator_id)
    source_limits = np.asarray(probe["joint_limits"], dtype=np.float64)
    mujoco_limits = np.asarray([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in sapien_joint_names
    ])
    joint_limit_error = float(np.max(np.abs(source_limits - mujoco_limits)))

    child_by_joint = joint_child_links(urdf)
    sapien_link_index = {name: index for index, name in enumerate(sapien_link_names)}
    comparable_links: list[str] = []
    for joint in sapien_joint_names:
        link = child_by_joint[joint]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0 or link not in sapien_link_index:
            raise ValueError(f"cannot compare active child-link frame {link!r}")
        comparable_links.append(link)

    position_errors: list[float] = []
    angle_errors: list[float] = []
    force_errors: list[float] = []
    arm_force_errors: list[float] = []
    hand_force_errors: list[float] = []
    velocity_drive_errors: list[float] = []
    position_drive_errors: list[float] = []
    arm_velocity_qpos_errors: list[float] = []
    arm_position_qpos_errors: list[float] = []
    rows: list[dict[str, float | int]] = []
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    for config_index, configuration in enumerate(configurations):
        mujoco.mj_resetData(model, data)
        for address, value in zip(qpos_addresses, configuration, strict=True):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)

        config_position: list[float] = []
        config_angle: list[float] = []
        for link in comparable_links:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            target = link_poses[config_index, sapien_link_index[link]]
            config_position.append(float(np.linalg.norm(data.xpos[body_id] - target[:3])))
            config_angle.append(quaternion_angle(data.xquat[body_id], target[3:]))
        position_errors.extend(config_position)
        angle_errors.extend(config_angle)

        mujoco.mj_fullM(model, full_mass, data.qM)
        robot_mass = full_mass[np.ix_(dof_addresses, dof_addresses)]
        mujoco_response = np.linalg.inv(robot_mass)
        target_response = force_response[config_index]
        all_error = relative_error(mujoco_response, target_response)
        arm_error = relative_error(mujoco_response[:6, :6], target_response[:6, :6])
        hand_error = relative_error(mujoco_response[6:, 6:], target_response[6:, 6:])
        force_errors.append(all_error)
        arm_force_errors.append(arm_error)
        hand_force_errors.append(hand_error)

        measured_velocity = np.empty((18, 18), dtype=np.float64)
        measured_position = np.empty((18, 18), dtype=np.float64)
        measured_velocity_qpos = np.empty((18, 18), dtype=np.float64)
        measured_position_qpos = np.empty((18, 18), dtype=np.float64)
        for axis in range(18):
            mujoco.mj_resetData(model, data)
            for address, value in zip(qpos_addresses, configuration, strict=True):
                data.qpos[address] = value
            data.qvel[dof_addresses[axis]] = velocity_amplitude[axis]
            for actuator_id, value in zip(actuator_ids, configuration, strict=True):
                data.ctrl[actuator_id] = value
            base_qpos = data.qpos[qpos_addresses].copy()
            for _ in range(args.source_step_substeps):
                mujoco.mj_step(model, data)
            measured_velocity[:, axis] = (
                data.qvel[dof_addresses] / velocity_amplitude[axis]
            )
            measured_velocity_qpos[:, axis] = (
                (data.qpos[qpos_addresses] - base_qpos) / velocity_amplitude[axis]
            )

            mujoco.mj_resetData(model, data)
            for address, value in zip(qpos_addresses, configuration, strict=True):
                data.qpos[address] = value
            target = configuration.copy()
            target[axis] += (
                position_sign[config_index, axis] * position_amplitude[axis]
            )
            for actuator_id, value in zip(actuator_ids, target, strict=True):
                data.ctrl[actuator_id] = value
            base_qpos = data.qpos[qpos_addresses].copy()
            for _ in range(args.source_step_substeps):
                mujoco.mj_step(model, data)
            measured_position[:, axis] = data.qvel[dof_addresses] / (
                position_sign[config_index, axis] * position_amplitude[axis]
            )
            measured_position_qpos[:, axis] = (
                (data.qpos[qpos_addresses] - base_qpos)
                / (position_sign[config_index, axis] * position_amplitude[axis])
            )
        velocity_error = relative_error(
            measured_velocity, velocity_response[config_index],
        )
        position_error = relative_error(
            measured_position, position_response[config_index],
        )
        velocity_drive_errors.append(velocity_error)
        position_drive_errors.append(position_error)
        arm_velocity_qpos_error = None
        arm_position_qpos_error = None
        if source_velocity_qpos is not None and source_position_qpos is not None:
            arm_velocity_qpos_error = relative_error(
                measured_velocity_qpos[:6, :6],
                source_velocity_qpos[config_index, :6, :6],
            )
            arm_position_qpos_error = relative_error(
                measured_position_qpos[:6, :6],
                source_position_qpos[config_index, :6, :6],
            )
            arm_velocity_qpos_errors.append(arm_velocity_qpos_error)
            arm_position_qpos_errors.append(arm_position_qpos_error)
        rows.append({
            "configuration": config_index,
            "link_position_error_m_max": max(config_position),
            "link_angle_error_rad_max": max(config_angle),
            "inverse_mass_response_relative_error": all_error,
            "arm_inverse_mass_response_relative_error": arm_error,
            "hand_inverse_mass_response_relative_error": hand_error,
            "velocity_drive_response_relative_error": velocity_error,
            "position_drive_response_relative_error": position_error,
            "arm_velocity_qpos_response_relative_error": arm_velocity_qpos_error,
            "arm_position_qpos_response_relative_error": arm_position_qpos_error,
        })

    report = {
        "schema": "deximit_sapien_mujoco_equivalence_audit_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "sapien_probe": str(probe_path),
        "sapien_probe_sha256": sha256(probe_path),
        "sapien_urdf": str(urdf),
        "sapien_urdf_sha256": sha256(urdf),
        "configuration_count": len(configurations),
        "mujoco_substeps_per_sapien_step": args.source_step_substeps,
        "compared_active_link_count": len(comparable_links),
        "active_links": comparable_links,
        "joint_limit_error_rad_max": joint_limit_error,
        "link_position_error_m_max": max(position_errors),
        "link_position_error_m_rms": float(np.sqrt(np.mean(np.square(position_errors)))),
        "link_angle_error_rad_max": max(angle_errors),
        "link_angle_error_rad_rms": float(np.sqrt(np.mean(np.square(angle_errors)))),
        "inverse_mass_response_relative_error_mean": float(np.mean(force_errors)),
        "inverse_mass_response_relative_error_max": max(force_errors),
        "arm_inverse_mass_response_relative_error_mean": float(np.mean(arm_force_errors)),
        "hand_inverse_mass_response_relative_error_mean": float(np.mean(hand_force_errors)),
        "velocity_drive_response_relative_error_mean": float(np.mean(velocity_drive_errors)),
        "velocity_drive_response_relative_error_max": max(velocity_drive_errors),
        "position_drive_response_relative_error_mean": float(np.mean(position_drive_errors)),
        "position_drive_response_relative_error_max": max(position_drive_errors),
        "arm_velocity_qpos_response_relative_error_mean": (
            None if not arm_velocity_qpos_errors
            else float(np.mean(arm_velocity_qpos_errors))
        ),
        "arm_position_qpos_response_relative_error_mean": (
            None if not arm_position_qpos_errors
            else float(np.mean(arm_position_qpos_errors))
        ),
        "per_configuration": rows,
    }
    mujoco.mj_resetData(model, data)
    for address, value in zip(qpos_addresses, configurations[0], strict=True):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    visual_rows = collision_geometry_errors(
        model, data, urdf, sapien_link_names, link_poses[0],
    )
    report["visual_geometry_not_used_for_contact"] = {
        "compared_mesh_count": len(visual_rows),
        "symmetric_nearest_vertex_error_m_max": max(
            float(row["symmetric_nearest_vertex_error_m_max"])
            for row in visual_rows
        ),
        "per_link": visual_rows,
    }
    if cooked_report is not None:
        collision_rows = cooked_collision_geometry_errors(
            model, data, cooked_report, sapien_link_names, link_poses[0],
        )
        report["collision_geometry"] = {
            "source": str(cooked_report),
            "source_sha256": sha256(cooked_report),
            "compared_collision_count": len(collision_rows),
            "symmetric_nearest_vertex_error_m_max": max(
                float(row["symmetric_nearest_vertex_error_m_max"])
                for row in collision_rows
            ),
            "per_link": collision_rows,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
