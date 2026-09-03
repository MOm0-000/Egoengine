#!/usr/bin/env python3
"""Calibrate opposing-contact pinch/lift behavior without a grasp candidate.

The profile search covers both friction-solver settings and the two contact
responses independently calibrated from object-only impact/slide probes.  A
moving opposed pinch is the candidate-free test that reveals which contact
response remains accurate when normal loads come from two moving surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SOURCE_SCHEMA = "deximit_sapien_pinch_probe_v1_diagnostic_only"
SCENE_SCHEMA = (
    "deximit_exact_urdf_mujoco_scene_v9_bound_friction_drive_limits_"
    "diagnostic_only"
)
SUBSTEPS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(array: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.asarray(array).reshape(-1))


def rotation_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    error = (
        Rotation.from_quat(actual, scalar_first=True)
        * Rotation.from_quat(expected, scalar_first=True).inv()
    ).magnitude()
    return float(np.sqrt(np.mean(np.square(error))))


def add_paddles(
    source_scene: Path, output_scene: Path, half_size: np.ndarray,
    initial_position: np.ndarray, quaternion: np.ndarray,
) -> None:
    tree = ET.parse(source_scene)
    root = tree.getroot()
    world = root.find("worldbody")
    contact = root.find("contact")
    if world is None or contact is None:
        raise ValueError("exact scene lacks worldbody or explicit contacts")
    table_pair = contact.find("pair[@name='table_right_object']")
    if table_pair is None:
        raise ValueError("exact scene lacks calibrated table/object pair")
    for index, name in enumerate(("left_probe_pad", "right_probe_pad")):
        body = ET.SubElement(world, "body", {
            "name": name, "pos": values(initial_position[index]),
            "quat": values(quaternion), "gravcomp": "1",
        })
        ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})
        ET.SubElement(body, "inertial", {
            "pos": "0 0 0", "mass": "1000000",
            "diaginertia": "1000000 1000000 1000000",
        })
        ET.SubElement(body, "geom", {
            "name": f"{name}_geom", "type": "box", "size": values(half_size),
            "contype": "0", "conaffinity": "0", "rgba": "0.2 0.7 0.9 0.5",
        })
        attributes = {
            "name": f"{name}_right_object",
            "geom1": f"{name}_geom", "geom2": "right_object_single",
            "condim": "3", "friction": "0.5 0.5 0 0 0",
        }
        for key in ("solref", "solimp", "margin", "gap"):
            if key in table_pair.attrib:
                attributes[key] = table_pair.attrib[key]
        ET.SubElement(contact, "pair", attributes)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_scene, encoding="unicode", xml_declaration=False)


def body_point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: np.ndarray,
) -> np.ndarray:
    spatial = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0,
    )
    return spatial[3:] + np.cross(spatial[:3], point - data.xpos[body])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    probe = args.sapien_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pinch audit {output}")
    source_report = json.loads(
        probe.with_name("report.json").read_text(encoding="utf-8"),
    )
    provenance_path = scene.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        source_report.get("schema") != SOURCE_SCHEMA
        or source_report.get("strict_gate", {}).get("passed") is not True
        or source_report.get("grasp_candidate_executed") is not False
        or provenance.get("schema") != SCENE_SCHEMA
        or provenance.get("diagnostic_only") is not True
        or provenance.get("formal_renderer_3_3_eligible") is not False
    ):
        raise ValueError("pinch inputs violate their isolated diagnostic contracts")
    with np.load(probe, allow_pickle=False) as source:
        if (
            str(np.asarray(source["schema"]).item()) != SOURCE_SCHEMA
            or not bool(source["diagnostic_only"])
            or bool(source["formal_renderer_3_3_eligible"])
            or bool(source["grasp_candidate_executed"])
        ):
            raise ValueError("SAPIEN pinch trace is not diagnostic-only")
        dt = float(source["physics_dt_s"])
        phase = np.asarray(source["phase"]).astype("U16")
        source_pose = np.asarray(source["object_pose_world_wxyz"], dtype=np.float64)
        source_linear = np.asarray(source["object_linear_velocity"], dtype=np.float64)
        source_angular = np.asarray(source["object_angular_velocity"], dtype=np.float64)
        rest = np.asarray(source["object_rest_position"], dtype=np.float64)
        pad_position = np.asarray(source["pad_position"], dtype=np.float64)
    count = len(phase)
    if (
        not np.isclose(dt, 1.0 / 240.0) or source_pose.shape != (count, 7)
        or pad_position.shape != (count, 2, 3)
    ):
        raise ValueError("SAPIEN pinch timing or arrays are malformed")
    half_size = np.asarray(source_report["paddle_half_size_m"], dtype=np.float64)
    summary_path = Path(source_report["source_summary"]).resolve(strict=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = int(source_report["candidate_used_only_for_recorded_object_properties"])
    rows = [
        row for row in summary["original_sapien_passes"]
        if int(row["source_candidate_index"]) == candidate
    ]
    if len(rows) != 1:
        raise ValueError("pinch source summary lacks one physical-properties row")
    initial_transform = np.asarray(rows[0]["metrics"]["object_initial_pose"])
    pad_quaternion = Rotation.from_matrix(
        initial_transform[:3, :3],
    ).as_quat(scalar_first=True)
    initial_pad = pad_position[0] - (pad_position[1] - pad_position[0])

    output_scene = output / "pinch_scene.xml"
    add_paddles(scene, output_scene, half_size, initial_pad, pad_quaternion)
    base_model = mujoco.MjModel.from_xml_path(str(output_scene))
    if base_model.nq != 39 or base_model.nv != 36:
        raise RuntimeError("generated pinch scene has unexpected state dimensions")

    table_threshold = float(
        provenance["contact_calibration"]["table_static_speed_threshold_m_s"],
    )
    contact_profiles = (
        ("object_probe_v6", 0.00045, 16.0, 0.8),
        ("object_probe_v7", 0.00035, 16.0, 0.95),
        ("hand_grid_d4", 0.00035, 4.0, 0.95),
    )
    profiles = [
        (
            contact_name, contact_time, contact_damping, contact_impedance,
            cone, noslip, threshold,
        )
        for (
            contact_name, contact_time, contact_damping, contact_impedance,
        ) in contact_profiles
        for cone in ("pyramidal", "elliptic")
        for noslip in (0, 1, 2, 5, 10)
        for threshold in (0.001, 0.005, 0.01, 0.02)
    ]
    traces: dict[tuple[str, str, int, float], dict[str, np.ndarray]] = {}
    reports: list[dict[str, object]] = []
    for (
        contact_name, contact_time, contact_damping, contact_impedance,
        cone, noslip, threshold,
    ) in profiles:
        model = mujoco.MjModel.from_xml_path(str(output_scene))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        model.opt.cone = (
            mujoco.mjtCone.mjCONE_PYRAMIDAL
            if cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
        )
        model.opt.noslip_iterations = noslip
        data = mujoco.MjData(model)
        object_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
        )
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
        )
        object_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
        )
        table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        object_qpos = int(model.jnt_qposadr[object_joint])
        object_dof = int(model.jnt_dofadr[object_joint])
        pad_joint = []
        pad_dof = []
        pad_body = []
        pad_geom = []
        pad_pair = []
        for name in ("left_probe_pad", "right_probe_pad"):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint",
            )
            pad_joint.append(int(model.jnt_qposadr[joint]))
            pad_dof.append(int(model.jnt_dofadr[joint]))
            pad_body.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            pad_geom.append(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom",
            ))
            pad_pair.append(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_PAIR, f"{name}_right_object",
            ))
        table_pair = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_PAIR, "table_right_object",
        )
        allowed_pairs = {table_pair, *pad_pair}
        for pair in range(model.npair):
            if pair not in allowed_pairs:
                model.pair_margin[pair] = -1.0
                model.pair_gap[pair] = 0.0
        for pair in allowed_pairs:
            model.pair_solref[pair] = (contact_time, contact_damping)
            model.pair_solimp[pair] = (
                contact_impedance, contact_impedance, 0.001, 0.5, 2.0,
            )
        mujoco.mj_resetData(model, data)
        data.qpos[object_qpos : object_qpos + 3] = rest
        data.qpos[object_qpos + 3 : object_qpos + 7] = source_pose[0, 3:]
        for index in range(2):
            address = pad_joint[index]
            data.qpos[address : address + 3] = initial_pad[index]
            data.qpos[address + 3 : address + 7] = pad_quaternion
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pose = np.empty_like(source_pose)
        linear = np.empty_like(source_linear)
        angular = np.empty_like(source_angular)
        pad_impulse = np.zeros((count, 2), dtype=np.float64)
        minimum_gap = np.full(count, np.inf, dtype=np.float64)
        previous_pad = initial_pad.copy()
        for frame in range(count):
            velocity = (pad_position[frame] - previous_pad) / dt
            data.qvel[object_dof : object_dof + 6] *= 1.0 - 20.0 * dt
            for substep in range(SUBSTEPS):
                fraction = (substep + 1) / SUBSTEPS
                current_pad = previous_pad + fraction * (
                    pad_position[frame] - previous_pad
                )
                for index in range(2):
                    address = pad_joint[index]
                    dof = pad_dof[index]
                    data.qpos[address : address + 3] = current_pad[index]
                    data.qpos[address + 3 : address + 7] = pad_quaternion
                    data.qvel[dof : dof + 3] = velocity[index]
                    data.qvel[dof + 3 : dof + 6] = 0.0
                for pair in pad_pair:
                    model.pair_friction[pair, :2] = 0.5
                model.pair_friction[table_pair, :2] = 0.75
                mujoco.mj_forward(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    if object_geom not in geoms:
                        continue
                    other_geom = next(iter(geoms - {object_geom}))
                    if other_geom not in {*pad_geom, table_geom}:
                        continue
                    other_body = int(model.geom_bodyid[other_geom])
                    relative = body_point_velocity(
                        model, data, object_body, contact.pos,
                    ) - body_point_velocity(model, data, other_body, contact.pos)
                    normal = np.asarray(contact.frame[:3], dtype=np.float64)
                    tangential = relative - float(relative @ normal) * normal
                    speed = float(np.linalg.norm(tangential))
                    if other_geom == table_geom:
                        friction = 0.85 if speed <= table_threshold else 0.75
                        model.pair_friction[table_pair, :2] = friction
                    else:
                        friction = 0.7 if speed <= threshold else 0.5
                        model.pair_friction[
                            pad_pair[pad_geom.index(other_geom)], :2
                        ] = friction
                mujoco.mj_forward(model, data)
                data.qvel[:] += data.qacc * float(model.opt.timestep)
                mujoco.mj_integratePos(
                    model, data.qpos, data.qvel, float(model.opt.timestep),
                )
                mujoco.mj_normalizeQuat(model, data.qpos)
                data.time += float(model.opt.timestep)
                mujoco.mj_forward(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    if object_geom not in geoms:
                        continue
                    minimum_gap[frame] = min(
                        minimum_gap[frame], float(contact.dist),
                    )
                    for index, geom in enumerate(pad_geom):
                        if geom not in geoms or int(contact.efc_address) < 0:
                            continue
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            model, data, contact_index, force,
                        )
                        pad_impulse[frame, index] += (
                            float(np.linalg.norm(force[:3])) * model.opt.timestep
                        )
            previous_pad = pad_position[frame]
            pose[frame] = data.qpos[object_qpos : object_qpos + 7]
            linear[frame] = data.qvel[object_dof : object_dof + 3]
            angular[frame] = data.qvel[object_dof + 3 : object_dof + 6]
        hold = np.flatnonzero(phase == "hold")
        lift = pose[:, 2] - rest[2]
        opposed = (pad_impulse[:, 0] > 1.0e-8) & (pad_impulse[:, 1] > 1.0e-8)
        strict = bool(
            np.any(opposed) and np.all(opposed[hold])
            and np.min(lift[hold]) >= 0.06
            and np.ptp(pose[hold, 2]) <= 0.005
            and np.min(minimum_gap) >= -0.002
        )
        position_rmse = float(np.sqrt(np.mean(np.square(
            pose[:, :3] - source_pose[:, :3],
        ))))
        angle_rmse = rotation_rmse(pose[:, 3:], source_pose[:, 3:])
        report = {
            "contact_profile": contact_name,
            "contact_time_constant_s": contact_time,
            "contact_damping_ratio": contact_damping,
            "contact_impedance": contact_impedance,
            "cone": cone, "noslip_iterations": noslip,
            "hand_static_speed_threshold_m_s": threshold,
            "strict_gate_passed": strict,
            "opposed_contact_all_hold_steps": bool(np.all(opposed[hold])),
            "hold_minimum_lift_m": float(np.min(lift[hold])),
            "hold_vertical_span_m": float(np.ptp(pose[hold, 2])),
            "minimum_paddle_object_gap_m": float(np.min(minimum_gap)),
            "object_position_rmse_m": position_rmse,
            "object_rotation_rmse_rad": angle_rmse,
            "score": position_rmse / 0.005 + angle_rmse / 0.1,
        }
        reports.append(report)
        traces[(contact_name, cone, noslip, threshold)] = {
            "pose": pose, "linear": linear, "angular": angular,
            "pad_impulse": pad_impulse, "gap": minimum_gap,
        }
    reports.sort(key=lambda row: (not row["strict_gate_passed"], row["score"]))
    selected = reports[0]
    key = (
        str(selected["contact_profile"]),
        str(selected["cone"]), int(selected["noslip_iterations"]),
        float(selected["hand_static_speed_threshold_m_s"]),
    )
    trace_values = traces[key]
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.npz"
    np.savez_compressed(
        trace_path,
        schema=np.asarray("deximit_mujoco_pinch_audit_v3_hand_contact_diagnostic_only"),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), source_probe_sha256=np.asarray(sha256(probe)),
        phase=phase, object_pose_world_wxyz=trace_values["pose"],
        object_linear_velocity=trace_values["linear"],
        object_angular_velocity=trace_values["angular"],
        pad_impulse_norm_sum_ns=trace_values["pad_impulse"],
        paddle_object_gap_m=trace_values["gap"],
    )
    payload = {
        "schema": "deximit_mujoco_pinch_audit_v3_hand_contact_diagnostic_only",
        "diagnostic_only": True, "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "generated_pinch_scene": str(output_scene),
        "generated_pinch_scene_sha256": sha256(output_scene),
        "sapien_probe": str(probe), "sapien_probe_sha256": sha256(probe),
        "tested_profile_count": len(reports),
        "selected_profile": selected,
        "any_strict_pass": any(bool(row["strict_gate_passed"]) for row in reports),
        "all_profiles": reports,
        "trace": str(trace_path), "trace_sha256": sha256(trace_path),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["any_strict_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
