#!/usr/bin/env python3
"""Test one full-frame MuJoCo solve against recorded SAPIEN contact motion.

This is a candidate-free attribution audit.  Each sample is reset to one
recorded SAPIEN state and predicts only the next 1/240 s state.  It tests
whether treating 25 PhysX TGS position iterations as 25 independent MuJoCo
physics steps is the remaining articulation/contact coupling error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "deximit_frame_solver_audit_v1_diagnostic_only"
TRACE_SCHEMAS = {
    "deximit_sapien_hand_trace_v6_pair_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v7_link_contact_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only",
}
DT = 1.0 / 240.0
FINGERS = ("thumb", "index", "mid", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def vector_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    return {
        "rmse": rms(actual - expected),
        "actual_norm_sum": float(np.sum(np.linalg.norm(actual, axis=1))),
        "expected_norm_sum": float(np.sum(np.linalg.norm(expected, axis=1))),
        "norm_sum_ratio": (
            None if np.sum(np.linalg.norm(expected, axis=1)) <= 1.0e-12
            else float(
                np.sum(np.linalg.norm(actual, axis=1))
                / np.sum(np.linalg.norm(expected, axis=1))
            )
        ),
    }


def rotation_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(actual, scalar_first=True)
        * Rotation.from_quat(expected, scalar_first=True).inv()
    ).magnitude()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--home-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    trace_path = args.sapien_trace.expanduser().resolve(strict=True)
    summary_path = args.sapien_summary.expanduser().resolve(strict=True)
    probe_path = args.home_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frame audit {output}")

    with np.load(probe_path, allow_pickle=False) as values:
        names = [str(value) for value in values["joint_names"]]
    with np.load(trace_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) not in TRACE_SCHEMAS
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or str(np.asarray(values["sample_semantics"]).item())
            != "post-step state under same-index drive target"
        ):
            raise ValueError("input is not an isolated SAPIEN trace")
        phase = np.asarray(values["phase"]).astype("U32")
        pose = np.asarray(values["object_pose_sapien_wxyz"], dtype=np.float64)
        linear = np.asarray(values["object_linear_velocity_sapien"], dtype=np.float64)
        angular = np.asarray(values["object_angular_velocity_sapien"], dtype=np.float64)
        robot = np.asarray(values["right_robot_qpos_sapien"], dtype=np.float64)
        robot_velocity = np.asarray(
            values["right_robot_qvel_sapien"], dtype=np.float64,
        )
        target = np.asarray(values["right_drive_target_sapien"], dtype=np.float64)
        channels = [str(value) for value in values["contact_channel_order"]]
        source_load = np.asarray(values["contact_channel_load_bearing"], dtype=bool)
        source_normal = np.asarray(
            values["contact_channel_normal_impulse_net_on_object_ns"],
            dtype=np.float64,
        )
        candidate = int(np.asarray(values["candidate_index"]).item())
        depth = int(np.asarray(values["pool_depth"]).item())
    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError("home probe does not define 18 unique source joints")
    finger_indices = [channels.index(name) for name in FINGERS]
    any_load = source_load[:, finger_indices].any(axis=1)
    available = np.flatnonzero(any_load & (np.arange(len(phase)) > 0))
    if len(available) < 100:
        raise ValueError("trace lacks loaded contact samples")
    early = available[available <= available[0] + 40]
    late = available[available > early[-1]]
    spread = late[np.unique(np.linspace(0, len(late) - 1, 120).round().astype(int))]
    rows = np.unique(np.r_[early, spread])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [
        row for row in summary.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == candidate
        and int(row.get("depth", -1)) == depth
    ]
    if len(matches) != 1:
        raise ValueError("summary is not bound to the source candidate")
    metrics = matches[0]["metrics"]
    mass = float(metrics["object_mass_kg"])
    linear_decay = 1.0 - float(metrics["object_linear_damping"]) * DT
    angular_decay = 1.0 - float(metrics["object_angular_damping"]) * DT

    profiles = (
        {
            "name": "full_frame_native_pd_refsafe",
            "disable_refsafe": False, "contact": "source_positive",
            "activation_clearance_m": 0.0,
        },
        {
            "name": "full_frame_native_pd_source_contact",
            "disable_refsafe": True, "contact": "source_positive",
            "activation_clearance_m": 0.0,
        },
        {
            "name": "full_frame_native_pd_critical_safe",
            "disable_refsafe": False, "contact": "critical_safe",
            "activation_clearance_m": 0.0,
        },
        {
            "name": "full_frame_native_pd_source_contact_predictive_025mm",
            "disable_refsafe": True, "contact": "source_positive",
            "activation_clearance_m": 0.00025,
        },
        {
            "name": "full_frame_native_pd_direct_contact",
            "disable_refsafe": True, "contact": "direct",
            "activation_clearance_m": 0.0,
        },
        {
            "name": "full_frame_average_velocity_refsafe",
            "disable_refsafe": False, "contact": "source_positive",
            "activation_clearance_m": 0.0,
            "robot_velocity_mode": "backward_position_difference",
        },
        {
            "name": "full_frame_average_velocity_critical_safe",
            "disable_refsafe": False, "contact": "critical_safe",
            "activation_clearance_m": 0.0,
            "robot_velocity_mode": "backward_position_difference",
        },
        {
            "name": "full_frame_all_average_velocity_refsafe",
            "disable_refsafe": False, "contact": "source_positive",
            "activation_clearance_m": 0.0,
            "robot_velocity_mode": "backward_position_difference",
            "object_velocity_mode": "backward_pose_difference",
        },
    )
    reports: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {"sample_rows": rows, "phase": phase[rows]}
    for profile in profiles:
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = DT
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        if profile["disable_refsafe"]:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
        else:
            model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.noslip_iterations = 1
        data = mujoco.MjData(model)

        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        actuator_ids: list[int] = []
        for name in names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{name}",
            )
            if min(joint, actuator) < 0:
                raise ValueError(f"scene lacks drive mapping for {name!r}")
            qpos_addresses.append(int(model.jnt_qposadr[joint]))
            dof_addresses.append(int(model.jnt_dofadr[joint]))
            actuator_ids.append(actuator)
            model.actuator_biasprm[actuator, 2] = -(100.0 + DT * 1000.0)
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
        model.dof_damping[object_dof : object_dof + 6] = 0.0
        hand_geoms = {
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
            .startswith("collision_hand_")
        }
        pair_channel: dict[frozenset[int], str] = {}
        hand_pairs: list[int] = []
        table_pair = -1
        for pair in range(model.npair):
            geom1 = int(model.pair_geom1[pair])
            geom2 = int(model.pair_geom2[pair])
            geoms = frozenset((geom1, geom2))
            if geoms == frozenset((table_geom, object_geom)):
                table_pair = pair
                pair_channel[geoms] = "table"
            elif object_geom in geoms and geoms & hand_geoms:
                hand_pairs.append(pair)
                hand_geom = next(iter(geoms & hand_geoms))
                hand_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom,
                ) or ""
                pair_channel[geoms] = next(
                    (finger for finger in FINGERS if finger in hand_name), "palm",
                )
        if table_pair < 0 or not hand_pairs:
            raise ValueError("scene lacks object contact pairs")
        clearance = float(profile["activation_clearance_m"])
        for pair in hand_pairs:
            model.pair_gap[pair] = model.pair_margin[pair] - clearance
            if profile["contact"] == "critical_safe":
                model.pair_solref[pair] = (2.0 * DT, 1.0)
            elif profile["contact"] == "direct":
                model.pair_solref[pair] = (-9.0e6, -1.2e4)
        if profile["contact"] == "critical_safe":
            model.pair_solref[table_pair] = (2.0 * DT, 1.0)
        elif profile["contact"] == "direct":
            model.pair_solref[table_pair] = (-9.0e6, -1.2e4)

        predicted_pose = np.empty((len(rows), 7), dtype=np.float64)
        predicted_linear = np.empty((len(rows), 3), dtype=np.float64)
        predicted_angular = np.empty((len(rows), 3), dtype=np.float64)
        predicted_robot = np.empty((len(rows), 18), dtype=np.float64)
        recorded_channel = np.zeros(
            (len(rows), len(channels), 3), dtype=np.float64,
        )
        warnings = 0
        for sample, row in enumerate(rows):
            previous = int(row - 1)
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = robot[previous]
            velocity_mode = str(profile.get("robot_velocity_mode", "sapien_api"))
            if velocity_mode == "sapien_api":
                data.qvel[dof_addresses] = robot_velocity[previous]
            elif velocity_mode == "backward_position_difference":
                if previous < 1:
                    raise ValueError("backward velocity needs two source positions")
                data.qvel[dof_addresses] = (
                    robot[previous] - robot[previous - 1]
                ) / DT
            else:
                raise ValueError(f"unknown robot velocity mode {velocity_mode!r}")
            data.qpos[object_qpos : object_qpos + 7] = pose[previous]
            object_velocity_mode = str(
                profile.get("object_velocity_mode", "sapien_api"),
            )
            if object_velocity_mode == "sapien_api":
                initial_linear = linear[previous]
                initial_angular = angular[previous]
            elif object_velocity_mode == "backward_pose_difference":
                if previous < 1:
                    raise ValueError("backward object velocity needs two poses")
                initial_linear = (pose[previous, :3] - pose[previous - 1, :3]) / DT
                initial_angular = (
                    Rotation.from_quat(pose[previous, 3:], scalar_first=True)
                    * Rotation.from_quat(
                        pose[previous - 1, 3:], scalar_first=True,
                    ).inv()
                ).as_rotvec() / DT
            else:
                raise ValueError(
                    f"unknown object velocity mode {object_velocity_mode!r}",
                )
            data.qvel[object_dof : object_dof + 3] = linear_decay * initial_linear
            data.qvel[object_dof + 3 : object_dof + 6] = (
                angular_decay * initial_angular
            )
            data.ctrl[actuator_ids] = target[row]
            mujoco.mj_step1(model, data)
            mujoco.mj_step2(model, data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                channel = pair_channel.get(
                    frozenset((int(contact.geom1), int(contact.geom2))),
                )
                if channel not in channels or int(contact.efc_address) < 0:
                    continue
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, force)
                normal = np.asarray(contact.frame[:3], dtype=np.float64)
                toward = data.xpos[object_body] - np.asarray(contact.pos)
                if float(normal @ toward) < 0.0:
                    normal = -normal
                recorded_channel[sample, channels.index(channel)] += (
                    max(0.0, float(force[0])) * DT * normal
                )
            predicted_pose[sample] = data.qpos[object_qpos : object_qpos + 7]
            predicted_linear[sample] = data.qvel[object_dof : object_dof + 3]
            predicted_angular[sample] = data.qvel[object_dof + 3 : object_dof + 6]
            predicted_robot[sample] = data.qpos[qpos_addresses]
            warnings += sum(int(warning.number) for warning in data.warning)

        gravity = np.asarray((0.0, 0.0, -9.81), dtype=np.float64)
        source_total = mass * (
            linear[rows] - linear_decay * linear[rows - 1]
        ) - mass * gravity * DT
        predicted_total = mass * (
            predicted_linear - linear_decay * linear[rows - 1]
        ) - mass * gravity * DT
        profile_report = {
            **profile,
            "source_dt_s": DT,
            "native_pd_kp": 1000.0,
            "native_pd_kd_implicit_full_frame": 100.0 + DT * 1000.0,
            "warning_count": warnings,
            "object_position_rmse_m": rms(predicted_pose[:, :3] - pose[rows, :3]),
            "object_rotation_rmse_rad": rms(rotation_error(
                predicted_pose[:, 3:], pose[rows, 3:],
            )),
            "object_linear_velocity_rmse_m_s": rms(
                predicted_linear - linear[rows]
            ),
            "object_angular_velocity_rmse_rad_s": rms(
                predicted_angular - angular[rows]
            ),
            "robot_qpos_rmse_rad": rms(predicted_robot - robot[rows]),
            "total_contact_impulse_from_momentum": vector_error(
                predicted_total, source_total,
            ),
            "normal_impulse_by_channel": {
                name: vector_error(
                    recorded_channel[:, index], source_normal[rows, index],
                )
                for index, name in enumerate(channels)
                if name in ("table", "thumb", "mid", "ring")
            },
        }
        reports.append(profile_report)
        prefix = str(profile["name"])
        saved[f"{prefix}_object_pose_wxyz"] = predicted_pose
        saved[f"{prefix}_object_linear_velocity"] = predicted_linear
        saved[f"{prefix}_object_angular_velocity"] = predicted_angular
        saved[f"{prefix}_robot_qpos"] = predicted_robot
        saved[f"{prefix}_normal_impulse_by_channel"] = recorded_channel

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_advanced": False,
        "rendered": False,
        "question": (
            "does one full-frame coupled MuJoCo solve match SAPIEN better than "
            "25 independent MuJoCo contact steps"
        ),
        "scene": str(scene), "scene_sha256": sha256(scene),
        "source_trace": str(trace_path), "source_trace_sha256": sha256(trace_path),
        "sample_count": len(rows),
        "profiles": reports,
    }
    output.mkdir(parents=True)
    trace_out = output / "one_frame_predictions.npz"
    np.savez_compressed(
        trace_out, schema=np.asarray(SCHEMA), diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False), **saved,
    )
    report["trace"] = str(trace_out)
    report["trace_sha256"] = sha256(trace_out)
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
