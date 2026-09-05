#!/usr/bin/env python3
"""Hold one recorded airborne grasp state without replaying the candidate.

The object, hand, and velocities are reset to one recorded passing SAPIEN
state.  The drive target is then held constant for 0.5 s.  This isolates
loaded contact persistence from approach timing and never advances a grasp
candidate or renders a video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = Path(__file__).resolve().parent
for index, path in enumerate((ROOT, DIAGNOSTICS)):
    if str(path) not in sys.path:
        sys.path.insert(index, str(path))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from replay_exact_deximit_mujoco import body_point_velocity, friction_for_speed
from sapien_equiv import (
    PhysxStrongFriction,
    PhysxTgsForceDrive,
    advance_physx_tgs_microstep,
    apply_physx_frame_start_gravity,
    apply_physx_velocity_decay,
)


SCHEMA = "deximit_loaded_hold_audit_v1_diagnostic_only"
TRACE_SCHEMAS = {
    "deximit_sapien_hand_trace_v6_pair_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v7_link_contact_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only",
}
DT = 1.0 / 240.0
SUBSTEPS = 25
FINGERS = ("thumb", "index", "mid", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--home-probe", type=Path, required=True)
    parser.add_argument("--row", type=int, default=1500)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--focus", choices=("full", "force", "strong"), default="full",
        help="Run the complete grid or three force-attribution profiles.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    trace_path = args.sapien_trace.expanduser().resolve(strict=True)
    summary_path = args.sapien_summary.expanduser().resolve(strict=True)
    probe_path = args.home_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite loaded hold audit {output}")
    if args.frames <= 0:
        raise ValueError("hold frame count must be positive")

    with np.load(probe_path, allow_pickle=False) as values:
        names = [str(value) for value in values["joint_names"]]
    with np.load(trace_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) not in TRACE_SCHEMAS
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
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
        candidate = int(np.asarray(values["candidate_index"]).item())
        depth = int(np.asarray(values["pool_depth"]).item())
    row = int(args.row)
    if (
        len(names) != 18 or row < 1 or row >= len(phase)
        or phase[row] not in ("demonstrated_object_motion", "hold")
    ):
        raise ValueError("loaded hold row or joint mapping is invalid")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [
        value for value in summary.get("original_sapien_passes", [])
        if int(value.get("source_candidate_index", -1)) == candidate
        and int(value.get("depth", -1)) == depth
    ]
    if len(matches) != 1:
        raise ValueError("summary is not bound to this trace")
    metrics = matches[0]["metrics"]
    object_mass = float(metrics["object_mass_kg"])
    linear_damping = float(metrics["object_linear_damping"])
    angular_damping = float(metrics["object_angular_damping"])

    reports: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {}
    profiles = tuple(
        (damping, "sapien_api", "speed_threshold", 1)
        for damping in (1.0, 2.0, 4.0, 8.0, 16.0)
    ) + (
        (1.0, "backward_difference", "speed_threshold", 1),
        (4.0, "backward_difference", "speed_threshold", 1),
        (1.0, "backward_difference", "fixed_static", 5),
        (4.0, "backward_difference", "fixed_static", 5),
        (1.0, "backward_difference", "fixed_static", 25),
    )
    if args.focus == "force":
        profiles = (
            (1.0, "sapien_api", "fixed_static", 1),
            (2.0, "sapien_api", "fixed_static", 1),
            (4.0, "sapien_api", "fixed_static", 1),
        )
    profiles = tuple((*profile, None) for profile in profiles)
    if args.focus == "strong":
        profiles = tuple(
            (1.0, "sapien_api", "strong_anchor", 1, time_constant)
            for time_constant in (0.003, DT, 0.005)
        )
    for damping, velocity_mode, friction_mode, noslip, anchor_time in profiles:
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.noslip_iterations = noslip
        data = mujoco.MjData(model)
        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        for name in names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"scene lacks source joint {name!r}")
            qpos_addresses.append(int(model.jnt_qposadr[joint]))
            dof_addresses.append(int(model.jnt_dofadr[joint]))
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
        hand_geoms: dict[int, str] = {}
        for geom in range(model.ngeom):
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom,
            ) or ""
            if not geom_name.startswith("collision_hand_"):
                continue
            hand_geoms[geom] = next(
                (finger for finger in FINGERS if finger in geom_name), "palm",
            )
        pair_by_geoms: dict[frozenset[int], int] = {}
        pair_channel: dict[int, str] = {}
        hand_pairs: list[int] = []
        table_pair = -1
        for pair in range(model.npair):
            geoms = frozenset((
                int(model.pair_geom1[pair]), int(model.pair_geom2[pair]),
            ))
            pair_by_geoms[geoms] = pair
            if geoms == frozenset((table_geom, object_geom)):
                table_pair = pair
            elif object_geom in geoms and geoms & hand_geoms.keys():
                hand_pairs.append(pair)
                pair_channel[pair] = hand_geoms[
                    next(iter(geoms & hand_geoms.keys()))
                ]
        if table_pair < 0 or not hand_pairs:
            raise ValueError("scene lacks hand/object/table contact mapping")
        pair_hand_bodies = {
            pair: int(model.geom_bodyid[next(iter(
                frozenset((
                    int(model.pair_geom1[pair]),
                    int(model.pair_geom2[pair]),
                )) & hand_geoms.keys()
            ))])
            for pair in hand_pairs
        }
        for pair in hand_pairs:
            model.pair_solref[pair] = (0.00035, damping)
            model.pair_solimp[pair] = (0.95, 0.95, 0.001, 0.5, 2.0)

        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addresses] = robot[row]
        data.qpos[object_qpos : object_qpos + 7] = pose[row]
        if velocity_mode == "sapien_api":
            data.qvel[dof_addresses] = robot_velocity[row]
            data.qvel[object_dof : object_dof + 3] = linear[row]
            data.qvel[object_dof + 3 : object_dof + 6] = angular[row]
        elif velocity_mode == "backward_difference":
            data.qvel[dof_addresses] = (robot[row] - robot[row - 1]) / DT
            data.qvel[object_dof : object_dof + 3] = (
                pose[row, :3] - pose[row - 1, :3]
            ) / DT
            data.qvel[object_dof + 3 : object_dof + 6] = (
                Rotation.from_quat(pose[row, 3:], scalar_first=True)
                * Rotation.from_quat(
                    pose[row - 1, 3:], scalar_first=True,
                ).inv()
            ).as_rotvec() / DT
        else:
            raise ValueError(f"unknown initial velocity mode {velocity_mode!r}")
        drive = PhysxTgsForceDrive(
            model, mujoco, names, stiffness=1000.0, damping=100.0,
        )
        strong_friction = None
        if friction_mode == "strong_anchor":
            assert anchor_time is not None
            strong_friction = PhysxStrongFriction(
                model, mujoco, object_body=object_body,
                object_geom=object_geom, pair_hand_bodies=pair_hand_bodies,
                static_friction=0.7, time_constant=float(anchor_time),
                correlation_distance=0.025,
            )
            if not np.isclose(
                float(model.body_mass[object_body]), object_mass,
                atol=1.0e-12, rtol=0.0,
            ):
                raise ValueError("scene object mass differs from the SAPIEN trace")
        drive.set_target(target[row])
        scene_gravity = model.opt.gravity.copy()
        object_trace = np.empty((args.frames, 7), dtype=np.float64)
        robot_trace = np.empty((args.frames, 18), dtype=np.float64)
        active_trace = np.zeros((args.frames, len(FINGERS)), dtype=bool)
        normal_impulse_trace = np.zeros(
            (args.frames, len(FINGERS)), dtype=np.float64,
        )
        friction_impulse_trace = np.zeros_like(normal_impulse_trace)
        normal_vector_trace = np.zeros(
            (args.frames, len(FINGERS), 3), dtype=np.float64,
        )
        friction_vector_trace = np.zeros_like(normal_vector_trace)
        minimum_gap = np.inf
        warning_count = 0
        anchor_force_trace = np.zeros(
            (args.frames, len(FINGERS), 3), dtype=np.float64,
        )
        anchor_stiffness = (
            0.0 if strong_friction is None else strong_friction.stiffness
        )
        anchor_damping = (
            0.0 if strong_friction is None else strong_friction.damping
        )
        for frame in range(args.frames):
            apply_physx_velocity_decay(
                data.qvel, object_dof, linear_rate=linear_damping,
                angular_rate=angular_damping,
            )
            apply_physx_frame_start_gravity(
                model, data, mujoco, scene_gravity, source_dt=DT,
            )
            drive.begin_frame(data)
            for _ in range(SUBSTEPS):
                for pair in hand_pairs:
                    model.pair_friction[pair, :2] = (
                        0.0 if friction_mode == "strong_anchor" else 0.5
                    )
                model.pair_friction[table_pair, :2] = 0.75
                mujoco.mj_forward(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geoms = frozenset((int(contact.geom1), int(contact.geom2)))
                    pair = pair_by_geoms.get(geoms)
                    if pair is None or object_geom not in geoms:
                        continue
                    other_geom = next(iter(geoms - {object_geom}))
                    other_body = int(model.geom_bodyid[other_geom])
                    relative = body_point_velocity(
                        model, data, object_body, contact.pos,
                    ) - body_point_velocity(
                        model, data, other_body, contact.pos,
                    )
                    normal = np.asarray(contact.frame[:3], dtype=np.float64)
                    tangent = relative - float(relative @ normal) * normal
                    speed = float(np.linalg.norm(tangent))
                    if pair == table_pair:
                        model.pair_friction[pair, :2] = friction_for_speed(
                            speed, static=0.85, dynamic=0.75, threshold=0.02,
                        )
                    elif pair in pair_channel:
                        if friction_mode == "speed_threshold":
                            friction = friction_for_speed(
                                speed, static=0.7, dynamic=0.5, threshold=0.02,
                            )
                        elif friction_mode == "fixed_static":
                            friction = 0.7
                        elif friction_mode == "strong_anchor":
                            friction = 0.0
                        else:
                            raise ValueError(
                                f"unknown friction mode {friction_mode!r}",
                            )
                        model.pair_friction[pair, :2] = friction
                data.qfrc_applied[:] = 0.0
                if strong_friction is not None:
                    # Rebuild normal constraints with zero native tangential
                    # friction, then apply the retained anchor force as an
                    # equal-and-opposite body force.  The anchor is removed
                    # when contact is lost or its 25 mm PhysX correlation
                    # distance is exceeded.
                    mujoco.mj_forward(model, data)
                    for pair, desired in strong_friction.apply(data).items():
                        # The physical anchor also covers the palm pair, while
                        # the audit trace has one column per named finger.
                        channel = pair_channel[pair]
                        if channel in FINGERS:
                            finger = FINGERS.index(channel)
                            anchor_force_trace[frame, finger] += (
                                desired * float(model.opt.timestep)
                            )
                advance_physx_tgs_microstep(model, data, mujoco, drive)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geoms = frozenset((int(contact.geom1), int(contact.geom2)))
                    pair = pair_by_geoms.get(geoms)
                    if pair is None or object_geom not in geoms:
                        continue
                    minimum_gap = min(minimum_gap, float(contact.dist))
                    channel = pair_channel.get(pair)
                    if channel in FINGERS and int(contact.efc_address) >= 0:
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            model, data, contact_index, force,
                        )
                        if float(force[0]) > 1.0e-6:
                            active_trace[frame, FINGERS.index(channel)] = True
                        finger = FINGERS.index(channel)
                        contact_frame = np.asarray(
                            contact.frame, dtype=np.float64,
                        ).reshape(3, 3)
                        world_force = contact_frame.T @ force[:3]
                        toward_object = (
                            data.xpos[object_body] - np.asarray(contact.pos)
                        )
                        if float(world_force @ toward_object) < 0.0:
                            world_force = -world_force
                        normal_world = np.asarray(contact.frame[:3], dtype=np.float64)
                        if float(normal_world @ toward_object) < 0.0:
                            normal_world = -normal_world
                        normal_force = max(0.0, float(force[0])) * normal_world
                        friction_force = world_force - normal_force
                        normal_impulse_trace[frame, finger] += (
                            max(0.0, float(force[0])) * float(model.opt.timestep)
                        )
                        friction_impulse_trace[frame, finger] += (
                            float(np.linalg.norm(force[1:3]))
                            * float(model.opt.timestep)
                        )
                        normal_vector_trace[frame, finger] += (
                            normal_force * float(model.opt.timestep)
                        )
                        friction_vector_trace[frame, finger] += (
                            friction_force * float(model.opt.timestep)
                        )
            object_trace[frame] = data.qpos[object_qpos : object_qpos + 7]
            robot_trace[frame] = data.qpos[qpos_addresses]
            warning_count += sum(int(warning.number) for warning in data.warning)

        lift = object_trace[:, 2] - pose[row, 2]
        # Keep the loaded-hold gate aligned with the common grasp gate:
        # thumb plus any named non-thumb finger is a valid opposed pair.
        other_fingers = [
            index for index, finger in enumerate(FINGERS) if finger != "thumb"
        ]
        opposed = active_trace[:, FINGERS.index("thumb")] & np.any(
            active_trace[:, other_fingers], axis=1,
        )
        tail = slice(max(0, args.frames - 24), args.frames)
        passed = bool(
            warning_count == 0
            and np.all(opposed[tail])
            and np.min(lift[tail]) >= -0.005
            and np.ptp(object_trace[tail, 2]) <= 0.003
            and minimum_gap >= -0.002
        )
        name = (
            f"hand_d{damping:g}_vel_{velocity_mode}_"
            f"friction_{friction_mode}_noslip_{noslip}"
            + (
                "" if anchor_time is None
                else f"_anchor_t{float(anchor_time):.9g}"
            )
        )
        reports.append({
            "name": name,
            "hand_contact_time_constant_s": 0.00035,
            "hand_contact_damping_ratio": damping,
            "initial_velocity_mode": velocity_mode,
            "hand_friction_mode": friction_mode,
            "noslip_iterations": noslip,
            "strong_anchor_time_constant_s": anchor_time,
            "strong_anchor_stiffness_n_m": anchor_stiffness,
            "strong_anchor_damping_n_s_m": anchor_damping,
            "passed": passed,
            "opposed_contact_all_tail_frames": bool(np.all(opposed[tail])),
            "minimum_height_change_m": float(np.min(lift)),
            "tail_minimum_height_change_m": float(np.min(lift[tail])),
            "tail_vertical_span_m": float(np.ptp(object_trace[tail, 2])),
            "minimum_hand_object_gap_m": float(minimum_gap),
            "robot_qpos_rmse_from_reset_rad": float(np.sqrt(np.mean(np.square(
                robot_trace - robot[row],
            )))),
            "tail_normal_force_mean_n": {
                finger: float(np.mean(normal_impulse_trace[tail, index]) / DT)
                for index, finger in enumerate(FINGERS)
            },
            "tail_friction_force_mean_n": {
                finger: float(np.mean(friction_impulse_trace[tail, index]) / DT)
                for index, finger in enumerate(FINGERS)
            },
            "tail_net_normal_force_mean_xyz_n": (
                np.mean(np.sum(normal_vector_trace[tail], axis=1), axis=0) / DT
            ).tolist(),
            "tail_net_friction_force_mean_xyz_n": (
                np.mean(np.sum(friction_vector_trace[tail], axis=1), axis=0) / DT
            ).tolist(),
            "tail_net_anchor_force_mean_xyz_n": (
                np.mean(np.sum(anchor_force_trace[tail], axis=1), axis=0) / DT
            ).tolist(),
            "warning_count": warning_count,
        })
        saved[f"{name}_object_pose_wxyz"] = object_trace
        saved[f"{name}_robot_qpos"] = robot_trace
        saved[f"{name}_active_fingers"] = active_trace
        saved[f"{name}_normal_impulse_ns"] = normal_impulse_trace
        saved[f"{name}_friction_impulse_ns"] = friction_impulse_trace
        saved[f"{name}_normal_vector_impulse_ns"] = normal_vector_trace
        saved[f"{name}_friction_vector_impulse_ns"] = friction_vector_trace
        saved[f"{name}_anchor_vector_impulse_ns"] = anchor_force_trace

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_advanced": False,
        "rendered": False,
        "state_source_candidate": candidate,
        "state_source_row": row,
        "state_source_phase": str(phase[row]),
        "hold_duration_s": args.frames * DT,
        "profile_focus": args.focus,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "source_trace": str(trace_path), "source_trace_sha256": sha256(trace_path),
        "profiles": reports,
    }
    output.mkdir(parents=True)
    trace_out = output / "loaded_hold.npz"
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
