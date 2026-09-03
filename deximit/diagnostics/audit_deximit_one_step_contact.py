#!/usr/bin/env python3
"""Separate MuJoCo contact and friction errors without trajectory drift.

Each trial starts from one recorded post-step SAPIEN state and predicts only
the following 1/240 s state.  Restarting at every sample prevents an early
contact error from contaminating all later comparisons.  This is an isolated
diagnostic: it never renders and never changes the formal renderer/3.3 chain.
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
    PhysxTgsForceDrive,
    advance_physx_tgs_microstep,
    apply_physx_frame_start_gravity,
    apply_physx_velocity_decay,
    incoming_joint_wrenches_at_loader_frames,
)


SCHEMA = (
    "deximit_one_step_contact_audit_"
    "v18_joint_wrench_coupling_diagnostic_only"
)
TRACE_SCHEMAS = {
    "deximit_sapien_hand_trace_v6_pair_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v7_link_contact_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only",
}
DT = 1.0 / 240.0
SUBSTEPS = 25
FINGERS = ("thumb", "index", "mid", "ring", "pinky")
ACTIVE_LINK_JOINTS = (
    ("right_hand_thumb_rota_link2", "right_hand_thumb_rota_joint2"),
    ("right_hand_mid_link2", "right_hand_mid_joint2"),
    ("right_hand_ring_link2", "right_hand_ring_joint2"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(actual, scalar_first=True)
        * Rotation.from_quat(expected, scalar_first=True).inv()
    ).magnitude()


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def advance_drive_before_contact(
    model: mujoco.MjModel, data: mujoco.MjData, drive: PhysxTgsForceDrive,
    record: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Test the opposite row order without changing the runtime converter."""
    mujoco.mj_forward(model, data)
    drive_impulses, limit_impulses = drive.apply_impulses(data)
    # Recompute velocity-dependent contact constraints after the drive row.
    mujoco.mj_forward(model, data)
    if record is not None:
        record(model, data, drive.dt)
    data.qvel[:] += data.qacc * drive.dt
    mujoco.mj_integratePos(model, data.qpos, data.qvel, drive.dt)
    drive.project_hard_limits(data)
    mujoco.mj_normalizeQuat(model, data.qpos)
    data.time += drive.dt
    mujoco.mj_forward(model, data)
    return drive_impulses, limit_impulses


def advance_symmetric_contact_drive_contact(
    model: mujoco.MjModel, data: mujoco.MjData, drive: PhysxTgsForceDrive,
    record: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Test a time-symmetric contact/drive/contact split."""
    mujoco.mj_forward(model, data)
    if record is not None:
        record(model, data, 0.5 * drive.dt)
    data.qvel[:] += data.qacc * (0.5 * drive.dt)
    drive_impulses, limit_impulses = drive.apply_impulses(data)
    mujoco.mj_forward(model, data)
    if record is not None:
        record(model, data, 0.5 * drive.dt)
    data.qvel[:] += data.qacc * (0.5 * drive.dt)
    mujoco.mj_integratePos(model, data.qpos, data.qvel, drive.dt)
    drive.project_hard_limits(data)
    mujoco.mj_normalizeQuat(model, data.qpos)
    data.time += drive.dt
    mujoco.mj_forward(model, data)
    return drive_impulses, limit_impulses


def advance_coupled_drive_force(
    model: mujoco.MjModel, data: mujoco.MjData, drive: PhysxTgsForceDrive,
    record: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Let MuJoCo solve the converted drive force with contact constraints."""
    mujoco.mj_forward(model, data)
    velocity_before = data.qvel.copy()
    drive_impulses, limit_impulses = drive.apply_impulses(data)
    drive_delta = data.qvel - velocity_before
    data.qvel[:] = velocity_before
    mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass, data.qM)
    applied_before = data.qfrc_applied.copy()
    data.qfrc_applied[:] += mass @ drive_delta / drive.dt
    mujoco.mj_forward(model, data)
    if record is not None:
        record(model, data, drive.dt)
    data.qvel[:] += data.qacc * drive.dt
    data.qfrc_applied[:] = applied_before
    mujoco.mj_integratePos(model, data.qpos, data.qvel, drive.dt)
    drive.project_hard_limits(data)
    mujoco.mj_normalizeQuat(model, data.qpos)
    data.time += drive.dt
    mujoco.mj_forward(model, data)
    return drive_impulses, limit_impulses


def advance_contact_before_drive_recorded(
    model: mujoco.MjModel, data: mujoco.MjData, drive: PhysxTgsForceDrive,
    record: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the production order and record the forces actually integrated."""
    mujoco.mj_forward(model, data)
    record(model, data, drive.dt)
    data.qvel[:] += data.qacc * drive.dt
    drive_impulses, limit_impulses = drive.apply_impulses(data)
    mujoco.mj_integratePos(model, data.qpos, data.qvel, drive.dt)
    drive.project_hard_limits(data)
    mujoco.mj_normalizeQuat(model, data.qpos)
    data.time += drive.dt
    mujoco.mj_forward(model, data)
    return drive_impulses, limit_impulses


def apply_contact_velocity_iteration(
    model: mujoco.MjModel, data: mujoco.MjData, duration_s: float,
    record: object | None = None,
) -> None:
    """Approximate PhysX's final velocity-only TGS constraint iteration.

    The released scene requests one velocity iteration after its 25 position
    iterations.  That pass does not integrate position or repeat gravity and
    drive impulses.  MuJoCo exposes the contact-only acceleration as the
    difference between constrained and smooth accelerations, so integrate only
    that difference for one TGS substep.
    """
    mujoco.mj_forward(model, data)
    if record is not None:
        record(model, data, duration_s)
    data.qvel[:] += (data.qacc - data.qacc_smooth) * duration_s
    mujoco.mj_forward(model, data)


def vector_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    delta = actual - expected
    return {
        "rmse_ns": rms(delta),
        "bias_ns_xyz": np.mean(delta, axis=0).tolist(),
        "actual_sum_ns_xyz": np.sum(actual, axis=0).tolist(),
        "expected_sum_ns_xyz": np.sum(expected, axis=0).tolist(),
    }


def wrench_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Summarize force:torque wrench agreement without mixing units."""
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 6:
        raise ValueError("wrench arrays must have matching (sample, 6) shapes")
    actual_force = actual[:, :3]
    expected_force = expected[:, :3]
    actual_torque = actual[:, 3:]
    expected_torque = expected[:, 3:]
    expected_force_norm_sum = float(np.sum(np.linalg.norm(expected_force, axis=1)))
    expected_torque_norm_sum = float(np.sum(np.linalg.norm(expected_torque, axis=1)))
    return {
        "force_vector_rmse_n": rms(actual_force - expected_force),
        "torque_vector_rmse_nm": rms(actual_torque - expected_torque),
        "force_vector_rmse_n_if_sign_flipped": rms(actual_force + expected_force),
        "torque_vector_rmse_nm_if_sign_flipped": rms(actual_torque + expected_torque),
        "actual_force_norm_mean_n": float(np.mean(np.linalg.norm(actual_force, axis=1))),
        "expected_force_norm_mean_n": float(np.mean(np.linalg.norm(expected_force, axis=1))),
        "actual_torque_norm_mean_nm": float(np.mean(np.linalg.norm(actual_torque, axis=1))),
        "expected_torque_norm_mean_nm": float(np.mean(np.linalg.norm(expected_torque, axis=1))),
        "force_norm_sum_ratio": (
            None if expected_force_norm_sum <= 1.0e-12 else float(
                np.sum(np.linalg.norm(actual_force, axis=1))
                / expected_force_norm_sum
            )
        ),
        "torque_norm_sum_ratio": (
            None if expected_torque_norm_sum <= 1.0e-12 else float(
                np.sum(np.linalg.norm(actual_torque, axis=1))
                / expected_torque_norm_sum
            )
        ),
    }


def summarize(
    rows: np.ndarray, predicted_pose: np.ndarray, predicted_linear: np.ndarray,
    predicted_angular: np.ndarray, predicted_robot: np.ndarray,
    source_pose: np.ndarray, source_linear: np.ndarray,
    source_angular: np.ndarray, source_robot: np.ndarray,
) -> dict[str, float | int | list[float]]:
    pos_delta = predicted_pose[:, :3] - source_pose[rows, :3]
    linear_delta = predicted_linear - source_linear[rows]
    angular_delta = predicted_angular - source_angular[rows]
    robot_delta = predicted_robot - source_robot[rows]
    return {
        "samples": int(len(rows)),
        "position_rmse_m": rms(pos_delta),
        "position_error_m_mean": float(np.mean(np.linalg.norm(pos_delta, axis=1))),
        "position_bias_m_xyz": np.mean(pos_delta, axis=0).tolist(),
        "rotation_error_rad_mean": float(np.mean(rotation_error(
            predicted_pose[:, 3:], source_pose[rows, 3:],
        ))),
        "linear_velocity_rmse_m_s": rms(linear_delta),
        "linear_velocity_bias_m_s_xyz": np.mean(linear_delta, axis=0).tolist(),
        "angular_velocity_rmse_rad_s": rms(angular_delta),
        "robot_qpos_rmse_rad": rms(robot_delta),
        "arm_qpos_rmse_rad": rms(robot_delta[:, :6]),
        "hand_qpos_rmse_rad": rms(robot_delta[:, 6:]),
        "robot_qpos_rmse_by_axis_rad": np.sqrt(
            np.mean(np.square(robot_delta), axis=0),
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--home-probe", type=Path, required=True)
    parser.add_argument(
        "--sapien-joint-frames", type=Path,
        help=(
            "Isolated SAPIEN loader-frame probe; required by --focus coupling "
            "because PhysX incoming-joint frames can differ from URDF links."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--focus", choices=(
            "full", "backend", "solver", "finger_response", "activation",
            "order", "impulse", "direct_contact", "direct_friction",
            "patch_friction", "force_timing", "coupling", "velocity_state",
            "speculative", "baseline", "velocity_iteration",
        ), default="full",
        help="Run the complete material grid or one focused comparison.",
    )
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    trace_path = args.sapien_trace.expanduser().resolve(strict=True)
    summary_path = args.sapien_summary.expanduser().resolve(strict=True)
    probe_path = args.home_probe.expanduser().resolve(strict=True)
    joint_frame_path = (
        args.sapien_joint_frames.expanduser().resolve(strict=True)
        if args.sapien_joint_frames is not None else None
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-step audit {output}")

    loader_frame_by_link: dict[str, tuple[str, np.ndarray]] = {}
    if joint_frame_path is not None:
        with np.load(joint_frame_path, allow_pickle=False) as values:
            if (
                str(np.asarray(values["schema"]).item())
                != "deximit_sapien_joint_frame_probe_v1_diagnostic_only"
                or not bool(np.asarray(values["diagnostic_only"]).item())
                or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
                or bool(np.asarray(values["candidate_executed"]).item())
            ):
                raise ValueError("SAPIEN joint-frame probe contract is invalid")
            loader_links = [str(value) for value in values["link_order"]]
            loader_joints = [
                str(value) for value in values["incoming_joint_order"]
            ]
            loader_poses = np.asarray(
                values["incoming_joint_pose_in_child_link_p_wxyz"],
                dtype=np.float64,
            )
        if (
            len(loader_links) != len(loader_joints)
            or loader_poses.shape != (len(loader_links), 7)
            or len(set(loader_links)) != len(loader_links)
            or not np.isfinite(loader_poses).all()
        ):
            raise ValueError("SAPIEN loader-frame arrays are malformed")
        loader_frame_by_link = {
            link: (joint, loader_poses[index])
            for index, (link, joint) in enumerate(zip(loader_links, loader_joints))
        }

    with np.load(trace_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) not in TRACE_SCHEMAS
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or not np.isclose(float(values["physics_dt_s"]), DT)
            or str(np.asarray(values["sample_semantics"]).item())
            != "post-step state under same-index drive target"
        ):
            raise ValueError("input is not the detailed post-step SAPIEN trace")
        phase = np.asarray(values["phase"]).astype("U32")
        source_pose = np.asarray(values["object_pose_sapien_wxyz"], dtype=np.float64)
        source_linear = np.asarray(
            values["object_linear_velocity_sapien"], dtype=np.float64,
        )
        source_angular = np.asarray(
            values["object_angular_velocity_sapien"], dtype=np.float64,
        )
        source_robot = np.asarray(
            values["right_robot_qpos_sapien"], dtype=np.float64,
        )
        source_robot_velocity = np.asarray(
            values["right_robot_qvel_sapien"], dtype=np.float64,
        )
        source_target = np.asarray(
            values["right_drive_target_sapien"], dtype=np.float64,
        )
        source_normal_impulse = np.asarray(
            values["contact_channel_normal_impulse_net_on_object_ns"],
            dtype=np.float64,
        )
        source_tangent_impulse = np.asarray(
            values["contact_channel_tangent_impulse_net_on_object_ns"],
            dtype=np.float64,
        )
        source_patch_radius = np.asarray(
            values["contact_channel_patch_rms_radius_m_mean"],
            dtype=np.float64,
        )
        channel_order = [str(value) for value in values["contact_channel_order"]]
        channel_load = np.asarray(values["contact_channel_load_bearing"], dtype=bool)
        if "right_robot_link_incoming_joint_force_child_frame" in values:
            if (
                tuple(str(value) for value in values[
                    "incoming_joint_force_component_order"
                ]) != (
                    "force_x", "force_y", "force_z",
                    "torque_x", "torque_y", "torque_z",
                )
                or str(np.asarray(values["incoming_joint_force_frame"]).item())
                != "child incoming-joint frame"
            ):
                raise ValueError("SAPIEN incoming-joint wrench convention changed")
            source_link_order = [
                str(value) for value in values["right_robot_link_order"]
            ]
            source_joint_wrench = np.asarray(
                values["right_robot_link_incoming_joint_force_child_frame"],
                dtype=np.float64,
            )
            if source_joint_wrench.shape != (
                len(phase), len(source_link_order), 6,
            ) or not np.isfinite(source_joint_wrench).all():
                raise ValueError("SAPIEN incoming-joint wrench trace is malformed")
        else:
            source_link_order = []
            source_joint_wrench = None
    finger_channels = [channel_order.index(name) for name in FINGERS]
    channel_indices = {name: index for index, name in enumerate(channel_order)}
    any_load = channel_load[:, finger_channels].any(axis=1)
    channel_patch_radius = {
        name: float(np.nanmedian(source_patch_radius[channel_load[:, index], index]))
        for name, index in zip(FINGERS, finger_channels)
        if np.any(channel_load[:, index])
    }
    available = np.flatnonzero(any_load & (np.arange(len(phase)) > 0))
    if len(available) < 100:
        raise ValueError("trace does not contain enough load-bearing contact samples")
    # Keep every onset sample, dense coverage of the first one-sided contact,
    # and a deterministic spread over the later opposed/hold interval.
    early = available[available <= available[0] + 40]
    late = available[available > early[-1]]
    spread = late[np.unique(np.linspace(0, len(late) - 1, 120).round().astype(int))]
    rows = np.unique(np.r_[early, spread])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_candidate = int(np.asarray(np.load(trace_path)["candidate_index"]).item())
    source_depth = int(np.asarray(np.load(trace_path)["pool_depth"]).item())
    matches = [
        row for row in summary.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == source_candidate
        and int(row.get("depth", -1)) == source_depth
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("metrics"), dict):
        raise ValueError("SAPIEN summary is not bound to this candidate")
    source_metrics = matches[0]["metrics"]
    linear_damping = float(source_metrics["object_linear_damping"])
    angular_damping = float(source_metrics["object_angular_damping"])
    object_mass = float(source_metrics["object_mass_kg"])
    with np.load(probe_path, allow_pickle=False) as values:
        names = [str(value) for value in values["joint_names"]]
    if len(names) != 18:
        raise ValueError("home probe does not define the 18 source joints")

    material_profiles = (
        {
            "name": "old_contact_old_friction",
            "time": 0.00045, "impedance": 0.8,
            "cone": "pyramidal", "noslip": 0,
            "hand_threshold": 0.001, "table_threshold": 0.0,
        },
        {
            "name": "old_contact_new_friction",
            "time": 0.00045, "impedance": 0.8,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.02, "table_threshold": 0.02,
        },
        {
            "name": "old_contact_pinch_friction",
            "time": 0.00045, "impedance": 0.8,
            "cone": "pyramidal", "noslip": 1,
            "hand_threshold": 0.02, "table_threshold": 0.02,
        },
        {
            "name": "new_contact_old_friction",
            "time": 0.00035, "impedance": 0.95,
            "cone": "pyramidal", "noslip": 0,
            "hand_threshold": 0.001, "table_threshold": 0.0,
        },
        {
            "name": "new_contact_new_friction",
            "time": 0.00035, "impedance": 0.95,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.02, "table_threshold": 0.02,
        },
    )
    hand_contact_profiles = tuple(
        {
            "name": f"hand_d{damping:g}_i{impedance:g}_t{time:g}",
            # Keep table/object response at the independently passing v8
            # profile while varying only the hand/object constraint.
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": time, "hand_damping": damping,
            "hand_impedance": impedance,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "orders": ("contact_before_drive",),
        }
        for time in (0.00035, 0.00045)
        for damping in (1.0, 2.0, 4.0, 8.0, 16.0)
        for impedance in (0.95, 0.99)
    )
    hand_friction_profiles = tuple(
        {
            "name": f"hand_friction_speed_t{threshold:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": threshold, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "orders": ("contact_before_drive",),
        }
        for threshold in (0.0, 0.005, 0.02, 0.05, 0.1, 0.2)
    ) + (
        {
            "name": "hand_friction_fixed_dynamic",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.0, "table_threshold": 0.02,
            "hand_friction_mode": "fixed_dynamic",
            "orders": ("contact_before_drive",),
        },
        {
            "name": "hand_friction_fixed_static",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.0, "table_threshold": 0.02,
            "hand_friction_mode": "fixed_static",
            "orders": ("contact_before_drive",),
        },
    )
    backend_profiles = tuple(
        {
            "name": f"backend_{backend}_{'multi' if multi else 'single'}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": backend == "native", "multiccd": multi,
            "orders": ("contact_before_drive",),
        }
        for backend in ("native", "legacy") for multi in (True, False)
    )
    solver_profiles = tuple(
        {
            "name": f"solver_noslip{noslip:g}_impratio1",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": noslip, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "orders": ("contact_before_drive",),
        }
        for noslip in (0, 1, 2, 5, 10, 25)
    ) + tuple(
        {
            "name": f"solver_noslip1_impratio{impratio:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": impratio,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "orders": ("contact_before_drive",),
        }
        for impratio in (2.0, 5.0, 10.0, 20.0)
    )
    finger_response_profiles = tuple(
        {
            "name": f"finger_thumb_d{thumb:g}_opposed_d{opposed:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "channel_damping": {
                "thumb": thumb, "mid": opposed, "ring": opposed,
            },
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "orders": ("contact_before_drive",),
        }
        for thumb in (1.0, 2.0, 4.0)
        for opposed in (8.0, 16.0, 32.0)
    ) + (
        {
            "name": "finger_uniform_d4_baseline",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "channel_damping": {},
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "orders": ("contact_before_drive",),
        },
    )
    activation_profiles = tuple(
        {
            "name": f"activation_clearance_{clearance:.9g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "activation_clearance_m": clearance,
            "gravity_timing": "physx_frame_start",
            "record_impulse_decomposition": True,
            "orders": ("contact_before_drive",),
        }
        for clearance in (
            0.0, 2.5e-6, 5.0e-6, 1.0e-5,
            2.5e-5, 5.0e-5, 1.0e-4, 1.5e-4, 2.5e-4, 5.0e-4,
        )
    )
    speculative_profiles = ({
        "name": "physx_velocity_predicted_activation",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "activation_mode": "physx_velocity_prediction",
        "gravity_timing": "physx_frame_start",
        "record_impulse_decomposition": True,
        "orders": ("contact_before_drive",),
    },)
    baseline_profiles = ({
        "name": "physx_equivalent_baseline",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "gravity_timing": "physx_frame_start",
        "record_impulse_decomposition": True,
        "orders": ("contact_before_drive",),
    },)
    velocity_iteration_profiles = tuple({
        "name": f"final_velocity_iteration_{'on' if enabled else 'off'}",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "gravity_timing": "physx_frame_start",
        "record_impulse_decomposition": True,
        "record_joint_wrench": True,
        "final_velocity_iteration": enabled,
        "orders": ("contact_before_drive",),
    } for enabled in (False, True))
    order_profiles = ({
        "name": "constraint_order_v21_contact",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "orders": (
            "contact_before_drive", "drive_before_contact",
            "symmetric_contact_drive_contact", "coupled_drive_force",
        ),
    },)
    impulse_profiles = tuple(
        {
            "name": f"impulse_hand_d{hand_damping:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": hand_damping,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "gravity_timing": "physx_frame_start",
            "record_impulse_decomposition": True,
            "orders": ("contact_before_drive",),
        }
        for hand_damping in (1.0, 2.0, 4.0, 8.0, 16.0)
    )
    direct_contact_profiles = tuple(
        {
            "name": f"direct_k{stiffness:g}_b{damping:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 16.0,
            "hand_direct_stiffness": stiffness,
            "hand_direct_damping": damping,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "record_impulse_decomposition": True,
            "orders": ("contact_before_drive",),
        }
        for stiffness in (9.0e6, 36.0e6, 144.0e6)
        for damping in (6.0e3, 12.0e3, 24.0e3)
    )
    direct_friction_profiles = tuple(
        {
            "name": f"direct_friction_scale_{scale:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 16.0,
            "hand_direct_stiffness": 9.0e6,
            "hand_direct_damping": 12.0e3,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_dynamic_coefficient": 0.5 * scale,
            "hand_static_coefficient": 0.7 * scale,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "record_impulse_decomposition": True,
            "orders": ("contact_before_drive",),
        }
        for scale in (1.0, 1.5, 2.0, 2.5, 3.0)
    )
    patch_friction_profiles = ({
        "name": "patch_friction_condim3_baseline",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "hand_condim": 3,
        "orders": ("contact_before_drive",),
    },) + tuple(
        {
            "name": f"patch_friction_condim4_scale_{scale:g}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "hand_condim": 4,
            "torsional_radius_scale": scale,
            "channel_patch_radius_m": channel_patch_radius,
            "orders": ("contact_before_drive",),
        }
        for scale in (0.5, 1.0, 2.0)
    )
    force_timing_profiles = tuple(
        {
            "name": f"force_timing_{timing}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "gravity_timing": timing,
            "record_impulse_decomposition": True,
            "orders": ("contact_before_drive",),
        }
        for timing in ("distributed", "physx_frame_start")
    )
    coupling_profiles = ({
        "name": "joint_wrench_constraint_order",
        "time": 0.00045, "damping": 16.0, "impedance": 0.8,
        "hand_time": 0.00035, "hand_damping": 4.0,
        "hand_impedance": 0.95,
        "cone": "elliptic", "noslip": 1, "impratio": 1.0,
        "hand_threshold": 0.02, "table_threshold": 0.02,
        "hand_friction_mode": "speed_threshold",
        "native_ccd": True, "multiccd": True,
        "gravity_timing": "physx_frame_start",
        "record_impulse_decomposition": True,
        "record_joint_wrench": True,
        "orders": (
            "contact_before_drive", "drive_before_contact",
            "symmetric_contact_drive_contact", "coupled_drive_force",
        ),
    },)
    velocity_state_profiles = tuple(
        {
            "name": f"velocity_robot_{robot_mode}_object_{object_mode}",
            "time": 0.00045, "damping": 16.0, "impedance": 0.8,
            "hand_time": 0.00035, "hand_damping": 4.0,
            "hand_impedance": 0.95,
            "cone": "elliptic", "noslip": 1, "impratio": 1.0,
            "hand_threshold": 0.02, "table_threshold": 0.02,
            "hand_friction_mode": "speed_threshold",
            "native_ccd": True, "multiccd": True,
            "gravity_timing": "physx_frame_start",
            "record_impulse_decomposition": True,
            "record_joint_wrench": True,
            "robot_velocity_mode": robot_mode,
            "object_velocity_mode": object_mode,
            "orders": ("contact_before_drive", "coupled_drive_force"),
        }
        for robot_mode, object_mode in (
            ("sapien_api", "sapien_api"),
            ("backward_qpos_difference", "sapien_api"),
            ("sapien_api", "backward_pose_difference"),
            ("backward_qpos_difference", "backward_pose_difference"),
            ("zero", "zero"),
        )
    )
    if args.focus == "backend":
        material_profiles = backend_profiles
    elif args.focus == "solver":
        material_profiles = solver_profiles
    elif args.focus == "finger_response":
        material_profiles = finger_response_profiles
    elif args.focus == "activation":
        material_profiles = activation_profiles
    elif args.focus == "speculative":
        material_profiles = speculative_profiles
    elif args.focus == "baseline":
        material_profiles = baseline_profiles
    elif args.focus == "velocity_iteration":
        if source_joint_wrench is None or joint_frame_path is None:
            raise ValueError(
                "velocity-iteration audit requires joint-force trace and frame probe",
            )
        material_profiles = velocity_iteration_profiles
    elif args.focus == "order":
        material_profiles = order_profiles
    elif args.focus == "impulse":
        material_profiles = impulse_profiles
    elif args.focus == "direct_contact":
        material_profiles = direct_contact_profiles
    elif args.focus == "direct_friction":
        material_profiles = direct_friction_profiles
    elif args.focus == "patch_friction":
        material_profiles = patch_friction_profiles
    elif args.focus == "force_timing":
        material_profiles = force_timing_profiles
    elif args.focus == "coupling":
        if source_joint_wrench is None:
            raise ValueError("coupling audit requires a v8 SAPIEN joint-force trace")
        if joint_frame_path is None:
            raise ValueError("coupling audit requires the SAPIEN joint-frame probe")
        material_profiles = coupling_profiles
    elif args.focus == "velocity_state":
        if source_joint_wrench is None or joint_frame_path is None:
            raise ValueError(
                "velocity-state audit requires joint-force trace and frame probe",
            )
        material_profiles = velocity_state_profiles
    else:
        material_profiles = (
            material_profiles + hand_contact_profiles + hand_friction_profiles
        )
    profiles = tuple(
        {
            **{key: value for key, value in profile.items() if key != "orders"},
            "constraint_order": order,
        }
        for profile in material_profiles
        for order in profile.get(
            "orders", ("contact_before_drive", "drive_before_contact"),
        )
    )
    profile_reports: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {"sample_rows": rows, "phase": phase[rows]}
    scene_object_contact_kind: str | None = None
    for profile in profiles:
        model = mujoco.MjModel.from_xml_path(str(scene))
        if profile.get("native_ccd", True):
            model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        else:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        if profile.get("multiccd", True):
            model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        else:
            model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        model.opt.cone = (
            mujoco.mjtCone.mjCONE_PYRAMIDAL
            if profile["cone"] == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
        )
        model.opt.noslip_iterations = int(profile["noslip"])
        model.opt.impratio = float(profile.get("impratio", 1.0))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        data = mujoco.MjData(model)
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names
        ]
        qpos_addresses = [int(model.jnt_qposadr[joint]) for joint in joint_ids]
        dof_addresses = [int(model.jnt_dofadr[joint]) for joint in joint_ids]
        object_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
        )
        object_qpos = int(model.jnt_qposadr[object_joint])
        object_dof = int(model.jnt_dofadr[object_joint])
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
        )
        object_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
        )
        object_flex = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_FLEX, "right_object_flex",
        )
        use_rigid_flex = object_flex >= 0
        if use_rigid_flex and not bool(model.flex_rigid[object_flex]):
            raise ValueError("right_object_flex must be rigid")
        hand_flex_channels: dict[int, str] = {}
        hand_flex_bodies: dict[int, int] = {}
        for flex in range(model.nflex):
            flex_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_FLEX, flex,
            ) or ""
            if not (
                flex_name.startswith("right_hand_")
                and flex_name.endswith("_flex")
            ):
                continue
            link_name = flex_name.removesuffix("_flex")
            hand_flex_channels[flex] = next(
                (finger for finger in FINGERS if finger in link_name),
                "palm",
            )
            hand_flex_bodies[flex] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, link_name,
            )
            if (
                hand_flex_bodies[flex] < 0
                or not bool(model.flex_rigid[flex])
            ):
                raise ValueError(f"invalid rigid hand flex {flex_name!r}")
        use_hand_rigid_flex = bool(hand_flex_channels)
        contact_kind = (
            "object_rigid_flex" if use_rigid_flex
            else "hand_rigid_flex" if use_hand_rigid_flex
            else "geom"
        )
        if scene_object_contact_kind is None:
            scene_object_contact_kind = contact_kind
        elif scene_object_contact_kind != contact_kind:
            raise AssertionError("scene contact representation changed between profiles")
        table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        hand_geoms = {
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
            .startswith("collision_hand_")
        }
        object_pairs: dict[frozenset[int], int] = {}
        hand_pairs: list[int] = []
        hand_pair_channels: dict[int, str] = {}
        for pair in range(model.npair):
            geoms = frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair])))
            object_pairs[geoms] = pair
            if object_geom in geoms and geoms & hand_geoms:
                hand_pairs.append(pair)
                hand_geom = next(iter(geoms & hand_geoms))
                hand_name = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom,
                    ) or ""
                )
                hand_pair_channels[pair] = next(
                    (finger for finger in FINGERS if finger in hand_name),
                    "palm",
                )
        table_pair = object_pairs[frozenset((table_geom, object_geom))]
        activation_clearance = float(
            profile.get("activation_clearance_m", 0.0),
        )
        activation_mode = str(profile.get("activation_mode", "fixed"))
        if activation_mode not in ("fixed", "physx_velocity_prediction"):
            raise ValueError(f"unknown activation mode {activation_mode!r}")
        if activation_clearance < 0.0:
            raise ValueError("activation clearance must be nonnegative")
        for pair in hand_pairs:
            if activation_clearance > float(model.pair_margin[pair]):
                raise ValueError("activation clearance exceeds pair margin")
            model.pair_gap[pair] = (
                model.pair_margin[pair] - activation_clearance
            )
        for flex in hand_flex_channels:
            if activation_clearance > float(model.flex_margin[flex]):
                raise ValueError("activation clearance exceeds hand flex margin")
            model.flex_gap[flex] = (
                model.flex_margin[flex] - activation_clearance
            )
        if use_rigid_flex:
            if activation_clearance > float(model.flex_margin[object_flex]):
                raise ValueError("activation clearance exceeds flex margin")
            model.flex_gap[object_flex] = (
                model.flex_margin[object_flex] - activation_clearance
            )
        table_damping = float(profile.get("damping", 16.0))
        model.pair_solref[table_pair] = (
            float(profile["time"]), table_damping,
        )
        model.pair_solimp[table_pair] = (
            float(profile["impedance"]), float(profile["impedance"]),
            0.001, 0.5, 2.0,
        )
        hand_time = float(profile.get("hand_time", profile["time"]))
        hand_damping = float(profile.get("hand_damping", table_damping))
        hand_impedance = float(
            profile.get("hand_impedance", profile["impedance"]),
        )
        channel_damping = profile.get("channel_damping", {})
        direct_stiffness = profile.get("hand_direct_stiffness")
        direct_damping = profile.get("hand_direct_damping")
        hand_condim = int(profile.get("hand_condim", 3))
        if hand_condim not in (3, 4):
            raise ValueError("hand contact dimension must be 3 or 4")
        if use_rigid_flex and channel_damping:
            raise ValueError(
                "one rigid flex cannot represent different response per finger",
            )
        if use_rigid_flex and hand_condim == 4:
            raise ValueError(
                "one rigid flex cannot represent different torsional radii per finger",
            )
        for pair in hand_pairs:
            model.pair_dim[pair] = hand_condim
            if direct_stiffness is not None or direct_damping is not None:
                if (
                    direct_stiffness is None or direct_damping is None
                    or float(direct_stiffness) <= 0.0
                    or float(direct_damping) <= 0.0
                ):
                    raise ValueError("direct contact coefficients must be positive")
                model.pair_solref[pair] = (
                    -float(direct_stiffness), -float(direct_damping),
                )
            else:
                pair_damping = float(
                    channel_damping.get(hand_pair_channels[pair], hand_damping),
                )
                model.pair_solref[pair] = (hand_time, pair_damping)
            model.pair_solimp[pair] = (
                hand_impedance, hand_impedance, 0.001, 0.5, 2.0,
            )
        if use_rigid_flex:
            model.flex_condim[object_flex] = hand_condim
            if direct_stiffness is not None or direct_damping is not None:
                if (
                    direct_stiffness is None or direct_damping is None
                    or float(direct_stiffness) <= 0.0
                    or float(direct_damping) <= 0.0
                ):
                    raise ValueError("direct contact coefficients must be positive")
                model.flex_solref[object_flex] = (
                    -float(direct_stiffness), -float(direct_damping),
                )
            else:
                model.flex_solref[object_flex] = (hand_time, hand_damping)
            model.flex_solimp[object_flex] = (
                hand_impedance, hand_impedance, 0.001, 0.5, 2.0,
            )
        for flex, flex_channel in hand_flex_channels.items():
            model.flex_condim[flex] = hand_condim
            if direct_stiffness is not None or direct_damping is not None:
                if (
                    direct_stiffness is None or direct_damping is None
                    or float(direct_stiffness) <= 0.0
                    or float(direct_damping) <= 0.0
                ):
                    raise ValueError("direct contact coefficients must be positive")
                model.flex_solref[flex] = (
                    -float(direct_stiffness), -float(direct_damping),
                )
            else:
                flex_damping = float(
                    channel_damping.get(flex_channel, hand_damping),
                )
                model.flex_solref[flex] = (hand_time, flex_damping)
            model.flex_solimp[flex] = (
                hand_impedance, hand_impedance, 0.001, 0.5, 2.0,
            )
        scene_gravity = model.opt.gravity.copy()
        drive = PhysxTgsForceDrive(
            model, mujoco, names, stiffness=1000.0, damping=100.0,
        )
        record_joint_wrench = bool(profile.get("record_joint_wrench", False))
        active_link_names = [link for link, _ in ACTIVE_LINK_JOINTS]
        active_joint_names = [joint for _, joint in ACTIVE_LINK_JOINTS]
        active_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            for link in active_link_names
        ]
        active_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for joint in active_joint_names
        ]
        if (
            any(value < 0 for value in active_body_ids + active_joint_ids)
            or any(joint not in names for joint in active_joint_names)
        ):
            raise ValueError("active SAPIEN/MuJoCo joint-wrench mapping is incomplete")
        active_drive_axes = [names.index(joint) for joint in active_joint_names]
        active_joint_axes_child = np.asarray(
            [model.jnt_axis[joint] for joint in active_joint_ids],
            dtype=np.float64,
        )
        if record_joint_wrench:
            if source_joint_wrench is None or any(
                link not in source_link_order for link in active_link_names
            ):
                raise ValueError("source trace lacks active incoming-joint wrenches")
            if any(link not in loader_frame_by_link for link in active_link_names):
                raise ValueError("SAPIEN loader-frame probe lacks active links")
            for link_name, joint_name in ACTIVE_LINK_JOINTS:
                if loader_frame_by_link[link_name][0] != joint_name:
                    raise ValueError(
                        f"SAPIEN loader maps {link_name!r} to an unexpected joint",
                    )
            active_source_links = [
                source_link_order.index(link) for link in active_link_names
            ]
            active_loader_poses = np.asarray([
                loader_frame_by_link[link][1] for link in active_link_names
            ], dtype=np.float64)
            active_joint_axes_loader = np.empty_like(active_joint_axes_child)
            for active_index, pose in enumerate(active_loader_poses):
                rotation_child_from_joint = Rotation.from_quat(
                    pose[3:], scalar_first=True,
                ).as_matrix()
                active_joint_axes_loader[active_index] = (
                    rotation_child_from_joint.T
                    @ active_joint_axes_child[active_index]
                )
        else:
            active_source_links = []
            active_loader_poses = np.empty((0, 7), dtype=np.float64)
            active_joint_axes_loader = np.empty((0, 3), dtype=np.float64)
        predicted_pose = np.empty((len(rows), 7), dtype=np.float64)
        predicted_linear = np.empty((len(rows), 3), dtype=np.float64)
        predicted_angular = np.empty((len(rows), 3), dtype=np.float64)
        predicted_robot = np.empty((len(rows), 18), dtype=np.float64)
        recorded_normal_impulse = np.zeros((len(rows), 3), dtype=np.float64)
        recorded_full_impulse = np.zeros((len(rows), 3), dtype=np.float64)
        recorded_channel_normal_impulse = np.zeros(
            (len(rows), len(channel_order), 3), dtype=np.float64,
        )
        recorded_joint_wrench_average = np.zeros(
            (len(rows), len(ACTIVE_LINK_JOINTS), 6), dtype=np.float64,
        )
        recorded_joint_wrench_final = np.zeros_like(
            recorded_joint_wrench_average,
        )
        recorded_drive_force_average = np.zeros(
            (len(rows), 18), dtype=np.float64,
        )
        recorded_limit_force_average = np.zeros_like(
            recorded_drive_force_average,
        )
        initial_object_linear_after_decay = np.zeros(
            (len(rows), 3), dtype=np.float64,
        )
        hand_static_microsteps = 0
        hand_dynamic_microsteps = 0
        predicted_clearance_by_channel: dict[str, list[float]] = {
            name: [] for name in channel_order
        }
        instability: dict[str, int] | None = None
        for sample, row in enumerate(rows):
            previous = int(row - 1)
            model.opt.gravity[:] = scene_gravity
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = source_robot[previous]
            robot_velocity_mode = str(
                profile.get("robot_velocity_mode", "sapien_api"),
            )
            if robot_velocity_mode == "sapien_api":
                data.qvel[dof_addresses] = source_robot_velocity[previous]
            elif robot_velocity_mode == "backward_qpos_difference":
                if previous < 1:
                    raise ValueError("backward robot velocity needs two source states")
                data.qvel[dof_addresses] = (
                    source_robot[previous] - source_robot[previous - 1]
                ) / DT
            elif robot_velocity_mode == "zero":
                data.qvel[dof_addresses] = 0.0
            else:
                raise ValueError(
                    f"unknown robot velocity mode {robot_velocity_mode!r}",
                )
            data.qpos[object_qpos : object_qpos + 7] = source_pose[previous]
            object_velocity_mode = str(
                profile.get("object_velocity_mode", "sapien_api"),
            )
            if object_velocity_mode == "sapien_api":
                data.qvel[object_dof : object_dof + 3] = source_linear[previous]
                data.qvel[object_dof + 3 : object_dof + 6] = (
                    source_angular[previous]
                )
            elif object_velocity_mode == "backward_pose_difference":
                if previous < 1:
                    raise ValueError("backward object velocity needs two source states")
                data.qvel[object_dof : object_dof + 3] = (
                    source_pose[previous, :3]
                    - source_pose[previous - 1, :3]
                ) / DT
                rotation_delta = (
                    Rotation.from_quat(
                        source_pose[previous, 3:], scalar_first=True,
                    )
                    * Rotation.from_quat(
                        source_pose[previous - 1, 3:], scalar_first=True,
                    ).inv()
                )
                data.qvel[object_dof + 3 : object_dof + 6] = (
                    rotation_delta.as_rotvec() / DT
                )
            elif object_velocity_mode == "zero":
                data.qvel[object_dof : object_dof + 6] = 0.0
            else:
                raise ValueError(
                    f"unknown object velocity mode {object_velocity_mode!r}",
                )
            drive.set_target(source_target[row])
            apply_physx_velocity_decay(
                data.qvel, object_dof, linear_rate=linear_damping,
                angular_rate=angular_damping,
            )
            initial_object_linear_after_decay[sample] = (
                data.qvel[object_dof : object_dof + 3]
            )
            gravity_timing = str(
                profile.get("gravity_timing", "distributed"),
            )
            if gravity_timing == "physx_frame_start":
                # PhysX TGS defaults to applying gravity/external forces once
                # at the beginning of the full 1/240 s frame.  Isolate the
                # generalized acceleration caused by gravity, apply its full
                # frame impulse, then keep gravity off during the 25 internal
                # TGS-sized substeps.  Contact and velocity-dependent dynamics
                # are still recomputed every substep.
                apply_physx_frame_start_gravity(
                    model, data, mujoco, scene_gravity, source_dt=DT,
                )
            elif gravity_timing != "distributed":
                raise ValueError(f"unknown gravity timing {gravity_timing!r}")
            mujoco.mj_forward(model, data)
            if activation_mode == "physx_velocity_prediction":
                # PhysX speculative contacts become load-bearing when the
                # current normal closing speed predicts contact within the
                # full 1/240 s frame.  MuJoCo exposes only one activation gap
                # per explicit geom pair, so use the largest pointwise
                # prediction on that pair and hold it for the 25 substeps.
                relevant_pairs = hand_pairs + [table_pair]
                for pair in relevant_pairs:
                    model.pair_gap[pair] = model.pair_margin[pair]
                mujoco.mj_forward(model, data)
                pair_clearance = {pair: 0.0 for pair in relevant_pairs}
                for contact_index in range(data.ncon):
                    current = data.contact[contact_index]
                    geoms = frozenset((int(current.geom1), int(current.geom2)))
                    pair = object_pairs.get(geoms)
                    if pair not in pair_clearance:
                        continue
                    body1 = int(model.geom_bodyid[int(current.geom1)])
                    body2 = int(model.geom_bodyid[int(current.geom2)])
                    relative = body_point_velocity(
                        model, data, body2, current.pos,
                    ) - body_point_velocity(model, data, body1, current.pos)
                    normal = np.asarray(current.frame[:3], dtype=np.float64)
                    predicted = max(0.0, -float(relative @ normal) * DT)
                    pair_clearance[pair] = min(
                        float(model.pair_margin[pair]),
                        max(pair_clearance[pair], predicted),
                    )
                for pair, clearance in pair_clearance.items():
                    model.pair_gap[pair] = model.pair_margin[pair] - clearance
                    channel = (
                        "table" if pair == table_pair
                        else hand_pair_channels[pair]
                    )
                    predicted_clearance_by_channel.setdefault(channel, []).append(
                        clearance
                    )
                mujoco.mj_forward(model, data)
            drive.begin_frame(data)
            sample_normal_impulse = np.zeros(3, dtype=np.float64)
            sample_full_impulse = np.zeros(3, dtype=np.float64)
            sample_channel_normal_impulse = np.zeros(
                (len(channel_order), 3), dtype=np.float64,
            )
            sample_joint_wrench_impulse = np.zeros(
                (len(ACTIVE_LINK_JOINTS), 6), dtype=np.float64,
            )
            sample_joint_wrench_final = np.zeros_like(
                sample_joint_wrench_impulse,
            )
            sample_drive_impulse = np.zeros(18, dtype=np.float64)
            sample_limit_impulse = np.zeros(18, dtype=np.float64)

            def record_solver_forces(
                current_model: mujoco.MjModel, current_data: mujoco.MjData,
                duration_s: float,
            ) -> None:
                duration_s = float(duration_s)
                if not np.isfinite(duration_s) or duration_s <= 0.0:
                    raise ValueError("solver-force recording duration is invalid")
                if record_joint_wrench:
                    measured = incoming_joint_wrenches_at_loader_frames(
                        current_model, current_data, mujoco,
                        active_body_ids, active_joint_ids,
                        active_loader_poses,
                    )
                    sample_joint_wrench_impulse[:] += duration_s * measured
                    sample_joint_wrench_final[:] = measured
                center = np.asarray(current_data.xpos[object_body])
                for current_index in range(current_data.ncon):
                    current = current_data.contact[current_index]
                    current_geoms = {
                        int(current.geom1), int(current.geom2),
                    }
                    current_flexes = {
                        int(value) for value in np.asarray(current.flex)
                    }
                    if (
                        object_geom not in current_geoms
                        and not (
                            use_rigid_flex and object_flex in current_flexes
                        )
                    ) or (
                        int(current.efc_address) < 0
                    ):
                        continue
                    force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(
                        current_model, current_data, current_index, force,
                    )
                    normal = np.asarray(current.frame[:3], dtype=np.float64)
                    toward_object = center - np.asarray(current.pos)
                    if float(normal @ toward_object) < 0.0:
                        normal = -normal
                    world_force = (
                        np.asarray(current.frame, dtype=np.float64).reshape(3, 3).T
                        @ force[:3]
                    )
                    if float(world_force @ toward_object) < 0.0:
                        world_force = -world_force
                    sample_normal_impulse[:] += (
                        max(0.0, float(force[0])) * duration_s * normal
                    )
                    sample_full_impulse[:] += duration_s * world_force
                    if (
                        object_geom in current_geoms
                        and table_geom in current_geoms
                    ):
                        contact_channel = "table"
                    else:
                        current_hand_flexes = [
                            flex for flex in current_flexes
                            if flex in hand_flex_channels
                        ]
                        current_hand_geoms = current_geoms & hand_geoms
                        if current_hand_flexes:
                            contact_channel = hand_flex_channels[
                                current_hand_flexes[0]
                            ]
                        elif current_hand_geoms:
                            current_hand_geom = next(iter(current_hand_geoms))
                            current_hand_name = (
                                mujoco.mj_id2name(
                                    current_model, mujoco.mjtObj.mjOBJ_GEOM,
                                    current_hand_geom,
                                ) or ""
                            )
                            contact_channel = next((
                                finger for finger in FINGERS
                                if finger in current_hand_name
                            ), "palm")
                        else:
                            contact_channel = "other"
                    channel_index = channel_indices.get(
                        contact_channel, channel_indices.get("other"),
                    )
                    if channel_index is not None:
                        sample_channel_normal_impulse[channel_index] += (
                            max(0.0, float(force[0])) * duration_s * normal
                        )

            for substep in range(SUBSTEPS):
                for pair in hand_pairs:
                    model.pair_friction[pair, :2] = 0.5
                    model.pair_friction[pair, 2:] = 0.0
                if use_rigid_flex:
                    model.flex_friction[object_flex, :2] = 0.5
                    model.flex_friction[object_flex, 2:] = 0.0
                for flex in hand_flex_channels:
                    model.flex_friction[flex, :2] = 0.5
                    model.flex_friction[flex, 2:] = 0.0
                model.pair_friction[table_pair, :2] = 0.75
                hand_static_this_step = False
                hand_dynamic_this_step = False
                flex_hand_frictions: dict[int, list[float]] = {
                    flex: [] for flex in hand_flex_channels
                }
                if use_rigid_flex:
                    flex_hand_frictions[object_flex] = []
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geoms = frozenset((int(contact.geom1), int(contact.geom2)))
                    flexes = {
                        int(value) for value in np.asarray(contact.flex)
                    }
                    if geoms == frozenset((table_geom, object_geom)):
                        other_body = int(model.geom_bodyid[table_geom])
                        threshold = float(profile["table_threshold"])
                        pair = table_pair
                        dynamic, static = 0.75, 0.85
                        hand_contact = False
                        contact_flex = None
                    elif object_geom in geoms and geoms & hand_geoms:
                        hand_geom = next(iter(geoms & hand_geoms))
                        other_body = int(model.geom_bodyid[hand_geom])
                        threshold = float(profile["hand_threshold"])
                        pair = object_pairs[geoms]
                        dynamic = float(
                            profile.get("hand_dynamic_coefficient", 0.5),
                        )
                        static = float(
                            profile.get("hand_static_coefficient", 0.7),
                        )
                        hand_contact = True
                        hand_channel = hand_pair_channels[pair]
                        contact_flex = None
                    elif (
                        use_rigid_flex
                        and object_flex in flexes
                        and geoms & hand_geoms
                    ):
                        hand_geom = next(iter(geoms & hand_geoms))
                        other_body = int(model.geom_bodyid[hand_geom])
                        threshold = float(profile["hand_threshold"])
                        pair = None
                        dynamic = float(
                            profile.get("hand_dynamic_coefficient", 0.5),
                        )
                        static = float(
                            profile.get("hand_static_coefficient", 0.7),
                        )
                        hand_contact = True
                        contact_flex = object_flex
                        hand_name = (
                            mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom,
                            ) or ""
                        )
                        hand_channel = next(
                            (finger for finger in FINGERS if finger in hand_name),
                            "palm",
                        )
                    elif object_geom in geoms and any(
                        flex in hand_flex_channels for flex in flexes
                    ):
                        contact_flex = next(
                            flex for flex in flexes
                            if flex in hand_flex_channels
                        )
                        other_body = hand_flex_bodies[contact_flex]
                        threshold = float(profile["hand_threshold"])
                        pair = None
                        dynamic = float(
                            profile.get("hand_dynamic_coefficient", 0.5),
                        )
                        static = float(
                            profile.get("hand_static_coefficient", 0.7),
                        )
                        hand_contact = True
                        hand_channel = hand_flex_channels[contact_flex]
                    else:
                        continue
                    relative = body_point_velocity(
                        model, data, object_body, contact.pos,
                    ) - body_point_velocity(model, data, other_body, contact.pos)
                    normal = np.asarray(contact.frame[:3], dtype=np.float64)
                    tangential = relative - float(relative @ normal) * normal
                    if hand_contact:
                        friction_mode = str(
                            profile.get("hand_friction_mode", "speed_threshold"),
                        )
                        if friction_mode == "fixed_static":
                            friction = static
                        elif friction_mode == "fixed_dynamic":
                            friction = dynamic
                        elif friction_mode == "speed_threshold":
                            friction = friction_for_speed(
                                float(np.linalg.norm(tangential)),
                                threshold=threshold, dynamic=dynamic,
                                static=static,
                            )
                        else:
                            raise ValueError(
                                f"unknown hand friction mode {friction_mode!r}",
                            )
                        hand_static_this_step |= friction == static
                        hand_dynamic_this_step |= friction == dynamic
                        if hand_condim == 4:
                            radius = float(channel_patch_radius.get(
                                hand_channel, 0.0,
                            ))
                            torsional = (
                                friction * radius * float(
                                    profile.get("torsional_radius_scale", 1.0),
                                )
                            )
                            if pair is not None:
                                model.pair_friction[pair, 2] = torsional
                            else:
                                assert contact_flex is not None
                                model.flex_friction[contact_flex, 2] = torsional
                    else:
                        friction = friction_for_speed(
                            float(np.linalg.norm(tangential)),
                            threshold=threshold, dynamic=dynamic, static=static,
                        )
                    if pair is None:
                        assert contact_flex is not None
                        flex_hand_frictions[contact_flex].append(friction)
                    else:
                        model.pair_friction[pair, :2] = friction
                for flex, friction_values in flex_hand_frictions.items():
                    if not friction_values:
                        continue
                    # A flex has one material for its whole surface.  If any
                    # finger is sliding, use the dynamic coefficient, matching
                    # the conservative branch of the PhysX material law.
                    model.flex_friction[flex, :2] = min(friction_values)
                hand_static_microsteps += int(hand_static_this_step)
                hand_dynamic_microsteps += int(hand_dynamic_this_step)
                force_recorder = (
                    record_solver_forces
                    if profile.get("record_impulse_decomposition", False)
                    or record_joint_wrench
                    else None
                )
                if profile["constraint_order"] == "contact_before_drive":
                    if force_recorder is not None:
                        drive_impulses, limit_impulses = (
                            advance_contact_before_drive_recorded(
                                model, data, drive, force_recorder,
                            )
                        )
                    else:
                        drive_impulses, limit_impulses = (
                            advance_physx_tgs_microstep(
                                model, data, mujoco, drive,
                            )
                        )
                elif profile["constraint_order"] == "drive_before_contact":
                    drive_impulses, limit_impulses = advance_drive_before_contact(
                        model, data, drive, force_recorder,
                    )
                elif (
                    profile["constraint_order"]
                    == "symmetric_contact_drive_contact"
                ):
                    drive_impulses, limit_impulses = (
                        advance_symmetric_contact_drive_contact(
                            model, data, drive, force_recorder,
                        )
                    )
                elif profile["constraint_order"] == "coupled_drive_force":
                    drive_impulses, limit_impulses = advance_coupled_drive_force(
                        model, data, drive, force_recorder,
                    )
                else:
                    raise ValueError(
                        f"unknown constraint order {profile['constraint_order']!r}",
                    )
                sample_drive_impulse += drive_impulses
                sample_limit_impulse += limit_impulses
                object_quat = data.qpos[object_qpos + 3 : object_qpos + 7]
                if (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or float(np.linalg.norm(object_quat)) < 0.5
                    or any(int(warning.number) > 0 for warning in data.warning)
                ):
                    instability = {
                        "sample_index": sample,
                        "source_row": int(row),
                        "substep": substep,
                    }
                    break
            if instability is not None:
                break
            if bool(profile.get("final_velocity_iteration", False)):
                force_recorder = (
                    record_solver_forces
                    if profile.get("record_impulse_decomposition", False)
                    or record_joint_wrench
                    else None
                )
                apply_contact_velocity_iteration(
                    model, data, drive.dt, force_recorder,
                )
            predicted_pose[sample] = data.qpos[object_qpos : object_qpos + 7]
            predicted_linear[sample] = data.qvel[object_dof : object_dof + 3]
            predicted_angular[sample] = data.qvel[object_dof + 3 : object_dof + 6]
            predicted_robot[sample] = data.qpos[qpos_addresses]
            recorded_normal_impulse[sample] = sample_normal_impulse
            recorded_full_impulse[sample] = sample_full_impulse
            recorded_channel_normal_impulse[sample] = (
                sample_channel_normal_impulse
            )
            recorded_joint_wrench_average[sample] = (
                sample_joint_wrench_impulse / DT
            )
            recorded_joint_wrench_final[sample] = sample_joint_wrench_final
            recorded_drive_force_average[sample] = sample_drive_impulse / DT
            recorded_limit_force_average[sample] = sample_limit_impulse / DT
        if instability is not None:
            profile_reports.append({
                **profile,
                "stability_passed": False,
                "instability": instability,
                "hand_static_contact_microsteps": hand_static_microsteps,
                "hand_dynamic_contact_microsteps": hand_dynamic_microsteps,
            })
            continue
        unilateral = rows < 983
        report = {
            **profile,
            "stability_passed": True,
            "hand_static_contact_microsteps": hand_static_microsteps,
            "hand_dynamic_contact_microsteps": hand_dynamic_microsteps,
            "all_contact": summarize(
                rows, predicted_pose, predicted_linear, predicted_angular,
                predicted_robot, source_pose, source_linear, source_angular,
                source_robot,
            ),
            "early_one_sided_contact": summarize(
                rows[unilateral], predicted_pose[unilateral],
                predicted_linear[unilateral], predicted_angular[unilateral],
                predicted_robot[unilateral], source_pose, source_linear,
                source_angular, source_robot,
            ),
            "opposed_or_later": summarize(
                rows[~unilateral], predicted_pose[~unilateral],
                predicted_linear[~unilateral], predicted_angular[~unilateral],
                predicted_robot[~unilateral], source_pose, source_linear,
                source_angular, source_robot,
            ),
        }
        if activation_mode == "physx_velocity_prediction":
            report["predicted_activation_clearance_m"] = {
                name: {
                    "sample_count": len(values),
                    "positive_sample_count": int(np.count_nonzero(values)),
                    "mean": float(np.mean(values)) if values else 0.0,
                    "p95": float(np.percentile(values, 95)) if values else 0.0,
                    "maximum": float(np.max(values)) if values else 0.0,
                }
                for name, values in predicted_clearance_by_channel.items()
                if values
            }
        if profile.get("record_impulse_decomposition", False):
            previous_rows = rows - 1
            decay = 1.0 - linear_damping * DT
            gravity = np.asarray((0.0, 0.0, -9.81), dtype=np.float64)
            source_total_impulse = (
                object_mass * (
                    source_linear[rows] - decay * source_linear[previous_rows]
                ) - object_mass * gravity * DT
            )
            source_normal = np.sum(source_normal_impulse[rows], axis=1)
            # PhysX/SAPIEN's per-point ``impulse`` is the solver's normal
            # impulse.  The v4 trace resolves it against the reported normal;
            # the orthogonal component below is numerical leakage, not the
            # unreported friction impulse.  Recover total tangential impulse
            # from rigid-body momentum balance instead.
            source_reported_tangent_leakage = np.sum(
                source_tangent_impulse[rows], axis=1,
            )
            source_tangent = source_total_impulse - source_normal
            recorded_tangent = recorded_full_impulse - recorded_normal_impulse
            predicted_total_from_momentum = (
                object_mass * (
                    predicted_linear - initial_object_linear_after_decay
                ) - object_mass * gravity * DT
            )
            report["impulse_decomposition"] = {
                "source_total_vs_mujoco_recorded": vector_error(
                    recorded_full_impulse, source_total_impulse,
                ),
                "source_normal_vs_mujoco_normal": vector_error(
                    recorded_normal_impulse, source_normal,
                ),
                "source_tangent_vs_mujoco_tangent": vector_error(
                    recorded_tangent, source_tangent,
                ),
                "source_point_impulse_tangent_leakage": {
                    "maximum_norm_ns": float(np.max(np.linalg.norm(
                        source_reported_tangent_leakage, axis=1,
                    ))),
                    "sum_ns_xyz": np.sum(
                        source_reported_tangent_leakage, axis=0,
                    ).tolist(),
                    "interpretation": (
                        "PhysX point.impulse is normal-only; friction is the "
                        "rigid-body momentum residual"
                    ),
                },
                "recorded_vs_momentum_consistency": vector_error(
                    recorded_full_impulse, predicted_total_from_momentum,
                ),
                "normal_impulse_by_channel": {
                    name: {
                        **vector_error(
                            recorded_channel_normal_impulse[:, index],
                            source_normal_impulse[rows, index],
                        ),
                        "source_load_bearing_samples": int(np.count_nonzero(
                            channel_load[rows, index],
                        )),
                        "mujoco_norm_sum_ns": float(np.sum(np.linalg.norm(
                            recorded_channel_normal_impulse[:, index], axis=1,
                        ))),
                        "sapien_norm_sum_ns": float(np.sum(np.linalg.norm(
                            source_normal_impulse[rows, index], axis=1,
                        ))),
                        "mujoco_to_sapien_norm_sum_ratio": (
                            None if np.sum(np.linalg.norm(
                                source_normal_impulse[rows, index], axis=1,
                            )) <= 1.0e-12 else float(
                                np.sum(np.linalg.norm(
                                    recorded_channel_normal_impulse[:, index],
                                    axis=1,
                                )) / np.sum(np.linalg.norm(
                                    source_normal_impulse[rows, index], axis=1,
                                ))
                            )
                        ),
                    }
                    for name, index in channel_indices.items()
                },
            }
        if record_joint_wrench:
            assert source_joint_wrench is not None
            source_active_wrench = source_joint_wrench[rows][
                :, active_source_links, :
            ]
            custom_joint_force = (
                recorded_drive_force_average[:, active_drive_axes]
                + recorded_limit_force_average[:, active_drive_axes]
            )
            combined_joint_wrench = recorded_joint_wrench_average.copy()
            custom_rows_already_in_internal_wrench = (
                profile["constraint_order"] == "coupled_drive_force"
            )
            if not custom_rows_already_in_internal_wrench:
                combined_joint_wrench[:, :, 3:] += (
                    custom_joint_force[:, :, None]
                    * active_joint_axes_loader[None, :, :]
                )
            per_link: dict[str, object] = {}
            for active_index, link_name in enumerate(active_link_names):
                axis = active_joint_axes_loader[active_index]
                source_axial = (
                    source_active_wrench[:, active_index, 3:] @ axis
                )
                internal_axial = (
                    recorded_joint_wrench_average[:, active_index, 3:] @ axis
                )
                combined_axial = (
                    combined_joint_wrench[:, active_index, 3:] @ axis
                )
                custom_axial = custom_joint_force[:, active_index]
                per_link[link_name] = {
                    "incoming_internal_average_over_25_substeps": wrench_error(
                        recorded_joint_wrench_average[:, active_index],
                        source_active_wrench[:, active_index],
                    ),
                    "incoming_internal_at_final_substep": wrench_error(
                        recorded_joint_wrench_final[:, active_index],
                        source_active_wrench[:, active_index],
                    ),
                    "incoming_with_custom_rows_counted_once": wrench_error(
                        combined_joint_wrench[:, active_index],
                        source_active_wrench[:, active_index],
                    ),
                    "joint_axis_incoming_frame": axis.tolist(),
                    "axial_torque_nm": {
                        "sapien_mean_abs": float(np.mean(np.abs(source_axial))),
                        "mujoco_internal_mean_abs": float(
                            np.mean(np.abs(internal_axial))
                        ),
                        "custom_drive_plus_limit_mean_abs": float(
                            np.mean(np.abs(custom_axial))
                        ),
                        "mujoco_combined_mean_abs": float(
                            np.mean(np.abs(combined_axial))
                        ),
                        "mujoco_combined_rmse": rms(
                            combined_axial - source_axial
                        ),
                        "mujoco_combined_rmse_if_sign_flipped": rms(
                            combined_axial + source_axial
                        ),
                    },
                }
            report["joint_wrench_comparison"] = {
                "source_semantics": (
                    "PhysX total parent-to-child incoming-joint wrench, "
                    "sampled after the full 1/240 s frame"
                ),
                "mujoco_internal_semantics": (
                    "MuJoCo cfrc_int transformed from the root subtree center "
                    "of mass to the child incoming-joint frame"
                ),
                "average_semantics": (
                    "time-weighted average over 25 TGS-sized MuJoCo substeps"
                ),
                "custom_row_semantics": (
                    "the converted drive and hard-limit rows are velocity "
                    "impulses outside MuJoCo's native force solver; their "
                    "generalized impulse divided by 1/240 s is added only to "
                    "the matching joint-axis torque for split orders.  In the "
                    "coupled-drive profile qfrc_applied is already included in "
                    "cfrc_int, so it is deliberately not added a second time"
                ),
                "custom_rows_already_in_internal_wrench": (
                    custom_rows_already_in_internal_wrench
                ),
                "links": per_link,
            }
        profile_reports.append(report)
        prefix = f"{profile['name']}_{profile['constraint_order']}"
        saved[f"{prefix}_object_pose_wxyz"] = predicted_pose
        saved[f"{prefix}_object_linear_velocity"] = predicted_linear
        saved[f"{prefix}_object_angular_velocity"] = predicted_angular
        saved[f"{prefix}_robot_qpos"] = predicted_robot
        if profile.get("record_impulse_decomposition", False):
            saved[f"{prefix}_recorded_normal_impulse_ns"] = (
                recorded_normal_impulse
            )
            saved[f"{prefix}_recorded_full_impulse_ns"] = recorded_full_impulse
            saved[f"{prefix}_recorded_channel_normal_impulse_ns"] = (
                recorded_channel_normal_impulse
            )
            saved[f"{prefix}_source_total_impulse_ns"] = source_total_impulse
            saved[f"{prefix}_source_normal_impulse_ns"] = source_normal
        if record_joint_wrench:
            saved[f"{prefix}_joint_wrench_average_force_torque"] = (
                recorded_joint_wrench_average
            )
            saved[f"{prefix}_joint_wrench_final_force_torque"] = (
                recorded_joint_wrench_final
            )
            saved[f"{prefix}_drive_force_average"] = (
                recorded_drive_force_average
            )
            saved[f"{prefix}_limit_force_average"] = (
                recorded_limit_force_average
            )
            saved[f"{prefix}_source_joint_wrench_force_torque"] = (
                source_active_wrench
            )

    stable_reports = [
        item for item in profile_reports if item.get("stability_passed") is True
    ]
    if not stable_reports:
        raise RuntimeError("every one-step profile was numerically unstable")
    selected = min(
        stable_reports,
        key=lambda item: (
            float(item["early_one_sided_contact"]["linear_velocity_rmse_m_s"]),
            float(item["all_contact"]["linear_velocity_rmse_m_s"]),
        ),
    )
    output.mkdir(parents=True)
    trace_output = output / "one_step_predictions.npz"
    np.savez_compressed(
        trace_output, schema=np.asarray(SCHEMA), diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False), **saved,
    )
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "rendered": False,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "source_trace": str(trace_path), "source_trace_sha256": sha256(trace_path),
        "sapien_joint_frames": (
            str(joint_frame_path) if joint_frame_path is not None else None
        ),
        "sapien_joint_frames_sha256": (
            sha256(joint_frame_path) if joint_frame_path is not None else None
        ),
        "sample_count": int(len(rows)),
        "sample_policy": "all first 41 load rows plus 120 evenly spread later rows",
        "profile_focus": args.focus,
        "object_contact_kind": scene_object_contact_kind,
        "profiles": profile_reports,
        "selected_by_early_linear_response": {
            "material_profile": str(selected["name"]),
            "constraint_order": str(selected["constraint_order"]),
        },
        "trace": str(trace_output), "trace_sha256": sha256(trace_output),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
