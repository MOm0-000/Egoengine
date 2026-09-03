#!/usr/bin/env python3
"""Calibrate MuJoCo contact against a candidate-free SAPIEN v2 object probe.

The calibration first enforces two hard conditions: penetration must stay near
the PhysX value and the *net* contact impulse must use the same definition in
both engines.  Only profiles that pass those conditions are ranked by drop and
slide trajectory errors.  No grasp candidate or hand trajectory is executed.
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


SCHEMA = "deximit_mujoco_contact_calibration_v4_fixed_profile_diagnostic_only"
SOURCE_SCHEMA = "deximit_sapien_object_probe_v2_diagnostic_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - target))))


def rotation_rmse(actual: np.ndarray, target: np.ndarray) -> float:
    errors = (
        Rotation.from_quat(actual, scalar_first=True)
        * Rotation.from_quat(target, scalar_first=True).inv()
    ).magnitude()
    return float(np.sqrt(np.mean(np.square(errors))))


def relative_error(actual: float, target: float) -> float:
    if target <= 0.0:
        raise ValueError("relative-error target must be positive")
    return abs(actual - target) / target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--substeps", type=int, default=25)
    parser.add_argument(
        "--integration-order", choices=("native", "physx_tgs"),
        default="native",
        help="Use MuJoCo's native step or contact-then-drive TGS microstep order.",
    )
    parser.add_argument(
        "--cone", choices=("pyramidal", "elliptic"), default="pyramidal",
    )
    parser.add_argument("--noslip-iterations", type=int, default=0)
    parser.add_argument("--fixed-time-constant", type=float)
    parser.add_argument("--fixed-damping-ratio", type=float)
    parser.add_argument("--fixed-impedance", type=float)
    args = parser.parse_args()
    scene_path = args.scene.expanduser().resolve(strict=True)
    probe_path = args.sapien_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite contact calibration {output}")
    fixed_values = (
        args.fixed_time_constant, args.fixed_damping_ratio, args.fixed_impedance,
    )
    fixed_count = sum(value is not None for value in fixed_values)
    if (
        args.substeps <= 0 or args.noslip_iterations < 0
        or fixed_count not in (0, 3)
        or (
            fixed_count == 3
            and (
                not np.isfinite(np.asarray(fixed_values, dtype=float)).all()
                or float(args.fixed_time_constant) <= 0.0
                or float(args.fixed_damping_ratio) <= 0.0
                or not 0.0 < float(args.fixed_impedance) <= 1.0
            )
        )
    ):
        raise ValueError("substeps must be positive and noslip non-negative")

    source = np.load(probe_path, allow_pickle=False)
    if (
        str(np.asarray(source["schema"]).item()) != SOURCE_SCHEMA
        or not bool(np.asarray(source["diagnostic_only"]).item())
        or bool(np.asarray(source["formal_renderer_3_3_eligible"]).item())
        or bool(np.asarray(source["grasp_candidate_executed"]).item())
    ):
        raise ValueError("input is not the candidate-free SAPIEN v2 object probe")
    source_dt = float(np.asarray(source["physics_dt_s"]).item())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = source_dt / args.substeps
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    model.opt.cone = (
        mujoco.mjtCone.mjCONE_PYRAMIDAL
        if args.cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
    )
    model.opt.noslip_iterations = args.noslip_iterations
    data = mujoco.MjData(model)

    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
    )
    pair_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "table_right_object",
    )
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
    )
    if min(object_joint, pair_id, table_geom, object_geom) < 0:
        raise ValueError("exact scene lacks the object/table calibration contract")
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    for pair in range(model.npair):
        if pair != pair_id:
            model.pair_margin[pair] = -1.0
            model.pair_gap[pair] = 0.0
    model.pair_margin[pair_id] = 0.04
    model.pair_gap[pair_id] = 0.04
    model.pair_friction[pair_id, 2:] = 0.0

    initial = np.asarray(source["initial_pose"], dtype=np.float64)
    initial_quat = Rotation.from_matrix(
        initial[:3, :3],
    ).as_quat(scalar_first=True)
    velocity_decay = 1.0 - 20.0 * source_dt
    if not 0.0 < velocity_decay < 1.0:
        raise ValueError("SAPIEN object damping has no valid one-step multiplier")

    def reset(position: np.ndarray, quaternion: np.ndarray) -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[object_qpos : object_qpos + 3] = position
        data.qpos[object_qpos + 3 : object_qpos + 7] = (
            quaternion / np.linalg.norm(quaternion)
        )
        data.qvel[object_dof : object_dof + 6] = 0.0

    def advance_microstep() -> None:
        if args.integration_order == "native":
            mujoco.mj_step(model, data)
            return
        # The converted PhysX force drive is applied after the unconstrained
        # and contact velocity update.  This object-only calibration has no
        # robot drive, so retain the same ordering without adding an impulse.
        mujoco.mj_forward(model, data)
        data.qvel[:] += data.qacc * float(model.opt.timestep)
        mujoco.mj_integratePos(
            model, data.qpos, data.qvel, float(model.opt.timestep),
        )
        mujoco.mj_normalizeQuat(model, data.qpos)
        data.time += float(model.opt.timestep)
        mujoco.mj_forward(model, data)

    def rollout(
        frames: int, position: np.ndarray, quaternion: np.ndarray,
        linear_velocity: np.ndarray, angular_velocity: np.ndarray, *,
        gravity: bool, time_constant: float, damping_ratio: float,
        impedance: float, static_speed: float,
    ) -> dict[str, np.ndarray]:
        reset(position, quaternion)
        data.qvel[object_dof : object_dof + 3] = linear_velocity
        data.qvel[object_dof + 3 : object_dof + 6] = angular_velocity
        model.opt.gravity[:] = [0.0, 0.0, -9.81] if gravity else 0.0
        model.pair_solref[pair_id] = [time_constant, damping_ratio]
        # Equal first and second values make impedance constant over distance;
        # this isolates stiffness/damping from the default nonlinear ramp.
        model.pair_solimp[pair_id] = [impedance, impedance, 0.001, 0.5, 2.0]
        pose = np.empty((frames, 7), dtype=np.float64)
        linear = np.empty((frames, 3), dtype=np.float64)
        angular = np.empty((frames, 3), dtype=np.float64)
        separation = np.full(frames, np.nan, dtype=np.float64)
        impulse_norm_sum = np.zeros(frames, dtype=np.float64)
        impulse_net = np.zeros((frames, 3), dtype=np.float64)
        point_count = np.zeros(frames, dtype=np.int32)
        for frame in range(frames):
            # The source probe measured one PhysX damping multiplication per
            # 1/240 s step, not one multiplication per TGS iteration.
            data.qvel[object_dof : object_dof + 6] *= velocity_decay
            frame_separations: list[float] = []
            frame_norm_sum = 0.0
            frame_net = np.zeros(3, dtype=np.float64)
            frame_points = 0
            for _ in range(args.substeps):
                planar_speed = float(np.linalg.norm(
                    data.qvel[object_dof : object_dof + 2],
                ))
                friction = 0.85 if planar_speed <= static_speed else 0.75
                model.pair_friction[pair_id, :2] = friction
                advance_microstep()
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if {int(contact.geom1), int(contact.geom2)} != {
                        table_geom, object_geom,
                    }:
                        continue
                    frame_separations.append(float(contact.dist))
                    frame_points += 1
                    if int(contact.efc_address) < 0:
                        continue
                    local_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, contact_index, local_force)
                    world_force = (
                        np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T
                        @ local_force[:3]
                    )
                    point_impulse = world_force * float(model.opt.timestep)
                    frame_norm_sum += float(np.linalg.norm(point_impulse))
                    frame_net += point_impulse
            pose[frame] = np.concatenate((
                data.qpos[object_qpos : object_qpos + 3],
                data.qpos[object_qpos + 3 : object_qpos + 7],
            ))
            linear[frame] = data.qvel[object_dof : object_dof + 3]
            angular[frame] = data.qvel[object_dof + 3 : object_dof + 6]
            if frame_separations:
                separation[frame] = min(frame_separations)
            impulse_norm_sum[frame] = frame_norm_sum
            impulse_net[frame] = frame_net
            point_count[frame] = frame_points
        return {
            "pose": pose, "linear": linear, "angular": angular,
            "separation": separation, "impulse_norm_sum": impulse_norm_sum,
            "impulse_net": impulse_net, "point_count": point_count,
        }

    source_drop_pose = np.asarray(source["drop_pose_world_wxyz"])
    source_drop_linear = np.asarray(source["drop_linear_velocity"])
    source_drop_angular = np.asarray(source["drop_angular_velocity"])
    source_slide_pose = np.asarray(source["slide_pose_world_wxyz"])
    source_slide_linear = np.asarray(source["slide_linear_velocity"])
    source_slide_angular = np.asarray(source["slide_angular_velocity"])
    source_slide_relative = source_slide_pose[:, :3] - source_slide_pose[0, :3]
    source_minimum_separation = float(np.nanmin(
        source["drop_contact_separation_m_min"],
    ))
    source_peak_norm_sum = float(np.max(
        source["drop_contact_impulse_norm_sum_ns"],
    ))
    source_peak_net = float(np.max(np.linalg.norm(
        source["drop_contact_impulse_net_ns"], axis=1,
    )))
    drop_position = initial[:3, 3].copy()
    drop_position[2] += 0.05

    physical_rows: list[dict[str, object]] = []
    drop_traces: dict[tuple[float, float, float], dict[str, np.ndarray]] = {}
    # After replacing the input hulls with PhysX's actual cooked convex meshes,
    # the broad optimum moved to damping=24.  Extend that boundary locally so
    # the final choice cannot be a clipped search result.
    if fixed_count:
        time_constants = (float(args.fixed_time_constant),)
        damping_ratios = (float(args.fixed_damping_ratio),)
        impedances = (float(args.fixed_impedance),)
    else:
        time_constants = (0.00035, 0.0004, 0.00045, 0.0005, 0.0006)
        damping_ratios = (16.0, 24.0, 32.0, 48.0, 64.0)
        impedances = (0.75, 0.8, 0.85, 0.9, 0.95)
    for time_constant in time_constants:
        for damping_ratio in damping_ratios:
            for impedance in impedances:
                drop = rollout(
                    len(source_drop_pose), drop_position, initial_quat,
                    np.zeros(3), np.zeros(3), gravity=True,
                    time_constant=time_constant, damping_ratio=damping_ratio,
                    impedance=impedance, static_speed=0.01,
                )
                finite = drop["separation"][np.isfinite(drop["separation"])]
                minimum_separation = float(np.min(finite))
                peak_norm_sum = float(np.max(drop["impulse_norm_sum"]))
                peak_net = float(np.max(np.linalg.norm(drop["impulse_net"], axis=1)))
                separation_error = abs(minimum_separation - source_minimum_separation)
                norm_sum_error = relative_error(peak_norm_sum, source_peak_norm_sum)
                net_error = relative_error(peak_net, source_peak_net)
                penetration_pass = minimum_separation >= -2.5e-4
                impulse_pass = net_error <= 0.25
                drop_score = float(
                    rmse(drop["pose"][:, :3], source_drop_pose[:, :3]) / 5.0e-4
                    + rotation_rmse(drop["pose"][:, 3:], source_drop_pose[:, 3:]) / 0.05
                    + rmse(drop["linear"], source_drop_linear) / 5.0e-2
                    + rmse(drop["angular"], source_drop_angular) / 5.0e-1
                    + separation_error / 1.0e-4
                    + 0.5 * norm_sum_error + 0.5 * net_error
                )
                row = {
                    "time_constant_s": time_constant,
                    "damping_ratio": damping_ratio,
                    "constant_constraint_impedance": impedance,
                    "drop_minimum_separation_m": minimum_separation,
                    "source_drop_minimum_separation_m": source_minimum_separation,
                    "drop_minimum_separation_error_m": separation_error,
                    "drop_peak_impulse_norm_sum_ns": peak_norm_sum,
                    "source_drop_peak_impulse_norm_sum_ns": source_peak_norm_sum,
                    "drop_peak_impulse_norm_sum_relative_error": norm_sum_error,
                    "drop_peak_impulse_net_ns": peak_net,
                    "source_drop_peak_impulse_net_ns": source_peak_net,
                    "drop_peak_impulse_net_relative_error": net_error,
                    "drop_position_rmse_m": rmse(
                        drop["pose"][:, :3], source_drop_pose[:, :3],
                    ),
                    "drop_rotation_rmse_rad": rotation_rmse(
                        drop["pose"][:, 3:], source_drop_pose[:, 3:],
                    ),
                    "drop_linear_velocity_rmse_m_s": rmse(
                        drop["linear"], source_drop_linear,
                    ),
                    "drop_angular_velocity_rmse_rad_s": rmse(
                        drop["angular"], source_drop_angular,
                    ),
                    "penetration_hard_condition_passed": penetration_pass,
                    "impulse_hard_condition_passed": impulse_pass,
                    "hard_conditions_passed": penetration_pass and impulse_pass,
                    "drop_score": drop_score,
                }
                physical_rows.append(row)
                drop_traces[(time_constant, damping_ratio, impedance)] = drop

    passed_physical = [row for row in physical_rows if row["hard_conditions_passed"]]
    if not passed_physical:
        raise RuntimeError("no contact profile satisfies penetration and impulse conditions")
    passed_physical.sort(key=lambda row: float(row["drop_score"]))
    finalists = passed_physical[: min(16, len(passed_physical))]
    final_rows: list[dict[str, object]] = []
    slide_traces: dict[tuple[float, float, float, float], dict[str, np.ndarray]] = {}
    for physical in finalists:
        key3 = (
            float(physical["time_constant_s"]),
            float(physical["damping_ratio"]),
            float(physical["constant_constraint_impedance"]),
        )
        drop = drop_traces[key3]
        settled = drop["pose"][-1]
        for static_speed in (0.0, 0.001, 0.005, 0.01, 0.02):
            slide = rollout(
                len(source_slide_pose), settled[:3], settled[3:],
                np.asarray([0.2, 0.0, 0.0]), np.zeros(3), gravity=True,
                time_constant=key3[0], damping_ratio=key3[1],
                impedance=key3[2], static_speed=static_speed,
            )
            slide_relative = slide["pose"][:, :3] - slide["pose"][0, :3]
            slide_score = float(
                rmse(slide_relative, source_slide_relative) / 5.0e-4
                + rmse(slide["linear"], source_slide_linear) / 5.0e-2
                + rmse(slide["angular"], source_slide_angular) / 5.0e-1
            )
            row = {
                **physical,
                "static_friction_speed_threshold_m_s": static_speed,
                "dynamic_friction": 0.75,
                "static_friction": 0.85,
                "slide_relative_position_rmse_m": rmse(
                    slide_relative, source_slide_relative,
                ),
                "slide_linear_velocity_rmse_m_s": rmse(
                    slide["linear"], source_slide_linear,
                ),
                "slide_angular_velocity_rmse_rad_s": rmse(
                    slide["angular"], source_slide_angular,
                ),
                "slide_score": slide_score,
                "total_score": float(physical["drop_score"]) + slide_score,
            }
            final_rows.append(row)
            slide_traces[(*key3, static_speed)] = slide
    final_rows.sort(key=lambda row: float(row["total_score"]))
    selected = final_rows[0]
    selected_key3 = (
        float(selected["time_constant_s"]),
        float(selected["damping_ratio"]),
        float(selected["constant_constraint_impedance"]),
    )
    selected_key4 = (
        *selected_key3,
        float(selected["static_friction_speed_threshold_m_s"]),
    )
    selected_drop = drop_traces[selected_key3]
    selected_slide = slide_traces[selected_key4]

    output.mkdir(parents=True)
    trace_path = output / "mujoco_object_probe.npz"
    np.savez_compressed(
        trace_path, schema=np.asarray(SCHEMA), diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), source_dt_s=np.asarray(source_dt),
        mujoco_dt_s=np.asarray(model.opt.timestep), substeps=np.asarray(args.substeps),
        integration_order=np.asarray(args.integration_order),
        cone=np.asarray(args.cone),
        noslip_iterations=np.asarray(args.noslip_iterations),
        drop_pose_world_wxyz=selected_drop["pose"],
        drop_linear_velocity=selected_drop["linear"],
        drop_angular_velocity=selected_drop["angular"],
        drop_contact_separation_m_min=selected_drop["separation"],
        drop_contact_impulse_norm_sum_ns=selected_drop["impulse_norm_sum"],
        drop_contact_impulse_net_ns=selected_drop["impulse_net"],
        slide_pose_world_wxyz=selected_slide["pose"],
        slide_linear_velocity=selected_slide["linear"],
        slide_angular_velocity=selected_slide["angular"],
    )
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "scene": str(scene_path), "scene_sha256": sha256(scene_path),
        "sapien_probe": str(probe_path), "sapien_probe_sha256": sha256(probe_path),
        "mujoco_version": mujoco.__version__,
        "source_dt_s": source_dt,
        "mujoco_substep_dt_s": float(model.opt.timestep),
        "mujoco_substeps_per_source_step": args.substeps,
        "integration_order": args.integration_order,
        "friction_cone": args.cone,
        "noslip_iterations": args.noslip_iterations,
        "fixed_physical_profile_requested": bool(fixed_count),
        "tested_physical_profile_count": len(physical_rows),
        "hard_condition_pass_count": len(passed_physical),
        "finalist_physical_profile_count": len(finalists),
        "selected_contact_profile": selected,
        "selection_policy": (
            "candidate-free fixed physical profile with five slide thresholds; "
            if fixed_count else "candidate-free two-stage grid; "
        ) + (
            "require penetration >= -0.25 mm and like-for-like net peak impulse "
            "error <= 25%, then rank drop/slide traces"
        ),
        "all_physical_profiles": sorted(
            physical_rows,
            key=lambda row: (
                not bool(row["hard_conditions_passed"]), float(row["drop_score"]),
            ),
        ),
        "all_finalists": final_rows,
        "trace": str(trace_path), "trace_sha256": sha256(trace_path),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps({
        key: report[key] for key in (
            "schema", "diagnostic_only", "formal_renderer_3_3_eligible",
            "grasp_candidate_executed", "tested_physical_profile_count",
            "hard_condition_pass_count", "selected_contact_profile", "trace",
        )
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
