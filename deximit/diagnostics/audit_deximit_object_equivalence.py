#!/usr/bin/env python3
"""Fit MuJoCo object/table behavior to a candidate-free SAPIEN micro probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_errors(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(actual, scalar_first=True)
        * Rotation.from_quat(target, scalar_first=True).inv()
    ).magnitude()


def rmse(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - target))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--substeps", type=int, default=25)
    parser.add_argument(
        "--solver", choices=("scene", "newton", "cg", "pgs"), default="scene",
    )
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    sapien_probe = args.sapien_probe.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite object equivalence audit {output}")
    if args.substeps <= 0:
        raise ValueError("substeps must be positive")

    source = np.load(sapien_probe, allow_pickle=False)
    source_dt = float(np.asarray(source["physics_dt_s"]).item())
    model = mujoco.MjModel.from_xml_path(str(scene))
    if args.solver != "scene":
        model.opt.solver = int({
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "pgs": mujoco.mjtSolver.mjSOL_PGS,
        }[args.solver])
    model.opt.timestep = source_dt / args.substeps
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    data = mujoco.MjData(model)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
    )
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    table_object_pair = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_PAIR, "table_right_object",
    )
    if min(object_joint, table_object_pair) < 0:
        raise ValueError("scene lacks exact object joint or table-object pair")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
    )
    for pair in range(model.npair):
        if pair != table_object_pair:
            model.pair_margin[pair] = -1.0
            model.pair_gap[pair] = 0.0
    model.pair_margin[table_object_pair] = 0.04
    model.pair_gap[table_object_pair] = 0.04
    model.pair_friction[table_object_pair, 2:] = 0.0

    initial = np.asarray(source["initial_pose"], dtype=np.float64)
    initial_quat = Rotation.from_matrix(initial[:3, :3]).as_quat(scalar_first=True)
    full_step_decay = 1.0 - 20.0 * source_dt
    if not 0.0 < full_step_decay < 1.0:
        raise ValueError("recorded PhysX damping has no finite decay factor")

    def reset(position: np.ndarray, quat: np.ndarray) -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[object_qpos : object_qpos + 3] = position
        data.qpos[object_qpos + 3 : object_qpos + 7] = quat / np.linalg.norm(quat)
        data.qvel[object_dof : object_dof + 6] = 0.0

    def rollout(
        frames: int, position: np.ndarray, quat: np.ndarray,
        linear_velocity: np.ndarray, angular_velocity: np.ndarray, *,
        gravity: bool, time_constant: float, static_speed: float,
    ) -> dict[str, np.ndarray]:
        reset(position, quat)
        data.qvel[object_dof : object_dof + 3] = linear_velocity
        data.qvel[object_dof + 3 : object_dof + 6] = angular_velocity
        model.opt.gravity[:] = [0.0, 0.0, -9.81] if gravity else 0.0
        model.pair_solref[table_object_pair] = [time_constant, 1.0]
        pose = np.empty((frames, 7), dtype=np.float64)
        linear = np.empty((frames, 3), dtype=np.float64)
        angular = np.empty((frames, 3), dtype=np.float64)
        separation = np.full(frames, np.nan, dtype=np.float64)
        impulse = np.zeros(frames, dtype=np.float64)
        point_count = np.zeros(frames, dtype=np.int32)
        for frame in range(frames):
            # This is not an arbitrary stabilizer: the isolated SAPIEN probe
            # measures exactly one multiplier (1 - rate*dt) per 1/240 s step.
            data.qvel[object_dof : object_dof + 6] *= full_step_decay
            frame_separations: list[float] = []
            frame_impulse = 0.0
            frame_points = 0
            for _ in range(args.substeps):
                planar_speed = float(np.linalg.norm(
                    data.qvel[object_dof : object_dof + 2],
                ))
                friction = 0.85 if planar_speed <= static_speed else 0.75
                model.pair_friction[table_object_pair, :2] = friction
                mujoco.mj_step(model, data)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if {int(contact.geom1), int(contact.geom2)} != {
                        table_geom, object_geom,
                    }:
                        continue
                    frame_separations.append(float(contact.dist))
                    frame_points += 1
                    if int(contact.efc_address) >= 0:
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            model, data, contact_index, force,
                        )
                        frame_impulse += abs(float(force[0])) * float(model.opt.timestep)
            pose[frame] = np.concatenate((
                data.qpos[object_qpos : object_qpos + 3],
                data.qpos[object_qpos + 3 : object_qpos + 7],
            ))
            linear[frame] = data.qvel[object_dof : object_dof + 3]
            angular[frame] = data.qvel[object_dof + 3 : object_dof + 6]
            if frame_separations:
                separation[frame] = min(frame_separations)
            impulse[frame] = frame_impulse
            point_count[frame] = frame_points
        return {
            "pose": pose, "linear": linear, "angular": angular,
            "separation": separation, "impulse": impulse,
            "point_count": point_count,
        }

    high_position = initial[:3, 3].copy()
    high_position[2] += 0.25
    decay_linear = rollout(
        120, high_position, initial_quat, np.asarray([0.2, -0.1, 0.05]),
        np.zeros(3), gravity=False, time_constant=0.02, static_speed=0.0,
    )
    decay_angular = rollout(
        120, high_position, initial_quat, np.zeros(3),
        np.asarray([0.5, -0.3, 0.2]), gravity=False,
        time_constant=0.02, static_speed=0.0,
    )

    time_constants = (0.02, 0.015, 0.01, 0.0075, 0.005, 0.0035, 0.002)
    static_speeds = (0.0, 0.001, 0.005, 0.01, 0.02)
    rows: list[dict[str, object]] = []
    traces: dict[tuple[float, float], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    drop_position = initial[:3, 3].copy()
    drop_position[2] += 0.05
    source_drop_pose = np.asarray(source["drop_pose_world_wxyz"])
    source_drop_linear = np.asarray(source["drop_linear_velocity"])
    source_drop_angular = np.asarray(source["drop_angular_velocity"])
    source_slide_pose = np.asarray(source["slide_pose_world_wxyz"])
    source_slide_linear = np.asarray(source["slide_linear_velocity"])
    source_slide_angular = np.asarray(source["slide_angular_velocity"])
    source_slide_relative = source_slide_pose[:, :3] - source_slide_pose[0, :3]
    source_drop_peak_impulse = float(np.max(source["drop_contact_impulse_ns_max"]))

    for time_constant in time_constants:
        for static_speed in static_speeds:
            drop = rollout(
                len(source_drop_pose), drop_position, initial_quat,
                np.zeros(3), np.zeros(3), gravity=True,
                time_constant=time_constant, static_speed=static_speed,
            )
            settled = drop["pose"][-1]
            slide = rollout(
                len(source_slide_pose), settled[:3], settled[3:],
                np.asarray([0.2, 0.0, 0.0]), np.zeros(3), gravity=True,
                time_constant=time_constant, static_speed=static_speed,
            )
            slide_relative = slide["pose"][:, :3] - slide["pose"][0, :3]
            drop_position_rmse = rmse(drop["pose"][:, :3], source_drop_pose[:, :3])
            drop_linear_rmse = rmse(drop["linear"], source_drop_linear)
            drop_angular_rmse = rmse(drop["angular"], source_drop_angular)
            slide_relative_rmse = rmse(slide_relative, source_slide_relative)
            slide_linear_rmse = rmse(slide["linear"], source_slide_linear)
            slide_angular_rmse = rmse(slide["angular"], source_slide_angular)
            drop_peak_impulse = float(np.max(drop["impulse"]))
            impulse_relative_error = abs(
                drop_peak_impulse - source_drop_peak_impulse
            ) / source_drop_peak_impulse
            score = float(
                drop_position_rmse / 5.0e-4
                + drop_linear_rmse / 5.0e-2
                + drop_angular_rmse / 5.0e-1
                + slide_relative_rmse / 5.0e-4
                + slide_linear_rmse / 5.0e-2
                + slide_angular_rmse / 5.0e-1
                + 0.25 * impulse_relative_error
            )
            finite_separation = drop["separation"][np.isfinite(drop["separation"])]
            row = {
                "time_constant_s": time_constant,
                "static_friction_speed_threshold_m_s": static_speed,
                "dynamic_friction": 0.75,
                "static_friction": 0.85,
                "drop_position_rmse_m": drop_position_rmse,
                "drop_rotation_rmse_rad": float(np.sqrt(np.mean(np.square(
                    rotation_errors(drop["pose"][:, 3:], source_drop_pose[:, 3:]),
                )))),
                "drop_linear_velocity_rmse_m_s": drop_linear_rmse,
                "drop_angular_velocity_rmse_rad_s": drop_angular_rmse,
                "drop_minimum_separation_m": float(np.min(finite_separation)),
                "drop_peak_impulse_ns": drop_peak_impulse,
                "drop_peak_impulse_relative_error": impulse_relative_error,
                "slide_relative_position_rmse_m": slide_relative_rmse,
                "slide_linear_velocity_rmse_m_s": slide_linear_rmse,
                "slide_angular_velocity_rmse_rad_s": slide_angular_rmse,
                "score": score,
            }
            rows.append(row)
            traces[(time_constant, static_speed)] = (drop, slide)

    rows.sort(key=lambda row: float(row["score"]))
    selected = rows[0]
    key = (
        float(selected["time_constant_s"]),
        float(selected["static_friction_speed_threshold_m_s"]),
    )
    selected_drop, selected_slide = traces[key]
    output.mkdir(parents=True)
    trace_path = output / "mujoco_object_probe.npz"
    np.savez_compressed(
        trace_path,
        schema=np.asarray("deximit_mujoco_object_equivalence_v1_diagnostic_only"),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), source_dt_s=np.asarray(source_dt),
        mujoco_dt_s=np.asarray(model.opt.timestep), substeps=np.asarray(args.substeps),
        linear_decay_pose_world_wxyz=decay_linear["pose"],
        linear_decay_linear_velocity=decay_linear["linear"],
        angular_decay_pose_world_wxyz=decay_angular["pose"],
        angular_decay_angular_velocity=decay_angular["angular"],
        drop_pose_world_wxyz=selected_drop["pose"],
        drop_linear_velocity=selected_drop["linear"],
        drop_angular_velocity=selected_drop["angular"],
        drop_contact_separation_m_min=selected_drop["separation"],
        drop_contact_impulse_ns_sum=selected_drop["impulse"],
        slide_pose_world_wxyz=selected_slide["pose"],
        slide_linear_velocity=selected_slide["linear"],
        slide_angular_velocity=selected_slide["angular"],
    )
    report = {
        "schema": "deximit_mujoco_object_equivalence_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "scene": str(scene), "scene_sha256": sha256(scene),
        "sapien_probe": str(sapien_probe),
        "sapien_probe_sha256": sha256(sapien_probe),
        "mujoco_version": mujoco.__version__,
        "mujoco_solver": mujoco.mjtSolver(model.opt.solver).name,
        "source_dt_s": source_dt,
        "mujoco_substep_dt_s": float(model.opt.timestep),
        "mujoco_substeps_per_source_step": args.substeps,
        "physx_full_step_velocity_decay": full_step_decay,
        "damping_application": "multiply free-object velocity once before each source-sized step",
        "linear_decay_velocity_rmse_m_s": rmse(
            decay_linear["linear"], source["linear_decay_linear_velocity"],
        ),
        "linear_decay_position_rmse_m": rmse(
            decay_linear["pose"][:, :3], source["linear_decay_pose_world_wxyz"][:, :3],
        ),
        "angular_decay_velocity_rmse_rad_s": rmse(
            decay_angular["angular"], source["angular_decay_angular_velocity"],
        ),
        "selected_contact_profile": selected,
        "selection_policy": (
            "fixed candidate-free drop/slide grid; normalized pose, velocity, angular "
            "velocity and peak-impulse errors"
        ),
        "all_contact_profiles": rows,
        "irreducible_model_difference": (
            "PhysX TGS/PCM and MuJoCo convex multi-contact use different contact "
            "manifold and friction solvers; the selected profile is a measured compromise"
        ),
        "trace": str(trace_path), "trace_sha256": sha256(trace_path),
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
