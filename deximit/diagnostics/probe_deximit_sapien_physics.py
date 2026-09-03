#!/usr/bin/env python3
"""Measure DexImit's SAPIEN physics without executing any grasp candidate.

The probe removes the table and objects, then measures the response of the
released 18-joint robot to force, velocity, position-error, sustained
external-load, and hard joint-limit inputs at deterministic configurations.
This provides an engine-conversion target for MuJoCo without using candidate
21 as a parameter probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deximit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    deximit = args.deximit_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SAPIEN probe {output}")
    package = deximit / "third_party/any2dex/any2dex"
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(package / "third_party/BODex_api/src"))

    import sapien
    import yaml
    import env.base_env_for_render as render_env
    from env.base_env_for_render import BaseEnv

    class DisabledSyntheticPC:
        def __init__(self, *unused_args: object, **unused_kwargs: object) -> None:
            pass

    render_env.SyntheticPC = DisabledSyntheticPC

    def setup_raster_physics(self: object) -> None:
        sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
        sapien.physx.set_body_config(
            solver_position_iterations=25, solver_velocity_iterations=1,
            sleep_threshold=0.005,
        )
        sapien.physx.set_scene_config(
            gravity=np.array([0.0, 0.0, -9.81]), bounce_threshold=2.0,
            enable_pcm=True, enable_tgs=True, enable_ccd=False,
            enable_enhanced_determinism=False,
            enable_friction_every_iteration=True, cpu_workers=0,
        )
        sapien.physx.set_default_material(
            static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
        )
        self.scene = sapien.Scene([
            sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(),
        ])
        self.scene.set_timestep(self.timestep)
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_viewer_shader_dir("default")

    BaseEnv.set_up_physics_and_render = setup_raster_physics
    config_path = package / "env/config/env_hand.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env = BaseEnv(config)
    env.reset()

    # Contact-free drive response only.  This does not change the released
    # environment on disk and no candidate/object trajectory is executed.
    for entity in list(env.scene.get_entities()):
        if str(getattr(entity, "name", "")) == "table":
            env.scene.remove_entity(entity)
    for articulation in (env.robot_left, env.robot_right):
        for link in articulation.get_links():
            for shape in link.get_collision_shapes():
                shape.set_collision_groups((0, 0, 0, 0))

    robot = env.robot_right
    joints = list(env.active_joints_right)
    names = [str(joint.get_name()) for joint in joints]
    if len(names) != 18:
        raise ValueError(f"expected 18 right-robot joints, got {names}")
    q0 = np.asarray(robot.get_qpos(), dtype=np.float64)
    left_target = np.asarray(env.robot_left.get_qpos(), dtype=np.float64)
    dt = float(env.timestep)

    stiffness = np.asarray([float(joint.get_stiffness()) for joint in joints])
    damping = np.asarray([float(joint.get_damping()) for joint in joints])
    force_limit = np.asarray([float(joint.get_force_limit()) for joint in joints])
    friction = np.asarray([float(joint.get_friction()) for joint in joints])
    armature = np.asarray([
        float(np.asarray(joint.get_armature(), dtype=np.float64).reshape(-1)[0])
        for joint in joints
    ])
    limits = np.asarray([np.asarray(joint.get_limits(), dtype=np.float64)[0] for joint in joints])

    rng = np.random.default_rng(20260830)
    center = limits.mean(axis=1)
    half = 0.5 * (limits[:, 1] - limits[:, 0])
    configurations = [q0]
    for _ in range(6):
        configurations.append(center + 0.7 * half * rng.uniform(-1.0, 1.0, 18))
    configurations = np.asarray(configurations, dtype=np.float64)

    links = list(robot.get_links())
    link_names = [str(link.get_name()) for link in links]
    link_pose_world_wxyz = np.empty(
        (len(configurations), len(links), 7), dtype=np.float64,
    )
    for config_index, qpos in enumerate(configurations):
        robot.set_qpos(qpos)
        robot.set_qvel(np.zeros(18, dtype=np.float64))
        for link_index, link in enumerate(links):
            pose = link.get_entity_pose()
            link_pose_world_wxyz[config_index, link_index] = np.concatenate((
                np.asarray(pose.p, dtype=np.float64),
                np.asarray(pose.q, dtype=np.float64),
            ))

    def reset(qpos: np.ndarray, target: np.ndarray, velocity: np.ndarray) -> None:
        # PhysX warm-starts articulation constraints.  Teleporting directly
        # from one basis experiment to the next otherwise leaks the previous
        # experiment's cached drive impulse into the new result.  First let a
        # zero-load drive at the requested configuration replace that cache,
        # then restore the exact requested state before measuring.
        robot.set_qpos(qpos)
        robot.set_qvel(np.zeros(18, dtype=np.float64))
        robot.set_qf(np.zeros(18, dtype=np.float64))
        env.apply_action(np.concatenate((left_target, qpos)))
        for _ in range(8):
            env.scene.step()
        robot.set_qpos(qpos)
        robot.set_qvel(np.asarray(velocity, dtype=np.float64))
        robot.set_qf(np.zeros(18, dtype=np.float64))
        env.apply_action(np.concatenate((left_target, target)))

    zero = np.zeros(18, dtype=np.float64)
    zero_qpos = np.empty_like(configurations)
    zero_qvel = np.empty_like(configurations)
    velocity_response = np.empty((len(configurations), 18, 18), dtype=np.float64)
    position_response = np.empty((len(configurations), 18, 18), dtype=np.float64)
    velocity_qpos_response = np.empty_like(velocity_response)
    position_qpos_response = np.empty_like(position_response)
    # The XHand links are orders of magnitude lighter than the UR5e links.
    # A common basis amplitude drives fingers into PhysX's 100 rad/s cap and
    # no longer measures the linear drive.  Scale the two blocks separately.
    velocity_amplitude = np.concatenate((
        np.full(6, 1.0e-1, dtype=np.float64),
        np.full(12, 1.0e-5, dtype=np.float64),
    ))
    position_amplitude = np.concatenate((
        np.full(6, 1.0e-2, dtype=np.float64),
        np.full(12, 1.0e-6, dtype=np.float64),
    ))
    position_sign = np.ones((len(configurations), 18), dtype=np.float64)
    for config_index, qpos in enumerate(configurations):
        reset(qpos, qpos, zero)
        env.scene.step()
        zero_qpos[config_index] = np.asarray(robot.get_qpos())
        zero_qvel[config_index] = np.asarray(robot.get_qvel())
        for axis in range(18):
            basis = np.zeros(18, dtype=np.float64)
            basis[axis] = velocity_amplitude[axis]
            reset(qpos, qpos, basis)
            env.scene.step()
            velocity_response[config_index, :, axis] = (
                np.asarray(robot.get_qvel()) / velocity_amplitude[axis]
            )
            velocity_qpos_response[config_index, :, axis] = (
                (np.asarray(robot.get_qpos()) - zero_qpos[config_index])
                / velocity_amplitude[axis]
            )

            low, high = limits[axis]
            if qpos[axis] + position_amplitude[axis] > high:
                position_sign[config_index, axis] = -1.0
            target = qpos.copy()
            target[axis] += (
                position_sign[config_index, axis] * position_amplitude[axis]
            )
            if not low <= target[axis] <= high:
                raise ValueError(f"joint {names[axis]} lacks room for a drive basis probe")
            reset(qpos, target, zero)
            env.scene.step()
            position_response[config_index, :, axis] = (
                np.asarray(robot.get_qvel())
                / (position_sign[config_index, axis] * position_amplitude[axis])
            )
            position_qpos_response[config_index, :, axis] = (
                (np.asarray(robot.get_qpos()) - zero_qpos[config_index])
                / (position_sign[config_index, axis] * position_amplitude[axis])
            )

    # Remove the drive and measure inverse-mass response directly.  This
    # separates model/inertia differences from drive-algorithm differences.
    for joint in joints:
        joint.set_drive_property(
            stiffness=0.0, damping=0.0, force_limit=1.0e10, mode="force",
        )
    force_response = np.empty((len(configurations), 18, 18), dtype=np.float64)
    # Unit torque saturates PhysX's 100 rad/s articulation velocity cap on the
    # very light finger links.  Use small, recorded basis amplitudes and divide
    # them back out so this remains a linear inverse-mass measurement.
    force_amplitude = np.concatenate((
        np.full(6, 1.0e-2, dtype=np.float64),
        np.full(12, 1.0e-5, dtype=np.float64),
    ))
    for config_index, qpos in enumerate(configurations):
        for axis in range(18):
            reset(qpos, qpos, zero)
            applied = np.zeros(18, dtype=np.float64)
            applied[axis] = force_amplitude[axis]
            robot.set_qf(applied)
            env.scene.step()
            force_response[config_index, :, axis] = (
                np.asarray(robot.get_qvel()) / (dt * force_amplitude[axis])
            )
    for joint in joints:
        joint.set_drive_property(
            stiffness=1000.0, damping=100.0, force_limit=1.0e10, mode="force",
        )

    # A small free-space target perturbation does not identify how the drive
    # yields under an opposing constraint.  Keep the released drive active and
    # apply known generalized loads for 0.1 s.  This contains no contact,
    # object, table, grasp pose, or candidate-specific state.
    load_steps = 24
    load_scales = np.asarray((0.1, 0.5, 1.0), dtype=np.float64)
    load_base = np.concatenate((
        np.full(6, 20.0, dtype=np.float64),
        np.full(12, 0.5, dtype=np.float64),
    ))
    load_amplitude = load_scales[:, None] * load_base[None, :]
    load_sign = np.empty((len(configurations), 18), dtype=np.float64)
    loaded_qpos = np.empty(
        (len(configurations), len(load_scales), 18, load_steps, 18),
        dtype=np.float64,
    )
    loaded_qvel = np.empty_like(loaded_qpos)
    for config_index, qpos in enumerate(configurations):
        lower_room = qpos - limits[:, 0]
        upper_room = limits[:, 1] - qpos
        load_sign[config_index] = np.where(upper_room >= lower_room, 1.0, -1.0)
        for scale_index in range(len(load_scales)):
            for axis in range(18):
                reset(qpos, qpos, zero)
                applied = np.zeros(18, dtype=np.float64)
                applied[axis] = (
                    load_sign[config_index, axis]
                    * load_amplitude[scale_index, axis]
                )
                robot.set_qf(applied)
                for step in range(load_steps):
                    env.scene.step()
                    loaded_qpos[config_index, scale_index, axis, step] = (
                        np.asarray(robot.get_qpos(), dtype=np.float64)
                    )
                    loaded_qvel[config_index, scale_index, axis, step] = (
                        np.asarray(robot.get_qvel(), dtype=np.float64)
                    )
    robot.set_qf(zero)

    # Drive every scalar joint into both of its limits.  The prior free-space
    # and loaded probes stayed strictly inside the range and therefore could
    # not reveal how the PhysX limit constraint competes with the drive.
    limit_steps = 24
    limit_sides = np.asarray((-1.0, 1.0), dtype=np.float64)
    limit_start = np.empty((18, 2, 18), dtype=np.float64)
    limit_target = np.empty_like(limit_start)
    limit_qpos = np.empty((18, 2, limit_steps, 18), dtype=np.float64)
    limit_qvel = np.empty_like(limit_qpos)
    for axis in range(18):
        width = float(limits[axis, 1] - limits[axis, 0])
        clearance = min(0.02, 0.1 * width)
        overshoot = min(0.03, 0.15 * width)
        for side_index, side in enumerate(limit_sides):
            qpos = q0.copy()
            target = q0.copy()
            boundary = limits[axis, side_index]
            qpos[axis] = boundary - side * clearance
            target[axis] = boundary + side * overshoot
            limit_start[axis, side_index] = qpos
            limit_target[axis, side_index] = target
            reset(qpos, target, zero)
            for step in range(limit_steps):
                env.scene.step()
                limit_qpos[axis, side_index, step] = np.asarray(
                    robot.get_qpos(), dtype=np.float64,
                )
                limit_qvel[axis, side_index, step] = np.asarray(
                    robot.get_qvel(), dtype=np.float64,
                )
    robot.set_qf(zero)

    link_rows: list[dict[str, object]] = []
    for link in links:
        mass = float(link.get_mass())
        inertia = np.asarray(link.get_inertia(), dtype=np.float64)
        cmass = link.get_cmass_local_pose()
        link_rows.append({
            "name": str(link.get_name()), "mass_kg": mass,
            "inertia_kg_m2": inertia.tolist(),
            "cmass_local_pose_wxyz": np.concatenate((cmass.p, cmass.q)).tolist(),
            "gravity_disabled": bool(link.disable_gravity),
            "collision_shape_count": len(link.get_collision_shapes()),
        })

    output.mkdir(parents=True)
    trace = output / "joint_drive_probe.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray(
            "deximit_sapien_joint_drive_probe_v9_clean_loaded_limits_diagnostic_only",
        ),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        physics_dt_s=np.asarray(dt), joint_names=np.asarray(names, dtype="U64"),
        link_names=np.asarray(link_names, dtype="U64"),
        link_pose_world_wxyz=link_pose_world_wxyz,
        qpos0=q0, configurations=configurations,
        joint_limits=limits, stiffness=stiffness, damping=damping,
        force_limit=force_limit, friction=friction, armature=armature,
        velocity_probe_amplitude=velocity_amplitude,
        position_probe_amplitude=position_amplitude,
        position_probe_sign=position_sign,
        zero_qpos_after_step=zero_qpos, zero_qvel_after_step=zero_qvel,
        force_probe_amplitude=force_amplitude, force_response=force_response,
        velocity_response=velocity_response,
        position_response=position_response,
        velocity_qpos_response=velocity_qpos_response,
        position_qpos_response=position_qpos_response,
        loaded_response_steps=np.asarray(load_steps),
        loaded_response_scales=load_scales,
        loaded_response_amplitude_nm=load_amplitude,
        loaded_response_sign=load_sign,
        loaded_response_qpos=loaded_qpos,
        loaded_response_qvel=loaded_qvel,
        limit_response_steps=np.asarray(limit_steps),
        limit_response_sides=limit_sides,
        limit_response_start_qpos=limit_start,
        limit_response_drive_target=limit_target,
        limit_response_qpos=limit_qpos,
        limit_response_qvel=limit_qvel,
    )
    urdf = package / "asset" / config["robot"]["ur5e_with_right_hand"]["urdf_path"]
    report = {
        "schema": (
            "deximit_sapien_joint_drive_probe_v9_clean_loaded_limits_diagnostic_only"
        ),
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "table_and_objects_present_during_probe": False,
        "sapien_version": sapien.__version__,
        "physics_dt_s": dt,
        "robot_urdf": str(urdf.resolve()),
        "robot_urdf_sha256": sha256(urdf),
        "joint_names": names,
        "runtime_stiffness": stiffness.tolist(),
        "runtime_damping": damping.tolist(),
        "runtime_force_limit": force_limit.tolist(),
        "runtime_friction": friction.tolist(),
        "runtime_armature": armature.tolist(),
        "configuration_count": int(len(configurations)),
        "configuration_policy": "home plus six seeded interior joint configurations",
        "physx_warm_start_cache_policy": (
            "eight zero-load target-equals-position steps before every independent probe"
        ),
        "loaded_drive_probe": {
            "contact_free": True,
            "object_free": True,
            "candidate_free": True,
            "duration_steps": load_steps,
            "duration_s": load_steps * dt,
            "load_scales": load_scales.tolist(),
            "arm_base_load_nm": float(load_base[0]),
            "hand_base_load_nm": float(load_base[6]),
            "direction_policy": "toward the farther joint limit at each configuration",
        },
        "joint_limit_probe": {
            "contact_free": True,
            "object_free": True,
            "candidate_free": True,
            "duration_steps": limit_steps,
            "duration_s": limit_steps * dt,
            "sides_per_joint": 2,
            "start_clearance_rad_max": 0.02,
            "target_overshoot_rad_max": 0.03,
        },
        "zero_state_drift_qpos_linf": float(np.max(np.abs(zero_qpos - configurations))),
        "zero_state_drift_qvel_linf": float(np.max(np.abs(zero_qvel))),
        "robot_links": link_rows,
        "trace": str(trace),
        "trace_sha256": sha256(trace),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
