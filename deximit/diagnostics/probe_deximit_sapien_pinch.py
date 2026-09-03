#!/usr/bin/env python3
"""Run a candidate-free opposing-contact pinch/lift probe in SAPIEN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_metrics(path: Path, candidate: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == candidate
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("metrics"), dict):
        raise ValueError("summary lacks one selected physical-properties row")
    return rows[0]["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--candidate", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mesh_path = args.object_mesh.expanduser().resolve(strict=True)
    summary = args.sapien_summary.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SAPIEN pinch probe {output}")

    import sapien

    dt = 1.0 / 240.0
    sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
    sapien.physx.set_body_config(
        solver_position_iterations=25, solver_velocity_iterations=1,
        sleep_threshold=0.005,
    )
    sapien.physx.set_scene_config(
        gravity=np.asarray([0.0, 0.0, -9.81]), bounce_threshold=2.0,
        enable_pcm=True, enable_tgs=True, enable_ccd=False,
        enable_enhanced_determinism=False,
        enable_friction_every_iteration=True, cpu_workers=0,
    )
    sapien.physx.set_default_material(
        static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
    )
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(dt)

    table_builder = scene.create_actor_builder()
    table_builder.add_box_collision(
        half_size=[0.6, 0.8, 0.03],
        material=sapien.physx.PhysxMaterial(1.0, 1.0, 0.0),
    )
    table = table_builder.build_kinematic(name="table")
    table.set_pose(sapien.Pose([0.3, 0.0, 0.684]))

    material = sapien.physx.PhysxMaterial(0.7, 0.5, 0.0)
    object_builder = scene.create_actor_builder()
    object_builder.add_convex_collision_from_file(
        filename=str(mesh_path), scale=[1.0, 1.0, 1.0], density=100.0,
        material=material,
    )
    actor = object_builder.build(name="eraser")
    body = actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
    if body is None:
        raise RuntimeError("SAPIEN object builder produced no rigid body")
    metrics = selected_metrics(summary, args.candidate)
    body.set_mass(float(metrics["object_mass_kg"]))
    body.set_linear_damping(float(metrics["object_linear_damping"]))
    body.set_angular_damping(float(metrics["object_angular_damping"]))
    initial = np.asarray(metrics["object_initial_pose"], dtype=np.float64)
    quat = Rotation.from_matrix(initial[:3, :3]).as_quat(scalar_first=True)
    actor.set_pose(sapien.Pose(initial[:3, 3], quat))

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    local_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    world_vertices = (
        np.c_[np.asarray(mesh.vertices), np.ones(len(mesh.vertices))] @ initial.T
    )[:, :3]
    bounds = np.stack((world_vertices.min(axis=0), world_vertices.max(axis=0)))
    half_size = np.asarray([0.08, 0.01, 0.03], dtype=np.float64)
    open_gap = 0.006
    compression = 0.001
    local_center = local_bounds.mean(axis=0)
    left_local = local_center.copy()
    right_local = local_center.copy()
    left_local[1] = local_bounds[0, 1] - open_gap - half_size[1]
    right_local[1] = local_bounds[1, 1] + open_gap + half_size[1]
    left_open = initial[:3, 3] + initial[:3, :3] @ left_local
    right_open = initial[:3, 3] + initial[:3, :3] @ right_local
    # Keep the lower paddle faces just above the horizontal table despite the
    # object's small roll/pitch angle.
    z_shift = 0.714 + half_size[2] - min(left_open[2], right_open[2])
    left_open[2] += z_shift
    right_open[2] += z_shift
    left_closed = left_open.copy()
    right_closed = right_open.copy()
    pinch_axis = initial[:3, 1]
    left_closed += pinch_axis * (open_gap + compression)
    right_closed -= pinch_axis * (open_gap + compression)

    pads = []
    pad_bodies = []
    for name, position in (("left_pad", left_open), ("right_pad", right_open)):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size, material=material)
        pad = builder.build_kinematic(name=name)
        pad.set_pose(sapien.Pose(position, quat))
        component = pad.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
        if component is None or not component.get_kinematic():
            raise RuntimeError("pinch paddle is not kinematic")
        pads.append(pad)
        pad_bodies.append(component)

    # Stabilize only the object/table before moving either paddle.
    for _ in range(120):
        scene.step()
    settled = actor.get_pose()
    rest_position = np.asarray(settled.p, dtype=np.float64)

    close_steps, squeeze_steps, lift_steps, hold_steps = 72, 48, 120, 60
    total_steps = close_steps + squeeze_steps + lift_steps + hold_steps
    phase = np.empty(total_steps, dtype="U16")
    object_pose = np.empty((total_steps, 7), dtype=np.float64)
    object_linear = np.empty((total_steps, 3), dtype=np.float64)
    object_angular = np.empty((total_steps, 3), dtype=np.float64)
    pad_position = np.empty((total_steps, 2, 3), dtype=np.float64)
    pad_impulse_norm = np.zeros((total_steps, 2), dtype=np.float64)
    table_impulse_norm = np.zeros(total_steps, dtype=np.float64)

    for step in range(total_steps):
        if step < close_steps:
            phase[step] = "close"
            fraction = (step + 1) / close_steps
            left = left_open + fraction * (left_closed - left_open)
            right = right_open + fraction * (right_closed - right_open)
        elif step < close_steps + squeeze_steps:
            phase[step] = "squeeze"
            left, right = left_closed.copy(), right_closed.copy()
        elif step < close_steps + squeeze_steps + lift_steps:
            phase[step] = "lift"
            fraction = (step + 1 - close_steps - squeeze_steps) / lift_steps
            left, right = left_closed.copy(), right_closed.copy()
            left[2] += 0.08 * fraction
            right[2] += 0.08 * fraction
        else:
            phase[step] = "hold"
            left, right = left_closed.copy(), right_closed.copy()
            left[2] += 0.08
            right[2] += 0.08
        for component, position in zip(pad_bodies, (left, right), strict=True):
            component.set_kinematic_target(sapien.Pose(position, quat))
        scene.step()
        state = actor.get_pose()
        object_pose[step] = np.concatenate((state.p, state.q))
        object_linear[step] = body.get_linear_velocity()
        object_angular[step] = body.get_angular_velocity()
        pad_position[step] = np.stack((left, right))
        for contact in scene.get_contacts():
            if body not in contact.bodies:
                continue
            impulse = float(sum(
                np.linalg.norm(np.asarray(point.impulse, dtype=np.float64))
                for point in contact.points
            ))
            if table.find_component_by_type(
                sapien.physx.PhysxRigidBodyComponent,
            ) in contact.bodies:
                table_impulse_norm[step] += impulse
            for index, pad_body in enumerate(pad_bodies):
                if pad_body in contact.bodies:
                    pad_impulse_norm[step, index] += impulse

    hold = np.flatnonzero(phase == "hold")
    lift = object_pose[:, 2] - rest_position[2]
    opposed = (pad_impulse_norm[:, 0] > 1.0e-8) & (
        pad_impulse_norm[:, 1] > 1.0e-8
    )
    passed = bool(
        np.any(opposed)
        and np.all(opposed[hold])
        and np.min(lift[hold]) >= 0.06
        and np.ptp(object_pose[hold, 2]) <= 0.005
    )
    output.mkdir(parents=True)
    trace = output / "pinch_probe.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray("deximit_sapien_pinch_probe_v1_diagnostic_only"),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), physics_dt_s=np.asarray(dt),
        phase=phase, object_rest_position=rest_position,
        object_pose_world_wxyz=object_pose, object_linear_velocity=object_linear,
        object_angular_velocity=object_angular, pad_position=pad_position,
        pad_impulse_norm_sum_ns=pad_impulse_norm,
        table_impulse_norm_sum_ns=table_impulse_norm,
    )
    report = {
        "schema": "deximit_sapien_pinch_probe_v1_diagnostic_only",
        "diagnostic_only": True, "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "candidate_used_only_for_recorded_object_properties": args.candidate,
        "object_mesh": str(mesh_path), "object_mesh_sha256": sha256(mesh_path),
        "source_summary": str(summary), "source_summary_sha256": sha256(summary),
        "physics_dt_s": dt,
        "solver_position_iterations": 25,
        "friction_every_iteration": True,
        "persistent_contact_manifold": True,
        "strong_friction_default_enabled": True,
        "object_and_pad_static_dynamic_friction": [0.7, 0.5],
        "table_static_dynamic_friction": [1.0, 1.0],
        "object_rest_position_m": rest_position.tolist(),
        "object_bounds_from_initial_pose_m": bounds.tolist(),
        "paddle_half_size_m": half_size.tolist(),
        "paddle_compression_per_side_m": compression,
        "paddle_lift_m": 0.08,
        "strict_gate": {
            "passed": passed,
            "opposed_contact": bool(np.any(opposed)),
            "opposed_contact_all_hold_steps": bool(np.all(opposed[hold])),
            "hold_minimum_lift_m": float(np.min(lift[hold])),
            "hold_vertical_span_m": float(np.ptp(object_pose[hold, 2])),
        },
        "trace": str(trace), "trace_sha256": sha256(trace),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
