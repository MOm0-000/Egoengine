#!/usr/bin/env python3
"""Run candidate-free object, damping, table-contact, and sliding probes in SAPIEN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_metrics(path: Path, candidate: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in value.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == candidate
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("metrics"), dict):
        raise ValueError("summary lacks one selected SAPIEN metrics row")
    return rows[0]["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--candidate", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mesh = args.object_mesh.expanduser().resolve(strict=True)
    summary = args.sapien_summary.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SAPIEN object probe {output}")

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

    table_material = sapien.physx.PhysxMaterial(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
    )
    table_builder = scene.create_actor_builder()
    table_builder.add_box_collision(
        half_size=[0.6, 0.8, 0.03], material=table_material,
    )
    table = table_builder.build_kinematic(name="table")
    table.set_pose(sapien.Pose([0.3, 0.0, 0.684]))

    object_material = sapien.physx.PhysxMaterial(
        static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
    )
    object_builder = scene.create_actor_builder()
    object_builder.add_convex_collision_from_file(
        filename=str(mesh), scale=[1.0, 1.0, 1.0], density=100.0,
        material=object_material,
    )
    actor = object_builder.build(name="eraser")
    body = actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
    if body is None:
        raise RuntimeError("SAPIEN object builder produced no rigid body")
    metrics = selected_metrics(summary, args.candidate)
    body.set_linear_damping(float(metrics["object_linear_damping"]))
    body.set_angular_damping(float(metrics["object_angular_damping"]))
    body.set_mass(float(metrics["object_mass_kg"]))
    initial = np.asarray(metrics["object_initial_pose"], dtype=np.float64)
    initial_quat = Rotation.from_matrix(initial[:3, :3]).as_quat(scalar_first=True)

    def reset(position: np.ndarray, quat: np.ndarray, *, gravity: bool) -> None:
        actor.set_pose(sapien.Pose(position, quat))
        body.set_linear_velocity(np.zeros(3, dtype=np.float64))
        body.set_angular_velocity(np.zeros(3, dtype=np.float64))
        body.disable_gravity = not gravity

    def run(
        name: str, steps: int, position: np.ndarray, quat: np.ndarray,
        linear_velocity: np.ndarray, angular_velocity: np.ndarray, *, gravity: bool,
    ) -> dict[str, np.ndarray]:
        reset(position, quat, gravity=gravity)
        body.set_linear_velocity(linear_velocity)
        body.set_angular_velocity(angular_velocity)
        pose = np.empty((steps, 7), dtype=np.float64)
        linear = np.empty((steps, 3), dtype=np.float64)
        angular = np.empty((steps, 3), dtype=np.float64)
        separation = np.full(steps, np.nan, dtype=np.float64)
        impulse_max = np.zeros(steps, dtype=np.float64)
        impulse_norm_sum = np.zeros(steps, dtype=np.float64)
        impulse_net = np.zeros((steps, 3), dtype=np.float64)
        point_count = np.zeros(steps, dtype=np.int32)
        for step in range(steps):
            scene.step()
            state = actor.get_pose()
            pose[step] = np.concatenate((state.p, state.q))
            linear[step] = body.get_linear_velocity()
            angular[step] = body.get_angular_velocity()
            separations: list[float] = []
            impulses: list[np.ndarray] = []
            for contact in scene.get_contacts():
                if body not in contact.bodies:
                    continue
                for point in contact.points:
                    separations.append(float(point.separation))
                    impulses.append(np.asarray(point.impulse, dtype=np.float64))
            if separations:
                separation[step] = min(separations)
                impulse_norms = np.linalg.norm(np.asarray(impulses), axis=1)
                impulse_max[step] = float(np.max(impulse_norms))
                impulse_norm_sum[step] = float(np.sum(impulse_norms))
                impulse_net[step] = np.sum(impulses, axis=0)
                point_count[step] = len(separations)
        return {
            f"{name}_pose_world_wxyz": pose,
            f"{name}_linear_velocity": linear,
            f"{name}_angular_velocity": angular,
            f"{name}_contact_separation_m_min": separation,
            f"{name}_contact_impulse_ns_max": impulse_max,
            f"{name}_contact_impulse_norm_sum_ns": impulse_norm_sum,
            f"{name}_contact_impulse_net_ns": impulse_net,
            f"{name}_contact_point_count": point_count,
        }

    traces: dict[str, np.ndarray] = {}
    high_position = initial[:3, 3].copy()
    high_position[2] += 0.25
    traces.update(run(
        "linear_decay", 120, high_position, initial_quat,
        np.asarray([0.2, -0.1, 0.05]), np.zeros(3), gravity=False,
    ))
    traces.update(run(
        "angular_decay", 120, high_position, initial_quat,
        np.zeros(3), np.asarray([0.5, -0.3, 0.2]), gravity=False,
    ))
    drop_position = initial[:3, 3].copy()
    drop_position[2] += 0.05
    traces.update(run(
        "drop", 240, drop_position, initial_quat,
        np.zeros(3), np.zeros(3), gravity=True,
    ))
    # Start the slide from the same stable pose obtained at the end of the drop.
    settled_pose = traces["drop_pose_world_wxyz"][-1]
    traces.update(run(
        "slide", 240, settled_pose[:3], settled_pose[3:],
        np.asarray([0.2, 0.0, 0.0]), np.zeros(3), gravity=True,
    ))

    cmass = body.get_cmass_local_pose()
    shapes = list(body.get_collision_shapes())
    runtime = {
        "mass_kg": float(body.get_mass()),
        "inertia_kg_m2": np.asarray(body.get_inertia(), dtype=np.float64).tolist(),
        "cmass_local_pose_wxyz": np.concatenate((cmass.p, cmass.q)).tolist(),
        "linear_damping": float(body.get_linear_damping()),
        "angular_damping": float(body.get_angular_damping()),
        "shape_count": len(shapes),
        "shape_contact_offset_m": [float(shape.get_contact_offset()) for shape in shapes],
        "shape_rest_offset_m": [float(shape.get_rest_offset()) for shape in shapes],
    }
    expected_inertia = np.asarray(metrics["object_inertia_kg_m2"], dtype=np.float64)
    expected_cmass = np.asarray(metrics["object_cmass_local_pose_wxyz"], dtype=np.float64)
    actual_cmass = np.concatenate((cmass.p, cmass.q))
    if (
        not np.isclose(runtime["mass_kg"], float(metrics["object_mass_kg"]), rtol=1e-6)
        or not np.allclose(body.get_inertia(), expected_inertia, rtol=2e-6, atol=1e-10)
        or not np.allclose(actual_cmass, expected_cmass, rtol=2e-6, atol=1e-8)
    ):
        raise RuntimeError("candidate-free object does not reproduce recorded SAPIEN inertia")

    output.mkdir(parents=True)
    trace = output / "object_probe.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray("deximit_sapien_object_probe_v2_diagnostic_only"),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        grasp_candidate_executed=np.asarray(False), physics_dt_s=np.asarray(dt),
        initial_pose=initial, **traces,
    )
    report = {
        "schema": "deximit_sapien_object_probe_v2_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "object_mesh": str(mesh),
        "object_mesh_sha256": sha256(mesh),
        "source_summary": str(summary),
        "source_summary_sha256": sha256(summary),
        "physics_dt_s": dt,
        "solver_position_iterations": 25,
        "solver_velocity_iterations": 1,
        "tgs": True,
        "persistent_contact_manifold": True,
        "contact_impulse_semantics": {
            "max": "largest Euclidean point-impulse norm in one source step",
            "norm_sum": "sum of all point-impulse norms in one source step",
            "net": "vector sum of all point impulses in one source step",
        },
        "table_static_dynamic_friction": [1.0, 1.0],
        "object_static_dynamic_friction": [0.7, 0.5],
        "runtime_object": runtime,
        "trace": str(trace),
        "trace_sha256": sha256(trace),
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
