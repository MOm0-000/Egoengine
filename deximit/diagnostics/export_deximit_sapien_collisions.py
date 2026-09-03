#!/usr/bin/env python3
"""Export PhysX-cooked DexImit convex meshes without running a candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


SCHEMA = "deximit_sapien_cooked_collisions_v1_diagnostic_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deximit-root", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    deximit = args.deximit_root.expanduser().resolve(strict=True)
    object_mesh = args.object_mesh.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cooked collisions {output}")
    package = deximit / "third_party/any2dex/any2dex"
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(package / "third_party/BODex_api/src"))

    import sapien
    import trimesh
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
            gravity=np.asarray([0.0, 0.0, -9.81]), bounce_threshold=2.0,
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
    config = yaml.safe_load(
        (package / "env/config/env_hand.yaml").read_text(encoding="utf-8"),
    )
    env = BaseEnv(config)
    env.reset()
    material = sapien.physx.PhysxMaterial(
        static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
    )
    builder = env.scene.create_actor_builder()
    builder.add_convex_collision_from_file(
        filename=str(object_mesh), scale=[1.0, 1.0, 1.0], density=100.0,
        material=material,
    )
    object_actor = builder.build(name="diagnostic_object_geometry_only")
    object_body = object_actor.find_component_by_type(
        sapien.physx.PhysxRigidBodyComponent,
    )
    if object_body is None:
        raise RuntimeError("object collision export has no rigid body")

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []

    def export_shape(owner: str, shape_index: int, shape: object) -> None:
        if not isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
            return
        vertices = np.asarray(shape.get_vertices(), dtype=np.float64)
        triangles = np.asarray(shape.get_triangles(), dtype=np.int64).reshape(-1, 3)
        scale = np.asarray(shape.get_scale(), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or scale.shape != (3,):
            raise RuntimeError(f"cooked convex shape {owner!r} has malformed vertices")
        scaled = vertices * scale
        mesh = trimesh.Trimesh(vertices=scaled, faces=triangles, process=False)
        safe_owner = owner.replace("/", "_")
        path = mesh_dir / f"{safe_owner}_{shape_index:02d}.obj"
        mesh.export(path, file_type="obj")
        pose = shape.get_local_pose()
        rows.append({
            "owner": owner,
            "shape_index": shape_index,
            "shape_type": type(shape).__name__,
            "cooked_vertex_count": len(vertices),
            "cooked_triangle_count": len(triangles),
            "scale": scale.tolist(),
            "local_pose_wxyz": np.concatenate((
                np.asarray(pose.p, dtype=np.float64),
                np.asarray(pose.q, dtype=np.float64),
            )).tolist(),
            "mesh": str(path),
            "mesh_sha256": sha256(path),
        })

    for link in env.robot_right.get_links():
        for shape_index, shape in enumerate(link.get_collision_shapes()):
            export_shape(str(link.get_name()), shape_index, shape)
    for shape_index, shape in enumerate(object_body.get_collision_shapes()):
        export_shape("right_object", shape_index, shape)
    if not rows or not any(row["owner"] == "right_object" for row in rows):
        raise RuntimeError("no cooked object/robot convex shapes were exported")

    payload = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_executed": False,
        "object_input_mesh": str(object_mesh),
        "object_input_mesh_sha256": sha256(object_mesh),
        "sapien_version": sapien.__version__,
        "shape_count": len(rows),
        "shapes": rows,
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
