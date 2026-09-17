"""Check sampled released object surfaces at GT poses, without rendering."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-mesh", type=Path, default=ROOT / "data/taco_v1/dev4/object_models/object_models_released/071_cm.obj")
    parser.add_argument("--target-mesh", type=Path, default=ROOT / "data/taco_v1/dev4/object_models/object_models_released/146_cm.obj")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.human, allow_pickle=False) as data:
        poses = data["T_sim_object_reference"]
    meshes = []
    inputs = [args.human]
    for path in (args.tool_mesh, args.target_mesh):
        mesh = trimesh.load_mesh(path, process=True)
        mesh.apply_scale(0.01)
        if not mesh.is_watertight:
            raise ValueError(f"signed containment requires a watertight mesh: {path}")
        meshes.append(mesh)
        inputs.append(path)
    # Deterministic vertex samples quantify evidence, not a global intersection
    # bound. Full vertex sets are used for table clearance, which is exact for
    # a triangulated surface against a plane.
    samples = [m.vertices[np.linspace(0, len(m.vertices) - 1, 2048, dtype=int)] for m in meshes]
    frames = []
    for frame in range(len(poses)):
        floors = []
        for index, mesh in enumerate(meshes):
            world = mesh.vertices @ poses[frame, index, :3, :3].T + poses[frame, index, :3, 3]
            floors.append(float(world[:, 2].min() - 0.72))
        frames.append(dict(frame=frame, mesh_floor_clearance_m=floors))
    # Inspect both endpoints and the strongest collision-model violations;
    # the collision report is only used to select diagnostic frames.
    audit_path = args.human.parent / "collision_audit.json"
    selected = {0, len(poses) - 1}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        selected.update(item["frame"] for item in audit["tool_target"]["worst_pairs"])
        inputs.append(audit_path)
    for frame in sorted(selected):
        penetrations = []
        for a, b in ((0, 1), (1, 0)):
            relative = np.linalg.inv(poses[frame, b]) @ poses[frame, a]
            points = samples[a] @ relative[:3, :3].T + relative[:3, 3]
            signed = closed_mesh_signed_distance(meshes[b], points)
            penetrations.append(dict(sampled_inside_count=int((signed > 5e-5).sum()),
                                      sampled_penetration_max_m=float(max(0, signed.max()))))
        frames[frame]["sampled_surface_penetration_tool_into_target_then_reverse"] = penetrations
        print(f"surface audit {frame}: {penetrations}", flush=True)
    report = dict(status="sampled_surface_diagnostic_not_full_collision_certificate",
                  samples_per_object=2048, tolerance_m=5e-5,
                  table_height_m=0.72, frames=frames,
                  inputs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in inputs], strict_gate_passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
