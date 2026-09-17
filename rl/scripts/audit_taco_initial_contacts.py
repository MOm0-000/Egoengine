"""Cross-check initial hand/object shell overlap against released native surfaces."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from audit_taco_initialization import visual_meshes, world_vertices
from egoengine_repro.retarget.collision_audit import collision_families, distances, validate_qpos
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def overlapping_bounds(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all() for p in (first, second)):
        raise ValueError("finite nonempty (N,3) vertices required for AABB checks")
    return bool(np.all(first.min(axis=0) <= second.max(axis=0)) and
                np.all(second.min(axis=0) <= first.max(axis=0)))


def inspect(scene: Path, reference: Path, *, native_aabb=False) -> dict:
    mesh_snapshot = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    with np.load(reference, allow_pickle=False) as robot:
        data.qpos[:] = validate_qpos(model, robot["qpos"][0])
        projection_candidate = bool(robot.get("diagnostic_only", np.asarray(False)))
    mujoco.mj_forward(model, data)
    meshes, mesh_paths = visual_meshes(scene, model)
    by_body = {int(model.geom_bodyid[g]): g for g in meshes}
    vertices = {g: world_vertices(model, data, g, mesh) for g, mesh in meshes.items()}
    native_support = [dict(geom=model.geom(g).name,
                           table_clearance_m=float(points[:, 2].min() - model.geom("floor").pos[2]))
                      for g, points in vertices.items()]
    aabb_checks = []
    records = []
    for family, pairs in collision_families(model).items():
        if family not in ("hand_tool", "hand_target"):
            continue
        native_pairs = set()
        if native_aabb:
            object_geom = model.geom("right_object_visual" if family == "hand_tool" else "left_object_visual").id
            for g in meshes:
                if model.geom(g).name in ("right_object_visual", "left_object_visual"):
                    continue
                overlap = overlapping_bounds(vertices[g], vertices[object_geom])
                aabb_checks.append(dict(hand_visual=model.geom(g).name, object_visual=model.geom(object_geom).name,
                                        native_aabb_overlap=overlap))
                if overlap:
                    native_pairs.add((g, object_geom))
        else:
            values = distances(model, data, pairs)
            for (a, b), distance in zip(pairs, values):
                if distance < -5e-5:
                    native_pairs.add((by_body[int(model.geom_bodyid[a])], by_body[int(model.geom_bodyid[b])]))
        for a, b in sorted(native_pairs):
            record = dict(family=family, hand_visual=model.geom(a).name,
                          object_visual=model.geom(b).name, directions=[])
            for ga, gb in ((a, b), (b, a)):
                result = dict(source_visual=model.geom(ga).name, target_visual=model.geom(gb).name)
                target_mesh = meshes[gb]
                if not target_mesh.is_watertight:
                    result["status"] = "target_open_mesh_signed_containment_not_used"
                else:
                    points = vertices[ga]
                    points = points[np.linspace(0, len(points) - 1, min(256, len(points)), dtype=int)]
                    body = model.geom_bodyid[gb]
                    local = (points - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                    signed = closed_mesh_signed_distance(target_mesh, local)
                    if not np.isfinite(signed).all():
                        raise ValueError("nonfinite native signed distance")
                    result.update(status="sampled_surface_evidence_not_global_bound", samples=len(points),
                                  inside_samples_50um=int((signed > 5e-5).sum()),
                                  sampled_penetration_max_m=float(max(0, signed.max())))
                record["directions"].append(result)
            records.append(record)
            print(json.dumps(record), flush=True)
    verify_artifacts(mesh_snapshot)
    return dict(status="initial_native_contact_diagnostic_not_reset_or_full_collision_certificate",
                native_distance_backend="Open3D signed distance, 11 rays, positive inside",
                distance_code=artifact(Path(__file__).resolve().parents[1] / "src/egoengine_repro/retarget/mesh_distance.py"),
                source_row_zero_based=0, tolerance_m=5e-5, max_samples_per_surface=256,
                selection=("all native hand/object AABBs, independent of collision-shell clearance" if native_aabb else
                           "unique native body pairs with initial shell penetration over 50 micrometers"),
                native_aabb_checks=aabb_checks, native_visual_support=native_support,
                input_is_projection_diagnostic_candidate=projection_candidate,
                records=records, state_projection_applied=False, simulation_steps_executed=0,
                strict_gate_passed=False, physics_validated=False,
                source_assets=[artifact(p) for p in mesh_paths], scene_meshes=mesh_snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-aabb", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preserved = [artifact(p) for p in (args.scene, args.reference)]
    report = inspect(args.scene, args.reference, native_aabb=args.native_aabb)
    if preserved != [artifact(p) for p in (args.scene, args.reference)]:
        raise RuntimeError("scene/reference changed during audit")
    report["preserved_artifacts"] = preserved
    report["audit_code"] = artifact(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
