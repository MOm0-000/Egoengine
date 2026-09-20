"""Check coordinates and native triangles before comparing reset candidates.

FCL detects surface contact/intersection, not material penetration depth. Open
CAD parts are not silently closed. No collision exemption follows from a joint
being adjacent or from its surfaces also touching in the zero-finger pose.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import fcl
import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_collision_coverage import coverage_inventory
from audit_taco_initialization import visual_meshes, world_vertices
from audit_taco_thumb_assembly import native_intersection
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance, solid_angle_winding
from egoengine_repro.retarget.initial_hand import state_summary
from egoengine_repro.retarget.paper_audit import (
    alignment_invariants, artifact, scene_mesh_artifacts, transform_report, verify_artifacts,
)
from egoengine_repro.retarget.taco_bimanual import pose7

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
BASELINE = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1"
DATA = ROOT / "data/taco_v1/pour_bowl_plate"
SEQUENCE = "(pour in some, bowl, plate)/20230927_017"


def triangle_object(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("finite nonempty triangle mesh required")
    geometry = fcl.BVHModel()
    geometry.beginModel(len(vertices), len(faces))
    geometry.addSubModel(vertices, faces)
    geometry.endModel()
    return fcl.CollisionObject(geometry)


def surface_intersects(a, b):
    result = fcl.CollisionResult()
    fcl.collide(a, b, fcl.CollisionRequest(num_max_contacts=1), result)
    return bool(result.is_collision)


def native_surface_scan(model, meshes, qpos):
    qpos = validate_qpos(model, qpos, trajectory=True)
    geoms = list(meshes)
    objects = {g: triangle_object(meshes[g]) for g in geoms}
    pairs = list(combinations(geoms, 2))  # Includes shell-less index roots.
    declared = {tuple(sorted(map(int, model.geom_bodyid[[a, b]])))
                for a, b in zip(model.pair_geom1, model.pair_geom2)}
    records = []
    for a, b in pairs:
        names = [model.geom(g).name for g in (a, b)]
        ba, bb = map(int, model.geom_bodyid[[a, b]])
        wa, wb = model.body_weldid[[ba, bb]]
        pa, pb = model.body_weldid[model.body_parentid[[wa, wb]]]
        nobjects = sum(name.endswith("object_visual") for name in names)
        family = ("object_object" if nobjects == 2 else "hand_object" if nobjects == 1
                  else "intrahand" if names[0].split("_")[0] == names[1].split("_")[0] else "interhand")
        records.append(dict(geoms=names, family=family,
            declared_body_pair=tuple(sorted((ba, bb))) in declared,
            assembly_adjacent=bool(wa == wb or wa == pb or wb == pa)))

    zero = qpos[0].copy()
    for j in range(model.njnt):
        if "_hand_" in model.joint(j).name and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            zero[model.jnt_qposadr[j]] = model.qpos0[model.jnt_qposadr[j]]
    flags = np.zeros((len(qpos) + 1, len(pairs)), dtype=bool)
    support = np.empty((len(qpos), len(geoms)))
    data = mujoco.MjData(model)
    for frame, q in enumerate([zero, *qpos]):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        bounds = {}
        floor = model.geom("floor").id
        normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        for col, g in enumerate(geoms):
            body = model.geom_bodyid[g]
            objects[g].setTransform(fcl.Transform(data.xmat[body].reshape(3, 3), data.xpos[body]))
            vertices = world_vertices(model, data, g, meshes[g])
            bounds[g] = (vertices.min(0), vertices.max(0))
            if frame:
                support[frame - 1, col] = ((vertices - data.geom_xpos[floor]) @ normal).min()
        for col, (a, b) in enumerate(pairs):
            if np.all(bounds[a][0] <= bounds[b][1]) and np.all(bounds[b][0] <= bounds[a][1]):
                flags[frame, col] = surface_intersects(objects[a], objects[b])
        if frame % 30 == 0:
            print(f"native triangle audit {max(0, frame)}/{len(qpos)}", flush=True)
    for col, record in enumerate(records):
        rows = np.flatnonzero(flags[1:, col])
        record.update(zero_finger_pose_surface_contact=bool(flags[0, col]),
                      initial_surface_contact=bool(flags[1, col]),
                      surface_contact_frames=len(rows), source_rows=rows.tolist())
    return dict(
        method="FCL full native triangle surfaces at every reference frame; no shell-based pair filtering",
        python_fcl_version=fcl.__version__, frames=len(qpos), pairs=records,
        initial_surface_contact_counts=dict(Counter(r["family"] for r in records if r["initial_surface_contact"])),
        trajectory_surface_contact_pair_counts=dict(Counter(r["family"] for r in records if r["surface_contact_frames"])),
        native_table=[dict(geom=model.geom(g).name, initial_clearance_m=float(support[0, col]),
                          minimum_clearance_m=float(support[:, col].min()),
                          minimum_row=int(support[:, col].argmin())) for col, g in enumerate(geoms)],
        limitations=["touching and triangle crossing are not distinguished by this boolean test",
                     "no surface crossing does not exclude one solid wholly contained in another",
                     "open CAD surfaces do not define material interiors",
                     "zero-finger pose contacts and assembly adjacency are not collision exemptions",
                     "discrete reference frames, not continuous swept motion"],
    ), flags[1:], support


def coordinate_check(model, meshes, human, robot):
    raw_paths = [DATA / "hand_poses/Hand_Poses" / SEQUENCE / "hand_joints.npy",
                 DATA / "object_poses/Object_Poses" / SEQUENCE / "tool_022.npy",
                 DATA / "object_poses/Object_Poses" / SEQUENCE / "target_135.npy",
                 DATA / "camera/Egocentric_Camera_Parameters" / SEQUENCE / "egocentric_frame_extrinsic.npy"]
    joints, tool, target, camera = [np.load(p, allow_pickle=False).astype(float) for p in raw_paths]
    transform = human["T_sim_world"]
    if not transform_report(transform[None])["rigid_within_float32_tolerance"]:
        raise ValueError("non-rigid shared coordinate transform")
    poses = np.stack([tool, target], axis=1)
    report = alignment_invariants(joints, poses, camera, transform)
    report["exported_joints_error_m"] = float(np.abs(
        joints[:, [1, 0]] @ transform[:3, :3].T + transform[:3, 3] - human["joint_positions_sim"]).max())
    report["exported_object_transform_error"] = float(np.abs(transform @ poses - human["T_sim_object_reference"]).max())
    for col, side in enumerate(("right", "left")):
        adr = int(model.joint(f"{side}_object_joint").qposadr[0])
        expected = np.stack([pose7(p) for p in human["T_sim_object_reference"][:, col]])
        report[f"{side}_object_qpos_error"] = float(np.abs(expected - robot["qpos"][:, adr:adr + 7]).max())
    # Released float32 rotations are slightly nonorthogonal; conversion to unit
    # quaternions can differ at ~1e-8 between the exporter and MuJoCo. This is a
    # numerical export check, not a physical millimetre collision tolerance.
    if max(report.values()) > 1e-6:
        raise ValueError(f"inconsistent coordinate export: {report}")
    data = mujoco.MjData(model)
    data.qpos[:] = robot["qpos"][0]
    mujoco.mj_forward(model, data)
    mesh_error = {}
    for g, mesh in meshes.items():
        native = world_vertices(model, data, g, mesh)
        mid = model.geom_dataid[g]
        start, size = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        compiled = model.mesh_vert[start:start + size] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        mesh_error[model.geom(g).name] = float(max(cKDTree(native).query(compiled)[0].max(),
                                                 cKDTree(compiled).query(native)[0].max()))
    if max(mesh_error.values()) > 2e-6:
        raise ValueError("native/compiled visual vertices differ by over 2 micrometres")
    return dict(algebra=report, native_compiled_vertex_set_error_m=mesh_error,
                coordinate_export_consistent=True, export_numeric_tolerance=1e-6,
                original_sensor_registration_certified=False,
                raw_inputs=[artifact(p) for p in raw_paths],
                T_sim_world=transform.tolist(), table_z_m=float(data.geom_xpos[model.geom("floor").id, 2]),
                paper_table_source="EgoEngine Appendix A.1 PDF p15; matched real-world robot setup",
                vertical_anchor_source="local first plate minimum at table; not specified in paper" )


def object_overfill_samples(model, meshes):
    """Positive distance proves sampled collision material outside closed native mesh.

    Samples may be inside other pieces; all are still contained in the union.
    This reports sampled overfill, not its maximum or a complete cavity audit.
    """
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    result = {}
    for side in ("right", "left"):
        native_id = model.geom(f"{side}_object_visual").id
        native = meshes[native_id]
        if not native.is_volume:
            raise ValueError("overfill signed-distance test needs closed oriented object mesh")
        body = model.geom_bodyid[native_id]
        samples = []
        for g in range(model.ngeom):
            if not model.geom(g).name.startswith(f"{side}_object_") or g == native_id:
                continue
            mid = model.geom_dataid[g]
            start, size = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            world = model.mesh_vert[start:start + size] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            hull = trimesh.convex.convex_hull(local)
            samples.extend([hull.vertices, hull.triangles_center])
        points = np.concatenate(samples)
        signed = closed_mesh_signed_distance(native, points)
        if not np.isfinite(signed).all():
            raise ValueError("nonfinite native surface distance")
        excess = np.maximum(0, -signed)
        worst = np.argsort(excess)[-5:]
        winding = solid_angle_winding(native, points[worst])
        if np.any((excess[worst] > 5e-5) & (np.abs(winding) > .01)):
            raise ValueError("overfill sign disagrees with independent solid-angle check")
        result[side] = dict(samples=len(points),
            sampled_outside_native_over_50um=int((excess > 5e-5).sum()),
            sampled_outside_native_over_1mm=int((excess > .001).sum()),
            sampled_overfill_distance_m_percentiles=np.percentile(excess, [50, 95, 99, 100]).tolist(),
            worst_sample_object_coordinates_m=points[excess.argmax()].tolist(),
            distance_backend="Open3D signed distance, 11 rays, positive inside",
            extreme_probe_winding_crosscheck=dict(indices=worst.tolist(), values=winding.tolist()),
            maximum_error_certified=False, underfill_checked=False,
            sampling="all compiled convex hull vertices and triangle centroids, no random sampling")
        print(f"{side} object: sampled maximum overfill {excess.max() * 1000:.3f} mm", flush=True)
    return result


def run(scene, baseline, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [artifact(scene)] + [artifact(baseline / name) for name in
        ("human_reference.npz", "robot_reference.npz", "retarget_report.json")]
    source = json.loads((baseline / "retarget_report.json").read_text())
    if source["scene_sha256"] != inputs[0]["sha256"] or source["human_reference_sha256"] != inputs[1]["sha256"]:
        raise ValueError("retarget source hashes differ")
    dependencies = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(baseline / "human_reference.npz", allow_pickle=False) as data:
        human = dict(data)
    with np.load(baseline / "robot_reference.npz", allow_pickle=False) as data:
        robot = dict(data)
    if (len(robot["qpos"]) != 198 or not np.array_equal(human["frame_indices"], robot["frame_indices"])
            or not np.array_equal(robot["frame_indices"], np.arange(198))):
        raise ValueError("requires all original Pour rows without cropping")
    meshes, _ = visual_meshes(scene, model)
    coordinates = coordinate_check(model, meshes, human, robot)
    inventory = coverage_inventory(model, meshes)
    surfaces, flags, support = native_surface_scan(model, meshes, robot["qpos"])
    data = mujoco.MjData(model)
    data.qpos[:] = robot["qpos"][0]
    mujoco.mj_forward(model, data)
    report = dict(status="preflight_not_a_reset_or_physics_success", baseline=str(baseline.resolve()),
        mujoco_version=mujoco.__version__, coordinates=coordinates, inventory=inventory,
        initial_declared=state_summary(model, robot["qpos"][0]), native_surfaces=surfaces,
        initial_palm_thumb_closed_intersection={s: native_intersection(model, data, meshes, s) for s in ("right", "left")},
        object_collision_overfill=object_overfill_samples(model, meshes),
        collision_model_validated=False, candidate_comparison_ready=False, replay_rl_ready=False,
        preserved_artifacts=inputs, scene_meshes=dependencies, code=artifact(Path(__file__)),
        original_scene_or_reference_modified=False, simulation_steps_executed=0)
    verify_artifacts(inputs + dependencies + coordinates["raw_inputs"])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native_checks.npz", surface_contact=flags, table_clearance_m=support,
                        frame_indices=robot["frame_indices"])
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1",
    )
    args = parser.parse_args()
    run(args.scene, args.baseline, args.output)
