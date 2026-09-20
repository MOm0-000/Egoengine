"""Localize residual shape/contact discrepancies without changing a scene/reset."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import fcl
import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from build_taco_collision_repair import SCENE, OUTPUT, check_unchanged_dynamics
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance, solid_angle_winding
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

REFERENCE = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1/robot_reference.npz"
PREFLIGHT = ROOT / "runs/taco_pour_initialization_preflight_right_guard_v1/report.json"
URDFS = ROOT / "runs/taco_pour_thumb_assembly_v1"


def relative_mesh(model, data, meshes, source, target):
    a, b = model.geom_bodyid[[source, target]]
    rotation = data.xmat[b].reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :3] = rotation.T @ data.xmat[a].reshape(3, 3)
    transform[:3, 3] = rotation.T @ (data.xpos[a] - data.xpos[b])
    result = meshes[source].copy()
    result.apply_transform(transform)
    return result


def closed_component(mesh):
    if mesh.is_volume:
        return mesh, "full_native"
    components = [m for m in mesh.split(only_watertight=False, repair=False) if m.is_volume]
    return (components[0], "auxiliary_closed_component") if len(components) == 1 else (None, "no_unique_closed_component")


def native_pair_evidence(first, second, anchor=None, axis=None):
    """Both meshes already share first-body coordinates; no shell substitution."""
    result = fcl.CollisionResult()
    fcl.collide(triangle_object(first), triangle_object(second),
                fcl.CollisionRequest(num_max_contacts=4096, enable_contact=True), result)
    records = dict(surface_crossing=bool(result.is_collision), contact_triangles=len(result.contacts),
                   contact_list_capped=len(result.contacts) >= 4096)
    if result.contacts:
        points = np.asarray([c.pos for c in result.contacts])
        records["contact_bounds_m"] = [points.min(0).tolist(), points.max(0).tolist()]
        if anchor is not None:
            radial = np.linalg.norm(np.cross(points - anchor, axis), axis=1)
            records["surface_contact_radius_from_joint_axis_m"] = [float(radial.min()), float(radial.max())]
    cores = [closed_component(m) for m in (first, second)]
    records["solid_sources"] = [tag for _, tag in cores]
    if all(core is not None for core, _ in cores):
        # Millimetre units improve the absolute numerical scale of tiny patches.
        scaled = [m.copy() for m, _ in cores]
        for m in scaled:
            m.apply_scale(1000)
        intersection = trimesh.boolean.intersection(scaled, engine="manifold", check_volume=True)
        records["intersection_volume_mm3"] = float(max(0, intersection.volume)) if len(intersection.faces) else 0.0
        if len(intersection.faces):
            records["intersection_bounds_m"] = (intersection.bounds / 1000).tolist()
    # Closed targets permit positive containment evidence even when the other
    # CAD surface is open. This is sampled penetration, not a global depth.
    samples = []
    for source, (target, tag) in zip((first, second), cores[::-1]):
        if target is None:
            samples.append(dict(status=tag))
            continue
        points = np.concatenate([source.vertices, source.triangles_center])
        signed = closed_mesh_signed_distance(target, points)
        worst = int(signed.argmax())
        row = dict(status=tag, probes=len(points), inside_over_1um=int((signed > 1e-6).sum()),
                   sampled_max_inside_m=float(max(0, signed[worst])))
        if signed[worst] > 1e-6:
            row["extreme_winding"] = float(solid_angle_winding(target, points[worst:worst + 1])[0])
            row["extreme_position_m"] = points[worst].tolist()
            if anchor is not None:
                row["extreme_radius_from_joint_axis_m"] = float(np.linalg.norm(np.cross(points[worst] - anchor, axis)))
        samples.append(row)
    records["surface_containment_first_in_second_then_reverse"] = samples
    return records


def physical_geoms(model, body):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and
            (model.geom(g).name.startswith("collision_hand_") or
             (model.geom(g).name.startswith("left_object_") and not model.geom(g).name.endswith("visual")))]


def two_contact_differences(model, data, meshes, qpos):
    results = []
    declared = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    for row, names in [(93, ("left_index_rota2_visual", "left_middle_link1_visual")),
                       (177, ("left_index_rota2_visual", "left_middle_link2_visual"))]:
        a, b = [model.geom(n).id for n in names]
        bodies = model.geom_bodyid[[a, b]]
        data.qpos[:] = qpos[row]
        mujoco.mj_forward(model, data)
        contacts = [float(c.dist) for c in data.contact if
                    set(model.geom_bodyid[[c.geom1, c.geom2]]) == set(bodies)]
        distances = []
        for ga in physical_geoms(model, bodies[0]):
            for gb in physical_geoms(model, bodies[1]):
                distances.append(float(mujoco.mj_geomDistance(model, data, ga, gb, .1, None)))
                if frozenset((ga, gb)) not in declared:
                    raise ValueError("required physical pair is absent")
        evidence = native_pair_evidence(meshes[a], relative_mesh(model, data, meshes, b, a))
        record = dict(source_row=row, geoms=list(names), all_piece_pairs_declared=True,
                      runtime_contacts=len(contacts), minimum_runtime_contact_distance_m=min(contacts, default=None),
                      independent_geom_distance_min_m=min(distances), native=evidence,
                      comparison_statistic_m=5e-5, statistic_is_not_contact_activation_threshold=True)
        record["classification"] = ("detected_contact_below_50um_statistic" if contacts and min(contacts) < 0
                                     else "native_surface_record_without_runtime_penetrating_contact_requires_review")
        results.append(record)
        print("contact difference", row, record["classification"], min(contacts, default=None), flush=True)
    return results


def source_joint_check(model, joint_id, urdf):
    name = model.joint(joint_id).name
    joint = urdf.find(f"joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"source URDF joint absent: {name}")
    child = int(model.jnt_bodyid[joint_id])
    origin = joint.find("origin")
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    compiled = Rotation.from_quat(model.body_quat[child][[1, 2, 3, 0]])
    values = dict(parent_matches=model.body(int(model.body_parentid[child])).name == joint.find("parent").get("link"),
        child_matches=model.body(child).name == joint.find("child").get("link"),
        origin_error_m=float(np.linalg.norm(np.fromstring(origin.get("xyz"), sep=" ") - model.body_pos[child])),
        rotation_error_rad=float((rotation.inv() * compiled).magnitude()),
        axis_error=float(np.linalg.norm(np.fromstring(joint.find("axis").get("xyz"), sep=" ") - model.jnt_axis[joint_id])),
        limits_error_rad=float(np.abs(model.jnt_range[joint_id] -
            [float(joint.find("limit").get(k)) for k in ("lower", "upper")]).max()))
    values["matches"] = (values["parent_matches"] and values["child_matches"] and values["origin_error_m"] < 1e-9
                          and values["rotation_error_rad"] < 2e-6 and values["axis_error"] < 1e-9
                          and values["limits_error_rad"] < 1e-9)
    return values


def adjacent_assemblies(model, data, meshes, qpos, preflight):
    urdfs = {s: ET.parse(URDFS / f"xhand_{s}.urdf").getroot() for s in ("right", "left")}
    results = []
    for pair in preflight["native_surfaces"]["pairs"]:
        if not pair["assembly_adjacent"] or pair["family"] != "intrahand":
            continue
        a, b = [model.geom(n).id for n in pair["geoms"]]
        ba, bb = map(int, model.geom_bodyid[[a, b]])
        if model.body_parentid[bb] != ba:
            raise ValueError("this audit expects an immediate ordered parent/child pair")
        ids = np.flatnonzero(model.jnt_bodyid == bb)
        if len(ids) != 1 or model.jnt_type[ids[0]] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("adjacent scan requires a single relative hinge")
        joint = int(ids[0]); adr = int(model.jnt_qposadr[joint])
        # Five declared-range angles plus actual first-frame angle. This is a
        # local classification probe, not a full continuous-motion certificate.
        angles = sorted(set([0.0, float(qpos[0, adr]), *np.linspace(*model.jnt_range[joint], 5)]))
        rows = []
        for angle in angles:
            data.qpos[:] = qpos[0]
            data.qpos[adr] = angle
            mujoco.mj_kinematics(model, data)
            rotation = data.xmat[ba].reshape(3, 3)
            anchor = rotation.T @ (data.xanchor[joint] - data.xpos[ba])
            axis = rotation.T @ data.xaxis[joint]
            evidence = native_pair_evidence(meshes[a], relative_mesh(model, data, meshes, b, a), anchor, axis)
            rows.append(dict(joint_angle_rad=angle, **evidence))
        zero = next(r for r in rows if r["joint_angle_rad"] == 0)
        tags = {t for r in rows for t in r["solid_sources"]}
        category = ("angle_dependent_not_zero_pose_assembly_overlap" if not zero["surface_crossing"]
                    and any(r["surface_crossing"] for r in rows) else "persistent_joint_surface_contact")
        result = dict(geoms=pair["geoms"], joint=model.joint(joint).name, classification=category,
            original_reference_contact_frames=pair["surface_contact_frames"], angles=rows,
            full_solid_claim_available=tags == {"full_native"},
            source_check=source_joint_check(model, joint, urdfs[pair["geoms"][0].split("_")[0]]),
            exemption_applied=False, sweep_is_not_global_certificate=True)
        results.append(result)
        print("assembly", pair["geoms"], category, flush=True)
    return results


def plate_edge(model, data, meshes, qpos):
    visual = model.geom("left_object_visual").id
    body = int(model.geom_bodyid[visual]); native = meshes[visual]
    ray = o3d.t.geometry.RaycastingScene(nthreads=4)
    ray.add_triangles(o3d.core.Tensor(np.asarray(native.vertices), dtype=o3d.core.Dtype.Float32),
                      o3d.core.Tensor(np.asarray(native.faces), dtype=o3d.core.Dtype.UInt32))
    mujoco.mj_kinematics(model, data)
    hulls, point_groups, geometry_ids = {}, [], []
    for geom in physical_geoms(model, body):
        mid = model.geom_dataid[geom]; start, size = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        world = model.mesh_vert[start:start + size] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        points = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        hull = trimesh.convex.convex_hull(points); hulls[geom] = hull
        probes = np.concatenate([hull.vertices, hull.triangles_center])
        point_groups.append(probes); geometry_ids.extend([geom] * len(probes))
    points = np.concatenate(point_groups); geometry_ids = np.asarray(geometry_ids)
    error = np.maximum(0, -closed_mesh_signed_distance(native, points))
    worst = np.argsort(error)[-10:]
    query = ray.compute_closest_points(o3d.core.Tensor(points[worst].astype(np.float32)))
    closest = query["points"].numpy().astype(float)
    faces = query["primitive_ids"].numpy().astype(int)
    severe_ids = set(geometry_ids[error > .001].tolist())
    witnesses = []
    for row, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        checked = set()
        for contact in data.contact:
            ga, gb = int(contact.geom1), int(contact.geom2)
            if ga not in hulls and gb not in hulls:
                continue
            plate_geom = ga if ga in hulls else gb
            if (ga, gb) in checked:
                continue
            checked.add((ga, gb))
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(model, data, ga, gb, .5, segment)
            world = segment[:3] if ga in hulls else segment[3:]
            local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            other = gb if ga in hulls else ga
            witnesses.append(dict(source_row=row, plate_geom=model.geom(plate_geom).name,
                other_geom=model.geom(other).name, minimum_pair_distance_m=float(distance),
                plate_witness_local_m=local.tolist(), piece_has_sampled_overfill_over_1mm=plate_geom in severe_ids))
    if witnesses:
        positions = np.array([r["plate_witness_local_m"] for r in witnesses])
        signed = closed_mesh_signed_distance(native, positions)
        for row, value in zip(witnesses, signed):
            row["witness_outside_native_m"] = float(max(0, -value))
    top = sorted(witnesses, key=lambda r: r["witness_outside_native_m"], reverse=True)[:20]
    return dict(native_bounds_m=native.bounds.tolist(), native_extents_m=native.extents.tolist(),
        shell_samples=len(points), sampled_overfill_over_1mm=int((error > .001).sum()),
        pieces_with_overfill_over_1mm=len(severe_ids),
        extrema=[dict(point_m=points[i].tolist(), geom=model.geom(int(geometry_ids[i])).name,
            outside_m=float(error[i]), nearest_native_point_m=c.tolist(), nearest_native_normal=native.face_normals[f].tolist())
            for i, c, f in zip(worst, closest, faces)],
        extrema_winding=solid_angle_winding(native, points[worst]).tolist(),
        runtime_pair_witnesses=len(witnesses),
        runtime_on_over_1mm_pieces=sum(r["piece_has_sampled_overfill_over_1mm"] for r in witnesses),
        runtime_witness_over_1mm=sum(r["witness_outside_native_m"] > .001 for r in witnesses),
        worst_runtime_witnesses=top,
        contact_point_caveat="one mj_geomDistance witness per active pair; not all contact manifold points or unseen RL states",
        frames=len(qpos), shape_or_reference_changed=False)


def run(scene, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [artifact(p) for p in (scene, SCENE, REFERENCE, PREFLIGHT,
        URDFS / "xhand_right.urdf", URDFS / "xhand_left.urdf")]
    inputs += scene_mesh_artifacts(scene)
    print("Compiling unchanged candidate once; explicit-pair compilation takes several minutes", flush=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    dynamics = check_unchanged_dynamics(mujoco.MjModel.from_xml_path(str(SCENE)), model)
    data = mujoco.MjData(model)
    meshes, _ = visual_meshes(scene, model)
    with np.load(REFERENCE, allow_pickle=False) as source:
        qpos = source["qpos"]
    preflight = json.loads(PREFLIGHT.read_text())
    result = dict(status="remaining_geometry_classification_not_reset_or_training",
        unchanged_dynamics=dynamics, mujoco_version=mujoco.__version__,
        two_contact_differences=two_contact_differences(model, data, meshes, qpos),
        plate=plate_edge(model, data, meshes, qpos),
        adjacent_assemblies=adjacent_assemblies(model, data, meshes, qpos, preflight),
        source_artifacts=inputs, code=artifact(Path(__file__)),
        geometry_or_reference_modified=False, simulation_steps=0, training_ready=False)
    result["assembly_class_counts"] = dict(Counter(r["classification"] for r in result["adjacent_assemblies"]))
    verify_artifacts(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved", output, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=OUTPUT / "scene_refined.xml")
    parser.add_argument("--output", type=Path, default=OUTPUT / "remaining_contacts.json")
    args = parser.parse_args()
    run(args.scene, args.output)
