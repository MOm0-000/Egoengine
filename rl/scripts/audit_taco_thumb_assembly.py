"""Compare source thumb kinematics and full native mesh intersections; no reset."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def native_intersection(model, data, meshes, side):
    validate_qpos(model, data.qpos)
    palm_id = model.geom(f"{side}_hand_link_visual").id
    thumb_id = model.geom(f"{side}_thumb_rota1_visual").id
    palm_body, thumb_body = model.geom_bodyid[[palm_id, thumb_id]]
    palm_rotation = data.xmat[palm_body].reshape(3, 3)
    relative = np.eye(4)
    relative[:3, :3] = palm_rotation.T @ data.xmat[thumb_body].reshape(3, 3)
    relative[:3, 3] = palm_rotation.T @ (data.xpos[thumb_body] - data.xpos[palm_body])
    palm, thumb = meshes[palm_id], meshes[thumb_id].copy()
    if not palm.is_volume or not thumb.is_volume:
        raise ValueError("full intersection requires closed, positively oriented native solids")
    thumb.apply_transform(relative)
    intersection = trimesh.boolean.intersection([palm, thumb], engine="manifold", check_volume=True)
    nonempty = len(intersection.faces) > 0
    axes = {}
    if nonempty:
        for suffix in ("bend_joint", "rota_joint1"):
            joint = model.joint(f"{side}_hand_thumb_{suffix}").id
            anchor = palm_rotation.T @ (data.xanchor[joint] - data.xpos[palm_body])
            axis = palm_rotation.T @ data.xaxis[joint]
            radial = np.linalg.norm(np.cross(intersection.vertices - anchor, axis), axis=1)
            axes[suffix] = dict(intersection_vertex_radial_distance_min_m=float(radial.min()),
                                intersection_vertex_radial_distance_max_m=float(radial.max()))
    return dict(intersection_nonempty=nonempty, intersection_volume_m3=float(intersection.volume) if nonempty else 0.0,
                intersection_faces=len(intersection.faces),
                intersection_bounds_in_palm_m=intersection.bounds.tolist() if nonempty else None,
                joint_axis_distances=axes, T_palm_thumb=relative.tolist(),
                method="full closed native meshes; Manifold boolean in palm coordinates, finite numerical precision")


def kinematic_source_check(model, urdf, side):
    root = ET.parse(urdf).getroot()
    checks = []
    tolerances = dict(translation_m=1e-9, rotation_rad=2e-6, axis=1e-9, limit_rad=1e-9)
    for suffix in ("bend_joint", "rota_joint1"):
        name = f"{side}_hand_thumb_{suffix}"
        joint = root.find(f"joint[@name='{name}']")
        origin = joint.find("origin")
        child = joint.find("child").get("link")
        body = model.body(child).id
        parent = joint.find("parent").get("link")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        quat = model.body_quat[body]
        relative_rotation = Rotation.from_euler("xyz", rpy).inv() * Rotation.from_quat(quat[[1, 2, 3, 0]])
        limit = joint.find("limit")
        checks.append(dict(joint=name, parent=parent, child=child,
            parent_matches=model.body(int(model.body_parentid[body])).name == parent,
            joint_child_matches=int(model.jnt_bodyid[model.joint(name).id]) == body,
            joint_type_matches=joint.get("type") == "revolute" and int(model.joint(name).type[0]) == mujoco.mjtJoint.mjJNT_HINGE,
            limit_enabled=bool(model.joint(name).limited[0]),
            joint_local_anchor_error_m=float(np.linalg.norm(model.joint(name).pos)),
            joint_reference_error_rad=float(abs(model.qpos0[int(model.joint(name).qposadr[0])])),
            origin_translation_error_m=float(np.linalg.norm(xyz - model.body_pos[body])),
            origin_rotation_error_rad=float(relative_rotation.magnitude()),
            joint_axis_error=float(np.linalg.norm(np.fromstring(joint.find("axis").get("xyz"), sep=" ") - model.joint(name).axis)),
            limit_max_error_rad=float(np.abs(model.joint(name).range - [float(limit.get("lower")), float(limit.get("upper"))]).max())))
    matches = all(c["parent_matches"] and c["joint_child_matches"] and c["joint_type_matches"] and c["limit_enabled"]
                  and c["joint_local_anchor_error_m"] <= tolerances["translation_m"]
                  and c["joint_reference_error_rad"] <= tolerances["limit_rad"]
                  and c["origin_translation_error_m"] <= tolerances["translation_m"]
                  and c["origin_rotation_error_rad"] <= tolerances["rotation_rad"]
                  and c["joint_axis_error"] <= tolerances["axis"]
                  and c["limit_max_error_rad"] <= tolerances["limit_rad"] for c in checks)
    return dict(joints=checks, source_has_pair_exemption=False,
                matches_source_within_tolerance=matches, local_comparison_tolerances=tolerances,
                interpretation=("source URDF agrees within the disclosed export-comparison tolerance" if matches else
                                "source/model kinematic mismatch; inspect failing numerical or topology checks") +
                               "; omission from runtime pairs is not permission to intersect")


def run(scene, baseline, candidate, source_assets, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [scene, baseline, candidate, *[source_assets / f"xhand_{side}.urdf" for side in ("right", "left")]]
    preserved = [artifact(p) for p in inputs]
    mesh_snapshot = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    meshes, mesh_paths = visual_meshes(scene, model)
    with np.load(baseline, allow_pickle=False) as source:
        original = source["qpos"][0]
    with np.load(candidate, allow_pickle=False) as source:
        initial = source["qpos"][0]
    states = dict(source_zero=model.qpos0.copy(), original_first_frame=original, diagnostic_candidate=initial)
    records = []
    for label, qpos in states.items():
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for side in ("right", "left"):
            result = dict(state=label, side=side, **native_intersection(model, data, meshes, side))
            records.append(result)
            print(f"{label} {side}: intersection {result['intersection_volume_m3'] * 1e9:.6f} mm^3", flush=True)
    sweep = []
    side = "left"
    bend = model.joint(f"{side}_hand_thumb_bend_joint")
    rotate = model.joint(f"{side}_hand_thumb_rota_joint1")
    for bend_value in np.linspace(*bend.range, 5):
        for rotate_value in np.linspace(*rotate.range, 7):
            data.qpos[:] = initial
            data.qpos[int(bend.qposadr[0])] = bend_value
            data.qpos[int(rotate.qposadr[0])] = rotate_value
            mujoco.mj_forward(model, data)
            result = dict(bend_rad=float(bend_value), rotate_rad=float(rotate_value),
                          **native_intersection(model, data, meshes, side))
            sweep.append(result)
        print(f"left native joint sweep bend={bend_value:.4f}: {len(sweep)}/35", flush=True)
    source_checks = {side: kinematic_source_check(model, source_assets / f"xhand_{side}.urdf", side)
                     for side in ("right", "left")}
    if preserved != [artifact(p) for p in inputs]:
        raise RuntimeError("source inputs changed during audit")
    verify_artifacts(mesh_snapshot)
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for source in inputs[3:]:
        destination = output / source.name
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
        copied.append(dict(source=artifact(source), local_copy=artifact(destination)))
    report = dict(status="native_thumb_assembly_audit_not_collision_exemption_or_reset", source_checks=source_checks,
        states=records, left_joint_sweep=sweep, sweep_is_not_all_configurations_certificate=True,
        geometry_or_pairs_modified=False, state_candidate_modified=False, simulation_steps_executed=0,
        collision_exemption_applied=False, accepted_as_reset=False, source_copies=copied,
        preserved_artifacts=preserved, source_meshes=[artifact(p) for p in mesh_paths],
        scene_meshes=mesh_snapshot,
        boolean_library_version=version("manifold3d"), audit_code=artifact(Path(__file__)))
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-assets", type=Path, default=Path("/data_all/zzx/egoengine/spider/spider/assets/robots/xhand"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.baseline, args.candidate, args.source_assets, args.output)


if __name__ == "__main__":
    main()
