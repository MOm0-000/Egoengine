"""One isolated surface-tip/longitudinal-axis convention, not author calibration."""

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_tip_semantics import point_geometry, unit
from audit_taco_finger_axis_compatibility import pairwise_orientation_rms_lower_bound
from diagnose_taco_retarget_objectives import load_inputs, RUN, UPSTREAM
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from egoengine_repro.retarget.schema import validate_human_reference
from egoengine_repro.retarget.taco_bimanual import FINGERS, SIDES, geometric_frame, retarget

CONVENTION = "local_native_surface_tip_urdf_longitudinal_fixed_palm_roll_v1"


def rotation_steps(rotations):
    relative = rotations[:-1].swapaxes(-1, -2) @ rotations[1:]
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()


def build_contract(scene, human):
    """Derive points/axes from models only; never fit offsets to motion errors."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    meshes, _ = visual_meshes(scene, model)
    target = {key: value.copy() for key, value in human.items()}
    records = {}
    for hi, side in enumerate(SIDES):
        urdf = ET.parse(UPSTREAM / f"xhand_{side}.urdf").getroot()
        root = model.body(f"{side}_hand_link").id
        palm = model.site(f"{side}_palm").id
        middle = model.body(f"{side}_hand_mid_link1").id
        palm_frame = geometric_frame(zero.site_xmat[palm].reshape(3, 3)[:, 0],
                                     zero.xpos[middle] - zero.xpos[root])
        for fi, finger in enumerate(FINGERS):
            name = f"{side}_{finger}_tip"
            sid = model.site(name).id
            bid = int(model.site_bodyid[sid])
            candidates = [mesh for gid, mesh in meshes.items() if model.geom_bodyid[gid] == bid]
            if len(candidates) != 1:
                raise ValueError(f"expected one native distal surface for {name}")
            mesh = candidates[0]
            joints = [j for j in urdf.findall("joint") if j.get("type") == "fixed"
                      and j.find("parent").get("link") == model.body(bid).name
                      and j.find("child").get("link").endswith("tip")]
            if len(joints) != 1:
                raise ValueError(f"ambiguous URDF endpoint for {name}")
            endpoint = np.fromstring(joints[0].find("origin").get("xyz"), sep=" ")
            axis = unit(endpoint)
            projection = point_geometry(mesh, endpoint, axis)
            point = np.asarray(projection["nearest_surface_m"])
            distance = trimesh.proximity.closest_point(mesh, point[None])[1][0]
            if distance > 1e-10:
                raise ValueError(f"projected point is not on native surface: {name}")
            Rsite = zero.site_xmat[sid].reshape(3, 3)
            Rbody = zero.xmat[bid].reshape(3, 3)
            old = Rsite.T @ geometric_frame(palm_frame[:, 0], zero.site_xpos[sid] - zero.xpos[bid])
            new = Rsite.T @ geometric_frame(palm_frame[:, 0], Rbody @ axis)
            old_target = human["T_sim_fingertip_target"][:, hi, fi, :3, :3]
            target["T_sim_fingertip_target"][:, hi, fi, :3, :3] = old_target @ old @ new.T
            records[name] = dict(body=model.body(bid).name, urdf_joint=joints[0].get("name"),
                inherited_site_local_m=model.site_pos[sid].tolist(),
                surface_site_local_m=point.tolist(), urdf_endpoint_local_m=endpoint.tolist(),
                longitudinal_axis_body_local=axis.tolist(),
                inherited_anatomical_frame_site_local=old.tolist(),
                candidate_anatomical_frame_site_local=new.tolist(),
                endpoint_projection=projection, projected_point_surface_distance_m=float(distance),
                site_displacement_m=float(np.linalg.norm(point - model.site_pos[sid])),
                calibration_change_deg=float(np.rad2deg(Rotation.from_matrix(old.T @ new).magnitude())))

    target["fingertip_orientation_source"] = np.asarray(CONVENTION)
    target["fingertip_geometry_contract"] = np.asarray(CONVENTION)
    target["diagnostic_only"] = np.asarray(True)
    validate_human_reference(target)
    for key in human:
        if key not in ("T_sim_fingertip_target", "fingertip_orientation_source"):
            np.testing.assert_array_equal(target[key], human[key])
    np.testing.assert_array_equal(target["T_sim_fingertip_target"][..., :3, 3],
                                  human["T_sim_fingertip_target"][..., :3, 3])
    step_error = np.abs(rotation_steps(target["T_sim_fingertip_target"][..., :3, :3]) -
                        rotation_steps(human["T_sim_fingertip_target"][..., :3, :3])).max()
    if step_error > 1e-12:
        raise ValueError("constant calibration changed angular step sizes")
    contract = dict(convention=CONVENTION, not_author_calibration=True, not_formal_initializer=True,
        fitted_to_episode=False, human_GT_positions_unchanged=True,
        human_point="released MANO tip surface vertex, not inferred task contact center",
        robot_point="nearest native distal-mesh triangle point to URDF named tip endpoint",
        human_z="neutral DIP-to-GT-tip unit vector transported by MANO distal rotational FK",
        robot_z="distal-link origin to URDF named tip endpoint, transported by robot FK",
        x="neutral palm normal projected perpendicular to z, then transported by distal FK",
        y="cross(z, x), right-handed frame; roll is a local convention, not measured pad/nail normal",
        position_comparison="both origins in the same fixed TACO-to-simulator world transform, meters",
        rotation_comparison="R_robot_site @ C_robot versus R_MANO_distal @ C_MANO (in simulator world)",
        target_change="R_target_new = R_target_old @ C_old @ C_new.T",
        angular_step_max_change_rad=float(step_error), sites=records,
        limitations=["No exact author point/roll calibration was recovered from section 3.2 or Appendix C.",
            "Point projection is deterministic geometry, not evidence of an identical functional contact point.",
            "MANO is skinned: the released surface point need not be rigid relative to its distal joint.",
            "No collision geometry, contact alias, table transform, cost or initialization rule is changed."])
    return target, contract


def write_scene(source, destination, contract):
    if destination.exists():
        raise FileExistsError(destination)
    tree = ET.parse(source)
    root = tree.getroot()
    if root.findall(".//include"):
        raise ValueError("diagnostic requires flattened scene")
    compiler = root.find("compiler")
    directory = (source.parent / compiler.get("meshdir", "")).resolve(strict=True)
    compiler.set("meshdir", str(directory))
    for name, record in contract["sites"].items():
        sites = root.findall(f".//site[@name='{name}']")
        if len(sites) != 1:
            raise ValueError(f"expected exactly one site {name}")
        sites[0].set("pos", " ".join(format(v, ".17g") for v in record["surface_site_local_m"]))
    with destination.open("xb") as stream:
        tree.write(stream, encoding="utf-8")
    return verify_scene(source, destination, contract)


def verify_scene(source, candidate, contract):
    """Check the complete XML except allowed site positions and meshdir spelling."""
    original = ET.parse(source).getroot()
    revised = ET.parse(candidate).getroot()
    normalized = copy.deepcopy(revised)
    normalized.find("compiler").attrib = dict(original.find("compiler").attrib)
    for name, record in contract["sites"].items():
        old_site = original.find(f".//site[@name='{name}']")
        new_site = normalized.find(f".//site[@name='{name}']")
        np.testing.assert_allclose(np.fromstring(new_site.get("pos"), sep=" "),
                                   record["surface_site_local_m"], atol=1e-15, rtol=0)
        new_site.set("pos", old_site.get("pos"))
    if ET.tostring(original) != ET.tostring(normalized):
        raise ValueError("scene changed beyond ten tip positions and compiler path")
    if scene_mesh_artifacts(source) != scene_mesh_artifacts(candidate):
        raise ValueError("scene mesh assets differ")
    first, second = [mujoco.MjModel.from_xml_path(str(p)) for p in (source, candidate)]
    checked = []
    for name in dir(first):
        if not name.startswith(("body_", "geom_", "mesh_", "jnt_", "dof_", "pair_", "exclude_", "actuator_", "eq_", "qpos")):
            continue
        a, b = getattr(first, name), getattr(second, name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=f"physical model changed: {name}")
            checked.append(name)
    changed = set(np.flatnonzero(np.any(first.site_pos != second.site_pos, axis=1)))
    if changed != {first.site(name).id for name in contract["sites"]}:
        raise ValueError("unexpected site changes")
    np.testing.assert_array_equal(first.site_quat, second.site_quat)
    return dict(physical_XML_unchanged=True, source_mesh_hashes_unchanged=True,
        changed_site_count=len(changed), contact_and_trace_aliases_unchanged=True,
        compiled_physical_arrays_identical=checked)


def measure(scene, human, qpos):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    position, orientation, wrist = [], [], []
    for row, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ps, rs, ws = [], [], []
        for hi, side in enumerate(SIDES):
            ids = [model.site(f"{side}_{finger}_tip").id for finger in FINGERS]
            targets = human["T_sim_fingertip_target"][row, hi]
            ps.append(np.linalg.norm(data.site_xpos[ids] - targets[:, :3, 3], axis=-1))
            relative = data.site_xmat[ids].reshape(5, 3, 3).swapaxes(-1, -2) @ targets[:, :3, :3]
            rs.append(Rotation.from_matrix(relative).magnitude())
            bid = model.body(f"{side}_hand_link").id
            ws.append(Rotation.from_matrix(data.xmat[bid].reshape(3, 3).T @
                      human["T_sim_wrist_target"][row, hi, :3, :3]).magnitude())
        position.append(ps)
        orientation.append(rs)
        wrist.append(ws)
    position, orientation, wrist = map(np.asarray, (position, orientation, wrist))
    return dict(mean_position_m_by_hand=position.mean(axis=(0, 2)).tolist(),
        mean_position_m_by_hand_finger=position.mean(axis=0).tolist(),
        max_position_m_by_hand=position.max(axis=(0, 2)).tolist(),
        mean_orientation_deg_by_hand=np.rad2deg(orientation.mean(axis=(0, 2))).tolist(),
        mean_orientation_deg_by_hand_finger=np.rad2deg(orientation.mean(axis=0)).tolist(),
        mean_wrist_deg_by_hand=np.rad2deg(wrist.mean(axis=0)).tolist(),
        first_position_m_by_hand_finger=position[0].tolist(),
        position_error_m=position.tolist())


def run(output):
    if output.exists():
        raise FileExistsError(output)
    scene, settings, human, robot, inputs = load_inputs()
    semantics = ROOT / "runs/taco_pour_tip_semantics_v1/report.json"
    semantic_report = json.loads(semantics.read_text())
    verify_artifacts(semantic_report["inputs"] + semantic_report["code"])
    inputs += [artifact(semantics), *semantic_report["inputs"]]
    code = [artifact(Path(__file__)), artifact(ROOT / "scripts/audit_taco_tip_semantics.py"),
            artifact(ROOT / "src/egoengine_repro/retarget/taco_bimanual.py"),
            artifact(ROOT / "src/egoengine_repro/retarget/mink.py")]
    target, contract = build_contract(scene, human)
    output.mkdir(parents=True)
    candidate_scene = output / "diagnostic_scene.xml"
    contract["scene_invariants"] = write_scene(scene, candidate_scene, contract)
    human_path = output / "human_reference.npz"
    with human_path.open("xb") as stream:
        np.savez_compressed(stream, **target)
    contract.update(inputs=inputs, code=code, baseline=str(RUN))
    with (output / "geometry_contract.json").open("x") as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
    result = retarget(candidate_scene, human_path, settings, output)
    with np.load(output / "robot_reference.npz", allow_pickle=False) as src:
        candidate = dict(src)
    np.testing.assert_allclose(candidate["qpos"][:, 36:], robot["qpos"][:, 36:], atol=1e-12, rtol=0)
    comparisons = {}
    for point_label, model_path, reference in (("inherited", scene, human),
                                               ("surface", candidate_scene, target)):
        for state_label, states in (("original", robot["qpos"]), ("candidate", candidate["qpos"])):
            comparisons[f"{point_label}_points_at_{state_label}_qpos"] = measure(model_path, reference, states)
    np.testing.assert_allclose(comparisons["surface_points_at_candidate_qpos"]["position_error_m"],
                               candidate["fingertip_position_error_m"], atol=1e-12, rtol=0)
    bounds = {}
    for hi, side in enumerate(SIDES):
        rotations = target["T_sim_fingertip_target"][:, hi, 2:, :3, :3]
        angles, lower = pairwise_orientation_rms_lower_bound(rotations)
        bounds[side] = dict(pair_angle_deg_mean=angles.mean(axis=0).tolist(),
                           rms_orientation_lower_bound_deg_mean=float(lower.mean()))
    verify_artifacts(inputs + code)
    summary = dict(status="isolated_common_geometry_diagnostic_not_adopted", frames=len(candidate["qpos"]),
        hand_order=list(SIDES), settings_identical_to_baseline=True,
        candidate_kinematic_model_feasible=result["kinematic_model_feasible"],
        inputs_unchanged=True, object_qpos_unchanged=True, reference_overwritten=False,
        physics_validated=False, rl_validation_completed=False, comparison=comparisons,
        candidate_shared_axis_lower_bounds=bounds, geometry_contract=artifact(output / "geometry_contract.json"),
        retarget_report=artifact(output / "retarget_report.json"),
        interpretation="Different point conventions measure different physical points; the four-way comparison keeps this explicit.")
    with (output / "comparison.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value["mean_position_m_by_hand"] for key, value in comparisons.items()}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
