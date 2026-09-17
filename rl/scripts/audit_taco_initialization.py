"""Inspect a selected candidate without resetting, stepping or repairing it."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
import mink
from egoengine_repro.retarget.collision_audit import (
    collision_families, distances, explicit_hand_pairs, hand_ids, nonadjacent_hand_pairs, validate_qpos,
)
from egoengine_repro.retarget.mink import _enable_planning_collision_masks, _explicit_collision_groups
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.paper_audit import artifact, alignment_invariants
from egoengine_repro.retarget.taco_bimanual import pose7

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_brush_brush_bowl_20230927_027/scene_source_contacts_mass.xml"
RUN = ROOT / "runs/taco_brush_bimanual_gt_v4"
TEMPLATE = ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"


def visual_meshes(scene, model):
    root = ET.parse(scene).getroot()
    mesh_dir = scene.parent / root.find("compiler").get("meshdir", "")
    assets = {e.get("name"): e for e in root.findall("asset/mesh")}
    result, inputs = {}, []
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if not name.endswith("visual") or geom.get("type") != "mesh":
            continue
        if not name.startswith(("right_", "left_")):
            continue
        entry = assets[geom.get("mesh")]
        path = mesh_dir / entry.get("file")
        mesh = trimesh.load_mesh(path, process=True)
        mesh.apply_scale(np.fromstring(entry.get("scale", "1 1 1"), sep=" "))
        # These source visual geoms have identity local transforms. Fail instead
        # of silently confusing native vertices with MuJoCo's centered mesh frame.
        if any(geom.get(k) is not None for k in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError(f"explicit visual transform needs handling: {name}")
        gid = model.geom(name).id
        result[gid] = mesh
        inputs.append(path)
    return result, inputs


def world_vertices(model, data, geom, mesh):
    body = model.geom_bodyid[geom]
    return mesh.vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]


def topology(model, qpos, meshes):
    qpos = validate_qpos(model, qpos, trajectory=True)
    hands = hand_ids(model)
    explicit = set(explicit_hand_pairs(model))
    nonadjacent = set(nonadjacent_hand_pairs(model))
    omitted = [pair for pair in combinations(hands, 2) if pair not in explicit
               and model.geom(pair[0]).name.split("_")[2] == model.geom(pair[1]).name.split("_")[2]]
    values = np.empty((len(qpos), len(omitted)))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    zero = distances(model, data, omitted)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = distances(model, data, omitted)
    result = []
    mesh_by_body = {int(model.geom_bodyid[g]): (g, m) for g, m in meshes.items()}
    for index, pair in enumerate(omitted):
        a, b = pair
        frame = int(values[:, index].argmin())
        kind = "nonadjacent_unclassified" if pair in nonadjacent else "assembly_adjacent_not_automatically_excluded"
        report = dict(geom1=model.geom(a).name, geom2=model.geom(b).name, classification=kind,
                      zero_configuration_shell_distance_m=float(zero[index]),
                      minimum_shell_distance_m=float(values[frame, index]),
                      worst_source_row_zero_based=frame,
                      penetrating_rows_50um=int((values[:, index] < -5e-5).sum()))
        # Limit expensive native surface samples to penetrating nonadjacent pairs.
        # Positive watertight containment is evidence; absent samples are not a
        # no-collision certificate, and open meshes do not support signed claims.
        if pair in nonadjacent and values[frame, index] < -5e-5:
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            sampled = []
            for ga, gb in ((a, b), (b, a)):
                source = mesh_by_body.get(int(model.geom_bodyid[ga]))
                target = mesh_by_body.get(int(model.geom_bodyid[gb]))
                if source is None or target is None:
                    sampled.append(dict(status="native_mesh_missing"))
                    continue
                sa, ma = source
                sb, mb = target
                if not mb.is_watertight:
                    sampled.append(dict(status="target_open_mesh_signed_containment_not_used"))
                    continue
                points = world_vertices(model, data, sa, ma)
                points = points[np.linspace(0, len(points) - 1, min(256, len(points)), dtype=int)]
                body = model.geom_bodyid[sb]
                local = (points - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                signed = closed_mesh_signed_distance(mb, local)
                if not np.isfinite(signed).all():
                    raise ValueError("nonfinite native signed distance")
                sampled.append(dict(status="sampled_native_surface_not_global_bound", samples=len(points),
                    source_visual=model.geom(sa).name, target_visual=model.geom(sb).name,
                    inside_samples_50um=int((signed > 5e-5).sum()),
                    sampled_penetration_max_m=float(max(0, signed.max()))))
            report["native_surface_samples_forward_reverse"] = sampled
        result.append(report)
    return dict(omitted_intrahand_pairs=len(omitted),
                classifications=dict(Counter(r["classification"] for r in result)),
                omitted_nonadjacent_penetrating_pairs=sum(
                    r["classification"] == "nonadjacent_unclassified" and r["penetrating_rows_50um"] > 0
                    for r in result),
                decisions_applied=False, pairs=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/taco_v1/dev4")
    parser.add_argument("--sequence", default="(brush, brush, bowl)/20230927_027")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scene, run = args.scene.resolve(strict=True), args.run.resolve(strict=True)
    inputs = [scene, TEMPLATE, run / "human_reference.npz", run / "robot_reference.npz",
              run / "retarget_report.json"]
    preserved = [artifact(p) for p in inputs]
    retarget_report = json.loads((run / "retarget_report.json").read_text())
    if retarget_report["scene_sha256"] != artifact(scene)["sha256"]:
        raise ValueError("reference was generated against a different scene")
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(run / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    with np.load(run / "robot_reference.npz", allow_pickle=False) as source:
        robot = dict(source)
    qpos = robot["qpos"]
    if len(qpos) != len(human["frame_indices"]) or not np.array_equal(robot["frame_indices"], human["frame_indices"]):
        raise ValueError("audit requires the full matching human/robot source timeline")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = robot["qvel"][0]
    data.ctrl[:] = robot["ctrl"][0]
    mujoco.mj_forward(model, data)
    families = collision_families(model)
    initial = {}
    for key, pairs in families.items():
        d = distances(model, data, pairs)
        worst = int(d.argmin())
        a, b = pairs[worst]
        initial[key] = dict(min_distance_m=float(d[worst]), geom1=model.geom(a).name,
                            geom2=model.geom(b).name, pairs_below_50um=int((d < -5e-5).sum()))

    meshes, mesh_inputs = visual_meshes(scene, model)
    native = []
    for g, mesh in meshes.items():
        points = world_vertices(model, data, g, mesh)
        mesh_id = model.geom_dataid[g]
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        compiled = model.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        native.append(dict(geom=model.geom(g).name, watertight=bool(mesh.is_watertight),
            initial_table_clearance_m=float(points[:, 2].min() - .72),
            native_compiled_world_bounds_max_error_m=float(np.abs(
                np.stack([points.min(0), points.max(0)]) - np.stack([compiled.min(0), compiled.max(0)])).max())))
    print("Initial shell and native support measurements complete", flush=True)

    scene_root, template_root = ET.parse(scene).getroot(), ET.parse(TEMPLATE).getroot()
    geometry_keys = ("type", "size", "pos", "quat", "mesh", "fromto")
    shell_attributes = lambda root: {g.get("name"): [g.get(k) for k in geometry_keys]
        for g in root.findall(".//geom") if g.get("name", "").startswith("collision_hand_")}
    runtime = set(explicit_hand_pairs(model))
    planning = mujoco.MjModel.from_xml_path(str(scene))
    hands = set(hand_ids(planning))
    _enable_planning_collision_masks(planning, list(hands), [])
    mink_pairs = set(mink.CollisionAvoidanceLimit(planning, _explicit_collision_groups(
        planning, mujoco, hand_geom_ids=hands), include_explicit_pairs=True).geom_id_pairs)
    limited_joints = [j for j in range(model.njnt) if model.jnt_limited[j]]
    addresses = model.jnt_qposadr[limited_joints]
    ranges = model.jnt_range[limited_joints]
    margins = np.minimum(qpos[:, addresses] - ranges[:, 0], ranges[:, 1] - qpos[:, addresses])
    pose_errors = []
    for index, side in enumerate(("right", "left")):
        address = int(model.joint(f"{side}_object_joint").qposadr[0])
        expected = np.stack([pose7(p) for p in human["T_sim_object_reference"][:, index]])
        pose_errors.append(float(np.abs(qpos[:, address:address + 7] - expected).max()))
    raw, sequence = args.data_root, args.sequence
    joint_path = raw / "hand_poses/Hand_Poses" / sequence / "hand_joints.npy"
    object_paths = []
    for role in ("tool", "target"):
        paths = list((raw / "object_poses/Object_Poses" / sequence).glob(f"{role}_*.npy"))
        if len(paths) != 1:
            raise ValueError(f"expected one {role} pose source")
        object_paths.append(paths[0])
    camera_path = raw / "camera/Egocentric_Camera_Parameters" / sequence / "egocentric_frame_extrinsic.npy"
    joints = np.load(joint_path).astype(float)
    objects = np.stack([np.load(p).astype(float) for p in object_paths], axis=1)
    transform = human["T_sim_world"]
    aligned_joints = joints[:, [1, 0]] @ transform[:3, :3].T + transform[:3, 3]
    alignment = alignment_invariants(joints, objects, np.load(camera_path).astype(float), transform)
    alignment.update(exported_joints_max_error_m=float(np.abs(aligned_joints - human["joint_positions_sim"]).max()),
        exported_object_transforms_max_error=float(np.abs(transform @ objects - human["T_sim_object_reference"]).max()),
        reference_object_qpos_max_errors=pose_errors, table_z_m=float(model.geom("floor").pos[2]),
        native_compiled_scale_check_max_error_m=max(r["native_compiled_world_bounds_max_error_m"] for r in native))

    # These are lower bounds for vertical translations, not accepted reset states.
    hand_shifts = {side: max(0.0, -min(mujoco.mj_geomDistance(model, data, g, model.geom("floor").id, .05, None)
        for g in hand_ids(model) if model.geom(g).name.startswith(f"collision_hand_{side}_"))) for side in ("right", "left")}
    tool_shift = max(0.0, -initial["tool_floor"]["min_distance_m"])
    target_shift = max(0.0, -initial["target_floor"]["min_distance_m"])
    alternatives = dict(status="analytic_translation_lower_bounds_only_not_selected_or_applied",
        raise_hands_independently_m=hand_shifts, raise_tool_for_shell_table_clearance_m=tool_shift,
        raise_target_for_shell_table_clearance_m=target_shift,
        raise_all_dynamic_bodies_m=max(*hand_shifts.values(), tool_shift, target_shift),
        common_raise_consequence="preserves relative poses but lifts the initially supported target off the table",
        independent_raise_consequence="changes initial hand/object relations; does not establish all-pair feasibility or continuity",
        qvel_choice="unpublished; neither zero nor reference derivative selected as validated reset")
    print("Auditing every omitted intra-hand pair and native surface samples", flush=True)
    coverage = topology(model, qpos, meshes)
    assets = {p.resolve() for p in mesh_inputs}
    report = dict(status="initialization_unresolved_no_physics_rollout", frames=len(qpos),
        native_distance_backend="Open3D signed distance, 11 rays, positive inside",
        distance_code=artifact(ROOT / "src/egoengine_repro/retarget/mesh_distance.py"),
        sequence=sequence, data_root=str(raw.resolve()), scene=str(scene),
        paper_basis=["3.2.1 Eq.1", "A.1-A.3", "3.2.2 contact refinement", "6 limitations"],
        alignment=alignment, initial_distances=initial, native_visual_support=native,
        topology=dict(runtime_hand_pairs=len(runtime), mink_hand_pairs=len(mink_pairs),
            ik_runtime_pairs_equal=runtime == mink_pairs, physical_pairs=model.npair,
            hand_shell_geometry_equal_pinned_source=shell_attributes(scene_root) == shell_attributes(template_root),
            complete_intrahand_geometry_coverage=False, omitted=coverage),
        joint_limit_min_margin=float(margins.min()), joint_limit_violating_rows=int((margins < -1e-6).any(axis=1).sum()),
        initial_qvel=dict(hand_max_abs=float(np.abs(robot["qvel"][0, :36]).max()),
            tool=robot["qvel"][0, 36:42].tolist(), target=robot["qvel"][0, 42:48].tolist(),
            provenance="existing_reference_finite_difference_not_author_reset"),
        minimal_alternatives=alternatives, strict_gate_passed=False, rl_validation_completed=False,
        state_projection_applied=False, simulation_steps_executed=0,
        preserved_artifacts=preserved, source_assets=[artifact(p) for p in sorted(assets)],
        raw_inputs=[artifact(p) for p in [joint_path, *object_paths, camera_path]],
        audit_code=[artifact(Path(__file__))])
    if preserved != [artifact(p) for p in inputs]:
        raise RuntimeError("preserved source artifacts changed during the audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(initial=initial, alignment=alignment,
                         coverage_summary={k: v for k, v in coverage.items() if k != "pairs"})), flush=True)


if __name__ == "__main__":
    main()
