"""Build and audit the right palm/index-root semantic collision guards.

The native CAD solids are the offline oracle.  The emitted MuJoCo spheres
only participate in two explicit parent/child pairs; they never replace the
external palm or finger collision shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import fcl
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "external/mink/src")]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from audit_taco_remaining_contacts import native_pair_evidence, relative_mesh
from egoengine_repro.retarget.paper_audit import artifact

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
REFERENCE = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1/robot_reference.npz"
JOINT = "right_hand_index_bend_joint"
PALM_VISUAL = "right_hand_link_visual"
ROOT_VISUAL = "right_index_bend_visual"
PALM_BODY = "right_hand_link"
ROOT_BODY = "right_hand_index_bend_link"
RANGE = (-0.175, 0.175)


def _native_collision(model, data, meshes, palm, root, address, seed, angle):
    data.qpos[:] = seed
    data.qpos[address] = angle
    mujoco.mj_kinematics(model, data)
    result = fcl.CollisionResult()
    fcl.collide(
        triangle_object(meshes[palm]),
        triangle_object(relative_mesh(model, data, meshes, root, palm)),
        fcl.CollisionRequest(num_max_contacts=1, enable_contact=True),
        result,
    )
    return bool(result.is_collision)


def _transition(predicate, lo, hi, lo_value, iterations=34):
    if predicate(lo) != lo_value or predicate(hi) == lo_value:
        raise ValueError("native CAD transition is not bracketed")
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if predicate(mid) == lo_value:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _fmt(values):
    return " ".join(f"{float(value):.12g}" for value in np.atleast_1d(values))


def _guard_spec(label, endpoint, onset, bounds, joint_origin):
    bounds = np.asarray(bounds, dtype=float)
    palm_center = bounds.mean(axis=0)
    # Put the centre-coincidence angle just outside the physical joint range.
    # This leaves a finite contact shell at the endpoint while preserving the
    # CAD-measured clear/interference transition inside the range.
    coincidence = endpoint + np.sign(endpoint) * 0.05
    root_center = _rotation_y(-coincidence) @ (palm_center - joint_origin)
    distance_at_onset = np.linalg.norm(
        palm_center - (joint_origin + _rotation_y(onset) @ root_center)
    )
    distance_at_endpoint = np.linalg.norm(
        palm_center - (joint_origin + _rotation_y(endpoint) @ root_center)
    )
    radius = distance_at_onset / 2.0
    y_span = float(bounds[1, 1] - bounds[0, 1])
    return dict(
        label=label,
        endpoint_rad=float(endpoint),
        native_onset_rad=float(onset),
        coincidence_angle_rad=float(coincidence),
        palm_center_m=palm_center.tolist(),
        root_center_m=root_center.tolist(),
        sphere_radius_m=float(radius),
        native_patch_y_span_m=y_span,
        endpoint_penetration_m=float(distance_at_onset - distance_at_endpoint),
    )


def _add_guards(root, specs):
    palm = root.find(f".//body[@name='{PALM_BODY}']")
    index = root.find(f".//body[@name='{ROOT_BODY}']")
    contact = root.find("contact")
    if palm is None or index is None or contact is None:
        raise ValueError("right palm/index-root bodies or contact table are missing")
    if any("index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")
           for geom in root.findall(".//geom")):
        raise ValueError("right index-root guards already exist")
    for spec in specs:
        label = spec["label"]
        common = dict(
            type="sphere",
            size=_fmt(spec["sphere_radius_m"]),
            group="3",
            density="0",
            rgba="1 0.75 0 1",
            contype="0",
            conaffinity="0",
            condim="3",
            friction="1 0.1 0",
        )
        palm_name = f"collision_hand_right_palm_index_root_{label}_guard"
        root_name = f"collision_hand_right_index_root_{label}_guard"
        ET.SubElement(palm, "geom", name=palm_name, pos=_fmt(spec["palm_center_m"]), **common)
        ET.SubElement(index, "geom", name=root_name, pos=_fmt(spec["root_center_m"]), **common)
        ET.SubElement(
            contact,
            "pair",
            name=f"semantic_right_palm_index_root_{label}_guard",
            geom1=palm_name,
            geom2=root_name,
            condim="3",
            friction="1 1 0.1 0 0",
        )


def _guard_transition(model, data, seed, address, pair, clear_angle, overlap_angle):
    def penetrating(angle):
        data.qpos[:] = seed
        data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        return mujoco.mj_geomDistance(model, data, *pair, 0.05, None) < 0.0

    if penetrating(clear_angle) or not penetrating(overlap_angle):
        raise ValueError("guard transition is not bracketed")
    lo, hi = clear_angle, overlap_angle
    for _ in range(34):
        mid = (lo + hi) / 2.0
        if penetrating(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _unchanged_non_guard_contract(before, after):
    def snapshot(root):
        geoms = []
        for body in root.findall(".//body"):
            for geom in body.findall("geom"):
                if "guard" not in geom.get("name", ""):
                    geoms.append((body.get("name"), tuple(sorted(geom.attrib.items()))))
        pairs = [tuple(sorted(pair.attrib.items())) for pair in root.findall("contact/pair")
                 if "guard" not in pair.get("name", "")]
        assets = [tuple(sorted(node.attrib.items())) for node in root.findall("asset/*")]
        return geoms, pairs, assets
    return snapshot(before) == snapshot(after)


def run(scene, reference, output_scene, report_path, baseline_scene=None,
        retarget_run=None, comparison_run=None):
    source_model = mujoco.MjModel.from_xml_path(str(scene))
    source_data = mujoco.MjData(source_model)
    meshes, _ = visual_meshes(scene, source_model)
    with np.load(reference, allow_pickle=False) as archive:
        seed = np.asarray(archive["qpos"][0], dtype=float)
    address = int(source_model.joint(JOINT).qposadr[0])
    palm = source_model.geom(PALM_VISUAL).id
    index = source_model.geom(ROOT_VISUAL).id
    predicate = lambda q: _native_collision(
        source_model, source_data, meshes, palm, index, address, seed, q
    )
    if not predicate(RANGE[0]) or predicate(0.0) or not predicate(RANGE[1]):
        raise ValueError("unexpected native CAD collision topology")
    negative_onset = _transition(predicate, RANGE[0], 0.0, True)
    positive_onset = _transition(predicate, 0.0, RANGE[1], False)

    endpoint_evidence = {}
    for label, angle in (("negative", RANGE[0]), ("neutral", 0.0), ("positive", RANGE[1])):
        source_data.qpos[:] = seed
        source_data.qpos[address] = angle
        mujoco.mj_kinematics(source_model, source_data)
        endpoint_evidence[label] = native_pair_evidence(
            meshes[palm], relative_mesh(source_model, source_data, meshes, index, palm)
        )
    specs = [
        _guard_spec("negative", RANGE[0], negative_onset,
                    endpoint_evidence["negative"]["intersection_bounds_m"],
                    source_model.body(ROOT_BODY).pos),
        _guard_spec("positive", RANGE[1], positive_onset,
                    endpoint_evidence["positive"]["intersection_bounds_m"],
                    source_model.body(ROOT_BODY).pos),
    ]

    tree = ET.parse(scene)
    root = tree.getroot()
    formal_audit = output_scene is None
    if formal_audit:
        if baseline_scene is None:
            raise ValueError("formal audit requires the superseded baseline scene")
        baseline_root = ET.parse(baseline_scene).getroot()
        audited_scene = scene
        source_pairs = len(baseline_root.findall("contact/pair"))
        external_unchanged = _unchanged_non_guard_contract(baseline_root, root)
    else:
        _add_guards(root, specs)
        output_scene.parent.mkdir(parents=True, exist_ok=True)
        # A candidate outside the source model directory needs an absolute meshdir.
        compiler = root.find("compiler")
        source_meshdir = (scene.parent / compiler.get("meshdir", ".")).resolve()
        compiler.set("meshdir", str(source_meshdir))
        ET.indent(tree, space="  ")
        tree.write(output_scene, encoding="unicode")
        audited_scene = output_scene
        source_pairs = len(ET.parse(scene).getroot().findall("contact/pair"))
        external_unchanged = True

    model = mujoco.MjModel.from_xml_path(str(audited_scene))
    data = mujoco.MjData(model)
    candidate_address = int(model.joint(JOINT).qposadr[0])
    transitions = {}
    sampled = np.linspace(RANGE[0], RANGE[1], 141)
    mismatch = {}
    guard_ids = {}
    for spec in specs:
        label = spec["label"]
        pair = (
            model.geom(f"collision_hand_right_palm_index_root_{label}_guard").id,
            model.geom(f"collision_hand_right_index_root_{label}_guard").id,
        )
        guard_ids[label] = pair
        if label == "negative":
            transition = _guard_transition(model, data, seed, candidate_address, pair, 0.0, RANGE[0])
            guard_label = sampled < transition
            native_label = sampled < negative_onset
        else:
            transition = _guard_transition(model, data, seed, candidate_address, pair, 0.0, RANGE[1])
            guard_label = sampled > transition
            native_label = sampled > positive_onset
        transitions[label] = float(transition)
        mismatch[label] = dict(
            sampled_false_positive_count=int(np.count_nonzero(guard_label & ~native_label)),
            sampled_false_negative_count=int(np.count_nonzero(~guard_label & native_label)),
            onset_error_rad=float(abs(transition - spec["native_onset_rad"])),
        )

    guard_names = {geom.get("name") for geom in root.findall(".//geom") if "index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")}
    guard_pairs = [pair for pair in root.findall("contact/pair")
                   if pair.get("geom1") in guard_names or pair.get("geom2") in guard_names]
    isolated = len(guard_names) == 4 and len(guard_pairs) == 2 and all(
        pair.get("geom1") in guard_names and pair.get("geom2") in guard_names for pair in guard_pairs
    )
    comparison_range = (np.fromstring(baseline_root.find(f".//joint[@name='{JOINT}']").get("range"), sep=" ")
                        if formal_audit else source_model.joint(JOINT).range)
    joint_range_unchanged = np.array_equal(comparison_range, model.joint(JOINT).range)
    passed = (
        isolated
        and external_unchanged
        and joint_range_unchanged
        and len(root.findall("contact/pair")) == source_pairs + 2
        and all(row["sampled_false_positive_count"] == 0
                and row["sampled_false_negative_count"] == 0
                and row["onset_error_rad"] < 1e-8 for row in mismatch.values())
    )
    trajectory = None
    comparison = None
    extra_sources = []
    if formal_audit:
        if retarget_run is None or comparison_run is None:
            raise ValueError("formal audit requires active and pre-guard retarget runs")
        retarget_report = json.loads((retarget_run / "retarget_report.json").read_text())
        if retarget_report["scene_sha256"] != artifact(scene)["sha256"]:
            raise ValueError("active retarget was not generated from the formal guarded scene")
        with np.load(retarget_run / "robot_reference.npz", allow_pickle=False) as archive:
            trajectory_qpos = np.asarray(archive["qpos"], dtype=float)
        angles = trajectory_qpos[:, candidate_address]
        distance_rows = {label: [] for label in guard_ids}
        for qpos in trajectory_qpos:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            for label, pair in guard_ids.items():
                distance_rows[label].append(
                    mujoco.mj_geomDistance(model, data, *pair, 0.05, None)
                )
        guard_minimum = {label: float(np.min(values))
                         for label, values in distance_rows.items()}
        native_invalid = (angles < negative_onset) | (angles > positive_onset)
        trajectory = dict(
            frames=len(trajectory_qpos),
            right_index_angle_range_rad=[float(angles.min()), float(angles.max())],
            native_interference_frames=int(np.count_nonzero(native_invalid)),
            guard_minimum_distance_m=guard_minimum,
            guard_penetrating_frames={label: int(np.count_nonzero(np.asarray(values) < 0.0))
                                      for label, values in distance_rows.items()},
            kinematic_model_feasible=retarget_report["kinematic_model_feasible"],
            solver_primal_tolerance=retarget_report["solver_primal_tolerance"],
            solver_dual_tolerance=retarget_report["solver_dual_tolerance"],
            self_collision_clearance_m=retarget_report["effective_settings"]["self_collision_clearance_m"],
        )
        old_report = json.loads((comparison_run / "retarget_report.json").read_text())
        with np.load(comparison_run / "robot_reference.npz", allow_pickle=False) as archive:
            old_angles = np.asarray(archive["qpos"], dtype=float)[:, candidate_address]
        metrics = ("fingertip_mean_error_m", "fingertip_max_error_m", "wrist_mean_error_rad")
        comparison = dict(
            pre_guard_native_interference_frames=int(np.count_nonzero(
                (old_angles < negative_onset) | (old_angles > positive_onset))),
            post_guard_native_interference_frames=trajectory["native_interference_frames"],
            baseline={key: old_report[key] for key in metrics},
            guarded={key: retarget_report[key] for key in metrics},
            guarded_minus_baseline={key: (np.asarray(retarget_report[key]) -
                                           np.asarray(old_report[key])).tolist()
                                    for key in metrics},
        )
        extra_sources = [artifact(retarget_run / name) for name in
                         ("robot_reference.npz", "retarget_report.json")]
        extra_sources += [artifact(comparison_run / name) for name in
                          ("robot_reference.npz", "retarget_report.json")]
        passed = (passed and trajectory["kinematic_model_feasible"]
                  and trajectory["native_interference_frames"] == 0
                  and min(trajectory["guard_minimum_distance_m"].values()) >= 0.0)
    report = dict(
        status=(("right_index_root_guard_formal_audit_passed" if formal_audit
                 else "right_index_root_guard_candidate_passed") if passed else
                ("right_index_root_guard_formal_audit_failed" if formal_audit
                 else "right_index_root_guard_candidate_failed")),
        native_clear_interval_rad=[float(negative_onset), float(positive_onset)],
        native_endpoint_evidence=endpoint_evidence,
        guards=specs,
        guard_transition_rad=transitions,
        classification_check=mismatch,
        sampled_angles=141,
        guards_isolated_to_two_explicit_pairs=isolated,
        source_contact_pair_count=source_pairs,
        candidate_contact_pair_count=len(root.findall("contact/pair")),
        joint_range_rad=model.joint(JOINT).range.tolist(),
        joint_range_unchanged=joint_range_unchanged,
        external_collision_geometries_unchanged=external_unchanged,
        active_retarget_trajectory=trajectory,
        retarget_comparison=comparison,
        audited_scene=artifact(audited_scene),
        sources=[artifact(scene), artifact(reference)] +
                ([artifact(baseline_scene)] if baseline_scene is not None else []) + extra_sources,
        generation_code=artifact(Path(__file__)),
        formal_scene_modified=formal_audit,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError("right index-root guard candidate did not pass")
    print(json.dumps({key: report[key] for key in (
        "status", "native_clear_interval_rad", "guard_transition_rad",
        "classification_check", "guards_isolated_to_two_explicit_pairs",
        "joint_range_unchanged")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output-scene", type=Path)
    parser.add_argument("--baseline-scene", type=Path)
    parser.add_argument("--retarget-run", type=Path)
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.reference, args.output_scene, args.report, args.baseline_scene,
        args.retarget_run, args.comparison_run)
