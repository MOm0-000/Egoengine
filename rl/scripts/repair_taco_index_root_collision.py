"""Build and audit one side's palm/index-root semantic collision guards.

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
from egoengine_repro.retarget.paper_audit import artifact, resolve_artifact_path

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
REFERENCE = ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/robot_reference.npz"
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


def _add_guards(root, specs, side, palm_body, index_body):
    palm = root.find(f".//body[@name='{palm_body}']")
    index = root.find(f".//body[@name='{index_body}']")
    contact = root.find("contact")
    if palm is None or index is None or contact is None:
        raise ValueError(f"{side} palm/index-root bodies or contact table are missing")
    if any(f"collision_hand_{side}_" in geom.get("name", "")
           and "index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")
           for geom in root.findall(".//geom")):
        raise ValueError(f"{side} index-root guards already exist")
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
        palm_name = f"collision_hand_{side}_palm_index_root_{label}_guard"
        root_name = f"collision_hand_{side}_index_root_{label}_guard"
        ET.SubElement(palm, "geom", name=palm_name, pos=_fmt(spec["palm_center_m"]), **common)
        ET.SubElement(index, "geom", name=root_name, pos=_fmt(spec["root_center_m"]), **common)
        ET.SubElement(
            contact,
            "pair",
            name=f"semantic_{side}_palm_index_root_{label}_guard",
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


def _guard_distances(model, data, seed, address, pair, angles):
    values = []
    for angle in angles:
        data.qpos[:] = seed
        data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        values.append(mujoco.mj_geomDistance(model, data, *pair, 0.05, None))
    return np.asarray(values)


def _classification(predicate, model, data, seed, address, pairs, angles):
    native = np.asarray([predicate(float(angle)) for angle in angles], dtype=bool)
    guard = np.any(
        np.stack([
            _guard_distances(model, data, seed, address, pair, angles) < 0.0
            for pair in pairs
        ]),
        axis=0,
    )
    return dict(
        samples=len(angles),
        native_collision_count=int(np.count_nonzero(native)),
        guard_collision_count=int(np.count_nonzero(guard)),
        false_positive_count=int(np.count_nonzero(guard & ~native)),
        false_negative_count=int(np.count_nonzero(~guard & native)),
    )


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


def _same_scene_with_rebased_meshdir(baseline_scene, candidate_scene, original_parent):
    baseline = ET.parse(baseline_scene).getroot()
    candidate = ET.parse(candidate_scene).getroot()
    baseline_compiler = baseline.find("compiler")
    candidate_compiler = candidate.find("compiler")
    baseline_assets = (original_parent / baseline_compiler.get("meshdir", ".")).resolve()
    candidate_meshdir = Path(candidate_compiler.get("meshdir", "."))
    candidate_assets = (candidate_meshdir if candidate_meshdir.is_absolute()
                        else original_parent / candidate_meshdir).resolve()
    baseline_compiler.set("meshdir", "<ASSET_ROOT>")
    candidate_compiler.set("meshdir", "<ASSET_ROOT>")
    return baseline_assets == candidate_assets and ET.tostring(baseline) == ET.tostring(candidate)


def run(scene, reference, output_scene, report_path, baseline_scene=None,
        retarget_run=None, comparison_run=None, controlled_guard_run=None, side="right"):
    if side not in ("right", "left"):
        raise ValueError("side must be right or left")
    joint = f"{side}_hand_index_bend_joint"
    palm_visual = f"{side}_hand_link_visual"
    root_visual = f"{side}_index_bend_visual"
    palm_body = f"{side}_hand_link"
    index_body = f"{side}_hand_index_bend_link"
    source_model = mujoco.MjModel.from_xml_path(str(scene))
    source_data = mujoco.MjData(source_model)
    meshes, _ = visual_meshes(scene, source_model)
    with np.load(reference, allow_pickle=False) as archive:
        seed = np.asarray(archive["qpos"][0], dtype=float)
    address = int(source_model.joint(joint).qposadr[0])
    palm = source_model.geom(palm_visual).id
    index = source_model.geom(root_visual).id
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
                    source_model.body(index_body).pos),
        _guard_spec("positive", RANGE[1], positive_onset,
                    endpoint_evidence["positive"]["intersection_bounds_m"],
                    source_model.body(index_body).pos),
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
        baseline_pair_names = {pair.get("name") for pair in baseline_root.findall("contact/pair")}
    else:
        _add_guards(root, specs, side, palm_body, index_body)
        output_scene.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the portable relative mesh path for a sibling formal scene.
        # A diagnostic candidate elsewhere needs an absolute path to compile.
        if output_scene.parent.resolve() != scene.parent.resolve():
            compiler = root.find("compiler")
            source_meshdir = (scene.parent / compiler.get("meshdir", ".")).resolve()
            compiler.set("meshdir", str(source_meshdir))
        ET.indent(tree, space="  ")
        tree.write(output_scene, encoding="unicode")
        audited_scene = output_scene
        source_pairs = len(ET.parse(scene).getroot().findall("contact/pair"))
        external_unchanged = True
        baseline_pair_names = {pair.get("name") for pair in ET.parse(scene).getroot().findall("contact/pair")}

    model = mujoco.MjModel.from_xml_path(str(audited_scene))
    data = mujoco.MjData(model)
    candidate_address = int(model.joint(joint).qposadr[0])
    transitions = {}
    sampled = np.linspace(RANGE[0], RANGE[1], 141)
    midpoint_holdout = (sampled[:-1] + sampled[1:]) / 2.0
    guard_ids = {}
    for spec in specs:
        label = spec["label"]
        pair = (
            model.geom(f"collision_hand_{side}_palm_index_root_{label}_guard").id,
            model.geom(f"collision_hand_{side}_index_root_{label}_guard").id,
        )
        guard_ids[label] = pair
        if label == "negative":
            transition = _guard_transition(model, data, seed, candidate_address, pair, 0.0, RANGE[0])
        else:
            transition = _guard_transition(model, data, seed, candidate_address, pair, 0.0, RANGE[1])
        transitions[label] = float(transition)
    classification = dict(
        sampled_grid=_classification(
            predicate, model, data, seed, candidate_address, list(guard_ids.values()), sampled
        ),
        midpoint_holdout_grid=_classification(
            predicate, model, data, seed, candidate_address,
            list(guard_ids.values()), midpoint_holdout
        ),
    )

    guard_names = {
        geom.get("name") for geom in root.findall(".//geom")
        if f"collision_hand_{side}_" in geom.get("name", "")
        and "index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")
    }
    guard_pairs = [pair for pair in root.findall("contact/pair")
                   if pair.get("geom1") in guard_names or pair.get("geom2") in guard_names]
    isolated = len(guard_names) == 4 and len(guard_pairs) == 2 and all(
        pair.get("geom1") in guard_names and pair.get("geom2") in guard_names for pair in guard_pairs
    )
    comparison_range = (np.fromstring(baseline_root.find(f".//joint[@name='{joint}']").get("range"), sep=" ")
                        if formal_audit else source_model.joint(joint).range)
    joint_range_unchanged = np.array_equal(comparison_range, model.joint(joint).range)
    all_guard_pairs = [pair for pair in root.findall("contact/pair")
                       if "index_root_" in pair.get("name", "") and "guard" in pair.get("name", "")]
    added_guard_pair_count = sum(
        pair.get("name") not in baseline_pair_names for pair in all_guard_pairs
    )
    passed = (
        isolated
        and external_unchanged
        and joint_range_unchanged
        and added_guard_pair_count >= 2
        and len(root.findall("contact/pair")) == source_pairs + added_guard_pair_count
        and all(
            classification[grid]["false_positive_count"] == 0
            and classification[grid]["false_negative_count"] == 0
            for grid in ("sampled_grid", "midpoint_holdout_grid")
        )
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
            index_angle_range_rad=[float(angles.min()), float(angles.max())],
            native_interference_frames=int(np.count_nonzero(native_invalid)),
            guard_minimum_distance_m=guard_minimum,
            guard_penetrating_frames={label: int(np.count_nonzero(np.asarray(values) < 0.0))
                                      for label, values in distance_rows.items()},
            kinematic_model_feasible=retarget_report["kinematic_model_feasible"],
            solver_primal_tolerance=retarget_report["solver_primal_tolerance"],
            solver_dual_tolerance=retarget_report["solver_dual_tolerance"],
            planning_collision_buffer_m=retarget_report["effective_settings"]["planning_collision_buffer_m"],
            accepted_min_self_collision_distance_m=retarget_report["effective_settings"]["accepted_min_self_collision_distance_m"],
        )
        controlled_guard_run = controlled_guard_run or retarget_run
        old_report = json.loads((comparison_run / "retarget_report.json").read_text())
        controlled_report = json.loads((controlled_guard_run / "retarget_report.json").read_text())
        comparison_scene = resolve_artifact_path(
            {"path": old_report["scene"], "sha256": old_report["scene_sha256"]}
        )
        comparison_scene_equivalent = _same_scene_with_rebased_meshdir(
            baseline_scene, comparison_scene, scene.parent
        )
        numerical_keys = (
            "planning_collision_buffer_m",
            "accepted_min_self_collision_distance_m",
            "source_dt_s",
            "qp_integration_dt_s",
            "iterations_per_frame",
        )
        solver_numerics_equal = (
            old_report["solver_primal_tolerance"] == controlled_report["solver_primal_tolerance"]
            and old_report["solver_dual_tolerance"] == controlled_report["solver_dual_tolerance"]
            and all(old_report["effective_settings"][key]
                    == controlled_report["effective_settings"][key] for key in numerical_keys)
        )
        with np.load(comparison_run / "robot_reference.npz", allow_pickle=False) as archive:
            old_angles = np.asarray(archive["qpos"], dtype=float)[:, candidate_address]
        with np.load(controlled_guard_run / "robot_reference.npz", allow_pickle=False) as archive:
            controlled_angles = np.asarray(archive["qpos"], dtype=float)[:, candidate_address]
        metrics = ("fingertip_mean_error_m", "fingertip_max_error_m", "wrist_mean_error_rad")
        comparison = dict(
            interpretation="controlled scene constraint comparison; report feasibility, not a general tracking benefit",
            baseline_scene_only_rebases_compiler_meshdir=comparison_scene_equivalent,
            solver_numerics_equal=solver_numerics_equal,
            pre_guard_native_interference_frames=int(np.count_nonzero(
                (old_angles < negative_onset) | (old_angles > positive_onset))),
            post_guard_native_interference_frames=int(np.count_nonzero(
                (controlled_angles < negative_onset) | (controlled_angles > positive_onset))),
            baseline={key: old_report[key] for key in metrics},
            guarded={key: controlled_report[key] for key in metrics},
            guarded_minus_baseline={key: (np.asarray(controlled_report[key]) -
                                           np.asarray(old_report[key])).tolist()
                                    for key in metrics},
        )
        extra_sources = [artifact(retarget_run / name) for name in
                         ("robot_reference.npz", "retarget_report.json")]
        extra_sources += [artifact(comparison_run / name) for name in
                          ("robot_reference.npz", "retarget_report.json")]
        if controlled_guard_run != retarget_run:
            extra_sources += [artifact(controlled_guard_run / name) for name in
                              ("robot_reference.npz", "retarget_report.json")]
        passed = (passed and trajectory["kinematic_model_feasible"]
                  and trajectory["native_interference_frames"] == 0
                  and comparison_scene_equivalent
                  and solver_numerics_equal
                  and min(trajectory["guard_minimum_distance_m"].values()) >= 0.0)
    report = dict(
        status=((f"{side}_index_root_guard_formal_audit_passed" if formal_audit
                 else f"{side}_index_root_guard_candidate_passed") if passed else
                (f"{side}_index_root_guard_formal_audit_failed" if formal_audit
                 else f"{side}_index_root_guard_candidate_failed")),
        side=side,
        native_clear_interval_rad=[float(negative_onset), float(positive_onset)],
        native_endpoint_evidence=endpoint_evidence,
        guards=specs,
        guard_transition_rad=transitions,
        construction_onset_error_rad={
            spec["label"]: float(abs(transitions[spec["label"]] - spec["native_onset_rad"]))
            for spec in specs
        },
        classification_check=classification,
        sampled_angles=141,
        midpoint_holdout_angles=140,
        classification_method="direct_native_FCL_predicate_and_actual_MuJoCo_guard_distance",
        guards_isolated_to_two_explicit_pairs=isolated,
        source_contact_pair_count=source_pairs,
        candidate_contact_pair_count=len(root.findall("contact/pair")),
        joint_range_rad=model.joint(joint).range.tolist(),
        joint_range_unchanged=joint_range_unchanged,
        external_collision_geometries_unchanged=external_unchanged,
        active_retarget_trajectory=trajectory,
        retarget_comparison=comparison,
        audited_scene=artifact(audited_scene),
        sources=[artifact(scene), artifact(reference)] +
                ([artifact(baseline_scene)] if baseline_scene is not None else []) + extra_sources,
        generation_code=artifact(Path(__file__)),
        scene_modified_by_audit=False,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"{side} index-root guard candidate did not pass")
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
    parser.add_argument("--controlled-guard-run", type=Path)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.reference, args.output_scene, args.report, args.baseline_scene,
        args.retarget_run, args.comparison_run, args.controlled_guard_run, args.side)
