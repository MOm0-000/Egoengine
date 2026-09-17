"""Recompute historical Pour evidence after audit fixes; never optimize or step."""

import argparse
import json
from numbers import Real
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initial_contacts import inspect as inspect_native
from audit_taco_initial_controls import run as inspect_controls
from audit_taco_initialization import topology, visual_meshes, world_vertices
from audit_taco_thumb_assembly import native_intersection, run as inspect_thumb
from egoengine_repro.retarget.collision_audit import audit_trajectory, validate_qpos
from egoengine_repro.retarget.initial_hand import change_summary, next_reference_diagnostic, state_summary
from egoengine_repro.retarget.mink import _joint_velocity_limits
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from egoengine_repro.retarget.taco_bimanual import differentiate, pose7

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
BASELINE = ROOT / "runs/taco_pour_bimanual_gt_v1"
ASSEMBLY = ROOT / "runs/taco_pour_thumb_assembly_v1"
ATTEMPTS = [ROOT / f"runs/taco_pour_initial_hand_v{i}" for i in range(1, 5)]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metric_differences(expected, actual, path="", *, atol=1e-12, rtol=1e-9):
    """Compare recorded fields, allowing added metadata but never missing fields."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [dict(field=path, issue="type_changed")]
        return [item for key, value in expected.items() for item in
                (metric_differences(value, actual[key], f"{path}/{key}", atol=atol, rtol=rtol)
                 if key in actual else [dict(field=f"{path}/{key}", issue="missing_recomputed_field")])]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [dict(field=path, issue="shape_or_length_changed")]
        return [item for i, (a, b) in enumerate(zip(expected, actual))
                for item in metric_differences(a, b, f"{path}/{i}", atol=atol, rtol=rtol)]
    if isinstance(expected, bool) or isinstance(actual, bool):
        equal = type(expected) is type(actual) and expected == actual
    elif isinstance(expected, Real) and isinstance(actual, Real):
        equal = bool(np.isfinite(expected) and np.isfinite(actual) and
                     (expected == actual if isinstance(expected, int) else np.isclose(expected, actual, atol=atol, rtol=rtol)))
    else:
        equal = expected == actual
    return [] if equal else [dict(field=path, issue="value_changed", recorded=expected, recomputed=actual)]


def finite_archive(model, path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    qpos = validate_qpos(model, arrays["qpos"], trajectory=True)
    for key, width in (("qvel", model.nv), ("ctrl", model.nu)):
        if key in arrays and (arrays[key].shape != (len(qpos), width) or not np.isfinite(arrays[key]).all()):
            raise ValueError(f"invalid {key}: {path}")
    quaternion_error = 0.0
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            adr = int(model.jnt_qposadr[joint]) + (3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 0)
            quaternion_error = max(quaternion_error, float(np.abs(np.linalg.norm(qpos[:, adr:adr + 4], axis=1) - 1).max()))
    return arrays, dict(input=artifact(path), qpos_rows=len(qpos), finite=True,
                         quaternion_norm_max_error=quaternion_error, corrected_or_normalized=False)


def reference_metrics(model, robot, human, settings):
    qpos = robot["qpos"]
    if not np.array_equal(robot["frame_indices"], human["frame_indices"]):
        raise ValueError("human/robot source frames differ")
    times = human["timestamps_s"]
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("invalid source timestamps")
    dt = float(times[1] - times[0])
    if not np.allclose(np.diff(times), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("reference finite difference requires the original uniform timeline")
    velocity = _joint_velocity_limits(model, mujoco, settings)
    addresses = [int(model.joint(name).qposadr[0]) for name in velocity]
    ranges = np.array([model.joint(name).range for name in velocity])
    margins = np.minimum(qpos[:, addresses] - ranges[:, 0], ranges[:, 1] - qpos[:, addresses]).min(axis=1)
    ratios = (np.abs(np.diff(qpos[:, addresses], axis=0)) / (dt * np.array(list(velocity.values())))).max(axis=1)
    tips = np.empty((len(qpos), 2, 5))
    wrists = np.empty((len(qpos), 2))
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for h, side in enumerate(("right", "left")):
            ids = [model.site(f"{side}_{f}_tip").id for f in ("thumb", "index", "middle", "ring", "pinky")]
            tips[frame, h] = np.linalg.norm(data.site_xpos[ids] - human["T_sim_fingertip_target"][frame, h, :, :3, 3], axis=1)
            relative = data.xmat[model.body(f"{side}_hand_link").id].reshape(3, 3).T @ human["T_sim_wrist_target"][frame, h, :3, :3]
            wrists[frame, h] = Rotation.from_matrix(relative).magnitude()
    return dict(fingertip_position_error_m=tips.tolist(), wrist_orientation_error_rad=wrists.tolist(),
        joint_limit_min_margin=margins.tolist(), frame_velocity_max_ratio=ratios.tolist(),
        qvel=differentiate(model, qpos, dt).tolist(), ctrl=qpos[:, :36].tolist()), dict(
        fingertip_mean_error_m=tips.mean(axis=(0, 2)).tolist(), fingertip_max_error_m=tips.max(axis=(0, 2)).tolist(),
        wrist_mean_error_rad=wrists.mean(axis=0).tolist(), joint_limit_min_margin=float(margins.min()),
        joint_limit_violating_frames=int((margins < -1e-6).sum()), frame_velocity_max_ratio=float(ratios.max()),
        frame_velocity_violating_intervals=int((ratios > 1 + 1e-6).sum()))


def run(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    legacy_paths = sorted({p for root in [BASELINE, ASSEMBLY, *ATTEMPTS]
                           for p in root.iterdir() if p.suffix in (".json", ".npz", ".urdf")})
    preserved = [artifact(p) for p in [SCENE, *legacy_paths]]
    scene_meshes = scene_mesh_artifacts(SCENE)
    provenance = []
    recorded_meshes = {}
    for path in legacy_paths:
        if path.suffix != ".json":
            continue
        report = read(path)
        groups = ("preserved_artifacts", "source_assets", "source_meshes", "scene_meshes", "raw_inputs", "inputs")
        records = [dict(path=r["path"], sha256=r["sha256"]) for key in groups for r in report.get(key, [])]
        verify_artifacts(records)
        for record in records:
            recorded_meshes[record["path"]] = record["sha256"]
        provenance.append(dict(report=artifact(path), checked_input_records=len(records),
                               legacy_input_hashes_match=True, code_hashes_not_compared_because_code_was_fixed=True))
    for object_id in ("022", "135"):
        path = ROOT / f"models/taco_xhand/assets/objects/{object_id}/convex_m/provenance.json"
        report = read(path)
        records = [dict(path=r["path"], sha256=r["sha256"]) for r in report["artifacts"]]
        verify_artifacts(records)
        preserved.append(artifact(path))
        recorded_meshes.update({r["path"]: r["sha256"] for r in records})
    covered = [r for r in scene_meshes if recorded_meshes.get(r["path"]) == r["sha256"]]
    if len(covered) != len(scene_meshes):
        raise ValueError("some current scene meshes cannot be matched to historical recorded hashes")
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []

    def compare(label, old, fresh, fields=None):
        if fields is not None:
            old = {key: old[key] for key in fields}
            fresh = {key: fresh[key] for key in fields}
        differences = metric_differences(old, fresh)
        comparisons.append(dict(check=label, outcome="recomputed_equal" if not differences else "discrepancy",
                                compared_top_level_fields=list(old) if isinstance(old, dict) else None,
                                differences=differences))
        print(f"{label}: {'MATCH' if not differences else f'{len(differences)} DIFFERENCES'}", flush=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    meshes, _ = visual_meshes(SCENE, model)
    robot, numeric = finite_archive(model, BASELINE / "robot_reference.npz")
    numeric_checks = [numeric]
    with np.load(BASELINE / "human_reference.npz", allow_pickle=False) as source:
        human = dict(source)
    retarget = read(BASELINE / "retarget_report.json")
    settings = retarget["inherited_settings"]["velocity_limits"]
    dt = float(human["timestamps_s"][1] - human["timestamps_s"][0])
    metrics, summary = reference_metrics(model, robot, human, settings)
    compare("reference_all_198_fk_velocity_command_and_joint_metrics", {k: robot[k].tolist() for k in metrics}, metrics)
    compare("reference_retarget_report_metrics", retarget, summary, list(summary))
    write(output / "reference_metrics.json", summary)
    for h, side in enumerate(("right", "left")):
        adr = int(model.joint(f"{side}_object_joint").qposadr[0])
        expected = np.stack([pose7(t) for t in human["T_sim_object_reference"][:, h]])
        compare(f"reference_{side}_object_aligned_gt", expected.tolist(), robot["qpos"][:, adr:adr + 7].tolist())
    print("Recomputing all reference collision families over 198 rows", flush=True)
    collisions = audit_trajectory(model, robot["qpos"])
    compare("reference_all_198_collision_families", read(BASELINE / "collision_audit.json"), collisions, list(collisions))
    write(output / "reference_collision_audit.json", collisions)
    print("Recomputing omitted-pair/native topology over all 198 rows", flush=True)
    initial = read(BASELINE / "initialization_audit.json")
    full_topology = topology(model, robot["qpos"], meshes)
    compare("reference_all_198_omitted_pair_topology", initial["topology"]["omitted"], full_topology)
    write(output / "reference_omitted_pairs.json", full_topology)
    original_state = state_summary(model, robot["qpos"][0], tolerance=5e-5)
    for family, old in initial["initial_distances"].items():
        current = original_state["families"][family]
        compare(f"original_first_frame_{family}", old, dict(min_distance_m=current["min_distance_m"],
                geom1=current["worst_pair"][0], geom2=current["worst_pair"][1], pairs_below_50um=current["violations"]))
    native = inspect_native(SCENE, BASELINE / "robot_reference.npz")
    compare("original_first_frame_native_hand_object", read(BASELINE / "initial_native_contact_audit.json"), native,
            ["records", "tolerance_m", "max_samples_per_surface"])
    write(output / "original_native_contacts.json", native)
    candidate_results = []
    for index, root in enumerate(ATTEMPTS, 1):
        old = read(root / "report.json")
        candidate_path = Path(old["candidate"]["path"])
        arrays, numeric = finite_archive(model, candidate_path)
        numeric_checks.append(numeric)
        q = arrays["qpos"][0]
        if not bool(arrays["diagnostic_only"]) or bool(arrays["accepted_as_reset"]):
            raise ValueError("historical temporary candidate has invalid acceptance flags")
        compare(f"v{index}_object_coordinates", robot["qpos"][0, 36:].tolist(), q[36:].tolist())
        fresh = dict(after=state_summary(model, q), changes=change_summary(model, robot["qpos"][0], q, human),
                     next_reference=next_reference_diagnostic(model, q, robot["qpos"][1], settings, dt))
        compare(f"v{index}_state_posture_and_connection", old, fresh, list(fresh))
        candidate_results.append(dict(version=index, declared_state_feasible=fresh["after"]["declared_state_feasible"]))
        trace_path = root / "solver_iterations.npz"
        if trace_path.exists():
            trace, numeric = finite_archive(model, trace_path)
            numeric_checks.append(numeric)
            if not np.array_equal(trace["qpos"][:, 36:], np.tile(robot["qpos"][0, 36:], (len(trace["qpos"]), 1))):
                raise ValueError("historical numerical iterations changed objects")
        if index >= 3:
            native = inspect_native(SCENE, candidate_path, native_aabb=True)
            compare(f"v{index}_native_contacts_and_table", read(root / "native_geometry_audit.json"), native,
                    ["native_visual_support", "native_aabb_checks", "records"])
            fresh["native_geometry"] = native
            fresh["omitted_intrahand_coverage"] = topology(model, arrays["qpos"], meshes)
            compare(f"v{index}_omitted_pair_native_samples", read(root / "audit.json")["omitted_intrahand_coverage"], fresh["omitted_intrahand_coverage"])
        write(output / f"v{index}_recomputed.json", fresh)
    print("Recomputing all 35 saved thumb-sweep configurations", flush=True)
    thumb = inspect_thumb(SCENE, BASELINE / "robot_reference.npz", ATTEMPTS[2] / "initial_hand_candidate.npz", ASSEMBLY, output / "thumb")
    previous_thumb = read(ASSEMBLY / "report.json")
    compare("thumb_all_saved_states_and_35_sweep_points", previous_thumb, thumb, ["states", "left_joint_sweep"])
    for side in ("right", "left"):
        old_joints = previous_thumb["source_checks"][side]["joints"]
        compare(f"source_{side}_joint_numeric_values", old_joints, thumb["source_checks"][side]["joints"])
        if not thumb["source_checks"][side]["matches_source_within_tolerance"]:
            raise ValueError("actual source kinematics fail the corrected check")
    with np.load(ATTEMPTS[3] / "initial_hand_candidate.npz", allow_pickle=False) as source:
        candidate = source["qpos"][0]
    saved_v4 = read(ATTEMPTS[3] / "report.json")
    data = mujoco.MjData(model)
    data.qpos[:] = candidate
    mujoco.mj_forward(model, data)
    final_thumb = {side: native_intersection(model, data, meshes, side) for side in ("right", "left")}
    compare("v4_full_native_thumb_intersections", saved_v4["native_thumb_final"], final_thumb)
    old_controls = read(ATTEMPTS[3] / "initial_control_audit.json")
    controls = inspect_controls(SCENE, BASELINE / "robot_reference.npz", ATTEMPTS[3] / "initial_hand_candidate.npz", output / "initial_controls.json")
    compare("v4_all_four_instantaneous_control_cases", old_controls["cases"], controls["cases"])
    surface = output / "object_surfaces.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_taco_object_surfaces.py"),
        "--human", str(BASELINE / "human_reference.npz"), "--output", str(surface),
        "--tool-mesh", str(ROOT / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/022_cm.obj"),
        "--target-mesh", str(ROOT / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/135_cm.obj")], check=True)
    compare("object_all_198_table_clearances_and_selected_native_samples", read(BASELINE / "object_surface_audit.json"), read(surface),
            ["frames", "samples_per_object", "tolerance_m", "table_height_m"])
    brush_screen = []
    for path in sorted((ROOT / "runs").glob("taco_brush*/robot_reference.npz")):
        old = read(path.parent / "retarget_report.json")
        scene = Path(old["scene"])
        _, record = finite_archive(mujoco.MjModel.from_xml_path(str(scene)), path)
        brush_screen.append(record)
        preserved.extend([artifact(path), artifact(scene), artifact(path.parent / "retarget_report.json")])
    verify_artifacts(preserved)
    verify_artifacts(scene_meshes)
    differences = [c for c in comparisons if c["outcome"] != "recomputed_equal"]
    result = dict(status="recomputed_with_discrepancies" if differences else "legacy_pour_metrics_recomputed_equal_after_bug_fixes",
        comparisons=comparisons, comparison_count=len(comparisons), discrepancy_count=len(differences),
        comparison_tolerances=dict(absolute=1e-12, relative=1e-9, provenance="numerical comparison only, not task thresholds"),
        numeric_archives=numeric_checks, candidate_results=candidate_results,
        actual_pour_invalid_numeric_state_found=False, source_thumb_kinematics_match=True,
        hash_provenance_checks=provenance, scene_mesh_dependencies=scene_meshes,
        scene_meshes_matched_to_historical_hashes=len(covered), actual_recorded_asset_hash_mismatch_found=False,
        historical_brush_numeric_screen_only=brush_screen,
        original_reference_collision_rows_recomputed=len(robot["qpos"]),
        bug_reproductions_were_injected_faults_not_observed_bad_dataset_rows=True,
        limits=["legacy tests used synthetic corruptions; no claim of actual historical corruption follows from those tests",
                "hash equality compares recorded endpoints, not an unobserved continuous history",
                "numerical legacy agreement is not a complete collision certificate or physical success",
                "brush history received numeric screening only, not full collision revalidation"],
        model_changed=False, reference_changed=False, candidates_changed=False, legacy_reports_overwritten=False,
        simulation_steps_executed=0, rl_or_physical_rollout_run=False, formal_initialization_selected=False,
        collision_coverage_plan_progressed=False, training_ready=False, preserved_artifacts=preserved,
        revalidation_code=artifact(Path(__file__)))
    write(output / "report.json", result)
    print(json.dumps(dict(status=result["status"], comparisons=len(comparisons), discrepancy_count=len(differences))), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
