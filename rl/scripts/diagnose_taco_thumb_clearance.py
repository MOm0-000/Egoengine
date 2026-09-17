"""Separate first-frame, single-joint diagnostic; never install a physical reset."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_thumb_assembly import native_intersection
from egoengine_repro.retarget.initial_hand import change_summary, next_reference_diagnostic, state_summary
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def bisect_clear_endpoint(intersects, clear, blocked, tolerance=1e-6):
    """Retain a tested clear endpoint, not a global optimum or clearance margin."""
    if not np.isfinite([clear, blocked, tolerance]).all() or not 0 < tolerance < blocked - clear:
        raise ValueError("finite ordered bracket and positive smaller tolerance required")
    if intersects(clear) or not intersects(blocked):
        raise ValueError("bracket must have a clear lower and intersecting upper endpoint")
    while blocked - clear > tolerance:
        middle = (clear + blocked) / 2
        if intersects(middle):
            blocked = middle
        else:
            clear = middle
    return clear, blocked


def run(scene, baseline, previous, sweep_path, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [scene, baseline / "robot_reference.npz", baseline / "human_reference.npz",
              baseline / "retarget_report.json", previous, sweep_path]
    preserved = [artifact(p) for p in inputs]
    sweep = json.loads(sweep_path.read_text())
    if not all(artifact(Path(p["path"])) == p for p in sweep["preserved_artifacts"]):
        raise ValueError("thumb sweep inputs changed")
    verify_artifacts(sweep["source_meshes"])
    verify_artifacts(sweep.get("scene_meshes", []))
    if [artifact(scene), artifact(inputs[1]), artifact(previous)] != sweep["preserved_artifacts"][:3]:
        raise ValueError("thumb sweep belongs to another scene or candidate")
    settings = json.loads(inputs[3].read_text())["inherited_settings"]["velocity_limits"]
    with np.load(inputs[1], allow_pickle=False) as source:
        robot = dict(source)
    with np.load(inputs[2], allow_pickle=False) as source:
        human = dict(source)
    with np.load(previous, allow_pickle=False) as source:
        if (source["qpos"].shape != (1, 50) or not bool(source["diagnostic_only"])
                or bool(source["accepted_as_reset"]) or "ctrl" in source or "qvel" in source):
            raise ValueError("expected one unaccepted diagnostic posture")
        initial = source["qpos"][0]
    mesh_snapshot = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    meshes, mesh_paths = visual_meshes(scene, model)
    mesh_hashes = [artifact(p) for p in mesh_paths]
    joint_name = "left_hand_thumb_rota_joint1"
    address = int(model.joint(joint_name).qposadr[0])
    bend_address = int(model.joint("left_hand_thumb_bend_joint").qposadr[0])
    clear_angles = [s["rotate_rad"] for s in sweep["left_joint_sweep"]
                    if np.isclose(s["bend_rad"], initial[bend_address], rtol=0, atol=1e-12)
                    and not s["intersection_nonempty"] and s["rotate_rad"] < initial[address]]
    if not clear_angles:
        raise ValueError("no previously audited lower clear angle at this bend value")
    history = []

    def intersects(angle):
        if not model.joint(joint_name).range[0] <= angle <= model.joint(joint_name).range[1]:
            raise ValueError("angle outside source limits")
        data.qpos[:] = initial
        data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        result = native_intersection(model, data, meshes, "left")
        history.append(dict(angle_rad=float(angle), **result))
        print(f"thumb angle={angle:.9f}: {result['intersection_volume_m3'] * 1e9:.6f} mm^3", flush=True)
        return result["intersection_nonempty"]

    clear, blocked = bisect_clear_endpoint(intersects, max(clear_angles), float(initial[address]))
    candidate = initial.copy()
    candidate[address] = clear
    frozen = [i for i in range(model.nq) if i != address]
    if not np.array_equal(candidate[frozen], initial[frozen]):
        raise RuntimeError("single-joint diagnostic changed other coordinates")
    if not np.array_equal(candidate[36:], robot["qpos"][0, 36:]):
        raise ValueError("object poses differ from baseline")
    data.qpos[:] = candidate
    mujoco.mj_forward(model, data)
    final_native = {s: native_intersection(model, data, meshes, s) for s in ("right", "left")}
    after = state_summary(model, candidate)
    usable = after["declared_state_feasible"] and not any(r["intersection_nonempty"] for r in final_native.values())
    dt = float(human["timestamps_s"][1] - human["timestamps_s"][0])
    report = dict(status="single_thumb_first_frame_diagnostic_not_accepted_reset",
        authorization="continuation_of_authorized_separate_first_frame_hand_only_diagnostic",
        source_row_zero_based=0, baseline_frames=len(robot["qpos"]),
        candidate_declared_state_feasible=after["declared_state_feasible"],
        candidate_declared_and_audited_thumb_feasible=usable,
        before=state_summary(model, initial), after=after,
        solver=dict(method="single_joint_native_intersection_bracket_bisection",
            provenance="local diagnostic, not a published EgoEngine reset or retargeting objective",
            joint=joint_name, previous_angle_rad=float(initial[address]), candidate_angle_rad=clear,
            angle_delta_rad=float(clear - initial[address]), clear_endpoint_rad=clear,
            blocked_endpoint_rad=blocked, angular_bracket_tolerance_rad=1e-6,
            positive_geometric_clearance_certified=False, monotonicity_proven=False,
            globally_minimum_correction_proven=False, frozen_qpos_addresses=frozen,
            search_evaluations=history),
        native_thumb_final=final_native,
        changes=change_summary(model, robot["qpos"][0], candidate, human),
        next_reference=next_reference_diagnostic(model, candidate, robot["qpos"][1], settings, dt),
        complete_intrahand_geometry_coverage=False, native_geometry_audit_pending=True,
        accepted_as_reset=False, physics_validated=False, rl_validation_completed=False,
        strict_gate_passed=False, simulation_steps_executed=0, source_reference_modified=False,
        qvel_selected=False, geometry_or_pairs_modified=False,
        preserved_artifacts=preserved, source_assets=mesh_hashes, scene_meshes=mesh_snapshot,
        audit_code=[artifact(Path(__file__)), artifact(ROOT / "scripts/audit_taco_thumb_assembly.py")])
    if preserved != [artifact(p) for p in inputs] or mesh_hashes != [artifact(p) for p in mesh_paths]:
        raise RuntimeError("source inputs changed during diagnostic")
    verify_artifacts(mesh_snapshot)
    output.mkdir(parents=True, exist_ok=False)
    name = "initial_hand_candidate.npz" if usable else "failed_initial_hand_candidate.npz"
    np.savez_compressed(output / name, qpos=candidate[None], source_frame_index=np.asarray(0),
                        diagnostic_only=np.asarray(True), accepted_as_reset=np.asarray(False))
    report["candidate"] = artifact(output / name)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(candidate=report["candidate"], feasible=usable,
                         angle_delta_rad=report["solver"]["angle_delta_rad"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene, args.baseline, args.previous, args.sweep, args.output)


if __name__ == "__main__":
    main()
