"""Produce a separate initial-hand candidate; never install it as an RL reset."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from egoengine_repro.retarget.initial_hand import (
    change_summary, next_reference_diagnostic, solve_initial_hands, state_summary,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def run(scene: Path, baseline: Path, output: Path, *, max_iterations=128, backtrack_separation=False,
        seed_mode="original"):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = [scene, baseline / "human_reference.npz", baseline / "robot_reference.npz",
              baseline / "retarget_report.json"]
    preserved = [artifact(p) for p in inputs]
    mesh_snapshot = scene_mesh_artifacts(scene)
    original_report = json.loads(inputs[-1].read_text())
    if original_report["scene_sha256"] != preserved[0]["sha256"]:
        raise ValueError("reference scene hash differs from requested scene")
    if original_report["human_reference_sha256"] != preserved[1]["sha256"]:
        raise ValueError("human reference no longer matches the MINK run")
    with np.load(inputs[1], allow_pickle=False) as data:
        human = dict(data)
    with np.load(inputs[2], allow_pickle=False) as data:
        robot = dict(data)
    if len(robot["qpos"]) < 2 or not np.array_equal(robot["frame_indices"], human["frame_indices"]):
        raise ValueError("full matching human/robot reference required")
    source_dt = float(np.diff(human["timestamps_s"])[0])
    settings = original_report["inherited_settings"]["velocity_limits"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    before = state_summary(model, robot["qpos"][0])
    candidate, solver, trace = solve_initial_hands(model, robot["qpos"][0], settings,
        source_dt=source_dt, max_iterations=max_iterations, backtrack_separation=backtrack_separation,
        seed_mode=seed_mode)
    # Audits reload the unmodified runtime model, not MINK's private masks.
    runtime = mujoco.MjModel.from_xml_path(str(scene))
    after = state_summary(runtime, candidate)
    report = dict(status="first_frame_hand_only_diagnostic_not_accepted_reset",
        authorization="user_allowed_separate_first_frame_hand_pose_candidate",
        source_row_zero_based=0, baseline_frames=len(robot["qpos"]),
        candidate_declared_state_feasible=after["declared_state_feasible"], before=before, after=after,
        solver=solver, changes=change_summary(runtime, robot["qpos"][0], candidate, human),
        next_reference=next_reference_diagnostic(runtime, candidate, robot["qpos"][1], settings, source_dt),
        complete_intrahand_geometry_coverage=False, native_geometry_audit_pending=True,
        accepted_as_reset=False, physics_validated=False, rl_validation_completed=False,
        strict_gate_passed=False, simulation_steps_executed=0, source_reference_modified=False,
        qvel_selected=False, qvel_provenance="no physical velocity chosen; baseline derivative remains untouched",
        preserved_artifacts=preserved, scene_meshes=mesh_snapshot, audit_code=[artifact(Path(__file__))])
    if preserved != [artifact(p) for p in inputs]:
        raise RuntimeError("preserved baseline changed during diagnostic")
    verify_artifacts(mesh_snapshot)
    output.mkdir(parents=True, exist_ok=False)
    filename = "initial_hand_candidate.npz" if after["declared_state_feasible"] else "failed_initial_hand_candidate.npz"
    np.savez_compressed(output / filename, qpos=candidate[None], source_frame_index=np.asarray(0),
                        diagnostic_only=np.asarray(True), accepted_as_reset=np.asarray(False))
    np.savez_compressed(output / "solver_iterations.npz", qpos=trace,
                        numerical_iterations_not_physical_time=np.asarray(True))
    report["candidate"] = artifact(output / filename)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "candidate_declared_state_feasible", "changes", "candidate")}, indent=2), flush=True)
    print(f"Termination: {solver['termination_reason']}; iterations: {solver['iterations']}", flush=True)
    return report


def audit_saved_candidate(scene: Path, baseline: Path, output: Path, *, frozen_reference: Path | None = None):
    from audit_taco_initialization import topology, visual_meshes

    destination = output / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    candidate_path = Path(report["candidate"]["path"])
    if artifact(candidate_path) != report["candidate"]:
        raise ValueError("candidate hash differs from the solver result")
    native_path = output / "native_geometry_audit.json"
    native = json.loads(native_path.read_text())
    for payload in (report, native):
        verify_artifacts(payload["preserved_artifacts"])
        verify_artifacts(payload.get("source_assets", []))
        verify_artifacts(payload.get("scene_meshes", []))
    if not native.get("source_assets"):
        raise ValueError("native audit has no source mesh provenance")
    if native["preserved_artifacts"] != [artifact(scene), artifact(candidate_path)]:
        raise ValueError("native audit belongs to a different scene or candidate")
    inputs = [scene, report_path, candidate_path, native_path, baseline / "robot_reference.npz"]
    if frozen_reference is not None:
        if artifact(frozen_reference) not in report["preserved_artifacts"]:
            raise ValueError("frozen reference is not a preserved input of this diagnostic")
        inputs.append(frozen_reference)
    preserved = [artifact(p) for p in inputs]
    with np.load(candidate_path, allow_pickle=False) as data:
        qpos = data["qpos"]
        if (qpos.shape != (1, 50) or not bool(data["diagnostic_only"])
                or bool(data["accepted_as_reset"]) or "qvel" in data or "ctrl" in data):
            raise ValueError("expected a single diagnostic posture, not a reset/trajectory")
    with np.load(baseline / "robot_reference.npz", allow_pickle=False) as data:
        original = data["qpos"][0]
    frozen = report["solver"]["frozen_qpos_addresses"]
    lock_origin = original
    if frozen_reference is not None:
        with np.load(frozen_reference, allow_pickle=False) as data:
            lock_origin = data["qpos"][0]
    if not np.array_equal(lock_origin[frozen], qpos[0, frozen]):
        raise ValueError("a coordinate frozen by this diagnostic changed")
    if not np.array_equal(original[36:], qpos[0, 36:]):
        raise ValueError("fixed object pose changed")
    mesh_snapshot = scene_mesh_artifacts(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, mesh_paths = visual_meshes(scene, model)
    if sorted(native["source_assets"], key=lambda r: r["path"]) != sorted([artifact(p) for p in mesh_paths], key=lambda r: r["path"]):
        raise ValueError("native mesh manifest does not cover the current scene's visual inputs")
    print("Independent candidate audit: omitted shell pairs and native surfaces", flush=True)
    coverage = topology(model, qpos, meshes)
    supports = [r for r in native["native_visual_support"] if "object_visual" not in r["geom"]]
    sampled = [d for pair in native["records"] for d in pair["directions"] if "inside_samples_50um" in d]
    payload = dict(status="audited_initial_hand_candidate_not_accepted_reset",
        independent_state_check=state_summary(model, qpos[0]),
        native_hand_table_min_clearance_m=min(r["table_clearance_m"] for r in supports),
        native_hand_object_sampled_inside_points=sum(r["inside_samples_50um"] for r in sampled),
        native_hand_object_aabb_pairs_checked=len(native["native_aabb_checks"]),
        native_surface_sampling_is_not_complete_intersection_test=True,
        omitted_intrahand_coverage=coverage, complete_intrahand_geometry_coverage=False,
        object_poses_byte_identical=True, source_reference_modified=False,
        frozen_coordinate_reference=artifact(frozen_reference) if frozen_reference is not None else artifact(baseline / "robot_reference.npz"),
        accepted_as_reset=False, physical_initial_velocity_selected=False, simulation_steps_executed=0,
        training_ready=False, strict_gate_passed=False, native_geometry_audit_pending=False,
        next_reference_replay_claim="none; interpolation is diagnostic and the original reference stays unchanged",
        preserved_artifacts=preserved, source_assets=[artifact(p) for p in mesh_paths],
        scene_meshes=mesh_snapshot,
        inherited_full_mesh_snapshot_available=bool(report.get("scene_meshes") and native.get("scene_meshes")))
    if preserved != [artifact(p) for p in inputs]:
        raise RuntimeError("sources changed during independent candidate audit")
    verify_artifacts(mesh_snapshot)
    with destination.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in payload.items() if k in (
        "status", "native_hand_table_min_clearance_m", "native_hand_object_sampled_inside_points",
        "native_hand_object_aabb_pairs_checked", "accepted_as_reset")}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=128)
    parser.add_argument("--backtrack-separation", action="store_true")
    parser.add_argument("--seed-mode", choices=["original", "above_fixed_geometry"], default="original")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--frozen-reference", type=Path,
                        help="audit only: preserved parent posture for a chained diagnostic's coordinate locks")
    args = parser.parse_args()
    if args.audit_only:
        audit_saved_candidate(args.scene, args.baseline, args.output, frozen_reference=args.frozen_reference)
        return
    if args.frozen_reference is not None:
        parser.error("--frozen-reference requires --audit-only")
    run(args.scene, args.baseline, args.output, max_iterations=args.max_iterations,
        backtrack_separation=args.backtrack_separation, seed_mode=args.seed_mode)


if __name__ == "__main__":
    main()
