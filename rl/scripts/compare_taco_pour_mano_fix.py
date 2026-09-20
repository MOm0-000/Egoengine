"""Compare preserved and FK-corrected references on the same scene and metrics."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from audit_taco_pour_input_logic import rotation_steps_degrees
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
OLD = ROOT / "runs/taco_pour_bimanual_gt_v1"
NEW = ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1"


def inspect():
    prior_audit = ROOT / "runs/taco_pour_input_logic_v1/final_report.json"
    prior_inputs = json.loads(prior_audit.read_text())["inputs"]
    verify_artifacts(prior_inputs)
    inputs = [artifact(SCENE)] + scene_mesh_artifacts(SCENE)
    inputs.append(artifact(prior_audit))
    human, robot, reports = [], [], []
    for directory in (OLD, NEW):
        for filename in ("human_reference.npz", "robot_reference.npz", "retarget_report.json",
                         "collision_audit.json", "initial_native_contact_audit.json"):
            inputs.append(artifact(directory / filename))
        with np.load(directory / "human_reference.npz", allow_pickle=False) as source:
            human.append(dict(source))
        with np.load(directory / "robot_reference.npz", allow_pickle=False) as source:
            robot.append(dict(source))
        reports.append(json.loads((directory / "retarget_report.json").read_text()))
    for key in human[0]:
        if key != "T_sim_fingertip_target":
            np.testing.assert_array_equal(human[0][key], human[1][key], err_msg=key)
    for index in ((Ellipsis, slice(None, 3), 3), (Ellipsis, 3, slice(None))):
        np.testing.assert_array_equal(human[0]["T_sim_fingertip_target"][index], human[1]["T_sim_fingertip_target"][index])
    for key in ("inherited_settings", "effective_settings", "scene_sha256", "wrist_position_cost"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"comparison changes more than fingertip orientations: {key}")
    historical_scene = dict(path=reports[0]["scene"], sha256=reports[0]["scene_sha256"])
    verify_artifacts([historical_scene])
    guard_audit = json.loads((ROOT / "runs/taco_pour_bilateral_index_guard_v1/right_index_guard_audit.json").read_text())
    if (not guard_audit["external_collision_geometries_unchanged"]
            or not guard_audit["joint_range_unchanged"]):
        raise ValueError("active scene cannot be used for historical kinematic measurements")
    inputs += [historical_scene, artifact(ROOT / "runs/taco_pour_bilateral_index_guard_v1/right_index_guard_audit.json")]
    np.testing.assert_array_equal(robot[0]["qpos"][:, 36:], robot[1]["qpos"][:, 36:])
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    meshes, _ = visual_meshes(SCENE, model)
    data = mujoco.MjData(model)
    result = {}
    for ri, (name, directory) in enumerate((("old", OLD), ("corrected", NEW))):
        qpos = validate_qpos(model, robot[ri]["qpos"], trajectory=True)
        gids = list(meshes)
        heights = np.empty((len(qpos), len(gids)))
        for row, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            for index, gid in enumerate(gids):
                body = model.geom_bodyid[gid]
                heights[row, index] = (meshes[gid].vertices @ data.xmat[body].reshape(3, 3)[2]).min() + data.xpos[body, 2] - .72
        native = {}
        for side in ("left", "right"):
            indices = [i for i, gid in enumerate(gids) if model.geom(gid).name.startswith(side + "_")
                       and model.geom(gid).name != side + "_object_visual"]
            palm = gids.index(model.geom(f"{side}_hand_link_visual").id)
            values = heights[:, indices]
            first = int(values[0].argmin())
            worst_frame, worst_geom = np.unravel_index(values.argmin(), values.shape)
            native[side] = dict(initial_palm_clearance_m=float(heights[0, palm]),
                initial_whole_hand_min_clearance_m=float(values[0, first]),
                initial_worst_visual=model.geom(gids[indices[first]]).name,
                whole_trajectory_min_clearance_m=float(values.min()),
                worst_source_row_zero_based=int(worst_frame), worst_visual=model.geom(gids[indices[worst_geom]]).name,
                native_hand_below_table_rows_50um=int((values < -5e-5).any(axis=1).sum()),
                native_palm_below_table_rows_50um=int((heights[:, palm] < -5e-5).sum()))
        contacts = json.loads((directory / "initial_native_contact_audit.json").read_text())
        samples = [direction for record in contacts["records"] for direction in record["directions"]
                   if direction.get("target_visual") == "left_object_visual"
                   and direction.get("source_visual", "").startswith("left_")
                   and "sampled_penetration_max_m" in direction]
        collision = json.loads((directory / "collision_audit.json").read_text())
        steps = rotation_steps_degrees(human[ri]["T_sim_fingertip_target"][..., :3, :3])
        result[name] = dict(native_table=native,
            fingertip_mean_error_m=reports[ri]["fingertip_mean_error_m"],
            fingertip_max_error_m=reports[ri]["fingertip_max_error_m"],
            wrist_mean_orientation_error_rad=reports[ri]["wrist_mean_error_rad"],
            fingertip_orientation_steps_over_90deg=int((steps > 90).sum()),
            maximum_fingertip_orientation_step_deg=float(steps.max()),
            initial_left_hand_plate_max_sampled_penetration_m=max(s["sampled_penetration_max_m"] for s in samples),
            initial_native_sample_selection=contacts["selection"],
            initial_left_hand_plate_samples=samples,
            collision={key: {k: value[k] for k in ("pair_count", "min_distance_m", "penetrating_frames")}
                       for key, value in collision.items() if isinstance(value, dict) and "pair_count" in value})
    verify_artifacts(inputs)
    verify_artifacts(prior_inputs)
    return dict(status="FK_target_fix_compared_not_physics_validated", frames=len(robot[0]["qpos"]),
        only_human_fingertip_rotations_changed=True, settings_identical=True, object_qpos_identical=True,
        historical_input_hashes_unchanged=len(prior_inputs),
        original_inputs=inputs, results=result,
        limitations=["No physical time integration or reset has been performed.",
                     "The archived scene hash is verified through TRASH relocation; active-scene FK is used only after the guard audit proves all non-guard geometry and joint ranges unchanged.",
                     "Native signed contact checks are samples, not full-surface penetration bounds.",
                     "The table is the same independently uncalibrated horizontal plane in both branches.",
                     "Whole-trajectory native checks here measure table penetration, not all native pair intersections."],
        code=artifact(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
