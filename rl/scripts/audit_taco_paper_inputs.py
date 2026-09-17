"""Audit exactly the four approved TACO inputs without altering data or scenes."""

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import (
    alignment_invariants, artifact, input_status, project_world, support_clearance,
    transform_report, video_info,
)

EPISODES = (
    ("brush", "taco_brush_brush_bowl_20230927_027", "(brush, brush, bowl)/20230927_027"),
    ("cut", "taco_cut_spatula_plate_20230917_020", "(cut, spatula, plate)/20230917_020"),
    ("skim", "taco_skim_spatula_plate_20230926_004", "(skim off, spatula, plate)/20230926_004"),
    ("smear", "taco_smear_eraser_box_20231103_071", "(smear, eraser, box)/20231103_071"),
)


def audit_episode(dev4, task, episode, sequence, row, *, video_kinds=None):
    inputs = []

    def load(path):
        inputs.append(path)
        return np.load(path, allow_pickle=False).astype(float)

    hand_dir = dev4 / "hand_poses/Hand_Poses" / sequence
    joints = load(hand_dir / "hand_joints.npy")
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"invalid hand joint shape: {joints.shape}")
    hand_audit = {}
    for h, side in enumerate(("left", "right")):
        path = hand_dir / f"{side}_hand.pkl"
        inputs.append(path)
        with path.open("rb") as stream:
            source = pickle.load(stream)
        keys = sorted(source, key=int)
        translations = np.stack([np.asarray(source[k]["hand_trans"]) for k in keys])
        aligned = [int(k) for k in keys] == list(range(1, len(joints) + 1))
        hand_audit[side] = dict(frames=len(keys), source_ids_contiguous_one_based=aligned,
            wrist_translation_error_m=(float(np.linalg.norm(translations - joints[:, h, 0], axis=-1).max())
                                       if aligned else None))

    poses, meshes, object_reports = [], [], {}
    for role in ("tool", "target"):
        paths = list((dev4 / "object_poses/Object_Poses" / sequence).glob(f"{role}_*.npy"))
        if len(paths) != 1:
            raise ValueError(f"expected one {role} file for {sequence}")
        pose = load(paths[0])
        object_id = paths[0].stem.split("_")[1]
        mesh_path = dev4 / "object_models/object_models_released" / f"{object_id}_cm.obj"
        inputs.append(mesh_path)
        mesh = trimesh.load_mesh(mesh_path, process=True)
        mesh.apply_scale(0.01)
        poses.append(pose)
        meshes.append(mesh)
        object_reports[role] = dict(object_id=object_id, frames=len(pose),
            pose=transform_report(pose), metric_extents_m=mesh.extents.tolist(),
            scale_applied=0.01, watertight=bool(mesh.is_watertight),
            volume_m3=float(mesh.volume), vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces), source_pose_translation_span_m=np.ptp(pose[:, :3, 3], axis=0).tolist())

    camera_dir = dev4 / "camera/Egocentric_Camera_Parameters" / sequence
    camera = load(camera_dir / "egocentric_frame_extrinsic.npy")
    intrinsic_path = camera_dir / "egocentric_intrinsic.txt"
    inputs.append(intrinsic_path)
    intrinsic = np.loadtxt(intrinsic_path)
    videos = {}
    if video_kinds is None:
        video_kinds = (("rgb", ".mp4"), ("depth_original", ".avi"), ("depth_resized", ".avi"))
    for kind, suffix in video_kinds:
        path = dev4 / kind / f"{episode}{suffix}"
        inputs.append(path)
        videos[kind] = video_info(path)
    counts = dict(hands=len(joints), tool=len(poses[0]), target=len(poses[1]), camera=len(camera),
                  **{k: v["decoded_frames"] for k, v in videos.items()})
    if len(set(counts[k] for k in ("hands", "tool", "target", "camera"))) != 1:
        raise ValueError(f"GT arrays require explicit alignment: {counts}")
    objects = np.stack(poses, axis=1)
    image_size = [videos["rgb"]["width"], videos["rgb"]["height"]]
    projections = dict(
        released_as_T_camera_world=project_world(joints, camera, intrinsic, image_size),
        inverse_diagnostic_only=project_world(joints, np.linalg.inv(camera), intrinsic, image_size))
    camera_report = dict(transforms=transform_report(camera), hand_projection=projections,
                         camera_modified=False, inverse_selected=False)

    # Reproduce the EXISTING convention for audit, not a new calibration/reset.
    target_bottom = support_clearance(meshes[1].vertices, poses[1][:1], 0.0)[0]
    transform = np.eye(4)
    transform[:2, 3] = np.array([0.6, 0]) - objects[0, :, :2, 3].mean(axis=0)
    transform[2, 3] = 0.72 - target_bottom
    aligned = transform @ objects
    support = {}
    for i, role in enumerate(("tool", "target")):
        clearance = support_clearance(meshes[i].vertices, aligned[:, i], 0.72)
        support[role] = dict(initial_clearance_m=float(clearance[0]),
            minimum_clearance_m=float(clearance.min()), minimum_source_row_zero_based=int(clearance.argmin()),
            rows_below_plane_50um=int((clearance < -5e-5).sum()))
    alignment = dict(T_sim_world=transform.tolist(), table_height_m=0.72,
        initial_pair_center_xy_m=aligned[0, :, :2, 3].mean(axis=0).tolist(),
        vertical_anchor="first_frame_target_bottom_local_convention_not_published_recipe",
        invariants=alignment_invariants(joints, objects, camera, transform), object_support=support)

    rigid = camera_report["transforms"]["rigid_within_float32_tolerance"] and all(
        r["pose"]["rigid_within_float32_tolerance"] for r in object_reports.values())
    hand_valid = bool(np.isfinite(joints).all()) and all(
        r["source_ids_contiguous_one_based"] and r["wrist_translation_error_m"] < 1e-5
        for r in hand_audit.values())
    status = input_status(counts, rigid, hand_valid,
                          projections["released_as_T_camera_world"]["positive_depth_fraction"] == 1)
    return dict(task=task, episode=episode, sequence=sequence, counts=counts,
                metadata={k: row[k] for k in ("action", "tool", "object", "n_frames", "fps", "calib_status")},
                hands=hand_audit, objects=object_reports, videos=videos,
                audited_video_modalities=list(videos), camera=camera_report,
                alignment=alignment, status=status, inputs=[artifact(p) for p in inputs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dev4 = ROOT / "data/taco_v1/dev4"
    with (dev4 / "taco_info.csv").open() as stream:
        rows = {r["sequence_id"]: r for r in csv.DictReader(stream)}
    reports = []
    for task, episode, sequence in EPISODES:
        reports.append(audit_episode(dev4, task, episode, sequence, rows[sequence]))
        print(json.dumps(dict(task=task, counts=reports[-1]["counts"], status=reports[-1]["status"])), flush=True)
    report = dict(status="paper_first_input_audit_not_physics_validation",
        paper_basis=["4.1 compatibility premise", "A.1 fixed offset/table", "A.3 approximate alignment"],
        checks_provenance="local_diagnostics_not_unpublished_author_selection_rules",
        artifacts_modified=False, frames_trimmed=False, episodes_added=False,
        metadata=artifact(dev4 / "taco_info.csv"), audit_code=[artifact(Path(__file__)),
            artifact(ROOT / "src/egoengine_repro/retarget/paper_audit.py")], episodes=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
