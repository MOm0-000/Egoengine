"""Copy one preregistered Pour/Bowl/Plate episode from local release archives.

No source project or existing output is modified. No robot rendering, scene
generation, GT editing, physics initialization or training occurs here.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import stat
import sys
from zipfile import ZipFile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import artifact
from audit_taco_paper_inputs import audit_episode

SOURCE = Path("/data_all/zzx/egoengine_new/datasets/taco_v1")
HANDS = ROOT / "data/taco_v1/hand_poses_v1/Hand_Poses.zip"
DEPTH = ROOT / "data/taco_v1/depth_resized/Egocentric_Depth_Videos.zip"


def select_candidate(rows):
    matching = sorted((r for r in rows if "pour" in r["action"].lower()
                       and r["tool"] == "bowl" and r["object"] == "plate"),
                      key=lambda r: r["sequence_id"])
    eligible = [r for r in matching if r["all_modalities_complete"] == "True"
                and r["calib_status"] == "good"]
    if not eligible:
        raise ValueError("no complete, metadata-good Pour/Bowl/Plate candidate")
    return eligible[0], matching


def copy_member(archive_path, member, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with ZipFile(archive_path) as archive:
        entries = [e for e in archive.infolist() if e.filename == member]
        if len(entries) != 1:
            raise ValueError(f"expected one archive member: {archive_path}: {member}")
        entry = entries[0]
        if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError(f"expected a regular file: {member}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry) as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if output.stat().st_size != entry.file_size:
            raise ValueError(f"member length mismatch: {output}")
    info = archive_path.stat()
    return dict(source_archive=str(archive_path.resolve()), source_archive_bytes=info.st_size,
                source_archive_mtime_ns=info.st_mtime_ns, member=member,
                member_crc32=f"{entry.CRC:08x}", crc_verified_by_zip_reader=True,
                output=artifact(output))


def object_members(archive_path, sequence):
    prefix = f"Object_Poses/{sequence}/"
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
    result = {}
    for role in ("tool", "target"):
        matches = [n for n in names if n.startswith(prefix + role + "_") and n.endswith(".npy")]
        if len(matches) != 1:
            raise ValueError(f"expected one {role} for {sequence}")
        object_id = Path(matches[0]).stem.split("_", 1)[1]
        if len(object_id) != 3 or not object_id.isdigit():
            raise ValueError(f"unexpected object id: {object_id}")
        result[role] = (matches[0], object_id)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/taco_v1/pour_bowl_plate")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    metadata = SOURCE / "taco_info.csv"
    with metadata.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        row, candidates = select_candidate(list(reader))
    sequence = row["sequence_id"]
    source_id = sequence.rsplit("/", 1)[1]
    episode = "taco_pour_bowl_plate_" + source_id
    roles = object_members(SOURCE / "Object_Poses.zip", sequence)
    output.mkdir(parents=True)
    selection = dict(task="Pour, Bowl, Plate", dataset_task=row["triplet"],
        sequence=sequence, episode=episode, exact_author_episode=False,
        selection_rule="lexicographically_first_metadata_complete_good_calibration_candidate_before_rollout",
        candidate_count=len(candidates),
        candidates=[{k: r[k] for k in ("sequence_id", "n_frames", "all_modalities_complete", "calib_status")}
                    for r in candidates], metadata=artifact(metadata),
        authority="user_requested_Pour_Bowl_Plate_and_authorized_sample_acquisition_2026_09_06")
    with (output / "selection.json").open("x") as stream:
        json.dump(selection, stream, indent=2)
        stream.write("\n")
    records = []

    def extract(archive, member, destination):
        records.append(copy_member(archive, member, output / destination))

    extract(SOURCE / "Egocentric_RGB_Videos.zip", row["egocentric_rgb_path"], f"rgb/{episode}.mp4")
    extract(DEPTH, f"Egocentric_Depth_Videos/{sequence}/egocentric_depth.avi", f"depth_resized/{episode}.avi")
    for filename in ("left_hand.pkl", "right_hand.pkl", "left_hand_shape.pkl", "right_hand_shape.pkl", "hand_joints.npy"):
        member = f"Hand_Poses/{sequence}/{filename}"
        extract(HANDS, member, "hand_poses/" + member)
    # Cross-check the independently packaged joint-only release without making
    # a second alias or overwriting either source array.
    with ZipFile(SOURCE / "Hand_Poses_3D.zip") as archive:
        independent = np.load(io.BytesIO(archive.read(f"Hand_Poses_3D/{sequence}/hand_joints.npy")), allow_pickle=False)
    joints = np.load(output / f"hand_poses/Hand_Poses/{sequence}/hand_joints.npy", allow_pickle=False)
    if not np.array_equal(joints, independent):
        raise ValueError("full hand archive and joint-only archive disagree")
    for role, (member, object_id) in roles.items():
        extract(SOURCE / "Object_Poses.zip", member, "object_poses/" + member)
        mesh_member = f"object_models_released/{object_id}_cm.obj"
        extract(SOURCE / "Object_Models.zip", mesh_member, "object_models/" + mesh_member)
    for filename in ("egocentric_frame_extrinsic.npy", "egocentric_intrinsic.txt"):
        member = f"Egocentric_Camera_Parameters/{sequence}/{filename}"
        extract(SOURCE / "Egocentric_Camera_Parameters.zip", member, "camera/" + member)
    with (output / "taco_info.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    report = audit_episode(output, "pour", episode, sequence, row,
                           video_kinds=(("rgb", ".mp4"), ("depth_resized", ".avi")))
    motions = {}
    for role, (member, _) in roles.items():
        poses = np.load(output / ("object_poses/" + member), allow_pickle=False).astype(float)
        angles = Rotation.from_matrix(poses[0, :3, :3].T @ poses[:, :3, :3]).magnitude()
        motions[role] = dict(max_rotation_from_initial_rad=float(angles.max()),
            max_rotation_source_row_zero_based=int(angles.argmax()),
            max_height_increase_from_initial_m=float((poses[:, 2, 3] - poses[0, 2, 3]).max()))
    report.update(status_label="input_audit_not_robot_or_physics_validation",
        independent_hand_archives_agree=True, reference_motion=motions,
        missing_modalities={"depth_original": "not_part_of_this_initial_local_archive_acquisition; add_as_a_separately_hashed_supplement"},
        paper_parameters=dict(source="Appendix C.2, PDF page 22, task-level example",
            position_threshold_m=0.12, rotation_threshold_rad=1.5,
            lambda_p=None, lambda_R=None, C=None, exact_author_episode=False),
        source_files_modified=False, frames_trimmed=False, physics_validated=False,
        code=[artifact(Path(__file__)), artifact(ROOT / "scripts/audit_taco_paper_inputs.py")])
    with (output / "input_audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = dict(status="copied_from_local_release_archives_not_network_download",
        sequence=sequence, episode=episode, records=records, files=len(records),
        copied_bytes=sum(Path(r["output"]["path"]).stat().st_size for r in records),
        original_depth_included=False, source_projects_modified=False,
        symlinks_created=False, joint_only_archive_crosscheck_equal=True)
    with (output / "acquisition_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), sequence=sequence, counts=report["counts"],
                          status=report["status"], motion=motions, copied_bytes=manifest["copied_bytes"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
