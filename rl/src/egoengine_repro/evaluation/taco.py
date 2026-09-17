"""Selective TACO V1 ground-truth materialization directly from release ZIPs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree

from .. import SCHEMA_VERSION
from ..artifacts import artifact_record


TACO_ARCHIVES = {
    "camera": "Egocentric_Camera_Parameters.zip",
    "hand": "Hand_Poses_3D.zip",
    "object_pose": "Object_Poses.zip",
    "mesh": "Object_Models.zip",
    "segmentation": "2D_Segmentation.zip",
}
FINGERTIP_INDICES = np.asarray([4, 8, 12, 16, 20])
MANO_JOINT_NAMES = (
    "Wrist", "ThumbMCP", "ThumbPIP", "ThumbDIP", "ThumbTip",
    "IndexMCP", "IndexPIP", "IndexDIP", "IndexTip",
    "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip",
    "RingMCP", "RingPIP", "RingDIP", "RingTip",
    "LittleMCP", "LittlePIP", "LittleDIP", "LittleTip",
)


@dataclass(frozen=True)
class TacoSetSpec:
    path: Path
    data: dict[str, Any]
    dataset_root: Path
    selection_sha256: str

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_taco_set_path(name: str) -> Path:
    if name != "taco_object_gt_16":
        raise ValueError(f"unknown bundled TACO set: {name}")
    return Path(str(files("egoengine_repro").joinpath(
        "configs", "formal_3_1_taco_object_gt_16.yaml",
    )))


def _zip_array(archive: ZipFile, member: str) -> np.ndarray:
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(f"archive member missing: {member}") from error
    return np.asarray(np.load(io.BytesIO(payload), allow_pickle=False))


def _members(archive: ZipFile, prefix: str, suffix: str = "") -> list[str]:
    return sorted(
        name for name in archive.namelist()
        if name.startswith(prefix) and not name.endswith("/") and name.endswith(suffix)
    )


def load_taco_set(path_or_name: str | Path) -> TacoSetSpec:
    candidate = Path(path_or_name)
    path = candidate if candidate.exists() else bundled_taco_set_path(str(path_or_name))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO set must use schema version 1.0")
    root = Path(data["dataset_root"]).expanduser().resolve()
    for archive in TACO_ARCHIVES.values():
        if not (root / archive).is_file():
            raise FileNotFoundError(root / archive)
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != int(data.get("episode_count", 0)):
        raise ValueError("TACO episode_count must match its episode list")
    identifiers = [str(episode.get("episode_id", "")) for episode in episodes]
    sequences = [str(episode.get("sequence_id", "")) for episode in episodes]
    if (
        any(not value for value in (*identifiers, *sequences))
        or len(set(identifiers)) != len(identifiers)
        or len(set(sequences)) != len(sequences)
    ):
        raise ValueError("TACO episode and sequence IDs must be non-empty and unique")
    rows = {
        row["sequence_id"]: row
        for row in csv.DictReader((root / "taco_info.csv").open(encoding="utf-8"))
    }
    for episode in episodes:
        row = rows.get(str(episode["sequence_id"]))
        if row is None:
            raise ValueError(f"TACO sequence missing from metadata: {episode['sequence_id']}")
        if row["all_modalities_complete"] != "True" or row["calib_status"] != "good":
            raise ValueError(f"TACO sequence is not complete/good: {episode['sequence_id']}")
        if str(episode.get("evaluation_object")) not in {"tool", "target"}:
            raise ValueError("TACO evaluation_object must be tool or target")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return TacoSetSpec(
        path=path.resolve(), data=data, dataset_root=root,
        selection_sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    )


def _hand_transforms(joints: np.ndarray) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (2, 21, 3):
        raise ValueError(f"TACO hand joints must have shape (T,2,21,3), got {values.shape}")
    transforms = np.broadcast_to(np.eye(4), values.shape[:-1] + (4, 4)).copy()
    transforms[..., :3, 3] = values
    return transforms


def _mesh_from_archive(archive: ZipFile, object_id: str) -> trimesh.Trimesh:
    member = f"object_models_released/{object_id}_cm.obj"
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(f"TACO object model missing: {member}") from error
    mesh = trimesh.load_mesh(io.BytesIO(payload), file_type="obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"unsupported TACO object mesh: {member}")
    mesh.apply_scale(0.01)
    return mesh


def _derived_contact(
    hand_joints_world: np.ndarray, T_world_object: np.ndarray,
    mesh: trimesh.Trimesh, threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    fingertips = np.asarray(hand_joints_world, dtype=np.float64)[..., FINGERTIP_INDICES, :]
    poses = np.asarray(T_world_object, dtype=np.float64)
    rotation_inv = np.swapaxes(poses[:, :3, :3], 1, 2)
    local_tips = np.einsum(
        "tij,thfj->thfi", rotation_inv,
        fingertips - poses[:, None, None, :3, 3],
    )
    distances = cKDTree(np.asarray(mesh.vertices, dtype=np.float64)).query(
        local_tips.reshape(-1, 3), workers=-1,
    )[0].reshape(local_tips.shape[:-1])
    valid = np.isfinite(distances)
    return (distances <= threshold_m) & valid, distances


def materialize_taco_set(
    spec: TacoSetSpec, output_dir: str | Path, *, force: bool = False,
) -> Path:
    """Materialize 16 selected TACO bundles without extracting the full release."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = spec.dataset_root
    rows = {
        row["sequence_id"]: row
        for row in csv.DictReader((root / "taco_info.csv").open(encoding="utf-8"))
    }
    threshold = float(spec.data.get("contact_threshold_m", 0.005))
    if threshold <= 0:
        raise ValueError("contact_threshold_m must be positive")
    archive_paths = {name: root / filename for name, filename in TACO_ARCHIVES.items()}
    archive_records = {name: artifact_record(path) for name, path in archive_paths.items()}
    with (
        ZipFile(archive_paths["camera"]) as camera_zip,
        ZipFile(archive_paths["hand"]) as hand_zip,
        ZipFile(archive_paths["object_pose"]) as pose_zip,
        ZipFile(archive_paths["mesh"]) as mesh_zip,
        ZipFile(archive_paths["segmentation"]) as segmentation_zip,
    ):
        episodes = []
        for episode in spec.data["episodes"]:
            episode_id = str(episode["episode_id"])
            sequence_id = str(episode["sequence_id"])
            role = str(episode["evaluation_object"])
            row = rows[sequence_id]
            destination = output / "episodes" / episode_id
            manifest_path = destination / "ground_truth_manifest.json"
            if manifest_path.is_file() and not force:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                destination.mkdir(parents=True, exist_ok=True)
                camera_prefix = f"Egocentric_Camera_Parameters/{sequence_id}/"
                raw_extrinsic = _zip_array(
                    camera_zip, camera_prefix + "egocentric_frame_extrinsic.npy",
                ).astype(np.float64)
                intrinsic_text = camera_zip.read(
                    camera_prefix + "egocentric_intrinsic.txt",
                ).decode("utf-8")
                intrinsic = np.loadtxt(io.StringIO(intrinsic_text), dtype=np.float64)
                joints = _zip_array(
                    hand_zip, f"Hand_Poses_3D/{sequence_id}/hand_joints.npy",
                ).astype(np.float64)
                pose_prefix = f"Object_Poses/{sequence_id}/"
                pose_members = _members(pose_zip, pose_prefix, ".npy")
                selected_members = [
                    member for member in pose_members
                    if Path(member).stem.startswith(f"{role}_")
                ]
                if len(selected_members) != 1:
                    raise ValueError(f"expected one {role} pose for {sequence_id}")
                pose_member = selected_members[0]
                T_world_object = _zip_array(pose_zip, pose_member).astype(np.float64)
                object_id = Path(pose_member).stem.split("_", 1)[1]
                mesh = _mesh_from_archive(mesh_zip, object_id)
                frame_count = min(len(raw_extrinsic), len(joints), len(T_world_object))
                expected = int(row["n_frames"])
                if frame_count != expected:
                    raise ValueError(
                        f"TACO modality length mismatch for {sequence_id}: {frame_count} != {expected}"
                    )
                frames = np.arange(frame_count, dtype=np.int64)
                timestamps = frames.astype(np.float64) / float(row["fps"])

                # Release object poses and hand joints share the mocap world frame;
                # egocentric extrinsics map that world frame into the camera.
                T_world_camera = np.linalg.inv(raw_extrinsic)
                T_camera_object = raw_extrinsic @ T_world_object
                contact, distances = _derived_contact(joints, T_world_object, mesh, threshold)

                hand_path = destination / "hand_ground_truth.npz"
                np.savez_compressed(
                    hand_path, frame_indices=frames, timestamps_s=timestamps,
                    hand_order=np.asarray(["left", "right"]),
                    joint_names=np.asarray(MANO_JOINT_NAMES),
                    T_world_joint=_hand_transforms(joints),
                    confidence=np.ones((frame_count, 2, 21), dtype=np.float64),
                    rotation_available=np.asarray(False),
                )
                camera_path = destination / "camera_ground_truth.npz"
                np.savez_compressed(
                    camera_path, frame_indices=frames, timestamps_s=timestamps,
                    intrinsics=intrinsic, T_world_camera=T_world_camera,
                    raw_T_camera_world=raw_extrinsic,
                )
                trajectory_path = destination / "object_trajectory_ground_truth.npz"
                np.savez_compressed(
                    trajectory_path, frame_indices=frames, timestamps_s=timestamps,
                    T_camera_object=T_camera_object, T_world_object=T_world_object,
                    valid=np.isfinite(T_world_object).all(axis=(1, 2)),
                    object_role=np.asarray(role), object_id=np.asarray(object_id),
                )
                contact_path = destination / "contact_ground_truth.npz"
                np.savez_compressed(
                    contact_path, frame_indices=frames, timestamps_s=timestamps,
                    contact=contact, fingertip_surface_distance_m=distances,
                    valid=np.isfinite(distances), threshold_m=np.asarray(threshold),
                )
                mesh_path = destination / f"{role}_{object_id}_meters.ply"
                mesh.export(mesh_path)
                segmentation_prefix = f"2D_Segmentation/{sequence_id}/"
                segmentation_members = _members(
                    segmentation_zip, segmentation_prefix, "_masks.npy",
                )
                segmentation_source = destination / "allocentric_segmentation_source.json"
                segmentation_source.write_text(json.dumps({
                    "archive": archive_records["segmentation"],
                    "members": segmentation_members,
                    "camera_ids": [Path(member).stem.removesuffix("_masks") for member in segmentation_members],
                    "fps": 6.0,
                    "registered_as_egocentric_gt": False,
                    "reason": "TACO V1 masks are allocentric and cannot score egocentric predictions",
                }, indent=2) + "\n", encoding="utf-8")
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": "TACO V1", "episode_id": episode_id,
                    "sequence_id": sequence_id,
                    "uses_ground_truth": True, "scope": "evaluation_only",
                    "benchmark_role": str(spec.data["benchmark_role"]),
                    "exact_paper_test_set": False,
                    "frame_count": frame_count, "fps": float(row["fps"]),
                    "object_role": role, "object_id": object_id,
                    "camera_extrinsic_interpretation": "release_array_is_T_camera_world",
                    "object_pose_interpretation": "release_array_is_T_world_object",
                    "quality_labels": {
                        "hand": "dataset_ground_truth_positions_only",
                        "camera": "dataset_calibration_ground_truth",
                        "mesh": "dataset_ground_truth",
                        "object_trajectory": "dataset_ground_truth",
                        "contact": "derived_from_ground_truth_geometry",
                    },
                    "artifacts": {
                        "hand": artifact_record(hand_path),
                        "camera": artifact_record(camera_path),
                        "mesh": artifact_record(mesh_path),
                        "object_trajectory": artifact_record(trajectory_path),
                        "contact": artifact_record(contact_path),
                    },
                    "auxiliary_artifacts": {
                        "allocentric_segmentation": artifact_record(segmentation_source),
                    },
                    "unavailable_modalities": ["segmentation", "depth"],
                    "contact_definition": {
                        "threshold_m": threshold,
                        "distance": "nearest released mesh vertex in object coordinates",
                    },
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            episodes.append({
                "episode_id": episode_id,
                "sequence_id": sequence_id,
                "triplet": row["triplet"],
                "frame_count": int(manifest["frame_count"]),
                "object_role": str(manifest["object_role"]),
                "ground_truth_manifest": artifact_record(manifest_path),
                "available_modalities": sorted(manifest["artifacts"]),
                "unavailable_modalities": manifest["unavailable_modalities"],
            })
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "dataset": "TACO V1",
        "benchmark_role": str(spec.data["benchmark_role"]),
        "exact_paper_test_set": False,
        "exact_paper_episode_ids_published": False,
        "selection_sha256": spec.selection_sha256,
        "episode_count": len(episodes),
        "contact_threshold_m": threshold,
        "spec": artifact_record(spec.path),
        "metadata": artifact_record(root / "taco_info.csv"),
        "source_archives": archive_records,
        "episodes": episodes,
        "gt_policy": {"scope": "evaluation_only", "may_enter_automatic_inference": False},
    }
    destination = output / "frozen_taco_evaluation_set_manifest.json"
    if destination.is_file() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        for key in ("selection_sha256", "episodes"):
            if existing.get(key) != frozen.get(key):
                raise ValueError("frozen TACO inputs changed; use a new output directory")
        return destination
    destination.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return destination


def slice_taco_ground_truth_bundle(
    ground_truth_manifest: str | Path, frame_indices: np.ndarray,
    output_dir: str | Path, *, force: bool = False,
) -> Path:
    """Create an evaluation-only TACO GT bundle on an exact RGB interval."""
    source_path = Path(ground_truth_manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("dataset") != "TACO V1" or source.get("scope") != "evaluation_only":
        raise ValueError("TACO interval slicing requires an evaluation-only TACO GT bundle")
    selected_frames = np.asarray(frame_indices, dtype=np.int64)
    if (
        selected_frames.ndim != 1 or not len(selected_frames)
        or len(np.unique(selected_frames)) != len(selected_frames)
    ):
        raise ValueError("selected TACO frame indices must be a non-empty unique vector")
    output = Path(output_dir).resolve()
    destination = output / "ground_truth_manifest.json"
    if destination.is_file() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("selected_frame_indices") != selected_frames.tolist():
            raise ValueError("frozen TACO GT interval differs; use a new output directory")
        return destination
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    sliced_modalities = {"hand", "camera", "object_trajectory", "contact"}
    for name, record in source["artifacts"].items():
        source_artifact = Path(record["path"]).resolve()
        if name not in sliced_modalities:
            artifacts[name] = artifact_record(source_artifact)
            continue
        with np.load(source_artifact, allow_pickle=False) as artifact:
            arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
        source_frames = np.asarray(arrays["frame_indices"], dtype=np.int64)
        lookup = {int(frame): index for index, frame in enumerate(source_frames)}
        missing = [int(frame) for frame in selected_frames if int(frame) not in lookup]
        if missing:
            raise ValueError(f"TACO {name} GT misses selected frames: {missing[:5]}")
        selected = np.asarray([lookup[int(frame)] for frame in selected_frames], dtype=np.int64)
        sliced = {
            key: value[selected] if value.ndim >= 1 and len(value) == len(source_frames) else value
            for key, value in arrays.items()
        }
        sliced["frame_indices"] = selected_frames
        artifact_path = output / source_artifact.name
        np.savez_compressed(artifact_path, **sliced)
        artifacts[name] = artifact_record(artifact_path)
    manifest = {
        **source,
        "parent_ground_truth_manifest": artifact_record(source_path),
        "frame_count": len(selected_frames),
        "selected_frame_indices": selected_frames.tolist(),
        "interval_materialization": "exact_intersection_with_decodable_egocentric_RGB",
        "artifacts": artifacts,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination
