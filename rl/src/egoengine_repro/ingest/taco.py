"""TACO first-person ingest and explicit paper-input materialization.

Ground-truth hand, object-pose, and contact arrays stay outside automatic run
directories. Camera calibration is copied into the normal ingest contract;
known meshes and scene-derived base alignment require separate opt-in calls.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np
import trimesh
import yaml

from video_to_spider.manifest import RunManifest, stage_cache_key
from video_to_spider.schemas import SCHEMA_VERSION, validate_transforms

from ..artifacts import artifact_record


@dataclass(frozen=True)
class TacoFirstPersonSpec:
    path: Path
    data: dict[str, Any]
    dataset_root: Path
    selection_sha256: str

    @property
    def name(self) -> str:
        return str(self.data["name"])


def bundled_taco_first_person_path(name: str) -> Path:
    if name != "taco_first_person_dev4":
        raise ValueError(f"unknown bundled TACO first-person set: {name}")
    return Path(__file__).resolve().parents[1] / "configs/taco_first_person_dev4.yaml"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _metadata_rows(root: Path) -> dict[str, dict[str, str]]:
    with (root / "taco_info.csv").open(encoding="utf-8") as handle:
        return {row["sequence_id"]: row for row in csv.DictReader(handle)}


def load_taco_first_person_set(path_or_name: str | Path) -> TacoFirstPersonSpec:
    candidate = Path(path_or_name)
    path = candidate if candidate.exists() else bundled_taco_first_person_path(str(path_or_name))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema_version")) != "1.0":
        raise ValueError("TACO first-person set must use schema version 1.0")
    root = Path(data["dataset_root"]).expanduser().resolve()
    required = (
        "taco_info.csv", "Egocentric_Camera_Parameters.zip",
        "Object_Models.zip", "Object_Poses.zip",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != int(data.get("episode_count", 0)):
        raise ValueError("TACO first-person episode_count must match its episode list")
    rows = _metadata_rows(root)
    episode_ids: set[str] = set()
    sequence_ids: set[str] = set()
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        sequence_id = str(episode.get("sequence_id", ""))
        if not episode_id or episode_id in episode_ids:
            raise ValueError("TACO first-person episode IDs must be non-empty and unique")
        if not sequence_id or sequence_id in sequence_ids:
            raise ValueError("TACO first-person sequence IDs must be non-empty and unique")
        episode_ids.add(episode_id)
        sequence_ids.add(sequence_id)
        row = rows.get(sequence_id)
        if row is None or row.get("has_egocentric_rgb") != "True":
            raise ValueError(f"TACO first-person RGB is unavailable: {sequence_id}")
        start = int(episode.get("start_frame", 0))
        end = int(episode.get("end_frame", row["n_frames"]))
        if start < 0 or end <= start or end > int(row["n_frames"]):
            raise ValueError(f"invalid TACO interval [{start}, {end}) for {sequence_id}")
        if str(episode.get("evaluation_object")) not in {"tool", "target"}:
            raise ValueError("TACO evaluation_object must be tool or target")
    base = data.get("base_frame", {})
    if float(base.get("workspace_offset_m", 0.0)) <= 0 or float(base.get("table_height_m", 0.0)) <= 0:
        raise ValueError("TACO base frame requires positive workspace offset and table height")
    return TacoFirstPersonSpec(
        path=path.resolve(), data=data, dataset_root=root,
        selection_sha256=_canonical_hash(data),
    )


def _extract_member(archive: ZipFile, member: str, destination: Path) -> None:
    try:
        source = archive.open(member)
    except KeyError as error:
        raise FileNotFoundError(f"archive member missing: {member}") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    temporary.replace(destination)


def extract_taco_rgb_videos(
    spec: TacoFirstPersonSpec, output_dir: str | Path, *, force: bool = False,
) -> Path:
    """Extract only the frozen episode videos from the 24.7 GB RGB archive."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _metadata_rows(spec.dataset_root)
    records = []
    archive_path = spec.dataset_root / "Egocentric_RGB_Videos.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with ZipFile(archive_path) as archive:
        for episode in spec.data["episodes"]:
            row = rows[str(episode["sequence_id"])]
            destination = output / str(episode["episode_id"]) / "color.mp4"
            if force or not destination.is_file():
                _extract_member(archive, row["egocentric_rgb_path"], destination)
            records.append({
                "episode_id": str(episode["episode_id"]),
                "sequence_id": str(episode["sequence_id"]),
                "archive_member": row["egocentric_rgb_path"],
                "video": artifact_record(destination),
            })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "selection_sha256": spec.selection_sha256,
        "source_archive": artifact_record(archive_path),
        "episode_count": len(records),
        "episodes": records,
    }
    manifest_path = output / "selected_rgb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _zip_array(archive: ZipFile, member: str) -> np.ndarray:
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(f"archive member missing: {member}") from error
    return np.asarray(np.load(io.BytesIO(payload), allow_pickle=False))


def _video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    result = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if result["frame_count"] <= 0 or result["fps"] <= 0:
        raise ValueError(f"invalid video metadata: {result}")
    return result


def _ingest_episode(
    spec: TacoFirstPersonSpec, episode: dict[str, Any], video_path: Path,
    run_dir: Path, *, overwrite: bool,
) -> Path:
    rows = _metadata_rows(spec.dataset_root)
    sequence_id = str(episode["sequence_id"])
    row = rows[sequence_id]
    video = _video_info(video_path)
    expected_count = int(row["n_frames"])
    start = int(episode.get("start_frame", 0))
    end = int(episode.get("end_frame", expected_count))
    if video["frame_count"] < end or video["frame_count"] > expected_count:
        raise ValueError(
            f"TACO RGB cannot cover [{start}, {end}) for {sequence_id}: "
            f"video={video['frame_count']} metadata={expected_count}"
        )
    expected_fps = float(row["fps"])
    if not np.isclose(video["fps"], expected_fps, atol=1e-3):
        raise ValueError(f"TACO RGB FPS mismatch for {sequence_id}: {video['fps']} != {expected_fps}")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"run already exists: {run_dir}; pass force=True")
    for relative in ("input", "frames/rgb", "calibration"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    camera_archive_path = spec.dataset_root / "Egocentric_Camera_Parameters.zip"
    camera_prefix = f"Egocentric_Camera_Parameters/{sequence_id}/"
    with ZipFile(camera_archive_path) as archive:
        raw_T_camera_world = _zip_array(
            archive, camera_prefix + "egocentric_frame_extrinsic.npy",
        ).astype(np.float64)
        intrinsic = np.loadtxt(io.StringIO(archive.read(
            camera_prefix + "egocentric_intrinsic.txt",
        ).decode("utf-8")), dtype=np.float64)
    if raw_T_camera_world.shape[0] != expected_count:
        raise ValueError("TACO camera trajectory does not match RGB frame count")
    T_world_camera = np.linalg.inv(raw_T_camera_world)
    validate_transforms("T_world_camera", T_world_camera)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"invalid TACO camera intrinsics: {intrinsic}")

    config = {
        "dataset": "TACO V1", "start_frame": start, "end_frame": end,
        "selection_sha256": spec.selection_sha256,
    }
    manifest = RunManifest.create(
        manifest_path, run_id=run_dir.name, source_episode=sequence_id,
        config=config, frame_count=end - start, fps=video["fps"],
    )
    manifest.data.update({
        "dataset": "TACO V1",
        "input_profile": "first_person_rgb_and_camera_only",
        "ground_truth_isolation": {
            "hand_object_pose_contact_in_run": False,
            "camera_calibration_is_dataset_input": True,
        },
    })
    manifest.save()
    cache_key = stage_cache_key("taco_ingest", config, [video_path, camera_archive_path])
    manifest.start_stage(
        "ingest", cache_key=cache_key,
        command=["egoengine-repro", "ingest-taco-devset", str(spec.path)],
        environment="v2s-core",
    )
    instruction = f"{row['action']} {row['object']} with {row['tool']}"
    source = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "TACO V1",
        "task_directory": row["triplet"],
        "episode_id": str(episode["episode_id"]),
        "sequence_id": sequence_id,
        "mp4_path": str(video_path.resolve()),
        "instruction_text": instruction,
        "instruction_source": "taco_info.csv",
        "object_keyword_candidates": [
            row["tool"] if str(episode["evaluation_object"]) == "tool" else row["object"]
        ],
        "video": video,
        "selected_frame_interval": [start, end],
        "public_attributes": {
            "action": row["action"], "tool": row["tool"], "object": row["object"],
            "evaluation_object": str(episode["evaluation_object"]),
        },
        "excluded_ground_truth": ["hand_pose", "object_pose", "contact", "segmentation"],
    }
    (run_dir / "input/source.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8",
    )
    np.save(run_dir / "calibration/intrinsics.npy", intrinsic.astype(np.float32))
    np.save(
        run_dir / "calibration/T_world_camera.npy",
        T_world_camera[start:end].astype(np.float32),
    )
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_rows = []
    for source_index in range(start, end):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"TACO RGB decode failed at source frame {source_index}")
        relative = f"frames/rgb/{source_index:06d}.jpg"
        if not cv2.imwrite(str(run_dir / relative), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"failed to write {relative}")
        frame_rows.append({
            "frame_index": source_index - start,
            "source_frame_index": source_index,
            "timestamp_s": (source_index - start) / video["fps"],
            "source_timestamp_s": source_index / video["fps"],
            "rgb_path": relative,
        })
    capture.release()
    (run_dir / "frames/frame_index.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "fps": video["fps"], "frames": frame_rows,
    }, indent=2) + "\n", encoding="utf-8")
    outputs = [
        "input/source.json", "frames/frame_index.json", "calibration/intrinsics.npy",
        "calibration/T_world_camera.npy",
    ]
    manifest.finish_stage(
        "ingest", success=True, outputs=outputs,
        quality_metrics={
            "decoded_frames": len(frame_rows),
            "dataset_frames": expected_count,
            "ground_truth_arrays_copied": 0,
        },
        warnings=["TACO camera calibration is a dataset input shared by both A/B branches"],
    )
    if video["frame_count"] != expected_count:
        manifest = RunManifest.load(manifest_path)
        manifest.data["stages"]["ingest"]["warnings"].append(
            f"RGB has {video['frame_count']} decodable frames while TACO metadata has "
            f"{expected_count}; the frozen interval is [{start}, {end})"
        )
        manifest.save()
    return manifest_path


def ingest_taco_devset(
    spec: TacoFirstPersonSpec, video_dir: str | Path, output_dir: str | Path,
    *, force: bool = False,
) -> Path:
    video_root = Path(video_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode in spec.data["episodes"]:
        episode_id = str(episode["episode_id"])
        video = video_root / episode_id / "color.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        run_dir = output / "episodes" / episode_id
        manifest = _ingest_episode(spec, episode, video, run_dir, overwrite=force)
        episodes.append({
            "episode_id": episode_id,
            "sequence_id": str(episode["sequence_id"]),
            "source_run": artifact_record(manifest),
        })
    result = output / "taco_first_person_ingest_manifest.json"
    result.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "selection_sha256": spec.selection_sha256,
        "episode_count": len(episodes),
        "ground_truth_policy": "camera calibration only; no hand/object/contact GT in source runs",
        "episodes": episodes,
    }, indent=2) + "\n", encoding="utf-8")
    return result


def materialize_known_mesh_proposal(
    run_dir: str | Path, ground_truth_manifest: str | Path, *, overwrite: bool = False,
) -> Path:
    """Expose a dataset mesh through the existing FoundationPose proposal contract."""
    root = Path(run_dir).resolve()
    gt_path = Path(ground_truth_manifest).resolve()
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if gt.get("scope") != "evaluation_only" or not gt.get("uses_ground_truth"):
        raise ValueError("known mesh input must come from an explicit evaluation-only GT bundle")
    source_mesh = Path(gt["artifacts"]["mesh"]["path"]).resolve()
    output_dir = root / "mesh_proposals"
    ranking_path = output_dir / "mesh_ranking.json"
    if ranking_path.exists() and not overwrite:
        raise FileExistsError(f"mesh proposal already exists: {ranking_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = output_dir / "known_mesh"
    proposal_dir.mkdir(exist_ok=True)
    visual_path = proposal_dir / "visual.ply"
    collision_path = proposal_dir / "collision_source.ply"
    shutil.copy2(source_mesh, visual_path)
    shutil.copy2(source_mesh, collision_path)
    mesh = trimesh.load_mesh(visual_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"known mesh is invalid: {source_mesh}")
    masks_path = root / "segmentation/object_masks.npz"
    with np.load(masks_path, allow_pickle=False) as masks:
        values = np.asarray(masks["masks"], dtype=bool)
        valid = np.asarray(masks["valid"], dtype=bool)
        frame_indices = np.asarray(masks["frame_indices"], dtype=np.int64)
    areas = values.reshape(values.shape[0], -1).sum(axis=1)
    usable = np.flatnonzero(valid & (areas > 0))
    if not usable.size:
        raise ValueError("known-mesh FoundationPose route requires at least one valid object mask")
    anchor_at = int(usable[np.argmax(areas[usable])])
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    proposal = {
        "proposal_id": "known_mesh",
        "frame_index": int(frame_indices[anchor_at]),
        "mask_index": anchor_at,
        "seed": -1,
        "qualified": True,
        "static_score": 1.0,
        "rank": 1,
        "visual_mesh": str(visual_path.relative_to(output_dir)),
        "collision_source_mesh": str(collision_path.relative_to(output_dir)),
        "selected_scale_m": 1.0,
        "integrity": {
            "qualified": True, "score": 1.0,
            "vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces)),
            "extents_m": extents.tolist(),
        },
        "fit": {"source": "deferred_to_foundationpose"},
        "canonical": {
            "source": "TACO released metric object model",
            "debug_only": False,
            "already_metric": True,
        },
        "input_provenance": {
            "uses_ground_truth": True,
            "role": "known_object_mesh_allowed_by_paper_faithful_profile",
            "ground_truth_manifest": str(gt_path),
            "source_mesh": artifact_record(source_mesh),
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "known_mesh_proposal",
        "ranking_policy": "single dataset-known metric mesh",
        "foundationpose_used": False,
        "keyframes": [{
            "frame_index": int(frame_indices[anchor_at]),
            "mask_index": anchor_at,
            "score": 1.0,
            "mask_area_px": int(areas[anchor_at]),
        }],
        "proposals": [proposal],
        "qualified_count": 1,
        "success": True,
        "failure_reason": None,
        "uses_ground_truth": True,
    }
    ranking_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest = RunManifest.load(root / "manifest.json")
    cache_key = stage_cache_key("known_mesh_proposal", {}, [gt_path, source_mesh, masks_path])
    manifest.start_stage(
        "known_mesh_proposal", cache_key=cache_key,
        command=["egoengine-repro", "prepare-taco-known-mesh", str(gt_path)],
        environment="v2s-core",
    )
    manifest.finish_stage(
        "known_mesh_proposal", success=True,
        outputs=[
            str(ranking_path.relative_to(root)), str(visual_path.relative_to(root)),
            str(collision_path.relative_to(root)),
        ],
        quality_metrics={"qualified_count": 1, "scale_to_m": 1.0},
        warnings=["This opt-in stage uses the TACO released object mesh"],
    )
    return ranking_path


def materialize_foundationpose_object_reference(
    run_dir: str | Path, T_sim_world_path: str | Path, output_path: str | Path,
    *, overwrite: bool = False,
) -> Path:
    """Transform raw FoundationPose output into the shared simulator reference contract."""
    root = Path(run_dir).resolve()
    source_path = root / "object_tracking/foundationpose_raw.npz"
    with np.load(source_path, allow_pickle=False) as artifact:
        frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(artifact["timestamps_s"], dtype=np.float64)
        T_camera_object = np.asarray(artifact["T_camera_object"], dtype=np.float64)
        valid = np.asarray(artifact["valid"], dtype=bool)
    T_world_camera = np.load(root / "calibration/T_world_camera.npy").astype(np.float64)
    if len(T_world_camera) != len(T_camera_object):
        raise ValueError("FoundationPose and camera trajectories must share one timeline")
    transform_source = Path(T_sim_world_path).resolve()
    if transform_source.suffix == ".json":
        T_sim_world = np.asarray(json.loads(
            transform_source.read_text(encoding="utf-8"),
        )["T_sim_world"], dtype=np.float64)
    elif transform_source.suffix == ".npz":
        with np.load(transform_source, allow_pickle=False) as artifact:
            T_sim_world = np.asarray(artifact["T_sim_world"], dtype=np.float64)
    else:
        T_sim_world = np.asarray(np.load(transform_source, allow_pickle=False), dtype=np.float64)
    if T_sim_world.shape != (4, 4) or not np.isfinite(T_sim_world).all():
        raise ValueError("T_sim_world must be a finite (4,4) matrix")
    T_world_object = np.einsum("tij,tjk->tik", T_world_camera, T_camera_object)
    T_sim_object = np.einsum("ij,tjk->tik", T_sim_world, T_world_object)
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        T_sim_object_reference=T_sim_object[:, None],
        valid=valid & np.isfinite(T_sim_object).all(axis=(1, 2)),
        T_sim_world=T_sim_world,
        source_foundationpose=np.asarray(str(source_path)),
        coordinate_alignment_source=np.asarray(str(transform_source)),
    )
    report = destination.with_suffix(".json")
    report.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source": artifact_record(source_path),
        "alignment": artifact_record(transform_source),
        "output": artifact_record(destination),
        "frame_count": len(frame_indices),
        "valid_rate": float(np.mean(valid)),
        "contact_scale_calibration": False,
        "per_frame_floor_shift": False,
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def reuse_run_artifacts(
    source_run: str | Path, target_run: str | Path, artifacts: list[str],
    *, overwrite: bool = False,
) -> Path:
    """Link immutable upstream outputs across paired ablation branches."""
    allowed = {
        "segmentation", "hands", "depth", "mesh_proposals",
        "object_tracking", "optimization", "rendered_gt_proxy",
    }
    if not artifacts or set(artifacts) - allowed:
        raise ValueError(f"reusable artifacts must be selected from {sorted(allowed)}")
    source = Path(source_run).resolve()
    target = Path(target_run).resolve()
    linked = []
    for name in artifacts:
        source_path = source / name
        target_path = target / name
        if not source_path.is_dir():
            raise FileNotFoundError(source_path)
        if target_path.exists() or target_path.is_symlink():
            if not overwrite:
                raise FileExistsError(target_path)
            if target_path.is_symlink() or target_path.is_file():
                target_path.unlink()
            else:
                shutil.rmtree(target_path)
        target_path.symlink_to(source_path, target_is_directory=True)
        linked.append(str(target_path.relative_to(target)))
    manifest = RunManifest.load(target / "manifest.json")
    cache_key = stage_cache_key(
        "reuse_paired_ablation_artifacts", {"artifacts": artifacts},
        [source / "manifest.json"],
    )
    stage_name = "reuse_" + "_".join(artifacts)
    manifest.start_stage(
        stage_name, cache_key=cache_key,
        command=["egoengine-repro", "reuse-run-artifacts", *artifacts],
        environment="v2s-core",
    )
    manifest.finish_stage(
        stage_name, success=True, outputs=linked,
        quality_metrics={"source_run": str(source), "artifact_count": len(linked)},
        warnings=["Linked outputs are immutable shared inputs for a paired ablation"],
    )
    report = target / f"{stage_name}.json"
    report.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "source_run": str(source),
        "target_run": str(target), "artifacts": artifacts,
        "policy": "read_only_paired_ablation_reuse",
    }, indent=2) + "\n", encoding="utf-8")
    return report


def _pose_members(archive: ZipFile, sequence_id: str) -> dict[str, str]:
    prefix = f"Object_Poses/{sequence_id}/"
    result: dict[str, str] = {}
    for member in archive.namelist():
        if not member.startswith(prefix) or not member.endswith(".npy"):
            continue
        stem = Path(member).stem
        role, _, object_id = stem.partition("_")
        if role in {"tool", "target"} and object_id:
            if role in result:
                raise ValueError(f"multiple TACO {role} poses for {sequence_id}")
            result[role] = member
    if set(result) != {"tool", "target"}:
        raise ValueError(f"TACO scene needs one tool and one target pose: {sequence_id}")
    return result


def _mesh_from_zip(archive: ZipFile, object_id: str) -> trimesh.Trimesh:
    member = f"object_models_released/{object_id}_cm.obj"
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(member) from error
    mesh = trimesh.load_mesh(io.BytesIO(payload), file_type="obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"invalid TACO mesh: {member}")
    mesh.apply_scale(0.01)
    return mesh


def freeze_taco_base_frame(
    spec: TacoFirstPersonSpec, episode_id: str, output_path: str | Path,
    *, overwrite: bool = False,
) -> Path:
    """Freeze the paper's object-centered 0.6 m TACO pseudo-base heuristic.

    The source paper gives the offset magnitude and table height but does not
    publish a full axis convention. This implementation places the object-pair
    center 0.6 m along simulator +X and centers it at Y=0.
    """
    matches = [item for item in spec.data["episodes"] if str(item["episode_id"]) == episode_id]
    if len(matches) != 1:
        raise KeyError(f"expected one TACO episode named {episode_id}, found {len(matches)}")
    episode = matches[0]
    sequence_id = str(episode["sequence_id"])
    start = int(episode.get("start_frame", 0))
    pose_archive_path = spec.dataset_root / "Object_Poses.zip"
    mesh_archive_path = spec.dataset_root / "Object_Models.zip"
    with ZipFile(pose_archive_path) as pose_archive, ZipFile(mesh_archive_path) as mesh_archive:
        members = _pose_members(pose_archive, sequence_id)
        poses = {
            role: _zip_array(pose_archive, member).astype(np.float64)
            for role, member in members.items()
        }
        meshes = {
            role: _mesh_from_zip(mesh_archive, Path(member).stem.split("_", 1)[1])
            for role, member in members.items()
        }
    centers = np.stack([poses[role][start, :3, 3] for role in ("tool", "target")])
    object_pair_center = centers.mean(axis=0)
    target_vertices = meshes["target"].vertices
    T_world_target = poses["target"][start]
    target_world = (
        target_vertices @ T_world_target[:3, :3].T + T_world_target[:3, 3]
    )
    source_table_height = float(np.min(target_world[:, 2]))
    base = spec.data["base_frame"]
    offset = float(base["workspace_offset_m"])
    table_height = float(base["table_height_m"])
    desired_center_xy = np.asarray([offset, 0.0], dtype=np.float64)
    T_sim_world = np.eye(4, dtype=np.float64)
    T_sim_world[:2, 3] = desired_center_xy - object_pair_center[:2]
    T_sim_world[2, 3] = table_height - source_table_height
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        T_sim_world=T_sim_world,
        source_object_pair_center_world=object_pair_center,
        desired_object_pair_center_sim=np.asarray([
            offset, 0.0, object_pair_center[2] + T_sim_world[2, 3],
        ]),
        source_table_height_world=np.asarray(source_table_height),
        table_height_sim=np.asarray(table_height),
        source_frame_index=np.asarray(start),
        sequence_id=np.asarray(sequence_id),
    )
    report_path = destination.with_suffix(".json")
    report_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "dataset": "TACO V1",
        "episode_id": episode_id,
        "sequence_id": sequence_id,
        "method": "paper_appendix_A.1_object_pair_center_fixed_offset",
        "T_sim_world": T_sim_world.tolist(),
        "workspace_offset_m": offset,
        "table_height_m": table_height,
        "axis_convention": "object-pair center at simulator x=+offset, y=0; +Z up",
        "axis_disclosure": "paper publishes magnitude/table height but not the complete axis convention",
        "source_frame_index": start,
        "uses_ground_truth": True,
        "ground_truth_role": "dataset_scene_calibration_only",
        "per_frame_floor_shift": False,
        "contact_scale_calibration": False,
        "inputs": {
            "selection": artifact_record(spec.path),
            "object_poses": artifact_record(pose_archive_path),
            "object_models": artifact_record(mesh_archive_path),
        },
        "output": artifact_record(destination),
    }, indent=2) + "\n", encoding="utf-8")
    return destination
