"""Read-only input diagnostics; these checks are not EgoEngine success metrics."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


def artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest())


def verify_artifacts(records):
    for record in records:
        if artifact(Path(record["path"])) != record:
            raise ValueError(f"an audited source artifact changed: {record['path']}")


def scene_mesh_artifacts(scene: Path) -> list[dict]:
    """Snapshot external meshes of the project's flattened MJCF scenes."""
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    if root.findall(".//include") or (compiler is not None and compiler.get("strippath", "false") != "false"):
        raise ValueError("mesh provenance requires flattened MJCF without strippath")
    directory = scene.parent / (compiler.get("meshdir", compiler.get("assetdir", "")) if compiler is not None else "")
    paths = sorted({(directory / mesh.get("file")).resolve(strict=True)
                    for mesh in root.findall("asset/mesh") if mesh.get("file")})
    return [artifact(path) for path in paths]


def video_info(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,pix_fmt",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1 or result.stderr.strip():
        raise ValueError(f"video decode diagnostic failed: {path}: {result.stderr}")
    stream = streams[0]
    return dict(**stream, decoded_frames=int(stream["nb_read_frames"]),
                fps=float(Fraction(stream["avg_frame_rate"])))


def transform_report(transforms: np.ndarray) -> dict:
    transforms = np.asarray(transforms, dtype=float)
    if transforms.ndim < 3 or transforms.shape[-2:] != (4, 4) or not transforms.size:
        raise ValueError("expected nonempty (...,4,4) transforms")
    if not np.isfinite(transforms).all():
        return dict(finite=False, rigid_within_float32_tolerance=False)
    r = transforms[..., :3, :3]
    orthogonal = float(np.abs(r.swapaxes(-2, -1) @ r - np.eye(3)).max())
    determinant = float(np.abs(np.linalg.det(r) - 1).max())
    homogeneous = float(np.abs(transforms[..., 3, :] - [0, 0, 0, 1]).max())
    return dict(finite=True, orthogonality_max_error=orthogonal,
                determinant_max_error=determinant, homogeneous_max_error=homogeneous,
                rigid_within_float32_tolerance=max(orthogonal, determinant, homogeneous) < 1e-5)


def project_world(points, extrinsics, intrinsic, image_size) -> dict:
    points = np.asarray(points, dtype=float).reshape(len(extrinsics), -1, 3)
    camera = np.einsum("tij,tkj->tki", extrinsics[:, :3, :3], points)
    camera += extrinsics[:, None, :3, 3]
    projected = camera @ intrinsic.T
    valid = np.abs(projected[..., 2]) > 1e-12
    uv = np.full(projected.shape[:-1] + (2,), np.nan)
    np.divide(projected[..., :2], projected[..., 2:3], out=uv, where=valid[..., None])
    front = camera[..., 2] > 0
    width, height = image_size
    inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < width)
    inside &= (uv[..., 1] >= 0) & (uv[..., 1] < height)
    return dict(positive_depth_fraction=float(front.mean()),
                front_and_in_image_fraction=float(inside.mean()),
                camera_z_range_m=[float(camera[..., 2].min()), float(camera[..., 2].max())])


def alignment_invariants(joints, objects, camera, transform) -> dict:
    """Check change-of-frame algebra, not accuracy of the original calibration."""
    aligned_objects = transform @ objects
    aligned_joints = joints @ transform[:3, :3].T + transform[:3, 3]
    before = np.linalg.inv(objects[:, 0]) @ objects[:, 1]
    after = np.linalg.inv(aligned_objects[:, 0]) @ aligned_objects[:, 1]
    local_before = np.einsum("toij,thkj->tohki", np.linalg.inv(objects)[..., :3, :3], joints)
    local_before += np.linalg.inv(objects)[:, :, None, None, :3, 3]
    local_after = np.einsum("toij,thkj->tohki", np.linalg.inv(aligned_objects)[..., :3, :3], aligned_joints)
    local_after += np.linalg.inv(aligned_objects)[:, :, None, None, :3, 3]
    aligned_camera = camera @ np.linalg.inv(transform)
    camera_before = camera[:, None] @ objects
    camera_after = aligned_camera[:, None] @ aligned_objects
    return dict(object_relative_transform_max_error=float(np.abs(before - after).max()),
                hand_in_object_frame_max_error_m=float(np.abs(local_before - local_after).max()),
                camera_object_transform_max_error=float(np.abs(camera_before - camera_after).max()))


def support_clearance(vertices, poses, table_height) -> np.ndarray:
    """Exact minimum vertex/triangle height against a horizontal plane."""
    vertices, poses = np.asarray(vertices), np.asarray(poses)
    return np.asarray([(vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() - table_height
                       for pose in poses])


def input_status(counts, rigid_valid, hand_order_valid, camera_front) -> dict:
    gt_counts = [counts[k] for k in ("hands", "tool", "target", "camera")]
    gt_ready = len(set(gt_counts)) == 1 and rigid_valid and hand_order_valid
    media_aligned = len(set(counts.values())) == 1
    return dict(gt_structurally_usable=bool(gt_ready),
                media_frame_counts_match=media_aligned,
                released_camera_front_check_passed=bool(camera_front),
                frame_count_match_is_not_temporal_correspondence_proof=True,
                physical_compatibility="unvalidated",
                task_success="not_evaluated")
