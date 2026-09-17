"""Derive an auditable TACO camera-calibration proxy from MANO/WiLoR correspondences."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_to_spider.schemas import SCHEMA_VERSION, validate_transforms

from ..artifacts import artifact_record
from .metrics import summary


PROXY_LABEL = "derived_calibration_proxy"


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.asarray(points, dtype=np.float64) @ np.asarray(K, dtype=np.float64).T
    return projected[..., :2] / projected[..., 2:3]


def derive_taco_camera_calibration_proxy(
    ground_truth_manifest: str | Path, wilor_artifact: str | Path,
    intrinsics_path: str | Path, output_dir: str | Path, *, overwrite: bool = False,
) -> Path:
    """Fit per-frame world-to-camera SE(3) from GT MANO3D to WiLoR image points.

    This repair is used only for a released TACO episode whose nominal camera
    transform projects the complete scene behind the camera. Because WiLoR is
    involved, the result is a derived calibration proxy and not independent GT.
    """
    source_path = Path(ground_truth_manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("dataset") != "TACO V1" or source.get("scope") != "evaluation_only":
        raise ValueError("camera proxy requires an evaluation-only TACO bundle")
    hand_path = Path(source["artifacts"]["hand"]["path"]).resolve()
    trajectory_path = Path(source["artifacts"]["object_trajectory"]["path"]).resolve()
    wilor_path = Path(wilor_artifact).resolve()
    K_path = Path(intrinsics_path).resolve()
    hand, trajectory, wilor = _npz(hand_path), _npz(trajectory_path), _npz(wilor_path)
    K = np.asarray(np.load(K_path, allow_pickle=False), dtype=np.float64)
    frames = np.asarray(hand["frame_indices"], dtype=np.int64)
    if not (
        np.array_equal(frames, trajectory["frame_indices"])
        and np.array_equal(frames, wilor["frame_indices"])
    ):
        raise ValueError("camera proxy inputs must share an exact frame timeline")
    world_joints = np.asarray(hand["T_world_joint"][..., :3, 3], dtype=np.float64)
    predicted_camera = (
        np.asarray(wilor["joints_camera_rootrel"], dtype=np.float64)
        + np.asarray(wilor["translation_camera"], dtype=np.float64)[:, :, None]
    )
    image_joints = _project(K, predicted_camera)
    valid_hands = np.asarray(wilor["valid"], dtype=bool)
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(frames), axis=0)
    reprojection_median = np.full(len(frames), np.nan, dtype=np.float64)
    inlier_count = np.zeros(len(frames), dtype=np.int64)
    successful = np.zeros(len(frames), dtype=bool)
    for index in range(len(frames)):
        use = np.repeat(valid_hands[index, :, None], 21, axis=1).reshape(-1)
        object_points = world_joints[index].reshape(-1, 3)[use]
        image_points = image_joints[index].reshape(-1, 2)[use]
        finite = np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
        object_points, image_points = object_points[finite], image_points[finite]
        if len(object_points) < 6:
            continue
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, None, flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=200, reprojectionError=15.0, confidence=0.999,
        )
        if not ok or inliers is None or len(inliers) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, None, rvec, tvec, True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        rotation = cv2.Rodrigues(rvec)[0]
        transforms[index, :3, :3] = rotation
        transforms[index, :3, 3] = tvec[:, 0]
        reprojection, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
        error = np.linalg.norm(reprojection[:, 0] - image_points, axis=1)
        reprojection_median[index] = float(np.median(error))
        inlier_count[index] = len(inliers)
        successful[index] = True
    if not successful.all():
        missing = np.flatnonzero(~successful)
        raise RuntimeError(f"camera proxy PnP failed on frames: {missing[:10].tolist()}")
    validate_transforms("derived T_camera_world", transforms)
    T_world_camera = np.linalg.inv(transforms)
    T_world_object = np.asarray(trajectory["T_world_object"], dtype=np.float64)
    T_camera_object = np.einsum("tij,tjk->tik", transforms, T_world_object)
    validate_transforms("derived T_camera_object", T_camera_object)

    output = Path(output_dir).resolve()
    destination = output / "ground_truth_manifest.json"
    if output.exists() and overwrite:
        shutil.rmtree(output)
    elif destination.exists():
        return destination
    output.mkdir(parents=True, exist_ok=True)
    camera_path = output / "camera_derived_calibration_proxy.npz"
    np.savez_compressed(
        camera_path, frame_indices=frames, timestamps_s=hand["timestamps_s"],
        intrinsics=K, T_world_camera=T_world_camera,
        raw_T_camera_world=transforms,
        reprojection_median_px=reprojection_median, pnp_inlier_count=inlier_count,
        quality_label=np.asarray(PROXY_LABEL),
    )
    object_path = output / "object_trajectory_derived_calibration_proxy.npz"
    np.savez_compressed(
        object_path, **{
            **trajectory,
            "T_camera_object": T_camera_object,
            "camera_quality_label": np.asarray(PROXY_LABEL),
        },
    )
    artifacts = dict(source["artifacts"])
    artifacts["camera"] = artifact_record(camera_path)
    artifacts["object_trajectory"] = artifact_record(object_path)
    manifest: dict[str, Any] = {
        **source,
        "parent_ground_truth_manifest": artifact_record(source_path),
        "artifacts": artifacts,
        "quality_labels": {
            **source.get("quality_labels", {}),
            "camera": PROXY_LABEL,
            "object_trajectory": "dataset_GT_object_pose_transformed_by_derived_calibration_proxy",
        },
        "camera_calibration_proxy": {
            "quality_label": PROXY_LABEL,
            "independent_ground_truth": False,
            "method": "per_frame_PnP_GT_MANO3D_to_WiLoR_image_projection",
            "reason": "released egocentric extrinsic places all MANO/object points behind camera",
            "reprojection_median_px": summary(reprojection_median),
            "pnp_inlier_count": summary(inlier_count),
            "inputs": {
                "wilor": artifact_record(wilor_path),
                "intrinsics": artifact_record(K_path),
                "hand_ground_truth": artifact_record(hand_path),
            },
        },
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def install_taco_camera_calibration_proxy(
    run_dir: str | Path, proxy_manifest: str | Path,
) -> Path:
    """Replace only a branched run's calibration link with the proxy camera."""
    root = Path(run_dir).resolve()
    manifest_path = Path(proxy_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("camera_calibration_proxy", {}).get("quality_label") != PROXY_LABEL:
        raise ValueError("manifest does not contain a derived TACO camera proxy")
    source_camera = Path(manifest["artifacts"]["camera"]["path"]).resolve()
    with np.load(source_camera, allow_pickle=False) as artifact:
        K = np.asarray(artifact["intrinsics"], dtype=np.float64)
        T_world_camera = np.asarray(artifact["T_world_camera"], dtype=np.float64)
    calibration = root / "calibration"
    if calibration.is_symlink():
        calibration.unlink()
    calibration.mkdir(parents=True, exist_ok=True)
    np.save(calibration / "intrinsics.npy", K.astype(np.float32))
    np.save(calibration / "T_world_camera.npy", T_world_camera.astype(np.float32))
    report = calibration / "derived_calibration_proxy.json"
    report.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "quality_label": PROXY_LABEL,
        "independent_ground_truth": False,
        "source_manifest": artifact_record(manifest_path),
    }, indent=2) + "\n", encoding="utf-8")
    return report
