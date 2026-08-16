#!/usr/bin/env python3
"""Generate P1/P2 FOV-preserving rectified virtual-pinhole runs.

P0 is the existing same-size pinhole benchmark. P1 is a wider common pinhole.
P2 is an ultra-wide common pinhole with a larger output resolution.
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image


CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
SOURCE_MANIFEST = (
    REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/fov_rectification_ablation/runs"
)
FRAME_COUNT = 30
FPS = 30.0
LEFT_STREAM = "1201-1"
RIGHT_STREAM = "1201-2"

PROTOCOLS = {
    "P1": {
        "name": "wide_pinhole_640x480_f160_rectified",
        "width": 640,
        "height": 480,
        "f": 160.0,
        "rectified": True,
    },
    "P2": {
        "name": "ultrawide_pinhole_1600x1200_f80_rectified",
        "width": 1600,
        "height": 1200,
        "f": 80.0,
        "rectified": True,
    },
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _rectified_rotation(
    left: camera.CameraModel,
    right: camera.CameraModel,
) -> np.ndarray:
    left_center = left.pos()
    right_center = right.pos()
    x_axis = _normalize(right_center - left_center)
    forward_left = left.orient()[:, 2]
    forward_right = right.orient()[:, 2]
    forward_avg = _normalize(forward_left + forward_right)
    z_axis = _normalize(
        forward_avg - x_axis * float(np.dot(forward_avg, x_axis))
    )
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    # T_world_from_eye is world<-eye: camera axes are columns of R.
    return np.column_stack([x_axis, y_axis, z_axis])


def _pinhole_from_source(
    source: camera.CameraModel,
    protocol: dict[str, Any],
    camera_center: np.ndarray | None = None,
) -> camera.PinholePlaneCameraModel:
    width = int(protocol["width"])
    height = int(protocol["height"])
    focal = float(protocol["f"])
    if protocol.get("rectified"):
        # Rotation is set by the caller once and shared by both views.
        rotation = source.orient().copy()
    else:
        rotation = source.orient().copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = (
        camera_center
        if camera_center is not None
        else source.pos()
    )
    return camera.PinholePlaneCameraModel(
        width=width,
        height=height,
        f=[focal, focal],
        c=(width / 2.0, height / 2.0),
        distort_coeffs=[],
        T_world_from_eye=transform,
    )


def _intrinsics(cam: camera.PinholePlaneCameraModel) -> np.ndarray:
    return np.asarray(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _warp_frame(
    src_image: np.ndarray,
    src_camera: camera.CameraModel,
    dst_camera: camera.PinholePlaneCameraModel,
) -> np.ndarray:
    src = np.asarray(src_image)
    if src.ndim == 2:
        src = np.stack([src, src, src], axis=-1)
    warped = warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=src,
        interpolation=cv2.INTER_LINEAR,
        depth_check=True,
    )
    if warped.ndim == 2:
        warped = np.stack([warped, warped, warped], axis=-1)
    return warped.astype(np.uint8)


def _prepare_run(
    source_entry: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames/rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames/right").mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    source_run = Path(source_entry["run_dir"])
    clip_name = source_entry["clip"]
    tar_path = CLIPS_ROOT / clip_name

    rows: list[dict[str, Any]] = []
    T_left_all: list[np.ndarray] = []
    T_right_all: list[np.ndarray] = []
    with tarfile.open(tar_path, "r") as tar:
        numbers = sorted(
            {
                int(name.split(".image_1201-1.jpg")[0])
                for name in tar.getnames()
                if name.endswith(".image_1201-1.jpg")
            }
        )[:FRAME_COUNT]
        for offset, frame_number in enumerate(numbers):
            cams = json.load(tar.extractfile(f"{frame_number:06d}.cameras.json"))
            left_source = camera.from_json(cams[LEFT_STREAM])
            right_source = camera.from_json(cams[RIGHT_STREAM])
            if protocol.get("rectified"):
                rotation = _rectified_rotation(left_source, right_source)
                left_source_copy = left_source.copy()
                right_source_copy = right_source.copy()
                left_source_copy.T_world_from_eye[:3, :3] = rotation
                right_source_copy.T_world_from_eye[:3, :3] = rotation
                left_dst = _pinhole_from_source(
                    left_source_copy,
                    protocol,
                    camera_center=left_source.pos(),
                )
                right_dst = _pinhole_from_source(
                    right_source_copy,
                    protocol,
                    camera_center=right_source.pos(),
                )
                # Replace the destination rotation with the shared one explicitly.
                left_dst.T_world_from_eye[:3, :3] = rotation
                right_dst.T_world_from_eye[:3, :3] = rotation
            else:
                left_dst = _pinhole_from_source(left_source, protocol)
                right_dst = _pinhole_from_source(right_source, protocol)

            raw_left = imageio.imread(
                tar.extractfile(f"{frame_number:06d}.image_{LEFT_STREAM}.jpg")
            )
            raw_right = imageio.imread(
                tar.extractfile(f"{frame_number:06d}.image_{RIGHT_STREAM}.jpg")
            )
            left_image = _warp_frame(raw_left, left_source, left_dst)
            right_image = _warp_frame(raw_right, right_source, right_dst)
            relative_name = f"{offset:06d}.png"
            cv2.imwrite(str(output_dir / "frames/rgb" / relative_name), left_image)
            cv2.imwrite(str(output_dir / "frames/right" / relative_name), right_image)
            rows.append(
                {
                    "frame_index": offset,
                    "source_frame_index": int(frame_number),
                    "timestamp_s": offset / FPS,
                    "source_timestamp_s": offset / FPS,
                    "rgb_path": f"frames/rgb/{relative_name}",
                    "right_rgb_path": f"frames/right/{relative_name}",
                    "source_relative_name": relative_name,
                }
            )
            T_left_all.append(left_dst.T_world_from_eye.copy())
            T_right_all.append(right_dst.T_world_from_eye.copy())

    np.save(output_dir / "calibration/intrinsics.npy", _intrinsics(left_dst))
    np.save(
        output_dir / "calibration/intrinsics_right.npy",
        _intrinsics(right_dst),
    )
    np.save(
        output_dir / "calibration/T_world_camera.npy",
        np.stack(T_left_all),
    )
    np.save(
        output_dir / "calibration/T_world_camera_right.npy",
        np.stack(T_right_all),
    )
    baseline = float(
        np.linalg.norm(
            np.asarray(right_dst.pos()) - np.asarray(left_dst.pos())
        )
    )
    stereo = {
        "schema_version": "1.0",
        "source_type": "hot3d_clips_fov_rectified_virtual_pinhole",
        "reference_camera": "left",
        "left_right_order": "physical left then physical right",
        "baseline_m": baseline,
        "T_left_right_semantics": (
            "column-vector transform from right pinhole camera to left pinhole camera"
        ),
        "accepted": False,
        "protocol": protocol["name"],
        "note": "FOV-preserving benchmark rectification; not production ADT rectification",
    }
    (output_dir / "calibration/stereo.json").write_text(
        json.dumps(stereo, indent=2) + "\n", encoding="utf-8"
    )
    source = {
        "schema_version": "1.0",
        "source_type": "hot3d_clips_fov_rectified_virtual_pinhole",
        "task_directory": "hot3d_clips",
        "episode_id": clip_name,
        "instruction_text": "hand FOV rectification diagnostic clip",
        "instruction_source": "fixed_hot3d_subset",
        "object_keyword_candidates": [],
        "video": {
            "frame_count": len(rows),
            "fps": FPS,
            "width": int(protocol["width"]),
            "height": int(protocol["height"]),
        },
        "selected_frame_interval": [0, len(rows)],
        "reference_camera": "pinhole_left",
        "right_camera": "pinhole_right",
        "depth_route_policy": "not applicable",
    }
    (output_dir / "input/source.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "frames/frame_index.json").write_text(
        json.dumps(
            {"schema_version": "1.0", "fps": FPS, "frames": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Reuse the frozen GT artifacts from the P0 run.
    (output_dir / "hot3d_gt").mkdir(parents=True, exist_ok=True)
    for name in (
        "hand_pose.npz",
        "hand_boxes_visibility.json",
        "metadata.json",
        "objects.json",
    ):
        shutil.copy2(source_run / "hot3d_gt" / name, output_dir / "hot3d_gt" / name)


def main() -> int:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    manifests: dict[str, list[dict[str, Any]]] = {"P1": [], "P2": []}
    for protocol_name, protocol in PROTOCOLS.items():
        for entry in source_manifest["items"]:
            run_name = Path(entry["run_dir"]).name
            output_dir = OUTPUT_ROOT / protocol_name / run_name
            print(f"preparing {protocol_name} {run_name}", flush=True)
            _prepare_run(entry, protocol, output_dir)
            manifests[protocol_name].append(
                {
                    "clip": entry["clip"],
                    "selected_side": entry["selected_side"],
                    "selected_object_id": entry["selected_object_id"],
                    "run_dir": str(output_dir),
                    "protocol": protocol_name,
                }
            )
        manifest_path = OUTPUT_ROOT / protocol_name / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "frame_count": FRAME_COUNT,
                    "fps": FPS,
                    "protocol": protocol["name"],
                    "items": manifests[protocol_name],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
