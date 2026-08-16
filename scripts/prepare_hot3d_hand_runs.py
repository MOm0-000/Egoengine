#!/usr/bin/env python3
"""Prepare a fixed HOT3D-Clips subset as standard pinhole run directories.

This script only prepares data for the hand bottleneck diagnosis. It keeps the
official HOT3D GT in a separate ``hot3d_gt`` folder and never copies it into the
standard inference inputs. The SLAM left/right fisheye streams are undistorted to
pinhole cameras with the same sensor size, focal length, and principal point as
the source camera. This is a controlled benchmark-only conversion; it is not a
claim that the production ADT rectification is identical to HOT3D.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image


HOT3D_CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
OUTPUT_ROOT = Path("/data_all/zzx/egoengine/video_to_spider/runs/hot3d_hand_diagnosis")
FRAME_COUNT = 30
FPS = 30.0
LEFT_STREAM = "1201-1"
RIGHT_STREAM = "1201-2"


SUBSET: list[dict[str, Any]] = [
    {"clip": "clip-001852.tar", "side": "left", "object_id": 29},
    {"clip": "clip-001856.tar", "side": "left", "object_id": 24},
    {"clip": "clip-001891.tar", "side": "right", "object_id": 4},
    {"clip": "clip-002273.tar", "side": "right", "object_id": 22},
    {"clip": "clip-002316.tar", "side": "right", "object_id": 29},
    {"clip": "clip-002850.tar", "side": "right", "object_id": 17},
    {"clip": "clip-002921.tar", "side": "right", "object_id": 19},
    {"clip": "clip-002974.tar", "side": "left", "object_id": 10},
]


def _se3_from_dict(payload: dict[str, Any]) -> np.ndarray:
    q = np.asarray(payload["quaternion_wxyz"], dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    rotation = Rotation.from_quat(np.asarray([x, y, z, w])).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(payload["translation_xyz"], dtype=np.float64)
    return transform


def _pinhole_from_camera(camera_model: camera.CameraModel) -> camera.PinholePlaneCameraModel:
    return camera.PinholePlaneCameraModel(
        width=camera_model.width,
        height=camera_model.height,
        f=[camera_model.f[0], camera_model.f[1]],
        c=camera_model.c,
        distort_coeffs=[],
        T_world_from_eye=camera_model.T_world_from_eye.copy(),
    )


def _intrinsics_from_camera(camera_model: camera.CameraModel) -> np.ndarray:
    return np.asarray(
        [
            [float(camera_model.f[0]), 0.0, float(camera_model.c[0])],
            [0.0, float(camera_model.f[1]), float(camera_model.c[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _frame_numbers(tar) -> list[int]:
    numbers = sorted(
        {
            int(name.split(".image_")[0])
            for name in tar.getnames()
            if name.endswith(".image_1201-1.jpg")
        }
    )
    if len(numbers) < FRAME_COUNT:
        raise RuntimeError(f"clip has only {len(numbers)} frames")
    return numbers[:FRAME_COUNT]


def _load_camera(tar, frame_number: int) -> dict[str, camera.CameraModel]:
    payload = json.load(tar.extractfile(f"{frame_number:06d}.cameras.json"))
    return {
        stream_id: camera.from_json(raw)
        for stream_id, raw in payload.items()
    }


def _warp_frame(tar, frame_number: int, stream_id: str, camera_model: camera.CameraModel) -> np.ndarray:
    raw = imageio.imread(tar.extractfile(f"{frame_number:06d}.image_{stream_id}.jpg"))
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=-1)
    pinhole = _pinhole_from_camera(camera_model)
    warped = warp_image(
        src_camera=camera_model,
        dst_camera=pinhole,
        src_image=raw,
        interpolation=cv2.INTER_LINEAR,
        depth_check=True,
    )
    if warped.ndim == 2:
        warped = np.stack([warped, warped, warped], axis=-1)
    return warped.astype(np.uint8)


def _collect_gt_hand(
    tar,
    frame_numbers: list[int],
    selected_side: str,
) -> dict[str, Any]:
    shape_raw = json.load(tar.extractfile("__hand_shapes.json__"))
    mano_beta = np.asarray(shape_raw["mano"], dtype=np.float32)
    count = len(frame_numbers)
    hand_side = 0 if selected_side == "left" else 1
    mano_theta = np.zeros((count, 15), dtype=np.float32)
    wrist_xform = np.zeros((count, 6), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    boxes = {}
    visibilities = {}
    for offset, frame_number in enumerate(frame_numbers):
        raw = json.load(tar.extractfile(f"{frame_number:06d}.hands.json"))
        hand = raw.get(selected_side)
        if not hand:
            continue
        mano = hand.get("mano_pose")
        if not mano:
            continue
        mano_theta[offset] = np.asarray(mano["thetas"], dtype=np.float32)
        wrist_xform[offset] = np.asarray(mano["wrist_xform"], dtype=np.float32)
        valid[offset] = True
        boxes[str(frame_number)] = hand.get("boxes_amodal", {})
        visibilities[str(frame_number)] = {
            "modeled": hand.get("visibilities_modeled", {}),
            "predicted": hand.get("visibilities_predicted", {}),
        }
    return {
        "frame_numbers": np.asarray(frame_numbers, dtype=np.int64),
        "hand_side": np.asarray([hand_side], dtype=np.int8),
        "mano_beta": mano_beta,
        "mano_theta": mano_theta,
        "wrist_xform": wrist_xform,
        "valid": valid,
        "boxes_amodal": boxes,
        "visibilities": visibilities,
    }


def _collect_gt_objects(
    tar,
    frame_numbers: list[int],
    selected_object_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_number in frame_numbers:
        raw = json.load(tar.extractfile(f"{frame_number:06d}.objects.json"))
        selected = None
        for instances in raw.values():
            for instance in instances:
                if int(instance.get("object_bop_id", -1)) == int(selected_object_id):
                    selected = instance
                    break
            if selected is not None:
                break
        rows.append(
            {
                "frame_number": int(frame_number),
                "object_bop_id": int(selected_object_id),
                "object_pose": selected,
            }
        )
    return rows


def prepare_clip(
    tar_path: Path,
    output_dir: Path,
    *,
    selected_side: str,
    object_id: int,
    overwrite: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        frame_numbers = _frame_numbers(tar)
        cameras_by_frame = {
            frame_number: _load_camera(tar, frame_number)
            for frame_number in frame_numbers
        }

        (output_dir / "frames/rgb").mkdir(parents=True, exist_ok=True)
        (output_dir / "frames/right").mkdir(parents=True, exist_ok=True)
        (output_dir / "calibration").mkdir(parents=True, exist_ok=True)
        (output_dir / "input").mkdir(parents=True, exist_ok=True)
        (output_dir / "hot3d_gt").mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        world_left: list[np.ndarray] = []
        world_right: list[np.ndarray] = []
        for offset, frame_number in enumerate(frame_numbers):
            cameras = cameras_by_frame[frame_number]
            left = cameras[LEFT_STREAM]
            right = cameras[RIGHT_STREAM]
            left_image = _warp_frame(tar, frame_number, LEFT_STREAM, left)
            right_image = _warp_frame(tar, frame_number, RIGHT_STREAM, right)
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
            world_left.append(_se3_from_dict(
                json.loads(
                    tar.extractfile(f"{frame_number:06d}.cameras.json").read()
                )[LEFT_STREAM]["T_world_from_camera"]
            ))
            world_right.append(_se3_from_dict(
                json.loads(
                    tar.extractfile(f"{frame_number:06d}.cameras.json").read()
                )[RIGHT_STREAM]["T_world_from_camera"]
            ))

        left_camera = cameras_by_frame[frame_numbers[0]][LEFT_STREAM]
        right_camera = cameras_by_frame[frame_numbers[0]][RIGHT_STREAM]
        np.save(output_dir / "calibration/intrinsics.npy", _intrinsics_from_camera(left_camera))
        np.save(output_dir / "calibration/intrinsics_right.npy", _intrinsics_from_camera(right_camera))
        np.save(output_dir / "calibration/T_world_camera.npy", np.stack(world_left))
        np.save(output_dir / "calibration/T_world_camera_right.npy", np.stack(world_right))

        left_pose = _se3_from_dict(
            json.loads(
                tar.extractfile(f"{frame_numbers[0]:06d}.cameras.json").read()
            )[LEFT_STREAM]["T_world_from_camera"]
        )
        right_pose = _se3_from_dict(
            json.loads(
                tar.extractfile(f"{frame_numbers[0]:06d}.cameras.json").read()
            )[RIGHT_STREAM]["T_world_from_camera"]
        )
        relative_pose = np.linalg.inv(left_pose) @ right_pose
        baseline_m = float(np.linalg.norm(relative_pose[:3, 3]))
        stereo = {
            "schema_version": "1.0",
            "source_type": "hot3d_clips_slam_pair_pinhole_benchmark",
            "reference_camera": "left",
            "left_right_order": "physical left then physical right",
            "baseline_m": baseline_m,
            "T_left_right_semantics": "column-vector transform from right pinhole camera to left pinhole camera",
            "T_left_right": relative_pose.tolist(),
            "accepted": False,
            "note": "HOT3D benchmark pair is not rectified to a shared pinhole plane; do not use production stereo gate",
        }
        (output_dir / "calibration/stereo.json").write_text(
            json.dumps(stereo, indent=2) + "\n", encoding="utf-8"
        )

        source = {
            "schema_version": "1.0",
            "source_type": "hot3d_clips_slam_pair_pinhole_benchmark",
            "task_directory": "hot3d_clips",
            "episode_id": tar_path.name,
            "instruction_text": "hand bottleneck diagnostic clip",
            "instruction_source": "fixed_hot3d_subset",
            "object_keyword_candidates": [],
            "video": {
                "frame_count": len(rows),
                "fps": FPS,
                "width": left_camera.width,
                "height": left_camera.height,
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

        hand_gt = _collect_gt_hand(tar, frame_numbers, selected_side)
        np.savez_compressed(
            output_dir / "hot3d_gt/hand_pose.npz",
            frame_numbers=hand_gt["frame_numbers"],
            hand_side=hand_gt["hand_side"],
            mano_beta=hand_gt["mano_beta"],
            mano_theta=hand_gt["mano_theta"],
            wrist_xform=hand_gt["wrist_xform"],
            valid=hand_gt["valid"],
        )
        (output_dir / "hot3d_gt/hand_boxes_visibility.json").write_text(
            json.dumps(
                {
                    "selected_side": selected_side,
                    "boxes_amodal": hand_gt["boxes_amodal"],
                    "visibilities": hand_gt["visibilities"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        objects_gt = _collect_gt_objects(tar, frame_numbers, object_id)
        (output_dir / "hot3d_gt/objects.json").write_text(
            json.dumps(
                {
                    "selected_object_id": int(object_id),
                    "frames": objects_gt,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "hot3d_gt/metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source_tar": str(tar_path),
                    "frame_count": len(frame_numbers),
                    "frame_numbers": frame_numbers,
                    "selected_side": selected_side,
                    "selected_object_id": int(object_id),
                    "pinhole_conversion": "same-size pinhole with source focal length/principal point; benchmark only",
                    "gt_in_inference_inputs": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-root", type=Path, default=HOT3D_CLIPS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest: list[dict[str, Any]] = []
    for item in SUBSET:
        clip_name = item["clip"]
        side = item["side"]
        run_name = clip_name.replace(".tar", "") + f"_{side}"
        output_dir = args.output_root / run_name
        tar_path = args.clips_root / clip_name
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        print(f"preparing {run_name}")
        prepare_clip(
            tar_path,
            output_dir,
            selected_side=side,
            object_id=int(item["object_id"]),
            overwrite=args.overwrite,
        )
        manifest.append(
            {
                "clip": clip_name,
                "selected_side": side,
                "selected_object_id": int(item["object_id"]),
                "run_dir": str(output_dir),
            }
        )
    manifest_path = args.output_root / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "items": manifest,
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
