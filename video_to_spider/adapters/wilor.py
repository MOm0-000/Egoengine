"""WiLoR adapter using EgoDex intrinsics and the frozen wilor_raw schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from ..video import FFmpegVideoWriter, VIDEO_ENCODING


def _camera_translation(pred_cam, box_center, box_size, K):
    """Convert crop weak-perspective camera using the episode's real K."""
    import torch

    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def _reflect_rotations(rotations: np.ndarray) -> np.ndarray:
    reflection = np.diag([-1.0, 1.0, 1.0])
    return reflection @ rotations @ reflection


def run(args: argparse.Namespace) -> Path:
    wilor_root = Path(__file__).resolve().parents[2] / "third_party/WiLoR"
    sys.path.insert(0, str(wilor_root))
    import torch
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("WiLoR requested CUDA but no device is visible")
    run_dir = args.run_dir.resolve()
    frame_rows = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))["frames"]
    start, end = args.start_frame, args.end_frame if args.end_frame is not None else len(frame_rows)
    if start < 0 or end <= start or end > len(frame_rows):
        raise ValueError(f"invalid adapter interval [{start}, {end})")
    rows = frame_rows[start:end]
    camera_view = getattr(args, "camera_view", "left")
    if camera_view not in {"left", "right"}:
        raise ValueError("camera_view must be 'left' or 'right'")
    image_key = "rgb_path" if camera_view == "left" else "right_rgb_path"
    if any(image_key not in row for row in rows):
        raise ValueError(f"run has no {camera_view} stereo image for every frame")
    intrinsics_name = "intrinsics.npy" if camera_view == "left" else "intrinsics_right.npy"
    K = np.load(run_dir / "calibration" / intrinsics_name).astype(np.float64)
    default_output = run_dir / ("hands" if camera_view == "left" else "hands_right")
    output_dir = args.output_dir.resolve() if args.output_dir else default_output
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"WiLoR output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model_config = args.model_config.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    if args.dry_run:
        metadata_path.write_text(json.dumps({
            "dry_run": True, "checkpoint": str(checkpoint), "detector_checkpoint": str(detector_checkpoint),
            "frame_count": len(rows), "intrinsics": K.tolist(),
        }, indent=2) + "\n")
        return metadata_path
    previous_cwd = Path.cwd()
    os.chdir(wilor_root)
    try:
        model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(model_config))
        detector = YOLO(str(detector_checkpoint))
    finally:
        os.chdir(previous_cwd)
    device = torch.device(args.device)
    model = model.to(device).eval()
    detector = detector.to(device)
    t, h = len(rows), 2  # frozen order: left, right
    identity = np.eye(3, dtype=np.float32)
    side = np.broadcast_to(np.array([0, 1], dtype=np.int8), (t, h)).copy()
    valid = np.zeros((t, h), dtype=bool)
    score = np.zeros((t, h), dtype=np.float32)
    global_orient = np.broadcast_to(identity, (t, h, 3, 3)).copy()
    hand_pose = np.broadcast_to(identity, (t, h, 15, 3, 3)).copy()
    betas = np.zeros((t, h, 10), dtype=np.float32)
    joints = np.zeros((t, h, 21, 3), dtype=np.float32)
    vertices = np.zeros((t, h, 778, 3), dtype=np.float32)
    translation = np.zeros((t, h, 3), dtype=np.float32)
    boxes_by_frame: list[list[list[float]]] = []
    overlay_frames = []
    for frame_offset, row in enumerate(rows):
        image = cv2.imread(str(run_dir / row[image_key]))
        if image is None:
            raise RuntimeError(f"cannot read {row[image_key]}")
        detections = detector(image, conf=args.detection_threshold, verbose=False)[0]
        boxes, is_right, detection_scores = [], [], []
        for detection in detections:
            data = detection.boxes.data.detach().cpu().reshape(-1).numpy()
            boxes.append(data[:4].tolist())
            detection_scores.append(float(data[4]))
            is_right.append(int(data[5]))
        boxes_by_frame.append(boxes)
        if boxes:
            box_array = np.asarray(boxes, dtype=np.float32)
            right_array = np.asarray(is_right, dtype=np.float32)
            dataset = ViTDetDataset(
                model_cfg, image, box_array, right_array, rescale_factor=args.rescale_factor, fp16=False
            )
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))
            batch = recursive_to(batch, device)
            with torch.no_grad():
                prediction = model(batch)
            pred_cam = prediction["pred_cam"].clone()
            pred_cam[:, 1] *= (2 * batch["right"] - 1)
            camera_t = _camera_translation(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), K
            ).detach().cpu().numpy()
            for detection_index in range(len(boxes)):
                hand_index = 1 if is_right[detection_index] else 0
                if valid[frame_offset, hand_index] and score[frame_offset, hand_index] >= detection_scores[detection_index]:
                    continue
                root_rotation = prediction["pred_mano_params"]["global_orient"][detection_index, 0].detach().cpu().numpy()
                pose_rotation = prediction["pred_mano_params"]["hand_pose"][detection_index].detach().cpu().numpy()
                vertex = prediction["pred_vertices"][detection_index].detach().cpu().numpy()
                joint = prediction["pred_keypoints_3d"][detection_index].detach().cpu().numpy()
                if not is_right[detection_index]:
                    root_rotation = _reflect_rotations(root_rotation)
                    pose_rotation = _reflect_rotations(pose_rotation)
                    vertex[:, 0] *= -1
                    joint[:, 0] *= -1
                valid[frame_offset, hand_index] = True
                score[frame_offset, hand_index] = detection_scores[detection_index]
                global_orient[frame_offset, hand_index] = root_rotation
                hand_pose[frame_offset, hand_index] = pose_rotation
                betas[frame_offset, hand_index] = prediction["pred_mano_params"]["betas"][detection_index].detach().cpu().numpy()
                vertices[frame_offset, hand_index] = vertex
                joints[frame_offset, hand_index] = joint
                translation[frame_offset, hand_index] = camera_t[detection_index]
        overlay = image.copy()
        for hand_index, color in ((0, (255, 120, 0)), (1, (0, 220, 255))):
            if not valid[frame_offset, hand_index]:
                continue
            points_camera = joints[frame_offset, hand_index] + translation[frame_offset, hand_index]
            projected = (K @ points_camera.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            for point in projected:
                if np.all(np.isfinite(point)):
                    cv2.circle(overlay, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
        overlay_frames.append(overlay)
    artifact_path = output_dir / "wilor_raw.npz"
    np.savez_compressed(
        artifact_path,
        frame_indices=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64),
        timestamps_s=np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64),
        side=side, valid=valid, score=score, mano_global_orient=global_orient,
        mano_hand_pose=hand_pose, mano_betas=betas,
        joints_camera_rootrel=joints, vertices_camera_rootrel=vertices,
        translation_camera=translation,
    )
    overlay_path = output_dir / "wilor_overlay.mp4"
    source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
    height, width = overlay_frames[0].shape[:2]
    writer = FFmpegVideoWriter(
        overlay_path, float(source["video"]["fps"]), (width, height), overwrite=True,
    )
    try:
        for frame in overlay_frames:
            writer.write(frame)
    finally:
        writer.release()
    metadata = {
        "schema_version": "1.0", "model": "WiLoR", "checkpoint": str(checkpoint),
        "detector_checkpoint": str(detector_checkpoint), "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "camera_view": camera_view,
        "image_field": image_key,
        "camera_intrinsics_source": f"calibration/{intrinsics_name}",
        "camera_translation_conversion": "crop weak-perspective to full OpenCV camera using EgoDex fx/fy/cx/cy",
        "video_encoding": VIDEO_ENCODING,
        "hand_order": ["left", "right"], "frame_count": t,
        "valid_rate_left": float(np.mean(valid[:, 0])), "valid_rate_right": float(np.mean(valid[:, 1])),
        "shared_beta_initial": {
            side_name: np.median(betas[valid[:, hand_index] & (score[:, hand_index] >= 0.5), hand_index], axis=0).tolist()
            if np.any(valid[:, hand_index] & (score[:, hand_index] >= 0.5)) else None
            for hand_index, side_name in enumerate(("left", "right"))
        },
        "detections_per_frame": [len(item) for item in boxes_by_frame],
        "outputs": ["wilor_raw.npz", "wilor_overlay.mp4"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--camera-view", choices=["left", "right"], default="left")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    print(run(build_parser().parse_args()))
