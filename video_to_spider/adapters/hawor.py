"""Isolated HaWoR adapter using the run's calibrated camera trajectory.

The official HaWoR demo estimates camera motion with DROID-SLAM and metric
scale with Metric3D.  EgoDex runs already contain calibrated intrinsics and an
Aria camera trajectory, so this adapter intentionally runs only HaWoR's hand
motion network and keeps those existing camera observations.  This makes the
comparison against the current per-frame WiLoR adapter controlled and avoids a
second camera/depth system changing several variables at once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


HAND_ORDER = ("left", "right")


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def interpolate_detection_boxes(
    boxes: np.ndarray, detected: np.ndarray, image_size: tuple[int, int],
) -> np.ndarray:
    """Fill missed detections without inventing a hand track from one frame."""
    values = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(detected, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 4 or valid.shape != (len(values),):
        raise ValueError("boxes/detected have incompatible shapes")
    indices = np.flatnonzero(valid & np.isfinite(values).all(axis=1))
    if indices.size < 2:
        raise ValueError("HaWoR requires at least two detected frames for a hand")
    timeline = np.arange(len(values))
    result = values.copy()
    for coordinate in range(4):
        result[:, coordinate] = np.interp(
            timeline, indices, values[indices, coordinate],
        )
    width, height = image_size
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width - 1)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height - 1)
    if np.any(result[:, 2] - result[:, 0] < 4) or np.any(
        result[:, 3] - result[:, 1] < 4
    ):
        raise ValueError("interpolated HaWoR boxes contain a degenerate crop")
    return result.astype(np.float32)


def _frame_paths(root: Path) -> list[Path]:
    rows = json.loads(
        (root / "frames/frame_index.json").read_text(encoding="utf-8")
    )["frames"]
    paths = [(root / row["rgb_path"]).resolve() for row in rows]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing RGB frame: {missing[0]}")
    return paths


def _detect_right_hand(
    frame_paths: list[Path], detector_checkpoint: Path, device: str,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ultralytics import YOLO

    detector = YOLO(str(detector_checkpoint)).to(device)
    boxes = np.full((len(frame_paths), 4), np.nan, dtype=np.float64)
    scores = np.zeros(len(frame_paths), dtype=np.float64)
    detected = np.zeros(len(frame_paths), dtype=bool)
    for frame, path in enumerate(frame_paths):
        result = detector(
            cv2.imread(str(path)), conf=confidence_threshold, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        candidates: list[tuple[float, np.ndarray]] = []
        for item in result.boxes:
            side = int(item.cls.detach().cpu().reshape(-1)[0])
            if side != 1:
                continue
            score = float(item.conf.detach().cpu().reshape(-1)[0])
            box = item.xyxy.detach().cpu().reshape(-1, 4)[0].numpy()
            candidates.append((score, box))
        if candidates:
            score, box = max(candidates, key=lambda item: item[0])
            boxes[frame] = box
            scores[frame] = score
            detected[frame] = True
    return boxes, scores, detected


def _load_hawor(checkpoint: Path) -> tuple[Any, Any]:
    import torch
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR

    model_config = checkpoint.parent.parent / "model_config.yaml"
    config = get_config(str(model_config), update_cachedir=True)
    if config.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in config.MODEL:
        config.defrost()
        config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()
    model = HAWOR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=config,
    )
    return model, torch


def run_hawor(
    run_dir: str | Path, *, hawor_root: str | Path,
    checkpoint: str | Path | None = None,
    detector_checkpoint: str | Path | None = None,
    gpu: int = 0, confidence_threshold: float = 0.20,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Run a right-hand HaWoR ablation without consulting hand/object GT."""
    root = Path(run_dir).resolve()
    external = Path(hawor_root).resolve()
    checkpoint_path = Path(
        checkpoint or external / "weights/hawor/checkpoints/hawor.ckpt"
    ).resolve()
    detector_path = Path(
        detector_checkpoint or external / "weights/external/detector.pt"
    ).resolve()
    for required in (checkpoint_path, detector_path, external / "_DATA/data/mano/MANO_RIGHT.pkl"):
        if not required.exists():
            raise FileNotFoundError(required)
    if checkpoint_path.stat().st_size < 3_000_000_000:
        raise ValueError(
            f"HaWoR checkpoint is incomplete ({checkpoint_path.stat().st_size} bytes): "
            f"{checkpoint_path}"
        )
    aria2_state = checkpoint_path.with_suffix(checkpoint_path.suffix + ".aria2")
    if aria2_state.exists():
        raise ValueError(
            f"HaWoR checkpoint still has an incomplete-download state: {aria2_state}"
        )
    output_path = root / "hands/hawor_raw.npz"
    metrics_path = root / "hands/hawor_metrics.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass overwrite=True")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in (0, 1]")

    frame_paths = _frame_paths(root)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"failed to read {frame_paths[0]}")
    height, width = first.shape[:2]
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    frame_rows = json.loads(
        (root / "frames/frame_index.json").read_text(encoding="utf-8")
    )["frames"]
    frame_indices = np.asarray(
        [int(row["source_frame_index"]) for row in frame_rows], dtype=np.int64,
    )
    source = json.loads((root / "input/source.json").read_text(encoding="utf-8"))
    fps = float(source["video"]["fps"])
    timestamps = np.arange(len(frame_paths), dtype=np.float64) / fps

    device = f"cuda:{gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # CUDA_VISIBLE_DEVICES remaps the selected physical device to local cuda:0.
    device = "cuda:0"
    boxes_raw, scores, detected = _detect_right_hand(
        frame_paths, detector_path, device, confidence_threshold,
    )
    boxes = interpolate_detection_boxes(
        boxes_raw, detected, (width, height),
    )
    detected_indices = np.flatnonzero(detected)
    track_start = int(detected_indices[0])
    track_end = int(detected_indices[-1]) + 1
    track_scores = np.interp(
        np.arange(track_start, track_end), detected_indices, scores[detected_indices],
    )

    external_string = str(external)
    if external_string not in sys.path:
        sys.path.insert(0, external_string)
    with _working_directory(external):
        model, torch = _load_hawor(checkpoint_path)
        model = model.to(device).eval()
        results = model.inference(
            np.asarray([str(path) for path in frame_paths[track_start:track_end]]),
            boxes[track_start:track_end],
            img_focal=float((K[0, 0] + K[1, 1]) * 0.5),
            img_center=[float(K[0, 2]), float(K[1, 2])],
            device=device, do_flip=False,
        )
        mano_input = {
            "pred_rotmat": results["pred_rotmat"].to(device),
            "pred_shape": results["pred_shape"].to(device),
        }
        with torch.no_grad():
            mano = model.mano.query(mano_input)
        joints_rootrel = mano.joints.detach().cpu().numpy()
        vertices_rootrel = mano.vertices.detach().cpu().numpy()
    translation = results["pred_trans"].detach().cpu().numpy()
    if translation.ndim == 3 and translation.shape[1] == 1:
        translation = translation[:, 0]
    rotations = results["pred_rotmat"].detach().cpu().numpy()
    betas = results["pred_shape"].detach().cpu().numpy()
    if (
        joints_rootrel.shape[1:] != (21, 3)
        or rotations.shape[1:] != (16, 3, 3)
        or translation.shape[1:] != (3,)
    ):
        raise ValueError(
            "unexpected HaWoR output shapes: "
            f"joints={joints_rootrel.shape}, rotations={rotations.shape}, "
            f"translation={translation.shape}"
        )

    count = len(frame_paths)
    identities = np.broadcast_to(np.eye(3), (count, 2, 3, 3)).copy()
    hand_pose = np.broadcast_to(np.eye(3), (count, 2, 15, 3, 3)).copy()
    valid = np.zeros((count, 2), dtype=bool)
    confidence = np.zeros((count, 2), dtype=np.float32)
    joints = np.zeros((count, 2, 21, 3), dtype=np.float32)
    vertices = np.zeros((count, 2, vertices_rootrel.shape[1], 3), dtype=np.float32)
    translations = np.zeros((count, 2, 3), dtype=np.float32)
    shapes = np.zeros((count, 2, 10), dtype=np.float32)
    track_slice = slice(track_start, track_end)
    valid[track_slice, 1] = True
    confidence[track_slice, 1] = track_scores.astype(np.float32)
    joints[track_slice, 1] = joints_rootrel.astype(np.float32)
    vertices[track_slice, 1] = vertices_rootrel.astype(np.float32)
    translations[track_slice, 1] = translation.astype(np.float32)
    identities[track_slice, 1] = rotations[:, 0]
    hand_pose[track_slice, 1] = rotations[:, 1:]
    shapes[track_slice, 1] = betas.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame_indices=frame_indices,
        timestamps_s=timestamps.astype(np.float64),
        side=np.broadcast_to(np.asarray(HAND_ORDER), (count, 2)),
        valid=valid,
        score=confidence,
        mano_global_orient=identities.astype(np.float32),
        mano_hand_pose=hand_pose.astype(np.float32),
        mano_betas=shapes,
        joints_camera_rootrel=joints,
        vertices_camera_rootrel=vertices,
        translation_camera=translations,
    )
    repo_commit = subprocess.run(
        ["git", "-C", str(external), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    wrist_to_middle = np.linalg.norm(
        joints_rootrel[:, 12] - joints_rootrel[:, 0], axis=-1,
    )
    metrics = {
        "schema_version": "1.0",
        "method": "official HaWoR hand network with existing calibrated camera inputs",
        "paper_alignment": (
            "approximates EgoEngine's shared-coordinate Aria 3D hand trajectory; "
            "does not modify MINK targets"
        ),
        "ground_truth_used": False,
        "camera_policy": "existing run intrinsics and T_world_camera; HaWoR DROID-SLAM bypassed",
        "depth_policy": "HaWoR Metric3D bypassed; no contact-based hand rescaling",
        "supported_hands_in_this_ablation": ["right"],
        "detected_frame_count": int(detected.sum()),
        "inferred_track_frame_count": int(track_end - track_start),
        "track_frame_interval_half_open": [track_start, track_end],
        "frame_count": count,
        "detected_rate": float(detected.mean()),
        "detector_confidence_threshold": confidence_threshold,
        "confidence_policy": (
            "threshold defines observed boxes; interpolated track scores are soft weights"
        ),
        "detection_score_median": float(np.median(scores[detected])),
        "wrist_to_middle_tip_median_m": float(np.median(wrist_to_middle)),
        "hawor_commit": repo_commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "detector_checkpoint": str(detector_path),
        "intrinsics": K.tolist(),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return output_path, metrics_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run_hawor(
        args.run_dir, hawor_root=args.hawor_root,
        checkpoint=args.checkpoint,
        detector_checkpoint=args.detector_checkpoint,
        gpu=args.gpu, confidence_threshold=args.confidence_threshold,
        overwrite=args.overwrite,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
