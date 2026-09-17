"""SAM2 paper route: point-tracked object and keypoint-prompted hands."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_to_spider.manifest import RunManifest, stage_cache_key
from video_to_spider.schemas import SCHEMA_VERSION


def _frame_rows(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))
    rows = payload["frames"]
    if not rows:
        raise ValueError("SAM2 route requires at least one RGB frame")
    return rows


def project_hand_keypoints(
    run_dir: str | Path, hand_ground_truth: str | Path,
    *, confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    """Project evaluation-only MANO21 joints into the ingested RGB timeline."""
    root = Path(run_dir).resolve()
    rows = _frame_rows(root)
    source_frames = np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64)
    with np.load(hand_ground_truth, allow_pickle=False) as artifact:
        gt_frames = np.asarray(artifact["frame_indices"], dtype=np.int64)
        hand_order = np.asarray(artifact["hand_order"]).astype(str)
        T_world_joint = np.asarray(artifact["T_world_joint"], dtype=np.float64)
        confidence = np.asarray(artifact["confidence"], dtype=np.float64)
    if T_world_joint.ndim != 5 or T_world_joint.shape[2:] != (21, 4, 4):
        raise ValueError(f"hand GT must contain (T,H,21,4,4) transforms, got {T_world_joint.shape}")
    if confidence.shape != T_world_joint.shape[:3]:
        raise ValueError("hand GT confidence shape does not match joint transforms")
    lookup = {int(frame): index for index, frame in enumerate(gt_frames)}
    try:
        selected = np.asarray([lookup[int(frame)] for frame in source_frames], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"hand GT is missing source frame {error.args[0]}") from error
    joints_world = T_world_joint[selected, ..., :3, 3]
    joint_confidence = confidence[selected]
    K = np.load(root / "calibration/intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(root / "calibration/T_world_camera.npy").astype(np.float64)
    if len(T_world_camera) != len(rows):
        raise ValueError("camera calibration does not match RGB timeline")
    T_camera_world = np.linalg.inv(T_world_camera)
    points_camera = np.einsum(
        "tij,thnj->thni", T_camera_world[:, :3, :3], joints_world,
    ) + T_camera_world[:, None, None, :3, 3]
    # Two released TACO 2023-09-26 sequences contain an egocentric camera
    # calibration whose homogeneous camera coordinates have the opposite
    # projective sign.  Multiplying xyz by -1 preserves the image projection
    # while restoring positive optical depth.  This is deliberately recorded
    # as a projective repair, not represented as an SE(3) extrinsic correction.
    robust_z = np.median(points_camera[..., 2], axis=(1, 2))
    camera_coordinate_sign = np.where(robust_z < 0.0, -1.0, 1.0)
    points_camera = points_camera * camera_coordinate_sign[:, None, None, None]
    z = points_camera[..., 2]
    pixels_h = np.einsum("ij,thnj->thni", K, points_camera)
    pixels = pixels_h[..., :2] / np.maximum(pixels_h[..., 2:3], 1e-12)
    first_image = cv2.imread(str(root / rows[0]["rgb_path"]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError("cannot read first RGB frame")
    height, width = first_image.shape[:2]
    valid = (
        np.isfinite(pixels).all(axis=-1)
        & (z > 1e-5)
        & (joint_confidence > confidence_threshold)
        & (pixels[..., 0] >= 0.0) & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0.0) & (pixels[..., 1] < height)
    )
    return {
        "frame_indices": source_frames,
        "hand_order": hand_order,
        "points_xy": pixels.astype(np.float32),
        "valid": valid,
        "confidence": joint_confidence.astype(np.float32),
        "image_size": np.asarray([width, height], dtype=np.int64),
        "camera_coordinate_sign": camera_coordinate_sign.astype(np.int8),
        "projection_repair": np.asarray("homogeneous_camera_xyz_sign_if_negative_depth"),
    }


def _mask_confidence(logits: Any, mask: np.ndarray) -> float:
    values = logits.detach().float().cpu().numpy()
    while values.ndim > 2:
        values = values[0]
    if not mask.any():
        return 0.0
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values[mask], -30.0, 30.0)))
    return float(np.mean(probabilities))


def _track_object(
    predictor: Any, frame_dir: Path, frame_count: int, prompt_frame: int,
    point_xy: tuple[float, float],
) -> dict[str, np.ndarray]:
    state = predictor.init_state(
        video_path=str(frame_dir), offload_video_to_cpu=True,
        offload_state_to_cpu=True, async_loading_frames=False,
    )
    _, object_ids, logits = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=prompt_frame, obj_id=1,
        points=np.asarray([point_xy], dtype=np.float32),
        labels=np.ones(1, dtype=np.int32),
    )
    height = int(state["video_height"])
    width = int(state["video_width"])
    masks = np.zeros((frame_count, height, width), dtype=bool)
    confidence = np.zeros(frame_count, dtype=np.float32)
    object_ids_out = np.full(frame_count, -1, dtype=np.int64)

    def store(frame_index: int, ids: Any, scores: Any) -> None:
        ids_array = np.asarray(ids)
        matches = np.flatnonzero(ids_array == 1)
        if not matches.size:
            return
        at = int(matches[0])
        mask = scores[at].detach().float().cpu().numpy()
        while mask.ndim > 2:
            mask = mask[0]
        selected = mask > 0.0
        masks[frame_index] = selected
        confidence[frame_index] = _mask_confidence(scores[at], selected)
        object_ids_out[frame_index] = 1

    store(prompt_frame, object_ids, logits)
    for frame_index, ids, scores in predictor.propagate_in_video(
        state, start_frame_idx=prompt_frame, max_frame_num_to_track=frame_count - prompt_frame,
        reverse=False,
    ):
        store(int(frame_index), ids, scores)
    if prompt_frame > 0:
        for frame_index, ids, scores in predictor.propagate_in_video(
            state, start_frame_idx=prompt_frame, max_frame_num_to_track=prompt_frame + 1,
            reverse=True,
        ):
            store(int(frame_index), ids, scores)
    valid = masks.reshape(frame_count, -1).any(axis=1)
    confidence[~valid] = 0.0
    return {
        "masks": masks, "valid": valid, "confidence": confidence,
        "object_ids": object_ids_out,
    }


def _segment_hands(
    model: Any, frame_paths: list[Path], prompts: dict[str, Any],
) -> dict[str, np.ndarray]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_predictor = SAM2ImagePredictor(model)
    frame_count = len(frame_paths)
    width, height = [int(value) for value in prompts["image_size"]]
    masks_out = np.zeros((frame_count, height, width), dtype=bool)
    confidence_out = np.zeros(frame_count, dtype=np.float32)
    object_ids = np.full(frame_count, -1, dtype=np.int64)
    for frame_index, frame_path in enumerate(frame_paths):
        image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"cannot read SAM2 hand frame: {frame_path}")
        image_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        frame_scores = []
        for hand_index in range(len(prompts["hand_order"])):
            valid = prompts["valid"][frame_index, hand_index]
            points = prompts["points_xy"][frame_index, hand_index, valid]
            if len(points) < 2:
                continue
            masks, scores, _ = image_predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points), dtype=np.int32),
                multimask_output=True,
            )
            selected = int(np.argmax(scores))
            masks_out[frame_index] |= np.asarray(masks[selected], dtype=bool)
            frame_scores.append(float(scores[selected]))
        if frame_scores and masks_out[frame_index].any():
            confidence_out[frame_index] = float(np.mean(frame_scores))
            object_ids[frame_index] = 1
    valid_out = masks_out.reshape(frame_count, -1).any(axis=1)
    return {
        "masks": masks_out, "valid": valid_out,
        "confidence": confidence_out, "object_ids": object_ids,
    }


def _save_mask_artifact(
    path: Path, rows: list[dict[str, Any]], result: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        frame_indices=np.asarray([row["source_frame_index"] for row in rows], dtype=np.int64),
        timestamps_s=np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64),
        masks=result["masks"].astype(np.uint8),
        valid=result["valid"], confidence=result["confidence"],
        object_ids=result["object_ids"],
    )


def _metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    masks = np.asarray(result["masks"], dtype=bool)
    valid = np.asarray(result["valid"], dtype=bool)
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    return {
        "valid_rate": float(np.mean(valid)),
        "mean_confidence": float(np.mean(result["confidence"][valid])) if valid.any() else 0.0,
        "mean_area_ratio": float(np.mean(areas[valid])) if valid.any() else 0.0,
    }


def _write_overlay(
    frame_paths: list[Path], objects: np.ndarray, hands: np.ndarray,
    output: Path, fps: float,
) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("cannot render SAM2 overlay")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open overlay writer: {output}")
    try:
        for path, object_mask, hand_mask in zip(frame_paths, objects, hands):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            overlay = frame.copy()
            overlay[object_mask] = (40, 210, 50)
            overlay[hand_mask] = (40, 90, 235)
            writer.write(cv2.addWeighted(frame, 0.58, overlay, 0.42, 0.0))
    finally:
        writer.release()


def run_sam2_paper_masks(
    run_dir: str | Path, hand_ground_truth: str | Path, checkpoint: str | Path,
    object_point_xy: tuple[float, float], *,
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    prompt_frame: int = 0, output_dir: str | Path | None = None,
    confidence_threshold: float = 0.0, overwrite: bool = False,
    dry_run: bool = False, write_overlay: bool = True,
) -> Path:
    import torch

    if not torch.cuda.is_available() and not dry_run:
        raise RuntimeError("SAM2 paper route requires a visible CUDA device")
    root = Path(run_dir).resolve()
    gt_path = Path(hand_ground_truth).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    rows = _frame_rows(root)
    if prompt_frame < 0 or prompt_frame >= len(rows):
        raise ValueError(f"invalid object prompt frame: {prompt_frame}")
    frame_paths = [root / row["rgb_path"] for row in rows]
    first = cv2.imread(str(frame_paths[prompt_frame]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("cannot read object prompt frame")
    height, width = first.shape[:2]
    point_x, point_y = [float(value) for value in object_point_xy]
    if not (0.0 <= point_x < width and 0.0 <= point_y < height):
        raise ValueError(f"object prompt {object_point_xy} lies outside {width}x{height}")
    output = Path(output_dir).resolve() if output_dir else root / "segmentation"
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"SAM2 output exists: {metadata_path}")
    output.mkdir(parents=True, exist_ok=True)
    hand_prompts = project_hand_keypoints(
        root, gt_path, confidence_threshold=confidence_threshold,
    )
    prompt_report = output / "sam2_prompts.npz"
    np.savez_compressed(prompt_report, **hand_prompts)
    if dry_run:
        dry_path = output / "sam2_dry_run.json"
        dry_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(checkpoint_path),
            "model_config": model_config,
            "object_prompt_frame": prompt_frame,
            "object_point_xy": [point_x, point_y],
            "hand_prompt_counts": hand_prompts["valid"].sum(axis=2).tolist(),
            "uses_ground_truth": True,
            "ground_truth_role": "oracle_hand_keypoint_prompts",
        }, indent=2) + "\n", encoding="utf-8")
        return dry_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    source = json.loads((root / "input/source.json").read_text(encoding="utf-8"))
    config = {
        "model_config": model_config,
        "object_prompt_frame": prompt_frame,
        "object_point_xy": [point_x, point_y],
        "confidence_threshold": confidence_threshold,
    }
    manifest = RunManifest.load(root / "manifest.json")
    manifest.start_stage(
        "sam2_paper", cache_key=stage_cache_key(
            "sam2_paper", config,
            [checkpoint_path, gt_path, root / "frames/frame_index.json",
             root / "calibration/intrinsics.npy", root / "calibration/T_world_camera.npy"],
        ),
        command=sys.argv, environment="v2s-sam3+sam2",
    )
    try:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            model_config, str(checkpoint_path), device="cuda",
            apply_postprocessing=False,
        )
        with tempfile.TemporaryDirectory(prefix="egoengine-sam2-") as temporary:
            frame_dir = Path(temporary)
            for index, frame_path in enumerate(frame_paths):
                (frame_dir / f"{index:06d}.jpg").symlink_to(frame_path)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported() else nullcontext()
            )
            with torch.inference_mode(), autocast:
                object_result = _track_object(
                    predictor, frame_dir, len(rows), prompt_frame, (point_x, point_y),
                )
                hand_result = _segment_hands(predictor, frame_paths, hand_prompts)
        object_path = output / "object_masks.npz"
        hand_path = output / "hand_masks.npz"
        _save_mask_artifact(object_path, rows, object_result)
        _save_mask_artifact(hand_path, rows, hand_result)
        overlay_path = output / "perception_overlay.mp4"
        if write_overlay:
            _write_overlay(
                frame_paths, object_result["masks"], hand_result["masks"],
                overlay_path, float(source["video"]["fps"]),
            )
        output_names = ["object_masks.npz", "hand_masks.npz", "sam2_prompts.npz"]
        if write_overlay:
            output_names.append("perception_overlay.mp4")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model": "SAM 2.1 Hiera Large",
            "model_config": model_config,
            "checkpoint": str(checkpoint_path),
            "device": str(torch.cuda.get_device_name()),
            "paper_route": True,
            "overlay_skipped": not write_overlay,
            "object_prompt": {
                "type": "single_positive_point",
                "frame_relative": prompt_frame,
                "point_xy": [point_x, point_y],
                "propagation": "bidirectional_video_memory",
            },
            "hand_prompt": {
                "type": "per_frame_projected_MANO21_positive_points",
                "source": str(gt_path),
                "uses_ground_truth": True,
                "role": "oracle_hand",
                "projective_sign_repair_frame_count": int(np.count_nonzero(
                    hand_prompts["camera_coordinate_sign"] < 0
                )),
                "projective_sign_repair_is_se3": False,
            },
            "object_metrics": _metrics(object_result),
            "hand_metrics": _metrics(hand_result),
            "outputs": output_names,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage_outputs = [object_path, hand_path, prompt_report, metadata_path]
        if write_overlay:
            stage_outputs.append(overlay_path)
        manifest.finish_stage(
            "sam2_paper", success=True,
            outputs=[str(path.relative_to(root)) for path in stage_outputs],
            quality_metrics={
                "object_valid_rate": metadata["object_metrics"]["valid_rate"],
                "hand_valid_rate": metadata["hand_metrics"]["valid_rate"],
            },
            warnings=["Hand masks use evaluation-only TACO MANO21 keypoints as oracle prompts"],
        )
        return metadata_path
    except Exception as error:
        manifest = RunManifest.load(root / "manifest.json")
        manifest.finish_stage(
            "sam2_paper", success=False,
            outputs=[str(prompt_report.relative_to(root))],
            quality_metrics={"error_type": type(error).__name__, "error": str(error)},
            warnings=["SAM2 paper route did not produce complete mask artifacts"],
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hand-ground-truth", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--object-point", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(run_sam2_paper_masks(
        args.run_dir, args.hand_ground_truth, args.checkpoint,
        tuple(args.object_point), model_config=args.model_config,
        prompt_frame=args.prompt_frame, output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        overwrite=args.overwrite, dry_run=args.dry_run,
        write_overlay=not args.skip_overlay,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
