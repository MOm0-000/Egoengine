"""SAM 3.1 text-only video segmentation adapter.

This module intentionally imports SAM 3 only inside ``run`` so core tooling can
inspect artifacts without importing third-party model code.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
import tempfile
import time
import types
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_name(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_") or "prompt"


def _archive_prior_failure(output_dir: Path, overwrite: bool) -> None:
    failure_path = output_dir / "failure.json"
    if overwrite and failure_path.exists():
        history_dir = output_dir / "failure_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        failure_path.replace(history_dir / f"failure_{time.time_ns()}.json")


def _output_arrays(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = _numpy(outputs.get("out_binary_masks", np.zeros((0, 1, 1), dtype=bool))).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    ids = _numpy(outputs.get("out_obj_ids", np.arange(masks.shape[0]))).reshape(-1)
    probabilities = outputs.get("out_probs", outputs.get("out_scores", outputs.get("scores", None)))
    if probabilities is None:
        probabilities = np.ones(ids.shape[0], dtype=np.float32)
    probabilities = _numpy(probabilities).reshape(-1)
    if probabilities.size != ids.size:
        probabilities = np.resize(probabilities, ids.size)
    # Multiplex uses large negative sentinels for suppressed instances even in
    # the field named ``out_probs``. They are not probabilities and must not
    # contaminate confidence metrics or count as valid observations.
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return ids.astype(np.int64), masks, probabilities.astype(np.float32)


def _invalid_spans(valid: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(np.asarray(valid, dtype=bool)):
        if not is_valid and start is None:
            start = index
        elif is_valid and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(valid)))
    return spans


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def _choose_anchor(frame_paths: list[Path]) -> int:
    scores = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read frame {path}")
        scores.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    # Avoid boundary anchors so bidirectional propagation exercises both directions.
    margin = 1 if len(scores) > 2 else 0
    eligible = np.asarray(scores[margin : len(scores) - margin or None])
    return int(np.argmax(eligible)) + margin


def _start_session_compat(predictor: Any, frame_dir: Path) -> str:
    """Work around the 2026-07-29 OSS SAM 3.1 base/multiplex signature mismatch."""
    try:
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
        session_id = str(response["session_id"])
    except TypeError as error:
        if "offload_state_to_cpu" not in str(error):
            raise
        state = predictor.model.init_state(
            resource_path=str(frame_dir), offload_video_to_cpu=True,
            async_loading_frames=getattr(predictor, "async_loading_frames", False),
        )
        session_id = str(uuid.uuid4())
        predictor._all_inference_states[session_id] = {
            "state": state, "session_id": session_id,
            "start_time": time.time(), "last_use_time": time.time(),
        }
    _offload_session_frames(predictor, session_id)
    return session_id


def _set_max_objects(predictor: Any, count: int) -> None:
    """Set the multiplex padding cap consistently for the detector and tracker."""
    predictor.model.max_num_objects = count
    if hasattr(predictor.model, "tracker"):
        predictor.model.tracker.max_num_objects = count


def _configure_state_offload(predictor: Any) -> None:
    """Restore the upstream tracker CPU-state offload path for multiplex sessions."""
    model = predictor.model
    if not hasattr(model, "_init_new_sam2_state") or not hasattr(model, "tracker"):
        return

    def _init_new_sam2_state(low_vram_model: Any, inference_state: dict[str, Any]) -> Any:
        return low_vram_model.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

    model._init_new_sam2_state = types.MethodType(_init_new_sam2_state, model)


def _offload_session_frames(predictor: Any, session_id: str) -> None:
    """Keep the full resized clip on CPU; SAM already transfers each indexed frame on demand."""
    state = predictor._all_inference_states[session_id]["state"]
    nested = state.get("input_batch", None)
    if nested is None:
        return
    image_batch = nested.img_batch
    tensors = getattr(image_batch, "tensors", None)
    if tensors is not None and getattr(tensors, "is_cuda", False):
        image_batch.tensors = tensors.cpu()


def _run_prompt(
    predictor: Any, frame_dir: Path, prompt: str, anchor: int, frame_count: int,
    anchor_hand_mask: np.ndarray,
) -> dict[str, Any]:
    # Preserve several detections for automatic hand-proximity ranking. The
    # multiplex implementation owns a coupled detector/tracker cache, so object
    # removal before its first propagation can invalidate cached frame outputs.
    # Tracker state and the full input clip are CPU-offloaded instead.
    _set_max_objects(predictor, 4)
    session_id = _start_session_compat(predictor, frame_dir)
    try:
        response = predictor.handle_request({
            "type": "add_prompt", "session_id": session_id, "frame_index": anchor, "text": prompt,
        })
        initial_ids, initial_masks, initial_probabilities = _output_arrays(response["outputs"])
        if initial_ids.size:
            nonempty = initial_masks.reshape(initial_masks.shape[0], -1).sum(axis=1)
            distance_to_hand = cv2.distanceTransform((~anchor_hand_mask).astype(np.uint8), cv2.DIST_L2, 3)
            distances = np.asarray([
                float(np.min(distance_to_hand[mask])) if mask.any() else float("inf") for mask in initial_masks
            ])
            proximity = np.exp(-np.minimum(distances, 1000.0) / 80.0)
            instance_scores = 0.55 * initial_probabilities + 0.45 * proximity + (nonempty > 0) * 1e-4
            target_at = int(np.argmax(instance_scores))
            target_id = int(initial_ids[target_at])
            target_anchor_confidence = float(initial_probabilities[target_at])
            target_anchor_hand_distance_px = float(distances[target_at])
            target_instance_score = float(instance_scores[target_at])
        else:
            target_id = -1
            target_anchor_confidence = 0.0
            target_anchor_hand_distance_px = -1.0
            target_instance_score = 0.0
        per_frame = {anchor: response["outputs"]}
        for item in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session_id,
            "propagation_direction": "both", "start_frame_index": anchor,
        }):
            per_frame[int(item["frame_index"])] = item["outputs"]
        sample_masks = initial_masks
        if sample_masks.size:
            height, width = sample_masks.shape[-2:]
        else:
            first = cv2.imread(str(sorted(frame_dir.glob("*.jpg"))[0]))
            height, width = first.shape[:2]
        masks = np.zeros((frame_count, height, width), dtype=bool)
        valid = np.zeros(frame_count, dtype=bool)
        confidence = np.zeros(frame_count, dtype=np.float32)
        object_ids = np.full(frame_count, -1, dtype=np.int64)
        for frame_index, outputs in per_frame.items():
            ids, output_masks, probabilities = _output_arrays(outputs)
            matches = np.flatnonzero(ids == target_id)
            if matches.size:
                selected = int(matches[0])
                masks[frame_index] = output_masks[selected]
                valid[frame_index] = bool(masks[frame_index].any() and probabilities[selected] > 0.0)
                confidence[frame_index] = probabilities[selected]
                object_ids[frame_index] = target_id
        return {
            "prompt": prompt, "anchor_frame": anchor, "target_object_id": target_id,
            "target_anchor_confidence": target_anchor_confidence,
            "target_anchor_hand_distance_px": target_anchor_hand_distance_px,
            "target_instance_score": target_instance_score,
            "masks": masks, "valid": valid, "confidence": confidence, "object_ids": object_ids,
        }
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    masks, valid, confidence = result["masks"], result["valid"], result["confidence"]
    areas = masks.reshape(masks.shape[0], -1).mean(axis=1)
    valid_areas = areas[valid]
    area_cv = float(np.std(valid_areas) / max(np.mean(valid_areas), 1e-8)) if valid_areas.size else float("inf")
    boundary = np.zeros(masks.shape[0], dtype=bool)
    if masks.size:
        boundary = masks[:, 0].any(1) | masks[:, -1].any(1) | masks[:, :, 0].any(1) | masks[:, :, -1].any(1)
    valid_rate = float(np.mean(valid))
    mean_confidence = float(np.mean(confidence[valid])) if valid.any() else 0.0
    area_stability = float(np.exp(-min(area_cv, 20.0)))
    boundary_safety = 1.0 - float(np.mean(boundary[valid])) if valid.any() else 0.0
    score = 0.35 * valid_rate + 0.30 * mean_confidence + 0.20 * area_stability + 0.15 * boundary_safety
    return {
        "valid_rate": valid_rate, "mean_confidence": mean_confidence,
        "mean_area_ratio": float(np.mean(valid_areas)) if valid_areas.size else 0.0,
        "area_cv": area_cv if np.isfinite(area_cv) else -1.0,
        "area_stability": area_stability, "boundary_safety": boundary_safety,
        "selection_score": float(score),
    }


def _save_mask_artifact(path: Path, frame_indices: np.ndarray, timestamps: np.ndarray, result: dict[str, Any]) -> None:
    np.savez_compressed(
        path, frame_indices=frame_indices, timestamps_s=timestamps,
        masks=result["masks"].astype(np.uint8), valid=result["valid"],
        confidence=result["confidence"], object_ids=result["object_ids"],
    )


def _load_mask_artifact(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a prior mask artifact and normalize legacy multiplex sentinels."""
    with np.load(path, allow_pickle=False) as artifact:
        frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(artifact["timestamps_s"], dtype=np.float64)
        masks = np.asarray(artifact["masks"], dtype=bool)
        confidence = np.clip(np.asarray(artifact["confidence"], dtype=np.float32), 0.0, 1.0)
        valid = np.asarray(artifact["valid"], dtype=bool)
        valid &= masks.reshape(masks.shape[0], -1).any(1)
        valid &= confidence > 0.0
        result = {
            "masks": masks, "valid": valid, "confidence": confidence,
            "object_ids": np.asarray(artifact["object_ids"], dtype=np.int64),
        }
    return frame_indices, timestamps, result


def _hand_union(predictor: Any, frame_dir: Path, anchor: int, frame_count: int) -> dict[str, Any]:
    # A hand prompt needs at most the two visible hands in this bimanual V1
    # pipeline. Avoid padding every tracking tensor to the object-ranking cap.
    _set_max_objects(predictor, 2)
    session_id = _start_session_compat(predictor, frame_dir)
    try:
        response = predictor.handle_request({
            "type": "add_prompt", "session_id": session_id, "frame_index": anchor, "text": "hand",
        })
        per_frame = {anchor: response["outputs"]}
        for item in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session_id,
            "propagation_direction": "both", "start_frame_index": anchor,
        }):
            per_frame[int(item["frame_index"])] = item["outputs"]
        _, sample, _ = _output_arrays(response["outputs"])
        if sample.size:
            height, width = sample.shape[-2:]
        else:
            image = cv2.imread(str(sorted(frame_dir.glob("*.jpg"))[0]))
            height, width = image.shape[:2]
        masks = np.zeros((frame_count, height, width), dtype=bool)
        confidence = np.zeros(frame_count, dtype=np.float32)
        for frame_index, outputs in per_frame.items():
            _, object_masks, probabilities = _output_arrays(outputs)
            if object_masks.size:
                masks[frame_index] = np.any(object_masks, axis=0)
                confidence[frame_index] = float(np.max(probabilities))
        return {"masks": masks, "valid": masks.reshape(frame_count, -1).any(1), "confidence": confidence,
                "object_ids": np.full(frame_count, -1), "prompt": "hand"}
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def _recover_invalid_spans(
    predictor: Any, frame_paths: list[Path], frame_dir: Path, prompt: str,
    result: dict[str, Any], hand: dict[str, Any],
) -> list[dict[str, Any]]:
    """Automatically re-ground invalid temporal spans and merge identity-consistent masks."""
    recoveries: list[dict[str, Any]] = []
    spans = _invalid_spans(result["valid"])
    for recovery_index, (start, end) in enumerate(spans):
        expanded_start = max(0, start - 1)
        expanded_end = min(len(frame_paths), end + 1)
        subset_dir = frame_dir / f"recovery_{recovery_index:02d}"
        subset_dir.mkdir()
        for local_index, source_path in enumerate(frame_paths[expanded_start:expanded_end]):
            (subset_dir / f"{local_index:06d}.jpg").symlink_to(source_path)
        # The anchor remains automatic and is selected only from the invalid
        # span. Adjacent valid frames are included solely for identity QC.
        anchor_global = start + _choose_anchor(frame_paths[start:end])
        anchor_local = anchor_global - expanded_start
        local_count = expanded_end - expanded_start
        local_hand = _hand_union(predictor, subset_dir, anchor_local, local_count)
        local_result = _run_prompt(
            predictor, subset_dir, prompt, anchor_local, local_count,
            local_hand["masks"][anchor_local],
        )
        overlap_ious = []
        for global_index in range(expanded_start, expanded_end):
            if start <= global_index < end or not result["valid"][global_index]:
                continue
            local_index = global_index - expanded_start
            if local_result["valid"][local_index]:
                overlap_ious.append(_mask_iou(
                    result["masks"][global_index], local_result["masks"][local_index]
                ))
        filled_local = np.flatnonzero(local_result["valid"][
            start - expanded_start : end - expanded_start
        ]) + (start - expanded_start)
        max_overlap_iou = max(overlap_ious, default=0.0)
        hand_distance = float(local_result["target_anchor_hand_distance_px"])
        accepted = bool(
            local_result["target_object_id"] >= 0 and filled_local.size > 0
            and (max_overlap_iou >= 0.10 or 0.0 <= hand_distance <= 40.0)
        )
        filled_global: list[int] = []
        if accepted:
            for local_index in filled_local:
                global_index = expanded_start + int(local_index)
                result["masks"][global_index] = local_result["masks"][local_index]
                result["valid"][global_index] = local_result["valid"][local_index]
                result["confidence"][global_index] = local_result["confidence"][local_index]
                result["object_ids"][global_index] = local_result["object_ids"][local_index]
                if local_hand["valid"][local_index] and not hand["valid"][global_index]:
                    hand["masks"][global_index] = local_hand["masks"][local_index]
                    hand["valid"][global_index] = True
                    hand["confidence"][global_index] = local_hand["confidence"][local_index]
                filled_global.append(global_index)
        recoveries.append({
            "invalid_span_relative": [start, end],
            "expanded_span_relative": [expanded_start, expanded_end],
            "anchor_frame_relative": anchor_global,
            "target_object_id": int(local_result["target_object_id"]),
            "target_anchor_hand_distance_px": hand_distance,
            "max_overlap_iou": max_overlap_iou,
            "filled_frames_relative": filled_global,
            "accepted": accepted,
            "metrics": _metrics(local_result),
        })
    result["metrics"] = _metrics(result)
    return recoveries


def _write_overlay(frame_paths: list[Path], object_masks: np.ndarray, hand_masks: np.ndarray, path: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        overlay = frame.copy()
        overlay[object_masks[index]] = (0, 220, 0)
        overlay[hand_masks[index]] = (0, 0, 220)
        frame = cv2.addWeighted(frame, 0.70, overlay, 0.30, 0)
        writer.write(frame)
    writer.release()


def run(args: argparse.Namespace) -> Path:
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 video predictor requires a visible CUDA device")
    run_dir = args.run_dir.resolve()
    source = json.loads((run_dir / "input/source.json").read_text(encoding="utf-8"))
    index = json.loads((run_dir / "frames/frame_index.json").read_text(encoding="utf-8"))["frames"]
    start, end = args.start_frame, args.end_frame if args.end_frame is not None else len(index)
    if start < 0 or end <= start or end > len(index):
        raise ValueError(f"invalid adapter interval [{start}, {end})")
    selected_rows = index[start:end]
    frame_paths = [run_dir / row["rgb_path"] for row in selected_rows]
    prompts = source["object_keyword_candidates"][: args.max_candidates]
    if not prompts:
        raise RuntimeError("no instruction-derived object keyword candidates")
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "segmentation"
    if args.resume_existing and not args.overwrite:
        raise ValueError("--resume-existing mutates the selected artifact and requires --overwrite")
    if output_dir.exists() and (output_dir / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"segmentation output exists: {output_dir}")
    if args.resume_existing and not (output_dir / "metadata.json").exists():
        raise FileNotFoundError(f"cannot resume segmentation without metadata: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _archive_prior_failure(output_dir, args.overwrite)
    if args.dry_run:
        (output_dir / "dry_run.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "prompts": prompts, "frame_count": len(frame_paths),
        }, indent=2) + "\n")
        return output_dir / "dry_run.json"
    with tempfile.TemporaryDirectory(prefix="v2s-sam3-") as temporary:
        frame_dir = Path(temporary)
        for relative_index, source_path in enumerate(frame_paths):
            (frame_dir / f"{relative_index:06d}.jpg").symlink_to(source_path)
        anchor = _choose_anchor(frame_paths)
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(args.checkpoint), use_fa3=False, use_rope_real=False,
            max_num_objects=4, compile=False, warm_up=False, async_loading_frames=False,
        )
        _configure_state_offload(predictor)
        # The upstream multiplex default batches 16 video frames for grounding,
        # which creates a >1 GiB transient allocation at 1080p. This V1 adapter
        # tracks one instruction target (plus a separate hand session), so a
        # single-frame grounding batch and four-instance cap preserve semantics
        # while making full-clip inference viable on a shared 40 GiB GPU.
        predictor.model.batched_grounding_batch_size = 1
        _set_max_objects(predictor, 4)
        try:
            candidate_dir = output_dir / "candidates"
            candidate_dir.mkdir(exist_ok=True)
            expected_frame_indices = np.asarray(
                [row["source_frame_index"] for row in selected_rows], dtype=np.int64
            )
            expected_timestamps = np.asarray([row["timestamp_s"] for row in selected_rows], dtype=np.float64)
            if args.resume_existing:
                prior_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
                frame_indices, timestamps, resumed_result = _load_mask_artifact(
                    output_dir / "object_masks.npz"
                )
                hand_frame_indices, hand_timestamps, hand = _load_mask_artifact(
                    output_dir / "hand_masks.npz"
                )
                if not (
                    np.array_equal(frame_indices, expected_frame_indices)
                    and np.array_equal(hand_frame_indices, expected_frame_indices)
                    and np.allclose(timestamps, expected_timestamps)
                    and np.allclose(hand_timestamps, expected_timestamps)
                ):
                    raise ValueError("existing segmentation timeline does not match the requested interval")
                selected_prompt = str(prior_metadata["selected_prompt"])
                if selected_prompt not in prompts:
                    raise ValueError("existing selected prompt is not instruction-derived for this run")
                prior_candidate = next(
                    (item for item in prior_metadata.get("candidates", [])
                     if item.get("prompt") == selected_prompt),
                    {},
                )
                resumed_result.update({
                    "prompt": selected_prompt,
                    "anchor_frame": int(prior_metadata["anchor_frame_relative"]),
                    "target_object_id": int(prior_candidate.get("target_object_id", -1)),
                    "target_anchor_confidence": float(prior_candidate.get("target_anchor_confidence", 0.0)),
                    "target_anchor_hand_distance_px": float(
                        prior_candidate.get("target_anchor_hand_distance_px", -1.0)
                    ),
                    "target_instance_score": float(prior_candidate.get("target_instance_score", 0.0)),
                })
                hand.update({"prompt": "hand"})
                resumed_result["metrics"] = _metrics(resumed_result)
                selected = {
                    "result": resumed_result,
                    "artifact": str(prior_candidate.get(
                        "artifact", f"candidates/{_safe_name(selected_prompt)}.npz"
                    )),
                }
                candidates = [selected]
                anchor = int(prior_metadata["anchor_frame_relative"])
            else:
                frame_indices = expected_frame_indices
                timestamps = expected_timestamps
                candidates = []
                hand = _hand_union(predictor, frame_dir, anchor, len(frame_paths))
                for prompt in prompts:
                    result = _run_prompt(
                        predictor, frame_dir, prompt, anchor, len(frame_paths), hand["masks"][anchor]
                    )
                    metrics = _metrics(result)
                    result["metrics"] = metrics
                    artifact_path = candidate_dir / f"{_safe_name(prompt)}.npz"
                    _save_mask_artifact(artifact_path, frame_indices, timestamps, result)
                    candidates.append({"result": result, "artifact": str(artifact_path.relative_to(output_dir))})
                selected = max(candidates, key=lambda item: item["result"]["metrics"]["selection_score"])
            recoveries = _recover_invalid_spans(
                predictor, frame_paths, frame_dir, selected["result"]["prompt"],
                selected["result"], hand,
            )
            _save_mask_artifact(
                output_dir / selected["artifact"], frame_indices, timestamps, selected["result"]
            )
        finally:
            if hasattr(predictor, "shutdown"):
                predictor.shutdown()
    object_path, hand_path = output_dir / "object_masks.npz", output_dir / "hand_masks.npz"
    _save_mask_artifact(object_path, frame_indices, timestamps, selected["result"])
    _save_mask_artifact(hand_path, frame_indices, timestamps, hand)
    overlay_path = output_dir / "perception_overlay.mp4"
    _write_overlay(frame_paths, selected["result"]["masks"], hand["masks"], overlay_path, float(source["video"]["fps"]))
    metadata = {
        "schema_version": "1.0", "model": "SAM 3.1 multiplex", "checkpoint": str(args.checkpoint.resolve()),
        "device": str(torch.cuda.get_device_name(0)), "text_only": True, "anchor_frame_relative": anchor,
        "offload_video_to_cpu": True, "offload_input_batch_to_cpu": True,
        "offload_tracker_state_to_cpu": True,
        "resumed_existing_artifact": bool(args.resume_existing),
        "batched_grounding_batch_size": 1,
        "grounding_max_num_objects": 4, "hand_max_num_objects": 2,
        "object_tracking_max_num_objects": 4,
        "selected_prompt": selected["result"]["prompt"], "frame_count": len(frame_paths),
        "automatic_invalid_span_recovery": recoveries,
        "candidates": [{
            "prompt": item["result"]["prompt"], "artifact": item["artifact"],
            "target_object_id": item["result"]["target_object_id"], "metrics": item["result"]["metrics"],
            "target_anchor_confidence": item["result"]["target_anchor_confidence"],
            "target_anchor_hand_distance_px": item["result"]["target_anchor_hand_distance_px"],
            "target_instance_score": item["result"]["target_instance_score"],
        } for item in candidates],
        "hand_metrics": _metrics(hand),
        "outputs": ["object_masks.npz", "hand_masks.npz", "perception_overlay.mp4"],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument(
        "--resume-existing", action="store_true",
        help="automatically rerun only invalid spans from an existing artifact",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    try:
        print(run(parsed_args))
    except Exception as error:
        output_dir = (
            parsed_args.output_dir.resolve() if parsed_args.output_dir
            else parsed_args.run_dir.resolve() / "segmentation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_text(json.dumps({
            "schema_version": "1.0", "success": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "interval": [parsed_args.start_frame, parsed_args.end_frame],
            "checkpoint": str(parsed_args.checkpoint.resolve()),
            "max_candidates": int(parsed_args.max_candidates),
        }, indent=2) + "\n", encoding="utf-8")
        raise
