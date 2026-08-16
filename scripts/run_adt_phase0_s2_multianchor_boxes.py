#!/usr/bin/env python3
"""Generate temporally spread Grounding DINO boxes for SAM3 multi-anchor S2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from groundingdino.util.inference import Model

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
CONFIG_PATH = Path(
    "/home/zzx/miniconda3/envs/v2s-grounding/lib/python3.11/site-packages/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
WEIGHTS_PATH = REPO_ROOT / "third_party/groundingdino/checkpoints/groundingdino_swint_ogc.pth"
OUTPUT_JSON = RUNS_ROOT / "adt_phase0_s2_multianchor_boxes.json"
VIS_DIR = RUNS_ROOT / "adt_phase0_s2_multianchor_vis"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0  # noqa: E402

BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10
NMS_IOU_THRESHOLD = 0.5
ASPECT_PENALTY = 0.8


def _compactness_score(xyxy: np.ndarray, confidence: float) -> float:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    width = max(x2 - x1, 1e-3)
    height = max(y2 - y1, 1e-3)
    aspect = max(width / height, height / width)
    return confidence * float(np.exp(-abs(np.log(aspect)) * ASPECT_PENALTY))


def _normalized_xywh(xyxy: np.ndarray, image_w: int, image_h: int) -> list[float]:
    x1 = float(np.clip(xyxy[0], 0.0, image_w))
    y1 = float(np.clip(xyxy[1], 0.0, image_h))
    x2 = float(np.clip(xyxy[2], 0.0, image_w))
    y2 = float(np.clip(xyxy[3], 0.0, image_h))
    x = x1 / image_w
    y = y1 / image_h
    width = max(x2 / image_w - x, 1e-4)
    height = max(y2 / image_h - y, 1e-4)
    return [float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1)), float(np.clip(width, 1e-4, 1 - x)), float(np.clip(height, 1e-4, 1 - y))]


def _visualize(path: Path, image: np.ndarray, xyxy: np.ndarray, phrase: str, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 3)
    cv2.putText(overlay, f"{phrase} {score:.3f}", (max(x1, 0), max(y1 - 12, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.imwrite(str(path), overlay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num-anchors", type=int, default=6)
    parser.add_argument("--only-failed", action="store_true")
    args = parser.parse_args(argv)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[gd multianchor] loading {WEIGHTS_PATH} device={device}", flush=True)
    model = Model(model_config_path=str(CONFIG_PATH), model_checkpoint_path=str(WEIGHTS_PATH), device=device)
    rows = phase0._rows()
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        if args.only_failed and row["prototype"] not in {"BlackCeramicMug", "Flask", "StepStool"}:
            continue
        run_dir = phase0._object_run(row)
        if run_dir is None:
            continue
        source = json.loads((run_dir / "input" / "source.json").read_text(encoding="utf-8"))
        frame_index = json.loads((run_dir / "frames" / "frame_index.json").read_text(encoding="utf-8"))["frames"]
        prompts = phase0._s1_prompts(row["prototype"], source.get("object_keyword_candidates", []))
        caption = ". ".join(prompts)
        best_per_frame: dict[int, dict[str, Any]] = {}
        for frame, item in enumerate(frame_index):
            image = cv2.imread(str(run_dir / item["rgb_path"]))
            if image is None:
                raise RuntimeError(f"cannot read {run_dir / item['rgb_path']}")
            detections, phrases = model.predict_with_caption(
                image=image, caption=caption, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
            )
            if len(detections) == 0:
                continue
            xyxy = np.asarray(detections.xyxy, dtype=np.float32)
            confidence = np.asarray(detections.confidence, dtype=np.float32)
            keep = nms(torch.from_numpy(xyxy), torch.from_numpy(confidence), NMS_IOU_THRESHOLD).numpy()
            h, w = image.shape[:2]
            for index in keep:
                index = int(index)
                candidate = {
                    "frame": frame,
                    "source_frame": int(item["source_frame_index"]),
                    "xyxy": [float(value) for value in xyxy[index]],
                    "xywh_normalized": _normalized_xywh(xyxy[index], w, h),
                    "confidence": float(confidence[index]),
                    "phrase": phrases[index],
                    "compactness_score": _compactness_score(xyxy[index], float(confidence[index])),
                }
                previous = best_per_frame.get(frame)
                if previous is None or candidate["compactness_score"] > previous["compactness_score"]:
                    best_per_frame[frame] = candidate
        n_frames = len(frame_index)
        selected: list[dict[str, Any]] = []
        for bin_index in range(args.num_anchors):
            lo = int(n_frames * bin_index / args.num_anchors)
            hi = int(n_frames * (bin_index + 1) / args.num_anchors)
            frames_in_bin = [frame for frame in range(lo, hi) if frame in best_per_frame]
            if not frames_in_bin:
                continue
            best_frame = max(frames_in_bin, key=lambda frame: best_per_frame[frame]["compactness_score"])
            selected.append(best_per_frame[best_frame])
        selected = sorted(selected, key=lambda item: item["frame"])
        for item in selected:
            image = cv2.imread(str(run_dir / frame_index[item["frame"]]["rgb_path"]))
            _visualize(VIS_DIR / f"{run_dir.name}_frame{item['frame']:02d}.jpg", image, np.asarray(item["xyxy"]), item["phrase"], item["compactness_score"])
        output_rows.append({
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
            "prompts": prompts,
            "anchors": selected,
        })
        print(f"[gd multianchor] {row['prototype']}: {len(selected)} anchors {[item['frame'] for item in selected]}", flush=True)
    OUTPUT_JSON.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[gd multianchor] wrote {OUTPUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
