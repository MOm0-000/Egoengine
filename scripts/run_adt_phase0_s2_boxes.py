#!/usr/bin/env python3
"""Generate Grounding DINO anchor boxes for the 10 ADT Phase 0 samples.

Run inside the ``v2s-grounding`` conda env. The output JSON is consumed by the
S2 SAM3 run so text-only anchor failures can be retried with a normalized
positive box prompt.
"""

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
OUTPUT_JSON = RUNS_ROOT / "adt_phase0_s2_boxes.json"
VIS_DIR = RUNS_ROOT / "adt_phase0_s2_boxes_vis"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0  # noqa: E402

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
TOP_K = 5


def _box_to_normalized_xywh(xyxy: np.ndarray, image_w: int, image_h: int) -> list[float]:
    x1 = float(np.clip(xyxy[0], 0.0, image_w))
    y1 = float(np.clip(xyxy[1], 0.0, image_h))
    x2 = float(np.clip(xyxy[2], 0.0, image_w))
    y2 = float(np.clip(xyxy[3], 0.0, image_h))
    x = x1 / image_w
    y = y1 / image_h
    w = max(x2 / image_w - x, 1e-4)
    h = max(y2 / image_h - y, 1e-4)
    # Keep the prompt fully inside [0, 1]^2.
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 1e-4, 1.0 - x))
    h = float(np.clip(h, 1e-4, 1.0 - y))
    return [x, y, w, h]


def _visualize(path: Path, image: np.ndarray, xyxy: np.ndarray, phrase: str, confidence: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 3)
    label = f"{phrase} {confidence:.3f}"
    cv2.putText(overlay, label, (max(x1, 0), max(y1 - 12, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.imwrite(str(path), overlay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"missing Grounding DINO checkpoint: {WEIGHTS_PATH}")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[groundingdino] loading {WEIGHTS_PATH}", flush=True)
    model = Model(model_config_path=str(CONFIG_PATH), model_checkpoint_path=str(WEIGHTS_PATH), device=device)
    print(f"[groundingdino] device={device}", flush=True)

    rows = phase0._rows()
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        run_dir = phase0._object_run(row)
        if run_dir is None:
            print(f"[missing run] {row['prototype']} {row['window']}", flush=True)
            continue
        source = json.loads((run_dir / "input" / "source.json").read_text(encoding="utf-8"))
        frame_index = json.loads((run_dir / "frames" / "frame_index.json").read_text(encoding="utf-8"))["frames"]
        frame_paths = [run_dir / item["rgb_path"] for item in frame_index]
        anchor = phase0._choose_anchor(frame_paths) if hasattr(phase0, "_choose_anchor") else _local_anchor(frame_paths)
        anchor_path = frame_paths[anchor]
        image = cv2.imread(str(anchor_path))
        if image is None:
            raise RuntimeError(f"cannot read anchor frame {anchor_path}")
        prompts = phase0._s1_prompts(row["prototype"], source.get("object_keyword_candidates", []))
        caption = ". ".join(prompts)
        print(f"[gd] {run_dir.name} anchor={anchor} prompts={len(prompts)}", flush=True)
        detections, phrases = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        entry: dict[str, Any] = {
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
            "anchor": anchor,
            "anchor_path": str(anchor_path),
            "prompts": prompts,
            "detections": [],
            "selected_box_xywh": None,
            "selected_box_xyxy": None,
            "selected_confidence": None,
            "selected_phrase": None,
        }
        if len(detections) == 0:
            print(f"[gd] {run_dir.name}: no detections", flush=True)
            output_rows.append(entry)
            continue
        xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        confidence = np.asarray(detections.confidence, dtype=np.float32)
        boxes_t = torch.from_numpy(xyxy)
        scores_t = torch.from_numpy(confidence)
        keep = nms(boxes_t, scores_t, NMS_IOU_THRESHOLD).numpy()
        kept_indices = sorted((int(index) for index in keep), key=lambda index: float(confidence[index]), reverse=True)
        h, w = image.shape[:2]
        for rank, index in enumerate(kept_indices[:TOP_K]):
            det = {
                "rank": rank,
                "xyxy": [float(value) for value in xyxy[index]],
                "xywh_normalized": _box_to_normalized_xywh(xyxy[index], w, h),
                "confidence": float(confidence[index]),
                "phrase": phrases[index],
            }
            entry["detections"].append(det)
            if rank == 0:
                entry["selected_box_xywh"] = det["xywh_normalized"]
                entry["selected_box_xyxy"] = det["xyxy"]
                entry["selected_confidence"] = det["confidence"]
                entry["selected_phrase"] = det["phrase"]
                _visualize(VIS_DIR / f"{run_dir.name}_anchor{anchor}.jpg", image, xyxy[index], phrases[index], confidence[index])
        print(f"[gd] {run_dir.name}: top={entry['selected_phrase']!r} conf={entry['selected_confidence']:.3f}", flush=True)
        output_rows.append(entry)

    OUTPUT_JSON.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[gd] wrote {OUTPUT_JSON}", flush=True)
    return 0


def _local_anchor(frame_paths: list[Path]) -> int:
    scores = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read frame {path}")
        scores.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    margin = 1 if len(scores) > 2 else 0
    eligible = np.asarray(scores[margin : len(scores) - margin or None])
    return int(np.argmax(eligible)) + margin


if __name__ == "__main__":
    raise SystemExit(main())
