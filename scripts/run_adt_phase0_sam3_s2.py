#!/usr/bin/env python3
"""S2: Grounding DINO candidate boxes + SAM3 box prompts.

Reads ``runs/adt_phase0_s2_boxes.json`` and runs a small per-box SAM3 ablation.
For reporting, masks are compared against ADT RGB GT; GT is never used by the
box selection step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
BOX_PATH = RUNS_ROOT / "adt_phase0_s2_boxes.json"
CHECKPOINT = REPO_ROOT / "third_party/sam3/checkpoints/sam3.1_multiplex.pt"
OUTPUT_SUMMARY = RUNS_ROOT / "adt_phase0_sam3_s2_summary.json"
OUTPUT_CSV = RUNS_ROOT / "adt_phase0_sam3_s2_matrix.csv"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0  # noqa: E402

FAILED_PROTOTYPES = {"BlackCeramicMug", "Flask", "StepStool"}


def _mask_metrics_for(run_dir: Path, subdir: str) -> dict[str, Any]:
    import zarr

    masks_path = run_dir / subdir / "object_masks.npz"
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)
    with np.load(masks_path, allow_pickle=False) as artifact:
        pred_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        pred_masks = np.asarray(artifact["masks"], dtype=bool)
        pred_valid = np.asarray(artifact["valid"], dtype=bool)
    valid_rate = float(np.mean(pred_valid)) if pred_valid.size else float("nan")
    result: dict[str, Any] = {
        "valid_rate": valid_rate,
        "coverage": valid_rate,
        "frames_with_mask": int(np.sum(pred_valid)),
        "frame_count": int(pred_valid.size),
    }
    gt_path = run_dir / "evaluation" / "adt_rgb_gt" / "adt_gt.zarr"
    if not gt_path.is_dir():
        result.update({
            "matched_frames": None, "mean_iou": None, "median_iou": None,
            "p05_iou": None, "p95_iou": None,
        })
        return result
    gt_group = zarr.open_group(str(gt_path), mode="r")
    gt_indices = np.asarray(gt_group["frame_indices"], dtype=np.int64)
    gt_masks = np.asarray(gt_group["object_mask"], dtype=bool)
    gt_valid = np.asarray(gt_group["valid"], dtype=bool)
    gt_lookup = {int(frame): index for index, frame in enumerate(gt_indices)}
    ious: list[float] = []
    for index, frame in enumerate(pred_indices):
        if int(frame) not in gt_lookup:
            continue
        gt_index = gt_lookup[int(frame)]
        pred = pred_masks[index] & bool(pred_valid[index])
        gt = gt_masks[gt_index] & gt_valid[gt_index]
        union = np.logical_or(pred, gt).sum()
        iou = float(np.logical_and(pred, gt).sum() / union) if union else 0.0
        ious.append(iou)
    arr = np.asarray(ious, dtype=np.float64)
    result.update({
        "matched_frames": int(len(ious)),
        "mean_iou": float(arr.mean()) if arr.size else float("nan"),
        "median_iou": float(np.median(arr)) if arr.size else float("nan"),
        "p05_iou": float(np.percentile(arr, 5)) if arr.size else float("nan"),
        "p95_iou": float(np.percentile(arr, 95)) if arr.size else float("nan"),
    })
    return result


def _metadata_quality(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return {"selected_confidence": None, "selection_score": None}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = metadata.get("candidates", [])
    selected_prompt = metadata.get("selected_prompt")
    selected = next((item for item in candidates if item.get("prompt") == selected_prompt), {})
    return {
        "selected_confidence": selected.get("target_anchor_confidence"),
        "selection_score": selected.get("metrics", {}).get("selection_score"),
    }


def _selection_key(metrics: dict[str, Any], quality: dict[str, Any]) -> tuple[float, float, float]:
    valid_rate = float(metrics.get("valid_rate") or 0.0)
    confidence = float(quality.get("selected_confidence") or 0.0)
    # The S2 acceptance objective is mask coverage across the 30-frame window,
    # so valid_rate dominates; confidence is only a tie-breaker.
    return (valid_rate, confidence, float(metrics.get("frames_with_mask") or 0))


def _run_one_box(
    run_dir: Path, rank: int, box: list[float], prompt: str, args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output_dir = run_dir / "segmentation_s2_boxes" / f"{rank:02d}"
    if (output_dir / "metadata.json").is_file() and not args.force:
        print(f"[skip s2 box] {run_dir.name} rank={rank}", flush=True)
        return output_dir, _mask_metrics_for(run_dir, str(output_dir.relative_to(run_dir))), _metadata_quality(output_dir)
    cmd = [
        "scripts/run_model_adapter.sh",
        "v2s-sam3",
        "video_to_spider.adapters.sam3",
        "--run-dir", str(run_dir),
        "--checkpoint", str(CHECKPOINT),
        "--max-candidates", "1",
        "--max-instances", "1",
        "--output-dir", str(output_dir),
        "--overwrite",
        "--no-recovery",
        "--prompt", prompt,
        "--box", *[str(value) for value in box],
    ]
    print(f"[s2 box] {run_dir.name} rank={rank} prompt={prompt!r}", flush=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONNOUSERSITE"] = "1"
    started = time.time()
    log_path = RUNS_ROOT / "logs" / f"phase0_s2_{run_dir.name}_box{rank}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(item) for item in cmd],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout)
        log.write(f"\n[returncode] {proc.returncode}\n")
    elapsed = time.time() - started
    print(f"[s2 box] {run_dir.name} rank={rank}: rc={proc.returncode} {elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"S2 box run failed rc={proc.returncode}:\n{tail}")
    subdir = str(output_dir.relative_to(run_dir))
    return output_dir, _mask_metrics_for(run_dir, subdir), _metadata_quality(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    rows = phase0._rows()
    box_rows = json.loads(BOX_PATH.read_text(encoding="utf-8"))
    by_key = {
        (row["prototype"], row["sequence"], row["window_start"], row["window_end"]): row
        for row in box_rows
    }
    results: list[dict[str, Any]] = []
    for row in rows:
        key = (row["prototype"], row["sequence"], row["window"][0], row["window"][1])
        box_row = by_key.get(key)
        if box_row is None or box_row.get("selected_box_xywh") is None:
            print(f"[s2] no box for {row['prototype']} {row['window']}", flush=True)
            continue
        if args.only_failed and row["prototype"] not in FAILED_PROTOTYPES:
            continue
        run_dir = phase0._object_run(row)
        if run_dir is None:
            print(f"[missing run] {row['prototype']} {row['window']}", flush=True)
            continue
        prompts = box_row["prompts"]
        prompt = prompts[0] if prompts else phase0._s1_prompts(row["prototype"], [])[0]
        detections = box_row["detections"][: max(1, min(args.max_boxes, len(box_row["detections"])))]
        candidates: list[dict[str, Any]] = []
        for detection in detections:
            rank = int(detection["rank"])
            box = [float(value) for value in detection["xywh_normalized"]]
            output_dir, metrics, quality = _run_one_box(run_dir, rank, box, prompt, args)
            candidates.append({
                "rank": rank,
                "box_xywh": box,
                "box_xyxy": detection["xyxy"],
                "confidence": detection["confidence"],
                "phrase": detection["phrase"],
                "output_dir": str(output_dir),
                "metrics": metrics,
                "quality": quality,
            })
        best = max(candidates, key=lambda item: _selection_key(item["metrics"], item["quality"]))
        s0 = phase0._mask_metrics(run_dir) if (run_dir / "segmentation" / "object_masks.npz").is_file() else None
        s1 = phase0._mask_metrics_s1(run_dir) if (run_dir / "segmentation_s1" / "object_masks.npz").is_file() else None
        entry = {
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
            "selected_box_rank": best["rank"],
            "selected_box_xywh": best["box_xywh"],
            "selected_box_xyxy": best["box_xyxy"],
            "selected_box_confidence": best["confidence"],
            "selected_box_phrase": best["phrase"],
            "s2": best["metrics"],
            "candidates": candidates,
        }
        if s0 is not None:
            entry["s0"] = s0
            entry["s0_valid_rate"] = s0["valid_rate"]
            entry["s0_mean_iou"] = s0["mean_iou"]
        if s1 is not None:
            entry["s1"] = s1
            entry["s1_valid_rate"] = s1["valid_rate"]
            entry["s1_mean_iou"] = s1["mean_iou"]
        entry.update({
            "s2_valid_rate": best["metrics"]["valid_rate"],
            "s2_mean_iou": best["metrics"]["mean_iou"],
            "s2_median_iou": best["metrics"]["median_iou"],
        })
        results.append(entry)
        _emit(results)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def _emit(results: list[dict[str, Any]]) -> None:
    OUTPUT_SUMMARY.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    headers = [
        "prototype", "sequence", "window_start", "window_end",
        "s0_valid_rate", "s0_mean_iou",
        "s1_valid_rate", "s1_mean_iou",
        "s2_valid_rate", "s2_mean_iou", "s2_median_iou",
        "delta_s1_s2_valid_rate", "delta_s1_s2_mean_iou",
        "selected_box_rank", "selected_box_confidence", "selected_box_phrase",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in results:
            values = []
            for header in headers:
                if header == "delta_s1_s2_valid_rate":
                    values.append(str(_delta(row.get("s1_valid_rate"), row.get("s2_valid_rate"))))
                elif header == "delta_s1_s2_mean_iou":
                    values.append(str(_delta(row.get("s1_mean_iou"), row.get("s2_mean_iou"))))
                else:
                    values.append(str(row.get(header, "")))
            handle.write(",".join(values) + "\n")


def _delta(before: Any, after: Any) -> Any:
    try:
        before_f = float(before)
        after_f = float(after)
        if np.isfinite(before_f) and np.isfinite(after_f):
            return after_f - before_f
    except (TypeError, ValueError):
        pass
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
