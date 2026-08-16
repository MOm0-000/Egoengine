#!/usr/bin/env python3
"""S2 multi-anchor: run SAM3 once per Grounding DINO anchor, then merge masks."""

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
BOX_PATH = RUNS_ROOT / "adt_phase0_s2_multianchor_boxes.json"
CHECKPOINT = REPO_ROOT / "third_party/sam3/checkpoints/sam3.1_multiplex.pt"
OUTPUT_SUMMARY = RUNS_ROOT / "adt_phase0_sam3_s2_multianchor_summary.json"
OUTPUT_CSV = RUNS_ROOT / "adt_phase0_sam3_s2_multianchor_matrix.csv"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_adt_phase0_sam3_ablation as phase0  # noqa: E402
import run_adt_phase0_sam3_s2 as s2  # noqa: E402


def _run_one_anchor(
    run_dir: Path, anchor: int, box: list[float], prompt: str, args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output_dir = run_dir / "segmentation_s2_multianchor" / f"{anchor:02d}"
    if (output_dir / "metadata.json").is_file() and not args.force:
        subdir = str(output_dir.relative_to(run_dir))
        return output_dir, s2._mask_metrics_for(run_dir, subdir), s2._metadata_quality(output_dir)
    cmd = [
        "scripts/run_model_adapter.sh",
        "v2s-sam3",
        "video_to_spider.adapters.sam3",
        "--run-dir", str(run_dir),
        "--checkpoint", str(CHECKPOINT),
        "--max-candidates", "1",
        "--max-instances", "4",
        "--output-dir", str(output_dir),
        "--overwrite",
        "--no-recovery",
        "--anchor", str(anchor),
        "--prompt", prompt,
        "--box", *[str(value) for value in box],
    ]
    print(f"[s2 ma] {run_dir.name} anchor={anchor} prompt={prompt!r}", flush=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONNOUSERSITE"] = "1"
    started = time.time()
    log_path = RUNS_ROOT / "logs" / f"phase0_s2ma_{run_dir.name}_a{anchor}.log"
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
    print(f"[s2 ma] {run_dir.name} anchor={anchor}: rc={proc.returncode} {time.time() - started:.1f}s", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"S2 multianchor run failed rc={proc.returncode}:\n{tail}")
    subdir = str(output_dir.relative_to(run_dir))
    return output_dir, s2._mask_metrics_for(run_dir, subdir), s2._metadata_quality(output_dir)


def _merge_artifacts(run_dir: Path, candidate_dirs: list[Path]) -> Path:
    merged_dir = run_dir / "segmentation_s2_multianchor_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    frame_indices = None
    timestamps = None
    for candidate_dir in candidate_dirs:
        path = candidate_dir / "object_masks.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as artifact:
            frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
            timestamps = np.asarray(artifact["timestamps_s"], dtype=np.float64)
            arrays.append((
                np.asarray(artifact["masks"], dtype=bool),
                np.asarray(artifact["valid"], dtype=bool),
                np.asarray(artifact["confidence"], dtype=np.float32),
            ))
    if frame_indices is None or not arrays:
        raise RuntimeError(f"no valid candidate masks for {run_dir}")
    merged_masks = np.zeros_like(arrays[0][0], dtype=bool)
    merged_valid = np.zeros_like(arrays[0][1], dtype=bool)
    merged_confidence = np.zeros_like(arrays[0][2], dtype=np.float32)
    for masks, valid, confidence in arrays:
        masked = masks & valid[:, None, None]
        merged_masks |= masked
        merged_valid |= valid
        merged_confidence = np.maximum(merged_confidence, np.where(valid[:, None, None], confidence[:, None, None], 0.0))
    object_ids = np.where(merged_valid, 0, -1).astype(np.int64)
    np.savez_compressed(
        merged_dir / "object_masks.npz",
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        masks=merged_masks.astype(np.uint8),
        valid=merged_valid,
        confidence=merged_confidence,
        object_ids=object_ids,
    )
    return merged_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--skip-prototypes", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    skip_prototypes = {item.strip() for item in args.skip_prototypes.split(",") if item.strip()}

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
        if box_row is None or not box_row.get("anchors"):
            continue
        if args.only_failed and row["prototype"] not in {"BlackCeramicMug", "Flask", "StepStool"}:
            continue
        if row["prototype"] in skip_prototypes:
            continue
        run_dir = phase0._object_run(row)
        if run_dir is None:
            continue
        prompts = box_row["prompts"]
        prompt = prompts[0] if prompts else phase0._s1_prompts(row["prototype"], [])[0]
        candidates: list[dict[str, Any]] = []
        candidate_dirs: list[Path] = []
        for anchor_item in box_row["anchors"]:
            anchor = int(anchor_item["frame"])
            box = [float(value) for value in anchor_item["xywh_normalized"]]
            try:
                output_dir, metrics, quality = _run_one_anchor(run_dir, anchor, box, prompt, args)
            except Exception as error:
                print(f"[s2 ma skip] {run_dir.name} anchor={anchor}: {type(error).__name__}: {error}", flush=True)
                candidates.append({
                    "anchor": anchor,
                    "source_frame": anchor_item.get("source_frame"),
                    "box_xywh": box,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            candidates.append({
                "anchor": anchor,
                "source_frame": anchor_item.get("source_frame"),
                "box_xywh": box,
                "box_xyxy": anchor_item.get("xyxy"),
                "confidence": anchor_item.get("confidence"),
                "phrase": anchor_item.get("phrase"),
                "output_dir": str(output_dir),
                "metrics": metrics,
                "quality": quality,
            })
            if (output_dir / "object_masks.npz").is_file():
                candidate_dirs.append(output_dir)
        merged_dir = _merge_artifacts(run_dir, candidate_dirs)
        merged_metrics = s2._mask_metrics_for(run_dir, str(merged_dir.relative_to(run_dir)))
        s0 = phase0._mask_metrics(run_dir) if (run_dir / "segmentation" / "object_masks.npz").is_file() else None
        s1 = phase0._mask_metrics_s1(run_dir) if (run_dir / "segmentation_s1" / "object_masks.npz").is_file() else None
        entry = {
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
            "merged_output_dir": str(merged_dir),
            "s2_multianchor": merged_metrics,
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
            "s2_valid_rate": merged_metrics["valid_rate"],
            "s2_mean_iou": merged_metrics["mean_iou"],
            "s2_median_iou": merged_metrics["median_iou"],
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
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in results:
            values = []
            for header in headers:
                if header == "delta_s1_s2_valid_rate":
                    values.append(str(s2._delta(row.get("s1_valid_rate"), row.get("s2_valid_rate"))))
                elif header == "delta_s1_s2_mean_iou":
                    values.append(str(s2._delta(row.get("s1_mean_iou"), row.get("s2_mean_iou"))))
                else:
                    values.append(str(row.get(header, "")))
            handle.write(",".join(values) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
