#!/usr/bin/env python3
"""Phase 0 S0/S1 SAM3 object-mask ablation on the 10 ADT benchmark samples.

S0 uses the already-generated instruction-derived SAM3 segmentation artifacts.
S1 runs SAM 3.1 again with an expanded text-prompt ensemble derived from the
prototype class name, its super-category, and short appearance descriptors.

The same 10 rows, frame windows, and downstream artifact schema are used as the
existing ``runs/adt_depth_benchmark_summary.json``. GT segmentation is only used
for reporting IoU; it is not passed to SAM3.
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
CHECKPOINT = REPO_ROOT / "third_party/sam3/checkpoints/sam3.1_multiplex.pt"
OUTPUT_SUMMARY = RUNS_ROOT / "adt_phase0_sam3_s0_s1_summary.json"
OUTPUT_CSV = RUNS_ROOT / "adt_phase0_sam3_s0_s1_matrix.csv"

PROMPT_ENSEMBLE: dict[str, list[str]] = {
    "WhiteLiddedTrashBin": [
        "trash bin",
        "white trash bin with lid",
        "cylindrical waste bin",
        "garbage can",
        "household container",
    ],
    "WoodenBowl": [
        "wooden bowl",
        "round serving bowl",
        "shallow wooden bowl",
        "bowl",
        "kitchen dish",
    ],
    "BookDeepLearning": [
        "book",
        "hardcover book",
        "rectangular book",
        "textbook",
        "reading material",
    ],
    "BlackCeramicMug": [
        "mug",
        "ceramic mug",
        "black coffee mug",
        "cylindrical cup with handle",
        "drinking cup",
    ],
    "WoodenSpoon": [
        "spoon",
        "wooden spoon",
        "long handled cooking spoon",
        "kitchen utensil",
        "stirring spoon",
    ],
    "DinoToy": [
        "dinosaur toy",
        "toy dinosaur",
        "plastic dinosaur figure",
        "animal toy",
        "figurine",
    ],
    "Flask": [
        "flask",
        "metal flask",
        "vacuum flask",
        "thermos",
        "water bottle",
        "cylindrical bottle",
    ],
    "StepStool": [
        "step stool",
        "stool",
        "portable step stool",
        "small step ladder",
        "two step stool",
        "household step",
    ],
}

PROMPT_ENSEMBLE["WoodenBowl"] = list(dict.fromkeys(
    ["wooden bowl", "round serving bowl", "shallow wooden bowl", "bowl", "kitchen dish"]
))
PROMPT_ENSEMBLE["BookDeepLearning"] = list(dict.fromkeys(
    ["book", "hardcover book", "rectangular book", "textbook", "deep learning book", "reading material"]
))


def _s1_prompts(prototype: str, source_candidates: list[str]) -> list[str]:
    merged = list(source_candidates)
    merged.extend(PROMPT_ENSEMBLE.get(prototype, []))
    spaced = prototype
    for token, repl in [
        (r"(?<=[a-z0-9])(?=[A-Z])", " "),
        (r"[_-]+", " "),
    ]:
        import re
        spaced = re.sub(token, repl, spaced)
    merged.append(prototype)
    merged.append(spaced.strip().lower())
    deduped: list[str] = []
    for prompt in merged:
        prompt = " ".join(prompt.split())
        if prompt and prompt not in deduped:
            deduped.append(prompt)
    return deduped[:14]


def _object_run(row: dict[str, Any]) -> Path | None:
    base = Path(row["run_dir"])
    start, end = row["window"]
    candidates = [
        base,
        base.parent / (base.name + "_rgbobject"),
    ]
    candidates.extend(sorted(base.parent.glob(
        f"adt_*_{row['prototype']}_rgbobject_f{start}_{end}"
    )))
    seen: set[str] = set()
    rgb_mask: list[Path] = []
    other_mask: list[Path] = []
    with_source: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "segmentation" / "object_masks.npz").is_file():
            if "_rgbobject" in candidate.name:
                rgb_mask.append(candidate)
            else:
                other_mask.append(candidate)
        elif (candidate / "input" / "source.json").is_file():
            with_source.append(candidate)
    if rgb_mask:
        return rgb_mask[0]
    if other_mask:
        return other_mask[0]
    return with_source[0] if with_source else None


def _gt_dir(run_dir: Path) -> Path:
    candidate = run_dir / "evaluation" / "adt_rgb_gt"
    if (candidate / "adt_gt.zarr").is_dir():
        return candidate
    raise FileNotFoundError(f"no ADT RGB GT for {run_dir}")


def _mask_metrics(run_dir: Path) -> dict[str, Any]:
    import zarr

    masks_path = run_dir / "segmentation" / "object_masks.npz"
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)
    with np.load(masks_path, allow_pickle=False) as artifact:
        pred_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
        pred_masks = np.asarray(artifact["masks"], dtype=bool)
        pred_valid = np.asarray(artifact["valid"], dtype=bool)
    valid_rate = float(np.mean(pred_valid)) if pred_valid.size else float("nan")
    coverage = valid_rate
    result: dict[str, Any] = {
        "valid_rate": valid_rate,
        "coverage": coverage,
        "frames_with_mask": int(np.sum(pred_valid)),
        "frame_count": int(pred_valid.size),
    }
    gt_dir = run_dir / "evaluation" / "adt_rgb_gt"
    gt_path = gt_dir / "adt_gt.zarr"
    if not gt_path.is_dir():
        result.update({
            "matched_frames": None,
            "mean_iou": None,
            "median_iou": None,
            "p05_iou": None,
            "p95_iou": None,
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


def _run_sam3_s1(run_dir: Path, prototype: str, args: argparse.Namespace) -> None:
    source = json.loads((run_dir / "input" / "source.json").read_text(encoding="utf-8"))
    candidates = list(source.get("object_keyword_candidates", []))
    prompts = _s1_prompts(prototype, candidates)
    output_dir = run_dir / "segmentation_s1"
    if (output_dir / "metadata.json").is_file() and not args.force:
        print(f"[skip s1] {run_dir.name}", flush=True)
        return
    cmd = [
        "scripts/run_model_adapter.sh",
        "v2s-sam3",
        "video_to_spider.adapters.sam3",
        "--run-dir", str(run_dir),
        "--checkpoint", str(CHECKPOINT),
        "--max-candidates", str(len(prompts)),
        "--max-instances", "4",
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    for prompt in prompts:
        cmd.extend(["--prompt", prompt])
    print(f"[s1] {run_dir.name}: {len(prompts)} prompts", flush=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONNOUSERSITE"] = "1"
    started = time.time()
    log_path = RUNS_ROOT / "logs" / f"phase0_s1_{run_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout)
        log.write(f"\n[returncode] {proc.returncode}\n")
    elapsed = time.time() - started
    print(f"[s1] {run_dir.name}: rc={proc.returncode} {elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"S1 failed rc={proc.returncode}:\n{tail}")


def _rows() -> list[dict[str, Any]]:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else []
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 ADT rows, got {len(rows)}")
    return rows


def _emit_summary(results: list[dict[str, Any]]) -> None:
    OUTPUT_SUMMARY.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    headers = [
        "prototype", "sequence", "window_start", "window_end",
        "s0_valid_rate", "s0_mean_iou", "s0_median_iou",
        "s1_valid_rate", "s1_mean_iou", "s1_median_iou",
        "delta_valid_rate", "delta_mean_iou",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in results:
            values = [str(row.get(header, "")) for header in headers]
            handle.write(",".join(values) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--s0-only", action="store_true")
    parser.add_argument("--s1-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    rows = _rows()
    results: list[dict[str, Any]] = []
    for row in rows:
        run_dir = _object_run(row)
        if run_dir is None:
            print(f"[missing run] {row['prototype']} {row['window']}", flush=True)
            continue
        entry = {
            "prototype": row["prototype"],
            "sequence": row["sequence"],
            "window_start": row["window"][0],
            "window_end": row["window"][1],
            "run_dir": str(run_dir),
        }
        s0 = _mask_metrics(run_dir)
        entry["s0"] = s0
        if not args.s1_only:
            entry.update({
                "s0_valid_rate": s0["valid_rate"],
                "s0_mean_iou": s0["mean_iou"],
                "s0_median_iou": s0["median_iou"],
            })
        if not args.s0_only:
            try:
                _run_sam3_s1(run_dir, row["prototype"], args)
                s1 = _mask_metrics_s1(run_dir)
                entry["s1"] = s1
                entry.update({
                    "s1_valid_rate": s1["valid_rate"],
                    "s1_mean_iou": s1["mean_iou"],
                    "s1_median_iou": s1["median_iou"],
                })
                if not args.s1_only:
                    entry["delta_valid_rate"] = (
                        float(s1["valid_rate"]) - float(s0["valid_rate"])
                        if np.isfinite(s0["valid_rate"]) and np.isfinite(s1["valid_rate"]) else None
                    )
                    entry["delta_mean_iou"] = (
                        float(s1["mean_iou"]) - float(s0["mean_iou"])
                        if np.isfinite(s0["mean_iou"]) and np.isfinite(s1["mean_iou"]) else None
                    )
            except Exception as error:
                entry["s1_error"] = f"{type(error).__name__}: {error}"
        results.append(entry)
        _emit_summary(results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def _mask_metrics_s1(run_dir: Path) -> dict[str, Any]:
    masks_path = run_dir / "segmentation_s1" / "object_masks.npz"
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
    import zarr
    gt_path = run_dir / "evaluation" / "adt_rgb_gt" / "adt_gt.zarr"
    if not gt_path.is_dir():
        result.update({
            "matched_frames": None,
            "mean_iou": None,
            "median_iou": None,
            "p05_iou": None,
            "p95_iou": None,
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


if __name__ == "__main__":
    raise SystemExit(main())
