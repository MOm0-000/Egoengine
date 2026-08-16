#!/usr/bin/env python3
"""Run the ADT P4 RGB object branch and P5 GT replacement across pilot rows.

This is an operational benchmark driver, not a production pipeline change.
It reads the existing ``runs/adt_depth_benchmark_summary.json`` (which already
contains the selected sequence/object/frame window), prepares the RGB pinhole
object branch when needed, runs the standard SAM3 / SAM3D / FoundationPose
adapters in their fixed environments, then runs P5 depth/mask/mesh
replacements. Existing artifacts are skipped unless ``--force`` is given.

GPU policy: only ``CUDA_VISIBLE_DEVICES`` is set; no process is killed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
ADT_ROOT = Path("/data_all/zzx/egoengine/adt_data")
OBJECT_ROOT = Path("/data_all/zzx/egoengine/adt_object_library")
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
STATE_PATH = RUNS_ROOT / "adt_p4_p5_batch_state.json"
LOG_ROOT = RUNS_ROOT / "logs"

V2S_CORE_PY = Path("/home/zzx/miniconda3/envs/v2s-core/bin/python")
SAM3_CHECKPOINT = REPO_ROOT / "third_party/sam3/checkpoints/sam3.1_multiplex.pt"
SAM3D_CONFIG = REPO_ROOT / "third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml"
FOUNDATIONPOSE_ROOT = REPO_ROOT / "third_party/FoundationPose"

GENERIC_KEYWORDS = {
    "BookDeepLearning": ["book", "deep learning book"],
    "WoodenBowl": ["bowl", "wooden bowl"],
    "WoodenSpoon": ["spoon", "wooden spoon"],
    "BlackCeramicMug": ["mug", "ceramic mug"],
    "WhiteLiddedTrashBin": ["trash bin", "white trash bin"],
    "StepStool": ["stool", "step stool"],
    "Flask": ["flask", "metal flask"],
    "DinoToy": ["dinosaur toy", "toy dinosaur"],
}


def _slug_keywords(prototype: str) -> list[str]:
    keywords = list(GENERIC_KEYWORDS.get(prototype, []))
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", prototype).lower()
    spaced = re.sub(r"[_]+", " ", spaced)
    keywords.append(prototype)
    if spaced and spaced != prototype.lower() and spaced not in keywords:
        keywords.append(spaced)
    return keywords


def _existing_object_run(row: dict[str, Any]) -> Path | None:
    prototype = row["prototype"]
    start, end = row["window"]
    suffix = f"{prototype}_rgbobject_f{start}_{end}"
    matches = sorted(RUNS_ROOT.glob(f"adt_*_{suffix}"))
    if matches:
        return matches[0]
    source = Path(row["run_dir"])
    candidate = RUNS_ROOT / f"{source.name}_rgbobject"
    return candidate if candidate.is_dir() else None


def _run(cmd: list[str], *, label: str, state: dict[str, Any]) -> dict[str, Any]:
    print(f"[{label}] + {' '.join(str(c) for c in cmd)}", flush=True)
    log_path = LOG_ROOT / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "6"
    environment["PYTHONNOUSERSITE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout)
        log.write(f"\n[returncode] {proc.returncode}\n")
    elapsed = time.time() - started
    result = {
        "label": label,
        "command": [str(c) for c in cmd],
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
    }
    state.setdefault("steps", []).append(result)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-30:])
        raise RuntimeError(f"{label} failed rc={proc.returncode}:\n{tail}")
    print(f"[{label}] ok {elapsed:.1f}s", flush=True)
    return result


def _artifact_ready(path: Path) -> bool:
    if path.suffix == ".npz":
        return path.is_file()
    return path.exists()


def _prepare_row(row: dict[str, Any], object_run: Path, force: bool, state: dict[str, Any]) -> bool:
    if object_run.is_dir() and any(object_run.iterdir()) and not force:
        return False
    seq_dir = ADT_ROOT / row["sequence"]
    prepared_dir = ADT_ROOT / row["sequence"] / (
        f"stereo_prepared_{row['prototype']}_f{row['window'][0]}_{row['window'][1]}"
    )
    source_run = Path(row["run_dir"])
    if not (seq_dir / "vrs_files/video.vrs").is_file():
        raise RuntimeError(f"sequence not ready: {seq_dir}")
    if not (prepared_dir / "adt_stereo_prepare.json").is_file():
        raise RuntimeError(f"prepared_dir missing: {prepared_dir}")
    cmd = [
        V2S_CORE_PY,
        "-m",
        "video_to_spider.benchmark.adt_rgb_object",
        "--sequence-dir", seq_dir,
        "--prepared-dir", prepared_dir,
        "--source-run-dir", source_run,
        "--output-run-dir", object_run,
        "--object-uid", str(row["object_uid"]),
        "--gt-output-dir", object_run / "evaluation/adt_rgb_gt",
    ]
    for keyword in _slug_keywords(row["prototype"]):
        cmd.extend(["--object-keyword", keyword])
    if force:
        cmd.append("--overwrite")
    _run(cmd, label=f"{object_run.name}_prepare", state=state)
    return True


def _run_sam3(object_run: Path, force: bool, state: dict[str, Any]) -> bool:
    masks = object_run / "segmentation/object_masks.npz"
    if _artifact_ready(masks) and not force:
        return False
    cmd = [
        str(REPO_ROOT / "scripts/run_model_adapter.sh"),
        "v2s-sam3",
        "video_to_spider.adapters.sam3",
        "--run-dir", object_run,
        "--checkpoint", SAM3_CHECKPOINT,
        "--max-candidates", "12",
        "--max-instances", "2",
        "--overwrite",
    ]
    _run(cmd, label=f"{object_run.name}_sam3", state=state)
    return True


def _run_sam3d(object_run: Path, force: bool, state: dict[str, Any]) -> bool:
    ranking = object_run / "mesh_proposals/mesh_ranking.json"
    if ranking.is_file() and not force:
        return False
    cmd = [
        str(REPO_ROOT / "scripts/run_model_adapter.sh"),
        "v2s-sam3d",
        "video_to_spider.adapters.sam3d_objects",
        "--run-dir", object_run,
        "--config-path", SAM3D_CONFIG,
        "--seeds", "45", "46", "47",
        "--max-keyframes", "1",
        "--max-proposals", "3",
        "--low-vram",
        "--moge-resolution-level", "4",
        "--max-slat-coords", "20000",
        "--overwrite",
    ]
    _run(cmd, label=f"{object_run.name}_sam3d", state=state)
    return True


def _run_foundationpose(object_run: Path, force: bool, state: dict[str, Any]) -> bool:
    tracking = object_run / "object_tracking/foundationpose_raw.npz"
    if _artifact_ready(tracking) and not force:
        return False
    cmd = [
        str(REPO_ROOT / "scripts/run_model_adapter.sh"),
        "v2s-foundationpose",
        "video_to_spider.adapters.foundationpose",
        "--run-dir", object_run,
        "--foundationpose-root", FOUNDATIONPOSE_ROOT,
        "--max-candidates", "3",
        "--screening-radius", "1",
        "--register-iter", "2",
        "--track-iter", "1",
        "--max-input-side", "640",
        "--overwrite",
    ]
    _run(cmd, label=f"{object_run.name}_foundationpose", state=state)
    return True


def _run_p5_variant(
    object_run: Path, row: dict[str, Any], variant: str, force: bool, state: dict[str, Any]
) -> bool:
    variant_run = object_run.parent / f"{object_run.name}_p5_{variant}"
    metrics = variant_run / "p5_metrics.json" if variant != "f0" else object_run / "p5_metrics.json"
    if metrics.is_file() and not force:
        return False
    seq_dir = ADT_ROOT / row["sequence"]
    prepared_dir = seq_dir / f"stereo_prepared_{row['prototype']}_f{row['window'][0]}_{row['window'][1]}"
    cmd = [
        V2S_CORE_PY,
        "scripts/run_adt_gt_replacement.py",
        "--base-run", object_run,
        "--sequence-dir", seq_dir,
        "--prepared-dir", prepared_dir,
        "--object-uid", str(row["object_uid"]),
        "--prototype", row["prototype"],
        "--variant", variant,
        "--gpu", "6",
        "--max-input-side", "640",
        "--overwrite",
    ]
    _run(cmd, label=f"{object_run.name}_p5_{variant}", state=state)
    return True


def _has_gt_mesh(prototype: str) -> bool:
    normalized = prototype.replace("_", "-").replace(" ", "-").lower()
    for directory in OBJECT_ROOT.iterdir():
        if directory.is_dir() and (
            directory.name.lower() == normalized or normalized in directory.name.lower()
        ):
            if (directory / "3d-asset.glb").is_file():
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", action="append", dest="only_runs", default=[])
    args = parser.parse_args()

    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    state: dict[str, Any] = {
        "schema_version": "1.0",
        "force": args.force,
        "rows": {},
    }
    if STATE_PATH.is_file():
        previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if previous.get("force") != args.force:
            previous["rows"] = {}
        state.update(previous)

    for row in rows:
        source_name = Path(row["run_dir"]).name
        if args.only_runs and source_name not in args.only_runs:
            continue
        row_key = f"{row['sequence']}__{row['prototype']}__f{row['window'][0]}_{row['window'][1]}"
        print(f"\n===== {row_key} =====", flush=True)
        object_run = _existing_object_run(row)
        if object_run is None:
            source_run = Path(row["run_dir"])
            object_run = RUNS_ROOT / f"{source_run.name}_rgbobject"
        row_state: dict[str, Any] = {
            "sequence": row["sequence"],
            "prototype": row["prototype"],
            "object_uid": row["object_uid"],
            "window": row["window"],
            "object_run": str(object_run),
            "steps": [],
        }
        state["rows"][row_key] = row_state
        state["_current"] = row_key
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        try:
            _prepare_row(row, object_run, args.force, row_state)
            _run_sam3(object_run, args.force, row_state)
            _run_sam3d(object_run, args.force, row_state)
            _run_foundationpose(object_run, args.force, row_state)
            _run_p5_variant(object_run, row, "f0", args.force, row_state)
            for variant in ("depth_gt", "mask_gt", "mesh_gt"):
                if variant == "mesh_gt" and not _has_gt_mesh(row["prototype"]):
                    print(f"[skip] {variant}: no ADT object model for {row['prototype']}", flush=True)
                    row_state.setdefault("skipped_variants", []).append(variant)
                    continue
                try:
                    _run_p5_variant(object_run, row, variant, args.force, row_state)
                except Exception as error:
                    print(f"[skip] {variant}: {error}", flush=True)
                    row_state.setdefault("failed_variants", []).append(
                        {"variant": variant, "error": f"{type(error).__name__}: {error}"}
                    )
            row_state["status"] = "ok"
        except Exception as error:
            row_state["status"] = "error"
            row_state["error"] = f"{type(error).__name__}: {error}"
            print(f"[row-error] {error}", flush=True)
        finally:
            STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[all-rows-finished]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
