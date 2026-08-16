#!/usr/bin/env python3
"""P5 G1: GT calibration rescue on the full-upstream stereo branch.

This isolates the production camera trajectory / rectified calibration from the
object Digital Twin. It clones a source stereo run, replaces only
``calibration/T_world_camera.npy`` with ADT GT ``aria_trajectory.csv`` expressed
in the same rectified-left frame, and reruns the strict sequence optimizer with
``--contact-similarity-mode validate_only``. FoundationPose raw camera-frame
tracking is intentionally not rerun because it does not consume camera
extrinsics; the rescue is measured on the full-upstream object trajectory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNS_ROOT = REPO_ROOT / "runs"
ADT_ROOT = Path("/data_all/zzx/egoengine/adt_data")
OBJECT_ROOT = Path("/data_all/zzx/egoengine/adt_object_library")
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
OUTPUT_PATH = RUNS_ROOT / "adt_calibration_rescue_summary.json"
MATRIX_PATH = RUNS_ROOT / "calibration_rescue_matrix.csv"
LOG_ROOT = RUNS_ROOT / "logs"
V2S_CORE_PY = Path("/home/zzx/miniconda3/envs/v2s-core/bin/python")


def _load_sequence_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "adt_sequence_benchmark_helpers",
        REPO_ROOT / "scripts" / "run_adt_sequence_benchmark.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEQ = _load_sequence_module()
P5 = SEQ.P5


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove(target)
    target.symlink_to(source, target_is_directory=True)


def _symlink_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove(target)
    target.symlink_to(source)


def _clone_with_gt_calibration(source: Path, target: Path, gt_cameras: np.ndarray) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for directory in (
        "frames", "depth", "segmentation", "mesh_proposals",
        "object_tracking", "input", "evaluation",
    ):
        if (source / directory).exists():
            _symlink_dir(source / directory, target / directory)

    hands_source = source / "hands"
    hands_target = target / "hands"
    hands_target.mkdir(parents=True, exist_ok=True)
    for name in ("wilor_raw.npz", "wilor_overlay.mp4", "metadata.json"):
        source_file = hands_source / name
        if source_file.exists():
            _symlink_file(source_file, hands_target / name)
    if (source / "hands_right").exists():
        _symlink_dir(source / "hands_right", target / "hands_right")

    calibration_target = target / "calibration"
    calibration_target.mkdir(parents=True, exist_ok=True)
    for child in (source / "calibration").iterdir():
        if child.is_file():
            shutil.copy2(child, calibration_target / child.name)
    np.save(calibration_target / "T_world_camera.npy", gt_cameras.astype(np.float32))

    manifest = source / "manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, target / "manifest.json")
    return target


def _prepared_dir(row: dict[str, Any]) -> Path:
    sequence = row["sequence"]
    prototype = row["prototype"]
    start, end = row["window"]
    path = ADT_ROOT / sequence / f"stereo_prepared_{prototype}_f{start}_{end}"
    if path.is_dir():
        return path
    matches = list((ADT_ROOT / sequence).glob(f"stereo_prepared_{prototype}_f*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"prepared_dir missing for {sequence}/{prototype}")


def _run_sequence(clone: Path) -> tuple[int, Path]:
    log_path = LOG_ROOT / f"{clone.name}_calibration_gt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(V2S_CORE_PY),
        "-m", "video_to_spider.optimization.sequence",
        "--run-dir", str(clone),
        "--contact-similarity-mode", "validate_only",
        "--overwrite",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
        log.write(f"\n[returncode] {proc.returncode}\n")
    return proc.returncode, log_path


def _sequence_record(clone: Path, returncode: int, log_path: Path) -> dict[str, Any]:
    metrics = clone / "optimization/optimization_metrics.json"
    aligned = clone / "optimization/aligned_trajectory.npz"
    if not metrics.is_file():
        reason = "no_optimization_artifact"
        if returncode != 0 and log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = [line.strip() for line in text.splitlines() if "RuntimeError:" in line or "Error:" in line]
            reason = " | ".join(lines[-3:]) if lines else reason
        return {"status": "failed_before_sequence_artifacts", "reason": reason, "has_aligned": aligned.is_file()}
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    optimization = payload.get("optimization", {})
    qc = payload.get("quality_control", {})
    contact_mode = optimization.get("contact_similarity_mode")
    export_ready = bool(qc.get("export_ready", False))
    has_aligned = aligned.is_file()
    status = "completed" if has_aligned and export_ready else "export_qc_failed" if has_aligned else "no_aligned_artifact"
    return {
        "status": status,
        "reason": "export_ready=false" if status == "export_qc_failed" else None,
        "contact_similarity_mode": contact_mode,
        "strict_shared_metric": contact_mode == "validate_only",
        "hand_roles": payload.get("hands", {}).get("roles"),
        "has_aligned": has_aligned,
    }


def _aligned_transforms(clone: Path, gt_cameras: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    metrics = json.loads((clone / "optimization/optimization_metrics.json").read_text(encoding="utf-8"))
    aligned = np.load(clone / "optimization/aligned_trajectory.npz", allow_pickle=False)
    T_sim_world = np.asarray(metrics["T_sim_world"], dtype=np.float64)
    T_sim_world_inv = np.linalg.inv(T_sim_world)
    T_world_camera_inv = np.linalg.inv(gt_cameras)
    T_sim_object = np.asarray(aligned["T_sim_object"], dtype=np.float64)
    valid = np.asarray(aligned["valid_object"], dtype=bool).reshape(-1)
    scale = float(np.asarray(aligned["object_scale_to_m"], dtype=np.float64).reshape(-1)[0])
    T_world_object = np.einsum("ij,tjk->tik", T_sim_world_inv, T_sim_object[:, 0])
    T_camera_object = T_world_camera_inv @ T_world_object
    return T_camera_object, valid, scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    args = parser.parse_args()

    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in rows:
        source = Path(row["run_dir"])
        required = [
            source / "object_tracking/foundationpose_raw.npz",
            source / "object_tracking/selected_mesh.json",
            source / "hands/wilor_raw.npz",
            source / "segmentation/object_masks.npz",
            source / "depth/depth_gate.json",
        ]
        if not all(path.exists() for path in required):
            continue
        sequence_dir = ADT_ROOT / row["sequence"]
        prepared_dir = _prepared_dir(row)
        timestamps, gt_cameras, _ = SEQ._gt_camera_and_object(
            sequence_dir, prepared_dir, str(row["object_uid"])
        )
        clone = RUNS_ROOT / f"{source.name}_p5_calibration_gt"
        _clone_with_gt_calibration(source, clone, gt_cameras)
        returncode, log = _run_sequence(clone)
        seq_record = _sequence_record(clone, returncode, log)

        raw_npz = np.load(source / "object_tracking/foundationpose_raw.npz", allow_pickle=False)
        raw_valid = np.asarray(raw_npz["valid"], dtype=bool)
        raw_transforms = np.asarray(raw_npz["T_camera_object"], dtype=np.float64)
        selected = json.loads((source / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
        raw_scale = float(selected["scale_to_m"])
        raw_metrics = SEQ._pose_metrics(
            transforms=raw_transforms, valid=raw_valid, run_dir=source,
            sequence_dir=sequence_dir, prepared_dir=prepared_dir,
            object_uid=str(row["object_uid"]), prototype=row["prototype"], pred_scale=raw_scale,
        )

        aligned_metrics = None
        scale_delta = math.nan
        if seq_record.get("has_aligned"):
            try:
                aligned_transforms, aligned_valid, aligned_scale = _aligned_transforms(clone, gt_cameras)
                aligned_metrics = SEQ._pose_metrics(
                    transforms=aligned_transforms, valid=aligned_valid, run_dir=source,
                    sequence_dir=sequence_dir, prepared_dir=prepared_dir,
                    object_uid=str(row["object_uid"]), prototype=row["prototype"], pred_scale=aligned_scale,
                )
                scale_delta = (aligned_scale - raw_scale) / max(abs(raw_scale), 1e-12)
            except Exception as error:
                aligned_metrics = {"status": "evaluation_failed", "error": f"{type(error).__name__}: {error}"}

        record = {
            "sequence": row["sequence"],
            "prototype": row["prototype"],
            "object_uid": row["object_uid"],
            "window": row["window"],
            "source_run": str(source),
            "clone_run": str(clone),
            "sequence_returncode": returncode,
            "sequence": seq_record,
            "raw": raw_metrics,
            "aligned": aligned_metrics,
            "scale_delta": scale_delta,
        }
        records.append(record)

    summary = {"schema_version": "1.0", "stage": "P5 G1 GT calibration rescue", "rows": records}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "run", "sequence_status", "raw_centroid_median_m", "aligned_centroid_median_m",
            "centroid_delta_m", "raw_add_s_median_m", "aligned_add_s_median_m", "scale_delta",
        ])
        for record in records:
            raw = record["raw"]
            aligned = record.get("aligned") or {}
            writer.writerow([
                Path(record["source_run"]).name,
                record["sequence"]["status"],
                raw.get("centroid_translation_error_m", {}).get("median", math.nan),
                aligned.get("centroid_translation_error_m", {}).get("median", math.nan),
                (aligned.get("centroid_translation_error_m", {}).get("median", math.nan)
                 - raw.get("centroid_translation_error_m", {}).get("median", math.nan))
                if aligned.get("centroid_translation_error_m") else math.nan,
                raw.get("add_s_m", {}).get("median", math.nan),
                aligned.get("add_s_m", {}).get("median", math.nan),
                record.get("scale_delta", math.nan),
            ])
    print(f"summary -> {args.output}")
    print(f"matrix   -> {args.matrix}")
    print(json.dumps(records, indent=2, ensure_ascii=False)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
