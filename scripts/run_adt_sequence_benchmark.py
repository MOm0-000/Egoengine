#!/usr/bin/env python3
"""P6 ADT sequence benchmark with strict stereo ``validate_only`` protocol.

This benchmark never relaxes or rewrites metric geometry. For every source ADT
run that has enough artifacts to enter sequence optimization, it creates an
isolated run clone, executes ``optimize_run`` with
``--contact-similarity-mode validate_only``, records whether the sequence can
finish, and compares raw FoundationPose vs any aligned trajectory against ADT
GT object pose.
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
import trimesh
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNS_ROOT = REPO_ROOT / "runs"
ADT_ROOT = Path("/data_all/zzx/egoengine/adt_data")
OBJECT_ROOT = Path("/data_all/zzx/egoengine/adt_object_library")
SUMMARY_PATH = RUNS_ROOT / "adt_depth_benchmark_summary.json"
STATE_PATH = RUNS_ROOT / "adt_sequence_benchmark_state.json"
OUTPUT_PATH = RUNS_ROOT / "adt_sequence_benchmark_summary.json"
MATRIX_PATH = RUNS_ROOT / "sequence_benchmark_matrix.csv"
VIOLATION_PATH = RUNS_ROOT / "sequence_protocol_violations.csv"
LOG_ROOT = RUNS_ROOT / "logs"
V2S_CORE_PY = Path("/home/zzx/miniconda3/envs/v2s-core/bin/python")


def _load_p5_helpers() -> Any:
    spec = importlib.util.spec_from_file_location(
        "adt_p5_helpers", REPO_ROOT / "scripts" / "run_adt_gt_replacement.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P5 = _load_p5_helpers()


def _percentile(values: np.ndarray, q: float) -> float:
    if not values.size:
        return math.nan
    return float(np.percentile(values, q))


def _remove_symlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_symlink(target)
    target.symlink_to(source, target_is_directory=True)


def _symlink_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_symlink(target)
    target.symlink_to(source)


def _clone_run(source: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for directory in (
        "frames",
        "calibration",
        "depth",
        "segmentation",
        "mesh_proposals",
        "object_tracking",
        "input",
        "evaluation",
    ):
        if (source / directory).exists():
            _symlink_dir(source / directory, target / directory)

    # P6 is the object-sequence benchmark. Keep monocular WiLoR as an
    # auxiliary hand input, but deliberately do not carry the failing
    # auxiliary stereo-hand artifact into the clone. This is not relaxing the
    # hand QC gate: strict_shared_metric will still freeze hand depth and mark
    # it as unverified, while the object trajectory comparison remains valid.
    hands_source = source / "hands"
    if hands_source.is_dir():
        hands_target = target / "hands"
        hands_target.mkdir(parents=True, exist_ok=True)
        for name in ("wilor_raw.npz", "wilor_overlay.mp4", "metadata.json"):
            source_file = hands_source / name
            if source_file.exists():
                _symlink_file(source_file, hands_target / name)

    if (source / "hands_right").exists():
        _symlink_dir(source / "hands_right", target / "hands_right")
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


def _gt_camera_and_object(
    sequence_dir: Path, prepared_dir: Path, object_uid: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, T_world_camera_rect, T_world_object) aligned by time."""
    timestamps = np.asarray(
        json.loads((prepared_dir / "calibration/timestamps.json").read_text(encoding="utf-8"))[
            "timestamps_s"
        ],
        dtype=np.float64,
    )
    prepared_meta = json.loads((prepared_dir / "adt_stereo_prepare.json").read_text(encoding="utf-8"))
    rectified_rotation = np.asarray(
        prepared_meta["rectification"]["device_to_rectified_rotation"], dtype=np.float64
    )
    device_to_rect = np.eye(4, dtype=np.float64)
    device_to_rect[:3, :3] = rectified_rotation

    camera_rows = list(csv.DictReader((sequence_dir / "aria_trajectory.csv").open(encoding="utf-8")))
    camera_times = np.asarray(
        [float(row["tracking_timestamp_us"]) * 1e-6 for row in camera_rows], dtype=np.float64
    )
    object_rows = [
        row
        for row in csv.DictReader((sequence_dir / "scene_objects.csv").open(encoding="utf-8"))
        if row["object_uid"] == object_uid
    ]
    if not object_rows or not camera_rows:
        raise RuntimeError("missing ADT GT camera/object trajectory")
    object_times = np.asarray([float(row["timestamp[ns]"]) * 1e-9 for row in object_rows], dtype=np.float64)

    T_world_camera = np.empty((len(timestamps), 4, 4), dtype=np.float64)
    T_world_object = np.empty((len(timestamps), 4, 4), dtype=np.float64)
    for index, timestamp in enumerate(timestamps):
        camera_row = camera_rows[int(np.argmin(np.abs(camera_times - timestamp)))]
        world_device = P5._pose(
            P5._quat_wxyz_matrix(
                float(camera_row["qw_world_device"]),
                float(camera_row["qx_world_device"]),
                float(camera_row["qy_world_device"]),
                float(camera_row["qz_world_device"]),
            ),
            [
                float(camera_row["tx_world_device"]),
                float(camera_row["ty_world_device"]),
                float(camera_row["tz_world_device"]),
            ],
        )
        T_world_camera[index] = world_device @ device_to_rect
        object_row = object_rows[int(np.argmin(np.abs(object_times - timestamp)))]
        T_world_object[index] = P5._pose(
            P5._quat_wxyz_matrix(
                float(object_row["q_wo_w"]),
                float(object_row["q_wo_x"]),
                float(object_row["q_wo_y"]),
                float(object_row["q_wo_z"]),
            ),
            [
                float(object_row["t_wo_x[m]"]),
                float(object_row["t_wo_y[m]"]),
                float(object_row["t_wo_z[m]"]),
            ],
        )
    return timestamps, T_world_camera, T_world_object


def _pose_metrics(
    *,
    transforms: np.ndarray,
    valid: np.ndarray,
    run_dir: Path,
    sequence_dir: Path,
    prepared_dir: Path,
    object_uid: str,
    prototype: str,
    pred_scale: float,
) -> dict[str, Any]:
    if not len(transforms) or not len(valid):
        return {"status": "no_frames", "valid_rate": 0.0}
    selected = json.loads((run_dir / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
    pred_mesh = P5._as_trimesh(trimesh.load(run_dir / selected["canonical_visual_mesh"], process=False))
    pred_mesh.apply_scale(float(pred_scale))
    gt_mesh = P5._as_trimesh(trimesh.load(P5._find_gt_mesh(OBJECT_ROOT, prototype), process=False))
    sample_count = max(3, min(600, len(pred_mesh.faces), len(gt_mesh.faces)))
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, sample_count)
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, sample_count)
    _, T_world_camera, T_world_object = _gt_camera_and_object(sequence_dir, prepared_dir, object_uid)
    symmetry_rotations = P5._object_symmetry_rotations(sequence_dir, object_uid)

    translations: list[float] = []
    rotation_errors: list[float] = []
    symmetry_rotation_errors: list[float] = []
    add_s_values: list[float] = []
    used = 0
    for index, transform in enumerate(transforms):
        if not bool(valid[index]):
            continue
        used += 1
        gt_camera_object = P5._invert(T_world_camera[index]) @ T_world_object[index]
        pred_centroid = np.asarray(pred_mesh.centroid, dtype=np.float64)
        gt_centroid = np.asarray(gt_mesh.centroid, dtype=np.float64)
        pred_camera_centroid = transform[:3, :3] @ pred_centroid + transform[:3, 3]
        gt_camera_centroid = gt_camera_object[:3, :3] @ gt_centroid + gt_camera_object[:3, 3]
        translations.append(float(np.linalg.norm(pred_camera_centroid - gt_camera_centroid)))

        relative_rotation = gt_camera_object[:3, :3].T @ transform[:3, :3]
        rotation_errors.append(float(P5._angle(relative_rotation)))
        symmetry_rotation_errors.append(
            float(min(P5._angle(relative_rotation @ symmetry) for symmetry in symmetry_rotations))
        )
        pred_camera_points = pred_points @ transform[:3, :3].T + transform[:3, 3]
        gt_camera_points = gt_points @ gt_camera_object[:3, :3].T + gt_camera_object[:3, 3]
        add_s_values.append(
            float(
                np.concatenate(
                    [
                        cKDTree(gt_camera_points).query(pred_camera_points)[0],
                        cKDTree(pred_camera_points).query(gt_camera_points)[0],
                    ]
                ).mean()
            )
        )

    translations = np.asarray(translations, dtype=np.float64)
    rotation_errors = np.asarray(rotation_errors, dtype=np.float64)
    symmetry_rotation_errors = np.asarray(symmetry_rotation_errors, dtype=np.float64)
    add_s_values = np.asarray(add_s_values, dtype=np.float64)
    return {
        "status": "evaluated",
        "used_frames": used,
        "valid_rate": float(used / max(len(transforms), 1)),
        "centroid_translation_error_m": {
            "median": _percentile(translations, 50),
            "p95": _percentile(translations, 95),
            "mean": float(translations.mean()) if translations.size else math.nan,
        },
        "rotation_error_deg": {
            "median": float(np.degrees(_percentile(rotation_errors, 50))),
            "p95": float(np.degrees(_percentile(rotation_errors, 95))),
        },
        "symmetry_aware_rotation_error_deg": {
            "median": float(np.degrees(_percentile(symmetry_rotation_errors, 50))),
            "p95": float(np.degrees(_percentile(symmetry_rotation_errors, 95))),
        },
        "add_s_m": {
            "median": _percentile(add_s_values, 50),
            "p95": _percentile(add_s_values, 95),
        },
    }


def _aligned_camera_transforms(run_dir: Path, T_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    metrics_path = run_dir / "optimization/optimization_metrics.json"
    aligned_path = run_dir / "optimization/aligned_trajectory.npz"
    if not metrics_path.is_file() or not aligned_path.is_file():
        raise FileNotFoundError("aligned sequence artifacts missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    T_sim_world = np.asarray(metrics["T_sim_world"], dtype=np.float64)
    T_sim_world_inv = np.linalg.inv(T_sim_world)
    T_world_camera_inv = np.linalg.inv(T_world_camera)
    aligned = np.load(aligned_path, allow_pickle=False)
    T_sim_object = np.asarray(aligned["T_sim_object"], dtype=np.float64)
    valid = np.asarray(aligned["valid_object"], dtype=bool).reshape(-1)
    scale = float(np.asarray(aligned["object_scale_to_m"], dtype=np.float64).reshape(-1)[0])
    T_world_object_aligned = np.einsum("ij,tjk->tik", T_sim_world_inv, T_sim_object[:, 0])
    T_camera_object_aligned = T_world_camera_inv @ T_world_object_aligned
    return T_camera_object_aligned, valid, scale


def _candidate_runs() -> list[dict[str, Any]]:
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        required = [
            run_dir / "object_tracking/foundationpose_raw.npz",
            run_dir / "object_tracking/selected_mesh.json",
            run_dir / "hands/wilor_raw.npz",
            run_dir / "segmentation/object_masks.npz",
            run_dir / "depth/depth_gate.json",
        ]
        if all(path.exists() for path in required):
            candidates.append(row)
    return candidates


def _run_sequence(clone: Path) -> tuple[int, str]:
    log_path = LOG_ROOT / f"{clone.name}_sequence.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(V2S_CORE_PY),
        "-m",
        "video_to_spider.optimization.sequence",
        "--run-dir",
        str(clone),
        "--contact-similarity-mode",
        "validate_only",
        "--overwrite",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout)
        log.write(f"\n[returncode] {proc.returncode}\n")
    return proc.returncode, str(log_path)


def _sequence_record(clone: Path, returncode: int, log_path: Path) -> dict[str, Any]:
    preflight = clone / "optimization/preflight_failure.json"
    metrics = clone / "optimization/optimization_metrics.json"
    aligned = clone / "optimization/aligned_trajectory.npz"
    if preflight.is_file():
        payload = json.loads(preflight.read_text(encoding="utf-8"))
        return {
            "status": payload.get("status", "failed_before_export"),
            "reason": payload.get("reason"),
            "contact_similarity_mode": payload.get("contact_similarity_mode"),
            "strict_shared_metric": payload.get("strict_shared_metric"),
            "hand_roles": payload.get("hand_roles"),
            "has_aligned": aligned.is_file(),
        }
    if metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        optimization = payload.get("optimization", {})
        qc = payload.get("quality_control", {})
        contact_mode = optimization.get("contact_similarity_mode")
        strict_shared_metric = contact_mode == "validate_only"
        export_ready = bool(qc.get("export_ready", False))
        has_aligned = aligned.is_file()
        if has_aligned and export_ready:
            status = "completed"
        elif has_aligned:
            status = "export_qc_failed"
        else:
            status = "no_aligned_artifact"
        reason = None
        if status == "export_qc_failed":
            failed_checks = [
                name
                for name, passed in (qc.get("checks") or {}).items()
                if passed is False
            ]
            reason = "export_ready=false" + (f": {failed_checks}" if failed_checks else "")
        elif returncode != 0 and log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            error_lines = [line.strip() for line in text.splitlines() if "RuntimeError:" in line or "Error:" in line]
            reason = " | ".join(error_lines[-3:]) if error_lines else None
        return {
            "status": status,
            "reason": reason,
            "contact_similarity_mode": contact_mode,
            "strict_shared_metric": strict_shared_metric,
            "hand_roles": payload.get("hands", {}).get("roles"),
            "has_aligned": has_aligned,
        }
    reason = "no_optimization_artifact"
    if returncode != 0 and log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        error_lines = [line.strip() for line in text.splitlines() if "RuntimeError:" in line or "Error:" in line]
        if error_lines:
            reason = " | ".join(error_lines[-3:])
        else:
            tail = text.splitlines()[-8:]
            reason = " | ".join(line.strip() for line in tail if line.strip())
    return {
        "status": "failed_before_sequence_artifacts",
        "reason": reason,
        "contact_similarity_mode": None,
        "strict_shared_metric": None,
        "hand_roles": None,
        "has_aligned": aligned.is_file(),
    }


def _run_row(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    key = f"{row['sequence']}__{row['prototype']}__f{row['window'][0]}_{row['window'][1]}"
    source = Path(row["run_dir"])
    clone = RUNS_ROOT / f"{source.name}_p6_sequence"
    _clone_run(source, clone)
    state["rows"][key] = {"source_run": str(source), "clone_run": str(clone)}
    returncode, log = _run_sequence(clone)
    state["rows"][key]["sequence_returncode"] = returncode
    state["rows"][key]["sequence_log"] = log

    seq_dir = ADT_ROOT / row["sequence"]
    prepared_dir = _prepared_dir(row)
    _, T_world_camera, _ = _gt_camera_and_object(seq_dir, prepared_dir, str(row["object_uid"]))

    raw_npz = np.load(source / "object_tracking/foundationpose_raw.npz", allow_pickle=False)
    raw_valid = np.asarray(raw_npz["valid"], dtype=bool)
    raw_transforms = np.asarray(raw_npz["T_camera_object"], dtype=np.float64)
    selected = json.loads((source / "object_tracking/selected_mesh.json").read_text(encoding="utf-8"))
    raw_scale = float(selected["scale_to_m"])
    raw_metrics = _pose_metrics(
        transforms=raw_transforms,
        valid=raw_valid,
        run_dir=source,
        sequence_dir=seq_dir,
        prepared_dir=prepared_dir,
        object_uid=str(row["object_uid"]),
        prototype=row["prototype"],
        pred_scale=raw_scale,
    )

    sequence = _sequence_record(clone, returncode, Path(log))
    aligned_metrics: dict[str, Any] | None = None
    deltas: dict[str, Any] = {}
    scale_delta = math.nan
    if sequence.get("has_aligned"):
        try:
            aligned_transforms, aligned_valid, aligned_scale = _aligned_camera_transforms(
                clone, T_world_camera
            )
            aligned_metrics = _pose_metrics(
                transforms=aligned_transforms,
                valid=aligned_valid,
                run_dir=source,
                sequence_dir=seq_dir,
                prepared_dir=prepared_dir,
                object_uid=str(row["object_uid"]),
                prototype=row["prototype"],
                pred_scale=aligned_scale,
            )
            if raw_metrics.get("status") == "evaluated" and aligned_metrics.get("status") == "evaluated":
                deltas = {
                    "centroid_translation_error_m_delta": (
                        aligned_metrics["centroid_translation_error_m"]["median"]
                        - raw_metrics["centroid_translation_error_m"]["median"]
                    ),
                    "add_s_m_delta": (
                        aligned_metrics["add_s_m"]["median"] - raw_metrics["add_s_m"]["median"]
                    ),
                    "symmetry_aware_rotation_error_deg_delta": (
                        aligned_metrics["symmetry_aware_rotation_error_deg"]["median"]
                        - raw_metrics["symmetry_aware_rotation_error_deg"]["median"]
                    ),
                }
            scale_delta = (aligned_scale - raw_scale) / max(abs(raw_scale), 1e-12)
        except Exception as error:
            aligned_metrics = {"status": "evaluation_failed", "error": f"{type(error).__name__}: {error}"}

    violations: list[str] = []
    depth_gate = json.loads((source / "depth/depth_gate.json").read_text(encoding="utf-8"))
    if depth_gate.get("gate_kind") != "calibrated_stereo_native_metric":
        violations.append("not_calibrated_stereo_native_metric")
    if not depth_gate.get("accepted", False):
        violations.append("stereo_depth_gate_not_accepted")
    if sequence.get("contact_similarity_mode") is None:
        violations.append("sequence_did_not_reach_contact_similarity_protocol")
    elif sequence.get("contact_similarity_mode") != "validate_only":
        violations.append("contact_similarity_mode_not_validate_only")
    if sequence.get("strict_shared_metric") is None:
        violations.append("sequence_did_not_reach_strict_shared_metric_protocol")
    elif sequence.get("strict_shared_metric") is not True:
        violations.append("strict_shared_metric_not_true")
    if sequence.get("status") == "failed_before_export":
        violations.append("sequence_failed_before_export")
    if sequence.get("status") == "export_qc_failed":
        violations.append("sequence_export_qc_failed")
    if sequence.get("has_aligned") and not math.isfinite(scale_delta):
        violations.append("aligned_scale_not_available")
    if sequence.get("has_aligned") and abs(scale_delta) > 1e-6:
        violations.append(f"metric_scale_rewritten_{scale_delta:+.6g}")

    row_record = {
        "sequence_name": row["sequence"],
        "prototype": row["prototype"],
        "object_uid": row["object_uid"],
        "window": row["window"],
        "source_run": str(source),
        "clone_run": str(clone),
        "sequence_returncode": returncode,
        "sequence_optimization": sequence,
        "raw": raw_metrics,
        "aligned": aligned_metrics,
        "deltas": deltas,
        "scale_delta": scale_delta,
        "violations": violations,
    }
    state["rows"][key]["record"] = row_record
    return row_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidates = _candidate_runs()
    state: dict[str, Any] = {"schema_version": "1.0", "rows": {}}
    if args.state.is_file() and not args.force:
        state = json.loads(args.state.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in candidates:
        key = f"{row['sequence']}__{row['prototype']}__f{row['window'][0]}_{row['window'][1]}"
        if key in state.get("rows", {}) and state["rows"][key].get("record") and not args.force:
            record = state["rows"][key]["record"]
        else:
            record = _run_row(row, state)
        records.append(record)
        args.state.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "stage": "P6 Sequence Benchmark",
        "candidate_count": len(records),
        "rows": records,
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "sequence_status",
                "raw_centroid_median_m",
                "aligned_centroid_median_m",
                "centroid_delta_m",
                "raw_add_s_median_m",
                "aligned_add_s_median_m",
                "scale_delta",
                "violations",
            ]
        )
        for record in records:
            raw = record["raw"]
            aligned = record.get("aligned") or {}
            writer.writerow(
                [
                    Path(record["source_run"]).name,
                    record["sequence_optimization"]["status"],
                    raw.get("centroid_translation_error_m", {}).get("median", math.nan),
                    aligned.get("centroid_translation_error_m", {}).get("median", math.nan),
                    record.get("deltas", {}).get("centroid_translation_error_m_delta", math.nan),
                    raw.get("add_s_m", {}).get("median", math.nan),
                    aligned.get("add_s_m", {}).get("median", math.nan),
                    record.get("scale_delta", math.nan),
                    "|".join(record.get("violations", [])),
                ]
            )

    with VIOLATION_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "violation"])
        for record in records:
            for violation in record.get("violations", []):
                writer.writerow([Path(record["source_run"]).name, violation])

    print(f"summary -> {args.output}")
    print(f"matrix   -> {MATRIX_PATH}")
    print(f"violations -> {VIOLATION_PATH}")
    print(json.dumps(records, indent=2, ensure_ascii=False)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
