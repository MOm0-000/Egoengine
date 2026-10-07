#!/usr/bin/env python3
"""Re-audit frozen Brush corrections against the same frozen table plane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))
sys.path.insert(0, str(RL_ROOT / "scripts"))

from audit_taco_brush_single_frame_calibration_penetration_v1 import (  # noqa: E402
    artifact, entity_distances, load_contracts, load_geometry, resolve_sample,
    sha256, write_json,
)
from egoengine_repro.evaluation.taco_calibration_residual import invert_transform  # noqa: E402
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


DEFAULT_OUTPUT = RL_ROOT / "runs/taco_brush_measurement_code_repair_v1"
SAMPLE = "brush_brush_bowl_20230927_027"
MODELS = ("WORLD_FIXED", "CAMERA_LOCAL")
ENTITIES = ("brush", "bowl", "left_hand", "right_hand", "hand")
RUNS = {
    "CURRENT_FIXED_LEGACY": RL_ROOT / "runs/taco_brush_single_frame_calibration_penetration_v1",
    "OPEN3D_OFFICIAL_LEGACY": RL_ROOT / "runs/taco_brush_open3d_single_frame_calibration_penetration_v1",
}
PLANE_ROWS = RL_ROOT / "runs/taco_multiframe_table_depth_estimation_v1/per_frame_table_plane_estimates.jsonl"
BASE_CONFIG = RL_ROOT / "configs/taco_brush_single_frame_calibration_penetration_v1.yaml"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        if item.is_file():
            digest.update(item.name.encode())
            digest.update(sha256(item).encode())
    return digest.hexdigest()


def _load_planes() -> dict[int, dict[str, Any]]:
    rows = {}
    for line in PLANE_ROWS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["sample"] == SAMPLE:
            rows[int(row["frame"])] = row
    if set(rows) != set(range(209)):
        raise ValueError("frozen plane records do not cover all Brush frames")
    return rows


def _legacy_rows(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with (path / "per_frame_results.csv").open(encoding="utf-8") as stream:
        return {
            (int(row["frame"]), row["coordinate_model"]): row
            for row in csv.DictReader(stream)
        }


def _world_correction(correction: np.ndarray, extrinsic: np.ndarray, model: str) -> np.ndarray:
    if model == "WORLD_FIXED":
        return correction
    if model == "CAMERA_LOCAL":
        return invert_transform(extrinsic) @ correction @ extrinsic
    raise ValueError(model)


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _plane_values(prefix: str, plane: Plane | None) -> dict[str, Any]:
    if plane is None:
        return {
            f"{prefix}_normal_x": None, f"{prefix}_normal_y": None,
            f"{prefix}_normal_z": None, f"{prefix}_offset_m": None,
            f"{prefix}_tilt_deg": None,
        }
    return {
        f"{prefix}_normal_x": float(plane.normal[0]),
        f"{prefix}_normal_y": float(plane.normal[1]),
        f"{prefix}_normal_z": float(plane.normal[2]),
        f"{prefix}_offset_m": plane.offset,
        f"{prefix}_tilt_deg": float(np.degrees(np.arccos(np.clip(plane.normal[2], -1, 1)))),
    }


def _acceptance(parameters: np.ndarray, legacy_success: bool, run_name: str) -> tuple[str, bool]:
    if not np.isfinite(parameters).all():
        return "CANDIDATE_NOT_SAVED_CANNOT_RECOMPUTE", False
    translation = float(np.linalg.norm(parameters[:3]))
    rotation = float(np.degrees(np.linalg.norm(parameters[3:])))
    if not legacy_success:
        return "LEGACY_REJECTED_OR_FAILED_CANDIDATE_NOT_SAVED", False
    if translation > 0.1 + 1e-12 or rotation > 10.0 + 1e-9:
        return "DIAGNOSTIC_ONLY_CANDIDATE_ACCEPTANCE_BOUND_EXCEEDED", False
    return "ACCEPTED_FOR_USE_UNDER_FROZEN_RANGE", True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run_name in RUNS:
        result[run_name] = {}
        for model in MODELS:
            local = [row for row in rows if row["legacy_run"] == run_name and row["coordinate_model"] == model]
            audited = [row for row in local if row["same_plane_recomputed"]]
            authorized = [row for row in audited if row["authorized_for_use"]]
            entities = {}
            for entity in ENTITIES:
                values = [row[f"{entity}_same_plane_after_mm"] for row in authorized]
                diagnostic_values = [
                    row[f"{entity}_same_plane_after_mm"] for row in audited
                    if row[f"{entity}_same_plane_after_mm"] is not None
                ]
                legacy_differences = [
                    row[f"{entity}_legacy_minus_same_plane_after_mm"] for row in audited
                    if row[f"{entity}_legacy_minus_same_plane_after_mm"] is not None
                ]
                entities[entity] = {
                    "authorized_common_frames": len(values),
                    "missing_or_not_authorized_frames": 209 - len(values),
                    "penetrating_frames": sum(value < -0.05 for value in values),
                    "minimum_mm": None if not values else float(min(values)),
                    "diagnostic_same_plane_frames": len(diagnostic_values),
                    "legacy_refit_minus_same_plane_after_mm": {
                        "count": len(legacy_differences),
                        "minimum": None if not legacy_differences else float(min(legacy_differences)),
                        "median": None if not legacy_differences else float(np.median(legacy_differences)),
                        "maximum": None if not legacy_differences else float(max(legacy_differences)),
                        "maximum_absolute": (
                            None if not legacy_differences
                            else float(np.max(np.abs(legacy_differences)))
                        ),
                    },
                }
            result[run_name][model] = {
                "frame_count": len(local),
                "saved_candidate_frames": sum(row["correction_exists"] for row in local),
                "authorized_candidate_frames": sum(row["authorized_for_use"] for row in local),
                "diagnostic_only_candidate_frames": sum(
                    str(row["usage_status"]).startswith("DIAGNOSTIC_ONLY") for row in local
                ),
                "candidate_not_saved_frames": sum(not row["correction_exists"] for row in local),
                "same_plane_recomputed_frames": len(audited),
                "entities_on_authorized_frames": entities,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    cfg, _, taco, _ = load_contracts(BASE_CONFIG.resolve())
    sample = resolve_sample(cfg, taco)
    geometry = load_geometry(taco, sample)
    extrinsic = np.load(sample["extrinsic"], mmap_mode="r")
    planes = _load_planes()
    before_hashes = {name: _manifest_hash(path) for name, path in RUNS.items()}
    before_hashes["TABLE_ESTIMATION_LEGACY"] = _manifest_hash(PLANE_ROWS.parent)
    plane_rows_sha256 = sha256(PLANE_ROWS)
    correction_sha256 = {
        name: sha256(path / "per_frame_corrections.npz") for name, path in RUNS.items()
    }
    rows: list[dict[str, Any]] = []

    for run_name, run_path in RUNS.items():
        corrections = np.load(run_path / "per_frame_corrections.npz")
        legacy = _legacy_rows(run_path)
        for model in MODELS:
            prefix = model.lower()
            for frame in range(209):
                plane_row = planes[frame]
                plane_data = plane_row.get("plane") if plane_row["status"] == "TABLE_PLANE_ESTIMATED" else None
                original = None if plane_data is None else Plane(
                    normal=plane_data["normal"], offset=plane_data["offset_m"], frame="world",
                )
                correction = np.asarray(corrections[f"{prefix}_transform"][frame], dtype=np.float64)
                parameters = np.asarray(
                    corrections[f"{prefix}_parameters_translation_m_then_rotvec_rad"][frame],
                    dtype=np.float64,
                )
                legacy_success = bool(corrections[f"{prefix}_success"][frame])
                correction_exists = bool(np.isfinite(correction).all() and np.isfinite(parameters).all())
                usage_status, authorized = _acceptance(parameters, legacy_success, run_name)
                transformed = None
                if original is not None and correction_exists:
                    world_correction = _world_correction(correction, extrinsic[frame], model)
                    transformed = original.transform(world_correction, target_frame="world")
                original_distances = (
                    entity_distances(original, geometry, frame) if original is not None else None
                )
                transformed_distances = (
                    entity_distances(transformed, geometry, frame) if transformed is not None else None
                )
                old = legacy[(frame, model)]
                row: dict[str, Any] = {
                    "legacy_run": run_name,
                    "frame": frame,
                    "coordinate_model": model,
                    "correction_exists": correction_exists,
                    "legacy_reason": str(corrections[f"{prefix}_failure_reason"][frame]),
                    "usage_status": usage_status,
                    "authorized_for_use": authorized,
                    "same_plane_recomputed": transformed is not None,
                    "missing_reason": (
                        None if transformed is not None
                        else "ORIGINAL_PLANE_UNAVAILABLE" if original is None
                        else "CANDIDATE_NOT_SAVED"
                    ),
                    "translation_norm_mm": None if not correction_exists else float(np.linalg.norm(parameters[:3]) * 1000),
                    "rotation_angle_deg": None if not correction_exists else float(np.degrees(np.linalg.norm(parameters[3:]))),
                    "original_plane_source": str(PLANE_ROWS),
                    "original_plane_artifact_sha256": plane_rows_sha256,
                    "correction_artifact_sha256": correction_sha256[run_name],
                    "original_plane_status": plane_row["status"],
                    "original_plane_has_10deg_estimation_prior": True,
                    "transformed_plane_source": "RIGID_TRANSFORM_OF_EXACT_SAME_ORIGINAL_PLANE",
                    **_plane_values("original_plane", original),
                    **_plane_values("transformed_same_plane", transformed),
                    "legacy_refit_plane_offset_m": _float(old.get("after_plane_offset_m")),
                    "legacy_refit_plane_tilt_deg": _float(old.get("after_plane_tilt_deg")),
                }
                for entity in ENTITIES:
                    before = None if original_distances is None else original_distances[entity] * 1000
                    after = None if transformed_distances is None else transformed_distances[entity] * 1000
                    old_before = _float(old.get(f"{entity}_before_mm"))
                    old_after = _float(old.get(f"{entity}_after_mm"))
                    row[f"{entity}_original_plane_mm"] = before
                    row[f"{entity}_same_plane_after_mm"] = after
                    row[f"{entity}_legacy_before_mm"] = old_before
                    row[f"{entity}_legacy_refit_after_mm"] = old_after
                    row[f"{entity}_legacy_minus_same_plane_after_mm"] = (
                        None if old_after is None or after is None else old_after - after
                    )
                rows.append(row)

    frame139 = next(
        row for row in rows
        if row["legacy_run"] == "OPEN3D_OFFICIAL_LEGACY"
        and row["frame"] == 139 and row["coordinate_model"] == "WORLD_FIXED"
    )
    if not np.isclose(frame139["transformed_same_plane_tilt_deg"], 28.6271405689, atol=1e-10):
        raise AssertionError("frame139 transformed-plane tilt regression")
    if not np.isclose(frame139["transformed_same_plane_offset_m"], 0.5297505385, atol=1e-10):
        raise AssertionError("frame139 transformed-plane offset regression")

    _write_csv(output / "per_frame_same_plane_reaudit.csv", rows)
    summary = {
        "schema": "taco_brush_measurement_code_repair_summary_v1",
        "sample": SAMPLE,
        "historical_runs_mutated": False,
        "calibration_solver_rerun": False,
        "depth_video_read": False,
        "table_refit_performed": False,
        "distance_threshold_mm": -0.05,
        "summary_by_run_and_model": _summarize(rows),
        "frame139_regression": frame139,
        "limitations": [
            "The original table plane retains the historical 10-degree estimation prior.",
            "Failed legacy CURRENT_FIXED rows did not save full candidates and cannot be reconstructed.",
            "Legacy corrections were selected with the old target-point mask; no real calibration was rerun.",
            "Nominal hand projection does not prove contamination outside the projected silhouette is absent.",
            "No independent table region or absolute position reference is available.",
        ],
        "conclusion": "MEASUREMENT_AND_REPORTING_REPAIRED_REAL_GEOMETRIC_CAUSE_UNRESOLVED",
    }
    write_json(output / "summary.json", summary)
    table_lines = [
        "| 历史运行 | 坐标设置 | 保存候选 | 获准使用 | 仅诊断 | 无候选 | 同平面可重核 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name in RUNS:
        for model in MODELS:
            item = summary["summary_by_run_and_model"][run_name][model]
            table_lines.append(
                f"| {run_name} | {model} | {item['saved_candidate_frames']} | "
                f"{item['authorized_candidate_frames']} | {item['diagnostic_only_candidate_frames']} | "
                f"{item['candidate_not_saved_frames']} | {item['same_plane_recomputed_frames']} |"
            )
    open3d_world = summary["summary_by_run_and_model"]["OPEN3D_OFFICIAL_LEGACY"]["WORLD_FIXED"]
    open3d_difference = open3d_world["entities_on_authorized_frames"]["bowl"][
        "legacy_refit_minus_same_plane_after_mm"
    ]
    (output / "summary.md").write_text(
        "# Brush 测量代码修复与同一平面重核 v1\n\n"
        "**结论：测量/报告代码已修复；真实几何误差原因仍未确定，约两厘米差异没有被证明解决。**\n\n"
        "## 六项修复\n\n"
        "- **A 同一平面：** 修正前后跟踪同一张完整平面，不再修正点云后另找近水平平面。\n"
        "- **B 归一化：** normal 和 offset 同除以法向量长度，不再移动平面。旧 Brush 实验是否触发此缺陷仍无证据。\n"
        "- **C 验证不截断：** 背景验证点不再按候选平面 ±10 mm 截断；独立桌面验证和绝对位置精度均未完成。\n"
        "- **D 强制排除：** 左右手名义投影不问深度前后均禁用；独立手/不确定区域 mask 不存在，轮廓外污染仍未知。\n"
        "- **E 候选与准入：** 有限候选保留完整矩阵和求解状态；未收敛、向量范数越界、证据不足不再混成一个“数值失败”。\n"
        "- **F 完整汇总：** 刷子、碗、左右手分别统计；改善与恶化并存时明确报混合结果，`0.0` 不再当成缺失。\n\n"
        "## 历史候选可重核性\n\n"
        + "\n".join(table_lines) + "\n\n"
        "原创算法失败帧当时没有保存完整候选，无法补算且本轮没有重跑。"
        "Open3D 越出原向量准入范围的候选仍做同平面诊断，但不计入有效校准。\n\n"
        "## Frame 139 回归\n\n"
        f"Open3D WORLD_FIXED 同一原平面变换后 tilt=`{frame139['transformed_same_plane_tilt_deg']:.10f}°`，"
        f"offset=`{frame139['transformed_same_plane_offset_m']:.10f} m`；旧报告另行重拟合得到 "
        f"tilt=`{frame139['legacy_refit_plane_tilt_deg']:.10f}°`，offset=`{frame139['legacy_refit_plane_offset_m']:.10f} m`。"
        "该候选的平移和旋转均超出原准入范围，因此只是诊断回归；旧数字不能再代表同一桌面。\n\n"
        "## 旧报告换平面的影响\n\n"
        f"Open3D WORLD_FIXED 的碗距离在 `{open3d_difference['count']}` 帧同时有旧值与同平面新值；"
        f"旧重拟合值减新值的最大绝对差为 `{open3d_difference['maximum_absolute']:.3f} mm`。"
        "该值只表明旧报告换平面可大幅改变结果，不证明原平面是现实真值。\n\n"
        "## 必须降级的旧结论\n\n"
        "- 筛选后的约 ±8 mm 只是候选平面附近残差，不能证明绝对桌面精度。\n"
        "- 旧严重穿透数字混入了重新选平面的影响，必须以逐帧同平面重核表解释。\n"
        "- 碗底/刷子底距离可报告，但不能单独归因相机标定错误。\n"
        "- 历史 `hand` 是真人手模型，不是 MINK v2 机器人手。\n\n"
        "本轮没有读取 Depth 视频，没有重新求解校准、重拟合桌面或运行 MINK/physics/Replay/MPC/RL。\n",
        encoding="utf-8",
    )
    after_hashes = {name: _manifest_hash(path) for name, path in RUNS.items()}
    after_hashes["TABLE_ESTIMATION_LEGACY"] = _manifest_hash(PLANE_ROWS.parent)
    write_json(output / "historical_integrity.json", {
        "before": before_hashes, "after": after_hashes,
        "all_unchanged": before_hashes == after_hashes,
    })
    write_json(output / "source_pins.json", {
        "schema": "taco_brush_measurement_code_repair_source_pins_v1",
        "repository_head": _git_head(),
        "artifacts": {
            "plane_rows": artifact(PLANE_ROWS),
            **{
                f"{name}:{file}": artifact(path / file)
                for name, path in RUNS.items()
                for file in ("per_frame_corrections.npz", "per_frame_results.csv", "summary.json")
            },
            "extrinsic": artifact(sample["extrinsic"]),
            "tool_pose": artifact(sample["tool_pose"]),
            "target_pose": artifact(sample["target_pose"]),
            "left_hand": artifact(sample["left_hand"]),
            "right_hand": artifact(sample["right_hand"]),
        },
        "missing_referenced_audit_attachments": [
            "Brush穿透问题_代码只读审计_2026-10-07.md",
            "reproduce_checks.py", "checks.json",
        ],
    })
    print(json.dumps({
        "rows": len(rows),
        "frame139_tilt_deg": frame139["transformed_same_plane_tilt_deg"],
        "frame139_offset_m": frame139["transformed_same_plane_offset_m"],
        "historical_inputs_unchanged": before_hashes == after_hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
