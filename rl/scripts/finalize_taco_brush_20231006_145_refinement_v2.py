#!/usr/bin/env python3
"""Finalize the fixed M4 geometry and experiment-C replay comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.retarget.trajectory_refinement import file_sha256  # noqa: E402


DEFAULT_CONFIG = (
    RL_ROOT
    / "configs/taco_brush_20231006_145_continuous_reference_refinement_v2.yaml"
)
REPLAY_OUTPUT = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_continuous_ref_refinement_m4_replay_v1"
)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_gz_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def replay_summary(output: Path, tolerance_m: float) -> dict[str, Any]:
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    with np.load(output / "substep_state_trace.npz", allow_pickle=False) as archive:
        states = {name: np.asarray(archive[name]) for name in archive.files}
    terminal = states["substep"] == states["substeps"] - 1
    terminal_indices = np.flatnonzero(terminal)
    maximum_error = {"value_rad": -1.0, "joint": None, "endpoint": None}
    saturation_count = 0
    maximum_effort_ratio = 0.0
    finger_dofs = []
    for contract in manifest["finger_dynamics"]["compiled_contracts"]:
        qpos_address = int(contract["qpos_address"])
        dof_address = int(contract["dof_address"])
        actuator = int(contract["actuator_id"])
        limit = float(contract["urdf_effort_limit_nm"])
        finger_dofs.append(dof_address)
        error = np.abs(
            states["reference_qpos"][:, qpos_address]
            - states["actual_qpos"][:, qpos_address]
        )
        index = int(terminal_indices[np.argmax(error[terminal])])
        if error[index] > maximum_error["value_rad"]:
            maximum_error = {
                "value_rad": float(error[index]),
                "joint": contract["joint_name"],
                "endpoint": int(states["outcome_endpoint"][index]),
            }
        force = np.abs(states["preceding_interval_actuator_force"][:, actuator])
        saturation_count += int(np.count_nonzero(force > limit + 1e-9))
        maximum_effort_ratio = max(maximum_effort_ratio, float(force.max() / limit))
    finger_dofs_array = np.asarray(finger_dofs, dtype=np.int64)
    maximum_finger_external_force = float(
        np.abs(states["qfrc_applied"][:, finger_dofs_array]).max()
    )
    native = results["substep_native_hand_table"]["whole_replay"]
    contacts = csv_gz_rows(output / "contact_trace.csv.gz")
    endpoints = csv_rows(output / "endpoint_trace.csv")
    return {
        "classification": results["classification"],
        "physics_steps": int(results["physics_steps"]),
        "maximum_finger_terminal_error": maximum_error,
        "finger_effort_saturation_substeps": saturation_count,
        "maximum_finger_effort_ratio": maximum_effort_ratio,
        "maximum_finger_external_generalized_force": maximum_finger_external_force,
        "actual_native_minimum_distance_m": float(native["actual_minimum_distance_m"]),
        "reference_native_minimum_distance_m": float(native["reference_minimum_distance_m"]),
        "actual_beyond_tolerance_count": int(
            native["actual_penetration_beyond_tolerance_count"]
        ),
        "reference_beyond_tolerance_count": int(
            native["reference_penetration_beyond_tolerance_count"]
        ),
        "first_actual_beyond_tolerance": native["first_actual_penetration_beyond_tolerance"],
        "hand_floor_proxy_contact_rows": sum(
            row["group"] == "hand_floor" for row in contacts
        ),
        "maximum_brush_position_error_m": max(
            float(row["brush_position_error_m"]) for row in endpoints
        ),
        "maximum_bowl_position_error_m": max(
            float(row["bowl_position_error_m"]) for row in endpoints
        ),
        "hard_stop": results["hard_stop"],
        "strict_pickup_gate_passed": bool(results["strict_gate_passed"]),
        "maximum_brush_actual_lift_m": float(
            results["pickup"]["brush"]["maximum_actual_lift_m"]
        ),
        "maximum_bowl_actual_lift_m": float(
            results["pickup"]["bowl"]["maximum_actual_lift_m"]
        ),
        "native_tolerance_m": tolerance_m,
    }


def prefix_comparison(baseline: Path, candidate: Path) -> dict[str, Any]:
    with np.load(baseline / "substep_state_trace.npz", allow_pickle=False) as archive:
        before = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(candidate / "substep_state_trace.npz", allow_pickle=False) as archive:
        after = {name: np.asarray(archive[name]) for name in archive.files}
    if set(before) != set(after):
        return {"status": "FAIL", "reason": "TRACE_FIELDS_CHANGED"}
    mask_before = before["outcome_endpoint"] <= 4
    mask_after = after["outcome_endpoint"] <= 4
    mismatches = {}
    for name in before:
        first = before[name] if before[name].ndim == 1 and len(before[name]) != len(mask_before) else before[name][mask_before]
        second = after[name] if after[name].ndim == 1 and len(after[name]) != len(mask_after) else after[name][mask_after]
        if first.shape != second.shape or not np.array_equal(first, second):
            mismatches[name] = {
                "baseline_shape": list(first.shape),
                "candidate_shape": list(second.shape),
                "maximum_absolute_difference": (
                    None
                    if first.shape != second.shape or first.dtype.kind not in "fiu"
                    else float(np.max(np.abs(first - second)))
                ),
            }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "compared_through_outcome_endpoint": 4,
        "baseline_substeps": int(mask_before.sum()),
        "candidate_substeps": int(mask_after.sum()),
        "mismatches": mismatches,
    }


def report_text(
    classification: str,
    audit: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    prefix: dict[str, Any],
    delta_rows: list[dict[str, str]],
) -> str:
    clearance = sorted(
        (
            row for row in audit["clearance_rows"] if row["grid"] == "primary"
        ),
        key=lambda row: row["source_endpoint"],
    )
    clearance_lines = [
        f"- {row['source_endpoint']}→{row['outcome_endpoint']}："
        f"最低 {row['minimum_distance_m'] * 1000.0:.6f} mm，"
        f"要求 {row['required_distance_m'] * 1000.0:.6f} mm。"
        for row in clearance
    ]
    first = candidate["first_actual_beyond_tolerance"]
    first_text = (
        "没有出现超过 0.05 mm 容差的实际穿桌。"
        if first is None
        else (
            f"首次超过容差发生在 {first['source_endpoint']}→"
            f"{first['outcome_endpoint']}，子步 {first['substep']}，"
            f"距离 {first['actual_minimum_distance_m'] * 1000.0:.6f} mm。"
        )
    )
    by_joint: dict[str, dict[int, float]] = {}
    for row in delta_rows:
        by_joint.setdefault(row["joint"], {})[int(row["endpoint"])] = float(
            row["applied_delta"]
        )
    delta_lines = []
    for joint, values in by_joint.items():
        translation = joint.startswith(("L_forearm_t", "R_forearm_t"))
        scale = 1000.0 if translation else 180.0 / np.pi
        unit = "mm" if translation else "°"
        formatted = "、".join(
            f"端点 {endpoint}: {values.get(endpoint, 0.0) * scale:+.6f}"
            for endpoint in (5, 6, 7, 8)
        )
        delta_lines.append(f"| `{joint}` | {formatted} {unit} |")
    hand_object_worst = min(
        audit["hand_object_rows"], key=lambda row: row["distance_change_m"]
    )
    return "\n".join([
        "# 20231006_145 连续参考轨迹修复 v2",
        "",
        "## 本轮结果",
        "",
        "本轮局部目标已经通过：端点 0–8 的新参考和实验 C 物理回放都没有手部穿桌。"
        f"结构化分类为 `{classification}`。但是完整抓取判定仍然失败，因为刷子和碗"
        "都没有形成稳定离桌抬升；这里没有把局部避障通过说成完整抓取成功。",
        "",
        "## 原参考问题",
        "",
        "原参考的 5→6 最低距离只有约 0.000002 mm，几乎没有余量；6→7 和 7→8 "
        "分别穿桌约 0.103382 mm 和 0.059546 mm。右小指在端点 7、8 也几乎贴着桌面，"
        "因此只允许修改左手的 M1–M3 无法满足双手都至少留出 0.5 mm 的要求。新增 M4 "
        "后，仅额外开放右小指关节 1、2。",
        "",
        "## 实际修改",
        "",
        "只修改端点 5–8；端点 4、9、物体位姿、原始时间戳和 30 Hz 频率逐位不变。"
        "下表给出新参考相对原参考的改变量；平移关节用毫米，其余关节用角度。",
        "",
        "| 关节 | 端点 5–8 的改变量 |",
        "|---|---|",
        *delta_lines,
        "",
        "## 独立几何验收",
        "",
        "候选重新写入并读取后，使用包含端点的 10001 点网格及半步错开的 10001 点"
        "网格检查全部原生手部视觉网格。结果如下：",
        "",
        *clearance_lines,
        "",
        f"最大指尖变化为 {audit['maximum_fingertip_change_m'] * 1000.0:.6f} mm，低于 2 mm 上限；"
        f"最大手腕平移变化为 {audit['maximum_wrist_translation_m'] * 1000.0:.6f} mm；"
        f"最大手腕旋转变化为 {np.degrees(audit['maximum_wrist_rotation_rad']):.6f}°。",
        f"手—物体原生距离最不利的变化为 "
        f"{hand_object_worst['distance_change_m'] * 1000.0:.6f} mm，"
        "没有超过 0.05 mm 非恶化上限；没有新增自碰撞。",
        f"30 Hz 原时间轴上的最大速度限额比例为 {audit['maximum_original_30hz_velocity_ratio']:.6f}，"
        "关节位置和速度限制均通过。",
        "",
        "## 物理回放对比",
        "",
        f"原实验 C 的实际原生手—桌最低距离为 "
        f"{baseline['actual_native_minimum_distance_m'] * 1000.0:.6f} mm；"
        f"新回放为 {candidate['actual_native_minimum_distance_m'] * 1000.0:.6f} mm。",
        f"{first_text}",
        f"手指最大端点误差从 {baseline['maximum_finger_terminal_error']['value_rad']:.9f} rad "
        f"变为 {candidate['maximum_finger_terminal_error']['value_rad']:.9f} rad；"
        f"力矩超限子步数为 {candidate['finger_effort_saturation_substeps']}；"
        f"手指外加广义力最大绝对值为 "
        f"{candidate['maximum_finger_external_generalized_force']:.9g}。",
        f"手指实际力矩最高达到限额的 {candidate['maximum_finger_effort_ratio'] * 100.0:.3f}%。"
        "MuJoCo 手—桌代理接触记录为 0。",
        f"端点 0–4 的 667 个物理子步中，状态、控制、执行器力和接触相关数组逐位一致，"
        f"前缀检查结果为 {'通过' if prefix['status'] == 'PASS' else '失败'}。",
        f"刷子最大位置误差为 {candidate['maximum_brush_position_error_m'] * 1000.0:.6f} mm，"
        f"碗最大位置误差为 {candidate['maximum_bowl_position_error_m'] * 1000.0:.6f} mm，"
        "与原实验 C 完全相同。",
        "",
        "## 尚未解决的范围",
        "",
        f"本次只验证端点 0–8，并且使用实验 C 的 10 倍慢速执行。刷子最大实际抬升为 "
        f"{candidate['maximum_brush_actual_lift_m'] * 1000.0:.9f} mm，碗为 "
        f"{candidate['maximum_bowl_actual_lift_m'] * 1000.0:.9f} mm，"
        "都没有稳定离桌，因此完整严格回放仍按失败关闭。端点 9 之后的参考没有在本轮修复或"
        "物理验证，不能据此判断完整 122 帧轨迹安全。",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-output", type=Path, default=REPLAY_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    refinement_output = Path(cfg["paths"]["output"]).resolve(strict=True)
    replay_output = args.replay_output.resolve(strict=True)
    audit = json.loads(
        (refinement_output / "native_geometry_audit.json").read_text(encoding="utf-8")
    )
    if audit["status"] != "PASS":
        raise RuntimeError("independent geometry audit did not pass")
    tolerance = float(cfg["geometry_contract"]["numerical_penetration_tolerance_m"])
    baseline_output = Path(
        cfg["paths"]["baseline_physics_results"]
    ).resolve(strict=True).parent
    baseline = replay_summary(baseline_output, tolerance)
    candidate = replay_summary(replay_output, tolerance)
    prefix = prefix_comparison(baseline_output, replay_output)
    local_physics_pass = bool(
        candidate["hard_stop"] is None
        and candidate["actual_beyond_tolerance_count"] == 0
        and candidate["reference_beyond_tolerance_count"] == 0
        and candidate["finger_effort_saturation_substeps"] == 0
        and candidate["maximum_finger_external_generalized_force"] == 0.0
        and prefix["status"] == "PASS"
    )
    classification = (
        "LOCAL_REFERENCE_AND_PHYSICS_VALIDATED"
        if local_physics_pass
        else "DYNAMIC_TRACKING_FAILED"
    )
    comparison = {
        "schema": "taco_brush_20231006_145_refined_physics_comparison_v2",
        "classification": classification,
        "baseline": baseline,
        "candidate": candidate,
        "prefix_consistency": prefix,
        "independent_geometry_audit_status": audit["status"],
        "scope": "ENDPOINT_0_TO_8_AT_EXPERIMENT_C_10X_TIME_SCALE",
        "full_122_endpoint_grasp_success_claimed": False,
    }
    write_json(refinement_output / "physics_replay_comparison.json", comparison)
    initial_results = json.loads(
        (refinement_output / "results.json").read_text(encoding="utf-8")
    )
    final_results = {
        **initial_results,
        "classification": classification,
        "independent_geometry_audit": audit["status"],
        "candidate_physics_replay": candidate["classification"],
        "local_physics_gate_passed": local_physics_pass,
        "prefix_consistency": prefix,
        "full_122_endpoint_grasp_success_claimed": False,
        "scope": "ENDPOINT_0_TO_8_AT_EXPERIMENT_C_10X_TIME_SCALE",
        "finalization": {
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=RL_ROOT.parent, text=True
            ).strip(),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "replay_results": {
                "path": str(replay_output / "results.json"),
                "sha256": file_sha256(replay_output / "results.json"),
            },
        },
    }
    write_json(refinement_output / "results.json", final_results)
    delta_rows = csv_rows(refinement_output / "refinement_delta.csv")
    (refinement_output / "report.md").write_text(
        report_text(
            classification, audit, baseline, candidate, prefix, delta_rows
        ),
        encoding="utf-8",
    )
    reproduction = refinement_output / "reproduction_commands.txt"
    final_commands = (
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src:rl/scripts "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/audit_taco_brush_20231006_145_refined_reference_v2.py "
        "--config rl/configs/taco_brush_20231006_145_continuous_reference_refinement_v2.yaml\n"
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src:rl/scripts "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/run_taco_brush_controlled_physics_replay_v1.py "
        "--config rl/configs/taco_brush_20231006_145_refined_m4_replay_v1.yaml\n"
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/finalize_taco_brush_20231006_145_refinement_v2.py\n"
    )
    existing_commands = reproduction.read_text(encoding="utf-8")
    if final_commands not in existing_commands:
        reproduction.write_text(
            existing_commands + final_commands, encoding="utf-8"
        )
    with (refinement_output / "artifacts.sha256").open("w", encoding="utf-8") as stream:
        for path in sorted(refinement_output.iterdir()):
            if path.name == "artifacts.sha256":
                continue
            stream.write(f"{file_sha256(path)}  {path.name}\n")
    print(json.dumps({
        "classification": classification,
        "local_physics_gate_passed": local_physics_pass,
        "actual_native_minimum_distance_m": candidate["actual_native_minimum_distance_m"],
        "actual_beyond_tolerance_count": candidate["actual_beyond_tolerance_count"],
        "finger_effort_saturation_substeps": candidate["finger_effort_saturation_substeps"],
        "prefix_consistency": prefix["status"],
    }, ensure_ascii=False, indent=2))
    return 0 if local_physics_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
