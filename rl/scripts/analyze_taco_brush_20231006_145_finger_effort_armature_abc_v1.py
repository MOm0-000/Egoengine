#!/usr/bin/env python3
"""Compare the frozen endpoint 0-8 XHand finger dynamics experiments."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    table_audit,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


RUNS = {
    "A": RL_ROOT / "runs/taco_brush_20231006_145_finger_effort_armature_a_v1",
    "B": RL_ROOT / "runs/taco_brush_20231006_145_finger_effort_armature_b_v1",
    "C": RL_ROOT / "runs/taco_brush_20231006_145_finger_effort_armature_c_retime_v1",
}
CONTRACT = (
    RL_ROOT
    / "configs/taco_brush_20231006_145_finger_effort_armature_ab_contract_v1.yaml"
)
OUTPUT = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_finger_effort_armature_abc_analysis_v1"
)
TOLERANCE_M = 0.00005
PHYSICS_DT_S = 0.002
PLOT_JOINT = "left_hand_pinky_joint2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def first_index(values: np.ndarray, predicate: np.ndarray) -> int | None:
    indices = np.flatnonzero(predicate)
    return None if len(indices) == 0 else int(indices[0])


def native_event(
    rows: list[dict[str, str]], values: str, index: int | None,
) -> dict[str, Any] | None:
    if index is None:
        return None
    row = rows[index]
    left = float(row[f"{values}_left_distance_m"])
    right = float(row[f"{values}_right_distance_m"])
    side = "left" if left <= right else "right"
    distance = min(left, right)
    return {
        "trace_index": index,
        "source_endpoint": int(row["source_endpoint"]),
        "outcome_endpoint": int(row["outcome_endpoint"]),
        "substep": int(row["substep"]),
        "substeps": int(row["substeps"]),
        "control_alpha": float(row["control_alpha"]),
        "geometric_endpoint_coordinate": (
            int(row["source_endpoint"]) + float(row["control_alpha"])
        ),
        "state_time_s": float(row["state_time_s"]),
        "side": side,
        "distance_m": distance,
        "distance_mm": distance * 1000.0,
    }


def detailed_native_location(
    scene: Path, qpos: np.ndarray, plane: Plane, side: str,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    audited = table_audit(native_world_objects(model, data), plane, TOLERANCE_M)
    entity = audited["entities"][f"{side}_xhand"]
    return {
        "body": entity["worst_location"]["body"],
        "native_visual_geom": entity["worst_location"]["native_visual_geom"],
        "vertex_index": entity["worst_location"]["vertex_index"],
        "point_sim_m": entity["worst_location"]["point_sim_m"],
        "minimum_signed_distance_m": entity["minimum_signed_distance_m"],
    }


def write_rows(path: Path, rows: list[dict[str, Any]], *, gzip_output: bool) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contact_summary(path: Path) -> dict[str, Any]:
    rows = read_csv_gz(path)
    hand_floor = [row for row in rows if row["group"] == "hand_floor"]
    if hand_floor:
        first = hand_floor[0]
        first_contact = {
            "source_endpoint": int(first["source_endpoint"]),
            "outcome_endpoint": int(first["outcome_endpoint"]),
            "substep": int(first["substep"]),
            "force_interval_start_time_s": float(first["interval_start_time_s"]),
            "force_interval_end_time_s": float(first["interval_end_time_s"]),
            "geom1": first["geom1"],
            "geom2": first["geom2"],
            "proxy_distance_m": float(first["distance_m"]),
            "force_norm_n": float(first["force_norm_n"]),
        }
    else:
        first_contact = None
    return {
        "hand_floor_contact_rows": len(hand_floor),
        "first_hand_floor_contact": first_contact,
        "maximum_hand_floor_force_n": max(
            (float(row["force_norm_n"]) for row in hand_floor), default=0.0,
        ),
        "hand_floor_normal_impulse_ns": sum(
            float(row["normal_force_n"]) * PHYSICS_DT_S for row in hand_floor
        ),
        "maximum_any_contact_force_n": max(
            (float(row["force_norm_n"]) for row in rows), default=0.0,
        ),
    }


def analyze_run(label: str, run: Path, plane: Plane) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(run / "input_manifest.json")
    result = load_json(run / "results.json")
    with np.load(run / "substep_state_trace.npz", allow_pickle=False) as archive:
        states = {name: np.asarray(archive[name]).copy() for name in archive.files}
    native_rows = read_csv_gz(run / "substep_native_hand_table_trace.csv.gz")
    if len(native_rows) != len(states["substep"]):
        raise ValueError(f"state/native row mismatch in experiment {label}")
    contracts = manifest["finger_dynamics"]["compiled_contracts"]
    time_scale = float(manifest["physics"].get("reference_time_scale", 1.0))
    detailed_rows: list[dict[str, Any]] = []
    saturated_step_indices: set[int] = set()
    maximum_raw_ratio = -np.inf
    maximum_raw_event: dict[str, Any] | None = None
    maximum_constraint = -np.inf
    maximum_constraint_event: dict[str, Any] | None = None
    maximum_endpoint_position_error = -np.inf
    maximum_endpoint_position_event: dict[str, Any] | None = None
    maximum_velocity_relative_error = -np.inf
    maximum_velocity_event: dict[str, Any] | None = None
    frozen_b_violations: list[dict[str, Any]] = []
    saturated_joint_substeps = 0
    terminal = states["substep"] == states["substeps"] - 1

    for contract in contracts:
        joint = contract["joint_name"]
        qpos_address = int(contract["qpos_address"])
        dof = int(contract["dof_address"])
        actuator = int(contract["actuator_id"])
        limit = float(contract["urdf_effort_limit_nm"])
        raw = states["preceding_interval_actuator_force"][:, actuator]
        limited = states["preceding_interval_qfrc_actuator"][:, dof]
        external = states["qfrc_applied"][:, dof]
        constraint = states["preceding_interval_qfrc_constraint"][:, dof]
        position_error = (
            states["reference_qpos"][:, qpos_address]
            - states["actual_qpos"][:, qpos_address]
        )
        saturated = np.abs(raw) > limit + 1e-9
        saturated_joint_substeps += int(np.count_nonzero(saturated))
        saturated_step_indices.update(np.flatnonzero(saturated).tolist())

        ratio = np.abs(raw) / limit
        ratio_index = int(np.argmax(ratio))
        if ratio[ratio_index] > maximum_raw_ratio:
            maximum_raw_ratio = float(ratio[ratio_index])
            maximum_raw_event = {
                "joint": joint,
                "source_endpoint": int(states["source_endpoint"][ratio_index]),
                "outcome_endpoint": int(states["outcome_endpoint"][ratio_index]),
                "substep": int(states["substep"][ratio_index]),
                "state_time_s": float(states["state_time_s"][ratio_index]),
                "raw_actuator_force_nm": float(raw[ratio_index]),
                "effort_limit_nm": limit,
                "ratio": float(ratio[ratio_index]),
            }
        constraint_index = int(np.argmax(np.abs(constraint)))
        if abs(constraint[constraint_index]) > maximum_constraint:
            maximum_constraint = float(abs(constraint[constraint_index]))
            maximum_constraint_event = {
                "joint": joint,
                "source_endpoint": int(states["source_endpoint"][constraint_index]),
                "outcome_endpoint": int(states["outcome_endpoint"][constraint_index]),
                "substep": int(states["substep"][constraint_index]),
                "state_time_s": float(states["state_time_s"][constraint_index]),
                "constraint_force_nm": float(constraint[constraint_index]),
            }

        for index in np.flatnonzero(terminal):
            error = abs(float(position_error[index]))
            if error > maximum_endpoint_position_error:
                maximum_endpoint_position_error = error
                maximum_endpoint_position_event = {
                    "joint": joint,
                    "endpoint": int(states["outcome_endpoint"][index]),
                    "position_error_rad": float(position_error[index]),
                }
            reference_velocity = float(states["reference_qvel"][index, dof])
            authored_velocity = reference_velocity * time_scale
            if abs(authored_velocity) >= 0.5:
                actual_velocity = float(states["actual_qvel"][index, dof])
                relative_error = abs(actual_velocity - reference_velocity) / abs(
                    reference_velocity
                )
                if relative_error > maximum_velocity_relative_error:
                    maximum_velocity_relative_error = relative_error
                    maximum_velocity_event = {
                        "joint": joint,
                        "endpoint": int(states["outcome_endpoint"][index]),
                        "reference_velocity_rad_s": reference_velocity,
                        "actual_velocity_rad_s": actual_velocity,
                        "relative_error": relative_error,
                    }
                allowed_position_error = abs(reference_velocity) * PHYSICS_DT_S + 0.0001
                same_direction = actual_velocity * reference_velocity > 0.0
                position_pass = error <= allowed_position_error
                velocity_pass = relative_error <= 0.10 and same_direction
                if not position_pass or not velocity_pass:
                    frozen_b_violations.append({
                        "joint": joint,
                        "endpoint": int(states["outcome_endpoint"][index]),
                        "position_error_rad": float(position_error[index]),
                        "allowed_position_error_rad": allowed_position_error,
                        "velocity_relative_error": relative_error,
                        "same_velocity_direction": same_direction,
                    })

        for index, native in enumerate(native_rows):
            detailed_rows.append({
                "experiment": label,
                "source_endpoint": int(states["source_endpoint"][index]),
                "outcome_endpoint": int(states["outcome_endpoint"][index]),
                "substep": int(states["substep"][index]),
                "substeps": int(states["substeps"][index]),
                "control_alpha": float(states["control_alpha"][index]),
                "state_time_s": float(states["state_time_s"][index]),
                "force_interval_start_time_s": float(
                    states["force_interval_start_time_s"][index]
                ),
                "force_interval_end_time_s": float(
                    states["force_interval_end_time_s"][index]
                ),
                "joint": joint,
                "effort_limit_nm": limit,
                "reference_position_post_state_rad": float(
                    states["reference_qpos"][index, qpos_address]
                ),
                "control_target_pre_step_rad": float(
                    states["control_target"][index, actuator]
                ),
                "actual_position_post_state_rad": float(
                    states["actual_qpos"][index, qpos_address]
                ),
                "position_error_post_state_rad": float(position_error[index]),
                "reference_velocity_rad_s": float(states["reference_qvel"][index, dof]),
                "actual_velocity_post_state_rad_s": float(
                    states["actual_qvel"][index, dof]
                ),
                "raw_actuator_force_pre_step_nm": float(raw[index]),
                "limited_joint_actuator_force_pre_step_nm": float(limited[index]),
                "external_applied_force_pre_step_nm": float(external[index]),
                "total_commanded_force_pre_step_nm": float(limited[index] + external[index]),
                "constraint_force_pre_step_nm": float(constraint[index]),
                "raw_request_saturated": bool(saturated[index]),
                "reference_left_table_distance_post_state_m": float(
                    native["reference_left_distance_m"]
                ),
                "reference_right_table_distance_post_state_m": float(
                    native["reference_right_distance_m"]
                ),
                "actual_left_table_distance_post_state_m": float(
                    native["actual_left_distance_m"]
                ),
                "actual_right_table_distance_post_state_m": float(
                    native["actual_right_distance_m"]
                ),
            })

    reference_distance = np.asarray([
        float(row["reference_minimum_distance_m"]) for row in native_rows
    ])
    actual_distance = np.asarray([
        float(row["actual_minimum_distance_m"]) for row in native_rows
    ])
    first_reference_index = first_index(
        reference_distance, reference_distance < -TOLERANCE_M,
    )
    first_actual_index = first_index(actual_distance, actual_distance < -TOLERANCE_M)
    minimum_reference_index = int(np.argmin(reference_distance))
    minimum_actual_index = int(np.argmin(actual_distance))
    first_reference = native_event(native_rows, "reference", first_reference_index)
    first_actual = native_event(native_rows, "actual", first_actual_index)
    minimum_reference = native_event(native_rows, "reference", minimum_reference_index)
    minimum_actual = native_event(native_rows, "actual", minimum_actual_index)
    scene = run / "experiment_local_scene.xml"
    for event, state_name in (
        (first_reference, "reference_qpos"),
        (first_actual, "actual_qpos"),
        (minimum_reference, "reference_qpos"),
        (minimum_actual, "actual_qpos"),
    ):
        if event is not None:
            event["native_location"] = detailed_native_location(
                scene,
                states[state_name][event["trace_index"]],
                plane,
                event["side"],
            )

    pinky_metrics: dict[str, Any] = {}
    for joint in ("left_hand_pinky_joint1", "left_hand_pinky_joint2"):
        contract = next(row for row in contracts if row["joint_name"] == joint)
        mask = (
            (states["source_endpoint"] == 5)
            & (states["outcome_endpoint"] == 6)
            & terminal
        )
        index = int(np.flatnonzero(mask)[0])
        dof = int(contract["dof_address"])
        qpos_address = int(contract["qpos_address"])
        pinky_metrics[joint] = {
            "reference_velocity_rad_s": float(states["reference_qvel"][index, dof]),
            "actual_velocity_rad_s": float(states["actual_qvel"][index, dof]),
            "actual_over_reference_speed": float(
                abs(states["actual_qvel"][index, dof])
                / abs(states["reference_qvel"][index, dof])
            ),
            "position_error_rad": float(
                states["reference_qpos"][index, qpos_address]
                - states["actual_qpos"][index, qpos_address]
            ),
        }

    total_joint_substeps = len(states["substep"]) * len(contracts)
    summary = {
        "experiment": label,
        "inputs": {
            "manifest": artifact(run / "input_manifest.json"),
            "results": artifact(run / "results.json"),
            "local_scene": artifact(run / "experiment_local_scene.xml"),
            "substep_state_trace": artifact(run / "substep_state_trace.npz"),
            "substep_native_hand_table_trace": artifact(
                run / "substep_native_hand_table_trace.csv.gz"
            ),
            "contact_trace": artifact(run / "contact_trace.csv.gz"),
        },
        "repository_head_recorded_by_run": manifest["repository_head"],
        "classification": result["classification"],
        "hard_stop": result["hard_stop"],
        "physics_steps": int(result["physics_steps"]),
        "state_duration_s": float(states["state_time_s"][-1]),
        "reference_time_scale": time_scale,
        "compiled_armature_kg_m2": sorted({
            float(row["compiled_armature_kg_m2"]) for row in contracts
        }),
        "compiled_velocity_gain_range_nm_s_rad": [
            min(float(row["compiled_velocity_gain_nm_s_rad"]) for row in contracts),
            max(float(row["compiled_velocity_gain_nm_s_rad"]) for row in contracts),
        ],
        "finger_external_applied_force_max_abs_nm": float(max(
            abs(row["external_applied_force_pre_step_nm"]) for row in detailed_rows
        )),
        "finger_total_commanded_force_within_limit": all(
            abs(row["total_commanded_force_pre_step_nm"])
            <= row["effort_limit_nm"] + 1e-10
            for row in detailed_rows
        ),
        "saturation": {
            "raw_request_saturated_joint_substeps": saturated_joint_substeps,
            "total_joint_substeps": total_joint_substeps,
            "fraction": saturated_joint_substeps / total_joint_substeps,
            "physics_steps_with_any_saturation": len(saturated_step_indices),
            "maximum_raw_request_over_limit": maximum_raw_event,
        },
        "tracking": {
            "maximum_terminal_position_error": maximum_endpoint_position_event,
            "maximum_comparable_terminal_velocity_relative_error": (
                maximum_velocity_event
            ),
            "frozen_rule_violation_count": len(frozen_b_violations),
            "frozen_rule_violations": frozen_b_violations,
            "endpoint_5_to_6_left_pinky": pinky_metrics,
        },
        "native_hand_table": {
            "tolerance_m": TOLERANCE_M,
            "first_reference_beyond_tolerance": first_reference,
            "first_actual_beyond_tolerance": first_actual,
            "minimum_reference": minimum_reference,
            "minimum_actual": minimum_actual,
        },
        "proxy_contacts": contact_summary(run / "contact_trace.csv.gz"),
        "maximum_abs_finger_constraint_force": maximum_constraint_event,
    }
    return summary, detailed_rows


def make_plot(summaries: dict[str, dict[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 9})
    figure, axes = plt.subplots(4, 3, figsize=(15, 12), sharex="col")
    for column, (label, run) in enumerate(RUNS.items()):
        manifest = load_json(run / "input_manifest.json")
        with np.load(run / "substep_state_trace.npz", allow_pickle=False) as archive:
            states = {name: np.asarray(archive[name]) for name in archive.files}
        native = read_csv_gz(run / "substep_native_hand_table_trace.csv.gz")
        contract = next(
            row for row in manifest["finger_dynamics"]["compiled_contracts"]
            if row["joint_name"] == PLOT_JOINT
        )
        qpos = int(contract["qpos_address"])
        dof = int(contract["dof_address"])
        actuator = int(contract["actuator_id"])
        limit = float(contract["urdf_effort_limit_nm"])
        x = states["source_endpoint"] + states["control_alpha"]
        axes[0, column].plot(x, states["reference_qpos"][:, qpos], label="参考")
        axes[0, column].plot(x, states["actual_qpos"][:, qpos], label="实际")
        axes[1, column].plot(
            x,
            1000.0 * (states["reference_qpos"][:, qpos] - states["actual_qpos"][:, qpos]),
            color="tab:red",
        )
        axes[2, column].plot(
            x, states["preceding_interval_actuator_force"][:, actuator],
            label="限幅前请求", alpha=0.8,
        )
        axes[2, column].plot(
            x, states["preceding_interval_qfrc_actuator"][:, dof],
            label="限幅后力矩", linewidth=1.5,
        )
        axes[2, column].axhline(limit, color="black", linestyle="--", linewidth=0.8)
        axes[2, column].axhline(-limit, color="black", linestyle="--", linewidth=0.8)
        reference_distance = np.asarray([
            float(row["reference_minimum_distance_m"]) * 1000.0 for row in native
        ])
        actual_distance = np.asarray([
            float(row["actual_minimum_distance_m"]) * 1000.0 for row in native
        ])
        axes[3, column].plot(x, reference_distance, label="参考")
        axes[3, column].plot(x, actual_distance, label="实际")
        axes[3, column].axhline(
            -TOLERANCE_M * 1000.0, color="black", linestyle="--", linewidth=0.8,
        )
        axes[0, column].set_title(
            f"实验 {label}，时间放大 {summaries[label]['reference_time_scale']:g} 倍"
        )
        axes[3, column].set_xlabel("参考路径位置（端点编号 + 区间比例）")
        for row in range(4):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("左小指关节2位置 (rad)")
    axes[1, 0].set_ylabel("参考减实际 (mrad)")
    axes[2, 0].set_ylabel("关节力矩 (N·m)")
    axes[3, 0].set_ylabel("原生手—桌最小距离 (mm)")
    axes[0, 0].legend()
    axes[2, 0].legend()
    axes[3, 0].legend()
    figure.suptitle("20231006_145 手指动力学 A/B/C 对照", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUTPUT / "finger_dynamics_abc_comparison.png", dpi=180)
    plt.close(figure)


def fmt_event(event: dict[str, Any] | None) -> str:
    if event is None:
        return "未出现"
    location = event["native_location"]
    side = "左" if event["side"] == "left" else "右"
    return (
        f"{event['source_endpoint']}→{event['outcome_endpoint']} 子步 "
        f"{event['substep']}/{event['substeps'] - 1}，"
        f"路径坐标 {event['geometric_endpoint_coordinate']:.4f}，"
        f"状态时刻 {event['state_time_s']:.3f} s，"
        f"{side}手，{event['distance_mm']:.3f} mm，"
        f"部件 `{location['body']}`"
    )


def write_report(summaries: dict[str, dict[str, Any]]) -> None:
    a, b, c = (summaries[label] for label in ("A", "B", "C"))
    lines = [
        "# 20231006_145 手指力矩上限、惯量与重定时对照",
        "",
        "## 实验保持不变的部分",
        "",
        "三组都从参考端点 0 的完整安全进入状态开始，物体使用同一份自然落座状态，"
        "物理步长为 2 ms，控制目标均在每个物理子步做线性插值。桌面、碰撞几何、"
        "MINK 参考端点和手腕的重力前馈及参考速度反馈均未改变。",
        "",
        "手指力矩限制取自左右 XHand URDF：不同关节为 0.4 或 1.1 N·m。限制施加在"
        "关节的 `actuatorfrcrange`，即执行器传动后的总关节力矩。三组的手指"
        " `qfrc_applied` 最大绝对值都是 0，因此没有外加广义力绕过力矩上限。",
        "",
        "## 三组设置",
        "",
        "| 实验 | 手指附加转动惯量 | 编译后速度反馈系数 | 时间轴 | 物理子步 |",
        "|---|---:|---:|---:|---:|",
        f"| A | {a['compiled_armature_kg_m2'][0]:.9f} | "
        f"{a['compiled_velocity_gain_range_nm_s_rad'][0]:.3f}–"
        f"{a['compiled_velocity_gain_range_nm_s_rad'][1]:.3f} | 原速 | {a['physics_steps']} |",
        f"| B | {b['compiled_armature_kg_m2'][0]:.9f} | "
        f"{b['compiled_velocity_gain_range_nm_s_rad'][0]:.3f}–"
        f"{b['compiled_velocity_gain_range_nm_s_rad'][1]:.3f} | 原速 | {b['physics_steps']} |",
        f"| C | {c['compiled_armature_kg_m2'][0]:.9f} | "
        f"{c['compiled_velocity_gain_range_nm_s_rad'][0]:.3f}–"
        f"{c['compiled_velocity_gain_range_nm_s_rad'][1]:.3f} | 整体放慢 10 倍 | {c['physics_steps']} |",
        "",
        "这里的手指附加转动惯量就是 MuJoCo XML 中的 `armature`。B 的候选值在运行前"
        "已固定为稳定性下界与力矩上界的几何平均值，"
        "没有按碰撞结果挑选。C 的 10 倍来自 B 中“限幅前力矩请求 / URDF 上限”的"
        "最大值 9.966 向上取整，也没有扫描多个速度。C 同时延长机器人和物体参考的"
        "全部端点时刻，不是只放慢手指。",
        "",
        "## 跟踪与力矩",
        "",
        "| 实验 | 最大端点位置误差 | 最大可比速度相对误差 | 饱和的关节-子步 | 有饱和的物理子步 | 最大限幅前请求/上限 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in (a, b, c):
        position = item["tracking"]["maximum_terminal_position_error"]
        velocity = item["tracking"]["maximum_comparable_terminal_velocity_relative_error"]
        saturation = item["saturation"]
        lines.append(
            f"| {item['experiment']} | {abs(position['position_error_rad']):.6f} rad "
            f"(`{position['joint']}`, 参考端点 {position['endpoint']}) | "
            f"{100.0 * velocity['relative_error']:.2f}% | "
            f"{saturation['raw_request_saturated_joint_substeps']}/"
            f"{saturation['total_joint_substeps']} | "
            f"{saturation['physics_steps_with_any_saturation']}/"
            f"{item['physics_steps']} | "
            f"{saturation['maximum_raw_request_over_limit']['ratio']:.3f} |"
        )
    lines.extend([
        "",
        "速度相对误差只统计三组中对应原始参考速度不小于 0.5 rad/s 的同一批快速运动关节；"
        "C 的比较先把执行速度按 10 倍时间缩放关系换回原始参考速度范围。B 在预先固定的"
        f"端点跟踪规则下共有 {b['tracking']['frozen_rule_violation_count']} 个关节-端点超限，"
        "因此按实验合同触发 C。",
        "",
        "在 5→6 区间末端，左小指关节1/2 的实际速度占参考速度幅值分别为：",
        "",
        f"- A：{100*a['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint1']['actual_over_reference_speed']:.2f}% / "
        f"{100*a['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint2']['actual_over_reference_speed']:.2f}%。",
        f"- B：{100*b['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint1']['actual_over_reference_speed']:.2f}% / "
        f"{100*b['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint2']['actual_over_reference_speed']:.2f}%。",
        f"- C：{100*c['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint1']['actual_over_reference_speed']:.2f}% / "
        f"{100*c['tracking']['endpoint_5_to_6_left_pinky']['left_hand_pinky_joint2']['actual_over_reference_speed']:.2f}%。",
        "",
        "A 中限幅后的手指跟踪几乎停住，随后位置误差不断积累，限幅前请求最高达到"
        f" {a['saturation']['maximum_raw_request_over_limit']['raw_actuator_force_nm']:.3f} N·m。"
        "B 的速度跟踪接近参考，但仍存在位置滞后和 45 个饱和关节-子步。C 的限幅前"
        "请求最大只到上限的 39.86%，没有手指力矩饱和。",
        "",
        "三组限幅后的总手指控制力矩都没有超过 URDF 上限。手指约束力最大绝对值"
        f"依次为 {abs(a['maximum_abs_finger_constraint_force']['constraint_force_nm']):.3f}、"
        f"{abs(b['maximum_abs_finger_constraint_force']['constraint_force_nm']):.3f}、"
        f"{abs(c['maximum_abs_finger_constraint_force']['constraint_force_nm']):.4f} N·m。",
        "",
        "## 原生手—桌距离",
        "",
        "距离低于 -0.05 mm 记为超过既定容差。这里的“参考”是同一条关节位置插值路径，"
        "“实际”是 MuJoCo 动力学执行后的机器人状态。",
        "",
        "| 实验 | 参考最深距离 | 实际最深距离 | 参考首次超限 | 实际首次超限 |",
        "|---|---:|---:|---|---|",
    ])
    for item in (a, b, c):
        native = item["native_hand_table"]
        lines.append(
            f"| {item['experiment']} | {native['minimum_reference']['distance_mm']:.3f} mm | "
            f"{native['minimum_actual']['distance_mm']:.3f} mm | "
            f"{fmt_event(native['first_reference_beyond_tolerance'])} | "
            f"{fmt_event(native['first_actual_beyond_tolerance'])} |"
        )
    lines.extend([
        "",
        "A 和 B 以原速运行，所以参考首次超限完全相同：6→7 的第 2 个子步后。"
        "C 的采样更密，记录到同一路径在 6→7、路径坐标约 6.144 时首次超限。"
        "这三个参考结果描述的是同一条几何路径，不是三条不同参考。",
        "",
        "C 虽然没有手指力矩饱和，实际手仍在 5→6 内先越过 -0.05 mm；"
        "其最深距离从 B 的 -1.057 mm 减少到 -0.334 mm，但没有变成全程不穿桌。"
        "同一条参考路径自身随后也达到约 -0.103 mm。",
        "",
        "## 代理接触与原生网格不是同一个信号",
        "",
        "| 实验 | 首个 MuJoCo 手—桌代理接触 | 最大手—桌代理接触力 | 手—桌代理接触冲量 |",
        "|---|---|---:|---:|",
    ])
    for item in (a, b, c):
        contact = item["proxy_contacts"]
        first = contact["first_hand_floor_contact"]
        first_text = (
            "未出现" if first is None else
            f"{first['source_endpoint']}→{first['outcome_endpoint']} 子步 "
            f"{first['substep']}，`{first['geom1']}` / `{first['geom2']}`"
        )
        lines.append(
            f"| {item['experiment']} | {first_text} | "
            f"{contact['maximum_hand_floor_force_n']:.3f} N | "
            f"{contact['hand_floor_normal_impulse_ns']:.6f} N·s |"
        )
    lines.extend([
        "",
        "A 的原生网格在 5→6 已超限，代理接触到 6→7 才出现；B 的原生网格在 "
        "6→7 超限，代理接触到 7→8 才出现；C 整段没有任何手—桌代理接触，"
        "但原生网格仍穿入桌面。因此，C 中“不发生代理接触”不能解释为“手没有穿桌”。",
        "",
        "## 可复核文件",
        "",
        "- `finger_substep_trace.csv.gz`：每个实验、每个物理子步、每个手指关节的完整数值。",
        "- `finger_dynamics_abc_comparison.png`：左小指关节2及原生手—桌距离的三组对照。",
        "- `results.json`：上述表格使用的结构化汇总、首次超限位置和最差原生网格部件。",
        "- 三个原始运行目录保留各自的状态、接触、原生距离和实验局部 XML，未覆盖历史 run。",
        "",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    support = yaml.safe_load((
        RL_ROOT / "configs/taco_brush_20231006_145_issue14_support_candidate_v1.yaml"
    ).read_text(encoding="utf-8"))["simulator"]
    plane = Plane(
        normal=np.asarray(support["normal"], dtype=np.float64),
        offset=float(support["offset_m"]), frame=str(support["frame"]),
    )
    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for label, run in RUNS.items():
        summaries[label], detail = analyze_run(label, run, plane)
        rows.extend(detail)
    write_rows(OUTPUT / "finger_substep_trace.csv.gz", rows, gzip_output=True)
    make_plot(summaries)
    write_report(summaries)
    results = {
        "schema": "taco_brush_20231006_145_finger_effort_armature_abc_analysis_v1",
        "status": "COMPLETE_READ_ONLY_COMPARISON",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "script": artifact(Path(__file__)),
        "contract": artifact(CONTRACT),
        "experiments": summaries,
        "outputs": {
            "finger_substep_trace": artifact(OUTPUT / "finger_substep_trace.csv.gz"),
            "comparison_plot": artifact(OUTPUT / "finger_dynamics_abc_comparison.png"),
            "report": artifact(OUTPUT / "report.md"),
        },
    }
    write_json(OUTPUT / "results.json", results)
    with (OUTPUT / "server_artifacts.sha256").open("w", encoding="utf-8") as stream:
        for path in sorted(OUTPUT.iterdir()):
            if path.name == "server_artifacts.sha256":
                continue
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({
        "status": results["status"],
        "experiments": {
            label: {
                "physics_steps": summary["physics_steps"],
                "minimum_actual_hand_table_mm": summary["native_hand_table"][
                    "minimum_actual"
                ]["distance_mm"],
                "saturated_joint_substeps": summary["saturation"][
                    "raw_request_saturated_joint_substeps"
                ],
            }
            for label, summary in summaries.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
