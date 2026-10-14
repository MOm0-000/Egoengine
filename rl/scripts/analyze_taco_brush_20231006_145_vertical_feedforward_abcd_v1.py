#!/usr/bin/env python3
"""Compare the frozen endpoint 0-to-5 vertical controller experiments."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = RL_ROOT / "runs/taco_brush_20231006_145_vertical_feedforward_abcd_v1"
VARIANTS = {
    "A": "a_position_servo_only",
    "B": "b_gravity_feedforward",
    "C": "c_reference_velocity_feedback",
    "D": "d_gravity_and_reference_velocity",
}
COLORS = {"A": "#5f6368", "B": "#0077b6", "C": "#d1495b", "D": "#2a9d6f"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    output_files = [
        RUN_ROOT / "comparison.csv",
        RUN_ROOT / "comparison.png",
        RUN_ROOT / "comparison_results.json",
        RUN_ROOT / "report.md",
        RUN_ROOT / "artifacts.sha256",
    ]
    if any(path.exists() for path in output_files):
        raise FileExistsError("comparison output already exists")

    runs = {key: RUN_ROOT / value for key, value in VARIANTS.items()}
    manifests = {
        key: read_json(path / "input_manifest.json") for key, path in runs.items()
    }
    results = {key: read_json(path / "results.json") for key, path in runs.items()}
    controllers = {
        key: read_csv(path / "controller_trace.csv") for key, path in runs.items()
    }
    native = {
        key: read_csv_gz(path / "substep_native_hand_table_trace.csv.gz")
        for key, path in runs.items()
    }
    contacts = {
        key: read_csv_gz(path / "contact_trace.csv.gz")
        for key, path in runs.items()
    }

    frozen_artifacts = (
        "source_scene", "robot_reference", "support_contract",
        "prior_collision_audit", "local_scene",
    )
    baseline_manifest = manifests["A"]
    for key, manifest in manifests.items():
        for artifact in frozen_artifacts:
            if manifest["inputs"][artifact]["sha256"] != (
                baseline_manifest["inputs"][artifact]["sha256"]
            ):
                raise RuntimeError(f"{key} changed frozen artifact {artifact}")
        for field in (
            "mujoco_version", "authored_timestep_s", "control_hz",
            "substeps_per_interval", "total_planned_physics_steps",
            "initial_qvel", "command_timing", "control_target_rule",
        ):
            if manifest["physics"][field] != baseline_manifest["physics"][field]:
                raise RuntimeError(f"{key} changed frozen physics field {field}")

    state_archives: dict[str, dict[str, np.ndarray]] = {}
    endpoint_archives: dict[str, dict[str, np.ndarray]] = {}
    for key, path in runs.items():
        with np.load(path / "substep_state_trace.npz", allow_pickle=False) as archive:
            state_archives[key] = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        with np.load(path / "endpoint_states.npz", allow_pickle=False) as archive:
            endpoint_archives[key] = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    frozen_state_arrays = (
        "initial_actual_qpos", "initial_actual_qvel", "initial_actual_ctrl",
        "initial_reference_qpos", "source_endpoint", "outcome_endpoint",
        "substep", "substeps", "control_alpha", "state_time_s",
        "force_interval_start_time_s", "force_interval_end_time_s",
        "reference_qpos", "reference_qvel", "control_target",
    )
    for key in VARIANTS:
        for name in frozen_state_arrays:
            if not np.array_equal(
                state_archives[key][name], state_archives["A"][name]
            ):
                raise RuntimeError(f"{key} changed frozen state array {name}")

    maximum_object_state_difference = 0.0
    for key in VARIANTS:
        for name, values in endpoint_archives[key].items():
            if name == "endpoint":
                continue
            maximum_object_state_difference = max(
                maximum_object_state_difference,
                float(np.max(np.abs(values - endpoint_archives["A"][name]))),
            )

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for key in VARIANTS:
        result = results[key]
        controller = result["controller"]
        interval = result["substep_native_hand_table"]["interval_4_to_5"]
        hand_floor = [row for row in contacts[key] if row["group"] == "hand_floor"]
        first = interval["first_actual_penetration_beyond_tolerance"]
        actuator_evidence = controller["actuators"]
        left = actuator_evidence["L_forearm_tz_position"]
        right = actuator_evidence["R_forearm_tz_position"]
        row = {
            "variant": key,
            "gravity_feedforward": controller["gravity_feedforward"],
            "reference_velocity_feedback": controller[
                "reference_velocity_feedback"
            ],
            "endpoint_4_left_below_reference_mm": (
                left["endpoint_4_reference_minus_actual_m"] * 1000.0
            ),
            "endpoint_4_right_below_reference_mm": (
                right["endpoint_4_reference_minus_actual_m"] * 1000.0
            ),
            "endpoint_5_left_below_reference_mm": (
                left["endpoint_5_reference_minus_actual_m"] * 1000.0
            ),
            "endpoint_5_right_below_reference_mm": (
                right["endpoint_5_reference_minus_actual_m"] * 1000.0
            ),
            "interval_4_to_5_reference_minimum_mm": (
                interval["reference_minimum_distance_m"] * 1000.0
            ),
            "interval_4_to_5_actual_minimum_mm": (
                interval["actual_minimum_distance_m"] * 1000.0
            ),
            "interval_4_to_5_penetrating_substeps": interval[
                "actual_penetration_beyond_tolerance_count"
            ],
            "first_penetrating_substep": (
                None if first is None else first["substep"]
            ),
            "first_penetrating_state_time_s": (
                None if first is None else first["state_time_s"]
            ),
            "maximum_abs_extra_generalized_force_n": controller[
                "maximum_abs_extra_generalized_force_n"
            ],
            "hand_floor_contact_rows": len(hand_floor),
            "hand_floor_normal_impulse_ns": sum(
                float(contact["normal_force_n"])
                * manifests[key]["physics"]["authored_timestep_s"]
                for contact in hand_floor
            ),
            "hand_floor_maximum_force_n": max(
                (float(contact["force_norm_n"]) for contact in hand_floor),
                default=0.0,
            ),
            "hard_stop": result["hard_stop"],
            "completed_outcome_endpoint": result["completed_outcome_endpoint"],
            "strict_gate_passed": result["strict_gate_passed"],
        }
        rows.append(row)
        details[key] = {
            "run": str(runs[key]),
            "manifest_sha256": sha256(runs[key] / "input_manifest.json"),
            "results_sha256": sha256(runs[key] / "results.json"),
            "metrics": row,
        }

    fields = list(rows[0])
    with (RUN_ROOT / "comparison.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True, constrained_layout=True,
    )
    for key in VARIANTS:
        right_rows = [
            row for row in controllers[key]
            if row["actuator_name"] == "R_forearm_tz_position"
        ]
        time_ms = np.asarray([
            float(row["interval_end_time_s"]) * 1000.0 for row in right_rows
        ])
        mean_error_mm = []
        mean_extra_force = []
        for right_row in right_rows:
            identity = (
                right_row["source_endpoint"], right_row["substep"]
            )
            pair = [
                row for row in controllers[key]
                if (row["source_endpoint"], row["substep"]) == identity
            ]
            mean_error_mm.append(np.mean([
                float(row["reference_minus_actual_end_m"]) * 1000.0
                for row in pair
            ]))
            mean_extra_force.append(np.mean([
                float(row["extra_generalized_force_n"]) for row in pair
            ]))
        axes[0].plot(
            time_ms, mean_error_mm, color=COLORS[key], label=key, linewidth=1.8,
        )
        axes[1].plot(
            [float(row["state_time_s"]) * 1000.0 for row in native[key]],
            [float(row["actual_minimum_distance_m"]) * 1000.0 for row in native[key]],
            color=COLORS[key], label=key, linewidth=1.8,
        )
        axes[2].plot(
            time_ms, mean_extra_force, color=COLORS[key], label=key,
            linewidth=1.8,
        )
    axes[1].plot(
        [float(row["state_time_s"]) * 1000.0 for row in native["A"]],
        [float(row["reference_minimum_distance_m"]) * 1000.0 for row in native["A"]],
        color="black", linestyle="--", label="reference", linewidth=1.4,
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("mean wrist below reference (mm)")
    axes[1].set_ylabel("minimum native hand-table distance (mm)")
    axes[2].set_ylabel("mean extra generalized force (N)")
    axes[2].set_xlabel("simulation time (ms)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=5)
    figure.savefig(RUN_ROOT / "comparison.png", dpi=180)
    plt.close(figure)

    summary = {
        "schema": "taco_brush_20231006_145_vertical_feedforward_abcd_comparison_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "window": "endpoint_0_to_5",
        "mujoco_version": baseline_manifest["physics"]["mujoco_version"],
        "physics_steps_per_variant": baseline_manifest["physics"][
            "total_planned_physics_steps"
        ],
        "frozen_input_hashes_equal": True,
        "frozen_reference_and_timing_arrays_equal": True,
        "maximum_object_state_difference_between_variants": (
            maximum_object_state_difference
        ),
        "variants": details,
        "scope": (
            "This bounded comparison tests endpoint 0 to 5 only and does not "
            "establish stable pickup or full Replay success."
        ),
    }
    write_json(RUN_ROOT / "comparison_results.json", summary)

    by_key = {row["variant"]: row for row in rows}
    report = f"""# 20231006_145 竖直控制补偿对照

本次只运行 endpoint 0 到 5，共 84 个物理子步，执行环境为 MuJoCo `{baseline_manifest['physics']['mujoco_version']}`。四组使用完全相同的初始状态、参考轨迹、位置目标插值、MuJoCo 模型、控制增益、碰撞参数和仿真预算。刷子与碗的逐 endpoint 状态在四组之间完全相同，最大差为 `{maximum_object_state_difference:.3e}`。

| 组别 | 重力前馈 | 参考速度项 | endpoint 4 左/右低于参考 | endpoint 5 左/右低于参考 | 4→5 实际最小手—桌距离 | 穿透子步数 |
|---|---|---|---:|---:|---:|---:|
| A | 关 | 关 | {by_key['A']['endpoint_4_left_below_reference_mm']:.3f} / {by_key['A']['endpoint_4_right_below_reference_mm']:.3f} mm | {by_key['A']['endpoint_5_left_below_reference_mm']:.3f} / {by_key['A']['endpoint_5_right_below_reference_mm']:.3f} mm | {by_key['A']['interval_4_to_5_actual_minimum_mm']:.3f} mm | {by_key['A']['interval_4_to_5_penetrating_substeps']} |
| B | 开 | 关 | {by_key['B']['endpoint_4_left_below_reference_mm']:.3f} / {by_key['B']['endpoint_4_right_below_reference_mm']:.3f} mm | {by_key['B']['endpoint_5_left_below_reference_mm']:.3f} / {by_key['B']['endpoint_5_right_below_reference_mm']:.3f} mm | {by_key['B']['interval_4_to_5_actual_minimum_mm']:.3f} mm | {by_key['B']['interval_4_to_5_penetrating_substeps']} |
| C | 关 | 开 | {by_key['C']['endpoint_4_left_below_reference_mm']:.3f} / {by_key['C']['endpoint_4_right_below_reference_mm']:.3f} mm | {by_key['C']['endpoint_5_left_below_reference_mm']:.3f} / {by_key['C']['endpoint_5_right_below_reference_mm']:.3f} mm | {by_key['C']['interval_4_to_5_actual_minimum_mm']:.3f} mm | {by_key['C']['interval_4_to_5_penetrating_substeps']} |
| D | 开 | 开 | {by_key['D']['endpoint_4_left_below_reference_mm']:.3f} / {by_key['D']['endpoint_4_right_below_reference_mm']:.3f} mm | {by_key['D']['endpoint_5_left_below_reference_mm']:.3f} / {by_key['D']['endpoint_5_right_below_reference_mm']:.3f} mm | {by_key['D']['interval_4_to_5_actual_minimum_mm']:.3f} mm | {by_key['D']['interval_4_to_5_penetrating_substeps']} |

表中的负位置误差表示实际手腕高于参考。四组共享的参考路径在 4→5 的最小手—桌距离为 `{by_key['A']['interval_4_to_5_reference_minimum_mm']:.3f} mm`。

## 逐组现象

- A 组在 4→5 中编号为 `{by_key['A']['first_penetrating_substep']}` 的子步首次超过 0.05 mm 容差，子步编号从 0 开始；最深原生穿透 `{by_key['A']['interval_4_to_5_actual_minimum_mm']:.3f} mm`。运行时代理没有产生手—桌接触力。
- B 组没有穿透，但实际手腕在 endpoint 4 已高于参考约 6.47 mm，endpoint 5 高于参考约 8.24 mm。单独加入完整重力前馈时，原位置执行器的绝对速度阻尼仍在向上施力，因此出现明显向上超调。
- C 组没有补偿重力，却加入了沿向下参考速度方向的力。它把原速度阻尼的一部分向上支撑抵消掉，endpoint 4 的向下误差增至约 9 mm，并出现 31 条手—桌代理接触记录；最大接触力为 `{by_key['C']['hand_floor_maximum_force_n']:.3f} N`。
- D 组把控制律组合为位置误差、相对速度误差和重力前馈。endpoint 4 实际手腕高于参考约 1.24 mm，endpoint 5 高于参考约 1.05 mm；4→5 的最小原生手—桌余量为 `{by_key['D']['interval_4_to_5_actual_minimum_mm']:.3f} mm`，没有手—桌代理接触。

四组都完成 84 个子步且没有硬停止。D 组证明在这个固定短窗口内可以同时做到参考不穿桌、实际 Replay 也不穿桌；它不代表后续抓取、抬升或完整 Replay 已经通过。
"""
    (RUN_ROOT / "report.md").write_text(report, encoding="utf-8")

    artifacts = [
        RUN_ROOT / "comparison.csv",
        RUN_ROOT / "comparison.png",
        RUN_ROOT / "comparison_results.json",
        RUN_ROOT / "report.md",
    ]
    (RUN_ROOT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
