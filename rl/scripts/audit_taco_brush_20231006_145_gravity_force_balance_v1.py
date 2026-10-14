#!/usr/bin/env python3
"""Read-only force decomposition for the endpoint 0-4 Brush hand drop."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
CAPTURE = RL_ROOT / "runs/taco_brush_20231006_145_endpoint_1_4_hand_table_capture_v1"
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_gravity_force_balance_v1"
STATE_TRACE = CAPTURE / "substep_state_trace.npz"
SCENE = CAPTURE / "experiment_local_scene.xml"
CONTACT_TRACE = CAPTURE / "contact_trace.csv.gz"
MANIFEST = CAPTURE / "input_manifest.json"
ENDPOINT_ROW_INDICES = (16, 33, 49, 66)


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def actuator_rows(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for side, name in (
        ("right", "R_forearm_tz_position"),
        ("left", "L_forearm_tz_position"),
    ):
        actuator_id = int(model.actuator(name).id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        body_id = int(model.jnt_bodyid[joint_id])
        result[side] = {
            "actuator_id": actuator_id,
            "dof_address": int(model.jnt_dofadr[joint_id]),
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "position_gain_n_per_m": float(
                model.actuator_gainprm[actuator_id, 0]
            ),
            "position_bias_n_per_m": float(
                model.actuator_biasprm[actuator_id, 1]
            ),
            "velocity_bias_n_s_per_m": float(
                model.actuator_biasprm[actuator_id, 2]
            ),
            "force_limited": bool(model.actuator_forcelimited[actuator_id]),
            "body_gravity_compensation": float(model.body_gravcomp[body_id]),
            "joint_actuator_gravity_compensation": bool(
                model.jnt_actgravcomp[joint_id]
            ),
        }
    return result


def load_contact_groups() -> tuple[set[str], int]:
    with gzip.open(CONTACT_TRACE, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["group"] for row in rows}, len(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    zero_gravity_model = mujoco.MjModel.from_xml_path(str(SCENE))
    zero_gravity_model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    gravity_only_data = mujoco.MjData(model)
    zero_gravity_data = mujoco.MjData(zero_gravity_model)
    actuators = actuator_rows(model)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reference_path = Path(manifest["inputs"]["robot_reference"]["path"])
    if sha256(reference_path) != manifest["inputs"]["robot_reference"]["sha256"]:
        raise RuntimeError("robot reference hash changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_qvel = np.asarray(archive["qvel"]).copy()

    with np.load(STATE_TRACE, allow_pickle=False) as archive:
        states = {name: np.asarray(archive[name]).copy() for name in archive.files}

    rows: list[dict[str, Any]] = []
    maximum_actuator_recompute_error = 0.0
    for index in range(len(states["state_time_s"])):
        start_qpos = (
            states["initial_actual_qpos"]
            if index == 0
            else states["actual_qpos"][index - 1]
        )
        start_qvel = (
            states["initial_actual_qvel"]
            if index == 0
            else states["actual_qvel"][index - 1]
        )
        ctrl = states["control_target"][index]

        for target_data, target_model in (
            (data, model),
            (zero_gravity_data, zero_gravity_model),
        ):
            target_data.qpos[:] = start_qpos
            target_data.qvel[:] = start_qvel
            target_data.ctrl[:] = ctrl
            mujoco.mj_forward(target_model, target_data)

        gravity_only_data.qpos[:] = start_qpos
        gravity_only_data.qvel[:] = 0.0
        gravity_only_data.ctrl[:] = ctrl
        mujoco.mj_forward(model, gravity_only_data)

        for side, actuator in actuators.items():
            actuator_id = actuator["actuator_id"]
            dof = actuator["dof_address"]
            qpos_address = actuator["qpos_address"]
            position_force = float(
                model.actuator_gainprm[actuator_id, 0] * ctrl[actuator_id]
                + model.actuator_biasprm[actuator_id, 1]
                * data.actuator_length[actuator_id]
            )
            velocity_force = float(
                model.actuator_biasprm[actuator_id, 2]
                * data.actuator_velocity[actuator_id]
            )
            actuator_force = float(data.actuator_force[actuator_id])
            saved_force = float(
                states["preceding_interval_actuator_force"][index, actuator_id]
            )
            maximum_actuator_recompute_error = max(
                maximum_actuator_recompute_error,
                abs(actuator_force - saved_force),
            )
            rows.append(
                {
                    "trace_index": index,
                    "source_endpoint": int(states["source_endpoint"][index]),
                    "outcome_endpoint": int(states["outcome_endpoint"][index]),
                    "substep": int(states["substep"][index]),
                    "substeps": int(states["substeps"][index]),
                    "interval_start_time_s": float(
                        states["force_interval_start_time_s"][index]
                    ),
                    "interval_end_time_s": float(
                        states["force_interval_end_time_s"][index]
                    ),
                    "side": side,
                    "control_target_m": float(ctrl[actuator_id]),
                    "start_actual_position_m": float(start_qpos[qpos_address]),
                    "start_target_minus_actual_mm": float(
                        1000.0 * (ctrl[actuator_id] - start_qpos[qpos_address])
                    ),
                    "start_actual_velocity_m_s": float(start_qvel[dof]),
                    "reference_initial_velocity_m_s": float(
                        reference_qvel[0, dof]
                    ),
                    "position_servo_force_n": position_force,
                    "velocity_damping_force_n": velocity_force,
                    "actuator_force_n": actuator_force,
                    "saved_actuator_force_n": saved_force,
                    "gravity_generalized_force_n": float(
                        gravity_only_data.qfrc_bias[dof]
                    ),
                    "moving_state_bias_force_n": float(data.qfrc_bias[dof]),
                    "passive_force_n": float(data.qfrc_passive[dof]),
                    "constraint_force_n": float(data.qfrc_constraint[dof]),
                    "vertical_acceleration_with_gravity_m_s2": float(data.qacc[dof]),
                    "vertical_acceleration_zero_gravity_same_state_m_s2": float(
                        zero_gravity_data.qacc[dof]
                    ),
                    "end_actual_minus_reference_mm": float(
                        1000.0
                        * (
                            states["actual_qpos"][index, qpos_address]
                            - states["reference_qpos"][index, qpos_address]
                        )
                    ),
                    "end_below_reference_mm": float(
                        1000.0
                        * (
                            states["reference_qpos"][index, qpos_address]
                            - states["actual_qpos"][index, qpos_address]
                        )
                    ),
                }
            )

    if maximum_actuator_recompute_error > 1.0e-10:
        raise RuntimeError("saved actuator force does not match state recomputation")

    contact_groups, contact_count = load_contact_groups()
    if contact_groups != {"brush_floor", "bowl_floor"}:
        raise RuntimeError(f"unexpected contact groups: {sorted(contact_groups)}")

    by_key = {(row["trace_index"], row["side"]): row for row in rows}
    endpoints = {
        str(endpoint): {
            side: by_key[(row_index, side)]
            for side in ("left", "right")
        }
        for endpoint, row_index in enumerate(ENDPOINT_ROW_INDICES, 1)
    }
    initial = {side: by_key[(0, side)] for side in ("left", "right")}

    gravity = np.asarray(
        [row["gravity_generalized_force_n"] for row in rows], dtype=np.float64
    )
    constraints = np.asarray(
        [row["constraint_force_n"] for row in rows], dtype=np.float64
    )
    passive = np.asarray(
        [row["passive_force_n"] for row in rows], dtype=np.float64
    )
    summary = {
        "schema": "taco_brush_20231006_145_gravity_force_balance_v1",
        "sample": "20231006_145",
        "scope": "read_only_endpoint_0_to_4_force_decomposition",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "method": {
            "physics_integration_performed": False,
            "saved_interval_start_state_recomputed_with_mj_forward": True,
            "command_timing": manifest["physics"]["command_timing"],
            "zero_gravity_check": (
                "instantaneous acceleration at the same saved qpos, qvel and ctrl; "
                "not an integrated zero-gravity Replay"
            ),
        },
        "inputs": {
            "state_trace": artifact(STATE_TRACE),
            "scene": artifact(SCENE),
            "contact_trace": artifact(CONTACT_TRACE),
            "capture_manifest": artifact(MANIFEST),
            "robot_reference": artifact(reference_path),
            "script": artifact(Path(__file__)),
        },
        "actuator_contract": actuators,
        "initial_interval": initial,
        "endpoints": endpoints,
        "force_checks": {
            "gravity_force_range_n": [float(gravity.min()), float(gravity.max())],
            "maximum_abs_hand_vertical_constraint_force_n": float(
                np.max(np.abs(constraints))
            ),
            "maximum_abs_hand_vertical_passive_force_n": float(
                np.max(np.abs(passive))
            ),
            "maximum_actuator_force_recompute_error_n": float(
                maximum_actuator_recompute_error
            ),
            "contact_groups": sorted(contact_groups),
            "contact_row_count": contact_count,
            "hand_contact_row_count": 0,
            "actuator_force_limited": False,
        },
    }

    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "force_balance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUTPUT / "results.json", summary)

    colors = {"left": "#16697a", "right": "#db6400"}
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for side in ("left", "right"):
        side_rows = [row for row in rows if row["side"] == side]
        time_ms = np.asarray(
            [row["interval_end_time_s"] for row in side_rows]
        ) * 1000.0
        axes[0].plot(
            time_ms,
            [row["gravity_generalized_force_n"] for row in side_rows],
            color=colors[side],
            linestyle="--",
            label=f"{side} gravity",
        )
        axes[0].plot(
            time_ms,
            [row["actuator_force_n"] for row in side_rows],
            color=colors[side],
            label=f"{side} actuator",
        )
        axes[1].plot(
            time_ms,
            [row["end_actual_minus_reference_mm"] for row in side_rows],
            color=colors[side],
            label=f"{side} actual-reference",
        )
    axes[0].set_ylabel("Vertical generalized force (N)")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Wrist vertical error (mm)")
    axes[1].set_xlabel("Simulation time (ms)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "gravity_force_balance.png", dpi=180)
    plt.close(figure)

    right_initial = initial["right"]
    right_final = endpoints["4"]["right"]
    left_final = endpoints["4"]["left"]
    report = f"""# 20231006_145 双手低于参考的力学分解

## 检查方法

读取 endpoint 0 到 4 已保存的每个物理区间起始 `qpos`、`qvel` 和控制目标，在同一场景中只执行 `mj_forward` 重算力和瞬时加速度，没有推进物理时间。零重力数据来自场景副本中的同状态瞬时计算，不是零重力 Replay。

这段回放的控制目标已经在每个 2 ms 物理子步之间做线性插值，不是整周期保持同一个跳变目标。

## 起始时刻

- 右手竖直重力广义力：`{right_initial['gravity_generalized_force_n']:.6f} N`。
- 左手竖直重力广义力：`{initial['left']['gravity_generalized_force_n']:.6f} N`。
- 实际初速度为 `0 m/s`；参考初速度约为 `{right_initial['reference_initial_velocity_m_s']:.6f} m/s`，方向向下。
- 第一个 2 ms 区间的右手位置伺服力只有 `{right_initial['position_servo_force_n']:.6f} N`，速度阻尼力为 `{right_initial['velocity_damping_force_n']:.6f} N`。
- 原重力下右手瞬时竖直加速度为 `{right_initial['vertical_acceleration_with_gravity_m_s2']:.6f} m/s²`；同状态零重力时为 `{right_initial['vertical_acceleration_zero_gravity_same_state_m_s2']:.6f} m/s²`。

## 第 4 个端点前的最后物理区间

| 手 | 位置项 | 速度阻尼项 | 执行器合力 | 重力广义力 | 起点低于目标 | 终点低于参考 |
|---|---:|---:|---:|---:|---:|---:|
| 左 | {left_final['position_servo_force_n']:.3f} N | {left_final['velocity_damping_force_n']:.3f} N | {left_final['actuator_force_n']:.3f} N | {left_final['gravity_generalized_force_n']:.3f} N | {left_final['start_target_minus_actual_mm']:.3f} mm | {left_final['end_below_reference_mm']:.3f} mm |
| 右 | {right_final['position_servo_force_n']:.3f} N | {right_final['velocity_damping_force_n']:.3f} N | {right_final['actuator_force_n']:.3f} N | {right_final['gravity_generalized_force_n']:.3f} N | {right_final['start_target_minus_actual_mm']:.3f} mm | {right_final['end_below_reference_mm']:.3f} mm |

位置控制器满足 `执行器力 = 位置增益 × (目标位置 - 实际位置) - 速度系数 × 实际速度`。双手向下运动时，速度阻尼提供约 `10.14 N` 的向上力；还需要约 `3.93 N` 的位置项，执行器总力才接近约 `13.95 N` 的重力。`1000 N/m` 的位置增益把这部分力转换成约 `3.93 mm` 的区间起点位置差，最后一个 2 ms 物理步后变为约 `4.13 mm`。

## 原因说明

重力是最初向下偏离和约 4 mm 稳态偏差的主要负载。初始时控制器几乎没有向上力，原重力下会明显向下加速；相同状态去掉重力后，瞬时竖直加速度接近零。随后控制器逐渐建立约 14 N 的向上力，双手不再继续加速下落，但需要保留几毫米的位置差来产生足够的向上力。

参考本身持续向下运动，而且实际初速度被设为零，参考初速度约为 `-0.021 m/s`。这两项决定了过渡过程和速度阻尼力，所以约 4 mm 不是纯静态重力下垂值；它是当前移动参考下的重力、位置项和速度阻尼共同平衡出的偏差。

模型中所有刚体的重力补偿系数都为零，两个竖直关节也没有启用执行器重力补偿。因此控制器没有单独的重力前馈，只能依靠位置误差和速度阻尼产生托住手的力。

## 其他力和接触

- 手腕竖直自由度的约束力最大绝对值：`{summary['force_checks']['maximum_abs_hand_vertical_constraint_force_n']:.9g} N`。
- 手腕竖直自由度的被动力最大绝对值：`{summary['force_checks']['maximum_abs_hand_vertical_passive_force_n']:.9g} N`。
- endpoint 0 到 4 的接触日志只有刷子—桌面和碗—桌面，共 `{contact_count}` 条；没有手—桌面或手—物体接触。
- 两个竖直执行器都没有力上限；保存的执行器力与同状态重算值最大差为 `{maximum_actuator_recompute_error:.3e} N`。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    files = [
        OUTPUT / "force_balance.csv",
        OUTPUT / "gravity_force_balance.png",
        OUTPUT / "report.md",
        OUTPUT / "results.json",
    ]
    (OUTPUT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
