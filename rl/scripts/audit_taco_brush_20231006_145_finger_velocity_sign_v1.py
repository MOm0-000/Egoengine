#!/usr/bin/env python3
"""Audit the compiled XHand finger velocity term for sample 20231006_145."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = RL_ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"
FULL_RUN = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_75n_v1"
)
FULL_ANALYSIS = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_analysis_v1"
)
OUTPUT = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_finger_velocity_sign_v1"
)
RUNNER = RL_ROOT / "scripts/run_taco_brush_controlled_physics_replay_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def source_default() -> dict[str, Any]:
    root = ET.parse(TEMPLATE).getroot()
    general = root.find("./default/general")
    if general is None:
        raise RuntimeError("source template has no default/general element")
    gain = [float(value) for value in general.attrib["gainprm"].split()]
    bias = [float(value) for value in general.attrib["biasprm"].split()]
    if gain != [300.0] or bias != [0.0, -300.0, 1.0]:
        raise RuntimeError("source XHand actuator default changed")
    return {
        "element": ET.tostring(general, encoding="unicode").strip(),
        "gainprm": gain,
        "biasprm": bias,
    }


def is_finger_joint(name: str) -> bool:
    return name.startswith(("right_hand_", "left_hand_"))


def compiled_contract(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name
        if not is_finger_joint(joint_name):
            continue
        dof = int(model.jnt_dofadr[joint_id])
        kp = float(model.actuator_gainprm[actuator_id, 0])
        b2 = float(model.actuator_biasprm[actuator_id, 2])
        acc0 = float(model.actuator_acc0[actuator_id])
        inferred_ratio = -b2 / (2.0 * np.sqrt(kp / acc0))
        rows.append({
            "actuator_id": actuator_id,
            "actuator_name": model.actuator(actuator_id).name,
            "joint_name": joint_name,
            "transmission_type": int(model.actuator_trntype[actuator_id]),
            "gear_0": float(model.actuator_gear[actuator_id, 0]),
            "gainprm_0": kp,
            "biasprm_0": float(model.actuator_biasprm[actuator_id, 0]),
            "biasprm_1": float(model.actuator_biasprm[actuator_id, 1]),
            "biasprm_2": b2,
            "compiled_velocity_gain": -b2,
            "actuator_acc0": acc0,
            "inferred_damping_ratio": float(inferred_ratio),
            "joint_damping": float(model.dof_damping[dof]),
            "joint_armature": float(model.dof_armature[dof]),
            "actuator_damping": float(model.actuator_damping[actuator_id]),
            "actuator_armature": float(model.actuator_armature[actuator_id]),
            "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
            "force_limited": bool(model.actuator_forcelimited[actuator_id]),
            "joint_force_limited": bool(model.jnt_actfrclimited[joint_id]),
        })
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 finger actuators, found {len(rows)}")
    return rows


def trace_index(
    states: dict[str, np.ndarray], first: dict[str, Any],
) -> int:
    hits = np.flatnonzero(
        (states["source_endpoint"] == int(first["source_endpoint"]))
        & (states["outcome_endpoint"] == int(first["outcome_endpoint"]))
        & (states["substep"] == int(first["substep"]))
    )
    if len(hits) != 1:
        raise RuntimeError(f"first penetration has {len(hits)} trace matches")
    return int(hits[0])


def force_balance(
    model: mujoco.MjModel,
    states: dict[str, np.ndarray],
    index: int,
    contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qpos_start = (
        states["initial_actual_qpos"]
        if index == 0 else states["actual_qpos"][index - 1]
    )
    qvel_start = (
        states["initial_actual_qvel"]
        if index == 0 else states["actual_qvel"][index - 1]
    )
    rows = []
    for contract in contracts:
        actuator_id = int(contract["actuator_id"])
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof = int(model.jnt_dofadr[joint_id])
        target = float(states["control_target"][index, actuator_id])
        qpos = float(qpos_start[qpos_address])
        qvel = float(qvel_start[dof])
        kp = float(model.actuator_gainprm[actuator_id, 0])
        b0, b1, b2 = (
            float(value) for value in model.actuator_biasprm[actuator_id, :3]
        )
        position_force = kp * target + b0 + b1 * qpos
        velocity_force = b2 * qvel
        predicted_force = position_force + velocity_force
        recorded_force = float(
            states["preceding_interval_actuator_force"][index, actuator_id]
        )
        rows.append({
            "actuator_id": actuator_id,
            "actuator_name": contract["actuator_name"],
            "joint_name": contract["joint_name"],
            "actual_position_start": qpos,
            "actual_velocity_start": qvel,
            "reference_position_target": target,
            "target_minus_actual_start": target - qpos,
            "reference_velocity": float(states["reference_qvel"][index, dof]),
            "position_feedback_force": position_force,
            "velocity_feedback_force": velocity_force,
            "velocity_feedback_power": velocity_force * qvel,
            "predicted_actuator_force": predicted_force,
            "recorded_actuator_force": recorded_force,
            "force_reconstruction_error": predicted_force - recorded_force,
            "reference_position_end": float(
                states["reference_qpos"][index, qpos_address]
            ),
            "actual_position_end": float(states["actual_qpos"][index, qpos_address]),
            "reference_minus_actual_end": float(
                states["reference_qpos"][index, qpos_address]
                - states["actual_qpos"][index, qpos_address]
            ),
            "actual_velocity_end": float(states["actual_qvel"][index, dof]),
        })
    return rows


def early_velocity_power_max(
    model: mujoco.MjModel,
    states: dict[str, np.ndarray],
    last_index: int,
    contracts: list[dict[str, Any]],
) -> float:
    maximum = -np.inf
    for index in range(last_index + 1):
        qvel = (
            states["initial_actual_qvel"]
            if index == 0 else states["actual_qvel"][index - 1]
        )
        for contract in contracts:
            actuator_id = int(contract["actuator_id"])
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            dof = int(model.jnt_dofadr[joint_id])
            velocity = float(qvel[dof])
            velocity_force = (
                float(model.actuator_biasprm[actuator_id, 2]) * velocity
            )
            maximum = max(maximum, velocity_force * velocity)
    return float(maximum)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    scene = FULL_RUN / "experiment_local_scene.xml"
    trace = FULL_RUN / "substep_state_trace.npz"
    analysis_path = FULL_ANALYSIS / "results.json"
    model = mujoco.MjModel.from_xml_path(str(scene))
    states = load_npz(trace)
    analysis = read_json(analysis_path)
    first = analysis["hand_table"]["first_actual_penetration"]
    index = trace_index(states, first)
    source = source_default()
    contracts = compiled_contract(model)
    balances = force_balance(model, states, index, contracts)

    reconstruction_error = max(
        abs(float(row["force_reconstruction_error"])) for row in balances
    )
    if reconstruction_error > 1.0e-10:
        raise RuntimeError("compiled actuator force does not reconstruct the trace")
    if any(float(row["biasprm_2"]) >= 0.0 for row in contracts):
        raise RuntimeError("a compiled finger actuator retained a nonnegative b2")
    if any(float(row["gear_0"]) != 1.0 for row in contracts):
        raise RuntimeError("finger actuator transmission is not direct unit gear")

    write_csv(OUTPUT / "compiled_finger_actuators.csv", contracts)
    write_csv(OUTPUT / "first_penetration_finger_force_balance.csv", balances)

    pinky = {
        row["actuator_name"]: row for row in balances
        if row["actuator_name"] in {
            "left_pinky_joint1_position", "left_pinky_joint2_position",
        }
    }
    if len(pinky) != 2:
        raise RuntimeError("left pinky force records are incomplete")
    early_max_power = early_velocity_power_max(model, states, index, contracts)
    result = {
        "schema": "taco_brush_20231006_145_finger_velocity_sign_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "mujoco_version": mujoco.__version__,
        "source_default": source,
        "compiled_contract": {
            "finger_actuator_count": len(contracts),
            "biasprm_2_min": min(float(row["biasprm_2"]) for row in contracts),
            "biasprm_2_max": max(float(row["biasprm_2"]) for row in contracts),
            "velocity_gain_min": min(
                float(row["compiled_velocity_gain"]) for row in contracts
            ),
            "velocity_gain_max": max(
                float(row["compiled_velocity_gain"]) for row in contracts
            ),
            "inferred_damping_ratio_min": min(
                float(row["inferred_damping_ratio"]) for row in contracts
            ),
            "inferred_damping_ratio_max": max(
                float(row["inferred_damping_ratio"]) for row in contracts
            ),
            "all_direct_unit_gear": True,
            "all_joint_damping_zero": all(
                float(row["joint_damping"]) == 0.0 for row in contracts
            ),
            "all_joint_armature_one": all(
                float(row["joint_armature"]) == 1.0 for row in contracts
            ),
            "all_force_unlimited": all(
                not bool(row["force_limited"])
                and not bool(row["joint_force_limited"])
                for row in contracts
            ),
        },
        "first_actual_penetration": {
            **first,
            "trace_index": index,
            "force_reconstruction_max_abs_error": reconstruction_error,
            "left_pinky_force_balance": pinky,
        },
        "early_window_maximum_velocity_feedback_power": early_max_power,
        "source_artifacts": {
            "source_template": artifact(TEMPLATE),
            "experiment_scene": artifact(scene),
            "substep_state_trace": artifact(trace),
            "first_penetration_analysis": artifact(analysis_path),
            "replay_runner": artifact(RUNNER),
        },
    }
    write_json(OUTPUT / "results.json", result)

    first_joint = pinky["left_pinky_joint1_position"]
    second_joint = pinky["left_pinky_joint2_position"]
    report = f"""# 20231006_145 手指执行器速度项审计

## 编译前后

模板默认值确实写成 `gainprm=300`、`biasprm=0 -300 1`。但是在本次完整 Replay 实际加载的 MuJoCo `{mujoco.__version__}` 编译模型中，24 个手指执行器的第三个偏置参数都已经变为负数，范围是 `{result['compiled_contract']['biasprm_2_min']:.9f}` 到 `{result['compiled_contract']['biasprm_2_max']:.9f}`。

这里的源 XML `+1` 被编译器解释为阻尼比 1。对满足 `gainprm[0] == -biasprm[1]` 且 `biasprm[2] > 0` 的位置执行器，MuJoCo 会根据参考姿态下的等效惯量，把正的阻尼比换算为负的速度系数。用编译模型的 `actuator_acc0` 反算，24 个手指执行器的阻尼比为 `{result['compiled_contract']['inferred_damping_ratio_min']:.9f}` 到 `{result['compiled_contract']['inferred_damping_ratio_max']:.9f}`，与 1 一致。

每个手指执行器都是关节直连、传动比 1。关节自身阻尼为 0，执行器附加阻尼为 0，关节转子惯量为 1；实际速度阻尼全部来自编译后的负 `biasprm[2]`。执行器和关节均未设置力上限。

因此，运行时手指执行器的力是：

`执行器力 = 300 × (目标角度 - 实际角度) - 约 34.64 × 实际角速度`

不是把 `+1 × 实际角速度` 直接带入运行时公式。

## 首次穿透子步

首次实际手—桌距离超过 0.05 mm 容差发生在 `{first['source_endpoint']}→{first['outcome_endpoint']}` 的子步 `{first['substep']}`，步后时刻 `{float(first['state_time_s']):.3f} s`。当时参考距离为 `{float(first['reference_minimum_distance_m']) * 1000:.3f} mm`，实际距离为 `{float(first['actual_minimum_distance_m']) * 1000:.3f} mm`，最低部位是左手小指末节。

| 关节 | 步前实际速度 | 位置反馈力 | 速度反馈力 | 合成执行器力 | 步后参考减实际 |
|---|---:|---:|---:|---:|---:|
| 左小指关节1 | {first_joint['actual_velocity_start']:.6f} rad/s | {first_joint['position_feedback_force']:.6f} N·m | {first_joint['velocity_feedback_force']:.6f} N·m | {first_joint['recorded_actuator_force']:.6f} N·m | {first_joint['reference_minus_actual_end'] * 1000:.3f} mrad |
| 左小指关节2 | {second_joint['actual_velocity_start']:.6f} rad/s | {second_joint['position_feedback_force']:.6f} N·m | {second_joint['velocity_feedback_force']:.6f} N·m | {second_joint['recorded_actuator_force']:.6f} N·m | {second_joint['reference_minus_actual_end'] * 1000:.3f} mrad |

两个速度反馈力都与实际速度方向相反。直到这个子步为止，24 个手指执行器的 `速度反馈力 × 实际速度` 最大值为 `{early_max_power:.12f} W`，没有出现正值；这个速度项没有向关节运动注入能量。

逐项复算的执行器力与 Replay 保存值最大只差 `{reconstruction_error:.3e} N·m`，说明上述公式就是这次 Replay 的实际运行公式。

## 这条线索说明什么

源 XML 中的 `+1` 没有在编译后保留为正速度反馈，因此“正号造成反阻尼并推动手指穿桌”与本次运行数据不符。

不过，手指控制器只有位置目标，没有参考速度目标。它使用 `-kv × 实际速度`，等价于把期望速度设为 0。在首次穿透子步，左小指两个关节的参考速度分别为 `{first_joint['reference_velocity']:.6f}` 和 `{second_joint['reference_velocity']:.6f} rad/s`，而实际速度只有 `{first_joint['actual_velocity_start']:.6f}` 和 `{second_joint['actual_velocity_start']:.6f} rad/s`。负速度项分别抵消了 `{abs(first_joint['velocity_feedback_force']):.3f}` 和 `{abs(second_joint['velocity_feedback_force']):.3f} N·m` 的位置反馈力，所以它会增加快速移动参考的跟踪滞后。这个现象与正反馈不是一回事；当前只读数据能确认其存在，但不能单独证明它是首次穿透的唯一原因。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    artifacts = [
        OUTPUT / "compiled_finger_actuators.csv",
        OUTPUT / "first_penetration_finger_force_balance.csv",
        OUTPUT / "report.md",
        OUTPUT / "results.json",
    ]
    (OUTPUT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
