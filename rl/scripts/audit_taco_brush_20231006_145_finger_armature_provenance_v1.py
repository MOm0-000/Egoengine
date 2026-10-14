#!/usr/bin/env python3
"""Audit XHand finger armature provenance and its effect in sample 20231006_145."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = RL_ROOT / "models/taco_xhand/templates/xhand_bimanual_source.xml"
LEFT_MODEL = RL_ROOT / "models/taco_xhand/assets/robots/xhand/left.xml"
RIGHT_MODEL = RL_ROOT / "models/taco_xhand/assets/robots/xhand/right.xml"
URDFS = {
    "left": RL_ROOT / "models/taco_xhand/assets/robots/xhand/xhand_left.urdf",
    "right": RL_ROOT / "models/taco_xhand/assets/robots/xhand/xhand_right.urdf",
}
FULL_RUN = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_75n_v1"
)
FULL_ANALYSIS = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_analysis_v1"
)
REFERENCE = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_safe_entry_mink_v1/safe_entry/robot_reference.npz"
)
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_finger_armature_provenance_v1"

PINKY_JOINTS = (
    "left_hand_pinky_joint1",
    "left_hand_pinky_joint2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


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


def is_finger_joint(name: str) -> bool:
    return name.startswith(("left_hand_", "right_hand_"))


def urdf_joint_contracts() -> dict[str, dict[str, float]]:
    contracts: dict[str, dict[str, float]] = {}
    for path in URDFS.values():
        root = ET.parse(path).getroot()
        if root.findall(".//transmission"):
            raise RuntimeError(f"unexpected transmission element in {path}")
        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            if not is_finger_joint(name) or joint.attrib.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            dynamics = joint.find("dynamics")
            if limit is None or dynamics is None:
                raise RuntimeError(f"incomplete URDF contract for {name}")
            contracts[name] = {
                "effort_limit": float(limit.attrib["effort"]),
                "velocity_limit": float(limit.attrib["velocity"]),
                "damping": float(dynamics.attrib["damping"]),
                "friction": float(dynamics.attrib["friction"]),
            }
    if len(contracts) != 24:
        raise RuntimeError(f"expected 24 URDF finger joints, found {len(contracts)}")
    return contracts


def urdf_transmission_evidence() -> dict[str, int]:
    text = "\n".join(path.read_text(encoding="utf-8") for path in URDFS.values())
    lowered = text.lower()
    return {
        "transmission_element_count": lowered.count("<transmission"),
        "mechanical_reduction_field_count": lowered.count("mechanicalreduction"),
        "rotor_inertia_field_count": lowered.count("rotor"),
    }


def full_mass_matrix(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> np.ndarray:
    matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, matrix, data.qM)
    else:
        mujoco.mj_fullM(model, data, matrix)
    return matrix


def effective_inertia(matrix: np.ndarray, dof: int) -> float:
    return float(1.0 / np.linalg.inv(matrix)[dof, dof])


def critical_response_time(natural_frequency: float, fraction: float) -> float:
    """Time for 1-(1+x)exp(-x) to reach a requested response fraction."""
    lower, upper = 0.0, 20.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        response = 1.0 - (1.0 + middle) * np.exp(-middle)
        if response < fraction:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper) / natural_frequency)


def actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> int:
    matches = [
        actuator_id for actuator_id in range(model.nu)
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"joint {joint_id} has {len(matches)} actuators")
    return matches[0]


def trace_index(
    states: dict[str, np.ndarray], source: int, outcome: int, substep: int,
) -> int:
    hits = np.flatnonzero(
        (states["source_endpoint"] == source)
        & (states["outcome_endpoint"] == outcome)
        & (states["substep"] == substep)
    )
    if len(hits) != 1:
        raise RuntimeError(
            f"trace key {source}->{outcome} substep {substep} has {len(hits)} rows"
        )
    return int(hits[0])


def endpoint_index(states: dict[str, np.ndarray], endpoint: int) -> int | None:
    if endpoint == 0:
        return None
    hits = np.flatnonzero(
        (states["outcome_endpoint"] == endpoint)
        & (states["substep"] == states["substeps"] - 1)
    )
    if len(hits) != 1:
        raise RuntimeError(f"endpoint {endpoint} has {len(hits)} terminal rows")
    return int(hits[0])


def source_default_evidence() -> dict[str, Any]:
    template_root = ET.parse(TEMPLATE).getroot()
    default_joint = template_root.find("./default/joint")
    if default_joint is None or float(default_joint.attrib["armature"]) != 1.0:
        raise RuntimeError("template armature default changed")
    model_text = LEFT_MODEL.read_text(encoding="utf-8")
    response_comment = "with this setting, the response time is around 1s"
    if response_comment not in model_text:
        raise RuntimeError("source response-time comment changed")
    introduced = subprocess.check_output(
        [
            "git", "log", "--follow", "--diff-filter=A", "--format=%H",
            "--", str(TEMPLATE.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip().splitlines()
    if len(introduced) != 1:
        raise RuntimeError("template introduction commit is ambiguous")
    return {
        "template_default_armature": float(default_joint.attrib["armature"]),
        "source_response_time_comment": response_comment,
        "template_introduction_commit": introduced[0],
        "template_has_later_armature_history": bool(subprocess.check_output(
            [
                "git", "log", "--follow", "--format=%H", "-S", 'armature="1"',
                "--", str(TEMPLATE.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            text=True,
        ).strip().splitlines()[1:]),
    }


def compiled_joint_rows(
    model: mujoco.MjModel,
    matrix: np.ndarray,
    body_matrix: np.ndarray,
    urdf: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if not is_finger_joint(joint_name):
            continue
        dof = int(model.jnt_dofadr[joint_id])
        actuator_id = actuator_for_joint(model, joint_id)
        total = effective_inertia(matrix, dof)
        body_only = effective_inertia(body_matrix, dof)
        kp = float(model.actuator_gainprm[actuator_id, 0])
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        natural_frequency = float(np.sqrt(kp / total))
        damping_ratio = float(kv / (2.0 * np.sqrt(kp * total)))
        source = urdf[joint_name]
        rows.append({
            "actuator_id": actuator_id,
            "actuator_name": model.actuator(actuator_id).name,
            "joint_name": joint_name,
            "dof_address": dof,
            "joint_armature": float(model.dof_armature[dof]),
            "mass_matrix_diagonal": float(matrix[dof, dof]),
            "body_only_mass_matrix_diagonal": float(body_matrix[dof, dof]),
            "effective_inertia_with_armature": total,
            "effective_inertia_without_all_armatures": body_only,
            "effective_inertia_ratio": total / body_only,
            "armature_effective_fraction": 1.0 - body_only / total,
            "position_gain": kp,
            "compiled_velocity_gain": kv,
            "local_natural_frequency_rad_s": natural_frequency,
            "inverse_natural_frequency_s": 1.0 / natural_frequency,
            "local_damping_ratio": damping_ratio,
            "model_joint_damping": float(model.dof_damping[dof]),
            "model_joint_frictionloss": float(model.dof_frictionloss[dof]),
            "actuator_force_limited": bool(model.actuator_forcelimited[actuator_id]),
            "joint_force_limited": bool(model.jnt_actfrclimited[joint_id]),
            "urdf_effort_limit": source["effort_limit"],
            "urdf_velocity_limit": source["velocity_limit"],
            "urdf_joint_damping": source["damping"],
            "urdf_joint_friction": source["friction"],
        })
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 compiled finger rows, found {len(rows)}")
    return rows


def endpoint_rows(
    model: mujoco.MjModel,
    states: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in range(9):
        index = endpoint_index(states, endpoint)
        actual_qpos = (
            states["initial_actual_qpos"] if index is None
            else states["actual_qpos"][index]
        )
        actual_qvel = (
            states["initial_actual_qvel"] if index is None
            else states["actual_qvel"][index]
        )
        interval_index = (
            trace_index(states, endpoint, endpoint + 1, 0)
            if endpoint < 8 else None
        )
        for joint_name in PINKY_JOINTS:
            joint_id = model.joint(joint_name).id
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof = int(model.jnt_dofadr[joint_id])
            rows.append({
                "endpoint": endpoint,
                "joint_name": joint_name,
                "reference_position": float(reference["qpos"][endpoint, qpos_address]),
                "actual_position": float(actual_qpos[qpos_address]),
                "reference_minus_actual_position": float(
                    reference["qpos"][endpoint, qpos_address]
                    - actual_qpos[qpos_address]
                ),
                "actual_velocity": float(actual_qvel[dof]),
                "next_interval_reference_velocity": (
                    "" if interval_index is None else float(
                        states["reference_qvel"][interval_index, dof]
                    )
                ),
                "next_interval_duration_s": (
                    "" if interval_index is None else float(
                        states["substeps"][interval_index] * model.opt.timestep
                    )
                ),
            })
    return rows


def first_penetration_rows(
    model: mujoco.MjModel,
    states: dict[str, np.ndarray],
    index: int,
    contracts: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    qpos_start = states["actual_qpos"][index - 1]
    qvel_start = states["actual_qvel"][index - 1]
    rows: list[dict[str, Any]] = []
    for joint_name in PINKY_JOINTS:
        joint_id = model.joint(joint_name).id
        actuator_id = actuator_for_joint(model, joint_id)
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof = int(model.jnt_dofadr[joint_id])
        target = float(states["control_target"][index, actuator_id])
        actual_position = float(qpos_start[qpos_address])
        actual_velocity = float(qvel_start[dof])
        kp = float(model.actuator_gainprm[actuator_id, 0])
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        position_force = kp * (target - actual_position)
        velocity_force = -kv * actual_velocity
        recorded_force = float(
            states["preceding_interval_actuator_force"][index, actuator_id]
        )
        effort_limit = contracts[joint_name]["effort_limit"]
        rows.append({
            "joint_name": joint_name,
            "actuator_name": model.actuator(actuator_id).name,
            "reference_velocity": float(states["reference_qvel"][index, dof]),
            "actual_velocity_start": actual_velocity,
            "actual_to_reference_speed_ratio": abs(
                actual_velocity / states["reference_qvel"][index, dof]
            ),
            "actual_velocity_end": float(states["actual_qvel"][index, dof]),
            "reference_position_target": target,
            "actual_position_start": actual_position,
            "target_minus_actual_start": target - actual_position,
            "position_feedback_force": position_force,
            "velocity_feedback_force": velocity_force,
            "predicted_actuator_force": position_force + velocity_force,
            "recorded_actuator_force": recorded_force,
            "force_reconstruction_error": position_force + velocity_force - recorded_force,
            "urdf_effort_limit": effort_limit,
            "recorded_force_to_urdf_effort_ratio": abs(recorded_force) / effort_limit,
            "reference_position_end": float(
                states["reference_qpos"][index, qpos_address]
            ),
            "actual_position_end": float(states["actual_qpos"][index, qpos_address]),
            "reference_minus_actual_end": float(
                states["reference_qpos"][index, qpos_address]
                - states["actual_qpos"][index, qpos_address]
            ),
        })
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    scene = FULL_RUN / "experiment_local_scene.xml"
    trace_path = FULL_RUN / "substep_state_trace.npz"
    manifest_path = FULL_RUN / "input_manifest.json"
    analysis_path = FULL_ANALYSIS / "results.json"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    states = load_npz(trace_path)
    reference = load_npz(REFERENCE)
    manifest = read_json(manifest_path)
    analysis = read_json(analysis_path)
    first = analysis["hand_table"]["first_actual_penetration"]
    index = trace_index(
        states,
        int(first["source_endpoint"]),
        int(first["outcome_endpoint"]),
        int(first["substep"]),
    )
    if index == 0:
        raise RuntimeError("first penetration unexpectedly occurs at first substep")

    data.qpos[:] = states["actual_qpos"][index - 1]
    data.qvel[:] = states["actual_qvel"][index - 1]
    mujoco.mj_forward(model, data)
    matrix = full_mass_matrix(model, data)
    body_matrix = matrix.copy()
    body_matrix[np.diag_indices(model.nv)] -= model.dof_armature
    if np.min(np.linalg.eigvalsh(body_matrix)) <= 0.0:
        raise RuntimeError("body-only mass matrix is not positive definite")

    provenance = source_default_evidence()
    urdf = urdf_joint_contracts()
    transmission_evidence = urdf_transmission_evidence()
    if any(transmission_evidence.values()):
        raise RuntimeError("URDF unexpectedly contains motor transmission evidence")
    joint_rows = compiled_joint_rows(model, matrix, body_matrix, urdf)
    timeline_rows = endpoint_rows(model, states, reference)
    penetration_rows = first_penetration_rows(model, states, index, urdf)
    write_csv(OUTPUT / "finger_dynamics_contract.csv", joint_rows)
    write_csv(OUTPUT / "left_pinky_endpoint_0_8.csv", timeline_rows)
    write_csv(OUTPUT / "first_penetration_left_pinky.csv", penetration_rows)

    pinky_contract = {row["joint_name"]: row for row in joint_rows if row["joint_name"] in PINKY_JOINTS}
    pinky_force = {row["joint_name"]: row for row in penetration_rows}
    interval_start = min(
        float(states["force_interval_start_time_s"][trace_index(states, 5, 6, 0)]),
        float(states["force_interval_start_time_s"][index]),
    )
    elapsed = float(states["force_interval_start_time_s"][index]) - interval_start
    ramp_predictions: dict[str, Any] = {}
    for joint_name in PINKY_JOINTS:
        natural_frequency = pinky_contract[joint_name]["local_natural_frequency_rad_s"]
        ratio = 1.0 - (1.0 + natural_frequency * elapsed) * np.exp(
            -natural_frequency * elapsed
        )
        ramp_predictions[joint_name] = {
            "elapsed_since_interval_5_start_s": elapsed,
            "ideal_critical_second_order_speed_ratio": float(ratio),
            "observed_pre_step_speed_ratio": pinky_force[joint_name][
                "actual_to_reference_speed_ratio"
            ],
            "ideal_speed_ratio_after_one_control_period": float(
                1.0
                - (1.0 + natural_frequency / manifest["physics"]["control_hz"])
                * np.exp(-natural_frequency / manifest["physics"]["control_hz"])
            ),
            "ideal_time_to_95_percent_response_s": critical_response_time(
                natural_frequency, 0.95,
            ),
        }

    robot_joint_rows = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        dof = int(model.jnt_dofadr[joint_id])
        robot_joint_rows.append({
            "joint_name": model.joint(joint_id).name,
            "joint_type": int(model.jnt_type[joint_id]),
            "armature": float(model.dof_armature[dof]),
        })
    all_robot_armature_one = all(row["armature"] == 1.0 for row in robot_joint_rows)
    robot_joint_types = sorted({row["joint_type"] for row in robot_joint_rows})
    if not all_robot_armature_one or len(robot_joint_types) < 2:
        raise RuntimeError("expected one blanket armature across slide and hinge joints")

    maximum_reconstruction_error = max(
        abs(float(row["force_reconstruction_error"])) for row in penetration_rows
    )
    if maximum_reconstruction_error > 1.0e-10:
        raise RuntimeError("finger actuator force reconstruction failed")
    if manifest["controller"]["controlled_actuators"][0]["joint_name"] not in {
        "R_forearm_tz_link_joint", "L_forearm_tz_link_joint",
    }:
        raise RuntimeError("full replay controller contract changed")
    controlled_names = {
        item["joint_name"] for item in manifest["controller"]["controlled_actuators"]
    }
    if controlled_names != {"R_forearm_tz_link_joint", "L_forearm_tz_link_joint"}:
        raise RuntimeError("reference velocity feedback is no longer wrist-only")

    result = {
        "schema": "taco_brush_20231006_145_finger_armature_provenance_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "runtime_mujoco_version": mujoco.__version__,
        "recorded_replay_mujoco_version": manifest["physics"]["mujoco_version"],
        "source_provenance": {
            **provenance,
            "robot_actuated_joint_count": len(robot_joint_rows),
            "same_armature_one_on_all_actuated_joints": all_robot_armature_one,
            "joint_type_ids_receiving_same_value": robot_joint_types,
            "urdf_transmission_evidence": transmission_evidence,
            "hardware_basis_status": "NO_VERIFIABLE_ROTOR_INERTIA_OR_GEAR_RATIO",
            "source_intent_evidence": "GENERIC_RESPONSE_TIME_SETTING",
            "stability_motive_status": "NOT_DOCUMENTED_IN_XHAND_SOURCE",
        },
        "current_replay_control_scope": {
            "position_target_timing": manifest["physics"]["command_timing"],
            "physics_timestep_s": manifest["physics"]["authored_timestep_s"],
            "reference_frequency_hz": manifest["physics"]["control_hz"],
            "gravity_and_reference_velocity_controlled_joints": sorted(controlled_names),
            "finger_reference_velocity_feedforward": False,
        },
        "first_actual_penetration": {
            **first,
            "trace_index": index,
            "left_pinky_force": pinky_force,
            "force_reconstruction_maximum_abs_error": maximum_reconstruction_error,
        },
        "first_proxy_hand_floor_contact": analysis["hand_table"][
            "first_proxy_hand_floor_contact"
        ],
        "left_pinky_inertia": pinky_contract,
        "ramp_response_check": ramp_predictions,
        "source_artifacts": {
            "template": artifact(TEMPLATE),
            "left_model": artifact(LEFT_MODEL),
            "right_model": artifact(RIGHT_MODEL),
            "left_urdf": artifact(URDFS["left"]),
            "right_urdf": artifact(URDFS["right"]),
            "experiment_scene": artifact(scene),
            "replay_manifest": artifact(manifest_path),
            "substep_state_trace": artifact(trace_path),
            "penetration_analysis": artifact(analysis_path),
            "reference": artifact(REFERENCE),
        },
        "external_primary_sources": {
            "upstream_spider_left_model": (
                "https://github.com/facebookresearch/spider/blob/main/"
                "spider/assets/robots/xhand/left.xml"
            ),
            "upstream_spider_left_urdf": (
                "https://github.com/facebookresearch/spider/blob/main/"
                "spider/assets/robots/xhand/xhand_left.urdf"
            ),
            "mujoco_xml_reference": (
                "https://mujoco.readthedocs.io/en/latest/XMLreference.html"
            ),
            "xhand_product_manual": (
                "https://cdn.robotshop.com/media/E/ERA/RB-Era-01/pdf/"
                "x-hand-product-manual-1-0.pdf"
            ),
        },
    }
    write_json(OUTPUT / "results.json", result)

    first_joint = pinky_contract[PINKY_JOINTS[0]]
    second_joint = pinky_contract[PINKY_JOINTS[1]]
    first_force = pinky_force[PINKY_JOINTS[0]]
    second_force = pinky_force[PINKY_JOINTS[1]]
    first_prediction = ramp_predictions[PINKY_JOINTS[0]]
    second_prediction = ramp_predictions[PINKY_JOINTS[1]]
    report = f"""# 20231006_145 手指转子惯量来源与动态影响审计

## 先说结论边界

目前找不到 `armature=1` 的 XHand 硬件标定依据。仓库和上游 SPIDER 的 XHand URDF 都没有电机转子惯量、减速比或传动元素，公开产品手册也没有给出这两个参数。因此无法用 `电机转子惯量 × 减速比²` 复算出 1。

相反，仓库随附的左右手 MuJoCo 模型把 `kp=300`、`dampratio=1`、`armature=1` 放在同一个默认块中，并明确注释“使用这个设置时，响应时间约 1 秒”。`armature=1` 还被一刀切地赋给全部 {len(robot_joint_rows)} 个受控机器人自由度，既包括单位为 kg 的平移关节，也包括单位为 kg·m² 的转动关节。现有证据因此更符合“为了统一仿真响应而设置的通用经验参数”，不符合“由每个 XHand 电机和减速器参数逐关节折算得到”。源码没有说明它是否还兼顾数值稳定性，所以这一点仍不能确定。

## 当前 Replay 中它有多大影响

在首次实际穿桌前一个状态，用完整 MuJoCo 质量矩阵计算：

| 关节 | 含 `armature=1` 的等效惯量 | 去掉全部 armature 后的机械结构等效惯量 | 放大倍数 | `1/ωn` |
|---|---:|---:|---:|---:|
| 左小指关节1 | {first_joint['effective_inertia_with_armature']:.9f} kg·m² | {first_joint['effective_inertia_without_all_armatures']:.9f} kg·m² | {first_joint['effective_inertia_ratio']:.1f} 倍 | {first_joint['inverse_natural_frequency_s'] * 1000:.3f} ms |
| 左小指关节2 | {second_joint['effective_inertia_with_armature']:.9f} kg·m² | {second_joint['effective_inertia_without_all_armatures']:.9f} kg·m² | {second_joint['effective_inertia_ratio']:.1f} 倍 | {second_joint['inverse_natural_frequency_s'] * 1000:.3f} ms |

因此，`sqrt(300/1) ≈ 17.32 rad/s` 和约 `58 ms` 的估算是正确的，而且不是一个小修正：`armature=1` 分别占这两个关节当前等效惯量的 {first_joint['armature_effective_fraction']:.5%} 和 {second_joint['armature_effective_fraction']:.5%}。但 58 ms 只是 `1/ωn`，不是“已经跟上”的时间。对当前近似临界阻尼系统，经过一个 33.3 ms 控制周期，匀速目标下的关节速度理论上只到参考速度的 {first_prediction['ideal_speed_ratio_after_one_control_period']:.2%}；到 95% 大约需要 {first_prediction['ideal_time_to_95_percent_response_s'] * 1000:.1f} ms。

这里要补一个时间语义：当前 Replay 已经不是每 33.3 ms 跳一次目标。位置目标在每个 2 ms 物理子步上线性插值；但是相邻 endpoint 定义的参考速度会在边界处换挡。左小指在 endpoint 0→5 的参考速度接近 0，到 5→6 突然变为 `{first_force['reference_velocity']:.6f}` 和 `{second_force['reference_velocity']:.6f} rad/s`。

首次实际穿桌发生在 5→6 的子步 12。该力作用区间从 `{float(first['force_interval_start_time_s']):.3f} s` 开始，也就是进入 5→6 后 {elapsed * 1000:.1f} ms。若只用当前惯量、位置刚度和临界阻尼构成最简单的二阶模型，它预测此时关节速度只能达到参考速度的 {first_prediction['ideal_critical_second_order_speed_ratio']:.2%} 和 {second_prediction['ideal_critical_second_order_speed_ratio']:.2%}；Replay 实测为 {first_prediction['observed_pre_step_speed_ratio']:.2%} 和 {second_prediction['observed_pre_step_speed_ratio']:.2%}。两者很接近，说明这次早期慢响应可以直接由当前惯量和伺服参数解释，不需要先假设桌面接触把手指拖慢。MuJoCo 代理手—桌接触要到后面的 6→7 子步 4 才出现。

## 力矩约束也不一致

本次“重力前馈 + 参考速度反馈”只加在左右手腕的竖直平移关节，没有加在手指关节上。手指仍然只有位置目标和对实际速度的阻尼。

| 关节 | URDF 力矩上限 | 首次穿桌子步的实际执行器力矩 | 超出倍数 |
|---|---:|---:|---:|
| 左小指关节1 | {first_force['urdf_effort_limit']:.3f} N·m | {first_force['recorded_actuator_force']:.3f} N·m | {first_force['recorded_force_to_urdf_effort_ratio']:.2f} 倍 |
| 左小指关节2 | {second_force['urdf_effort_limit']:.3f} N·m | {second_force['recorded_actuator_force']:.3f} N·m | {second_force['recorded_force_to_urdf_effort_ratio']:.2f} 倍 |

当前 MuJoCo 模型没有执行器力上限，也没有关节总执行器力上限。因此它一方面用很大的附加惯量把响应压慢，另一方面又允许位置误差积累后产生远超 URDF 约束的力矩。这套动态参数可以用于仿真，但现有证据不足以把它称为经过 XHand 硬件标定的低层控制器。

## 可复核文件

- `finger_dynamics_contract.csv`：24 个手指关节的编译参数、等效惯量及 URDF 约束。
- `left_pinky_endpoint_0_8.csv`：左小指两个关节在 endpoint 0–8 的参考、实际位置和下一段参考速度。
- `first_penetration_left_pinky.csv`：首次穿桌子步的目标、实际状态、力矩分解和 URDF 上限比值。
- `results.json`：全部机器可读结论、输入哈希和来源链接。

外部来源：[上游 SPIDER XHand 模型](https://github.com/facebookresearch/spider/blob/main/spider/assets/robots/xhand/left.xml)、[上游 XHand URDF](https://github.com/facebookresearch/spider/blob/main/spider/assets/robots/xhand/xhand_left.urdf)、[MuJoCo armature 定义](https://mujoco.readthedocs.io/en/latest/XMLreference.html)、[X-Hand 1 产品手册](https://cdn.robotshop.com/media/E/ERA/RB-Era-01/pdf/x-hand-product-manual-1-0.pdf)。

本审计没有修改模型、参考轨迹、控制增益、碰撞参数或历史 run，也没有执行新的物理实验。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    artifacts = [
        OUTPUT / "finger_dynamics_contract.csv",
        OUTPUT / "left_pinky_endpoint_0_8.csv",
        OUTPUT / "first_penetration_left_pinky.csv",
        OUTPUT / "report.md",
        OUTPUT / "results.json",
    ]
    (OUTPUT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
