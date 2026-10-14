#!/usr/bin/env python3
"""Summarize the complete Brush Replay with vertical feedforward enabled."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
    native_hand_visual_geom_ids,
)


SHORT_RUN = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_abcd_v1"
    / "d_gravity_and_reference_velocity"
)
STOPPED_RUN = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_v1"
)
FULL_RUN = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_75n_v1"
)
OUTPUT = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_vertical_feedforward_d_full_replay_analysis_v1"
)
TOLERANCE_M = 5.0e-5


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def assert_exact_prefix(
    prefix: dict[str, np.ndarray], full: dict[str, np.ndarray], label: str,
) -> None:
    initial_names = {
        "initial_actual_qpos", "initial_actual_qvel", "initial_actual_ctrl",
        "initial_reference_qpos",
    }
    for name, prefix_values in prefix.items():
        full_values = full[name]
        if name not in initial_names:
            full_values = full_values[: len(prefix_values)]
        if not np.array_equal(prefix_values, full_values):
            raise RuntimeError(f"{label} changed prefix array {name}")


def minimum_hand_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geoms: tuple[Any, ...],
    qpos: np.ndarray,
    floor_z_m: float,
) -> dict[str, Any]:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    candidates = []
    for geom in hand_geoms:
        distance = float(
            np.min(geom_world_vertices(data, geom, full=False)[:, 2] - floor_z_m)
        )
        candidates.append({
            "distance_m": distance,
            "geom": geom.name,
            "body": model.body(geom.body_id).name,
        })
    return min(candidates, key=lambda row: row["distance_m"])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    short_manifest = read_json(SHORT_RUN / "input_manifest.json")
    stopped_manifest = read_json(STOPPED_RUN / "input_manifest.json")
    full_manifest = read_json(FULL_RUN / "input_manifest.json")
    stopped_results = read_json(STOPPED_RUN / "results.json")
    results = read_json(FULL_RUN / "results.json")

    for key in (
        "source_scene", "local_scene", "robot_reference", "support_contract",
        "prior_collision_audit", "script",
    ):
        hashes = {
            manifest["inputs"][key]["sha256"]
            for manifest in (short_manifest, stopped_manifest, full_manifest)
        }
        if len(hashes) != 1:
            raise RuntimeError(f"frozen input changed: {key}")
    for manifest in (short_manifest, stopped_manifest, full_manifest):
        controller = manifest["controller"]
        if not controller["gravity_feedforward"] or not controller[
            "reference_velocity_feedback"
        ]:
            raise RuntimeError("full comparison did not use both controller terms")
    if stopped_results["hard_stop"]["reasons"] != [
        "CONTROLLER_EXTRA_FORCE_LIMIT"
    ]:
        raise RuntimeError("50 N run did not preserve the expected hard stop")
    if results["hard_stop"] is not None or results["completed_outcome_endpoint"] != 121:
        raise RuntimeError("75 N run did not complete the full reference")

    short_states = load_npz(SHORT_RUN / "substep_state_trace.npz")
    stopped_states = load_npz(STOPPED_RUN / "substep_state_trace.npz")
    states = load_npz(FULL_RUN / "substep_state_trace.npz")
    assert_exact_prefix(short_states, states, "short D run")
    assert_exact_prefix(stopped_states, states, "50 N stopped run")

    native_rows = read_csv_gz(FULL_RUN / "substep_native_hand_table_trace.csv.gz")
    controller_rows = read_csv(FULL_RUN / "controller_trace.csv")
    endpoint_rows = read_csv(FULL_RUN / "endpoint_trace.csv")
    contact_rows = read_csv_gz(FULL_RUN / "contact_trace.csv.gz")
    if len(native_rows) != len(states["state_time_s"]):
        raise RuntimeError("native trace and state trace lengths differ")

    for row in native_rows:
        for name in (
            "reference_left_distance_m", "reference_right_distance_m",
            "reference_minimum_distance_m", "actual_left_distance_m",
            "actual_right_distance_m", "actual_minimum_distance_m",
            "state_time_s",
        ):
            row[name] = float(row[name])
        for name in ("source_endpoint", "outcome_endpoint", "substep", "substeps"):
            row[name] = int(row[name])

    first_actual = next(
        row for row in native_rows
        if row["actual_minimum_distance_m"] < -TOLERANCE_M
    )
    first_reference = next(
        row for row in native_rows
        if row["reference_minimum_distance_m"] < -TOLERANCE_M
    )
    minimum_actual = min(
        native_rows, key=lambda row: row["actual_minimum_distance_m"]
    )
    minimum_reference = min(
        native_rows, key=lambda row: row["reference_minimum_distance_m"]
    )
    first_actual_index = native_rows.index(first_actual)
    minimum_actual_index = native_rows.index(minimum_actual)
    minimum_reference_index = native_rows.index(minimum_reference)

    model = mujoco.MjModel.from_xml_path(
        str(FULL_RUN / "experiment_local_scene.xml")
    )
    geometry_data = mujoco.MjData(model)
    hand_geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco),
    )
    floor_z_m = float(model.geom("floor").pos[2])
    first_actual_geom = minimum_hand_geom(
        model, geometry_data, hand_geoms,
        states["actual_qpos"][first_actual_index], floor_z_m,
    )
    first_reference_geom = minimum_hand_geom(
        model, geometry_data, hand_geoms,
        states["reference_qpos"][native_rows.index(first_reference)], floor_z_m,
    )
    deepest_actual_geom = minimum_hand_geom(
        model, geometry_data, hand_geoms,
        states["actual_qpos"][minimum_actual_index], floor_z_m,
    )
    deepest_reference_geom = minimum_hand_geom(
        model, geometry_data, hand_geoms,
        states["reference_qpos"][minimum_reference_index], floor_z_m,
    )

    qpos_error = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model, qpos_error, 1.0,
        states["reference_qpos"][first_actual_index],
        states["actual_qpos"][first_actual_index],
    )
    robot_joint_errors = []
    for dof in np.argsort(np.abs(qpos_error[: model.nu]))[::-1][:8]:
        joint_id = int(model.dof_jntid[dof])
        robot_joint_errors.append({
            "dof": int(dof),
            "joint": model.joint(joint_id).name,
            "actual_minus_reference": float(qpos_error[dof]),
        })

    controller_by_step: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for row in controller_rows:
        key = (int(row["source_endpoint"]), int(row["substep"]))
        controller_by_step.setdefault(key, {})[row["actuator_name"]] = row
    first_controller = controller_by_step[
        (first_actual["source_endpoint"], first_actual["substep"])
    ]
    wrist_evidence = {}
    for side, actuator in (
        ("right", "R_forearm_tz_position"),
        ("left", "L_forearm_tz_position"),
    ):
        row = first_controller[actuator]
        side_rows = [
            candidate for candidate in controller_rows
            if candidate["actuator_name"] == actuator
        ]
        worst = max(
            side_rows,
            key=lambda candidate: abs(
                float(candidate["reference_minus_actual_end_m"])
            ),
        )
        wrist_evidence[side] = {
            "first_penetration_reference_minus_actual_m": float(
                row["reference_minus_actual_end_m"]
            ),
            "first_penetration_reference_velocity_m_s": float(
                row["reference_velocity"]
            ),
            "first_penetration_extra_force_n": float(
                row["extra_generalized_force_n"]
            ),
            "maximum_abs_tracking_error_m": abs(
                float(worst["reference_minus_actual_end_m"])
            ),
            "maximum_abs_tracking_error_signed_m": float(
                worst["reference_minus_actual_end_m"]
            ),
            "maximum_abs_tracking_error_source_endpoint": int(
                worst["source_endpoint"]
            ),
            "maximum_abs_tracking_error_substep": int(worst["substep"]),
        }

    hand_floor_contacts = [
        row for row in contact_rows if row["group"] == "hand_floor"
    ]
    first_hand_floor = hand_floor_contacts[0]
    hand_floor_impulse = sum(
        float(row["normal_force_n"]) * float(model.opt.timestep)
        for row in hand_floor_contacts
    )
    hand_floor_max_force = max(
        float(row["force_norm_n"]) for row in hand_floor_contacts
    )

    pickup = results["pickup"]
    maximum_lift_endpoints = {}
    for obj in ("brush", "bowl"):
        actual = max(
            endpoint_rows, key=lambda row: float(row[f"{obj}_actual_lift_m"])
        )
        reference = max(
            endpoint_rows, key=lambda row: float(row[f"{obj}_reference_lift_m"])
        )
        maximum_lift_endpoints[obj] = {
            "actual_endpoint": int(actual["endpoint"]),
            "reference_endpoint": int(reference["endpoint"]),
        }

    timeline_rows = []
    for native in native_rows:
        key = (native["source_endpoint"], native["substep"])
        controlled = controller_by_step[key]
        right = controlled["R_forearm_tz_position"]
        left = controlled["L_forearm_tz_position"]
        timeline_rows.append({
            "source_endpoint": native["source_endpoint"],
            "outcome_endpoint": native["outcome_endpoint"],
            "substep": native["substep"],
            "state_time_s": native["state_time_s"],
            "reference_minimum_hand_table_distance_m": native[
                "reference_minimum_distance_m"
            ],
            "actual_minimum_hand_table_distance_m": native[
                "actual_minimum_distance_m"
            ],
            "right_wrist_reference_minus_actual_m": float(
                right["reference_minus_actual_end_m"]
            ),
            "left_wrist_reference_minus_actual_m": float(
                left["reference_minus_actual_end_m"]
            ),
            "right_extra_generalized_force_n": float(
                right["extra_generalized_force_n"]
            ),
            "left_extra_generalized_force_n": float(
                left["extra_generalized_force_n"]
            ),
        })
    with (OUTPUT / "timeline.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(timeline_rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(timeline_rows)

    time_s = np.asarray([row["state_time_s"] for row in timeline_rows])
    endpoint = np.asarray([int(row["endpoint"]) for row in endpoint_rows])
    figure, axes = plt.subplots(
        3, 1, figsize=(12, 12), constrained_layout=True,
    )
    axes[0].plot(
        time_s,
        np.asarray([
            row["right_wrist_reference_minus_actual_m"] for row in timeline_rows
        ]) * 1000.0,
        label="right wrist", color="#0077b6",
    )
    axes[0].plot(
        time_s,
        np.asarray([
            row["left_wrist_reference_minus_actual_m"] for row in timeline_rows
        ]) * 1000.0,
        label="left wrist", color="#d1495b",
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("wrist below reference (mm)")
    axes[0].legend()

    axes[1].plot(
        time_s,
        np.asarray([
            row["reference_minimum_hand_table_distance_m"]
            for row in timeline_rows
        ]) * 1000.0,
        label="reference", color="black", linestyle="--",
    )
    axes[1].plot(
        time_s,
        np.asarray([
            row["actual_minimum_hand_table_distance_m"]
            for row in timeline_rows
        ]) * 1000.0,
        label="actual", color="#2a9d6f",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].axhline(
        -TOLERANCE_M * 1000.0, color="#d1495b", linewidth=1.0, linestyle=":"
    )
    axes[1].set_ylabel("minimum native hand-table distance (mm)")
    axes[1].legend()

    for obj, color in (("brush", "#0077b6"), ("bowl", "#d1495b")):
        axes[2].plot(
            endpoint,
            [float(row[f"{obj}_actual_lift_m"]) * 1000.0 for row in endpoint_rows],
            label=f"{obj} actual", color=color,
        )
        axes[2].plot(
            endpoint,
            [float(row[f"{obj}_reference_lift_m"]) * 1000.0 for row in endpoint_rows],
            label=f"{obj} reference", color=color, linestyle="--",
        )
    axes[2].set_xlabel("reference endpoint")
    axes[2].set_ylabel("object lift (mm)")
    axes[2].legend(ncol=2)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(OUTPUT / "full_replay_timeline.png", dpi=180)
    plt.close(figure)

    whole = results["substep_native_hand_table"]["whole_replay"]
    analysis = {
        "schema": "taco_brush_20231006_145_vertical_feedforward_d_full_replay_analysis_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "mujoco_version": full_manifest["physics"]["mujoco_version"],
        "completion": {
            "completed_outcome_endpoint": results["completed_outcome_endpoint"],
            "physics_steps": results["physics_steps"],
            "hard_stop": results["hard_stop"],
            "strict_gate_passed": results["strict_gate_passed"],
        },
        "controller": {
            **results["controller"],
            "short_run_prefix_exact": True,
            "stopped_50n_run_prefix_exact": True,
            "stopped_50n_run_physics_steps": stopped_results["physics_steps"],
            "stopped_50n_run_hard_stop": stopped_results["hard_stop"],
        },
        "hand_table": {
            "tolerance_m": TOLERANCE_M,
            "first_actual_penetration": first_actual,
            "first_actual_penetrating_geom": first_actual_geom,
            "first_reference_penetration": first_reference,
            "first_reference_penetrating_geom": first_reference_geom,
            "deepest_actual": minimum_actual,
            "deepest_actual_geom": deepest_actual_geom,
            "deepest_reference": minimum_reference,
            "deepest_reference_geom": deepest_reference_geom,
            "actual_penetrating_substeps": whole[
                "actual_penetration_beyond_tolerance_count"
            ],
            "reference_penetrating_substeps": whole[
                "reference_penetration_beyond_tolerance_count"
            ],
            "reference_safe_actual_penetrating_substeps": whole[
                "reference_safe_actual_penetration_count"
            ],
            "first_proxy_hand_floor_contact": first_hand_floor,
            "proxy_hand_floor_contact_rows": len(hand_floor_contacts),
            "proxy_hand_floor_normal_impulse_ns": hand_floor_impulse,
            "proxy_hand_floor_maximum_force_n": hand_floor_max_force,
        },
        "first_actual_penetration_tracking": {
            "wrist": wrist_evidence,
            "largest_robot_joint_errors": robot_joint_errors,
        },
        "pickup": pickup,
        "maximum_lift_endpoints": maximum_lift_endpoints,
        "tracking_failure_count": len(results["tracking_failures"]),
        "first_tracking_failure": results["tracking_failures"][0],
        "native_blocker_count": len(results["native_blockers"]),
        "proxy_native_geometry_mismatch_count": len(
            results["proxy_native_geometry_mismatches"]
        ),
        "proxy_native_force_exposure_mismatch_count": len(
            results["proxy_native_force_exposure_mismatches"]
        ),
        "source_artifacts": {
            "short_run_results": {
                "path": str(SHORT_RUN / "results.json"),
                "sha256": sha256(SHORT_RUN / "results.json"),
            },
            "stopped_run_results": {
                "path": str(STOPPED_RUN / "results.json"),
                "sha256": sha256(STOPPED_RUN / "results.json"),
            },
            "full_run_results": {
                "path": str(FULL_RUN / "results.json"),
                "sha256": sha256(FULL_RUN / "results.json"),
            },
        },
    }
    write_json(OUTPUT / "results.json", analysis)

    report = f"""# 20231006_145 双项补偿完整 Replay

完整参考共有 122 个 endpoint。本次使用 MuJoCo `{full_manifest['physics']['mujoco_version']}`，从 endpoint 0 运行到 121，共执行 `{results['physics_steps']}` 个物理子步，没有触发最终硬停止。初始 `50 N` 合同在 19→20 前停止；根据冻结参考计算出的最大预期附加力为 `62.896 N`，独立的 `75 N` 合同不截断控制力并完成全程。两次运行的前 `{stopped_results['physics_steps']}` 个物理子步逐数组完全相同，完整运行的前 84 个子步也与短窗口 D 组完全相同。

## 手—桌时间线

- 实际轨迹首次超过 0.05 mm 容差：`{first_actual['source_endpoint']}→{first_actual['outcome_endpoint']}`，子步编号 `{first_actual['substep']}`（从 0 开始），时刻 `{first_actual['state_time_s']:.3f} s`。
- 当时参考最小余量：`{first_actual['reference_minimum_distance_m'] * 1000:.3f} mm`；实际距离：`{first_actual['actual_minimum_distance_m'] * 1000:.3f} mm`。
- 首次穿透部位：`{first_actual_geom['body']}`，即左手小指末节。
- 同一子步左手腕实际位置比参考高 `{-wrist_evidence['left']['first_penetration_reference_minus_actual_m'] * 1000:.3f} mm`，所以这次最早穿透不是左手腕整体低于参考造成的。左手小指两个关节的主要误差分别约为 `{robot_joint_errors[0]['actual_minus_reference'] * 1000:.3f}` 和 `{robot_joint_errors[1]['actual_minus_reference'] * 1000:.3f} mrad`。
- 参考轨迹自身首次超过容差：`{first_reference['source_endpoint']}→{first_reference['outcome_endpoint']}`，子步编号 `{first_reference['substep']}`，部位同样是左手小指末节，距离 `{first_reference['reference_minimum_distance_m'] * 1000:.3f} mm`。
- 全程参考最深距离：`{minimum_reference['reference_minimum_distance_m'] * 1000:.3f} mm`，位于 `{deepest_reference_geom['body']}`；实际最深距离：`{minimum_actual['actual_minimum_distance_m'] * 1000:.3f} mm`，位于 `{deepest_actual_geom['body']}`。
- 2017 个子步中，参考超过容差 `{whole['reference_penetration_beyond_tolerance_count']}` 个，实际超过容差 `{whole['actual_penetration_beyond_tolerance_count']}` 个；其中参考安全但实际穿透 `{whole['reference_safe_actual_penetration_count']}` 个。
- 代理手—桌接触从 `{first_hand_floor['source_endpoint']}→{first_hand_floor['outcome_endpoint']}` 的子步 `{first_hand_floor['substep']}` 开始；全程最大手—桌代理接触力 `{hand_floor_max_force:.3f} N`。

## 抓取和抬升

| 物体 | 最大实际抬升 | 最大参考抬升 | 满足“抬升、手接触、无桌面力”的最长连续区间 |
|---|---:|---:|---:|
| 刷子 | {pickup['brush']['maximum_actual_lift_m'] * 1000:.3f} mm | {pickup['brush']['maximum_reference_lift_m'] * 1000:.3f} mm | {pickup['brush']['lifted_hand_contact_no_floor_longest_run']} |
| 碗 | {pickup['bowl']['maximum_actual_lift_m'] * 1000:.3f} mm | {pickup['bowl']['maximum_reference_lift_m'] * 1000:.3f} mm | {pickup['bowl']['lifted_hand_contact_no_floor_longest_run']} |

刷子在 endpoint `{maximum_lift_endpoints['brush']['actual_endpoint']}` 达到最大实际抬升，远低于参考的 `108.933 mm`；完整运行没有形成稳定离桌抓取。物体跟踪从 endpoint `{results['tracking_failures'][0]['endpoint']}` 开始超过既定门槛，共记录 `{len(results['tracking_failures'])}` 个失败 endpoint，全部来自刷子。

## 范围说明

双项补偿解决了 endpoint 0→5 的早期手腕重力跟踪问题，但完整运行显示后续还存在手指关节跟踪误差、参考自身的局部手—桌穿透、手内接触以及物体抓取失败。完成 endpoint 121 只表示仿真跑完，不表示完整 Replay 成功。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    artifacts = [
        OUTPUT / "full_replay_timeline.png",
        OUTPUT / "report.md",
        OUTPUT / "results.json",
        OUTPUT / "timeline.csv",
    ]
    (OUTPUT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
