#!/usr/bin/env python3
"""Dense native hand/table check for every interval in the 20231006_145 reference."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np


REPO = Path("/data_all/zzx/Egoengine_upload_3_2")
RL_ROOT = REPO / "rl"
OUTPUT = Path(__file__).resolve().parent
SCENE = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_initialization_controlled_replay_v1"
    / "experiment_local_scene.xml"
)
REFERENCE = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_safe_entry_mink_v1"
    / "safe_entry/robot_reference.npz"
)
SAMPLES_PER_INTERVAL = 10_001
WORKERS = 16
ROBOT_QPOS_STOP = 36
ROBOT_DOF_STOP = 36
PLANE_Z_M = 0.72
TOLERANCE_M = 5.0e-5

sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    native_hand_visual_geom_ids,
)


_MODEL: mujoco.MjModel | None = None
_QPOS: np.ndarray | None = None
_GEOMS_BY_SIDE: dict[str, tuple[Any, ...]] | None = None
_ALPHAS: np.ndarray | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def initialize_worker() -> None:
    global _MODEL, _QPOS, _GEOMS_BY_SIDE, _ALPHAS
    _MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
    with np.load(REFERENCE, allow_pickle=False) as archive:
        _QPOS = np.asarray(archive["qpos"], dtype=np.float64).copy()
    geoms = build_native_support_geoms(
        _MODEL, mujoco, native_hand_visual_geom_ids(_MODEL, mujoco)
    )
    _GEOMS_BY_SIDE = {
        side: tuple(geom for geom in geoms if geom.name.startswith(f"{side}_"))
        for side in ("left", "right")
    }
    _ALPHAS = np.linspace(0.0, 1.0, SAMPLES_PER_INTERVAL)


def audit_interval(interval: int) -> tuple[int, np.ndarray]:
    if _MODEL is None or _QPOS is None or _GEOMS_BY_SIDE is None or _ALPHAS is None:
        raise RuntimeError("worker was not initialized")
    model = _MODEL
    q0 = _QPOS[interval]
    q1 = _QPOS[interval + 1]
    tangent = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, tangent, 1.0, q0, q1)
    tangent[ROBOT_DOF_STOP:] = 0.0
    data = mujoco.MjData(model)
    distances = np.full((SAMPLES_PER_INTERVAL, 2), np.inf, dtype=np.float64)

    for sample_index, alpha in enumerate(_ALPHAS):
        data.qpos[:] = q0
        mujoco.mj_integratePos(model, data.qpos, tangent, float(alpha))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(("left", "right")):
            for geom in _GEOMS_BY_SIDE[side]:
                rotation = np.asarray(data.geom_xmat[geom.geom_id]).reshape(3, 3)
                translation_z = float(data.geom_xpos[geom.geom_id, 2])
                distance = float(
                    np.min(geom.support_vertices @ rotation[2] + translation_z - PLANE_Z_M)
                )
                distances[sample_index, side_index] = min(
                    distances[sample_index, side_index], distance
                )
    return interval, distances


def interpolated_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    interval: int,
    alpha: float,
) -> np.ndarray:
    tangent = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model, tangent, 1.0, qpos[interval], qpos[interval + 1]
    )
    tangent[ROBOT_DOF_STOP:] = 0.0
    result = qpos[interval].copy()
    mujoco.mj_integratePos(model, result, tangent, alpha)
    return result


def full_vertex_witness(
    model: mujoco.MjModel,
    qpos_value: np.ndarray,
    side: str,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco)
    )
    best: dict[str, object] | None = None
    for geom in geoms:
        if not geom.name.startswith(f"{side}_"):
            continue
        rotation = np.asarray(data.geom_xmat[geom.geom_id]).reshape(3, 3)
        translation = np.asarray(data.geom_xpos[geom.geom_id])
        world = geom.full_vertices @ rotation.T + translation
        signed = world[:, 2] - PLANE_Z_M
        vertex_index = int(np.argmin(signed))
        row = {
            "body": model.body(geom.body_id).name,
            "geom": geom.name,
            "vertex_index": vertex_index,
            "point_sim_m": world[vertex_index],
            "distance_m": float(signed[vertex_index]),
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    if best is None:
        raise RuntimeError(f"no native hand geometry found for {side}")
    return best


def unique_path_count(mask: np.ndarray) -> int:
    count = int(np.count_nonzero(mask[0]))
    for interval in range(1, len(mask)):
        count += int(np.count_nonzero(mask[interval, 1:]))
    return count


def first_unique_occurrence(mask: np.ndarray) -> tuple[int, int] | None:
    for interval in range(len(mask)):
        start = 0 if interval == 0 else 1
        indices = np.flatnonzero(mask[interval, start:])
        if len(indices):
            return interval, int(indices[0] + start)
    return None


def main() -> int:
    generated = (
        OUTPUT / "dense_samples.npz",
        OUTPUT / "interval_summary.csv",
        OUTPUT / "results.json",
        OUTPUT / "report.md",
    )
    existing = [path for path in generated if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("unexpected controlled-Replay model dimensions")
    if not np.isclose(float(model.opt.timestep), 0.002, rtol=0.0, atol=1.0e-15):
        raise ValueError("unexpected MuJoCo timestep")

    robot_joint_types: list[str] = []
    for joint_id in range(36):
        if int(model.jnt_qposadr[joint_id]) != joint_id:
            raise ValueError("robot qpos addresses are not contiguous 0:36")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            robot_joint_types.append("slide")
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            robot_joint_types.append("hinge")
        else:
            raise ValueError("robot contains an unexpected ball/free joint")

    with np.load(REFERENCE, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64).copy()
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64).copy()
        timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64).copy()
        archived_native = np.asarray(
            archive["native_support_min_distance_m"], dtype=np.float64
        ).copy()
    if qpos.shape != (122, 50):
        raise ValueError("latest formal reference is not the expected 122x50 array")
    interval_count = len(qpos) - 1
    alphas = np.linspace(0.0, 1.0, SAMPLES_PER_INTERVAL)

    interval_results: list[np.ndarray | None] = [None] * interval_count
    context = mp.get_context("fork")
    with context.Pool(processes=WORKERS, initializer=initialize_worker) as pool:
        completed = 0
        for interval, values in pool.imap_unordered(
            audit_interval, range(interval_count), chunksize=1
        ):
            interval_results[interval] = values
            completed += 1
            if completed % 10 == 0 or completed == interval_count:
                print(f"completed intervals: {completed}/{interval_count}", flush=True)
    if any(value is None for value in interval_results):
        raise RuntimeError("one or more intervals were not evaluated")
    distances = np.stack(interval_results, axis=0)
    if not np.isfinite(distances).all():
        raise RuntimeError("nonfinite native distance")

    np.savez_compressed(
        OUTPUT / "dense_samples.npz",
        distances_m=distances,
        side_order=np.asarray(["left", "right"]),
        alpha=alphas,
        source_endpoint=np.arange(interval_count, dtype=np.int64),
        outcome_endpoint=np.arange(1, interval_count + 1, dtype=np.int64),
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        table_z_m=np.asarray(PLANE_Z_M),
        tolerance_m=np.asarray(TOLERANCE_M),
    )

    overall = np.min(distances, axis=2)
    strict_negative = overall < 0.0
    beyond_tolerance = overall < -TOLERANCE_M
    strict_negative_points = unique_path_count(strict_negative)
    beyond_tolerance_points = unique_path_count(beyond_tolerance)
    first_negative = first_unique_occurrence(strict_negative)
    first_beyond_tolerance = first_unique_occurrence(beyond_tolerance)

    flat_index = int(np.argmin(distances))
    worst_interval, worst_sample, worst_side_index = np.unravel_index(
        flat_index, distances.shape
    )
    worst_alpha = float(alphas[worst_sample])
    worst_side = ("left", "right")[worst_side_index]
    worst_qpos = interpolated_qpos(
        model, qpos, int(worst_interval), worst_alpha
    )
    witness = full_vertex_witness(model, worst_qpos, worst_side)
    global_minimum = float(distances[worst_interval, worst_sample, worst_side_index])
    if not np.isclose(
        global_minimum, float(witness["distance_m"]), rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("support-vertex minimum does not match full-vertex witness")

    endpoint_computed = np.empty(len(qpos), dtype=np.float64)
    endpoint_computed[0] = float(overall[0, 0])
    for endpoint in range(1, len(qpos)):
        endpoint_computed[endpoint] = float(overall[endpoint - 1, -1])
    endpoint_cross_check_max_error = float(
        np.max(np.abs(endpoint_computed - archived_native))
    )
    if endpoint_cross_check_max_error > 1.0e-12:
        raise RuntimeError("dense endpoint results do not match archived native distances")

    with (OUTPUT / "interval_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=(
                "source_endpoint",
                "outcome_endpoint",
                "source_time_s",
                "outcome_time_s",
                "minimum_distance_m",
                "minimum_side",
                "minimum_alpha",
                "strict_negative_sample_count_with_duplicates",
                "beyond_0p05mm_tolerance_sample_count_with_duplicates",
            ),
        )
        writer.writeheader()
        for interval in range(interval_count):
            local_flat = int(np.argmin(distances[interval]))
            sample_index, side_index = np.unravel_index(
                local_flat, distances[interval].shape
            )
            writer.writerow(
                {
                    "source_endpoint": interval,
                    "outcome_endpoint": interval + 1,
                    "source_time_s": f"{timestamps[interval]:.17g}",
                    "outcome_time_s": f"{timestamps[interval + 1]:.17g}",
                    "minimum_distance_m": f"{distances[interval, sample_index, side_index]:.17g}",
                    "minimum_side": ("left", "right")[side_index],
                    "minimum_alpha": f"{alphas[sample_index]:.17g}",
                    "strict_negative_sample_count_with_duplicates": int(
                        np.count_nonzero(strict_negative[interval])
                    ),
                    "beyond_0p05mm_tolerance_sample_count_with_duplicates": int(
                        np.count_nonzero(beyond_tolerance[interval])
                    ),
                }
            )

    unique_sample_count = interval_count * (SAMPLES_PER_INTERVAL - 1) + 1
    first_negative_row = None
    if first_negative is not None:
        interval, sample = first_negative
        side_index = int(np.argmin(distances[interval, sample]))
        first_negative_row = {
            "source_endpoint": interval,
            "outcome_endpoint": interval + 1,
            "alpha": float(alphas[sample]),
            "reference_time_s": float(
                timestamps[interval]
                + alphas[sample] * (timestamps[interval + 1] - timestamps[interval])
            ),
            "side": ("left", "right")[side_index],
            "distance_m": float(distances[interval, sample, side_index]),
        }

    result = {
        "schema": "taco_brush_20231006_145_full_reference_dense_native_v1",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "sample": "(brush, brush, bowl)/20231006_145",
        "inputs": {
            "scene": {"path": str(SCENE), "sha256": sha256(SCENE)},
            "robot_reference": {
                "path": str(REFERENCE),
                "sha256": sha256(REFERENCE),
            },
        },
        "path_definition": {
            "reference_endpoint_count": len(qpos),
            "interval_count": interval_count,
            "reference_start_time_s": float(timestamps[0]),
            "reference_end_time_s": float(timestamps[-1]),
            "formula": "each interval uses mj_integratePos(q_ref[i], alpha*mj_differentiatePos(q_ref[i],q_ref[i+1]))",
            "robot_qpos_addresses": [0, ROBOT_QPOS_STOP],
            "robot_dof_addresses": [0, ROBOT_DOF_STOP],
            "object_qpos": "frozen at each interval source endpoint",
            "joint_type_counts": {
                "slide": robot_joint_types.count("slide"),
                "hinge": robot_joint_types.count("hinge"),
            },
            "samples_per_interval_including_both_endpoints": SAMPLES_PER_INTERVAL,
            "unique_path_sample_count": unique_sample_count,
            "alpha_step": float(alphas[1] - alphas[0]),
            "nominal_time_step_s": float(
                (timestamps[1] - timestamps[0]) * (alphas[1] - alphas[0])
            ),
            "table_z_m": PLANE_Z_M,
            "existing_numerical_tolerance_m": TOLERANCE_M,
            "physics_steps": 0,
            "controls_applied": False,
        },
        "endpoint_cross_check": {
            "maximum_absolute_error_m": endpoint_cross_check_max_error,
            "strictly_negative_endpoint_count": int(
                np.count_nonzero(endpoint_computed < 0.0)
            ),
            "below_existing_tolerance_endpoint_count": int(
                np.count_nonzero(endpoint_computed < -TOLERANCE_M)
            ),
        },
        "whole_path": {
            "minimum_distance_m": global_minimum,
            "minimum_source_endpoint": int(worst_interval),
            "minimum_outcome_endpoint": int(worst_interval + 1),
            "minimum_alpha": worst_alpha,
            "minimum_reference_time_s": float(
                timestamps[worst_interval]
                + worst_alpha
                * (timestamps[worst_interval + 1] - timestamps[worst_interval])
            ),
            "minimum_side": worst_side,
            "minimum_full_vertex_witness": witness,
            "strict_negative_unique_sample_count": strict_negative_points,
            "beyond_existing_tolerance_unique_sample_count": beyond_tolerance_points,
            "first_strict_negative": first_negative_row,
            "first_beyond_existing_tolerance": first_beyond_tolerance,
            "intervals_with_strict_negative_samples": np.flatnonzero(
                np.any(strict_negative, axis=1)
            ),
            "intervals_beyond_existing_tolerance": np.flatnonzero(
                np.any(beyond_tolerance, axis=1)
            ),
            "maximum_adjacent_sample_change_m": float(
                np.max(np.abs(np.diff(distances, axis=1)))
            ),
        },
        "plain_language_decision": {
            "completely_nonpenetrating": bool(global_minimum >= 0.0),
            "passes_existing_0p05mm_tolerance": bool(
                global_minimum >= -TOLERANCE_M
            ),
            "scope": "native hand visual meshes versus the table plane along the defined full robot-reference interpolation path",
        },
    }
    write_json(OUTPUT / "results.json", result)

    first = first_negative_row
    if first is None:
        first_text = "没有发现负距离。"
    else:
        first_side = {"left": "左", "right": "右"}[first["side"]]
        first_text = (
            f"最早负距离出现在区间 {first['source_endpoint']}->{first['outcome_endpoint']}，"
            f"alpha={first['alpha']:.4f}，参考时刻 {first['reference_time_s']:.9f} s，"
            f"{first_side}手距离 {first['distance_m'] * 1000:.6f} mm。"
        )
    strict_text = "是" if global_minimum >= 0.0 else "否"
    tolerance_text = "是" if global_minimum >= -TOLERANCE_M else "否"
    report = f"""# 20231006_145 全参考路径密集手-桌检查

## 结论

如果“完全安全”指手部原始网格在任何时刻都不能低于桌面，答案是：**{strict_text}**。

全路径最小距离为 `{global_minimum * 1000:.6f} mm`，位于区间 `{worst_interval}->{worst_interval + 1}` 的 `alpha={worst_alpha:.4f}`，对应参考时刻 `{result['whole_path']['minimum_reference_time_s']:.9f} s`。负号表示手部网格低于桌面。最深位置是 `{witness['body']}` 的 `{witness['geom']}`，顶点 `{witness['vertex_index']}`。

若沿用当前审计允许的 `0.05 mm` 数值容差，则整条路径是否通过：**{tolerance_text}**。这两个结论不矛盾：前者要求严格不穿透，后者允许小于 `0.05 mm` 的数值误差。

## 检查范围和方法

- 仅检查最新正式参考 `20231006_145`，共 `122` 个端点、`121` 个相邻区间，时间范围 `{timestamps[0]:.9f}` 到 `{timestamps[-1]:.9f} s`。
- 每个区间均匀检查 `{SAMPLES_PER_INTERVAL}` 个位置，相邻位置的 `alpha` 间隔为 `{alphas[1] - alphas[0]:.4g}`；去除区间共享端点后共有 `{unique_sample_count}` 个不同位置。
- 每个区间使用 MuJoCo 的关节位置差分与积分函数连接两个参考端点。机器人关节为 `6` 个直线关节和 `30` 个转动关节。
- 距离取手部原始可视网格到 `z={PLANE_Z_M:.2f} m` 桌面的最小有符号距离。负数表示穿入桌面。
- 只做位置计算，没有推进物理，没有施加控制器，也没有改变参考、桌面或容差。

## 关键数字

- 严格小于零的参考端点：`{result['endpoint_cross_check']['strictly_negative_endpoint_count']}` 个。
- 严格小于零的密集采样位置：`{strict_negative_points}` 个。
- 低于 `-0.05 mm` 的密集采样位置：`{beyond_tolerance_points}` 个。
- {first_text}
- 端点复算值与原参考文件记录值的最大差异：`{endpoint_cross_check_max_error * 1e12:.6f} pm`。

## 能说明什么

这条完整参考插值路径不是“处处严格零穿透”，因为它自身包含微小负距离。若全路径最深值仍在 `0.05 mm` 容差内，它只能说明参考满足当前数值容差，不能把它描述成具有正安全余量的路径。该检查只回答手和桌面的几何关系，不评价手和物体、自碰撞或真实动力学跟踪。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result["plain_language_decision"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
