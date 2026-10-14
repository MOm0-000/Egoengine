#!/usr/bin/env python3
"""Compare frozen endpoint-hold and substep-interpolated Brush Replays."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
BASELINE = RL_ROOT / "runs/taco_brush_20231006_145_initialization_controlled_replay_v1"
CANDIDATE = (
    RL_ROOT
    / "runs/taco_brush_20231006_145_initialization_controlled_replay_interpolated_v1"
)
DENSE_REFERENCE = (
    RL_ROOT / "runs/taco_brush_20231006_145_full_reference_dense_native_v1"
    / "dense_samples.npz"
)
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_control_timing_ab_v1"
TOLERANCE_M = 5.0e-5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_endpoint_table(path: Path) -> dict[int, np.ndarray]:
    rows = load_json(path / "native_geometry_trace.json")
    return {
        int(row["endpoint"]): np.asarray([
            row["table"]["left_xhand"]["minimum_signed_distance_m"],
            row["table"]["right_xhand"]["minimum_signed_distance_m"],
        ], dtype=np.float64)
        for row in rows
    }


def reference_endpoint_table(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        dense = np.asarray(archive["distances_m"], dtype=np.float64)
    result = np.empty((len(dense) + 1, 2), dtype=np.float64)
    result[0] = dense[0, 0]
    for endpoint in range(1, len(result)):
        result[endpoint] = dense[endpoint - 1, -1]
    return result


def hand_floor_stats(path: Path) -> dict[str, Any]:
    first: dict[str, str] | None = None
    deepest: dict[str, str] | None = None
    impulse = 0.0
    rows = 0
    with gzip.open(path / "contact_trace.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["group"] != "hand_floor":
                continue
            rows += 1
            impulse += max(0.0, float(row["normal_force_n"])) * 0.002
            if first is None:
                first = dict(row)
            if deepest is None or float(row["distance_m"]) < float(deepest["distance_m"]):
                deepest = dict(row)
    if first is None or deepest is None:
        raise RuntimeError(f"no hand-floor contacts in {path}")

    def compact(row: dict[str, str]) -> dict[str, Any]:
        return {
            "source_endpoint": int(row["source_endpoint"]),
            "outcome_endpoint": int(row["outcome_endpoint"]),
            "substep": int(row["substep"]),
            "contact_state_time_s": float(row["contact_state_time_s"]),
            "interval_end_time_s": float(row["interval_end_time_s"]),
            "geom1": row["geom1"],
            "geom2": row["geom2"],
            "distance_m": float(row["distance_m"]),
            "normal_force_n": float(row["normal_force_n"]),
        }

    return {
        "first": compact(first),
        "deepest": compact(deepest),
        "normal_impulse_ns": impulse,
        "contact_row_count": rows,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    baseline_result = load_json(BASELINE / "results.json")
    candidate_result = load_json(CANDIDATE / "results.json")
    baseline_manifest = load_json(BASELINE / "input_manifest.json")
    candidate_manifest = load_json(CANDIDATE / "input_manifest.json")
    baseline_native = native_endpoint_table(BASELINE)
    candidate_native = native_endpoint_table(CANDIDATE)
    reference_native = reference_endpoint_table(DENSE_REFERENCE)

    invariant_input_names = (
        "source_scene", "local_scene", "robot_reference", "support_contract",
        "prior_collision_audit",
    )
    invariant_physics_names = (
        "backend", "authored_timestep_s", "control_hz", "substeps_per_interval",
        "total_planned_physics_steps", "initial_qvel",
    )
    invariant_initial_names = (
        "robot", "objects", "object_velocity", "settled_object_qpos_simulator",
        "object_variants",
    )
    invariants = {
        "input_hashes_match": {
            name: baseline_manifest["inputs"][name]["sha256"]
            == candidate_manifest["inputs"][name]["sha256"]
            for name in invariant_input_names
        },
        "physics_contract_matches": {
            name: baseline_manifest["physics"][name]
            == candidate_manifest["physics"][name]
            for name in invariant_physics_names
        },
        "initial_state_matches": {
            name: baseline_manifest["initial_state_provenance"][name]
            == candidate_manifest["initial_state_provenance"][name]
            for name in invariant_initial_names
        },
        "hard_stop_matches": baseline_manifest["hard_stop"]
        == candidate_manifest["hard_stop"],
        "pickup_gate_matches": baseline_manifest["strict_pickup_gate"]
        == candidate_manifest["strict_pickup_gate"],
    }
    if not all(
        all(value.values()) if isinstance(value, dict) else value
        for value in invariants.values()
    ):
        raise RuntimeError("A/B invariant check failed")

    endpoint_rows: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    newly_unsafe: list[dict[str, Any]] = []
    for endpoint in sorted(set(baseline_native) & set(candidate_native)):
        for side_index, side in enumerate(("left", "right")):
            reference_distance = float(reference_native[endpoint, side_index])
            baseline_distance = float(baseline_native[endpoint][side_index])
            candidate_distance = float(candidate_native[endpoint][side_index])
            reference_safe = reference_distance >= -TOLERANCE_M
            baseline_safe = baseline_distance >= -TOLERANCE_M
            candidate_safe = candidate_distance >= -TOLERANCE_M
            if reference_safe and not baseline_safe:
                outcome = "became_safe" if candidate_safe else "still_penetrating"
            elif reference_safe and baseline_safe and not candidate_safe:
                outcome = "newly_unsafe"
            else:
                outcome = "not_a_baseline_reference_safe_failure"
            row = {
                "endpoint": endpoint,
                "side": side,
                "reference_distance_m": reference_distance,
                "baseline_actual_distance_m": baseline_distance,
                "interpolated_actual_distance_m": candidate_distance,
                "outcome": outcome,
            }
            endpoint_rows.append(row)
            if outcome == "became_safe":
                converted.append(row)
            elif outcome == "still_penetrating":
                remaining.append(row)
            elif outcome == "newly_unsafe":
                newly_unsafe.append(row)

    with (OUTPUT / "endpoint_hand_table_comparison.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(endpoint_rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(endpoint_rows)

    baseline_contacts = hand_floor_stats(BASELINE)
    candidate_contacts = hand_floor_stats(CANDIDATE)
    substep = candidate_result["substep_native_hand_table"]
    early = substep["interval_4_to_5"]
    whole = substep["whole_replay"]
    endpoint_5_baseline = float(baseline_native[5][0])
    endpoint_5_candidate = float(candidate_native[5][0])
    penetration_reduction = abs(endpoint_5_baseline) - abs(endpoint_5_candidate)
    reduction_fraction = penetration_reduction / abs(endpoint_5_baseline)

    result = {
        "schema": "taco_brush_20231006_145_control_timing_ab_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "comparison": {
            "baseline": "outcome endpoint target held for every physics substep",
            "candidate": "source-to-outcome target linearly interpolated by physics substep",
            "only_changed_factor": "actuator position-target timing",
            "invariants": invariants,
        },
        "interval_4_to_5": {
            "reference_minimum_m": early["reference_minimum_distance_m"],
            "baseline_endpoint_5_actual_m": endpoint_5_baseline,
            "interpolated_endpoint_5_actual_m": endpoint_5_candidate,
            "endpoint_5_penetration_reduction_m": penetration_reduction,
            "endpoint_5_penetration_reduction_fraction": reduction_fraction,
            "interpolated_first_actual_penetration_beyond_tolerance": early[
                "first_actual_penetration_beyond_tolerance"
            ],
            "interpolated_reference_safe_actual_penetration_substeps": early[
                "reference_safe_actual_penetration_count"
            ],
            "interpolated_reference_and_actual_both_safe": early[
                "all_reference_and_actual_safe_within_tolerance"
            ],
        },
        "audited_endpoint_observations": {
            "baseline_reference_safe_actual_penetration_count": len(remaining) + len(converted),
            "became_safe_with_interpolation_count": len(converted),
            "remained_penetrating_with_interpolation_count": len(remaining),
            "newly_unsafe_with_interpolation_count": len(newly_unsafe),
        },
        "whole_interpolated_replay": whole,
        "proxy_hand_floor": {
            "baseline": baseline_contacts,
            "interpolated": candidate_contacts,
            "first_contact_delay_s": candidate_contacts["first"]["contact_state_time_s"]
            - baseline_contacts["first"]["contact_state_time_s"],
        },
        "task_outcome": {
            "baseline_brush_maximum_lift_m": baseline_result["pickup"]["brush"][
                "maximum_actual_lift_m"
            ],
            "interpolated_brush_maximum_lift_m": candidate_result["pickup"]["brush"][
                "maximum_actual_lift_m"
            ],
            "reference_brush_maximum_lift_m": candidate_result["pickup"]["brush"][
                "maximum_reference_lift_m"
            ],
            "baseline_stable_brush_pickup": baseline_result["pickup"]["brush"][
                "stable_pickup_pass"
            ],
            "interpolated_stable_brush_pickup": candidate_result["pickup"]["brush"][
                "stable_pickup_pass"
            ],
        },
        "observations": {
            "interpolation_reduced_early_penetration": penetration_reduction > 0.0,
            "interpolation_eliminated_reference_safe_replay_penetrations": len(remaining) == 0,
            "interpolation_produced_reference_and_replay_both_safe": bool(
                early["all_reference_and_actual_safe_within_tolerance"]
            ),
            "stable_brush_pickup_achieved": bool(
                candidate_result["pickup"]["brush"]["stable_pickup_pass"]
            ),
        },
        "source_artifacts": {
            "baseline_results": str(BASELINE / "results.json"),
            "candidate_results": str(CANDIDATE / "results.json"),
            "candidate_substep_trace": str(
                CANDIDATE / "substep_native_hand_table_trace.csv.gz"
            ),
            "dense_reference": str(DENSE_REFERENCE),
        },
    }
    write_json(OUTPUT / "results.json", result)

    first = early["first_actual_penetration_beyond_tolerance"]
    report = f"""# 20231006_145 低层目标保持与子步插值对照

## 观测结果

把整周期保持改为逐物理子步插值后，记录到的早期穿桌深度发生变化，但穿桌仍然存在。

在关键区间 `4->5`，参考路径全程安全，最小余量为 `{early['reference_minimum_distance_m'] * 1000:.6f} mm`。原方案在 endpoint 5 的左手穿透为 `{endpoint_5_baseline * 1000:.6f} mm`；插值方案为 `{endpoint_5_candidate * 1000:.6f} mm`。穿透深度减少 `{penetration_reduction * 1000:.6f} mm`，约 `{reduction_fraction:.1%}`，但仍明显超过 `0.05 mm` 容差。

插值方案第一次超过容差发生在区间 `4->5` 的第 `{first['substep'] + 1}` 个物理子步，步后状态时刻 `{first['state_time_s']:.6f} s`。当时参考仍高于桌面 `{first['reference_minimum_distance_m'] * 1000:.6f} mm`，真实手部已经进入桌面 `{abs(first['actual_minimum_distance_m']) * 1000:.6f} mm`。

## 固定条件

- 两次运行使用相同场景、机器人参考、桌面、自然落座物体状态、物理时间步、控制周期、执行器增益、碰撞参数和停止条件。
- 唯一改变是控制周期内的位置目标时序。
- 原方案在每个周期开始时直接使用下一个端点目标，并保持 16 或 17 个物理子步。
- 插值方案从当前端点目标出发，每个物理子步前进 `1/N`，最后一个子步到达下一个端点。
- 没有修改参考轨迹，没有增加辅助力，没有运行强化学习或参数搜索。

## 是否出现“参考和 Replay 都不穿桌”

没有。原方案共有 `{len(remaining) + len(converted)}` 个已审计的“参考安全、Replay 穿桌”手侧观测。插值后：

- 变为安全：`{len(converted)}` 个；
- 仍然穿桌：`{len(remaining)}` 个；
- 新增穿桌：`{len(newly_unsafe)}` 个。

在插值 Replay 的全部 2017 个物理子步中，有 `{whole['reference_safe_actual_penetration_count']}` 个子步属于“参考安全但真实状态穿桌”。真实手部全程最深进入桌面 `{abs(whole['actual_minimum_distance_m']) * 1000:.6f} mm`。参考插值路径本身在后半段也会进入桌面，子步采样最深 `{abs(whole['reference_minimum_distance_m']) * 1000:.6f} mm`。

## 其他变化

- MuJoCo 代理手—桌首次接触所在物理区间的起点从 `{baseline_contacts['first']['contact_state_time_s']:.3f} s` 推迟到 `{candidate_contacts['first']['contact_state_time_s']:.3f} s`，推迟 `{(candidate_contacts['first']['contact_state_time_s'] - baseline_contacts['first']['contact_state_time_s']) * 1000:.1f} ms`。
- 最大接触力从 `{baseline_result['maximums']['contact_force_n']:.3f} N` 降到 `{candidate_result['maximums']['contact_force_n']:.3f} N`。
- 刷子最大实际抬升从 `{baseline_result['pickup']['brush']['maximum_actual_lift_m'] * 1000:.3f} mm` 变为 `{candidate_result['pickup']['brush']['maximum_actual_lift_m'] * 1000:.3f} mm`，参考要求仍为 `{candidate_result['pickup']['brush']['maximum_reference_lift_m'] * 1000:.3f} mm`，稳定离桌抓取仍未完成。

"""
    (OUTPUT / "report.md").write_text(report.rstrip() + "\n", encoding="utf-8")

    hashes = []
    for path in sorted(value for value in OUTPUT.iterdir() if value.is_file()):
        hashes.append(f"{sha256(path)}  {path.name}")
    (OUTPUT / "artifacts.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps(result["observations"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
