#!/usr/bin/env python3
"""Dense native hand/table audit over all intervals of the refined reference."""

from __future__ import annotations

import argparse
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
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT / "src"))

from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    native_hand_visual_geom_ids,
)


DEFAULT_CONFIG = (
    RL_ROOT
    / "configs/taco_brush_20231006_145_full_refined_reference_dense_native_v2.yaml"
)

_SCENE: Path | None = None
_REFERENCE: Path | None = None
_PRIMARY_ALPHAS: np.ndarray | None = None
_PHASE_ALPHAS: np.ndarray | None = None
_ROBOT_DOF_STOP: int | None = None
_PLANE_Z_M: float | None = None
_MODEL: mujoco.MjModel | None = None
_QPOS: np.ndarray | None = None
_GEOMS_BY_SIDE: dict[str, tuple[Any, ...]] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def initialize_worker() -> None:
    global _MODEL, _QPOS, _GEOMS_BY_SIDE
    if _SCENE is None or _REFERENCE is None:
        raise RuntimeError("worker paths were not configured")
    _MODEL = mujoco.MjModel.from_xml_path(str(_SCENE))
    with np.load(_REFERENCE, allow_pickle=False) as archive:
        _QPOS = np.asarray(archive["qpos"], dtype=np.float64).copy()
    geoms = build_native_support_geoms(
        _MODEL, mujoco, native_hand_visual_geom_ids(_MODEL, mujoco)
    )
    _GEOMS_BY_SIDE = {
        side: tuple(geom for geom in geoms if geom.name.startswith(f"{side}_"))
        for side in ("left", "right")
    }


def grid_distances(
    model: mujoco.MjModel,
    q0: np.ndarray,
    tangent: np.ndarray,
    alphas: np.ndarray,
    geoms_by_side: dict[str, tuple[Any, ...]],
    plane_z_m: float,
) -> np.ndarray:
    data = mujoco.MjData(model)
    result = np.full((len(alphas), 2), np.inf, dtype=np.float64)
    for sample_index, alpha in enumerate(alphas):
        data.qpos[:] = q0
        mujoco.mj_integratePos(model, data.qpos, tangent, float(alpha))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(("left", "right")):
            for geom in geoms_by_side[side]:
                rotation = np.asarray(
                    data.geom_xmat[geom.geom_id], dtype=np.float64
                ).reshape(3, 3)
                distance = float(
                    np.min(
                        geom.support_vertices @ rotation[2]
                        + float(data.geom_xpos[geom.geom_id, 2])
                        - plane_z_m
                    )
                )
                result[sample_index, side_index] = min(
                    result[sample_index, side_index], distance
                )
    return result


def audit_interval(interval: int) -> tuple[int, np.ndarray, np.ndarray]:
    if any(value is None for value in (
        _MODEL,
        _QPOS,
        _GEOMS_BY_SIDE,
        _PRIMARY_ALPHAS,
        _PHASE_ALPHAS,
        _ROBOT_DOF_STOP,
        _PLANE_Z_M,
    )):
        raise RuntimeError("worker was not initialized")
    model = _MODEL
    q0 = _QPOS[interval]
    q1 = _QPOS[interval + 1]
    tangent = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, tangent, 1.0, q0, q1)
    tangent[int(_ROBOT_DOF_STOP):] = 0.0
    primary = grid_distances(
        model, q0, tangent, _PRIMARY_ALPHAS, _GEOMS_BY_SIDE, float(_PLANE_Z_M)
    )
    phase = grid_distances(
        model, q0, tangent, _PHASE_ALPHAS, _GEOMS_BY_SIDE, float(_PLANE_Z_M)
    )
    return interval, primary, phase


def interpolated_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    interval: int,
    alpha: float,
    robot_dof_stop: int,
) -> np.ndarray:
    tangent = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model, tangent, 1.0, qpos[interval], qpos[interval + 1]
    )
    tangent[robot_dof_stop:] = 0.0
    result = qpos[interval].copy()
    mujoco.mj_integratePos(model, result, tangent, float(alpha))
    return result


def full_vertex_witness(
    model: mujoco.MjModel,
    state: np.ndarray,
    side: str,
    plane_z_m: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = state
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco)
    )
    best = None
    for geom in geoms:
        if not geom.name.startswith(f"{side}_"):
            continue
        rotation = np.asarray(
            data.geom_xmat[geom.geom_id], dtype=np.float64
        ).reshape(3, 3)
        translation = np.asarray(data.geom_xpos[geom.geom_id], dtype=np.float64)
        world = geom.full_vertices @ rotation.T + translation
        signed = world[:, 2] - plane_z_m
        vertex = int(np.argmin(signed))
        row = {
            "body": model.body(geom.body_id).name,
            "geom": geom.name,
            "vertex_index": vertex,
            "point_sim_m": world[vertex],
            "distance_m": float(signed[vertex]),
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    if best is None:
        raise RuntimeError(f"no native hand geometry for {side}")
    return best


def unique_primary_count(mask: np.ndarray) -> int:
    result = int(np.count_nonzero(mask[0]))
    for interval in range(1, len(mask)):
        result += int(np.count_nonzero(mask[interval, 1:]))
    return result


def first_occurrence(
    primary_mask: np.ndarray,
    phase_mask: np.ndarray,
    primary_alpha: np.ndarray,
    phase_alpha: np.ndarray,
) -> tuple[int, str, int] | None:
    for interval in range(len(primary_mask)):
        candidates = []
        primary_start = 0 if interval == 0 else 1
        primary_indices = np.flatnonzero(primary_mask[interval, primary_start:])
        if len(primary_indices):
            index = int(primary_indices[0] + primary_start)
            candidates.append((float(primary_alpha[index]), "primary", index))
        phase_indices = np.flatnonzero(phase_mask[interval])
        if len(phase_indices):
            index = int(phase_indices[0])
            candidates.append((float(phase_alpha[index]), "phase_offset", index))
        if candidates:
            _, grid, index = min(candidates)
            return interval, grid, index
    return None


def sample_row(
    occurrence: tuple[int, str, int] | None,
    primary: np.ndarray,
    phase: np.ndarray,
    primary_alpha: np.ndarray,
    phase_alpha: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any] | None:
    if occurrence is None:
        return None
    interval, grid, sample = occurrence
    values = primary if grid == "primary" else phase
    alphas = primary_alpha if grid == "primary" else phase_alpha
    side_index = int(np.argmin(values[interval, sample]))
    alpha = float(alphas[sample])
    return {
        "source_endpoint": interval,
        "outcome_endpoint": interval + 1,
        "grid": grid,
        "alpha": alpha,
        "reference_time_s": float(
            timestamps[interval]
            + alpha * (timestamps[interval + 1] - timestamps[interval])
        ),
        "side": ("left", "right")[side_index],
        "distance_m": float(values[interval, sample, side_index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "taco_brush_20231006_145_full_refined_reference_dense_native_v2":
        raise ValueError("unexpected full-reference audit schema")
    paths = {
        name: Path(value).resolve(strict=(name != "output"))
        for name, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    if sha256(paths["scene"]) != cfg["expected_inputs"]["scene_sha256"]:
        raise ValueError("scene hash changed")
    if sha256(paths["refined_reference"]) != cfg["expected_inputs"]["refined_reference_sha256"]:
        raise ValueError("refined reference hash changed")
    local = json.loads(paths["local_refinement_results"].read_text(encoding="utf-8"))
    local_audit = json.loads(paths["local_refinement_audit"].read_text(encoding="utf-8"))
    if not local["local_physics_gate_passed"] or local_audit["status"] != "PASS":
        raise ValueError("local refined reference did not pass its frozen gates")
    output.mkdir(parents=True)

    scene = paths["scene"]
    reference_path = paths["refined_reference"]
    sample = cfg["sample"]
    geometry = cfg["geometry"]
    robot_dof_stop = int(sample["robot_dof_stop"])
    plane_z_m = float(geometry["table_z_m"])
    tolerance_m = float(geometry["penetration_tolerance_m"])
    primary_count = int(geometry["primary_samples_per_interval"])
    phase_count = int(geometry["phase_offset_samples_per_interval"])
    primary_alpha = np.linspace(0.0, 1.0, primary_count)
    phase_alpha = (np.arange(phase_count, dtype=np.float64) + 0.5) / phase_count

    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy() for name in archive.files}
    qpos = reference["qpos"]
    timestamps = reference["timestamps_s"]
    archived_native = reference["native_support_min_distance_m"]
    if qpos.shape != (int(sample["endpoint_count"]), int(model.nq)):
        raise ValueError("refined reference dimensions changed")
    interval_count = len(qpos) - 1
    if interval_count != int(sample["interval_count"]):
        raise ValueError("refined reference interval count changed")

    global _SCENE, _REFERENCE, _PRIMARY_ALPHAS, _PHASE_ALPHAS
    global _ROBOT_DOF_STOP, _PLANE_Z_M
    _SCENE = scene
    _REFERENCE = reference_path
    _PRIMARY_ALPHAS = primary_alpha
    _PHASE_ALPHAS = phase_alpha
    _ROBOT_DOF_STOP = robot_dof_stop
    _PLANE_Z_M = plane_z_m

    primary_results: list[np.ndarray | None] = [None] * interval_count
    phase_results: list[np.ndarray | None] = [None] * interval_count
    context = mp.get_context("fork")
    with context.Pool(
        processes=int(cfg["runtime"]["workers"]), initializer=initialize_worker
    ) as pool:
        completed = 0
        for interval, primary_values, phase_values in pool.imap_unordered(
            audit_interval, range(interval_count), chunksize=1
        ):
            primary_results[interval] = primary_values
            phase_results[interval] = phase_values
            completed += 1
            if completed % 10 == 0 or completed == interval_count:
                print(f"completed intervals: {completed}/{interval_count}", flush=True)
    if any(value is None for value in primary_results + phase_results):
        raise RuntimeError("one or more intervals were not evaluated")
    primary = np.stack(primary_results)
    phase = np.stack(phase_results)
    if not np.isfinite(primary).all() or not np.isfinite(phase).all():
        raise RuntimeError("nonfinite native distance")

    np.savez_compressed(
        output / "dense_samples.npz",
        primary_distances_m=primary,
        phase_offset_distances_m=phase,
        side_order=np.asarray(["left", "right"]),
        primary_alpha=primary_alpha,
        phase_offset_alpha=phase_alpha,
        source_endpoint=np.arange(interval_count, dtype=np.int64),
        outcome_endpoint=np.arange(1, interval_count + 1, dtype=np.int64),
        timestamps_s=timestamps,
        table_z_m=np.asarray(plane_z_m),
        tolerance_m=np.asarray(tolerance_m),
    )

    primary_overall = np.min(primary, axis=2)
    phase_overall = np.min(phase, axis=2)
    primary_negative = primary_overall < 0.0
    phase_negative = phase_overall < 0.0
    primary_beyond = primary_overall < -tolerance_m
    phase_beyond = phase_overall < -tolerance_m

    candidates = []
    for grid, values, alphas in (
        ("primary", primary, primary_alpha),
        ("phase_offset", phase, phase_alpha),
    ):
        flat = int(np.argmin(values))
        interval, sample_index, side_index = np.unravel_index(flat, values.shape)
        candidates.append({
            "grid": grid,
            "source_endpoint": int(interval),
            "sample_index": int(sample_index),
            "side_index": int(side_index),
            "alpha": float(alphas[sample_index]),
            "distance_m": float(values[interval, sample_index, side_index]),
        })
    worst = min(candidates, key=lambda row: row["distance_m"])
    worst_state = interpolated_qpos(
        model,
        qpos,
        worst["source_endpoint"],
        worst["alpha"],
        robot_dof_stop,
    )
    witness = full_vertex_witness(
        model,
        worst_state,
        ("left", "right")[worst["side_index"]],
        plane_z_m,
    )
    if not np.isclose(
        worst["distance_m"], witness["distance_m"], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("support and full-vertex minima disagree")

    endpoint_computed = np.empty(len(qpos), dtype=np.float64)
    endpoint_computed[0] = float(primary_overall[0, 0])
    for endpoint in range(1, len(qpos)):
        endpoint_computed[endpoint] = float(primary_overall[endpoint - 1, -1])
    endpoint_error = float(np.max(np.abs(endpoint_computed - archived_native)))
    if endpoint_error > 1e-12:
        raise RuntimeError("endpoint distances disagree with the refined archive")

    interval_rows: list[dict[str, Any]] = []
    intervals_beyond = []
    intervals_negative = []
    for interval in range(interval_count):
        local_candidates = []
        for grid, values, alphas in (
            ("primary", primary, primary_alpha),
            ("phase_offset", phase, phase_alpha),
        ):
            flat = int(np.argmin(values[interval]))
            sample_index, side_index = np.unravel_index(
                flat, values[interval].shape
            )
            local_candidates.append({
                "grid": grid,
                "sample_index": sample_index,
                "side_index": side_index,
                "alpha": float(alphas[sample_index]),
                "distance_m": float(values[interval, sample_index, side_index]),
            })
        local = min(local_candidates, key=lambda row: row["distance_m"])
        negative = bool(
            np.any(primary_negative[interval]) or np.any(phase_negative[interval])
        )
        beyond = bool(
            np.any(primary_beyond[interval]) or np.any(phase_beyond[interval])
        )
        if negative:
            intervals_negative.append(interval)
        if beyond:
            intervals_beyond.append(interval)
        interval_rows.append({
            "source_endpoint": interval,
            "outcome_endpoint": interval + 1,
            "source_time_s": f"{timestamps[interval]:.17g}",
            "outcome_time_s": f"{timestamps[interval + 1]:.17g}",
            "minimum_distance_m": f"{local['distance_m']:.17g}",
            "minimum_side": ("left", "right")[local["side_index"]],
            "minimum_grid": local["grid"],
            "minimum_alpha": f"{local['alpha']:.17g}",
            "has_strict_negative_sample": negative,
            "has_sample_beyond_0p05mm_tolerance": beyond,
            "primary_strict_negative_sample_count": int(
                np.count_nonzero(primary_negative[interval])
            ),
            "phase_strict_negative_sample_count": int(
                np.count_nonzero(phase_negative[interval])
            ),
            "primary_beyond_tolerance_sample_count": int(
                np.count_nonzero(primary_beyond[interval])
            ),
            "phase_beyond_tolerance_sample_count": int(
                np.count_nonzero(phase_beyond[interval])
            ),
        })
    with (output / "interval_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(interval_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(interval_rows)

    first_negative = sample_row(
        first_occurrence(
            primary_negative, phase_negative, primary_alpha, phase_alpha
        ),
        primary,
        phase,
        primary_alpha,
        phase_alpha,
        timestamps,
    )
    first_beyond = sample_row(
        first_occurrence(primary_beyond, phase_beyond, primary_alpha, phase_alpha),
        primary,
        phase,
        primary_alpha,
        phase_alpha,
        timestamps,
    )
    primary_unique = interval_count * (primary_count - 1) + 1
    phase_unique = interval_count * phase_count
    strict_negative_samples = (
        unique_primary_count(primary_negative) + int(np.count_nonzero(phase_negative))
    )
    beyond_samples = (
        unique_primary_count(primary_beyond) + int(np.count_nonzero(phase_beyond))
    )
    result = {
        "schema": cfg["schema"],
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "sample": cfg["sample"],
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "scene": {"path": str(scene), "sha256": sha256(scene)},
            "refined_reference": {
                "path": str(reference_path),
                "sha256": sha256(reference_path),
            },
            "local_refinement_results": {
                "path": str(paths["local_refinement_results"]),
                "sha256": sha256(paths["local_refinement_results"]),
            },
            "local_refinement_audit": {
                "path": str(paths["local_refinement_audit"]),
                "sha256": sha256(paths["local_refinement_audit"]),
            },
        },
        "method": {
            "interpolation": (
                "mj_integratePos(q[i], alpha*mj_differentiatePos(q[i],q[i+1]))"
            ),
            "object_dofs": "frozen at each interval source endpoint",
            "geometry": "all native hand visual mesh support vertices",
            "full_vertex_witness_cross_check": True,
            "table_z_m": plane_z_m,
            "tolerance_m": tolerance_m,
            "primary_samples_per_interval": primary_count,
            "phase_offset_samples_per_interval": phase_count,
            "total_distinct_samples_across_both_grids": primary_unique + phase_unique,
            "physics_steps": 0,
            "controls_applied": False,
        },
        "endpoint_cross_check": {
            "maximum_absolute_error_m": endpoint_error,
            "strictly_negative_endpoint_count": int(
                np.count_nonzero(endpoint_computed < 0.0)
            ),
            "below_tolerance_endpoint_count": int(
                np.count_nonzero(endpoint_computed < -tolerance_m)
            ),
        },
        "whole_path": {
            "minimum_distance_m": worst["distance_m"],
            "minimum_source_endpoint": worst["source_endpoint"],
            "minimum_outcome_endpoint": worst["source_endpoint"] + 1,
            "minimum_grid": worst["grid"],
            "minimum_alpha": worst["alpha"],
            "minimum_reference_time_s": float(
                timestamps[worst["source_endpoint"]]
                + worst["alpha"]
                * (
                    timestamps[worst["source_endpoint"] + 1]
                    - timestamps[worst["source_endpoint"]]
                )
            ),
            "minimum_side": ("left", "right")[worst["side_index"]],
            "minimum_full_vertex_witness": witness,
            "strict_negative_sample_count_across_both_grids": strict_negative_samples,
            "beyond_tolerance_sample_count_across_both_grids": beyond_samples,
            "intervals_with_strict_negative_samples": intervals_negative,
            "intervals_beyond_tolerance": intervals_beyond,
            "first_strict_negative": first_negative,
            "first_beyond_tolerance": first_beyond,
        },
        "decision": {
            "completely_nonpenetrating": bool(worst["distance_m"] >= 0.0),
            "passes_0p05mm_tolerance": bool(worst["distance_m"] >= -tolerance_m),
            "local_5_to_8_core_nonpenetrating": all(
                float(interval_rows[index]["minimum_distance_m"]) >= 0.0
                for index in (5, 6, 7)
            ),
            "scope": "full 122-endpoint refined reference; kinematic native hand/table audit only",
        },
    }
    write_json(output / "results.json", result)

    first_text = "没有发现负距离。"
    if first_negative is not None:
        first_text = (
            f"最早负距离出现在 {first_negative['source_endpoint']}→"
            f"{first_negative['outcome_endpoint']}，alpha={first_negative['alpha']:.7f}，"
            f"距离 {first_negative['distance_m'] * 1000.0:.6f} mm。"
        )
    interval_lines = [
        f"- {row['source_endpoint']}→{row['outcome_endpoint']}："
        f"{float(row['minimum_distance_m']) * 1000.0:.6f} mm"
        for row in interval_rows
        if row["has_sample_beyond_0p05mm_tolerance"]
    ]
    report = "\n".join([
        "# 20231006_145 修复后参考的全样本密集手—桌检查",
        "",
        "## 结论",
        "",
        "局部修复后的 5→6、6→7、7→8 均保持正距离，不穿桌。",
        "",
        "但把同一检查扩展到全部 121 个区间后，整条参考仍会穿桌。"
        f"全路径最低距离为 {worst['distance_m'] * 1000.0:.6f} mm，位于 "
        f"{worst['source_endpoint']}→{worst['source_endpoint'] + 1} 的 "
        f"alpha={worst['alpha']:.7f}，最低部件是 {witness['body']}。",
        "",
        f"共有 {len(intervals_beyond)} 个区间低于 -0.05 mm 容差：",
        "",
        *interval_lines,
        "",
        "## 检查方法",
        "",
        f"- 检查 122 个端点之间的全部 121 个区间。",
        f"- 每个区间使用 {primary_count} 个含端点采样，再使用 {phase_count} 个半步错开采样。",
        f"- 两套网格合计 {primary_unique + phase_unique} 个不同的参考姿态。",
        "- 相邻端点使用 MuJoCo 的关节位置差分和积分函数连接。",
        f"- 检查全部原生手部视觉网格到 z={plane_z_m:.2f} m 桌面的距离。",
        "- 没有推进物理、施加控制器或修改参考。",
        "",
        "## 关键数字",
        "",
        f"- {first_text}",
        f"- 严格负距离区间数：{len(intervals_negative)}。",
        f"- 超过 0.05 mm 穿透容差的区间数：{len(intervals_beyond)}。",
        f"- 严格负距离参考端点数：{result['endpoint_cross_check']['strictly_negative_endpoint_count']}。",
        f"- 端点复算与新参考记录的最大差异：{endpoint_error * 1e12:.6f} pm。",
        "",
        "## 解释范围",
        "",
        "该结果只说明修复后的完整参考在运动学插值下仍存在其他手—桌穿透。"
        "它不包含物理控制、手—物体接触或完整抓取判定。",
        "",
    ])
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "reproduction_commands.txt").write_text(
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/audit_taco_brush_20231006_145_full_refined_reference_v2.py "
        "--config rl/configs/taco_brush_20231006_145_full_refined_reference_dense_native_v2.yaml\n",
        encoding="utf-8",
    )
    with (output / "artifacts.sha256").open("w", encoding="utf-8") as stream:
        for path in sorted(output.iterdir()):
            if path.name == "artifacts.sha256":
                continue
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
