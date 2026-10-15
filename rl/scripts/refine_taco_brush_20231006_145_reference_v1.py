#!/usr/bin/env python3
"""Run the frozen local continuous-reference refinement contract."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
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
from egoengine_repro.retarget.trajectory_refinement import (  # noqa: E402
    file_sha256,
    immutable_clearance_blockers,
    resolve_variables,
)


DEFAULT_CONFIG = (
    RL_ROOT
    / "configs/taco_brush_20231006_145_continuous_reference_refinement_v1.yaml"
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


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
            value, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False, default=json_default,
        ) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def joint_budget(name: str, cfg: dict[str, Any]) -> float:
    budgets = cfg["modification_budgets"]
    if name.startswith("L_forearm_t"):
        return float(budgets["wrist_translation_m"])
    if name.startswith("L_forearm_"):
        return float(budgets["wrist_rotation_rad"])
    return float(budgets["finger_joint_rad"])


def fingertip_baseline_rows(
    model: mujoco.MjModel, qpos: np.ndarray, endpoints: list[int],
) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        data.qpos[:] = qpos[endpoint]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side in ("right", "left"):
            object_body = model.body(f"{side}_object").id
            rotation = np.asarray(data.xmat[object_body], dtype=np.float64).reshape(3, 3)
            translation = np.asarray(data.xpos[object_body], dtype=np.float64)
            for finger in FINGERS:
                site = model.site(f"{side}_{finger}_tip").id
                world = np.asarray(data.site_xpos[site], dtype=np.float64).copy()
                relative = rotation.T @ (world - translation)
                rows.append({
                    "endpoint": endpoint,
                    "side": side,
                    "finger": finger,
                    "original_world_x_m": world[0],
                    "original_world_y_m": world[1],
                    "original_world_z_m": world[2],
                    "original_object_relative_x_m": relative[0],
                    "original_object_relative_y_m": relative[1],
                    "original_object_relative_z_m": relative[2],
                    "refined_world_x_m": "",
                    "refined_world_y_m": "",
                    "refined_world_z_m": "",
                    "refined_object_relative_x_m": "",
                    "refined_object_relative_y_m": "",
                    "refined_object_relative_z_m": "",
                    "position_change_m": "",
                    "contact_status": "UNKNOWN_EXISTING_NATIVE_EVIDENCE_ONLY",
                    "comparison_status": "NO_REFINED_REFERENCE_EMITTED",
                })
    return rows


def baseline_physics_summary(paths: dict[str, Path], tolerance_m: float) -> dict[str, Any]:
    manifest = json.loads(paths["baseline_physics_manifest"].read_text(encoding="utf-8"))
    results = json.loads(paths["baseline_physics_results"].read_text(encoding="utf-8"))
    with np.load(paths["baseline_physics_state_trace"], allow_pickle=False) as archive:
        states = {name: np.asarray(archive[name]) for name in archive.files}
    terminal = states["substep"] == states["substeps"] - 1
    maximum_error = {"value_rad": -1.0, "joint": None, "endpoint": None}
    saturated = 0
    for contract in manifest["finger_dynamics"]["compiled_contracts"]:
        qpos_address = int(contract["qpos_address"])
        actuator = int(contract["actuator_id"])
        limit = float(contract["urdf_effort_limit_nm"])
        error = np.abs(
            states["reference_qpos"][:, qpos_address]
            - states["actual_qpos"][:, qpos_address]
        )
        terminal_indices = np.flatnonzero(terminal)
        index = int(terminal_indices[np.argmax(error[terminal])])
        if error[index] > maximum_error["value_rad"]:
            maximum_error = {
                "value_rad": float(error[index]),
                "joint": contract["joint_name"],
                "endpoint": int(states["outcome_endpoint"][index]),
            }
        saturated += int(np.count_nonzero(
            np.abs(states["preceding_interval_actuator_force"][:, actuator])
            > limit + 1e-9
        ))
    native_rows = read_csv_gz(paths["baseline_physics_native_trace"])
    actual = np.asarray([
        float(row["actual_minimum_distance_m"]) for row in native_rows
    ])
    violations = np.flatnonzero(actual < -tolerance_m)
    first = None
    if len(violations):
        row = native_rows[int(violations[0])]
        first = {
            "source_endpoint": int(row["source_endpoint"]),
            "outcome_endpoint": int(row["outcome_endpoint"]),
            "substep": int(row["substep"]),
            "state_time_s": float(row["state_time_s"]),
            "distance_m": float(row["actual_minimum_distance_m"]),
        }
    contacts = read_csv_gz(paths["baseline_physics_contact_trace"])
    hand_floor = [row for row in contacts if row["group"] == "hand_floor"]
    endpoint_rows = read_csv(paths["baseline_physics_endpoint_trace"])
    return {
        "classification": results["classification"],
        "physics_steps": int(results["physics_steps"]),
        "maximum_finger_terminal_error": maximum_error,
        "saturated_finger_joint_substeps": saturated,
        "minimum_actual_native_hand_table_distance_m": float(actual.min()),
        "first_actual_beyond_tolerance": first,
        "hand_floor_proxy_contact_rows": len(hand_floor),
        "maximum_brush_position_error_m": max(
            float(row["brush_position_error_m"]) for row in endpoint_rows
        ),
        "maximum_bowl_position_error_m": max(
            float(row["bowl_position_error_m"]) for row in endpoint_rows
        ),
        "reference_time_scale": float(manifest["physics"]["reference_time_scale"]),
        "physics_timestep_s": float(manifest["physics"]["authored_timestep_s"]),
    }


def report_text(
    cfg: dict[str, Any], candidates: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]], baseline: dict[str, Any],
) -> str:
    right_blockers = [
        blocker
        for candidate in candidates
        for blocker in candidate["immutable_clearance_blockers"]
        if blocker["geom"].startswith("right_")
    ]
    unique = {
        (row["endpoint"], row["geom"], row["vertex_index"]): row
        for row in right_blockers
    }
    witness_lines = [
        f"- 参考端点 {row['endpoint']}：`{row['body']}` / `{row['geom']}`，"
        f"最低距离 {row['actual_clearance_m'] * 1000.0:.9f} mm，"
        f"距离 0.5 mm 目标还差 {row['deficit_m'] * 1000.0:.6f} mm。"
        for row in sorted(unique.values(), key=lambda value: value["endpoint"])
    ]
    interval_lines = [
        f"- {row['source_endpoint']}→{row['outcome_endpoint']}："
        f"原参考最小距离 {float(row['original_minimum_distance_m']) * 1000.0:.6f} mm。"
        for row in interval_rows
    ]
    return "\n".join([
        "# 20231006_145 局部连续参考轨迹修复 v1",
        "",
        "## 执行状态",
        "",
        "本轮在优化器求解前的可达性检查处停止，状态为 "
        "`OPTIMIZATION_INFEASIBLE`。没有生成修复参考，也没有运行候选物理回放。"
        "这是几何硬约束与授权自由度之间的冲突，不是求解器未收敛。",
        "",
        "## 最早阻塞点",
        "",
        "核心区间要求左右手全部原生网格至少高于桌面 0.5 mm。M1、M2、M3 "
        "授权的关节全部位于左手运动链；右手任何关节都被冻结。原参考的右手小指"
        "在核心端点 7 和 8 已经低于 0.5 mm 目标，因此这些距离对三个候选都是"
        "不可改变的常量。",
        "",
        *witness_lines,
        "",
        "每个阻塞证据都通过模型父子链核对：危险右手部件的祖先自由度与候选自由度"
        "交集为空。即使左手变量取满预算，右手这两个端点仍保持原距离。",
        "",
        "## M1、M2、M3 记录",
        "",
        "| 候选 | 授权范围 | 不可改变的超限证据数 | 结果 |",
        "|---|---|---:|---|",
        *[
            f"| {row['candidate']} | {', '.join(row['authorized_joints'])} | "
            f"{len(row['immutable_clearance_blockers'])} | 优化前即不可行 |"
            for row in candidates
        ],
        "",
        "M1 失败后按冻结顺序检查 M2，M2 失败后检查 M3。M3 仍然只包含左手腕、"
        "左小指和左无名指，因此同一右手硬约束继续阻塞。未加入未经授权的右手自由度，"
        "也未降低 0.5 mm 目标。",
        "",
        "## 原参考 4→9 几何情况",
        "",
        *interval_lines,
        "",
        "5→6 的最小距离虽然没有超过负容差，但只有约 0.00000154 mm，实际等于没有"
        "动力学余量。6→7 和 7→8 的左小指插值弧线分别穿入桌面约 0.103382 mm 和"
        "0.059546 mm。除此之外，右手端点 7、8 本身也不满足本轮新增的 0.5 mm 正余量。",
        "",
        "## 为什么没有生成新参考和物理回放",
        "",
        "按照冻结流程，只有局部几何审计通过后才能将新参考交给实验 C。当前没有任何"
        "授权候选能满足几何硬约束，因此输出 `robot_reference_refined_v1.npz` 会造成"
        "“已有可行参考”的错误印象。独立候选审计和 Candidate Replay 均标记为未运行。",
        "",
        "作为对照，原实验 C 的记录仍是：",
        "",
        f"- 物理子步：{baseline['physics_steps']}；",
        f"- 手指最大端点误差：{baseline['maximum_finger_terminal_error']['value_rad']:.6f} rad；",
        f"- 手指力矩饱和关节子步：{baseline['saturated_finger_joint_substeps']}；",
        f"- 实际原生手—桌最深距离：{baseline['minimum_actual_native_hand_table_distance_m'] * 1000.0:.3f} mm；",
        f"- MuJoCo 手—桌代理接触记录：{baseline['hand_floor_proxy_contact_rows']}。",
        "",
        "## 任务要求的十项回答",
        "",
        "1. 原参考不安全，是因为 5→6 几乎零余量，6→7 和 7→8 的左小指连续插值"
        "穿桌；同时右手端点 7、8 不满足新增的 0.5 mm 正余量。",
        "2. 没有 endpoint 或关节被修改；三个授权候选都在求解前被证明不可行。",
        "3. 所有关节实际修改量均为零；没有发布修复参考。",
        "4. 连续参考安全余量没有变化。",
        "5. 指尖、手腕和手—物体关系没有变化，因为没有候选被写出。",
        "6. 没有运行候选物理 Replay，因此不能声称首次穿桌已消除。",
        "7. 候选不存在，无法评价候选力矩饱和；原实验 C 的饱和数仍为零。",
        "8. 没有新的候选物体运动；原实验数据未改动。",
        "9. 未解决的是授权范围与双手 0.5 mm 核心余量之间的冲突，以及原参考左小指"
        "的连续穿桌。",
        "10. 本轮没有形成可验证结果；原实验 C 对照仍只适用于 10 倍慢速控制。",
        "",
        "## 产物说明",
        "",
        "`optimization_trace.csv` 和 `native_geometry_audit.json` 保存三个候选的精确"
        "阻塞点、原生网格顶点及运动链证据。`refinement_delta.csv` 明确记录没有应用"
        "任何改动。`physics_replay_comparison.json` 将 Candidate 标记为未运行。",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "taco_brush_20231006_145_continuous_reference_refinement_v1":
        raise ValueError("unexpected refinement config schema")
    paths = {
        name: Path(value).resolve(strict=(name != "output"))
        for name, value in cfg["paths"].items()
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    if file_sha256(paths["robot_reference"]) != cfg["expected_inputs"]["robot_reference_sha256"]:
        raise ValueError("original robot reference hash changed")
    if file_sha256(paths["scene"]) != cfg["expected_inputs"]["scene_sha256"]:
        raise ValueError("scene hash changed")
    if file_sha256(paths["finger_dynamics_contract"]) != cfg["expected_inputs"]["finger_dynamics_contract_sha256"]:
        raise ValueError("finger dynamics contract hash changed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["base_commit"], "HEAD"],
        cwd=REPO_ROOT, check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("configured base commit is not an ancestor of HEAD")

    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    if (model.nq, model.nv, model.nu) != (50, 48, 36):
        raise ValueError("compiled model dimensions changed")
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if reference["qpos"].shape != (122, 50) or reference["ctrl"].shape != (122, 36):
        raise ValueError("original reference dimensions changed")
    if not np.array_equal(reference["ctrl"], reference["qpos"][:, :36]):
        raise ValueError("original control/qpos mapping changed")
    if float(reference["frequency"]) != 30.0:
        raise ValueError("original reference frequency changed")

    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco),
    )
    geometry = cfg["geometry_contract"]
    endpoints = [int(value) for value in cfg["modified_endpoints"]]
    candidates: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for candidate_cfg in cfg["candidate_order"]:
        if not isinstance(candidate_cfg, dict):
            continue
        candidate_id = str(candidate_cfg["id"])
        names = [str(value) for value in candidate_cfg["authorized_joints"]]
        budgets = {name: joint_budget(name, cfg) for name in names}
        variables = resolve_variables(
            model,
            endpoints=endpoints,
            joint_names=names,
            budgets=budgets,
            robot_qpos_stop=int(cfg["sample"]["robot_qpos_stop"]),
        )
        blockers = immutable_clearance_blockers(
            model,
            mujoco,
            reference["qpos"],
            variables,
            geoms,
            endpoints=[int(value) for value in geometry["core_endpoints"]],
            plane_z_m=float(geometry["table_z_m"]),
            minimum_clearance_m=float(geometry["core_minimum_clearance_m"]),
        )
        status = "PRE_QP_IMMUTABLE_GEOMETRY_INFEASIBLE" if blockers else "READY_FOR_SQP"
        candidates.append({
            "candidate": candidate_id,
            "authorized_joints": names,
            "variable_count": len(variables),
            "immutable_clearance_blockers": blockers,
            "status": status,
            "qp_attempted": False,
        })
        if not blockers:
            raise RuntimeError(
                f"{candidate_id} passed the frozen precheck; SQP execution is required"
            )
        for blocker in blockers:
            optimization_rows.append({
                "candidate": candidate_id,
                "stage": "immutable_geometry_reachability_precheck",
                "iteration": -1,
                "status": status,
                "endpoint": blocker["endpoint"],
                "geom": blocker["geom"],
                "body": blocker["body"],
                "vertex_index": blocker["vertex_index"],
                "actual_clearance_m": f"{blocker['actual_clearance_m']:.17g}",
                "required_clearance_m": f"{blocker['required_clearance_m']:.17g}",
                "constraint_residual_m": f"{-blocker['deficit_m']:.17g}",
                "objective_total": "",
                "trust_region_fraction": "",
                "candidate_joint_count": len(names),
                "candidate_variable_count": len(variables),
                "candidate_influencing_dof_count": 0,
                "reason": blocker["reason"],
            })
        for variable in variables:
            delta_rows.append({
                "candidate": candidate_id,
                "endpoint": variable.endpoint,
                "joint": variable.joint_name,
                "budget": f"{variable.budget:.17g}",
                "applied_delta": "",
                "status": "NO_DELTA_CANDIDATE_INFEASIBLE_BEFORE_QP",
            })

    if any(row["status"] != "PRE_QP_IMMUTABLE_GEOMETRY_INFEASIBLE" for row in candidates):
        raise RuntimeError("candidate sequence did not fail at the expected frozen gate")

    prior_intervals = read_csv(paths["prior_dense_interval_summary"])
    relevant = [
        row for row in prior_intervals
        if int(row["source_endpoint"]) in {4, 5, 6, 7, 8}
    ]
    interval_rows = [{
        "source_endpoint": int(row["source_endpoint"]),
        "outcome_endpoint": int(row["outcome_endpoint"]),
        "original_minimum_distance_m": row["minimum_distance_m"],
        "original_minimum_side": row["minimum_side"],
        "original_minimum_alpha": row["minimum_alpha"],
        "refined_minimum_distance_m": "",
        "refined_minimum_side": "",
        "refined_minimum_alpha": "",
        "comparison_status": "NO_REFINED_REFERENCE_EMITTED",
    } for row in relevant]
    fingertip_rows = fingertip_baseline_rows(model, reference["qpos"], endpoints)
    baseline = baseline_physics_summary(
        paths, float(geometry["numerical_penetration_tolerance_m"]),
    )

    manifest = {
        "schema": "taco_brush_20231006_145_continuous_reference_refinement_v1_input_manifest",
        "status": "FROZEN_INPUTS_VERIFIED",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "base_commit": cfg["base_commit"],
        "software": {
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "qp_solver": cfg["optimizer"]["qp_solver"],
        },
        "random_seed": int(cfg["random_seed"]),
        "inputs": {
            "config": artifact(config_path),
            "script": artifact(Path(__file__)),
            **{
                name: artifact(path)
                for name, path in paths.items() if name != "output"
            },
        },
        "compiled_model": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "timestep_s": float(model.opt.timestep),
            "native_hand_geom_count": len(geoms),
        },
        "authorization": cfg["authorization"],
    }
    write_json(output / "input_manifest.json", manifest)
    write_csv(output / "optimization_trace.csv", optimization_rows, list(optimization_rows[0]))
    write_csv(output / "refinement_delta.csv", delta_rows, list(delta_rows[0]))
    write_csv(
        output / "interval_clearance_comparison.csv", interval_rows,
        list(interval_rows[0]),
    )
    write_csv(
        output / "fingertip_contact_comparison.csv", fingertip_rows,
        list(fingertip_rows[0]),
    )
    native_audit = {
        "schema": "taco_brush_20231006_145_refinement_native_geometry_gate_v1",
        "status": "CANDIDATE_AUDIT_NOT_RUN_NO_FEASIBLE_CANDIDATE",
        "pre_qp_method": (
            "full native mesh vertices at every core endpoint; exact body ancestor "
            "DOF intersection with each frozen candidate"
        ),
        "table_z_m": float(geometry["table_z_m"]),
        "core_minimum_clearance_m": float(geometry["core_minimum_clearance_m"]),
        "candidates": candidates,
        "prior_independent_dense_reference_evidence": {
            "samples_per_interval": int(geometry["independent_audit_samples_per_interval"]),
            "intervals": interval_rows,
            "dense_samples": artifact(paths["prior_dense_samples"]),
            "report": artifact(paths["prior_dense_report"]),
        },
        "independent_refined_reference_audit": {
            "status": "NOT_RUN",
            "reason": "NO_REFINED_REFERENCE_EMITTED",
        },
    }
    write_json(output / "native_geometry_audit.json", native_audit)
    physics = {
        "schema": "taco_brush_20231006_145_refined_physics_comparison_v1",
        "status": "CANDIDATE_REPLAY_NOT_RUN_GEOMETRY_GATE_FAILED",
        "baseline": baseline,
        "candidate": None,
        "comparison": None,
        "gate_reason": "ALL_AUTHORIZED_CANDIDATES_OPTIMIZATION_INFEASIBLE",
    }
    write_json(output / "physics_replay_comparison.json", physics)
    results = {
        "schema": "taco_brush_20231006_145_continuous_reference_refinement_v1",
        "classification": "OPTIMIZATION_INFEASIBLE",
        "earliest_blocking_stage": "PRE_QP_IMMUTABLE_GEOMETRY_REACHABILITY",
        "selected_candidate": None,
        "refined_reference_emitted": False,
        "refined_reference_path": None,
        "independent_candidate_geometry_audit": "NOT_RUN",
        "candidate_physics_replay": "NOT_RUN",
        "candidates": candidates,
        "baseline_physics": baseline,
        "original_reference_sha256_after_run": file_sha256(paths["robot_reference"]),
        "original_reference_unchanged": (
            file_sha256(paths["robot_reference"])
            == cfg["expected_inputs"]["robot_reference_sha256"]
        ),
        "missing_by_design": {
            "robot_reference_refined_v1.npz": "NO_FEASIBLE_CANDIDATE",
        },
    }
    write_json(output / "results.json", results)
    (output / "report.md").write_text(
        report_text(cfg, candidates, interval_rows, baseline), encoding="utf-8",
    )
    (output / "reproduction_commands.txt").write_text(
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python "
        "rl/scripts/refine_taco_brush_20231006_145_reference_v1.py "
        "--config rl/configs/taco_brush_20231006_145_continuous_reference_refinement_v1.yaml\n"
        "PYTHONPATH=/tmp/egoengine_mujoco_370.3FmVPI:rl/src "
        "/data_all/zzx/deximit_isolated/env-py311-cu118/bin/python -m pytest -q "
        "rl/tests/core/test_reference_trajectory_refinement_v1.py\n",
        encoding="utf-8",
    )
    with (output / "artifacts.sha256").open("w", encoding="utf-8") as stream:
        for path in sorted(output.iterdir()):
            if path.name == "artifacts.sha256":
                continue
            stream.write(f"{file_sha256(path)}  {path.name}\n")
    print(json.dumps({
        "classification": results["classification"],
        "candidate_statuses": {
            row["candidate"]: row["status"] for row in candidates
        },
        "refined_reference_emitted": False,
        "candidate_physics_replay": "NOT_RUN",
    }, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
