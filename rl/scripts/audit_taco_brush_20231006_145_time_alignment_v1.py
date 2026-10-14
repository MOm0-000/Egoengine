#!/usr/bin/env python3
"""Audit state, geometry, force and timestamp alignment for sample 20231006_145."""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_time_alignment_code_audit_v1"
RUNS = {
    "整周期保持": RL_ROOT / "runs/taco_brush_20231006_145_initialization_controlled_replay_v1",
    "子步插值": RL_ROOT / "runs/taco_brush_20231006_145_initialization_controlled_replay_interpolated_v1",
    "端点0到4捕获": RL_ROOT / "runs/taco_brush_20231006_145_endpoint_1_4_hand_table_capture_v1",
}
MJWARP_FORWARD = Path(
    "/data_all/zzx/3.2RL/.env_mjwp313_overlay/mujoco_warp/_src/forward.py"
)


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def line_of(path: Path, text: str) -> int:
    for line, value in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if text in value:
            return line
    raise RuntimeError(f"text not found in {path}: {text}")


def function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return {
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = first / np.linalg.norm(first)
    right = second / np.linalg.norm(second)
    return float(2.0 * np.arccos(np.clip(abs(float(left @ right)), 0.0, 1.0)))


def chinese_bool(value: bool) -> str:
    return "是" if value else "否"


def check_run(label: str, path: Path) -> dict[str, Any]:
    manifest = load_json(path / "input_manifest.json")
    results = load_json(path / "results.json")
    dt = float(manifest["physics"]["authored_timestep_s"])
    contacts = read_gzip_csv(path / "contact_trace.csv.gz")
    contact_start_matches = all(
        float(row["contact_state_time_s"]) == float(row["interval_start_time_s"])
        for row in contacts
    )
    contact_duration_error = max((
        abs(
            float(row["interval_end_time_s"])
            - float(row["interval_start_time_s"])
            - dt
        )
        for row in contacts
    ), default=0.0)
    endpoint_rows = list(csv.DictReader(
        (path / "endpoint_trace.csv").open(encoding="utf-8", newline="")
    ))
    expected_final_time = int(results["physics_steps"]) * dt
    final_time_error = abs(float(endpoint_rows[-1]["time_s"]) - expected_final_time)

    native_path = path / "substep_native_hand_table_trace.csv.gz"
    native_check = None
    if native_path.exists():
        native_rows = read_gzip_csv(native_path)
        state_time_matches_end = all(
            float(row["state_time_s"]) == float(row["force_interval_end_time_s"])
            for row in native_rows
        )
        native_duration_error = max(
            abs(
                float(row["force_interval_end_time_s"])
                - float(row["force_interval_start_time_s"])
                - dt
            )
            for row in native_rows
        )
        native_check = {
            "row_count": len(native_rows),
            "matches_physics_step_count": len(native_rows) == int(results["physics_steps"]),
            "state_time_equals_interval_end": state_time_matches_end,
            "maximum_interval_duration_error_s": native_duration_error,
        }

    reconstruction = manifest["initial_state_provenance"][
        "natural_seating_state_reconstruction"
    ]
    corrections = {}
    for name, row in reconstruction.items():
        before = np.asarray(row["pre_integration_world_pose"], dtype=np.float64)
        after = np.asarray(row["reconstructed_world_pose"], dtype=np.float64)
        corrections[name] = {
            "translation_m": (after[:3] - before[:3]).tolist(),
            "translation_norm_m": float(np.linalg.norm(after[:3] - before[:3])),
            "rotation_rad": rotation_distance(before[3:7], after[3:7]),
            "reconstructed_pose_time_s": float(row["reconstructed_pose_time_s"]),
        }

    return {
        "label": label,
        "backend": manifest["physics"]["backend"],
        "physics_steps": int(results["physics_steps"]),
        "physics_timestep_s": dt,
        "native_geometry_alignment_contract": manifest["physics"][
            "native_geometry_state_alignment"
        ],
        "force_alignment_contract": manifest["physics"]["force_time_alignment"],
        "contact_row_count": len(contacts),
        "contact_state_time_equals_interval_start": contact_start_matches,
        "maximum_contact_interval_duration_error_s": contact_duration_error,
        "endpoint_final_time_error_s": final_time_error,
        "substep_native_trace": native_check,
        "natural_seating_final_state_correction": corrections,
        "artifacts": {
            "manifest": artifact(path / "input_manifest.json"),
            "results": artifact(path / "results.json"),
            "contact_trace": artifact(path / "contact_trace.csv.gz"),
            **({"native_trace": artifact(native_path)} if native_path.exists() else {}),
        },
    }


def mjwp_one_step_probe() -> dict[str, Any]:
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body><freejoint/><geom type="sphere" size="0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    initial_qpos_x = float(data.qpos[0])
    initial_geom_xpos_x = float(data.geom_xpos[0, 0])
    wp.init()
    with wp.ScopedDevice("cpu"):
        model_wp = mjwarp.put_model(model)
        data_wp = mjwarp.put_data(
            model, data, nworld=1, nconmax=32, njmax=64,
        )
        mjwarp.step(model_wp, data_wp)
        wp.synchronize()
        post_step_qpos_x = float(wp.to_torch(data_wp.qpos)[0, 0])
        cached_geom_xpos_x = float(wp.to_torch(data_wp.geom_xpos)[0, 0, 0])
    return {
        "device": "cpu",
        "timestep_s": float(model.opt.timestep),
        "initial_linear_velocity_x_m_s": float(data.qvel[0]),
        "initial_qpos_x_m": initial_qpos_x,
        "initial_geom_xpos_x_m": initial_geom_xpos_x,
        "post_step_qpos_x_m": post_step_qpos_x,
        "post_step_cached_geom_xpos_x_m": cached_geom_xpos_x,
        "state_geometry_difference_m": post_step_qpos_x - cached_geom_xpos_x,
        "one_step_time_skew_reproduced": bool(
            np.isclose(post_step_qpos_x, 0.01, atol=1.0e-7)
            and np.isclose(cached_geom_xpos_x, 0.0, atol=1.0e-7)
        ),
        "versions": {
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(getattr(mjwarp, "__version__", "unknown")),
            "warp": str(wp.__version__),
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    run_checks = {label: check_run(label, path) for label, path in RUNS.items()}
    endpoint_audit_path = (
        RL_ROOT / "runs/taco_brush_20231006_145_endpoint_1_4_hand_table_audit_v1"
        / "results.json"
    )
    endpoint_audit = load_json(endpoint_audit_path)
    mjwp_probe = mjwp_one_step_probe()

    controlled = RL_ROOT / "scripts/run_taco_brush_controlled_physics_replay_v1.py"
    seating = RL_ROOT / "scripts/audit_taco_brush_20231006_145_issue14_initialization_v1.py"
    fallback = RL_ROOT / "scripts/run_taco_brush_controlled_rl_fallback_v1.py"
    generic_replay = RL_ROOT / "src/egoengine_repro/action/replay.py"
    mjwp_env = RL_ROOT / "src/video_to_spider/rl/mjwp_env.py"
    mjwp_backend = RL_ROOT / "external/spider_compat/spider/simulators/mjwp.py"

    mjwp_setup_calls = function_calls(mjwp_backend, "setup_env")
    mjwp_step_calls = function_calls(mjwp_backend, "step_env")
    warp_text = MJWARP_FORWARD.read_text(encoding="utf-8")
    warp_step_order = (
        "def step(m: Model, d: Data):" in warp_text
        and warp_text.index("  forward(m, d)", warp_text.index("def step(m: Model, d: Data):"))
        < warp_text.index("    euler(m, d)", warp_text.index("def step(m: Model, d: Data):"))
    )

    code_paths = {
        "本次CPU回放": {
            "运行状态": "已执行",
            "时间同步状态": "已对齐",
            "代码位置": {
                "物理步": [str(controlled), line_of(controlled, "mujoco.mj_step(model, data)")],
                "步后几何副本": [
                    str(controlled), line_of(controlled, "synchronized_forward_snapshot(")
                ],
                "接触区间起点": [
                    str(controlled), line_of(controlled, "interval_start_time_s = float(data.time)")
                ],
            },
            "说明": (
                "qpos、qvel和原生几何使用步后时刻；接触和执行器力使用前一物理区间，"
                "两类数据分别保存时刻。"
            ),
        },
        "自然落座初态输入": {
            "运行状态": "未重复执行自然落座物理审计",
            "时间同步状态": "本次Replay输入已恢复到记录时刻",
            "代码位置": {
                "末步恢复": [str(controlled), line_of(controlled, "integrate_freejoint_pose(")],
                "今后记录": [str(seating), line_of(seating, "mujoco.mj_forward(model, sample_data)")],
            },
            "说明": (
                "本次Replay从已保存的步前自由关节位姿和步后速度恢复最后一个2 ms物理步后的位姿。"
            ),
        },
        "CPU回退路径": {
            "运行状态": "本次未执行",
            "时间同步状态": "记录代码已对齐",
            "代码位置": {
                "步后几何副本": [
                    str(fallback), line_of(fallback, "synchronized_forward_snapshot(")
                ],
                "接触区间起点": [
                    str(fallback), line_of(fallback, "interval_start_time_s = float(data.time)")
                ],
            },
        },
        "通用MuJoCo Replay": {
            "运行状态": "本次未执行",
            "时间同步状态": "代码已有步后前向计算",
            "代码位置": {
                "步后前向计算": [
                    str(generic_replay), line_of(generic_replay, "self.mujoco.mj_forward(self.model, self.data)")
                ],
            },
        },
        "MuJoCo-Warp观察路径": {
            "运行状态": "本次未执行",
            "时间同步状态": "仍存在一步错位",
            "初始化不推进物理": (
                "mujoco.mj_forward" in mjwp_setup_calls
                and "mujoco.mj_step" not in mjwp_setup_calls
            ),
            "step_env调用mjwarp.step": "mjwarp.step" in mjwp_step_calls,
            "依赖实现先forward再积分": warp_step_order,
            "代码位置": {
                "物理步": [str(mjwp_backend), line_of(mjwp_backend, "mjwarp.step(env.model_wp, env.data_wp)")],
                "子步观察回调": [str(mjwp_env), line_of(mjwp_env, "substep_observer(substep)")],
                "步后qpos读取": [str(mjwp_env), line_of(mjwp_env, "qpos = self._mjwp.get_qpos")],
                "派生site读取": [str(mjwp_env), line_of(mjwp_env, "site_xpos = wp.to_torch(self.env.data_wp.site_xpos)")],
                "依赖step": [str(MJWARP_FORWARD), line_of(MJWARP_FORWARD, "def step(m: Model, d: Data):")],
            },
            "说明": (
                "mjwarp.step先计算派生量和接触，再积分qpos/qvel。当前观察构造直接把步后qpos/qvel"
                "与积分前的site_xpos、contact和力放在同一个观察中。"
            ),
        },
    }

    all_current_run_checks_pass = all(
        row["backend"] == "mujoco_cpu"
        and row["contact_state_time_equals_interval_start"]
        and row["maximum_contact_interval_duration_error_s"] <= 1.0e-12
        and row["endpoint_final_time_error_s"] <= 1.0e-12
        and (
            row["substep_native_trace"] is None
            or (
                row["substep_native_trace"]["matches_physics_step_count"]
                and row["substep_native_trace"]["state_time_equals_interval_end"]
                and row["substep_native_trace"]["maximum_interval_duration_error_s"]
                <= 1.0e-12
            )
        )
        for row in run_checks.values()
    )

    result = {
        "schema": "taco_brush_20231006_145_time_alignment_code_audit_v1",
        "sample": "(brush, brush, bowl)/20231006_145",
        "current_replay_backend": "mujoco_cpu",
        "current_run_log_checks_all_passed": all_current_run_checks_pass,
        "endpoint_0_to_4": {
            "capture_matches_full_replay_prefix": endpoint_audit[
                "capture_matches_parent_prefix"
            ],
            "state_and_native_geometry_share_the_same_timestamp": endpoint_audit[
                "state_and_native_geometry_share_the_same_timestamp"
            ],
            "native_mesh_recompute_max_abs_error_m": endpoint_audit[
                "native_mesh_trace_recompute_max_abs_error_m"
            ],
        },
        "mjwp_one_step_runtime_probe": mjwp_probe,
        "run_checks": run_checks,
        "code_paths": code_paths,
        "source_artifacts": {
            "controlled_replay": artifact(controlled),
            "natural_seating": artifact(seating),
            "cpu_fallback": artifact(fallback),
            "generic_replay": artifact(generic_replay),
            "mjwp_environment": artifact(mjwp_env),
            "mjwp_backend": artifact(mjwp_backend),
            "mujoco_warp_forward": artifact(MJWARP_FORWARD),
            "endpoint_audit": artifact(endpoint_audit_path),
        },
    }
    write_json(OUTPUT / "results.json", result)

    interpolated = run_checks["子步插值"]
    corrections = interpolated["natural_seating_final_state_correction"]
    lines = [
        "# 20231006_145 时间同步代码审计", "",
        "## 本次运行链路", "",
        "- 当前正式对照使用 CPU MuJoCo，不使用 MuJoCo-Warp。",
        f"- 三个当前运行的日志时刻检查全部通过：{chinese_bool(all_current_run_checks_pass)}。",
        "- 每个物理子步先保存区间起点，再调用 `mj_step`；步后的 `qpos/qvel` 和原生几何记在区间终点。",
        "- 原生几何由独立 `MjData` 复制步后积分状态，再调用 `mj_forward` 计算，不读取实时 `MjData` 中尚未刷新的几何缓存。",
        "- `contact`、接触力和执行器力描述刚完成的物理区间。日志将区间起点、区间终点和步后状态时刻分成独立字段。", "",
        "## 日志一致性", "",
    ]
    for label, row in run_checks.items():
        native = row["substep_native_trace"]
        native_text = "未配置逐子步原生距离"
        if native is not None:
            native_text = (
                f"{native['row_count']} 行，步后状态时刻等于区间终点："
                f"{chinese_bool(native['state_time_equals_interval_end'])}"
            )
        lines.append(
            f"- {label}：{row['physics_steps']} 个物理步；接触时刻等于区间起点："
            f"{chinese_bool(row['contact_state_time_equals_interval_start'])}；{native_text}。"
        )
    lines += [
        f"- endpoint 0–4 捕获与完整插值回放前缀逐字段一致："
        f"{chinese_bool(endpoint_audit['capture_matches_parent_prefix'])}。",
        f"- 保存姿态离线重算与在线原生距离的最大绝对差："
        f"`{endpoint_audit['native_mesh_trace_recompute_max_abs_error_m']:.3e} m`。", "",
        "## 初态时刻", "",
        "- 自然落座结果没有再次运行。Replay 读取保存的自由关节位姿和同一物理步的步后速度，用 MuJoCo 位置积分恢复记录时刻的完整位姿。",
        f"- 刷子末步平移恢复量：`{corrections['brush']['translation_norm_m'] * 1000:.9f} mm`；"
        f"碗末步平移恢复量：`{corrections['bowl']['translation_norm_m'] * 1000:.9f} mm`。",
        "- 恢复后的物体姿态同时写入三次当前 Replay 的 `input_manifest.json`。", "",
        "## 代码路径清点", "",
        "| 路径 | 本次是否执行 | 当前时间同步状态 |",
        "|---|---|---|",
    ]
    for name, row in code_paths.items():
        lines.append(
            f"| {name} | {row['运行状态']} | {row['时间同步状态']} |"
        )
    lines += [
        "", "## MuJoCo-Warp 路径中的错位", "",
        "- 初始化已使用 `mj_forward`，不会为填充派生量而把参考初态提前推进一个物理步。",
        "- `mjwarp.step` 的实现顺序仍是：先根据当前状态执行 `forward`，再积分到下一状态。",
        "- 环境在 `step_env` 返回后直接调用子步观察器；控制周期结束后也直接读取步后 `qpos/qvel`、`site_xpos`、接触和力。",
        "- 因此这条未参与本次 CPU Replay 的路径中，`qpos/qvel` 属于步后，`site_xpos`、接触和力仍属于积分前，代码层面仍有一个物理子步的时间错位。", "",
        "## 最小运行验证", "",
        f"- 单球模型步长 `{mjwp_probe['timestep_s'] * 1000:.0f} ms`，水平初速度 "
        f"`{mjwp_probe['initial_linear_velocity_x_m_s']:.0f} m/s`。",
        f"- 一步后 `qpos.x = {mjwp_probe['post_step_qpos_x_m']:.9f} m`，"
        f"缓存的 `geom_xpos.x = {mjwp_probe['post_step_cached_geom_xpos_x_m']:.9f} m`。",
        f"- 两者相差 `{mjwp_probe['state_geometry_difference_m']:.9f} m`，"
        f"恰好是一个物理步的位移；最小程序复现错位："
        f"{chinese_bool(mjwp_probe['one_step_time_skew_reproduced'])}。", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "artifacts.sha256"
    ]
    (OUTPUT / "artifacts.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "current_run_log_checks_all_passed": all_current_run_checks_pass,
        "mjwp_observation_time_alignment": code_paths["MuJoCo-Warp观察路径"][
            "时间同步状态"
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
