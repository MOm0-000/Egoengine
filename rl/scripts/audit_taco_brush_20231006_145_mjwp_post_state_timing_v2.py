#!/usr/bin/env python3
"""Audit MJWP post-state geometry and last-step dynamics time separation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from spider.simulators import mjwp


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_mjwp_post_state_timing_v2"
BACKEND = RL_ROOT / "external/spider_compat/spider/simulators/mjwp.py"
ENVIRONMENT = RL_ROOT / "src/video_to_spider/rl/mjwp_env.py"
STARTUP = RL_ROOT / "src/video_to_spider/rl/core/startup_planner.py"
TRACE = RL_ROOT / "scripts/trace_replay_0_20.py"
FALLBACK = RL_ROOT / "scripts/run_taco_brush_controlled_rl_fallback_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(function)


def line_of(path: Path, text: str) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if text in line:
            return index
    raise RuntimeError(f"missing expected source text: {path}: {text}")


def exact_torch_copy(value: Any) -> torch.Tensor:
    return wp.to_torch(value).clone()


def run_probe() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body name="slider" pos="0 0 .04">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom name="ball" type="sphere" size=".05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)

    wp.set_device("cpu")
    model_wp = mjwarp.put_model(model)
    live = mjwarp.put_data(model, data, nworld=1, nconmax=16, njmax=32)
    observation = mjwarp.put_data(model, data, nworld=1, nconmax=16, njmax=32)
    rollback = mjwarp.put_data(model, data, nworld=1, nconmax=16, njmax=32)
    env = mjwp.MJWPEnv(
        model_cpu=model,
        data_cpu=data,
        model_wp=model_wp,
        data_wp=live,
        data_wp_observation=observation,
        data_wp_prev=rollback,
        graph=None,
        device="cpu",
        num_worlds=1,
    )
    wp.copy(rollback.qpos, wp.from_torch(torch.tensor([[7.0]], dtype=torch.float32)))

    mjwarp.step(model_wp, live)
    live_names = (
        "qpos",
        "qvel",
        "time",
        "geom_xpos",
        "site_xpos",
        "qfrc_actuator",
        "actuator_force",
        "qfrc_constraint",
        "nacon",
        "nefc",
    )
    contact_names = ("dist", "geom", "worldid", "efc_address")
    live_before = {name: exact_torch_copy(getattr(live, name)) for name in live_names}
    contact_before = {
        name: exact_torch_copy(getattr(live.contact, name)) for name in contact_names
    }
    efc_before = exact_torch_copy(live.efc.force)
    rollback_before = exact_torch_copy(rollback.qpos)

    mjwp.refresh_post_state_kinematics(env)
    wp.synchronize()

    live_after = {name: exact_torch_copy(getattr(live, name)) for name in live_names}
    contact_after = {
        name: exact_torch_copy(getattr(live.contact, name)) for name in contact_names
    }
    efc_after = exact_torch_copy(live.efc.force)
    rollback_after = exact_torch_copy(rollback.qpos)

    unchanged = {
        **{
            name: bool(torch.equal(live_before[name], live_after[name]))
            for name in live_names
        },
        **{
            f"contact.{name}": bool(
                torch.equal(contact_before[name], contact_after[name])
            )
            for name in contact_names
        },
        "efc.force": bool(torch.equal(efc_before, efc_after)),
    }
    rollback_unchanged = bool(torch.equal(rollback_before, rollback_after))

    cpu_post = mujoco.MjData(model)
    cpu_post.qpos[:] = wp.to_torch(observation.qpos)[0].numpy()
    cpu_post.qvel[:] = wp.to_torch(observation.qvel)[0].numpy()
    mujoco.mj_forward(model, cpu_post)
    observation_geom = wp.to_torch(observation.geom_xpos)[0].numpy().copy()
    observation_site = wp.to_torch(observation.site_xpos)[0].numpy().copy()
    cpu_geom_error = float(np.max(np.abs(observation_geom - cpu_post.geom_xpos)))
    cpu_site_error = float(
        np.max(np.abs(observation_site - cpu_post.site_xpos))
        if observation_site.size
        else 0.0
    )

    live_qpos = float(wp.to_torch(live.qpos)[0, 0])
    live_geom_x = float(wp.to_torch(live.geom_xpos)[0, 1, 0])
    observation_qpos = float(wp.to_torch(observation.qpos)[0, 0])
    observation_geom_x = float(wp.to_torch(observation.geom_xpos)[0, 1, 0])
    results = {
        "physics_timestep_s": float(model.opt.timestep),
        "live_qpos_after_step": live_qpos,
        "live_geom_cache_x_after_step": live_geom_x,
        "observation_qpos_after_refresh": observation_qpos,
        "observation_geom_x_after_refresh": observation_geom_x,
        "observation_qpos_geometry_error_m": abs(
            observation_qpos - observation_geom_x
        ),
        "active_contact_count": int(wp.to_torch(live.nacon)[0]),
        "maximum_constraint_force_component": float(torch.max(torch.abs(efc_before))),
        "live_fields_unchanged_by_observation_refresh": unchanged,
        "all_live_fields_unchanged": all(unchanged.values()),
        "rollback_buffer_unchanged": rollback_unchanged,
        "rollback_qpos_after_refresh": float(rollback_after[0, 0]),
        "cpu_warp_post_state_geom_max_error_m": cpu_geom_error,
        "cpu_warp_post_state_site_max_error_m": cpu_site_error,
    }
    arrays = {
        "live_qpos_before_refresh": live_before["qpos"].numpy(),
        "live_qpos_after_refresh": live_after["qpos"].numpy(),
        "live_geom_xpos_before_refresh": live_before["geom_xpos"].numpy(),
        "live_geom_xpos_after_refresh": live_after["geom_xpos"].numpy(),
        "observation_qpos_after_refresh": wp.to_torch(observation.qpos).numpy(),
        "observation_geom_xpos_after_refresh": wp.to_torch(
            observation.geom_xpos
        ).numpy(),
        "live_contact_dist_before_refresh": contact_before["dist"].numpy(),
        "live_contact_dist_after_refresh": contact_after["dist"].numpy(),
        "live_efc_force_before_refresh": efc_before.numpy(),
        "live_efc_force_after_refresh": efc_after.numpy(),
        "rollback_qpos_before_refresh": rollback_before.numpy(),
        "rollback_qpos_after_refresh": rollback_after.numpy(),
        "cpu_post_geom_xpos": cpu_post.geom_xpos.copy(),
    }
    return results, arrays


def run_graph_capture_probe() -> dict[str, Any]:
    cuda_devices = [device for device in wp.get_devices() if device.is_cuda]
    if not cuda_devices:
        return {"cuda_available": False, "capture_time_unchanged": None}
    device = cuda_devices[0]
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body>
              <joint type="slide" axis="1 0 0"/>
              <geom type="sphere" size=".05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice(device):
        model_wp = mjwarp.put_model(model)
        data_wp = mjwarp.put_data(model, data, nworld=1, nconmax=16, njmax=32)
        time_before = float(data_wp.time.numpy()[0])
        qpos_before = float(data_wp.qpos.numpy()[0, 0])
        graph = mjwp._compile_step(model_wp, data_wp)
        time_after_capture = float(data_wp.time.numpy()[0])
        qpos_after_capture = float(data_wp.qpos.numpy()[0, 0])
        if graph is None:
            raise RuntimeError("CUDA graph capture unexpectedly returned no graph")
        wp.capture_launch(graph)
        wp.synchronize()
        time_after_launch = float(data_wp.time.numpy()[0])
        qpos_after_launch = float(data_wp.qpos.numpy()[0, 0])
    return {
        "cuda_available": True,
        "device": str(device),
        "time_before_capture_s": time_before,
        "time_after_capture_s": time_after_capture,
        "time_after_first_launch_s": time_after_launch,
        "qpos_before_capture": qpos_before,
        "qpos_after_capture": qpos_after_capture,
        "qpos_after_first_launch": qpos_after_launch,
        "capture_time_unchanged": time_before == time_after_capture,
        "capture_qpos_unchanged": qpos_before == qpos_after_capture,
        "first_launch_advanced_one_timestep": bool(
            np.isclose(time_after_launch - time_after_capture, model.opt.timestep)
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    step_source = function_source(BACKEND, "step_env")
    refresh_source = function_source(BACKEND, "refresh_post_state_kinematics")
    build_source = function_source(ENVIRONMENT, "_build_observations")
    contacts_source = function_source(ENVIRONMENT, "_last_step_contact_features")
    environment_step_source = function_source(ENVIRONMENT, "step")
    restore_source = function_source(ENVIRONMENT, "set_env_state")
    reset_source = function_source(ENVIRONMENT, "_write_state")

    checks = {
        "普通物理步后刷新独立观测缓冲区": (
            "refresh_post_state_kinematics(env)" in step_source
        ),
        "普通物理步没有对正式缓冲区额外执行完整前向计算": (
            "mjwarp.forward(env.model_wp, env.data_wp)" not in step_source
        ),
        "独立缓冲区执行步后前向计算": (
            "mjwarp.forward(env.model_wp, env.data_wp_observation)" in refresh_source
        ),
        "观测刷新拒绝复用正式或回滚缓冲区": (
            "env.data_wp_observation is env.data_wp" in refresh_source
            and "env.data_wp_observation is env.data_wp_prev" in refresh_source
        ),
        "关节状态和手部位姿来自步后观测缓冲区": (
            "post_state = self.env.data_wp_observation" in build_source
            and "site_xpos = wp.to_torch(post_state.site_xpos)" in build_source
        ),
        "接触和执行器力保留刚完成区间的求解结果": (
            "contact = self.env.data_wp.contact" in contacts_source
            and "efc = self.env.data_wp.efc" in contacts_source
            and "self.env.data_wp.qfrc_actuator" in build_source
        ),
        "子步回调显式提供状态时刻和力区间": all(
            text in environment_step_source
            for text in (
                "state_time_s=state_time_s",
                "force_interval_start_time_s=force_interval_start_time_s",
                "force_interval_end_time_s=force_interval_end_time_s",
                "post_state=self.env.data_wp_observation",
                "last_step_dynamics=self.env.data_wp",
            )
        ),
        "快照恢复后重建观测缓冲区": (
            "self._mjwp.refresh_post_state_kinematics(self.env)" in restore_source
        ),
        "正式缓冲区完整前向计算只保留在状态重置路径": (
            "mjwarp.forward(self.env.model_wp, self.env.data_wp)" in reset_source
            and ENVIRONMENT.read_text(encoding="utf-8").count(
                "mjwarp.forward(self.env.model_wp, self.env.data_wp)"
            )
            == 1
        ),
        "启动规划日志按两类数据源读取": (
            "observation.last_step_dynamics" in STARTUP.read_text(encoding="utf-8")
            and "world.env.data_wp_observation" in STARTUP.read_text(encoding="utf-8")
        ),
        "子步跟踪日志按两类数据源读取": (
            "observation.post_state" in TRACE.read_text(encoding="utf-8")
            and "observation.last_step_dynamics" in TRACE.read_text(encoding="utf-8")
        ),
        "训练安全观察器按两类数据源读取": (
            "observation.post_state" in FALLBACK.read_text(encoding="utf-8")
            and "observation.last_step_dynamics" in FALLBACK.read_text(encoding="utf-8")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"source audit failed: {failed}")

    probe, arrays = run_probe()
    graph_probe = run_graph_capture_probe()
    if not probe["all_live_fields_unchanged"]:
        raise RuntimeError("observation refresh changed live solver data")
    if not probe["rollback_buffer_unchanged"]:
        raise RuntimeError("observation refresh changed rollback data")
    if probe["observation_qpos_geometry_error_m"] != 0.0:
        raise RuntimeError("post-state geometry is not aligned with post-state qpos")
    if probe["cpu_warp_post_state_geom_max_error_m"] >= 1.0e-6:
        raise RuntimeError("CPU and Warp post-state geometry differ")
    if graph_probe["cuda_available"] and not all(
        graph_probe[name]
        for name in (
            "capture_time_unchanged",
            "capture_qpos_unchanged",
            "first_launch_advanced_one_timestep",
        )
    ):
        raise RuntimeError("CUDA graph capture changed the initial physics state")

    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "probe_arrays.npz", **arrays)
    result = {
        "schema": "taco_brush_20231006_145_mjwp_post_state_timing_v2",
        "sample": "20231006_145",
        "repository_head_before_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_files": {
            str(path.relative_to(REPO_ROOT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (BACKEND, ENVIRONMENT, STARTUP, TRACE, FALLBACK)
        },
        "source_checks": checks,
        "source_lines": {
            "independent_buffer_allocation": line_of(
                BACKEND, "data_wp_observation = mjwarp.put_data("
            ),
            "independent_forward": line_of(
                BACKEND, "mjwarp.forward(env.model_wp, env.data_wp_observation)"
            ),
            "post_state_site_read": line_of(
                ENVIRONMENT, "site_xpos = wp.to_torch(post_state.site_xpos)"
            ),
            "last_step_contact_read": line_of(
                ENVIRONMENT, "contact = self.env.data_wp.contact"
            ),
            "substep_time_view": line_of(
                ENVIRONMENT, "MJWPSubstepObservation("
            ),
        },
        "synthetic_runtime_probe": probe,
        "cuda_graph_capture_probe": graph_probe,
        "sample_level_validation": {
            "updated_observation_replay_executed": False,
            "checkpoint_compatibility_checked": False,
            "cold_replay_executed": False,
            "sample_cpu_warp_comparison_executed": False,
            "training_chain_regression_executed": False,
            "checkpoint_compatibility": "未验证",
        },
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# 20231006_145 MJWP 时间同步代码检查

## 检查范围

本记录只检查当前 MJWP 低层代码中“步后状态”和“刚完成物理区间的力”是否分开保存和读取。没有运行样本级 Replay、旧 checkpoint、冷启动 Replay 或训练。

## 当前数据含义

- `data_wp` 是正式物理缓冲区。`mjwarp.step` 后，其中的接触、约束力和执行器力保留为刚完成物理区间的求解结果。
- `data_wp_observation` 是独立观测缓冲区。它复制正式缓冲区的当前积分状态，再在自己的缓冲区中执行前向计算；手掌、指尖、刚体和几何位置都从这里读取。
- `data_wp_prev` 只用于回滚。观测刷新不读取或覆盖它。
- 子步观察器收到三个明确时间量：步后状态时刻、力区间起点、力区间终点。步后状态时刻等于力区间终点。
- 状态重置会在正式缓冲区执行一次完整前向计算，用来建立新的初始求解状态；普通物理步后没有这项操作。
- CUDA 图捕获只记录物理步，不推进初始状态；第一次图启动才推进一个物理步。

## 运行时探针

- 物理步长：`{probe['physics_timestep_s']:.6f} s`
- 步后关节位置：`{probe['live_qpos_after_step']:.9f} m`
- 正式缓冲区中尚未更新的几何 x：`{probe['live_geom_cache_x_after_step']:.9f} m`
- 独立观测缓冲区的几何 x：`{probe['observation_geom_x_after_refresh']:.9f} m`
- 观测缓冲区中关节位置与几何位置之差：`{probe['observation_qpos_geometry_error_m']:.9g} m`
- 探针中的有效接触数：`{probe['active_contact_count']}`
- 探针中的最大约束力分量：`{probe['maximum_constraint_force_component']:.6f}`
- 刷新前后正式物理缓冲区逐项完全相同：`{'是' if probe['all_live_fields_unchanged'] else '否'}`
- 回滚缓冲区保持不变：`{'是' if probe['rollback_buffer_unchanged'] else '否'}`
- 独立观测几何与 CPU MuJoCo 前向结果的最大差：`{probe['cpu_warp_post_state_geom_max_error_m']:.9g} m`
- CUDA 图捕获前后的时间都为：`{graph_probe.get('time_after_capture_s', float('nan')):.6f} s`
- CUDA 图第一次启动后的时间：`{graph_probe.get('time_after_first_launch_s', float('nan')):.6f} s`

## 尚未覆盖的验证

- 使用新观测含义重新运行 `20231006_145`：未执行。
- 旧 checkpoint 与新观测含义的兼容性：未验证。
- 新语义下的冷启动 Replay：未执行。
- `20231006_145` 完整场景的 CPU 与 Warp 数值对照：未执行。
- 训练链回归：未执行。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    command = (
        'PYTHONPATH="/data_all/zzx/3.2RL/.env_mjwp313_overlay:$PWD/src:'
        '$PWD/scripts:$PWD/external/mink/src:$PWD/external/human2sim2robot:'
        '$PWD/external/spider_compat" OMP_NUM_THREADS=4 '
        "/data_all/zzx/egoengine/spider/.venv/bin/python "
        "scripts/audit_taco_brush_20231006_145_mjwp_post_state_timing_v2.py"
    )
    (OUTPUT / "validation_command.txt").write_text(command + "\n", encoding="utf-8")

    artifacts = [OUTPUT / name for name in ("probe_arrays.npz", "report.md", "results.json")]
    (OUTPUT / "artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
