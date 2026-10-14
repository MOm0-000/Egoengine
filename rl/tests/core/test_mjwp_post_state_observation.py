from __future__ import annotations

import ast
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from spider.simulators import mjwp


ROOT = Path(__file__).resolve().parents[2]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_step_refreshes_only_the_independent_observation_buffer() -> None:
    path = ROOT / "external/spider_compat/spider/simulators/mjwp.py"
    step = _function(path, "step_env")
    calls = [ast.unparse(node) for node in ast.walk(step) if isinstance(node, ast.Call)]
    assert "refresh_post_state_kinematics(env)" in calls
    assert "mjwarp.forward(env.model_wp, env.data_wp)" not in calls


def test_observation_and_last_step_dynamics_have_separate_read_sources() -> None:
    path = ROOT / "src/video_to_spider/rl/mjwp_env.py"
    build = ast.unparse(_function(path, "_build_observations"))
    contacts = ast.unparse(_function(path, "_last_step_contact_features"))
    assert "post_state = self.env.data_wp_observation" in build
    assert "site_xpos = wp.to_torch(post_state.site_xpos)" in build
    assert "self.env.data_wp.qfrc_actuator" in build
    assert "contact = self.env.data_wp.contact" in contacts
    assert "efc = self.env.data_wp.efc" in contacts


def test_independent_forward_aligns_geometry_without_mutating_live_dynamics() -> None:
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
    assert int(wp.to_torch(live.nacon)[0]) == 1
    assert bool(torch.any(wp.to_torch(live.efc.force) != 0.0))
    live_fields = {
        name: wp.to_torch(getattr(live, name)).clone()
        for name in (
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
    }
    contact_fields = {
        name: wp.to_torch(getattr(live.contact, name)).clone()
        for name in ("dist", "geom", "worldid", "efc_address")
    }
    efc_force = wp.to_torch(live.efc.force).clone()

    mjwp.refresh_post_state_kinematics(env)
    wp.synchronize()

    cpu_post = mujoco.MjData(model)
    cpu_post.qpos[:] = wp.to_torch(observation.qpos)[0].numpy()
    cpu_post.qvel[:] = wp.to_torch(observation.qvel)[0].numpy()
    mujoco.mj_forward(model, cpu_post)

    assert float(wp.to_torch(live.qpos)[0, 0]) == float(
        wp.to_torch(observation.qpos)[0, 0]
    )
    assert float(wp.to_torch(observation.geom_xpos)[0, 1, 0]) == float(
        wp.to_torch(observation.qpos)[0, 0]
    )
    assert float(wp.to_torch(live.geom_xpos)[0, 1, 0]) == 0.0
    assert float(wp.to_torch(rollback.qpos)[0, 0]) == 7.0
    assert np.max(
        np.abs(wp.to_torch(observation.geom_xpos)[0].numpy() - cpu_post.geom_xpos)
    ) < 1.0e-6
    for name, expected in live_fields.items():
        assert torch.equal(wp.to_torch(getattr(live, name)), expected), name
    for name, expected in contact_fields.items():
        assert torch.equal(wp.to_torch(getattr(live.contact, name)), expected), name
    assert torch.equal(wp.to_torch(live.efc.force), efc_force)
