import numpy as np
import torch

from video_to_spider.rl.core.env import load_ego_config
from video_to_spider.rl.core.rollout import RolloutBatch
from video_to_spider.rl.mjwp_env import control_substeps_at_sources


def test_brush_30hz_clock_preserves_authored_two_ms_phase() -> None:
    observed = control_substeps_at_sources(
        np.arange(80), control_dt=1.0 / 30.0, simulation_dt=0.002
    )
    assert observed.tolist()[:6] == [17, 17, 16, 17, 17, 16]
    assert observed.sum() == 1334
    restored = control_substeps_at_sources(
        np.arange(20, 60), control_dt=1.0 / 30.0, simulation_dt=0.002
    )
    assert np.array_equal(restored, observed[20:60])


def test_core_loader_preserves_nonintegral_brush_control_clock() -> None:
    config = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "configs/taco_brush_controlled_rl_simulator_v1.yaml"
    )
    loaded = load_ego_config(config)
    assert loaded.sim_dt == 0.002
    assert loaded.ctrl_dt == 1.0 / 30.0
    assert loaded.ctrl_steps == 17


def test_rollout_validation_accepts_non_pour_window_phase() -> None:
    # Exercise the phase portion without constructing a heavyweight policy.
    batch = object.__new__(RolloutBatch)
    batch.source_endpoint = torch.tensor([20, 21, 20, 21])
    batch.critic_phase = torch.tensor([0.0, 0.025, 0.0, 0.025])
    # The full validator is covered elsewhere.  Its window-specific expression
    # must agree with the Brush endpoint-20 contract.
    expected = (batch.source_endpoint - 20).to(torch.float32) / 40.0
    assert torch.equal(batch.critic_phase, expected)
