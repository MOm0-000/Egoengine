# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the SPIDER MuJoCo+Warp MPC loop with EgoEngine 3.2.2 mode switching.

This is a paper-aligned variant of ``examples/run_mjwp.py``.  It keeps the same
setup and receding-horizon MPC loop, but at every control tick it first checks
whether deterministic Replay of the reference controls already stays inside the
object-tracking feasibility boundary (paper Eq. 2).  Only when Replay fails does
it run the sampling MPC optimizer, and only when MPC is also insufficient does
it attempt RL.  RL is a stub here and raises ``NotImplementedError`` when
``use_rl_reward`` is enabled without an injected policy.

The decision logic is imported from ``video_to_spider.export.egoengine_mode_switch``
and does not modify the cloned SPIDER checkout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path


# Make the surrounding ``video_to_spider`` package importable regardless of the
# working directory.  This script normally lives at ``video_to_spider/scripts/``,
# so the package root is one level up; an explicit override is honoured when the
# script is materialized elsewhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
_V2S_ROOT = Path(os.environ.get("VIDEO_TO_SPIDER_ROOT", _SCRIPT_DIR.parent)).resolve()
if str(_V2S_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2S_ROOT))

_H2S2R_ROOT = Path(
    os.environ.get("H2S2R_ROOT", _V2S_ROOT.parent / "reference" / "human2sim2robot")
).resolve()
if str(_H2S2R_ROOT) not in sys.path:
    sys.path.insert(0, str(_H2S2R_ROOT))

# Hydra resolves ``config_path`` relative to this file by default.  SPIDER's
# config lives next to ``examples/run_mjwp.py``; default there and allow an
# explicit ``SPIDER_CONFIG_DIR`` override for materialized copies.
_SPIDER_CONFIG_DIR = Path(
    os.environ.get(
        "SPIDER_CONFIG_DIR",
        _V2S_ROOT.parent / "spider" / "examples" / "config",
    )
).resolve()

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    _initial_state_sanity_check,
    compute_contact_point_delta,
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,
    step_env,
    sync_env,
)
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)

from video_to_spider.export.egoengine_mode_switch import (
    SOLVER_MPC,
    SOLVER_REPLAY,
    SOLVER_RL,
    ModeSwitchConfig,
    feasibility_constant,
    is_feasible,
)
from video_to_spider.rl.h2s2r import make_anchor_points, XHandResidualPolicySpec
from video_to_spider.rl.mjwp_env import (
    _object_pose_from_qpos,
    _transform_anchors_torch,
    _XHAND_FINGERTIP_SITE_IDS,
    _XHAND_PALM_SITE_ID,
)
from video_to_spider.rl.residual_policy import ResidualPolicy, load_residual_policy

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}


def _parse_override_tokens(tokens: list[str]) -> dict:
    allowed = {field.name for field in fields(Config)}
    override_dict: dict = {}
    for item in tokens:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        if key not in allowed:
            continue
        parsed = OmegaConf.to_container(
            OmegaConf.from_dotlist([f"{key}={value}"]), resolve=True
        )
        if isinstance(parsed, dict) and key in parsed:
            override_dict[key] = parsed[key]
    return override_dict


def _extract_cli_overrides(cfg: DictConfig) -> dict:
    overrides = OmegaConf.select(cfg, "hydra.overrides.task") or []
    override_dict = _parse_override_tokens(overrides)
    if override_dict:
        return override_dict
    return _parse_override_tokens(sys.argv[1:])


def _assert_object_actuator_gains_zero(env, config: Config, stage: str, atol: float = 1e-4) -> None:
    if not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol), f"Object actuator Kp not near zero at {stage}"
    assert np.allclose(kd, 0.0, atol=atol), f"Object actuator Kd not near zero at {stage}"


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return "" if name is None else str(name)


def _is_object_collision_name(name: str, side: str) -> bool:
    """Return True for the right/left object collision geoms, not the visual mesh."""
    prefix = f"{side}_object"
    return name.startswith(prefix) and not name.endswith("_visual")


def _object_has_environment_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    min_force_n: float = 0.05,
) -> bool:
    """Use the current CPU MuJoCo contacts to detect floor/table support."""
    support_force = 0.0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        first_name = _geom_name(model, first)
        second_name = _geom_name(model, second)
        first_object = _is_object_collision_name(first_name, side)
        second_object = _is_object_collision_name(second_name, side)
        if first_object == second_object:
            continue
        object_geom = first if first_object else second
        other_geom = second if first_object else first
        other_name = second_name if first_object else first_name
        other_body = int(model.geom_bodyid[other_geom])
        other_is_fixed_environment = bool(
            other_name == "floor"
            and int(model.body_jntnum[other_body]) == 0
        )
        if not other_is_fixed_environment:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        support_force += max(float(wrench[0]), 0.0)
    return support_force >= min_force_n


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = Path(config.output_dir) / f"config{'_act' if config.contact_guidance else ''}.yaml"
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = int(config.object_action_dims) if config.object_action_dims > 0 else 12
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    return list(range(0, half)), list(range(half, robot_nu))


def _apply_noise_mask(base_noise_scale: torch.Tensor, zero_indices: list[int]) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(zero_indices, device=base_noise_scale.device, dtype=torch.long)
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def _mode_switch_config(config: Config) -> ModeSwitchConfig:
    lookahead = max(0, int(config.horizon_steps // max(1, int(config.ctrl_steps))) - 1)
    return ModeSwitchConfig(
        chunk_steps=max(1, int(config.ctrl_steps)),
        lookahead_chunks=lookahead,
        object_pos_threshold_m=float(config.object_pos_threshold),
        object_rot_threshold_rad=float(config.object_rot_threshold),
        lambda_pos=float(config.pos_rew_scale),
        lambda_rot=float(config.rot_rew_scale),
        rl_enabled=bool(config.use_rl_reward),
    )


def _replay_feasible(rollout, config: Config, env, ref_slice, mode_switch_config: ModeSwitchConfig):
    """Check whether deterministic Replay of the reference controls is feasible."""
    replay_ctrls = ref_slice[2]
    if replay_ctrls.shape[0] != config.horizon_steps:
        return False, None
    # MJWP environments are materialized with ``config.num_samples`` worlds.
    # Evaluate Replay with every world running the same reference controls so
    # the rollout stays batch-consistent with the fixed world count.  This is
    # one parallel rollout, which is much cheaper than the full MPC iteration
    # count it can skip.
    replay_batch = replay_ctrls.repeat(int(config.num_samples), 1, 1)
    _, _, terminate, rollout_info = rollout(config, env, replay_batch, ref_slice, {})
    error = rollout_info.get("object_tracking_error")
    if isinstance(error, torch.Tensor):
        error = error.detach().cpu().numpy()
    else:
        error = np.asarray(error if error is not None else [float("inf")])
    if isinstance(terminate, torch.Tensor):
        terminate = terminate.detach().cpu().numpy()
    else:
        terminate = np.asarray(terminate)
    feasible = bool(not bool(terminate.any())) and is_feasible(
        float(error.max()), feasibility_constant(mode_switch_config)
    )
    return feasible, float(error.max())


def _rl_obs_state(
    config: Config,
    env,
    qpos_ref: torch.Tensor,
    sim_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the single-world actor obs and privileged state for RL inference."""
    qpos = get_qpos(config, env)[0]
    qvel = get_qvel(config, env)[0]
    site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]
    anchors = torch.as_tensor(
        make_anchor_points(0.2), dtype=torch.float32, device=config.device
    )

    current_object = _object_pose_from_qpos(qpos.unsqueeze(0), int(config.nq_obj))
    goal_qpos = qpos_ref[min(int(sim_step), qpos_ref.shape[0] - 1)].unsqueeze(0)
    goal_object = _object_pose_from_qpos(goal_qpos, int(config.nq_obj))
    current_anchors = _transform_anchors_torch(current_object[0], current_object[1], anchors)
    goal_anchors = _transform_anchors_torch(goal_object[0], goal_object[1], anchors)

    fingertip = site_xpos[list(_XHAND_FINGERTIP_SITE_IDS)]
    palm = site_xpos[_XHAND_PALM_SITE_ID]
    hand_qpos = qpos[:18]
    hand_qvel = qvel[:18]
    observation = torch.cat(
        [
            hand_qpos,
            hand_qvel,
            fingertip.reshape(-1),
            palm,
            current_anchors.reshape(-1),
            goal_anchors.reshape(-1),
        ],
        dim=0,
    )

    object_linear = qvel[-6:-3]
    object_angular = qvel[-3:]
    joint_forces = torch.zeros(18, device=config.device)
    try:
        qfrc = wp.to_torch(env.data_wp.qfrc_actuator)[0]
        joint_forces = qfrc[:18].clone()
    except AttributeError:
        pass
    contact_forces = torch.zeros(len(_XHAND_FINGERTIP_SITE_IDS), 3, device=config.device)
    state = torch.cat(
        [object_linear, object_angular, joint_forces, contact_forces.reshape(-1)],
        dim=0,
    )
    return observation.detach().cpu().numpy(), state.detach().cpu().numpy()


def _rl_solver_factory(config: Config, qpos_ref: torch.Tensor) -> callable | None:
    """Return an RL residual solver, or ``None`` when no checkpoint is configured."""
    checkpoint = os.environ.get("MJWP_RL_CHECKPOINT", "")
    metadata = os.environ.get("MJWP_RL_TRAIN_METADATA", "")
    if not checkpoint:
        return None
    if not metadata:
        metadata = str(Path(checkpoint).resolve().parent / "train_metadata.json")
    policy: ResidualPolicy = load_residual_policy(
        checkpoint_path=checkpoint,
        train_metadata_path=metadata,
        device=config.device,
        deterministic=True,
    )
    residual_spec = XHandResidualPolicySpec()

    def solve(env, sim_step: int, ref_ctrls: torch.Tensor) -> torch.Tensor:
        obs, state = _rl_obs_state(config, env, qpos_ref, sim_step)
        delta = policy.predict_delta(obs, state)
        delta_t = torch.as_tensor(delta, dtype=torch.float32, device=config.device)
        delta_t = torch.clamp(
            residual_spec.residual_scale * delta_t,
            -residual_spec.residual_clip,
            residual_spec.residual_clip,
        )
        ctrls = ref_ctrls.clone()
        if ctrls.dim() == 2 and ctrls.shape[0] < int(config.horizon_steps):
            # Near the end of a short trajectory ``get_slice`` returns fewer
            # than ``horizon_steps`` rows.  SPIDER's sampling optimizer still
            # expects a full horizon control tensor, so pad the reference with
            # its final row; the trailing controls are never stepped because
            # the loop stops at ``max_sim_steps``.
            pad = ctrls[-1:].repeat(int(config.horizon_steps) - ctrls.shape[0], 1)
            ctrls = torch.cat([ctrls, pad], dim=0)
        elif ctrls.dim() == 3 and ctrls.shape[1] < int(config.horizon_steps):
            pad = ctrls[:, -1:].repeat(1, int(config.horizon_steps) - ctrls.shape[1], 1)
            ctrls = torch.cat([ctrls, pad], dim=1)
        hand_indices = list(range(min(18, ctrls.shape[-1])))
        ctrls[..., hand_indices] += delta_t[: len(hand_indices)]
        return ctrls

    return solve


def main(config: Config):
    """Run the SPIDER MuJoCo+Warp loop with Replay -> MPC -> RL switching."""
    conditional_contact_guidance = os.environ.get(
        "MJWP_CONDITIONAL_CONTACT_GUIDANCE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    conditional_min_support_force_n = float(
        os.environ.get("MJWP_CONDITIONAL_CONTACT_MIN_SUPPORT_FORCE_N", "0.05")
    )
    conditional_max_delta_m = float(
        os.environ.get("MJWP_CONDITIONAL_CONTACT_MAX_DELTA_M", "0.01")
    )
    config = process_config(config)
    if conditional_contact_guidance and config.contact_guidance:
        raise ValueError(
            "MJWP_CONDITIONAL_CONTACT_GUIDANCE must be used with contact_guidance=false; "
            "it injects contact deltas into the free-joint hand controls instead of "
            "re-enabling the object-actuator guidance scene."
        )
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning("contact_guidance requires improvement_threshold <= 0; overriding to 0.0.")
        config.improvement_threshold = 0.0

    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(config, config.data_path)
    if (
        config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        loguru.logger.info("Using qpos as ctrl reference for contact guidance.")
        ctrl_ref = qpos_ref[:, : config.nu]
    if config.contact_guidance and torch.all(contact <= 0):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    if conditional_contact_guidance and torch.all(contact <= 0):
        raise ValueError("conditional contact guidance is enabled, but contact mask is all zeros.")
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    env = setup_env(config, ref_data)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    _assert_object_actuator_gains_zero(env, config, "start")

    if config.sanity_check_seconds > 0.0:
        _initial_state_sanity_check(
            mj_model, mj_data, qpos_ref, qvel_ref, ctrl_ref, config,
            save_video_dir=config.output_dir,
        )
        mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
        mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
        mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
        mujoco.mj_step(mj_model, mj_data)
        mj_data.time = 0.0

    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    contact_guidance_enabled = config.contact_guidance and len(config.object_actuator_ids) > 0
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning("contact_guidance is enabled but no object actuators were resolved.")
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(min(contact.shape[1], contact_pos.shape[1], len(config.contact_order)))
        if config.contact_len != len(config.contact_order) or config.contact_len != contact.shape[1]:
            loguru.logger.warning("Contact length mismatch; truncating to {}.".format(config.contact_len))
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[: config.contact_len]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)
    elif conditional_contact_guidance:
        if config.contact_len <= 0:
            config.contact_len = int(
                min(
                    contact.shape[1],
                    contact_pos.shape[1],
                    len(config.contact_order or [None] * contact.shape[1]),
                    len(config.hand_contact_site_ids or [None] * contact.shape[1]),
                )
            )
        if config.contact_len <= 0:
            raise ValueError("conditional contact guidance requires contact fields and site ids")
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[: config.contact_len]
        if not config.right_contact_indices:
            config.right_contact_indices = [
                idx
                for idx, (side, finger) in enumerate(config.contact_order)
                if side == "right"
            ]
        if not config.right_pos_ctrl_ids:
            config.right_pos_ctrl_ids = [0, 1, 2]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr)
        pair_margin_list = np.linspace(config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr)
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array([
            config.init_rot_actuator_gain if ("_rot_" in (name or "")) else config.init_pos_actuator_gain
            for name in actuator_names
        ], dtype=np.float32)
        base_kd = np.array([
            config.init_rot_actuator_bias if ("_rot_" in (name or "")) else config.init_pos_actuator_bias
            for name in actuator_names
        ], dtype=np.float32)
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                kp_i = np.zeros_like(base_kp, dtype=np.float32)
                kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {"xy_offset": xy_offset_list[j], "pair_margin": pair_margin_list[j]}
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    rollout = make_rollout_fn(
        step_env, save_state, load_state, get_reward, get_terminal_reward,
        get_terminate, get_trace, save_env_params, load_env_params, copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)
    base_noise_scale = config.noise_scale.clone()
    gibbs_enabled = config.gibbs_sampling and config.embodiment_type == "bimanual"
    if config.gibbs_sampling and not gibbs_enabled:
        loguru.logger.warning("gibbs_sampling is enabled but embodiment_type is {}, disabling.".format(config.embodiment_type))
    if gibbs_enabled:
        right_ids, left_ids = _get_bimanual_hand_indices(config)
        right_only_zero = left_ids
        left_only_zero = right_ids

    mode_switch_config = _mode_switch_config(config)
    feasibility_c = feasibility_constant(mode_switch_config)
    rl_solver = _rl_solver_factory(config, qpos_ref)
    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    mode_decisions = []
    conditional_contact_decisions = []

    t_start = time.perf_counter()
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            sim_step = int(np.round(mj_data.time / config.sim_dt))
            ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
            ref_slice = tuple(
                (
                    torch.cat(
                        [
                            x,
                            x[-1:].repeat(
                                int(config.horizon_steps) - x.shape[0],
                                *([1] * max(0, x.dim() - 1)),
                            ),
                        ],
                        dim=0,
                    )
                    if x.dim() >= 1 and x.shape[0] < int(config.horizon_steps)
                    else x
                )
                for x in ref_slice
            )
            if ctrls.dim() == 2 and ctrls.shape[0] < int(config.horizon_steps):
                pad = ctrls[-1:].repeat(int(config.horizon_steps) - ctrls.shape[0], 1)
                ctrls = torch.cat([ctrls, pad], dim=0)
            elif ctrls.dim() == 3 and ctrls.shape[1] < int(config.horizon_steps):
                pad = ctrls[:, -1:].repeat(1, int(config.horizon_steps) - ctrls.shape[1], 1)
                ctrls = torch.cat([ctrls, pad], dim=1)
            ctrls_for_opt = ctrls
            contact_injection_enabled = contact_guidance_enabled or conditional_contact_guidance
            conditional_support = True
            conditional_delta_norm = 0.0
            conditional_applied = False
            if contact_injection_enabled and config.contact_len > 0:
                contact_mask_step = contact[sim_step][contact_offset : contact_offset + config.contact_len]
                contact_pos_ref_step = contact_pos[sim_step]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]
                right_delta = compute_contact_point_delta(
                    contact_mask_step, contact_pos_ref_step, site_xpos,
                    config.hand_contact_site_ids, config.right_contact_indices,
                )
                left_delta = compute_contact_point_delta(
                    contact_mask_step, contact_pos_ref_step, site_xpos,
                    config.hand_contact_site_ids, config.left_contact_indices,
                )
                if conditional_contact_guidance:
                    conditional_support = _object_has_environment_support(
                        mj_model, mj_data, "right", conditional_min_support_force_n
                    )
                    apply_contact_delta = not conditional_support
                else:
                    apply_contact_delta = True

                ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                if apply_contact_delta and right_delta is not None and config.right_pos_ctrl_ids and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]:
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                    clipped_delta = torch.clip(
                        right_delta,
                        -conditional_max_delta_m if conditional_contact_guidance else -0.01,
                        conditional_max_delta_m if conditional_contact_guidance else 0.01,
                    )
                    conditional_delta_norm = float(torch.norm(clipped_delta).detach().cpu())
                    conditional_applied = True
                    if conditional_contact_guidance:
                        ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[:, config.right_pos_ctrl_ids] - clipped_delta
                    else:
                        ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[:, config.right_pos_ctrl_ids] + clipped_delta
                if left_delta is not None and config.left_pos_ctrl_ids and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]:
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                    ctrls_for_opt[:, config.left_pos_ctrl_ids] = ref_ctrl_slice[:, config.left_pos_ctrl_ids] + torch.clip(left_delta, -0.01, 0.01)
            if conditional_contact_guidance:
                conditional_contact_decisions.append({
                    "sim_step": int(sim_step),
                    "supported": bool(conditional_support),
                    "applied": bool(conditional_applied),
                    "delta_norm_m": conditional_delta_norm,
                })

            replay_ok, replay_error = _replay_feasible(
                rollout, config, env, ref_slice, mode_switch_config
            )
            mpc_feasible = False
            rl_feasible = False
            tick_object_error = float("inf")
            if replay_ok:
                ctrls = ref_slice[2]
                mode = SOLVER_REPLAY
                tick_object_error = float(replay_error)
                infos = {"opt_steps": np.asarray([0])}
            else:
                if gibbs_enabled:
                    config.noise_scale = _apply_noise_mask(base_noise_scale, right_only_zero)
                    ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                    config.noise_scale = _apply_noise_mask(base_noise_scale, left_only_zero)
                    ctrls, infos = optimize(config, env, ctrls, ref_slice)
                    config.noise_scale = base_noise_scale
                else:
                    config.noise_scale = base_noise_scale
                    ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                mode = SOLVER_MPC
                mpc_error = float(np.asarray(infos.get("object_tracking_error_max", np.asarray([float("inf")]))).max())
                mpc_feasible = is_feasible(mpc_error, feasibility_c)
                tick_object_error = float(mpc_error)
                if not mpc_feasible and mode_switch_config.rl_enabled:
                    if rl_solver is None:
                        raise NotImplementedError(
                            "RL residual policy is enabled (use_rl_reward=True) but no "
                            "MJWP_RL_CHECKPOINT policy was injected into run_mjwp_modeswitch."
                        )
                    ctrls = rl_solver(env, sim_step, ref_slice[2])
                    mode = SOLVER_RL
                    rl_feasible = True
                    tick_object_error = float(mpc_error)

            if len(config.trace_site_ids) > 0:
                trace_ref = []
                qpos_ref_horizon = ref_slice[0]
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    site_xpos = np.array([mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids])
                    trace_ref.append(site_xpos)
                trace_ref_np = np.stack(trace_ref, axis=0)[None, None, :, :, :]
                infos["trace_ref"] = trace_ref_np

            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]
                step_env(config, env, ctrl_step)
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                        image = render_image(config, renderer, mj_model, mj_data, mj_data_ref)
                        images.append(image)
                if "rerun" in config.viewer or "viser" in config.viewer:
                    mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    log_frame(
                        mj_data, sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                    )
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            sync_env(config, env, mj_data)

            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step + prev_ctrl.shape[0] + config.ctrl_steps
            ]
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}, mode: {mode}",
                end="\r",
            )

            info_list.append({
                "qpos": step_info["qpos"],
                "qvel": step_info["qvel"],
                "time": step_info["time"],
                "ctrl": step_info["ctrl"],
                "opt_steps": np.asarray([int(infos["opt_steps"][0])]),
                "object_tracking_error_max": np.asarray([tick_object_error]),
            })
            mode_decisions.append({
                "sim_step": int(sim_step),
                "mode": mode,
                "replay_feasible": bool(replay_ok),
                "mpc_feasible": bool(mpc_feasible),
                "rl_feasible": bool(rl_feasible),
                "object_tracking_error": float(tick_object_error),
            })

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"Total time: {t_end - t_start:.4f}s")

    if config.save_info and len(info_list) > 0:
        all_keys = set().union(*(info.keys() for info in info_list))
        info_aggregated = {}
        for k in all_keys:
            values = [info[k] for info in info_list if k in info]
            if values:
                info_aggregated[k] = np.stack(values, axis=0)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz")

    mode_report = {
        "schema_version": "1.0",
        "solver_order": [SOLVER_REPLAY, SOLVER_MPC, SOLVER_RL],
        "rl_enabled": bool(mode_switch_config.rl_enabled),
        "feasibility_constant": feasibility_c,
        "chunk_steps": mode_switch_config.chunk_steps,
        "lookahead_chunks": mode_switch_config.lookahead_chunks,
        "decisions": mode_decisions,
    }
    if conditional_contact_guidance:
        mode_report["conditional_contact_guidance"] = {
            "min_support_force_n": conditional_min_support_force_n,
            "max_delta_m": conditional_max_delta_m,
            "applied_frame_count": int(
                sum(1 for item in conditional_contact_decisions if item["applied"])
            ),
            "unsupported_frame_count": int(
                sum(1 for item in conditional_contact_decisions if not item["supported"])
            ),
            "decisions": conditional_contact_decisions,
        }
    mode_report_path = Path(config.output_dir) / "mode_switch_report.json"
    mode_report_path.write_text(json.dumps(mode_report, indent=2) + "\n", encoding="utf-8")
    loguru.logger.info(f"Saved mode-switch report to {mode_report_path}")

    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / config.render_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(qpos_traj, qpos_ref_np, config.embodiment_type, data_type)
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"], errors["obj_quat_err"],
        )

    _assert_object_actuator_gains_zero(env, config, "end")

    if "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info("Optimization complete! Keeping Viser server alive.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors


@hydra.main(version_base=None, config_path=str(_SPIDER_CONFIG_DIR), config_name="default")
def run_main(cfg: DictConfig) -> None:
    config_dict = dict(cfg)
    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
