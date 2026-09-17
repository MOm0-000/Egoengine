"""Vectorized Spider MJWP environment matching the H2S2R PpoAgent interface.

This is the adapter between the cloned Spider MuJoCo+Warp backend and the
verified PPO trainer from the official Human2Sim2Robot repository.  It does not
reimplement PPO; it only produces the ``get_env_info / reset / step`` surface
that ``human2sim2robot.ppo.ppo_agent.PpoAgent`` expects.

Action-space adaptation: H2S2R outputs palm/PCA actions for a fabric controller.
EgoEngine needs a residual on the upstream reference.  We expose one residual
per actuated xHand coordinate (18 unilateral, 36 bimanual) and add it to the
reference control inside :meth:`step`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from video_to_spider.rl.h2s2r import (
    AnchorRewardConfig,
    DomainRandomizationConfig,
    XHandResidualPolicySpec,
    make_anchor_points,
)
from video_to_spider.rl.reset_sampler import PreGraspResetSampler, PreGraspSamplerConfig
from egoengine_repro.action.paper_rewards import (
    TrackingObjective,
    lifting,
    object_tracking,
    opposition_contact,
)

try:  # Gym is an H2S2R PpoAgent dependency, but keep this module importable for tests.
    import gym
    from gym import spaces
except ModuleNotFoundError:  # pragma: no cover - only used in the training runtime
    gym = None
    spaces = None


_XHAND_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_WP_STATE_FIELDS = (
    "qpos", "qvel", "qacc", "time", "ctrl", "act", "act_dot",
    "qacc_warmstart", "qfrc_applied", "xfrc_applied", "energy",
    "mocap_pos", "mocap_quat", "xpos", "xquat", "xmat", "xipos", "ximat",
    "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cacc", "cfrc_int",
    "cfrc_ext", "sensordata", "actuator_length", "actuator_velocity",
    "actuator_force", "actuator_moment", "ten_length", "ten_velocity",
    "qacc_smooth", "qfrc_actuator", "qfrc_bias", "qfrc_constraint",
    "qfrc_damper", "qfrc_fluid", "qfrc_gravcomp", "qfrc_inverse",
    "qfrc_passive", "qfrc_smooth", "qfrc_spring", "cdof", "cdof_dot",
    "cvel", "cinert", "crb", "subtree_angmom", "subtree_com",
    "subtree_linvel", "xanchor", "xaxis", "qLD", "qLDiagInv", "qM",
    "moment_colind", "moment_rowadr", "moment_rownnz", "ne", "nefc", "nf",
    "nisland", "nl", "solver_niter", "tree_island", "eq_active", "nacon", "ncollision",
)
_WP_CONTACT_FIELDS = (
    "dist", "pos", "frame", "includemargin", "friction", "solref",
    "solreffriction", "solimp", "dim", "geom", "efc_address", "worldid",
    "type", "flex", "vert", "geomcollisionid",
)
_WP_EFC_FIELDS = (
    "type", "id", "J", "J_colind", "J_rowadr", "J_rownnz", "pos", "margin",
    "D", "vel", "aref", "frictionloss", "force", "state", "Ma",
)


def _quat_wxyz_to_rot(q: torch.Tensor) -> torch.Tensor:
    """MuJoCo quaternion ``(w, x, y, z)`` -> rotation matrix, batch-safe."""
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _object_pose_parts(
    qpos: torch.Tensor, nq_obj: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract one or two free-joint poses from the qpos tail.

    SPIDER encodes bimanual objects as ``[right, left]`` with either
    position-plus-axis-angle (12) or position-plus-quaternion (14) values.
    """
    if nq_obj in (6, 7):
        chunks = (qpos[..., -nq_obj:],)
    elif nq_obj in (12, 14):
        chunks = (qpos[..., -nq_obj : -nq_obj // 2], qpos[..., -nq_obj // 2 :])
    else:
        raise ValueError(f"Unsupported object qpos dimension {nq_obj}")
    result = []
    for obj in chunks:
        translation = obj[..., :3]
        if obj.shape[-1] == 6:
            rotation = _axis_angle_to_rot(obj[..., 3:6])
        else:
            rotation = _quat_wxyz_to_rot(obj[..., 3:7])
        result.append((translation, rotation))
    return result


def _object_pose_from_qpos(qpos: torch.Tensor, nq_obj: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible first-object accessor."""
    return _object_pose_parts(qpos, nq_obj)[0]


def _axis_angle_to_rot(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = torch.cos(angle).squeeze(-1)
    s = torch.sin(angle).squeeze(-1)
    one_c = 1 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], dim=-1),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], dim=-1),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], dim=-1),
        ],
        dim=-2,
    )


def _transform_anchors_torch(
    translation: torch.Tensor, rotation: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """``R @ k + t`` for batched poses and ``(N, 3)`` anchors."""
    return translation.unsqueeze(-2) + torch.einsum("...ij,kj->...ki", rotation, anchors)


@dataclass
class MJWPVectorEnvConfig:
    anchor: AnchorRewardConfig = field(default_factory=AnchorRewardConfig)
    domain: DomainRandomizationConfig = field(default_factory=lambda: DomainRandomizationConfig(
        obs_noise_std=0.0,
        action_noise_std=0.0,
        gravity_noise_std=0.0,
        scale_range=(1.0, 1.0),
        random_force_prob=0.0,
    ))
    residual: XHandResidualPolicySpec = field(default_factory=XHandResidualPolicySpec)
    fingertip_site_ids: tuple[int, ...] | None = None
    palm_site_ids: tuple[int, ...] | None = None
    hand_qpos_dof: int | None = None
    pre_grasp: PreGraspSamplerConfig = field(default_factory=PreGraspSamplerConfig)
    reference_start_index: int | None = None
    asymmetric_critic: bool = False
    reset_hand_noise_std: float = 0.0
    reset_object_pos_noise_std: float = 0.0
    reset_object_rot_noise_std: float = 0.0
    fingertip_reset_distance_max_m: float = 0.5
    max_episode_length: int = 240
    tracked_object_indices: tuple[int, ...] | None = None


class MJWPVectorEnv:
    """Wrap Spider's batched MJWP world as an H2S2R-compatible vectorized env."""

    def __init__(
        self,
        config: Any,
        ref_data: tuple[torch.Tensor, ...],
        *,
        num_envs: int | None = None,
        env_config: MJWPVectorEnvConfig | None = None,
        seed: int = 0,
    ) -> None:
        from spider.simulators import mjwp as spider_mjwp

        self._mjwp = spider_mjwp
        self.ego_cfg = replace(config, num_samples=int(num_envs or config.num_samples))
        self.num_envs = int(self.ego_cfg.num_samples)
        self.env_cfg = env_config or MJWPVectorEnvConfig()
        if self.env_cfg.hand_qpos_dof is None:
            object_ctrl_dims = int(getattr(self.ego_cfg, "object_action_dims", 0))
            inferred = int(self.ego_cfg.nu) - (object_ctrl_dims if object_ctrl_dims > 0 else 0)
            self.env_cfg.hand_qpos_dof = inferred if inferred > 0 else 18
        if self.env_cfg.hand_qpos_dof > 18 and self.env_cfg.residual.hand_dof == 18:
            self.env_cfg.residual = replace(
                self.env_cfg.residual,
                hand_dof=int(self.env_cfg.hand_qpos_dof),
                hand_control_indices=tuple(range(int(self.env_cfg.hand_qpos_dof))),
            )
        self.env_cfg.pre_grasp.validate()
        self.env_cfg.domain.validate()
        self.anchors = torch.as_tensor(
            make_anchor_points(self.env_cfg.anchor.length),
            dtype=torch.float32,
            device=str(config.device),
        )
        self.qpos_ref, self.qvel_ref, self.ctrl_ref = ref_data[:3]
        if len(self.qpos_ref) < 2:
            raise ValueError("reference must contain at least two endpoint states")
        if not np.isclose(float(self.ego_cfg.ctrl_dt), float(self.ego_cfg.ref_dt)):
            raise ValueError("MJWP PPO requires one reference row per control step")
        if int(self.ego_cfg.ctrl_steps) < 1:
            raise ValueError("ctrl_dt must contain at least one physics step")
        if (
            self.ego_cfg.contact_guidance
            and self.ctrl_ref.shape[1] != self.ego_cfg.nu
            and self.qpos_ref.shape[1] >= self.ego_cfg.nu
        ):
            self.ctrl_ref = self.qpos_ref[:, : self.ego_cfg.nu]
        self.contact_ref, self.contact_pos_ref = ref_data[3:5] if len(ref_data) >= 5 else (None, None)
        env_ref_data = (
            self.qpos_ref,
            self.qvel_ref,
            self.ctrl_ref,
            self.contact_ref,
            self.contact_pos_ref,
        )
        n_objects = 2 if int(self.ego_cfg.nq_obj) in (12, 14) else 1
        if self.env_cfg.tracked_object_indices is None:
            self.tracked_object_indices = tuple(range(n_objects))
        else:
            self.tracked_object_indices = tuple(self.env_cfg.tracked_object_indices)
            if not self.tracked_object_indices or any(
                index < 0 or index >= n_objects for index in self.tracked_object_indices
            ):
                raise ValueError("tracked_object_indices must select existing objects")
        object_linear_velocity = self.qvel_ref[:, -6 * n_objects:].reshape(-1, n_objects, 6)[..., :3].mean(1).cpu().numpy()
        self.pre_grasp_sampler = PreGraspResetSampler(self.env_cfg.pre_grasp, seed=seed)
        self.start_indices = self.pre_grasp_sampler.sample_indices(
            object_linear_velocity, self.num_envs
        )
        if self.env_cfg.reference_start_index is not None:
            index = self.env_cfg.reference_start_index
            if not isinstance(index, int) or not 0 <= index < len(self.qpos_ref) - 1:
                raise ValueError("explicit reference start must leave a valid transition")
            self.start_indices[:] = index
        if np.any(self.start_indices >= len(self.qpos_ref) - 1):
            raise ValueError("pre-grasp reset must leave at least one reference transition")
        self.episode_lengths = np.minimum(
            int(self.env_cfg.max_episode_length),
            len(self.qpos_ref) - 1 - self.start_indices,
        ).astype(np.int32)
        self.time_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.rng = np.random.default_rng(seed)
        self._chunk_reset_state = None
        # Work spent on rejected lookahead and PPO must not disappear on restore.
        self.simulation_control_intervals = 0
        self.simulation_physics_steps = 0
        self._last_action = torch.zeros(
            self.num_envs, self.env_cfg.residual.hand_dof,
            dtype=torch.float32, device=str(self.ego_cfg.device),
        )
        self._last_ctrl = torch.zeros(
            self.num_envs, int(self.ego_cfg.nu),
            dtype=torch.float32, device=str(self.ego_cfg.device),
        )
        self._initial_object_heights = torch.zeros(
            self.num_envs, n_objects, dtype=torch.float32, device=str(self.ego_cfg.device),
        )
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=str(self.ego_cfg.device))
        self._last_contact_score = torch.zeros(self.num_envs, dtype=torch.float32, device=str(self.ego_cfg.device))

        self.env = self._mjwp.setup_env(self.ego_cfg, env_ref_data)
        self._resolve_sites()
        self._resolve_contact_maps()
        self.obs_dim = self._compute_obs_dim()
        self.priv_dim = self._compute_privileged_dim()
        self.tracking_boundary = float(
            (
                float(getattr(self.ego_cfg, "pos_rew_scale", 1.0))
                * float(getattr(self.ego_cfg, "object_pos_threshold", 0.12)) ** 2
                + float(getattr(self.ego_cfg, "rot_rew_scale", 1.0))
                * float(getattr(self.ego_cfg, "object_rot_threshold", 1.5)) ** 2
            ) ** 0.5
        )
        self._last_tracking_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=str(self.ego_cfg.device)
        )
        self._reset_worlds()

    # ------------------------------------------------------------------
    # PpoAgent-compatible API
    # ------------------------------------------------------------------
    def get_env_info(self) -> dict[str, Any]:
        assert spaces is not None, "gym is required for the H2S2R PpoAgent runtime"
        return {
            "observation_space": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "action_space": spaces.Box(low=-1.0, high=1.0, shape=(self.env_cfg.residual.hand_dof,), dtype=np.float32),
            "state_space": spaces.Box(low=-np.inf, high=np.inf, shape=(self.priv_dim,), dtype=np.float32),
            "agents": 1,
            "value_size": 1,
        }

    def get_number_of_agents(self) -> int:
        return 1

    def reset(self) -> np.ndarray | dict[str, np.ndarray]:
        self._reset_worlds()
        obs, privileged = self._build_observations()
        return self._pack_observation(obs, privileged)

    def step(self, actions: np.ndarray, *, auto_reset: bool = True) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.env_cfg.residual.hand_dof):
            raise ValueError(f"actions must have shape {(self.num_envs, self.env_cfg.residual.hand_dof)}")
        if not np.isfinite(actions).all():
            raise ValueError("residual actions must be finite")
        if self.env_cfg.domain.action_noise_std > 0.0:
            actions = actions + self.rng.normal(
                0.0,
                self.env_cfg.domain.action_noise_std,
                size=actions.shape,
            ).astype(np.float32)
        # A saved ctrl row is the command that produced the matching next state.
        # Use row t+1 to advance from reference state t to t+1.
        reference_ctrls = self._reference_ctrls(self.time_indices, offset=1)
        delta = torch.as_tensor(actions, dtype=torch.float32, device=str(self.ego_cfg.device))
        full_ctrl = self._apply_residual(reference_ctrls, delta)
        for _ in range(max(int(self.ego_cfg.ctrl_steps), 1)):
            self._mjwp.step_env(self.ego_cfg, self.env, full_ctrl)
            self.simulation_physics_steps += self.num_envs
            self._check_capacity()
        if str(self.ego_cfg.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.ego_cfg.device)

        self._last_action = delta.detach().clone()
        self._last_ctrl = full_ctrl.detach().clone()
        self.time_indices += 1
        self.simulation_control_intervals += self.num_envs
        obs, privileged = self._build_observations()
        reward = self._compute_reward(privileged["object_pose"], privileged["goal_object_pose"], privileged["contact_flags"])
        done = self._compute_done()

        done_np = done.cpu().numpy()
        time_outs = self.time_indices >= self.episode_lengths
        terminal_contact_score = self._last_contact_score.cpu().numpy().copy()
        terminal_tracking_error = self._last_tracking_error.cpu().numpy().copy()
        terminal_terminated = self._last_terminated.cpu().numpy().copy()
        reset_mask = done_np
        if auto_reset and reset_mask.any():
            self._reset_worlds(reset_mask)
            obs, next_privileged = self._build_observations()
        else:
            next_privileged = privileged

        infos: dict[str, Any] = {
            "reward": reward.cpu().numpy(),
            "contact_score": terminal_contact_score,
            "contact_flags": privileged["contact_flags"].cpu().numpy(),
            "object_tracking_error": terminal_tracking_error,
            "time_outs": time_outs,
            "terminated": terminal_terminated,
        }
        return self._pack_observation(obs, next_privileged), reward.cpu().numpy(), done_np, infos

    def set_train_info(self, frame: int, agent: Any) -> None:
        return None

    def set_chunk_reset(self, *, start: int, end: int) -> None:
        """Train a bounded window from its exact incoming state, not reference qpos.

        The initial implementation deliberately supports one world. Packed Warp
        contact buffers cannot be partially copied as ordinary per-world arrays.
        This prevents a seemingly vectorized reset from corrupting other worlds.
        """
        if self.num_envs != 1 or np.any(self.start_indices != 0):
            raise ValueError("exact chunk reset currently requires one world starting at source row 0")
        if not 0 <= start < end < len(self.qpos_ref) or int(self.time_indices[0]) != start:
            raise ValueError("chunk boundary must match current reference cursor and leave valid endpoints")
        if any((self.env_cfg.reset_hand_noise_std, self.env_cfg.reset_object_pos_noise_std,
                self.env_cfg.reset_object_rot_noise_std, self.env_cfg.domain.action_noise_std,
                self.env_cfg.domain.obs_noise_std)):
            raise ValueError("exact chunk comparison requires disabled reset/domain noise")
        self.episode_lengths[:] = end
        self._chunk_reset_state = self.get_env_state()

    def get_env_state(self) -> Any:
        state: dict[str, Any] = {
            "time_indices": self.time_indices.copy(),
            "start_indices": self.start_indices.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "last_action": self._last_action.cpu().clone(),
            "last_ctrl": self._last_ctrl.cpu().clone(),
            "initial_object_heights": self._initial_object_heights.cpu().clone(),
            "last_tracking_error": self._last_tracking_error.cpu().clone(),
            "last_terminated": self._last_terminated.cpu().clone(),
            "last_contact_score": self._last_contact_score.cpu().clone(),
        }
        for prefix, data in (("", self.env.data_wp), ("prev.", self.env.data_wp_prev)):
            for name in _WP_STATE_FIELDS:
                if hasattr(data, name):
                    state[f"{prefix}{name}"] = wp.to_torch(getattr(data, name)).cpu().clone()
            for name in _WP_CONTACT_FIELDS:
                if hasattr(data.contact, name):
                    state[f"{prefix}contact.{name}"] = wp.to_torch(getattr(data.contact, name)).cpu().clone()
            for name in _WP_EFC_FIELDS:
                if hasattr(data.efc, name):
                    state[f"{prefix}efc.{name}"] = wp.to_torch(getattr(data.efc, name)).cpu().clone()
        return state

    def set_env_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("MJWP state must be the dictionary returned by get_env_state")
        self.time_indices = np.asarray(state["time_indices"], dtype=np.int32).copy()
        self.start_indices = np.asarray(state["start_indices"], dtype=np.int64).copy()
        self.episode_lengths = np.asarray(state["episode_lengths"], dtype=np.int32).copy()
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self._last_action = state["last_action"].to(str(self.ego_cfg.device)).clone()
        self._last_ctrl = state["last_ctrl"].to(str(self.ego_cfg.device)).clone()
        self._initial_object_heights = state["initial_object_heights"].to(str(self.ego_cfg.device)).clone()
        self._last_tracking_error = state["last_tracking_error"].to(str(self.ego_cfg.device)).clone()
        self._last_terminated = state["last_terminated"].to(str(self.ego_cfg.device)).clone()
        self._last_contact_score = state["last_contact_score"].to(str(self.ego_cfg.device)).clone()
        with wp.ScopedDevice(self.env.device):
            for key, value in state.items():
                if key in {
                    "time_indices", "start_indices", "episode_lengths", "rng_state",
                    "last_action", "last_ctrl", "initial_object_heights",
                    "last_tracking_error", "last_terminated", "last_contact_score",
                }:
                    continue
                data = self.env.data_wp_prev if key.startswith("prev.") else self.env.data_wp
                key = key[5:] if key.startswith("prev.") else key
                parts = key.split(".")
                target = data if len(parts) == 1 else getattr(data, parts[0])
                target = getattr(target, parts[-1])
                wp.copy(target, wp.from_torch(value.to(str(self.env.device))))
        wp.synchronize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_obs_dim(self) -> int:
        n_hand = self.env_cfg.hand_qpos_dof
        n_anchors = self.anchors.shape[0]
        n_objects = 2 if int(self.ego_cfg.nq_obj) in (12, 14) else 1
        n_contact_pairs = len(self.env_cfg.fingertip_site_ids) * n_objects
        return (
            n_hand * 2
            + len(self.env_cfg.fingertip_site_ids) * 3
            + len(self.env_cfg.palm_site_ids) * 3
            + n_anchors * 3 * 2 * n_objects
            + n_hand * 2
            + n_contact_pairs
        )

    def _compute_privileged_dim(self) -> int:
        n_objects = 2 if int(self.ego_cfg.nq_obj) in (12, 14) else 1
        n_contact_pairs = len(self.env_cfg.fingertip_site_ids) * n_objects
        return 6 * n_objects + self.env_cfg.hand_qpos_dof + n_contact_pairs * 3

    def _reference_ctrls(self, time_indices: np.ndarray, *, offset: int = 0) -> torch.Tensor:
        """Return one reference control row per vectorized environment."""
        indices = np.minimum(
            self.start_indices + np.asarray(time_indices, dtype=np.int64) + int(offset),
            self.ctrl_ref.shape[0] - 1,
        )
        return self.ctrl_ref[indices].to(str(self.ego_cfg.device))

    def _apply_residual(self, reference_ctrls: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        spec = self.env_cfg.residual
        residual = torch.clamp(spec.residual_scale * delta, -spec.residual_clip, spec.residual_clip)
        out = reference_ctrls.clone()
        out[:, list(spec.hand_control_indices)] += residual
        return out

    def _write_state(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
        reset_mask: np.ndarray,
    ) -> None:
        qpos = qpos.to(str(self.ego_cfg.device), torch.float32).contiguous()
        qvel = qvel.to(str(self.ego_cfg.device), torch.float32).contiguous()
        ctrl = ctrl.to(str(self.ego_cfg.device), torch.float32).contiguous()
        rows = torch.as_tensor(reset_mask, dtype=torch.bool, device=str(self.ego_cfg.device))
        with wp.ScopedDevice(self.env.device):
            preserved = {}
            if not bool(np.all(reset_mask)):
                for name in _WP_STATE_FIELDS:
                    if hasattr(self.env.data_wp, name):
                        current = wp.to_torch(getattr(self.env.data_wp, name))
                        if current.ndim and current.shape[0] == self.num_envs:
                            preserved[name] = current[~rows].clone()
                for name in _WP_EFC_FIELDS:
                    if hasattr(self.env.data_wp.efc, name):
                        current = wp.to_torch(getattr(self.env.data_wp.efc, name))
                        preserved[f"efc.{name}"] = current[~rows].clone()
            wp.copy(self.env.data_wp.qpos, wp.from_torch(qpos))
            wp.copy(self.env.data_wp.qvel, wp.from_torch(qvel))
            wp.copy(self.env.data_wp.ctrl, wp.from_torch(ctrl))
            for name in ("qacc", "qacc_warmstart", "act", "act_dot", "qfrc_applied", "xfrc_applied"):
                if hasattr(self.env.data_wp, name):
                    current = wp.to_torch(getattr(self.env.data_wp, name)).clone()
                    current[rows] = 0
                    wp.copy(getattr(self.env.data_wp, name), wp.from_torch(current))
            time = wp.to_torch(self.env.data_wp.time).clone()
            time[rows] = 0
            wp.copy(self.env.data_wp.time, wp.from_torch(time))

            contact_world = wp.to_torch(self.env.data_wp.contact.worldid).long()
            valid_world = (contact_world >= 0) & (contact_world < self.num_envs)
            contact_rows = valid_world & rows[contact_world.clamp(0, self.num_envs - 1)]
            for name in _WP_CONTACT_FIELDS:
                if hasattr(self.env.data_wp.contact, name):
                    current = wp.to_torch(getattr(self.env.data_wp.contact, name)).clone()
                    current[contact_rows] = -1 if name in {"worldid", "geom", "efc_address"} else 0
                    wp.copy(getattr(self.env.data_wp.contact, name), wp.from_torch(current))
            for name in _WP_EFC_FIELDS:
                if hasattr(self.env.data_wp.efc, name):
                    current = wp.to_torch(getattr(self.env.data_wp.efc, name)).clone()
                    current[rows] = 0
                    wp.copy(getattr(self.env.data_wp.efc, name), wp.from_torch(current))
            mjwarp.forward(self.env.model_wp, self.env.data_wp)
            for key, value in preserved.items():
                owner, name = (
                    (self.env.data_wp.efc, key[4:])
                    if key.startswith("efc.")
                    else (self.env.data_wp, key)
                )
                current = wp.to_torch(getattr(owner, name)).clone()
                current[~rows] = value
                wp.copy(getattr(owner, name), wp.from_torch(current))
        wp.synchronize()

    def _resolve_sites(self) -> None:
        hands = 2 if self.ego_cfg.embodiment_type == "bimanual" else 1
        sides = ("right", "left") if hands == 2 else (self.ego_cfg.embodiment_type,)
        if self.env_cfg.fingertip_site_ids is None or len(self.env_cfg.fingertip_site_ids) != 5 * hands:
            self.env_cfg.fingertip_site_ids = tuple(
                int(mujoco.mj_name2id(self.env.model_cpu, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip"))
                for side in sides for finger in _XHAND_FINGERS
            )
        if self.env_cfg.palm_site_ids is None or len(self.env_cfg.palm_site_ids) != hands:
            self.env_cfg.palm_site_ids = tuple(
                int(mujoco.mj_name2id(self.env.model_cpu, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm"))
                for side in sides
            )
        if any(site_id < 0 for site_id in (*self.env_cfg.fingertip_site_ids, *self.env_cfg.palm_site_ids)):
            raise ValueError("XHand palm/fingertip sites are missing from the scene")

    def _resolve_contact_maps(self) -> None:
        if not getattr(self.ego_cfg, "force_closure_geom_finger_map", None):
            from spider.config import build_force_closure_geom_maps
            maps = build_force_closure_geom_maps(self.env.model_cpu, self.ego_cfg.embodiment_type)
            self.ego_cfg.force_closure_geom_finger_map = maps[0]
            self.ego_cfg.force_closure_geom_hand_group_map = maps[1]
            self.ego_cfg.force_closure_geom_object_group_map = maps[2]

    def _live_contact_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_hands = len(self.env_cfg.palm_site_ids)
        num_objects = 2 if int(self.ego_cfg.nq_obj) in (12, 14) else 1
        shape = (self.num_envs, num_hands, num_objects, 5)
        zeros = torch.zeros(shape, dtype=torch.float32, device=str(self.ego_cfg.device))
        force_zeros = torch.zeros(*shape, 3, dtype=torch.float32, device=str(self.ego_cfg.device))
        contact = self.env.data_wp.contact
        efc = self.env.data_wp.efc
        required = ("geom", "dist", "worldid", "efc_address")
        if not all(hasattr(contact, name) for name in required) or not hasattr(efc, "force"):
            if float(getattr(self.ego_cfg, "physical_contact_rew_scale", 0.0)) > 0.0:
                raise RuntimeError("MJWarp contact fields are unavailable; cannot train with physical contact reward")
            return zeros.bool(), force_zeros
        geom = wp.to_torch(contact.geom).long()
        dist = wp.to_torch(contact.dist)
        worldid = wp.to_torch(contact.worldid).long()
        address = wp.to_torch(contact.efc_address).long()
        frame = wp.to_torch(contact.frame) if hasattr(contact, "frame") else None
        efc_force = wp.to_torch(efc.force)
        finger_map = torch.as_tensor(self.ego_cfg.force_closure_geom_finger_map, dtype=torch.long, device=geom.device)
        object_map = torch.as_tensor(self.ego_cfg.force_closure_geom_object_group_map, dtype=torch.long, device=geom.device)
        safe_geom = geom.clamp(0, max(finger_map.numel() - 1, 0))
        f0, f1 = finger_map[safe_geom[..., 0]], finger_map[safe_geom[..., 1]]
        o0, o1 = object_map[safe_geom[..., 0]], object_map[safe_geom[..., 1]]
        first = (f0 >= 0) & (o1 >= 0)
        second = (f1 >= 0) & (o0 >= 0)
        finger = torch.where(first, f0, f1)
        object_index = torch.where(first, o1, o0)
        valid = (
            (first | second)
            & (finger >= 0)
            & (finger < num_hands * 5)
            & (object_index >= 0)
            & (object_index < num_objects)
        )
        valid_world = (worldid >= 0) & (worldid < self.num_envs)
        safe_world = worldid.clamp(0, max(self.num_envs - 1, 0))
        addr_valid = (address >= 0) & (address < efc_force.shape[1])
        safe_addr = address.clamp(0, max(efc_force.shape[1] - 1, 0))
        components = torch.relu(efc_force[safe_world[:, None], safe_addr])
        components = torch.where(addr_valid, components, torch.zeros_like(components))
        normal_force = components.sum(dim=-1)
        touching = valid & valid_world & addr_valid.any(dim=-1) & (dist <= 0.0) & (normal_force > 0.0)
        # Warp retains unused packed-buffer entries. They are not live contacts.
        count = wp.to_torch(self.env.data_wp.nacon)[0]
        touching &= torch.arange(len(dist), device=dist.device) < count
        hand_index = (finger // 5).clamp(0, max(num_hands - 1, 0))
        finger_index = (finger % 5).clamp(0, 4)
        safe_object = object_index.clamp(0, max(num_objects - 1, 0))
        flat_index = (((safe_world * num_hands + hand_index) * num_objects + safe_object) * 5 + finger_index)
        flat_size = self.num_envs * num_hands * num_objects * 5
        flat = torch.zeros(flat_size, dtype=normal_force.dtype, device=normal_force.device)
        flat.scatter_reduce_(0, flat_index.reshape(-1), torch.where(touching, normal_force, torch.zeros_like(normal_force)).reshape(-1), reduce="amax", include_self=True)
        flags = flat.reshape(shape) > 0.0
        if frame is None:
            return flags, force_zeros
        normal = frame[:, 0, :]
        normal = torch.where(first[:, None], normal, -normal)
        flat_vec = torch.zeros(flat_size, 3, dtype=normal.dtype, device=normal.device)
        weighted = torch.where(touching[:, None], normal * normal_force[:, None], torch.zeros_like(normal))
        flat_vec.scatter_add_(0, flat_index.reshape(-1, 1).expand(-1, 3), weighted.reshape(-1, 3))
        force = flat_vec.reshape(*shape, 3)
        return flags, force

    def _check_capacity(self) -> None:
        data = self.env.data_wp
        contacts = int(data.nacon.numpy()[0])
        broadphase = int(data.ncollision.numpy()[0])
        constraints = data.nefc.numpy()
        if max(contacts, broadphase) > data.naconmax or np.any(constraints > data.njmax):
            raise RuntimeError(f"MJWP capacity overflow: contacts={contacts}, broadphase={broadphase}, "
                               f"constraints={constraints.tolist()}, capacities={data.naconmax}/{data.njmax}")

    def _reset_worlds(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        if self._chunk_reset_state is not None:
            if mask.any():
                if self.num_envs != 1 or not mask.all():
                    raise ValueError("partial packed-contact chunk reset is not implemented")
                self.set_env_state(self._chunk_reset_state)
            return
        base_qpos = self.qpos_ref[self.start_indices].cpu().numpy()
        base_qvel = self.qvel_ref[self.start_indices].cpu().numpy()
        if mask.all():
            qpos = base_qpos.astype(np.float32).copy()
            qvel = base_qvel.astype(np.float32).copy()
        else:
            qpos = self._mjwp.get_qpos(self.ego_cfg, self.env).detach().cpu().numpy()
            qvel = self._mjwp.get_qvel(self.ego_cfg, self.env).detach().cpu().numpy()

        for idx in np.nonzero(mask)[0]:
            qpos[idx] = base_qpos[idx]
            qvel[idx] = base_qvel[idx]
            qpos[idx, : self.env_cfg.hand_qpos_dof] += self.rng.normal(
                0.0, self.env_cfg.reset_hand_noise_std, size=(self.env_cfg.hand_qpos_dof,)
            ).astype(np.float32)
            n_objects = 2 if int(self.ego_cfg.nq_obj) in (12, 14) else 1
            object_width = int(self.ego_cfg.nq_obj) // n_objects
            first_object = qpos.shape[1] - int(self.ego_cfg.nq_obj)
            for object_index in range(n_objects):
                start = first_object + object_index * object_width
                qpos[idx, start : start + 3] += self.rng.normal(
                    0.0, self.env_cfg.reset_object_pos_noise_std, size=(3,)
                ).astype(np.float32)
                if object_width == 6:
                    qpos[idx, start + 3 : start + 6] += self.rng.normal(
                        0.0, self.env_cfg.reset_object_rot_noise_std, size=(3,)
                    ).astype(np.float32)
                else:
                    q = qpos[idx, start + 3 : start + 7].astype(np.float32)
                    q += self.rng.normal(0.0, self.env_cfg.reset_object_rot_noise_std, size=(4,)).astype(np.float32)
                    qpos[idx, start + 3 : start + 7] = q / max(np.linalg.norm(q), 1e-12)

        self.time_indices[mask] = 0
        object_start = qpos.shape[1] - int(self.ego_cfg.nq_obj)
        object_width = int(self.ego_cfg.nq_obj) // n_objects
        for object_index in range(n_objects):
            start = object_start + object_index * object_width
            self._initial_object_heights[mask, object_index] = torch.as_tensor(
                qpos[mask, start + 2], dtype=torch.float32, device=str(self.ego_cfg.device)
            )
        self._last_action[mask] = 0.0
        self._last_terminated[mask] = False
        self._last_contact_score[mask] = 0.0
        reset_ctrl = self._last_ctrl.clone()
        reference_ctrl = self._reference_ctrls(self.time_indices)
        reset_ctrl[mask] = reference_ctrl[mask]
        self._last_ctrl[mask] = reference_ctrl[mask]
        self._write_state(
            torch.as_tensor(qpos, device=str(self.ego_cfg.device)),
            torch.as_tensor(qvel, device=str(self.ego_cfg.device)),
            reset_ctrl,
            mask,
        )

    def _build_observations(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        qpos = self._mjwp.get_qpos(self.ego_cfg, self.env)
        qvel = self._mjwp.get_qvel(self.ego_cfg, self.env)
        site_xpos = wp.to_torch(self.env.data_wp.site_xpos)
        fingertip = site_xpos[:, list(self.env_cfg.fingertip_site_ids)]
        palm = site_xpos[:, list(self.env_cfg.palm_site_ids)]
        contact_flags, contact_forces = self._live_contact_features()

        current_objects = _object_pose_parts(qpos, int(self.ego_cfg.nq_obj))
        goal_indices = np.minimum(
            self.start_indices + self.time_indices,
            self.qpos_ref.shape[0] - 1,
        )
        goal_qpos = self.qpos_ref[goal_indices].to(qpos.device)
        goal_objects = _object_pose_parts(goal_qpos, int(self.ego_cfg.nq_obj))
        current_anchors = torch.cat([
            _transform_anchors_torch(pose[0], pose[1], self.anchors)
            for pose in current_objects
        ], dim=1)
        goal_anchors = torch.cat([
            _transform_anchors_torch(pose[0], pose[1], self.anchors)
            for pose in goal_objects
        ], dim=1)

        hand_qpos = qpos[:, : self.env_cfg.hand_qpos_dof]
        hand_qvel = qvel[:, : self.env_cfg.hand_qpos_dof]
        current_reference = self._reference_ctrls(self.time_indices, offset=1)[:, : self.env_cfg.hand_qpos_dof]
        next_reference = self._reference_ctrls(self.time_indices, offset=2)[:, : self.env_cfg.hand_qpos_dof]
        observation = torch.cat(
            [
                hand_qpos,
                hand_qvel,
                fingertip.reshape(self.num_envs, -1),
                palm.reshape(self.num_envs, -1),
                current_anchors.reshape(self.num_envs, -1),
                goal_anchors.reshape(self.num_envs, -1),
                current_reference,
                next_reference,
                contact_flags.reshape(self.num_envs, -1).to(qpos.dtype),
            ],
            dim=1,
        )
        if self.env_cfg.domain.obs_noise_std > 0.0:
            noise = self.rng.normal(
                0.0, self.env_cfg.domain.obs_noise_std, size=observation.shape
            ).astype(np.float32)
            observation = observation + torch.as_tensor(noise, device=observation.device)

        object_velocities = qvel[:, -6 * len(current_objects) :].reshape(self.num_envs, len(current_objects), 6)
        object_linear = object_velocities[:, :, :3].reshape(self.num_envs, -1)
        object_angular = object_velocities[:, :, 3:].reshape(self.num_envs, -1)
        joint_forces = torch.zeros(self.num_envs, self.env_cfg.hand_qpos_dof, device=qpos.device)
        try:
            joint_forces = wp.to_torch(self.env.data_wp.qfrc_actuator)[:, : self.env_cfg.hand_qpos_dof]
        except AttributeError:
            # Some mjwarp builds expose actuator forces under a different name;
            # keep the privileged state zero rather than failing the rollout.
            joint_forces = torch.zeros_like(joint_forces)
        privileged = {
            "object_linear_velocity": object_linear,
            "object_angular_velocity": object_angular,
            "joint_forces": joint_forces,
            "fingertip_contact_forces": contact_forces,
            "contact_flags": contact_flags,
            "fingertip_positions": fingertip,
            "object_pose": current_objects,
            "goal_object_pose": goal_objects,
            "palm_position": palm,
        }
        return observation, privileged

    def _build_privileged_vector(self, privileged: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten the privileged critic state into the ``state_space`` vector.

        The asymmetric critic only consumes the privileged physics information:
        object linear/angular velocity, actuator forces on the hand DOFs and
        per hand-object-fingertip contact forces.  The extra keys in ``privileged`` remain
        internal inputs for reward/done computation and are not exposed as the
        critic observation.
        """
        return torch.cat(
            [
                privileged["object_linear_velocity"],
                privileged["object_angular_velocity"],
                privileged["joint_forces"],
                privileged["fingertip_contact_forces"].reshape(self.num_envs, -1),
            ],
            dim=1,
        )

    def _pack_observation(
        self, observation: torch.Tensor, privileged: dict[str, torch.Tensor]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Return actor-only or actor+critic observation depending on config."""
        if not self.env_cfg.asymmetric_critic:
            return observation.cpu().numpy()
        return {
            "obs": observation.cpu().numpy(),
            "states": self._build_privileged_vector(privileged).cpu().numpy(),
        }

    def _object_distances(self, current_objects, goal_objects) -> torch.Tensor:
        distances = []
        for current_object, goal_object in zip(current_objects, goal_objects, strict=True):
            current_anchors = _transform_anchors_torch(current_object[0], current_object[1], self.anchors)
            goal_anchors = _transform_anchors_torch(goal_object[0], goal_object[1], self.anchors)
            distances.append(torch.sum(torch.norm(goal_anchors - current_anchors, dim=-1), dim=-1))
        return torch.stack(distances, dim=1)

    def _compute_reward(self, current_objects, goal_objects, contact_flags) -> torch.Tensor:
        objective = TrackingObjective(
            lambda_p=float(getattr(self.ego_cfg, "pos_rew_scale", 1.0)),
            lambda_r=float(getattr(self.ego_cfg, "rot_rew_scale", 1.0)),
            boundary=float(getattr(self, "tracking_boundary", 1.0)),
        )
        scores = [
            object_tracking(current_objects[index][0], current_objects[index][1],
                            goal_objects[index][0], goal_objects[index][1], objective)
            for index in self.tracked_object_indices
        ]
        self._last_tracking_error = torch.stack([score.error for score in scores], dim=1).mean(dim=1)
        self._last_terminated = torch.stack([score.terminated for score in scores], dim=1).any(dim=1)
        self._last_contact_score = opposition_contact(
            contact_flags[:, :, self.tracked_object_indices],
            coefficient=float(getattr(self.ego_cfg, "physical_contact_rew_scale", 0.0)),
        ).mean(dim=(1, 2))
        reward = torch.stack([score.reward for score in scores], dim=1).mean(dim=1) + self._last_contact_score
        lift_scale = float(getattr(self.ego_cfg, "lift_rew_scale", 0.0))
        if lift_scale > 0.0:
            heights = torch.stack([pose[0][:, 2] for pose in current_objects], dim=1)
            reward = reward + lifting(heights[:, 0], self._initial_object_heights[:, 0], lambda_z=lift_scale)
        return reward

    def _compute_done(self) -> torch.Tensor:
        done = self._last_terminated | torch.as_tensor(
            self.time_indices >= self.episode_lengths,
            device=self._last_terminated.device,
        )
        return done
