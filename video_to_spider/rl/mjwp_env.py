"""Vectorized Spider MJWP environment matching the H2S2R PpoAgent interface.

This is the adapter between the cloned Spider MuJoCo+Warp backend and the
verified PPO trainer from the official Human2Sim2Robot repository.  It does not
reimplement PPO; it only produces the ``get_env_info / reset / step`` surface
that ``human2sim2robot.ppo.ppo_agent.PpoAgent`` expects.

Action-space adaptation: H2S2R outputs palm/PCA actions for a fabric controller.
EgoEngine needs a residual on the upstream reference.  We expose an 18-DoF
xHand residual action in ``[-1, 1]`` and add it to the reference control on the
hand dimensions inside :meth:`step`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import mujoco_warp as mjwarp
import numpy as np
import torch

from video_to_spider.rl.h2s2r import (
    AnchorRewardConfig,
    DomainRandomizationConfig,
    XHandResidualPolicySpec,
    make_anchor_points,
)
from video_to_spider.rl.reset_sampler import PreGraspResetSampler, PreGraspSamplerConfig

try:  # Gym is an H2S2R PpoAgent dependency, but keep this module importable for tests.
    import gym
    from gym import spaces
except ModuleNotFoundError:  # pragma: no cover - only used in the training runtime
    gym = None
    spaces = None


_XHAND_FINGERTIP_SITE_IDS = (1, 4, 7, 10, 13)
_XHAND_PALM_SITE_ID = 0


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


def _object_pose_from_qpos(qpos: torch.Tensor, nq_obj: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract translation and rotation matrix from the free-joint object tail."""
    obj = qpos[..., -nq_obj:]
    translation = obj[..., :3]
    if nq_obj == 6:
        # Contact-guidance scene uses axis-angle rather than a quaternion.
        rotation = _axis_angle_to_rot(obj[..., 3:6])
    elif nq_obj == 7:
        rotation = _quat_wxyz_to_rot(obj[..., 3:7])
    else:
        raise ValueError(f"Unsupported object qpos dimension {nq_obj}")
    return translation, rotation


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
    domain: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    residual: XHandResidualPolicySpec = field(default_factory=XHandResidualPolicySpec)
    fingertip_site_ids: tuple[int, ...] = _XHAND_FINGERTIP_SITE_IDS
    palm_site_id: int = _XHAND_PALM_SITE_ID
    hand_qpos_dof: int = 18
    pre_grasp: PreGraspSamplerConfig = field(default_factory=PreGraspSamplerConfig)
    asymmetric_critic: bool = False
    reset_hand_noise_std: float = 0.0
    reset_object_pos_noise_std: float = 0.0
    reset_object_rot_noise_std: float = 0.0
    fingertip_reset_distance_max_m: float = 0.5
    max_episode_length: int = 240


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
        self.env_cfg.pre_grasp.validate()
        self.env_cfg.domain.validate()
        self.anchors = torch.as_tensor(
            make_anchor_points(self.env_cfg.anchor.length),
            dtype=torch.float32,
            device=str(config.device),
        )
        self.qpos_ref, self.qvel_ref, self.ctrl_ref = ref_data[:3]
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
        self.obs_dim = self._compute_obs_dim()
        self.priv_dim = self._compute_privileged_dim()
        object_linear_velocity = self.qvel_ref[:, -6:-3].cpu().numpy()
        self.pre_grasp_sampler = PreGraspResetSampler(self.env_cfg.pre_grasp, seed=seed)
        self.start_indices = self.pre_grasp_sampler.sample_indices(
            object_linear_velocity, self.num_envs
        )
        self.time_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

        self.env = self._mjwp.setup_env(self.ego_cfg, env_ref_data)
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

    def step(self, actions: np.ndarray) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.env_cfg.residual.hand_dof):
            raise ValueError(f"actions must have shape {(self.num_envs, self.env_cfg.residual.hand_dof)}")
        if self.env_cfg.domain.action_noise_std > 0.0:
            actions = actions + self.rng.normal(
                0.0,
                self.env_cfg.domain.action_noise_std,
                size=actions.shape,
            ).astype(np.float32)
        reference_ctrls = self._reference_ctrls(self.time_indices)
        delta = torch.as_tensor(actions, dtype=torch.float32, device=str(self.ego_cfg.device))
        full_ctrl = self._apply_residual(reference_ctrls, delta)
        self._mjwp.step_env(self.ego_cfg, self.env, full_ctrl)
        torch.cuda.synchronize(self.ego_cfg.device)

        obs, privileged = self._build_observations()
        reward = self._compute_reward(privileged["object_pose"], privileged["goal_object_pose"])
        done = self._compute_done(reward, privileged)

        done_np = done.cpu().numpy()
        self.time_indices += 1
        reset_mask = done_np | (self.time_indices >= self.env_cfg.max_episode_length)
        if reset_mask.any():
            self._reset_worlds(reset_mask)

        infos: dict[str, Any] = {"reward": reward.cpu().numpy()}
        return self._pack_observation(obs, privileged), reward.cpu().numpy(), done_np, infos

    def set_train_info(self, frame: int, agent: Any) -> None:
        return None

    def get_env_state(self) -> Any:
        qpos = self._mjwp.get_qpos(self.ego_cfg, self.env).clone()
        qvel = self._mjwp.get_qvel(self.ego_cfg, self.env).clone()
        return (qpos.cpu(), qvel.cpu(), self.time_indices.copy())

    def set_env_state(self, state: Any) -> None:
        qpos, qvel, time_indices = state
        self.time_indices = np.asarray(time_indices, dtype=np.int32).copy()
        self._write_state(qpos.to(str(self.ego_cfg.device)), qvel.to(str(self.ego_cfg.device)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_obs_dim(self) -> int:
        n_hand = self.env_cfg.hand_qpos_dof
        n_anchors = self.anchors.shape[0]
        return n_hand * 2 + len(self.env_cfg.fingertip_site_ids) * 3 + 3 + n_anchors * 3 * 2

    def _compute_privileged_dim(self) -> int:
        return 6 + self.env_cfg.hand_qpos_dof + len(self.env_cfg.fingertip_site_ids) * 3

    def _reference_ctrls(self, time_indices: np.ndarray) -> torch.Tensor:
        """Return one reference control row per vectorized environment."""
        indices = np.minimum(
            self.start_indices + np.asarray(time_indices, dtype=np.int64),
            self.ctrl_ref.shape[0] - 1,
        )
        return self.ctrl_ref[indices].to(str(self.ego_cfg.device))

    def _apply_residual(self, reference_ctrls: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        spec = self.env_cfg.residual
        residual = torch.clamp(spec.residual_scale * delta, -spec.residual_clip, spec.residual_clip)
        out = reference_ctrls.clone()
        out[:, list(spec.hand_control_indices)] += residual
        return out

    def _write_state(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        import warp as wp

        qpos = qpos.to(str(self.ego_cfg.device), torch.float32).contiguous()
        qvel = qvel.to(str(self.ego_cfg.device), torch.float32).contiguous()
        ctrl = self._reference_ctrls(self.time_indices)
        time = torch.zeros(self.num_envs, dtype=torch.float32, device=str(self.ego_cfg.device))
        with wp.ScopedDevice(self.env.device):
            wp.copy(self.env.data_wp.qpos, wp.from_torch(qpos))
            wp.copy(self.env.data_wp.qvel, wp.from_torch(qvel))
            wp.copy(self.env.data_wp.ctrl, wp.from_torch(ctrl))
            wp.copy(self.env.data_wp.time, wp.from_torch(time))
            mjwarp.kinematics(self.env.model_wp, self.env.data_wp)
        wp.synchronize()

    def _reset_worlds(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
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
            if self.ego_cfg.nq_obj == 6:
                pos_slice = slice(-6, -3)
            else:
                pos_slice = slice(-7, -4)
            qpos[idx, pos_slice] += self.rng.normal(
                0.0, self.env_cfg.reset_object_pos_noise_std, size=(3,)
            ).astype(np.float32)
            if self.ego_cfg.nq_obj == 6:
                qpos[idx, -3:] += self.rng.normal(
                    0.0, self.env_cfg.reset_object_rot_noise_std, size=(3,)
                ).astype(np.float32)
            else:
                # Small random rotation on the quaternion tail, then renormalize.
                q = qpos[idx, -4:].astype(np.float32)
                q += self.rng.normal(0.0, self.env_cfg.reset_object_rot_noise_std, size=(4,)).astype(np.float32)
                qpos[idx, -4:] = q / np.linalg.norm(q)

        self.time_indices[mask] = 0
        self._write_state(
            torch.as_tensor(qpos, device=str(self.ego_cfg.device)),
            torch.as_tensor(qvel, device=str(self.ego_cfg.device)),
        )

    def _build_observations(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import warp as wp

        qpos = self._mjwp.get_qpos(self.ego_cfg, self.env)
        qvel = self._mjwp.get_qvel(self.ego_cfg, self.env)
        site_xpos = wp.to_torch(self.env.data_wp.site_xpos)
        fingertip = site_xpos[:, list(self.env_cfg.fingertip_site_ids)]
        palm = site_xpos[:, self.env_cfg.palm_site_id]

        current_object = _object_pose_from_qpos(qpos, int(self.ego_cfg.nq_obj))
        goal_indices = np.minimum(
            self.start_indices + self.time_indices,
            self.qpos_ref.shape[0] - 1,
        )
        goal_qpos = self.qpos_ref[goal_indices].to(qpos.device)
        goal_object = _object_pose_from_qpos(goal_qpos, int(self.ego_cfg.nq_obj))
        current_anchors = _transform_anchors_torch(current_object[0], current_object[1], self.anchors)
        goal_anchors = _transform_anchors_torch(
            goal_object[0], goal_object[1], self.anchors
        )

        hand_qpos = qpos[:, : self.env_cfg.hand_qpos_dof]
        hand_qvel = qvel[:, : self.env_cfg.hand_qpos_dof]
        observation = torch.cat(
            [
                hand_qpos,
                hand_qvel,
                fingertip.reshape(self.num_envs, -1),
                palm,
                current_anchors.reshape(self.num_envs, -1),
                goal_anchors.reshape(self.num_envs, -1),
            ],
            dim=1,
        )
        if self.env_cfg.domain.obs_noise_std > 0.0:
            observation = observation + torch.randn_like(
                observation, device=observation.device
            ) * self.env_cfg.domain.obs_noise_std

        object_linear = qvel[:, -6:-3]
        object_angular = qvel[:, -3:]
        joint_forces = torch.zeros(self.num_envs, self.env_cfg.hand_qpos_dof, device=qpos.device)
        try:
            joint_forces = wp.to_torch(self.env.data_wp.qfrc_actuator)[:, : self.env_cfg.hand_qpos_dof]
        except AttributeError:
            # Some mjwarp builds expose actuator forces under a different name;
            # keep the privileged state zero rather than failing the rollout.
            joint_forces = torch.zeros_like(joint_forces)
        contact_forces = torch.zeros(self.num_envs, len(self.env_cfg.fingertip_site_ids), 3, device=qpos.device)
        privileged = {
            "object_linear_velocity": object_linear,
            "object_angular_velocity": object_angular,
            "joint_forces": joint_forces,
            "fingertip_contact_forces": contact_forces,
            "fingertip_positions": fingertip,
            "object_pose": current_object,
            "goal_object_pose": goal_object,
            "palm_position": palm,
        }
        return observation, privileged

    def _build_privileged_vector(self, privileged: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten the privileged critic state into the ``state_space`` vector.

        The asymmetric critic only consumes the privileged physics information:
        object linear/angular velocity, actuator forces on the 18 hand DOFs and
        per-fingertip contact forces.  The extra keys in ``privileged`` remain
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

    def _compute_reward(self, current_object, goal_object) -> torch.Tensor:
        current_anchors = _transform_anchors_torch(current_object[0], current_object[1], self.anchors)
        goal_anchors = _transform_anchors_torch(goal_object[0], goal_object[1], self.anchors)
        distance = torch.sum(torch.norm(goal_anchors - current_anchors, dim=-1), dim=-1)
        return torch.exp(-self.env_cfg.anchor.alpha * distance)

    def _compute_done(self, reward: torch.Tensor, privileged: dict[str, torch.Tensor]) -> torch.Tensor:
        distance = -torch.log(reward.clamp_min(1e-12)) / self.env_cfg.anchor.alpha
        fingertip_to_object = torch.norm(
            privileged["fingertip_positions"] - privileged["object_pose"][0].unsqueeze(1),
            dim=-1,
        )
        min_fingertip_to_object = fingertip_to_object.min(dim=1).values
        done = (distance > self.env_cfg.anchor.reset_distance_max_m) | (
            min_fingertip_to_object > self.env_cfg.fingertip_reset_distance_max_m
        )
        done = done | torch.as_tensor(
            self.time_indices + 1 >= self.env_cfg.max_episode_length,
            device=reward.device,
        )
        return done
