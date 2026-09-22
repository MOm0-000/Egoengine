"""Actual MJWP transitions for the paper's two-chunk Replay→RL scheduler."""

import numpy as np


class MJWPChunkBackend:
    """One-world correctness path; no object actuation or reference teleporting."""

    def __init__(self, env):
        if env.num_envs != 1 or np.any(env.start_indices != 0):
            raise ValueError("chunk validation requires one world and source row 0")
        self.env = env
        self.last_info = None
        self.validation_traces = []
        self._active_trace = None

    def begin_trial(self, mode, start, end):
        if self._active_trace is not None:
            raise RuntimeError("a validation trace is already active")
        self._active_trace = {
            "mode": mode,
            "start": int(start),
            "lookahead_end": int(end),
            "steps": [],
        }

    def end_trial(self, feasible, validated_steps, *, error=None):
        if self._active_trace is None:
            raise RuntimeError("no validation trace is active")
        trace = self._active_trace
        trace.update(feasible=bool(feasible), validated_steps=int(validated_steps))
        if error is not None:
            trace["error"] = error
        first = next((step for step in trace["steps"] if step["terminated"]), None)
        if first is not None:
            failed = [
                role for role, terminated in zip(
                    first["tracked_object_roles"], first["object_terminated"], strict=True
                ) if terminated
            ]
            trace["first_failure"] = {
                "control_interval": first["control_interval"],
                "endpoint": first["endpoint"],
                "object_roles": failed,
                "reason": "tracking_boundary",
            }
        elif not feasible and trace["steps"]:
            first = trace["steps"][-1]
            trace["first_failure"] = {
                "control_interval": first["control_interval"],
                "endpoint": first["endpoint"],
                "object_roles": [],
                "reason": "nonfinite_state" if not first["finite"] else "backend_rejected",
            }
        else:
            trace["first_failure"] = None
        self.validation_traces.append(trace)
        self._active_trace = None

    def snapshot(self):
        return self.env.get_env_state()

    def restore(self, state):
        self.env.set_env_state(state)

    def observation(self):
        obs, privileged = self.env._build_observations()
        return self.env._pack_observation(obs, privileged)

    def step(self, action, reference_step):
        if int(self.env.time_indices[0]) != reference_step:
            raise ValueError("scheduler and simulator reference cursors differ")
        _, reward, _, self.last_info = self.env.step(action, auto_reset=False)
        qpos = self.env._mjwp.get_qpos(self.env.ego_cfg, self.env.env)
        qvel = self.env._mjwp.get_qvel(self.env.ego_cfg, self.env.env)
        finite = bool(qpos.isfinite().all() and qvel.isfinite().all()
                      and np.isfinite(reward).all()
                      and np.isfinite(self.last_info["object_tracking_error"]).all())
        if self._active_trace is not None:
            def row(name):
                return np.asarray(self.last_info[name][0]).tolist()

            endpoint_qpos = qpos[0].detach().cpu().numpy()
            endpoint_qvel = qvel[0].detach().cpu().numpy()
            commanded_ctrl = self.env._last_ctrl[0].detach().cpu().numpy()
            raw_residual = self.env._last_action[0].detach().cpu().numpy()
            reference_ctrl = self.env._reference_ctrls(
                self.env.time_indices, offset=0
            )[0].detach().cpu().numpy()
            position_error = np.asarray(self.last_info["object_position_error"][0])
            rotation_error = np.asarray(self.last_info["object_rotation_error"][0])
            position_threshold = self.env.objective.independent_position_threshold_m
            rotation_threshold = self.env.objective.independent_rotation_threshold_rad
            if position_threshold is None or rotation_threshold is None:
                independent_position_pass = None
                independent_rotation_pass = None
                independent_threshold_pass = None
            else:
                independent_position_pass = (position_error <= position_threshold).tolist()
                independent_rotation_pass = (rotation_error <= rotation_threshold).tolist()
                independent_threshold_pass = (
                    (position_error <= position_threshold)
                    & (rotation_error <= rotation_threshold)
                ).tolist()

            self._active_trace["steps"].append({
                "control_interval": int(reference_step),
                "endpoint": int(self.env.time_indices[0]),
                "tracked_object_indices": list(self.last_info["tracked_object_indices"]),
                "tracked_object_roles": list(self.last_info["tracked_object_roles"]),
                "position_error_m": row("object_position_error"),
                "rotation_error_rad": row("object_rotation_error"),
                "tracking_error": row("object_tracking_error_per_object"),
                "objective_metric_name": self.env.objective.tracking_metric_name,
                "objective_score": row("object_tracking_error_per_object"),
                "independent_position_threshold_m": position_threshold,
                "independent_rotation_threshold_rad": rotation_threshold,
                "independent_position_pass": independent_position_pass,
                "independent_rotation_pass": independent_rotation_pass,
                "independent_threshold_pass": independent_threshold_pass,
                "tracking_reward": row("object_tracking_reward_per_object"),
                "object_terminated": row("object_terminated"),
                "contact_bonus_per_hand_object": row("contact_bonus_per_hand_object"),
                "aggregate_tracking_error": float(self.last_info["object_tracking_error"][0]),
                "aggregate_tracking_reward": float(self.last_info["aggregate_tracking_reward"][0]),
                "aggregate_contact_bonus": float(self.last_info["contact_score"][0]),
                "lift_reward": float(self.last_info["lift_reward"][0]),
                "total_reward": float(reward[0]),
                "terminated": bool(self.last_info["terminated"][0]),
                "finite": finite,
                "endpoint_qpos": endpoint_qpos.tolist(),
                "endpoint_qvel": endpoint_qvel.tolist(),
                "commanded_ctrl": commanded_ctrl.tolist(),
                "raw_residual_action": raw_residual.tolist(),
                "applied_residual": (commanded_ctrl - reference_ctrl).tolist(),
            })
        # A window timeout is not object-tracking failure. No autoreset may hide
        # the physical state at either the commit point or a failed endpoint.
        return finite and not bool(self.last_info["terminated"][0])


def replay_action(backend, reference_step):
    """Zero residual uses ctrl[t+1] in the adapter, never repeated endpoint t."""
    return np.zeros((1, backend.env.env_cfg.residual.hand_dof), dtype=np.float32)


def train_chunk_ppo(backend, start, end, output, *, epochs=2, horizon=40, seed=0):
    """Reuse the existing official trainer; reset every rollout to this boundary.

    Network, optimizer and recurrent state are fresh for each failed chunk in
    this local diagnostic. This is not an unpublished EgoEngine hyperparameter.
    """
    import torch
    from run_mjwp_ppo import (PpoAgent, _build_ppo_config, _build_network_config,
                             _build_asymmetric_critic_config)

    env = backend.env
    env.set_chunk_reset(start=start, end=end)
    if horizon < 4 or horizon % 4 or epochs < 1:
        raise ValueError("positive epochs and horizon divisible by recurrent sequence length 4 required")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = _build_ppo_config(num_envs=1, horizon_length=horizon, seq_length=4,
        max_epochs=epochs, learning_rate=1e-4, device=str(env.ego_cfg.device),
        asymmetric_critic=_build_asymmetric_critic_config(horizon))
    agent = PpoAgent(experiment_dir=output, ppo_config=config,
                     network_config=_build_network_config(4), env=env)
    try:
        agent.train()
    finally:
        agent.writer.close()
    agent.set_eval()
    agent.rnn_states = [s.to(agent.device).zero_() for s in agent.model.get_default_rnn_state()]

    def policy(current, reference_step):
        result = agent.get_action_values(agent.obs_to_tensors(current.observation()))
        agent.rnn_states = result["rnn_states"]
        # Deterministic mean action for validation; training remains stochastic.
        return agent.preprocess_actions(result["mus"])

    return policy
