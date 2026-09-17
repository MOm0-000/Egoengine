"""Actual MJWP transitions for the paper's two-chunk Replay→RL scheduler."""

import numpy as np


class MJWPChunkBackend:
    """One-world correctness path; no object actuation or reference teleporting."""

    def __init__(self, env):
        if env.num_envs != 1 or np.any(env.start_indices != 0):
            raise ValueError("chunk validation requires one world and source row 0")
        self.env = env
        self.last_info = None

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
