"""Actual MJWP transitions for the paper's two-chunk Replay→RL scheduler."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np

from video_to_spider.rl.residual_semantics import control_target_residuals


def _snapshot_value_equal(left, right):
    if hasattr(left, "detach") and hasattr(right, "detach"):
        left = left.detach().cpu().numpy()
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_snapshot_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_snapshot_value_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


class MJWPChunkBackend:
    """One-world correctness path; no object actuation or reference teleporting."""

    def __init__(self, env):
        if env.num_envs != 1 or np.any(env.start_indices != 0):
            raise ValueError("chunk validation requires one world and source row 0")
        self.env = env
        self.last_info = None
        self.validation_traces = []
        self._active_trace = None
        self.verified_restore_count = 0

    def begin_trial(self, mode, start, end):
        if self._active_trace is not None:
            raise RuntimeError("a validation trace is already active")
        self._active_trace = {
            "schema": "taco_replay_rl_validation_trace_v4",
            "mode": mode,
            "start": int(start),
            "lookahead_end": int(end),
            "residual_semantics": {
                "requested_residual": (
                    "requested control target minus reference control target"
                ),
                "effective_residual_after_ctrlrange": (
                    "control-target offset remaining after enabled actuator ctrlrange; "
                    "not realized qpos motion"
                ),
                "residual_lost_to_ctrlrange": (
                    "signed requested residual minus effective residual"
                ),
            },
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

    def verify_restored_snapshot(self, expected):
        """Fail closed if a CPU/GPU boundary transfer changes any saved field."""
        actual = self.snapshot()
        if expected.keys() != actual.keys():
            raise ValueError("restored MJWP snapshot keys differ")
        mismatches = [
            key for key in expected
            if not _snapshot_value_equal(expected[key], actual[key])
        ]
        if mismatches:
            preview = ", ".join(mismatches[:8])
            raise ValueError(f"restored MJWP snapshot differs: {preview}")
        self.verified_restore_count += 1

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
            residuals = control_target_residuals(
                self.env.env.model_cpu,
                reference_ctrl[None],
                commanded_ctrl[None],
            )
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
                "command_reference_endpoint": int(
                    self.last_info["command_reference_endpoint"][0]
                ),
                "reward_reference_endpoint": int(
                    self.last_info["reward_reference_endpoint"][0]
                ),
                "next_observation_goal_reference_endpoint": int(
                    self.last_info["next_observation_goal_reference_endpoint"][0]
                ),
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
                "requested_residual": residuals["requested_residual"][0].tolist(),
                "effective_residual_after_ctrlrange": (
                    residuals["effective_residual_after_ctrlrange"][0].tolist()
                ),
                "residual_lost_to_ctrlrange": (
                    residuals["residual_lost_to_ctrlrange"][0].tolist()
                ),
            })
        # A window timeout is not object-tracking failure. No autoreset may hide
        # the physical state at either the commit point or a failed endpoint.
        return finite and not bool(self.last_info["terminated"][0])


class MJWPIndependentTrainingBackend:
    """Non-authoritative PPO backend with one complete MJWP buffer per world."""

    def __init__(self, env):
        if env.num_envs < 2:
            raise ValueError("independent training backend requires multiple worlds")
        self.env = env
        self.verified_restore_count = 0
        self.restore_audits = []

    def snapshot(self):
        return self.env.get_env_states()

    def restore(self, state):
        self.env.set_env_state(state)

    def verify_restored_snapshot(self, expected):
        states = self.snapshot()
        mismatches = {}
        for world_index, actual in enumerate(states):
            if expected.keys() != actual.keys():
                mismatches[world_index] = ["<snapshot keys differ>"]
                continue
            failed = [
                key for key in expected
                if not _snapshot_value_equal(expected[key], actual[key])
            ]
            if failed:
                mismatches[world_index] = failed
        if mismatches:
            preview = "; ".join(
                f"world {index}: {', '.join(names[:4])}"
                for index, names in mismatches.items()
            )
            raise ValueError(f"independent training restore differs: {preview}")
        self.restore_audits.append({
            "worlds": len(states),
            "snapshot_keys_per_world": len(expected),
            "warp_state_fields_per_world": len(expected["warp_state_keys"]),
            "all_worlds_bitwise_equal_to_source": True,
        })
        self.verified_restore_count += 1


def replay_action(backend, reference_step):
    """Zero residual uses ctrl[t+1] in the adapter, never repeated endpoint t."""
    return np.zeros((1, backend.env.env_cfg.residual.hand_dof), dtype=np.float32)


def _model_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _AgentPolicy:
    """Own one evaluation agent and its optional temporary CPU workspace."""

    def __init__(self, agent, audit, temporary_directory=None):
        self.agent = agent
        self.audit = audit
        self.temporary_directory = temporary_directory
        self.closed = False

    def __call__(self, current, reference_step):
        del reference_step
        observation = self.agent.obs_to_tensors(current.observation())
        deterministic = getattr(
            self.agent, "get_deterministic_action_values", None
        )
        result = (
            deterministic(observation)
            if deterministic is not None
            else self.agent.get_action_values(observation)
        )
        self.agent.rnn_states = result["rnn_states"]
        action = result.get("deterministic_actions", result["mus"])
        return self.agent.preprocess_actions(action)

    def close(self):
        if self.closed:
            return
        if self.agent.writer is not None:
            self.agent.writer.close()
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
        self.closed = True


def train_chunk_ppo(
    backend,
    start,
    end,
    output,
    *,
    epochs=2,
    horizon=40,
    seed=0,
    validation_env=None,
    action_distribution_spec=None,
    credit_audit_dir=None,
    credit_probe_panel=None,
):
    """Reuse the existing official trainer; reset every rollout to this boundary.

    Network, optimizer and recurrent state are fresh for each failed chunk in
    this local diagnostic. This is not an unpublished EgoEngine hyperparameter.
    """
    from dataclasses import replace
    import torch
    from run_mjwp_ppo import (PpoAgent, _build_ppo_config, _build_network_config,
                             _build_asymmetric_critic_config)

    if action_distribution_spec is None:
        agent_class = PpoAgent
        agent_kwargs = {}
    else:
        if not action_distribution_spec.optimizer_training_authorized:
            raise ValueError(
                "state-feasible optimizer training lacks an exact authorization"
            )
        agent_kwargs = {"distribution_spec": action_distribution_spec}
        if credit_audit_dir is None:
            from video_to_spider.rl.state_feasible_truncated_gaussian import (
                StateFeasibleTruncatedGaussianPpoAgent,
            )
            agent_class = StateFeasibleTruncatedGaussianPpoAgent
        else:
            from video_to_spider.rl.credit_audit import (
                CreditInstrumentedTruncatedGaussianPpoAgent,
            )
            agent_class = CreditInstrumentedTruncatedGaussianPpoAgent
            agent_kwargs.update(
                credit_audit_dir=Path(credit_audit_dir),
                credit_probe_panel=credit_probe_panel,
            )

    env = backend.env
    env.set_chunk_reset(start=start, end=end)
    if horizon < 4 or horizon % 4 or epochs < 1:
        raise ValueError("positive epochs and horizon divisible by recurrent sequence length 4 required")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    training_worlds = int(env.num_envs)
    config = _build_ppo_config(num_envs=training_worlds, horizon_length=horizon, seq_length=4,
        max_epochs=epochs, learning_rate=1e-4, device=str(env.ego_cfg.device),
        asymmetric_critic=_build_asymmetric_critic_config(training_worlds * horizon))
    if action_distribution_spec is not None:
        config = replace(config, clip_actions=False)
    agent = agent_class(experiment_dir=output, ppo_config=config,
                        network_config=_build_network_config(4), env=env,
                        **agent_kwargs)
    env.enable_training_trace(output / "training_visitation")
    training_visitation = None
    try:
        agent.train()
    except BaseException as error:
        try:
            env.finalize_training_trace(completed=False)
        except Exception as trace_error:
            error.add_note(f"training-trace finalization also failed: {trace_error}")
        raise
    else:
        training_visitation = env.finalize_training_trace(completed=True)
    finally:
        if agent.writer is not None:
            agent.writer.close()

    credit_audit = (
        agent.finalize_credit_audit()
        if credit_audit_dir is not None
        else None
    )

    actor_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in agent.model.state_dict().items()
    }
    actor_sha256 = _model_state_sha256(actor_state)
    curriculum_audit = (
        env.tail_curriculum_audit()
        if hasattr(env, "tail_curriculum_audit")
        else None
    )
    state_feasible_action_audit = (
        env.state_feasible_action_audit()
        if action_distribution_spec is not None
        and hasattr(env, "state_feasible_action_audit")
        else None
    )
    observation_normalization_audit = (
        agent.observation_normalization_audit()
        if hasattr(agent, "observation_normalization_audit")
        else None
    )
    checkpoints = []
    for checkpoint in sorted((output / "nn").glob("*.pth")):
        checkpoints.append({
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })

    if validation_env is None:
        agent.set_eval()
        agent.rnn_states = [
            state.to(agent.device).zero_()
            for state in agent.model.get_default_rnn_state()
        ]
        return _AgentPolicy(agent, {
            "training_device": str(env.ego_cfg.device),
            "training_worlds": training_worlds,
            "policy_inference_device": str(agent.device),
            "actor_state_sha256": actor_sha256,
            "checkpoint_artifacts": checkpoints,
            "training_visitation": training_visitation,
            "credit_audit": credit_audit,
            "tail_curriculum": curriculum_audit,
            "state_feasible_action": state_feasible_action_audit,
            "observation_normalization": observation_normalization_audit,
        })

    if str(validation_env.ego_cfg.device) != "cpu":
        raise ValueError("deterministic validation policy requires a CPU MJWP environment")
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="egoengine_cpu_policy_validation_"
    )
    cpu_config = _build_ppo_config(
        num_envs=1,
        horizon_length=horizon,
        seq_length=4,
        max_epochs=epochs,
        learning_rate=1e-4,
        device="cpu",
        asymmetric_critic=None,
    )
    if action_distribution_spec is not None:
        cpu_config = replace(cpu_config, clip_actions=False)
    cpu_agent_class = agent_class
    cpu_agent_kwargs = dict(agent_kwargs)
    if credit_audit_dir is not None:
        from video_to_spider.rl.state_feasible_truncated_gaussian import (
            StateFeasibleTruncatedGaussianPpoAgent,
        )
        cpu_agent_class = StateFeasibleTruncatedGaussianPpoAgent
        cpu_agent_kwargs = {"distribution_spec": action_distribution_spec}
    cpu_agent = cpu_agent_class(
        experiment_dir=Path(temporary_directory.name),
        ppo_config=cpu_config,
        network_config=_build_network_config(4),
        env=validation_env,
        **cpu_agent_kwargs,
    )
    cpu_agent.model.load_state_dict(actor_state)
    transferred_sha256 = _model_state_sha256(cpu_agent.model.state_dict())
    if transferred_sha256 != actor_sha256:
        cpu_agent.writer.close()
        temporary_directory.cleanup()
        raise ValueError("GPU-trained actor changed during transfer to CPU validation")
    cpu_agent.set_eval()
    cpu_agent.rnn_states = [
        state.to("cpu").zero_()
        for state in cpu_agent.model.get_default_rnn_state()
    ]
    return _AgentPolicy(cpu_agent, {
        "training_device": str(env.ego_cfg.device),
        "training_worlds": training_worlds,
        "policy_inference_device": "cpu",
        "actor_state_sha256": actor_sha256,
        "transferred_actor_state_sha256": transferred_sha256,
        "actor_transfer_bitwise_equal": True,
        "deterministic_mean_action": action_distribution_spec is None,
        "deterministic_truncated_mode_action": action_distribution_spec is not None,
        "recurrent_state_reset_to_zero": True,
        "checkpoint_artifacts": checkpoints,
        "training_visitation": training_visitation,
        "credit_audit": credit_audit,
        "tail_curriculum": curriculum_audit,
        "state_feasible_action": state_feasible_action_audit,
        "observation_normalization": observation_normalization_audit,
    }, temporary_directory)
