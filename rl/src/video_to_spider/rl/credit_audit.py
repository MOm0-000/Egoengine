"""Passive, lossless credit logging for the corrected Pour PPO experiment.

This module does not define a new optimizer or objective.  It subclasses the
already-gated state-feasible PPO only to persist tensors that the existing
rollout, GAE and optimizer code already computes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .state_feasible_truncated_gaussian import (
    StateFeasibleTruncatedGaussianPpoAgent,
    deterministic_truncated_action,
)


SCHEMA = "taco_corrected_fresh_ppo_credit_instrumentation_v1"


def load_fixed_probe_panel(path: Path) -> dict[str, Any]:
    """Load immutable PPO/Replay observations from the prior attribution run."""
    with np.load(path, allow_pickle=False) as archive:
        sources = np.asarray(archive["source_endpoints"], dtype=np.int64)
        selected = np.flatnonzero((sources >= 42) & (sources <= 47))
        if sources[selected].tolist() != [42, 43, 44, 45, 46, 47]:
            raise ValueError("fixed probe artifact must contain sources 42..47")
        observations = []
        lows = []
        highs = []
        hidden_0 = []
        hidden_1 = []
        labels = []
        for prefix in ("ppo", "replay"):
            observations.append(np.asarray(archive[f"{prefix}_raw_observation"])[selected])
            lows.append(np.asarray(archive[f"{prefix}_state_low"])[selected])
            highs.append(np.asarray(archive[f"{prefix}_state_high"])[selected])
            hidden_0.append(np.asarray(archive[f"{prefix}_pre_forward_hidden_0"])[selected])
            hidden_1.append(np.asarray(archive[f"{prefix}_pre_forward_hidden_1"])[selected])
            labels.extend(f"{prefix}_source_{source}" for source in sources[selected])
        # The zero-memory panel distinguishes current-observation sensitivity
        # from the historical final-actor recurrent context without inventing a
        # new physical state.
        ppo_obs = np.asarray(archive["ppo_raw_observation"])[selected]
        ppo_low = np.asarray(archive["ppo_state_low"])[selected]
        ppo_high = np.asarray(archive["ppo_state_high"])[selected]
        observations.append(ppo_obs)
        lows.append(ppo_low)
        highs.append(ppo_high)
        hidden_0.append(np.zeros_like(hidden_0[0]))
        hidden_1.append(np.zeros_like(hidden_1[0]))
        labels.extend(f"zero_hidden_ppo_source_{source}" for source in sources[selected])

    def hidden(values: list[np.ndarray]) -> np.ndarray:
        # Stored panel rows are probe x layer x hidden; the actor expects
        # layer x probe x hidden.
        return np.concatenate(values, axis=0).transpose(1, 0, 2)

    return {
        "labels": labels,
        "raw_observation": np.concatenate(observations, axis=0).astype(np.float32),
        "action_low": np.concatenate(lows, axis=0).astype(np.float32),
        "action_high": np.concatenate(highs, axis=0).astype(np.float32),
        "rnn_states": [
            hidden(hidden_0).astype(np.float32),
            hidden(hidden_1).astype(np.float32),
        ],
        "source_artifact": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "limitations": [
            "Probe hidden states come from the frozen final corrected actor, not the fresh actor.",
            "No exact full observation-plus-hidden artifact exists for the prior rescue branches; they are not fabricated here.",
        ],
    }


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
    }


def write_state(path: Path, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    raw = buffer.getvalue()
    artifact = gzip.compress(raw, compresslevel=9, mtime=0)
    path.write_bytes(artifact)
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "model_state_sha256": model_state_sha256(state),
        "bytes": len(artifact),
    }


def write_xor_patch(
    path: Path,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Write an exact, compact patch from one model state to the next."""
    if before.keys() != after.keys():
        raise ValueError("actor state keys changed during one update")
    arrays: dict[str, np.ndarray] = {}
    manifest = []
    for index, name in enumerate(sorted(before)):
        left = before[name].contiguous().numpy()
        right = after[name].contiguous().numpy()
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"actor tensor contract changed: {name}")
        left_bytes = left.reshape(-1).view(np.uint8)
        right_bytes = right.reshape(-1).view(np.uint8)
        key = f"tensor_{index:03d}"
        arrays[key] = np.bitwise_xor(left_bytes, right_bytes)
        manifest.append({
            "key": key,
            "name": name,
            "dtype": str(left.dtype),
            "shape": list(left.shape),
            "byte_count": int(left_bytes.size),
        })
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "before_model_state_sha256": model_state_sha256(before),
        "after_model_state_sha256": model_state_sha256(after),
    }


def apply_xor_patch(
    before: dict[str, torch.Tensor], path: Path
) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        result = {}
        for row in manifest:
            tensor = before[row["name"]].detach().cpu().contiguous()
            source = tensor.numpy()
            xor = np.asarray(archive[row["key"]], dtype=np.uint8)
            restored = np.bitwise_xor(source.reshape(-1).view(np.uint8), xor)
            restored = restored.view(np.dtype(row["dtype"])).reshape(row["shape"])
            result[row["name"]] = torch.from_numpy(restored.copy())
    return result


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _world_major(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert time x world x ... into the official world-major flat order."""
    array = _numpy(value)
    if array.ndim < 2:
        raise ValueError("world-major conversion requires time and world axes")
    return array.swapaxes(0, 1).reshape((-1,) + array.shape[2:])


def _termination_outcomes(
    outcome_endpoint: np.ndarray, done: np.ndarray, worlds: int, horizon: int
) -> np.ndarray:
    endpoints = outcome_endpoint.reshape(worlds, horizon)
    dones = done.reshape(worlds, horizon)
    result = np.full((worlds, horizon), -1, dtype=np.int32)
    for world in range(worlds):
        for time in range(horizon):
            future = np.flatnonzero(dones[world, time:])
            if len(future):
                index = time + int(future[0])
                result[world, time] = int(endpoints[world, index])
    return result.reshape(-1)


class CreditAuditRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        worlds: int,
        horizon: int,
        right_wrist_y_index: int,
        probe_panel: dict[str, Any] | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self.patch_dir = self.output_dir / "actor_state_patches"
        self.patch_dir.mkdir()
        self.worlds = int(worlds)
        self.horizon = int(horizon)
        self.right_wrist_y_index = int(right_wrist_y_index)
        self.probe_panel = probe_panel
        self.epoch = 0
        self.update_in_epoch = 0
        self.global_actor_update = 0
        self.rollout_steps: list[dict[str, Any]] = []
        self.discount: dict[str, np.ndarray] | None = None
        self.dataset: dict[str, np.ndarray] | None = None
        self.update_reports: list[dict[str, Any]] = []
        self.probe_reports: list[dict[str, Any]] = []
        self.epoch_reports: list[dict[str, Any]] = []
        self.initial_actor: dict[str, Any] | None = None
        self._initial_state: dict[str, torch.Tensor] | None = None
        self._last_state: dict[str, torch.Tensor] | None = None

    def begin_epoch(self, agent: Any) -> None:
        if self.rollout_steps:
            raise RuntimeError("previous credit-audit epoch was not finalized")
        self.epoch = int(agent.epoch_num)
        self.update_in_epoch = 0
        self.discount = None
        self.dataset = None
        if self.initial_actor is None:
            self._initial_state = cpu_state(agent.model.state_dict())
            self._last_state = self._initial_state
            self.initial_actor = write_state(
                self.output_dir / "initial_actor_state.pt.gz", self._initial_state
            )
        self.record_probes(agent, stage="epoch_start")

    def actor_forward(
        self,
        *,
        raw_observation: Any,
        processed_observation: torch.Tensor,
        normalized_observation: torch.Tensor,
        rnn_input: list[torch.Tensor],
        rnn_output: list[torch.Tensor],
        result: dict[str, torch.Tensor],
    ) -> None:
        states = raw_observation.get("states") if isinstance(raw_observation, dict) else None
        self._pending = {
            "raw_observation": _numpy(raw_observation["obs"]),
            "privileged_state": None if states is None else _numpy(states),
            "processed_observation": _numpy(processed_observation),
            "normalized_observation": _numpy(normalized_observation),
            "rnn_input": [_numpy(value) for value in rnn_input],
            "rnn_output": [_numpy(value) for value in rnn_output],
            "actor_mu": _numpy(result["mus"]),
            "actor_sigma": _numpy(result["sigmas"]),
            "critic_value": _numpy(result["values"]),
        }

    def sampled_action(self, result: dict[str, torch.Tensor]) -> None:
        if not hasattr(self, "_pending"):
            raise RuntimeError("sampled action has no matching actor forward")
        self._pending.update({
            "sampled_action": _numpy(result["actions"]),
            "action_low": _numpy(result["action_lows"]),
            "action_high": _numpy(result["action_highs"]),
            "old_neglogp": _numpy(result["neglogpacs"]),
        })

    def env_result(self, result: tuple[Any, torch.Tensor, torch.Tensor, dict]) -> None:
        if not hasattr(self, "_pending"):
            raise RuntimeError("environment result has no matching sampled action")
        _, reward, done, info = result
        row = self._pending
        del self._pending
        row.update({
            "reward": _numpy(reward),
            "done": _numpy(done),
            "source_endpoint": np.asarray(info["source_reference_endpoint"]),
            "outcome_endpoint": np.asarray(info["outcome_reference_endpoint"]),
            "command_reference_endpoint": np.asarray(info["command_reference_endpoint"]),
            "reward_reference_endpoint": np.asarray(info["reward_reference_endpoint"]),
            "next_observation_goal_reference_endpoint": np.asarray(
                info["next_observation_goal_reference_endpoint"]
            ),
            "position_error": np.asarray(info["object_position_error"]),
            "rotation_error": np.asarray(info["object_rotation_error"]),
            "tracking_score": np.asarray(info["object_tracking_error_per_object"]),
            "tracking_reward": np.asarray(info["object_tracking_reward_per_object"]),
            "aggregate_tracking_reward": np.asarray(info["aggregate_tracking_reward"]),
            "aggregate_contact_bonus": np.asarray(info["aggregate_contact_bonus"]),
            "lift_reward": np.asarray(info["lift_reward"]),
            "contact_flags": np.asarray(info["contact_flags"]),
            "tracking_terminated": np.asarray(info["terminated"]),
            "time_out": np.asarray(info["time_outs"]),
        })
        self.rollout_steps.append(row)

    def gae(
        self,
        *,
        fdones: torch.Tensor,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        gamma: float,
        tau: float,
    ) -> None:
        delta = torch.zeros_like(rewards)
        for time in range(self.horizon):
            if time == self.horizon - 1:
                next_done = fdones
                next_value = last_values
            else:
                next_done = dones[time + 1].float()
                next_value = values[time + 1]
            nonterminal = (1.0 - next_done).unsqueeze(1)
            delta[time] = rewards[time] + gamma * next_value * nonterminal - values[time]
        self.discount = {
            "bootstrap_value": _numpy(last_values),
            "final_done": _numpy(fdones),
            "gae_delta": _world_major(delta),
            "gae_advantage": _world_major(advantages),
        }

    def prepared_dataset(
        self,
        *,
        raw_advantage: torch.Tensor,
        normalized_advantage: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> None:
        self.dataset = {
            "raw_advantage": _numpy(raw_advantage),
            "normalized_advantage": _numpy(normalized_advantage),
            "return": _numpy(returns),
            "dataset_old_value": _numpy(old_values),
            "sample_id": _numpy(sample_ids),
        }

    def record_probes(self, agent: Any, *, stage: str) -> dict[str, Any] | None:
        if self.probe_panel is None:
            return None
        raw = torch.as_tensor(
            self.probe_panel["raw_observation"], dtype=torch.float32, device=agent.device
        )
        hidden = [
            torch.as_tensor(value, dtype=torch.float32, device=agent.device)
            for value in self.probe_panel["rnn_states"]
        ]
        rms = getattr(agent.model, "running_mean_std", None)
        rms_training = None if rms is None else rms.training
        if rms is not None:
            rms.eval()
        with torch.no_grad():
            processed = agent._preproc_obs(raw)
            normalized = agent.model.norm_obs(processed)
            mu, logstd, _, _ = agent.model.a2c_network({
                "obs": normalized,
                "rnn_states": hidden,
            })
            low = torch.as_tensor(
                self.probe_panel["action_low"], dtype=mu.dtype, device=agent.device
            )
            high = torch.as_tensor(
                self.probe_panel["action_high"], dtype=mu.dtype, device=agent.device
            )
            deterministic = deterministic_truncated_action(mu, low, high)
        if rms is not None and rms_training:
            rms.train()
        report = {
            "epoch": self.epoch,
            "global_actor_update": self.global_actor_update,
            "stage": stage,
            "labels": list(self.probe_panel["labels"]),
            "mu_y": _numpy(mu[:, self.right_wrist_y_index]).tolist(),
            "sigma_y": _numpy(torch.exp(logstd)[:, self.right_wrist_y_index]).tolist(),
            "deterministic_action_y": _numpy(
                deterministic[:, self.right_wrist_y_index]
            ).tolist(),
            "normalization_preclip_z_y_sensitive_inputs": None,
        }
        self.probe_reports.append(report)
        return report

    def actor_update(
        self,
        agent: Any,
        *,
        input_dict: dict[str, Any],
        current: dict[str, torch.Tensor],
        ratio: torch.Tensor,
        surr1: torch.Tensor,
        surr2: torch.Tensor,
        actor_loss_rows: torch.Tensor,
        critic_loss_rows: torch.Tensor,
        entropy_rows: torch.Tensor,
        total_loss: torch.Tensor,
        gradient_norm: float,
        before_forward: dict[str, torch.Tensor],
        after_forward: dict[str, torch.Tensor],
        after_update: dict[str, torch.Tensor],
    ) -> None:
        self.global_actor_update += 1
        self.update_in_epoch += 1
        stem = f"epoch_{self.epoch:04d}_update_{self.update_in_epoch:02d}"
        forward_patch = write_xor_patch(
            self.patch_dir / f"{stem}_forward_state_patch.npz",
            before_forward,
            after_forward,
        )
        optimizer_patch = write_xor_patch(
            self.patch_dir / f"{stem}_optimizer_state_patch.npz",
            after_forward,
            after_update,
        )
        if forward_patch["before_model_state_sha256"] != model_state_sha256(self._last_state):
            raise RuntimeError("actor patch chain does not begin at the previous state")
        self._last_state = after_update
        ids = _numpy(input_dict["audit_sample_id"]).astype(np.int32)
        arrays = {
            "sample_id": ids,
            "ratio": _numpy(ratio),
            "unclipped_surrogate": _numpy(surr1),
            "clipped_surrogate": _numpy(surr2),
            "actor_loss_per_sample": _numpy(actor_loss_rows),
            "critic_loss_per_sample": _numpy(critic_loss_rows),
            "entropy_per_sample": _numpy(entropy_rows),
            "new_neglogp": _numpy(current["neglogp"]),
            "new_mu": _numpy(current["mu"]),
            "new_sigma": _numpy(current["sigma"]),
        }
        npz = self.output_dir / f"{stem}_ppo_rows.npz"
        np.savez_compressed(npz, **arrays)
        probes = self.record_probes(agent, stage=f"after_actor_update_{self.update_in_epoch}")
        report = {
            "epoch": self.epoch,
            "update_in_epoch": self.update_in_epoch,
            "global_actor_update": self.global_actor_update,
            "minibatch_index": 0,
            "sample_count": int(len(ids)),
            "sample_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "actor_loss": float(actor_loss_rows.mean().detach().cpu()),
            "critic_loss": float(critic_loss_rows.mean().detach().cpu()),
            "entropy": float(entropy_rows.mean().detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "gradient_norm_before_clip": float(gradient_norm),
            "ratio": {
                "min": float(ratio.min().detach().cpu()),
                "mean": float(ratio.mean().detach().cpu()),
                "max": float(ratio.max().detach().cpu()),
            },
            "forward_state_patch": forward_patch,
            "optimizer_state_patch": optimizer_patch,
            "row_artifact": {
                "path": str(npz.resolve()),
                "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            },
            "probe": probes,
        }
        path = self.output_dir / f"{stem}_summary.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        self.update_reports.append(report)

    def critic_update(
        self,
        *,
        mini_epoch: int,
        loss: float,
        before_sha256: str,
        after_sha256: str,
    ) -> None:
        path = self.output_dir / f"epoch_{self.epoch:04d}_critic_updates.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({
                "epoch": self.epoch,
                "mini_epoch": int(mini_epoch),
                "minibatch_index": 0,
                "loss": float(loss),
                "before_model_state_sha256": before_sha256,
                "after_model_state_sha256": after_sha256,
            }, sort_keys=True) + "\n")

    def finalize_epoch(self, agent: Any) -> None:
        if len(self.rollout_steps) != self.horizon:
            raise RuntimeError("credit audit did not receive the complete rollout")
        if self.discount is None or self.dataset is None:
            raise RuntimeError("credit audit is missing GAE or dataset tensors")
        if self.update_in_epoch != int(agent.cfg.mini_epochs):
            raise RuntimeError("credit audit did not observe every actor update")
        names = sorted(self.rollout_steps[0])
        rollout = {}
        for name in names:
            if name in {"rnn_input", "rnn_output"}:
                for state_index in range(len(self.rollout_steps[0][name])):
                    stacked = np.stack([
                        row[name][state_index].squeeze(0)
                        for row in self.rollout_steps
                    ])
                    rollout[f"{name}_{state_index}"] = _world_major(stacked)
                continue
            value = self.rollout_steps[0][name]
            if value is None:
                continue
            rollout[name] = _world_major(np.stack([row[name] for row in self.rollout_steps]))
        rollout.update(self.discount)
        rollout.update(self.dataset)
        rollout["termination_outcome_endpoint"] = _termination_outcomes(
            rollout["outcome_endpoint"], rollout["done"], self.worlds, self.horizon
        )
        npz = self.output_dir / f"epoch_{self.epoch:04d}_credit_rows.npz"
        np.savez_compressed(npz, **rollout)
        y = rollout["sampled_action"][:, self.right_wrist_y_index]
        bins = (-1.0, -0.5, 0.0, 0.5, 1.0)
        bin_rows = []
        for index in range(len(bins) - 1):
            if index == len(bins) - 2:
                selected = (y >= bins[index]) & (y <= bins[index + 1])
            else:
                selected = (y >= bins[index]) & (y < bins[index + 1])
            def mean(name: str) -> float | None:
                return float(np.asarray(rollout[name])[selected].mean()) if selected.any() else None
            bin_rows.append({
                "interval": [bins[index], bins[index + 1]],
                "count": int(selected.sum()),
                "mean_reward": mean("reward"),
                "mean_return": mean("return"),
                "mean_value": mean("critic_value"),
                "mean_raw_advantage": mean("raw_advantage"),
                "mean_normalized_advantage": mean("normalized_advantage"),
                "termination_outcome_counts": {
                    str(endpoint): int((rollout["termination_outcome_endpoint"][selected] == endpoint).sum())
                    for endpoint in sorted(set(rollout["termination_outcome_endpoint"][selected].tolist()))
                },
            })
        epoch_report = {
            "epoch": self.epoch,
            "samples": self.worlds * self.horizon,
            "actor_updates": self.update_in_epoch,
            "credit_rows": {
                "path": str(npz.resolve()),
                "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            },
            "wrist_y_preupdate_bins": bin_rows,
            "actor_state_sha256_after_epoch": model_state_sha256(agent.model.state_dict()),
        }
        path = self.output_dir / f"epoch_{self.epoch:04d}_summary.json"
        path.write_text(json.dumps(epoch_report, indent=2) + "\n")
        self.epoch_reports.append(epoch_report)
        self.record_probes(agent, stage="epoch_end")
        self.rollout_steps = []

    def finalize(self, agent: Any) -> dict[str, Any]:
        if self.rollout_steps:
            raise RuntimeError("cannot finalize credit audit during an epoch")
        final = write_state(
            self.output_dir / "final_actor_state.pt.gz",
            cpu_state(agent.model.state_dict()),
        )
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "paper_faithful": False,
            "worlds": self.worlds,
            "horizon": self.horizon,
            "epochs": len(self.epoch_reports),
            "actor_updates": len(self.update_reports),
            "right_wrist_y_action_index": self.right_wrist_y_index,
            "initial_actor": self.initial_actor,
            "final_actor": final,
            "exact_intermediate_actor_reconstruction": (
                "initial state plus ordered lossless XOR forward-state and optimizer-state patches"
            ),
            "epoch_reports": self.epoch_reports,
            "update_reports": self.update_reports,
            "probe_reports": self.probe_reports,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": SCHEMA,
            "actor_updates": len(self.update_reports),
        }


class CreditInstrumentedTruncatedGaussianPpoAgent(
    StateFeasibleTruncatedGaussianPpoAgent
):
    """The corrected agent with passive credit evidence attached."""

    def __init__(
        self,
        *args,
        credit_audit_dir: Path,
        credit_probe_panel: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(self.minibatch_size) != int(self.batch_size):
            raise ValueError("credit audit is frozen to one full-rollout minibatch")
        self.credit_audit = CreditAuditRecorder(
            credit_audit_dir,
            worlds=int(self.cfg.num_actors),
            horizon=int(self.cfg.horizon_length),
            right_wrist_y_index=1,
            probe_panel=credit_probe_panel,
        )
        self._credit_gradient_norm = float("nan")

    def _observe_actor_forward(self, **kwargs) -> None:
        self.credit_audit.actor_forward(**kwargs)

    def get_action_values(self, obs) -> dict:
        result = super().get_action_values(obs)
        self.credit_audit.sampled_action(result)
        return result

    def env_step(self, actions: torch.Tensor) -> tuple:
        result = super().env_step(actions)
        self.credit_audit.env_result(result)
        return result

    def discount_values(
        self,
        fdones: torch.Tensor,
        last_extrinsic_values: torch.Tensor,
        mb_fdones: torch.Tensor,
        mb_extrinsic_values: torch.Tensor,
        mb_rewards: torch.Tensor,
    ) -> torch.Tensor:
        advantages = super().discount_values(
            fdones,
            last_extrinsic_values,
            mb_fdones,
            mb_extrinsic_values,
            mb_rewards,
        )
        self.credit_audit.gae(
            fdones=fdones,
            last_values=last_extrinsic_values,
            dones=mb_fdones,
            values=mb_extrinsic_values,
            rewards=mb_rewards,
            advantages=advantages,
            gamma=float(self.cfg.gamma),
            tau=float(self.cfg.tau),
        )
        return advantages

    def prepare_dataset(self, batch_dict) -> None:
        raw_advantage = (batch_dict["returns"] - batch_dict["values"]).sum(dim=1)
        super().prepare_dataset(batch_dict)
        sample_ids = torch.arange(self.batch_size, device=self.device, dtype=torch.int64)
        self.dataset.values_dict["audit_sample_id"] = sample_ids
        self.credit_audit.prepared_dataset(
            raw_advantage=raw_advantage,
            normalized_advantage=self.dataset.values_dict["advantages"],
            returns=batch_dict["returns"],
            old_values=batch_dict["values"],
            sample_ids=sample_ids,
        )

    def train_asymmetric_critic(self) -> float:
        critic = self.asymmetric_critic_net
        critic.train()
        loss = 0.0
        for mini_epoch in range(self.cfg.mini_epochs):
            if self.cfg.freeze_critic:
                break
            for index in range(len(critic.dataset)):
                before = model_state_sha256(critic.model.state_dict())
                value = critic.train_critic(critic.dataset[index])
                after = model_state_sha256(critic.model.state_dict())
                self.credit_audit.critic_update(
                    mini_epoch=mini_epoch,
                    loss=value,
                    before_sha256=before,
                    after_sha256=after,
                )
                loss += value
            if self.cfg.normalize_input:
                critic.model.running_mean_std.eval()
        average = loss / (self.cfg.mini_epochs * critic.num_minibatches)
        critic.epoch_num += 1
        critic.lr, _ = critic.scheduler.update(critic.lr, 0, critic.epoch_num, 0, 0)
        critic.update_lr(critic.lr)
        critic.frame += critic.batch_size
        if critic.writer is not None:
            critic.writer.add_scalar("losses/cval_loss", average, critic.frame)
            critic.writer.add_scalar("info/cval_lr", critic.lr, critic.frame)
        return average

    def truncate_gradients_and_step(self) -> None:
        if self.cfg.multi_gpu:
            raise ValueError("credit audit does not support multi-process PPO")
        if self.cfg.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)
            self._credit_gradient_norm = float(norm.detach().cpu())
        else:
            squares = [
                parameter.grad.detach().float().square().sum()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            self._credit_gradient_norm = float(torch.stack(squares).sum().sqrt().cpu())
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_actor_critic(self, input_dict):
        if not self.distribution_spec.optimizer_training_authorized:
            raise RuntimeError("optimizer training is not authorized")
        value_preds = input_dict["old_values"]
        old_neglogp = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        returns = input_dict["returns"]
        before_forward = cpu_state(self.model.state_dict())
        current = self.evaluate_ppo_distribution(input_dict)
        after_forward = cpu_state(self.model.state_dict())
        ratio = torch.exp(old_neglogp - current["neglogp"])
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.cfg.e_clip, 1.0 + self.cfg.e_clip
        )
        surr1 = advantage * ratio
        surr2 = advantage * clipped_ratio
        actor_loss_rows = torch.max(-surr1, -surr2)
        actor_loss = actor_loss_rows.mean()
        value_clipped = value_preds + (current["values"] - value_preds).clamp(
            -self.cfg.e_clip, self.cfg.e_clip
        )
        critic_loss_rows = torch.max(
            (current["values"] - returns).square(),
            (value_clipped - returns).square(),
        ).squeeze(dim=1)
        critic_loss = critic_loss_rows.mean()
        if self.cfg.bounds_loss_coef is None:
            bounds_loss = torch.zeros((), device=self.device)
        elif self.cfg.bound_loss_type == "regularisation":
            bounds_loss = current["mu"].square().sum(dim=-1).mean()
        elif self.cfg.bound_loss_type == "bound":
            soft_bound = 1.1
            lower = torch.clamp_max(current["mu"] + soft_bound, 0.0).square()
            upper = torch.clamp_min(current["mu"] - soft_bound, 0.0).square()
            bounds_loss = (lower + upper).sum(dim=-1).mean()
        else:
            raise ValueError(f"unknown bound loss type {self.cfg.bound_loss_type}")
        entropy_rows = current["entropy"]
        entropy = entropy_rows.mean()
        loss = (
            actor_loss
            + 0.5 * critic_loss * self.cfg.critic_coef
            - entropy * self.current_entropy_coef
            + bounds_loss * (self.cfg.bounds_loss_coef or 0.0)
        )
        for parameter in self.model.parameters():
            parameter.grad = None
        self.scaler.scale(loss).backward()
        self.truncate_gradients_and_step()
        after_update = cpu_state(self.model.state_dict())
        self.credit_audit.actor_update(
            self,
            input_dict=input_dict,
            current=current,
            ratio=ratio,
            surr1=surr1,
            surr2=surr2,
            actor_loss_rows=actor_loss_rows,
            critic_loss_rows=critic_loss_rows,
            entropy_rows=entropy_rows,
            total_loss=loss,
            gradient_norm=self._credit_gradient_norm,
            before_forward=before_forward,
            after_forward=after_forward,
            after_update=after_update,
        )
        with torch.no_grad():
            kl = torch.clamp((current["neglogp"] - old_neglogp).mean(), min=0.0)
        return (
            actor_loss,
            critic_loss,
            entropy,
            kl,
            self.current_lr,
            1.0,
            current["mu"].detach(),
            current["sigma"].detach(),
            bounds_loss,
        )

    def train_epoch(self):
        self.credit_audit.begin_epoch(self)
        result = super().train_epoch()
        self.credit_audit.finalize_epoch(self)
        return result

    def finalize_credit_audit(self) -> dict[str, Any]:
        return self.credit_audit.finalize(self)
