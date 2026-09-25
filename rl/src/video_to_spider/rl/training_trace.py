"""Lossless PPO visitation records plus small per-epoch audit summaries."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "taco_ppo_training_visitation_v5"
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _component_distributions(values: np.ndarray, names: tuple[str, ...]) -> dict:
    values = np.asarray(values, np.float64)
    return {
        name: _distribution(values[:, index])
        for index, name in enumerate(names)
    }


class PpoTrainingTrace:
    """Collect in memory during rollout; write only between PPO epochs."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        actuator_names: tuple[str, ...],
        actuator_units: tuple[str, ...],
        object_roles: tuple[str, ...],
        hand_roles: tuple[str, ...],
        residual_scale: float,
        residual_clip: float,
        ctrlrange_contract: dict[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.output_dir.mkdir(parents=True)
        expected = 18 * len(hand_roles)
        if len(actuator_names) != expected or len(actuator_units) != expected:
            raise ValueError("training trace requires all 18 action coordinates per hand")
        if any(unit not in {"m", "rad"} for unit in actuator_units):
            raise ValueError("actuator units must be metres or radians")
        self.actuator_names = actuator_names
        self.actuator_units = actuator_units
        self.object_roles = object_roles
        self.hand_roles = hand_roles
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        self.ctrlrange_contract = ctrlrange_contract
        self._active: dict[str, Any] | None = None
        self._epochs: list[dict[str, Any]] = []
        self._finalized = False

    def begin_epoch(self, epoch: int, frame: int) -> None:
        if self._finalized:
            raise RuntimeError("training trace is already finalized")
        self._flush_epoch()
        self._active = {
            "epoch": int(epoch),
            "frame_at_start": int(frame),
            "source_endpoint": [],
            "outcome_endpoint": [],
            "command_reference_endpoint": [],
            "reward_reference_endpoint": [],
            "next_observation_goal_reference_endpoint": [],
            "tool_position_error_m": [],
            "tool_rotation_error_rad": [],
            "tool_objective_score": [],
            "sampled_action_preclamp": [],
            "sampled_action_clamped": [],
            "actor_mu": [],
            "actor_sigma": [],
            "reference_ctrl": [],
            "requested_residual": [],
            "effective_residual_after_ctrlrange": [],
            "residual_lost_to_ctrlrange": [],
            "contact_flags": [],
            "tracking_terminated": [],
            "time_out": [],
        }

    def record(
        self,
        *,
        source_endpoint: np.ndarray,
        outcome_endpoint: np.ndarray,
        sampled_action_preclamp: np.ndarray,
        sampled_action_clamped: np.ndarray,
        actor_mu: np.ndarray,
        actor_sigma: np.ndarray,
        reference_ctrl: np.ndarray,
        requested_residual: np.ndarray,
        effective_residual_after_ctrlrange: np.ndarray,
        residual_lost_to_ctrlrange: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        if self._active is None:
            raise RuntimeError("PPO step occurred without an active training-trace epoch")
        arrays = {
            "source_endpoint": np.asarray(source_endpoint, np.int32),
            "outcome_endpoint": np.asarray(outcome_endpoint, np.int32),
            "command_reference_endpoint": np.asarray(
                info["command_reference_endpoint"], np.int32
            ),
            "reward_reference_endpoint": np.asarray(
                info["reward_reference_endpoint"], np.int32
            ),
            "next_observation_goal_reference_endpoint": np.asarray(
                info["next_observation_goal_reference_endpoint"], np.int32
            ),
            "tool_position_error_m": np.asarray(info["object_position_error"], np.float32)[:, 0],
            "tool_rotation_error_rad": np.asarray(info["object_rotation_error"], np.float32)[:, 0],
            "tool_objective_score": np.asarray(info["object_tracking_error_per_object"], np.float32)[:, 0],
            "sampled_action_preclamp": np.asarray(sampled_action_preclamp, np.float32),
            "sampled_action_clamped": np.asarray(sampled_action_clamped, np.float32),
            "actor_mu": np.asarray(actor_mu, np.float32),
            "actor_sigma": np.asarray(actor_sigma, np.float32),
            "reference_ctrl": np.asarray(reference_ctrl, np.float64),
            "requested_residual": np.asarray(requested_residual, np.float64),
            "effective_residual_after_ctrlrange": np.asarray(
                effective_residual_after_ctrlrange, np.float64
            ),
            "residual_lost_to_ctrlrange": np.asarray(
                residual_lost_to_ctrlrange, np.float64
            ),
            "contact_flags": np.asarray(info["contact_flags"], bool),
            "tracking_terminated": np.asarray(info["terminated"], bool),
            "time_out": np.asarray(info["time_outs"], bool),
        }
        batch = len(arrays["source_endpoint"])
        if any(len(value) != batch for value in arrays.values()):
            raise ValueError("training-trace fields have inconsistent world counts")
        action_shape = (batch, len(self.actuator_names))
        for name in (
            "sampled_action_preclamp",
            "sampled_action_clamped",
            "actor_mu",
            "actor_sigma",
            "reference_ctrl",
            "requested_residual",
            "effective_residual_after_ctrlrange",
            "residual_lost_to_ctrlrange",
        ):
            if arrays[name].shape != action_shape:
                raise ValueError(f"{name} must have shape {action_shape}")
        if np.any(arrays["actor_sigma"] <= 0.0):
            raise ValueError("actor sigma must be finite and strictly positive")
        if not np.array_equal(
            arrays["command_reference_endpoint"], arrays["outcome_endpoint"]
        ):
            raise ValueError("command reference endpoint must equal outcome endpoint")
        if not np.array_equal(
            arrays["reward_reference_endpoint"], arrays["outcome_endpoint"]
        ):
            raise ValueError("reward reference endpoint must equal outcome endpoint")
        if np.any(
            arrays["next_observation_goal_reference_endpoint"]
            < arrays["outcome_endpoint"]
        ) or np.any(
            arrays["next_observation_goal_reference_endpoint"]
            > arrays["outcome_endpoint"] + 1
        ):
            raise ValueError("next observation goal endpoint must be outcome or outcome plus one")
        if not np.allclose(
            arrays["requested_residual"],
            arrays["effective_residual_after_ctrlrange"]
            + arrays["residual_lost_to_ctrlrange"],
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        ):
            raise ValueError("requested residual does not equal effective plus lost")
        if arrays["contact_flags"].ndim != 4:
            raise ValueError("contact flags must be world x hand x object x finger")
        expected_contacts = (batch, len(self.hand_roles), len(self.object_roles), len(_FINGERS))
        if arrays["contact_flags"].shape != expected_contacts:
            raise ValueError(
                f"contact flags must have shape {expected_contacts}, got "
                f"{arrays['contact_flags'].shape}"
            )
        for name, value in arrays.items():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"non-finite training trace field: {name}")
            self._active[name].append(value.copy())

    def _contact_pattern(self, flags: np.ndarray) -> str:
        parts = []
        for hand, hand_role in enumerate(self.hand_roles):
            for obj, object_role in enumerate(self.object_roles):
                active = [
                    finger for index, finger in enumerate(_FINGERS)
                    if flags[hand, obj, index]
                ]
                if active:
                    parts.append(f"{hand_role}-{object_role}:{','.join(active)}")
        return "|".join(parts) if parts else "none"

    def _summary(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
        endpoints = data["outcome_endpoint"]
        endpoint_rows = {}
        for endpoint in sorted(set(map(int, endpoints))):
            select = endpoints == endpoint
            endpoint_rows[str(endpoint)] = {
                "visit_count": int(select.sum()),
                "tool_position_error_m": _distribution(data["tool_position_error_m"][select]),
                "tool_rotation_error_rad": _distribution(data["tool_rotation_error_rad"][select]),
                "tool_objective_score": _distribution(data["tool_objective_score"][select]),
                "tracking_termination_count": int(data["tracking_terminated"][select].sum()),
                "timeout_count": int(data["time_out"][select].sum()),
            }
        patterns = Counter(self._contact_pattern(flags) for flags in data["contact_flags"])
        sampled_preclamp = data["sampled_action_preclamp"]
        sampled_clamped = data["sampled_action_clamped"]
        requested = data["requested_residual"]
        effective = data["effective_residual_after_ctrlrange"]
        lost = data["residual_lost_to_ctrlrange"]
        translation_names = tuple(self.actuator_names[:3])
        rotation_names = tuple(self.actuator_names[3:6])
        result = {
            "sample_count": int(len(endpoints)),
            "source_endpoint_visit_counts": {
                str(key): value for key, value in sorted(Counter(map(int, data["source_endpoint"])).items())
            },
            "outcome_endpoint_visit_counts": {
                str(key): value for key, value in sorted(Counter(map(int, endpoints)).items())
            },
            "per_outcome_endpoint": endpoint_rows,
            "tool_position_error_m": _distribution(data["tool_position_error_m"]),
            "tool_rotation_error_rad": _distribution(data["tool_rotation_error_rad"]),
            "tool_objective_score": _distribution(data["tool_objective_score"]),
            "right_wrist_translation": {
                "unit": {"policy": "unitless", "control_target_residual": "m"},
                "sampled_action_preclamp": _component_distributions(
                    sampled_preclamp[:, :3], translation_names
                ),
                "sampled_action_clamped": _component_distributions(
                    sampled_clamped[:, :3], translation_names
                ),
                "actor_mu": _component_distributions(data["actor_mu"][:, :3], translation_names),
                "actor_sigma": _component_distributions(data["actor_sigma"][:, :3], translation_names),
                "requested_residual": _component_distributions(requested[:, :3], translation_names),
                "effective_residual_after_ctrlrange": _component_distributions(
                    effective[:, :3], translation_names
                ),
                "residual_lost_to_ctrlrange": _component_distributions(lost[:, :3], translation_names),
                "sampled_preclamp_fraction_abs_gt_1": float(
                    (np.abs(sampled_preclamp[:, :3]) > 1.0).mean()
                ),
                "sampled_clamped_fraction_at_limit": float(
                    np.isclose(np.abs(sampled_clamped[:, :3]), 1.0).mean()
                ),
                "requested_fraction_at_clip": float(
                    np.isclose(np.abs(requested[:, :3]), self.residual_clip, atol=1e-7).mean()
                ),
            },
            "right_wrist_rotation": {
                "unit": {"policy": "unitless", "control_target_residual": "rad"},
                "sampled_action_preclamp": _component_distributions(
                    sampled_preclamp[:, 3:6], rotation_names
                ),
                "sampled_action_clamped": _component_distributions(
                    sampled_clamped[:, 3:6], rotation_names
                ),
                "actor_mu": _component_distributions(data["actor_mu"][:, 3:6], rotation_names),
                "actor_sigma": _component_distributions(data["actor_sigma"][:, 3:6], rotation_names),
                "requested_residual": _component_distributions(requested[:, 3:6], rotation_names),
                "effective_residual_after_ctrlrange": _component_distributions(
                    effective[:, 3:6], rotation_names
                ),
                "residual_lost_to_ctrlrange": _component_distributions(lost[:, 3:6], rotation_names),
                "sampled_preclamp_fraction_abs_gt_1": float(
                    (np.abs(sampled_preclamp[:, 3:6]) > 1.0).mean()
                ),
                "sampled_clamped_fraction_at_limit": float(
                    np.isclose(np.abs(sampled_clamped[:, 3:6]), 1.0).mean()
                ),
                "requested_fraction_at_clip": float(
                    np.isclose(np.abs(requested[:, 3:6]), self.residual_clip, atol=1e-7).mean()
                ),
            },
            "coarse_contact_pattern_counts": dict(sorted(patterns.items())),
            "tracking_termination_endpoint_counts": {
                str(key): value for key, value in sorted(Counter(
                    map(int, endpoints[data["tracking_terminated"]])
                ).items())
            },
            "timeout_endpoint_counts": {
                str(key): value for key, value in sorted(Counter(
                    map(int, endpoints[data["time_out"]])
                ).items())
            },
        }
        groups = {}
        for hand_index, hand_role in enumerate(self.hand_roles):
            offset = 18 * hand_index
            for label, start, stop in (
                ("wrist_translation", 0, 3),
                ("wrist_rotation", 3, 6),
                ("fingers", 6, 18),
            ):
                indices = np.arange(offset + start, offset + stop)
                names = tuple(self.actuator_names[index] for index in indices)
                group_lost = lost[:, indices]
                group_requested = requested[:, indices]
                group_effective = effective[:, indices]
                group_sampled_preclamp = sampled_preclamp[:, indices]
                group_sampled_clamped = sampled_clamped[:, indices]
                groups[f"{hand_role}_{label}"] = {
                    "indices": indices.tolist(),
                    "unit": self.actuator_units[int(indices[0])],
                    "actuator_names": list(names),
                    "sampled_action_preclamp": _component_distributions(
                        group_sampled_preclamp, names
                    ),
                    "sampled_action_clamped": _component_distributions(
                        group_sampled_clamped, names
                    ),
                    "actor_mu": _component_distributions(data["actor_mu"][:, indices], names),
                    "actor_sigma": _component_distributions(data["actor_sigma"][:, indices], names),
                    "requested_residual": _component_distributions(group_requested, names),
                    "effective_residual_after_ctrlrange": _component_distributions(
                        group_effective, names
                    ),
                    "residual_lost_to_ctrlrange": _component_distributions(group_lost, names),
                    "requested_fraction_at_clip": float(
                        np.isclose(
                            np.abs(group_requested), self.residual_clip, atol=1e-7
                        ).mean()
                    ),
                    "range_truncated_component_count": int(
                        (np.abs(group_lost) > 1e-7).sum()
                    ),
                    "range_truncated_component_fraction": float(
                        (np.abs(group_lost) > 1e-7).mean()
                    ),
                    "fully_blocked_requested_component_count": int((
                        (np.abs(group_requested) > 1e-7)
                        & (np.abs(group_effective) <= 1e-7)
                    ).sum()),
                }
        result["residual_groups"] = groups
        return result

    def _flush_epoch(self, *, allow_empty: bool = False) -> None:
        if self._active is None:
            return
        active = self._active
        self._active = None
        if not active["source_endpoint"]:
            if allow_empty:
                return
            raise RuntimeError(f"epoch {active['epoch']} produced no PPO rollout samples")
        data = {
            name: np.concatenate(values, axis=0)
            for name, values in active.items()
            if isinstance(values, list)
        }
        epoch = int(active["epoch"])
        stem = f"epoch_{epoch:04d}"
        raw_path = self.output_dir / f"{stem}_visits.npz"
        summary_path = self.output_dir / f"{stem}_summary.json"
        np.savez_compressed(raw_path, **data)
        summary = {
            "schema": SCHEMA,
            "epoch": epoch,
            "frame_at_start": int(active["frame_at_start"]),
            **self._summary(data),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self._epochs.append({
            "epoch": epoch,
            "frame_at_start": int(active["frame_at_start"]),
            "sample_count": summary["sample_count"],
            "visits": {"path": str(raw_path.resolve()), "sha256": _sha256(raw_path)},
            "summary": {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
        })

    def finalize(
        self, *, completed: bool, incomplete_step_discarded: bool = False
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("training trace was finalized twice")
        self._flush_epoch(allow_empty=not completed)
        if completed and not self._epochs:
            raise RuntimeError("training completed without any visitation records")
        self._finalized = True
        status = "complete" if completed else (
            "training_failed_after_logged_rollout"
            if self._epochs else "training_failed_before_logged_rollout"
        )
        manifest = {
            "schema": SCHEMA,
            "status": status,
            "logging_semantics": {
                "endpoint_alignment": (
                    "transition t->t+1 commands ctrl[t+1], scores physical state[t+1] "
                    "against ref[t+1], and exposes ref[t+2] to the next actor observation "
                    "except for reference-tail clipping"
                ),
                "sampled_action_preclamp": (
                    "stochastic action sampled from the PPO policy distribution before the "
                    "official [-1,1] clamp; this is not actor mean mu"
                ),
                "sampled_action_clamped": (
                    "the same sampled action after the official [-1,1] clamp"
                ),
                "actor_mu": (
                    "mean of the Gaussian PPO policy distribution used to draw the recorded sample"
                ),
                "actor_sigma": (
                    "strictly-positive standard deviation (exp(logstd), not variance) of the "
                    "Gaussian PPO policy distribution used to draw the recorded sample"
                ),
                "reference_ctrl": (
                    "full residual-action control target at transition t->t+1 before the "
                    "policy residual and before actuator ctrlrange"
                ),
                "requested_residual": (
                    "full action-space signed control-target offset after local scale and "
                    "formal safety clip, before actuator ctrlrange"
                ),
                "effective_residual_after_ctrlrange": (
                    "full action-space signed control-target offset remaining after the "
                    "model's enabled actuator ctrlrange clamp; not realized qpos motion"
                ),
                "residual_lost_to_ctrlrange": (
                    "signed requested minus effective control-target residual"
                ),
                "legacy_v2_applied_residual": (
                    "historical v2 right_wrist_applied_residual equals requested_residual "
                    "for its recorded six dimensions; v2 artifacts remain immutable"
                ),
                "actor_distribution_parameters": (
                    "mu and sigma are passively captured from the same official get_action_values "
                    "result as the sampled action; no extra actor forward pass is performed"
                ),
                "contact_summary": "control-endpoint hand-object finger flags; not substep collision topology",
                "write_timing": (
                    "rollout samples are held in memory and written only at the next epoch "
                    "boundary or after training completes"
                ),
            },
            "incomplete_step_discarded": bool(incomplete_step_discarded),
            "action_contract": {
                "dimensions": len(self.actuator_names),
                "actuator_names": list(self.actuator_names),
                "coordinate_units": list(self.actuator_units),
                "residual_scale": self.residual_scale,
                "residual_clip": self.residual_clip,
                "ctrlrange": self.ctrlrange_contract,
            },
            "object_roles": list(self.object_roles),
            "hand_roles": list(self.hand_roles),
            "epochs": self._epochs,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {"path": str(path.resolve()), "sha256": _sha256(path), **manifest}
