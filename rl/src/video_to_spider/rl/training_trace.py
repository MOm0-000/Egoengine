"""Lossless PPO visitation records plus small per-epoch audit summaries."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "taco_ppo_training_visitation_v1"
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
        object_roles: tuple[str, ...],
        hand_roles: tuple[str, ...],
        residual_scale: float,
        residual_clip: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.output_dir.mkdir(parents=True)
        if len(actuator_names) < 6:
            raise ValueError("training trace requires six right-wrist actuators")
        self.actuator_names = actuator_names
        self.object_roles = object_roles
        self.hand_roles = hand_roles
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
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
            "tool_position_error_m": [],
            "tool_rotation_error_rad": [],
            "tool_objective_score": [],
            "right_wrist_policy_output_prelimit": [],
            "right_wrist_policy_output_bounded": [],
            "right_wrist_applied_residual": [],
            "contact_flags": [],
            "tracking_terminated": [],
            "time_out": [],
        }

    def record(
        self,
        *,
        source_endpoint: np.ndarray,
        outcome_endpoint: np.ndarray,
        policy_output_prelimit: np.ndarray,
        policy_output_bounded: np.ndarray,
        applied_residual: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        if self._active is None:
            raise RuntimeError("PPO step occurred without an active training-trace epoch")
        arrays = {
            "source_endpoint": np.asarray(source_endpoint, np.int32),
            "outcome_endpoint": np.asarray(outcome_endpoint, np.int32),
            "tool_position_error_m": np.asarray(info["object_position_error"], np.float32)[:, 0],
            "tool_rotation_error_rad": np.asarray(info["object_rotation_error"], np.float32)[:, 0],
            "tool_objective_score": np.asarray(info["object_tracking_error_per_object"], np.float32)[:, 0],
            "right_wrist_policy_output_prelimit": np.asarray(policy_output_prelimit, np.float32)[:, :6],
            "right_wrist_policy_output_bounded": np.asarray(policy_output_bounded, np.float32)[:, :6],
            "right_wrist_applied_residual": np.asarray(applied_residual, np.float32)[:, :6],
            "contact_flags": np.asarray(info["contact_flags"], bool),
            "tracking_terminated": np.asarray(info["terminated"], bool),
            "time_out": np.asarray(info["time_outs"], bool),
        }
        batch = len(arrays["source_endpoint"])
        if any(len(value) != batch for value in arrays.values()):
            raise ValueError("training-trace fields have inconsistent world counts")
        if arrays["right_wrist_policy_output_prelimit"].shape != (batch, 6):
            raise ValueError("right-wrist policy output must have six coordinates")
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
        prelimit = data["right_wrist_policy_output_prelimit"]
        bounded = data["right_wrist_policy_output_bounded"]
        applied = data["right_wrist_applied_residual"]
        translation_names = tuple(self.actuator_names[:3])
        rotation_names = tuple(self.actuator_names[3:6])
        return {
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
                "unit": {"policy": "unitless", "applied_residual": "m"},
                "policy_output_prelimit": _component_distributions(prelimit[:, :3], translation_names),
                "policy_output_bounded": _component_distributions(bounded[:, :3], translation_names),
                "applied_residual": _component_distributions(applied[:, :3], translation_names),
                "prelimit_fraction_abs_gt_1": float((np.abs(prelimit[:, :3]) > 1.0).mean()),
                "bounded_fraction_at_limit": float(np.isclose(np.abs(bounded[:, :3]), 1.0).mean()),
                "applied_fraction_at_clip": float(
                    np.isclose(np.abs(applied[:, :3]), self.residual_clip, atol=1e-7).mean()
                ),
            },
            "right_wrist_rotation": {
                "unit": {"policy": "unitless", "applied_residual": "rad"},
                "policy_output_prelimit": _component_distributions(prelimit[:, 3:6], rotation_names),
                "policy_output_bounded": _component_distributions(bounded[:, 3:6], rotation_names),
                "applied_residual": _component_distributions(applied[:, 3:6], rotation_names),
                "prelimit_fraction_abs_gt_1": float((np.abs(prelimit[:, 3:6]) > 1.0).mean()),
                "bounded_fraction_at_limit": float(np.isclose(np.abs(bounded[:, 3:6]), 1.0).mean()),
                "applied_fraction_at_clip": float(
                    np.isclose(np.abs(applied[:, 3:6]), self.residual_clip, atol=1e-7).mean()
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
                "policy_output_prelimit": "sampled PPO action before the official [-1,1] action clamp",
                "policy_output_bounded": "action passed by the official PPO adapter to MJWP",
                "applied_residual": "bounded residual after local scale and formal safety clip",
                "contact_summary": "control-endpoint hand-object finger flags; not substep collision topology",
                "write_timing": (
                    "rollout samples are held in memory and written only at the next epoch "
                    "boundary or after training completes"
                ),
            },
            "incomplete_step_discarded": bool(incomplete_step_discarded),
            "action_contract": {
                "right_wrist_actuator_names": list(self.actuator_names[:6]),
                "right_wrist_coordinate_units": ["m", "m", "m", "rad", "rad", "rad"],
                "residual_scale": self.residual_scale,
                "residual_clip": self.residual_clip,
            },
            "object_roles": list(self.object_roles),
            "hand_roles": list(self.hand_roles),
            "epochs": self._epochs,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {"path": str(path.resolve()), "sha256": _sha256(path), **manifest}
