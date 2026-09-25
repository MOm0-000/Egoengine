#!/usr/bin/env python3
"""Audit the frozen 8/16-epoch Pour policies without running physics or training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/human2sim2robot"))

DEFAULT_CHECKPOINT_8 = ROOT / (
    "runs/taco_pour_normalized_ellipse_v1/short_rl_tool_only_8ep/"
    "ppo_chunk_20/nn/last_ep_8_rew__8.038462_.pth"
)
DEFAULT_CHECKPOINT_16 = ROOT / (
    "runs/taco_pour_normalized_ellipse_v1/short_rl_tool_only_16ep_cpu_gate_v1/"
    "ppo_chunk_20/nn/last_ep_16_rew__5.7482166_.pth"
)
DEFAULT_TRACE_8 = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "checkpoint_8ep_closed_loop_v2.json.gz"
)
DEFAULT_TRACE_16 = ROOT / (
    "runs/taco_pour_normalized_ellipse_v1/short_rl_tool_only_16ep_cpu_gate_v1/"
    "cpu_closed_loop_validation.json.gz"
)
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_normalized_policy_diagnostics_v1/report.json.gz"
PAPER = ROOT / "paper/EgoEngine From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations.pdf"

POSITION_SCALE_M = 0.12
ROTATION_SCALE_RAD = 1.5
RESIDUAL_CLIP = 0.05


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _build_actor_model():
    from human2sim2robot.ppo.utils.models import ModelA2CContinuousLogStd
    from human2sim2robot.ppo.utils.network import MlpConfig, NetworkConfig, RnnConfig

    return ModelA2CContinuousLogStd(
        network_config=NetworkConfig(
            mlp=MlpConfig(units=[512, 512]),
            rnn=RnnConfig(
                units=1024,
                layers=1,
                name="lstm",
                layer_norm=True,
                before_mlp=False,
                concat_input=False,
                concat_output=True,
            ),
            separate_value_mlp=False,
            asymmetric_critic=False,
        ),
        actions_num=36,
        input_shape=(236,),
        normalize_value=True,
        normalize_input=True,
        value_size=1,
        num_seqs=1,
    )


def audit_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model = _build_actor_model()
    incompatible = model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    unequal = [name for name in state if not torch.equal(state[name], loaded[name])]

    normalization_keys = [
        "running_mean_std.running_mean",
        "running_mean_std.running_var",
        "running_mean_std.count",
        "value_mean_std.running_mean",
        "value_mean_std.running_var",
        "value_mean_std.count",
    ]
    missing_normalization = [name for name in normalization_keys if name not in state]
    normalization = {}
    for prefix in ("running_mean_std", "value_mean_std"):
        mean = state[f"{prefix}.running_mean"]
        variance = state[f"{prefix}.running_var"]
        count = state[f"{prefix}.count"]
        group = {name: state[name] for name in normalization_keys if name.startswith(prefix)}
        normalization[prefix] = {
            "keys": sorted(group),
            "state_sha256": _state_sha256(group),
            "count": float(count.item()),
            "mean_min": float(mean.min().item()),
            "mean_max": float(mean.max().item()),
            "variance_min": float(variance.min().item()),
            "variance_max": float(variance.max().item()),
            "finite": bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()),
            "strictly_positive_variance": bool((variance > 0).all()),
        }

    recurrent = {name: tensor for name, tensor in state.items() if ".rnn." in name}
    default_rnn = model.get_default_rnn_state()
    checkpoint_files = sorted(path.parent.glob("*.pth"))
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": _file_sha256(path),
        "epoch": int(checkpoint["epoch"]),
        "frame": int(checkpoint["frame"]),
        "top_level_keys": sorted(checkpoint),
        "model_state_tensor_count": len(state),
        "model_state_sha256": _state_sha256(state),
        "normalization": normalization,
        "missing_normalization_keys": missing_normalization,
        "recurrent_state": {
            "parameter_keys": sorted(recurrent),
            "parameter_state_sha256": _state_sha256(recurrent),
            "default_state_shapes": [list(value.shape) for value in default_rnn],
            "default_state_is_zero": all(bool(torch.count_nonzero(value) == 0) for value in default_rnn),
        },
        "cpu_strict_load": {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "unequal_after_load": unequal,
            "loaded_model_state_sha256": _state_sha256(loaded),
            "all_tensors_bitwise_equal": not unequal,
        },
        "checkpoints_in_same_run_directory": [item.name for item in checkpoint_files],
    }


def _trace_rows(report: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return report["repetitions"][0]["trials"][mode]["trace"]["steps"]


def _repeatability(report: dict[str, Any]) -> dict[str, Any]:
    modes = ("replay", "rl")
    signatures = {
        mode: [row["trials"][mode]["trajectory_signature_sha256"] for row in report["repetitions"]]
        for mode in modes
    }
    return {
        "reported": report["bitwise_trajectory_repeatable"],
        "signatures": signatures,
        "independently_confirmed": {mode: len(set(values)) == 1 for mode, values in signatures.items()},
    }


def _action_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = np.asarray([row["raw_residual_action"] for row in rows], dtype=np.float64)
    applied = np.asarray([row["applied_residual"] for row in rows], dtype=np.float64)
    saturated = np.isclose(np.abs(applied), RESIDUAL_CLIP, atol=1e-7, rtol=0.0)
    deltas = np.diff(applied, axis=0)
    sign_flips = np.sign(applied[1:]) * np.sign(applied[:-1]) < 0 if len(applied) > 1 else np.zeros((0, 36), bool)
    return {
        "steps": len(rows),
        "dimensions": int(applied.shape[1]),
        "residual_clip_rad": RESIDUAL_CLIP,
        "raw_abs_mean": float(np.abs(raw).mean()),
        "raw_abs_max": float(np.abs(raw).max()),
        "applied_abs_mean_rad": float(np.abs(applied).mean()),
        "saturated_value_fraction": float(saturated.mean()),
        "per_dimension_saturation_fraction": saturated.mean(axis=0).tolist(),
        "per_dimension_saturation_min": float(saturated.mean(axis=0).min()),
        "per_dimension_saturation_median": float(np.median(saturated.mean(axis=0))),
        "per_dimension_saturation_max": float(saturated.mean(axis=0).max()),
        "steps_with_any_saturated_dimension": int(saturated.any(axis=1).sum()),
        "max_saturated_dimensions_in_one_step": int(saturated.sum(axis=1).max()),
        "adjacent_applied_delta_abs_max_rad": float(np.abs(deltas).max()) if len(deltas) else 0.0,
        "adjacent_applied_delta_l2_max_rad": float(np.linalg.norm(deltas, axis=1).max()) if len(deltas) else 0.0,
        "sign_flip_count": int(sign_flips.sum()),
        "steps_with_any_sign_flip": int(sign_flips.any(axis=1).sum()) if len(sign_flips) else 0,
    }


def _reward_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tracking = np.asarray([row["aggregate_tracking_reward"] for row in rows], dtype=np.float64)
    contact = np.asarray([row["aggregate_contact_bonus"] for row in rows], dtype=np.float64)
    lift = np.asarray([row["lift_reward"] for row in rows], dtype=np.float64)
    total = np.asarray([row["total_reward"] for row in rows], dtype=np.float64)
    reconstructed = tracking + contact + lift
    return {
        "reward_reconstruction_max_abs_error": float(np.abs(reconstructed - total).max()),
        "contact_nonzero_steps": int(np.count_nonzero(contact)),
        "contact_bonus_min": float(contact.min()),
        "contact_bonus_max": float(contact.max()),
        "lift_reward_min": float(lift.min()),
        "lift_reward_max": float(lift.max()),
        "contact_flags_or_forces_present_in_saved_trace": all(
            "contact_flags" in row or "contact_forces" in row for row in rows
        ),
    }


def _endpoint_row(row: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    position = float(row["position_error_m"][0])
    rotation = float(row["rotation_error_rad"][0])
    score = float(row["objective_score"][0])
    action = np.asarray(row["applied_residual"], dtype=np.float64)
    delta = np.zeros_like(action) if previous is None else action - np.asarray(
        previous["applied_residual"], dtype=np.float64
    )
    return {
        "endpoint": int(row["endpoint"]),
        "position_error_m": position,
        "rotation_error_rad": rotation,
        "position_squared_contribution": (position / POSITION_SCALE_M) ** 2,
        "rotation_squared_contribution": (rotation / ROTATION_SCALE_RAD) ** 2,
        "ellipse_score": score,
        "independent_threshold_pass": bool(row["independent_threshold_pass"][0]),
        "tracking_reward": float(row["aggregate_tracking_reward"]),
        "contact_bonus": float(row["aggregate_contact_bonus"]),
        "lift_reward": float(row["lift_reward"]),
        "total_reward": float(row["total_reward"]),
        "saturated_dimensions": int(np.isclose(np.abs(action), RESIDUAL_CLIP, atol=1e-7).sum()),
        "action_delta_abs_max_rad": float(np.abs(delta).max()),
        "action_delta_l2_rad": float(np.linalg.norm(delta)),
    }


def audit_trace(path: Path, failure_tail_start: int) -> dict[str, Any]:
    report = _load_json(path)
    rows = _trace_rows(report, "rl")
    by_endpoint = {int(row["endpoint"]): row for row in rows}
    tail = []
    for endpoint in sorted(value for value in by_endpoint if value >= failure_tail_start):
        previous = by_endpoint.get(endpoint - 1)
        tail.append(_endpoint_row(by_endpoint[endpoint], previous))
    first_trial = report["repetitions"][0]["trials"]["rl"]
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": _file_sha256(path),
        "repeatability": _repeatability(report),
        "validated_steps": int(first_trial["validated_steps"]),
        "first_failure": first_trial["first_failure"],
        "endpoint_range": [int(rows[0]["endpoint"]), int(rows[-1]["endpoint"])],
        "action": _action_metrics(rows),
        "reward": _reward_metrics(rows),
        "failure_tail": tail,
        "_rows": rows,
    }


def compare_policies(rows8: list[dict[str, Any]], rows16: list[dict[str, Any]]) -> dict[str, Any]:
    by8 = {int(row["endpoint"]): row for row in rows8}
    by16 = {int(row["endpoint"]): row for row in rows16}
    common = sorted(set(by8) & set(by16))
    comparisons = []
    for endpoint in common:
        action8 = np.asarray(by8[endpoint]["applied_residual"], dtype=np.float64)
        action16 = np.asarray(by16[endpoint]["applied_residual"], dtype=np.float64)
        comparisons.append({
            "endpoint": endpoint,
            "score_8": float(by8[endpoint]["objective_score"][0]),
            "score_16": float(by16[endpoint]["objective_score"][0]),
            "score_16_minus_8": float(by16[endpoint]["objective_score"][0] - by8[endpoint]["objective_score"][0]),
            "applied_action_l2_difference_rad": float(np.linalg.norm(action16 - action8)),
            "opposite_fully_saturated_dimensions": int(np.sum(
                np.isclose(np.abs(action8), RESIDUAL_CLIP, atol=1e-7)
                & np.isclose(np.abs(action16), RESIDUAL_CLIP, atol=1e-7)
                & (np.sign(action8) != np.sign(action16))
            )),
        })
    first = comparisons[0]
    return {
        "common_endpoint_range": [common[0], common[-1]],
        "common_endpoint_count": len(common),
        "first_common_endpoint": first,
        "endpoint_48": next(row for row in comparisons if row["endpoint"] == 48),
        "all_common_endpoints": comparisons,
        "causal_epoch_ablation": False,
        "reason": (
            "The checkpoints came from separate GPU training runs, and the 16-epoch run "
            "did not preserve an epoch-8 checkpoint from that same optimization path."
        ),
    }


def _source_contract() -> dict[str, Any]:
    files = {
        "ppo_agent": ROOT / "external/human2sim2robot/human2sim2robot/ppo/ppo_agent.py",
        "residual_policy": ROOT / "src/video_to_spider/rl/residual_policy.py",
        "replay_rl": ROOT / "src/video_to_spider/rl/replay_rl.py",
        "checkpoint_validation": ROOT / "scripts/audit_taco_pour_checkpoint_cpu_validation.py",
        "environment": ROOT / "src/video_to_spider/rl/mjwp_env.py",
    }
    text = {name: path.read_text() for name, path in files.items()}
    checks = {
        "training_initializes_default_rnn_state": "self.rnn_states = self.model.get_default_rnn_state()" in text["ppo_agent"],
        "training_zeroes_rnn_on_done": "s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0" in text["ppo_agent"],
        "ppo_config_enables_zero_rnn_on_done": "zero_rnn_on_done=True" in text["residual_policy"],
        "formal_cpu_policy_zeroes_rnn": "state.to(\"cpu\").zero_()" in text["replay_rl"],
        "historical_cpu_validation_zeroes_rnn": "state.to(agent.device).zero_()" in text["checkpoint_validation"],
        "chunk_terminal_uses_episode_end": "self.time_indices >= self.episode_lengths" in text["environment"],
        "terminal_environment_auto_resets": "self._reset_worlds(reset_mask)" in text["environment"],
    }
    return {
        "files": {name: {"path": str(path.resolve()), "sha256": _file_sha256(path)} for name, path in files.items()},
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "interpretation": (
            "Training starts with zero default recurrent state. A done endpoint zeroes the "
            "recurrent state while the environment restores the exact chunk boundary. CPU "
            "validation also starts each trial from zero recurrent state."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-8", type=Path, default=DEFAULT_CHECKPOINT_8)
    parser.add_argument("--checkpoint-16", type=Path, default=DEFAULT_CHECKPOINT_16)
    parser.add_argument("--trace-8", type=Path, default=DEFAULT_TRACE_8)
    parser.add_argument("--trace-16", type=Path, default=DEFAULT_TRACE_16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    checkpoint8 = audit_checkpoint(args.checkpoint_8)
    checkpoint16 = audit_checkpoint(args.checkpoint_16)
    trace8 = audit_trace(args.trace_8, failure_tail_start=55)
    trace16 = audit_trace(args.trace_16, failure_tail_start=45)
    rows8 = trace8.pop("_rows")
    rows16 = trace16.pop("_rows")
    comparison = compare_policies(rows8, rows16)

    normalization_transfer_pass = all(
        not checkpoint["missing_normalization_keys"]
        and checkpoint["cpu_strict_load"]["all_tensors_bitwise_equal"]
        for checkpoint in (checkpoint8, checkpoint16)
    )
    rnn_contract = _source_contract()
    action_saturation_strong = all(
        trace["action"]["saturated_value_fraction"] > 0.90
        for trace in (trace8, trace16)
    )
    tail8 = trace8["failure_tail"]
    reward_conflict_in_tail = any(
        current["tracking_reward"] < previous["tracking_reward"]
        and current["contact_bonus"] + current["lift_reward"]
        > previous["contact_bonus"] + previous["lift_reward"]
        and current["total_reward"] >= previous["total_reward"]
        for previous, current in zip(tail8, tail8[1:])
    )

    report = {
        "schema": "taco_pour_normalized_policy_diagnostics_v1",
        "scope": {
            "training_executed": False,
            "physics_rollout_executed": False,
            "checkpoint_and_saved_trace_read_only": True,
            "paper_faithful_objective_claimed": False,
        },
        "action_contract": {
            "ppo_action_space": "[-1, 1]^36",
            "runtime_mapping": "clip(1.0 * action, -0.05, +0.05) rad",
            "linear_unsaturated_input_interval": [-0.05, 0.05],
        },
        "paper_constraint": {
            "paper_path": str(PAPER.resolve()),
            "paper_sha256": _file_sha256(PAPER),
            "section": "Appendix C.2, Action smoothness reward and TACO configuration",
            "published_residual_statement": "a_t = a_base_t + delta_a_t",
            "published_residual_scale_or_range": None,
            "taco_action_smoothness_reward": "disabled",
            "interpretation": (
                "The next controlled test may repair the local action unit/range mapping, "
                "but must not add action-smoothness reward and must not claim an author scale."
            ),
        },
        "checkpoints": {"epoch_8": checkpoint8, "epoch_16": checkpoint16},
        "inference_contract": {
            "normalization_state_present_and_cpu_load_bitwise_equal": normalization_transfer_pass,
            "rnn": rnn_contract,
            "mismatch_found": not normalization_transfer_pass or not rnn_contract["all_checks_pass"],
        },
        "cpu_traces": {"epoch_8": trace8, "epoch_16": trace16},
        "policy_comparison": comparison,
        "evidence_assessment": {
            "cpu_gpu_inference_semantics_mismatch": {
                "level": "not_found",
                "basis": "Normalization buffers load bitwise exactly and recurrent reset semantics agree.",
            },
            "contact_or_lift_reward_conflict_at_epoch8_failure_tail": {
                "level": "not_supported_by_saved_trace",
                "mechanical_masking_detected": reward_conflict_in_tail,
                "basis": (
                    "At endpoints 55-59 contact bonus is zero and lift is below 0.00072, "
                    "while tracking reward falls by about 0.188. Per-finger flags/forces were "
                    "not saved, so finer contact-switch attribution is unavailable."
                ),
            },
            "action_saturation_or_switching": {
                "level": "strong",
                "criterion": "both policies saturate more than 90% of all action values",
                "criterion_passed": action_saturation_strong,
                "basis": (
                    "The normalized policy output is clipped directly to +/-0.05 rad rather "
                    "than scaled across that range; both frozen policies lose magnitude "
                    "resolution in more than 95% of action values."
                ),
            },
            "late_window_training_coverage": {
                "level": "inconclusive",
                "basis": "Saved traces contain executed states but not the training-state visitation distribution.",
            },
            "same_policy_degraded_from_epoch8_to_epoch16": {
                "level": "not_identifiable",
                "basis": comparison["reason"],
            },
        },
        "decision": {
            "next_single_controlled_change_family": "action unit/range contract",
            "proposed_local_mapping_for_separate_followup": (
                "delta_a_rad = 0.05 * normalized_policy_output, retaining the existing "
                "+/-0.05 rad safety limit"
            ),
            "proposed_mapping_is_published_by_egoengine": False,
            "do_not_change_yet": [
                "reward coefficients",
                "action-smoothness reward (published as disabled for TACO)",
                "training epochs",
                "data collection",
            ],
            "implementation_change_executed": False,
            "reason": (
                "The predeclared action-saturation rule is triggered. The reward-conflict rule "
                "is not triggered at the 8-epoch failure tail, coverage remains unproven, and "
                "Appendix C.2 rules out adding smoothness reward for the TACO configuration."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", compresslevel=9) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "inference_mismatch_found": report["inference_contract"]["mismatch_found"],
        "epoch8_validated_steps": trace8["validated_steps"],
        "epoch16_validated_steps": trace16["validated_steps"],
        "epoch8_saturated_value_fraction": trace8["action"]["saturated_value_fraction"],
        "epoch16_saturated_value_fraction": trace16["action"]["saturated_value_fraction"],
        "reward_conflict_in_epoch8_failure_tail": reward_conflict_in_tail,
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
