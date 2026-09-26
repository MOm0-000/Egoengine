#!/usr/bin/env python3
"""Read-only audit of the valid post-fix Pour PPO policy extremization."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_corrected_endpoint44_48_failure_attribution import (  # noqa: E402
    _actuator_metadata,
    _active_contacts,
    _require_exact_trace,
)
from video_to_spider.rl.credit_audit import (  # noqa: E402
    apply_xor_patch,
    model_state_sha256,
)
from video_to_spider.rl.state_feasible_truncated_gaussian import (  # noqa: E402
    deterministic_truncated_action,
)


FAILURE_SOURCES = tuple(range(34, 40))
FAILURE_ENDPOINTS = tuple(range(35, 41))
PROBE_SOURCES = tuple(range(32, 40))
FOCUS_TRAINING_SOURCES = tuple(range(35, 40))
ACTION_GROUPS = {
    "right_wrist_translation": slice(0, 3),
    "right_wrist_rotation": slice(3, 6),
    "right_fingers": slice(6, 18),
    "left_wrist_translation": slice(18, 21),
    "left_wrist_rotation": slice(21, 24),
    "left_fingers": slice(24, 36),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_postfix_policy_extremization_audit_v1":
        raise ValueError("unsupported post-fix policy audit contract")
    if contract.get("status") != "authorized_read_only_audit":
        raise ValueError("post-fix policy audit is not authorized")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_steps": 0,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "action_bound_changed": False,
        "action_distribution_changed": False,
        "reference_timing_changed": False,
        "action_frame_changed": False,
        "PPO_hyperparameters_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
        "checkpoint_resume_for_training_allowed": False,
    }
    if any(contract["runtime"].get(key) != value for key, value in required.items()):
        raise ValueError("runtime contract is not fail-closed")
    if contract["failure_attribution"] != {
        "source_endpoints": list(FAILURE_SOURCES),
        "outcome_endpoints": list(FAILURE_ENDPOINTS),
        "fixed_probe_source_endpoints": list(PROBE_SOURCES),
    }:
        raise ValueError("failure attribution endpoints changed")
    paths = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_gzip_torch(path: Path):
    return torch.load(
        io.BytesIO(gzip.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )


def clone_hidden(values):
    return [value.detach().cpu().clone() for value in values]


def zero_hidden(policy):
    return [state.to("cpu").zero_() for state in policy.model.get_default_rnn_state()]


def contact_record(backend) -> dict:
    flags = np.asarray(backend.last_info["contact_flags"][0], dtype=bool)
    return {"flags": flags, "active": _active_contacts(flags)}


def run_trial(
    *, backend, policy, boundary, mode: str, deterministic: bool,
    seed: int | None = None, capture_sources: tuple[int, ...] = (),
) -> tuple[dict, dict[int, dict], dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    if policy is not None:
        policy.rnn_states = zero_hidden(policy)
    if seed is not None:
        torch.manual_seed(seed)
    backend.begin_trial(mode, 20, 60)
    contacts = {}
    captures = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if policy is None:
            action = np.zeros((1, 36), dtype=np.float32)
        else:
            raw = backend.observation()
            packed = policy.obs_to_tensors(raw)
            hidden = clone_hidden(policy.rnn_states)
            result = (
                policy.get_deterministic_action_values(packed)
                if deterministic else policy.get_action_values(packed)
            )
            policy.rnn_states = result["rnn_states"]
            selected = (
                result["deterministic_actions"] if deterministic else result["actions"]
            )
            action = policy.preprocess_actions(selected)
            if source in capture_sources:
                raw_observation = raw["obs"] if isinstance(raw, dict) else raw
                captures[source] = {
                    "raw_observation": np.asarray(raw_observation, np.float32)[0].copy(),
                    "hidden": hidden,
                    "actor_mu": result["mus"][0].detach().cpu().numpy().copy(),
                    "actor_sigma": result["sigmas"][0].detach().cpu().numpy().copy(),
                    "action_low": result["action_lows"][0].detach().cpu().numpy().copy(),
                    "action_high": result["action_highs"][0].detach().cpu().numpy().copy(),
                    "normalized_action": selected[0].detach().cpu().numpy().copy(),
                }
        feasible = backend.step(action, source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        contacts[endpoint] = contact_record(backend)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], contacts, captures


def qvel_metadata(model, control_indices: tuple[int, ...]) -> np.ndarray:
    result = []
    for actuator in control_indices:
        joint = int(model.actuator_trnid[actuator, 0])
        result.append(int(model.jnt_dofadr[joint]))
    if len(result) != 36 or len(set(result)) != 36:
        raise ValueError("expected 36 distinct scalar actuator dofs")
    return np.asarray(result, dtype=np.int64)


def group_values(values: np.ndarray, names: tuple[str, ...]) -> dict:
    values = np.asarray(values, np.float64)
    return {
        group: {
            "actuator_names": list(names[selection]),
            "values": values[selection].tolist(),
            "l2": float(np.linalg.norm(values[selection])),
            "max_abs": float(np.max(np.abs(values[selection]))),
        }
        for group, selection in ACTION_GROUPS.items()
    }


def failure_attribution(
    *, traces: dict, extras: dict, captures: dict, reference,
    actuator_names: tuple[str, ...], qvel_indices: np.ndarray,
) -> dict:
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    per_mode = {}
    step_maps = {}
    for mode in ("replay", "ppo"):
        steps = {int(row["endpoint"]): row for row in traces[mode]["steps"]}
        step_maps[mode] = steps
        rows = {}
        for endpoint in FAILURE_ENDPOINTS:
            row = steps[endpoint]
            position = float(row["position_error_m"][0])
            rotation = float(row["rotation_error_rad"][0])
            pos_term = (position / 0.12) ** 2
            rot_term = (rotation / 1.5) ** 2
            qpos = np.asarray(row["endpoint_qpos"], np.float64)
            qvel = np.asarray(row["endpoint_qvel"], np.float64)
            requested = np.asarray(row["requested_residual"], np.float64)
            effective = np.asarray(
                row["effective_residual_after_ctrlrange"], np.float64
            )
            lost = np.asarray(row["residual_lost_to_ctrlrange"], np.float64)
            if not np.array_equal(requested, effective + lost):
                raise RuntimeError("residual decomposition identity failed")
            item = {
                "source_endpoint": endpoint - 1,
                "outcome_endpoint": endpoint,
                "position_error_m": position,
                "rotation_error_rad": rotation,
                "objective_score": float(row["objective_score"][0]),
                "ellipse_squared_contributions": {
                    "position": pos_term,
                    "rotation": rot_term,
                    "position_fraction": pos_term / (pos_term + rot_term),
                },
                "tool_position_error_vector_actual_minus_reference_m": (
                    qpos[36:39] - reference_qpos[endpoint, 36:39]
                ).tolist(),
                "tool_linear_velocity_m_per_s": qvel[36:39].tolist(),
                "named_hand_qvel": group_values(qvel[qvel_indices], actuator_names),
                "reward": {
                    "tracking": float(row["aggregate_tracking_reward"]),
                    "contact": float(row["aggregate_contact_bonus"]),
                    "lift": float(row["lift_reward"]),
                    "total": float(row["total_reward"]),
                },
                "contact": {
                    "active": extras[mode][endpoint]["active"],
                    "flags": extras[mode][endpoint]["flags"].tolist(),
                },
                "residual": {
                    "requested": group_values(requested, actuator_names),
                    "effective_after_ctrlrange": group_values(effective, actuator_names),
                    "lost_to_ctrlrange": group_values(lost, actuator_names),
                    "lost_max_abs": float(np.max(np.abs(lost))),
                    "lost_above_1e_8_count": int(np.count_nonzero(np.abs(lost) > 1e-8)),
                },
                "terminated": bool(row["terminated"]),
            }
            if mode == "ppo":
                policy = captures[mode][endpoint - 1]
                action = policy["normalized_action"]
                low = policy["action_low"]
                high = policy["action_high"]
                at_state_bound = (action - low <= 2e-6) | (high - action <= 2e-6)
                at_unit_bound = np.abs(action) >= 1.0 - 2e-6
                item["policy"] = {
                    "actor_mu": policy["actor_mu"].tolist(),
                    "actor_sigma": policy["actor_sigma"].tolist(),
                    "deterministic_normalized_action": action.tolist(),
                    "state_feasible_low": low.tolist(),
                    "state_feasible_high": high.tolist(),
                    "unit_bound_count": int(at_unit_bound.sum()),
                    "state_feasible_bound_count": int(at_state_bound.sum()),
                    "unit_bound_actuators": [
                        actuator_names[index] for index in np.flatnonzero(at_unit_bound)
                    ],
                    "state_feasible_bound_actuators": [
                        actuator_names[index]
                        for index in np.flatnonzero(at_state_bound)
                    ],
                    "grouped_action": group_values(action, actuator_names),
                }
            rows[str(endpoint)] = item
        per_mode[mode] = rows

    comparison = {}
    first_score_worse = None
    for endpoint in FAILURE_ENDPOINTS:
        replay = per_mode["replay"][str(endpoint)]
        ppo = per_mode["ppo"][str(endpoint)]
        delta_score = ppo["objective_score"] - replay["objective_score"]
        if delta_score > 0 and first_score_worse is None:
            first_score_worse = endpoint
        comparison[str(endpoint)] = {
            "ppo_minus_replay_position_error_m": (
                ppo["position_error_m"] - replay["position_error_m"]
            ),
            "ppo_minus_replay_rotation_error_rad": (
                ppo["rotation_error_rad"] - replay["rotation_error_rad"]
            ),
            "ppo_minus_replay_objective_score": delta_score,
            "ppo_minus_replay_tool_position_error_vector_m": (
                np.asarray(ppo["tool_position_error_vector_actual_minus_reference_m"])
                - np.asarray(replay["tool_position_error_vector_actual_minus_reference_m"])
            ).tolist(),
            "full_qpos_l2_distance": float(np.linalg.norm(
                np.asarray(step_maps["ppo"][endpoint]["endpoint_qpos"], np.float64)
                - np.asarray(step_maps["replay"][endpoint]["endpoint_qpos"], np.float64)
            )),
            "full_qvel_l2_distance": float(np.linalg.norm(
                np.asarray(step_maps["ppo"][endpoint]["endpoint_qvel"], np.float64)
                - np.asarray(step_maps["replay"][endpoint]["endpoint_qvel"], np.float64)
            )),
            "contact_sets_differ": (
                per_mode["ppo"][str(endpoint)]["contact"]["active"]
                != per_mode["replay"][str(endpoint)]["contact"]["active"]
            ),
            "contacts_only_in_PPO": sorted(set(
                per_mode["ppo"][str(endpoint)]["contact"]["active"]
            ) - set(per_mode["replay"][str(endpoint)]["contact"]["active"])),
            "contacts_only_in_Replay": sorted(set(
                per_mode["replay"][str(endpoint)]["contact"]["active"]
            ) - set(per_mode["ppo"][str(endpoint)]["contact"]["active"])),
        }
    return {
        "per_mode": per_mode,
        "ppo_minus_replay": comparison,
        "first_outcome_endpoint_with_worse_PPO_score_in_window": first_score_worse,
        "failure_is_mixed_position_rotation": (
            per_mode["ppo"]["40"]["ellipse_squared_contributions"]["position"] > 0
            and per_mode["ppo"]["40"]["ellipse_squared_contributions"]["rotation"] > 0
        ),
    }


def write_probe_panel(path: Path, captures: dict[int, dict]) -> dict:
    if tuple(sorted(captures)) != PROBE_SOURCES:
        raise RuntimeError("final PPO trajectory did not provide source 32..39 probes")
    arrays = {
        "source_endpoint": np.asarray(PROBE_SOURCES, np.int64),
        "raw_observation": np.stack([
            captures[source]["raw_observation"] for source in PROBE_SOURCES
        ]),
        "actor_mu": np.stack([captures[source]["actor_mu"] for source in PROBE_SOURCES]),
        "actor_sigma": np.stack([
            captures[source]["actor_sigma"] for source in PROBE_SOURCES
        ]),
        "action_low": np.stack([
            captures[source]["action_low"] for source in PROBE_SOURCES
        ]),
        "action_high": np.stack([
            captures[source]["action_high"] for source in PROBE_SOURCES
        ]),
        "deterministic_action": np.stack([
            captures[source]["normalized_action"] for source in PROBE_SOURCES
        ]),
    }
    for index in range(len(captures[PROBE_SOURCES[0]]["hidden"])):
        arrays[f"rnn_hidden_{index}"] = np.stack([
            captures[source]["hidden"][index][:, 0, :].numpy()
            for source in PROBE_SOURCES
        ])
    np.savez_compressed(path, **arrays)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def state_l2(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], names) -> float:
    total = 0.0
    for name in names:
        delta = right[name].double() - left[name].double()
        total += float(torch.sum(delta * delta))
    return float(np.sqrt(total))


def build_probe_evolution(
    *, manifest: dict, panel_path: Path, output: Path, network_config,
) -> tuple[dict, dict[int, dict]]:
    from human2sim2robot.ppo.utils.models import ModelA2CContinuousLogStd

    with np.load(panel_path, allow_pickle=False) as panel:
        sources = panel["source_endpoint"].copy()
        raw = torch.as_tensor(panel["raw_observation"], dtype=torch.float32)
        low = torch.as_tensor(panel["action_low"], dtype=torch.float32)
        high = torch.as_tensor(panel["action_high"], dtype=torch.float32)
        hidden = [
            torch.as_tensor(panel[name].transpose(1, 0, 2), dtype=torch.float32)
            for name in sorted(key for key in panel.files if key.startswith("rnn_hidden_"))
        ]
    model = ModelA2CContinuousLogStd(
        network_config=network_config,
        actions_num=36,
        input_shape=(236,),
        normalize_value=True,
        normalize_input=True,
        value_size=1,
        num_seqs=len(sources),
    )
    model.eval()
    parameter_names = {name for name, _ in model.named_parameters()}

    def evaluate(state: dict[str, torch.Tensor]) -> dict:
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            normalized = model.norm_obs(raw, update_stats=False)
            mu, logstd, _, _ = model.a2c_network({
                "obs": normalized,
                "rnn_states": [value.clone() for value in hidden],
            })
            sigma = torch.exp(logstd)
            action = deterministic_truncated_action(mu, low, high)
        action_np = action.numpy()
        return {
            "mu": mu.numpy(),
            "sigma": sigma.numpy(),
            "action": action_np,
            "saturation": ((action_np - low.numpy() <= 2e-6)
                           | (high.numpy() - action_np <= 2e-6)),
        }

    initial_path = Path(manifest["initial_actor"]["path"])
    if sha256(initial_path) != manifest["initial_actor"]["artifact_sha256"]:
        raise RuntimeError("initial actor artifact changed")
    current = load_gzip_torch(initial_path)
    if model_state_sha256(current) != manifest["initial_actor"]["model_state_sha256"]:
        raise RuntimeError("initial actor hash mismatch")
    stages = []
    transitions = []
    update_transitions = {}

    def append_stage(label: str, kind: str, epoch: int, update: int, state) -> dict:
        result = evaluate(state)
        stages.append({
            "label": label,
            "kind": kind,
            "epoch": epoch,
            "update": update,
            "state_sha256": model_state_sha256(state),
            **result,
        })
        return result

    previous_eval = append_stage("initial", "initial", 0, 0, current)
    updates_by_epoch = {
        epoch: [
            row for row in manifest["update_reports"] if int(row["epoch"]) == epoch
        ]
        for epoch in range(1, 9)
    }
    normalization_by_epoch = {
        int(row["epoch"]): row for row in manifest["normalization_reports"]
    }
    for epoch in range(1, 9):
        for update in updates_by_epoch[epoch]:
            forward_path = Path(update["forward_state_patch"]["path"])
            optimizer_path = Path(update["optimizer_state_patch"]["path"])
            if sha256(forward_path) != update["forward_state_patch"]["sha256"]:
                raise RuntimeError("forward patch changed")
            if sha256(optimizer_path) != update["optimizer_state_patch"]["sha256"]:
                raise RuntimeError("optimizer patch changed")
            after_forward = apply_xor_patch(current, forward_path)
            forward_delta = state_l2(current, after_forward, current.keys())
            if forward_delta != 0.0:
                raise RuntimeError("post-fix actor forward mutated model state")
            after_optimizer = apply_xor_patch(after_forward, optimizer_path)
            after_eval = append_stage(
                f"epoch_{epoch:02d}_update_{int(update['update_in_epoch']):02d}",
                "optimizer", epoch, int(update["update_in_epoch"]), after_optimizer,
            )
            transition = {
                "kind": "optimizer",
                "epoch": epoch,
                "update_in_epoch": int(update["update_in_epoch"]),
                "global_actor_update": int(update["global_actor_update"]),
                "parameter_delta_l2": state_l2(
                    after_forward, after_optimizer, parameter_names
                ),
                "all_state_delta_l2": state_l2(
                    after_forward, after_optimizer, after_forward.keys()
                ),
                "maximum_abs_probe_mu_change": float(np.max(np.abs(
                    after_eval["mu"] - previous_eval["mu"]
                ))),
                "probe_saturation_count_before": int(previous_eval["saturation"].sum()),
                "probe_saturation_count_after": int(after_eval["saturation"].sum()),
            }
            transitions.append(transition)
            update_transitions[int(update["global_actor_update"])] = transition
            current = after_optimizer
            previous_eval = after_eval
        normalization = normalization_by_epoch[epoch]
        norm_path = Path(normalization["actor_state_patch"]["path"])
        if sha256(norm_path) != normalization["actor_state_patch"]["sha256"]:
            raise RuntimeError("normalization patch changed")
        after_norm = apply_xor_patch(current, norm_path)
        after_eval = append_stage(
            f"epoch_{epoch:02d}_RMS_commit", "RMS_commit", epoch, 0, after_norm
        )
        transitions.append({
            "kind": "RMS_commit",
            "epoch": epoch,
            "update_in_epoch": 0,
            "global_actor_update": epoch * 4,
            "parameter_delta_l2": state_l2(current, after_norm, parameter_names),
            "all_state_delta_l2": state_l2(current, after_norm, current.keys()),
            "maximum_abs_probe_mu_change": float(np.max(np.abs(
                after_eval["mu"] - previous_eval["mu"]
            ))),
            "probe_saturation_count_before": int(previous_eval["saturation"].sum()),
            "probe_saturation_count_after": int(after_eval["saturation"].sum()),
        })
        current = after_norm
        previous_eval = after_eval

    if model_state_sha256(current) != manifest["final_actor"]["model_state_sha256"]:
        raise RuntimeError("patch chain does not reach the final actor")
    np.savez_compressed(
        output,
        source_endpoint=sources,
        stage_label=np.asarray([row["label"] for row in stages]),
        stage_kind=np.asarray([row["kind"] for row in stages]),
        epoch=np.asarray([row["epoch"] for row in stages], np.int32),
        update_in_epoch=np.asarray([row["update"] for row in stages], np.int32),
        actor_mu=np.stack([row["mu"] for row in stages]),
        actor_sigma=np.stack([row["sigma"] for row in stages]),
        deterministic_action=np.stack([row["action"] for row in stages]),
        saturated=np.stack([row["saturation"] for row in stages]),
    )
    optimizer = [row for row in transitions if row["kind"] == "optimizer"]
    rms = [row for row in transitions if row["kind"] == "RMS_commit"]
    return {
        "artifact": {"path": str(output.resolve()), "sha256": sha256(output)},
        "probe_sources": sources.tolist(),
        "stages": len(stages),
        "optimizer_transitions": optimizer,
        "RMS_commit_transitions": rms,
        "optimizer_max_abs_mu_change": distribution(np.asarray([
            row["maximum_abs_probe_mu_change"] for row in optimizer
        ])),
        "RMS_commit_max_abs_mu_change": distribution(np.asarray([
            row["maximum_abs_probe_mu_change"] for row in rms
        ])),
        "largest_probe_change_transition": max(
            transitions, key=lambda row: row["maximum_abs_probe_mu_change"]
        ),
        "final_patch_chain_exact": True,
    }, update_transitions


def focus_probe_dynamics(
    path: Path, saturated_indices: np.ndarray, saturated_signs: np.ndarray
) -> dict:
    with np.load(path, allow_pickle=False) as data:
        sources = np.asarray(data["source_endpoint"])
        selected_sources = np.flatnonzero(np.isin(sources, FAILURE_SOURCES))
        mu = np.asarray(data["actor_mu"], np.float64)
        saturated = np.asarray(data["saturated"], bool)
        kinds = np.asarray(data["stage_kind"])
        labels = np.asarray(data["stage_label"])
    focus_mu = mu[:, selected_sources][:, :, saturated_indices]
    focus_saturation = saturated[:, selected_sources][:, :, saturated_indices]
    delta = np.diff(focus_mu, axis=0)
    toward_final = delta * saturated_signs.reshape(1, 1, -1)
    rows = []
    for index in range(len(delta)):
        rows.append({
            "from": str(labels[index]),
            "to": str(labels[index + 1]),
            "kind": str(kinds[index + 1]),
            "mean_signed_mu_change_toward_final_saturation": float(
                toward_final[index].mean()
            ),
            "maximum_signed_mu_change_toward_final_saturation": float(
                toward_final[index].max()
            ),
            "maximum_abs_mu_change": float(np.abs(delta[index]).max()),
            "positive_toward_final_fraction": float((toward_final[index] > 0).mean()),
            "focus_saturation_count_before": int(focus_saturation[index].sum()),
            "focus_saturation_count_after": int(focus_saturation[index + 1].sum()),
        })
    by_kind = {}
    for kind in ("optimizer", "RMS_commit"):
        selected = np.asarray([row["kind"] == kind for row in rows])
        by_kind[kind] = {
            "transitions": int(selected.sum()),
            "signed_mu_change_toward_final_saturation": distribution(
                toward_final[selected]
            ),
            "absolute_mu_change": distribution(np.abs(delta[selected])),
            "net_signed_mu_change_toward_final_saturation": float(
                toward_final[selected].sum()
            ),
            "transitions_increasing_focus_saturation_count": int(sum(
                row["focus_saturation_count_after"]
                > row["focus_saturation_count_before"]
                for row in rows if row["kind"] == kind
            )),
            "transitions_decreasing_focus_saturation_count": int(sum(
                row["focus_saturation_count_after"]
                < row["focus_saturation_count_before"]
                for row in rows if row["kind"] == kind
            )),
        }
    return {
        "source_endpoints": sources[selected_sources].tolist(),
        "final_saturated_action_indices": saturated_indices.tolist(),
        "initial_focus_saturation_count": int(focus_saturation[0].sum()),
        "final_focus_saturation_count": int(focus_saturation[-1].sum()),
        "by_transition_type": by_kind,
        "transitions": rows,
    }


def normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / np.sqrt(2.0 * np.pi)


def exact_truncated_kl(old_mu, old_sigma, new_mu, new_sigma, low, high) -> np.ndarray:
    tensors = [
        torch.as_tensor(value, dtype=torch.float64)
        for value in (old_mu, old_sigma, new_mu, new_sigma, low, high)
    ]
    mu0, sigma0, mu1, sigma1, low_t, high_t = tensors
    a0 = (low_t - mu0) / sigma0
    b0 = (high_t - mu0) / sigma0
    a1 = (low_t - mu1) / sigma1
    b1 = (high_t - mu1) / sigma1
    z0 = (torch.special.ndtr(b0) - torch.special.ndtr(a0)).clamp_min(1e-15)
    z1 = (torch.special.ndtr(b1) - torch.special.ndtr(a1)).clamp_min(1e-15)
    phi_a = normal_pdf(a0)
    phi_b = normal_pdf(b0)
    mean_z = (phi_a - phi_b) / z0
    second_z = 1.0 + (a0 * phi_a - b0 * phi_b) / z0
    mean_x = mu0 + sigma0 * mean_z
    second_x = mu0.square() + 2.0 * mu0 * sigma0 * mean_z + sigma0.square() * second_z
    new_quadratic = (
        second_x - 2.0 * mu1 * mean_x + mu1.square()
    ) / sigma1.square()
    kl = (
        torch.log(sigma1 * z1) - torch.log(sigma0 * z0)
        + 0.5 * (new_quadratic - second_z)
    )
    return kl.sum(dim=-1).numpy()


def update_dynamics(
    *, manifest: dict, transition_by_update: dict[int, dict], grad_clip: float,
) -> dict:
    rows = []
    for update in manifest["update_reports"]:
        artifact = Path(update["row_artifact"]["path"])
        if sha256(artifact) != update["row_artifact"]["sha256"]:
            raise RuntimeError("PPO row artifact changed")
        epoch_report = next(
            row for row in manifest["epoch_reports"]
            if int(row["epoch"]) == int(update["epoch"])
        )
        credit_path = Path(epoch_report["credit_rows"]["path"])
        if sha256(credit_path) != epoch_report["credit_rows"]["sha256"]:
            raise RuntimeError("credit row artifact changed")
        with np.load(credit_path, allow_pickle=False) as credit, np.load(
            artifact, allow_pickle=False
        ) as ppo:
            credit_ids = np.asarray(credit["sample_id"], np.int64)
            ppo_ids = np.asarray(ppo["sample_id"], np.int64)
            index = {int(sample): row for row, sample in enumerate(credit_ids)}
            order = np.asarray([index[int(sample)] for sample in ppo_ids], np.int64)
            old_neglogp = np.asarray(credit["old_neglogp"])[order].reshape(-1)
            new_neglogp = np.asarray(ppo["new_neglogp"]).reshape(-1)
            ratio = np.asarray(ppo["ratio"]).reshape(-1)
            exact_kl = exact_truncated_kl(
                np.asarray(credit["actor_mu"])[order],
                np.asarray(credit["actor_sigma"])[order],
                np.asarray(ppo["new_mu"]),
                np.asarray(ppo["new_sigma"]),
                np.asarray(credit["action_low"])[order],
                np.asarray(credit["action_high"])[order],
            )
            surr1 = np.asarray(ppo["unclipped_surrogate"]).reshape(-1)
            surr2 = np.asarray(ppo["clipped_surrogate"]).reshape(-1)
            before = float(update["gradient_norm_before_clip"])
            transition = transition_by_update[int(update["global_actor_update"])]
            rows.append({
                "epoch": int(update["epoch"]),
                "update_in_epoch": int(update["update_in_epoch"]),
                "global_actor_update": int(update["global_actor_update"]),
                "ratio": distribution(ratio),
                "ratio_outside_0_8_1_2_fraction": float(
                    ((ratio < 0.8) | (ratio > 1.2)).mean()
                ),
                "surrogate_changed_by_clip_fraction": float(
                    (np.abs(surr1 - surr2) > 1e-8).mean()
                ),
                "sample_log_ratio_KL_estimate": distribution(
                    new_neglogp - old_neglogp
                ),
                "exact_distribution_KL_old_to_current": distribution(exact_kl),
                "entropy": distribution(np.asarray(ppo["entropy_per_sample"])),
                "sigma": distribution(np.asarray(ppo["new_sigma"])),
                "gradient_norm_before_clip": before,
                "gradient_norm_after_clip_by_configured_contract": min(before, grad_clip),
                "gradient_after_clip_is_reconstructed_not_logged": True,
                "parameter_delta_l2": transition["parameter_delta_l2"],
                "maximum_abs_fixed_probe_mu_change": transition[
                    "maximum_abs_probe_mu_change"
                ],
                "fixed_probe_saturation_count_after": transition[
                    "probe_saturation_count_after"
                ],
            })
    return {
        "updates": rows,
        "ratio_outside_fraction": distribution(np.asarray([
            row["ratio_outside_0_8_1_2_fraction"] for row in rows
        ])),
        "exact_KL_mean_by_update": distribution(np.asarray([
            row["exact_distribution_KL_old_to_current"]["mean"] for row in rows
        ])),
        "parameter_delta_l2": distribution(np.asarray([
            row["parameter_delta_l2"] for row in rows
        ])),
        "gradient_norm_before_clip": distribution(np.asarray([
            row["gradient_norm_before_clip"] for row in rows
        ])),
    }


def subset_summary(data: dict[str, np.ndarray], selected: np.ndarray) -> dict:
    if not selected.any():
        return {"count": 0}
    termination = np.asarray(data["termination_outcome_endpoint"])[selected]
    return {
        "count": int(selected.sum()),
        "sampled_signed_action": distribution(data["signed_action"][selected]),
        "actor_signed_mu": distribution(data["signed_mu"][selected]),
        "raw_advantage": distribution(data["raw_advantage"][selected]),
        "normalized_advantage": distribution(data["normalized_advantage"][selected]),
        "return": distribution(data["return"][selected]),
        "critic_value": distribution(data["critic_value"][selected]),
        "termination_outcome_counts": {
            str(value): int((termination == value).sum())
            for value in sorted(set(termination.tolist()))
        },
    }


def credit_focus(
    *, manifest: dict, saturated_indices: np.ndarray, saturated_signs: np.ndarray,
    actuator_names: tuple[str, ...],
) -> dict:
    per_epoch = []
    combined = {name: [] for name in (
        "source", "sampled", "mu", "raw", "normalized", "return", "value", "termination"
    )}
    for epoch_report in manifest["epoch_reports"]:
        path = Path(epoch_report["credit_rows"]["path"])
        with np.load(path, allow_pickle=False) as source:
            source_endpoint = np.asarray(source["source_endpoint"]).reshape(-1)
            selected = np.isin(source_endpoint, FOCUS_TRAINING_SOURCES)
            sampled = np.asarray(source["sampled_action"])[selected][:, saturated_indices]
            mu = np.asarray(source["actor_mu"])[selected][:, saturated_indices]
            raw = np.asarray(source["raw_advantage"]).reshape(-1)[selected]
            normalized = np.asarray(source["normalized_advantage"]).reshape(-1)[selected]
            returns = np.asarray(source["return"]).reshape(-1)[selected]
            values = np.asarray(source["critic_value"]).reshape(-1)[selected]
            termination = np.asarray(source["termination_outcome_endpoint"])[selected]
            endpoint = source_endpoint[selected]
        dimensions = {}
        for local, index in enumerate(saturated_indices):
            signed_action = sampled[:, local] * saturated_signs[local]
            signed_mu = mu[:, local] * saturated_signs[local]
            data = {
                "signed_action": signed_action,
                "signed_mu": signed_mu,
                "raw_advantage": raw,
                "normalized_advantage": normalized,
                "return": returns,
                "critic_value": values,
                "termination_outcome_endpoint": termination,
            }
            dimensions[actuator_names[index]] = {
                "final_saturation_direction": int(saturated_signs[local]),
                "all_focus_samples": subset_summary(data, np.ones(len(raw), bool)),
                "opposing_or_zero": subset_summary(data, signed_action <= 0.0),
                "aligned_0_to_0_5": subset_summary(
                    data, (signed_action > 0.0) & (signed_action <= 0.5)
                ),
                "aligned_above_0_5": subset_summary(data, signed_action > 0.5),
                "sampled_action_normalized_advantage_correlation": (
                    float(np.corrcoef(signed_action, normalized)[0, 1])
                    if len(signed_action) > 1
                    and np.std(signed_action) > 0 and np.std(normalized) > 0
                    else None
                ),
            }
        per_epoch.append({
            "epoch": int(epoch_report["epoch"]),
            "focus_sample_count": int(selected.sum()),
            "source_endpoint_counts": {
                str(value): int((endpoint == value).sum())
                for value in FOCUS_TRAINING_SOURCES
            },
            "final_saturated_dimensions": dimensions,
        })
        combined["source"].append(endpoint)
        combined["sampled"].append(sampled)
        combined["mu"].append(mu)
        combined["raw"].append(raw)
        combined["normalized"].append(normalized)
        combined["return"].append(returns)
        combined["value"].append(values)
        combined["termination"].append(termination)
    joined = {name: np.concatenate(values, axis=0) for name, values in combined.items()}
    dimensions = {}
    for local, index in enumerate(saturated_indices):
        signed_action = joined["sampled"][:, local] * saturated_signs[local]
        data = {
            "signed_action": signed_action,
            "signed_mu": joined["mu"][:, local] * saturated_signs[local],
            "raw_advantage": joined["raw"],
            "normalized_advantage": joined["normalized"],
            "return": joined["return"],
            "critic_value": joined["value"],
            "termination_outcome_endpoint": joined["termination"],
        }
        dimensions[actuator_names[index]] = {
            "action_index": int(index),
            "final_saturation_direction": int(saturated_signs[local]),
            "all_focus_samples": subset_summary(data, np.ones(len(signed_action), bool)),
            "opposing_or_zero": subset_summary(data, signed_action <= 0.0),
            "aligned_0_to_0_5": subset_summary(
                data, (signed_action > 0.0) & (signed_action <= 0.5)
            ),
            "aligned_above_0_5": subset_summary(data, signed_action > 0.5),
            "sampled_action_normalized_advantage_correlation": (
                float(np.corrcoef(signed_action, joined["normalized"])[0, 1])
                if np.std(signed_action) > 0 and np.std(joined["normalized"]) > 0
                else None
            ),
        }
    return {
        "focus_source_endpoints": list(FOCUS_TRAINING_SOURCES),
        "final_endpoint40_saturated_dimensions": dimensions,
        "per_epoch": per_epoch,
        "limitation": (
            "These are observational on-policy associations. They do not identify a "
            "single action component's causal advantage because all 36 sampled actions "
            "and the visited state vary together."
        ),
    }


def stochastic_audit(*, backend, policy, boundary, seeds: list[int]) -> dict:
    actor_before = model_state_sha256(policy.model.state_dict())
    rows = []
    for seed in seeds:
        trace, _, _ = run_trial(
            backend=backend,
            policy=policy,
            boundary=boundary,
            mode=f"diagnostic_stochastic_seed_{seed}",
            deterministic=False,
            seed=seed,
        )
        steps = trace["steps"]
        endpoint_40 = next((row for row in steps if row["endpoint"] == 40), None)
        rows.append({
            "seed": seed,
            "attempted_intervals_including_failure": len(steps),
            "successful_intervals": len(steps) - int(trace["first_failure"] is not None),
            "first_failure": trace["first_failure"],
            "reached_endpoint_40": endpoint_40 is not None,
            "endpoint_40_score": (
                None if endpoint_40 is None else float(endpoint_40["objective_score"][0])
            ),
            "passed_full_40_step_window": trace["first_failure"] is None,
        })
    actor_after = model_state_sha256(policy.model.state_dict())
    if actor_before != actor_after:
        raise RuntimeError("stochastic read-only audit changed actor state")
    successful = np.asarray([row["successful_intervals"] for row in rows])
    endpoint_40_scores = np.asarray([
        row["endpoint_40_score"] for row in rows if row["endpoint_40_score"] is not None
    ])
    return {
        "seeds": seeds,
        "all_rollouts_reported": True,
        "selection_or_best_of_allowed": False,
        "rows": rows,
        "successful_interval_distribution": distribution(successful),
        "reached_endpoint_40_count": int(sum(row["reached_endpoint_40"] for row in rows)),
        "passed_endpoint_40_count": int(sum(
            row["reached_endpoint_40"]
            and row["first_failure"] != {
                "control_interval": 39,
                "endpoint": 40,
                "object_roles": ["tool"],
                "reason": "tracking_boundary",
            }
            for row in rows
        )),
        "full_40_step_pass_count": int(sum(
            row["passed_full_40_step_window"] for row in rows
        )),
        "endpoint_40_score_when_reached": (
            distribution(endpoint_40_scores) if len(endpoint_40_scores) else {"count": 0}
        ),
        "formal_CPU_acceptance_remains_deterministic": True,
    }


def write_summary(path: Path, report: dict) -> None:
    failure = report["failure_attribution"]
    dynamics = report["update_dynamics"]
    stochastic = report["deterministic_vs_stochastic"]["stochastic"]
    final = failure["per_mode"]["ppo"]["40"]
    lines = [
        "# Pour post-fix PPO policy-extremization audit v1",
        "",
        "Read-only audit. No training, optimizer step, checkpoint resume or chunk commit occurred.",
        "",
        "## Endpoint-40 failure",
        "",
        f"- position error: {final['position_error_m']:.9f} m",
        f"- rotation error: {final['rotation_error_rad']:.9f} rad",
        f"- normalized ellipse: {final['objective_score']:.9f}",
        f"- first worse PPO score in the 35--40 window: endpoint {failure['first_outcome_endpoint_with_worse_PPO_score_in_window']}",
        f"- unit-bound deterministic action components: {final['policy']['unit_bound_count']}",
        "",
        "## Saved update dynamics",
        "",
        f"- reconstructed actor updates: {len(dynamics['updates'])}",
        f"- largest exact-KL mean across updates: {dynamics['exact_KL_mean_by_update']['max']:.9g}",
        f"- largest ratio-outside-clip fraction: {dynamics['ratio_outside_fraction']['max']:.6f}",
        "",
        "## Frozen final-policy stochastic diagnostic",
        "",
        f"- stochastic rollouts: {len(stochastic['rows'])}",
        f"- reached endpoint 40: {stochastic['reached_endpoint_40_count']}",
        f"- passed full 40-step window: {stochastic['full_40_step_pass_count']}",
        "",
        "No algorithm change is selected by this report; it supplies evidence for the next single-variable decision.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_postfix_policy_extremization_audit_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_postfix_policy_extremization_audit_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)

    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    formal = json.loads(paths["formal_report"].read_text())
    if formal.get("status") != "completed_postfix_diagnostic_no_commit":
        raise ValueError("source run is not valid post-fix evidence")
    manifest = json.loads(paths["credit_manifest"].read_text())
    if manifest.get("actor_updates") != 32 or manifest.get("epochs") != 8:
        raise ValueError("credit manifest does not contain 32 updates and 8 epochs")
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    spec, _ = load_truncated_gaussian_profile(paths["distribution_profile"])
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(
            reference_start_index=0,
            asymmetric_critic=False,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,),
            object_roles=("tool", "target"),
            objective=objective,
            observation=observation,
            residual=residual,
        ),
        seed=0,
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])

    replay, replay_contacts, _ = run_trial(
        backend=backend, policy=None, boundary=boundary,
        mode="Replay", deterministic=True,
    )
    _require_exact_trace(replay, formal["replay_validation"], "post-fix Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    if model_state_sha256(checkpoint["model"]) != manifest["final_actor"]["model_state_sha256"]:
        raise ValueError("checkpoint actor differs from credit patch chain")
    temp = tempfile.TemporaryDirectory(prefix=".postfix_extremization_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temp.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        ppo, ppo_contacts, ppo_captures = run_trial(
            backend=backend, policy=policy, boundary=boundary,
            mode="PPO", deterministic=True, capture_sources=PROBE_SOURCES,
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "post-fix PPO")

        control_indices = tuple(env.env_cfg.residual.hand_control_indices)
        actuator_names, _, actuator_units = _actuator_metadata(
            env.env.model_cpu, control_indices
        )
        qvel_indices = qvel_metadata(env.env.model_cpu, control_indices)
        failure = failure_attribution(
            traces={"replay": replay, "ppo": ppo},
            extras={"replay": replay_contacts, "ppo": ppo_contacts},
            captures={"ppo": ppo_captures}, reference=reference,
            actuator_names=actuator_names, qvel_indices=qvel_indices,
        )

        args.output.mkdir(parents=True)
        shutil.copy2(args.contract, args.output / "contract.yaml")
        panel_path = args.output / contract["artifacts"]["fixed_probe_panel"]
        panel_artifact = write_probe_panel(panel_path, ppo_captures)
        evolution, update_transitions = build_probe_evolution(
            manifest=manifest,
            panel_path=panel_path,
            output=args.output / contract["artifacts"]["probe_evolution"],
            network_config=_build_network_config(4),
        )
        updates = update_dynamics(
            manifest=manifest,
            transition_by_update=update_transitions,
            grad_clip=float(contract["update_dynamics"]["configured_gradient_norm_clip"]),
        )
        final_policy = failure["per_mode"]["ppo"]["40"]["policy"]
        final_action = np.asarray(final_policy["deterministic_normalized_action"])
        saturated = np.flatnonzero(np.abs(final_action) >= 1.0 - 2e-6)
        signs = np.sign(final_action[saturated]).astype(np.int64)
        evolution["failure_focus"] = focus_probe_dynamics(
            Path(evolution["artifact"]["path"]), saturated, signs
        )
        credit = credit_focus(
            manifest=manifest, saturated_indices=saturated,
            saturated_signs=signs, actuator_names=actuator_names,
        )
        stochastic = stochastic_audit(
            backend=backend, policy=policy, boundary=boundary,
            seeds=list(contract["stochastic_diagnostic"]["seeds"]),
        )
        policy.writer.close()
    finally:
        temp.cleanup()

    report = {
        "schema": "taco_pour_postfix_policy_extremization_audit_v1",
        "status": "completed_read_only_no_algorithm_change",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "checkpoint_resume_for_training": False,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "source_run": {
            "report": {"path": str(paths["formal_report"]), "sha256": sha256(paths["formal_report"])},
            "credit_manifest": {"path": str(paths["credit_manifest"]), "sha256": sha256(paths["credit_manifest"])},
            "formal_trace_bitwise_reproduced": {"Replay": True, "PPO": True},
        },
        "actuator_contract": {
            "names": list(actuator_names),
            "units": list(actuator_units),
        },
        "failure_attribution": failure,
        "fixed_probe_panel": panel_artifact,
        "fixed_probe_evolution": evolution,
        "update_dynamics": updates,
        "training_credit_source35_39": credit,
        "deterministic_vs_stochastic": {
            "formal_deterministic": {
                "attempted_intervals_including_failure": len(ppo["steps"]),
                "successful_intervals": len(ppo["steps"]) - 1,
                "first_failure": ppo["first_failure"],
            },
            "stochastic": stochastic,
        },
        "decision_boundary": {
            "normalization_blocker_reopened": False,
            "new_algorithm_change_selected": False,
            "second_fresh_PPO_authorized": False,
            "reason": (
                "This report separates saved credit, optimizer-pass, RMS-commit and "
                "stochastic/deterministic evidence. It does not tune or select an "
                "algorithm change automatically."
            ),
        },
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path.resolve()),
        "failure_endpoint": ppo["first_failure"],
        "endpoint40_unit_bound_count": final_policy["unit_bound_count"],
        "largest_probe_transition": evolution["largest_probe_change_transition"],
        "stochastic_full_window_passes": stochastic["full_40_step_pass_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
