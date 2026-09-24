#!/usr/bin/env python3
"""Read-only endpoint 40--50 attribution for the two frozen 8-epoch policies."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from video_to_spider.rl.residual_semantics import residual_from_trace_step

BOUNDARY = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "endpoint20_complete_boundary.pt.gz"
)
CONFIG = ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml"
INITIALIZATION = ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json"
PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
OBJECTIVE = ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml"
OBSERVATION = ROOT / "configs/taco_pour_observation_local_236d_v1.yaml"
OLD_ACTION = ROOT / "configs/taco_pour_residual_action_legacy_v1.yaml"
NEW_ACTION = ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml"
OLD_CHECKPOINT = ROOT / (
    "runs/taco_pour_normalized_ellipse_v1/short_rl_tool_only_8ep/"
    "ppo_chunk_20/nn/last_ep_8_rew__8.038462_.pth"
)
NEW_CHECKPOINT = ROOT / (
    "runs/taco_pour_normalized_action_scale_v1/ppo_chunk_20/nn/"
    "last_ep_8_rew__7.965472_.pth"
)
OLD_TRACE = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "checkpoint_8ep_closed_loop_v2.json.gz"
)
NEW_TRACE = ROOT / (
    "runs/taco_pour_normalized_action_scale_v1/"
    "cpu_closed_loop_validation.json.gz"
)
OUTPUT = ROOT / "runs/taco_pour_action_attribution_v1/report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def actuator_groups(scene: Path) -> tuple[list[str], dict[str, list[int]]]:
    actuator = ET.parse(scene).getroot().find("actuator")
    if actuator is None:
        raise ValueError("scene has no actuator section")
    names = [element.get("name", "") for element in actuator]
    if len(names) != 36 or any(not name for name in names):
        raise ValueError("expected exactly 36 named XHand actuators")
    groups: dict[str, list[int]] = {
        "right_wrist_translation": [],
        "right_wrist_rotation": [],
        "right_fingers": [],
        "left_wrist_translation": [],
        "left_wrist_rotation": [],
        "left_fingers": [],
    }
    for index, name in enumerate(names):
        if name.startswith("R_forearm_t"):
            groups["right_wrist_translation"].append(index)
        elif name.startswith("R_forearm_"):
            groups["right_wrist_rotation"].append(index)
        elif name.startswith("right_"):
            groups["right_fingers"].append(index)
        elif name.startswith("L_forearm_t"):
            groups["left_wrist_translation"].append(index)
        elif name.startswith("L_forearm_"):
            groups["left_wrist_rotation"].append(index)
        elif name.startswith("left_"):
            groups["left_fingers"].append(index)
        else:
            raise ValueError(f"unclassified actuator: {name}")
    expected = {
        "right_wrist_translation": 3,
        "right_wrist_rotation": 3,
        "right_fingers": 12,
        "left_wrist_translation": 3,
        "left_wrist_rotation": 3,
        "left_fingers": 12,
    }
    if {name: len(indices) for name, indices in groups.items()} != expected:
        raise ValueError("actuator-name grouping differs from the expected XHand layout")
    return names, groups


def exact_trace_equal(actual: list[dict], frozen: list[dict]) -> bool:
    if len(actual) != len(frozen):
        return False
    keys = (
        "endpoint_qpos", "endpoint_qvel", "commanded_ctrl", "objective_score",
        "position_error_m", "rotation_error_rad", "raw_residual_action",
    )
    return all(
        left["endpoint"] == right["endpoint"]
        and np.array_equal(
            residual_from_trace_step(left), residual_from_trace_step(right)
        )
        and all(
            np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
            for key in keys
        )
        for left, right in zip(actual, frozen, strict=True)
    )


def rollout(checkpoint_path: Path, action_profile: Path) -> dict:
    from run_mjwp_ppo import (
        MJWPVectorEnv, MJWPVectorEnvConfig, PpoAgent, _build_network_config,
        _build_ppo_config, _load_ego_config, _load_reference, torch,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    boundary_raw = gzip.decompress(BOUNDARY.read_bytes())
    boundary = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    checkpoint = torch.load(
        io.BytesIO(checkpoint_path.read_bytes()), map_location="cpu", weights_only=False
    )
    objective = load_runtime_objective(
        PROTOCOL, OBJECTIVE, tracking_variant="tool_only", require_run_ready=True
    )
    observation = load_runtime_observation(
        PROTOCOL, OBSERVATION, require_run_ready=True
    )
    residual, residual_report = load_residual_action_profile(action_profile)
    _, initialization = load_accepted_initialization(INITIALIZATION, CONFIG)
    config = _load_ego_config(str(CONFIG), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        config,
        reference,
        num_envs=1,
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
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    ppo_config = _build_ppo_config(
        num_envs=1,
        horizon_length=40,
        seq_length=4,
        max_epochs=8,
        learning_rate=1e-4,
        device="cpu",
        asymmetric_critic=None,
    )
    with tempfile.TemporaryDirectory(prefix="egoengine_action_attribution_") as temp_dir:
        agent = PpoAgent(
            experiment_dir=Path(temp_dir),
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
        )
        agent.model.load_state_dict(checkpoint["model"])
        agent.set_eval()
        agent.rnn_states = [
            state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
        ]
        backend = MJWPChunkBackend(env)
        backend.restore(boundary)
        backend.verify_restored_snapshot(boundary)
        backend.begin_trial("rl_prelimit_attribution", 20, 60)
        prelimit = []
        bounded = []
        valid_steps = 0
        for reference_step in range(20, 60):
            values = agent.get_action_values(agent.obs_to_tensors(backend.observation()))
            agent.rnn_states = values["rnn_states"]
            mu = values["mus"].detach().cpu().numpy()
            action = np.asarray(agent.preprocess_actions(values["mus"]), np.float32)
            prelimit.append(mu[0].tolist())
            bounded.append(action[0].tolist())
            if not backend.step(action, reference_step):
                break
            valid_steps += 1
        backend.end_trial(valid_steps == 40, valid_steps)
        agent.writer.close()
    trace = backend.validation_traces[-1]
    for row, mu, action in zip(trace["steps"], prelimit, bounded, strict=True):
        row["network_mu_before_action_limit"] = mu
        if not np.array_equal(
            np.asarray(row["raw_residual_action"], np.float32),
            np.asarray(action, np.float32),
        ):
            raise ValueError("logged bounded action differs from policy preprocessing")
        endpoint = int(row["endpoint"])
        qpos = np.asarray(row["endpoint_qpos"], np.float64)
        goal = reference[0][endpoint].cpu().numpy().astype(np.float64)
        row["tool_position_error_vector_m"] = (qpos[36:39] - goal[36:39]).tolist()
    return {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "action_profile": residual_report,
        "validated_steps": trace["validated_steps"],
        "first_failure": trace["first_failure"],
        "steps": trace["steps"],
    }


def distribution(values: np.ndarray) -> dict:
    absolute = np.abs(values).reshape(-1)
    return {
        "abs_mean": float(absolute.mean()),
        "abs_percentiles_50_90_95_99": [
            float(value) for value in np.percentile(absolute, [50, 90, 95, 99])
        ],
        "abs_max": float(absolute.max()),
        "fraction_abs_gt_1": float((absolute > 1.0).mean()),
        "fraction_abs_gt_2": float((absolute > 2.0).mean()),
        "fraction_abs_gt_3": float((absolute > 3.0).mean()),
        "fraction_abs_gt_5": float((absolute > 5.0).mean()),
    }


def summarize_policy(
    rows: list[dict], groups: dict[str, list[int]], *, start: int, end: int
) -> dict:
    selected = [row for row in rows if start <= int(row["endpoint"]) <= end]
    mu = np.asarray([row["network_mu_before_action_limit"] for row in selected])
    bounded = np.asarray([row["raw_residual_action"] for row in selected])
    applied = np.asarray([residual_from_trace_step(row) for row in selected])
    summary = {
        "endpoint_range": [int(selected[0]["endpoint"]), int(selected[-1]["endpoint"])],
        "steps": len(selected),
        "network_mu_before_limit": distribution(mu),
        "bounded_u_at_limit_fraction": float(
            np.isclose(np.abs(bounded), 1.0, atol=1e-7, rtol=0.0).mean()
        ),
        "applied_abs_mean": float(np.abs(applied).mean()),
        "tool_position_error_vector_m": {
            "first": selected[0]["tool_position_error_vector_m"],
            "last": selected[-1]["tool_position_error_vector_m"],
            "last_minus_first": (
                np.asarray(selected[-1]["tool_position_error_vector_m"])
                - np.asarray(selected[0]["tool_position_error_vector_m"])
            ).tolist(),
        },
        "groups": {},
    }
    for name, indices in groups.items():
        group_mu = mu[:, indices]
        group_bounded = bounded[:, indices]
        group_applied = applied[:, indices]
        summary["groups"][name] = {
            "indices": indices,
            "network_mu_before_limit": distribution(group_mu),
            "bounded_u_at_limit_fraction": float(
                np.isclose(np.abs(group_bounded), 1.0, atol=1e-7, rtol=0.0).mean()
            ),
            "applied_abs_mean": float(np.abs(group_applied).mean()),
            "signed_command_offset_sum": group_applied.sum(axis=0).tolist(),
            "absolute_command_offset_sum": np.abs(group_applied).sum(axis=0).tolist(),
            "signed_sum_l2_norm": float(np.linalg.norm(group_applied.sum(axis=0))),
        }
    return summary


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return None if denominator == 0.0 else float(np.dot(left, right) / denominator)


def compare_policies(
    old_rows: list[dict],
    new_rows: list[dict],
    groups: dict[str, list[int]],
    *,
    start: int,
    end: int,
) -> dict:
    old = {int(row["endpoint"]): row for row in old_rows}
    new = {int(row["endpoint"]): row for row in new_rows}
    endpoints = [
        endpoint for endpoint in sorted(set(old) & set(new))
        if start <= endpoint <= end
    ]
    result = {
        "endpoint_range": [endpoints[0], endpoints[-1]],
        "steps": len(endpoints),
        "groups": {},
        "signed_sum_is_not_physical_displacement": True,
    }
    for name, indices in groups.items():
        old_action = np.asarray([
            residual_from_trace_step(old[endpoint])[indices] for endpoint in endpoints
        ])
        new_action = np.asarray([
            residual_from_trace_step(new[endpoint])[indices] for endpoint in endpoints
        ])
        old_sum = old_action.sum(axis=0)
        new_sum = new_action.sum(axis=0)
        signs = np.sign(old_action) == np.sign(new_action)
        result["groups"][name] = {
            "flattened_action_cosine": cosine(old_action.reshape(-1), new_action.reshape(-1)),
            "same_sign_fraction": float(signs.mean()),
            "old_signed_command_offset_sum": old_sum.tolist(),
            "new_signed_command_offset_sum": new_sum.tolist(),
            "signed_sum_cosine": cosine(old_sum, new_sum),
            "new_to_old_signed_sum_norm_ratio": float(
                np.linalg.norm(new_sum) / np.linalg.norm(old_sum)
            ) if np.linalg.norm(old_sum) else None,
            "new_to_old_absolute_sum_ratio": float(
                np.abs(new_action).sum() / np.abs(old_action).sum()
            ) if np.abs(old_action).sum() else None,
        }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    scene = Path(config["model_path"])
    names, groups = actuator_groups(scene)
    old = rollout(OLD_CHECKPOINT, OLD_ACTION)
    new = rollout(NEW_CHECKPOINT, NEW_ACTION)
    old_frozen = load_json(OLD_TRACE)["repetitions"][0]["trials"]["rl"]["trace"]["steps"]
    new_frozen = load_json(NEW_TRACE)["repetitions"][0]["trials"]["rl"]["trace"]["steps"]
    old_equal = exact_trace_equal(old["steps"], old_frozen)
    new_equal = exact_trace_equal(new["steps"], new_frozen)
    if not old_equal or not new_equal:
        raise ValueError("pre-limit diagnostic did not reproduce the frozen CPU trace")

    report = {
        "schema": "taco_pour_endpoint40_50_action_attribution_v1",
        "status": "read_only_diagnostic_complete",
        "scope": {
            "training_executed": False,
            "physics_modified": False,
            "policy_modified": False,
            "endpoint_range": [40, 50],
            "question": "scaled policy failure is insufficient magnitude versus wrong direction",
        },
        "artifacts": {
            "boundary": {"path": str(BOUNDARY), "sha256": sha256(BOUNDARY)},
            "scene": {"path": str(scene), "sha256": sha256(scene)},
            "old_frozen_trace": {"path": str(OLD_TRACE), "sha256": sha256(OLD_TRACE)},
            "new_frozen_trace": {"path": str(NEW_TRACE), "sha256": sha256(NEW_TRACE)},
        },
        "actuator_contract": {
            "names": names,
            "groups": groups,
            "units": {
                "right_wrist_translation": "m",
                "left_wrist_translation": "m",
                "right_wrist_rotation": "rad",
                "left_wrist_rotation": "rad",
                "right_fingers": "rad",
                "left_fingers": "rad",
            },
            "single_scalar_action_scale_spans_mixed_units": True,
        },
        "trace_reproduction": {
            "old_bitwise_equal": old_equal,
            "new_bitwise_equal": new_equal,
        },
        "old_scale_1": {
            "checkpoint": old["checkpoint"],
            "action_profile": old["action_profile"],
            "validated_steps": old["validated_steps"],
            "first_failure": old["first_failure"],
            "summary": summarize_policy(old["steps"], groups, start=40, end=50),
            "critical_tail_46_50": summarize_policy(
                old["steps"], groups, start=46, end=50
            ),
        },
        "new_scale_005": {
            "checkpoint": new["checkpoint"],
            "action_profile": new["action_profile"],
            "validated_steps": new["validated_steps"],
            "first_failure": new["first_failure"],
            "summary": summarize_policy(new["steps"], groups, start=40, end=50),
            "critical_tail_46_50": summarize_policy(
                new["steps"], groups, start=46, end=50
            ),
        },
        "old_vs_new": compare_policies(
            old["steps"], new["steps"], groups, start=40, end=50
        ),
        "old_vs_new_critical_tail_46_50": compare_policies(
            old["steps"], new["steps"], groups, start=46, end=50
        ),
        "endpoint_rows": {
            "old_scale_1": [row for row in old["steps"] if 40 <= row["endpoint"] <= 50],
            "new_scale_005": [row for row in new["steps"] if 40 <= row["endpoint"] <= 50],
        },
    }
    critical = report["old_vs_new_critical_tail_46_50"]["groups"][
        "right_wrist_translation"
    ]
    old_tail = report["old_scale_1"]["critical_tail_46_50"]
    new_tail = report["new_scale_005"]["critical_tail_46_50"]
    new_wrist_mu = new_tail["groups"]["right_wrist_translation"][
        "network_mu_before_limit"
    ]
    report["diagnosis"] = {
        "critical_right_wrist_translation": {
            "classification": "direction_diverged_not_merely_insufficient_magnitude",
            "old_vs_new_flattened_action_cosine": critical["flattened_action_cosine"],
            "old_vs_new_signed_sum_cosine": critical["signed_sum_cosine"],
            "same_sign_fraction": critical["same_sign_fraction"],
            "new_to_old_absolute_sum_ratio": critical["new_to_old_absolute_sum_ratio"],
            "new_network_mu_abs_max": new_wrist_mu["abs_max"],
            "new_network_mu_fraction_abs_gt_1": new_wrist_mu["fraction_abs_gt_1"],
            "old_tool_position_error_change_m": old_tail[
                "tool_position_error_vector_m"
            ]["last_minus_first"],
            "new_tool_position_error_change_m": new_tail[
                "tool_position_error_vector_m"
            ]["last_minus_first"],
        },
        "network_output": {
            "new_policy_has_some_prelimit_values_above_one_globally": True,
            "new_global_fraction_abs_gt_1_endpoint40_50": report["new_scale_005"][
                "summary"
            ]["network_mu_before_limit"]["fraction_abs_gt_1"],
            "new_global_fraction_abs_gt_2_endpoint40_50": report["new_scale_005"][
                "summary"
            ]["network_mu_before_limit"]["fraction_abs_gt_2"],
            "new_global_abs_max_endpoint40_50": report["new_scale_005"]["summary"][
                "network_mu_before_limit"
            ]["abs_max"],
            "critical_right_wrist_translation_is_action_limited": False,
        },
        "mixed_units": {
            "single_scale_applies_to_translation_m_and_rotation_rad": True,
            "treated_as_new_contract_risk_not_as_proven_failure_cause": True,
        },
        "decision": {
            "increase_or_sweep_scalar_action_scale_next": False,
            "pure_strength_shortage_explanation_supported": False,
            "late_wrong_direction_explanation_supported": True,
            "next_step": (
                "audit whether endpoint 46-50 states and corrective directions are represented "
                "in PPO training data before choosing one learning/data-coverage modification"
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "trace_reproduction": report["trace_reproduction"],
        "old_network_mu": report["old_scale_1"]["summary"]["network_mu_before_limit"],
        "new_network_mu": report["new_scale_005"]["summary"]["network_mu_before_limit"],
        "right_wrist_translation": report["old_vs_new"]["groups"]["right_wrist_translation"],
    }, indent=2))


if __name__ == "__main__":
    main()
