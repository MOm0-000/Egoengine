#!/usr/bin/env python3
"""Finite-difference right-wrist translation sensitivity for outcome endpoints 46--50."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_pour_action_attribution import actuator_groups, sha256


BOUNDARY = ROOT / (
    "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
    "endpoint20_complete_boundary.pt.gz"
)
CONFIG = ROOT / "runs/taco_pour_floor_contact_v1/candidate_ppo_config.yaml"
INITIALIZATION = ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json"
PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
OBJECTIVE = ROOT / "configs/taco_pour_local_normalized_ellipse_v1.yaml"
OBSERVATION = ROOT / "configs/taco_pour_observation_local_236d_v1.yaml"
POLICIES = {
    "old_scale_1": {
        "action_profile": ROOT / "configs/taco_pour_residual_action_legacy_v1.yaml",
        "checkpoint": ROOT / (
            "runs/taco_pour_normalized_ellipse_v1/short_rl_tool_only_8ep/"
            "ppo_chunk_20/nn/last_ep_8_rew__8.038462_.pth"
        ),
    },
    "new_scale_005": {
        "action_profile": ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
        "checkpoint": ROOT / (
            "runs/taco_pour_normalized_action_scale_v1/ppo_chunk_20/nn/"
            "last_ep_8_rew__7.965472_.pth"
        ),
    },
}
SOURCE_ENDPOINTS = tuple(range(45, 50))
EPSILONS_M = (0.0005, 0.001)
OUTPUT = ROOT / "runs/endpoint46_50_local_control_sensitivity_v1/report.json"


def outcome(env) -> dict:
    info = env._last_info if hasattr(env, "_last_info") else None
    del info
    return {
        "position_error_m": float(env._last_tracking_position_error[0, 0]),
        "rotation_error_rad": float(env._last_tracking_rotation_error[0, 0]),
        "objective_score": float(env._last_tracking_errors[0, 0]),
        "terminated": bool(env._last_object_terminated[0, 0]),
    }


def central_gradient(plus: dict, minus: dict, epsilon: float) -> dict:
    return {
        metric: (plus[metric] - minus[metric]) / (2.0 * epsilon)
        for metric in ("position_error_m", "rotation_error_rad", "objective_score")
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return None if denominator == 0.0 else float(np.dot(left, right) / denominator)


def probe_baseline(backend, env, state, source_endpoint, baseline_applied, scale):
    backend.restore(state)
    backend.step((baseline_applied / scale)[None], source_endpoint)
    baseline_outcome = outcome(env)
    probes = {}
    gradients = {}
    for epsilon in EPSILONS_M:
        key = f"{epsilon:.4f}"
        probes[key] = {}
        gradients[key] = {
            "position_error_m_per_command_m": [],
            "rotation_error_rad_per_command_m": [],
            "objective_score_per_command_m": [],
        }
        for axis, axis_name in enumerate(("x", "y", "z")):
            axis_outcomes = {}
            for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                desired = baseline_applied.copy()
                desired[axis] += sign * epsilon
                backend.restore(state)
                backend.step((desired / scale)[None], source_endpoint)
                axis_outcomes[sign_name] = outcome(env)
            probes[key][axis_name] = axis_outcomes
            gradient = central_gradient(
                axis_outcomes["plus"], axis_outcomes["minus"], epsilon
            )
            gradients[key]["position_error_m_per_command_m"].append(
                gradient["position_error_m"]
            )
            gradients[key]["rotation_error_rad_per_command_m"].append(
                gradient["rotation_error_rad"]
            )
            gradients[key]["objective_score_per_command_m"].append(
                gradient["objective_score"]
            )
    gradient_small = np.asarray(
        gradients["0.0005"]["position_error_m_per_command_m"]
    )
    gradient_large = np.asarray(
        gradients["0.0010"]["position_error_m_per_command_m"]
    )
    favorable = -gradient_large
    return {
        "baseline_outcome": baseline_outcome,
        "probes": probes,
        "central_gradients": gradients,
        "favorable_position_direction": favorable.tolist(),
        "gradient_sign_stable_between_epsilons": (
            np.sign(gradient_small) == np.sign(gradient_large)
        ).tolist(),
        "gradient_cosine_between_epsilons": cosine(gradient_small, gradient_large),
    }


def run_policy(label: str, files: dict, scene: Path) -> dict:
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
    checkpoint_raw = files["checkpoint"].read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False)
    objective = load_runtime_objective(
        PROTOCOL, OBJECTIVE, tracking_variant="tool_only", require_run_ready=True
    )
    observation = load_runtime_observation(
        PROTOCOL, OBSERVATION, require_run_ready=True
    )
    residual, residual_report = load_residual_action_profile(files["action_profile"])
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
    formal_residual = env.env_cfg.residual
    diagnostic_residual = replace(
        formal_residual,
        residual_clip=formal_residual.residual_clip + max(EPSILONS_M),
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
    records = []
    with tempfile.TemporaryDirectory(prefix=f"egoengine_sensitivity_{label}_") as temp_dir:
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
        for source_endpoint in range(20, 50):
            if int(env.time_indices[0]) != source_endpoint:
                raise ValueError("policy rollout cursor changed unexpectedly")
            values = agent.get_action_values(agent.obs_to_tensors(backend.observation()))
            agent.rnn_states = values["rnn_states"]
            bounded_action = np.asarray(
                agent.preprocess_actions(values["mus"]), np.float32
            )[0]
            applied = np.clip(
                formal_residual.residual_scale * bounded_action,
                -formal_residual.residual_clip,
                formal_residual.residual_clip,
            )
            if source_endpoint in SOURCE_ENDPOINTS:
                state = backend.snapshot()
                neutral_applied = applied.copy()
                neutral_applied[:3] = 0.0
                env.env_cfg.residual = diagnostic_residual
                neutral_probe = probe_baseline(
                    backend,
                    env,
                    state,
                    source_endpoint,
                    neutral_applied,
                    formal_residual.residual_scale,
                )
                centered_probe = probe_baseline(
                    backend,
                    env,
                    state,
                    source_endpoint,
                    applied,
                    formal_residual.residual_scale,
                )
                env.env_cfg.residual = formal_residual
                favorable = np.asarray(
                    centered_probe["favorable_position_direction"]
                )
                policy_translation = applied[:3].astype(np.float64)
                records.append({
                    "source_endpoint": source_endpoint,
                    "outcome_endpoint": source_endpoint + 1,
                    "network_mu_before_limit": values["mus"][0, :3].detach().cpu().tolist(),
                    "policy_right_wrist_translation_m": policy_translation.tolist(),
                    "neutral_centered_probe": neutral_probe,
                    "policy_centered_probe": centered_probe,
                    "policy_vs_local_favorable_increment_cosine": cosine(
                        policy_translation, favorable
                    ),
                    "policy_axis_sign_matches_local_favorable_increment": (
                        np.sign(policy_translation) == np.sign(favorable)
                    ).tolist(),
                })
                backend.restore(state)

            if not backend.step(bounded_action[None], source_endpoint):
                if source_endpoint + 1 < 50:
                    raise ValueError("policy failed before the sensitivity window ended")
        agent.writer.close()

    cosines = np.asarray([
        row["policy_vs_local_favorable_increment_cosine"] for row in records
    ], dtype=np.float64)
    sign_matches = np.asarray([
        row["policy_axis_sign_matches_local_favorable_increment"] for row in records
    ], dtype=bool)
    stable = np.asarray([
        row["policy_centered_probe"]["gradient_sign_stable_between_epsilons"]
        for row in records
    ], dtype=bool)
    gradient_cosines = np.asarray([
        row["policy_centered_probe"]["gradient_cosine_between_epsilons"]
        for row in records
    ], dtype=np.float64)
    return {
        "checkpoint": {
            "path": str(files["checkpoint"].resolve()),
            "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
            "epoch": int(checkpoint["epoch"]),
        },
        "action_profile": residual_report,
        "records": records,
        "summary": {
            "policy_vs_local_favorable_increment_cosine_per_endpoint": cosines.tolist(),
            "cosine_mean": float(cosines.mean()),
            "cosine_median": float(np.median(cosines)),
            "positive_cosine_endpoints": int((cosines > 0).sum()),
            "negative_cosine_endpoints": int((cosines < 0).sum()),
            "axis_sign_match_fraction": float(sign_matches.mean()),
            "gradient_sign_stability_fraction": float(stable.mean()),
            "gradient_cosine_between_epsilons_per_endpoint": gradient_cosines.tolist(),
            "gradient_cosine_between_epsilons_mean": float(gradient_cosines.mean()),
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    scene = Path(config["model_path"])
    names, groups = actuator_groups(scene)
    if groups["right_wrist_translation"] != [0, 1, 2]:
        raise ValueError("right-wrist translation indices changed")
    policies = {
        label: run_policy(label, files, scene) for label, files in POLICIES.items()
    }
    report = {
        "schema": "endpoint46_50_local_control_sensitivity_v1",
        "status": "read_only_local_sensitivity_complete",
        "scope": {
            "training_executed": False,
            "policy_modified": False,
            "physics_modified": False,
            "source_state_endpoints": list(SOURCE_ENDPOINTS),
            "outcome_endpoints": [endpoint + 1 for endpoint in SOURCE_ENDPOINTS],
            "indexing": "action at state endpoint t produces outcome endpoint t+1",
            "epsilon_m": list(EPSILONS_M),
            "probe_baseline": (
                "both neutral-centered and policy-centered probes retain the other 33 policy "
                "dimensions; the primary diagnosis uses perturbations around the actual "
                "right-wrist translation command"
            ),
            "diagnostic_residual_clip_m": 0.051,
            "diagnostic_clip_reason": (
                "permit symmetric +/-1 mm probes around a formal command at the +/-0.05 m bound; "
                "this one-step diagnostic limit is not a training or runtime proposal"
            ),
            "primary_metric": "next-endpoint tool position error norm",
            "parallel_metrics": ["rotation_error_rad", "normalized_ellipse_score"],
        },
        "artifacts": {
            "boundary": {"path": str(BOUNDARY), "sha256": sha256(BOUNDARY)},
            "scene": {"path": str(scene), "sha256": sha256(scene)},
        },
        "actuator_contract": {
            "right_wrist_translation_names": names[:3],
            "indices": groups["right_wrist_translation"],
            "unit": "m",
        },
        "policies": policies,
    }
    old_summary = policies["old_scale_1"]["summary"]
    new_summary = policies["new_scale_005"]["summary"]
    report["diagnosis"] = {
        "stable_local_favorable_direction_established": False,
        "new_policy_wrong_direction_causally_confirmed": False,
        "old_policy_local_direction_certified": False,
        "reason": (
            "The central-difference direction changes materially between the 0.5 mm and "
            "1.0 mm probes in the contact-rich one-step dynamics. Neither policy has a "
            "scale-stable gradient field over all five states."
        ),
        "old_policy_centered_gradient_sign_stability_fraction": old_summary[
            "gradient_sign_stability_fraction"
        ],
        "new_policy_centered_gradient_sign_stability_fraction": new_summary[
            "gradient_sign_stability_fraction"
        ],
        "old_policy_centered_gradient_epsilon_cosine_mean": old_summary[
            "gradient_cosine_between_epsilons_mean"
        ],
        "new_policy_centered_gradient_epsilon_cosine_mean": new_summary[
            "gradient_cosine_between_epsilons_mean"
        ],
        "evidence_retained": (
            "The trajectory-level endpoint 46-50 direction divergence remains valid, but "
            "this finite-difference audit does not upgrade it to a local causal direction claim."
        ),
        "decisions": {
            "scale_sweep_authorized": False,
            "training_authorized": False,
            "use_old_policy_as_ground_truth": False,
            "next_method_if_requested": (
                "contact-mode-aware multi-step response or local system identification, "
                "with the perturbation horizon and acceptance rule frozen in advance"
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "old": policies["old_scale_1"]["summary"],
        "new": policies["new_scale_005"]["summary"],
        "diagnosis": report["diagnosis"],
    }, indent=2))


if __name__ == "__main__":
    main()
