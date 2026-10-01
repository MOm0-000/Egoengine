#!/usr/bin/env python3
"""Read-only source-45 state-entry gate for the frozen Pour actor."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
import numpy as np
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
    _require_exact_trace,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
    zero_hidden,
)
from audit_taco_pour_single_pass_endpoint47_49 import (  # noqa: E402
    step_map,
    successful_intervals,
)
from audit_taco_pour_source46_state_entry import (  # noqa: E402
    outcome_state,
    physical_state,
    state_distance,
)


SOURCE = 45
RECORDED_ENDPOINTS = (45, 46, 47, 48, 49, 50)
EXPECTED_BRANCHES = {
    "zero_R_forearm_ty_residual_only": (1,),
    "zero_right_wrist_translation": tuple(range(0, 3)),
    "zero_complete_right_wrist": tuple(range(0, 6)),
    "zero_entire_right_hand_residual": tuple(range(0, 18)),
    "zero_full_36d_residual": tuple(range(0, 36)),
}


def all_right_tool_contact_state(env) -> dict:
    """Return every live right-hand/tool contact, including unmapped palm geoms."""
    import warp as wp

    finger_names = ("thumb", "index", "middle", "ring", "pinky")
    model = env.env.model_cpu
    finger_map = np.asarray(env.ego_cfg.force_closure_geom_finger_map, np.int64)
    hand_map = np.asarray(env.ego_cfg.force_closure_geom_hand_group_map, np.int64)
    object_map = np.asarray(env.ego_cfg.force_closure_geom_object_group_map, np.int64)
    contact = env.env.data_wp.contact
    geom = wp.to_torch(contact.geom).detach().cpu().numpy().astype(np.int64)
    distance = wp.to_torch(contact.dist).detach().cpu().numpy().astype(np.float64)
    address = wp.to_torch(contact.efc_address).detach().cpu().numpy().astype(np.int64)
    frame = wp.to_torch(contact.frame).detach().cpu().numpy().astype(np.float64)
    forces = wp.to_torch(env.env.data_wp.efc.force)[0].detach().cpu().numpy()
    count = int(wp.to_torch(env.env.data_wp.nacon)[0].item())
    live = []
    for index in range(count):
        first, second = map(int, geom[index])
        if hand_map[first] == 0 and object_map[second] == 0:
            hand, tool = first, second
            normal = frame[index, 0]
        elif hand_map[second] == 0 and object_map[first] == 0:
            hand, tool = second, first
            normal = -frame[index, 0]
        else:
            continue
        valid = address[index][
            (address[index] >= 0) & (address[index] < len(forces))
        ]
        normal_force = float(np.maximum(forces[valid], 0.0).sum())
        if distance[index] > 0.0 or normal_force <= 0.0:
            continue
        finger = int(finger_map[hand])
        live.append({
            "finger_role": finger_names[finger] if 0 <= finger < 5 else None,
            "hand_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, hand
            ),
            "tool_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, tool
            ),
            "contact_distance_m": float(distance[index]),
            "normal_force": normal_force,
            "hand_to_tool_contact_normal_world": normal.tolist(),
        })
    return {
        "has_any_live_right_hand_tool_contact": bool(live),
        "active_finger_roles": sorted({
            row["finger_role"] for row in live if row["finger_role"] is not None
        }),
        "unmapped_right_hand_contact_geoms": sorted({
            row["hand_geom"] for row in live if row["finger_role"] is None
        }),
        "live_contacts": live,
        "semantics": (
            "all compiled contacts between hand_group=right and object_group=tool "
            "with nonpositive distance and positive normal force; finger_role is "
            "null for palm or other right-hand geoms outside the finger map"
        ),
    }


def state_record(env) -> dict:
    state = physical_state(env)
    finger_mapped = state["contact"]
    state["contact"] = all_right_tool_contact_state(env)
    state["contact"]["finger_mapped_force_closure_view"] = finger_mapped
    return state


def has_live_right_tool_contact(state: dict) -> bool:
    return bool(state["contact"]["has_any_live_right_hand_tool_contact"])


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_source45_state_entry_gate_v1":
        raise ValueError("unsupported source-45 state-entry contract")
    if contract.get("status") != "authorized_read_only_gate":
        raise ValueError("source-45 state-entry gate is not authorized")
    required = {
        "backend": "CPU_MuJoCo_Warp",
        "training_allowed": False,
        "optimizer_steps": 0,
        "actor_frozen": True,
        "critic_frozen": True,
        "observation_normalization_frozen": True,
        "reward_changed": False,
        "objective_changed": False,
        "action_scale_changed": False,
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
        raise ValueError("source-45 runtime is not fail-closed")
    actual = {
        row["name"]: tuple(row["zero_normalized_action_indices"])
        for row in contract["branches"]
    }
    if actual != EXPECTED_BRANCHES:
        raise ValueError("source-45 branch set changed")
    if (
        contract["source"]["endpoint"] != SOURCE
        or contract["intervention_control_steps"] != 1
        or contract["continuation"] != "frozen_deterministic_PPO"
        or contract["rollout_end_endpoint"] != 60
        or contract["individual_actuator_sweep_allowed"] is not False
        or contract["scale_sweep_allowed"] is not False
        or contract["multi_step_intervention_allowed"] is not False
        or contract["primary_passing_condition"]["contact_required"] is not False
        or contract["frozen_half_LR_candidate"]["remains_blocked"] is not True
    ):
        raise ValueError("source-45 intervention definition changed")
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


def validate_source46_interpretation(report: dict) -> dict:
    by_name = {row["name"]: row for row in report["branches"]}
    no_contact_but_survives_49 = []
    contact_but_fails_49 = []
    for name in (
        "zero_R_forearm_ty_residual_only",
        "zero_right_wrist_translation",
        "zero_complete_right_wrist",
    ):
        row = by_name[name]
        if (
            row["endpoint_48_has_live_right_hand_tool_contact"]
            or not row["endpoint_49_survives"]
            or row["first_failure_endpoint_or_60"] != 50
        ):
            raise ValueError("source-46 tracking-rescue evidence changed")
        no_contact_but_survives_49.append(name)
    for name in (
        "zero_entire_right_hand_residual",
        "zero_full_36d_residual",
    ):
        row = by_name[name]
        if (
            not row["endpoint_48_has_live_right_hand_tool_contact"]
            or row["endpoint_49_survives"]
            or row["first_failure_endpoint_or_60"] != 49
        ):
            raise ValueError("source-46 contact-with-failure evidence changed")
        contact_but_fails_49.append(name)
    return {
        "source46_is_still_tracking_controllable": True,
        "source46_best_tracking_rescue_survives_endpoint49_and_fails_endpoint50": True,
        "endpoint48_contact_is_necessary_for_short_horizon_tracking_feasibility": False,
        "endpoint48_contact_is_sufficient_for_short_horizon_tracking_feasibility": False,
        "no_contact_but_survives_endpoint49": no_contact_but_survives_49,
        "contact_but_fails_endpoint49": contact_but_fails_49,
        "corrected_interpretation": (
            "Source 46 can still rescue endpoint-49 object tracking; it failed "
            "the earlier joint contact-plus-feasibility gate, not all task control."
        ),
    }


def actor_record(result: dict, action: np.ndarray) -> dict:
    return {
        "actor_mu": result["mus"][0].detach().cpu().numpy().astype(np.float64).tolist(),
        "actor_sigma": result["sigmas"][0].detach().cpu().numpy().astype(np.float64).tolist(),
        "state_feasible_low": result["action_lows"][0].detach().cpu().numpy().astype(np.float64).tolist(),
        "state_feasible_high": result["action_highs"][0].detach().cpu().numpy().astype(np.float64).tolist(),
        "deterministic_action": np.asarray(action, np.float64).tolist(),
    }


def actor_delta(counterfactual: dict, formal: dict) -> dict:
    mu_delta = np.asarray(counterfactual["actor_mu"]) - np.asarray(formal["actor_mu"])
    action_delta = (
        np.asarray(counterfactual["deterministic_action"])
        - np.asarray(formal["deterministic_action"])
    )
    groups = {
        "R_forearm_ty": (1,),
        "right_wrist_translation": tuple(range(0, 3)),
        "right_fingers": tuple(range(6, 18)),
    }
    grouped = {}
    for name, indices in groups.items():
        selected = np.asarray(indices, np.int64)
        grouped[name] = {
            "indices": list(indices),
            "formal_mu": np.asarray(formal["actor_mu"])[selected].tolist(),
            "counterfactual_mu": np.asarray(counterfactual["actor_mu"])[selected].tolist(),
            "mu_delta": mu_delta[selected].tolist(),
            "mu_delta_l2": float(np.linalg.norm(mu_delta[selected])),
            "formal_deterministic_action": np.asarray(
                formal["deterministic_action"]
            )[selected].tolist(),
            "counterfactual_deterministic_action": np.asarray(
                counterfactual["deterministic_action"]
            )[selected].tolist(),
            "deterministic_action_delta": action_delta[selected].tolist(),
            "deterministic_action_delta_l2": float(
                np.linalg.norm(action_delta[selected])
            ),
        }
    return {
        "full_36d_mu_delta": mu_delta.tolist(),
        "full_36d_mu_delta_l2": float(np.linalg.norm(mu_delta)),
        "full_36d_deterministic_action_delta": action_delta.tolist(),
        "full_36d_deterministic_action_delta_l2": float(
            np.linalg.norm(action_delta)
        ),
        "highlighted_groups": grouped,
    }


def run_replay_with_states(*, backend, boundary) -> tuple[dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    backend.begin_trial("Replay", 20, 60)
    states = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if source in RECORDED_ENDPOINTS:
            states[source] = state_record(backend.env)
        feasible = backend.step(np.zeros((1, 36), np.float32), source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint in RECORDED_ENDPOINTS:
            states[endpoint] = state_record(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    return backend.validation_traces[-1], states


def run_ppo_with_states(
    *, backend, policy, boundary,
) -> tuple[dict, dict[int, dict], dict, dict[int, dict]]:
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    policy.rnn_states = zero_hidden(policy)
    backend.begin_trial("PPO", 20, 60)
    states = {}
    capture = None
    actors = {}
    attempted = 0
    feasible = True
    for source in range(20, 60):
        if source in RECORDED_ENDPOINTS:
            states[source] = state_record(backend.env)
        if source == SOURCE:
            capture = {
                "environment_state": backend.snapshot(),
                "pre_forward_hidden": clone_hidden(policy.rnn_states),
            }
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if source in (SOURCE, 46):
            actors[source] = actor_record(result, action[0])
        feasible = backend.step(action, source)
        attempted += 1
        endpoint = int(backend.env.time_indices[0])
        if endpoint in RECORDED_ENDPOINTS:
            states[endpoint] = state_record(backend.env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    if capture is None or 46 not in actors:
        raise RuntimeError("formal PPO did not capture source 45/46")
    return backend.validation_traces[-1], states, capture, actors


def run_branch(
    *, env, backend, policy, capture: dict, name: str,
    zero_indices: tuple[int, ...],
) -> tuple[dict, dict[int, dict], dict, dict[str, np.ndarray]]:
    backend.restore(capture["environment_state"])
    backend.verify_restored_snapshot(capture["environment_state"])
    policy.rnn_states = clone_hidden(capture["pre_forward_hidden"])
    states = {SOURCE: state_record(env)}
    backend.begin_trial(name, SOURCE, 60)
    source46_actor = None
    original_action = None
    executed_action = None
    attempted = 0
    feasible = True
    for current in range(SOURCE, 60):
        packed = policy.obs_to_tensors(backend.observation())
        result = policy.get_deterministic_action_values(packed)
        policy.rnn_states = result["rnn_states"]
        action = policy.preprocess_actions(result["deterministic_actions"])
        if current == SOURCE:
            original_action = action[0].copy()
            low = result["action_lows"][0].detach().cpu().numpy()
            high = result["action_highs"][0].detach().cpu().numpy()
            indices = np.asarray(zero_indices, np.int64)
            if np.any(low[indices] > 1e-7) or np.any(high[indices] < -1e-7):
                raise RuntimeError(f"zero leaves state-feasible support for {name}")
            action[0, indices] = 0.0
            executed_action = action[0].copy()
        if current == 46:
            source46_actor = actor_record(result, action[0])
        feasible = backend.step(action, current)
        attempted += 1
        endpoint = int(env.time_indices[0])
        if endpoint in RECORDED_ENDPOINTS:
            states[endpoint] = state_record(env)
        if not feasible:
            break
    backend.end_trial(feasible, attempted)
    if source46_actor is None:
        raise RuntimeError(f"{name} did not reach source 46")
    trace = backend.validation_traces[-1]
    arrays = {
        f"{name}_endpoint": np.asarray(
            [row["endpoint"] for row in trace["steps"]], np.int64
        ),
        f"{name}_score": np.asarray(
            [row["objective_score"][0] for row in trace["steps"]], np.float64
        ),
        f"{name}_qpos": np.asarray(
            [row["endpoint_qpos"] for row in trace["steps"]], np.float32
        ),
        f"{name}_qvel": np.asarray(
            [row["endpoint_qvel"] for row in trace["steps"]], np.float32
        ),
        f"{name}_normalized_action": np.asarray(
            [row["raw_residual_action"] for row in trace["steps"]], np.float32
        ),
    }
    return trace, states, {
        "original_action": original_action,
        "executed_action": executed_action,
        "source46_actor": source46_actor,
    }, arrays


def summarize_branch(
    *, name: str, zero_indices: tuple[int, ...], trace: dict,
    states: dict[int, dict], detail: dict, formal_actors: dict[int, dict],
    formal_states: dict[str, dict[int, dict]], reference_qpos: np.ndarray,
    actuator_names: tuple[str, ...],
) -> dict:
    rows = step_map(trace)
    outcomes = {
        str(endpoint): outcome_state(states[endpoint], rows[endpoint], reference_qpos)
        for endpoint in RECORDED_ENDPOINTS[1:]
        if endpoint in rows and endpoint in states
    }
    endpoint_49 = outcomes.get("49")
    endpoint_50 = outcomes.get("50")
    survives_49 = endpoint_49 is not None and not endpoint_49["tracking"]["terminated"]
    survives_50 = endpoint_50 is not None and not endpoint_50["tracking"]["terminated"]
    failure = trace["first_failure"]
    original = np.asarray(detail["original_action"])
    executed = np.asarray(detail["executed_action"])
    contacts = {
        endpoint: (
            has_live_right_tool_contact(state),
            state["contact"]["active_finger_roles"],
            state["contact"]["unmapped_right_hand_contact_geoms"],
        )
        for endpoint, state in outcomes.items()
    }
    return {
        "name": name,
        "source_endpoint": SOURCE,
        "zero_normalized_action_indices": list(zero_indices),
        "zeroed_actuators": [actuator_names[index] for index in zero_indices],
        "formal_normalized_action": original.tolist(),
        "executed_normalized_action": executed.tolist(),
        "untargeted_action_dimensions_bitwise_unchanged": bool(np.array_equal(
            np.delete(original, zero_indices), np.delete(executed, zero_indices)
        )),
        "source_45_state": states[SOURCE],
        "outcome_states": outcomes,
        "contact_secondary_diagnostic": {
            endpoint: {
                "has_live_right_hand_tool_contact": active,
                "active_fingers": fingers,
                "unmapped_right_hand_contact_geoms": unmapped,
            }
            for endpoint, (active, fingers, unmapped) in contacts.items()
        },
        "source46_actor": detail["source46_actor"],
        "source46_actor_delta_vs_formal": actor_delta(
            detail["source46_actor"], formal_actors[46]
        ),
        "endpoint_46_distance_to_formal_Replay": state_distance(
            states[46], formal_states["Replay"][46]
        ),
        "endpoint_46_distance_to_formal_PPO": state_distance(
            states[46], formal_states["PPO"][46]
        ),
        "endpoint_49_survives": survives_49,
        "endpoint_50_survives": survives_50,
        "endpoint_50_score": (
            None if endpoint_50 is None else endpoint_50["tracking"]["objective_score"]
        ),
        "successful_intervals_from_source": successful_intervals(trace),
        "first_failure_endpoint_or_60": (
            60 if failure is None else int(failure["endpoint"])
        ),
        "passes_predeclared_primary_gate": bool(survives_49 and survives_50),
        "contact_used_as_acceptance_condition": False,
        "formal_acceptance_or_chunk_commit_allowed": False,
    }


def decide(branches: list[dict], contract: dict) -> dict:
    passed = [row["name"] for row in branches if row["passes_predeclared_primary_gate"]]
    if passed:
        selected = next(
            name for name in contract["candidate_precedence_if_multiple_pass"]
            if name in passed
        )
        direction = contract["candidate_direction_if_passed"][selected]
    else:
        selected = None
        direction = contract["decision_if_all_fail"]
    return {
        "passing_branches": passed,
        "gate_passed": bool(passed),
        "narrowest_passing_branch": selected,
        "next_read_only_or_candidate_direction": direction,
        "new_training_authorized": False,
        "actor_LR_5e_minus_5_unblocked": False,
        "reward_change_authorized": False,
        "chunk_acceptance_or_commit_authorized": False,
    }


def write_summary(path: Path, report: dict) -> None:
    lines = ["# Source-45 state-entry gate", ""]
    for row in report["branches"]:
        contact = row["contact_secondary_diagnostic"].get("50", {}).get(
            "has_live_right_hand_tool_contact"
        )
        lines.append(
            f"- {row['name']}: survive50={row['endpoint_50_survives']}, "
            f"score@50={row['endpoint_50_score']}, contact@50={contact}, "
            f"failure={row['first_failure_endpoint_or_60']}"
        )
    lines.extend([
        "",
        f"- gate passed: {report['decision']['gate_passed']}",
        "- next direction: "
        f"{report['decision']['next_read_only_or_candidate_direction']}",
        "",
        "Contact was diagnostic only. No training, task acceptance or chunk commit occurred.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_source45_state_entry_gate_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_source45_state_entry_gate_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    source46_correction = validate_source46_interpretation(
        json.loads(paths["source46_gate_report"].read_text())
    )

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
    objective = load_runtime_objective(
        paths["protocol"], paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        paths["protocol"], paths["observation_profile"], require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(paths["action_profile"])
    distribution_spec, _ = load_truncated_gaussian_profile(
        paths["distribution_profile"]
    )
    _, initialization = load_accepted_initialization(
        paths["initialization_report"], paths["simulator_config"]
    )
    config = _load_ego_config(str(paths["simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
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
    actuator_names, _, _ = _actuator_metadata(
        env.env.model_cpu, tuple(env.env_cfg.residual.hand_control_indices)
    )
    if actuator_names[1] != "R_forearm_ty_position":
        raise RuntimeError("action index 1 is not R_forearm_ty_position")
    backend = MJWPChunkBackend(env)
    boundary = load_gzip_torch(paths["boundary"])
    replay, replay_states = run_replay_with_states(backend=backend, boundary=boundary)
    _require_exact_trace(replay, formal["replay_validation"], "single-pass Replay")

    checkpoint = load_gzip_torch(paths["checkpoint"])
    temporary = tempfile.TemporaryDirectory(prefix=".source45_entry_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temporary.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=distribution_spec,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        ppo, ppo_states, capture, formal_actors = run_ppo_with_states(
            backend=backend, policy=policy, boundary=boundary
        )
        _require_exact_trace(ppo, formal["ppo_validation"], "single-pass PPO")

        baseline, baseline_states, baseline_detail, _ = run_branch(
            env=env, backend=backend, policy=policy, capture=capture,
            name="formal_source45_suffix", zero_indices=(),
        )
        expected_suffix = [
            row for row in ppo["steps"] if int(row["endpoint"]) > SOURCE
        ]
        if baseline["steps"] != expected_suffix:
            raise RuntimeError("restored formal source-45 suffix is not exact")
        for endpoint in (46, 47, 48, 49):
            if baseline_states[endpoint] != ppo_states[endpoint]:
                raise RuntimeError(
                    f"restored source-45 physical record differs at endpoint {endpoint}"
                )
        if baseline_detail["source46_actor"] != formal_actors[46]:
            raise RuntimeError("restored formal source-46 actor decision is not exact")

        branches = []
        arrays = {}
        formal_states = {"Replay": replay_states, "PPO": ppo_states}
        for name, indices in EXPECTED_BRANCHES.items():
            trace, states, detail, branch_arrays = run_branch(
                env=env, backend=backend, policy=policy, capture=capture,
                name=name, zero_indices=indices,
            )
            branches.append(summarize_branch(
                name=name,
                zero_indices=indices,
                trace=trace,
                states=states,
                detail=detail,
                formal_actors=formal_actors,
                formal_states=formal_states,
                reference_qpos=reference_qpos,
                actuator_names=actuator_names,
            ))
            arrays.update(branch_arrays)
        policy.writer.close()
    finally:
        temporary.cleanup()

    replay_rows = step_map(replay)
    ppo_rows = step_map(ppo)
    formal_state_report = {}
    for mode, states, rows in (
        ("Replay", replay_states, replay_rows),
        ("PPO", ppo_states, ppo_rows),
    ):
        formal_state_report[mode] = {
            str(endpoint): outcome_state(states[endpoint], rows[endpoint], reference_qpos)
            for endpoint in RECORDED_ENDPOINTS
            if endpoint in states and endpoint in rows
        }
    gate_decision = decide(branches, contract)

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    arrays_path = args.output / contract["artifacts"]["branch_arrays"]
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "schema": contract["schema"],
        "status": "completed_read_only_gate",
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "source46_interpretation_correction": source46_correction,
        "regression_gate": {
            "formal_Replay_trace_bitwise_reproduced": True,
            "formal_PPO_trace_bitwise_reproduced": True,
            "restored_source45_suffix_bitwise_reproduced": True,
            "restored_source45_physical_records_exact": True,
            "restored_formal_source46_actor_decision_exact": True,
            "formal_Replay_successful_intervals": successful_intervals(replay),
            "formal_PPO_successful_intervals": successful_intervals(ppo),
        },
        "actuator_contract": {
            "names": list(actuator_names),
            "R_forearm_ty_index": 1,
            "formal_source45_normalized_action": formal_actors[45][
                "deterministic_action"
            ],
            "formal_source46_actor": formal_actors[46],
        },
        "formal_state_entry": formal_state_report,
        "branches": branches,
        "branch_arrays": {
            "path": str(arrays_path.resolve()),
            "sha256": sha256(arrays_path),
        },
        "measurement_boundaries": {
            "contact_is_secondary_and_does_not_affect_gate": True,
            "positive_mesh_surface_gap_computed": False,
            "mj_geomDistance_used": False,
            "position_and_rotation_contributions_reported_separately": True,
            "endpoint46_state_distance_is_not_a_success_criterion": True,
            "mixed_unit_combined_distance_reported": False,
        },
        "decision": gate_decision,
    }
    report_path = args.output / contract["artifacts"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output / contract["artifacts"]["summary"], report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "passing_branches": gate_decision["passing_branches"],
        "next_direction": gate_decision[
            "next_read_only_or_candidate_direction"
        ],
        "branches": [
            {
                "name": row["name"],
                "survive50": row["endpoint_50_survives"],
                "score50": row["endpoint_50_score"],
                "failure": row["first_failure_endpoint_or_60"],
                "contact_sequence": {
                    endpoint: value["active_fingers"]
                    for endpoint, value in row[
                        "contact_secondary_diagnostic"
                    ].items()
                },
                "source46_mu_ty_delta": row[
                    "source46_actor_delta_vs_formal"
                ]["highlighted_groups"]["R_forearm_ty"]["mu_delta"][0],
            }
            for row in branches
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
