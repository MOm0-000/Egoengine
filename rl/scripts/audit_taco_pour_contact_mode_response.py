#!/usr/bin/env python3
"""Frozen three-interval wrist probes, grouped by exact substep contact mode."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_pour_action_attribution import actuator_groups, sha256
from video_to_spider.rl.replay_rl import _snapshot_value_equal


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
        "trace": ROOT / (
            "runs/taco_pour_normalized_ellipse_repeatability_cpu_v1/"
            "checkpoint_8ep_closed_loop_v2.json.gz"
        ),
    },
    "new_scale_005": {
        "action_profile": ROOT / "configs/taco_pour_residual_action_scaled_v1.yaml",
        "trace": ROOT / (
            "runs/taco_pour_normalized_action_scale_v1/"
            "cpu_closed_loop_validation.json.gz"
        ),
    },
}
PAIRED_SOURCES = (45, 46, 47)
OLD_ONLY_SOURCES = (48, 49)
EPSILONS_M = (0.0005, 0.001)
AXES = ("x", "y", "z")
OUTPUT_DIR = ROOT / "runs/endpoint45_49_contact_mode_response_v1"
FULL_REPORT = OUTPUT_DIR / "report.json.gz"
SUMMARY = OUTPUT_DIR / "summary.json"


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def frozen_rows(path: Path) -> dict[int, dict]:
    report = load_json(path)
    rows = report["repetitions"][0]["trials"]["rl"]["trace"]["steps"]
    result = {int(row["control_interval"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate control interval in {path}")
    return result


def outcome(info: dict, endpoint: int) -> dict:
    return {
        "endpoint": endpoint,
        "position_error_m": float(info["object_position_error"][0, 0]),
        "rotation_error_rad": float(info["object_rotation_error"][0, 0]),
        "objective_score": float(info["object_tracking_error_per_object"][0, 0]),
        "terminated": bool(info["object_terminated"][0, 0]),
    }


def geom_metadata(env, geom_id: int) -> dict:
    model = env.env.model_cpu
    name = model.geom(geom_id).name or f"geom_{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name or f"body_{body_id}"
    object_map = np.asarray(env.ego_cfg.force_closure_geom_object_group_map)
    object_index = int(object_map[geom_id]) if geom_id < len(object_map) else -1
    role = env.object_roles[object_index] if 0 <= object_index < len(env.object_roles) else None
    return {
        "id": geom_id,
        "name": name,
        "body": body_name,
        "object_role": role,
    }


def capture_contacts(env, control_offset: int, physics_substep: int) -> dict:
    data = env.env.data_wp
    packed_count = int(data.nacon.numpy()[0])
    geom = wp.to_torch(data.contact.geom).cpu().numpy()
    dist = wp.to_torch(data.contact.dist).cpu().numpy()
    world = wp.to_torch(data.contact.worldid).cpu().numpy()
    live = np.flatnonzero(world == 0)[:packed_count]
    touching = [int(index) for index in live if float(dist[index]) <= 0.0]
    counts = Counter(
        tuple(sorted((int(geom[index, 0]), int(geom[index, 1]))))
        for index in touching
    )
    pairs = []
    for pair, multiplicity in sorted(counts.items()):
        sides = [geom_metadata(env, geom_id) for geom_id in pair]
        pairs.append({
            "geom_ids": list(pair),
            "geom_names": [side["name"] for side in sides],
            "body_names": [side["body"] for side in sides],
            "object_roles": [side["object_role"] for side in sides],
            "multiplicity": multiplicity,
        })
    return {
        "control_offset": control_offset,
        "physics_substep": physics_substep,
        "global_substep": control_offset * int(env.ego_cfg.ctrl_steps) + physics_substep,
        "packed_contact_count": packed_count,
        "touching_contact_count": len(touching),
        "touching_pairs": pairs,
    }


def mode_signature(run: dict) -> tuple:
    return tuple(
        tuple(
            (tuple(pair["geom_ids"]), int(pair["multiplicity"]))
            for pair in row["touching_pairs"]
        )
        for row in run["physics_substeps"]
    )


def signature_sha256(signature: tuple) -> str:
    return hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode()
    ).hexdigest()


def differing_substeps(reference: tuple, candidate: tuple) -> list[int]:
    if len(reference) != len(candidate):
        raise ValueError("contact sequences have different horizons")
    return [index for index, pair in enumerate(zip(reference, candidate, strict=True)) if pair[0] != pair[1]]


def contact_change(reference: tuple, candidate: tuple) -> dict:
    changed = differing_substeps(reference, candidate)
    pair_set_changed = []
    multiplicity_only = []
    for index in changed:
        reference_pairs = {item[0] for item in reference[index]}
        candidate_pairs = {item[0] for item in candidate[index]}
        if reference_pairs != candidate_pairs:
            pair_set_changed.append(index)
        else:
            multiplicity_only.append(index)
    return {
        "changed": bool(changed),
        "differing_substep_count": len(changed),
        "first_differing_global_substep": changed[0] if changed else None,
        "pair_set_changed_substep_count": len(pair_set_changed),
        "multiplicity_only_changed_substep_count": len(multiplicity_only),
    }


def run_sequence(env, state: dict, actions: list[np.ndarray]) -> dict:
    env.set_env_state(state)
    substeps = []
    outcomes = []
    applied = []
    for offset, action in enumerate(actions):
        expected_source = int(state["time_indices"][0]) + offset

        def observe(substep: int, *, control_offset: int = offset) -> None:
            substeps.append(capture_contacts(env, control_offset, substep))

        _, _, _, info = env.step(
            np.asarray(action, np.float32)[None],
            auto_reset=False,
            substep_observer=observe,
        )
        if int(env.time_indices[0]) != expected_source + 1:
            raise ValueError("three-step probe cursor changed unexpectedly")
        reference_ctrl = env._reference_ctrls(env.time_indices, offset=0)[0]
        actual = (env._last_ctrl[0] - reference_ctrl).detach().cpu().numpy()
        applied.append(actual.tolist())
        outcomes.append(outcome(info, expected_source + 1))
    if len(substeps) != 3 * int(env.ego_cfg.ctrl_steps):
        raise ValueError("substep observer did not record the fixed three-interval horizon")
    result = {
        "outcomes": outcomes,
        "applied_residuals": applied,
        "physics_substeps": substeps,
    }
    result["contact_mode_sha256"] = signature_sha256(mode_signature(result))
    return result


def snapshot_mismatches(left: dict, right: dict) -> list[str]:
    if left.keys() != right.keys():
        return ["<snapshot keys differ>"]
    return [key for key in left if not _snapshot_value_equal(left[key], right[key])]


def observer_transparency(env, state: dict, action: np.ndarray) -> dict:
    env.set_env_state(state)
    env.step(action[None], auto_reset=False)
    without = env.get_env_state()
    env.set_env_state(state)
    captured = []
    env.step(
        action[None],
        auto_reset=False,
        substep_observer=lambda substep: captured.append(capture_contacts(env, 0, substep)),
    )
    with_observer = env.get_env_state()
    mismatches = snapshot_mismatches(without, with_observer)
    return {
        "observer_is_bitwise_transparent": not mismatches,
        "saved_state_field_count": len(without),
        "mismatching_fields": mismatches,
        "captured_physics_substeps": len(captured),
    }


def derivative(runs: dict, epsilon: float, scheme: str, horizon: int) -> dict:
    metric_names = ("position_error_m", "rotation_error_rad", "objective_score")
    base = runs["baseline"]["outcomes"][horizon - 1]
    if scheme == "central":
        plus = runs["plus"]["outcomes"][horizon - 1]
        minus = runs["minus"]["outcomes"][horizon - 1]
        return {name: (plus[name] - minus[name]) / (2.0 * epsilon) for name in metric_names}
    if scheme == "one_sided_plus":
        plus = runs["plus"]["outcomes"][horizon - 1]
        return {name: (plus[name] - base[name]) / epsilon for name in metric_names}
    if scheme == "one_sided_minus":
        minus = runs["minus"]["outcomes"][horizon - 1]
        return {name: (base[name] - minus[name]) / epsilon for name in metric_names}
    raise ValueError(f"unknown difference scheme: {scheme}")


def analyze_probe(baseline: dict, variants: dict, epsilon: float) -> dict:
    base_signature = mode_signature(baseline)
    signatures = {name: mode_signature(run) for name, run in variants.items()}
    changes = {
        name: contact_change(base_signature, signature)
        for name, signature in signatures.items()
    }
    feasible = set(variants)
    if feasible == {"plus", "minus"}:
        scheme = "central"
    elif feasible == {"plus"}:
        scheme = "one_sided_plus"
    elif feasible == {"minus"}:
        scheme = "one_sided_minus"
    else:
        scheme = None
    stable = scheme is not None and all(not change["changed"] for change in changes.values())
    result = {
        "difference_scheme": scheme,
        "contact_stable_for_local_estimate": stable,
        "contact_mode_changes_vs_baseline": changes,
        "derivative": None,
        "position_derivative_sign_same_at_1_and_3_steps": None,
    }
    if stable:
        runs = {"baseline": baseline, **variants}
        first = derivative(runs, epsilon, scheme, 1)
        third = derivative(runs, epsilon, scheme, 3)
        result["derivative"] = {"after_1_step": first, "after_3_steps": third}
        result["position_derivative_sign_same_at_1_and_3_steps"] = bool(
            np.sign(first["position_error_m"]) == np.sign(third["position_error_m"])
        )
    return result


def build_environment(action_profile: Path):
    from run_mjwp_ppo import MJWPVectorEnv, MJWPVectorEnvConfig, _load_ego_config, _load_reference, torch
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation

    objective = load_runtime_objective(
        PROTOCOL, OBJECTIVE, tracking_variant="tool_only", require_run_ready=True
    )
    observation = load_runtime_observation(PROTOCOL, OBSERVATION, require_run_ready=True)
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
    boundary = torch.load(
        io.BytesIO(gzip.decompress(BOUNDARY.read_bytes())),
        map_location="cpu",
        weights_only=False,
    )
    return env, boundary, residual_report


def source_states(env, boundary: dict, rows: dict[int, dict], sources: tuple[int, ...]) -> tuple[dict, dict]:
    env.set_env_state(boundary)
    states = {}
    exact_rows = []
    for control in range(20, max(sources) + 1):
        if control in sources:
            states[control] = env.get_env_state()
        if control == max(sources):
            break
        row = rows[control]
        env.step(np.asarray(row["raw_residual_action"], np.float32)[None], auto_reset=False)
        qpos = env._mjwp.get_qpos(env.ego_cfg, env.env)[0].detach().cpu().numpy()
        qvel = env._mjwp.get_qvel(env.ego_cfg, env.env)[0].detach().cpu().numpy()
        exact = (
            np.array_equal(qpos, np.asarray(row["endpoint_qpos"], qpos.dtype))
            and np.array_equal(qvel, np.asarray(row["endpoint_qvel"], qvel.dtype))
        )
        exact_rows.append({"control_interval": control, "endpoint": control + 1, "bitwise_equal": exact})
        if not exact:
            raise ValueError(f"frozen trajectory did not reproduce at endpoint {control + 1}")
    return states, {
        "checked_endpoints": [row["endpoint"] for row in exact_rows],
        "all_bitwise_equal": all(row["bitwise_equal"] for row in exact_rows),
    }


def probe_source(env, state: dict, rows: dict[int, dict], source: int) -> dict:
    actions = [
        np.asarray(rows[control]["raw_residual_action"], np.float32)
        for control in range(source, source + 3)
    ]
    baseline = run_sequence(env, state, actions)
    repeated_baseline = run_sequence(env, state, actions)
    repeatable = (
        mode_signature(baseline) == mode_signature(repeated_baseline)
        and baseline["outcomes"] == repeated_baseline["outcomes"]
        and baseline["applied_residuals"] == repeated_baseline["applied_residuals"]
    )
    if not repeatable:
        raise ValueError(f"baseline three-step response is not repeatable at source {source}")
    stored_applied = np.asarray(rows[source]["applied_residual"], np.float64)
    observed_applied = np.asarray(baseline["applied_residuals"][0], np.float64)
    if not np.allclose(stored_applied, observed_applied, rtol=0.0, atol=2e-7):
        raise ValueError(f"baseline applied action differs at source {source}")
    scale = float(env.env_cfg.residual.residual_scale)
    clip = float(env.env_cfg.residual.residual_clip)
    probes = {}
    for axis, axis_name in enumerate(AXES):
        probes[axis_name] = {}
        for epsilon in EPSILONS_M:
            key = f"{epsilon:.4f}"
            variants = {}
            feasibility = {}
            for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                desired = float(observed_applied[axis] + sign * epsilon)
                feasible = -clip <= desired <= clip
                feasibility[sign_name] = {
                    "feasible_under_formal_clip": feasible,
                    "desired_applied_residual_m": desired,
                }
                if not feasible:
                    continue
                perturbed = [action.copy() for action in actions]
                perturbed[0][axis] = np.float32(desired / scale)
                variants[sign_name] = run_sequence(env, state, perturbed)
            probes[axis_name][key] = {
                "feasibility": feasibility,
                "variants": variants,
                "analysis": analyze_probe(baseline, variants, epsilon),
            }
    return {
        "source_endpoint": source,
        "outcome_endpoints": [source + 1, source + 2, source + 3],
        "frozen_control_intervals": [source, source + 1, source + 2],
        "baseline_first_action_right_wrist_translation_m": observed_applied[:3].tolist(),
        "baseline_repeatability": {
            "bitwise_numeric_and_exact_contact_sequence": repeatable,
            "first_contact_mode_sha256": baseline["contact_mode_sha256"],
            "second_contact_mode_sha256": repeated_baseline["contact_mode_sha256"],
        },
        "baseline": baseline,
        "probes": probes,
    }


def response_summary(records: list[dict]) -> dict:
    eligible = []
    by_key = {}
    schemes = Counter()
    contact_changes = []
    changed_pairs = Counter()
    for record in records:
        for axis in AXES:
            for epsilon in EPSILONS_M:
                key = f"{epsilon:.4f}"
                analysis = record["probes"][axis][key]["analysis"]
                schemes[analysis["difference_scheme"]] += 1
                contact_changes.extend(analysis["contact_mode_changes_vs_baseline"].values())
                baseline_rows = record["baseline"]["physics_substeps"]
                for variant in record["probes"][axis][key]["variants"].values():
                    for baseline_row, variant_row in zip(
                        baseline_rows, variant["physics_substeps"], strict=True
                    ):
                        baseline_pairs = {
                            tuple(pair["geom_ids"]): pair for pair in baseline_row["touching_pairs"]
                        }
                        variant_pairs = {
                            tuple(pair["geom_ids"]): pair for pair in variant_row["touching_pairs"]
                        }
                        for pair_id in set(baseline_pairs) ^ set(variant_pairs):
                            pair = baseline_pairs.get(pair_id) or variant_pairs[pair_id]
                            key_pair = (
                                " <-> ".join(pair["geom_names"]),
                                tuple(role for role in pair["object_roles"] if role),
                            )
                            changed_pairs[key_pair] += 1
                item_key = (record["source_endpoint"], axis, key)
                if analysis["contact_stable_for_local_estimate"]:
                    eligible.append(item_key)
                    by_key[item_key] = analysis
    common_epsilon = []
    for record in records:
        for axis in AXES:
            small = by_key.get((record["source_endpoint"], axis, "0.0005"))
            large = by_key.get((record["source_endpoint"], axis, "0.0010"))
            if small is not None and large is not None:
                common_epsilon.append((small, large))

    def agreement(horizon: str) -> dict:
        matches = [
            np.sign(pair[0]["derivative"][horizon]["position_error_m"])
            == np.sign(pair[1]["derivative"][horizon]["position_error_m"])
            for pair in common_epsilon
        ]
        return {
            "comparison_count": len(matches),
            "matching_sign_count": int(sum(matches)),
            "matching_sign_fraction": float(np.mean(matches)) if matches else None,
        }

    one = agreement("after_1_step")
    three = agreement("after_3_steps")
    cross_horizon = [
        analysis["position_derivative_sign_same_at_1_and_3_steps"]
        for analysis in by_key.values()
    ]
    return {
        "probe_count": len(records) * len(AXES) * len(EPSILONS_M),
        "contact_stable_local_estimate_count": len(eligible),
        "contact_stable_fraction": len(eligible) / max(len(records) * 6, 1),
        "difference_scheme_counts": dict(sorted(schemes.items())),
        "feasible_signed_variant_count": len(contact_changes),
        "contact_changed_variant_count": sum(change["changed"] for change in contact_changes),
        "pair_set_changed_variant_count": sum(
            change["pair_set_changed_substep_count"] > 0 for change in contact_changes
        ),
        "multiplicity_only_changed_variant_count": sum(
            change["changed"] and change["pair_set_changed_substep_count"] == 0
            for change in contact_changes
        ),
        "differing_substeps_per_changed_variant_min_median_max": (
            [
                int(min(change["differing_substep_count"] for change in contact_changes if change["changed"])),
                float(np.median([
                    change["differing_substep_count"] for change in contact_changes if change["changed"]
                ])),
                int(max(change["differing_substep_count"] for change in contact_changes if change["changed"])),
            ]
            if any(change["changed"] for change in contact_changes) else None
        ),
        "most_common_changed_geom_pairs": [
            {"geom_pair": pair, "object_roles": list(roles), "substep_occurrences": count}
            for (pair, roles), count in changed_pairs.most_common(10)
        ],
        "baseline_three_step_repeatable_at_every_source": all(
            record["baseline_repeatability"]["bitwise_numeric_and_exact_contact_sequence"]
            for record in records
        ),
        "epsilon_sign_agreement_after_1_step": one,
        "epsilon_sign_agreement_after_3_steps": three,
        "same_sign_between_1_and_3_steps": {
            "comparison_count": len(cross_horizon),
            "matching_sign_count": int(sum(cross_horizon)),
            "matching_sign_fraction": float(np.mean(cross_horizon)) if cross_horizon else None,
        },
    }


def run_policy(label: str, files: dict) -> dict:
    rows = frozen_rows(files["trace"])
    paired_required = set().union(*(range(source, source + 3) for source in PAIRED_SOURCES))
    if not paired_required.issubset(rows):
        raise ValueError(f"{label} lacks a frozen action inside paired common support")
    extension_sources = OLD_ONLY_SOURCES if label == "old_scale_1" else ()
    all_sources = PAIRED_SOURCES + extension_sources
    env, boundary, action_report = build_environment(files["action_profile"])
    states, reproduction = source_states(env, boundary, rows, all_sources)
    transparency = observer_transparency(
        env,
        states[PAIRED_SOURCES[0]],
        np.asarray(rows[PAIRED_SOURCES[0]]["raw_residual_action"], np.float32),
    )
    if not transparency["observer_is_bitwise_transparent"]:
        raise ValueError("read-only contact observer changed the simulator endpoint state")
    paired = [probe_source(env, states[source], rows, source) for source in PAIRED_SOURCES]
    extension = [probe_source(env, states[source], rows, source) for source in extension_sources]
    return {
        "frozen_trace": {"path": str(files["trace"]), "sha256": sha256(files["trace"])},
        "action_profile": action_report,
        "frozen_trajectory_reproduction": reproduction,
        "substep_observer_audit": transparency,
        "paired_common_support": paired,
        "paired_summary": response_summary(paired),
        "old_policy_only_extension": extension,
        "old_policy_only_extension_summary": response_summary(extension) if extension else None,
        "new_policy_missing_followup_actions": list(OLD_ONLY_SOURCES) if label == "new_scale_005" else [],
    }


def main() -> None:
    if FULL_REPORT.exists() or SUMMARY.exists():
        raise FileExistsError("contact-mode response output already exists")
    import mujoco
    import mujoco_warp
    import warp
    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    scene = Path(config["model_path"])
    names, groups = actuator_groups(scene)
    if groups["right_wrist_translation"] != [0, 1, 2]:
        raise ValueError("right-wrist translation indices changed")
    policies = {label: run_policy(label, files) for label, files in POLICIES.items()}
    old = policies["old_scale_1"]["paired_summary"]
    new = policies["new_scale_005"]["paired_summary"]
    one_values = [
        item["epsilon_sign_agreement_after_1_step"]["matching_sign_fraction"]
        for item in (old, new)
        if item["epsilon_sign_agreement_after_1_step"]["matching_sign_fraction"] is not None
    ]
    three_values = [
        item["epsilon_sign_agreement_after_3_steps"]["matching_sign_fraction"]
        for item in (old, new)
        if item["epsilon_sign_agreement_after_3_steps"]["matching_sign_fraction"] is not None
    ]
    one_mean = float(np.mean(one_values)) if one_values else None
    three_mean = float(np.mean(three_values)) if three_values else None
    delay_evidence = one_mean is not None and three_mean is not None and three_mean > one_mean
    strict_linear_support = bool(three_values) and all(value == 1.0 for value in three_values)
    no_contact_stable_estimates = (
        old["contact_stable_local_estimate_count"] == 0
        and new["contact_stable_local_estimate_count"] == 0
    )
    report = {
        "schema": "endpoint45_49_contact_mode_response_v1",
        "status": "read_only_contact_mode_response_complete",
        "scope": {
            "training_executed": False,
            "policy_called_during_probes": False,
            "policy_modified": False,
            "physics_modified": False,
            "paired_common_support": list(PAIRED_SOURCES),
            "old_policy_only_extension": list(OLD_ONLY_SOURCES),
            "new_policy_missing_followup_actions": list(OLD_ONLY_SOURCES),
            "paired_outcome_endpoints": {
                str(source): [source + 1, source + 2, source + 3]
                for source in PAIRED_SOURCES
            },
            "epsilon_m": list(EPSILONS_M),
            "perturbed_actuators": names[:3],
            "formal_action_clip_retained": True,
            "response_horizon_control_intervals": 3,
            "response_horizon_seconds": 0.1,
            "physics_substeps_per_response": 30,
            "contact_mode_definition": (
                "exact per-substep multiset of unordered geom pairs with dist<=0, including multiplicity"
            ),
            "local_estimate_gate": (
                "baseline and every feasible signed perturbation must have identical contact mode "
                "at all 30 corresponding physics substeps"
            ),
            "stability_metric": (
                "position-error derivative sign agreement between 0.5 mm and 1.0 mm, "
                "evaluated on contact-stable estimates available at both sizes"
            ),
        },
        "artifacts": {
            "boundary": {"path": str(BOUNDARY), "sha256": sha256(BOUNDARY)},
            "scene": {"path": str(scene), "sha256": sha256(scene)},
        },
        "runtime": {
            "device": "cpu",
            "mujoco": mujoco.__version__,
            "mujoco_warp": mujoco_warp.__version__,
            "warp": warp.__version__,
            "spider_root": str((ROOT / "external/spider_compat").resolve()),
        },
        "policies": policies,
        "diagnosis": {
            "paired_old_one_step_epsilon_stability": old["epsilon_sign_agreement_after_1_step"],
            "paired_old_three_step_epsilon_stability": old["epsilon_sign_agreement_after_3_steps"],
            "paired_new_one_step_epsilon_stability": new["epsilon_sign_agreement_after_1_step"],
            "paired_new_three_step_epsilon_stability": new["epsilon_sign_agreement_after_3_steps"],
            "mean_one_step_stability_fraction": one_mean,
            "mean_three_step_stability_fraction": three_mean,
            "descriptive_response_delay_evidence": delay_evidence,
            "strict_contact_stable_local_linear_support": strict_linear_support,
            "no_contact_stable_local_estimates_on_common_support": no_contact_stable_estimates,
            "stop_further_local_linear_probing_of_frozen_policies": no_contact_stable_estimates,
            "reason": (
                "Every formally feasible wrist perturbation changed the exact 30-substep "
                "contact mode, while repeated unperturbed baselines were exact. The frozen "
                "trajectories therefore provide no contact-stable sample on which a local "
                "linear response or response-delay direction can be estimated."
                if no_contact_stable_estimates else
                "Some contact-stable estimates remain; inspect the reported stability metrics."
            ),
            "old_policy_only_extension_excluded_from_pairwise_claims": True,
            "scale_sweep_authorized": False,
            "training_authorized": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with gzip.open(FULL_REPORT, "wt", compresslevel=9) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    summary = {
        "schema": report["schema"],
        "status": report["status"],
        "full_report": {
            "path": str(FULL_REPORT),
            "artifact_sha256": sha256(FULL_REPORT),
        },
        "scope": report["scope"],
        "paired_summaries": {
            label: policy["paired_summary"] for label, policy in policies.items()
        },
        "old_policy_only_extension_summary": policies["old_scale_1"][
            "old_policy_only_extension_summary"
        ],
        "diagnosis": report["diagnosis"],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
