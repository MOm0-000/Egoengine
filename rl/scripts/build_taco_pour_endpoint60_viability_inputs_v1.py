#!/usr/bin/env python3
"""Build carryover evidence and the three exact endpoint-60 viability states."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external/human2sim2robot"),
    str(ROOT / "external/spider_compat"),
]

from run_taco_pour_algorithmic_candidate_C_v5 import (  # noqa: E402
    artifact,
    exact_equal,
    load_checkpoint,
    sha256,
)
from run_taco_pour_candidate_D_chunk_local_v1 import (  # noqa: E402
    _validation_summary,
)


CONTRACT = ROOT / "configs/taco_pour_endpoint60_viability_adjudication_v1.yaml"
OUTPUT = ROOT / "runs/taco_pour_endpoint60_viability_adjudication_v1"


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _write_snapshot(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    raw = buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)
    restored = load_checkpoint(path)
    if not exact_equal(state, restored):
        raise RuntimeError(f"snapshot serialization changed payload: {path}")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
        "payload_roundtrip_exact": True,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {
        "minimum": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _historical_source60(
    *, manifests: list[Path], output: Path, contact_coefficient: float
) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[Any]]] = {
        name: {
            "score": [],
            "position": [],
            "rotation": [],
            "tracking_reward": [],
            "contact_bonus": [],
            "action": [],
            "action_preclamp": [],
            "contact_flags": [],
            "seed": [],
            "epoch": [],
            "world_index": [],
            "row_index": [],
            "terminated": [],
        }
        for name in ("A", "B")
    }
    checked_files = 0
    for seed, manifest_path in enumerate(manifests):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "taco_ppo_training_visitation_v6_world_indexed":
            raise ValueError("historical visitation is not world-indexed v6")
        for epoch_row in manifest["epochs"]:
            epoch = int(epoch_row["epoch"])
            if not 63 <= epoch <= 125:
                continue
            visits = Path(epoch_row["visits"]["path"])
            if sha256(visits) != epoch_row["visits"]["sha256"]:
                raise ValueError(f"historical visitation artifact changed: {visits}")
            checked_files += 1
            with np.load(visits, allow_pickle=False) as data:
                source = np.asarray(data["source_endpoint"], np.int32)
                worlds = np.asarray(data["world_index"], np.int32)
                for name, world_index in (("A", 2), ("B", 3)):
                    indices = np.flatnonzero((source == 60) & (worlds == world_index))
                    if not len(indices):
                        raise RuntimeError(
                            f"seed {seed} epoch {epoch} boundary {name} has no source60 row"
                        )
                    flags = np.asarray(data["contact_flags"], bool)[indices]
                    # Tool is object role 0. opposition_contact is computed per
                    # hand and then averaged over the two hands and one tool.
                    opposition = flags[:, :, 0, 0] & flags[:, :, 0, 1:].any(axis=-1)
                    contact = contact_coefficient * opposition.mean(axis=1)
                    score = np.asarray(data["tool_objective_score"], np.float32)[indices]
                    bucket = buckets[name]
                    bucket["score"].extend(score.tolist())
                    bucket["position"].extend(
                        np.asarray(data["tool_position_error_m"], np.float32)[indices].tolist()
                    )
                    bucket["rotation"].extend(
                        np.asarray(data["tool_rotation_error_rad"], np.float32)[indices].tolist()
                    )
                    bucket["tracking_reward"].extend((1.0 - score).tolist())
                    bucket["contact_bonus"].extend(contact.tolist())
                    bucket["action"].extend(
                        np.asarray(data["sampled_action_clamped"], np.float32)[indices]
                    )
                    bucket["action_preclamp"].extend(
                        np.asarray(data["sampled_action_preclamp"], np.float32)[indices]
                    )
                    bucket["contact_flags"].extend(flags)
                    bucket["seed"].extend([seed] * len(indices))
                    bucket["epoch"].extend([epoch] * len(indices))
                    bucket["world_index"].extend([world_index] * len(indices))
                    bucket["row_index"].extend(indices.tolist())
                    bucket["terminated"].extend(
                        np.asarray(data["tracking_terminated"], bool)[indices].tolist()
                    )

    report: dict[str, Any] = {
        "schema": "taco_pour_candidate_F_source60_historical_samples_v1",
        "status": "completed_without_new_simulation",
        "simulation_control_intervals": 0,
        "visitation_files_hash_checked": checked_files,
        "mapping": {"A": "world_index_2", "B": "world_index_3"},
        "logger_limitations": {
            "lift_reward": "not_recorded_not_reconstructible_without_simulation",
            "total_reward": "not_recorded_not_reconstructible_without_simulation",
            "cross_artifact_credit_join_attempted": False,
        },
        "boundaries": {},
    }
    for name, raw in buckets.items():
        arrays = {
            key: np.asarray(value)
            for key, value in raw.items()
        }
        if len(arrays["score"]) != 7560:
            raise RuntimeError(f"boundary {name} expected 7560 source60 samples")
        best_index = int(np.argmin(arrays["score"]))
        if not bool(arrays["terminated"].all()):
            raise RuntimeError(f"boundary {name} historical source60 sample survived endpoint61")
        report["boundaries"][name] = {
            "sample_count": int(len(arrays["score"])),
            "all_tracking_terminated_at_endpoint61": True,
            "endpoint61_tracking_score": _distribution(arrays["score"]),
            "endpoint61_position_error_m": _distribution(arrays["position"]),
            "endpoint61_rotation_error_rad": _distribution(arrays["rotation"]),
            "raw_tracking_reward_C_minus_e": _distribution(arrays["tracking_reward"]),
            "contact_bonus": _distribution(arrays["contact_bonus"]),
            "lift_reward": None,
            "total_reward": None,
            "best_score_minus_1": float(arrays["score"][best_index] - 1.0),
            "best_historical_sample": {
                "score": float(arrays["score"][best_index]),
                "position_error_m": float(arrays["position"][best_index]),
                "rotation_error_rad": float(arrays["rotation"][best_index]),
                "raw_tracking_reward_C_minus_e": float(
                    arrays["tracking_reward"][best_index]
                ),
                "contact_bonus": float(arrays["contact_bonus"][best_index]),
                "sampled_action_clamped": arrays["action"][best_index].tolist(),
                "sampled_action_preclamp": arrays["action_preclamp"][best_index].tolist(),
                "seed": int(arrays["seed"][best_index]),
                "epoch": int(arrays["epoch"][best_index]),
                "world_index": int(arrays["world_index"][best_index]),
                "trace_row_index": int(arrays["row_index"][best_index]),
            },
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    if (
        contract.get("schema") != "taco_pour_endpoint60_viability_adjudication_v1"
        or contract.get("status") != "authorized_read_only_adjudication"
        or contract.get("paper_faithful") is not False
        or contract["carryover"]["candidate"] != "D"
        or contract["carryover"]["seed"] != 2
        or contract["carryover"]["epoch"] != 125
        or contract["one_step_search"]["sobol"]["actions"] != 4096
        or contract["one_step_search"]["CEM"]["total_actions"] != 3072
    ):
        raise ValueError("endpoint60 viability contract changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "carryover").mkdir()
    (args.output / "boundaries").mkdir()
    (args.output / "one_step_search").mkdir()

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen endpoint60 input changed: {name}")

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
    from video_to_spider.rl.algorithmic_training_v6 import (
        SupportAnchoredBoundedMeanPpoAgent,
        load_support_anchored_profile,
    )
    from video_to_spider.rl.curriculum_reset import capture_physics_rnn_boundary
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        CanonicalOldPolicyGateSpec,
        LikelihoodIdentityGateSpec,
    )

    objective = load_runtime_objective(
        inputs["protocol_at_authorization"], inputs["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        inputs["protocol_at_authorization"], inputs["observation_profile"],
        require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(inputs["action_profile"])
    distribution, _ = load_support_anchored_profile(inputs["distribution_profile"])
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    source_boundary = load_checkpoint(inputs["source_boundary_endpoint20"])
    checkpoint = load_checkpoint(inputs["carryover_checkpoint"])
    if (
        np.asarray(source_boundary["time_indices"]).tolist() != [20]
        or checkpoint.get("candidate") != "D"
        or checkpoint.get("seed") != 2
        or checkpoint.get("agent_epoch") != 125
    ):
        raise ValueError("carryover lineage changed")
    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)

    world = MJWPVectorEnv(
        cpu_config, reference, num_envs=1,
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
        seed=2,
    )
    verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
    world.set_env_state(source_boundary)
    ppo_config = replace(
        _build_ppo_config(
            num_envs=1, horizon_length=40, seq_length=4, max_epochs=1,
            learning_rate=1.0e-4, device="cpu", asymmetric_critic=None,
            actor_mini_epochs=1,
        ),
        clip_actions=False,
        bounds_loss_coef=0.0,
        bound_loss_type="regularisation",
        lr_schedule=None,
    )
    temporary = tempfile.TemporaryDirectory(prefix=".endpoint60_carryover_", dir=ROOT / "runs")
    try:
        agent = SupportAnchoredBoundedMeanPpoAgent(
            experiment_dir=Path(temporary.name) / "agent",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=world,
            distribution_spec=distribution,
            likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
            canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
            audit_dir=Path(temporary.name) / "unused_audit",
        )
        agent.model.load_state_dict(checkpoint["actor"], strict=True)
        agent.set_eval()
        agent.rnn_states = [
            state.to("cpu").zero_() for state in agent.model.get_default_rnn_state()
        ]
        agent.dones = torch.zeros(1, dtype=torch.uint8)
        agent.current_rewards = torch.zeros(1, 1, dtype=torch.float32)
        agent.current_shaped_rewards = torch.zeros(1, 1, dtype=torch.float32)
        agent.current_lengths = torch.zeros(1, dtype=torch.float32)

        rows: dict[str, list[Any]] = {
            "endpoint": [], "terminated": [], "tracking_score": [],
            "position_error": [], "rotation_error": [], "ctrl": [],
            "qpos": [], "qvel": [], "contact_flags": [],
            "deterministic_action": [], "raw_location": [], "bounded_mu": [],
            "actor_sigma": [], "action_low": [], "action_high": [],
            "reward": [], "tracking_reward": [], "contact_bonus": [],
            "lift_reward": [],
        }
        prefix: list[Any] = []
        endpoint40_boundary = None
        endpoint60_boundary = None
        for source in range(20, 80):
            raw_observation = world.current_observation()
            prefix.append(deepcopy(raw_observation))
            packed = agent.obs_to_tensors(raw_observation)
            result = agent.get_deterministic_action_values(packed)
            agent.rnn_states = result["rnn_states"]
            action = agent.preprocess_actions(result["deterministic_actions"])
            _, reward, _, info = world.step(action, auto_reset=False)
            rows["endpoint"].append(source + 1)
            rows["terminated"].append(bool(info["terminated"][0]))
            rows["tracking_score"].append(float(info["object_tracking_error"][0]))
            rows["position_error"].append(float(info["object_position_error"][0, 0]))
            rows["rotation_error"].append(float(info["object_rotation_error"][0, 0]))
            rows["ctrl"].append(world._last_ctrl[0].detach().cpu().numpy().copy())
            rows["qpos"].append(
                world._mjwp.get_qpos(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            )
            rows["qvel"].append(
                world._mjwp.get_qvel(cpu_config, world.env)[0].detach().cpu().numpy().copy()
            )
            rows["contact_flags"].append(np.asarray(info["contact_flags"][0]).copy())
            rows["deterministic_action"].append(np.asarray(action[0]).copy())
            rows["raw_location"].append(result["raw_locations"][0].detach().cpu().numpy().copy())
            rows["bounded_mu"].append(result["mus"][0].detach().cpu().numpy().copy())
            rows["actor_sigma"].append(result["sigmas"][0].detach().cpu().numpy().copy())
            rows["action_low"].append(result["action_lows"][0].detach().cpu().numpy().copy())
            rows["action_high"].append(result["action_highs"][0].detach().cpu().numpy().copy())
            rows["reward"].append(float(reward[0]))
            rows["tracking_reward"].append(float(info["aggregate_tracking_reward"][0]))
            rows["contact_bonus"].append(float(info["aggregate_contact_bonus"][0]))
            rows["lift_reward"].append(float(info["lift_reward"][0]))
            if source + 1 == 40:
                endpoint40_boundary = capture_physics_rnn_boundary(
                    agent, world, rollout_start_endpoint=20,
                    observation_prefix=prefix,
                    provenance={
                        "role": "previous_policy_carryover_endpoint40",
                        "candidate": "D", "seed": 2, "epoch": 125,
                        "RNN_reset": False,
                        "direct_chunk_commit_allowed": False,
                    },
                )
            if source + 1 == 60:
                endpoint60_boundary = capture_physics_rnn_boundary(
                    agent, world, rollout_start_endpoint=20,
                    observation_prefix=prefix,
                    provenance={
                        "role": "Boundary_P_previous_policy_carryover_endpoint60",
                        "candidate": "D", "seed": 2, "epoch": 125,
                        "natural_simulator_rollout": True,
                        "reference_or_GT_state_injected": False,
                        "RNN_reset": False,
                        "direct_chunk_commit_allowed": False,
                    },
                )
    finally:
        if "agent" in locals() and agent.writer is not None:
            agent.writer.close()
        temporary.cleanup()

    if endpoint40_boundary is None or endpoint60_boundary is None:
        raise RuntimeError("continuous rollout did not capture required boundaries")
    arrays = {
        key: np.asarray(value, dtype=np.int32) if key == "endpoint" else np.asarray(value)
        for key, value in rows.items()
    }
    arrays["actor_mu"] = arrays["bounded_mu"]
    arrays["object_pose"] = arrays["qpos"][:, -14:].copy()
    with np.load(inputs["promoted_historical_trajectory"], allow_pickle=False) as historical:
        exact_checks = {
            name: _array_exact(arrays[name][:40], historical[name])
            for name in historical.files
        }
    if not all(exact_checks.values()):
        failure = {
            "schema": "taco_pour_endpoint60_policy_carryover_failure_v1",
            "status": "failed_closed_historical_prefix_mismatch",
            "endpoint21_60_array_checks": exact_checks,
            "search_authorized": False,
            "training_executed": False,
            "chunk_commit_written": False,
        }
        (args.output / "carryover" / "failure_report.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        raise RuntimeError("previous policy did not reproduce promoted endpoint21-60")

    continuous_path = args.output / "carryover" / "continuous_20_80.npz"
    np.savez_compressed(continuous_path, **arrays)
    window_arrays = {key: value[20:] for key, value in arrays.items()}
    window_summary = _validation_summary(window_arrays)
    boundary40_artifact = _write_snapshot(
        args.output / "carryover" / "endpoint40_physics_rnn.pt.gz",
        endpoint40_boundary,
    )
    boundary_p_artifact = _write_snapshot(
        args.output / "boundaries" / "boundary_P_endpoint60.pt.gz",
        endpoint60_boundary,
    )
    carryover_report = {
        "schema": "taco_pour_endpoint60_policy_carryover_v1",
        "status": (
            "strict_40_of_40_policy_carryover"
            if window_summary["forty_of_forty"]
            else "continuous_policy_does_not_solve_next_window"
        ),
        "paper_faithful": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "endpoint21_60_all_historical_arrays_bitwise_equal": True,
        "endpoint21_60_array_checks": exact_checks,
        "endpoint41_80_summary": window_summary,
        "RNN_reset_at_endpoint40_or60": False,
        "continuous_trajectory": artifact(continuous_path),
        "endpoint40_physics_RNN": boundary40_artifact,
        "boundary_P_endpoint60": boundary_p_artifact,
        "chunk_commit_written": False,
        "endpoint20_to40_committed": True,
        "endpoint40_to60_committed": False,
    }
    (args.output / "carryover" / "report.json").write_text(
        json.dumps(carryover_report, indent=2) + "\n"
    )

    if window_summary["forty_of_forty"]:
        decision = {
            "schema": "taco_pour_endpoint60_viability_decision_v1",
            "classification": "C1",
            "label": "POLICY_CARRYOVER_SOLVES_NEXT_WINDOW",
            "one_step_search_executed": False,
            "separate_promotion_request_required": True,
            "chunk_commit_written": False,
        }
        (args.output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps({"carryover": carryover_report, "decision": decision}, indent=2))
        return

    # A/B are copied byte-for-byte from the accepted Candidate-F builder.
    boundary_artifacts = {"P": boundary_p_artifact}
    for name in ("A", "B"):
        source = inputs[f"boundary_{name}"]
        destination = args.output / "boundaries" / f"boundary_{name}_endpoint60.pt.gz"
        shutil.copyfile(source, destination)
        if sha256(destination) != contract["inputs"][f"boundary_{name}"]["sha256"]:
            raise RuntimeError(f"boundary {name} copy changed bytes")
        payload = load_checkpoint(destination)
        if (
            payload.get("schema") != "egoengine_physics_rnn_boundary_v1"
            or int(payload.get("reference_endpoint", -1)) != 60
            or bool(np.asarray(payload["physics_state"]["last_terminated"])[0])
        ):
            raise ValueError(f"boundary {name} is not a live endpoint60 state")
        boundary_artifacts[name] = artifact(destination)

    historical = _historical_source60(
        manifests=[
            inputs[f"candidate_F_seed{seed}_visitation_manifest"]
            for seed in (0, 1, 2)
        ],
        output=args.output / "historical_source60_samples.json",
        contact_coefficient=float(objective.contact_coefficient),
    )
    setup = {
        "schema": "taco_pour_endpoint60_viability_search_setup_v1",
        "status": "carryover_failed_search_authorized",
        "carryover": carryover_report,
        "boundaries": boundary_artifacts,
        "historical_source60": historical,
        "one_step_search_authorized": True,
        "training_executed": False,
        "chunk_commit_written": False,
    }
    (args.output / "setup_report.json").write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps({
        "status": setup["status"],
        "carryover_summary": window_summary,
        "historical_best_score_minus_1": {
            name: historical["boundaries"][name]["best_score_minus_1"]
            for name in ("A", "B")
        },
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
