#!/usr/bin/env python3
"""Bounded one-step endpoint-60 action search on one frozen physics state."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
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
    load_checkpoint,
    sha256,
)
from video_to_spider.rl.replay_rl import _model_state_sha256  # noqa: E402


CONTRACT = ROOT / "configs/taco_pour_endpoint60_viability_adjudication_v1.yaml"
OUTPUT = ROOT / "runs/taco_pour_endpoint60_viability_adjudication_v1"


def _project(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(action, np.float32), low), high)


def _array_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _result_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    scalar_keys = (
        "tracking_score", "position_error_m", "rotation_error_rad",
        "terminated", "reward", "tracking_reward", "contact_bonus", "lift_reward",
    )
    return all(left[key] == right[key] for key in scalar_keys) and all(
        _array_exact(left[key], right[key])
        for key in ("action", "qpos", "qvel", "ctrl", "contact_flags")
    )


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(payload, indent=2) + "\n").encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    restored = json.loads(gzip.decompress(path.read_bytes()))
    if restored != payload:
        raise RuntimeError("gzip JSON roundtrip changed payload")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_json_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", choices=("A", "B", "P"), required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    name = args.boundary
    contract = yaml.safe_load(args.contract.read_text())
    search = contract["one_step_search"]
    if (
        contract.get("schema") != "taco_pour_endpoint60_viability_adjudication_v1"
        or contract.get("status") != "authorized_read_only_adjudication"
        or contract["runtime"]["training_allowed"] is not False
        or search["sobol"] != {
            "seed": 600061, "actions": 4096,
            "distribution": "uniform_exact_state_support",
        }
        or search["CEM"]["seed"] != 610060
        or search["CEM"]["generations"] != 6
        or search["CEM"]["population"] != 512
        or search["CEM"]["total_actions"] != 3072
        or search["CEM"]["elite_count"] != 52
        or search["maximum_new_control_intervals_per_boundary"] != 7168
    ):
        raise ValueError("endpoint60 one-step search contract changed")
    setup = json.loads((args.output / "setup_report.json").read_text())
    if not setup.get("one_step_search_authorized"):
        raise ValueError("carryover stage did not authorize search")
    destination = args.output / "one_step_search" / f"boundary_{name}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)

    inputs = {key: Path(row["path"]) for key, row in contract["inputs"].items()}
    for key, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][key]["sha256"]:
            raise ValueError(f"frozen input changed: {key}")
    boundary_path = args.output / "boundaries" / f"boundary_{name}_endpoint60.pt.gz"
    boundary = load_checkpoint(boundary_path)
    if (
        boundary.get("schema") != "egoengine_physics_rnn_boundary_v1"
        or int(boundary.get("reference_endpoint", -1)) != 60
        or bool(np.asarray(boundary["physics_state"]["last_terminated"])[0])
    ):
        raise ValueError("search boundary is not a live endpoint60 state")
    checkpoint_key = {
        "A": "boundary_A_teacher_checkpoint",
        "B": "boundary_B_teacher_checkpoint",
        "P": "carryover_checkpoint",
    }[name]
    checkpoint = load_checkpoint(inputs[checkpoint_key])

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
    ego = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(ego.data_path, "cpu", expected_frequency=30)
    env = MJWPVectorEnv(
        ego, reference, num_envs=1,
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
        seed={"A": 0, "B": 1, "P": 2}[name],
    )
    verify_runtime_model(env.env.model_cpu, initialization["validated_physics_contract"])
    temporary = tempfile.TemporaryDirectory(
        prefix=f".endpoint60_{name}_", dir=ROOT / "runs"
    )
    try:
        ppo = replace(
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
        agent = SupportAnchoredBoundedMeanPpoAgent(
            experiment_dir=Path(temporary.name) / "agent",
            ppo_config=ppo,
            network_config=_build_network_config(4),
            env=env,
            distribution_spec=distribution,
            likelihood_identity_gate=LikelihoodIdentityGateSpec.float32_ulp_aware_v1(),
            canonical_old_policy_gate=CanonicalOldPolicyGateSpec.v1(),
            audit_dir=Path(temporary.name) / "unused_audit",
        )
        agent.model.load_state_dict(checkpoint["actor"], strict=True)
        agent.set_eval()
        actor_hash = _model_state_sha256(agent.model.state_dict())
        if actor_hash != boundary["actor_state_sha256"]:
            raise ValueError(f"boundary {name} RNN belongs to a different actor")
        env.set_env_state(boundary["physics_state"])
        observed = env.current_observation()
        saved_observation = boundary["observation"]
        if isinstance(observed, dict):
            if not isinstance(saved_observation, dict) or set(observed) != set(saved_observation):
                raise RuntimeError("boundary observation structure changed")
            observation_exact = all(
                _array_exact(np.asarray(observed[key]), np.asarray(saved_observation[key]))
                for key in observed
            )
        elif isinstance(saved_observation, dict):
            # Candidate-F A/B boundaries were captured in an asymmetric-critic
            # training environment.  The actor input is the `obs` member; the
            # additional `states` member belongs only to the critic.
            observation_exact = set(saved_observation) == {"obs", "states"} and _array_exact(
                np.asarray(observed), np.asarray(saved_observation["obs"])
            )
        else:
            observation_exact = _array_exact(
                np.asarray(observed), np.asarray(saved_observation)
            )
        if not observation_exact:
            raise RuntimeError("boundary observation changed")
        agent.rnn_states = [state.clone().to("cpu") for state in boundary["rnn_states"]]
        packed = agent.obs_to_tensors(observed)
        actor = agent.get_deterministic_action_values(packed)
        deterministic = np.asarray(
            agent.preprocess_actions(actor["deterministic_actions"])[0], np.float32
        )
        low_t, high_t = env.current_normalized_action_bounds()
        low = low_t[0].detach().cpu().numpy().astype(np.float32)
        high = high_t[0].detach().cpu().numpy().astype(np.float32)
        if not np.array_equal(low, actor["action_lows"][0].detach().cpu().numpy()):
            raise RuntimeError("actor and environment low bounds differ")
        if not np.array_equal(high, actor["action_highs"][0].detach().cpu().numpy()):
            raise RuntimeError("actor and environment high bounds differ")
        if np.any(low > 0.0) or np.any(high < 0.0):
            raise RuntimeError("zero residual is outside exact support")

        historical = json.loads((args.output / "historical_source60_samples.json").read_text())
        anchors = [
            ("zero_residual", np.zeros(36, np.float32)),
            ("teacher_or_current_deterministic", deterministic),
            ("historical_best_A", np.asarray(
                historical["boundaries"]["A"]["best_historical_sample"]["sampled_action_clamped"],
                np.float32,
            )),
            ("historical_best_B", np.asarray(
                historical["boundaries"]["B"]["best_historical_sample"]["sampled_action_clamped"],
                np.float32,
            )),
        ]
        anchor_rows = []
        unique_anchors: list[tuple[str, np.ndarray]] = []
        for label, raw_action in anchors:
            projected = _project(raw_action, low, high)
            anchor_rows.append({
                "name": label,
                "projection_Linf": float(np.max(np.abs(projected - raw_action))),
                "action": projected.tolist(),
            })
            if not any(np.array_equal(projected, prior) for _, prior in unique_anchors):
                unique_anchors.append((label, projected))

        def evaluate(action: np.ndarray) -> dict[str, Any]:
            env.set_env_state(boundary["physics_state"])
            action = np.asarray(action, np.float32)
            _, reward, _, info = env.step(
                torch.as_tensor(action[None], dtype=torch.float32), auto_reset=False
            )
            qpos = env._mjwp.get_qpos(ego, env.env)[0].detach().cpu().numpy().copy()
            qvel = env._mjwp.get_qvel(ego, env.env)[0].detach().cpu().numpy().copy()
            score = float(info["object_tracking_error"][0])
            terminated = bool(info["terminated"][0])
            return {
                "action": action.copy(),
                "tracking_score": score,
                "position_error_m": float(info["object_position_error"][0, 0]),
                "rotation_error_rad": float(info["object_rotation_error"][0, 0]),
                "terminated": terminated,
                "feasible": bool(np.isfinite(score) and score <= 1.0 and not terminated),
                "reward": float(reward[0]),
                "tracking_reward": float(info["aggregate_tracking_reward"][0]),
                "contact_bonus": float(info["aggregate_contact_bonus"][0]),
                "lift_reward": float(info["lift_reward"][0]),
                "qpos": qpos,
                "qvel": qvel,
                "ctrl": env._last_ctrl[0].detach().cpu().numpy().copy(),
                "contact_flags": np.asarray(info["contact_flags"][0]).copy(),
            }

        total = int(search["sobol"]["actions"])
        sobol = torch.quasirandom.SobolEngine(
            dimension=36, scramble=True, seed=int(search["sobol"]["seed"])
        ).draw(total).numpy().astype(np.float32)
        phase_actions = low + sobol * (high - low)
        phase_labels = ["sobol"] * total
        for index, (label, action) in enumerate(unique_anchors):
            phase_actions[index] = action
            phase_labels[index] = label

        records: list[dict[str, Any]] = []
        for index, action in enumerate(phase_actions):
            row = evaluate(action)
            row.update({"phase": phase_labels[index], "generation": -1, "index": index})
            records.append(row)
        best_index = min(range(len(records)), key=lambda index: records[index]["tracking_score"])
        mean = records[best_index]["action"].astype(np.float64)
        std = 0.25 * (high - low).astype(np.float64)
        rng = np.random.default_rng(int(search["CEM"]["seed"]))
        population = int(search["CEM"]["population"])
        elite_count = int(search["CEM"]["elite_count"])
        generation_reports = []
        for generation in range(int(search["CEM"]["generations"])):
            samples = rng.normal(mean, std, size=(population, 36))
            samples = np.clip(samples, low, high).astype(np.float32)
            samples[0] = mean.astype(np.float32)
            generation_rows = []
            for sample_index, action in enumerate(samples):
                row = evaluate(action)
                row.update({
                    "phase": "CEM", "generation": generation,
                    "index": sample_index,
                })
                records.append(row)
                generation_rows.append(row)
            order = np.argsort(
                np.asarray([row["tracking_score"] for row in generation_rows]),
                kind="stable",
            )
            elite = np.asarray(
                [generation_rows[int(index)]["action"] for index in order[:elite_count]],
                np.float64,
            )
            mean = elite.mean(axis=0)
            std = elite.std(axis=0)
            generation_reports.append({
                "generation": generation,
                "feasible_count": int(sum(row["feasible"] for row in generation_rows)),
                "best_tracking_score": float(generation_rows[int(order[0])]["tracking_score"]),
                "mean_standard_deviation": float(std.mean()),
            })
            print(
                f"boundary={name} generation={generation} "
                f"feasible={generation_reports[-1]['feasible_count']} "
                f"best={generation_reports[-1]['best_tracking_score']:.9f}",
                flush=True,
            )
        if len(records) != 7168:
            raise RuntimeError("one-step search did not consume exactly 7168 intervals")
        scores = np.asarray([row["tracking_score"] for row in records], np.float32)
        feasible = np.asarray([row["feasible"] for row in records], bool)
        order = np.argsort(scores, kind="stable")
        top_indices = order[: int(search["save_top_actions"])]
        best = records[int(order[0])]
        replay = evaluate(best["action"])
        if not _result_exact(best, replay):
            raise RuntimeError(f"boundary {name} best candidate replay is not bitwise exact")

        # The authorized output contract publishes one canonical NPZ per
        # boundary at one_step_search/{A,B,P}.npz.  Per-boundary directories
        # hold only human-readable report/top-32 metadata.
        npz_path = args.output / "one_step_search" / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            action=np.asarray([row["action"] for row in records], np.float32),
            tracking_score=scores,
            position_error_m=np.asarray([row["position_error_m"] for row in records], np.float32),
            rotation_error_rad=np.asarray([row["rotation_error_rad"] for row in records], np.float32),
            terminated=np.asarray([row["terminated"] for row in records], bool),
            feasible=feasible,
            phase=np.asarray([row["phase"] for row in records], "U40"),
            generation=np.asarray([row["generation"] for row in records], np.int16),
            phase_index=np.asarray([row["index"] for row in records], np.int32),
        )
        top_payload = {
            "schema": "taco_pour_endpoint60_one_step_top32_v1",
            "boundary": name,
            "rows": [
                {
                    "rank": rank,
                    "global_index": int(index),
                    "phase": records[int(index)]["phase"],
                    "generation": records[int(index)]["generation"],
                    "phase_index": records[int(index)]["index"],
                    "tracking_score": records[int(index)]["tracking_score"],
                    "position_error_m": records[int(index)]["position_error_m"],
                    "rotation_error_rad": records[int(index)]["rotation_error_rad"],
                    "terminated": records[int(index)]["terminated"],
                    "feasible": records[int(index)]["feasible"],
                    "action": records[int(index)]["action"].tolist(),
                }
                for rank, index in enumerate(top_indices.tolist(), start=1)
            ],
        }
        top_artifact = _write_gzip_json(destination / "top32.json.gz", top_payload)
        report = {
            "schema": "taco_pour_endpoint60_one_step_viability_v1",
            "status": "completed_read_only_search",
            "source_endpoint": 60,
            "outcome_endpoint": 61,
            "training_executed": False,
            "optimizer_updates": 0,
            "physics_control_intervals": 7169,
            "new_search_control_intervals": 7168,
            "exact_state_feasible_support": {
                "low": low.tolist(), "high": high.tolist(),
                "all_actions_within_support": True,
            },
            "actor_state_sha256": actor_hash,
            "boundary": {"name": name, "artifact": artifact(boundary_path)},
            "teacher_checkpoint": artifact(inputs[checkpoint_key]),
            "baseline_candidates": anchor_rows,
            "sobol": search["sobol"],
            "CEM": {**search["CEM"], "generation_reports": generation_reports},
            "evaluated_action_count": 7168,
            "feasible_action_count": int(feasible.sum()),
            "any_feasible_action_found": bool(feasible.any()),
            "best": {
                "global_index": int(order[0]),
                "phase": best["phase"],
                "generation": best["generation"],
                "phase_index": best["index"],
                "tracking_score": best["tracking_score"],
                "score_minus_1": best["tracking_score"] - 1.0,
                "position_error_m": best["position_error_m"],
                "rotation_error_rad": best["rotation_error_rad"],
                "terminated": best["terminated"],
                "feasible": best["feasible"],
                "action": best["action"].tolist(),
                "bitwise_exact_replay": True,
            },
            "arrays": artifact(npz_path),
            "top32": top_artifact,
            "finite_negative_is_mathematical_infeasibility_proof": False,
            "chunk_commit_written": False,
        }
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "boundary": name,
            "feasible_action_count": report["feasible_action_count"],
            "best": report["best"],
            "report": str((destination / "report.json").resolve()),
        }, indent=2))
    finally:
        if "agent" in locals() and agent.writer is not None:
            agent.writer.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
