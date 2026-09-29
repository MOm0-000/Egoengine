#!/usr/bin/env python3
"""Run a zero-residual Replay lookahead from a committed chunk boundary."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
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
    exact_equal,
    load_checkpoint,
    sha256,
)


CONTRACT = ROOT / "configs/taco_pour_progressive_chunk_execution_v1.yaml"
RUN_ROOT = ROOT / "runs/taco_pour_progressive_chunk_execution_v1"


def _write_snapshot(path: Path, state: dict[str, Any]) -> dict[str, object]:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    raw = buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    restored = load_checkpoint(path)
    if not exact_equal(state, restored):
        raise RuntimeError("Replay boundary serialization changed state")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
        "payload_roundtrip_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    stage = contract.get("active_stage", {})
    if (
        contract.get("schema") != "taco_pour_progressive_chunk_execution_v1"
        or contract.get("status") != "authorized_replay_from_promoted_endpoint40"
        or contract.get("paper_faithful") is not False
        or stage.get("mode") != "Replay_zero_residual"
        or stage.get("source_endpoint") != 40
        or stage.get("lookahead_endpoint") != 80
        or stage.get("commit_target_endpoint") != 60
        or stage.get("candidate_D_policy_allowed") is not False
        or stage.get("RNN_state_allowed") is not False
    ):
        raise ValueError("progressive Replay contract changed")
    if args.run_root.resolve() != Path(contract["output_directory"]).resolve():
        raise ValueError("progressive Replay output root changed")
    output = args.run_root / "chunk_source_40" / "replay"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    inputs = {name: Path(row["path"]) for name, row in contract["inputs"].items()}
    for name, path in inputs.items():
        if not path.is_file() or sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"frozen Replay input changed: {name}")
    implementations = {
        "MJWP_environment_sha256": ROOT / "src/video_to_spider/rl/mjwp_env.py",
        "PPO_config_builder_sha256": ROOT / "scripts/run_mjwp_ppo.py",
        "replay_runner_sha256": Path(__file__).resolve(),
    }
    for name, path in implementations.items():
        if sha256(path) != contract["implementation_contract"][name]:
            raise ValueError(f"Replay implementation changed: {name}")

    from run_mjwp_ppo import (
        MJWPVectorEnv,
        MJWPVectorEnvConfig,
        _load_ego_config,
        _load_reference,
    )
    from run_taco_replay_rl import load_accepted_initialization, verify_runtime_model
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation

    objective = load_runtime_objective(
        inputs["protocol_at_authorization"],
        inputs["objective_profile"],
        tracking_variant="tool_only",
        require_run_ready=False,
    )
    observation = load_runtime_observation(
        inputs["protocol_at_authorization"],
        inputs["observation_profile"],
        require_run_ready=False,
    )
    residual, residual_report = load_residual_action_profile(inputs["action_profile"])
    _, initialization = load_accepted_initialization(
        inputs["initialization_report"], inputs["simulator_config"]
    )
    boundary = load_checkpoint(inputs["source_boundary"])
    if (
        boundary.get("snapshot_schema")
        != "egoengine_mjwp_snapshot_v3_reward_aligned"
        or boundary.get("mujoco_warp_version") != "3.13.0"
        or np.asarray(boundary["time_indices"]).tolist() != [40]
    ):
        raise ValueError("progressive Replay requires exact endpoint-40 boundary")

    cpu_config = _load_ego_config(str(inputs["simulator_config"]), "cpu")
    reference = _load_reference(cpu_config.data_path, "cpu", expected_frequency=30)
    world = MJWPVectorEnv(
        cpu_config,
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
        seed=0,
    )
    verify_runtime_model(world.env.model_cpu, initialization["validated_physics_contract"])
    world.set_env_state(boundary)

    rows: dict[str, list[Any]] = {
        "source_endpoint": [],
        "endpoint": [],
        "terminated": [],
        "tracking_score": [],
        "position_error": [],
        "rotation_error": [],
        "ctrl": [],
        "qpos": [],
        "qvel": [],
        "contact_flags": [],
        "deterministic_action": [],
    }
    endpoint60_state: dict[str, Any] | None = None
    zero = torch.zeros((1, 36), dtype=torch.float32)
    for source in range(40, 80):
        _, _, _, info = world.step(zero, auto_reset=False)
        rows["source_endpoint"].append(source)
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
        rows["deterministic_action"].append(np.zeros(36, dtype=np.float32))
        if source + 1 == 60:
            endpoint60_state = world.get_env_state()

    arrays = {
        name: np.asarray(values, dtype=np.int32)
        if name in {"source_endpoint", "endpoint"}
        else np.asarray(values)
        for name, values in rows.items()
    }
    arrays["object_pose"] = arrays["qpos"][:, -14:].copy()
    validation_path = output / "replay_validation_40_80.npz"
    np.savez_compressed(validation_path, **arrays)
    failure_index = next(
        (index for index, done in enumerate(arrays["terminated"].tolist()) if done),
        None,
    )
    successful = 40 if failure_index is None else failure_index
    first_failure = None if failure_index is None else int(arrays["endpoint"][failure_index])
    summary = {
        "successful_intervals": successful,
        "first_failure_endpoint": first_failure,
        "forty_of_forty": failure_index is None,
        "endpoint_scores": {
            str(endpoint): float(arrays["tracking_score"][endpoint - 41])
            for endpoint in (60, 70, 80)
        },
        "first_failure_score": None
        if failure_index is None
        else float(arrays["tracking_score"][failure_index]),
        "deterministic_residual_bitwise_zero": bool(
            arrays["deterministic_action"].tobytes()
            == np.zeros_like(arrays["deterministic_action"]).tobytes()
        ),
    }
    common = {
        "schema": "taco_pour_progressive_replay_report_v1",
        "paper_faithful": False,
        "classification": "progressive_zero_residual_replay_gate",
        "source_endpoint": 40,
        "lookahead_endpoint": 80,
        "commit_target_endpoint": 60,
        "summary": summary,
        "candidate_D_policy_loaded": False,
        "RNN_state_used": False,
        "fresh_training_executed": False,
        "residual_action": residual_report,
        "provenance": {
            "contract": artifact(args.contract),
            "promotion_report": artifact(inputs["promotion_report"]),
            "source_boundary": artifact(inputs["source_boundary"]),
        },
        "validation": artifact(validation_path),
    }
    if failure_index is not None:
        report = {
            **common,
            "status": "replay_failed_no_chunk_commit",
            "chunk_commit_written": False,
            "next_action": "fresh_candidate_D_chunk_local_solve_required",
        }
    else:
        if endpoint60_state is None:
            raise RuntimeError("Replay passed but endpoint-60 state was not captured")
        chunk_path = output / "committed_chunk_40_60.npz"
        np.savez_compressed(
            chunk_path,
            **{name: value[:20].copy() for name, value in arrays.items()},
        )
        boundary_artifact = _write_snapshot(
            output / "committed_boundary_endpoint_60.pt.gz", endpoint60_state
        )
        report = {
            **common,
            "status": "replay_passed_chunk_40_60_committed",
            "chunk_commit_written": True,
            "lookahead_committed": False,
            "artifacts": {
                "committed_chunk_40_60": artifact(chunk_path),
                "committed_boundary_endpoint_60": boundary_artifact,
            },
            "next_action": "repeat_replay_from_endpoint60_to100",
        }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": summary, "report": artifact(report_path)}, indent=2))


if __name__ == "__main__":
    main()
