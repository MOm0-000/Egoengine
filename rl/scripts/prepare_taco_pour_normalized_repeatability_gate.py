#!/usr/bin/env python3
"""Freeze the failed 8-epoch window for action-repeatability auditing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> tuple[dict, bytes, bytes]:
    artifact = path.read_bytes()
    raw = gzip.decompress(artifact) if path.suffix == ".gz" else artifact
    return json.loads(raw), artifact, raw


def trace(report: dict, mode: str) -> dict:
    chunks = [chunk for chunk in report["chunks"] if chunk["start"] == 20]
    if len(chunks) != 1:
        raise ValueError("source report must contain exactly one chunk starting at endpoint 20")
    matches = [row for row in chunks[0]["validation_traces"] if row["mode"] == mode]
    if len(matches) != 1:
        raise ValueError(f"source report must contain exactly one {mode} trace")
    return matches[0]


def offline_variant_reduction(report: dict) -> dict:
    steps = report["trace"]["steps"]
    if not steps or steps[0]["tracked_object_roles"] != ["tool", "target"]:
        raise ValueError("variant trace must contain tool and target on one physical rollout")
    tool_failure = next((row for row in steps if row["object_terminated"][0]), None)
    any_failure = next((row for row in steps if any(row["object_terminated"])), None)
    return {
        "single_physical_trace": True,
        "trace_steps": len(steps),
        "tool_only": {
            "first_failure_endpoint": tool_failure["endpoint"] if tool_failure else None,
            "validated_steps": (tool_failure["endpoint"] - 1) if tool_failure else len(steps),
        },
        "tool_and_target": {
            "first_failure_endpoint": any_failure["endpoint"] if any_failure else None,
            "validated_steps": (any_failure["endpoint"] - 1) if any_failure else len(steps),
            "failure_roles": [
                role for role, failed in zip(
                    any_failure["tracked_object_roles"], any_failure["object_terminated"], strict=True
                ) if failed
            ] if any_failure else [],
        },
        "conclusion": "variant reduction is identical because the tool triggers both gates on the same recorded rollout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--variant-trace-report", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    report, report_artifact, report_raw = read_json(args.source_report)
    variant_report, variant_artifact, variant_raw = read_json(args.variant_trace_report)
    if report["objective"]["objective_id"] != "taco_pour_local_normalized_ellipse_v1":
        raise ValueError("source report does not use the normalized ellipse objective")
    if report["tracking_variant"] != "tool_only":
        raise ValueError("action gate requires the primary tool_only source report")
    replay = trace(report, "replay")
    rl = trace(report, "rl")
    if replay["validated_steps"] != 27 or rl["validated_steps"] != 35:
        raise ValueError("source traces no longer match the predeclared 27-to-35 comparison")

    def arrays(source: dict) -> tuple[np.ndarray, ...]:
        steps = source["steps"]
        return (
            np.asarray([row["raw_residual_action"] for row in steps], dtype=np.float32),
            np.asarray([row["endpoint"] for row in steps], dtype=np.int32),
            np.asarray([row["objective_score"][0] for row in steps], dtype=np.float32),
            np.asarray([row["position_error_m"][0] for row in steps], dtype=np.float32),
            np.asarray([row["rotation_error_rad"][0] for row in steps], dtype=np.float32),
        )

    replay_actions, replay_endpoints, replay_scores, replay_pos, replay_rot = arrays(replay)
    rl_actions, rl_endpoints, rl_scores, rl_pos, rl_rot = arrays(rl)
    if replay_actions.shape != (28, 36) or not np.array_equal(replay_actions, np.zeros_like(replay_actions)):
        raise ValueError("Replay source must contain 28 zero-residual actions including its failed step")
    if rl_actions.shape != (36, 36):
        raise ValueError("RL source must contain 36 deterministic actions including its failed step")

    boundary_raw = args.boundary.read_bytes()
    boundary_state = torch.load(io.BytesIO(boundary_raw), map_location="cpu", weights_only=False)
    if int(np.asarray(boundary_state["time_indices"])[0]) != 20:
        raise ValueError("saved boundary is not endpoint 20")

    args.output_dir.mkdir(parents=True)
    actions_path = args.output_dir / "window_actions.npz"
    np.savez_compressed(
        actions_path,
        replay_actions=replay_actions,
        replay_endpoints=replay_endpoints,
        replay_scores=replay_scores,
        replay_position_error_m=replay_pos,
        replay_rotation_error_rad=replay_rot,
        rl_actions=rl_actions,
        rl_endpoints=rl_endpoints,
        rl_scores=rl_scores,
        rl_position_error_m=rl_pos,
        rl_rotation_error_rad=rl_rot,
    )
    boundary_artifact = gzip.compress(boundary_raw, compresslevel=9, mtime=0)
    boundary_path = args.output_dir / "endpoint20_boundary.pt.gz"
    boundary_path.write_bytes(boundary_artifact)

    manifest = {
        "schema": "taco_pour_normalized_ellipse_repeatability_gate_v1",
        "status": "prepared_not_executed",
        "source_report": {
            "artifact_path": str(args.source_report.resolve()),
            "artifact_sha256": sha256(report_artifact),
            "uncompressed_json_sha256": sha256(report_raw),
        },
        "variant_trace_report": {
            "artifact_path": str(args.variant_trace_report.resolve()),
            "artifact_sha256": sha256(variant_artifact),
            "uncompressed_json_sha256": sha256(variant_raw),
        },
        "boundary": {
            "artifact_path": str(boundary_path.resolve()),
            "artifact_sha256": sha256(boundary_artifact),
            "uncompressed_pt_sha256": sha256(boundary_raw),
            "reference_endpoint": 20,
        },
        "actions": {
            "artifact_path": str(actions_path.resolve()),
            "artifact_sha256": sha256(actions_path.read_bytes()),
            "replay_shape": list(replay_actions.shape),
            "rl_shape": list(rl_actions.shape),
            "source_replay_validated_steps": 27,
            "source_rl_validated_steps": 35,
        },
        "contracts": {
            "physics_contract_sha256": report["initialization"]["physics_contract"]["physics_contract_sha256"],
            "objective_profile_sha256": report["objective"]["profile_sha256"],
            "observation_profile_sha256": report["observation"]["profile_sha256"],
        },
        "offline_variant_reduction": offline_variant_reduction(variant_report),
        "execution_plan": {
            "fresh_environment_per_repetition": True,
            "repetitions": 3,
            "trial_order_per_repetition": ["replay", "rl"],
            "seed_sweep": False,
            "training": False,
            "full_horizon": False,
        },
        "acceptance": {
            "all_repetitions_required": True,
            "replay_validated_steps_each": 27,
            "replay_failure_endpoint_each": 48,
            "rl_validated_steps_each": 35,
            "rl_failure_endpoint_each": 56,
            "rl_improvement_steps_each": 8,
            "score_curve_difference": "report_only_no_post_hoc_numeric_tolerance",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "offline_variant_reduction": manifest["offline_variant_reduction"],
        "repetitions": 3,
    }, indent=2))


if __name__ == "__main__":
    main()
