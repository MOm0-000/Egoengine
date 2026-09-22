"""Guarded full-source Replay→PPO runner; no implicit reset or task promotion."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.physics_contract import build_physics_contract, verify_runtime_model


def _scalar(initial, name, expected_type):
    if name not in initial or np.asarray(initial[name]).shape != ():
        raise ValueError(f"initial state is missing scalar metadata: {name}")
    value = np.asarray(initial[name]).item()
    if expected_type is bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"initial metadata {name} must be boolean")
        return bool(value)
    if expected_type is int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"initial metadata {name} must be integer")
        return int(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"initial metadata {name} must be a nonempty string")
    return value


def _state_contract(initial):
    contract = {
        "state_contract_version": _scalar(initial, "state_contract_version", str),
        "reference_index": _scalar(initial, "reference_index", int),
        "hand_qpos_provenance": _scalar(initial, "hand_qpos_provenance", str),
        "hand_qvel_provenance": _scalar(initial, "hand_qvel_provenance", str),
        "object_qpos_provenance": _scalar(initial, "object_qpos_provenance", str),
        "object_qvel_provenance": _scalar(initial, "object_qvel_provenance", str),
        "ctrl_provenance": _scalar(initial, "ctrl_provenance", str),
        "first_command_reference_index": _scalar(initial, "first_command_reference_index", int),
        "first_command_semantics": _scalar(initial, "first_command_semantics", str),
        "object_hold_method": _scalar(initial, "object_hold_method", str),
        "object_constraints_released": _scalar(initial, "object_constraints_released", bool),
        "release_timing": _scalar(initial, "release_timing", str),
        "post_release_validation_steps": _scalar(initial, "post_release_validation_steps", int),
    }
    if contract["state_contract_version"] != "egoengine_replay_rl_initial_state_v1":
        raise ValueError("unsupported initial-state contract version")
    if contract["object_qpos_provenance"] != "reference_endpoint_0":
        raise ValueError("object qpos must retain reference endpoint 0")
    if contract["first_command_reference_index"] != 1 or contract["first_command_semantics"] != "reference_endpoint_target_plus_residual":
        raise ValueError("first command must explicitly advance endpoint 0 to endpoint 1")
    if not contract["object_constraints_released"] or contract["release_timing"] != "before_accepted_snapshot":
        raise ValueError("accepted state must be captured after releasing object holds")
    if contract["post_release_validation_steps"] < 1:
        raise ValueError("accepted state requires a positive passive post-release validation")
    return contract


def _object_hold_constraints(root):
    """Return equality constraints attached to a free object subtree."""
    object_names = set()
    for body in root.findall(".//body"):
        joints = [*body.findall("freejoint"), *body.findall("joint")]
        if not any(
            joint.tag == "freejoint" or joint.get("type") == "free"
            for joint in joints
        ):
            continue
        if "object" not in body.get("name", "") and not any(
            "object" in joint.get("name", "") for joint in joints
        ):
            continue
        object_names.update(
            element.get("name") for element in body.iter()
            if element.tag in {"body", "joint", "freejoint"} and element.get("name")
        )
    equality = root.find("equality")
    if equality is None:
        return []
    return [
        constraint for constraint in equality
        if any(value in object_names for value in constraint.attrib.values())
    ]


def load_accepted_initialization(report_path, config_path):
    """Read the measured candidate, never accept an old qpos-only diagnostic."""
    report = json.loads(report_path.read_text())
    if not report.get("accepted_for_replay_rl", False):
        raise ValueError("initialization has not passed model and passive-object release checks")
    config = yaml.safe_load(config_path.read_text())
    physics_contract = build_physics_contract(config_path)
    if report.get("physics_contract") != physics_contract:
        raise ValueError("accepted initialization belongs to another physics contract")
    for key, config_key in (("scene", "model_path"), ("reference", "data_path")):
        path = Path(config[config_key]).resolve(strict=True)
        if str(path) != report[key]["path"] or hashlib.sha256(path.read_bytes()).hexdigest() != report[key]["sha256"]:
            raise ValueError(f"accepted initialization belongs to another {key}")
    path = Path(report["initial_state"]["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["initial_state"]["sha256"]:
        raise ValueError("initial state changed after validation")
    with np.load(path, allow_pickle=False) as source:
        initial = dict(source)
    contract = _state_contract(initial)
    if contract["reference_index"] != 0:
        raise ValueError("this runner preserves the full source trajectory from row 0")
    for name, shape in (("qpos", (50,)), ("qvel", (48,)), ("ctrl", (36,))):
        if initial[name].shape != shape or not np.isfinite(initial[name]).all():
            raise ValueError(f"initial {name} must be finite with shape {shape}")
    with np.load(config["data_path"], allow_pickle=False) as reference:
        if not np.array_equal(initial["qpos"][36:], reference["qpos"][0, 36:]):
            raise ValueError("initialization moved the source object's initial pose")
        if contract["hand_qpos_provenance"] == "reference_endpoint_0":
            if not np.array_equal(initial["qpos"][:36], reference["qpos"][0, :36]):
                raise ValueError("hand qpos does not match reference endpoint 0 provenance")
        elif not contract["hand_qpos_provenance"].startswith("local_procedure:"):
            raise ValueError("hand qpos provenance is unsupported")
        if contract["object_qvel_provenance"] == "zero":
            if not np.array_equal(initial["qvel"][36:], np.zeros(12, dtype=initial["qvel"].dtype)):
                raise ValueError("object qvel provenance says zero but values are nonzero")
        elif contract["object_qvel_provenance"] == "reference_finite_difference":
            if not np.array_equal(initial["qvel"][36:], reference["qvel"][0, 36:]):
                raise ValueError("object qvel does not match its reference provenance")
        elif not contract["object_qvel_provenance"].startswith(("measured:", "local_procedure:")):
            raise ValueError("object qvel provenance is unsupported")
        if contract["hand_qvel_provenance"] == "zero":
            if not np.array_equal(initial["qvel"][:36], np.zeros(36, dtype=initial["qvel"].dtype)):
                raise ValueError("hand qvel provenance says zero but values are nonzero")
        elif contract["hand_qvel_provenance"] == "reference_finite_difference":
            if not np.array_equal(initial["qvel"][:36], reference["qvel"][0, :36]):
                raise ValueError("hand qvel does not match its reference provenance")
        elif not contract["hand_qvel_provenance"].startswith("local_procedure:"):
            raise ValueError("hand qvel provenance is unsupported")
        if contract["ctrl_provenance"] == "candidate_hand_qpos":
            if not np.array_equal(initial["ctrl"], initial["qpos"][:36]):
                raise ValueError("ctrl does not match candidate hand qpos provenance")
        elif contract["ctrl_provenance"] == "reference_endpoint_0":
            if not np.array_equal(initial["ctrl"], reference["ctrl"][0]):
                raise ValueError("ctrl does not match reference endpoint 0 provenance")
        elif not contract["ctrl_provenance"].startswith("local_procedure:"):
            raise ValueError("ctrl provenance is unsupported")
    if report.get("state_contract") != contract:
        raise ValueError("initial-state report and artifact contract differ")
    release = report.get("release_validation", {})
    if (not release.get("passed", False)
            or release.get("object_constraints_active_after_release") is not False
            or release.get("requested_control_intervals") != contract["post_release_validation_steps"]
            or release.get("executed_control_intervals") != contract["post_release_validation_steps"]
            or release.get("physics_contract_sha256") != physics_contract["physics_contract_sha256"]):
        raise ValueError("passive post-release validation is missing or inconsistent")
    scene = ET.parse(config["model_path"]).getroot()
    if _object_hold_constraints(scene):
        raise ValueError("formal scene still contains an object hold constraint; release is not proven")
    report = dict(report)
    report["validated_state_contract"] = contract
    report["validated_physics_contract"] = physics_contract
    return initial, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/taco_pour_bimanual_ppo.yaml")
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/replay_rl_protocol.yaml")
    parser.add_argument("--objective-profile", type=Path)
    parser.add_argument("--observation-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--stop-after-first-ppo",
        action="store_true",
        help="Stop after the first PPO-selected chunk for a bounded smoke test.",
    )
    parser.add_argument("--tracking-variant", choices=("tool_only", "tool_and_target"), default="tool_only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.max_chunks < 1:
        raise ValueError("positive epoch/chunk budgets required")
    objective = load_runtime_objective(
        args.protocol, args.objective_profile,
        tracking_variant=args.tracking_variant, require_run_ready=True,
    )
    from video_to_spider.rl.observation_contract import load_runtime_observation
    observation = load_runtime_observation(
        args.protocol, args.observation_profile, require_run_ready=True
    )
    initial, provenance = load_accepted_initialization(args.initialization_report, args.config)

    # Refuse unaccepted initialization before allocating a GPU or loading PPO.
    from run_mjwp_ppo import _load_ego_config, _load_reference, MJWPVectorEnv, MJWPVectorEnvConfig, torch
    from egoengine_repro.action.replay_rl import solve_chunk
    from video_to_spider.rl.replay_rl import MJWPChunkBackend, replay_action, train_chunk_ppo

    config = _load_ego_config(str(args.config), "cuda:0")
    reference = _load_reference(config.data_path, "cuda:0", expected_frequency=30)
    env = MJWPVectorEnv(config, reference, num_envs=1,
        env_config=MJWPVectorEnvConfig(reference_start_index=0, asymmetric_critic=True,
            max_episode_length=len(reference[0]) - 1,
            tracked_object_indices=(0,) if args.tracking_variant == "tool_only" else None,
            object_roles=("tool", "target"), objective=objective,
            observation=observation))
    verify_runtime_model(env.env.model_cpu, provenance["validated_physics_contract"])
    tensors = [torch.as_tensor(initial[k][None], device="cuda:0", dtype=torch.float32)
               for k in ("qpos", "qvel", "ctrl")]
    env._write_state(*tensors, np.array([True]))
    env._last_ctrl = tensors[2].clone()
    env._check_capacity()
    backend = MJWPChunkBackend(env)
    accepted_initial_boundary = backend.snapshot()
    args.output.mkdir(parents=True)
    result = dict(status="running", initialization=provenance,
                  tracking_variant=args.tracking_variant,
                  objective=objective.as_report(),
                  observation=observation.as_report(),
                  config_artifact=dict(path=str(args.config.resolve()),
                      sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()),
                  local_settings=dict(worlds=1, ppo_epochs=args.epochs, ppo_horizon=40,
                      deterministic_mean_validation=True, fresh_policy_per_failed_chunk=True,
                      stop_after_first_ppo=args.stop_after_first_ppo,
                      tracking_boundary=env.tracking_boundary),
                  source_frames=len(reference[0]), control_intervals=len(reference[0]) - 1,
                  chunks=[], task_success=False)
    committed_qpos = [np.asarray(initial["qpos"], dtype=np.float32)]
    committed_qvel = [np.asarray(initial["qvel"], dtype=np.float32)]
    committed_ctrl = [np.asarray(initial["ctrl"], dtype=np.float32)]
    committed_raw_residual = []
    committed_applied_residual = []
    committed_modes = []
    start = 0
    try:
        for _ in range(args.max_chunks):
            trace_start = len(backend.validation_traces)
            chunk = solve_chunk(backend, replay_action,
                lambda current, first, end: train_chunk_ppo(current, first, end,
                    args.output / f"ppo_chunk_{first}", epochs=args.epochs),
                start=start, total_steps=len(reference[0]) - 1)
            chunk_report = asdict(chunk)
            chunk_report["validation_traces"] = backend.validation_traces[trace_start:]
            result["chunks"].append(chunk_report)
            if chunk.mode is None:
                result["status"] = "both_modes_failed_within_local_budget"
                break
            selected_trace = next(
                trace for trace in chunk_report["validation_traces"]
                if trace["mode"] == chunk.mode and trace["feasible"]
            )
            committed_count = chunk.committed_end - chunk.start
            committed_steps = selected_trace["steps"][:committed_count]
            if len(committed_steps) != committed_count:
                raise RuntimeError("selected validation trace does not contain the committed chunk")
            committed_qpos.extend(np.asarray(step["endpoint_qpos"], dtype=np.float32) for step in committed_steps)
            committed_qvel.extend(np.asarray(step["endpoint_qvel"], dtype=np.float32) for step in committed_steps)
            committed_ctrl.extend(np.asarray(step["commanded_ctrl"], dtype=np.float32) for step in committed_steps)
            committed_raw_residual.extend(
                np.asarray(step["raw_residual_action"], dtype=np.float32) for step in committed_steps
            )
            committed_applied_residual.extend(
                np.asarray(step["applied_residual"], dtype=np.float32) for step in committed_steps
            )
            committed_modes.extend([chunk.mode] * committed_count)
            start = chunk.committed_end
            torch.save(backend.snapshot(), args.output / "committed_boundary.pt")
            if args.stop_after_first_ppo and chunk.mode == "rl":
                result["status"] = "short_rl_smoke_passed"
                break
            if start == len(reference[0]) - 1:
                result.update(status="full_horizon_tracking_feasible", task_success=True)
                break
        else:
            result["status"] = "chunk_budget_reached_not_full_task_success"
        if result["task_success"]:
            if len(committed_raw_residual) != len(reference[0]) - 1:
                raise RuntimeError("full-horizon result does not contain one action per transition")
            final_committed_boundary = backend.snapshot()
            backend.restore(accepted_initial_boundary)
            backend.begin_trial("stitched_trajectory", 0, len(committed_raw_residual))
            stitched_steps = 0
            stitched_error = None
            try:
                for index, action in enumerate(committed_raw_residual):
                    if not backend.step(action[None], index):
                        break
                    stitched_steps += 1
            except Exception as error:
                stitched_error = f"{type(error).__name__}: {error}"
                raise
            finally:
                stitched_feasible = stitched_steps == len(committed_raw_residual)
                backend.end_trial(
                    stitched_feasible, stitched_steps, error=stitched_error
                )
                result["stitched_trajectory_validation"] = backend.validation_traces[-1]
                backend.restore(final_committed_boundary)
            if stitched_feasible:
                result["status"] = "full_horizon_tracking_feasible_and_stitched_replay_validated"
            else:
                result["status"] = "stitched_trajectory_replay_failed"
                result["task_success"] = False
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["simulation_control_intervals"] = env.simulation_control_intervals
        result["simulation_physics_steps"] = env.simulation_physics_steps
        result["committed_reference_index"] = int(start)
        trajectory_path = args.output / "optimized_trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            qpos=np.stack(committed_qpos),
            qvel=np.stack(committed_qvel),
            ctrl=np.stack(committed_ctrl),
            raw_residual_action=np.stack(committed_raw_residual) if committed_raw_residual else np.empty((0, 36), np.float32),
            applied_residual=np.stack(committed_applied_residual) if committed_applied_residual else np.empty((0, 36), np.float32),
            mode=np.asarray(committed_modes),
            reference_endpoint=np.arange(len(committed_qpos), dtype=np.int32),
            frequency=np.asarray(30.0, dtype=np.float32),
        )
        result["optimized_trajectory"] = {
            "path": str(trajectory_path.resolve()),
            "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
            "endpoints": len(committed_qpos),
            "transitions": len(committed_modes),
        }
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
