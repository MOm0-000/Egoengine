"""Validate reset candidates in fresh formal MJWP worlds without reward or cursor motion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPIDER = Path(os.environ.get("SPIDER_ROOT", ROOT / "external/spider_compat"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SPIDER)]

from audit_taco_initialization import visual_meshes
from build_taco_pour_initialization_candidates import (
    _fresh_mjwp_state, _json_default, _load_spider_config, load_protocol, native_geometry_gate,
)
from egoengine_repro.retarget.collision_audit import collision_families, distances
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts
from video_to_spider.rl.physics_contract import build_physics_contract, verify_runtime_model


def _object_hold_constraints(root):
    object_names = set()
    for body in root.findall(".//body"):
        joints = [*body.findall("freejoint"), *body.findall("joint")]
        if not any(joint.tag == "freejoint" or joint.get("type") == "free" for joint in joints):
            continue
        if "object" not in body.get("name", ""):
            continue
        object_names.update(element.get("name") for element in body.iter()
                            if element.get("name"))
    equality = root.find("equality")
    return [] if equality is None else [constraint for constraint in equality
        if any(value in object_names for value in constraint.attrib.values())]


def _rotation_delta(q0, q1):
    dot = float(np.clip(abs(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _object_motion(previous, current, initial):
    rows = []
    for index, side in enumerate(("bowl", "plate")):
        start = 36 + 7 * index
        rows.append({
            "role": side,
            "interval_translation_m": float(np.linalg.norm(current[start:start + 3] - previous[start:start + 3])),
            "interval_rotation_rad": _rotation_delta(previous[start + 3:start + 7], current[start + 3:start + 7]),
            "cumulative_translation_m": float(np.linalg.norm(current[start:start + 3] - initial[start:start + 3])),
            "cumulative_rotation_rad": _rotation_delta(initial[start + 3:start + 7], current[start + 3:start + 7]),
        })
    return rows


def _capacity(env):
    contacts = int(env.data_wp.nacon.numpy()[0])
    broadphase = int(env.data_wp.ncollision.numpy()[0])
    constraints = int(np.max(env.data_wp.nefc.numpy()))
    return contacts, broadphase, constraints


def _native_forbidden(native):
    return {key: value for key, value in native["failures"].items() if value}


def _classify_release_native_failures(model, native, runtime, release_cfg):
    raw = native["failures"]
    native_limit = float(release_cfg.get(
        "max_declared_floor_native_compression_m",
        release_cfg.get("require_no_new_native_material_penetration_over_m", 5.0e-5),
    ))
    runtime_limit = float(release_cfg.get(
        "max_declared_floor_runtime_compression_m", native_limit
    ))
    allowed, forbidden = {"hand_table": [], "object_table": []}, {
        "hand_table": [],
        "object_table": [],
        "hand_object": list(raw["hand_object"]),
        "omitted_nonadjacent": list(raw["omitted_nonadjacent"]),
        "unclassified_omitted_nonadjacent": list(
            raw["unclassified_omitted_nonadjacent"]
        ),
    }
    for family in ("hand_table", "object_table"):
        for geom_name in raw[family]:
            body_name = model.body(int(model.geom_bodyid[model.geom(geom_name).id])).name
            native_distance = float(native["table_clearance_m"][geom_name])
            runtime_distance = runtime["floor_distance_by_body_m"].get(body_name)
            record = {
                "geom": geom_name,
                "body": body_name,
                "native_distance_m": native_distance,
                "runtime_distance_m": runtime_distance,
            }
            if (
                native_distance >= -native_limit
                and runtime_distance is not None
                and runtime_distance >= -runtime_limit
            ):
                allowed[family].append(record)
            else:
                forbidden[family].append(record)
    return (
        {key: value for key, value in forbidden.items() if value},
        {key: value for key, value in allowed.items() if value},
    )


def _runtime_endpoint_diagnostic(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    families = collision_families(model)
    family_minimum = {
        name: float(distances(model, data, families[name]).min())
        for name in ("hand_floor", "tool_floor", "target_floor")
    }
    floor = model.geom("floor").id
    body_minimum = {}
    for pair in (
        families["hand_floor"] + families["tool_floor"] + families["target_floor"]
    ):
        other = pair[1] if pair[0] == floor else pair[0]
        body_name = model.body(int(model.geom_bodyid[other])).name
        value = float(mujoco.mj_geomDistance(
            model, data, pair[0], pair[1], 0.05, None
        ))
        body_minimum[body_name] = min(value, body_minimum.get(body_name, np.inf))
    limited = np.flatnonzero(model.jnt_limited)
    addresses = model.jnt_qposadr[limited]
    margins = np.minimum(
        qpos[addresses] - model.jnt_range[limited, 0],
        model.jnt_range[limited, 1] - qpos[addresses],
    )
    violations = [
        {
            "joint": model.joint(int(joint_id)).name,
            "margin_rad": float(margin),
        }
        for joint_id, margin in zip(limited, margins)
        if margin < -1.0e-6
    ]
    return {
        "family_minimum_distance_m": family_minimum,
        "floor_distance_by_body_m": body_minimum,
        "joint_limit_violations": violations,
    }


def validate_candidate(protocol_path: Path, root: Path, key: str, physics=None):
    from spider.simulators import mjwp
    import warp as wp

    protocol = load_protocol(protocol_path)
    directory = root / key
    report_path = directory / "report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(report_path)
    builder_path = directory / "builder_report.json"
    builder = json.loads(builder_path.read_text())
    initial_path = directory / "initial_state.npz"
    if artifact(initial_path) != builder["initial_state"]:
        raise ValueError("candidate state changed after the t0 audit")
    with np.load(initial_path, allow_pickle=False) as source:
        initial = {name: np.asarray(source[name]) for name in source.files}
    qpos, qvel, ctrl = (initial[name] for name in ("qpos", "qvel", "ctrl"))
    config_path = Path(protocol["formal_inputs"]["formal_simulator_config"])
    if physics is None:
        physics = build_physics_contract(config_path)
    reference_path = Path(protocol["formal_inputs"]["reference"])
    scene = Path(protocol["formal_inputs"]["scene"])
    holds = _object_hold_constraints(ET.parse(scene).getroot())
    release_cfg = protocol["passive_release_validation"]
    t0_passed = bool(builder["t0_legality_gate"]["passed"])
    trace, endpoints, motions, runtime_endpoint_diagnostics = [], [], [], []
    failed_reason = None
    formal_environment_created = False
    if t0_passed and not holds:
        config = _load_spider_config(config_path)
        with np.load(reference_path, allow_pickle=False) as source:
            reference = {name: np.asarray(source[name]) for name in source.files}
        zeros_contact = np.zeros((len(reference["qpos"]), 10), dtype=np.float32)
        zeros_position = np.zeros((len(reference["qpos"]), 10, 3), dtype=np.float32)
        ref_data = tuple(torch.as_tensor(value, device=config.device, dtype=torch.float32) for value in (
            reference["qpos"], reference["qvel"], reference["ctrl"], zeros_contact, zeros_position,
        ))
        env = mjwp.setup_env(config, ref_data)
        formal_environment_created = True
        verify_runtime_model(env.model_cpu, physics)
        _fresh_mjwp_state(env, qpos, qvel, ctrl)
        written = mjwp.get_qpos(config, env)[0].detach().cpu().numpy()
        state_write_max_error = float(np.max(np.abs(written - qpos)))
        constant_ctrl = torch.as_tensor(ctrl[None], device=config.device, dtype=torch.float32)
        previous = written.astype(float)
        for interval in range(int(release_cfg["control_intervals"])):
            for substep in range(int(protocol["timing"]["physics_steps_per_control"])):
                mjwp.step_env(config, env, constant_ctrl)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(config.device)
                q = mjwp.get_qpos(config, env)
                v = mjwp.get_qvel(config, env)
                with wp.ScopedDevice(env.device):
                    acc = wp.to_torch(env.data_wp.qacc)
                    live_ctrl = wp.to_torch(env.data_wp.ctrl)
                contacts, broadphase, constraints = _capacity(env)
                arrays_finite = bool(torch.isfinite(q).all() and torch.isfinite(v).all()
                                     and torch.isfinite(acc).all() and torch.isfinite(live_ctrl).all())
                limited = np.flatnonzero(env.model_cpu.jnt_limited)
                addresses = env.model_cpu.jnt_qposadr[limited]
                q_cpu = q[0].detach().cpu().numpy()
                margins = np.minimum(q_cpu[addresses] - env.model_cpu.jnt_range[limited, 0],
                                     env.model_cpu.jnt_range[limited, 1] - q_cpu[addresses])
                overflow = bool(max(contacts, broadphase) > env.data_wp.naconmax
                                or constraints > env.data_wp.njmax)
                trace.append({"control_interval": interval, "physics_substep": substep,
                              "finite": arrays_finite, "contacts": contacts,
                              "broadphase": broadphase, "constraints": constraints,
                              "capacity_overflow": overflow,
                              "joint_limit_min_margin": float(margins.min())})
                if not arrays_finite or overflow:
                    failed_reason = "nonfinite_state" if not arrays_finite else "capacity_overflow"
                    break
            if failed_reason:
                break
            endpoint = mjwp.get_qpos(config, env)[0].detach().cpu().numpy().astype(float)
            endpoints.append(endpoint)
            runtime_endpoint_diagnostics.append(
                _runtime_endpoint_diagnostic(env.model_cpu, endpoint)
            )
            motions.append(_object_motion(previous, endpoint, qpos))
            previous = endpoint
    else:
        state_write_max_error = None
        failed_reason = "t0_legality_gate_failed" if not t0_passed else "formal_scene_contains_object_hold"

    cpu_model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, _ = visual_meshes(scene, cpu_model)
    reporting_threshold = float(release_cfg.get(
        "native_material_penetration_reporting_threshold_m",
        release_cfg.get("require_no_new_native_material_penetration_over_m", 5.0e-5),
    ))
    endpoint_native = [native_geometry_gate(cpu_model, state, meshes,
        reporting_threshold)
        for state in endpoints]
    native_failures = [_native_forbidden(row) for row in endpoint_native]
    classified = [
        _classify_release_native_failures(
            cpu_model, native, runtime, release_cfg
        )
        for native, runtime in zip(endpoint_native, runtime_endpoint_diagnostics)
    ]
    unmodeled_native_failures = [row[0] for row in classified]
    allowed_modeled_compression = [row[1] for row in classified]
    all_motion = [row for interval in motions for row in interval]
    release_executed = len(endpoints) == int(release_cfg["control_intervals"])
    max_interval_translation = max((row["interval_translation_m"] for row in all_motion), default=None)
    max_interval_rotation = max((row["interval_rotation_rad"] for row in all_motion), default=None)
    max_cumulative_translation = max((row["cumulative_translation_m"] for row in all_motion), default=None)
    max_cumulative_rotation = max((row["cumulative_rotation_rad"] for row in all_motion), default=None)
    checks = {
        "t0_legality_gate": t0_passed,
        "all_physics_steps_executed": len(trace) == int(release_cfg["physics_steps"]),
        "all_states_finite": bool(trace and all(row["finite"] for row in trace)),
        "no_capacity_overflow": bool(trace and not any(row["capacity_overflow"] for row in trace)),
        "object_hold_absent": not holds,
        "joint_limits": (min(row["joint_limit_min_margin"] for row in trace)
                         >= -float(release_cfg.get(
                             "joint_limit_numerical_tolerance_rad", 1.0e-6
                         ))
                         if trace else None),
        "no_forbidden_native_penetration_at_endpoints": (
            not any(unmodeled_native_failures) if release_executed else None),
        "object_translation_per_interval": (
            max_interval_translation <= release_cfg["max_object_translation_per_interval_m"]
            if release_executed else None),
        "object_rotation_per_interval": (
            max_interval_rotation <= release_cfg["max_object_rotation_per_interval_rad"]
            if release_executed else None),
        "object_translation_cumulative": (
            max_cumulative_translation <= release_cfg["max_object_translation_cumulative_m"]
            if release_executed else None),
        "object_rotation_cumulative": (
            max_cumulative_rotation <= release_cfg["max_object_rotation_cumulative_rad"]
            if release_executed else None),
    }
    passed = bool(all(value is True for value in checks.values()))
    release = {
        "passed": passed,
        "checks": checks,
        "failure_reason": failed_reason,
        "requested_control_intervals": int(release_cfg["control_intervals"]),
        "executed_control_intervals": len(endpoints),
        "requested_physics_steps": int(release_cfg["physics_steps"]),
        "executed_physics_steps": len(trace),
        "formal_environment_created": formal_environment_created,
        "backend_setup_state_discarded_before_release": formal_environment_created,
        "control": "constant_initial_ctrl",
        "reference_cursor_advanced": False,
        "reward_or_objective_computed": False,
        "terminal_state_used_as_initial_state": False,
        "object_constraints_active_after_release": (bool(holds) if release_executed else None),
        "physics_contract_sha256": physics["physics_contract_sha256"],
        "fresh_state_write_max_abs_error": state_write_max_error,
        "max_contacts": max((row["contacts"] for row in trace), default=None),
        "max_broadphase": max((row["broadphase"] for row in trace), default=None),
        "max_constraints": max((row["constraints"] for row in trace), default=None),
        "max_object_translation_per_interval_m": max_interval_translation,
        "max_object_rotation_per_interval_rad": max_interval_rotation,
        "max_object_translation_cumulative_m": max_cumulative_translation,
        "max_object_rotation_cumulative_rad": max_cumulative_rotation,
        "object_motion_by_endpoint": motions,
        "native_endpoint_audits": endpoint_native,
        "native_endpoint_failures": native_failures,
        "unmodeled_native_endpoint_failures": unmodeled_native_failures,
        "allowed_modeled_floor_compression": allowed_modeled_compression,
        "runtime_endpoint_diagnostics": runtime_endpoint_diagnostics,
        "endpoint_qpos": [value.tolist() for value in endpoints],
        "trace": trace,
    }
    state_contract = {name: np.asarray(initial[name]).item() for name in (
        "state_contract_version", "reference_index", "hand_qpos_provenance",
        "hand_qvel_provenance", "object_qpos_provenance", "object_qvel_provenance",
        "ctrl_provenance", "first_command_reference_index", "first_command_semantics",
        "object_hold_method", "object_constraints_released", "release_timing",
        "post_release_validation_steps",
    )}
    report = {
        "status": "accepted_reset" if passed else "initialization_candidate_rejected",
        "candidate": key,
        "accepted_for_replay_rl": passed,
        "training_ready": False,
        "scene": artifact(scene), "reference": artifact(reference_path),
        "initial_state": artifact(initial_path), "state_contract": state_contract,
        "physics_contract": physics, "t0_legality_gate": builder["t0_legality_gate"],
        "native_collision_audit": builder["t0_legality_gate"]["native_collision_audit"],
        "first_command_diagnostic": builder["first_command_diagnostic"],
        "release_validation": release,
        "protocol": artifact(protocol_path), "builder_report": artifact(builder_path),
        "validation_code": artifact(Path(__file__)),
    }
    report_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    verify_artifacts([report["scene"], report["reference"], report["initial_state"],
                      report["protocol"], report["builder_report"]])
    return report


def run(protocol_path: Path, root: Path):
    protocol = load_protocol(protocol_path)
    physics = build_physics_contract(
        Path(protocol["formal_inputs"]["formal_simulator_config"])
    )
    reports = {key: validate_candidate(protocol_path, root, key, physics)
               for key in ("candidate_a", "candidate_b")}
    comparison_path = root / "comparison.json"
    if comparison_path.exists() or comparison_path.is_symlink():
        raise FileExistsError(comparison_path)
    comparison = {
        "status": f"{yaml.safe_load(protocol_path.read_text())['protocol_name']}_complete",
        "protocol": artifact(protocol_path),
        "candidates": {key: {
            "accepted_for_replay_rl": report["accepted_for_replay_rl"],
            "report": artifact(root / key / "report.json"),
            "metrics": json.loads((root / key / "builder_report.json").read_text())["metrics"],
            "minimum_declared_clearance_m": report["t0_legality_gate"]["minimum_declared_self_distance_m"],
            "native_hand_table_clearance_m": min(value for name, value in report["native_collision_audit"]["table_clearance_m"].items() if "object_visual" not in name),
            "first_command": report["first_command_diagnostic"],
            "release_object_motion": report["release_validation"]["object_motion_by_endpoint"],
            "release_max_contacts": report["release_validation"]["max_contacts"],
            "release_max_constraints": report["release_validation"]["max_constraints"],
        } for key, report in reports.items()},
        "retained_candidates": [key for key, report in reports.items()
                                if report["accepted_for_replay_rl"]],
        "selection_deferred_until_observation_contract": True,
        "training_ready": False,
    }
    comparison_path.write_text(json.dumps(comparison, indent=2, default=_json_default) + "\n")
    print(json.dumps({"status": comparison["status"],
                      "accepted": {key: value["accepted_for_replay_rl"]
                                   for key, value in comparison["candidates"].items()}}, indent=2))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/taco_pour_initialization_protocol_v1.yaml")
    parser.add_argument("--root", type=Path,
                        default=ROOT / "runs/taco_pour_initialization_protocol_v1")
    args = parser.parse_args()
    run(args.protocol, args.root)
