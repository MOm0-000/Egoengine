#!/usr/bin/env python3
"""Run Candidate A with a near-minimum safe floor-pair time constant on MJWP."""

from __future__ import annotations

import argparse
from dataclasses import fields
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
    _fresh_mjwp_state,
    native_geometry_gate,
)
from validate_taco_pour_release import _runtime_endpoint_diagnostic


def _build_candidate(source_scene: Path, source_config: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(source_scene)
    count = 0
    for pair in tree.getroot().findall("./contact/pair"):
        if "floor" in (pair.get("geom1"), pair.get("geom2")):
            pair.set("solref", "0.0068 1")
            count += 1
    scene = output / "candidate.xml"
    ET.indent(tree, space="  ")
    tree.write(scene, encoding="unicode")
    config = yaml.safe_load(source_config.read_text())
    config["model_path"] = str(scene.resolve())
    config_path = output / "candidate_ppo_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return scene, config_path, count


def run(source_config: Path, initial_path: Path, output: Path) -> dict:
    source = yaml.safe_load(source_config.read_text())
    scene, config_path, changed_pairs = _build_candidate(
        Path(source["model_path"]), source_config, output
    )
    from spider.config import Config, load_config_yaml, process_config
    from spider.simulators import mjwp
    import warp as wp

    raw = load_config_yaml(str(config_path))
    allowed = {field.name for field in fields(Config)}
    config_values = {
        key: value for key, value in raw.items() if key in allowed
    }
    config_values.update(device="cuda:0", num_samples=1)
    config = process_config(Config(**config_values))
    with np.load(source["data_path"], allow_pickle=False) as data:
        reference = {name: np.asarray(data[name]) for name in data.files}
    zeros_contact = np.zeros((len(reference["qpos"]), 10), dtype=np.float32)
    zeros_position = np.zeros((len(reference["qpos"]), 10, 3), dtype=np.float32)
    ref_data = tuple(torch.as_tensor(value, device=config.device, dtype=torch.float32)
                     for value in (reference["qpos"], reference["qvel"],
                                   reference["ctrl"], zeros_contact, zeros_position))
    env = mjwp.setup_env(config, ref_data)
    with np.load(initial_path, allow_pickle=False) as data:
        initial = {name: np.asarray(data[name]) for name in ("qpos", "qvel", "ctrl")}
    _fresh_mjwp_state(env, initial["qpos"], initial["qvel"], initial["ctrl"])
    control = torch.as_tensor(initial["ctrl"][None], device=config.device, dtype=torch.float32)
    meshes, _ = visual_meshes(scene, env.model_cpu)
    endpoints, trace = [], []
    for step in range(50):
        mjwp.step_env(config, env, control)
        wp.synchronize()
        qpos = mjwp.get_qpos(config, env)[0].detach().cpu().numpy().astype(float)
        qvel = mjwp.get_qvel(config, env)[0].detach().cpu().numpy().astype(float)
        contacts = int(env.data_wp.nacon.numpy()[0])
        broadphase = int(env.data_wp.ncollision.numpy()[0])
        constraints = int(np.max(env.data_wp.nefc.numpy()))
        trace.append({
            "physics_step": step + 1,
            "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
            "contacts": contacts,
            "broadphase": broadphase,
            "constraints": constraints,
            "capacity_overflow": bool(
                max(contacts, broadphase) > env.data_wp.naconmax
                or constraints > env.data_wp.njmax
            ),
        })
        if (step + 1) % 10 == 0:
            native = native_geometry_gate(env.model_cpu, qpos, meshes, 5.0e-5)
            endpoints.append({
                "control_interval": (step + 1) // 10 - 1,
                "native_table_clearance_m": native["table_clearance_m"],
                "native_failures": native["failures"],
                "runtime": _runtime_endpoint_diagnostic(env.model_cpu, qpos),
                "qpos": qpos.tolist(),
            })
    report = {
        "schema": "taco_pour_floor_contact_gpu_v1",
        "scene": str(scene.resolve()),
        "config": str(config_path.resolve()),
        "initial_state": str(initial_path.resolve()),
        "backend": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": __import__("mujoco_warp").__version__,
            "warp": wp.__version__,
        },
        "floor_pairs_changed": changed_pairs,
        "floor_pair_solref": [0.0068, 1.0],
        "minimum_native_table_clearance_m": min(
            min(row["native_table_clearance_m"].values()) for row in endpoints
        ),
        "minimum_runtime_floor_distance_m": min(
            min(row["runtime"]["family_minimum_distance_m"].values())
            for row in endpoints
        ),
        "joint_limit_failure_free": all(
            not row["runtime"]["joint_limit_violations"] for row in endpoints
        ),
        "finite": all(row["finite"] for row in trace),
        "capacity_overflow": any(row["capacity_overflow"] for row in trace),
        "max_contacts": max(row["contacts"] for row in trace),
        "max_broadphase": max(row["broadphase"] for row in trace),
        "max_constraints": max(row["constraints"] for row in trace),
        "endpoints": endpoints,
        "trace": trace,
        "formal_config_modified": False,
        "training_ready": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "minimum_native_table_clearance_m",
        "minimum_runtime_floor_distance_m",
        "joint_limit_failure_free",
        "finite",
        "capacity_overflow",
        "max_contacts",
        "max_constraints",
    )}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_config, args.initial_state, args.output)
