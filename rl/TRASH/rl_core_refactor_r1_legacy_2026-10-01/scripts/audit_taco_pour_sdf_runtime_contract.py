#!/usr/bin/env python3
"""Prove that the candidate config and Spider compile the same SDF model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from video_to_spider.rl.physics_contract import (
    build_physics_contract,
    verify_runtime_model,
)


def run(config_path: Path, output: Path, spider_root: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    os.environ["SPIDER_ROOT"] = str(spider_root.resolve())
    sys.path.insert(0, str(spider_root.resolve()))
    from run_mjwp_ppo import _load_ego_config
    from spider.simulators.mjwp import setup_mj_model

    started = time.monotonic()
    contract = build_physics_contract(config_path)
    contract_seconds = time.monotonic() - started

    config = _load_ego_config(str(config_path), "cuda:0")
    started = time.monotonic()
    runtime_model = setup_mj_model(config)
    runtime_compile_seconds = time.monotonic() - started
    verify_runtime_model(runtime_model, contract)

    requested = dict(sorted(config.sdf_octree_depths.items()))
    nodes = {
        name: int(runtime_model.mesh_octnum[runtime_model.mesh(name).id])
        for name in requested
    }
    report = {
        "schema": "taco_pour_sdf_runtime_contract_v1",
        "config": str(config_path.resolve()),
        "spider_root": str(spider_root.resolve()),
        "versions": contract["versions"],
        "sdf_octree_depths": requested,
        "sdf_octree_nodes": nodes,
        "contract_sdf_octree_nodes": contract["runtime"]["sdf_octree_nodes"],
        "physics_contract_sha256": contract["physics_contract_sha256"],
        "compiled_model_sha256": contract["compiled_model_sha256"],
        "contract_build_seconds": contract_seconds,
        "runtime_compile_seconds": runtime_compile_seconds,
        "runtime_matches_physics_contract": True,
        "passed": nodes == contract["runtime"]["sdf_octree_nodes"],
        "formal_config_modified": False,
        "training_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spider-root", type=Path, default=ROOT / "external/spider_compat"
    )
    args = parser.parse_args()
    result = run(args.config, args.output, args.spider_root)
    raise SystemExit(0 if result["passed"] else 1)
