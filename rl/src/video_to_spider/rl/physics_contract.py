"""Hash and verify the complete physics input used by an accepted reset."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any

import mujoco
import yaml

from egoengine_repro.action.contracts import mujoco_model_signature
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts


SCHEMA = "egoengine_replay_rl_physics_v1"


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _apply_mjwp_options(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    """Mirror the active Spider MJWP hand-model overrides."""
    model.opt.timestep = float(config["sim_dt"])
    if config.get("embodiment_type") not in {"left", "right", "bimanual"}:
        raise ValueError("the Replay→RL physics contract currently supports hand embodiments")
    model.opt.iterations = 20
    model.opt.ls_iterations = 50
    model.opt.o_solref = [0.02, 1.0]
    model.opt.o_solimp = [0.0, 0.95, 0.03, 0.5, 2.0]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def compile_mujoco_model(
    scene: str | Path, sdf_octree_depths: dict[str, int] | None = None,
) -> mujoco.MjModel:
    """Compile the exact model requested by the runtime physics contract."""
    depths = sdf_octree_depths or {}
    if not isinstance(depths, dict):
        raise ValueError("sdf_octree_depths must map mesh names to depths")
    if not depths:
        return mujoco.MjModel.from_xml_path(str(scene))
    if any(
        not isinstance(name, str)
        or not isinstance(depth, int)
        or depth < 1
        for name, depth in depths.items()
    ):
        raise ValueError(
            "sdf_octree_depths must map mesh names to positive integers"
        )
    if not all(
        hasattr(mujoco, name)
        for name in ("MjSpec", "mj_getCache", "mj_clearCache")
    ):
        raise RuntimeError("this MuJoCo version cannot compile configured SDF depths")
    mujoco.mj_clearCache(mujoco.mj_getCache())
    spec = mujoco.MjSpec.from_file(str(scene))
    for mesh_name, depth in sorted(depths.items()):
        mesh = spec.mesh(mesh_name)
        if not hasattr(mesh, "octree_maxdepth"):
            raise RuntimeError("this MuJoCo version lacks mesh.octree_maxdepth")
        mesh.needsdf = True
        mesh.octree_maxdepth = depth
    return spec.compile()


def build_physics_contract(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("simulator") != "mjwp":
        raise ValueError("physics contract requires an explicit MJWP config")
    required = ("model_path", "sim_dt", "ctrl_dt", "ref_dt", "nconmax_per_env", "njmax_per_env")
    if any(key not in config for key in required):
        raise ValueError("formal config is missing a physics-contract field")
    scene = Path(config["model_path"]).resolve(strict=True)
    sdf_depths = config.get("sdf_octree_depths", {}) or {}
    model = compile_mujoco_model(scene, sdf_depths)
    _apply_mjwp_options(model, config)
    assets = scene_mesh_artifacts(scene)
    payload = {
        "schema": SCHEMA,
        "config": artifact(config_path),
        "scene": artifact(scene),
        "assets": assets,
        "runtime": {
            "simulator": "mjwp",
            "sim_dt": float(config["sim_dt"]),
            "ctrl_dt": float(config["ctrl_dt"]),
            "ref_dt": float(config["ref_dt"]),
            "physics_steps_per_control": int(round(float(config["ctrl_dt"]) / float(config["sim_dt"]))),
            "nconmax_per_env": int(config["nconmax_per_env"]),
            "njmax_per_env": int(config["njmax_per_env"]),
            "sdf_octree_depths": dict(sorted(sdf_depths.items())),
            "sdf_octree_nodes": {
                name: int(model.mesh_octnum[model.mesh(name).id])
                for name in sorted(sdf_depths)
            },
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "o_solref": model.opt.o_solref.tolist(),
            "o_solimp": model.opt.o_solimp.tolist(),
        },
        "versions": {
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "mujoco_warp": _version("mujoco-warp"),
            "warp": _version("warp-lang"),
        },
        "compiled_model_sha256": mujoco_model_signature(model, mujoco),
    }
    if payload["runtime"]["physics_steps_per_control"] < 1 or not abs(
        payload["runtime"]["ctrl_dt"]
        - payload["runtime"]["physics_steps_per_control"] * payload["runtime"]["sim_dt"]
    ) < 1e-12:
        raise ValueError("ctrl_dt must be an integer number of physics steps")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "physics_contract_sha256": hashlib.sha256(encoded).hexdigest()}


def verify_runtime_model(model: mujoco.MjModel, contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported physics contract")
    if mujoco_model_signature(model, mujoco) != contract.get("compiled_model_sha256"):
        raise ValueError("runtime compiled physics model differs from release validation")
