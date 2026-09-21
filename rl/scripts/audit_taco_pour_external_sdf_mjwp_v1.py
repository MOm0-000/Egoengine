#!/usr/bin/env python3
"""Check whether the high-resolution Pour SDF model runs in MuJoCo-Warp."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import time
import traceback

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "runs/taco_pour_external_sdf_combined_v1/candidate.xml"
REFERENCE = ROOT / (
    "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"
)
OUTPUT = ROOT / "runs/taco_pour_external_sdf_mjwp_v1"
OBJECT_DEPTHS = {"right_visual": 8, "left_visual": 9}


def _compile_model() -> tuple[mujoco.MjModel, float]:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for mesh_name, depth in OBJECT_DEPTHS.items():
        mesh = spec.mesh(mesh_name)
        if not hasattr(mesh, "octree_maxdepth"):
            raise RuntimeError("MuJoCo does not expose mesh.octree_maxdepth")
        mesh.needsdf = True
        mesh.octree_maxdepth = depth
    started = time.monotonic()
    model = spec.compile()
    model.opt.timestep = 1.0 / 300.0
    model.opt.iterations = 20
    model.opt.ls_iterations = 50
    model.opt.o_solref = [0.02, 1.0]
    model.opt.o_solimp = [0.0, 0.95, 0.03, 0.5, 2.0]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model, time.monotonic() - started


def _cpu_data(model: mujoco.MjModel) -> mujoco.MjData:
    reference = np.load(REFERENCE, allow_pickle=False)
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = reference["qvel"][0]
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    return data


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "taco_pour_external_sdf_mjwp_audit_v1",
        "scene": str(SCENE),
        "reference": str(REFERENCE),
        "versions": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": getattr(mjwarp, "__version__", "unknown"),
            "warp": wp.__version__,
        },
        "object_octree_depths": OBJECT_DEPTHS,
        "nworld": 1,
        "nconmax_per_env": 128,
        "njmax_per_env": 512,
        "passed": False,
        "training_ready": False,
    }
    try:
        model, compile_seconds = _compile_model()
        data = _cpu_data(model)
        report.update({
            "compile_seconds": compile_seconds,
            "compiled": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "ngeom": int(model.ngeom),
                "npair": int(model.npair),
                "noct": int(model.noct),
            },
            "cpu_initial": {"ncon": int(data.ncon), "nefc": int(data.nefc)},
        })

        wp.init()
        device = "cuda:0"
        wp.set_device(device)
        started = time.monotonic()
        model_warp = mjwarp.put_model(model)
        wp.synchronize_device(device)
        report["put_model_seconds"] = time.monotonic() - started

        started = time.monotonic()
        data_warp = mjwarp.put_data(
            model,
            data,
            nworld=1,
            nconmax=128,
            njmax=512,
        )
        wp.synchronize_device(device)
        report["put_data_seconds"] = time.monotonic() - started

        started = time.monotonic()
        mjwarp.forward(model_warp, data_warp)
        wp.synchronize_device(device)
        report["forward_seconds"] = time.monotonic() - started
        report["gpu_forward"] = {
            "nacon": wp.to_torch(data_warp.nacon).cpu().tolist(),
            "ncollision": wp.to_torch(data_warp.ncollision).cpu().tolist(),
            "nefc": wp.to_torch(data_warp.nefc).cpu().tolist(),
            "overflow": wp.to_torch(data_warp.overflow).cpu().tolist(),
        }

        started = time.monotonic()
        mjwarp.step(model_warp, data_warp)
        wp.synchronize_device(device)
        report["step_seconds"] = time.monotonic() - started
        report["gpu_after_step"] = {
            "nacon": wp.to_torch(data_warp.nacon).cpu().tolist(),
            "ncollision": wp.to_torch(data_warp.ncollision).cpu().tolist(),
            "nefc": wp.to_torch(data_warp.nefc).cpu().tolist(),
            "overflow": wp.to_torch(data_warp.overflow).cpu().tolist(),
        }
        qpos = wp.to_torch(data_warp.qpos).cpu().numpy()
        qvel = wp.to_torch(data_warp.qvel).cpu().numpy()
        report["finite_after_step"] = bool(
            np.isfinite(qpos).all() and np.isfinite(qvel).all()
        )
        report["passed"] = report["finite_after_step"]
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["process_max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["passed"] else 1)
