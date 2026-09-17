"""Measure CPU and actual MJWarp allocation needs without adopting a reset.

Use the existing SPIDER Python for --gpu. Snapshot checks do not integrate time;
optional short stepping is only capacity stress, never a task-success rollout.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.collision_audit import validate_qpos
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from build_taco_collision_repair import SCENE, check_unchanged_dynamics


def allocation_usage(data):
    """Counters may exceed allocated arrays. Never clip before detecting overflow."""
    nacon, ncollision = int(data.nacon.numpy()[0]), int(data.ncollision.numpy()[0])
    nefc = data.nefc.numpy().astype(int)
    return dict(contacts_total=nacon, broadphase_pairs_total=ncollision,
                constraints_per_world=nefc.tolist(), contacts_capacity=data.naconmax,
                constraints_capacity=data.njmax,
                overflow=bool(max(nacon, ncollision) > data.naconmax or np.any(nefc > data.njmax)))


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.worlds < 1 or args.nconmax < 1 or args.njmax < 1 or args.steps < 0:
        raise ValueError("positive capacities/world count and nonnegative step count required")
    inputs = [artifact(args.scene), artifact(args.reference)] + scene_mesh_artifacts(args.scene)
    if args.spider_config:
        import yaml
        sys.path.insert(0, str(args.spider_root))
        from spider.simulators.mjwp import setup_mj_model
        config = yaml.safe_load(args.spider_config.read_text())
        inputs += [artifact(args.spider_config), artifact(args.spider_root / "spider/simulators/mjwp.py")]
        model = setup_mj_model(SimpleNamespace(model_path=str(args.scene), sim_dt=config["sim_dt"],
                                               embodiment_type=config["embodiment_type"]))
    else:
        model = mujoco.MjModel.from_xml_path(str(args.scene))
    inputs.append(artifact(SCENE))
    dynamics = check_unchanged_dynamics(mujoco.MjModel.from_xml_path(str(SCENE)), model)
    with np.load(args.reference, allow_pickle=False) as source:
        qpos = validate_qpos(model, source["qpos"], trajectory=True)
        qvel, ctrl = source["qvel"], source["ctrl"]
    if qvel.shape != (len(qpos), model.nv) or ctrl.shape != (len(qpos), model.nu):
        raise ValueError("reference state/control dimensions differ")
    if not np.isfinite(qvel).all() or not np.isfinite(ctrl).all():
        raise ValueError("nonfinite velocity/control")
    data = mujoco.MjData(model)
    cpu = []
    for frame, q in enumerate(qpos):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:] = q, qvel[frame], ctrl[frame]
        mujoco.mj_forward(model, data)
        cpu.append(dict(row=frame, contacts=data.ncon, constraints=data.nefc,
                        warnings=int(sum(w.number for w in data.warning))))
    result = dict(status="capacity_audit_not_reset_or_training_validation",
        unchanged_source_dynamics=dynamics,
        mujoco_version=mujoco.__version__, inputs=inputs, cpu=cpu,
        cpu_max_contacts=max(r["contacts"] for r in cpu),
        cpu_max_constraints=max(r["constraints"] for r in cpu),
        requested=dict(nworld=args.worlds, nconmax=args.nconmax, njmax=args.njmax),
        solver_setup="SPIDER setup_mj_model with selected scene" if args.spider_config else "unchanged XML options",
        solver_options=dict(timestep=model.opt.timestep, iterations=model.opt.iterations,
                            ls_iterations=model.opt.ls_iterations, integrator=int(model.opt.integrator)),
        gpu=None, simulation_steps=0, training_ready=False, source_modified=False,
        code=artifact(Path(__file__)))
    print(f"CPU {mujoco.__version__}: contacts {result['cpu_max_contacts']}, constraints {result['cpu_max_constraints']}", flush=True)
    if args.gpu:
        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        with wp.ScopedDevice(args.device):
            start = time.perf_counter()
            gpu_model = mjw.put_model(model)
            # Allocate empty data, so deliberately small capacity can be tested
            # without put_data rejecting the CPU state's contacts first.
            gpu_data = mjw.make_data(model, nworld=args.worlds, nconmax=args.nconmax, njmax=args.njmax)
            records = []
            # Repeating the same frame in every world exercises synchronized
            # resets, not an average that hides a single difficult frame.
            worst = max(range(len(cpu)), key=lambda i: max(cpu[i]["contacts"] / args.nconmax,
                                                           cpu[i]["constraints"] / args.njmax))
            rows = [worst, *[i for i in range(len(qpos)) if i != worst]]
            for frame in rows:
                for field, value in (("qpos", qpos[frame]), ("qvel", qvel[frame]), ("ctrl", ctrl[frame])):
                    getattr(gpu_data, field).assign(np.repeat(value[None], args.worlds, axis=0).astype(np.float32))
                gpu_data.qacc_warmstart.zero_()
                mjw.forward(gpu_model, gpu_data)
                wp.synchronize()
                usage = allocation_usage(gpu_data)
                usage["row"] = frame
                usage["finite_qacc"] = bool(np.isfinite(gpu_data.qacc.numpy()).all())
                if not usage["overflow"]:
                    count = usage["contacts_total"]
                    ids = gpu_data.contact.worldid.numpy()[:count]
                    usage["contacts_per_world"] = np.bincount(ids, minlength=args.worlds).tolist()
                records.append(usage)
                if usage["overflow"] or not usage["finite_qacc"]:
                    print(f"Stop at row {frame}: {usage}", flush=True)
                    break
            stress = []
            if len(records) == len(qpos) and all(not r["overflow"] and r["finite_qacc"] for r in records):
                stress_rows = {0, worst, max(records, key=lambda r: r["contacts_total"])["row"],
                               max(records, key=lambda r: r["broadphase_pairs_total"])["row"],
                               max(records, key=lambda r: max(r["constraints_per_world"]))["row"]}
                for frame in sorted(stress_rows):
                    for field, value in (("qpos", qpos[frame]), ("qvel", qvel[frame]), ("ctrl", ctrl[frame])):
                        getattr(gpu_data, field).assign(np.repeat(value[None], args.worlds, axis=0).astype(np.float32))
                    gpu_data.qacc_warmstart.zero_()
                    gpu_data.time.zero_()
                    for step in range(args.steps):
                        mjw.step(gpu_model, gpu_data)
                        wp.synchronize()
                        result["simulation_steps"] += args.worlds
                        usage = allocation_usage(gpu_data)
                        usage.update(start_row=frame, step=step + 1,
                                     finite_state=bool(np.isfinite(gpu_data.qpos.numpy()).all()
                                                       and np.isfinite(gpu_data.qvel.numpy()).all()))
                        stress.append(usage)
                        if usage["overflow"] or not usage["finite_state"]:
                            break
                    if stress and (stress[-1]["overflow"] or not stress[-1]["finite_state"]):
                        break
            result["gpu"] = dict(device=args.device, warp_version=wp.__version__,
                mujoco_warp_version=version("mujoco-warp"), snapshots=records, stress=stress,
                snapshots_complete=len(records) == len(qpos),
                all_snapshots_finite=all(r["finite_qacc"] for r in records),
                requested_stress_steps_per_start=args.steps,
                overflow_seen=any(r["overflow"] for r in records + stress),
                elapsed_including_compile_s=time.perf_counter() - start,
                scope="all reference states repeated across worlds; optional short held-control stress, not successful Replay")
    verify_artifacts(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/taco_pour_bimanual_mano_fk_v1/robot_reference.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--spider-config", type=Path)
    parser.add_argument("--spider-root", type=Path, default=Path("/data_all/zzx/egoengine/spider"))
    run(parser.parse_args())
