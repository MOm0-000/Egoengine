"""Build the two frozen reset candidates; never accept either without release validation."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import fcl
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPIDER = Path("/data_all/zzx/egoengine/spider")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src"), str(SPIDER)]

from audit_taco_initialization import visual_meshes, world_vertices
from audit_taco_initialization_preflight import triangle_object
from egoengine_repro.retarget.collision_audit import (
    collision_families, distances, explicit_hand_pairs, nonadjacent_hand_pairs,
)
from egoengine_repro.retarget.initial_hand import solve_initial_hands
from egoengine_repro.retarget.mesh_distance import closed_mesh_signed_distance
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def load_protocol(path: Path) -> dict:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("protocol_name") != "taco_pour_initialization_protocol_v1":
        raise ValueError("unsupported initialization protocol")
    if protocol.get("scope") != "reset_only" or protocol.get("training_ready") is not False:
        raise ValueError("initialization protocol must stay reset-only and training-blocked")
    formal = protocol["formal_inputs"]
    scene, reference = Path(formal["scene"]), Path(formal["reference"])
    if _sha(scene) != formal["scene_sha256"] or _sha(reference) != formal["reference_sha256"]:
        raise ValueError("frozen scene/reference hash differs")
    timing = protocol["timing"]
    if timing["physics_steps_per_control"] != round(timing["ctrl_dt_s"] / timing["sim_dt_s"]):
        raise ValueError("protocol timing is inconsistent")
    return protocol


def _world_mesh(model, data, geom, mesh):
    result = mesh.copy()
    body = int(model.geom_bodyid[geom])
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[body].reshape(3, 3)
    transform[:3, 3] = data.xpos[body]
    result.apply_transform(transform)
    return result


def _bounds_overlap(first, second):
    return bool(np.all(first[0] <= second[1]) and np.all(second[0] <= first[1]))


def _containment_direction(source, target, tolerance):
    if target.is_volume:
        points = np.concatenate([source.vertices, source.triangles_center])
        signed = closed_mesh_signed_distance(target, points)
        if not np.isfinite(signed).all():
            raise ValueError("nonfinite native signed distance")
        return {
            "classification": "closed_target_all_surface_vertices_and_centroids",
            "samples": len(points),
            "over_50um": int(np.count_nonzero(signed > tolerance)),
            "sampled_max_inside_m": float(max(0.0, signed.max())),
        }
    bounds_possible = bool(np.all(source.bounds[0] >= target.bounds[0]) and
                           np.all(source.bounds[1] <= target.bounds[1]))
    if not bounds_possible:
        return {"classification": "full_containment_impossible_by_aabb", "over_50um": 0}
    return {"classification": "unclassified_open_target", "over_50um": None}


def _native_pair(first, second, tolerance, *, allow_surface_contact):
    if not _bounds_overlap(first.bounds, second.bounds):
        return {"aabb_overlap": False, "surface_crossing": False,
                "material_interference_over_50um": False, "classified": True}
    result = fcl.CollisionResult()
    fcl.collide(triangle_object(first), triangle_object(second),
                fcl.CollisionRequest(num_max_contacts=1), result)
    crossing = bool(result.is_collision)
    directions = [
        _containment_direction(first, second, tolerance),
        _containment_direction(second, first, tolerance),
    ]
    evidence = any(row["over_50um"] not in (None, 0) for row in directions)
    unclassified = any(row["over_50um"] is None for row in directions)
    return {
        "aabb_overlap": True,
        "surface_crossing": crossing,
        "directions": directions,
        "material_interference_over_50um": bool(evidence or (crossing and not allow_surface_contact)),
        "classified": not unclassified or crossing,
    }


def native_geometry_gate(model, qpos, meshes, tolerance):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    world = {geom: _world_mesh(model, data, geom, mesh) for geom, mesh in meshes.items()}
    names = {geom: model.geom(geom).name for geom in world}
    hand = [geom for geom in world if not names[geom].endswith("object_visual")]
    objects = [model.geom(f"{side}_object_visual").id for side in ("right", "left")]
    floor = model.geom("floor").id
    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    origin = data.geom_xpos[floor]
    table = {names[g]: float(((world[g].vertices - origin) @ normal).min())
             for g in hand + objects}

    hand_object = []
    for a in hand:
        for b in objects:
            evidence = _native_pair(world[a], world[b], tolerance, allow_surface_contact=True)
            if evidence["aabb_overlap"]:
                hand_object.append({"geoms": [names[a], names[b]], **evidence})

    explicit = set(explicit_hand_pairs(model))
    omitted = [pair for pair in nonadjacent_hand_pairs(model)
               if tuple(sorted(pair)) not in explicit
               and model.geom(pair[0]).name.split("_")[2] == model.geom(pair[1]).name.split("_")[2]]
    body_pairs = {}
    for a, b in omitted:
        body_pair = tuple(sorted(map(int, model.geom_bodyid[[a, b]])))
        body_pairs[body_pair] = body_pairs.get(body_pair, 0) + 1
    visual_by_body = {int(model.geom_bodyid[g]): g for g in hand}
    omitted_native = []
    for bodies, shell_pairs in sorted(body_pairs.items()):
        if not all(body in visual_by_body for body in bodies):
            omitted_native.append({"bodies": [model.body(body).name for body in bodies],
                "shell_pairs": shell_pairs, "classified": False,
                "material_interference_over_50um": False,
                "reason": "native_visual_missing"})
            continue
        a, b = (visual_by_body[body] for body in bodies)
        evidence = _native_pair(world[a], world[b], tolerance, allow_surface_contact=False)
        omitted_native.append({"geoms": [names[a], names[b]], "shell_pairs": shell_pairs, **evidence})

    failures = {
        "hand_table": [name for name, value in table.items()
                       if "object_visual" not in name and value < -tolerance],
        "object_table": [name for name, value in table.items()
                         if name.endswith("object_visual") and value < -tolerance],
        "hand_object": [row["geoms"] for row in hand_object
                        if row["material_interference_over_50um"]],
        "omitted_nonadjacent": [row.get("geoms", row.get("bodies")) for row in omitted_native
                                if row["material_interference_over_50um"]],
        "unclassified_omitted_nonadjacent": [row.get("geoms", row.get("bodies"))
                                             for row in omitted_native if not row["classified"]],
    }
    return {
        "tolerance_m": float(tolerance),
        "tolerance_interpretation": "native audit reporting threshold, not runtime collision tolerance",
        "backends": {"mujoco": mujoco.__version__, "python_fcl": fcl.__version__,
                     "containment": "closed-mesh signed distance over all vertices and triangle centroids"},
        "table_clearance_m": table,
        "hand_object_overlapping_pairs": hand_object,
        "omitted_nonadjacent_native_body_pairs": omitted_native,
        "failures": failures,
        "passed": not any(failures.values()),
    }


def _state_contract(candidate_cfg, common):
    return {
        **common,
        "hand_qpos_provenance": candidate_cfg["hand_qpos_provenance"],
        "object_hold_method": candidate_cfg["object_hold_method"],
    }


def _save_state(path, qpos, contract):
    qvel = np.zeros(48, dtype=qpos.dtype)
    ctrl = qpos[:36].copy()
    np.savez_compressed(path, qpos=qpos, qvel=qvel, ctrl=ctrl,
                        **{key: np.asarray(value) for key, value in contract.items()})


def _hand_metrics(model, qpos, reference_qpos, human):
    before, after = mujoco.MjData(model), mujoco.MjData(model)
    before.qpos[:], after.qpos[:] = reference_qpos, qpos
    mujoco.mj_forward(model, before)
    mujoco.mj_forward(model, after)
    result = {}
    for index, side in enumerate(("right", "left")):
        body = model.body(f"{side}_hand_link").id
        target_wrist = human["T_sim_wrist_target"][0, index]
        wrist_error = Rotation.from_matrix(
            target_wrist[:3, :3].T @ after.xmat[body].reshape(3, 3)
        ).magnitude()
        relative = before.xmat[body].reshape(3, 3).T @ after.xmat[body].reshape(3, 3)
        tips = [model.site(f"{side}_{finger}_tip").id
                for finger in ("thumb", "index", "middle", "ring", "pinky")]
        target_tips = human["T_sim_fingertip_target"][0, index, :, :3, 3]
        errors = np.linalg.norm(after.site_xpos[tips] - target_tips, axis=1)
        result[side] = {
            "wrist_translation_from_ref0_m": float(np.linalg.norm(after.xpos[body] - before.xpos[body])),
            "wrist_rotation_from_ref0_rad": float(Rotation.from_matrix(relative).magnitude()),
            "max_finger_joint_delta_rad": float(np.max(np.abs(
                qpos[index * 18 + 6:index * 18 + 18]
                - reference_qpos[index * 18 + 6:index * 18 + 18]))),
            "fingertip_mean_error_m": float(errors.mean()),
            "fingertip_max_error_m": float(errors.max()),
            "wrist_orientation_error_rad": float(wrist_error),
        }
    return result


def _ctrlrange(model, ctrl, tolerance):
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    ranges = model.actuator_ctrlrange
    violation = np.where(limited, np.maximum(ranges[:, 0] - ctrl, ctrl - ranges[:, 1]), 0.0)
    violation = np.maximum(violation, 0.0)
    return bool(violation.max() <= tolerance), float(violation.max())


def _first_command(model, initial_ctrl, next_ctrl, tolerance):
    result = {}
    for index, side in enumerate(("right", "left")):
        delta = next_ctrl[index * 18:(index + 1) * 18] - initial_ctrl[index * 18:(index + 1) * 18]
        result[side] = {"l2_delta": float(np.linalg.norm(delta)),
                        "max_coordinate_delta": float(np.abs(delta).max())}
    valid, violation = _ctrlrange(model, next_ctrl, tolerance)
    result["ctrlrange_numerical_tolerance"] = float(tolerance)
    result["reference_ctrl_1_max_raw_violation"] = violation
    result["reference_ctrl_1_within_ctrlrange"] = valid
    return result


def t0_legality_gate(model, qpos, reference, contract, protocol, meshes):
    gate = protocol["t0_legality_gate"]
    qvel = np.zeros(48, dtype=qpos.dtype)
    ctrl = qpos[:36].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    families = collision_families(model)
    declared = distances(model, data, families["self_explicit"])
    external = {name: distances(model, data, families[name]) for name in
                ("hand_tool", "hand_target", "hand_floor", "tool_target", "tool_floor", "target_floor")}
    guards = []
    for a, b in families["self_explicit"]:
        pair_names = (model.geom(a).name, model.geom(b).name)
        if any("index_root_" in name and "guard" in name for name in pair_names):
            guards.append(float(mujoco.mj_geomDistance(model, data, a, b, 0.05, None)))
    limited = np.flatnonzero(model.jnt_limited)
    addresses = model.jnt_qposadr[limited]
    margins = np.minimum(qpos[addresses] - model.jnt_range[limited, 0],
                         model.jnt_range[limited, 1] - qpos[addresses])
    native = native_geometry_gate(
        model, qpos, meshes,
        float(gate["native_material_penetration_reporting_threshold_m"]),
    )
    initial_ctrl_valid, initial_ctrl_violation = _ctrlrange(
        model, ctrl, float(gate["ctrlrange_numerical_tolerance"])
    )
    first_ctrl_valid, first_ctrl_violation = _ctrlrange(
        model, reference["ctrl"][1], float(gate["ctrlrange_numerical_tolerance"])
    )
    checks = {
        "finite_shapes": qpos.shape == (50,) and qvel.shape == (48,) and ctrl.shape == (36,)
                         and np.isfinite(np.concatenate([qpos, qvel, ctrl])).all(),
        "object_qpos_bit_exact_reference_endpoint_0": bool(np.array_equal(qpos[36:], reference["qpos"][0, 36:])),
        "qvel_exact_zero": bool(np.array_equal(qvel, np.zeros_like(qvel))),
        "ctrl_exact_candidate_hand_qpos": bool(np.array_equal(ctrl, qpos[:36])),
        "initial_ctrl_within_ctrlrange": initial_ctrl_valid,
        "first_reference_command_within_ctrlrange": first_ctrl_valid,
        "joint_limits": bool(margins.min() >= -1e-12),
        "declared_self_pairs": bool(len(declared) == 178 and declared.min() >= gate["declared_pair_min_distance_m"]),
        "bilateral_index_guards": bool(len(guards) == 4 and min(guards) >= gate["bilateral_guard_min_distance_m"]),
        "declared_hand_environment_pairs": bool(all(values.min() >= gate["declared_pair_min_distance_m"]
                                                     for name, values in external.items()
                                                     if name.startswith("hand_"))),
        "object_object_and_table": bool(all(external[name].min() >= gate["declared_pair_min_distance_m"]
                                            for name in ("tool_target", "tool_floor", "target_floor"))),
        "native_geometry": bool(native["passed"]),
    }
    return {
        "passed": bool(all(checks.values())), "checks": checks,
        "declared_self_pair_count": len(declared),
        "minimum_declared_self_distance_m": float(declared.min()),
        "minimum_guard_distance_m": float(min(guards)),
        "minimum_external_distances_m": {key: float(value.min()) for key, value in external.items()},
        "joint_limit_min_margin": float(margins.min()),
        "initial_ctrl_max_raw_range_violation": initial_ctrl_violation,
        "first_reference_command_max_raw_range_violation": first_ctrl_violation,
        "native_collision_audit": native,
        "state_contract": contract,
    }


def _load_spider_config(path, device="cuda:0"):
    from spider.config import Config, load_config_yaml, process_config
    raw = load_config_yaml(str(path))
    allowed = {field.name for field in fields(Config)}
    data = {key: value for key, value in raw.items() if key in allowed}
    for key in ("pair_margin_range", "xy_offset_range"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    data.update(device=device, num_samples=1)
    return process_config(Config(**data))


def _fresh_mjwp_state(env, qpos, qvel, ctrl):
    import warp as wp
    import mujoco_warp as mjwarp
    arrays = {"qpos": qpos[None], "qvel": qvel[None], "ctrl": ctrl[None]}
    with wp.ScopedDevice(env.device):
        for name, value in arrays.items():
            tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32).contiguous()
            wp.copy(getattr(env.data_wp, name), wp.from_torch(tensor))
        for name in ("qacc", "qacc_warmstart", "act", "act_dot", "qfrc_applied", "xfrc_applied"):
            if hasattr(env.data_wp, name):
                value = wp.to_torch(getattr(env.data_wp, name)).clone()
                value.zero_()
                wp.copy(getattr(env.data_wp, name), wp.from_torch(value))
        time = wp.to_torch(env.data_wp.time).clone(); time.zero_()
        wp.copy(env.data_wp.time, wp.from_torch(time))
        mjwarp.forward(env.model_wp, env.data_wp)


def held_object_preroll(seed, reference, protocol):
    from spider.simulators import mjwp
    import warp as wp
    import mujoco_warp

    config_path = Path(protocol["formal_inputs"]["formal_simulator_config"])
    config = _load_spider_config(config_path)
    zeros_contact = np.zeros((len(reference["qpos"]), 10), dtype=np.float32)
    zeros_position = np.zeros((len(reference["qpos"]), 10, 3), dtype=np.float32)
    ref_data = tuple(torch.as_tensor(value, device=config.device, dtype=torch.float32) for value in (
        reference["qpos"], reference["qvel"], reference["ctrl"], zeros_contact, zeros_position,
    ))
    env = mjwp.setup_env(config, ref_data)
    start_qvel = np.zeros(48, dtype=np.float32)
    start_ctrl = seed[:36].astype(np.float32)
    _fresh_mjwp_state(env, seed.astype(np.float32), start_qvel, start_ctrl)
    object_qpos = torch.as_tensor(reference["qpos"][0, 36:], device=config.device, dtype=torch.float32)
    trace = []
    candidate = protocol["candidate_b"]
    substeps = int(protocol["timing"]["physics_steps_per_control"])
    for interval in range(int(candidate["preroll_control_intervals"])):
        alpha = interval / (int(candidate["ramp_intervals"]) - 1) if interval < candidate["ramp_intervals"] else 1.0
        ctrl = (1.0 - alpha) * seed[:36] + alpha * reference["ctrl"][0]
        ctrl_t = torch.as_tensor(ctrl[None], device=config.device, dtype=torch.float32)
        for substep in range(substeps):
            mjwp.step_env(config, env, ctrl_t)
            qpos = mjwp.get_qpos(config, env).clone()
            qvel = mjwp.get_qvel(config, env).clone()
            qpos[:, 36:] = object_qpos
            qvel[:, 36:] = 0.0
            with wp.ScopedDevice(env.device):
                wp.copy(env.data_wp.qpos, wp.from_torch(qpos.contiguous()))
                wp.copy(env.data_wp.qvel, wp.from_torch(qvel.contiguous()))
                if hasattr(env.data_wp, "qacc"):
                    qacc = wp.to_torch(env.data_wp.qacc).clone(); qacc[:, 36:] = 0.0
                    wp.copy(env.data_wp.qacc, wp.from_torch(qacc.contiguous()))
                mujoco_warp.forward(env.model_wp, env.data_wp)
            if torch.cuda.is_available():
                torch.cuda.synchronize(config.device)
            contacts = int(env.data_wp.nacon.numpy()[0])
            broadphase = int(env.data_wp.ncollision.numpy()[0])
            constraints = int(np.max(env.data_wp.nefc.numpy()))
            if max(contacts, broadphase) > env.data_wp.naconmax or constraints > env.data_wp.njmax:
                raise RuntimeError("held-object pre-roll exceeded formal MJWP capacity")
            values = torch.cat([qpos.reshape(-1), qvel.reshape(-1)]).detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("held-object pre-roll produced a nonfinite state")
            trace.append({"interval": interval, "physics_substep": substep,
                          "contacts": contacts, "broadphase": broadphase,
                          "constraints": constraints})
    final = mjwp.get_qpos(config, env)[0].detach().cpu().numpy().astype(float)
    result = seed.copy(); result[:36] = final[:36]
    result[36:] = reference["qpos"][0, 36:]
    return result, {
        "backend": "mujoco_warp", "control_intervals": candidate["preroll_control_intervals"],
        "physics_steps": len(trace), "object_qpos_rewritten_each_substep": True,
        "object_qvel_zeroed_each_substep": True,
        "max_contacts": max(row["contacts"] for row in trace),
        "max_broadphase": max(row["broadphase"] for row in trace),
        "max_constraints": max(row["constraints"] for row in trace),
        "trace": trace,
    }


def run(protocol_path: Path, output: Path):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    protocol = load_protocol(protocol_path)
    formal = protocol["formal_inputs"]
    scene, reference_path = Path(formal["scene"]), Path(formal["reference"])
    human_path = Path(formal["human_reference"])
    inputs = [protocol_path, scene, reference_path, human_path,
              Path(formal["formal_simulator_config"])]
    preserved = [artifact(path) for path in inputs]
    with np.load(reference_path, allow_pickle=False) as source:
        reference = {key: np.asarray(source[key]) for key in source.files}
    with np.load(human_path, allow_pickle=False) as source:
        human = {key: np.asarray(source[key]) for key in source.files}
    model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, mesh_paths = visual_meshes(scene, model)
    solver_cfg = protocol["hand_solver"]
    solver_kwargs = dict(
        source_dt=float(protocol["timing"]["ref_dt_s"]),
        max_iterations=int(solver_cfg["max_iterations"]),
        planning_collision_buffer=float(solver_cfg["planning_collision_buffer_m"]),
        accepted_min_distance=float(solver_cfg["accepted_min_self_collision_distance_m"]),
        solver_primal_tolerance=float(solver_cfg["solver_primal_tolerance"]),
        solver_dual_tolerance=float(solver_cfg["solver_dual_tolerance"]),
        depenetration_step=float(solver_cfg["depenetration_step_m"]),
    )
    common_contract = dict(protocol["state_contract"])
    output.mkdir(parents=True)
    reports = {}
    for key in ("candidate_a", "candidate_b"):
        cfg = protocol[key]
        candidate, solver, trace = solve_initial_hands(
            model, reference["qpos"][0], solver_cfg["velocity_limits"],
            seed_mode=cfg["numerical_seed"],
            backtrack_separation=bool(cfg["separation_backtracking"]),
            **solver_kwargs,
        )
        preroll = None
        construction_complete = solver["retained_feasible_iteration"] is not None
        if key == "candidate_b":
            if construction_complete:
                candidate, preroll = held_object_preroll(candidate, reference, protocol)
            else:
                preroll = {
                    "executed": False,
                    "reason": "collision_aware_depenetration_did_not_produce_a_legal_hand_seed",
                    "physics_steps": 0,
                }
        directory = output / key
        directory.mkdir()
        contract = _state_contract(cfg, common_contract)
        state_path = directory / "initial_state.npz"
        _save_state(state_path, candidate, contract)
        t0 = t0_legality_gate(model, candidate, reference, contract, protocol, meshes)
        metrics = _hand_metrics(model, candidate, reference["qpos"][0], human)
        first = _first_command(
            model, candidate[:36], reference["ctrl"][1],
            float(protocol["t0_legality_gate"]["ctrlrange_numerical_tolerance"]),
        )
        report = {
            "status": ("candidate_built_pending_release_validation" if construction_complete
                       else "candidate_construction_failed_before_release_validation"),
            "candidate": key, "method": cfg["name"],
            "construction_complete": construction_complete,
            "initial_state": artifact(state_path), "state_contract": contract,
            "solver": solver, "held_object_preroll": preroll,
            "t0_legality_gate": t0, "metrics": metrics,
            "first_command_diagnostic": first,
            "solver_trace": {"states": len(trace), "artifact_written": False,
                             "reason": "formal artifact needs only the selected endpoint"},
            "preserved_artifacts": preserved,
            "scene_meshes": scene_mesh_artifacts(scene),
            "source_assets": [artifact(path) for path in mesh_paths],
            "build_code": artifact(Path(__file__)),
            "accepted_for_replay_rl": False,
            "training_ready": False,
        }
        (directory / "builder_report.json").write_text(
            json.dumps(report, indent=2, default=_json_default) + "\n"
        )
        reports[key] = report
    summary = {
        "status": "both_candidates_built_pending_release_validation",
        "protocol": artifact(protocol_path),
        "candidate_a": reports["candidate_a"]["initial_state"],
        "candidate_b": reports["candidate_b"]["initial_state"],
        "t0_gate": {key: reports[key]["t0_legality_gate"]["passed"] for key in reports},
        "comparison": {key: {"metrics": reports[key]["metrics"],
                             "minimum_declared_clearance_m": reports[key]["t0_legality_gate"]["minimum_declared_self_distance_m"],
                             "native_hand_table_clearance_m": min(value for name, value in reports[key]["t0_legality_gate"]["native_collision_audit"]["table_clearance_m"].items() if "object_visual" not in name),
                             "first_command": reports[key]["first_command_diagnostic"]}
                       for key in reports},
        "accepted_for_replay_rl": False, "training_ready": False,
    }
    (output / "build_report.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n"
    )
    verify_artifacts(preserved + scene_mesh_artifacts(scene))
    print(json.dumps({"status": summary["status"], "t0_gate": summary["t0_gate"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/taco_pour_initialization_protocol_v1.yaml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/taco_pour_initialization_protocol_v1")
    args = parser.parse_args()
    run(args.protocol, args.output)
