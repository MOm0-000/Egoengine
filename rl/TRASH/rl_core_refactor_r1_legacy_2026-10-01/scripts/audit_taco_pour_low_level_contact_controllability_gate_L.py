#!/usr/bin/env python3
"""Finite Gate-L audit for low-level Pour contact controllability.

This is deliberately a diagnostic, not a replacement runtime controller.  It
replays two frozen high-level command sequences through (L0) an explicit
URDF-effort-limited joint-coordinate impedance layer and, only after L0 fails,
through (L1) the one predeclared contact-force extension.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_taco_pour_action_feasibility_gate_A1 import (  # noqa: E402
    HORIZON,
    SOURCE,
    load_contract as load_gate_A1_contract,
    read_physics_state,
    rollout_feedback,
)
from audit_taco_pour_action_feasibility_minimum_rho import capture_source57  # noqa: E402
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden,
    load_gzip_torch,
    sha256,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    load_contract as load_tail_contract,
)


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_low_level_contact_controllability_gate_L_v1":
        raise ValueError("unsupported Gate L contract")
    if contract.get("status") != "authorized_finite_read_only_gate_L":
        raise ValueError("Gate L is not authorized")
    runtime = contract["runtime"]
    required_runtime = {
        "backend": "CPU_MuJoCo_Warp",
        "source_state": "exact_tail_source57_snapshot",
        "source_state_provenance": "selected_tail_semantic_oracle_path_through_source56",
        "high_level_command_search_allowed": False,
        "policy_training_allowed": False,
        "optimizer_updates_policy": False,
        "reward_changed": False,
        "objective_changed": False,
        "reference_timing_changed": False,
        "action_frame_changed": False,
        "contact_model_changed": False,
        "chunk_acceptance_allowed": False,
        "chunk_commit_allowed": False,
    }
    stopping = contract["stopping_rule"]
    if (
        contract.get("paper_faithful") is not False
        or runtime != required_runtime
        or contract["free_space_equivalence"]["response_fit_or_gain_search_allowed"] is not False
        or contract["free_space_equivalence"]["actuators"] != "all_36"
        or contract["L0_impedance_equivalent"]["task_gain_sweep_allowed"] is not False
        or contract["L1_force_aware_impedance"]["execute_only_if_L0_has_no_passing_anchor"] is not True
        or contract["L1_force_aware_impedance"]["gain"]["gain_sweep_allowed"] is not False
        or stopping != {
            "Gate_L_closes_after_L0_and_conditional_L1": True,
            "additional_controller_gain_contact_force_axis_frame_or_per_source_sweep_allowed": False,
            "Gate_B_allowed_before_Gate_L_positive": False,
            "Gate_C_allowed_before_Gate_L_positive": False,
            "Gate_D_allowed_before_Gate_L_positive": False,
            "actor_learning_rate_5e_minus_5_allowed": False,
            "learned_gate_allowed": False,
            "reward_change_allowed": False,
            "PPO_retraining_allowed": False,
            "chunk_commit_allowed": False,
        }
    ):
        raise ValueError("Gate L definition changed")
    paths: dict[str, Path] = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def urdf_effort_limits(paths: tuple[Path, Path]) -> dict[str, float]:
    limits: dict[str, float] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is None or "effort" not in limit.attrib:
                continue
            name = joint.attrib["name"]
            effort = float(limit.attrib["effort"])
            if not np.isfinite(effort) or effort <= 0.0:
                raise ValueError(f"invalid URDF effort for {name}")
            if name in limits and limits[name] != effort:
                raise ValueError(f"inconsistent URDF effort for {name}")
            limits[name] = effort
    return limits


def derive_free_space_scene(*, source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        raise ValueError("formal scene has no explicit contact section")
    for child in list(contact):
        contact.remove(child)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def actuator_force_law(formal_model, efforts: dict[str, float]) -> dict[str, np.ndarray]:
    import mujoco

    if formal_model.nu != 36:
        raise ValueError("Gate L requires exactly 36 actuators")
    names = []
    qpos_address = []
    qvel_address = []
    kp = []
    kd = []
    limits = []
    joints = []
    for aid in range(formal_model.nu):
        formal_name = mujoco.mj_id2name(
            formal_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid
        )
        fjoint = int(formal_model.actuator_trnid[aid, 0])
        joint_name = mujoco.mj_id2name(
            formal_model, mujoco.mjtObj.mjOBJ_JOINT, fjoint
        )
        if joint_name not in efforts:
            raise ValueError(f"URDF effort missing for {joint_name}")
        if not np.array_equal(
            np.asarray(formal_model.actuator_gear[aid, :]),
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ):
            raise ValueError(f"Gate L requires unit direct joint gear: {formal_name}")
        gain = float(formal_model.actuator_gainprm[aid, 0])
        bias_position = float(formal_model.actuator_biasprm[aid, 1])
        bias_velocity = float(formal_model.actuator_biasprm[aid, 2])
        if not np.isclose(gain, -bias_position, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"formal actuator is not a position-target affine servo: {formal_name}")
        if gain <= 0.0 or bias_velocity >= 0.0:
            raise ValueError(f"formal actuator has non-impedance gains: {formal_name}")
        names.append(formal_name)
        joints.append(joint_name)
        qpos_address.append(int(formal_model.jnt_qposadr[fjoint]))
        qvel_address.append(int(formal_model.jnt_dofadr[fjoint]))
        kp.append(gain)
        kd.append(-bias_velocity)
        limits.append(efforts[joint_name])
    return {
        "names": np.asarray(names),
        "joints": np.asarray(joints),
        "qpos_address": np.asarray(qpos_address, np.int64),
        "qvel_address": np.asarray(qvel_address, np.int64),
        "kp": np.asarray(kp, np.float64),
        "kd": np.asarray(kd, np.float64),
        "effort_limit": np.asarray(limits, np.float64),
    }
class ImpedanceStepProxy:
    """Translate desired joint targets into bounded generalized effort each MJWP substep."""

    def __init__(self, base, law: dict[str, np.ndarray], *, force_callback=None):
        self.base = base
        self.law = law
        self.force_callback = force_callback
        self.target_override: torch.Tensor | None = None
        self.requested_effort: list[np.ndarray] = []
        self.applied_effort: list[np.ndarray] = []
        self.force_correction: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset_trace(self) -> None:
        self.requested_effort.clear()
        self.applied_effort.clear()
        self.force_correction.clear()

    def step_env(self, config, env, desired_target: torch.Tensor):
        import warp as wp

        target = self.target_override
        if target is None:
            target = desired_target
        if target.ndim == 1:
            target = target.unsqueeze(0)
        target = target.to(torch.float64)
        qpos = self.base.get_qpos(config, env).to(torch.float64)
        qvel = self.base.get_qvel(config, env).to(torch.float64)
        device = target.device
        qpa = torch.as_tensor(self.law["qpos_address"], dtype=torch.long, device=device)
        qva = torch.as_tensor(self.law["qvel_address"], dtype=torch.long, device=device)
        kp = torch.as_tensor(self.law["kp"], dtype=torch.float64, device=device)
        kd = torch.as_tensor(self.law["kd"], dtype=torch.float64, device=device)
        limits = torch.as_tensor(
            self.law["effort_limit"], dtype=torch.float64, device=device
        )
        requested = kp * (target - qpos[:, qpa]) - kd * qvel[:, qva]
        correction = torch.zeros((target.shape[0], 3), dtype=torch.float64, device=device)
        if self.force_callback is not None:
            correction = torch.as_tensor(
                self.force_callback(), dtype=torch.float64, device=device
            ).reshape(target.shape[0], 3)
            requested[:, :3] += correction
        applied = torch.clamp(requested, -limits, limits)
        # Keep the exact formal MJCF model.  Neutralizing each position target
        # at the live qpos leaves only its compiled velocity damping.  Apply
        # the remainder as generalized force so the total hand-DOF force is
        # exactly the bounded impedance request without introducing another
        # actuator model or changing contact physics.
        neutral_target = qpos[:, qpa]
        formal_damping = -kd * qvel[:, qva]
        generalized = applied - formal_damping
        qfrc_applied = wp.to_torch(env.data_wp.qfrc_applied).clone()
        qfrc_applied.zero_()
        qfrc_applied[:, qva] = generalized.to(qfrc_applied.dtype)
        wp.copy(
            env.data_wp.qfrc_applied,
            wp.from_torch(qfrc_applied, dtype=env.data_wp.qfrc_applied.dtype),
        )
        self.requested_effort.append(requested.detach().cpu().numpy())
        self.applied_effort.append(applied.detach().cpu().numpy())
        self.force_correction.append(correction.detach().cpu().numpy())
        self.base.step_env(config, env, neutral_target.to(torch.float32))

    def trace(self) -> dict[str, np.ndarray]:
        shape = (0, 1, 36)
        requested = np.concatenate(self.requested_effort, axis=0) if self.requested_effort else np.zeros(shape[0:1] + shape[2:])
        applied = np.concatenate(self.applied_effort, axis=0) if self.applied_effort else np.zeros(shape[0:1] + shape[2:])
        correction = np.concatenate(self.force_correction, axis=0) if self.force_correction else np.zeros((0, 3))
        return {
            "requested_effort": requested,
            "applied_effort": applied,
            "force_correction": correction,
        }


def build_env(config, reference, objective, observation, residual):
    from run_mjwp_ppo import MJWPVectorEnv, MJWPVectorEnvConfig

    return MJWPVectorEnv(
        config,
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


def fixed_target_replay(*, backend, snapshot: dict, actions: np.ndarray) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    scores, controls, qpos, qvel = [], [], [], []
    for offset, action in enumerate(np.asarray(actions, np.float32)):
        backend.step(action[None], SOURCE + offset)
        scores.append(float(np.asarray(backend.last_info["object_tracking_error"])[0]))
        controls.append(backend.env._last_ctrl[0].detach().cpu().numpy().copy())
        state = read_physics_state(backend.env)
        qpos.append(state[0])
        qvel.append(state[1])
    return {
        "scores": np.asarray(scores, np.float64),
        "controls": np.asarray(controls, np.float32),
        "qpos": np.asarray(qpos, np.float32),
        "qvel": np.asarray(qvel, np.float32),
    }


def pinky_tool_contact(env) -> dict:
    import warp as wp

    contact = env.env.data_wp.contact
    efc = env.env.data_wp.efc
    geom = wp.to_torch(contact.geom).detach().cpu().numpy().astype(np.int64)
    dist = wp.to_torch(contact.dist).detach().cpu().numpy().astype(np.float64)
    world = wp.to_torch(contact.worldid).detach().cpu().numpy().astype(np.int64)
    address = wp.to_torch(contact.efc_address).detach().cpu().numpy().astype(np.int64)
    frame = wp.to_torch(contact.frame).detach().cpu().numpy().astype(np.float64)
    forces = wp.to_torch(efc.force).detach().cpu().numpy().astype(np.float64)
    nacon = int(wp.to_torch(env.env.data_wp.nacon).detach().cpu().numpy()[0])
    finger_map = np.asarray(env.ego_cfg.force_closure_geom_finger_map, np.int64)
    object_map = np.asarray(env.ego_cfg.force_closure_geom_object_group_map, np.int64)
    rows = []
    vector = np.zeros(3, np.float64)
    scalar = 0.0
    for index in range(min(nacon, len(geom))):
        if world[index] != 0 or dist[index] > 0.0:
            continue
        g0, g1 = int(geom[index, 0]), int(geom[index, 1])
        if min(g0, g1) < 0 or max(g0, g1) >= len(finger_map):
            continue
        first = finger_map[g0] == 4 and object_map[g1] == 0
        second = finger_map[g1] == 4 and object_map[g0] == 0
        if not (first or second):
            continue
        valid = address[index][(address[index] >= 0) & (address[index] < forces.shape[1])]
        normal_force = float(np.maximum(forces[0, valid], 0.0).sum()) if len(valid) else 0.0
        if normal_force <= 0.0:
            continue
        normal = np.asarray(frame[index, 0], np.float64)
        if second:
            normal = -normal
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0 or not np.isfinite(norm):
            continue
        normal /= norm
        vector += normal_force * normal
        scalar += normal_force
        rows.append({
            "contact_index": index,
            "geom": [g0, g1],
            "distance_m": float(dist[index]),
            "normal_force_n": normal_force,
            "hand_to_tool_normal_world": normal.tolist(),
        })
    vector_norm = float(np.linalg.norm(vector))
    return {
        "live_contact_count": len(rows),
        "sum_normal_force_n": scalar,
        "force_weighted_normal_vector_world_n": vector.tolist(),
        "aggregate_normal_world": (
            (vector / vector_norm).tolist() if vector_norm > 0.0 else None
        ),
        "rows": rows,
    }


def rollout_impedance(*, backend, proxy: ImpedanceStepProxy, snapshot: dict,
                      actions: np.ndarray, controls: np.ndarray) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    proxy.reset_trace()
    scores, terminated, qpos, qvel, contacts = [], [], [], [], []
    for offset, (action, target) in enumerate(zip(actions, controls, strict=True)):
        proxy.target_override = torch.as_tensor(
            target[None], dtype=torch.float32, device=str(backend.env.ego_cfg.device)
        )
        backend.step(np.asarray(action, np.float32)[None], SOURCE + offset)
        scores.append(float(np.asarray(backend.last_info["object_tracking_error"])[0]))
        terminated.append(bool(np.asarray(backend.last_info["terminated"])[0]))
        state = read_physics_state(backend.env)
        qpos.append(state[0])
        qvel.append(state[1])
        contacts.append(pinky_tool_contact(backend.env))
    proxy.target_override = None
    trace = proxy.trace()
    limits = np.asarray(proxy.law["effort_limit"], np.float64)
    applied = trace["applied_effort"]
    requested = trace["requested_effort"]
    within = bool(np.all(np.abs(applied) <= limits[None, :] + 1.0e-9))
    feasible = bool(
        within
        and np.isfinite(scores).all()
        and all(score < 1.0 for score in scores)
    )
    return {
        "scores": np.asarray(scores, np.float64),
        "terminated": np.asarray(terminated, np.bool_),
        "qpos": np.asarray(qpos, np.float32),
        "qvel": np.asarray(qvel, np.float32),
        "contacts": contacts,
        "requested_effort": requested,
        "applied_effort": applied,
        "force_correction": trace["force_correction"],
        "requested_saturation_count": int(
            np.sum(np.abs(requested) > limits[None, :] + 1.0e-9)
        ),
        "maximum_applied_effort_fraction": float(
            np.max(np.abs(applied) / limits[None, :])
        ),
        "all_applied_effort_within_URDF_limits": within,
        "feasible_three_of_three": feasible,
    }


def free_space_equivalence(*, formal_env, impedance_env, law: dict,
                           substeps: int, amplitude_fraction: float,
                           qpos_threshold: float, qvel_threshold: float) -> tuple[dict, dict]:
    import warp as wp

    nq = formal_env.env.model_cpu.nq
    nv = formal_env.env.model_cpu.nv
    qpos = formal_env.qpos_ref[20].detach().cpu().numpy().astype(np.float32).copy()
    qvel = np.zeros(nv, np.float32)
    model = formal_env.env.model_cpu
    for aid, address in enumerate(law["qpos_address"]):
        joint = int(model.actuator_trnid[aid, 0])
        low, high = np.asarray(model.jnt_range[joint], np.float64)
        if not bool(model.jnt_limited[joint]) or not np.isfinite((low, high)).all():
            raise ValueError("free-space identification requires finite actuated joint ranges")
        qpos[address] = np.float32(0.5 * (low + high))
    controls = qpos[law["qpos_address"]].copy()
    reset = np.asarray([True])
    formal_env.time_indices[:] = 20
    formal_env._write_state(
        torch.as_tensor(qpos[None]), torch.as_tensor(qvel[None]),
        torch.as_tensor(controls[None]), reset,
    )
    formal_snapshot = formal_env.get_env_state()
    impedance_env.set_env_state(formal_snapshot)
    impedance_snapshot = impedance_env.get_env_state()
    proxy = ImpedanceStepProxy(impedance_env._mjwp, law)
    rows = []
    formal_qpos_all, impedance_qpos_all = [], []
    formal_qvel_all, impedance_qvel_all = [], []
    saturation_total = 0
    for aid in range(36):
        amplitude = amplitude_fraction * law["effort_limit"][aid] / law["kp"][aid]
        for direction in (-1.0, 1.0):
            target = controls.astype(np.float64).copy()
            target[aid] += direction * amplitude
            formal_env.set_env_state(formal_snapshot)
            formal_trace_qpos, formal_trace_qvel = [], []
            for _ in range(substeps):
                formal_env._mjwp.step_env(
                    formal_env.ego_cfg, formal_env.env,
                    torch.as_tensor(target[None], dtype=torch.float32),
                )
                formal_trace_qpos.append(
                    formal_env._mjwp.get_qpos(formal_env.ego_cfg, formal_env.env)[0].detach().cpu().numpy()
                )
                formal_trace_qvel.append(
                    formal_env._mjwp.get_qvel(formal_env.ego_cfg, formal_env.env)[0].detach().cpu().numpy()
                )
            impedance_env.set_env_state(impedance_snapshot)
            proxy.reset_trace()
            proxy.target_override = torch.as_tensor(target[None], dtype=torch.float32)
            impedance_trace_qpos, impedance_trace_qvel = [], []
            for _ in range(substeps):
                proxy.step_env(
                    impedance_env.ego_cfg, impedance_env.env,
                    torch.as_tensor(target[None], dtype=torch.float32),
                )
                impedance_trace_qpos.append(
                    impedance_env._mjwp.get_qpos(impedance_env.ego_cfg, impedance_env.env)[0].detach().cpu().numpy()
                )
                impedance_trace_qvel.append(
                    impedance_env._mjwp.get_qvel(impedance_env.ego_cfg, impedance_env.env)[0].detach().cpu().numpy()
                )
            proxy.target_override = None
            fqp = np.asarray(formal_trace_qpos)
            tq = np.asarray(impedance_trace_qpos)
            fqv = np.asarray(formal_trace_qvel)
            tv = np.asarray(impedance_trace_qvel)
            trace = proxy.trace()
            saturated = int(np.sum(
                np.abs(trace["requested_effort"])
                > law["effort_limit"][None, :] + 1.0e-9
            ))
            saturation_total += saturated
            rows.append({
                "actuator_index": aid,
                "actuator_name": str(law["names"][aid]),
                "direction": int(direction),
                "step_amplitude": float(amplitude),
                "max_qpos_difference": float(np.max(np.abs(fqp - tq))),
                "max_qvel_difference": float(np.max(np.abs(fqv - tv))),
                "effort_saturation_count": saturated,
            })
            formal_qpos_all.append(fqp)
            impedance_qpos_all.append(tq)
            formal_qvel_all.append(fqv)
            impedance_qvel_all.append(tv)
    max_qpos = max(row["max_qpos_difference"] for row in rows)
    max_qvel = max(row["max_qvel_difference"] for row in rows)
    passed = bool(
        saturation_total == 0
        and max_qpos <= qpos_threshold
        and max_qvel <= qvel_threshold
    )
    return {
        "passed": passed,
        "rollout_count": len(rows),
        "physics_substeps_per_rollout": substeps,
        "maximum_qpos_difference": max_qpos,
        "maximum_qvel_difference": max_qvel,
        "effort_saturation_count": saturation_total,
        "rows": rows,
    }, {
        "formal_qpos": np.asarray(formal_qpos_all, np.float32),
        "impedance_qpos": np.asarray(impedance_qpos_all, np.float32),
        "formal_qvel": np.asarray(formal_qvel_all, np.float32),
        "impedance_qvel": np.asarray(impedance_qvel_all, np.float32),
    }


def force_probe(*, env, backend, proxy: ImpedanceStepProxy, snapshot: dict,
                target: np.ndarray, epsilon: float, minimum_sensitivity: float) -> dict:
    backend.restore(snapshot)
    backend.verify_restored_snapshot(snapshot)
    initial = pinky_tool_contact(env)
    if initial["aggregate_normal_world"] is None or initial["sum_normal_force_n"] <= 0.0:
        return {"identifiable": False, "reason": "source57_has_no_live_right_pinky_tool_contact"}
    normal = np.asarray(initial["aggregate_normal_world"], np.float64)
    forces = []
    details = []
    for direction in (-1.0, 1.0):
        backend.restore(snapshot)
        backend.verify_restored_snapshot(snapshot)
        perturbed = np.asarray(target, np.float64).copy()
        perturbed[:3] += direction * epsilon * normal
        proxy.reset_trace()
        proxy.target_override = torch.as_tensor(perturbed[None], dtype=torch.float32)
        proxy.step_env(
            env.ego_cfg, env.env,
            torch.as_tensor(perturbed[None], dtype=torch.float32),
        )
        contact = pinky_tool_contact(env)
        forces.append(contact["sum_normal_force_n"])
        details.append({
            "direction": int(direction),
            "normal_force_n": contact["sum_normal_force_n"],
            "contact": contact,
        })
    proxy.target_override = None
    signed = (forces[1] - forces[0]) / (2.0 * epsilon)
    identifiable = bool(np.isfinite(signed) and abs(signed) >= minimum_sensitivity)
    return {
        "identifiable": identifiable,
        "reason": None if identifiable else "centered_contact_stiffness_below_fail_closed_minimum",
        "source_contact": initial,
        "epsilon_m": epsilon,
        "probe_rows": details,
        "signed_dFn_dxn_N_per_m": float(signed),
        "abs_dFn_dxn_N_per_m": float(abs(signed)),
        "pressing_direction_sign": float(np.sign(signed)) if identifiable else None,
    }


def serialize_rollout(row: dict) -> dict:
    return {
        "scores": row["scores"].tolist(),
        "terminated": row["terminated"].tolist(),
        "contacts": row["contacts"],
        "requested_saturation_count": row["requested_saturation_count"],
        "maximum_applied_effort_fraction": row["maximum_applied_effort_fraction"],
        "all_applied_effort_within_URDF_limits": row["all_applied_effort_within_URDF_limits"],
        "feasible_three_of_three": row["feasible_three_of_three"],
    }


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def write_summary(path: Path, report: dict) -> None:
    lines = [
        "# Gate L: low-level contact controllability",
        "",
        f"- free-space impedance equivalence passed: {report['free_space_equivalence']['passed']}",
        f"- L0 any 3/3 feasible anchor: {report['L0']['any_anchor_feasible']}",
        f"- L1 executed: {report['L1']['executed']}",
        f"- L1 any 3/3 feasible anchor: {report['L1']['any_anchor_feasible']}",
        f"- Gate L classification: {report['decision']['classification']}",
        f"- next blocker: {report['decision']['next_blocker']}",
        "",
        "Gate L is closed by this finite L0/L1 contract. No policy training,",
        "task-level gain sweep, chunk acceptance, or chunk commit occurred.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/taco_pour_low_level_contact_controllability_gate_L_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/taco_pour_low_level_contact_controllability_gate_L_v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_A1_report = json.loads(paths["gate_A1_report"].read_text())
    if (
        gate_A1_report["decision"]["Gate_A_closed"] is not True
        or gate_A1_report["decision"]["next_blocker"]
        != "low_level_impedance_or_force_aware_contact_controllability_required"
    ):
        raise ValueError("Gate L requires the closed negative Gate A1 result")
    gate_A1_contract, gate_A1_paths, _ = load_gate_A1_contract(paths["gate_A1_contract"])
    _, parent_paths, _ = load_tail_contract(gate_A1_paths["tail_contract"])
    tail_report = json.loads(paths["tail_report"].read_text())

    from run_mjwp_ppo import (
        _build_network_config,
        _build_ppo_config,
        _load_ego_config,
        _load_reference,
    )
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        StateFeasibleTruncatedGaussianPpoAgent,
        load_truncated_gaussian_profile,
    )

    objective = load_runtime_objective(
        parent_paths["protocol"], parent_paths["objective_profile"],
        tracking_variant="tool_only", require_run_ready=False,
    )
    observation = load_runtime_observation(
        parent_paths["protocol"], parent_paths["observation_profile"],
        require_run_ready=False,
    )
    residual, _ = load_residual_action_profile(parent_paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(parent_paths["distribution_profile"])
    config = _load_ego_config(str(paths["formal_simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    reference_qvel = reference[1].detach().cpu().numpy().astype(np.float64)

    args.output.mkdir(parents=True)
    shutil.copy2(args.contract, args.output / "contract.yaml")
    efforts = urdf_effort_limits((paths["right_XHand_URDF"], paths["left_XHand_URDF"]))
    free_space_scene = args.output / contract["artifacts"]["free_space_scene"]
    derive_free_space_scene(
        source=paths["formal_scene"], destination=free_space_scene,
    )

    formal_env = build_env(config, reference, objective, observation, residual)
    formal_backend = MJWPChunkBackend(formal_env)
    checkpoint = load_gzip_torch(parent_paths["checkpoint"])
    boundary = load_gzip_torch(parent_paths["boundary"])
    temporary = tempfile.TemporaryDirectory(prefix=".gate_L_", dir=ROOT / "runs")
    try:
        ppo_config = replace(
            _build_ppo_config(
                num_envs=1, horizon_length=40, seq_length=4, max_epochs=8,
                learning_rate=1e-4, device="cpu", asymmetric_critic=None,
            ),
            clip_actions=False,
        )
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=Path(temporary.name) / "policy",
            ppo_config=ppo_config,
            network_config=_build_network_config(4),
            env=formal_env,
            distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"])
        policy.set_eval()
        source = capture_source57(
            backend=formal_backend,
            policy=policy,
            boundary=boundary,
            parent_paths=parent_paths,
            tail_report=tail_report,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            objective=objective,
        )
        source_hidden = clone_hidden(policy.rnn_states)
        formal_actor = rollout_feedback(
            backend=formal_backend,
            policy=policy,
            snapshot=source["snapshot"],
            hidden=source_hidden,
            kp=0.0,
            kv=0.0,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
        )
        formal_anchor = fixed_target_replay(
            backend=formal_backend,
            snapshot=source["snapshot"],
            actions=formal_actor["actions"],
        )
        if not np.array_equal(formal_anchor["scores"].astype(np.float32), formal_actor["scores"].astype(np.float32)):
            raise RuntimeError("formal PPO fixed-action replay changed its scores")
        feedback_arrays = np.load(paths["gate_A1_feedback_best"])
        feedback_actions = np.asarray(feedback_arrays["action"], np.float32)
        feedback_anchor = fixed_target_replay(
            backend=formal_backend,
            snapshot=source["snapshot"],
            actions=feedback_actions,
        )
        if not np.array_equal(
            feedback_anchor["scores"].astype(np.float32),
            np.asarray(feedback_arrays["score"], np.float32),
        ):
            raise RuntimeError("Gate A1 feedback fixed-action replay changed its scores")
        policy.writer.close()
    finally:
        temporary.cleanup()

    impedance_env = build_env(config, reference, objective, observation, residual)
    impedance_backend = MJWPChunkBackend(impedance_env)
    law = actuator_force_law(formal_env.env.model_cpu, efforts)
    impedance_backend.restore(source["snapshot"])
    impedance_backend.verify_restored_snapshot(source["snapshot"])

    free_config = deepcopy(config)
    free_config.model_path = str(free_space_scene.resolve())
    free_position_env = build_env(
        free_config, reference, objective, observation, residual
    )
    free_impedance_env = build_env(
        free_config, reference, objective, observation, residual
    )
    fs = contract["free_space_equivalence"]
    free_report, free_arrays = free_space_equivalence(
        formal_env=free_position_env,
        impedance_env=free_impedance_env,
        law=law,
        substeps=int(fs["physics_substeps"]),
        amplitude_fraction=0.1,
        qpos_threshold=float(fs["acceptance"]["max_qpos_difference"]),
        qvel_threshold=float(fs["acceptance"]["max_qvel_difference"]),
    )
    if not free_report["passed"]:
        raise RuntimeError(f"free-space impedance equivalence gate failed: {free_report}")

    anchors = {
        "formal_frozen_PPO": {
            "actions": np.asarray(formal_actor["actions"], np.float32),
            "controls": formal_anchor["controls"],
            "formal_scores": formal_anchor["scores"],
        },
        "gate_A1_best_position_velocity_feedback": {
            "actions": feedback_actions,
            "controls": feedback_anchor["controls"],
            "formal_scores": feedback_anchor["scores"],
        },
    }
    l0_rows = {}
    l0_raw = {}
    for name, anchor in anchors.items():
        proxy = ImpedanceStepProxy(impedance_env._mjwp, law)
        impedance_env._mjwp = proxy
        try:
            first = rollout_impedance(
                backend=impedance_backend, proxy=proxy, snapshot=source["snapshot"],
                actions=anchor["actions"], controls=anchor["controls"],
            )
            second = rollout_impedance(
                backend=impedance_backend, proxy=proxy, snapshot=source["snapshot"],
                actions=anchor["actions"], controls=anchor["controls"],
            )
        finally:
            impedance_env._mjwp = proxy.base
        if not (
            np.array_equal(first["scores"], second["scores"])
            and np.array_equal(first["qpos"], second["qpos"])
            and np.array_equal(first["qvel"], second["qvel"])
            and np.array_equal(first["applied_effort"], second["applied_effort"])
        ):
            raise RuntimeError(f"L0 CPU rollout is not repeatable for {name}")
        l0_rows[name] = serialize_rollout(first) | {
            "formal_position_actuator_scores": anchor["formal_scores"].tolist(),
            "bitwise_repeatable_on_CPU": True,
        }
        l0_raw[name] = first
    l0_pass = any(row["feasible_three_of_three"] for row in l0_rows.values())

    l1_rows: dict[str, dict] = {}
    l1_raw: dict[str, dict] = {}
    probe = None
    if not l0_pass:
        probe_proxy = ImpedanceStepProxy(impedance_env._mjwp, law)
        probe = force_probe(
            env=impedance_env,
            backend=impedance_backend,
            proxy=probe_proxy,
            snapshot=source["snapshot"],
            target=anchors["formal_frozen_PPO"]["controls"][0],
            epsilon=float(contract["L1_force_aware_impedance"]["probe"]["displacement_m"]),
            minimum_sensitivity=float(
                contract["L1_force_aware_impedance"]["probe"][
                    "minimum_identifiable_abs_dFn_dxn_N_per_m"
                ]
            ),
        )
        if probe["identifiable"]:
            fstar = float(probe["source_contact"]["sum_normal_force_n"])
            sign = float(probe["pressing_direction_sign"])
            sensitivity = float(probe["abs_dFn_dxn_N_per_m"])
            kp_translation = float(np.mean(law["kp"][:3]))
            if not np.allclose(law["kp"][:3], kp_translation, rtol=0.0, atol=0.0):
                raise RuntimeError("right-wrist translation kp must be shared")
            kf = kp_translation / sensitivity

            def correction():
                contact = pinky_tool_contact(impedance_env)
                normal = contact["aggregate_normal_world"]
                if normal is None:
                    return np.zeros((1, 3), np.float64)
                error = fstar - float(contact["sum_normal_force_n"])
                return (kf * error * sign * np.asarray(normal, np.float64))[None]

            probe["force_target_n"] = fstar
            probe["kf_dimensionless"] = kf
            for name, anchor in anchors.items():
                proxy = ImpedanceStepProxy(
                    impedance_env._mjwp, law, force_callback=correction
                )
                impedance_env._mjwp = proxy
                try:
                    first = rollout_impedance(
                        backend=impedance_backend, proxy=proxy,
                        snapshot=source["snapshot"], actions=anchor["actions"],
                        controls=anchor["controls"],
                    )
                    second = rollout_impedance(
                        backend=impedance_backend, proxy=proxy,
                        snapshot=source["snapshot"], actions=anchor["actions"],
                        controls=anchor["controls"],
                    )
                finally:
                    impedance_env._mjwp = proxy.base
                if not (
                    np.array_equal(first["scores"], second["scores"])
                    and np.array_equal(first["qpos"], second["qpos"])
                    and np.array_equal(first["qvel"], second["qvel"])
                    and np.array_equal(first["applied_effort"], second["applied_effort"])
                ):
                    raise RuntimeError(f"L1 CPU rollout is not repeatable for {name}")
                l1_rows[name] = serialize_rollout(first) | {
                    "formal_position_actuator_scores": anchor["formal_scores"].tolist(),
                    "bitwise_repeatable_on_CPU": True,
                }
                l1_raw[name] = first
    l1_pass = any(row["feasible_three_of_three"] for row in l1_rows.values())

    if l0_pass:
        classification = "L0_impedance_equivalent_controller_demonstrates_three_step_feasibility"
        next_blocker = "separate_minimal_Gate_A_revalidation_under_frozen_impedance_required"
        gate_a_revalidation = True
    elif l1_pass:
        classification = "L1_force_aware_impedance_demonstrates_three_step_feasibility"
        next_blocker = "explicit_contact_force_layer_architecture_contract_required"
        gate_a_revalidation = False
    else:
        classification = (
            "low_level_contact_controllability_not_demonstrated_under_the_two_"
            "predeclared_controllers"
        )
        next_blocker = "physics_reference_and_XHand_contact_actuator_model_review_required"
        gate_a_revalidation = False

    commands_path = args.output / contract["artifacts"]["fixed_high_level_commands"]
    np.savez_compressed(
        commands_path,
        formal_PPO_action=anchors["formal_frozen_PPO"]["actions"],
        formal_PPO_control=anchors["formal_frozen_PPO"]["controls"],
        formal_PPO_formal_score=anchors["formal_frozen_PPO"]["formal_scores"].astype(np.float32),
        feedback_action=anchors["gate_A1_best_position_velocity_feedback"]["actions"],
        feedback_control=anchors["gate_A1_best_position_velocity_feedback"]["controls"],
        feedback_formal_score=anchors["gate_A1_best_position_velocity_feedback"]["formal_scores"].astype(np.float32),
    )
    free_path = args.output / contract["artifacts"]["free_space_equivalence"]
    np.savez_compressed(free_path, **free_arrays)
    rollout_path = args.output / contract["artifacts"]["rollout_arrays"]
    rollout_payload = {}
    for stage, rows in (("L0", l0_raw), ("L1", l1_raw)):
        for name, row in rows.items():
            prefix = f"{stage}_{name}"
            for field in (
                "scores", "terminated", "qpos", "qvel", "requested_effort",
                "applied_effort", "force_correction",
            ):
                rollout_payload[f"{prefix}_{field}"] = np.asarray(row[field])
    np.savez_compressed(rollout_path, **rollout_payload)

    report = {
        "schema": contract["schema"],
        "status": "completed_finite_read_only_gate_L",
        "paper_faithful": False,
        "classification": contract["classification"],
        "training_executed": False,
        "policy_optimizer_steps": 0,
        "chunk_commit_written": False,
        "contract": contract_artifact,
        "source57": {
            "source_state": source["source_state"],
            "snapshot_field_count": len(source["snapshot"]),
            "same_model_snapshot_fields_bitwise_restored": True,
        },
        "diagnostic_model": {
            "formal_scene": artifact(paths["formal_scene"]),
            "free_space_scene": artifact(free_space_scene),
            "formal_scene_model_is_unchanged_during_L0_and_L1": True,
            "actuator_names": law["names"].tolist(),
            "joint_names": law["joints"].tolist(),
            "kp": law["kp"].tolist(),
            "kd": law["kd"].tolist(),
            "URDF_effort_limits": law["effort_limit"].tolist(),
            "formal_actuators_are_already_affine_PD_but_unlimited": True,
            "diagnostic_change": (
                "explicit_qfrc_applied_joint_impedance_with_formal_actuators_"
                "neutralized_at_live_qpos_and_URDF_effort_limits"
            ),
        },
        "free_space_equivalence": free_report,
        "fixed_high_level_anchors": {
            name: {
                "actions_shape": list(anchor["actions"].shape),
                "controls_shape": list(anchor["controls"].shape),
                "formal_position_actuator_scores": anchor["formal_scores"].tolist(),
            }
            for name, anchor in anchors.items()
        },
        "L0": {
            "executed": True,
            "controller": "URDF_effort_limited_joint_coordinate_impedance",
            "task_gain_search_executed": False,
            "anchors": l0_rows,
            "any_anchor_feasible": l0_pass,
        },
        "L1": {
            "executed": not l0_pass,
            "skipped_reason": "L0_passed" if l0_pass else None,
            "probe": probe,
            "anchors": l1_rows,
            "any_anchor_feasible": l1_pass,
            "task_gain_or_force_target_search_executed": False,
        },
        "decision": {
            "Gate_L_closed": True,
            "classification": classification,
            "finite_negative_result_is_mathematical_infeasibility_proof": False,
            "separate_minimal_Gate_A_revalidation_authorized": gate_a_revalidation,
            "Gate_B_allowed_now": False,
            "Gate_C_allowed_now": False,
            "Gate_D_or_PPO_retraining_allowed_now": False,
            "RL_architecture_gates_remain_closed": True,
            "actor_LR_5e_minus_5_unblocked": False,
            "learned_gate_authorized": False,
            "reward_change_authorized": False,
            "chunk_acceptance_or_commit_authorized": False,
            "next_blocker": next_blocker,
        },
        "stopping_rule": {
            "only_L0_and_conditional_L1_executed": True,
            "additional_source57_controller_gain_contact_force_axis_frame_or_per_source_sweep_authorized": False,
        },
    }
    summary_path = args.output / contract["artifacts"]["summary"]
    report_path = args.output / contract["artifacts"]["report"]
    write_summary(summary_path, report)
    report["artifacts"] = {
        "contract": artifact(args.output / "contract.yaml"),
        "free_space_scene": artifact(free_space_scene),
        "fixed_high_level_commands": artifact(commands_path),
        "free_space_equivalence": artifact(free_path),
        "rollout_arrays": artifact(rollout_path),
        "summary": artifact(summary_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "free_space_passed": free_report["passed"],
        "L0_any_anchor_feasible": l0_pass,
        "L1_executed": not l0_pass,
        "L1_any_anchor_feasible": l1_pass,
        "classification": classification,
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
