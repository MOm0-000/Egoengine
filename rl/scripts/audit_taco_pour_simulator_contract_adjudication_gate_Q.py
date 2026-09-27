#!/usr/bin/env python3
"""Final finite Gate Q for the TACO Pour simulator contract.

Gate Q is deliberately an adjudication, not another tuning surface.  It
measures the force demand of the unchanged formal actuator (Q1), compares the
three predeclared collision representations in native MuJoCo and CPU MJWP
(Q2), and re-runs the reference-wrench feasibility test with a clustered,
bilateral contact graph and compiled MuJoCo contact semantics (Q3).
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
    str(ROOT / "src"), str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]

from audit_taco_pour_low_level_contact_controllability_gate_L import (  # noqa: E402
    actuator_force_law, build_env, urdf_effort_limits,
)
from audit_taco_pour_physics_reference_validity_gate_P import (  # noqa: E402
    FINGERS, P0_SOURCES, _geom_name, _mesh_projection, _rotation_matrix_wxyz,
    _native_from_snapshot, _snapshot_array, _tensor, audit_P0,
    actuator_groups, capture_gate_P_states, native_contacts,
)
from audit_taco_pour_postfix_policy_extremization import (  # noqa: E402
    clone_hidden, load_gzip_torch, sha256, zero_hidden,
)
from audit_taco_pour_tail_semantic_suppression_oracle import (  # noqa: E402
    CANDIDATES, load_contract as load_tail_contract,
)


HAND_OBJECT = lambda a, b: (  # noqa: E731
    (a.startswith("collision_hand_") and (b.startswith("right_object_") or b.startswith("left_object_")))
    or (b.startswith("collision_hand_") and (a.startswith("right_object_") or a.startswith("left_object_")))
)
OBJECT_NAMES = ("right_object_", "left_object_")
Q2_VARIANTS = ("current_mixed", "convex_only", "external_exact_SDF_only")


def load_contract(path: Path) -> tuple[dict, dict[str, Path], dict]:
    raw = path.read_bytes()
    contract = yaml.safe_load(raw)
    if contract.get("schema") != "taco_pour_simulator_contract_adjudication_gate_Q_v1":
        raise ValueError("unsupported Gate Q contract")
    if contract.get("status") != "authorized_final_finite_simulator_contract_adjudication":
        raise ValueError("Gate Q is not authorized")
    runtime = contract["runtime"]
    stopping = contract["stopping_rule"]
    if (
        contract.get("paper_faithful") is not False
        or runtime != {
            "training_allowed": False,
            "optimizer_updates_policy": False,
            "new_controller_allowed": False,
            "action_or_controller_search_allowed": False,
            "mass_friction_effort_gain_axis_frame_or_source_sweep_allowed": False,
            "reward_objective_reference_or_actuator_change_allowed": False,
            "chunk_acceptance_allowed": False,
            "chunk_commit_allowed": False,
            "order": ["Q1", "Q2", "Q3"],
        }
        or stopping["Gate_Q_is_final_physics_gate"] is not True
        or stopping["additional_physics_gate_allowed"] is not False
        or stopping["PPO_retraining_allowed"] is not False
        or stopping["chunk_commit_allowed"] is not False
    ):
        raise ValueError("Gate Q definition changed")
    paths: dict[str, Path] = {}
    for name, row in contract["inputs"].items():
        artifact = Path(row["path"])
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"contract input changed: {artifact}")
        paths[name] = artifact
    return contract, paths, {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
    }


class FormalActuatorTraceProxy:
    """Observe the unchanged formal position actuator at every MJWP substep."""

    def __init__(self, base, law: dict[str, np.ndarray]):
        self.base = base
        self.law = law
        self.source = -1
        self.rows: list[dict[str, np.ndarray | int]] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step_env(self, config, env, target: torch.Tensor):
        qpos = self.base.get_qpos(config, env).detach().cpu().numpy().astype(np.float64)
        qvel = self.base.get_qvel(config, env).detach().cpu().numpy().astype(np.float64)
        desired = target.detach().cpu().numpy().astype(np.float64)
        qpa, qva = self.law["qpos_address"], self.law["qvel_address"]
        position = self.law["kp"] * (desired[0] - qpos[0, qpa])
        damping = -self.law["kd"] * qvel[0, qva]
        requested = position + damping
        self.base.step_env(config, env, target)
        actual = _tensor(env.data_wp, "actuator_force")[0]
        self.rows.append({
            "source": int(self.source), "position": position, "damping": damping,
            "requested": requested, "actual": actual,
        })

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            raise RuntimeError("formal actuator trace is empty")
        return {
            "source": np.asarray([row["source"] for row in self.rows], np.int32),
            "position": np.asarray([row["position"] for row in self.rows], np.float64),
            "damping": np.asarray([row["damping"] for row in self.rows], np.float64),
            "requested": np.asarray([row["requested"] for row in self.rows], np.float64),
            "actual": np.asarray([row["actual"] for row in self.rows], np.float64),
        }


def _semantic_action(action: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(action, np.float32).copy()
    indices = CANDIDATES[mode]
    if indices:
        result[0, np.asarray(indices, np.int64)] = 0.0
    return result


def _tail_mode(source: int, binary_off: set[int], tail_report: dict) -> str:
    if source < 44:
        return "zero_right_wrist_translation" if source in binary_off else "complete_PPO"
    if source <= 55:
        return tail_report["tail_decisions"][source - 44]["selected"]
    if source == 56:
        return tail_report["source56_fork"]["selected_for_continuation"]
    return "complete_PPO"


def _run_Q1_path(*, name: str, env, backend, policy, boundary,
                 law: dict[str, np.ndarray], binary_off: set[int], tail_report: dict,
                 use_policy: bool, use_tail: bool) -> tuple[dict, dict[str, np.ndarray]]:
    proxy = FormalActuatorTraceProxy(env._mjwp, law)
    env._mjwp = proxy
    backend.restore(boundary)
    backend.verify_restored_snapshot(boundary)
    if policy is not None:
        policy.rnn_states = zero_hidden(policy)
    first_failure = None
    modes = []
    for source in range(20, 60):
        if use_policy:
            packed = policy.obs_to_tensors(backend.observation())
            values = policy.get_deterministic_action_values(packed)
            policy.rnn_states = clone_hidden(values["rnn_states"])
            action = policy.preprocess_actions(values["deterministic_actions"])
            mode = _tail_mode(source, binary_off, tail_report) if use_tail else "complete_PPO"
            action = _semantic_action(action, mode)
        else:
            action = np.zeros((1, 36), np.float32)
            mode = "Replay_zero_residual"
        proxy.source = source
        alive = backend.step(action, source)
        modes.append(mode)
        if not alive and first_failure is None:
            first_failure = int(source + 1)
    arrays = proxy.arrays()
    if arrays["actual"].shape != (400, 36):
        raise RuntimeError(f"Q1 {name} expected 400x36 substep trace, got {arrays['actual'].shape}")
    limit = np.asarray(law["effort_limit"], np.float64)
    arrays["requested_ratio"] = np.abs(arrays["requested"]) / limit
    arrays["actual_ratio"] = np.abs(arrays["actual"]) / limit
    groups = actuator_groups(list(law["names"]))
    focus = arrays["source"] >= 44

    def stats(selection: np.ndarray) -> dict:
        requested = arrays["requested_ratio"][selection]
        actual = arrays["actual_ratio"][selection]
        return {
            "physics_substeps": int(selection.sum()),
            "requested_over_limit": int((requested > 1.0 + 1e-12).sum()),
            "actual_over_limit": int((actual > 1.0 + 1e-12).sum()),
            "maximum_requested_to_limit_ratio": float(requested.max()),
            "maximum_actual_to_limit_ratio": float(actual.max()),
            "maximum_abs_actual_minus_affine_law_n": float(
                np.max(np.abs(arrays["actual"][selection] - arrays["requested"][selection]))
            ),
            "groups": {
                group: {
                    "requested_over_limit": int((requested[:, ids] > 1.0 + 1e-12).sum()),
                    "actual_over_limit": int((actual[:, ids] > 1.0 + 1e-12).sum()),
                    "maximum_requested_to_limit_ratio": float(requested[:, ids].max()),
                    "maximum_actual_to_limit_ratio": float(actual[:, ids].max()),
                } for group, ids in groups.items()
            },
        }
    return {
        "name": name, "first_tracking_failure_endpoint": first_failure,
        "diagnostic_continuation_after_failure": first_failure is not None,
        "modes_by_source": modes, "all_endpoints_20_60": stats(np.ones(400, bool)),
        "focus_sources_44_59": stats(focus),
    }, arrays


def derive_collision_variants(source: Path, destination_dir: Path) -> dict[str, Path]:
    outputs = {}
    for variant in Q2_VARIANTS:
        tree = ET.parse(source)
        root = tree.getroot()
        contact = root.find("contact")
        if contact is None:
            raise ValueError("formal scene has no contact section")
        removed = 0
        if variant != "current_mixed":
            for pair in list(contact.findall("pair")):
                names = (pair.attrib.get("geom1", ""), pair.attrib.get("geom2", ""))
                exact = any("external_exact" in name for name in names)
                remove = (
                    (variant == "convex_only" and exact)
                    or (variant == "external_exact_SDF_only" and HAND_OBJECT(*names) and not exact)
                )
                if remove:
                    contact.remove(pair); removed += 1
        output = destination_dir / f"collision_variant_{variant}.xml"
        tree.write(output, encoding="unicode")
        outputs[variant] = output
        if variant != "current_mixed" and removed == 0:
            raise RuntimeError(f"Q2 variant {variant} removed no pairs")
    return outputs


def _variant_snapshot(env, source: int, state_arrays) -> dict:
    import warp as wp

    qpos = np.asarray(state_arrays[f"source{source}_qpos"], np.float32)
    qvel = np.asarray(state_arrays[f"source{source}_qvel"], np.float32)
    ctrl = np.asarray(state_arrays[f"source{source}_ctrl"], np.float32)
    if qpos.ndim == 1: qpos = qpos[None]
    if qvel.ndim == 1: qvel = qvel[None]
    if ctrl.ndim == 1: ctrl = ctrl[None]
    env.time_indices[:] = source
    env.episode_lengths[:] = source
    env._write_state(torch.from_numpy(qpos), torch.from_numpy(qvel), torch.from_numpy(ctrl), np.asarray([True]))
    warm = np.asarray(state_arrays[f"source{source}_qacc_warmstart"], np.float32)
    if warm.ndim == 1: warm = warm[None]
    current = wp.to_torch(env.env.data_wp.qacc_warmstart).clone()
    current[:] = torch.from_numpy(warm).to(current.device)
    wp.copy(env.env.data_wp.qacc_warmstart,
            wp.from_torch(current, dtype=env.env.data_wp.qacc_warmstart.dtype))
    return env.get_env_state()


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0: return 180.0
    return float(np.degrees(np.arccos(np.clip(float(a @ b / denominator), -1.0, 1.0))))


def _duplicate_patch_count(parity_report: dict, parity_arrays: dict[str, np.ndarray],
                           bowl_mesh) -> tuple[int, list[dict]]:
    rows = []
    for source, audit in parity_report["sources"].items():
        for details in audit["contact_details"]:
            for backend in ("MJWP", "native_MuJoCo"):
                contacts = details[backend]
                array_backend = "MJWP" if backend == "MJWP" else "native"
                qpos = np.asarray(
                    parity_arrays[f"source{source}_{array_backend}_qpos"][
                        details["physics_substep"] - 1
                    ], np.float64,
                )
                object_position = qpos[36:39]
                object_rotation = _rotation_matrix_wxyz(qpos[39:43])
                for i, first in enumerate(contacts):
                    f_names = first["geom_names"]
                    f_hand = next((x for x in f_names if x.startswith("collision_hand_")), None)
                    f_obj = next((x for x in f_names if x.startswith(OBJECT_NAMES)), None)
                    if f_hand is None or f_obj is None: continue
                    for second in contacts[i + 1:]:
                        s_names = second["geom_names"]
                        s_hand = next((x for x in s_names if x.startswith("collision_hand_")), None)
                        s_obj = next((x for x in s_names if x.startswith(OBJECT_NAMES)), None)
                        if s_hand != f_hand or s_obj is None or s_obj == f_obj: continue
                        p1 = np.asarray(first["position_world_m"], np.float64)
                        p2 = np.asarray(second["position_world_m"], np.float64)
                        n1 = np.asarray(first["normal_geom1_to_geom2_world"], np.float64)
                        n2 = np.asarray(second["normal_geom1_to_geom2_world"], np.float64)
                        # Normal sign depends on pair ordering; compare unsigned patch normals.
                        angle = min(_angle_deg(n1, n2), _angle_deg(n1, -n2))
                        separation = float(np.linalg.norm(p1 - p2))
                        if separation > 5e-4 or angle > 5.0: continue
                        local = (np.vstack([p1, p2]) - object_position) @ object_rotation
                        nearest, _, _ = _mesh_projection(bowl_mesh, local)
                        visual_patch_separation = float(np.linalg.norm(nearest[0] - nearest[1]))
                        if visual_patch_separation > 5e-4: continue
                        rows.append({
                            "source": int(source), "physics_substep": details["physics_substep"],
                            "backend": backend, "hand_geom": f_hand,
                            "object_geoms": [f_obj, s_obj], "point_separation_m": separation,
                            "normal_angle_deg": angle,
                            "visual_surface_patch_separation_m": visual_patch_separation,
                        })
    return len(rows), rows


def audit_Q2(*, paths: dict[str, Path], variants: dict[str, Path], config,
             reference, objective, observation, residual, gate_P_contract: dict,
             reference_qpos: np.ndarray, state_arrays, bowl_mesh) -> tuple[dict, dict[str, np.ndarray]]:
    import mujoco
    from video_to_spider.rl.replay_rl import MJWPChunkBackend

    report, saved = {}, {}
    current_duplicates = None
    for variant, xml in variants.items():
        variant_config = deepcopy(config)
        variant_config.model_path = str(xml.resolve())
        env = build_env(variant_config, reference, objective, observation, residual)
        backend = MJWPChunkBackend(env)
        states = {}
        for source in P0_SOURCES:
            states[source] = {
                "snapshot": _variant_snapshot(env, source, state_arrays),
                "full_ctrl": np.asarray(state_arrays[f"source{source}_full_ctrl"], np.float64),
            }
        model = mujoco.MjModel.from_xml_path(str(xml))
        parity, arrays = audit_P0(
            contract=gate_P_contract, backend=backend, model=model, states=states,
            reference_qpos=reference_qpos, objective=objective,
        )
        duplicates, duplicate_rows = _duplicate_patch_count(parity, arrays, bowl_mesh)
        source45_pairs = parity["sources"]["45"]["contact_pair_sets"]
        source45_match = bool(all(row["exact_match"] for row in source45_pairs))
        if variant == "current_mixed": current_duplicates = duplicates
        report[variant] = {
            "xml": str(xml.resolve()), "compiled_pair_count": int(model.npair),
            "parity": parity, "source45_exact_pair_set_all_substeps": source45_match,
            "visual_surface_patch_duplicate_count": duplicates,
            "duplicate_patch_rows": duplicate_rows,
        }
        for key, value in arrays.items(): saved[f"{variant}_{key}"] = value
    if current_duplicates is None: raise AssertionError("missing current mixed variant")
    passing = []
    for variant, row in report.items():
        row["freeze_condition_passed"] = bool(
            row["source45_exact_pair_set_all_substeps"]
            and row["parity"]["all_sources_passed"]
            and row["visual_surface_patch_duplicate_count"] < current_duplicates
        )
        if row["freeze_condition_passed"]: passing.append(variant)
    selected = passing[0] if len(passing) == 1 else None
    return {
        "variants": report, "passing_variants": passing,
        "unique_selected_variant": selected,
        "collision_backend_contract_frozen": selected is not None,
        "backend_dependent_physics_acknowledged": selected is None,
        "limited_claim": "finite ten-substep parity at source45/50/57 for three predeclared pair representations",
    }, saved


def _side_surface_samples(model, data, side: str) -> list[dict]:
    import mujoco

    rows = []
    site_names = {
        f"{side}_palm": "palm", f"{side}_thumb_tip": "thumb",
        f"{side}_index_tip": "index", f"{side}_middle_tip": "middle",
        f"{side}_ring_tip": "ring", f"{side}_pinky_tip": "pinky",
    }
    for site_name, role in site_names.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site < 0:
            raise ValueError(f"missing declared Gate-Q surface site {site_name}")
        body = int(model.site_bodyid[site])
        paired_geoms = [
            geom for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) == body
            and _geom_name(model, geom).startswith(f"collision_hand_{side}")
            and _pair_semantics(model, geom) is not None
        ]
        if paired_geoms:
            rows.append({
                "sample": f"site:{site_name}", "side": side, "role": role,
                "geom_id": paired_geoms[0], "body_id": body,
                "point_world": np.asarray(data.site_xpos[site], np.float64).copy(),
            })
    for geom in range(model.ngeom):
        name = _geom_name(model, geom)
        if not name.startswith(f"collision_hand_{side}"): continue
        # Self-collision and floor-only guards are not hand--tool contact
        # surfaces.  Q3 samples every surface that has a compiled object pair,
        # rather than projecting thousands of unrelated semantic self guards.
        if _pair_semantics(model, geom) is None: continue
        lower = name.lower()
        role = next((f for f in FINGERS if f in lower), "palm")
        center = np.asarray(data.geom_xpos[geom], np.float64)
        rotation = np.asarray(data.geom_xmat[geom], np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[geom], np.float64)
        kind = int(model.geom_type[geom]); local = []
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            local = [axis * sign * size[0] for axis in np.eye(3) for sign in (-1., 1.)]
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            local = [np.array([sign * size[0], 0., 0.]) for sign in (-1., 1.)]
            local += [np.array([0., sign * size[0], 0.]) for sign in (-1., 1.)]
            local += [np.array([0., 0., sign * size[1]]) for sign in (-1., 1.)]
        elif kind in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            local = [np.eye(3)[axis] * sign * size[axis] for axis in range(3) for sign in (-1., 1.)]
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                local += [np.asarray(signs) * size for signs in (
                    (-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),
                    (1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1))]
        else:
            radius = float(model.geom_rbound[geom])
            local = [axis * sign * radius for axis in np.eye(3) for sign in (-1., 1.)]
        for index, point in enumerate(local):
            rows.append({
                "sample": f"geom:{name}:{index}", "side": side, "role": role,
                "geom_id": geom, "body_id": int(model.geom_bodyid[geom]),
                "point_world": center + rotation @ point,
            })
    return rows


def _pair_semantics(model, hand_geom: int) -> tuple[int, np.ndarray] | None:
    for pair in range(model.npair):
        g1, g2 = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        if hand_geom not in (g1, g2): continue
        other = g2 if hand_geom == g1 else g1
        if _geom_name(model, other).startswith(OBJECT_NAMES):
            return int(model.pair_dim[pair]), np.asarray(model.pair_friction[pair], np.float64)
    return None


def _pair_semantics_between(model, geom1: int, geom2: int) -> tuple[int, np.ndarray] | None:
    target = {int(geom1), int(geom2)}
    for pair in range(model.npair):
        if {int(model.pair_geom1[pair]), int(model.pair_geom2[pair])} == target:
            return int(model.pair_dim[pair]), np.asarray(model.pair_friction[pair], np.float64)
    return None


def _unique_candidates(rows: list[dict]) -> list[dict]:
    selected = []
    for row in sorted(rows, key=lambda item: item["visual_mesh_gap_m"]):
        duplicate = False
        for old in selected:
            if (old["side"], old["cluster_body"]) != (row["side"], row["cluster_body"]): continue
            if np.linalg.norm(np.asarray(old["nearest_world"]) - np.asarray(row["nearest_world"])) > 5e-4: continue
            if _angle_deg(np.asarray(old["normal_world"]), np.asarray(row["normal_world"])) > 5.0: continue
            duplicate = True; break
        if not duplicate: selected.append(row)
    return selected


def _cone_rays(normal: np.ndarray, frame: np.ndarray, condim: int,
               friction: np.ndarray) -> list[np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    if condim == 1: return [normal]
    if condim != 3:
        raise ValueError(f"Gate Q finite LP supports compiled condim 1 or 3, got {condim}")
    tangent1 = frame[1] / np.linalg.norm(frame[1])
    tangent2 = frame[2] / np.linalg.norm(frame[2])
    return [
        normal + friction[0] * tangent1, normal - friction[0] * tangent1,
        normal + friction[1] * tangent2, normal - friction[1] * tangent2,
    ]


def _contact_columns(*, model, data, tool_body: int, candidate: dict,
                     qvel_addresses: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    import mujoco

    point = np.asarray(candidate["point_world"], np.float64)
    normal = np.asarray(candidate["normal_world"], np.float64)
    frame = np.asarray(candidate["frame_world"], np.float64)
    rays = _cone_rays(normal, frame, candidate["condim"], np.asarray(candidate["friction"]))
    tool_jp = np.zeros((3, model.nv)); tool_jr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, tool_jp, tool_jr, point, tool_body)
    columns, effort, labels = [], [], []
    for ray in rays:
        columns.append(tool_jp[:, 36:42].T @ ray)
        hand_cost = np.zeros(36, np.float64)
        if candidate["kind"] == "hand":
            hand_jp = np.zeros((3, model.nv)); hand_jr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, hand_jp, hand_jr, point, candidate["body_id"])
            hand_cost[:] = hand_jp[:, qvel_addresses].T @ (-ray)
        effort.append(hand_cost)
        labels.append(candidate["label"])
    return columns, effort, labels


def audit_Q3(*, model, reference_qpos: np.ndarray, reference_qvel: np.ndarray,
             reference_ctrl: np.ndarray, residual_scale: float,
             effort_limits: np.ndarray, bowl_mesh) -> tuple[dict, dict[str, np.ndarray]]:
    import mujoco
    from scipy.optimize import linprog

    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("Gate Q requires compiled pyramidal cone; no substitute cone is allowed")
    tool_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    if tool_joint < 0:
        tool_joint = next(j for j in range(model.njnt) if int(model.jnt_qposadr[j]) == 36)
    tool_body = int(model.jnt_bodyid[tool_joint])
    qvel_addresses = np.asarray([
        model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]
    ], np.int64)
    data = mujoco.MjData(model)
    dt = 1.0 / 30.0
    qacc_ref = np.gradient(reference_qvel, dt, axis=0, edge_order=2)
    ctrlrange = np.asarray(model.actuator_ctrlrange, np.float64)
    limited = np.asarray(model.actuator_ctrllimited, bool)
    endpoints, required_all, counts, feasible = {}, [], [], []
    for endpoint in range(44, 61):
        data.qpos[:] = reference_qpos[endpoint]
        data.qvel[:] = reference_qvel[endpoint]
        data.qacc[:] = qacc_ref[endpoint]
        data.ctrl[:] = reference_ctrl[min(endpoint, len(reference_ctrl) - 1)]
        mujoco.mj_forward(model, data)
        data.qacc[:] = qacc_ref[endpoint]
        mujoco.mj_inverse(model, data)
        # MuJoCo inverse dynamics reports the applied generalized force after
        # subtracting its inverse constraint solution.  Q3 is about whether a
        # newly assembled unique contact graph can supply the free-body demand,
        # so recover M*qacc + bias/passive by adding that constraint force back.
        # Using qfrc_inverse alone would double-count the current contact graph
        # and manufacture O(10^2--10^3 N) requirements.
        required = np.asarray(
            data.qfrc_inverse[36:42] + data.qfrc_constraint[36:42], np.float64
        ).copy()
        object_position = np.asarray(data.xpos[tool_body], np.float64)
        object_rotation = np.asarray(data.xmat[tool_body], np.float64).reshape(3, 3)
        candidates = []
        control = np.asarray(reference_ctrl[min(endpoint, len(reference_ctrl) - 1)], np.float64)
        low = np.full(36, -1.0); high = np.full(36, 1.0)
        low[limited] = np.maximum(low[limited], (ctrlrange[limited, 0] - control[limited]) / residual_scale)
        high[limited] = np.minimum(high[limited], (ctrlrange[limited, 1] - control[limited]) / residual_scale)
        if np.any(low > high + 2e-7): raise RuntimeError(f"empty residual support at {endpoint}")
        for side in ("right", "left"):
            side_indices = np.arange(0, 18) if side == "right" else np.arange(18, 36)
            samples = _side_surface_samples(model, data, side)
            points = np.asarray([row["point_world"] for row in samples], np.float64)
            local = (points - object_position) @ object_rotation
            nearest_local, gaps, normals_local = _mesh_projection(bowl_mesh, local)
            for index, sample in enumerate(samples):
                semantics = _pair_semantics(model, sample["geom_id"])
                if semantics is None: continue
                nearest_world = object_position + object_rotation @ nearest_local[index]
                toward = nearest_world - points[index]
                gap = float(gaps[index])
                direction = toward / gap if gap > 1e-12 else -(object_rotation @ normals_local[index])
                jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, jr, points[index], sample["body_id"])
                coefficients = direction @ jp[:, qvel_addresses[side_indices]]
                authority = float(np.sum(np.maximum(
                    coefficients * residual_scale * low[side_indices],
                    coefficients * residual_scale * high[side_indices],
                )))
                if gap > authority + 1e-6: continue
                normal = -(object_rotation @ normals_local[index])
                normal /= np.linalg.norm(normal)
                # Stable tangent basis; pair friction and condim remain compiled semantics.
                seed = np.array([1.,0.,0.]) if abs(normal[0]) < .8 else np.array([0.,1.,0.])
                tangent1 = np.cross(normal, seed); tangent1 /= np.linalg.norm(tangent1)
                tangent2 = np.cross(normal, tangent1)
                condim, friction = semantics
                candidates.append({
                    "kind": "hand", "label": f"{side}:{sample['role']}:{sample['sample']}",
                    "side": side, "body_id": sample["body_id"],
                    "cluster_body": sample["body_id"],
                    "point_world": nearest_world, "nearest_world": nearest_world,
                    "normal_world": normal, "frame_world": np.vstack([normal,tangent1,tangent2]),
                    "condim": condim, "friction": friction, "visual_mesh_gap_m": gap,
                })
        candidates = _unique_candidates(candidates)
        hand_count = len(candidates)
        # Native contacts at the exact reference geometry add passive tool-target/floor support.
        for row in native_contacts(model, data, only_right_tool=False):
            names = row["geom_names"]
            tool_side = [i for i, name in enumerate(names) if name.startswith("right_object_")]
            if not tool_side: continue
            other = names[1 - tool_side[0]]
            if not (other.startswith("left_object_") or "floor" in other.lower() or "table" in other.lower()):
                continue
            normal = np.asarray(row["normal_geom1_to_geom2_world"], np.float64)
            force_on_tool = normal if tool_side[0] == 1 else -normal
            frame = np.asarray(row["frame_world"], np.float64)
            if float(force_on_tool @ frame[0]) < 0: frame = -frame
            semantics = _pair_semantics_between(model, *row["geom_ids"])
            if semantics is None:
                # Native contact without an explicit pair: use the compiled
                # contact dimension and MuJoCo's priority/solmix-combined geom
                # friction as observed on the contact row.
                semantics = (
                    int(row["condim"]),
                    np.maximum(
                        np.asarray(model.geom_friction[row["geom_ids"][0]], np.float64),
                        np.asarray(model.geom_friction[row["geom_ids"][1]], np.float64),
                    ),
                )
            counterpart_geom = row["geom_ids"][1 - tool_side[0]]
            candidates.append({
                "kind": "passive", "label": f"passive:{other}", "side": "passive",
                "body_id": -1, "cluster_body": int(model.geom_bodyid[counterpart_geom]),
                "point_world": np.asarray(row["position_world_m"]),
                "nearest_world": np.asarray(row["position_world_m"]),
                "normal_world": force_on_tool, "frame_world": frame,
                "condim": semantics[0], "friction": semantics[1],
                "visual_mesh_gap_m": 0.0,
            })
        candidates = _unique_candidates(candidates)
        columns, efforts, labels = [], [], []
        for candidate in candidates:
            c, e, l = _contact_columns(
                model=model, data=data, tool_body=tool_body, candidate=candidate,
                qvel_addresses=qvel_addresses,
            )
            columns += c; efforts += e; labels += l
        matrix = np.stack(columns, axis=1) if columns else np.zeros((6,0))
        effort = np.stack(efforts, axis=1) if efforts else np.zeros((36,0))
        cone_solution = None
        if matrix.shape[1]:
            cone_solution = linprog(
                c=np.zeros(matrix.shape[1]), A_eq=matrix, b_eq=required,
                bounds=[(0., None)] * matrix.shape[1], method="highs",
            )
            solution = linprog(
                c=np.zeros(matrix.shape[1]),
                A_ub=np.vstack([effort, -effort]),
                b_ub=np.concatenate([effort_limits, effort_limits]),
                A_eq=matrix, b_eq=required,
                bounds=[(0., None)] * matrix.shape[1], method="highs",
            )
            ok = bool(solution.success)
        else:
            solution = None; ok = False
        endpoint_row = {
            "required_inverse_dynamics_free_joint_force": required.tolist(),
            "hand_unique_patch_candidates": hand_count,
            "passive_unique_patch_candidates": len(candidates) - hand_count,
            "total_unique_patch_candidates": len(candidates),
            "pyramidal_rays": int(matrix.shape[1]),
            "contact_cone_feasible_without_hand_effort_limits": bool(
                cone_solution is not None and cone_solution.success
            ),
            "feasible": ok,
            "LP_status": solution.message if solution is not None else "no_contact_candidates",
            "active_candidate_labels": (
                sorted(set(labels[i] for i, value in enumerate(solution.x) if value > 1e-9))
                if ok else []
            ),
        }
        endpoints[str(endpoint)] = endpoint_row
        required_all.append(required); counts.append([hand_count, len(candidates)-hand_count, matrix.shape[1]])
        feasible.append(ok)
    return {
        "geometry_backend": "native_MuJoCo",
        "required_wrench_method": (
            "mj_inverse_free_body_smooth_demand_qfrc_inverse_plus_qfrc_constraint_"
            "on_tool_free_joint_dofs"
        ),
        "compiled_cone": "pyramidal",
        "artificial_16_sided_cone_used": False,
        "endpoints": endpoints,
        "all_endpoints_feasible": bool(all(feasible)),
        "infeasible_endpoints": [endpoint for endpoint, ok in zip(range(44,61), feasible) if not ok],
        "finite_model_statement": (
            "Finite bilateral collision-surface samples plus live tool-target/floor contacts, "
            "clustered by 0.5 mm/5 degree patch criteria; not a global contact proof."
        ),
    }, {
        "endpoint": np.arange(44,61,dtype=np.int32),
        "required_inverse_dynamics_free_joint_force": np.asarray(required_all),
        "candidate_counts_hand_passive_rays": np.asarray(counts, np.int32),
        "feasible": np.asarray(feasible, bool),
    }


def _write_heatmap(path: Path, Q1_arrays: dict[str, np.ndarray], names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = ("Replay_zero_residual", "frozen_single_pass_PPO", "frozen_binary_prefix_plus_tail_semantic_oracle")
    figure, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, paths, strict=True):
        ratio = Q1_arrays[f"{name}_actual_ratio"].T
        image = axis.imshow(np.log10(np.maximum(ratio,1e-3)), aspect="auto", origin="lower", cmap="magma")
        axis.contour(ratio, levels=[1.0], colors="cyan", linewidths=.7)
        axis.set_title(f"{name}: log10(|compiled actuator force| / URDF effort)")
        axis.set_yticks(np.arange(len(names))); axis.set_yticklabels(names, fontsize=5)
        figure.colorbar(image, ax=axis)
    axes[-1].set_xlabel("physics substep (sources 20--59; 10 each)")
    figure.savefig(path, dpi=180); plt.close(figure)


def _write_summary(path: Path, report: dict) -> None:
    q1, q2, q3 = report["Q1"], report["Q2"], report["Q3"]
    lines = [
        "# TACO Pour simulator-contract adjudication Gate Q v1", "",
        "Gate Q is the final finite physics gate. No training, controller/action search, physics-parameter sweep, chunk acceptance, or chunk commit occurred.", "",
        "## Q1 — formal actuator demand", "",
    ]
    for name, row in q1["paths"].items():
        focus = row["focus_sources_44_59"]
        lines.append(f"- `{name}`: {focus['actual_over_limit']}/{focus['physics_substeps']*36} focus-window actuator samples exceed declared effort; max ratio `{focus['maximum_actual_to_limit_ratio']:.6g}`.")
    lines += ["", "## Q2 — collision/backend contract", "",
              f"- Passing variants: `{q2['passing_variants']}`",
              f"- Unique frozen variant: `{q2['unique_selected_variant']}`",
              f"- Backend-dependent physics acknowledged: `{q2['backend_dependent_physics_acknowledged']}`", "",
              "## Q3 — complete-contact finite wrench", "",
              f"- All reference endpoints feasible: `{q3['all_endpoints_feasible']}`",
              f"- Infeasible endpoints: `{q3['infeasible_endpoints']}`", "",
              "## Final decision", "",
              f"- Gate B/C reopened: `{report['decision']['Gate_B_C_reopened']}`",
              f"- Exact reproduction status: `{report['decision']['exact_reproduction_status']}`",
              "- Gate Q is closed. The protocol forbids additional physics gates or automatic fallback experiments."]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/taco_pour_simulator_contract_adjudication_gate_Q_v1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/taco_pour_simulator_contract_adjudication_gate_Q_v1")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    contract, paths, contract_artifact = load_contract(args.contract)
    gate_P_contract = yaml.safe_load(paths["Gate_P_contract"].read_text())
    _, parent_paths, _ = load_tail_contract(paths["tail_contract"])
    tail_report = json.loads(paths["tail_report"].read_text())
    binary_report = json.loads(paths["binary_oracle_report"].read_text())
    binary_off = set(binary_report["oracle_result"]["OFF_sources"])

    from run_mjwp_ppo import _build_network_config, _build_ppo_config, _load_ego_config, _load_reference
    from video_to_spider.rl.action_contract import load_residual_action_profile
    from video_to_spider.rl.objective_contract import load_runtime_objective
    from video_to_spider.rl.observation_contract import load_runtime_observation
    from video_to_spider.rl.replay_rl import MJWPChunkBackend
    from video_to_spider.rl.state_feasible_truncated_gaussian import StateFeasibleTruncatedGaussianPpoAgent, load_truncated_gaussian_profile
    import mujoco

    objective = load_runtime_objective(parent_paths["protocol"], parent_paths["objective_profile"], tracking_variant="tool_only", require_run_ready=False)
    observation = load_runtime_observation(parent_paths["protocol"], parent_paths["observation_profile"], require_run_ready=False)
    residual, _ = load_residual_action_profile(parent_paths["action_profile"])
    distribution, _ = load_truncated_gaussian_profile(parent_paths["distribution_profile"])
    config = _load_ego_config(str(paths["formal_simulator_config"]), "cpu")
    reference = _load_reference(config.data_path, "cpu", expected_frequency=30)
    reference_qpos = reference[0].detach().cpu().numpy().astype(np.float64)
    reference_qvel = reference[1].detach().cpu().numpy().astype(np.float64)
    reference_ctrl = np.load(paths["robot_reference"])["ctrl"].astype(np.float64)
    checkpoint = load_gzip_torch(paths["checkpoint"])
    boundary = load_gzip_torch(paths["boundary"])
    effort_map = urdf_effort_limits((paths["right_XHand_URDF"], paths["left_XHand_URDF"]))

    temporary_root = Path(tempfile.mkdtemp(prefix=".gate_Q_", dir=ROOT / "runs"))
    try:
        variant_dir = temporary_root / "variants"; variant_dir.mkdir()
        variants = derive_collision_variants(paths["formal_scene"], variant_dir)
        # Q1 paths use independent formal environments and the same frozen actor.
        q1_report, q1_arrays = {}, {}
        law_for_report = None
        for name, use_policy, use_tail in (
            ("Replay_zero_residual", False, False),
            ("frozen_single_pass_PPO", True, False),
            ("frozen_binary_prefix_plus_tail_semantic_oracle", True, True),
        ):
            env = build_env(config, reference, objective, observation, residual)
            backend = MJWPChunkBackend(env)
            law = actuator_force_law(env.env.model_cpu, effort_map); law_for_report = law
            policy = None
            if use_policy:
                ppo_config = replace(_build_ppo_config(num_envs=1, horizon_length=40, seq_length=4, max_epochs=8, learning_rate=1e-4, device="cpu", asymmetric_critic=None), clip_actions=False)
                policy = StateFeasibleTruncatedGaussianPpoAgent(
                    experiment_dir=temporary_root / f"policy_{name}", ppo_config=ppo_config,
                    network_config=_build_network_config(4), env=env, distribution_spec=distribution,
                )
                policy.model.load_state_dict(checkpoint["model"]); policy.set_eval()
            row, arrays = _run_Q1_path(
                name=name, env=env, backend=backend, policy=policy, boundary=boundary,
                law=law, binary_off=binary_off, tail_report=tail_report,
                use_policy=use_policy, use_tail=use_tail,
            )
            q1_report[name] = row
            for key, value in arrays.items(): q1_arrays[f"{name}_{key}"] = value
            if policy is not None:
                policy.writer.close()
        Q1 = {
            "paths": q1_report,
            "actuator_names": list(law_for_report["names"]),
            "declared_URDF_effort_limits": np.asarray(law_for_report["effort_limit"]).tolist(),
            "formal_actuator_changed": False,
        }

        # Recreate exact Gate-P source states once for Q2 full_ctrl and compact state input.
        env_current = build_env(config, reference, objective, observation, residual)
        backend_current = MJWPChunkBackend(env_current)
        ppo_config = replace(_build_ppo_config(num_envs=1, horizon_length=40, seq_length=4, max_epochs=8, learning_rate=1e-4, device="cpu", asymmetric_critic=None), clip_actions=False)
        policy = StateFeasibleTruncatedGaussianPpoAgent(
            experiment_dir=temporary_root / "policy_Q2", ppo_config=ppo_config,
            network_config=_build_network_config(4), env=env_current, distribution_spec=distribution,
        )
        policy.model.load_state_dict(checkpoint["model"]); policy.set_eval()
        exact_states = capture_gate_P_states(
            backend=backend_current, policy=policy, boundary=boundary,
            parent_paths=parent_paths, tail_report=tail_report,
            reference_qpos=reference_qpos, reference_qvel=reference_qvel, objective=objective,
        )
        policy.writer.close()
        state_arrays = {}
        for source, state in exact_states.items():
            for field in ("qpos","qvel","qacc_warmstart","ctrl"):
                if field in state["snapshot"]: state_arrays[f"source{source}_{field}"] = _snapshot_array(state["snapshot"], field)
            if "full_ctrl" in state: state_arrays[f"source{source}_full_ctrl"] = state["full_ctrl"]
        import trimesh
        bowl_mesh = trimesh.load(paths["bowl_visual_mesh"], force="mesh", process=True)
        # The formal MJCF declares `scale="0.01 0.01 0.01"` for this OBJ.
        # Trimesh loads raw OBJ units, so Q3 must apply the identical compiled
        # geometry scale before any gap, patch, or reachability calculation.
        bowl_mesh.apply_scale(0.01)
        Q2, q2_arrays = audit_Q2(
            paths=paths, variants=variants, config=config, reference=reference,
            objective=objective, observation=observation, residual=residual,
            gate_P_contract=gate_P_contract, reference_qpos=reference_qpos,
            state_arrays=state_arrays, bowl_mesh=bowl_mesh,
        )
        q3_variant = Q2["unique_selected_variant"] or "current_mixed"
        q3_model = mujoco.MjModel.from_xml_path(str(variants[q3_variant]))
        q3_law = actuator_force_law(q3_model, effort_map)
        Q3, q3_arrays = audit_Q3(
            model=q3_model, reference_qpos=reference_qpos, reference_qvel=reference_qvel,
            reference_ctrl=reference_ctrl, residual_scale=float(residual.residual_scale),
            effort_limits=np.asarray(q3_law["effort_limit"], np.float64), bowl_mesh=bowl_mesh,
        )
        Q3["collision_variant"] = q3_variant

        reopen = bool(Q2["collision_backend_contract_frozen"] and Q3["all_endpoints_feasible"])
        report = {
            "schema": "taco_pour_simulator_contract_adjudication_gate_Q_report_v1",
            "status": "completed_final_finite_simulator_contract_adjudication_closed",
            "paper_faithful": False, "contract": contract_artifact,
            "runtime": contract["runtime"], "Q1": Q1, "Q2": Q2, "Q3": Q3,
            "decision": {
                "Gate_Q_closed": True, "Gate_Q_is_final_physics_gate": True,
                "Gate_B_C_reopened": reopen,
                "exact_reproduction_status": (
                    "finite_simulator_contract_supports_reopening_architecture_gates"
                    if reopen else
                    "blocked_pending_unpublished_simulator_contact_actuation_object_physics_and_objective_details"
                ),
                "additional_physics_or_parity_gate_allowed": False,
                "PPO_retraining_allowed": False,
                "chunk_commit_allowed": False,
            },
            "artifacts": {},
        }

        # Publish only after every computation succeeds.
        args.output.mkdir(parents=True)
        shutil.copy2(args.contract, args.output / "contract.yaml")
        for variant, temp in variants.items():
            destination = args.output / contract["artifacts"][f"collision_variant_{variant}"]
            shutil.copy2(temp, destination)
            report["Q2"]["variants"][variant]["xml"] = str(destination.resolve())
            report["artifacts"][f"collision_variant_{variant}"] = {"path": str(destination.resolve()), "sha256": sha256(destination)}
        for key, arrays in (("Q1_arrays", q1_arrays), ("Q2_arrays", q2_arrays), ("Q3_arrays", q3_arrays)):
            destination = args.output / contract["artifacts"][key]
            np.savez_compressed(destination, **arrays)
            report["artifacts"][key] = {"path": str(destination.resolve()), "sha256": sha256(destination)}
        heatmap = args.output / contract["artifacts"]["Q1_heatmap"]
        _write_heatmap(heatmap, q1_arrays, list(law_for_report["names"]))
        report["artifacts"]["Q1_heatmap"] = {"path": str(heatmap.resolve()), "sha256": sha256(heatmap)}
        report_path = args.output / contract["artifacts"]["report"]
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_summary(args.output / contract["artifacts"]["summary"], report)
        print(json.dumps({"report": str(report_path), "decision": report["decision"]}, indent=2))
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
