#!/usr/bin/env python3
"""Bounded, actor-free Replay 0->20 contact trace.

The script has an explicit preflight phase which writes the immutable input
manifest before any environment (and therefore any setup integration) is
constructed.  ``run`` refuses to proceed without that manifest, first proves
plain Replay identity (A), then restores the reconstructed complete s0 and
performs the observer-on trace (B).  It never trains or commits a chunk.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp
import yaml

from video_to_spider.rl.action_contract import load_residual_action_profile
from video_to_spider.rl.core.env import load_ego_config, load_reference
from video_to_spider.rl.mjwp_env import MJWPVectorEnv, MJWPVectorEnvConfig
from video_to_spider.rl.objective_contract import load_runtime_objective
from video_to_spider.rl.observation_contract import load_runtime_observation
from video_to_spider.rl.replay_contact_trace import (
    BudgetLedger,
    canonical_contact_group,
    classify_geom_role,
    clone_array,
    control_unit_labels,
    startup_control_sequences,
    transition_indices,
)


ASSET_ROOT = Path("/data_all/zzx/3.2RL")
HISTORICAL = ASSET_ROOT / "runs/taco_pour_corrected_replay_rebase_v1/tool_only"
INITIAL_REPORT = ASSET_ROOT / "runs/taco_pour_initialization_protocol_v2/candidate_a/report.json"
OUTPUT = ASSET_ROOT / "runs/taco_pour_replay_0_20_contact_trace_v1"
EXPECTED_HEAD = "4a35c280ff49fbd1bd216042e79bc03d91149221"
EXPECTED = {
    "historical_trajectory": "81712fd46d5bade497a2a98e8ce07c6f8f8b2f600510db8de42aed1a952ef4b3",
    "initial_state": "fe0c3b0620a44be47f22174d027c1150622dfd54e4b86407a9259515b95d4f36",
    "scene": "deb35689ce385aa30f361ddd1907f54bdbd4f55d4fd98a22409b68a9a691a215",
    "reference": "cf99b9846a63617ddd5e439e086708c80a1e9b43f614df63168ffb3236a7685c",
}
TRACE_SCHEMA = "taco_pour_replay_0_20_contact_trace_v1"
SUBSTEPS = 10
ENDPOINTS = 20
STARTUP_OUTPUT = ASSET_ROOT / "runs/taco_pour_startup_transition_isolation_v1"
STARTUP_SCHEMA = "taco_pour_startup_transition_isolation_v1"
STARTUP_BASELINE = "8b5e941f5a12d8823f53949e070e0efac36b94cc"
STARTUP_TRACE = OUTPUT
STARTUP_SNAPSHOT = STARTUP_TRACE / "snapshots/A_s0_reconstructed.pt.gz"
STARTUP_KEYS = (0, 1, 5, 10, 14, 15, 16, 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def torch_gzip_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        torch.save(value, stream)


def torch_gzip_read(path: Path) -> Any:
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_paths() -> dict[str, Path]:
    contracts = HISTORICAL / "input_contracts"
    report = json.loads(INITIAL_REPORT.read_text())
    return {
        "historical_trajectory": HISTORICAL / "optimized_trajectory.npz",
        "historical_s20": HISTORICAL / "committed_boundary_endpoint_20.pt.gz",
        "historical_report": HISTORICAL / "report.json",
        "simulator_config": contracts / "simulator_config.yaml",
        "protocol": contracts / "replay_rl_protocol.yaml",
        "objective_profile": contracts / "objective_profile.yaml",
        "observation_profile": contracts / "observation_profile.yaml",
        "action_profile": contracts / "action_profile.yaml",
        "backend_contract": contracts / "dual_backend_contract.yaml",
        "initialization_report": INITIAL_REPORT,
        "initial_state": Path(report["initial_state"]["path"]),
        "scene": Path(report["scene"]["path"]),
        "reference": Path(report["reference"]["path"]),
        "mjwp_support_source": Path(sys.modules[mjwarp.__name__].__file__).resolve().parent / "_src/support.py",
        "mjwp_forward_source": Path(sys.modules[mjwarp.__name__].__file__).resolve().parent / "_src/forward.py",
        "active_environment_source": repo_root() / "src/video_to_spider/rl/mjwp_env.py",
        "spider_backend_source": repo_root() / "external/spider_compat/spider/simulators/mjwp.py",
    }


def preflight(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"immutable diagnostic output already exists: {output}")
    output.mkdir(parents=True)
    paths = input_paths()
    artifacts = {name: artifact(path) for name, path in paths.items()}
    for name, digest in EXPECTED.items():
        if artifacts[name]["sha256"] != digest:
            raise RuntimeError(f"frozen {name} hash changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"diagnostic requires exact baseline {EXPECTED_HEAD}, got {head}")
    with np.load(paths["historical_trajectory"], allow_pickle=False) as archive:
        trajectory = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }
        if not np.array_equal(archive["reference_endpoint"], np.arange(21)):
            raise RuntimeError("historical endpoints are not 0..20")
    with np.load(paths["initial_state"], allow_pickle=False) as archive:
        initial = {name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)} for name in archive.files}
    config = yaml.safe_load(paths["simulator_config"].read_text())
    manifest = {
        "schema": TRACE_SCHEMA,
        "status": "preflight_complete_no_environment_constructed",
        "git_commit": head,
        "inputs": artifacts,
        "historical_trajectory_arrays": trajectory,
        "initial_state_arrays": initial,
        "s0_source": {
            "kind": "reconstructed_from_historical_initialization_path",
            "historical_full_s0_snapshot_found": False,
            "procedure": "construct original CPU MJWarp world; write accepted initial qpos/qvel/ctrl; zero reset fields and call the historical MJWarp forward path; set last_ctrl; capture complete snapshot immediately",
            "not_claimed": "restoration of a historically serialized full s0",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_warp": importlib.metadata.version("mujoco-warp"),
            "warp": wp.__version__,
            "device": "cpu",
        },
        "physics": {
            "sim_dt": config["sim_dt"],
            "ctrl_dt": config["ctrl_dt"],
            "ctrl_steps": SUBSTEPS,
            "nconmax_per_env": config["nconmax_per_env"],
            "njmax_per_env": config["njmax_per_env"],
            "object_assistance": None,
            "auto_reset": False,
            "historical_horizon_preserved": 197,
            "diagnostic_stop_endpoint": 20,
        },
        "time_semantics": {
            "state_before": "cached qpos/qvel/time immediately before this MJWarp step",
            "solve_for_this_transition": "contact and efc arrays left by forward/solver for state_before and the current control",
            "state_after": "post-integrator qpos/qvel/time observed by the callback",
        },
        "force_semantics": {
            "api": "installed mujoco_warp.contact_force",
            "contact_wrench_order": "force_xyz_then_torque_xyz",
            "contact_frame_normal": "positive x points geom1 toward geom2",
            "world_wrench_direction": "reported API wrench acts on geom2 by geom1; opposite on geom1",
        },
        "planned_cost": {
            "setup_native_mj_steps": 1,
            "A_physics_steps": 200,
            "B_physics_steps": 200,
            "total_physics_integrations": 401,
            "task_control_intervals": 40,
            "actor_or_critic_forwards": 0,
            "optimizer_updates": 0,
        },
        "authorization": {
            "diagnostic_only": True,
            "training": False,
            "chunk_commit": False,
            "actions_or_physics_modified": False,
        },
    }
    write_json(output / "input_manifest.json", manifest)


def load_initial(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    report = json.loads(paths["initialization_report"].read_text())
    if not report.get("accepted_for_replay_rl") or report["state_contract"]["reference_index"] != 0:
        raise RuntimeError("historical initialization is not the accepted endpoint-0 state")
    if report["state_contract"]["first_command_reference_index"] != 1:
        raise RuntimeError("historical first command is not reference row 1")
    with np.load(paths["initial_state"], allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in ("qpos", "qvel", "ctrl")}
    if values["qpos"].shape != (50,) or values["qvel"].shape != (48,) or values["ctrl"].shape != (36,):
        raise RuntimeError("accepted initial state dimensions changed")
    return values


def make_world(paths: dict[str, Path], initial: dict[str, np.ndarray]) -> MJWPVectorEnv:
    config = load_ego_config(paths["simulator_config"], device="cpu")
    reference = load_reference(config.data_path, device="cpu")
    objective = load_runtime_objective(paths["protocol"], paths["objective_profile"], tracking_variant="tool_only", require_run_ready=False)
    observation = load_runtime_observation(paths["protocol"], paths["observation_profile"], require_run_ready=False)
    residual, _ = load_residual_action_profile(paths["action_profile"])
    world = MJWPVectorEnv(
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
            object_assistance=None,
        ),
        seed=0,
    )
    tensors = [torch.as_tensor(initial[name][None], dtype=torch.float32) for name in ("qpos", "qvel", "ctrl")]
    world._write_state(*tensors, np.asarray([True]))
    world._last_ctrl = tensors[2].clone()
    # MJWarp 3.13 allocates ``geomcollisionid`` with ``wp.empty`` and marks it
    # TODO(set values) in io.py.  The historical reconstructed boundary had
    # this non-dynamical collision-sensor index initialized to -1, whereas a
    # fresh allocator can leave arbitrary integers in data_wp_prev.  Explicitly
    # reproduce the historical sentinel before capturing s0.  This field is
    # not used by the solver in this scene (there are no collision sensors),
    # but it is part of the complete snapshot contract and may not be ignored.
    for data in (world.env.data_wp, world.env.data_wp_prev):
        target = data.contact.geomcollisionid
        sentinel = torch.full(tuple(target.shape), -1, dtype=torch.int32)
        wp.copy(target, wp.from_torch(sentinel, dtype=target.dtype))
    wp.synchronize()
    world._check_capacity()
    return world


def array(world: MJWPVectorEnv, name: str, *, owner: str = "data") -> np.ndarray:
    target = world.env.data_wp if owner == "data" else getattr(world.env.data_wp, owner)
    return wp.to_torch(getattr(target, name)).cpu().numpy().copy()


def endpoint_row(world: MJWPVectorEnv) -> dict[str, np.ndarray]:
    return {name: array(world, name)[0] for name in ("qpos", "qvel", "ctrl")}


def equality_report(left: Any, right: Any) -> dict[str, Any]:
    if torch.is_tensor(left):
        left = left.detach().cpu().numpy()
    if torch.is_tensor(right):
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        same_shape = a.shape == b.shape
        equal = same_shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
        result: dict[str, Any] = {"equal": bool(equal), "shape_left": list(a.shape), "shape_right": list(b.shape), "dtype_left": str(a.dtype), "dtype_right": str(b.dtype)}
        if same_shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            result["max_abs"] = float(np.nanmax(diff)) if diff.size else 0.0
            result["unequal_count"] = int(np.count_nonzero(~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)))
        return result
    return {"equal": left == right, "left": left, "right": right}


def compare_snapshots(current: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    common = sorted(set(current) & set(historical))
    fields = {name: equality_report(current[name], historical[name]) for name in common}
    return {
        "common_field_count": len(common),
        "all_common_equal": all(row["equal"] for row in fields.values()),
        "unequal_fields": {name: row for name, row in fields.items() if not row["equal"]},
        "only_current": sorted(set(current) - set(historical)),
        "only_historical": sorted(set(historical) - set(current)),
        "fields": fields,
    }


def inactive_contact_padding_only(
    current: dict[str, Any], historical: dict[str, Any], comparison: dict[str, Any]
) -> bool:
    """Adjudicate—not hide—the sole undefined ``wp.empty`` padding mismatch."""
    unequal = set(comparison["unequal_fields"])
    if unequal != {"contact.geomcollisionid"} or comparison["only_current"] or comparison["only_historical"]:
        return False
    if not np.array_equal(np.asarray(current["nacon"]), np.asarray(historical["nacon"])):
        return False
    nacon = int(np.asarray(current["nacon"]).reshape(-1)[0])
    left = np.asarray(current["contact.geomcollisionid"])
    right = np.asarray(historical["contact.geomcollisionid"])
    return np.array_equal(left[:nacon], right[:nacon]) and not np.array_equal(left[nacon:], right[nacon:])


def save_snapshot(output: Path, label: str, state: dict[str, Any]) -> None:
    torch_gzip_write(output / "snapshots" / f"{label}.pt.gz", state)


def run_plain(world: MJWPVectorEnv, output: Path, ledger: BudgetLedger) -> tuple[dict[str, np.ndarray], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rows = [endpoint_row(world)]
    snapshots = {0: world.get_env_state()}
    infos: list[dict[str, Any]] = []
    zero = np.zeros((1, 36), dtype=np.float32)
    for source in range(ENDPOINTS):
        _, _, done, info = world.step(zero, auto_reset=False)
        ledger.charge(physics_steps=SUBSTEPS, control_intervals=1)
        if bool(done[0]):
            raise RuntimeError(f"plain Replay unexpectedly terminated at endpoint {source + 1}")
        rows.append(endpoint_row(world))
        infos.append({key: clone_array(value) for key, value in info.items()})
        if source + 1 in (15, 16, 20):
            snapshots[source + 1] = world.get_env_state()
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}, snapshots, infos


def model_roles(model: mujoco.MjModel) -> dict[str, Any]:
    def obj(kind: mujoco.mjtObj, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise RuntimeError(f"model is missing {name}")
        return value
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}" for i in range(model.ngeom)]
    return {
        "right_root": obj(mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"),
        "left_root": obj(mujoco.mjtObj.mjOBJ_BODY, "left_hand_link"),
        "tool_root": obj(mujoco.mjtObj.mjOBJ_BODY, "right_object"),
        "target_root": obj(mujoco.mjtObj.mjOBJ_BODY, "left_object"),
        "floor_geom": 0,
        "body_names": body_names,
        "geom_names": geom_names,
        "body_parentid": model.body_parentid.copy(),
        "geom_bodyid": model.geom_bodyid.copy(),
    }


class Observer:
    def __init__(self, world: MJWPVectorEnv):
        self.world = world
        self.rows: list[dict[str, Any]] = []
        self.contacts: list[dict[str, Any]] = []
        self.efc_force: list[np.ndarray] = []
        self.source = -1
        self.before_qpos = np.empty(0)
        self.before_qvel = np.empty(0)
        self.before_time = 0.0
        self.roles = model_roles(world.env.model_cpu)

    def begin(self, source: int) -> None:
        self.source = source
        self.before_qpos = array(self.world, "qpos")[0]
        self.before_qvel = array(self.world, "qvel")[0]
        self.before_time = float(array(self.world, "time")[0])

    def __call__(self, substep: int) -> None:
        index = transition_indices(self.source, substep)
        after_qpos = array(self.world, "qpos")[0]
        after_qvel = array(self.world, "qvel")[0]
        after_time = float(array(self.world, "time")[0])
        data = self.world.env.data_wp
        nacon = int(array(self.world, "nacon")[0])
        if nacon < 0 or nacon > int(self.world.ego_cfg.nconmax_per_env):
            raise RuntimeError(f"invalid active contact count {nacon}")
        local = np.zeros((nacon, 6), dtype=np.float32)
        world_force = np.zeros((nacon, 6), dtype=np.float32)
        if nacon:
            ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=wp.int32, device=self.world.env.device)
            local_wp = wp.zeros(nacon, dtype=wp.spatial_vector, device=self.world.env.device)
            world_wp = wp.zeros(nacon, dtype=wp.spatial_vector, device=self.world.env.device)
            mjwarp.contact_force(self.world.env.model_wp, data, ids, False, local_wp)
            mjwarp.contact_force(self.world.env.model_wp, data, ids, True, world_wp)
            wp.synchronize()
            local = local_wp.numpy().copy()
            world_force = world_wp.numpy().copy()
        contact_arrays = {
            name: wp.to_torch(getattr(data.contact, name)).cpu().numpy().copy()[:nacon]
            for name in ("worldid", "geom", "pos", "frame", "dist", "dim", "friction", "efc_address", "adhesion")
        }
        self.efc_force.append(array(self.world, "force", owner="efc")[0])
        for contact_id in range(nacon):
            geom1, geom2 = (int(value) for value in contact_arrays["geom"][contact_id])
            kwargs = dict(
                geom_body=self.roles["geom_bodyid"], body_parents=self.roles["body_parentid"],
                body_names=self.roles["body_names"], right_root=self.roles["right_root"],
                left_root=self.roles["left_root"], tool_root=self.roles["tool_root"],
                target_root=self.roles["target_root"], floor_geom=self.roles["floor_geom"],
            )
            role1 = classify_geom_role(geom1, **kwargs)
            role2 = classify_geom_role(geom2, **kwargs)
            self.contacts.append({
                **index, "contact_id": contact_id, "worldid": int(contact_arrays["worldid"][contact_id]),
                "geom1": geom1, "geom2": geom2, "geom1_name": self.roles["geom_names"][geom1],
                "geom2_name": self.roles["geom_names"][geom2], "role1": role1, "role2": role2,
                "group": canonical_contact_group(role1, role2), "pos": contact_arrays["pos"][contact_id].copy(),
                "frame": contact_arrays["frame"][contact_id].copy(), "dist": float(contact_arrays["dist"][contact_id]),
                "dim": int(contact_arrays["dim"][contact_id]), "friction": contact_arrays["friction"][contact_id].copy(),
                "efc_address": contact_arrays["efc_address"][contact_id].copy(), "adhesion": float(contact_arrays["adhesion"][contact_id]),
                "wrench_contact_force_torque": local[contact_id].copy(),
                "wrench_world_force_torque_on_geom2": world_force[contact_id].copy(),
            })
        selected = {}
        for name in ("ctrl", "actuator_force", "qfrc_actuator", "qfrc_constraint", "qfrc_applied", "xfrc_applied", "qfrc_passive", "qfrc_bias", "qacc", "cvel", "xpos", "xipos", "subtree_linvel", "overflow"):
            if hasattr(data, name):
                selected[name] = array(self.world, name)[0]
        self.rows.append({
            **index, "time_before": self.before_time, "time_after": after_time,
            "qpos_before": self.before_qpos.copy(), "qvel_before": self.before_qvel.copy(),
            "qpos_after": after_qpos.copy(), "qvel_after": after_qvel.copy(), "nacon": nacon, **selected,
        })
        self.before_qpos, self.before_qvel, self.before_time = after_qpos, after_qvel, after_time


def run_observed(world: MJWPVectorEnv, s0: dict[str, Any], ledger: BudgetLedger) -> tuple[dict[str, np.ndarray], dict[int, dict[str, Any]], Observer]:
    world.set_env_state(s0)
    rows = [endpoint_row(world)]
    snapshots = {0: world.get_env_state()}
    observer = Observer(world)
    zero = np.zeros((1, 36), dtype=np.float32)
    for source in range(ENDPOINTS):
        observer.begin(source)
        _, _, done, _ = world.step(zero, auto_reset=False, substep_observer=observer)
        ledger.charge(physics_steps=SUBSTEPS, control_intervals=1)
        if bool(done[0]):
            raise RuntimeError(f"observed Replay unexpectedly terminated at endpoint {source + 1}")
        rows.append(endpoint_row(world))
        if source + 1 in (15, 16, 20):
            snapshots[source + 1] = world.get_env_state()
    if len(observer.rows) != 200:
        raise RuntimeError("observer did not record exactly 200 substeps")
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}, snapshots, observer


def rows_to_npz(rows: list[dict[str, Any]], path: Path) -> None:
    keys = tuple(rows[0])
    arrays = {key: np.stack([np.asarray(row[key]) for row in rows]) for key in keys}
    np.savez_compressed(path, **arrays)


def contacts_to_npz(contacts: list[dict[str, Any]], path: Path) -> None:
    numeric = {}
    for key in ("source_endpoint", "outcome_endpoint", "command_reference_endpoint", "substep", "global_substep", "contact_id", "worldid", "geom1", "geom2", "pos", "frame", "dist", "dim", "friction", "efc_address", "adhesion", "wrench_contact_force_torque", "wrench_world_force_torque_on_geom2"):
        numeric[key] = np.stack([np.asarray(row[key]) for row in contacts]) if contacts else np.empty((0,))
    for key in ("geom1_name", "geom2_name", "role1", "role2", "group"):
        numeric[key] = np.asarray([row[key] for row in contacts])
    np.savez_compressed(path, **numeric)


def write_contact_csv(contacts: list[dict[str, Any]], path: Path) -> None:
    fields = ["global_substep", "source_endpoint", "outcome_endpoint", "substep", "contact_id", "geom1", "geom2", "geom1_name", "geom2_name", "role1", "role2", "group", "dist_m", "dim", "normal_force_N", "tangent_force_N", "world_fx_N", "world_fy_N", "world_fz_N", "pos_x_m", "pos_y_m", "pos_z_m"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in contacts:
            contact = row["wrench_contact_force_torque"]
            world = row["wrench_world_force_torque_on_geom2"]
            pos = row["pos"]
            writer.writerow({
                "global_substep": row["global_substep"], "source_endpoint": row["source_endpoint"],
                "outcome_endpoint": row["outcome_endpoint"], "substep": row["substep"], "contact_id": row["contact_id"],
                "geom1": row["geom1"], "geom2": row["geom2"], "geom1_name": row["geom1_name"], "geom2_name": row["geom2_name"],
                "role1": row["role1"], "role2": row["role2"], "group": row["group"], "dist_m": row["dist"], "dim": row["dim"],
                "normal_force_N": contact[0], "tangent_force_N": float(np.linalg.norm(contact[1:3])),
                "world_fx_N": world[0], "world_fy_N": world[1], "world_fz_N": world[2],
                "pos_x_m": pos[0], "pos_y_m": pos[1], "pos_z_m": pos[2],
            })


def rotation_error(actual: np.ndarray, reference: np.ndarray) -> float:
    a = Rotation.from_quat(actual[[1, 2, 3, 0]])
    r = Rotation.from_quat(reference[[1, 2, 3, 0]])
    return float((r.inv() * a).magnitude())


def write_motion_csv(endpoints: dict[str, np.ndarray], substeps: list[dict[str, Any]], reference_qpos: np.ndarray, path: Path) -> None:
    fields = ["sample", "endpoint", "global_substep", "time_s", "tool_error_x_m", "tool_error_y_m", "tool_error_z_m", "tool_error_norm_m", "tool_rotation_error_rad", "target_error_x_m", "target_error_y_m", "target_error_z_m", "target_error_norm_m", "target_rotation_error_rad", "pair_error_x_m", "pair_error_y_m", "pair_error_z_m", "pair_error_norm_m", "tool_speed_m_s", "target_speed_m_s"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        samples = [("endpoint", i, i * SUBSTEPS, i / 30.0, endpoints["qpos"][i], endpoints["qvel"][i]) for i in range(21)]
        samples += [("substep", row["outcome_endpoint"], row["global_substep"], row["time_after"], row["qpos_after"], row["qvel_after"]) for row in substeps]
        for kind, endpoint, global_step, time_s, qpos, qvel in samples:
            ref_index = endpoint if kind == "endpoint" else min(endpoint, len(reference_qpos) - 1)
            ref = reference_qpos[ref_index]
            te, ge = qpos[36:39] - ref[36:39], qpos[43:46] - ref[43:46]
            pair = (qpos[36:39] - qpos[43:46]) - (ref[36:39] - ref[43:46])
            writer.writerow({"sample": kind, "endpoint": endpoint, "global_substep": global_step, "time_s": time_s,
                "tool_error_x_m": te[0], "tool_error_y_m": te[1], "tool_error_z_m": te[2], "tool_error_norm_m": np.linalg.norm(te),
                "tool_rotation_error_rad": rotation_error(qpos[39:43], ref[39:43]),
                "target_error_x_m": ge[0], "target_error_y_m": ge[1], "target_error_z_m": ge[2], "target_error_norm_m": np.linalg.norm(ge),
                "target_rotation_error_rad": rotation_error(qpos[46:50], ref[46:50]),
                "pair_error_x_m": pair[0], "pair_error_y_m": pair[1], "pair_error_z_m": pair[2], "pair_error_norm_m": np.linalg.norm(pair),
                "tool_speed_m_s": np.linalg.norm(qvel[36:39]), "target_speed_m_s": np.linalg.norm(qvel[42:45])})


def execute(output: Path) -> None:
    manifest_path = output / "input_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("preflight input_manifest.json must exist before integration")
    manifest = json.loads(manifest_path.read_text())
    retry_path = output / "retry_authorization.json"
    retry = json.loads(retry_path.read_text()) if retry_path.is_file() else None
    allowed_status = {"preflight_complete_no_environment_constructed"}
    if retry is not None:
        allowed_status.add("replay_identity_not_established_stopped_before_observer_B")
    if manifest.get("status") not in allowed_status:
        raise RuntimeError("input manifest is not an unused preflight")
    paths = input_paths()
    for name, row in manifest["inputs"].items():
        if artifact(paths[name]) != row:
            raise RuntimeError(f"input changed after preflight: {name}")
    initial = load_initial(paths)
    prior = retry.get("prior_attempt_cost", {}) if retry is not None else {}
    ledger = BudgetLedger(
        physics_steps=int(prior.get("physics_steps", 0)),
        control_intervals=int(prior.get("control_intervals", 0)),
    )
    world = make_world(paths, initial)
    # One native MuJoCo setup step is part of the historical constructor.
    ledger.charge(physics_steps=1, control_intervals=0)
    s0 = world.get_env_state()
    save_snapshot(output, "A_s0_reconstructed", s0)
    a_endpoints, a_snapshots, _ = run_plain(world, output, ledger)
    for endpoint, state in a_snapshots.items():
        save_snapshot(output, f"A_s{endpoint}", state)
    with np.load(paths["historical_trajectory"], allow_pickle=False) as archive:
        historical = {name: archive[name].copy() for name in ("qpos", "qvel", "ctrl")}
    a_history = {name: equality_report(a_endpoints[name], historical[name]) for name in historical}
    historical_s20 = torch_gzip_read(paths["historical_s20"])
    s20_comparison = compare_snapshots(a_snapshots[20], historical_s20)
    parity = {
        "schema": TRACE_SCHEMA, "A_vs_historical_endpoints": a_history,
        "A_endpoint_arrays_bitwise_equal": all(row["equal"] for row in a_history.values()),
        "A_s20_vs_historical_full_snapshot": s20_comparison,
        "B_executed": False,
    }
    write_json(output / "replay_parity.json", parity)
    if not parity["A_endpoint_arrays_bitwise_equal"] or not s20_comparison["all_common_equal"] or s20_comparison["only_current"] or s20_comparison["only_historical"]:
        manifest["status"] = "replay_identity_not_established_stopped_before_observer_B"
        manifest["actual_cost"] = vars(ledger)
        write_json(manifest_path, manifest)
        raise RuntimeError("plain Replay A did not reproduce historical evidence bitwise")
    b_endpoints, b_snapshots, observer = run_observed(world, s0, ledger)
    for endpoint, state in b_snapshots.items():
        save_snapshot(output, f"B_s{endpoint}", state)
    b_a = {name: equality_report(b_endpoints[name], a_endpoints[name]) for name in a_endpoints}
    b_history = {name: equality_report(b_endpoints[name], historical[name]) for name in historical}
    parity.update({
        "B_executed": True, "B_vs_A_endpoints": b_a, "B_vs_historical_endpoints": b_history,
        "B_endpoint_arrays_bitwise_equal_A": all(row["equal"] for row in b_a.values()),
        "B_endpoint_arrays_bitwise_equal_historical": all(row["equal"] for row in b_history.values()),
        "B_s20_vs_A_s20": compare_snapshots(b_snapshots[20], a_snapshots[20]),
        "historical_substep_force_claim": "not available historically; forces are measured on this bitwise reproduced trajectory",
    })
    write_json(output / "replay_parity.json", parity)
    if not parity["B_endpoint_arrays_bitwise_equal_A"] or not parity["B_s20_vs_A_s20"]["all_common_equal"]:
        raise RuntimeError("passive observer changed Replay result")
    np.savez_compressed(output / "endpoints.npz", **{f"A_{k}": v for k, v in a_endpoints.items()}, **{f"B_{k}": v for k, v in b_endpoints.items()}, **{f"historical_{k}": v for k, v in historical.items()}, reference_endpoint=np.arange(21, dtype=np.int64))
    rows_to_npz(observer.rows, output / "substeps.npz")
    contacts_to_npz(observer.contacts, output / "contacts_raw.npz")
    np.savez_compressed(output / "efc_force_by_substep.npz", efc_force=np.stack(observer.efc_force))
    write_contact_csv(observer.contacts, output / "contact_forces.csv")
    with np.load(paths["reference"], allow_pickle=False) as archive:
        reference_qpos = archive["qpos"].copy()
    write_motion_csv(b_endpoints, observer.rows, reference_qpos, output / "object_pair_motion.csv")
    manifest["status"] = "replay_identity_established_contact_trace_recorded_visual_review_pending"
    if retry is not None:
        manifest["paired_retest"] = retry
    manifest["actual_cost"] = {**vars(ledger), "setup_native_mj_steps": 1, "A_task_physics_steps": 200, "B_task_physics_steps": 200, "actor_or_critic_forwards": 0, "optimizer_updates": 0}
    manifest["model"] = {
        "nq": world.env.model_cpu.nq, "nv": world.env.model_cpu.nv, "nu": world.env.model_cpu.nu,
        "nbody": world.env.model_cpu.nbody, "ngeom": world.env.model_cpu.ngeom,
        "timestep": world.env.model_cpu.opt.timestep, "integrator": int(world.env.model_cpu.opt.integrator),
        "cone": int(world.env.model_cpu.opt.cone), "iterations": world.env.model_cpu.opt.iterations,
        "ls_iterations": world.env.model_cpu.opt.ls_iterations,
        "body_mass": world.env.model_cpu.body_mass.tolist(), "body_inertia": world.env.model_cpu.body_inertia.tolist(),
        "dof_armature": world.env.model_cpu.dof_armature.tolist(), "dof_damping": world.env.model_cpu.dof_damping.tolist(),
    }
    manifest["recorded"] = {"endpoints": 21, "substeps": len(observer.rows), "raw_contacts": len(observer.contacts), "snapshots": 8}
    write_json(manifest_path, manifest)


def resume_observer_b(output: Path) -> None:
    """Finish observer B after the one authorized retest's A was adjudicated.

    This never repeats A.  The saved complete retest s0 is restored into a new
    instance and B is compared against the saved retest A s20 and the already
    proven historical endpoint arrays.
    """
    manifest_path = output / "input_manifest.json"
    parity_path = output / "replay_parity.json"
    authorization_path = output / "resume_b_authorization.json"
    if not (manifest_path.is_file() and parity_path.is_file() and authorization_path.is_file()):
        raise RuntimeError("resume-B requires manifest, retest-A parity, and explicit authorization")
    manifest = json.loads(manifest_path.read_text())
    parity = json.loads(parity_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    if parity.get("B_executed") or not parity.get("A_endpoint_arrays_bitwise_equal"):
        raise RuntimeError("saved A is not the expected endpoint-exact, B-pending retest")
    paths = input_paths()
    for name, row in manifest["inputs"].items():
        if artifact(paths[name]) != row:
            raise RuntimeError(f"input changed before resume-B: {name}")
    a_s0 = torch_gzip_read(output / "snapshots/A_s0_reconstructed.pt.gz")
    a_s20 = torch_gzip_read(output / "snapshots/A_s20.pt.gz")
    historical_s20 = torch_gzip_read(paths["historical_s20"])
    comparison = compare_snapshots(a_s20, historical_s20)
    padding_only = inactive_contact_padding_only(a_s20, historical_s20, comparison)
    if not padding_only:
        raise RuntimeError("saved retest A has more than the adjudicated inactive-padding difference")
    parity["A_s20_vs_historical_full_snapshot"] = comparison
    parity["A_s20_physics_identity_adjudication"] = {
        "status": "all_fields_equal_except_undefined_inactive_contact_padding",
        "field": "contact.geomcollisionid",
        "nacon": int(np.asarray(a_s20["nacon"]).reshape(-1)[0]),
        "mismatch_is_strictly_after_nacon": True,
        "active_contact_rows_equal": True,
        "reason": "installed MJWarp io.py allocates this field with wp.empty; padding is not an active contact and is never decoded",
        "evidence_level": "full reconstructed state with explicit nonphysical padding exception; not a historically serialized s0",
    }
    prior = authorization["prior_attempt_cost"]
    ledger = BudgetLedger(physics_steps=int(prior["physics_steps"]), control_intervals=int(prior["control_intervals"]))
    world = make_world(paths, load_initial(paths))
    ledger.charge(physics_steps=1, control_intervals=0)
    b_endpoints, b_snapshots, observer = run_observed(world, a_s0, ledger)
    for endpoint, state in b_snapshots.items():
        save_snapshot(output, f"B_s{endpoint}", state)
    with np.load(paths["historical_trajectory"], allow_pickle=False) as archive:
        historical = {name: archive[name].copy() for name in ("qpos", "qvel", "ctrl")}
    b_history = {name: equality_report(b_endpoints[name], historical[name]) for name in historical}
    b_a_s20 = compare_snapshots(b_snapshots[20], a_s20)
    parity.update({
        "B_executed": True,
        "B_vs_A_endpoints": b_history,
        "B_vs_historical_endpoints": b_history,
        "B_endpoint_arrays_bitwise_equal_A": all(row["equal"] for row in b_history.values()),
        "B_endpoint_arrays_bitwise_equal_historical": all(row["equal"] for row in b_history.values()),
        "B_s20_vs_A_s20": b_a_s20,
        "historical_substep_force_claim": "not available historically; forces are measured on this endpoint-bitwise reproduced trajectory",
        "resume_B_authorization": authorization,
    })
    write_json(parity_path, parity)
    if not parity["B_endpoint_arrays_bitwise_equal_A"] or not b_a_s20["all_common_equal"] or b_a_s20["only_current"] or b_a_s20["only_historical"]:
        raise RuntimeError("passive observer B changed the saved retest A result")
    np.savez_compressed(output / "endpoints.npz", **{f"A_{k}": v for k, v in historical.items()}, **{f"B_{k}": v for k, v in b_endpoints.items()}, **{f"historical_{k}": v for k, v in historical.items()}, reference_endpoint=np.arange(21, dtype=np.int64))
    rows_to_npz(observer.rows, output / "substeps.npz")
    contacts_to_npz(observer.contacts, output / "contacts_raw.npz")
    np.savez_compressed(output / "efc_force_by_substep.npz", efc_force=np.stack(observer.efc_force))
    write_contact_csv(observer.contacts, output / "contact_forces.csv")
    with np.load(paths["reference"], allow_pickle=False) as archive:
        reference_qpos = archive["qpos"].copy()
    write_motion_csv(b_endpoints, observer.rows, reference_qpos, output / "object_pair_motion.csv")
    manifest["status"] = "replay_identity_established_contact_trace_recorded_visual_review_pending"
    manifest["paired_retest"] = json.loads((output / "retry_authorization.json").read_text())
    manifest["resume_B"] = authorization
    manifest["actual_cost"] = {**vars(ledger), "setup_native_mj_steps": 3, "A_task_physics_steps": 400, "B_task_physics_steps": 200, "actor_or_critic_forwards": 0, "optimizer_updates": 0}
    manifest["recorded"] = {"endpoints": 21, "substeps": len(observer.rows), "raw_contacts": len(observer.contacts), "snapshots": 8}
    manifest["model"] = {"nq": world.env.model_cpu.nq, "nv": world.env.model_cpu.nv, "nu": world.env.model_cpu.nu, "nbody": world.env.model_cpu.nbody, "ngeom": world.env.model_cpu.ngeom, "timestep": world.env.model_cpu.opt.timestep, "integrator": int(world.env.model_cpu.opt.integrator), "cone": int(world.env.model_cpu.opt.cone), "iterations": world.env.model_cpu.opt.iterations, "ls_iterations": world.env.model_cpu.opt.ls_iterations, "body_mass": world.env.model_cpu.body_mass.tolist(), "body_inertia": world.env.model_cpu.body_inertia.tolist(), "dof_armature": world.env.model_cpu.dof_armature.tolist(), "dof_damping": world.env.model_cpu.dof_damping.tolist()}
    write_json(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Startup-transition isolation mode.  This is deliberately local to the
# already-audited Replay tracer: it reuses the exact observer and snapshot
# machinery without widening the active RL core.
# ---------------------------------------------------------------------------


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def startup_input_paths() -> dict[str, Path]:
    paths = input_paths()
    paths.update(
        {
            "startup_contract": repo_root() / "configs/taco_pour_startup_transition_isolation_v1.yaml",
            "trace_manifest": STARTUP_TRACE / "input_manifest.json",
            "trace_replay_parity": STARTUP_TRACE / "replay_parity.json",
            "trace_endpoints": STARTUP_TRACE / "endpoints.npz",
            "trace_substeps": STARTUP_TRACE / "substeps.npz",
            "trace_contacts": STARTUP_TRACE / "contacts_raw.npz",
            "trace_efc_force": STARTUP_TRACE / "efc_force_by_substep.npz",
            "trace_s0_reconstructed": STARTUP_SNAPSHOT,
            "trace_s0_a": STARTUP_TRACE / "snapshots/A_s0.pt.gz",
            "trace_s0_b": STARTUP_TRACE / "snapshots/B_s0.pt.gz",
            "trace_s20_b": STARTUP_TRACE / "snapshots/B_s20.pt.gz",
        }
    )
    return paths


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
    ).strip()


def _require_baseline_ancestor(head: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTUP_BASELINE, head],
        cwd=repo_root(),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"startup audit baseline {STARTUP_BASELINE} is not an ancestor of {head}")


def startup_preflight(output: Path) -> None:
    """Freeze inputs and controls without constructing a physics environment."""
    if output.exists():
        raise FileExistsError(f"immutable startup diagnostic output already exists: {output}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root(), text=True).strip():
        raise RuntimeError("startup preflight requires a clean implementation worktree")
    head = _git_head()
    _require_baseline_ancestor(head)
    paths = startup_input_paths()
    artifacts = {name: artifact(path) for name, path in paths.items()}
    trace_manifest = json.loads(paths["trace_manifest"].read_text())
    trace_parity = json.loads(paths["trace_replay_parity"].read_text())
    if not trace_parity.get("B_endpoint_arrays_bitwise_equal_historical"):
        raise RuntimeError("historical Replay 0->20 identity is not established")
    if trace_manifest.get("status") != "replay_identity_established_contact_trace_and_visual_review_complete":
        raise RuntimeError("historical Replay trace has not completed review")

    s0 = torch_gzip_read(paths["trace_s0_reconstructed"])
    s0_a = torch_gzip_read(paths["trace_s0_a"])
    s0_b = torch_gzip_read(paths["trace_s0_b"])
    a_parity = compare_snapshots(s0, s0_a)
    b_parity = compare_snapshots(s0, s0_b)
    if not (
        a_parity["all_common_equal"]
        and b_parity["all_common_equal"]
        and not a_parity["only_current"]
        and not a_parity["only_historical"]
        and not b_parity["only_current"]
        and not b_parity["only_historical"]
    ):
        raise RuntimeError("the three saved trace s0 snapshots are not bitwise identical")
    with np.load(paths["reference"], allow_pickle=False) as archive:
        reference_ctrl = archive["ctrl"].astype(np.float32)
    initial_ctrl = s0["ctrl"].detach().cpu().numpy()[0].astype(np.float32, copy=True)
    controls = startup_control_sequences(initial_ctrl, reference_ctrl)

    output.mkdir(parents=True)
    np.savez_compressed(output / "control_plan.npz", **controls)
    manifest = {
        "schema": STARTUP_SCHEMA,
        "status": "preflight_complete_no_environment_constructed",
        "baseline_git_commit": STARTUP_BASELINE,
        "implementation_git_commit": head,
        "inputs": artifacts,
        "s0_provenance": {
            "selected": artifacts["trace_s0_reconstructed"],
            "origin": trace_manifest["s0_source"],
            "complete_snapshot_field_count": len(s0),
            "A_s0_reconstructed_equals_A_s0_bitwise": True,
            "A_s0_reconstructed_equals_B_s0_bitwise": True,
            "comparison_A": a_parity,
            "comparison_B": b_parity,
        },
        "control_plan": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in controls.items()
        },
        "fixed_order": ["ORIGINAL", "HOLD_1", "BLEND_5", "BLEND_5_COLD"],
        "planned_cost": {
            "setup_native_mj_steps": 2,
            "ORIGINAL_physics_steps": 200,
            "HOLD_1_physics_steps": 10,
            "BLEND_5_physics_steps": 200,
            "BLEND_5_COLD_physics_steps": 200,
            "total_physics_steps": 612,
            "task_control_intervals": 61,
            "physics_ceiling": 1000,
            "control_interval_ceiling": 81,
        },
        "runtime_contract": {
            "device": "cpu",
            "actor_or_critic_forwards": 0,
            "optimizer_updates": 0,
            "object_assistance": None,
            "action_noise": 0.0,
            "auto_reset": False,
            "tracking_variant": "tool_only",
            "diagnostic_only": True,
            "chunk_commit": False,
        },
    }
    write_json(output / "input_manifest.json", manifest)


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _compare_array_maps(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    fields = {
        key: equality_report(left[key], right[key])
        for key in keys
        if key in left and key in right
    }
    return {
        "all_equal": (
            set(left) == set(right) and all(row["equal"] for row in fields.values())
        ),
        "only_left": sorted(set(left) - set(right)),
        "only_right": sorted(set(right) - set(left)),
        "unequal_fields": {key: row for key, row in fields.items() if not row["equal"]},
        "field_count": len(fields),
    }


def _observer_arrays(observer: Observer) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = {
        key: np.stack([np.asarray(row[key]) for row in observer.rows])
        for key in observer.rows[0]
    }
    contacts: dict[str, np.ndarray] = {}
    for key in (
        "source_endpoint", "outcome_endpoint", "command_reference_endpoint", "substep",
        "global_substep", "contact_id", "worldid", "geom1", "geom2", "pos", "frame",
        "dist", "dim", "friction", "efc_address", "adhesion",
        "wrench_contact_force_torque", "wrench_world_force_torque_on_geom2",
    ):
        contacts[key] = (
            np.stack([np.asarray(row[key]) for row in observer.contacts])
            if observer.contacts else np.empty((0,))
        )
    for key in ("geom1_name", "geom2_name", "role1", "role2", "group"):
        contacts[key] = np.asarray([row[key] for row in observer.contacts])
    efc = {"efc_force": np.stack(observer.efc_force)}
    return rows, contacts, efc


def _save_condition(
    output: Path,
    label: str,
    endpoints: dict[str, np.ndarray],
    snapshots: dict[int, dict[str, Any]],
    observer: Observer,
    result: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = output / "conditions" / label
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "endpoints.npz", **endpoints)
    rows, contacts, efc = _observer_arrays(observer)
    np.savez_compressed(directory / "substeps.npz", **rows)
    np.savez_compressed(directory / "contacts_raw.npz", **contacts)
    np.savez_compressed(directory / "efc_force_by_substep.npz", **efc)
    write_contact_csv(observer.contacts, directory / "contact_forces.csv")
    for endpoint, state in snapshots.items():
        torch_gzip_write(directory / "snapshots" / f"s{endpoint}.pt.gz", state)
    write_json(directory / "result.json", result)
    return rows, contacts, efc


def _exact_snapshot_restored(world: MJWPVectorEnv, s0: dict[str, Any]) -> dict[str, Any]:
    world.set_env_state(s0)
    comparison = compare_snapshots(world.get_env_state(), s0)
    if not (
        comparison["all_common_equal"]
        and not comparison["only_current"]
        and not comparison["only_historical"]
    ):
        raise RuntimeError("full s0 did not restore bitwise")
    return comparison


def _run_startup_condition(
    world: MJWPVectorEnv,
    s0: dict[str, Any],
    label: str,
    actions: np.ndarray,
    expected_ctrl: np.ndarray,
    ledger: BudgetLedger,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, Any]], Observer, dict[str, Any]]:
    restore = _exact_snapshot_restored(world, s0)
    s0_after_restore = world.get_env_state()
    rows = [endpoint_row(world)]
    snapshots = {0: s0_after_restore}
    observer = Observer(world)
    infos: list[dict[str, Any]] = []
    terminated = False
    failure_endpoint: int | None = None
    for source in range(len(actions)):
        observer.begin(source)
        _, _, done, info = world.step(
            actions[source : source + 1], auto_reset=False, substep_observer=observer
        )
        ledger.charge(physics_steps=SUBSTEPS, control_intervals=1)
        rows.append(endpoint_row(world))
        infos.append({key: clone_array(value) for key, value in info.items()})
        if not np.array_equal(rows[-1]["ctrl"], expected_ctrl[source], equal_nan=True):
            raise RuntimeError(f"{label} realized ctrl differs from its validated command at source {source}")
        endpoint = source + 1
        if endpoint in STARTUP_KEYS:
            snapshots[endpoint] = world.get_env_state()
        if bool(done[0]):
            terminated = True
            failure_endpoint = endpoint
            break
    endpoints = {
        name: np.stack([row[name] for row in rows]) for name in rows[0]
    }
    if len(observer.rows) != (len(rows) - 1) * SUBSTEPS:
        raise RuntimeError(f"{label} observer substep count is incomplete")
    result = {
        "schema": STARTUP_SCHEMA,
        "condition": label,
        "requested_control_intervals": int(len(actions)),
        "executed_control_intervals": int(len(rows) - 1),
        "physics_steps": int((len(rows) - 1) * SUBSTEPS),
        "terminated": terminated,
        "failure_endpoint": failure_endpoint,
        "full_s0_restore_bitwise": True,
        "s0_restore_comparison": restore,
        "actor_or_critic_forwards": 0,
        "optimizer_updates": 0,
    }
    return endpoints, snapshots, observer, result


def _validate_startup_controls(
    world: MJWPVectorEnv,
    s0: dict[str, Any],
    frozen_plan: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = world.ctrl_ref[: ENDPOINTS + 1].detach().cpu().numpy().astype(np.float32)
    initial = s0["ctrl"].detach().cpu().numpy()[0].astype(np.float32, copy=True)
    plan = startup_control_sequences(initial, reference)
    for name, value in frozen_plan.items():
        if name not in plan or not np.array_equal(value, plan[name], equal_nan=True):
            raise RuntimeError(f"runtime control plan changed after preflight: {name}")
    model = world.env.model_cpu
    indices = np.asarray(world.env_cfg.residual.hand_control_indices, dtype=np.int64)
    if not np.array_equal(indices, np.arange(model.nu)):
        raise RuntimeError("startup audit requires all 36 actuator controls in canonical order")
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)[indices]
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)[indices]
    joint_ids = np.asarray(model.actuator_trnid, dtype=np.int64)[indices, 0]
    joint_types = np.asarray(model.jnt_type, dtype=np.int64)[joint_ids]
    units = control_unit_labels(
        joint_types,
        slide_type=int(mujoco.mjtJoint.mjJNT_SLIDE),
        hinge_type=int(mujoco.mjtJoint.mjJNT_HINGE),
    )
    slide = joint_types == int(mujoco.mjtJoint.mjJNT_SLIDE)
    hinge = joint_types == int(mujoco.mjtJoint.mjJNT_HINGE)
    if int(slide.sum()) != 6 or int(hinge.sum()) != 30:
        raise RuntimeError("actuator unit partition changed from 6 slide + 30 hinge")

    realized: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "residual_scale": float(world.env_cfg.residual.residual_scale),
        "residual_clip": float(world.env_cfg.residual.residual_clip),
        "slide_components_m": int(slide.sum()),
        "hinge_components_rad": int(hinge.sum()),
        "per_component_units": units.tolist(),
        "conditions": {},
    }
    mapping = {
        "ORIGINAL": (plan["original_desired_ctrl"], plan["original_residual_action"], reference[1:21]),
        "HOLD_1": (plan["hold_1_desired_ctrl"], plan["hold_1_residual_action"], reference[1:2]),
        "BLEND_5": (plan["blend_5_desired_ctrl"], plan["blend_5_residual_action"], reference[1:21]),
    }
    for label, (desired, action, base) in mapping.items():
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        base_tensor = torch.as_tensor(base, dtype=torch.float32)
        actual = world._apply_residual(base_tensor, action_tensor).detach().cpu().numpy()
        residual = actual - base
        error = actual.astype(np.float64) - desired.astype(np.float64)
        if not np.isfinite(action).all() or float(np.max(np.abs(action))) > 1.0:
            raise RuntimeError(f"{label} action leaves normalized support")
        if float(np.max(np.abs(residual))) > 0.05 + 1e-8:
            raise RuntimeError(f"{label} physical residual leaves +/-0.05")
        below = np.maximum(ranges[None, :, 0] - actual.astype(np.float64), 0.0)
        above = np.maximum(actual.astype(np.float64) - ranges[None, :, 1], 0.0)
        violation = np.maximum(below, above)[:, limited]
        base_below = np.maximum(ranges[None, :, 0] - base.astype(np.float64), 0.0)
        base_above = np.maximum(base.astype(np.float64) - ranges[None, :, 1], 0.0)
        base_violation = np.maximum(base_below, base_above)[:, limited]
        maximum_violation = float(np.max(violation)) if violation.size else 0.0
        if maximum_violation > 2e-7 or np.any(violation > base_violation + 1e-12):
            raise RuntimeError(f"{label} desired control leaves or worsens actuator ctrlrange")
        slide_error = float(np.max(np.abs(error[:, slide])))
        hinge_error = float(np.max(np.abs(error[:, hinge])))
        if slide_error > 2e-7 or hinge_error > 2e-7:
            raise RuntimeError(f"{label} residual encoding exceeds 2e-7 tolerance")
        realized[label] = actual.astype(np.float32, copy=True)
        report["conditions"][label] = {
            "action_min": float(action.min()),
            "action_max": float(action.max()),
            "maximum_absolute_normalized_action": float(np.max(np.abs(action))),
            "maximum_absolute_physical_residual": float(np.max(np.abs(residual))),
            "maximum_encoding_error_slide_m": slide_error,
            "maximum_encoding_error_hinge_rad": hinge_error,
            "normalized_support_clips": 0,
            "physical_residual_clips": 0,
            "ctrlrange_clips": 0,
            "baseline_tolerance_level_ctrlrange_components": int(np.count_nonzero(violation)),
            "maximum_ctrlrange_violation": maximum_violation,
        }
    if not np.array_equal(realized["ORIGINAL"], reference[1:21]):
        raise RuntimeError("ORIGINAL is not exact zero-residual Replay")
    if not np.array_equal(plan["blend_5_desired_ctrl"][4:], reference[5:21]):
        raise RuntimeError("BLEND_5 is not exact Replay from source4->5 onward")
    return {**plan, **{f"{key}_realized_ctrl": value for key, value in realized.items()}}, report


def _historical_original_parity(
    output: Path,
    paths: dict[str, Path],
    endpoints: dict[str, np.ndarray],
    snapshots: dict[int, dict[str, Any]],
    observer: Observer,
) -> dict[str, Any]:
    historical_endpoints = _npz_arrays(paths["trace_endpoints"])
    expected_endpoints = {
        name: historical_endpoints[f"B_{name}"] for name in ("qpos", "qvel", "ctrl")
    }
    endpoint_parity = _compare_array_maps(endpoints, expected_endpoints)
    rows, contacts, efc = _observer_arrays(observer)
    substep_parity = _compare_array_maps(rows, _npz_arrays(paths["trace_substeps"]))
    contact_parity = _compare_array_maps(contacts, _npz_arrays(paths["trace_contacts"]))
    efc_parity = _compare_array_maps(efc, _npz_arrays(paths["trace_efc_force"]))
    s20_parity = compare_snapshots(snapshots[20], torch_gzip_read(paths["trace_s20_b"]))
    report = {
        "schema": STARTUP_SCHEMA,
        "historical_trace": artifact(paths["trace_manifest"]),
        "endpoints": endpoint_parity,
        "substeps": substep_parity,
        "contacts": contact_parity,
        "efc_force": efc_parity,
        "full_s20_snapshot": s20_parity,
        "all_bitwise_equal": bool(
            endpoint_parity["all_equal"]
            and substep_parity["all_equal"]
            and contact_parity["all_equal"]
            and efc_parity["all_equal"]
            and s20_parity["all_common_equal"]
            and not s20_parity["only_current"]
            and not s20_parity["only_historical"]
        ),
        "padding_exception_used": False,
    }
    write_json(output / "replay_parity.json", report)
    return report


def startup_execute(output: Path) -> None:
    manifest_path = output / "input_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("startup preflight must run before integration")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "preflight_complete_no_environment_constructed":
        raise RuntimeError("startup output is not an unused preflight")
    if _git_head() != manifest["implementation_git_commit"]:
        raise RuntimeError("implementation HEAD changed after startup preflight")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root(), text=True).strip():
        raise RuntimeError("startup integration requires a clean worktree")
    paths = startup_input_paths()
    for name, row in manifest["inputs"].items():
        if artifact(paths[name]) != row:
            raise RuntimeError(f"startup input changed after preflight: {name}")
    frozen_plan = _npz_arrays(output / "control_plan.npz")
    for name, row in manifest["control_plan"].items():
        if _array_sha256(frozen_plan[name]) != row["sha256"]:
            raise RuntimeError(f"startup control plan hash changed: {name}")

    s0 = torch_gzip_read(paths["trace_s0_reconstructed"])
    s0_before = {key: value.clone() if torch.is_tensor(value) else value for key, value in s0.items()}
    ledger = BudgetLedger(limit_physics_steps=1000, limit_control_intervals=81)
    ledger.require_capacity(physics_steps=612, control_intervals=61)
    world = make_world(paths, load_initial(paths))
    ledger.charge(physics_steps=1, control_intervals=0)
    controls, control_report = _validate_startup_controls(world, s0, frozen_plan)
    np.savez_compressed(output / "control_sequences.npz", **controls)
    write_json(output / "control_validation.json", control_report)

    results: dict[str, Any] = {}
    recorded: dict[str, tuple[dict[str, np.ndarray], dict[int, dict[str, Any]], Observer]] = {}
    ledger.require_capacity(physics_steps=611, control_intervals=61)
    original = _run_startup_condition(
        world, s0, "ORIGINAL", controls["original_residual_action"],
        controls["ORIGINAL_realized_ctrl"], ledger,
    )
    recorded["ORIGINAL"] = original[:3]
    results["ORIGINAL"] = original[3]
    _save_condition(output, "ORIGINAL", *original)
    parity = _historical_original_parity(output, paths, *original[:3])
    if not parity["all_bitwise_equal"]:
        manifest["status"] = "original_identity_failed_stopped"
        manifest["actual_cost"] = vars(ledger)
        write_json(manifest_path, manifest)
        raise RuntimeError("ORIGINAL did not reproduce the historical trace bitwise")

    ledger.require_capacity(physics_steps=411, control_intervals=41)
    hold = _run_startup_condition(
        world, s0, "HOLD_1", controls["hold_1_residual_action"],
        controls["HOLD_1_realized_ctrl"], ledger,
    )
    recorded["HOLD_1"] = hold[:3]
    results["HOLD_1"] = hold[3]
    _save_condition(output, "HOLD_1", *hold)

    ledger.require_capacity(physics_steps=401, control_intervals=40)
    blend = _run_startup_condition(
        world, s0, "BLEND_5", controls["blend_5_residual_action"],
        controls["BLEND_5_realized_ctrl"], ledger,
    )
    recorded["BLEND_5"] = blend[:3]
    results["BLEND_5"] = blend[3]
    _save_condition(output, "BLEND_5", *blend)

    ledger.require_capacity(physics_steps=201, control_intervals=20)
    cold_world = make_world(paths, load_initial(paths))
    ledger.charge(physics_steps=1, control_intervals=0)
    cold = _run_startup_condition(
        cold_world, s0, "BLEND_5_COLD", controls["blend_5_residual_action"],
        controls["BLEND_5_realized_ctrl"], ledger,
    )
    recorded["BLEND_5_COLD"] = cold[:3]
    results["BLEND_5_COLD"] = cold[3]
    _save_condition(output, "BLEND_5_COLD", *cold)

    blend_rows, blend_contacts, blend_efc = _observer_arrays(blend[2])
    cold_rows, cold_contacts, cold_efc = _observer_arrays(cold[2])
    cold_parity = {
        "schema": STARTUP_SCHEMA,
        "saved_action_array_sha256": _array_sha256(controls["blend_5_residual_action"]),
        "endpoints": _compare_array_maps(blend[0], cold[0]),
        "substeps": _compare_array_maps(blend_rows, cold_rows),
        "contacts": _compare_array_maps(blend_contacts, cold_contacts),
        "efc_force": _compare_array_maps(blend_efc, cold_efc),
        "full_s20_snapshot": compare_snapshots(blend[1][20], cold[1][20]),
    }
    cold_parity["all_bitwise_equal"] = bool(
        cold_parity["endpoints"]["all_equal"]
        and cold_parity["substeps"]["all_equal"]
        and cold_parity["contacts"]["all_equal"]
        and cold_parity["efc_force"]["all_equal"]
        and cold_parity["full_s20_snapshot"]["all_common_equal"]
        and not cold_parity["full_s20_snapshot"]["only_current"]
        and not cold_parity["full_s20_snapshot"]["only_historical"]
    )
    write_json(output / "cold_replay_parity.json", cold_parity)
    if not cold_parity["all_bitwise_equal"]:
        manifest["status"] = "cold_replay_parity_failed_stopped"
        manifest["actual_cost"] = vars(ledger)
        write_json(manifest_path, manifest)
        raise RuntimeError("BLEND_5 cold replay did not reproduce bitwise")

    # set_env_state must treat the saved source snapshot as immutable input.
    s0_after = torch_gzip_read(paths["trace_s0_reconstructed"])
    s0_immutable = compare_snapshots(s0_after, s0_before)
    if not s0_immutable["all_common_equal"]:
        raise RuntimeError("saved s0 input mutated during startup comparison")
    manifest["status"] = "physics_complete_analysis_and_visual_review_pending"
    manifest["results"] = results
    manifest["actual_cost"] = {
        **vars(ledger),
        "setup_native_mj_steps": 2,
        "task_physics_steps": 610,
        "actor_or_critic_forwards": 0,
        "optimizer_updates": 0,
    }
    manifest["s0_input_immutable"] = True
    write_json(manifest_path, manifest)


def _object_metrics(qpos: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    tool_error = qpos[36:39] - reference[36:39]
    target_error = qpos[43:46] - reference[43:46]
    pair_error = (qpos[36:39] - qpos[43:46]) - (
        reference[36:39] - reference[43:46]
    )
    tool_rotation = rotation_error(qpos[39:43], reference[39:43])
    target_rotation = rotation_error(qpos[46:50], reference[46:50])
    return {
        "tool_error_x_m": float(tool_error[0]),
        "tool_error_y_m": float(tool_error[1]),
        "tool_error_z_m": float(tool_error[2]),
        "tool_position_error_m": float(np.linalg.norm(tool_error)),
        "tool_rotation_error_rad": tool_rotation,
        "tool_ellipse_score": float(
            np.hypot(np.linalg.norm(tool_error) / 0.12, tool_rotation / 1.5)
        ),
        "target_error_x_m": float(target_error[0]),
        "target_error_y_m": float(target_error[1]),
        "target_error_z_m": float(target_error[2]),
        "target_position_error_m": float(np.linalg.norm(target_error)),
        "target_rotation_error_rad": target_rotation,
        "pair_error_x_m": float(pair_error[0]),
        "pair_error_y_m": float(pair_error[1]),
        "pair_error_z_m": float(pair_error[2]),
        "pair_position_error_m": float(np.linalg.norm(pair_error)),
    }


def _force_on_recipient(
    contacts: dict[str, np.ndarray], row: int, recipient: str
) -> np.ndarray:
    world_force = contacts["wrench_world_force_torque_on_geom2"][row, :3].astype(np.float64)
    if str(contacts["role2"][row]) == recipient:
        return world_force
    if str(contacts["role1"][row]) == recipient:
        return -world_force
    raise RuntimeError(f"contact row does not contain recipient {recipient}")


def _pair_contact_stats(
    contacts: dict[str, np.ndarray], role_a: str, role_b: str, recipient: str
) -> dict[str, Any]:
    mask = (
        ((contacts["role1"] == role_a) & (contacts["role2"] == role_b))
        | ((contacts["role1"] == role_b) & (contacts["role2"] == role_a))
    )
    rows = np.flatnonzero(mask)
    first = None
    if len(rows):
        index = int(rows[np.argmin(contacts["global_substep"][rows])])
        first = {
            "source_endpoint": int(contacts["source_endpoint"][index]),
            "outcome_endpoint": int(contacts["outcome_endpoint"][index]),
            "substep": int(contacts["substep"][index]),
            "global_substep": int(contacts["global_substep"][index]),
            "time_s": float(contacts["global_substep"][index] / 300.0),
        }
    dt = 1.0 / 300.0

    def window(selected: np.ndarray) -> dict[str, Any]:
        if not len(selected):
            return {
                "contact_rows": 0,
                "active_substeps": 0,
                "normal_impulse_Ns": 0.0,
                "signed_impulse_on_recipient_Ns": [0.0, 0.0, 0.0],
                "peak_summed_normal_force_N": 0.0,
            }
        impulse = sum(
            (_force_on_recipient(contacts, int(row), recipient) for row in selected),
            np.zeros(3, dtype=np.float64),
        ) * dt
        peak = 0.0
        for step in np.unique(contacts["global_substep"][selected]):
            local = selected[contacts["global_substep"][selected] == step]
            peak = max(
                peak,
                float(contacts["wrench_contact_force_torque"][local, 0].sum()),
            )
        return {
            "contact_rows": int(len(selected)),
            "active_substeps": int(len(np.unique(contacts["global_substep"][selected]))),
            "normal_impulse_Ns": float(
                contacts["wrench_contact_force_torque"][selected, 0].sum() * dt
            ),
            "signed_impulse_on_recipient_Ns": impulse.tolist(),
            "peak_summed_normal_force_N": peak,
        }

    return {
        "roles": [role_a, role_b],
        "signed_force_recipient": recipient,
        "first_occurrence": first,
        "first_control_cycle": window(rows[contacts["global_substep"][rows] <= 10]),
        "full_0_20": window(rows),
    }


def startup_analyze(output: Path) -> None:
    manifest_path = output / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "physics_complete_analysis_and_visual_review_pending":
        raise RuntimeError("startup analysis requires completed physics and cold parity")
    paths = startup_input_paths()
    with np.load(paths["reference"], allow_pickle=False) as archive:
        reference_qpos = archive["qpos"].copy()
    conditions: dict[str, dict[str, np.ndarray]] = {}
    substeps: dict[str, dict[str, np.ndarray]] = {}
    contacts: dict[str, dict[str, np.ndarray]] = {}
    for label in ("ORIGINAL", "HOLD_1", "BLEND_5", "BLEND_5_COLD"):
        directory = output / "conditions" / label
        conditions[label] = _npz_arrays(directory / "endpoints.npz")
        substeps[label] = _npz_arrays(directory / "substeps.npz")
        contacts[label] = _npz_arrays(directory / "contacts_raw.npz")

    fields = [
        "condition", "endpoint", "availability", "tool_error_x_m", "tool_error_y_m",
        "tool_error_z_m", "tool_position_error_m", "tool_rotation_error_rad",
        "tool_ellipse_score", "target_error_x_m", "target_error_y_m", "target_error_z_m",
        "target_position_error_m", "target_rotation_error_rad", "pair_error_x_m",
        "pair_error_y_m", "pair_error_z_m", "pair_position_error_m",
    ]
    rows: list[dict[str, Any]] = []
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    for label, arrays in conditions.items():
        endpoint_metrics[label] = {}
        for endpoint in range(ENDPOINTS + 1):
            if endpoint >= len(arrays["qpos"]):
                rows.append({"condition": label, "endpoint": endpoint, "availability": "N/A"})
                endpoint_metrics[label][str(endpoint)] = None
                continue
            metrics = _object_metrics(arrays["qpos"][endpoint], reference_qpos[endpoint])
            rows.append(
                {"condition": label, "endpoint": endpoint, "availability": "measured", **metrics}
            )
            endpoint_metrics[label][str(endpoint)] = metrics
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    first_cycle_fields = [
        "condition", "substep", "time_s", "tool_position_delta_vs_original_m",
        "target_position_delta_vs_original_m", "pair_position_delta_vs_original_m",
        "maximum_qpos_delta_vs_original",
    ]
    first_cycle_rows: list[dict[str, Any]] = []
    original_sub = substeps["ORIGINAL"]
    for label in ("ORIGINAL", "HOLD_1", "BLEND_5", "BLEND_5_COLD"):
        current = substeps[label]
        for step in range(10):
            qpos = current["qpos_after"][step]
            baseline = original_sub["qpos_after"][step]
            first_cycle_rows.append(
                {
                    "condition": label,
                    "substep": step,
                    "time_s": float(current["time_after"][step]),
                    "tool_position_delta_vs_original_m": float(np.linalg.norm(qpos[36:39] - baseline[36:39])),
                    "target_position_delta_vs_original_m": float(np.linalg.norm(qpos[43:46] - baseline[43:46])),
                    "pair_position_delta_vs_original_m": float(
                        np.linalg.norm((qpos[36:39] - qpos[43:46]) - (baseline[36:39] - baseline[43:46]))
                    ),
                    "maximum_qpos_delta_vs_original": float(np.max(np.abs(qpos - baseline))),
                }
            )
    with (output / "first_cycle_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=first_cycle_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(first_cycle_rows)

    pair_specs = {
        "left_ring_target": ("left_hand:ring", "target", "target"),
        "left_pinky_target": ("left_hand:pinky", "target", "target"),
        "target_floor": ("target", "floor", "target"),
        "right_thumb_floor": ("right_hand:thumb", "floor", "right_hand:thumb"),
    }
    contact_summary = {
        label: {
            name: _pair_contact_stats(contacts[label], *spec)
            for name, spec in pair_specs.items()
        }
        for label in conditions
    }

    blend_delta: dict[str, Any] = {}
    for endpoint in (1, 5, 10, 14, 15, 16, 20):
        oq = conditions["ORIGINAL"]["qpos"][endpoint]
        bq = conditions["BLEND_5"]["qpos"][endpoint]
        blend_delta[str(endpoint)] = {
            "maximum_absolute_qpos_delta": float(np.max(np.abs(bq - oq))),
            "tool_position_delta_m": float(np.linalg.norm(bq[36:39] - oq[36:39])),
            "target_position_delta_m": float(np.linalg.norm(bq[43:46] - oq[43:46])),
            "pair_position_delta_m": float(
                np.linalg.norm((bq[36:39] - bq[43:46]) - (oq[36:39] - oq[43:46]))
            ),
        }
    analysis = {
        "schema": STARTUP_SCHEMA,
        "endpoint_metrics": endpoint_metrics,
        "contact_pairs": contact_summary,
        "blend_minus_original": blend_delta,
        "first_cycle": {
            "blend_maximum_qpos_delta_vs_original": max(
                row["maximum_qpos_delta_vs_original"]
                for row in first_cycle_rows if row["condition"] == "BLEND_5"
            ),
            "hold_maximum_qpos_delta_vs_original": max(
                row["maximum_qpos_delta_vs_original"]
                for row in first_cycle_rows if row["condition"] == "HOLD_1"
            ),
        },
        "windows": {
            "startup_endpoints_1_5": {
                "blend_max_tool_position_delta_m": max(blend_delta[str(i)]["tool_position_delta_m"] for i in (1, 5)),
                "blend_max_target_position_delta_m": max(blend_delta[str(i)]["target_position_delta_m"] for i in (1, 5)),
            },
            "downstream_endpoints_10_20": {
                "blend_max_tool_position_delta_m": max(blend_delta[str(i)]["tool_position_delta_m"] for i in (10, 14, 15, 16, 20)),
                "blend_max_target_position_delta_m": max(blend_delta[str(i)]["target_position_delta_m"] for i in (10, 14, 15, 16, 20)),
            },
        },
        "hold_endpoint_gt_1_semantics": "N/A: HOLD_1 intentionally stops after one control interval",
        "tracking_relation": {
            label: json.loads((output / "conditions" / label / "result.json").read_text())
            for label in conditions
        },
    }
    write_json(output / "analysis.json", analysis)
    manifest["status"] = "physics_and_analysis_complete_visual_review_pending"
    manifest["recorded"] = {
        label: {
            "endpoints": int(len(conditions[label]["qpos"])),
            "substeps": int(len(substeps[label]["qpos_after"])),
            "active_contact_rows": int(len(contacts[label]["role1"])),
        }
        for label in conditions
    }
    write_json(manifest_path, manifest)


def hashes(output: Path) -> None:
    rows = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "hashes.json":
            rows[str(path.relative_to(output))] = artifact(path)
    write_json(output / "hashes.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight", "run", "resume-b", "hashes",
            "startup-preflight", "startup-run", "startup-analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    default_output = STARTUP_OUTPUT if args.stage.startswith("startup-") else OUTPUT
    output = (args.output or default_output).resolve()
    if args.stage == "preflight":
        preflight(output)
    elif args.stage == "run":
        execute(output)
    elif args.stage == "resume-b":
        resume_observer_b(output)
    elif args.stage == "startup-preflight":
        startup_preflight(output)
    elif args.stage == "startup-run":
        startup_execute(output)
    elif args.stage == "startup-analyze":
        startup_analyze(output)
    else:
        hashes(output)


if __name__ == "__main__":
    main()
