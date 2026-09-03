#!/usr/bin/env python3
"""Diagnostic table-contact ablation for a recorded DexImit SAPIEN trace.

This is deliberately outside the exact replay contract.  It reuses the exact
MuJoCo scene, source drive targets, and PhysX-equivalent drive/contact update,
then disables selected table pairs in memory to attribute a failed lift.  It
never writes an object pose during the rollout and never produces a formal
success artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(1, str(DIAGNOSTICS_ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from replay_exact_deximit_mujoco import (
    FINGERS,
    PHYSICS_DT_S,
    TGS_SUBSTEPS,
    body_point_velocity,
    friction_for_speed,
    load_source,
    sha256,
)
from sapien_equiv import (
    PhysxTgsForceDrive,
    advance_physx_tgs_microstep,
    apply_physx_frame_start_gravity,
    apply_physx_velocity_decay,
)


def load_failed_metrics(trace: Path) -> dict[str, object]:
    checkpoint = trace.parent.parent / "checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    with np.load(trace, allow_pickle=False) as values:
        candidate = int(np.asarray(values["candidate_index"]).item())
    rows = [
        row for row in payload.get("attempts", [])
        if int(row.get("source_candidate_index", -1)) == candidate
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("metrics"), dict):
        raise ValueError("trace checkpoint is not bound to one candidate metric row")
    return rows[0]["metrics"]


def load_home_probe(provenance_path: Path) -> tuple[list[str], np.ndarray]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = Path(str(provenance["drive_conversion"]["source"])).resolve(strict=True)
    with np.load(source, allow_pickle=False) as values:
        names = [str(value) for value in values["joint_names"]]
        qpos = np.asarray(values["qpos0"], dtype=np.float64)
    if len(names) != 18 or qpos.shape != (18,):
        raise ValueError("bound home probe does not define the 18-joint state")
    return names, qpos


def configure_scene(
    model: mujoco.MjModel,
    *,
    table_geom: int,
    object_geom: int,
    hand_geoms: set[int],
    table_mode: str,
) -> tuple[int, list[int], dict[int, int], dict[int, int]]:
    object_pair_by_geoms: dict[frozenset[int], int] = {}
    hand_object_pairs: list[int] = []
    table_pairs: list[int] = []
    for pair in range(model.npair):
        geoms = frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair])))
        object_pair_by_geoms[geoms] = pair
        if object_geom in geoms and hand_geoms & geoms:
            hand_object_pairs.append(pair)
        if table_geom in geoms:
            table_pairs.append(pair)
    table_object_pair = object_pair_by_geoms.get(
        frozenset((table_geom, object_geom)), -1,
    )
    if table_object_pair < 0 or len(hand_object_pairs) != 13:
        raise ValueError("scene lacks the expected explicit table/object/hand pairs")
    if table_mode not in {
        "baseline", "disable-table-object", "disable-all-table",
    }:
        raise ValueError(f"unknown table mode: {table_mode}")
    if table_mode == "disable-table-object":
        disabled = {table_object_pair}
    elif table_mode == "disable-all-table":
        disabled = set(table_pairs)
    else:
        disabled = set()
    for pair in disabled:
        # A negative margin moves the explicit pair's detection boundary below
        # any physically reachable penetration, without changing other pairs.
        model.pair_margin[pair] = -1.0
        model.pair_gap[pair] = 0.0
    pair_hand_bodies: dict[int, int] = {}
    pair_fingers: dict[int, int] = {}
    for pair in hand_object_pairs:
        geoms = frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair])))
        hand_geom = next(iter(geoms & hand_geoms))
        pair_hand_bodies[pair] = int(model.geom_bodyid[hand_geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom) or ""
        pair_fingers[pair] = next(
            (index for index, finger in enumerate(FINGERS) if finger in name), -1,
        )
    return table_object_pair, hand_object_pairs, pair_hand_bodies, pair_fingers


def run_variant(
    scene: Path,
    source: dict[str, object],
    metrics: dict[str, object],
    joint_names: list[str],
    home_qpos: np.ndarray,
    table_mode: str,
    gravity_disabled: bool,
) -> dict[str, object]:
    provenance = json.loads(scene.with_suffix(".provenance.json").read_text())
    contact = provenance["contact_calibration"]
    pinch = provenance["pinch_friction_calibration"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.noslip_iterations = int(contact["noslip_iterations"])

    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
    )
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
    )
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
    )
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    hand_geoms: set[int] = set()
    hand_by_geom: dict[int, int] = {}
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if not name.startswith("collision_hand_"):
            continue
        hand_geoms.add(geom)
        hand_by_geom[geom] = next(
            (index for index, finger in enumerate(FINGERS) if finger in name), -1,
        )
    table_object_pair, hand_object_pairs, pair_hand_bodies, pair_fingers = (
        configure_scene(
            model, table_geom=table_geom, object_geom=object_geom,
            hand_geoms=hand_geoms, table_mode=table_mode,
        )
    )
    object_pair_by_geoms = {
        frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair]))): pair
        for pair in range(model.npair)
    }
    drive = PhysxTgsForceDrive(
        model, mujoco, joint_names,
        stiffness=float(provenance["drive_conversion"]["source_stiffness"]),
        damping=float(provenance["drive_conversion"]["source_damping"]),
    )
    initial_transform = np.asarray(metrics["object_initial_pose"], dtype=np.float64)
    initial_quat = Rotation.from_matrix(initial_transform[:3, :3]).as_quat(
        scalar_first=True,
    )
    mujoco.mj_resetData(model, data)
    qpos_addresses = []
    dof_addresses = []
    for name in joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
        dof_addresses.append(int(model.jnt_dofadr[joint]))
    data.qpos[qpos_addresses] = home_qpos
    data.qpos[object_qpos : object_qpos + 3] = initial_transform[:3, 3]
    data.qpos[object_qpos + 3 : object_qpos + 7] = initial_quat
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    count = len(source["phase"])
    object_pose = np.empty((count, 7), dtype=np.float64)
    object_velocity = np.empty((count, 6), dtype=np.float64)
    finger_detected = np.zeros((count, 5), dtype=bool)
    finger_loaded = np.zeros((count, 5), dtype=bool)
    finger_impulse = np.zeros((count, 5, 3), dtype=np.float64)
    table_object_impulse = np.zeros((count, 3), dtype=np.float64)
    table_hand_impulse = np.zeros((count, 3), dtype=np.float64)
    table_object_contact = np.zeros(count, dtype=bool)
    table_hand_contact = np.zeros(count, dtype=bool)
    object_gap = np.full(count, np.inf, dtype=np.float64)
    table_object_gap = np.full(count, np.inf, dtype=np.float64)
    source_gravity = model.opt.gravity.copy()
    if gravity_disabled:
        source_gravity[:] = 0.0
    table_threshold = float(contact["table_static_speed_threshold_m_s"])
    hand_threshold = float(pinch["hand_static_speed_threshold_m_s"])

    for frame in range(count):
        data.qfrc_applied[:] = 0.0
        drive.set_target(np.asarray(source["drive_target"])[frame])
        apply_physx_velocity_decay(
            data.qvel, object_dof,
            linear_rate=float(metrics["object_linear_damping"]),
            angular_rate=float(metrics["object_angular_damping"]),
        )
        apply_physx_frame_start_gravity(
            model, data, mujoco, source_gravity, source_dt=PHYSICS_DT_S,
        )
        mujoco.mj_forward(model, data)
        drive.begin_frame(data)
        for _ in range(TGS_SUBSTEPS):
            for pair in hand_object_pairs:
                model.pair_friction[pair, :2] = 0.5
            model.pair_friction[table_object_pair, :2] = 0.75
            for contact_index in range(data.ncon):
                current = data.contact[contact_index]
                geoms = frozenset((int(current.geom1), int(current.geom2)))
                if geoms == frozenset((table_geom, object_geom)):
                    relative = body_point_velocity(
                        model, data, object_body, current.pos,
                    ) - body_point_velocity(
                        model, data, int(model.geom_bodyid[table_geom]), current.pos,
                    )
                    normal = np.asarray(current.frame[:3], dtype=np.float64)
                    tangential = relative - float(relative @ normal) * normal
                    model.pair_friction[table_object_pair, :2] = friction_for_speed(
                        float(np.linalg.norm(tangential)),
                        threshold=table_threshold, dynamic=0.75, static=0.85,
                    )
                if not (geoms & {object_geom} and geoms & hand_geoms):
                    continue
                hand_geom = next(iter(geoms & hand_geoms))
                hand_body = int(model.geom_bodyid[hand_geom])
                relative = body_point_velocity(
                    model, data, object_body, current.pos,
                ) - body_point_velocity(model, data, hand_body, current.pos)
                normal = np.asarray(current.frame[:3], dtype=np.float64)
                tangential = relative - float(relative @ normal) * normal
                friction = friction_for_speed(
                    float(np.linalg.norm(tangential)),
                    threshold=hand_threshold, dynamic=0.5, static=0.7,
                )
                if friction == 0.7:
                    pair = object_pair_by_geoms[geoms]
                    model.pair_friction[pair, :2] = friction
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            advance_physx_tgs_microstep(model, data, mujoco, drive)
            for contact_index in range(data.ncon):
                current = data.contact[contact_index]
                geoms = frozenset((int(current.geom1), int(current.geom2)))
                distance = float(current.dist)
                force = np.zeros(6, dtype=np.float64)
                active = int(current.efc_address) >= 0
                if object_geom in geoms and hand_geoms & geoms:
                    object_gap[frame] = min(object_gap[frame], distance)
                    hand_geom = next(iter(geoms & hand_geoms))
                    finger = hand_by_geom[hand_geom]
                    finger_detected[frame, finger] = True
                    if active:
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        world_force = (
                            np.asarray(current.frame, dtype=np.float64).reshape(3, 3).T
                            @ force[:3]
                        )
                        impulse = world_force * float(model.opt.timestep)
                        finger_impulse[frame, finger] += impulse
                        finger_loaded[frame, finger] |= (
                            float(np.linalg.norm(impulse)) > 1.0e-10
                        )
                if geoms != frozenset((table_geom, object_geom)):
                    if table_geom in geoms and hand_geoms & geoms:
                        table_hand_contact[frame] = True
                    continue
                table_object_contact[frame] = True
                table_object_gap[frame] = min(table_object_gap[frame], distance)
                if not active:
                    continue
                mujoco.mj_contactForce(model, data, contact_index, force)
                world_force = (
                    np.asarray(current.frame, dtype=np.float64).reshape(3, 3).T
                    @ force[:3]
                )
                impulse = world_force * float(model.opt.timestep)
                table_object_impulse[frame] += impulse
        object_pose[frame] = data.qpos[object_qpos : object_qpos + 7]
        object_velocity[frame] = data.qvel[object_dof : object_dof + 6]
        # Sum hand-table impulses after the frame's final contacts.  This is a
        # contact-presence diagnostic; the object support impulse above is the
        # quantity used for the causal comparison.
        for contact_index in range(data.ncon):
            current = data.contact[contact_index]
            geoms = frozenset((int(current.geom1), int(current.geom2)))
            if table_geom not in geoms or not (hand_geoms & geoms):
                continue
            if int(current.efc_address) < 0:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            table_hand_impulse[frame] += (
                np.asarray(current.frame, dtype=np.float64).reshape(3, 3).T
                @ force[:3]
            ) * float(model.opt.timestep)

    phase = np.asarray(source["phase"])
    phase_report: dict[str, object] = {}
    for name in ("pregrasp", "grasp", "squeeze", "demonstrated_object_motion", "hold"):
        rows = np.flatnonzero(phase == name)
        if len(rows) == 0:
            continue
        phase_report[name] = {
            "start": int(rows[0]), "end": int(rows[-1]),
            "object_z_min_m": float(np.min(object_pose[rows, 2])),
            "object_z_max_m": float(np.max(object_pose[rows, 2])),
            "loaded_fingers_any": [
                FINGERS[i] for i in range(5) if bool(np.any(finger_loaded[rows, i]))
            ],
            "loaded_fingers_all": [
                FINGERS[i] for i in range(5) if bool(np.all(finger_loaded[rows, i]))
            ],
            "table_object_contact_fraction": float(np.mean(table_object_contact[rows])),
            "table_hand_contact_fraction": float(np.mean(table_hand_contact[rows])),
        }
    pregrasp = np.flatnonzero(phase == "pregrasp")
    hold = np.flatnonzero(phase == "hold")
    rest_z = float(np.median(object_pose[pregrasp[-min(30, len(pregrasp)):], 2]))
    return {
        "table_mode": table_mode,
        "gravity_disabled": gravity_disabled,
        "diagnostic_only": True,
        "candidate_index": int(source["candidate"]),
        "sample_count": count,
        "rest_z_m": rest_z,
        "final_z_m": float(object_pose[-1, 2]),
        "maximum_lift_m": float(np.max(object_pose[:, 2]) - rest_z),
        "hold_min_lift_m": float(np.min(object_pose[hold, 2]) - rest_z),
        "hold_end_lift_m": float(object_pose[hold[-1], 2] - rest_z),
        "hold_vertical_span_m": float(np.ptp(object_pose[hold, 2])),
        "final_linear_velocity_m_s": object_velocity[-1, :3].tolist(),
        "maximum_hand_object_impulse_norm_ns": float(np.linalg.norm(finger_impulse, axis=2).sum(axis=1).max()),
        "hand_object_impulse_net_ns": finger_impulse.sum(axis=(0, 1)).tolist(),
        "table_object_impulse_net_ns": table_object_impulse.sum(axis=0).tolist(),
        "table_hand_impulse_net_ns": table_hand_impulse.sum(axis=0).tolist(),
        "loaded_finger_fraction": {
            FINGERS[i]: float(np.mean(finger_loaded[:, i])) for i in range(5)
        },
        "detected_finger_fraction": {
            FINGERS[i]: float(np.mean(finger_detected[:, i])) for i in range(5)
        },
        "table_object_contact_fraction": float(np.mean(table_object_contact)),
        "table_hand_contact_fraction": float(np.mean(table_hand_contact)),
        "minimum_object_gap_m": float(np.min(object_gap)),
        "minimum_table_object_gap_m": float(np.min(table_object_gap)),
        "phase_report": phase_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", action="append",
        choices=("baseline", "disable-table-object", "disable-all-table", "disable-table-object-gravity"),
        help="Run only selected variants; repeat the option to run several.",
    )
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    trace = args.sapien_trace.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite table ablation report {output}")
    source = load_source(trace)
    metrics = load_failed_metrics(trace)
    joint_names, home_qpos = load_home_probe(scene.with_suffix(".provenance.json"))
    selected = args.variant or [
        "baseline", "disable-table-object", "disable-all-table",
        "disable-table-object-gravity",
    ]
    variants = []
    for variant in selected:
        if variant == "baseline":
            mode, gravity_disabled = "baseline", False
        elif variant == "disable-table-object":
            mode, gravity_disabled = "disable-table-object", True
        elif variant == "disable-all-table":
            mode, gravity_disabled = "disable-all-table", True
        else:
            mode, gravity_disabled = "disable-table-object", False
        variants.append(
            run_variant(
                scene, source, metrics, joint_names, home_qpos,
                mode, gravity_disabled,
            )
        )
    report = {
        "schema": "deximit_table_contact_ablation_v1_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "source_trace": str(trace),
        "source_trace_sha256": sha256(trace),
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
