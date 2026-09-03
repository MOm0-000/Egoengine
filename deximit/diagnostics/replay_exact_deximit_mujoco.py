#!/usr/bin/env python3
"""Replay one released SAPIEN trace in the exact isolated MuJoCo scene.

The object is free after initialization.  This script never renders and never
enters the formal renderer/3.3 chain; a separate fixed-camera renderer may run
only if the strict physical gate in this audit passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Running this file directly makes ``diagnostics`` Python's first search path.
# Pin the workspace root first so an unrelated installed ``egoengine_repro``
# package cannot shadow the project sources used by the diagnostic gate.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(1, str(DIAGNOSTICS_ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bodex_triptych import strict_grasp_gate
from egoengine_repro.action.contracts import XHAND_SELF_FLOOR_TOLERANCE_M
from sapien_equiv import (
    PhysxStrongFriction,
    PhysxTgsForceDrive,
    advance_physx_tgs_microstep,
    apply_physx_frame_start_gravity,
    apply_physx_velocity_decay,
)


TRACE_SCHEMAS = {
    "deximit_sapien_hand_trace_v2_post_step_diagnostic_only",
    "deximit_sapien_hand_trace_v3_detailed_contact_diagnostic_only",
    "deximit_sapien_hand_trace_v4_resolved_contact_impulses_diagnostic_only",
    "deximit_sapien_hand_trace_v5_contact_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v6_pair_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v7_link_contact_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only",
}
PHYSICS_DT_S = 1.0 / 240.0
TGS_SUBSTEPS = 25
FINGERS = ("thumb", "index", "mid", "ring", "pinky")


def friction_for_speed(
    tangential_speed: float, *, threshold: float,
    dynamic: float, static: float,
) -> float:
    """Approximate PhysX's two friction coefficients in one MuJoCo pair."""
    values = np.asarray((tangential_speed, threshold, dynamic, static), dtype=float)
    if (
        not np.isfinite(values).all() or tangential_speed < 0.0 or threshold < 0.0
        or dynamic < 0.0 or static < dynamic
    ):
        raise ValueError("friction conversion values are invalid")
    return float(static if tangential_speed <= threshold else dynamic)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    result = path.expanduser().resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"input is not a file: {result}")
    return result


def load_source(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) not in TRACE_SCHEMAS
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or str(np.asarray(values["sample_semantics"]).item())
            != "post-step state under same-index drive target"
        ):
            raise ValueError("source is not the released post-step SAPIEN trace")
        result = {
            "candidate": int(np.asarray(values["candidate_index"]).item()),
            "depth": int(np.asarray(values["pool_depth"]).item()),
            "dt": float(np.asarray(values["physics_dt_s"]).item()),
            "frame_skip": int(np.asarray(values["control_frame_skip"]).item()),
            "phase": np.asarray(values["phase"]).astype("U32"),
            "object_pose": np.asarray(
                values["object_pose_sapien_wxyz"], dtype=np.float64,
            ),
            "robot_qpos": np.asarray(
                values["right_robot_qpos_sapien"], dtype=np.float64,
            ),
            "robot_qvel": np.asarray(
                values["right_robot_qvel_sapien"], dtype=np.float64,
            ),
            "drive_target": np.asarray(
                values["right_drive_target_sapien"], dtype=np.float64,
            ),
            "sample_time": np.asarray(
                values["physics_sample_time_s"], dtype=np.float64,
            ),
        }
    count = len(result["phase"])
    if (
        result["dt"] != PHYSICS_DT_S or result["frame_skip"] != 12
        or result["object_pose"].shape != (count, 7)
        or result["robot_qpos"].shape != (count, 18)
        or result["robot_qvel"].shape != (count, 18)
        or result["drive_target"].shape != (count, 18)
        or result["sample_time"].shape != (count,)
        or not all(np.isfinite(np.asarray(result[key])).all() for key in (
            "object_pose", "robot_qpos", "robot_qvel", "drive_target", "sample_time",
        ))
        or not np.allclose(
            result["sample_time"], (np.arange(count) + 1.0) * PHYSICS_DT_S,
            atol=1.0e-10, rtol=0.0,
        )
    ):
        raise ValueError("SAPIEN trace violates the exact timing contract")
    return result


def load_summary(path: Path, trace: Path, source: dict[str, object]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == source["candidate"]
        and int(row.get("depth", -1)) == source["depth"]
    ]
    if (
        len(rows) != 1 or rows[0].get("original_sapien_pass") is not True
        or Path(rows[0].get("exported_trajectory", "")).resolve() != trace
        or rows[0].get("exported_trajectory_sha256") != sha256(trace)
        or not isinstance(rows[0].get("metrics"), dict)
    ):
        raise ValueError("SAPIEN summary is not bound to the selected trace")
    return rows[0]["metrics"]


def body_point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: np.ndarray,
) -> np.ndarray:
    spatial = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0,
    )
    angular, linear = spatial[:3], spatial[3:]
    return linear + np.cross(angular, point - data.xpos[body])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--home-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-object-penetration-m", type=float, default=0.002)
    parser.add_argument("--static-speed-threshold-m-s", type=float)
    parser.add_argument(
        "--table-static-speed-threshold-m-s", type=float,
        help="Must match the bound candidate-free contact calibration when provided.",
    )
    parser.add_argument(
        "--hand-friction-mode",
        choices=("speed-threshold", "strong-anchor"),
        default="speed-threshold",
        help=(
            "Use the legacy velocity-only approximation or the isolated "
            "PhysX-style retained friction anchor."
        ),
    )
    args = parser.parse_args()
    scene = require_file(args.scene)
    trace_path = require_file(args.sapien_trace)
    summary_path = require_file(args.sapien_summary)
    home_probe_path = require_file(args.home_probe)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exact replay {output}")
    if (
        args.max_object_penetration_m <= 0.0
        or (
            args.static_speed_threshold_m_s is not None
            and args.static_speed_threshold_m_s < 0.0
        )
        or (
            args.table_static_speed_threshold_m_s is not None
            and args.table_static_speed_threshold_m_s < 0.0
        )
    ):
        raise ValueError("physical thresholds are invalid")

    source = load_source(trace_path)
    metrics = load_summary(summary_path, trace_path, source)
    with np.load(home_probe_path, allow_pickle=False) as home_values:
        if (
            bool(np.asarray(home_values["formal_renderer_3_3_eligible"]).item())
            or not bool(np.asarray(home_values["diagnostic_only"]).item())
        ):
            raise ValueError("home probe is not isolated diagnostic evidence")
        joint_names = [str(value) for value in home_values["joint_names"]]
        home_qpos = np.asarray(home_values["qpos0"], dtype=np.float64)
    if len(joint_names) != 18 or home_qpos.shape != (18,):
        raise ValueError("home probe does not define the exact 18-joint state")

    provenance_path = scene.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema")
        != (
            "deximit_exact_urdf_mujoco_scene_"
            "v10_bound_strong_friction_diagnostic_only"
        )
        or provenance.get("diagnostic_only") is not True
        or provenance.get("formal_renderer_3_3_eligible") is not False
        or provenance.get("mujoco_version_used_to_generate_contact_semantics")
        != mujoco.__version__
    ):
        raise ValueError("scene provenance is not the current exact diagnostic contract")
    contact_calibration = provenance.get("contact_calibration", {})
    calibration_path = Path(str(contact_calibration.get("source", ""))).resolve(
        strict=True,
    )
    calibrated_time = float(contact_calibration.get("mujoco_time_constant_s", np.nan))
    calibrated_damping = float(contact_calibration.get("damping_ratio", np.nan))
    calibrated_impedance = float(
        contact_calibration.get("constant_constraint_impedance", np.nan),
    )
    calibrated_table_speed = float(
        contact_calibration.get("table_static_speed_threshold_m_s", np.nan),
    )
    calibrated_cone = str(contact_calibration.get("friction_cone", ""))
    calibrated_noslip = int(contact_calibration.get("noslip_iterations", -1))
    if (
        contact_calibration.get("source_sha256") != sha256(calibration_path)
        or not np.isfinite((
            calibrated_time, calibrated_damping, calibrated_impedance,
            calibrated_table_speed,
        )).all()
        or calibrated_time <= 0.0 or calibrated_damping <= 0.0
        or not 0.0 < calibrated_impedance <= 1.0
        or calibrated_table_speed < 0.0
        or contact_calibration.get("integration_order") != "physx_tgs"
        or calibrated_cone not in ("pyramidal", "elliptic")
        or calibrated_noslip < 0
    ):
        raise ValueError("scene does not carry a bound candidate-free contact calibration")
    drive_conversion = provenance.get("drive_conversion", {})
    drive_probe_path = Path(str(drive_conversion.get("source", ""))).resolve(
        strict=True,
    )
    drive_stiffness = float(drive_conversion.get("source_stiffness", np.nan))
    drive_damping = float(drive_conversion.get("source_damping", np.nan))
    converter_path = Path(
        str(drive_conversion.get("runtime_converter_implementation", "")),
    ).resolve(strict=True)
    if (
        drive_probe_path != home_probe_path
        or drive_conversion.get("source_sha256") != sha256(drive_probe_path)
        or converter_path != DIAGNOSTICS_ROOT / "sapien_equiv.py"
        or drive_conversion.get("runtime_converter_implementation_sha256")
        != sha256(converter_path)
        or not np.allclose((drive_stiffness, drive_damping), (1000.0, 100.0))
        or drive_conversion.get("native_mujoco_actuators_disabled_at_runtime")
        is not True
        or drive_conversion.get("native_mujoco_joint_limits_disabled_at_runtime")
        is not True
    ):
        raise ValueError("scene is not bound to the selected loaded SAPIEN drive probe")
    strong_friction_binding = provenance.get("strong_friction_conversion")
    if args.hand_friction_mode == "strong-anchor":
        if not isinstance(strong_friction_binding, dict):
            raise ValueError("scene lacks bound strong-friction evidence")
        strong_source = Path(
            str(strong_friction_binding.get("source", "")),
        ).resolve(strict=True)
        strong_converter = Path(str(
            strong_friction_binding.get("runtime_converter_implementation", ""),
        )).resolve(strict=True)
        if (
            strong_friction_binding.get("source_sha256") != sha256(strong_source)
            or strong_converter != converter_path
            or strong_friction_binding.get(
                "runtime_converter_implementation_sha256",
            ) != sha256(strong_converter)
            or strong_friction_binding.get("loaded_hold_passed") is not True
            or not np.allclose((
                float(strong_friction_binding.get("static_friction", np.nan)),
                float(strong_friction_binding.get("time_constant_s", np.nan)),
                float(strong_friction_binding.get("correlation_distance_m", np.nan)),
                float(strong_friction_binding.get(
                    "runtime_hand_contact_damping_ratio", np.nan,
                )),
            ), (0.7, PHYSICS_DT_S, 0.025, 1.0), atol=1.0e-12, rtol=0.0)
        ):
            raise ValueError("strong-friction evidence differs from the runtime contract")
    if args.table_static_speed_threshold_m_s is None:
        args.table_static_speed_threshold_m_s = calibrated_table_speed
    elif not np.isclose(
        args.table_static_speed_threshold_m_s, calibrated_table_speed,
        atol=1.0e-12, rtol=0.0,
    ):
        raise ValueError("table friction threshold differs from the bound calibration")
    pinch_calibration = provenance.get("pinch_friction_calibration", {})
    pinch_path = Path(str(pinch_calibration.get("source", ""))).resolve(strict=True)
    calibrated_hand_speed = float(
        pinch_calibration.get("hand_static_speed_threshold_m_s", np.nan),
    )
    pinch_contact_time = float(
        pinch_calibration.get("validated_contact_time_constant_s", np.nan),
    )
    pinch_contact_damping = float(
        pinch_calibration.get("validated_contact_damping_ratio", np.nan),
    )
    pinch_contact_impedance = float(
        pinch_calibration.get("validated_contact_impedance", np.nan),
    )
    if (
        pinch_calibration.get("source_sha256") != sha256(pinch_path)
        or pinch_calibration.get("friction_cone") != calibrated_cone
        or int(pinch_calibration.get("noslip_iterations", -1)) != calibrated_noslip
        or not np.isfinite((
            calibrated_hand_speed, pinch_contact_time, pinch_contact_damping,
            pinch_contact_impedance,
        )).all()
        or calibrated_hand_speed < 0.0
        or not np.isclose(
            pinch_contact_time, calibrated_time, atol=1.0e-12, rtol=0.0,
        )
        or not np.isclose(
            pinch_contact_damping, calibrated_damping,
            atol=1.0e-12, rtol=0.0,
        )
        or not np.isclose(
            pinch_contact_impedance, calibrated_impedance,
            atol=1.0e-12, rtol=0.0,
        )
    ):
        raise ValueError("scene does not carry a bound candidate-free pinch calibration")
    if args.static_speed_threshold_m_s is None:
        args.static_speed_threshold_m_s = calibrated_hand_speed
    elif not np.isclose(
        args.static_speed_threshold_m_s, calibrated_hand_speed,
        atol=1.0e-12, rtol=0.0,
    ):
        raise ValueError("hand friction threshold differs from the bound calibration")
    hand_contact_binding = provenance.get("hand_contact_calibration", {})
    hand_contact_source = hand_contact_binding.get("source")
    hand_contact_time = calibrated_time
    hand_contact_damping = calibrated_damping
    hand_contact_impedance = calibrated_impedance
    if hand_contact_source is not None:
        hand_contact_path = Path(str(hand_contact_source)).resolve(strict=True)
        hand_pinch_path = Path(str(
            hand_contact_binding.get("candidate_free_pinch_source", ""),
        )).resolve(strict=True)
        hand_contact_time = float(
            hand_contact_binding.get("mujoco_time_constant_s", np.nan),
        )
        hand_contact_damping = float(
            hand_contact_binding.get("damping_ratio", np.nan),
        )
        hand_contact_impedance = float(
            hand_contact_binding.get("constant_constraint_impedance", np.nan),
        )
        hand_pinch_gap = float(
            hand_contact_binding.get("candidate_free_pinch_minimum_gap_m", np.nan),
        )
        if (
            hand_contact_binding.get("source_sha256") != sha256(hand_contact_path)
            or hand_contact_binding.get("candidate_free_pinch_source_sha256")
            != sha256(hand_pinch_path)
            or hand_contact_binding.get("candidate_free_pinch_strict_passed")
            is not True
            or int(hand_contact_binding.get("candidate_specific_diagnostic", -1))
            != int(source["candidate"])
            or not np.isfinite((
                hand_contact_time, hand_contact_damping, hand_contact_impedance,
                hand_pinch_gap,
            )).all()
            or hand_contact_time <= 0.0 or hand_contact_damping <= 0.0
            or not 0.0 < hand_contact_impedance <= 1.0
            or hand_pinch_gap < -args.max_object_penetration_m
        ):
            raise ValueError("hand contact response is not bound to this candidate")
    elif hand_contact_binding and hand_contact_binding.get(
        "inherits_object_contact_response",
    ) is not True:
        raise ValueError("hand contact response provenance is malformed")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if (
        model.ncam != 1
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0) != "front"
        or int(model.cam_mode[0]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        or not np.isclose(model.opt.timestep, PHYSICS_DT_S / TGS_SUBSTEPS)
        or int(model.opt.cone) != int(
            mujoco.mjtCone.mjCONE_PYRAMIDAL
            if calibrated_cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
        )
        or int(model.opt.noslip_iterations) != calibrated_noslip
    ):
        raise ValueError("exact scene timing or fixed-camera contract is malformed")

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_ids: list[int] = []
    for name in joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{name}",
        )
        if min(joint, actuator) < 0:
            raise ValueError(f"exact scene lacks joint drive {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
        dof_addresses.append(int(model.jnt_dofadr[joint]))
        actuator_ids.append(actuator)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
    )
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
    )
    object_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single"),
    }
    object_geom = next(iter(object_geoms))
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    hand_collision_geoms: set[int] = set()
    hand_geoms: dict[int, int] = {}
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if not name.startswith("collision_hand_"):
            continue
        hand_collision_geoms.add(geom)
        finger = next(
            (index for index, value in enumerate(FINGERS) if value in name), -1,
        )
        if finger >= 0:
            hand_geoms[geom] = finger
    if len(hand_collision_geoms) != 13 or len(hand_geoms) != 12:
        raise ValueError(
            f"expected palm plus 12 finger collision geoms, got {hand_collision_geoms}",
        )

    object_pair_by_geoms: dict[frozenset[int], int] = {}
    hand_object_pairs: list[int] = []
    for pair in range(model.npair):
        geoms = frozenset((int(model.pair_geom1[pair]), int(model.pair_geom2[pair])))
        object_pair_by_geoms[geoms] = pair
        if geoms & object_geoms and geoms & hand_collision_geoms:
            hand_object_pairs.append(pair)
    if len(hand_object_pairs) != 13:
        raise ValueError("exact scene lacks one explicit pair per hand collision")
    table_object_pair = object_pair_by_geoms.get(
        frozenset((table_geom, object_geom)), -1,
    )
    if table_object_pair < 0:
        raise ValueError("exact scene lacks the table-object material pair")
    table_response_matches = np.allclose(
        model.pair_solref[table_object_pair],
        (calibrated_time, calibrated_damping), atol=1.0e-12, rtol=0.0,
    ) and np.allclose(
        model.pair_solimp[table_object_pair, :2], calibrated_impedance,
        atol=1.0e-12, rtol=0.0,
    )
    hand_response_matches = all(
        np.allclose(
            model.pair_solref[pair],
            (hand_contact_time, hand_contact_damping),
            atol=1.0e-12, rtol=0.0,
        )
        and np.allclose(
            model.pair_solimp[pair, :2], hand_contact_impedance,
            atol=1.0e-12, rtol=0.0,
        )
        for pair in hand_object_pairs
    )
    if not table_response_matches or not hand_response_matches:
        raise ValueError("compiled contact pairs differ from the bound calibration")

    pair_hand_bodies: dict[int, int] = {}
    pair_fingers: dict[int, int] = {}
    for pair in hand_object_pairs:
        geoms = frozenset((
            int(model.pair_geom1[pair]), int(model.pair_geom2[pair]),
        ))
        hand_geom = next(iter(geoms & hand_collision_geoms))
        pair_hand_bodies[pair] = int(model.geom_bodyid[hand_geom])
        if hand_geom in hand_geoms:
            pair_fingers[pair] = hand_geoms[hand_geom]

    strong_friction = None
    runtime_hand_contact_damping = hand_contact_damping
    if args.hand_friction_mode == "strong-anchor":
        # PhysX's zero-restitution rigid contact is critically damped.  The
        # old ratio of four came from a velocity-only one-step fit whose
        # loaded-hold audit is now known to omit PhysX friction history.
        runtime_hand_contact_damping = 1.0
        for pair in hand_object_pairs:
            model.pair_solref[pair] = (
                hand_contact_time, runtime_hand_contact_damping,
            )
        strong_friction = PhysxStrongFriction(
            model, mujoco, object_body=object_body, object_geom=object_geom,
            pair_hand_bodies=pair_hand_bodies, static_friction=0.7,
            time_constant=PHYSICS_DT_S, correlation_distance=0.025,
        )

    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    drive = PhysxTgsForceDrive(
        model, mujoco, joint_names,
        stiffness=drive_stiffness, damping=drive_damping,
    )

    initial_transform = np.asarray(metrics["object_initial_pose"], dtype=np.float64)
    initial_quat = Rotation.from_matrix(
        initial_transform[:3, :3],
    ).as_quat(scalar_first=True)
    mujoco.mj_resetData(model, data)
    for address, value in zip(qpos_addresses, home_qpos, strict=True):
        data.qpos[address] = value
    data.qpos[object_qpos : object_qpos + 3] = initial_transform[:3, 3]
    data.qpos[object_qpos + 3 : object_qpos + 7] = initial_quat
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    count = len(source["phase"])
    qpos_trace = np.empty((count, model.nq), dtype=np.float64)
    qvel_trace = np.empty((count, model.nv), dtype=np.float64)
    finger_detected = np.zeros((count, 5), dtype=bool)
    finger_contact = np.zeros((count, 5), dtype=bool)
    finger_normal_impulse = np.zeros((count, 5), dtype=np.float64)
    finger_impulse_norm_sum = np.zeros((count, 5), dtype=np.float64)
    finger_impulse_net = np.zeros((count, 5, 3), dtype=np.float64)
    finger_contact_point_count = np.zeros((count, 5), dtype=np.int32)
    finger_contact_position_mean = np.full((count, 5, 3), np.nan, dtype=np.float64)
    finger_contact_normal_mean = np.full((count, 5, 3), np.nan, dtype=np.float64)
    finger_anchor_impulse = np.zeros((count, 5, 3), dtype=np.float64)
    anchor_impulse_net = np.zeros((count, 3), dtype=np.float64)
    anchor_count_max = np.zeros(count, dtype=np.int32)
    object_gap = np.full(count, np.inf, dtype=np.float64)
    table_hand_gap = np.full(count, np.inf, dtype=np.float64)
    table_object_gap = np.full(count, np.inf, dtype=np.float64)
    static_microsteps = 0
    dynamic_microsteps = 0
    table_static_microsteps = 0
    table_dynamic_microsteps = 0
    strong_anchor_active_microsteps = 0
    strong_anchor_pair_microsteps = 0
    numerical_warning_step = None
    source_gravity = model.opt.gravity.copy()

    for source_step in range(count):
        data.qfrc_applied[:] = 0.0
        target = source["drive_target"][source_step]
        drive.set_target(target)
        apply_physx_velocity_decay(
            data.qvel, object_dof,
            linear_rate=float(metrics["object_linear_damping"]),
            angular_rate=float(metrics["object_angular_damping"]),
        )
        apply_physx_frame_start_gravity(
            model, data, mujoco, source_gravity, source_dt=PHYSICS_DT_S,
        )
        # Refresh body velocities and contacts after the source rigid-body
        # damping and frame-start gravity updates so friction selection never
        # uses stale cvel.  Gravity remains disabled during the 25 TGS-sized
        # substeps, matching PhysX's default external-force timing.
        mujoco.mj_forward(model, data)
        drive.begin_frame(data)
        step_detected = np.zeros(5, dtype=bool)
        step_touched = np.zeros(5, dtype=bool)
        step_normal_impulse = np.zeros(5, dtype=np.float64)
        step_impulse_norm_sum = np.zeros(5, dtype=np.float64)
        step_impulse_net = np.zeros((5, 3), dtype=np.float64)
        step_point_count = np.zeros(5, dtype=np.int32)
        step_position_sum = np.zeros((5, 3), dtype=np.float64)
        step_normal_sum = np.zeros((5, 3), dtype=np.float64)
        for _ in range(TGS_SUBSTEPS):
            for pair in hand_object_pairs:
                model.pair_friction[pair, :2] = (
                    0.0 if strong_friction is not None else 0.5
                )
            model.pair_friction[table_object_pair, :2] = 0.75
            static_pairs: set[int] = set()
            table_is_static = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geoms = frozenset((int(contact.geom1), int(contact.geom2)))
                if geoms == frozenset((table_geom, object_geom)):
                    table_body = int(model.geom_bodyid[table_geom])
                    relative = body_point_velocity(
                        model, data, object_body, contact.pos,
                    ) - body_point_velocity(model, data, table_body, contact.pos)
                    normal = np.asarray(contact.frame[:3], dtype=np.float64)
                    tangential = relative - float(relative @ normal) * normal
                    table_friction = friction_for_speed(
                        float(np.linalg.norm(tangential)),
                        threshold=args.table_static_speed_threshold_m_s,
                        dynamic=0.75, static=0.85,
                    )
                    model.pair_friction[table_object_pair, :2] = table_friction
                    if table_friction == 0.85:
                        table_is_static = True
                if not (geoms & object_geoms and geoms & hand_collision_geoms):
                    continue
                if strong_friction is None:
                    hand_geom = next(iter(geoms & hand_collision_geoms))
                    hand_body = int(model.geom_bodyid[hand_geom])
                    relative = body_point_velocity(
                        model, data, object_body, contact.pos,
                    ) - body_point_velocity(model, data, hand_body, contact.pos)
                    normal = np.asarray(contact.frame[:3], dtype=np.float64)
                    tangential = relative - float(relative @ normal) * normal
                    hand_friction = friction_for_speed(
                        float(np.linalg.norm(tangential)),
                        threshold=args.static_speed_threshold_m_s,
                        dynamic=0.5, static=0.7,
                    )
                    if hand_friction == 0.7:
                        pair = object_pair_by_geoms[geoms]
                        model.pair_friction[pair, :2] = hand_friction
                        static_pairs.add(pair)
            if strong_friction is None:
                static_microsteps += int(bool(static_pairs))
                dynamic_microsteps += int(not static_pairs)
            table_static_microsteps += int(table_is_static)
            table_dynamic_microsteps += int(not table_is_static)
            data.qfrc_applied[:] = 0.0
            if strong_friction is not None:
                # Rebuild the normal-only contact constraints before reading
                # their force capacity and applying the retained tangential
                # anchor as an equal-and-opposite external force.
                mujoco.mj_forward(model, data)
                applied = strong_friction.apply(data)
                strong_anchor_active_microsteps += int(bool(applied))
                strong_anchor_pair_microsteps += len(applied)
                anchor_count_max[source_step] = max(
                    anchor_count_max[source_step], strong_friction.anchor_count,
                )
                for pair, force in applied.items():
                    impulse = force * float(model.opt.timestep)
                    anchor_impulse_net[source_step] += impulse
                    finger = pair_fingers.get(pair)
                    if finger is not None:
                        finger_anchor_impulse[source_step, finger] += impulse
            advance_physx_tgs_microstep(model, data, mujoco, drive)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                geoms = {int(contact.geom1), int(contact.geom2)}
                distance = float(contact.dist)
                if geoms & object_geoms and geoms & hand_collision_geoms:
                    object_gap[source_step] = min(object_gap[source_step], distance)
                    hand_geom = next(iter(geoms & hand_collision_geoms))
                    finger = hand_geoms.get(hand_geom)
                    if finger is not None:
                        position = np.asarray(contact.pos, dtype=np.float64)
                        toward_object = data.xpos[object_body] - position
                        normal = np.asarray(contact.frame[:3], dtype=np.float64)
                        if float(normal @ toward_object) < 0.0:
                            normal = -normal
                        step_detected[finger] = True
                        step_point_count[finger] += 1
                        step_position_sum[finger] += position
                        step_normal_sum[finger] += normal
                    if finger is not None and int(contact.efc_address) >= 0:
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            model, data, contact_index, force,
                        )
                        normal_impulse = (
                            max(0.0, float(force[0])) * float(model.opt.timestep)
                        )
                        world_force = (
                            np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T
                            @ force[:3]
                        )
                        impulse_vector = world_force * float(model.opt.timestep)
                        if float(impulse_vector @ toward_object) < 0.0:
                            impulse_vector = -impulse_vector
                        impulse_norm = float(np.linalg.norm(impulse_vector))
                        step_normal_impulse[finger] += normal_impulse
                        step_impulse_norm_sum[finger] += impulse_norm
                        step_impulse_net[finger] += impulse_vector
                        step_touched[finger] |= impulse_norm > 1.0e-10
                if table_geom in geoms and geoms & hand_collision_geoms:
                    table_hand_gap[source_step] = min(
                        table_hand_gap[source_step], distance,
                    )
                if table_geom in geoms and geoms & object_geoms:
                    table_object_gap[source_step] = min(
                        table_object_gap[source_step], distance,
                    )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            numerical_warning_step = source_step
            break
        if any(int(warning.number) > 0 for warning in data.warning):
            numerical_warning_step = source_step
            break
        qpos_trace[source_step] = data.qpos
        qvel_trace[source_step] = data.qvel
        finger_detected[source_step] = step_detected
        finger_contact[source_step] = step_touched
        finger_normal_impulse[source_step] = step_normal_impulse
        finger_impulse_norm_sum[source_step] = step_impulse_norm_sum
        finger_impulse_net[source_step] = step_impulse_net
        finger_contact_point_count[source_step] = step_point_count
        populated = step_point_count > 0
        finger_contact_position_mean[source_step, populated] = (
            step_position_sum[populated] / step_point_count[populated, None]
        )
        mean_normal = step_normal_sum[populated] / step_point_count[populated, None]
        normal_norm = np.linalg.norm(mean_normal, axis=1)
        nonzero = normal_norm > np.finfo(np.float64).eps
        mean_normal[nonzero] /= normal_norm[nonzero, None]
        finger_contact_normal_mean[
            source_step, np.flatnonzero(populated)[nonzero]
        ] = mean_normal[nonzero]

    if numerical_warning_step is not None:
        raise FloatingPointError(
            f"exact MuJoCo replay became unstable at source step {numerical_warning_step}",
        )
    object_pose_mujoco = qpos_trace[:, object_qpos : object_qpos + 7]
    source_object = np.asarray(source["object_pose"])
    object_position_error = np.linalg.norm(
        object_pose_mujoco[:, :3] - source_object[:, :3], axis=1,
    )
    object_rotation_error = (
        Rotation.from_quat(object_pose_mujoco[:, 3:], scalar_first=True)
        * Rotation.from_quat(source_object[:, 3:], scalar_first=True).inv()
    ).magnitude()
    robot_qpos_mujoco = qpos_trace[:, qpos_addresses]
    robot_qvel_mujoco = qvel_trace[:, dof_addresses]
    robot_qpos_error = np.linalg.norm(
        robot_qpos_mujoco - np.asarray(source["robot_qpos"]), axis=1,
    )
    robot_qvel_error = np.linalg.norm(
        robot_qvel_mujoco - np.asarray(source["robot_qvel"]), axis=1,
    )

    phase = np.asarray(source["phase"])
    pregrasp = np.flatnonzero(phase == "pregrasp")
    hold = np.flatnonzero(phase == "hold")
    closure_or_later = np.isin(
        phase, ("squeeze", "demonstrated_object_motion", "hold"),
    )
    opposed = finger_contact[:, 0] & finger_contact[:, 1:].any(axis=1)
    opposed_contact = bool((opposed & closure_or_later).any())
    hold_opposed = bool(opposed[hold].any())
    object_z = object_pose_mujoco[:, 2]
    rest_rows = pregrasp[-min(30, len(pregrasp)) :]
    rest_z = float(np.median(object_z[rest_rows]))
    hold_min_lift = float(np.min(object_z[hold]) - rest_z)
    hold_end_lift = float(object_z[hold[-1]] - rest_z)
    held_lift = bool(hold_min_lift >= 0.019 and hold_end_lift >= 0.02)
    hold_span = float(np.ptp(object_z[hold]))
    stable_hold = hold_span <= 0.01
    minimum_object_gap = float(np.min(object_gap))
    minimum_table_hand_gap = float(np.min(table_hand_gap))
    hard_legal = minimum_table_hand_gap >= -XHAND_SELF_FLOOR_TOLERANCE_M
    passed = strict_grasp_gate(
        hard_legal=hard_legal,
        opposed_contact=opposed_contact,
        hold_opposed_contact=hold_opposed,
        held_lift=held_lift,
        stable_hold=stable_hold,
        minimum_object_gap_m=minimum_object_gap,
        maximum_object_penetration_m=args.max_object_penetration_m,
    )

    output.mkdir(parents=True)
    result_trace = output / "trace.npz"
    np.savez_compressed(
        result_trace,
        schema=np.asarray(
            "deximit_exact_mujoco_replay_v6_strong_friction_diagnostic_only",
        ),
        diagnostic_only=np.asarray(True), formal_renderer_3_3_eligible=np.asarray(False),
        strict_gate_passed=np.asarray(passed), candidate_index=np.asarray(source["candidate"]),
        source_trace_sha256=np.asarray(sha256(trace_path)),
        scene_sha256=np.asarray(sha256(scene)),
        qpos=qpos_trace, qvel=qvel_trace, phase=phase,
        finger_contact_detected=finger_detected,
        finger_contact_load_bearing=finger_contact,
        finger_contact=finger_contact,
        finger_normal_impulse_ns=finger_normal_impulse,
        finger_impulse_norm_sum_ns=finger_impulse_norm_sum,
        finger_impulse_net_on_object_ns=finger_impulse_net,
        finger_contact_point_count=finger_contact_point_count,
        finger_contact_position_mean_m=finger_contact_position_mean,
        finger_contact_normal_toward_object_mean=finger_contact_normal_mean,
        hand_object_gap_m=object_gap, table_hand_gap_m=table_hand_gap,
        table_object_gap_m=table_object_gap,
        object_position_error_m=object_position_error,
        object_rotation_error_rad=object_rotation_error,
        robot_qpos_l2_error_rad=robot_qpos_error,
        robot_qvel_l2_error_rad_s=robot_qvel_error,
        finger_strong_anchor_impulse_ns=finger_anchor_impulse,
        strong_anchor_impulse_net_on_object_ns=anchor_impulse_net,
        strong_anchor_count_max=anchor_count_max,
    )
    report = {
        "schema": "deximit_exact_mujoco_replay_v6_strong_friction_diagnostic_only",
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "rendered": False,
        "render_policy": "only a strict pass may enter the separate fixed-camera renderer",
        "candidate_index": source["candidate"],
        "pool_depth": source["depth"],
        "scene": str(scene), "scene_sha256": sha256(scene),
        "scene_provenance": str(provenance_path),
        "source_trace": str(trace_path), "source_trace_sha256": sha256(trace_path),
        "source_summary": str(summary_path), "source_summary_sha256": sha256(summary_path),
        "home_probe": str(home_probe_path), "home_probe_sha256": sha256(home_probe_path),
        "control_timing": {
            "source_physics_dt_s": PHYSICS_DT_S,
            "mujoco_substep_dt_s": float(model.opt.timestep),
            "mujoco_substeps_per_source_step": TGS_SUBSTEPS,
            "source_control_frame_skip": source["frame_skip"],
            "mujoco_steps_per_control_target": TGS_SUBSTEPS * source["frame_skip"],
            "drive_method": drive_conversion.get("method"),
            "native_mujoco_actuators_disabled": True,
            "native_mujoco_joint_limits_disabled": True,
            "drive_stiffness": drive_stiffness,
            "drive_damping": drive_damping,
            "drive_target_changes": int(np.count_nonzero(np.r_[
                True,
                np.linalg.norm(np.diff(source["drive_target"], axis=0), axis=1) > 1.0e-10,
            ])),
        },
        "object_control": {
            "free_joint": True,
            "pose_writes_after_initialization": 0,
            "direct_object_actuators": 0,
            "physx_damping_applied_once_per_source_step": True,
            "physx_gravity_applied_once_at_source_frame_start": True,
            "gravity_disabled_during_tgs_substeps": True,
            "source_gravity_m_s2": source_gravity.tolist(),
        },
        "friction_conversion": {
            "hand_friction_mode": args.hand_friction_mode,
            "hand_object_dynamic": 0.5,
            "hand_object_static": 0.7,
            "hand_static_speed_threshold_m_s": args.static_speed_threshold_m_s,
            "friction_cone": calibrated_cone,
            "noslip_iterations": calibrated_noslip,
            "microsteps_with_static_hand_contact": static_microsteps,
            "microsteps_without_static_hand_contact": dynamic_microsteps,
            "strong_anchor_time_constant_s": (
                None if strong_friction is None else strong_friction.time_constant
            ),
            "strong_anchor_correlation_distance_m": (
                None if strong_friction is None
                else strong_friction.correlation_distance
            ),
            "strong_anchor_stiffness_n_m": (
                None if strong_friction is None else strong_friction.stiffness
            ),
            "strong_anchor_damping_n_s_m": (
                None if strong_friction is None else strong_friction.damping
            ),
            "microsteps_with_active_strong_anchor": (
                strong_anchor_active_microsteps
            ),
            "active_strong_anchor_pair_microsteps": (
                strong_anchor_pair_microsteps
            ),
            "table_object_dynamic": 0.75,
            "table_object_static": 0.85,
            "table_static_speed_threshold_m_s": args.table_static_speed_threshold_m_s,
            "microsteps_with_static_table_contact": table_static_microsteps,
            "microsteps_without_static_table_contact": table_dynamic_microsteps,
            "hand_contact_time_constant_s": hand_contact_time,
            "bound_hand_contact_damping_ratio": hand_contact_damping,
            "runtime_hand_contact_damping_ratio": runtime_hand_contact_damping,
            "hand_contact_constraint_impedance": hand_contact_impedance,
        },
        "strict_gate": {
            "passed": passed,
            "hard_legal_table_hand": hard_legal,
            "opposed_contact_after_closure": opposed_contact,
            "opposed_contact_during_hold": hold_opposed,
            "held_lift": held_lift,
            "stable_hold": stable_hold,
            "rest_object_z_m": rest_z,
            "hold_min_lift_m": hold_min_lift,
            "hold_end_lift_m": hold_end_lift,
            "hold_vertical_span_m": hold_span,
            "minimum_hand_object_gap_m": minimum_object_gap,
            "maximum_object_penetration_m": args.max_object_penetration_m,
            "minimum_table_hand_gap_m": minimum_table_hand_gap,
            "table_hand_tolerance_m": XHAND_SELF_FLOOR_TOLERANCE_M,
        },
        "mujoco_vs_sapien": {
            "object_position_error_m_mean": float(np.mean(object_position_error)),
            "object_position_error_m_max": float(np.max(object_position_error)),
            "object_rotation_error_rad_mean": float(np.mean(object_rotation_error)),
            "object_rotation_error_rad_max": float(np.max(object_rotation_error)),
            "robot_qpos_l2_error_rad_mean": float(np.mean(robot_qpos_error)),
            "robot_qpos_l2_error_rad_max": float(np.max(robot_qpos_error)),
            "robot_qvel_l2_error_rad_s_mean": float(np.mean(robot_qvel_error)),
            "robot_qvel_l2_error_rad_s_max": float(np.max(robot_qvel_error)),
        },
        "trace": str(result_trace), "trace_sha256": sha256(result_trace),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
