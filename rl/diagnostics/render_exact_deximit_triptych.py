#!/usr/bin/env python3
"""Render a passed exact replay as a fixed-view real | ref | sim diagnostic.

This script never advances physics.  It accepts only a strict-passing isolated
MuJoCo replay and reads its stored states without advancing physics.  The
formal reference is rendered in its native model after every non-front camera
has been removed in memory.  The stored simulation hand-root, finger-joint and
free-object states are mapped onto that same visual rig.  Both panels therefore
share one world-fixed camera and one rendering style; the mapping is display
only and never changes the recorded physics trace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = Path(__file__).resolve().parent
for index, path in enumerate((ROOT, DIAGNOSTICS)):
    if str(path) not in sys.path:
        sys.path.insert(index, str(path))

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bodex_triptych import (
    CANONICAL_JOINTS,
    ROOT_JOINTS,
    fit,
    label,
    read_video,
    require_fixed_camera,
    scalar_joint_qpos_addresses,
    sha256,
)
from build_deximit_mujoco_scene import keep_fixed_front_camera_only


REPLAY_SCHEMA = "deximit_exact_mujoco_replay_v6_strong_friction_diagnostic_only"
REPORT_SCHEMA = REPLAY_SCHEMA
OUTPUT_SCHEMA = (
    "deximit_exact_real_ref_sim_v2_reference_style_fixed_camera_diagnostic_only"
)
CAMERA = "front"
SOURCE_DT_S = 1.0 / 240.0
PHASE_LABEL = {
    "pregrasp": "预抓取",
    "grasp": "接近",
    "squeeze": "收紧",
    "demonstrated_object_motion": "抬起",
    "hold": "保持",
}

# The original renderer was authored for the right-hand, single-object scene.
# These names extend that contract without changing its defaults.  A bimanual
# scene may use the conventional L_* arm joints and left_hand_* finger links;
# explicit CLI names below remain available for projects with different names.
LEFT_ROOT_JOINTS = tuple(name.replace("R_", "L_", 1) for name in ROOT_JOINTS)
LEFT_FINGER_JOINTS = tuple(
    name.replace("right_hand_", "left_hand_", 1) for name in CANONICAL_JOINTS
)
LEFT_ROOT_LINKS = tuple(
    name.replace("right_hand_", "left_hand_", 1)
    for name in (
        "right_hand_thumb_bend_link", "right_hand_index_bend_link",
        "right_hand_mid_link1", "right_hand_ring_link1",
        "right_hand_pinky_link1",
    )
)


def require_file(path: Path) -> Path:
    result = path.expanduser().resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"input is not a file: {result}")
    return result


def aligned_real_rows(
    phase: np.ndarray, *, pregrasp: int, grasp: int, motion: int,
) -> np.ndarray:
    """Map each simulated phase to the human-reviewed pickup interval."""
    values = np.asarray(phase).astype("U32")
    if (
        values.ndim != 1 or not len(values)
        or not 0 <= pregrasp <= grasp <= motion
        or set(np.unique(values)) - set(PHASE_LABEL)
    ):
        raise ValueError("phase or human row contract is malformed")
    rows = np.full(len(values), pregrasp, dtype=np.int64)
    for name, first, last in (
        ("grasp", pregrasp, grasp),
        ("demonstrated_object_motion", grasp, motion),
    ):
        indices = np.flatnonzero(values == name)
        if len(indices):
            rows[indices] = np.rint(np.linspace(first, last, len(indices))).astype(int)
    rows[values == "squeeze"] = grasp
    rows[values == "hold"] = motion
    return rows


def load_reference_model(path: Path) -> tuple[mujoco.MjModel, list[str]]:
    """Compile the native formal-reference scene after removing every other camera."""
    tree = ET.parse(path)
    removed = keep_fixed_front_camera_only(tree.getroot())
    compiler = tree.getroot().find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir"):
            directory = compiler.get(attribute)
            if directory is not None and not Path(directory).is_absolute():
                compiler.set(attribute, str((path.resolve().parent / directory).resolve()))
    xml = ET.tostring(tree.getroot(), encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    camera = require_fixed_camera(model, CAMERA)
    if model.ncam != 1 or int(model.cam_bodyid[camera]) != 0:
        raise ValueError("sanitized formal-reference camera is not world-fixed")
    return model, removed


def support_height(model: mujoco.MjModel, geom_name: str) -> float:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom < 0:
        raise ValueError(f"render model lacks support geom {geom_name!r}")
    geom_type = int(model.geom_type[geom])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return float(model.geom_pos[geom, 2])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(model.geom_pos[geom, 2] + model.geom_size[geom, 2])
    raise ValueError(f"unsupported support geom type for {geom_name!r}")


def _body_id(model: mujoco.MjModel, name: str | None) -> int:
    return -1 if name is None else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name,
    )


def _joint_id(model: mujoco.MjModel, name: str | None) -> int:
    return -1 if name is None else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name,
    )


def _first_body(model: mujoco.MjModel, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if _body_id(model, name) >= 0), None)


def _first_joint(model: mujoco.MjModel, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if _joint_id(model, name) >= 0), None)


def map_sim_to_reference_style(
    sim_model: mujoco.MjModel, ref_model: mujoco.MjModel,
    sim_qpos: np.ndarray, *,
    sim_left_wrist_body: str | None = None,
    ref_left_hand_body: str | None = None,
    sim_left_object_joint: str | None = None,
    ref_left_object_joint: str | None = None,
    sim_second_object_joint: str | None = None,
    ref_second_object_joint: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Put stored states on the reference rig, optionally including left hand/objects."""
    source = np.asarray(sim_qpos, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != sim_model.nq:
        raise ValueError("simulation qpos does not match its physics model")

    def body_transform(data: mujoco.MjData, body: int) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = data.xmat[body].reshape(3, 3)
        result[:3, 3] = data.xpos[body]
        return result

    # Each hand entry describes the same fused-wrist mapping used by the
    # original script.  The optional left entry is added only when its full
    # named visual rig exists; explicit names turn a missing part into an error.
    hands: list[dict[str, object]] = []
    hand_requests = (
        {
            "side": "right", "sim_wrist": _first_body(sim_model, ("wrist_3_link", "right_wrist_3_link", "right_hand_link")), "ref_hand": "right_hand_link",
            "sim_roots": (), "ref_roots": ROOT_JOINTS,
            "sim_fingers": CANONICAL_JOINTS, "ref_fingers": CANONICAL_JOINTS,
            "sim_links": ("right_hand_thumb_bend_link", "right_hand_index_bend_link", "right_hand_mid_link1", "right_hand_ring_link1", "right_hand_pinky_link1"),
            "ref_links": ("right_hand_thumb_bend_link", "right_hand_index_bend_link", "right_hand_mid_link1", "right_hand_ring_link1", "right_hand_pinky_link1"),
            "required": True,
        },
        {
            "side": "left",
            "sim_wrist": sim_left_wrist_body or _first_body(sim_model, ("left_wrist_3_link", "wrist_3_link_left", "left_wrist_link", "left_hand_link")),
            "ref_hand": ref_left_hand_body or _first_body(ref_model, ("left_hand_link",)),
            "sim_roots": (), "ref_roots": LEFT_ROOT_JOINTS,
            "sim_fingers": LEFT_FINGER_JOINTS, "ref_fingers": LEFT_FINGER_JOINTS,
            "sim_links": LEFT_ROOT_LINKS, "ref_links": LEFT_ROOT_LINKS,
            "required": any(value is not None for value in (sim_left_wrist_body, ref_left_hand_body)),
        },
    )
    sim_zero = mujoco.MjData(sim_model)
    ref_zero = mujoco.MjData(ref_model)
    mujoco.mj_forward(sim_model, sim_zero)
    mujoco.mj_forward(ref_model, ref_zero)
    for request in hand_requests:
        side = str(request["side"])
        sim_wrist_name = request["sim_wrist"]
        ref_hand_name = request["ref_hand"]
        present = sim_wrist_name is not None and ref_hand_name is not None
        if not present:
            if bool(request["required"]):
                raise ValueError("physics or reference model lacks right-hand mapping bodies")
            continue
        sim_wrist = _body_id(sim_model, str(sim_wrist_name))
        ref_hand = _body_id(ref_model, str(ref_hand_name))
        try:
            sim_root = np.asarray((), dtype=np.int64)
            ref_root = scalar_joint_qpos_addresses(ref_model, tuple(request["ref_roots"]))
            sim_fingers = scalar_joint_qpos_addresses(sim_model, tuple(request["sim_fingers"]))
            ref_fingers = scalar_joint_qpos_addresses(ref_model, tuple(request["ref_fingers"]))
        except ValueError:
            if bool(request["required"]):
                raise
            continue
        joint_signs: list[float] = []
        for sim_name, ref_name in zip(tuple(request["sim_fingers"]), tuple(request["ref_fingers"]), strict=True):
            sim_joint = _joint_id(sim_model, str(sim_name))
            ref_joint = _joint_id(ref_model, str(ref_name))
            dot = float(sim_model.jnt_axis[sim_joint] @ ref_model.jnt_axis[ref_joint])
            if not np.isclose(abs(dot), 1.0, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"finger joint axis {sim_name!r} is not sign-equivalent")
            joint_signs.append(1.0 if dot > 0.0 else -1.0)
        signs = np.asarray(joint_signs, dtype=np.float64)
        sim_wrist_world = body_transform(sim_zero, sim_wrist)
        ref_hand_world = body_transform(ref_zero, ref_hand)
        wrist_to_hand = None
        fixed_position_error = 0.0
        fixed_rotation_error = 0.0
        for sim_name, ref_name in zip(tuple(request["sim_links"]), tuple(request["ref_links"]), strict=True):
            sim_link = _body_id(sim_model, sim_name)
            ref_link = _body_id(ref_model, ref_name)
            if min(sim_link, ref_link) < 0:
                if bool(request["required"]):
                    raise ValueError(f"visual-rig mapping lacks fixed {side} finger root {sim_name!r}")
                wrist_to_hand = None
                break
            wrist_to_link = np.linalg.inv(sim_wrist_world) @ body_transform(sim_zero, sim_link)
            hand_to_link = np.linalg.inv(ref_hand_world) @ body_transform(ref_zero, ref_link)
            candidate = wrist_to_link @ np.linalg.inv(hand_to_link)
            if wrist_to_hand is None:
                wrist_to_hand = candidate
                continue
            residual = np.linalg.inv(wrist_to_hand) @ candidate
            fixed_position_error = max(fixed_position_error, float(np.linalg.norm(residual[:3, 3])))
            fixed_rotation_error = max(fixed_rotation_error, float(Rotation.from_matrix(residual[:3, :3]).magnitude()))
        if wrist_to_hand is None:
            if bool(request["required"]):
                raise ValueError(f"visual-rig mapping lacks complete {side} hand roots")
            continue
        if fixed_position_error > 2.0e-6 or fixed_rotation_error > 2.0e-5:
            raise ValueError(f"fused {side} hand-root transform is inconsistent across fingers")
        hands.append({
            "side": side, "sim_wrist": sim_wrist, "ref_hand": ref_hand,
            "sim_root": sim_root, "ref_root": ref_root,
            "sim_fingers": sim_fingers, "ref_fingers": ref_fingers,
            "signs": signs, "wrist_to_hand": wrist_to_hand,
            "fixed_position_error": fixed_position_error,
            "fixed_rotation_error": fixed_rotation_error,
        })

    def object_pair(sim_name: str | None, ref_name: str | None, label_name: str, required: bool) -> dict[str, object] | None:
        if sim_name is None and ref_name is None:
            return None
        if sim_name is None:
            sim_name = _first_joint(sim_model, ("left_object_joint", "target_object_joint", "second_object_joint"))
        if ref_name is None:
            ref_name = _first_joint(ref_model, ("left_object_joint", "target_object_joint", "second_object_joint"))
        sim_joint = _joint_id(sim_model, sim_name)
        ref_joint = _joint_id(ref_model, ref_name)
        if min(sim_joint, ref_joint) < 0:
            if required:
                raise ValueError(f"requested {label_name} object mapping is missing")
            return None
        if int(sim_model.jnt_type[sim_joint]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(ref_model.jnt_type[ref_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"{label_name} object joint must be free")
        return {"label": label_name, "sim": int(sim_model.jnt_qposadr[sim_joint]), "ref": int(ref_model.jnt_qposadr[ref_joint]), "sim_name": sim_name, "ref_name": ref_name}

    right_object = object_pair("right_object_joint", "right_object_joint", "right_object", True)
    objects = [right_object]
    left_object = object_pair(sim_left_object_joint, ref_left_object_joint, "left_object", sim_left_object_joint is not None or ref_left_object_joint is not None)
    second_object = object_pair(sim_second_object_joint, ref_second_object_joint, "second_object", sim_second_object_joint is not None or ref_second_object_joint is not None)
    for item in (left_object, second_object):
        if item is not None and not any(item["sim"] == existing["sim"] and item["ref"] == existing["ref"] for existing in objects):
            objects.append(item)

    sim_support = "table" if mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_GEOM, "table") >= 0 else "floor"
    z_shift = support_height(ref_model, "floor") - support_height(sim_model, sim_support)
    mapped = np.repeat(ref_model.qpos0[None], len(source), axis=0)
    expected_hands = [(hand, np.empty((len(source), 3), dtype=np.float64), np.empty((len(source), 3, 3), dtype=np.float64)) for hand in hands]
    sim_data = mujoco.MjData(sim_model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        for index, qpos in enumerate(source):
            sim_data.qpos[:] = qpos
            sim_data.qvel[:] = 0.0
            mujoco.mj_forward(sim_model, sim_data)
            for hand, positions, rotations in expected_hands:
                world_hand = body_transform(sim_data, int(hand["sim_wrist"])) @ hand["wrist_to_hand"]
                position = world_hand[:3, 3].copy()
                position[2] += z_shift
                rotation = world_hand[:3, :3].copy()
                roll, pitch, negative_yaw = Rotation.from_matrix(rotation).as_euler("ZXY")
                ref_root = hand["ref_root"]
                mapped[index, ref_root[:3]] = position
                mapped[index, ref_root[3:]] = (roll, pitch, -negative_yaw)
                mapped[index, hand["ref_fingers"]] = qpos[hand["sim_fingers"]] * hand["signs"]
                positions[index] = position
                rotations[index] = rotation
            for item in objects:
                mapped[index, item["ref"] : item["ref"] + 7] = qpos[item["sim"] : item["sim"] + 7]
                mapped[index, item["ref"] + 2] += z_shift

    ref_data = mujoco.MjData(ref_model)
    hand_errors = []
    for hand, expected_position, expected_rotation in expected_hands:
        position_error = 0.0
        rotation_error = 0.0
        for index in np.linspace(0, len(mapped) - 1, min(65, len(mapped)), dtype=int):
            ref_data.qpos[:] = mapped[index]
            ref_data.qvel[:] = 0.0
            mujoco.mj_forward(ref_model, ref_data)
            ref_hand = int(hand["ref_hand"])
            position_error = max(position_error, float(np.linalg.norm(ref_data.xpos[ref_hand] - expected_position[index])))
            rotation_error = max(rotation_error, float(Rotation.from_matrix(expected_rotation[index].T @ ref_data.xmat[ref_hand].reshape(3, 3)).magnitude()))
        if position_error > 1.0e-9 or rotation_error > 1.0e-9:
            raise RuntimeError(f"simulation state does not map exactly to reference {hand['side']} visual rig")
        hand_errors.append({"side": hand["side"], "sampled_position_error_m_max": position_error, "sampled_rotation_error_rad_max": rotation_error})
    right = hands[0]
    right_error = next(item for item in hand_errors if item["side"] == "right")
    return mapped, {
        "kind": "render-only bimanual-capable hand-root, finger-joint and free-object mapping",
        "physics_state_changed": False,
        "reference_support_height_m": support_height(ref_model, "floor"),
        "simulation_support_height_m": support_height(sim_model, sim_support),
        "world_z_translation_m": z_shift,
        "hands": [{"side": hand["side"], "sim_wrist_body": hand["sim_wrist"], "reference_hand_body": hand["ref_hand"], "fixed_finger_root_position_residual_m_max": hand["fixed_position_error"], "fixed_finger_root_rotation_residual_rad_max": hand["fixed_rotation_error"]} for hand in hands],
        "objects": [{key: value for key, value in item.items() if key in {"label", "sim_name", "ref_name"}} for item in objects],
        "finger_joint_axis_sign": {name: float(sign) for name, sign in zip(CANONICAL_JOINTS, right["signs"], strict=True)},
        "fused_wrist_to_hand_transform": right["wrist_to_hand"].tolist(),
        # Preserve the original single-hand report keys for downstream audits.
        "fixed_finger_root_position_residual_m_max": right["fixed_position_error"],
        "fixed_finger_root_rotation_residual_rad_max": right["fixed_rotation_error"],
        "sampled_hand_position_error_m_max": right_error["sampled_position_error_m_max"],
        "sampled_hand_rotation_error_rad_max": right_error["sampled_rotation_error_rad_max"],
        "sampled_hand_errors": hand_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--replay-trace", type=Path, required=True)
    parser.add_argument("--formal-reference", "--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-kind", choices=("formal", "mink-kinematic"), default="formal",
    )
    parser.add_argument("--real-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--real-pregrasp-row", type=int, required=True)
    parser.add_argument("--real-grasp-row", type=int, required=True)
    parser.add_argument("--real-motion-row", type=int, required=True)
    parser.add_argument("--camera-x-m", type=float, default=0.70)
    parser.add_argument("--sim-left-wrist-body")
    parser.add_argument("--ref-left-hand-body")
    parser.add_argument("--sim-left-object-joint")
    parser.add_argument("--ref-left-object-joint")
    parser.add_argument("--sim-second-object-joint")
    parser.add_argument("--ref-second-object-joint")
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.camera_x_m):
        raise ValueError("render fps or fixed camera position is invalid")

    scene = require_file(args.scene)
    reference_scene = require_file(args.reference_scene)
    replay_report_path = require_file(args.replay_report)
    replay_trace_path = require_file(args.replay_trace)
    reference_path = require_file(args.formal_reference)
    real_path = require_file(args.real_video)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite triptych output: {output}")

    replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    if (
        replay_report.get("schema") != REPORT_SCHEMA
        or replay_report.get("diagnostic_only") is not True
        or replay_report.get("formal_renderer_3_3_eligible") is not False
        or replay_report.get("rendered") is not False
        or replay_report.get("strict_gate", {}).get("passed") is not True
        or Path(str(replay_report.get("scene", ""))).resolve() != scene
        or replay_report.get("scene_sha256") != sha256(scene)
        or Path(str(replay_report.get("trace", ""))).resolve() != replay_trace_path
        or replay_report.get("trace_sha256") != sha256(replay_trace_path)
    ):
        raise ValueError("only an unrendered strict-passing exact replay is accepted")

    with np.load(replay_trace_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) != REPLAY_SCHEMA
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or not bool(np.asarray(values["strict_gate_passed"]).item())
            or str(np.asarray(values["scene_sha256"]).item()) != sha256(scene)
        ):
            raise ValueError("replay trace does not carry the passing render contract")
        sim_qpos = np.asarray(values["qpos"], dtype=np.float64)
        phase = np.asarray(values["phase"]).astype("U32")
    with np.load(reference_path, allow_pickle=False) as values:
        if "qpos" not in values.files:
            raise ValueError("formal reference lacks qpos")
        ref_qpos = np.asarray(values["qpos"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(scene))
    camera = require_fixed_camera(model, CAMERA)
    ref_model, removed_reference_cameras = load_reference_model(reference_scene)
    ref_camera = require_fixed_camera(ref_model, CAMERA)
    source_sim_camera_position = model.cam_pos[camera].copy()
    source_ref_camera_position = ref_model.cam_pos[ref_camera].copy()
    ref_model.cam_pos[ref_camera, 0] = args.camera_x_m
    if (
        model.ncam != 1 or int(model.cam_bodyid[camera]) != 0
        or sim_qpos.ndim != 2 or sim_qpos.shape[1] != model.nq
        or len(phase) != len(sim_qpos)
        or ref_qpos.ndim != 2 or ref_qpos.shape[1] != ref_model.nq
        or not np.isfinite(sim_qpos).all() or not np.isfinite(ref_qpos).all()
    ):
        raise ValueError("fixed-camera scene or recorded qpos contract is malformed")
    sim_display_qpos, sim_display_mapping = map_sim_to_reference_style(
        model, ref_model, sim_qpos,
        sim_left_wrist_body=args.sim_left_wrist_body,
        ref_left_hand_body=args.ref_left_hand_body,
        sim_left_object_joint=args.sim_left_object_joint,
        ref_left_object_joint=args.ref_left_object_joint,
        sim_second_object_joint=args.sim_second_object_joint,
        ref_second_object_joint=args.ref_second_object_joint,
    )

    real = read_video(real_path)
    real_rows = aligned_real_rows(
        phase, pregrasp=args.real_pregrasp_row,
        grasp=args.real_grasp_row, motion=args.real_motion_row,
    )
    if int(real_rows.max()) >= len(real) or len(ref_qpos) != len(real):
        raise ValueError("real video and formal reference row counts do not align")
    stride = max(1, int(round(1.0 / (args.fps * SOURCE_DT_S))))
    indices = np.arange(0, len(sim_qpos), stride, dtype=np.int64)
    if indices[-1] != len(sim_qpos) - 1:
        indices = np.append(indices, len(sim_qpos) - 1)

    output.mkdir(parents=True)
    video = output / "real_ref_sim_passed_fixed_third_person.mp4"
    ref_model.vis.global_.offwidth = 720
    ref_model.vis.global_.offheight = 480
    ref_renderer = mujoco.Renderer(ref_model, height=480, width=720)
    ref_data = mujoco.MjData(ref_model)
    sim_data = mujoco.MjData(ref_model)
    try:
        with imageio.get_writer(
            video, fps=args.fps, codec="libx264", pixelformat="yuv420p",
        ) as writer:
            for index in indices.tolist():
                real_row = int(real_rows[index])
                ref_data.qpos[:] = ref_qpos[real_row]
                ref_data.qvel[:] = 0.0
                mujoco.mj_forward(ref_model, ref_data)
                ref_renderer.update_scene(ref_data, camera=ref_camera)
                reference = ref_renderer.render().copy()

                sim_data.qpos[:] = sim_display_qpos[index]
                sim_data.qvel[:] = 0.0
                mujoco.mj_forward(ref_model, sim_data)
                ref_renderer.update_scene(sim_data, camera=ref_camera)
                simulation = ref_renderer.render().copy()
                phase_name = PHASE_LABEL[str(phase[index])]
                time_s = (index + 1) * SOURCE_DT_S
                panels = (
                    label(
                        fit(real[real_row]), "真人：原始视频",
                        f"人工核对行 {real_row}；按抓取阶段对齐",
                    ),
                    label(
                        reference,
                        "reference：正式参考动作" if args.reference_kind == "formal"
                        else "reference：MINK 运动学参考",
                        f"原视频行 {real_row}；参考场景原生显示"
                        if args.reference_kind == "formal"
                        else f"原视频行 {real_row}；非物理成功证据",
                    ),
                    label(
                        simulation, "sim：自由物体（严格通过）",
                        f"复刻 reference 外观；阶段：{phase_name}；{time_s:.2f} 秒",
                    ),
                )
                writer.append_data(np.concatenate(panels, axis=1))
    finally:
        ref_renderer.close()

    report = {
        "schema": OUTPUT_SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "physics_advanced_during_render": False,
        "candidate_index": int(replay_report["candidate_index"]),
        "strict_gate_passed": True,
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "reference_scene": str(reference_scene),
        "reference_scene_sha256": sha256(reference_scene),
        "replay_report": str(replay_report_path),
        "replay_report_sha256": sha256(replay_report_path),
        "replay_trace": str(replay_trace_path),
        "replay_trace_sha256": sha256(replay_trace_path),
        "formal_reference": str(reference_path),
        "formal_reference_sha256": sha256(reference_path),
        "reference_kind": args.reference_kind,
        "real_video": str(real_path),
        "real_video_sha256": sha256(real_path),
        "camera": {
            "name": CAMERA,
            "mode": "world-fixed third-person",
            "simulation_scene_camera_count": int(model.ncam),
            "sanitized_reference_scene_camera_count": int(ref_model.ncam),
            "removed_reference_cameras": removed_reference_cameras,
            "following_camera_present": False,
            "source_simulation_position_m": source_sim_camera_position.tolist(),
            "source_reference_position_m": source_ref_camera_position.tolist(),
            "shared_render_position_m": ref_model.cam_pos[ref_camera].tolist(),
            "render_only_horizontal_reframing": True,
        },
        "reference_render_transform": {
            "kind": "none; qpos rendered in its native formal-reference scene",
            "joint_or_relative_motion_changed": False,
        },
        "simulation_display_mapping": sim_display_mapping,
        "style_contract": {
            "reference_and_sim_share_model": True,
            "reference_and_sim_share_materials": True,
            "reference_and_sim_share_lighting": True,
            "reference_and_sim_share_support_surface": True,
            "reference_and_sim_share_camera": True,
            "simulation_physics_scene_used_for_rendering": False,
            "simulation_physics_trace_remains_source_of_motion": True,
        },
        "real_alignment": {
            "kind": "human-reviewed phase alignment; not frame-synchronous",
            "pregrasp_row": args.real_pregrasp_row,
            "grasp_row": args.real_grasp_row,
            "motion_row": args.real_motion_row,
        },
        "frames": int(len(indices)),
        "fps": int(args.fps),
        "duration_s": float(len(indices) / args.fps),
        "source_state_dt_s": SOURCE_DT_S,
        "source_state_stride": int(stride),
        "video": str(video),
        "video_sha256": sha256(video),
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
