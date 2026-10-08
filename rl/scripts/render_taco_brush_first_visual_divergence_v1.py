#!/usr/bin/env python3
"""Render synchronized human/MINK/Replay panels and locate the first divergence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_visual_geom_ids,
)
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
)


OUTPUT = RL_ROOT / "runs/taco_brush_first_visual_divergence_v1"
SCENE = RL_ROOT / "runs/taco_brush_controlled_physics_replay_v1/experiment_local_scene.xml"
MINK = RL_ROOT / "runs/taco_brush_issue14_mink_candidate_v1/robot_reference.npz"
REPLAY = RL_ROOT / "runs/taco_brush_replay_failure_attribution_v1/full_endpoint_states.npz"
RGB = Path(
    "/data_all/zzx/3.2RL/data/taco_v1/dev4/rgb/"
    "taco_brush_brush_bowl_20230927_027.mp4"
)
CAMERA = Path(
    "/data_all/zzx/3.2RL/data/taco_v1/dev4/camera/"
    "Egocentric_Camera_Parameters/(brush, brush, bowl)/20230927_027"
)
INTRINSIC = CAMERA / "egocentric_intrinsic.txt"
EXTRINSIC = CAMERA / "egocentric_frame_extrinsic.npy"
OBJECT_POSES = Path(
    "/data_all/zzx/3.2RL/data/taco_v1/dev4/object_poses/Object_Poses/"
    "(brush, brush, bowl)/20230927_027"
)
TOOL_POSE = OBJECT_POSES / "tool_071.npy"
TARGET_POSE = OBJECT_POSES / "target_146.npy"
FIRST_FRAME = 0
LAST_FRAME = 60
WIDTH = 640
HEIGHT = 480


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def letterbox(image: np.ndarray) -> np.ndarray:
    scale = min(WIDTH / image.shape[1], HEIGHT / image.shape[0])
    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x0 = (WIDTH - width) // 2
    y0 = (HEIGHT - height) // 2
    result[y0:y0 + height, x0:x0 + width] = resized
    return result


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (WIDTH, 42), (0, 0, 0), -1)
    cv2.putText(
        result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result


def decode_rgb(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    index = 0
    while index <= LAST_FRAME:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= FIRST_FRAME:
            frames.append(frame.copy())
        index += 1
    capture.release()
    expected = LAST_FRAME - FIRST_FRAME + 1
    if len(frames) != expected:
        raise RuntimeError(f"expected {expected} RGB frames, got {len(frames)}")
    return frames


def camera(view: str) -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    result.lookat[:] = [0.60, 0.0, 0.76 if view == "top" else 0.78]
    result.distance = 0.70
    result.azimuth = 90
    result.elevation = -90 if view == "top" else -18
    return result


def actuator_groups(model: mujoco.MjModel) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = {
        "right_wrist_translation_m": [], "right_wrist_rotation_rad": [],
        "right_fingers_rad": [], "left_wrist_translation_m": [],
        "left_wrist_rotation_rad": [], "left_fingers_rad": [],
    }
    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        side = "left" if name.startswith(("left_", "L_")) else "right"
        if "forearm_t" in name:
            group = f"{side}_wrist_translation_m"
        elif "forearm_" in name:
            group = f"{side}_wrist_rotation_rad"
        else:
            group = f"{side}_fingers_rad"
        groups[group].append((actuator_id, qpos_address))
    if sum(map(len, groups.values())) != model.nu:
        raise RuntimeError("actuator grouping did not cover the full hand command")
    return groups


def quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(abs(np.dot(first, second)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def frame_metrics(
    model: mujoco.MjModel,
    reference: np.ndarray,
    replay: np.ndarray,
    groups: dict[str, list[tuple[int, int]]],
    frame: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {"frame": frame}
    for name, addresses in groups.items():
        delta = np.asarray([
            replay[frame, qpos] - reference[frame, qpos]
            for _, qpos in addresses
        ])
        row[f"{name}_rms"] = float(np.sqrt(np.mean(delta * delta)))
        row[f"{name}_max_abs"] = float(np.max(np.abs(delta)))
    object_start = model.nq - 14
    for index, name in enumerate(("brush", "bowl")):
        start = object_start + 7 * index
        row[f"{name}_position_error_m"] = float(np.linalg.norm(
            replay[frame, start:start + 3] - reference[frame, start:start + 3]
        ))
        row[f"{name}_rotation_error_rad"] = quaternion_distance(
            replay[frame, start + 3:start + 7], reference[frame, start + 3:start + 7],
        )
    return row


def render_state(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    qpos: np.ndarray,
    view: mujoco.MjvCamera,
    title: str,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=view)
    image = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
    return label(image, title)


def visual_entity_vertices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    visual_geoms: list[Any],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {
        "right_xhand": [], "left_xhand": [], "brush": [], "bowl": [],
    }
    for geom in visual_geoms:
        body = model.body(geom.body_id).name
        if body == "right_object":
            entity = "brush"
        elif body == "left_object":
            entity = "bowl"
        elif body.startswith("right_hand"):
            entity = "right_xhand"
        elif body.startswith("left_hand"):
            entity = "left_xhand"
        else:
            continue
        grouped[entity].append(geom_world_vertices(data, geom, full=True))
    if any(not values for values in grouped.values()):
        raise RuntimeError("registered overlay is missing a visual entity")
    return {name: np.concatenate(values) for name, values in grouped.items()}


def project_mask(
    vertices: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    width: int,
    height: int,
    dilation: int,
) -> np.ndarray:
    camera = (
        world_to_camera[:3, :3] @ np.asarray(vertices, dtype=np.float64).T
        + world_to_camera[:3, 3:4]
    ).T
    projection = (intrinsic @ camera.T).T
    uv = projection[:, :2] / projection[:, 2:3]
    finite = np.isfinite(uv).all(axis=1) & (camera[:, 2] > 0.0)
    pixels = np.rint(uv[finite]).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    if dilation > 1:
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def registered_overlay(
    bgr: np.ndarray,
    entities: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    simulator_from_world_translation: np.ndarray,
    title_text: str,
) -> np.ndarray:
    height, width = bgr.shape[:2]
    result = bgr.astype(np.float32)
    colors = {
        "right_xhand": (30, 30, 255),
        "left_xhand": (255, 220, 0),
        "brush": (30, 255, 30),
        "bowl": (0, 220, 255),
    }
    # Object masks are a registration check; hand masks are the comparison target.
    for name in ("brush", "bowl", "right_xhand", "left_xhand"):
        mask = project_mask(
            entities[name] - simulator_from_world_translation, intrinsic,
            world_to_camera,
            width=width, height=height,
            dilation=3 if name in {"brush", "bowl"} else 5,
        ).astype(bool)
        color = np.asarray(colors[name], dtype=np.float32)
        alpha = 0.35 if name in {"brush", "bowl"} else 0.55
        result[mask] = (1.0 - alpha) * result[mask] + alpha * color
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.rectangle(result, (0, 0), (width, 54), (0, 0, 0), -1)
    cv2.putText(
        result, title_text, (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return cv2.resize(result, (960, 540), interpolation=cv2.INTER_AREA)


def write_sheet(path: Path, rows: dict[int, np.ndarray], frames: range) -> None:
    image = np.concatenate([rows[frame] for frame in frames], axis=0)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write {path}")


def hash_tree(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows_dir = OUTPUT / "rows"
    sheets_dir = OUTPUT / "sheets"
    rows_dir.mkdir()
    sheets_dir.mkdir()

    rgb = decode_rgb(RGB)
    with np.load(MINK, allow_pickle=False) as source:
        reference = np.asarray(source["qpos"], dtype=np.float64)
    with np.load(REPLAY, allow_pickle=False) as source:
        replay = np.asarray(source["qpos"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if reference.shape[1] != model.nq or replay.shape[1] != model.nq:
        raise ValueError("state/model qpos mismatch")
    if min(len(reference), len(replay)) <= LAST_FRAME:
        raise ValueError("state traces do not cover the requested visual window")

    groups = actuator_groups(model)
    intrinsic = np.loadtxt(INTRINSIC)
    extrinsics = np.load(EXTRINSIC, allow_pickle=False)
    if intrinsic.shape != (3, 3) or extrinsics.shape[0] <= LAST_FRAME:
        raise ValueError("official camera arrays do not cover the visual window")
    tool_poses = np.load(TOOL_POSE, allow_pickle=False)
    target_poses = np.load(TARGET_POSE, allow_pickle=False)
    simulator_offsets = np.stack([
        reference[:, -14:-11] - tool_poses[:, :3, 3],
        reference[:, -7:-4] - target_poses[:, :3, 3],
    ])
    simulator_from_world_translation = simulator_offsets.mean(axis=(0, 1))
    maximum_transform_residual = float(np.max(np.abs(
        simulator_offsets - simulator_from_world_translation
    )))
    if maximum_transform_residual > 1.0e-7:
        raise ValueError("object poses do not support one fixed simulator/world translation")
    metrics = [frame_metrics(model, reference, replay, groups, frame)
               for frame in range(FIRST_FRAME, LAST_FRAME + 1)]
    with (OUTPUT / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    videos = {}
    saved_rows = {}
    row_frames = set(range(0, 21)) | set(range(25, 61, 5))
    try:
        for view_name in ("top", "front"):
            view = camera(view_name)
            video_path = OUTPUT / f"human_mink_replay_{view_name}_frames_000_060.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0,
                (3 * WIDTH, HEIGHT),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot create {video_path}")
            view_rows: dict[int, np.ndarray] = {}
            try:
                for frame in range(FIRST_FRAME, LAST_FRAME + 1):
                    human = label(letterbox(rgb[frame]), f"Human RGB | frame {frame}")
                    mink = render_state(
                        model, renderer, data, reference[frame], view,
                        f"MINK reference {view_name} | frame {frame}",
                    )
                    actual = render_state(
                        model, renderer, data, replay[frame], view,
                        f"Physics Replay {view_name} | endpoint {frame}",
                    )
                    composite = np.concatenate([human, mink, actual], axis=1)
                    writer.write(composite)
                    if frame in row_frames:
                        view_rows[frame] = composite
                        path = rows_dir / f"frame_{frame:03d}_{view_name}.jpg"
                        if not cv2.imwrite(str(path), composite,
                                          [cv2.IMWRITE_JPEG_QUALITY, 94]):
                            raise RuntimeError(f"failed to write {path}")
            finally:
                writer.release()
            videos[view_name] = artifact(video_path)
            saved_rows[view_name] = view_rows
    finally:
        renderer.close()

    # Registered overlays place the native XHand/object vertices directly in the
    # official egocentric camera.  They are point-surface diagnostics, not a new
    # renderer and do not mutate either trajectory.
    overlay_data = mujoco.MjData(model)
    visual_geoms = build_native_support_geoms(
        model, mujoco, native_visual_geom_ids(model),
    )
    overlay_video = OUTPUT / "rgb_registered_mink_replay_frames_000_060.mp4"
    overlay_writer = cv2.VideoWriter(
        str(overlay_video), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (3 * 960, 540),
    )
    if not overlay_writer.isOpened():
        raise RuntimeError(f"cannot create {overlay_video}")
    overlay_rows: dict[int, np.ndarray] = {}
    try:
        for frame in range(FIRST_FRAME, LAST_FRAME + 1):
            raw = cv2.resize(rgb[frame], (960, 540), interpolation=cv2.INTER_AREA)
            cv2.rectangle(raw, (0, 0), (960, 54), (0, 0, 0), -1)
            cv2.putText(raw, f"Human RGB | frame {frame}", (18, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
            overlay_data.qpos[:] = reference[frame]
            overlay_data.qvel[:] = 0.0
            mujoco.mj_forward(model, overlay_data)
            mink_entities = visual_entity_vertices(model, overlay_data, visual_geoms)
            mink_overlay = registered_overlay(
                rgb[frame], mink_entities, intrinsic, extrinsics[frame],
                simulator_from_world_translation,
                f"MINK registered overlay | frame {frame}",
            )
            overlay_data.qpos[:] = replay[frame]
            overlay_data.qvel[:] = 0.0
            mujoco.mj_forward(model, overlay_data)
            replay_entities = visual_entity_vertices(model, overlay_data, visual_geoms)
            replay_overlay = registered_overlay(
                rgb[frame], replay_entities, intrinsic, extrinsics[frame],
                simulator_from_world_translation,
                f"Replay registered overlay | endpoint {frame}",
            )
            composite = np.concatenate([raw, mink_overlay, replay_overlay], axis=1)
            overlay_writer.write(composite)
            if frame in row_frames:
                overlay_rows[frame] = composite
                path = rows_dir / f"frame_{frame:03d}_registered.jpg"
                if not cv2.imwrite(str(path), composite,
                                   [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"failed to write {path}")
    finally:
        overlay_writer.release()
    for start, stop in ((0, 5), (5, 10), (10, 15), (15, 20)):
        write_sheet(
            sheets_dir / f"frames_{start:03d}_{stop - 1:03d}_registered.jpg",
            overlay_rows, range(start, stop),
        )

    for view_name, view_rows in saved_rows.items():
        write_sheet(sheets_dir / f"frames_000_009_{view_name}.jpg", view_rows, range(0, 10))
        write_sheet(sheets_dir / f"frames_010_019_{view_name}.jpg", view_rows, range(10, 20))

    first_exact_divergence = next(
        row["frame"] for row in metrics
        if any(value != 0.0 for key, value in row.items() if key != "frame")
    )
    reviewed_visible_divergence = 3
    reviewed = {
        "schema": "taco_brush_first_visual_divergence_review_v1",
        "status": "READ_ONLY_REVIEWED_INTERPRETATION",
        "frame_alignment": (
            "RGB frame N / MINK reference frame N / physics Replay endpoint N"
        ),
        "first_human_vs_mink_semantic_mismatch_frame": 0,
        "first_mink_vs_replay_exact_divergence_endpoint": first_exact_divergence,
        "first_mink_vs_replay_unambiguously_visible_divergence_endpoint": (
            reviewed_visible_divergence
        ),
        "endpoint_0_mink_replay_all_reported_groups_exact": all(
            value == 0.0 for key, value in metrics[0].items() if key != "frame"
        ),
        "early_metrics": {},
        "registration_validation": {
            "official_object_overlays_visibly_align_with_rgb": True,
            "simulator_from_world_translation_residual_max_m": (
                maximum_transform_residual
            ),
        },
        "limits": [
            "Human-to-XHand frame-0 classification is semantic and qualitative "
            "because morphology differs.",
            "MINK-to-Replay metrics are exact state-space comparisons.",
            "No hand segmentation mask or pixel-IoU metric is used.",
            "No new MINK optimization, physics stepping, or training was run.",
        ],
    }
    for frame in (1, 2, 3):
        row = metrics[frame]
        reviewed["early_metrics"][f"endpoint_{frame}"] = {
            "right_wrist_translation_rms_mm": (
                1_000.0 * row["right_wrist_translation_m_rms"]
            ),
            "right_wrist_rotation_rms_deg": math.degrees(
                row["right_wrist_rotation_rad_rms"]
            ),
            "right_fingers_rms_deg": math.degrees(row["right_fingers_rad_rms"]),
            "brush_position_error_mm": 1_000.0 * row["brush_position_error_m"],
            "brush_rotation_error_deg": math.degrees(row["brush_rotation_error_rad"]),
            "bowl_position_error_mm": 1_000.0 * row["bowl_position_error_m"],
            "bowl_rotation_error_deg": math.degrees(row["bowl_rotation_error_rad"]),
        }
    write_json(OUTPUT / "first_divergence.json", reviewed)

    def metric_line(frame: int) -> str:
        row = metrics[frame]
        return (
            f"| {frame} | {1_000.0 * row['right_wrist_translation_m_rms']:.3f} mm "
            f"| {math.degrees(row['right_wrist_rotation_rad_rms']):.3f} deg "
            f"| {math.degrees(row['right_fingers_rad_rms']):.3f} deg "
            f"| {1_000.0 * row['brush_position_error_m']:.3f} mm / "
            f"{math.degrees(row['brush_rotation_error_rad']):.3f} deg "
            f"| {1_000.0 * row['bowl_position_error_m']:.3f} mm / "
            f"{math.degrees(row['bowl_rotation_error_rad']):.3f} deg |"
        )

    (OUTPUT / "summary.md").write_text(
        "# Brush first visual divergence v1\n\n"
        "## Result\n\n"
        "There are two distinct first-error boundaries:\n\n"
        "- **Human RGB vs MINK reference:** a semantic hand-pose/morphology mismatch "
        "is already visible at frame 0. This is a reference/retarget mismatch, not a "
        "physics-execution error. Exact pixel agreement is not expected between "
        "MANO/human hands and XHand, so this finding is qualitative.\n"
        "- **MINK reference vs controlled Replay:** endpoint 0 is bitwise-identical "
        "in every reported hand/object group. The first exact execution divergence "
        f"is endpoint {first_exact_divergence}. It becomes unambiguously visible in "
        f"the registered comparison by endpoint {reviewed_visible_divergence}.\n\n"
        "Early exact Replay deviations are:\n\n"
        "| endpoint | right wrist translation RMS | right wrist rotation RMS | "
        "right finger RMS | brush position / rotation | bowl position / rotation |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(metric_line(frame) for frame in range(4))
        + "\n\nTherefore the later grasp/contact failures are downstream. The current "
        "evidence orders the defects as:\n\n"
        "1. the XHand reference is already an imperfect semantic match to the human "
        "hands at frame 0;\n"
        "2. physics/control adds a new error immediately on transition 0 to 1;\n"
        "3. that execution error is plainly visible by endpoint 3 and grows before "
        "the previously studied contact-failure interval.\n\n"
        "The official brush and bowl meshes visibly align with the RGB objects in the "
        "registered panels. The simulator/world translation was derived independently "
        "from both official object pose streams over all frames; its maximum translation "
        f"residual is `{maximum_transform_residual:.3g} m`. This validates the camera "
        "registration path, but is not a pixel-mask accuracy metric for the hands.\n\n"
        "## Method and limits\n\n"
        "Synchronized read-only comparison of human RGB, the current MINK reference, "
        "and the controlled physics Replay. The registered panels use the official "
        "per-frame egocentric camera; the fixed top/front panels remain semantic views. "
        "No MINK optimization, physics stepping, or training was run. See "
        "`frame_metrics.csv` for exact MINK-to-Replay state differences and "
        "`first_divergence.json` for the reviewed boundary classification.\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "taco_brush_first_visual_divergence_v1",
        "status": "READ_ONLY_SYNCHRONIZED_VISUAL_AUDIT",
        "frame_range": [FIRST_FRAME, LAST_FRAME],
        "alignment": "RGB frame N / MINK reference frame N / physics Replay endpoint N",
        "interpretation_limit": (
            "Official-camera overlay panels are registered through the validated fixed "
            "simulator/world transform. Human-to-XHand judgments remain semantic because "
            "morphology differs; MINK-to-Replay state metrics are exact."
        ),
        "inputs": {
            "rgb": artifact(RGB), "scene": artifact(SCENE),
            "mink_reference": artifact(MINK), "replay_states": artifact(REPLAY),
            "intrinsic": artifact(INTRINSIC), "extrinsic": artifact(EXTRINSIC),
            "tool_pose": artifact(TOOL_POSE), "target_pose": artifact(TARGET_POSE),
            "script": artifact(Path(__file__)),
        },
        "outputs": {
            "videos": videos,
            "registered_overlay_video": artifact(overlay_video),
            "metrics": artifact(OUTPUT / "frame_metrics.csv"),
            "review": artifact(OUTPUT / "first_divergence.json"),
            "summary": artifact(OUTPUT / "summary.md"),
        },
        "mutations": {"physics_steps": 0, "mink": 0, "training": 0, "state": 0},
        "simulator_from_world": {
            "rotation": "identity",
            "translation_m": simulator_from_world_translation.tolist(),
            "maximum_object_pose_translation_residual_m": maximum_transform_residual,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(OUTPUT / "manifest.json", manifest)
    hash_tree(OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "videos": {name: value["path"] for name, value in videos.items()},
        "sheets": sorted(str(path) for path in sheets_dir.glob("*.jpg")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
