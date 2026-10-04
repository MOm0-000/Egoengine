#!/usr/bin/env python3
"""Render frozen evidence for the bounded control-aware startup experiment.

This module is deliberately offline: it loads saved arrays, sets qpos for
kinematics/rendering, and never advances MuJoCo time or constructs an RL policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zipfile

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from video_to_spider.rl.core.visual_evidence import (
    FRAME_MAP_FIELDS,
    contact_transitions,
    endpoint_rows,
    endpoint_status,
    pending_review,
    resolve_frame_map,
    semantic_contact_sets,
    sha256_file,
    sha256_manifest,
    top_positive_increments,
    validate_publication,
    video_frame_map,
)


ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ROOT / "runs/taco_pour_control_aware_startup_v1"
REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("A_REPLAY", "A_PLAN", "L_REPLAY", "L_PLAN")
KEYS = (0, 1, 5, 10, 14, 15, 16, 20, 25, 30, 35, 40)
COLORS = {
    "A_REPLAY": "#6b7280",
    "A_PLAN": "#2563eb",
    "L_REPLAY": "#d97706",
    "L_PLAN": "#059669",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def annotate(image: np.ndarray, *lines: str) -> np.ndarray:
    return vismod.annotate_panel(image, lines)


def save_sheet(images: list[np.ndarray], path: Path, columns: int = 2) -> None:
    if not images:
        raise ValueError("cannot save an empty review sheet")
    tiles = [cv2.resize(image, (1200, 360), interpolation=cv2.INTER_AREA) for image in images]
    blank = np.full_like(tiles[0], 245)
    rows = []
    for first in range(0, len(tiles), columns):
        row = tiles[first:first + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    vismod.save_rgb(path, np.concatenate(rows, axis=0))


def add_contact_overlay(renderer, contacts: dict[str, np.ndarray], rows: np.ndarray) -> None:
    for row in rows:
        if renderer.scene.ngeom + 2 >= renderer.scene.maxgeom:
            raise RuntimeError("contact overlay exhausted render capacity")
        position = contacts["pos"][row].astype(np.float64)
        color = np.asarray([1.0, 0.1, 0.05, 1.0], dtype=np.float32)
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.0045), position,
            np.eye(3).reshape(-1), color,
        )
        renderer.scene.ngeom += 1
        force = contacts["wrench_world_force_torque_on_geom2"][row, :3].astype(np.float64)
        magnitude = float(np.linalg.norm(force))
        if magnitude <= 1e-8:
            continue
        end = position + force / magnitude * min(0.060, 0.001 * magnitude)
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).reshape(-1), color,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, position, end)
        renderer.scene.ngeom += 1


def contact_event(vis, condition: str, output: Path) -> tuple[np.ndarray, dict[str, Any]]:
    contacts = arrays(output / condition / "contacts_raw.npz")
    substeps = arrays(output / condition / "substeps.npz")
    near = (contacts["source_endpoint"] >= 13) & (contacts["source_endpoint"] <= 15)
    hand_object = np.isin(contacts["group"], ["right_hand_tool", "left_hand_target"])
    choices = np.flatnonzero(near & hand_object)
    if not len(choices):
        return vismod.missing_panel(14), {
            "condition": condition, "available": False, "reason": "no hand-object event at sources 13..15"
        }
    force = np.linalg.norm(contacts["wrench_world_force_torque_on_geom2"][choices, :3], axis=1)
    selected = int(choices[int(np.argmax(force))])
    global_substep = int(contacts["global_substep"][selected])
    sub = np.flatnonzero(substeps["global_substep"] == global_substep)
    if len(sub) != 1:
        raise RuntimeError(f"{condition}: contact substep lookup is not unique")
    rows = choices[contacts["global_substep"][choices] == global_substep]
    vis.actual.set_qpos(substeps["qpos"][int(sub[0])])
    vis.actual_renderer.update_scene(
        vis.actual.data,
        camera=vismod.camera_object(vismod.CAMERAS["oblique"]),
        scene_option=vismod.scene_option(collision=True),
    )
    add_contact_overlay(vis.actual_renderer, contacts, rows)
    vis.render_updates += 1
    image = annotate(
        vis.actual_renderer.render().copy(),
        condition,
        f"own source14-near event | source {int(contacts['source_endpoint'][selected])}",
        f"substep {global_substep} | peak {float(force.max()):.2f} N | red saved contacts",
    )
    return image, {
        "condition": condition,
        "available": True,
        "source_endpoint": int(contacts["source_endpoint"][selected]),
        "outcome_endpoint": int(contacts["outcome_endpoint"][selected]),
        "global_substep": global_substep,
        "peak_hand_object_force_N": float(force.max()),
        "contact_rows_at_substep": int(len(rows)),
    }


def comparison_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    result: dict[str, dict[int, dict[str, float]]] = {name: {} for name in CONDITIONS}
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw["availability"] != "measured":
                continue
            condition = raw["condition"]
            endpoint = int(raw["endpoint"])
            result.setdefault(condition, {})[endpoint] = {
                key: float(value) for key, value in raw.items()
                if key not in {"condition", "endpoint", "availability"} and value
            }
    return result


def plot_metric_curves(output: Path, trajectories: dict[str, dict[str, np.ndarray]]) -> list[str]:
    plots = output / "visuals/curves"
    plots.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(output / "comparison.csv")
    metric_groups = {
        "object_errors": (
            ("tool_position_error_m", "tool position (m)"),
            ("target_position_error_m", "target position (m)"),
            ("tool_rotation_error_rad", "tool rotation (rad)"),
            ("target_rotation_error_rad", "target rotation (rad)"),
        ),
        "relative_errors": (
            ("world_pair_translation_error_m", "world pair translation (m)"),
            ("pair_translation_error_m", "target-frame relative translation (m)"),
            ("pair_rotation_error_rad", "target-frame relative rotation (rad)"),
        ),
        "cost_components": (
            ("D_obj", "D_obj"), ("D_pair", "D_pair"),
            ("D_hand", "D_hand"), ("D_control", "D_control"), ("L", "L"),
        ),
    }
    outputs: list[str] = []
    for group, specs in metric_groups.items():
        columns = 2
        rows_n = (len(specs) + columns - 1) // columns
        fig, axes = plt.subplots(rows_n, columns, figsize=(12, 3.7 * rows_n), squeeze=False)
        for axis, (metric, label) in zip(axes.flat, specs):
            for condition in CONDITIONS:
                if metric in trajectories[condition]:
                    x = np.arange(1, 41)
                    y = trajectories[condition][metric]
                else:
                    x = np.arange(0, 41)
                    y = [rows[condition][endpoint][metric] for endpoint in x]
                axis.plot(x, y, label=condition, color=COLORS[condition], linewidth=1.6)
            axis.set_xlim(0, 40)
            axis.set_title(label)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(specs):]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        relative = f"curves/{group}.png"
        fig.savefig(output / "visuals" / relative, dpi=160)
        plt.close(fig)
        outputs.append(relative)
    return outputs


def selected_candidate_index(candidate: dict[str, np.ndarray], selection: dict[str, Any], source: int) -> int:
    records = [item for item in selection["round_selections"] if int(item["source"]) == source]
    record = max(records, key=lambda item: int(item["round"]))
    matches = np.flatnonzero(
        (candidate["round"] == int(record["round"]))
        & (candidate["slot"] == int(record["selected_slot"]))
    )
    if len(matches) != 1:
        raise RuntimeError("selected candidate lookup is not unique")
    return int(matches[0])


def plot_planning_sources(output: Path, trajectories: dict[str, dict[str, np.ndarray]]) -> list[str]:
    directory = output / "visuals/planning"
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for arm in ("A", "L"):
        selection = json.loads((output / f"{arm}_PLAN/selection.json").read_text())
        actual = trajectories[f"{arm}_PLAN"]
        for source in (0, 5, 10, 15):
            candidate = arrays(output / f"{arm}_PLAN/search/source_{source:02d}/candidates.npz")
            nominal_matches = np.flatnonzero((candidate["round"] == 0) & (candidate["slot"] == 0))
            if len(nominal_matches) != 1:
                raise RuntimeError("round-0 nominal lookup is not unique")
            nominal = int(nominal_matches[0])
            best = selected_candidate_index(candidate, selection, source)
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for label, row, style in (("round0 nominal", nominal, "--"), ("selected best prediction", best, "-")):
                mask = candidate["executed_mask"][row]
                x = candidate["outcome_endpoint"][row][mask]
                axes[0].plot(x, candidate["tracking_score"][row][mask], style, label=label)
                axes[1].plot(x, np.linalg.norm(candidate["action"][row][mask], axis=1), style, label=label)
            end = 40 if source == 15 else source + 5
            actual_slice = slice(source, end)
            x_actual = np.arange(source + 1, end + 1)
            axes[0].plot(x_actual, actual["tracking_score"][actual_slice], "o", markersize=3, label="actual executed prefix")
            axes[1].plot(x_actual, np.linalg.norm(actual["action"][actual_slice], axis=1), "o", markersize=3, label="actual executed prefix")
            axes[0].set_ylabel("tool tracking score")
            axes[1].set_ylabel("36-D action L2")
            axes[1].set_xlabel("outcome endpoint")
            for axis in axes:
                axis.grid(alpha=0.25)
                axis.legend()
                axis.set_xlim(source, 40)
            fig.suptitle(f"{arm}_PLAN source {source}: frozen candidate arrays")
            fig.tight_layout()
            relative = f"planning/{arm}_source_{source:02d}.png"
            fig.savefig(output / "visuals" / relative, dpi=160)
            plt.close(fig)
            outputs.append(relative)
    return outputs


def render_overlap(output: Path) -> dict[str, Any]:
    """Render the 0->60 overlap continuation with the existing fixed viewer."""
    status = json.loads((output / "status.json").read_text())
    if status.get("status") not in {
        "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING", "COMPLETE_NO_PROMOTION",
    }:
        raise RuntimeError("overlap rendering requires completed analysis")
    parity = json.loads((output / "cold_replay_parity.json").read_text())
    if not parity.get("all_bitwise_equal"):
        raise RuntimeError("overlap cold replay parity is not bitwise")
    names = ("FROZEN_SUFFIX_BASELINE", "OVERLAP_PLAN")
    trajectories = {name: arrays(output / name / "trajectory.npz") for name in names}
    reference = arrays(ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz")
    reference_model = vismod.StaticModel.load(
        "reference", ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml"
    )
    actual_model = vismod.StaticModel.load(
        "actual", ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml"
    )
    vis = vismod.Visualizer(reference_model, actual_model)
    visuals = output / "visuals"
    for child in ("keyframes", "videos", "review_sheets", "curves"):
        (visuals / child).mkdir(parents=True, exist_ok=True)
    keys = (0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
    sheets = {"oblique": [], "top": []}
    endpoints = {
        name: {int(endpoint): index for index, endpoint in enumerate(data["endpoint"])}
        for name, data in trajectories.items()
    }
    writers = {}
    frame_width = vismod.PANEL_WIDTH * 3
    for view in ("oblique", "top"):
        writer = cv2.VideoWriter(
            str(visuals / f"videos/reference_baseline_overlap_{view}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
            (frame_width, vismod.PANEL_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create overlap {view} video")
        writers[view] = writer
    try:
        for endpoint in range(61):
            for view in ("oblique", "top"):
                panels = [annotate(
                    vis.render_reference(reference["qpos"][endpoint], view),
                    "REFERENCE", f"endpoint {endpoint} | {view}",
                )]
                for name in names:
                    if endpoint in endpoints[name]:
                        index = endpoints[name][endpoint]
                        panels.append(annotate(
                            vis.render_actual(trajectories[name]["qpos"][index], view),
                            name, f"endpoint {endpoint}",
                        ))
                    else:
                        panels.append(annotate(
                            vismod.missing_panel(endpoint), name,
                            f"endpoint {endpoint} not reached",
                        ))
                frame = np.concatenate(panels, axis=1)
                repetitions = 4 if endpoint in {19, 20, 21, 39, 40, 41} else 1
                for _ in range(repetitions):
                    writers[view].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if endpoint in keys:
                    vismod.save_rgb(visuals / f"keyframes/endpoint_{endpoint:03d}_{view}.png", frame)
                    sheets[view].append(frame)
    finally:
        for writer in writers.values():
            writer.release()
        vis.close()
    for view in ("oblique", "top"):
        save_sheet(sheets[view], visuals / f"review_sheets/keyframes_{view}.png", columns=2)

    table = comparison_rows(output / "comparison.csv")
    colors = {"FROZEN_SUFFIX_BASELINE": "#6b7280", "OVERLAP_PLAN": "#2563eb"}
    groups = {
        "absolute_object_errors": (
            "tool_position_error_m", "target_position_error_m",
            "tool_rotation_error_rad", "target_rotation_error_rad",
        ),
        "relative_and_tracking": (
            "pair_translation_error_m", "world_pair_translation_error_m",
            "pair_rotation_error_rad", "tracking_score",
        ),
        "object_speeds": (
            "tool_linear_speed_m_s", "target_linear_speed_m_s",
            "tool_angular_speed_rad_s", "target_angular_speed_rad_s",
        ),
    }
    curve_outputs = []
    for group, metric_names in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
        for axis, metric in zip(axes.flat, metric_names):
            for name in names:
                x = sorted(table[name])
                y = [table[name][endpoint].get(metric, np.nan) for endpoint in x]
                axis.plot(x, y, label=name, color=colors[name], linewidth=1.7)
            axis.set_title(metric); axis.grid(alpha=0.25); axis.set_xlim(0, 60)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        relative = f"curves/{group}.png"
        fig.savefig(visuals / relative, dpi=160); plt.close(fig)
        curve_outputs.append(relative)
    result = {
        "schema": "taco_pour_aplan_overlap_continuation_visuals_v1",
        "offline_only": True, "physics_steps": 0,
        "rendered_endpoints": list(keys), "video_endpoints": [0, 60],
        "fixed_views": ["oblique", "top"],
        "columns": ["reference", *names],
        "transition_slowdowns": ["18..22", "38..42"],
        "curve_outputs": curve_outputs, "render_updates": vis.render_updates,
        "human_review_status": "complete" if (output / "visual_review.json").is_file() else "pending",
    }
    (visuals / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def render(output: Path) -> dict[str, Any]:
    status = json.loads((output / "status.json").read_text())
    if status.get("schema") == "taco_pour_aplan_overlap_continuation_20_60_v1":
        return render_overlap(output)
    if status.get("status") not in {
        "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING",
        "COMPLETE_NO_PROMOTION",
    }:
        raise RuntimeError("rendering requires completed analysis")
    for arm in ("A", "L"):
        parity = json.loads((output / f"{arm}_PLAN/cold_replay_parity.json").read_text())
        if not parity.get("all_bitwise_equal"):
            raise RuntimeError(f"{arm} cold replay parity is not bitwise")

    trajectories = {condition: arrays(output / condition / "trajectory.npz") for condition in CONDITIONS}
    reference = arrays(ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz")
    human = arrays(ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/human_reference.npz")
    rgb_path = ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4"
    rgb, rgb_info = vismod.read_rgb_frames(rgb_path, 41)
    reference_model = vismod.StaticModel.load(
        "reference", ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml"
    )
    actual_model = vismod.StaticModel.load(
        "actual", ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml"
    )
    vis = vismod.Visualizer(reference_model, actual_model)
    visuals = output / "visuals"
    for child in ("initial", "keyframes", "events", "videos", "review_sheets"):
        (visuals / child).mkdir(parents=True, exist_ok=True)
    key_oblique: list[np.ndarray] = []
    key_top: list[np.ndarray] = []
    event_records: list[dict[str, Any]] = []
    try:
        initial = np.concatenate([
            annotate(vismod.letterbox(rgb[0], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT), "REAL RGB", "independent camera; not pixel-registered"),
            annotate(vis.render_human(human["T_sim_object_reference"][0], human["joint_positions_sim"][0], "oblique"), "HUMAN + OBJECT SOURCE", "shared sim coordinates"),
            annotate(vis.render_reference(reference["qpos"][0], "oblique"), "ROBOT REFERENCE", "endpoint 0"),
            annotate(vis.render_actual(trajectories["A_REPLAY"]["qpos"][0], "oblique"), "A START", "accepted initial state"),
            annotate(vis.render_actual(trajectories["L_REPLAY"]["qpos"][0], "oblique"), "L START", "left-aligned candidate"),
        ], axis=1)
        vismod.save_rgb(visuals / "initial/source_reference_A_L.png", initial)

        writers = {}
        frame_width = vismod.PANEL_WIDTH * 5
        for view in ("oblique", "top"):
            writer = cv2.VideoWriter(
                str(visuals / f"videos/reference_four_cells_{view}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
                (frame_width, vismod.PANEL_HEIGHT),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot create {view} video")
            writers[view] = writer
        try:
            for endpoint in range(41):
                for view in ("oblique", "top"):
                    panels = [annotate(vis.render_reference(reference["qpos"][endpoint], view), "REFERENCE", f"endpoint {endpoint} | {view}")]
                    panels.extend(
                        annotate(vis.render_actual(trajectories[name]["qpos"][endpoint], view), name, f"endpoint {endpoint}")
                        for name in CONDITIONS
                    )
                    frame = np.concatenate(panels, axis=1)
                    writers[view].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if endpoint in KEYS:
                        vismod.save_rgb(visuals / f"keyframes/endpoint_{endpoint:03d}_{view}.png", frame)
                        (key_oblique if view == "oblique" else key_top).append(frame)
        finally:
            for writer in writers.values():
                writer.release()

        for condition in CONDITIONS:
            image, record = contact_event(vis, condition, output)
            relative = f"events/{condition.lower()}_source14_near.png"
            vismod.save_rgb(visuals / relative, image)
            record["image"] = relative
            event_records.append(record)
    finally:
        vis.close()

    save_sheet(key_oblique, visuals / "review_sheets/keyframes_oblique.png")
    save_sheet(key_top, visuals / "review_sheets/keyframes_top.png")
    curve_outputs = plot_metric_curves(output, trajectories)
    planning_outputs = plot_planning_sources(output, trajectories)
    result = {
        "schema": "taco_pour_control_aware_startup_visuals_v1",
        "offline_only": True,
        "physics_steps": 0,
        "rendered_endpoints": list(KEYS),
        "video_endpoints": [0, 40],
        "fixed_views": ["oblique", "top"],
        "four_cells": list(CONDITIONS),
        "rgb": rgb_info,
        "contact_events": event_records,
        "curve_outputs": curve_outputs,
        "planning_outputs": planning_outputs,
        "render_updates": vis.render_updates,
        "human_review_status": "complete" if (output / "visual_review.json").is_file() else "pending",
    }
    (visuals / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


VISUAL_SCHEMA = "taco_pour_visual_evidence_standard_v1"
VISUAL_CONDITIONS = ("FROZEN_SUFFIX_BASELINE", "OVERLAP_PLAN")
MEANINGFUL_CONTACT_GROUPS = {
    "right_hand_tool", "left_hand_target", "tool_target", "floor_tool", "floor_target",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty rows: {path}")
        fields = tuple(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _resolve_asset(asset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else asset_root / path


def _mesh_dependencies(scene: Path) -> list[Path]:
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", str(scene.parent))) if compiler is not None else scene.parent
    if not meshdir.is_absolute():
        meshdir = scene.parent / meshdir
    result = []
    for element in root.iter():
        filename = element.get("file")
        if filename:
            candidate = Path(filename)
            if not candidate.is_absolute():
                candidate = meshdir / candidate
            if candidate.is_file():
                result.append(candidate.resolve())
    return sorted(set(result))


def _ffprobe_frames(video: Path) -> tuple[list[float | None], dict[str, Any]]:
    version = subprocess.run(
        ["ffprobe", "-version"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()[0]
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: list[float | None] = []
    for line in process.stdout.splitlines():
        text = line.strip().rstrip(",")
        values.append(None if not text or text == "N/A" else float(text.split(",")[0]))
    return values, {"version": version, "probed_frames": len(values), "command": "ffprobe -show_frames best_effort_timestamp_time"}


def _decode_rgb(video: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open RGB video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, {
        "sequential_decode": True,
        "declared_frames": declared,
        "decoded_frames": len(frames),
        "fps": fps,
        "width": int(frames[0].shape[1]),
        "height": int(frames[0].shape[0]),
        "opencv_version": cv2.__version__,
    }


def _text_strip(width: int, height: int, lines: list[str], *, background=(20, 22, 28)) -> np.ndarray:
    strip = np.full((height, width, 3), background, dtype=np.uint8)
    if not lines:
        return strip
    spacing = max(18, min(28, (height - 8) // len(lines)))
    scale = 0.50 if width <= 640 else 0.72
    for index, line in enumerate(lines):
        cv2.putText(
            strip, str(line), (10, 20 + index * spacing), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (236, 239, 244), 1 if width <= 640 else 2, cv2.LINE_AA,
        )
    return strip


def _panel(image: np.ndarray, title: str, footer: list[str]) -> np.ndarray:
    width = int(image.shape[1])
    return np.concatenate(
        [_text_strip(width, 42, [title], background=(10, 40, 65)), image, _text_strip(width, 106, footer)],
        axis=0,
    )


def _missing_image(width: int, height: int, label: str) -> np.ndarray:
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(image, label, (max(16, width // 10), height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 if width <= 640 else 1.1, (255, 190, 70), 2, cv2.LINE_AA)
    return image


def _metric_lines(metrics: Mapping[str, float], status: str) -> list[str]:
    def number(key: str, scale: float = 1.0, suffix: str = "") -> str:
        value = metrics.get(key, float("nan"))
        return "NA" if not np.isfinite(value) else f"{value * scale:.3f}{suffix}"
    return [
        status,
        f"tool(bowl): pos {number('tool_position_error_m', 1000, ' mm')} | rot {number('tool_rotation_error_rad', 1, ' rad')}",
        f"target(tray): pos {number('target_position_error_m', 1000, ' mm')} | rot {number('target_rotation_error_rad', 1, ' rad')}",
        f"pair world-pos {number('world_pair_translation_error_m', 1000, ' mm')} | target-frame pos {number('pair_translation_error_m', 1000, ' mm')}",
        f"pair target-frame rot {number('pair_rotation_error_rad', 1, ' rad')} | tracking {number('tracking_score')}",
    ]


def _compose_frame(
    endpoint: int,
    view: str,
    rgb: np.ndarray,
    reference_image: np.ndarray,
    actual_images: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    statuses: Mapping[str, str],
    frame_row: Mapping[str, Any],
    *,
    experiment: str,
) -> np.ndarray:
    width, height = reference_image.shape[1], reference_image.shape[0]
    rgb_image = vismod.letterbox(rgb, width, height)
    panels = [
        _panel(
            rgb_image,
            "REAL RGB",
            [
                "time-aligned; independent camera; not pixel-registered",
                f"source frame {frame_row['source_frame_id']} | PTS {float(frame_row['rgb_pts_s']):.6f}s",
                f"alignment {frame_row['alignment_status']} | error {float(frame_row['alignment_error_s']):.2e}s",
            ],
        ),
        _panel(
            reference_image,
            "ROBOT REFERENCE - KINEMATIC REFERENCE",
            [f"endpoint {endpoint} | t_ref={float(frame_row['reference_time_s']):.6f}s", "static FK; not a physics rollout"],
        ),
    ]
    for condition in VISUAL_CONDITIONS:
        panels.append(_panel(actual_images[condition], condition, _metric_lines(metrics[condition], statuses[condition])))
    body = np.concatenate(panels, axis=1)
    header = _text_strip(
        body.shape[1], 52,
        [f"{experiment} | endpoint {endpoint} | view {view} | baseline a4e9d01b | alignment {frame_row['alignment_status']}"],
        background=(45, 18, 50),
    )
    return np.concatenate([header, body], axis=0)


def _redact_rgb_column(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    column = result.shape[1] // 4
    result[52:, :column] = 25
    lines = ["REAL RGB WITHHELD", "repository policy: controlled TACO data", "see controlled server package"]
    for index, line in enumerate(lines):
        cv2.putText(result, line, (18, 160 + index * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (255, 210, 90), 2, cv2.LINE_AA)
    return result


def _resize_comparison(image: np.ndarray, source_panel_width: int, target_panel_width: int) -> np.ndarray:
    scale = target_panel_width / source_panel_width
    return cv2.resize(image, (target_panel_width * 4, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def _build_events(
    source_root: Path,
    table: dict[str, dict[int, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(condition: str, endpoint: int, category: str, metric: str = "", value: Any = "", detail: str = "") -> None:
        rows.append(
            {
                "event_id": f"event_{len(rows):05d}", "condition": condition, "endpoint": endpoint,
                "source_endpoint": max(0, endpoint - 1), "category": category, "metric": metric,
                "value": value, "detail": detail, "navigation_only": True,
                "provenance": "saved comparison/contact evidence",
            }
        )

    for endpoint, detail in ((0, "initialization"), (20, "overlap start"), (25, "rolling source"),
                             (30, "rolling source"), (35, "rolling source"), (40, "parent boundary"),
                             (60, "last saved endpoint")):
        add("ALL", endpoint, "declared_boundary", detail=detail)
    error_metrics = (
        "tool_position_error_m", "tool_rotation_error_rad", "target_position_error_m",
        "target_rotation_error_rad", "world_pair_translation_error_m",
        "pair_translation_error_m", "pair_rotation_error_rad", "tracking_score",
    )
    speed_metrics = (
        "tool_linear_speed_m_s", "tool_angular_speed_rad_s",
        "target_linear_speed_m_s", "target_angular_speed_rad_s",
    )
    for condition in VISUAL_CONDITIONS:
        values = table[condition]
        for metric in error_metrics:
            metric_values = {endpoint: row[metric] for endpoint, row in values.items() if np.isfinite(row.get(metric, np.nan))}
            for endpoint, delta in top_positive_increments(metric_values, 3):
                add(condition, endpoint, "positive_error_increment", metric, delta, "top-3 positive adjacent increment; navigation only")
        for metric in speed_metrics:
            metric_values = {endpoint: row[metric] for endpoint, row in values.items() if np.isfinite(row.get(metric, np.nan))}
            if metric_values:
                maximum = min(((-value, endpoint) for endpoint, value in metric_values.items()))
                add(condition, maximum[1], "actual_speed_global_max", metric, -maximum[0], "actual speed peak; not an overspeed threshold")
                increments = top_positive_increments(metric_values, 1)
                if increments:
                    add(condition, increments[0][0], "actual_speed_largest_positive_increment", metric, increments[0][1], "navigation only")
        result = json.loads((source_root / condition / "result.json").read_text())
        if result.get("first_failure_endpoint") is not None:
            add(condition, int(result["first_failure_endpoint"]), "saved_tracking_termination", "tracking_score", "", "saved legacy criterion")
        contacts = arrays(source_root / condition / "contacts_raw.npz")
        states = semantic_contact_sets(
            contacts["source_endpoint"], contacts["group"], contacts["role1"], contacts["role2"],
            meaningful_groups=MEANINGFUL_CONTACT_GROUPS,
        )
        for event in contact_transitions(states, start=0, stop=60):
            add(condition, event["endpoint"], f"semantic_contact_{event['transition']}",
                "semantic_contact", "", event["semantic_identity"])
    return rows


def _make_sheet(images: list[Path], output: Path) -> None:
    tiles = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot decode preview for sheet: {path}")
        tiles.append(cv2.resize(image, (960, 260), interpolation=cv2.INTER_AREA))
    blank = np.full_like(tiles[0], 245)
    while len(tiles) < 10:
        tiles.append(blank)
    rows = [np.concatenate(tiles[index:index + 2], axis=1) for index in range(0, 10, 2)]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"cannot write sheet {output}")


def _write_indexes(root: Path, frame_rows: list[dict[str, Any]], events: list[dict[str, Any]], *, rgb_public: bool) -> None:
    event_endpoints = sorted({int(row["endpoint"]) for row in events})
    lines = [
        "# TACO Pour visual evidence 0–60", "",
        f"RGB distribution: {'controlled placeholder in this package' if not rgb_public else 'time-aligned real RGB'}.",
        "Review status: see `visual_review.json`. Automatic events are navigation only.", "",
        "## Contact sheets", "",
    ]
    for view in ("oblique", "top"):
        links = " ".join(f"[page {page}](sheets/{view}/page_{page:03d}.jpg)" for page in range(7))
        lines += [f"- {view}: {links}"]
    lines += ["", "## Events", "", ", ".join(f"[{e}](previews/oblique/endpoint_{e:06d}.jpg)" for e in event_endpoints), "",
              "## Every endpoint", "", "| endpoint | reference time | RGB frame/PTS | oblique | top |", "|---:|---:|---:|---|---|"]
    for row in frame_rows:
        endpoint = int(row["endpoint"])
        lines.append(
            f"| {endpoint} | {float(row['reference_time_s']):.6f}s | {row['source_frame_id']} / {float(row['rgb_pts_s']):.6f}s | "
            f"[open](previews/oblique/endpoint_{endpoint:06d}.jpg) | [open](previews/top/endpoint_{endpoint:06d}.jpg) |"
        )
    lines += ["", "Videos and full-resolution controlled evidence are listed in `completeness.json`."]
    (root / "VISUAL_INDEX.md").write_text("\n".join(lines) + "\n")

    data = [
        {"endpoint": int(row["endpoint"]), "time": float(row["reference_time_s"]), "event": int(row["endpoint"]) in event_endpoints}
        for row in frame_rows
    ]
    payload = json.dumps(data, separators=(",", ":"))
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>TACO Pour evidence</title>
<style>body{{font:16px sans-serif;background:#111;color:#eee;margin:18px}}button,select,input{{font-size:16px}}img{{max-width:100%;display:block;margin-top:12px}}.row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}a{{color:#8cf}}</style>
<h1>TACO Pour visual evidence 0–60</h1><p>{'RGB is controlled and redacted in this public package.' if not rgb_public else 'RGB is time-aligned, independent, and not pixel-registered.'}</p>
<div class=row><button onclick=\"step(-1)\">Previous</button><input id=e type=range min=0 max=60 value=0 oninput=\"show(+this.value)\"><button onclick=\"step(1)\">Next</button><select id=v onchange=\"show(+e.value)\"><option>oblique</option><option>top</option></select><select id=j onchange=\"show(+this.value)\"><option value=0>events</option>{''.join(f'<option value={item}>{item}</option>' for item in event_endpoints)}</select><span id=l></span></div><a id=o><img id=i></a>
<script>const data={payload};function show(n){{n=Math.max(0,Math.min(60,n));e.value=n;let p=`previews/${{v.value}}/endpoint_${{String(n).padStart(6,'0')}}.jpg`;i.src=p;o.href=p;l.textContent=`endpoint ${{n}} · t=${{data[n].time.toFixed(6)}}s${{data[n].event?' · event':''}}`;}}function step(d){{show(+e.value+d)}}show(0);</script>"""
    (root / "index.html").write_text(page)


def _write_videos(root: Path, frame_rows: list[dict[str, Any]], fps: int) -> tuple[list[dict[str, Any]], int]:
    mapping_rows: list[dict[str, Any]] = []
    encoded = 0
    for view in ("oblique", "top"):
        paths = [root / "previews" / view / f"endpoint_{int(row['endpoint']):06d}.jpg" for row in frame_rows]
        first = cv2.imread(str(paths[0]))
        if first is None:
            raise RuntimeError(f"cannot decode video source {paths[0]}")
        size = (first.shape[1], first.shape[0])
        for label, factor in (("realtime", 1), ("slow4x", 4)):
            target = root / "videos" / f"comparison_{view}_{label}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
            if not writer.isOpened():
                raise RuntimeError(f"cannot create video: {target}")
            try:
                local_map = video_frame_map(frame_rows, slowdown_factor=factor)
                for record in local_map:
                    source = cv2.imread(str(paths[int(record["endpoint"])]))
                    if source is None:
                        raise RuntimeError(f"cannot decode video source for endpoint {record['endpoint']}")
                    writer.write(source)
                    mapping_rows.append({"view": view, "speed": label, **record})
                    encoded += 1
            finally:
                writer.release()
    return mapping_rows, encoded


def _image_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records, sha256_manifest(records)


def _write_hashes(root: Path) -> None:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifacts.sha256" or "handoff" in path.parts:
            continue
        records.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    (root / "artifacts.sha256").write_text("\n".join(records) + "\n")


def _zip_tree(root: Path, archive: Path, include_dirs: tuple[str, ...]) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        "README.md", "VISUAL_INDEX.md", "index.html", "frame_map.csv", "events.csv",
        "completeness.json", "render_accounting.json", "visual_review.json", "findings.md",
        "image_manifest.json", "input_manifest.json", "video_frame_map.csv",
        "publication_validation.json", "artifacts.sha256",
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in selected:
            path = root / relative
            if path.is_file():
                bundle.write(path, relative)
        for directory in include_dirs:
            for path in sorted((root / directory).rglob("*")):
                if path.is_file():
                    if directory == "keyframes":
                        try:
                            endpoint = int(path.stem.split("_")[-1])
                        except ValueError:
                            continue
                        if endpoint % 5:
                            continue
                    bundle.write(path, str(path.relative_to(root)))


def render_visual_standard(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.time()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema") != VISUAL_SCHEMA:
        raise ValueError("unexpected visual-evidence schema")
    limits = config["execution_limits"]
    zero_fields = ("physics_steps", "control_intervals", "action_candidates", "mink_solve_calls",
                   "actor_critic_forward", "vision_estimation_forward", "optimizer_steps")
    if any(int(limits[field]) != 0 for field in zero_fields) or limits.get("chunk_commit"):
        raise ValueError("visual evidence requires a zero-execution contract")
    asset_root = Path(config["asset_root"]).resolve(strict=True)
    source_root = _resolve_asset(asset_root, config["source_run"]).resolve(strict=True)
    output = output.resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("visual evidence output must not overwrite the source run")
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"asset_root": str(asset_root), "source_run": str(source_root), "output": str(output)}, indent=2))

    input_paths: dict[str, Path] = {
        "config": config_path.resolve(strict=True),
        "reference": _resolve_asset(asset_root, config["inputs"]["reference"]).resolve(strict=True),
        "reference_scene": _resolve_asset(asset_root, config["inputs"]["reference_scene"]).resolve(strict=True),
        "actual_scene": _resolve_asset(asset_root, config["inputs"]["actual_scene"]).resolve(strict=True),
        "rgb_video": _resolve_asset(asset_root, config["inputs"]["rgb_video"]).resolve(strict=True),
        "metrics": _resolve_asset(asset_root, config["inputs"]["metrics"]).resolve(strict=True),
        "renderer": Path(__file__).resolve(strict=True),
        "visual_helper": (REPO / "scripts/inspect_hand_object_trajectory.py").resolve(strict=True),
    }
    for condition in config["inputs"]["conditions"]:
        condition_id = condition["id"]
        for key in ("trajectory", "substeps", "contacts"):
            input_paths[f"{condition_id}.{key}"] = _resolve_asset(asset_root, condition[key]).resolve(strict=True)
        input_paths[f"{condition_id}.result"] = (source_root / condition_id / "result.json").resolve(strict=True)
    for index, path in enumerate(config["inputs"]["provenance"]):
        input_paths[f"provenance.{index}"] = _resolve_asset(asset_root, path).resolve(strict=True)
    for label in ("reference_scene", "actual_scene"):
        for index, path in enumerate(_mesh_dependencies(input_paths[label])):
            input_paths[f"{label}.mesh.{index:03d}"] = path
    before = {label: _artifact(path) for label, path in input_paths.items()}

    reference = arrays(input_paths["reference"])
    if not np.array_equal(reference["frame_indices"], np.arange(len(reference["frame_indices"]))):
        raise ValueError("this run expected the preserved 0-based explicit source mapping")
    if len(reference["qpos"]) < 61:
        raise ValueError("reference does not cover endpoint 60")
    pts, ffprobe_info = _ffprobe_frames(input_paths["rgb_video"])
    rgb_frames, decode_info = _decode_rgb(input_paths["rgb_video"])
    frame_rows = resolve_frame_map(
        reference["frame_indices"], reference["timestamps_s"], pts,
        endpoint_start=0, endpoint_stop=60,
        provenance="robot_reference.npz explicit decoded-video frame ids; action.contracts.validate_reference_timeline",
    )
    if any(row["alignment_status"] != "aligned" for row in frame_rows):
        raise RuntimeError("RGB/reference frame alignment is unresolved")
    if len(rgb_frames) != len(pts) or len(rgb_frames) != 198:
        raise RuntimeError("sequential RGB decode and PTS frame count disagree")
    _write_csv(output / "frame_map.csv", frame_rows, FRAME_MAP_FIELDS)
    rgb_cache = output / "rgb_cache"
    for row in frame_rows:
        endpoint = int(row["endpoint"])
        vismod.save_rgb(rgb_cache / f"source_frame_{int(row['source_frame_id']):06d}.jpg", rgb_frames[int(row["decoded_video_frame_index"])])

    trajectories = {name: arrays(input_paths[f"{name}.trajectory"]) for name in VISUAL_CONDITIONS}
    endpoint_lookup = {name: endpoint_rows(data["endpoint"]) for name, data in trajectories.items()}
    results = {name: json.loads(input_paths[f"{name}.result"].read_text()) for name in VISUAL_CONDITIONS}
    table = comparison_rows(input_paths["metrics"])
    if any(set(range(61)) - set(table[name]) for name in VISUAL_CONDITIONS):
        raise ValueError("comparison table lacks a condition endpoint")
    events = _build_events(source_root, table)
    _write_csv(output / "events.csv", events)
    event_endpoints = {int(row["endpoint"]) for row in events}
    highres_endpoints = set(range(0, 61, int(config["render"]["high_resolution_key_interval"]))) | {0, 60}
    for endpoint in event_endpoints:
        highres_endpoints.update(range(max(0, endpoint - 2), min(60, endpoint + 2) + 1))

    for function_name in ("mj_step", "mj_step1", "mj_step2"):
        if hasattr(mujoco, function_name):
            setattr(mujoco, function_name, lambda *args, _name=function_name, **kwargs: (_ for _ in ()).throw(RuntimeError(f"forbidden {_name} call")))

    reference_model = vismod.StaticModel.load("reference", input_paths["reference_scene"])
    actual_model = vismod.StaticModel.load("actual", input_paths["actual_scene"])
    server_width, server_height = map(int, config["render"]["server_panel_image_size"])
    preview_width, preview_height = map(int, config["render"]["preview_panel_image_size"])
    high_width, high_height = map(int, config["render"]["high_resolution_panel_image_size"])
    # Renderer size is a visual-only property of these fresh StaticModel copies.
    # Raise the offscreen framebuffer before constructing any renderer so the
    # 960x720 keyframes are native renders rather than enlarged 640x480 pixels.
    for model in (reference_model.model, actual_model.model):
        model.vis.global_.offwidth = max(server_width, high_width)
        model.vis.global_.offheight = max(server_height, high_height)
    static_calls = 0
    main_vis = vismod.Visualizer(reference_model, actual_model, width=server_width, height=server_height)
    try:
        for endpoint, frame_row in enumerate(frame_rows):
            rgb = rgb_frames[int(frame_row["decoded_video_frame_index"])]
            metrics = {name: table[name][endpoint] for name in VISUAL_CONDITIONS}
            statuses = {
                name: endpoint_status(endpoint, endpoint_lookup[name], first_failure_endpoint=results[name].get("first_failure_endpoint"))
                for name in VISUAL_CONDITIONS
            }
            for view in config["render"]["views"]:
                reference_image = main_vis.render_reference(reference["qpos"][endpoint], view)
                actual_images = {}
                for name in VISUAL_CONDITIONS:
                    row = endpoint_lookup[name].get(endpoint)
                    actual_images[name] = (
                        main_vis.render_actual(trajectories[name]["qpos"][row], view)
                        if row is not None else _missing_image(server_width, server_height, statuses[name])
                    )
                frame = _compose_frame(
                    endpoint, view, rgb, reference_image, actual_images, metrics, statuses, frame_row,
                    experiment=VISUAL_SCHEMA,
                )
                vismod.save_rgb(output / "frames" / view / f"endpoint_{endpoint:06d}.png", frame)
                preview = _resize_comparison(frame, server_width, preview_width)
                target = output / "previews" / view / f"endpoint_{endpoint:06d}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise RuntimeError(f"cannot write preview {target}")
                public_target = output / "public" / "previews" / view / target.name
                public_target.parent.mkdir(parents=True, exist_ok=True)
                public = _redact_rgb_column(preview)
                if not cv2.imwrite(str(public_target), cv2.cvtColor(public, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise RuntimeError(f"cannot write public preview {public_target}")
        static_calls += main_vis.render_updates
    finally:
        main_vis.close()

    high_vis = vismod.Visualizer(reference_model, actual_model, width=high_width, height=high_height)
    public_key_endpoints = set(range(0, 61, 5)) | {20, 40, 60}
    try:
        for endpoint in sorted(highres_endpoints):
            frame_row = frame_rows[endpoint]
            rgb = rgb_frames[int(frame_row["decoded_video_frame_index"])]
            metrics = {name: table[name][endpoint] for name in VISUAL_CONDITIONS}
            statuses = {
                name: endpoint_status(endpoint, endpoint_lookup[name], first_failure_endpoint=results[name].get("first_failure_endpoint"))
                for name in VISUAL_CONDITIONS
            }
            for view in config["render"]["views"]:
                reference_image = high_vis.render_reference(reference["qpos"][endpoint], view)
                actual_images = {
                    name: high_vis.render_actual(trajectories[name]["qpos"][endpoint_lookup[name][endpoint]], view)
                    for name in VISUAL_CONDITIONS
                }
                frame = _compose_frame(endpoint, view, rgb, reference_image, actual_images, metrics, statuses, frame_row, experiment=VISUAL_SCHEMA)
                path = output / "keyframes" / view / f"endpoint_{endpoint:06d}.png"
                vismod.save_rgb(path, frame)
                if endpoint in public_key_endpoints:
                    vismod.save_rgb(output / "public" / "keyframes" / view / path.name, _redact_rgb_column(frame))
        static_calls += high_vis.render_updates
    finally:
        high_vis.close()

    sub_vis = vismod.Visualizer(reference_model, actual_model, width=server_width, height=server_height)
    substep_count = 0
    try:
        substeps = {name: arrays(input_paths[f"{name}.substeps"]) for name in VISUAL_CONDITIONS}
        contacts = {name: arrays(input_paths[f"{name}.contacts"]) for name in VISUAL_CONDITIONS}
        lookups = {
            name: {int(global_step): row for row, global_step in enumerate(data["global_substep"])}
            for name, data in substeps.items()
        }
        selected_sources = {
            source for start, stop in config["substep_review"]["source_ranges_half_open"] for source in range(int(start), int(stop))
        }
        for source in sorted(selected_sources):
            for substep in range(1, int(config["substep_review"]["expected_physics_substeps_per_control"]) + 1):
                global_substep = source * 10 + substep
                actual_time = global_substep / 300.0
                rgb_endpoint = int(np.clip(round(actual_time * 30.0), 0, 60))
                frame_row = frame_rows[rgb_endpoint]
                rgb = rgb_frames[int(frame_row["decoded_video_frame_index"])]
                outcome = source + 1
                for view in config["render"]["views"]:
                    reference_image = sub_vis.render_reference(reference["qpos"][outcome], view)
                    actual_images = {}
                    metrics = {}
                    statuses = {}
                    for name in VISUAL_CONDITIONS:
                        row = lookups[name].get(global_substep)
                        if row is None:
                            actual_images[name] = _missing_image(server_width, server_height, "SUBSTEP NOT RECORDED")
                            metrics[name] = table[name][outcome]
                            statuses[name] = "SUBSTEP NOT RECORDED"
                            continue
                        actual_images[name] = sub_vis.render_actual(substeps[name]["qpos"][row], view)
                        metrics[name] = table[name][outcome]
                        contact_rows = contacts[name]["global_substep"] == global_substep
                        forces = np.linalg.norm(contacts[name]["wrench_world_force_torque_on_geom2"][contact_rows, :3], axis=1)
                        peak = float(forces.max()) if len(forces) else 0.0
                        statuses[name] = f"POST qpos substep {substep}/10; contacts {int(contact_rows.sum())}; peak saved force {peak:.2f} N"
                    frame = _compose_frame(source, view, rgb, reference_image, actual_images, metrics, statuses, frame_row, experiment=f"SUBSTEP source {source} -> {outcome} | global {global_substep} | t={actual_time:.6f}s | RGB nearest dt={abs(frame_row['reference_time_s']-actual_time):.6f}s | RGB NOT SUBSTEP-SYNCHRONIZED | reference endpoint {outcome} HELD")
                    vismod.save_rgb(
                        output / "substeps" / f"source_{source:06d}" / f"substep_{substep:03d}_post_{view}.png", frame,
                    )
                substep_count += 1
        static_calls += sub_vis.render_updates
    finally:
        sub_vis.close()

    for view in config["render"]["views"]:
        for page in range(7):
            endpoints = list(range(page * 10, min(61, (page + 1) * 10)))
            _make_sheet([output / "previews" / view / f"endpoint_{endpoint:06d}.jpg" for endpoint in endpoints],
                        output / "sheets" / view / f"page_{page:03d}.jpg")
            _make_sheet([output / "public" / "previews" / view / f"endpoint_{endpoint:06d}.jpg" for endpoint in endpoints],
                        output / "public" / "sheets" / view / f"page_{page:03d}.jpg")

    video_rows, encoded_frames = _write_videos(output, frame_rows, fps=30)
    _write_csv(output / "video_frame_map.csv", video_rows)
    for curve in (source_root / "visuals" / "curves").glob("*.png"):
        (output / "curves").mkdir(parents=True, exist_ok=True)
        shutil.copy2(curve, output / "curves" / curve.name)

    _write_indexes(output, frame_rows, events, rgb_public=True)
    _write_indexes(output / "public", frame_rows, events, rgb_public=False)
    input_after = {label: _artifact(path) for label, path in input_paths.items()}
    if before != input_after:
        raise RuntimeError("an input hash changed during offline rendering")
    input_manifest = {
        "schema": f"{VISUAL_SCHEMA}_input_manifest", "asset_root": str(asset_root),
        "source_run": str(source_root), "baseline_commit": config["baseline"]["commit"],
        "renderer_git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "inputs_before": before, "inputs_after": input_after,
        "frame_mapping_contract": "explicit decoded-video frame IDs from robot_reference.npz",
        "ffprobe": ffprobe_info, "decode": decode_info,
        "cameras": config["render"]["cameras"], "visual_style": "fixed native model materials; RGB letterboxed; titles outside image",
        "role_mapping": {"tool": "bowl", "target": "tray/plate"},
        "distribution": {"controlled_server": "real RGB included", "git_public": "RGB derivative withheld per repository data policy"},
    }
    _write_json(output / "input_manifest.json", input_manifest)
    shutil.copy2(REPO / "docs/VISUAL_EVIDENCE_STANDARD.md", output / "README.md")
    shutil.copy2(REPO / "docs/VISUAL_EVIDENCE_SOURCE_NOTES.md", output / "SOURCE_NOTES.md")

    image_records, image_hash = _image_manifest(output)
    _write_json(output / "image_manifest.json", {"records": image_records, "sha256": image_hash})
    pending = pending_review(image_hash)
    _write_json(
        output / "visual_review.json",
        {
            "schema": f"{VISUAL_SCHEMA}_review", "status": pending.status,
            "image_manifest_sha256": pending.image_manifest_sha256,
            "reviewer": None, "viewed_coverage": [], "observations": [], "limitations": [],
            "earliest_suspicious_intervals": {}, "next_minimum_evidence": None,
            "old_review_inherited": False,
        },
    )
    (output / "findings.md").write_text(
        "# Findings\n\nStatus: PENDING visual review. Rendering completeness does not imply task validity.\n"
    )
    accounting = {
        "schema": f"{VISUAL_SCHEMA}_render_accounting", "physics_steps": 0, "control_intervals": 0,
        "action_candidates": 0, "mink_solve_calls": 0, "actor_critic_forward": 0,
        "vision_estimation_forward": 0, "optimizer_steps": 0, "chunk_commit": False,
        "static_fk_calls": reference_model.kinematics_calls + actual_model.kinematics_calls,
        "static_render_calls": static_calls, "decoded_rgb_frames": len(rgb_frames),
        "encoded_video_frames": encoded_frames, "substep_states_rendered": substep_count,
        "elapsed_s": time.time() - started,
    }
    _write_json(output / "render_accounting.json", accounting)
    completeness = {
        "schema": f"{VISUAL_SCHEMA}_completeness", "render_status": "COMPLETE", "visual_review_status": "PENDING",
        "frame_map_rows": len(frame_rows), "endpoint_previews": 122, "server_frames": 122,
        "sheets": 14, "videos": 4, "substep_comparisons": substep_count * 2,
        "rgb_alignment": "aligned", "rgb_publication": "controlled server only; Git placeholder",
        "parent_decision_modified": False, "promotion_authorized": False,
    }
    _write_json(output / "completeness.json", completeness)
    for name in ("frame_map.csv", "events.csv", "completeness.json", "render_accounting.json", "visual_review.json",
                 "findings.md", "image_manifest.json", "input_manifest.json", "video_frame_map.csv", "README.md", "SOURCE_NOTES.md"):
        shutil.copy2(output / name, output / "public" / name)
    validation = validate_publication(output, endpoint_count=61, views=("oblique", "top"), expected_sheet_count=14)
    public_validation = validate_publication(output / "public", endpoint_count=61, views=("oblique", "top"), expected_sheet_count=14)
    publication_validation = {"controlled": validation, "public": public_validation}
    _write_json(output / "publication_validation.json", publication_validation)
    _write_json(output / "public" / "publication_validation.json", publication_validation)
    if not validation["valid"] or not public_validation["valid"]:
        raise RuntimeError("publication validation failed")
    _write_hashes(output / "public")
    _write_hashes(output)
    _zip_tree(output, output / "handoff" / "visual_review_light_controlled.zip", ("previews", "sheets", "keyframes", "curves"))
    _zip_tree(output / "public", output / "handoff" / "visual_review_light_public.zip", ("previews", "sheets", "keyframes"))
    return {**completeness, "output": str(output), "image_manifest_sha256": image_hash, "publication_validation": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--visual-config", type=Path)
    args = parser.parse_args()
    vismod = load_module(REPO / "scripts/inspect_hand_object_trajectory.py", "startup_control_visual")
    result = (
        render_visual_standard(args.visual_config, args.output)
        if args.visual_config is not None else render(args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
