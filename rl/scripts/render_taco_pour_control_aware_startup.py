#!/usr/bin/env python3
"""Render frozen evidence for the bounded control-aware startup experiment.

This module is deliberately offline: it loads saved arrays, sets qpos for
kinematics/rendering, and never advances MuJoCo time or constructs an RL policy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


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
            result[condition][endpoint] = {
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


def render(output: Path) -> dict[str, Any]:
    status = json.loads((output / "status.json").read_text())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    vismod = load_module(REPO / "scripts/inspect_hand_object_trajectory.py", "startup_control_visual")
    print(json.dumps(render(args.output), indent=2, sort_keys=True))
