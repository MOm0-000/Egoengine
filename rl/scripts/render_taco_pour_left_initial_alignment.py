#!/usr/bin/env python3
"""Render the bounded left-initial-alignment comparison with fixed cameras."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


ASSET_ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ASSET_ROOT / "runs/taco_pour_left_initial_alignment_v1"
REPO = Path(__file__).resolve().parents[1]
KEYS = (0, 1, 5, 10, 14, 15, 16, 20)


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


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    return vismod.annotate_panel(image, lines)


def collision_view(vis, qpos: np.ndarray, view: str, label: str) -> np.ndarray:
    vis.actual.set_qpos(qpos)
    vis.actual_renderer.update_scene(
        vis.actual.data,
        camera=vismod.camera_object(vismod.CAMERAS[view]),
        scene_option=vismod.scene_option(collision=True),
    )
    vis.render_updates += 1
    return annotate(vis.actual_renderer.render().copy(), [label, f"{view} | collision geometry"])


def save_sheet(images: list[np.ndarray], path: Path, columns: int = 2) -> None:
    if not images:
        return
    width, height = 960, 300
    tiles = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]
    blank = np.full_like(tiles[0], 245)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    vismod.save_rgb(path, np.concatenate(rows, axis=0))


def render(output: Path) -> dict[str, object]:
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "physics_and_analysis_complete_visual_review_pending":
        raise RuntimeError("rendering requires completed analysis")
    parity = json.loads((output / "replay_parity.json").read_text())
    cold = json.loads((output / "cold_replay_parity.json").read_text())
    if not parity.get("all_bitwise_equal") or not cold.get("all_bitwise_equal"):
        raise RuntimeError("rendering requires original and cold parity")

    labels = ("ORIGINAL", "LEFT_ALIGNED_REPLAY")
    data = {}
    for label in labels:
        directory = output / "conditions" / label
        data[label] = {
            "endpoints": arrays(directory / "endpoints.npz"),
            "substeps": arrays(directory / "substeps.npz"),
        }
    hold_path = output / "conditions/LEFT_ALIGNED_HOLD_1/endpoints.npz"
    if hold_path.is_file():
        data["LEFT_ALIGNED_HOLD_1"] = {
            "endpoints": arrays(hold_path),
            "substeps": arrays(output / "conditions/LEFT_ALIGNED_HOLD_1/substeps.npz"),
        }
    reference = arrays(ASSET_ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz")
    rgb_path = ASSET_ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4"
    rgb_frames, rgb_info = vismod.read_rgb_frames(rgb_path, 21)
    reference_model = vismod.StaticModel.load(
        "MINK reference", ASSET_ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml"
    )
    actual_model = vismod.StaticModel.load(
        "executed", ASSET_ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml"
    )
    vis = vismod.Visualizer(reference_model, actual_model)
    review = output / "review"
    for child in ("initial", "first_cycle", "keyframes", "collision", "videos", "sheets"):
        (review / child).mkdir(parents=True, exist_ok=True)

    first_frames = []
    keyframes = []
    full_video = review / "videos/reference_original_left_aligned_0_20.mp4"
    first_video = review / "videos/original_hold_replay_first_cycle.mp4"
    full_writer = cv2.VideoWriter(
        str(full_video), cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
        (vismod.PANEL_WIDTH * 3, vismod.PANEL_HEIGHT),
    )
    first_writer = cv2.VideoWriter(
        str(first_video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
        (vismod.PANEL_WIDTH * 4, vismod.PANEL_HEIGHT),
    )
    if not full_writer.isOpened() or not first_writer.isOpened():
        raise RuntimeError("cannot open left-alignment video writer")
    try:
        initial = np.concatenate([
            annotate(vismod.letterbox(rgb_frames[0], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT), ["REAL RGB", "source frame 0 | independent view"]),
            annotate(vis.render_reference(reference["qpos"][0], "oblique"), ["ROBOT REFERENCE", "endpoint 0"]),
            annotate(vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][0], "oblique"), ["ACCEPTED A", "original s0"]),
            annotate(vis.render_actual(data["LEFT_ALIGNED_REPLAY"]["endpoints"]["qpos"][0], "oblique"), ["LEFT ALIGNED", "new s0L"]),
        ], axis=1)
        vismod.save_rgb(review / "initial/s0_four_panel.png", initial)
        initial_top = np.concatenate([
            annotate(vis.render_reference(reference["qpos"][0], "top"), ["REFERENCE TOP", "endpoint 0"]),
            annotate(vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][0], "top"), ["ACCEPTED A TOP", "left/tray detail"]),
            annotate(vis.render_actual(data["LEFT_ALIGNED_REPLAY"]["endpoints"]["qpos"][0], "top"), ["LEFT ALIGNED TOP", "left/tray detail"]),
        ], axis=1)
        vismod.save_rgb(review / "initial/s0_left_tray_top.png", initial_top)

        for step in range(10):
            ref = annotate(vis.render_reference(reference["qpos"][1], "oblique"), ["REFERENCE TARGET", "endpoint 1"])
            original = annotate(vis.render_actual(data["ORIGINAL"]["substeps"]["qpos_after"][step], "oblique"), ["ORIGINAL", f"0→1 substep {step + 1}/10"])
            replay = annotate(vis.render_actual(data["LEFT_ALIGNED_REPLAY"]["substeps"]["qpos_after"][step], "oblique"), ["LEFT ALIGNED REPLAY", f"same Replay ctrl | substep {step + 1}/10"])
            if "LEFT_ALIGNED_HOLD_1" in data:
                hold = annotate(vis.render_actual(data["LEFT_ALIGNED_HOLD_1"]["substeps"]["qpos_after"][step], "oblique"), ["LEFT ALIGNED HOLD", f"new initial ctrl | substep {step + 1}/10"])
            else:
                hold = np.full_like(ref, 245)
                hold = annotate(hold, ["LEFT ALIGNED HOLD", "SKIPPED: residual not representable"])
            frame = np.concatenate([ref, original, hold, replay], axis=1)
            first_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vismod.save_rgb(review / f"first_cycle/substep_{step + 1:02d}.png", frame)
            first_frames.append(frame)

        for endpoint in range(21):
            ref = annotate(vis.render_reference(reference["qpos"][endpoint], "oblique"), ["ROBOT REFERENCE", f"endpoint {endpoint}"])
            original = annotate(vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][endpoint], "oblique"), ["ORIGINAL", f"endpoint {endpoint}"])
            aligned = annotate(vis.render_actual(data["LEFT_ALIGNED_REPLAY"]["endpoints"]["qpos"][endpoint], "oblique"), ["LEFT ALIGNED REPLAY", f"endpoint {endpoint}"])
            frame = np.concatenate([ref, original, aligned], axis=1)
            full_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if endpoint in KEYS:
                vismod.save_rgb(review / f"keyframes/endpoint_{endpoint:03d}_oblique.png", frame)
                keyframes.append(frame)
                top = np.concatenate([
                    annotate(vis.render_reference(reference["qpos"][endpoint], "top"), ["REFERENCE TOP", f"endpoint {endpoint}"]),
                    annotate(vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][endpoint], "top"), ["ORIGINAL TOP", f"endpoint {endpoint}"]),
                    annotate(vis.render_actual(data["LEFT_ALIGNED_REPLAY"]["endpoints"]["qpos"][endpoint], "top"), ["LEFT ALIGNED TOP", f"endpoint {endpoint}"]),
                ], axis=1)
                vismod.save_rgb(review / f"keyframes/endpoint_{endpoint:03d}_top.png", top)

        for endpoint in (0, 1, 10, 14, 15, 16, 20):
            collision = np.concatenate([
                collision_view(vis, data["ORIGINAL"]["endpoints"]["qpos"][endpoint], "oblique", f"ORIGINAL endpoint {endpoint}"),
                collision_view(vis, data["LEFT_ALIGNED_REPLAY"]["endpoints"]["qpos"][endpoint], "oblique", f"LEFT ALIGNED endpoint {endpoint}"),
            ], axis=1)
            vismod.save_rgb(review / f"collision/endpoint_{endpoint:03d}.png", collision)
    finally:
        full_writer.release()
        first_writer.release()
        vis.close()

    save_sheet(first_frames, review / "sheets/first_cycle.png")
    save_sheet(keyframes, review / "sheets/keyframes.png")
    result = {
        "schema": "taco_pour_left_initial_alignment_visuals_v1",
        "rgb": rgb_info,
        "fixed_cameras": ["oblique", "top"],
        "endpoint_review_scope": list(range(21)),
        "first_cycle_substeps": list(range(1, 11)),
        "collision_keyframes": [0, 1, 10, 14, 15, 16, 20],
        "render_updates": vis.render_updates,
        "human_review_status": "pending",
    }
    (review / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    vismod = load_module(REPO / "scripts/inspect_hand_object_trajectory.py", "left_alignment_visual")
    print(json.dumps(render(args.output), indent=2))
