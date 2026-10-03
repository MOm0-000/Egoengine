#!/usr/bin/env python3
"""Offline visual evidence for the bounded startup-transition isolation."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ROOT / "runs/taco_pour_startup_transition_isolation_v1"
REPO = Path(__file__).resolve().parents[1]
KEYFRAMES = (0, 1, 5, 10, 14, 15, 16, 20)
PAIR_SPECS = {
    "left_ring_target": ("left_hand:ring", "target"),
    "left_pinky_target": ("left_hand:pinky", "target"),
    "target_floor": ("target", "floor"),
    "right_thumb_floor": ("right_hand:thumb", "floor"),
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


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    return vismod.annotate_panel(image, lines)


def pair_mask(contacts: dict[str, np.ndarray], role_a: str, role_b: str) -> np.ndarray:
    return (
        ((contacts["role1"] == role_a) & (contacts["role2"] == role_b))
        | ((contacts["role1"] == role_b) & (contacts["role2"] == role_a))
    )


def add_contact_overlay(renderer, contacts: dict[str, np.ndarray], indices: np.ndarray) -> None:
    for index in indices:
        pos = contacts["pos"][index].astype(np.float64)
        color = np.asarray([1.0, 0.15, 0.1, 1.0], dtype=np.float32)
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.0045), pos,
            np.eye(3).reshape(-1), color,
        )
        renderer.scene.ngeom += 1
        force = contacts["wrench_world_force_torque_on_geom2"][index, :3].astype(np.float64)
        norm = float(np.linalg.norm(force))
        if norm > 1e-7:
            end = pos + force / norm * min(0.065, 0.001 * norm)
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                np.eye(3).reshape(-1), color,
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0013, pos, end
            )
            renderer.scene.ngeom += 1


def render_contact(vis, qpos: np.ndarray, indices: np.ndarray, contacts: dict[str, np.ndarray]) -> np.ndarray:
    vis.actual.set_qpos(qpos)
    vis.actual_renderer.update_scene(
        vis.actual.data,
        camera=vismod.camera_object(vismod.CAMERAS["oblique"]),
        scene_option=vismod.scene_option(collision=True),
    )
    add_contact_overlay(vis.actual_renderer, contacts, indices)
    vis.render_updates += 1
    return vis.actual_renderer.render().copy()


def save_sheet(images: list[np.ndarray], path: Path, columns: int = 4) -> None:
    if not images:
        return
    width, height = 560, 315
    tiles = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]
    rows = []
    blank = np.full_like(tiles[0], 245)
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    vismod.save_rgb(path, np.concatenate(rows, axis=0))


def render(output: Path) -> dict[str, object]:
    parity = json.loads((output / "replay_parity.json").read_text())
    cold = json.loads((output / "cold_replay_parity.json").read_text())
    manifest = json.loads((output / "input_manifest.json").read_text())
    if not parity.get("all_bitwise_equal") or not cold.get("all_bitwise_equal"):
        raise RuntimeError("rendering requires ORIGINAL identity and cold replay parity")
    if manifest.get("status") != "physics_and_analysis_complete_visual_review_pending":
        raise RuntimeError("rendering requires completed startup analysis")

    data = {}
    for label in ("ORIGINAL", "HOLD_1", "BLEND_5"):
        directory = output / "conditions" / label
        data[label] = {
            "endpoints": arrays(directory / "endpoints.npz"),
            "substeps": arrays(directory / "substeps.npz"),
            "contacts": arrays(directory / "contacts_raw.npz"),
        }
    reference = arrays(
        ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"
    )
    human = arrays(
        ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/human_reference.npz"
    )
    rgb_path = ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4"
    rgb_frames, rgb_info = vismod.read_rgb_frames(rgb_path, 21)
    reference_model = vismod.StaticModel.load(
        "MINK reference",
        ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml",
    )
    actual_model = vismod.StaticModel.load(
        "executed", ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml"
    )
    vis = vismod.Visualizer(reference_model, actual_model)
    visuals = output / "visuals"
    for child in ("first_cycle", "keyframes", "events", "videos", "review_sheets"):
        (visuals / child).mkdir(parents=True, exist_ok=True)

    first_images: list[np.ndarray] = []
    key_images: list[np.ndarray] = []
    event_records: list[dict[str, object]] = []
    full_video_path = visuals / "videos/endpoint_0_20_oblique.mp4"
    first_video_path = visuals / "videos/first_cycle_substeps_oblique.mp4"
    full_writer = cv2.VideoWriter(
        str(full_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
        (vismod.PANEL_WIDTH * 4, vismod.PANEL_HEIGHT),
    )
    first_writer = cv2.VideoWriter(
        str(first_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
        (vismod.PANEL_WIDTH * 4, vismod.PANEL_HEIGHT),
    )
    if not full_writer.isOpened() or not first_writer.isOpened():
        raise RuntimeError("cannot open startup comparison video writer")
    try:
        # The simulator defines only endpoint reference poses; display the
        # exact endpoint-1 target throughout the first control cycle.
        for step in range(10):
            target = annotate(
                vis.render_reference(reference["qpos"][1], "oblique"),
                ["REFERENCE COMMAND TARGET", "endpoint 1 held (no substep reference defined)"],
            )
            original = annotate(
                vis.render_actual(data["ORIGINAL"]["substeps"]["qpos_after"][step], "oblique"),
                ["ORIGINAL", f"0→1 substep {step + 1}/10"],
            )
            hold = annotate(
                vis.render_actual(data["HOLD_1"]["substeps"]["qpos_after"][step], "oblique"),
                ["HOLD_1", f"initial ctrl | substep {step + 1}/10"],
            )
            blend = annotate(
                vis.render_actual(data["BLEND_5"]["substeps"]["qpos_after"][step], "oblique"),
                ["BLEND_5", f"weight 0.05792 | substep {step + 1}/10"],
            )
            frame = np.concatenate([target, original, hold, blend], axis=1)
            first_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vismod.save_rgb(visuals / f"first_cycle/substep_{step + 1:02d}.png", frame)
            first_images.append(frame)

        for endpoint in range(21):
            rgb = annotate(
                vismod.letterbox(rgb_frames[endpoint], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT),
                ["REAL RGB", f"source frame {endpoint}"],
            )
            ref = annotate(
                vis.render_reference(reference["qpos"][endpoint], "oblique"),
                ["MINK REFERENCE", f"endpoint {endpoint}"],
            )
            original = annotate(
                vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][endpoint], "oblique"),
                ["ORIGINAL", f"endpoint {endpoint}"],
            )
            blend = annotate(
                vis.render_actual(data["BLEND_5"]["endpoints"]["qpos"][endpoint], "oblique"),
                ["BLEND_5", f"endpoint {endpoint}"],
            )
            frame = np.concatenate([rgb, ref, original, blend], axis=1)
            full_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if endpoint in KEYFRAMES:
                vismod.save_rgb(
                    visuals / f"keyframes/endpoint_{endpoint:03d}_oblique.png", frame
                )
                key_images.append(frame)
                # A second fixed camera makes overlap/depth judgments auditable.
                ref_top = annotate(
                    vis.render_reference(reference["qpos"][endpoint], "top"),
                    ["MINK REFERENCE TOP", f"endpoint {endpoint}"],
                )
                orig_top = annotate(
                    vis.render_actual(data["ORIGINAL"]["endpoints"]["qpos"][endpoint], "top"),
                    ["ORIGINAL TOP", f"endpoint {endpoint}"],
                )
                blend_top = annotate(
                    vis.render_actual(data["BLEND_5"]["endpoints"]["qpos"][endpoint], "top"),
                    ["BLEND_5 TOP", f"endpoint {endpoint}"],
                )
                top = np.concatenate([ref_top, orig_top, blend_top], axis=1)
                vismod.save_rgb(
                    visuals / f"keyframes/endpoint_{endpoint:03d}_top.png", top
                )

        for label in ("ORIGINAL", "HOLD_1", "BLEND_5"):
            contact = data[label]["contacts"]
            sub = data[label]["substeps"]
            for pair_name, (role_a, role_b) in PAIR_SPECS.items():
                selected = np.flatnonzero(pair_mask(contact, role_a, role_b))
                if not len(selected):
                    event_records.append(
                        {"condition": label, "pair": pair_name, "first_global_substep": None, "image": None}
                    )
                    continue
                global_step = int(contact["global_substep"][selected].min())
                row = int(np.flatnonzero(sub["global_substep"] == global_step)[0])
                event_contacts = selected[contact["global_substep"][selected] == global_step]
                image = render_contact(vis, sub["qpos_before"][row], event_contacts, contact)
                image = annotate(
                    image,
                    [
                        f"{label} | {pair_name}",
                        f"first active solve | global substep {global_step}",
                        "red: saved contact position/wrench on geom2",
                    ],
                )
                relative = f"events/{label.lower()}_{pair_name}_first.png"
                vismod.save_rgb(visuals / relative, image)
                event_records.append(
                    {"condition": label, "pair": pair_name, "first_global_substep": global_step, "image": relative}
                )
    finally:
        full_writer.release()
        first_writer.release()
        vis.close()

    save_sheet(first_images, visuals / "review_sheets/first_cycle.png", columns=2)
    save_sheet(key_images, visuals / "review_sheets/keyframes_oblique.png", columns=2)
    cards = "".join(
        f'<a href="keyframes/endpoint_{endpoint:03d}_oblique.png"><img src="keyframes/endpoint_{endpoint:03d}_oblique.png"><br>endpoint {endpoint}</a>'
        for endpoint in KEYFRAMES
    )
    event_cards = "".join(
        f'<a href="{html.escape(str(row["image"]))}"><img src="{html.escape(str(row["image"]))}"><br>{html.escape(str(row["condition"]))} {html.escape(str(row["pair"]))}</a>'
        for row in event_records if row["image"] is not None
    )
    (visuals / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Startup transition isolation</title>"
        "<style>body{font-family:sans-serif}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}img{width:100%}</style>"
        "<h1>Startup transition isolation</h1>"
        "<p>All simulator panels are offline renders of saved states. Reference in the first-cycle panels is endpoint-1 command target, not an invented 300 Hz trajectory.</p>"
        "<p><a href='videos/first_cycle_substeps_oblique.mp4'>first-cycle video</a> · <a href='videos/endpoint_0_20_oblique.mp4'>0→20 video</a></p>"
        f"<h2>Keyframes</h2><div class='grid'>{cards}</div>"
        f"<h2>First contact events</h2><div class='grid'>{event_cards}</div>"
    )
    report = {
        "status": "images_generated_review_not_yet_recorded",
        "rgb": rgb_info,
        "fixed_primary_camera": "oblique",
        "secondary_keyframe_camera": "top",
        "first_cycle_substeps": list(range(1, 11)),
        "keyframe_endpoints": list(KEYFRAMES),
        "event_records": event_records,
        "render_updates": vis.render_updates,
    }
    (output / "visual_review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    global vismod
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    vismod = load_module(REPO / "scripts/inspect_hand_object_trajectory.py", "startup_visual")
    render(args.output.resolve())


if __name__ == "__main__":
    main()
