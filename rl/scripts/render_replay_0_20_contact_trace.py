#!/usr/bin/env python3
"""Offline analysis and rendering for the recorded Replay 0->20 trace."""

from __future__ import annotations

import csv
import html
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/data_all/zzx/3.2RL")
OUTPUT = ROOT / "runs/taco_pour_replay_0_20_contact_trace_v1"
REPO = Path(__file__).resolve().parents[1]
KEYS = (0, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20)


def load_visual_module():
    path = REPO / "scripts/inspect_hand_object_trajectory.py"
    spec = importlib.util.spec_from_file_location("hand_object_visual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score(actual: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    position = float(np.linalg.norm(actual[36:39] - reference[36:39]))
    qa, qr = actual[39:43], reference[39:43]
    ra = Rotation.from_quat(qa[[1, 2, 3, 0]])
    rr = Rotation.from_quat(qr[[1, 2, 3, 0]])
    rotation = float((rr.inv() * ra).magnitude())
    return float(np.hypot(position / 0.12, rotation / 1.5)), position, rotation


def contact_color(group: str) -> np.ndarray:
    colors = {
        "right_hand_tool": [0.1, 0.9, 0.2, 1],
        "left_hand_target": [0.2, 0.6, 1.0, 1],
        "floor_tool": [1.0, 0.25, 0.1, 1],
        "floor_target": [1.0, 0.65, 0.1, 1],
        "right_hand_target": [0.9, 0.1, 0.9, 1],
        "left_hand_tool": [0.2, 0.9, 0.9, 1],
        "tool_target": [1.0, 1.0, 0.1, 1],
    }
    return np.asarray(colors.get(group, [0.8, 0.8, 0.8, 1]), dtype=np.float32)


def add_contact_overlay(renderer, contacts: dict[str, np.ndarray], indices: np.ndarray) -> None:
    for index in indices:
        pos = contacts["pos"][index].astype(np.float64)
        group = str(contacts["group"][index])
        color = contact_color(group)
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.004), pos, np.eye(3).reshape(-1), color)
        renderer.scene.ngeom += 1
        force = contacts["wrench_world_force_torque_on_geom2"][index, :3].astype(np.float64)
        norm = float(np.linalg.norm(force))
        if norm > 1e-7:
            end = pos + force / norm * min(0.065, 0.001 * norm)
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0013, pos, end)
            renderer.scene.ngeom += 1


def render_contact(vis, qpos: np.ndarray, view: str, contacts: dict[str, np.ndarray], indices: np.ndarray, *, collision: bool = False) -> np.ndarray:
    vis.actual.set_qpos(qpos)
    vis.actual_renderer.update_scene(
        vis.actual.data,
        camera=vismod.camera_object(vismod.CAMERAS[view]),
        scene_option=vismod.scene_option(collision=collision),
    )
    add_contact_overlay(vis.actual_renderer, contacts, indices)
    vis.render_updates += 1
    return vis.actual_renderer.render().copy()


def force_on_object(row: int, contacts: dict[str, np.ndarray], role: str) -> np.ndarray:
    force = contacts["wrench_world_force_torque_on_geom2"][row, :3].astype(np.float64)
    role1, role2 = str(contacts["role1"][row]).split(":")[0], str(contacts["role2"][row]).split(":")[0]
    if role2 == role:
        return force
    if role1 == role:
        return -force
    return np.zeros(3)


def write_events(sub: dict[str, np.ndarray], contacts: dict[str, np.ndarray], endpoints: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> list[dict[str, object]]:
    fields = ["event", "source_endpoint", "substep", "global_substep", "time_s", "evidence"]
    events: list[dict[str, object]] = []
    right_tool = contacts["group"] == "right_hand_tool"
    first = int(np.flatnonzero(right_tool)[0])
    events.append({"event": "first_right_hand_tool_contact", "source_endpoint": int(contacts["source_endpoint"][first]), "substep": int(contacts["substep"][first]), "global_substep": int(contacts["global_substep"][first]), "time_s": int(contacts["global_substep"][first]) / 300.0, "evidence": "first recorded right-hand/tool contact row"})
    for global_step in range(1, 201):
        mask = contacts["global_substep"] == global_step
        groups = set(contacts["group"][mask].tolist())
        previous = set(contacts["group"][contacts["global_substep"] == global_step - 1].tolist()) if global_step > 1 else set()
        if groups != previous and 135 <= global_step <= 170:
            events.append({"event": "contact_group_change", "source_endpoint": (global_step - 1) // 10, "substep": (global_step - 1) % 10, "global_substep": global_step, "time_s": global_step / 300.0, "evidence": f"added={sorted(groups-previous)} removed={sorted(previous-groups)}"})
    with (OUTPUT / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(events)
    return events


def write_group_summary(sub: dict[str, np.ndarray], contacts: dict[str, np.ndarray]) -> list[dict[str, object]]:
    groups = sorted(set(contacts["group"].tolist()))
    rows = []
    dt = 1.0 / 300.0
    for global_step in range(1, 201):
        subrow = int(np.flatnonzero(sub["global_substep"] == global_step)[0])
        for group in groups:
            ids = np.flatnonzero((contacts["global_substep"] == global_step) & (contacts["group"] == group))
            if not len(ids):
                continue
            forces = contacts["wrench_world_force_torque_on_geom2"][ids, :3]
            rows.append({"global_substep": global_step, "source_endpoint": int(sub["source_endpoint"][subrow]), "substep": int(sub["substep"][subrow]), "group": group, "contact_count": len(ids), "normal_force_sum_N": float(contacts["wrench_contact_force_torque"][ids, 0].sum()), "interface_force_vector_sum_x_N": float(forces[:, 0].sum()), "interface_force_vector_sum_y_N": float(forces[:, 1].sum()), "interface_force_vector_sum_z_N": float(forces[:, 2].sum()), "normal_impulse_Ns": float(contacts["wrench_contact_force_torque"][ids, 0].sum() * dt), "deepest_dist_m": float(contacts["dist"][ids].min())})
    with (OUTPUT / "contact_groups_by_substep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return rows


def make_curves(sub: dict[str, np.ndarray], contacts: dict[str, np.ndarray], endpoint_qpos: np.ndarray, ref_qpos: np.ndarray) -> None:
    (OUTPUT / "curves").mkdir(exist_ok=True)
    endpoint = np.arange(21)
    pair = []
    tool = []
    target = []
    for i in endpoint:
        pair.append(np.linalg.norm((endpoint_qpos[i, 36:39] - endpoint_qpos[i, 43:46]) - (ref_qpos[i, 36:39] - ref_qpos[i, 43:46])))
        tool.append(np.linalg.norm(endpoint_qpos[i, 36:39] - ref_qpos[i, 36:39]))
        target.append(np.linalg.norm(endpoint_qpos[i, 43:46] - ref_qpos[i, 43:46]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(endpoint, np.asarray(pair) * 1000, label="tool-target relative")
    ax.plot(endpoint, np.asarray(tool) * 1000, label="tool")
    ax.plot(endpoint, np.asarray(target) * 1000, label="target")
    ax.axvspan(15, 16, alpha=.15, color="red"); ax.set(xlabel="endpoint", ylabel="position error (mm)"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT / "curves/object_errors.png", dpi=160); plt.close(fig)
    steps = sub["global_substep"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(steps, np.linalg.norm(sub["qvel_after"][:, 36:39], axis=1), label="tool")
    axes[0].plot(steps, np.linalg.norm(sub["qvel_after"][:, 42:45], axis=1), label="target")
    axes[0].set_ylabel("free-joint linear\nspeed (m/s)"); axes[0].legend()
    for group in ("right_hand_tool", "left_hand_target", "floor_tool", "floor_target"):
        values = np.zeros(200)
        for g in steps:
            mask = (contacts["global_substep"] == g) & (contacts["group"] == group)
            values[int(g)-1] = contacts["wrench_contact_force_torque"][mask, 0].sum()
        axes[1].plot(steps, values, label=group)
    axes[1].set_ylabel("summed normal\ninterface force (N)"); axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(steps, sub["nacon"], color="black"); axes[2].set_ylabel("all active contacts"); axes[2].set_xlabel("global physics substep")
    for ax in axes: ax.axvspan(151, 160, alpha=.12, color="red"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUTPUT / "curves/substep_contacts_motion.png", dpi=160); plt.close(fig)


def render_all(endpoints: dict[str, np.ndarray], sub: dict[str, np.ndarray], contacts: dict[str, np.ndarray], reference: dict[str, np.ndarray], human: dict[str, np.ndarray]) -> dict[str, object]:
    for directory in ("keyframes", "collision", "videos", "substeps_15_16", "thumbnails"):
        (OUTPUT / directory).mkdir(exist_ok=True)
    rgb_path = ROOT / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4"
    rgb_frames, rgb_info = vismod.read_rgb_frames(rgb_path, 21)
    reference_model = vismod.StaticModel.load("MINK reference", ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml")
    actual_model = vismod.StaticModel.load("executed", ROOT / "runs/taco_pour_floor_contact_v1/candidate.xml")
    vis = vismod.Visualizer(reference_model, actual_model)
    writers = {}
    for view in vismod.CAMERAS:
        path = OUTPUT / f"videos/endpoints_0_20_{view}.mp4"
        writers[view] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (vismod.PANEL_WIDTH * 4, vismod.PANEL_HEIGHT))
        if not writers[view].isOpened(): raise RuntimeError(f"cannot open {path}")
    try:
        for endpoint in range(21):
            s, _, _ = score(endpoints["B_qpos"][endpoint], reference["qpos"][endpoint])
            state = vismod.SavedState(endpoint, endpoints["B_qpos"][endpoint], endpoints["B_qvel"][endpoint], s, "OLD TRACKING BELOW BOUNDARY (offline)", "bitwise-reproduced Replay B")
            for view in vismod.CAMERAS:
                hp = vis.render_human(human["T_sim_object_reference"][endpoint], human["joint_positions_sim"][endpoint], view)
                rp = vis.render_reference(reference["qpos"][endpoint], view)
                ap = vis.render_actual(endpoints["B_qpos"][endpoint], view)
                comp = vismod.make_comparison_frame(endpoint, rgb_frames[endpoint], hp, rp, ap, state)
                writers[view].write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
                if endpoint in KEYS:
                    vismod.save_rgb(OUTPUT / f"keyframes/endpoint_{endpoint:03d}_{view}.png", comp)
                if endpoint in (0, 10, *range(12, 21)):
                    collision = vis.render_actual(endpoints["B_qpos"][endpoint], view, collision=True)
                    panel = np.concatenate([vismod.annotate_panel(ap, ["VISUAL GEOMETRY", f"endpoint {endpoint}"]), vismod.annotate_panel(collision, ["ORIGINAL COLLISION GEOMETRY", "offline kinematics; no integration"])], axis=1)
                    vismod.save_rgb(OUTPUT / f"collision/endpoint_{endpoint:03d}_{view}.png", panel)
        # 15->16: solve data is drawn on the cached pre-step geometry.  The last
        # frame is endpoint16 state_after and deliberately has no stale arrow.
        rows = np.flatnonzero(sub["source_endpoint"] == 15)
        for view in vismod.CAMERAS:
            for order, row in enumerate(rows):
                ids = np.flatnonzero(contacts["global_substep"] == sub["global_substep"][row])
                image = render_contact(vis, sub["qpos_before"][row], view, contacts, ids, collision=True)
                image = vismod.annotate_panel(image, ["SOLVE INPUT GEOMETRY + SAVED WRENCH", f"15->16 substep {order}/10 | t={sub['time_before'][row]:.6f}s", "arrows: official MJWarp wrench on geom2"])
                vismod.save_rgb(OUTPUT / f"substeps_15_16/{view}_{order:02d}.png", image)
            image = vis.render_actual(sub["qpos_after"][rows[-1]], view, collision=True)
            image = vismod.annotate_panel(image, ["STATE AFTER INTEGRATOR", f"endpoint16 | t={sub['time_after'][rows[-1]]:.6f}s", "no stale solve arrow drawn"])
            vismod.save_rgb(OUTPUT / f"substeps_15_16/{view}_10.png", image)
        # 12->20 slow motion, pre-step solve-aligned actual/contact plus held RGB.
        for view in vismod.CAMERAS:
            path = OUTPUT / f"videos/substeps_12_20_{view}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (vismod.PANEL_WIDTH * 2, vismod.PANEL_HEIGHT))
            if not writer.isOpened(): raise RuntimeError(f"cannot open {path}")
            for row in np.flatnonzero(sub["source_endpoint"] >= 12):
                source = int(sub["source_endpoint"][row]); ids = np.flatnonzero(contacts["global_substep"] == sub["global_substep"][row])
                actual = render_contact(vis, sub["qpos_before"][row], view, contacts, ids, collision=True)
                actual = vismod.annotate_panel(actual, ["REPLAY SOLVE INPUT + CONTACT", f"{source}->{source+1} substep {int(sub['substep'][row])} | {sub['time_before'][row]:.6f}s"])
                rgb = vismod.annotate_panel(vismod.letterbox(rgb_frames[source], vismod.PANEL_WIDTH, vismod.PANEL_HEIGHT), ["REAL RGB HELD AT SOURCE FRAME", "30Hz source; not 300Hz interpolation"])
                writer.write(cv2.cvtColor(np.concatenate([rgb, actual], axis=1), cv2.COLOR_RGB2BGR))
            writer.release()
    finally:
        for writer in writers.values(): writer.release()
        vis.close()
    return {"rgb": rgb_info, "render_updates": vis.render_updates, "endpoint_frames_review_scope": list(range(21)), "substep_15_16_frames_review_scope": list(range(11)), "cameras": list(vismod.CAMERAS)}


def write_index(render_info: dict[str, object]) -> None:
    cards = []
    for endpoint in KEYS:
        cards.append(f'<a href="keyframes/endpoint_{endpoint:03d}_oblique.png"><img src="keyframes/endpoint_{endpoint:03d}_oblique.png"><br>endpoint {endpoint}</a>')
    subcards = [f'<a href="substeps_15_16/oblique_{i:02d}.png"><img src="substeps_15_16/oblique_{i:02d}.png"><br>15→16 step {i}</a>' for i in range(11)]
    body = f"""<!doctype html><meta charset='utf-8'><title>Replay 0→20 contact trace</title><style>body{{font-family:sans-serif}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}img{{width:100%}}</style><h1>Replay 0→20 contact trace</h1><p>Tracking labels are the old local numerical contract, not a task-quality certificate. RGB is an independent camera. Substep contact arrows are drawn on solve-input geometry.</p><p><a href='videos/endpoints_0_20_oblique.mp4'>endpoint video oblique</a> · <a href='videos/substeps_12_20_oblique.mp4'>12→20 substep video</a></p><h2>Endpoints</h2><div class='grid'>{''.join(cards)}</div><h2>15→16 solve-aligned substeps</h2><div class='grid'>{''.join(subcards)}</div>"""
    (OUTPUT / "index.html").write_text(body)


def main() -> None:
    global vismod
    parity = json.loads((OUTPUT / "replay_parity.json").read_text())
    if not parity.get("B_endpoint_arrays_bitwise_equal_A") or not parity["B_s20_vs_A_s20"]["all_common_equal"]:
        raise RuntimeError("rendering forbidden until passive trace identity is established")
    with np.load(OUTPUT / "endpoints.npz", allow_pickle=False) as archive: endpoints = {k: archive[k].copy() for k in archive.files}
    with np.load(OUTPUT / "substeps.npz", allow_pickle=False) as archive: sub = {k: archive[k].copy() for k in archive.files}
    with np.load(OUTPUT / "contacts_raw.npz", allow_pickle=False) as archive: contacts = {k: archive[k].copy() for k in archive.files}
    with np.load(ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz", allow_pickle=False) as archive: reference = {k: archive[k].copy() for k in archive.files}
    with np.load(ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/human_reference.npz", allow_pickle=False) as archive: human = {k: archive[k].copy() for k in archive.files}
    write_events(sub, contacts, endpoints, reference)
    write_group_summary(sub, contacts)
    make_curves(sub, contacts, endpoints["B_qpos"], reference["qpos"])
    vismod = load_visual_module()
    render_info = render_all(endpoints, sub, contacts, reference, human)
    write_index(render_info)
    (OUTPUT / "visual_review.json").write_text(json.dumps({"status": "images_generated_review_not_yet_recorded", **render_info}, indent=2) + "\n")


if __name__ == "__main__":
    main()
