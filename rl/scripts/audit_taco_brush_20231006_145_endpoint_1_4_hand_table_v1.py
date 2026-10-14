#!/usr/bin/env python3
"""Compare reference and executed hand/table geometry through endpoint 1-4."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RL_ROOT / "src")]

from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
    native_hand_visual_geom_ids,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402


CAPTURE = RL_ROOT / "runs/taco_brush_20231006_145_endpoint_1_4_hand_table_capture_v1"
PARENT = RL_ROOT / "runs/taco_brush_20231006_145_initialization_controlled_replay_interpolated_v1"
CONFIG = RL_ROOT / "configs/taco_brush_20231006_145_endpoint_1_4_hand_table_capture_v1.yaml"
OUTPUT = RL_ROOT / "runs/taco_brush_20231006_145_endpoint_1_4_hand_table_audit_v1"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
WIDTH = 560
HEIGHT = 420
TOLERANCE_M = 0.00005


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


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False,
            default=json_default,
        ) + "\n",
        encoding="utf-8",
    )


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def side_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "right" in lowered or lowered.startswith("r_"):
        return "right"
    if "left" in lowered or lowered.startswith("l_"):
        return "left"
    return None


def actuator_contract(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for actuator_id in range(model.nu):
        actuator_name = model.actuator(actuator_id).name
        if actuator_name not in {
            "R_forearm_tz_position", "L_forearm_tz_position",
        }:
            continue
        side = side_from_name(actuator_name)
        if side is None:
            raise RuntimeError(f"cannot classify actuator side: {actuator_name}")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        result[side] = {
            "actuator_id": actuator_id,
            "actuator_name": actuator_name,
            "joint_name": model.joint(joint_id).name,
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "dof_address": int(model.jnt_dofadr[joint_id]),
            "position_gain_n_per_m": float(model.actuator_gainprm[actuator_id, 0]),
            "velocity_gain_n_s_per_m": float(-model.actuator_biasprm[actuator_id, 2]),
            "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
            "force_limited": bool(model.actuator_forcelimited[actuator_id]),
        }
    if set(result) != {"left", "right"}:
        raise RuntimeError("vertical wrist actuator contract is incomplete")
    return result


def geometry_minima(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geoms: tuple[Any, ...],
    plane: Plane,
    qpos: np.ndarray,
) -> dict[str, dict[str, Any]]:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    result: dict[str, dict[str, Any]] = {}
    for geom in geoms:
        side = geom.name.split("_", 1)[0]
        vertices = geom_world_vertices(data, geom, full=True)
        distances = plane.signed_distance(vertices)
        index = int(np.argmin(distances))
        record = {
            "distance_m": float(distances[index]),
            "geom": geom.name,
            "body": model.body(geom.body_id).name,
            "point_sim_m": vertices[index].copy(),
        }
        if side not in result or record["distance_m"] < result[side]["distance_m"]:
            result[side] = record
    if set(result) != {"left", "right"}:
        raise RuntimeError("native geometry check did not cover both hands")
    return result


def contact_lookup(rows: list[dict[str, str]]) -> dict[tuple[int, int, str], dict[str, float]]:
    result: dict[tuple[int, int, str], dict[str, float]] = {}
    for row in rows:
        if row["group"] != "hand_floor":
            continue
        side = side_from_name(row["geom1"] + " " + row["geom2"])
        if side is None:
            raise RuntimeError("hand-floor contact has no identifiable hand side")
        key = (int(row["source_endpoint"]), int(row["substep"]), side)
        record = result.setdefault(key, {
            "count": 0, "maximum_force_n": 0.0,
            "minimum_proxy_distance_m": np.inf,
        })
        record["count"] += 1
        record["maximum_force_n"] = max(
            record["maximum_force_n"], float(row["force_norm_n"]),
        )
        record["minimum_proxy_distance_m"] = min(
            record["minimum_proxy_distance_m"], float(row["distance_m"]),
        )
    return result


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result, "RGBA")
    draw.rectangle((0, 0, result.width, 76), fill=(0, 0, 0, 205))
    title_font = ImageFont.truetype(str(FONT), 22)
    detail_font = ImageFont.truetype(str(FONT), 17)
    draw.text((12, 7), lines[0], font=title_font, fill=(255, 255, 255, 255))
    draw.text((12, 42), lines[1], font=detail_font, fill=(255, 255, 255, 255))
    return np.asarray(result)


def add_marker(
    renderer: mujoco.Renderer,
    point: np.ndarray,
    plane: Plane,
    distance_m: float,
    color: np.ndarray,
) -> None:
    scene = renderer.scene
    if scene.ngeom + 2 >= scene.maxgeom:
        raise RuntimeError("renderer scene has no room for diagnostic markers")
    marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        marker, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.004),
        point, np.eye(3).reshape(-1), color,
    )
    scene.ngeom += 1
    table_point = point - distance_m * plane.normal
    connector = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        connector, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
        np.zeros(3), np.eye(3).reshape(-1), color,
    )
    mujoco.mjv_connector(
        connector, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012,
        point, table_point,
    )
    scene.ngeom += 1


def camera(view: str) -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    if view == "side":
        result.lookat[:] = [0.646, -0.093, 0.765]
        result.distance = 0.67
        result.azimuth = 90
        result.elevation = -5
    elif view == "oblique":
        result.lookat[:] = [0.646, -0.093, 0.765]
        result.distance = 0.72
        result.azimuth = 118
        result.elevation = -22
    elif view == "left_close":
        result.lookat[:] = [0.434, -0.143, 0.755]
        result.distance = 0.25
        result.azimuth = 90
        result.elevation = -3
    elif view == "right_close":
        result.lookat[:] = [0.774, 0.080, 0.755]
        result.distance = 0.25
        result.azimuth = 90
        result.elevation = -3
    else:
        raise ValueError(view)
    return result


class HandRenderer:
    def __init__(
        self, model: mujoco.MjModel, geoms: tuple[Any, ...], plane: Plane,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.geoms = geoms
        self.plane = plane
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[0] = 1
        self.option.geomgroup[1] = 1
        object_roots = {
            int(model.body("right_object").id), int(model.body("left_object").id),
        }
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) in object_roots:
                model.geom_rgba[geom_id, 3] = 0.0

    def render(
        self,
        qpos: np.ndarray,
        minima: dict[str, dict[str, Any]],
        view: str,
        color: np.ndarray,
    ) -> np.ndarray:
        for geom in self.geoms:
            self.model.geom_rgba[geom.geom_id] = color
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data, camera=camera(view), scene_option=self.option,
        )
        for side in ("left", "right"):
            row = minima[side]
            add_marker(
                self.renderer, np.asarray(row["point_sim_m"]), self.plane,
                float(row["distance_m"]), color,
            )
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def overlay(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(reference, 0.5, actual, 0.5, 0.0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot write image: {path}")


def render_evidence(
    model: mujoco.MjModel,
    geoms: tuple[Any, ...],
    plane: Plane,
    states: dict[str, np.ndarray],
    geometry_rows: list[dict[str, dict[str, Any]]],
    endpoint_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    visual_dir = output / "visuals"
    keyframe_dir = visual_dir / "keyframes"
    visual_dir.mkdir()
    keyframe_dir.mkdir()
    render_model = mujoco.MjModel.from_xml_path(str(CAPTURE / "experiment_local_scene.xml"))
    render_geoms = build_native_support_geoms(
        render_model, mujoco, native_hand_visual_geom_ids(render_model, mujoco),
    )
    renderer = HandRenderer(render_model, render_geoms, plane)
    reference_color = np.asarray([0.05, 0.70, 0.95, 1.0], dtype=np.float32)
    actual_color = np.asarray([1.0, 0.28, 0.03, 1.0], dtype=np.float32)
    sheets: dict[str, list[np.ndarray]] = {"side": [], "oblique": []}
    closeup_rows: list[np.ndarray] = []
    endpoint_indices = {
        endpoint: int(np.flatnonzero(
            (states["outcome_endpoint"] == endpoint)
            & (states["substep"] == states["substeps"] - 1)
        )[0])
        for endpoint in range(1, 5)
    }
    writer_path = visual_dir / "endpoint_0_to_4_side_slow.mp4"
    writer = cv2.VideoWriter(
        str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
        (WIDTH * 3, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot open diagnostic video writer")
    try:
        for index in range(len(states["state_time_s"])):
            reference_minima = geometry_rows[index]["reference"]
            actual_minima = geometry_rows[index]["actual"]
            ref = renderer.render(
                states["reference_qpos"][index], reference_minima,
                "side", reference_color,
            )
            act = renderer.render(
                states["actual_qpos"][index], actual_minima,
                "side", actual_color,
            )
            endpoint = int(states["outcome_endpoint"][index])
            step = int(states["substep"][index]) + 1
            steps = int(states["substeps"][index])
            ref = annotate(ref, [
                f"参考姿态  {int(states['source_endpoint'][index])}→{endpoint}",
                f"子步 {step}/{steps}  最低离桌 {min(row['distance_m'] for row in reference_minima.values())*1000:.3f} mm",
            ])
            act = annotate(act, [
                f"实际姿态  {int(states['source_endpoint'][index])}→{endpoint}",
                f"子步 {step}/{steps}  最低离桌 {min(row['distance_m'] for row in actual_minima.values())*1000:.3f} mm",
            ])
            combined = annotate(overlay(ref, act), [
                "叠加：蓝色为参考，橙色为实际",
                f"步后状态时刻 {states['state_time_s'][index]*1000:.0f} ms",
            ])
            frame = np.concatenate((ref, act, combined), axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        for endpoint, index in endpoint_indices.items():
            summary = {
                row["side"]: row
                for row in endpoint_rows if row["endpoint"] == endpoint
            }
            for view in sheets:
                reference_minima = geometry_rows[index]["reference"]
                actual_minima = geometry_rows[index]["actual"]
                ref = renderer.render(
                    states["reference_qpos"][index], reference_minima,
                    view, reference_color,
                )
                act = renderer.render(
                    states["actual_qpos"][index], actual_minima,
                    view, actual_color,
                )
                ref = annotate(ref, [
                    f"参考姿态  endpoint {endpoint}",
                    f"左/右离桌 {summary['left']['reference_distance_mm']:.3f} / {summary['right']['reference_distance_mm']:.3f} mm",
                ])
                act = annotate(act, [
                    f"实际姿态  endpoint {endpoint}",
                    f"左/右离桌 {summary['left']['actual_distance_mm']:.3f} / {summary['right']['actual_distance_mm']:.3f} mm",
                ])
                combined = annotate(overlay(ref, act), [
                    "叠加：蓝色为参考，橙色为实际",
                    f"左右手腕竖直误差 {summary['left']['wrist_tz_error_mm']:.3f} / {summary['right']['wrist_tz_error_mm']:.3f} mm",
                ])
                row_image = np.concatenate((ref, act, combined), axis=1)
                sheets[view].append(row_image)
                save_rgb(
                    keyframe_dir / f"endpoint_{endpoint}_{view}.png",
                    row_image,
                )
            closeup_panels = []
            for side, side_name in (("left", "左手"), ("right", "右手")):
                view = f"{side}_close"
                reference_minima = geometry_rows[index]["reference"]
                actual_minima = geometry_rows[index]["actual"]
                ref = renderer.render(
                    states["reference_qpos"][index], reference_minima,
                    view, reference_color,
                )
                act = renderer.render(
                    states["actual_qpos"][index], actual_minima,
                    view, actual_color,
                )
                closeup_panels.append(annotate(overlay(ref, act), [
                    f"endpoint {endpoint}  {side_name}最低点近景",
                    f"蓝色参考 {summary[side]['reference_distance_mm']:.3f} mm；橙色实际 {summary[side]['actual_distance_mm']:.3f} mm",
                ]))
            closeup_row = np.concatenate(closeup_panels, axis=1)
            closeup_rows.append(closeup_row)
            save_rgb(
                keyframe_dir / f"endpoint_{endpoint}_closeup.png",
                closeup_row,
            )
    finally:
        writer.release()
        renderer.close()
    for view, rows in sheets.items():
        save_rgb(visual_dir / f"endpoint_1_4_{view}.png", np.concatenate(rows, axis=0))
    save_rgb(
        visual_dir / "endpoint_1_4_lowest_point_closeup.png",
        np.concatenate(closeup_rows, axis=0),
    )
    return {
        "side_sheet": str((visual_dir / "endpoint_1_4_side.png").relative_to(output)),
        "oblique_sheet": str((visual_dir / "endpoint_1_4_oblique.png").relative_to(output)),
        "side_video": str(writer_path.relative_to(output)),
        "lowest_point_closeup_sheet": str(
            (visual_dir / "endpoint_1_4_lowest_point_closeup.png").relative_to(output)
        ),
        "reference_color_rgb": [13, 179, 242],
        "actual_color_rgb": [255, 71, 8],
        "minimum_point_marker_and_line": True,
        "objects_hidden_to_isolate_hand_table_relation": True,
        "camera_motion": False,
    }


def plot_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    time_ms = np.asarray([
        row["state_time_s"] * 1000 for row in rows if row["side"] == "left"
    ])
    by_side = {
        side: [row for row in rows if row["side"] == side]
        for side in ("left", "right")
    }
    figure, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True,
    )
    colors = {"left": "#c23b22", "right": "#246a73"}
    labels = {"left": "左手", "right": "右手"}
    for side in ("left", "right"):
        values = by_side[side]
        axes[0].plot(
            time_ms, [row["reference_distance_mm"] for row in values],
            color=colors[side], linestyle="--", linewidth=2,
            label=f"{labels[side]}参考",
        )
        axes[0].plot(
            time_ms, [row["actual_distance_mm"] for row in values],
            color=colors[side], linewidth=2, label=f"{labels[side]}实际",
        )
        axes[1].plot(
            time_ms, [row["clearance_difference_mm"] for row in values],
            color=colors[side], linewidth=2,
            label=f"{labels[side]}实际离桌量 - 参考离桌量",
        )
        axes[1].plot(
            time_ms, [row["wrist_tz_error_mm"] for row in values],
            color=colors[side], linestyle=":", linewidth=2,
            label=f"{labels[side]}手腕竖直位置误差",
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axhline(-TOLERANCE_M * 1000, color="#666666", linestyle=":", linewidth=1)
    axes[0].set_ylabel("原生手部网格最低点离桌距离（mm）")
    axes[0].set_title("endpoint 0→4：参考手与实际手的离桌距离")
    axes[0].legend(ncol=2)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("相对参考的误差（mm）")
    axes[1].set_xlabel("仿真时间（ms）")
    axes[1].legend(ncol=2)
    for endpoint_time in (34.0, 68.0, 100.0, 134.0):
        for axis in axes:
            axis.axvline(endpoint_time, color="#999999", linewidth=0.8)
    figure.savefig(output / "endpoint_0_4_hand_table_metrics.png", dpi=180)
    plt.close(figure)


def hash_tree(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "artifacts.sha256").write_text("\n".join(rows) + "\n")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    scene = CAPTURE / "experiment_local_scene.xml"
    state_path = CAPTURE / "substep_state_trace.npz"
    trace_path = CAPTURE / "substep_native_hand_table_trace.csv.gz"
    contact_path = CAPTURE / "contact_trace.csv.gz"
    parent_trace_path = PARENT / "substep_native_hand_table_trace.csv.gz"
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    support_cfg = yaml.safe_load(
        Path(cfg["paths"]["support_contract"]).read_text(encoding="utf-8")
    )
    support = support_cfg["simulator"]
    plane = Plane(
        normal=np.asarray(support["normal"], dtype=np.float64),
        offset=float(support["offset_m"]), frame=str(support["frame"]),
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    geoms = build_native_support_geoms(
        model, mujoco, native_hand_visual_geom_ids(model, mujoco),
    )
    actuators = actuator_contract(model)
    with np.load(state_path, allow_pickle=False) as archive:
        states = {name: archive[name].copy() for name in archive.files}
    trace = read_csv_gz(trace_path)
    parent_trace = read_csv_gz(parent_trace_path)
    parent_prefix_matches = all(
        capture_row[key] == parent_row[key]
        for capture_row, parent_row in zip(
            trace, parent_trace[:len(trace)], strict=True,
        )
        for key in capture_row
    )
    if not parent_prefix_matches:
        raise RuntimeError(
            "state-capture Replay differs from the parent Replay prefix"
        )
    contacts = contact_lookup(read_csv_gz(contact_path))

    geometry_rows: list[dict[str, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    trace_recompute_max_error_m = 0.0
    for index in range(len(states["state_time_s"])):
        reference_minima = geometry_minima(
            model, data, geoms, plane, states["reference_qpos"][index],
        )
        actual_minima = geometry_minima(
            model, data, geoms, plane, states["actual_qpos"][index],
        )
        geometry_rows.append({
            "reference": reference_minima, "actual": actual_minima,
        })
        source = int(states["source_endpoint"][index])
        substep = int(states["substep"][index])
        trace_row = trace[index]
        for side in ("left", "right"):
            actuator = actuators[side]
            qpos_address = int(actuator["qpos_address"])
            dof_address = int(actuator["dof_address"])
            actuator_id = int(actuator["actuator_id"])
            contact = contacts.get((source, substep, side), {
                "count": 0, "maximum_force_n": 0.0,
                "minimum_proxy_distance_m": None,
            })
            reference_distance = float(reference_minima[side]["distance_m"])
            actual_distance = float(actual_minima[side]["distance_m"])
            trace_recompute_max_error_m = max(
                trace_recompute_max_error_m,
                abs(reference_distance - float(trace_row[f"reference_{side}_distance_m"])),
                abs(actual_distance - float(trace_row[f"actual_{side}_distance_m"])),
            )
            row = {
                "source_endpoint": source,
                "outcome_endpoint": int(states["outcome_endpoint"][index]),
                "substep": substep,
                "substeps": int(states["substeps"][index]),
                "control_alpha": float(states["control_alpha"][index]),
                "state_time_s": float(states["state_time_s"][index]),
                "force_interval_start_time_s": float(
                    states["force_interval_start_time_s"][index]
                ),
                "force_interval_end_time_s": float(
                    states["force_interval_end_time_s"][index]
                ),
                "side": side,
                "reference_distance_mm": 1000 * reference_distance,
                "actual_distance_mm": 1000 * actual_distance,
                "clearance_difference_mm": 1000 * (actual_distance - reference_distance),
                "reference_lowest_geom": reference_minima[side]["geom"],
                "reference_lowest_body": reference_minima[side]["body"],
                "actual_lowest_geom": actual_minima[side]["geom"],
                "actual_lowest_body": actual_minima[side]["body"],
                **{
                    f"reference_lowest_point_{axis}_m": float(reference_minima[side]["point_sim_m"][coordinate])
                    for coordinate, axis in enumerate("xyz")
                },
                **{
                    f"actual_lowest_point_{axis}_m": float(actual_minima[side]["point_sim_m"][coordinate])
                    for coordinate, axis in enumerate("xyz")
                },
                "reference_wrist_tz_m": float(states["reference_qpos"][index, qpos_address]),
                "control_target_wrist_tz_m": float(states["control_target"][index, actuator_id]),
                "actual_wrist_tz_m": float(states["actual_qpos"][index, qpos_address]),
                "wrist_tz_error_mm": 1000 * float(
                    states["actual_qpos"][index, qpos_address]
                    - states["reference_qpos"][index, qpos_address]
                ),
                "actual_wrist_tz_velocity_m_s": float(
                    states["actual_qvel"][index, dof_address]
                ),
                "preceding_interval_wrist_tz_actuator_force_n": float(
                    states["preceding_interval_actuator_force"][index, actuator_id]
                ),
                "proxy_hand_floor_contact_count": int(contact["count"]),
                "proxy_hand_floor_maximum_force_n": float(contact["maximum_force_n"]),
                "proxy_hand_floor_minimum_distance_mm": (
                    None if contact["minimum_proxy_distance_m"] is None
                    else 1000 * float(contact["minimum_proxy_distance_m"])
                ),
            }
            rows.append(row)

    with (OUTPUT / "substep_hand_table_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    endpoint_rows: list[dict[str, Any]] = []
    initial_reference = states["initial_reference_qpos"]
    initial_actual = states["initial_actual_qpos"]
    previous_reference = initial_reference
    previous_actual = initial_actual
    previous_time = 0.0
    for endpoint in range(1, 5):
        index = int(np.flatnonzero(
            (states["outcome_endpoint"] == endpoint)
            & (states["substep"] == states["substeps"] - 1)
        )[0])
        time_s = float(states["state_time_s"][index])
        duration = time_s - previous_time
        by_side = {
            row["side"]: row for row in rows
            if row["outcome_endpoint"] == endpoint
            and row["substep"] == row["substeps"] - 1
        }
        for side in ("left", "right"):
            actuator = actuators[side]
            qpos_address = int(actuator["qpos_address"])
            row = dict(by_side[side])
            row.update({
                "endpoint": endpoint,
                "reference_interval_average_wrist_tz_velocity_m_s": float(
                    (states["reference_qpos"][index, qpos_address]
                     - previous_reference[qpos_address]) / duration
                ),
                "actual_interval_average_wrist_tz_velocity_m_s": float(
                    (states["actual_qpos"][index, qpos_address]
                     - previous_actual[qpos_address]) / duration
                ),
                "clearance_minus_wrist_error_mm": (
                    row["clearance_difference_mm"] - row["wrist_tz_error_mm"]
                ),
            })
            endpoint_rows.append(row)
        previous_reference = states["reference_qpos"][index]
        previous_actual = states["actual_qpos"][index]
        previous_time = time_s
    with (OUTPUT / "endpoint_1_4_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(endpoint_rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(endpoint_rows)

    plot_metrics(rows, OUTPUT)
    visual = render_evidence(
        model, geoms, plane, states, geometry_rows, endpoint_rows, OUTPUT,
    )
    first_deficit = next(row for row in rows if row["clearance_difference_mm"] < 0.0)
    first_tolerance_deficit = next(
        row for row in rows if row["clearance_difference_mm"] < -TOLERANCE_M * 1000
    )
    proxy_contact_count = sum(row["proxy_hand_floor_contact_count"] for row in rows)
    endpoint_four = {
        row["side"]: row for row in endpoint_rows if row["endpoint"] == 4
    }
    all_clear = all(
        row["reference_distance_mm"] >= -TOLERANCE_M * 1000
        and row["actual_distance_mm"] >= -TOLERANCE_M * 1000
        for row in rows
    )
    correlation = {
        side: float(np.corrcoef(
            [row["clearance_difference_mm"] for row in rows if row["side"] == side],
            [row["wrist_tz_error_mm"] for row in rows if row["side"] == side],
        )[0, 1])
        for side in ("left", "right")
    }
    result = {
        "schema": "taco_brush_20231006_145_endpoint_1_4_hand_table_audit_v1",
        "scope": "interpolated_control_replay_endpoint_0_to_4",
        "capture_matches_parent_prefix": parent_prefix_matches,
        "physics_substeps": len(states["state_time_s"]),
        "native_mesh_trace_recompute_max_abs_error_m": trace_recompute_max_error_m,
        "state_and_native_geometry_share_the_same_timestamp": True,
        "force_fields_describe_the_preceding_physics_interval": True,
        "reference_and_actual_both_safe_through_endpoint_4": all_clear,
        "first_strict_actual_below_reference": first_deficit,
        "first_actual_below_reference_by_more_than_tolerance": first_tolerance_deficit,
        "proxy_hand_floor_contact_count": proxy_contact_count,
        "endpoint_4": endpoint_four,
        "clearance_difference_vs_wrist_tz_error_correlation": correlation,
        "vertical_actuators": actuators,
        "visuals": visual,
        "inputs": {
            "capture_manifest": artifact(CAPTURE / "input_manifest.json"),
            "capture_state_trace": artifact(state_path),
            "capture_native_trace": artifact(trace_path),
            "capture_contact_trace": artifact(contact_path),
            "parent_native_trace": artifact(parent_trace_path),
            "scene": artifact(scene),
            "config": artifact(CONFIG),
        },
    }
    write_json(OUTPUT / "results.json", result)

    side_label = {"left": "左手", "right": "右手"}
    lines = [
        "# endpoint 1–4 手—桌关系专项对比", "",
        "## 现象概述", "",
        "参考手和实际手在 endpoint 0→4 内都还没有穿桌，但实际手从第一个控制周期开始逐渐落后于参考手。",
        "这段误差几乎全部表现为左右手腕一起向下滞后；到 endpoint 4，实际手比参考手少了约 4 mm 的离桌空间。",
        "随后在 endpoint 4→5 内记录到参考仍在桌面上方、实际手进入桌面的现象。", "",
        "## 数据可信度", "",
        f"- 本次短回放共 `{len(states['state_time_s'])}` 个物理子步，全部在线字段与完整插值回放的同长度前缀逐字段一致：`{parent_prefix_matches}`。",
        f"- 在线原生距离与保存姿态离线重算的最大差为 `{trace_recompute_max_error_m:.3e} m`。",
        "- 下表和全部渲染都从保存的步后 `qpos` 重新执行前向计算，姿态、原生几何和 `state_time_s` 对应同一时刻。",
        "- 执行器力和代理接触属于紧邻该步后状态之前的 2 ms 物理区间，日志另存了区间起点与终点，没有把它们标成终点瞬时量。",
        "- 渲染只隐藏刷子和碗，便于观察手与桌面；没有修改物理运行。", "",
        "## 四个端点", "",
        "| 端点 | 手 | 参考离桌 | 实际离桌 | 实际比参考少 | 手腕竖直误差 | 最低部件 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for endpoint in range(1, 5):
        for side in ("left", "right"):
            row = next(
                item for item in endpoint_rows
                if item["endpoint"] == endpoint and item["side"] == side
            )
            lines.append(
                f"| {endpoint} | {side_label[side]} | {row['reference_distance_mm']:.3f} mm | "
                f"{row['actual_distance_mm']:.3f} mm | {row['clearance_difference_mm']:.3f} mm | "
                f"{row['wrist_tz_error_mm']:.3f} mm | `{row['actual_lowest_body']}` |"
            )
    lines += [
        "", "## 最早偏离", "",
        f"- 第一次出现实际离桌量小于参考：步后 `{first_deficit['state_time_s']*1000:.0f} ms`，0→1 的第 `{first_deficit['substep']+1}/{first_deficit['substeps']}` 个子步，差 `{first_deficit['clearance_difference_mm']:.3f} mm`。",
        f"- 差值第一次超过 0.05 mm：步后 `{first_tolerance_deficit['state_time_s']*1000:.0f} ms`，0→1 的第 `{first_tolerance_deficit['substep']+1}/{first_tolerance_deficit['substeps']}` 个子步，差 `{first_tolerance_deficit['clearance_difference_mm']:.3f} mm`。",
        f"- 这 67 个子步中的手—桌代理接触数：`{proxy_contact_count}`；因此 endpoint 0→4 的向下偏离发生在桌面接触之前。",
        "", "## 为什么主要是竖直跟踪滞后", "",
        f"- endpoint 4 左/右手腕竖直位置误差分别为 `{endpoint_four['left']['wrist_tz_error_mm']:.3f}` / `{endpoint_four['right']['wrist_tz_error_mm']:.3f} mm`。",
        f"- 同一时刻左/右手的离桌空间差分别为 `{endpoint_four['left']['clearance_difference_mm']:.3f}` / `{endpoint_four['right']['clearance_difference_mm']:.3f} mm`。",
        f"- 离桌空间差与手腕竖直误差的相关系数为左手 `{correlation['left']:.6f}`、右手 `{correlation['right']:.6f}`。两条曲线几乎重合。",
        f"- 两个竖直执行器的位置增益均为 `{actuators['left']['position_gain_n_per_m']:.1f} N/m`，速度反馈系数约 `{actuators['left']['velocity_gain_n_s_per_m']:.3f} N·s/m`，且没有执行器力上限；现有数据不支持“达到力饱和”这一解释。",
        "- 初始实际速度被设为零，而参考手在前四段持续向下运动。位置控制器逐步追赶移动目标，形成约 4 mm 的稳定滞后；这是当前数据最直接支持的解释。",
        "", "## 肉眼检查材料", "",
        "- `endpoint_0_4_hand_table_metrics.png`：离桌距离和竖直误差曲线。",
        "- `visuals/endpoint_1_4_side.png`：四个端点的低角度侧视并排图。",
        "- `visuals/endpoint_1_4_oblique.png`：四个端点的斜视并排图。",
        "- `visuals/endpoint_1_4_lowest_point_closeup.png`：每只手最低点的放大叠加图。",
        "- `visuals/endpoint_0_to_4_side_slow.mp4`：全部 67 个物理子步的慢放视频。",
        "- 蓝色表示参考姿态，橙色表示实际姿态；球和短线标出每只手的原生网格最低点及其到桌面的距离。",
        "", "渲染来自保存的真实回放状态，不参与物理计算，也没有改变参考轨迹、控制增益、碰撞参数或桌面位置。", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    hash_tree(OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "parent_prefix_matches": parent_prefix_matches,
        "physics_substeps": len(states["state_time_s"]),
        "all_reference_and_actual_safe_through_endpoint_4": all_clear,
        "endpoint_4_left_clearance_reference_actual_mm": [
            endpoint_four["left"]["reference_distance_mm"],
            endpoint_four["left"]["actual_distance_mm"],
        ],
        "endpoint_4_left_wrist_tz_error_mm": endpoint_four["left"]["wrist_tz_error_mm"],
        "proxy_hand_floor_contact_count": proxy_contact_count,
        "visuals": visual,
    }, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
