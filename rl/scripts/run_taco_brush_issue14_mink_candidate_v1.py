#!/usr/bin/env python3
"""Fresh Brush MINK candidate using the unpromoted Issue #14 support plane."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
sys.path[:0] = [str(RL_ROOT / "src"), str(RL_ROOT / "scripts")]

from audit_taco_brush_issue14_frame0_static_penetration_v1 import (  # noqa: E402
    native_world_objects,
    pair_group_audit,
    table_audit,
)
from egoengine_repro.retarget.collision_audit import (  # noqa: E402
    collision_families,
    distances,
)
from egoengine_repro.retarget.support_plane_limit import (  # noqa: E402
    build_native_support_geoms,
    geom_world_vertices,
)
from egoengine_repro.retarget.taco_bimanual import retarget  # noqa: E402
from egoengine_repro.retarget.taco_bimanual_settings import (  # noqa: E402
    UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
    NativeSupportSettings,
    load_taco_bimanual_settings,
)
from egoengine_repro.scene.support_surface import Plane  # noqa: E402
from run_taco_brush_environment_aware_mink_v2 import (  # noqa: E402
    hash_tree,
    native_floor_audit,
    self_collision_audit,
)


SCHEMA = "taco_brush_issue14_mink_candidate_v1"
DEFAULT_CONFIG = RL_ROOT / "configs/taco_brush_issue14_mink_candidate_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_plane(path: Path) -> Plane:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "support_surface_contract_v1":
        raise ValueError("unexpected support contract schema")
    if raw.get("promotion_authorized") is not False:
        raise ValueError("candidate support must remain explicitly unpromoted")
    value = raw["simulator"]
    return Plane(normal=value["normal"], offset=value["offset_m"], frame=value["frame"])


def relevant_pairs(
    model: mujoco.MjModel, family: list[tuple[int, int]], hand_side: str,
    object_side: str,
) -> list[tuple[int, int]]:
    result = []
    for first, second in family:
        names = (model.geom(first).name or "", model.geom(second).name or "")
        if (any(name.startswith(f"collision_hand_{hand_side}_") for name in names)
                and any(name.startswith(f"{object_side}_object_")
                        and not name.endswith("visual") for name in names)):
            result.append((first, second))
    if not result:
        raise ValueError(f"no relevant {hand_side}/{object_side} collision pairs")
    return result


def pickup_screen(
    clearance_m: np.ndarray, hand_object_distance_m: np.ndarray,
    *, lift_delta_m: float, near_distance_m: float, fraction_min: float,
) -> dict[str, Any]:
    clearance = np.asarray(clearance_m, dtype=np.float64)
    distance = np.asarray(hand_object_distance_m, dtype=np.float64)
    if clearance.shape != distance.shape or clearance.ndim != 1 or not len(clearance):
        raise ValueError("pickup screen expects equal nonempty 1-D trajectories")
    if not np.isfinite(clearance).all() or not np.isfinite(distance).all():
        raise ValueError("pickup screen inputs must be finite")
    elevated = clearance >= clearance[0] + lift_delta_m
    near = distance <= near_distance_m
    elevated_count = int(elevated.sum())
    fraction = None if not elevated_count else float(np.mean(near[elevated]))
    passed = bool(elevated_count and fraction is not None and fraction >= fraction_min)
    return {
        "initial_table_clearance_m": float(clearance[0]),
        "maximum_table_clearance_m": float(clearance.max()),
        "maximum_lift_from_initial_m": float(clearance.max() - clearance[0]),
        "first_elevated_frame": (
            None if not elevated_count else int(np.flatnonzero(elevated)[0])
        ),
        "elevated_frame_count": elevated_count,
        "near_frame_count_during_elevation": int(np.logical_and(elevated, near).sum()),
        "near_fraction_during_elevation": fraction,
        "minimum_hand_object_proxy_distance_m": float(distance.min()),
        "penetrating_proxy_frame_count": int((distance < -5e-5).sum()),
        "screen_pass": passed,
        "interpretation": "KINEMATIC_ALIGNMENT_ONLY_NOT_PHYSICAL_LIFT_PROOF",
    }


def trajectory_audit(
    model: mujoco.MjModel, qpos: np.ndarray, plane: Plane,
    screen_cfg: dict[str, Any], output: Path,
) -> dict[str, Any]:
    visual_geoms = {
        "brush": build_native_support_geoms(
            model, mujoco, [model.geom("right_object_visual").id])[0],
        "bowl": build_native_support_geoms(
            model, mujoco, [model.geom("left_object_visual").id])[0],
    }
    families = collision_families(model)
    pairs = {
        "brush": relevant_pairs(model, families["hand_tool"], "right", "right"),
        "bowl": relevant_pairs(model, families["hand_target"], "left", "left"),
    }
    clearances = {name: np.empty(len(qpos)) for name in visual_geoms}
    hand_distances = {name: np.empty(len(qpos)) for name in visual_geoms}
    data = mujoco.MjData(model)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        for name, geom in visual_geoms.items():
            vertices = geom_world_vertices(data, geom, full=True)
            clearances[name][frame] = float(np.min(plane.signed_distance(vertices)))
            hand_distances[name][frame] = float(distances(
                model, data, pairs[name], detection=0.1).min())
    screens = {
        name: pickup_screen(
            clearances[name], hand_distances[name],
            lift_delta_m=float(screen_cfg["object_lift_delta_m"]),
            near_distance_m=float(screen_cfg["relevant_hand_object_near_distance_m"]),
            fraction_min=float(screen_cfg["elevated_near_fraction_min"]),
        ) for name in ("brush", "bowl")
    }
    with (output / "kinematic_pickup_alignment.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "frame", "brush_table_clearance_m", "brush_right_hand_proxy_distance_m",
            "bowl_table_clearance_m", "bowl_left_hand_proxy_distance_m",
        ))
        writer.writeheader()
        for frame in range(len(qpos)):
            writer.writerow({
                "frame": frame,
                "brush_table_clearance_m": clearances["brush"][frame],
                "brush_right_hand_proxy_distance_m": hand_distances["brush"][frame],
                "bowl_table_clearance_m": clearances["bowl"][frame],
                "bowl_left_hand_proxy_distance_m": hand_distances["bowl"][frame],
            })
    return {
        "screen_contract": screen_cfg,
        "objects": screens,
        "all_objects_pass": all(value["screen_pass"] for value in screens.values()),
        "proxy_geometry_caveat": (
            "MuJoCo convex collision proxies are a proximity diagnostic; this screen does "
            "not prove contact force or causal lifting."
        ),
    }


def frame0_static_audit(
    model: mujoco.MjModel, qpos0: np.ndarray, plane: Plane, tolerance_m: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    native = native_world_objects(model, data)
    table = table_audit(native, plane, tolerance_m)
    families = collision_families(model)
    pairs = {
        name: pair_group_audit(model, data, native, families[name], tolerance_m)
        for name in ("self_explicit", "hand_tool", "hand_target", "tool_target")
    }
    penetration_count = sum(
        len(report["native_penetrating_pairs"]) for report in pairs.values())
    unknown_count = sum(len(report["unknown_pairs"]) for report in pairs.values())
    return {
        "table": table,
        "pair_groups": pairs,
        "native_pair_penetration_count": penetration_count,
        "unknown_pair_count": unknown_count,
        "pass": bool(
            not table["penetrating_entities"] and not penetration_count and not unknown_count
        ),
    }


def render_candidate(
    model: mujoco.MjModel, qpos: np.ndarray, frames: list[int], output: Path,
) -> None:
    visual = output / "visuals"
    visual.mkdir()
    data = mujoco.MjData(model)
    # The source scene keeps MuJoCo's default 640-pixel offscreen framebuffer.
    # Stay within that immutable model limit instead of mutating the source XML.
    renderer = mujoco.Renderer(model, height=480, width=640)
    index_lines = ["# Issue #14 MINK candidate visual evidence", "",
                   "Each image shows the fixed front view (left) and top view (right).", ""]
    for frame in frames:
        if not 0 <= frame < len(qpos):
            raise ValueError(f"render frame outside candidate: {frame}")
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        panels = []
        front = mujoco.MjvCamera()
        front.lookat[:] = [0.60, 0.0, 0.78]
        front.distance = 0.70
        front.azimuth = 90
        front.elevation = -18
        top = mujoco.MjvCamera()
        top.lookat[:] = [0.60, 0.0, 0.75]
        top.distance = 0.70
        top.azimuth = 90
        top.elevation = -90
        for camera in (front, top):
            renderer.update_scene(data, camera=camera)
            panels.append(renderer.render().copy())
        image = np.concatenate(panels, axis=1)
        name = f"frame_{frame:03d}_front_top.jpg"
        if not cv2.imwrite(str(visual / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {name}")
        index_lines.append(f"- frame {frame}: [front/top](visuals/{name})")
    renderer.close()
    (output / "VISUAL_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unexpected schema")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", cfg["minimum_baseline"], "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    if cfg["authorization"]["warm_start"]:
        raise ValueError("warm start is forbidden")
    forbidden = (
        "modify_original_data", "modify_active_support_contract", "modify_object_pose",
        "physics", "replay", "mpc", "reinforcement_learning", "promotion", "chunk_commit",
    )
    if any(cfg["authorization"][key] for key in forbidden):
        raise ValueError("forbidden operation authorized")
    paths = {key: Path(value).resolve(strict=True) for key, value in cfg["paths"].items()
             if key != "output"}
    output = Path(cfg["paths"]["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    candidate_plane = load_plane(paths["candidate_support_contract"])
    active_raw = yaml.safe_load(paths["active_support_contract"].read_text(encoding="utf-8"))
    active_z = float(active_raw["simulator"]["offset_m"])
    expected = cfg["only_scientific_change"]
    if active_z != float(expected["from_support_simulator_z_m"]):
        raise ValueError("active support no longer matches the frozen comparison")
    if candidate_plane.offset != float(expected["to_support_simulator_z_m"]):
        raise ValueError("candidate support height mismatch")
    issue14 = json.loads(paths["issue14_static_audit"].read_text(encoding="utf-8"))
    settling = json.loads(paths["isolated_settling_audit"].read_text(encoding="utf-8"))
    if (float(issue14["candidate_support"]["simulator_plane"]["offset_m"])
            != candidate_plane.offset
            or issue14["decision"]["issue14_candidate_has_penetration"]
            or settling["verdict"] != "STABLE_SEATING"):
        raise ValueError("candidate support prerequisite evidence is not satisfied")

    settings, consumption = load_taco_bimanual_settings(paths["retarget_settings"])
    if consumption["unknown_key_count"] or consumption["unused_key_count"]:
        raise ValueError("retarget settings are not exactly consumed")
    support_cfg = cfg["frozen_native_support"]
    support = NativeSupportSettings(
        normal=tuple(float(value) for value in candidate_plane.normal),
        offset_m=float(candidate_plane.offset), frame=candidate_plane.frame,
        minimum_clearance_m=float(support_cfg["minimum_clearance_m"]),
        activation_distance_m=support_cfg["activation_distance_m"],
        gain=float(support_cfg["gain"]),
        depenetration_step_m=float(support_cfg["depenetration_step_m"]),
        validation_tolerance_m=float(support_cfg["validation_tolerance_m"]),
    )
    candidate_settings = replace(
        settings,
        final_feasibility_algorithm=UNIFIED_SELF_AND_NATIVE_SUPPORT_FEASIBILITY,
        native_support=support,
    )
    failure = None
    try:
        retarget(paths["scene"], paths["human_reference"], candidate_settings, output)
    except RuntimeError as error:
        failure = str(error)

    if failure is not None:
        result = {
            "schema": SCHEMA, "classification": "MINK_TRAJECTORY_INFEASIBLE",
            "failure": failure, "promotion_authorized": False,
            "physics_pickup_validated": False,
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        }
        write_json(output / "results.json", result)
        (output / "summary.md").write_text(
            "# Issue #14 Brush MINK candidate\n\n"
            f"Fresh MINK failed closed: `{failure}`.\n",
            encoding="utf-8",
        )
        hash_tree(output)
        print(json.dumps(result, indent=2))
        return 0

    with np.load(output / "robot_reference.npz", allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.shape != (cfg["sample"]["frames"], 50):
        raise ValueError("fresh MINK did not write the complete trajectory")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    model.geom_pos[model.geom("floor").id, 2] = candidate_plane.offset
    floor = native_floor_audit(
        model, candidate_plane, qpos, support.validation_tolerance_m, output)
    self_report = self_collision_audit(model, qpos, output)
    frame0 = frame0_static_audit(
        model, qpos[0], candidate_plane, support.validation_tolerance_m)
    write_json(output / "frame0_static_geometry_audit.json", frame0)
    pickup = trajectory_audit(
        model, qpos, candidate_plane, cfg["kinematic_pickup_screen"], output)
    write_json(output / "kinematic_pickup_alignment.json", pickup)
    render_candidate(
        model, qpos, [int(value) for value in cfg["visual_evidence"]["frame_indices"]],
        output,
    )

    static_pass = bool(
        floor["status"] == "PASS" and self_report["status"] == "PASS" and frame0["pass"]
    )
    if not static_pass:
        classification = "MINK_COMPLETE_STATIC_GEOMETRY_GATE_FAIL"
    elif pickup["all_objects_pass"]:
        classification = "MINK_COMPLETE_KINEMATIC_PICKUP_ALIGNMENT_PASS"
    else:
        classification = "MINK_COMPLETE_KINEMATIC_PICKUP_ALIGNMENT_INCONCLUSIVE"
    retarget_report = json.loads((output / "retarget_report.json").read_text(encoding="utf-8"))
    result = {
        "schema": SCHEMA,
        "classification": classification,
        "candidate_complete": True,
        "candidate_frames": len(qpos),
        "candidate_support": candidate_plane.to_dict(),
        "active_support_unchanged_m": active_z,
        "only_scientific_change": cfg["only_scientific_change"],
        "retarget_summary": {
            key: retarget_report[key] for key in (
                "fingertip_mean_error_m", "fingertip_max_error_m",
                "wrist_mean_error_rad", "self_collision_min_distance_m",
                "joint_limit_min_margin", "joint_limit_violating_frames",
                "frame_velocity_max_ratio", "frame_velocity_violating_intervals",
                "native_support_minimum_distance_m",
            )
        },
        "native_floor_status": floor["status"],
        "self_collision_status": self_report["status"],
        "frame0_static_geometry_pass": frame0["pass"],
        "kinematic_pickup_alignment": pickup,
        "physics_pickup_validated": False,
        "promotion_authorized": False,
        "forbidden_runtime_counts": {
            "physics": 0, "replay": 0, "mpc": 0, "reinforcement_learning": 0,
            "promotion": 0, "chunk_commit": 0,
        },
        "source_artifacts": {
            key: artifact(path) for key, path in paths.items()
        } | {"config": artifact(config_path)},
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }
    write_json(output / "results.json", result)
    brush = pickup["objects"]["brush"]
    bowl = pickup["objects"]["bowl"]
    summary = [
        "# Issue #14 Brush fresh MINK candidate", "",
        f"- Classification: `{classification}`",
        f"- Complete MINK trajectory: `{len(qpos)}/209` frames",
        f"- Native hand-table gate: `{floor['status']}`",
        f"- Self-collision gate: `{self_report['status']}`",
        f"- Frame-0 full native static geometry gate: `{'PASS' if frame0['pass'] else 'FAIL'}`",
        "- Active SupportSurfaceContract modified: `NO`", "",
        "## Kinematic pickup-alignment screen", "",
        "| object | maximum lift from frame 0 (mm) | elevated frames | near fraction while elevated | minimum relevant-hand proxy distance (mm) | screen |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in (("brush", brush), ("bowl", bowl)):
        fraction = value["near_fraction_during_elevation"]
        summary.append(
            f"| {name} | {value['maximum_lift_from_initial_m']*1000:.6f} | "
            f"{value['elevated_frame_count']} | "
            f"{'N/A' if fraction is None else f'{fraction:.3%}'} | "
            f"{value['minimum_hand_object_proxy_distance_m']*1000:.6f} | "
            f"{'PASS' if value['screen_pass'] else 'INCONCLUSIVE'} |"
        )
    summary += ["", "This is a fresh kinematic MINK result. Object poses follow the official "
                "reference and MINK freezes object DOFs; therefore proximity synchronized with "
                "object lift is evidence of hand-object alignment, not proof that contact forces "
                "physically lift either object. Physics, Replay, MPC, and RL did not run.", ""]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    hash_tree(output)
    print(json.dumps({
        "classification": classification,
        "candidate_frames": len(qpos),
        "static_pass": static_pass,
        "pickup_screens": pickup["objects"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
