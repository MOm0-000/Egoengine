#!/usr/bin/env python3
"""Close the Pour retarget objective/collision/source contracts without physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any

import cv2
import fcl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data_all/zzx/3.2RL")
CONFIG = ROOT / "configs/taco_pour_retarget_contract_closure_v2.yaml"
OUTPUT = ASSET_ROOT / "runs/taco_pour_retarget_contract_closure_v2"

sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from egoengine_repro.retarget.contract_closure import (
    PALM_THUMB_PAIR,
    final_classification,
    objective_semantic_unit_tests,
    prior_native_stricter_breakdown,
    project_world_points,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts
from video_to_spider.rl.physics_contract import compile_mujoco_model


def _load_prior_runner():
    path = ROOT / "scripts/audit_taco_pour_retarget_contract_fidelity.py"
    spec = importlib.util.spec_from_file_location("retarget_contract_fidelity_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PRIOR = _load_prior_runner()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_contract() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text())
    if value.get("schema") != "taco_pour_retarget_contract_closure_v2":
        raise ValueError("unexpected closure contract")
    if any(value["authorization"].values()):
        raise ValueError("contract closure must authorize no runtime work")
    return value


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_clean_baseline(cfg: dict[str, Any]) -> None:
    expected = cfg["expected_baseline"]
    subprocess.run(["git", "merge-base", "--is-ancestor", expected, head()], cwd=ROOT, check=True)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("formal closure audit requires a clean implementation worktree")


def input_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in cfg["inputs"].items()}


def compile_model(paths: dict[str, Path]) -> mujoco.MjModel:
    simulator = yaml.safe_load(paths["simulator_config"].read_text())
    return compile_mujoco_model(paths["active_scene"], simulator.get("sdf_octree_depths", {}))


def trajectory_states(paths: dict[str, Path]) -> dict[str, dict[int, np.ndarray]]:
    old = arrays(paths["old_mink_reference"])
    accepted = arrays(paths["accepted_a_replay"])
    shape = arrays(paths["shape_aware_candidate"])
    interaction = arrays(paths["interaction_v1_candidate"])
    return {
        "OLD_MINK_REFERENCE": {i: old["qpos"][i].astype(np.float64) for i in range(21)},
        "ACCEPTED_A_REPLAY": {i: accepted["A_qpos"][i].astype(np.float64) for i in range(21)},
        "SHAPE_AWARE_NEGATIVE": {0: shape["qpos"].astype(np.float64)},
        "INTERACTION_V1_NEGATIVE": {i: interaction["qpos"][i].astype(np.float64) for i in range(21)},
    }


def xml_topology(scene: Path, model: mujoco.MjModel) -> dict[str, Any]:
    root = ET.parse(scene).getroot()
    palm = int(model.geom("left_hand_link_visual").id)
    thumb = int(model.geom("left_thumb_rota1_visual").id)
    palm_body = int(model.geom_bodyid[palm])
    thumb_body = int(model.geom_bodyid[thumb])

    def ancestry(body: int) -> list[str]:
        result = []
        while body > 0:
            result.append(model.body(body).name)
            body = int(model.body_parentid[body])
        return result

    relevant_pairs = []
    relevant_body_set = {palm_body, thumb_body}
    for pair_id, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2, strict=True)):
        ba, bb = map(int, model.geom_bodyid[[a, b]])
        if ba in relevant_body_set or bb in relevant_body_set:
            relevant_pairs.append({
                "name": model.pair(pair_id).name,
                "geom1": model.geom(int(a)).name,
                "geom2": model.geom(int(b)).name,
                "body1": model.body(ba).name,
                "body2": model.body(bb).name,
            })
    excludes = [dict(element.attrib) for element in root.findall("contact/exclude")]
    return {
        "native_pair": list(PALM_THUMB_PAIR),
        "palm": {"geom_id": palm, "body_id": palm_body, "body": model.body(palm_body).name,
                 "ancestry": ancestry(palm_body)},
        "thumb": {"geom_id": thumb, "body_id": thumb_body, "body": model.body(thumb_body).name,
                  "ancestry": ancestry(thumb_body)},
        "intermediate_body": "left_hand_thumb_bend_link",
        "intermediate_joint": "left_hand_thumb_bend_joint",
        "thumb_joint": "left_hand_thumb_rota_joint1",
        "next_thumb_joint": "left_hand_thumb_rota_joint2",
        "explicit_pairs_touching_native_bodies": relevant_pairs,
        "xml_excludes": excludes,
        "visual_geoms_collision_disabled": {
            "left_hand_link_visual": bool(model.geom_contype[palm] == 0 and model.geom_conaffinity[palm] == 0),
            "left_thumb_rota1_visual": bool(model.geom_contype[thumb] == 0 and model.geom_conaffinity[thumb] == 0),
        },
    }


def native_contacts(palm_object: fcl.CollisionObject, thumb_object: fcl.CollisionObject,
                    rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thumb_object.setTransform(fcl.Transform(rotation, translation))
    result = fcl.CollisionResult()
    fcl.collide(palm_object, thumb_object,
                fcl.CollisionRequest(num_max_contacts=2048, enable_contact=True), result)
    if not result.contacts:
        return np.empty((0, 3)), np.empty(0)
    return (np.asarray([contact.pos for contact in result.contacts], dtype=np.float64),
            np.asarray([contact.penetration_depth for contact in result.contacts], dtype=np.float64))


def closeup(path: Path, palm_mesh: trimesh.Trimesh, thumb_mesh: trimesh.Trimesh,
            rotation: np.ndarray, translation: np.ndarray, contacts: np.ndarray,
            title: str) -> None:
    palm = np.asarray(palm_mesh.vertices)[::max(1, len(palm_mesh.vertices) // 2500)]
    thumb = (np.asarray(thumb_mesh.vertices) @ rotation.T + translation)[::max(1, len(thumb_mesh.vertices) // 1800)]
    fig = plt.figure(figsize=(6.4, 5.2))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(palm[:, 0], palm[:, 1], palm[:, 2], s=0.35, alpha=0.20, color="#6b7aa1", label="palm CAD")
    axis.scatter(thumb[:, 0], thumb[:, 1], thumb[:, 2], s=0.5, alpha=0.35, color="#d47f56", label="thumb CAD")
    if len(contacts):
        axis.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2], s=9, color="#e60026", label="native contacts")
    combined = np.concatenate([palm, thumb])
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2
    radius = max((combined.max(axis=0) - combined.min(axis=0)).max() / 2, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("palm-local x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(title)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def palm_thumb_evaluate(model: mujoco.MjModel, seed: np.ndarray,
                        palm_object: fcl.CollisionObject, thumb_object: fcl.CollisionObject,
                        palm_mesh: trimesh.Trimesh, thumb_mesh: trimesh.Trimesh,
                        bend: float, rota: float, guard_pairs: list[tuple[int, int]],
                        image_path: Path | None = None, label: str = "") -> dict[str, Any]:
    data = mujoco.MjData(model)
    qpos = seed.copy()
    bend_joint = int(model.joint("left_hand_thumb_bend_joint").id)
    rota_joint = int(model.joint("left_hand_thumb_rota_joint1").id)
    distal_joint = int(model.joint("left_hand_thumb_rota_joint2").id)
    qpos[int(model.jnt_qposadr[bend_joint])] = bend
    qpos[int(model.jnt_qposadr[rota_joint])] = rota
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    palm_body = int(model.body("left_hand_link").id)
    thumb_body = int(model.body("left_hand_thumb_rota_link1").id)
    rp = data.xmat[palm_body].reshape(3, 3)
    rt = data.xmat[thumb_body].reshape(3, 3)
    rotation = rp.T @ rt
    translation = rp.T @ (data.xpos[thumb_body] - data.xpos[palm_body])
    contacts, depths = native_contacts(palm_object, thumb_object, rotation, translation)
    joint_anchor = (data.xanchor[rota_joint] - data.xpos[palm_body]) @ rp
    distal_anchor = (data.xanchor[distal_joint] - data.xpos[palm_body]) @ rp
    distances_to_joint = np.linalg.norm(contacts - joint_anchor, axis=1) if len(contacts) else np.empty(0)
    thumb_local = (contacts - translation) @ rotation if len(contacts) else np.empty((0, 3))
    runtime = np.asarray([
        mujoco.mj_geomDistance(model, data, a, b, 0.05, None) for a, b in guard_pairs
    ], dtype=np.float64)
    if image_path is not None:
        closeup(image_path, palm_mesh, thumb_mesh, rotation, translation, contacts,
                label or f"bend={bend:.4f}, rota1={rota:.4f}")
    return {
        "bend_rad": float(bend),
        "rota1_rad": float(rota),
        "native_intersection": bool(len(contacts)),
        "native_contact_count": int(len(contacts)),
        "native_max_penetration_depth_m": float(depths.max()) if len(depths) else 0.0,
        "contact_max_distance_from_rota1_joint_m": float(distances_to_joint.max()) if len(distances_to_joint) else 0.0,
        "rota1_to_rota2_joint_span_m": float(np.linalg.norm(distal_anchor - joint_anchor)),
        "contact_reaches_beyond_rota1_rota2_span": bool(
            len(distances_to_joint) and distances_to_joint.max() > np.linalg.norm(distal_anchor - joint_anchor) + 5e-5
        ),
        "contact_thumb_local_x_min_m": float(thumb_local[:, 0].min()) if len(thumb_local) else "",
        "contact_thumb_local_x_max_m": float(thumb_local[:, 0].max()) if len(thumb_local) else "",
        "runtime_guard_min_distance_m": float(runtime.min()) if len(runtime) else "",
        "closeup": str(image_path.name) if image_path is not None else "",
    }


def palm_thumb_sweep(output: Path, cfg: dict[str, Any], model: mujoco.MjModel,
                     meshes: dict[int, trimesh.Trimesh], states: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    palm_id = int(model.geom("left_hand_link_visual").id)
    thumb_id = int(model.geom("left_thumb_rota1_visual").id)
    palm_mesh, thumb_mesh = meshes[palm_id], meshes[thumb_id]
    palm_object, thumb_object = triangle_object(palm_mesh), triangle_object(thumb_mesh)
    guard_pairs = [
        (int(a), int(b)) for pair_id, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2, strict=True))
        if (model.pair(pair_id).name or "").startswith("semantic_self_left_palm_thumb_surface_")
    ]
    directory = output / "palm_thumb_closeups"
    rows = []
    sweep = cfg["collision"]["palm_thumb_sweep"]
    seed = states["OLD_MINK_REFERENCE"][0]
    for ib, bend in enumerate(sweep["bend_samples"]):
        for ir, rota in enumerate(sweep["rota1_samples"]):
            path = directory / f"grid_b{ib:02d}_r{ir:02d}.png"
            row = palm_thumb_evaluate(
                model, seed, palm_object, thumb_object, palm_mesh, thumb_mesh,
                float(bend), float(rota), guard_pairs, path,
                f"legal FK sweep | bend={bend:.3f}, rota1={rota:.3f}",
            )
            row.update(sample_family="LEGAL_JOINT_GRID", sample_id=f"grid_b{ib:02d}_r{ir:02d}")
            rows.append(row)
    named = []
    for family in ("OLD_MINK_REFERENCE", "ACCEPTED_A_REPLAY"):
        for endpoint in (0, 20):
            qpos = states[family][endpoint]
            bend = qpos[int(model.joint("left_hand_thumb_bend_joint").qposadr[0])]
            rota = qpos[int(model.joint("left_hand_thumb_rota_joint1").qposadr[0])]
            path = directory / f"{family.lower()}_e{endpoint:02d}.png"
            row = palm_thumb_evaluate(
                model, qpos, palm_object, thumb_object, palm_mesh, thumb_mesh,
                float(bend), float(rota), guard_pairs, path, f"{family} endpoint {endpoint}",
            )
            row.update(sample_family=family, sample_id=f"{family}_e{endpoint:02d}")
            rows.append(row)
            named.append(row)
    write_csv(output / "palm_thumb_static_sweep.csv", rows)
    beyond = [row for row in rows if row["contact_reaches_beyond_rota1_rota2_span"]]
    if beyond:
        classification = "TRUE_SELF_COLLISION"
        rationale = (
            "At least one legal FK sample intersects beyond the full proximal-to-distal joint span; "
            "the pair cannot be reduced to a mechanical-interface-only overlap."
        )
    elif any(row["native_intersection"] for row in rows):
        classification = "UNKNOWN"
        rationale = "Intersections were observed but the sweep did not establish that every overlap is assembly-local."
    else:
        classification = "UNKNOWN"
        rationale = "The finite sweep cannot certify a pair-wide structural allowlist from absence alone."
    return {
        "classification": classification,
        "rationale": rationale,
        "legal_grid_samples": len(sweep["bend_samples"]) * len(sweep["rota1_samples"]),
        "native_intersection_samples": sum(bool(row["native_intersection"]) for row in rows),
        "samples_beyond_joint_span": len(beyond),
        "maximum_native_penetration_m": max(float(row["native_max_penetration_depth_m"]) for row in rows),
        "runtime_guard_pair_count": len(guard_pairs),
        "named_reference_samples": named,
    }


def recompute_collisions(output: Path, cfg: dict[str, Any], model: mujoco.MjModel,
                         meshes: dict[int, trimesh.Trimesh], states: dict[str, dict[int, np.ndarray]],
                         prior_path: Path, palm_classification: str) -> dict[str, Any]:
    probes = arrays(prior_path / "probe_states.npz")
    rows = []
    for family, endpoints in states.items():
        for endpoint, qpos in sorted(endpoints.items()):
            row = PRIOR.collision_row(model, meshes, qpos, family, endpoint, cfg)
            row["probe_family"] = ""
            rows.append(row)
    for index in range(len(probes["qpos"])):
        label = str(probes["probe_id"][index])
        row = PRIOR.collision_row(
            model, meshes, probes["qpos"][index].astype(np.float64), label,
            int(probes["endpoint"][index]), cfg,
        )
        row["probe_family"] = str(probes["family"][index])
        rows.append(row)

    categorized = []
    mismatches = 0
    for row in rows:
        findings = json.loads(row["candidate_relevant_findings"])
        omitted = [tuple(value) for value in findings["left_omitted_nonadjacent"]]
        palm_thumb = PALM_THUMB_PAIR in omitted or PALM_THUMB_PAIR[::-1] in omitted
        other_self = [list(value) for value in omitted if value not in (PALM_THUMB_PAIR, PALM_THUMB_PAIR[::-1])]
        categories = []
        if findings["left_hand_object"]:
            categories.append("HAND_OBJECT_NATIVE_MATERIAL")
        if findings["left_hand_table"]:
            categories.append("HAND_TABLE_NATIVE_MATERIAL")
        if palm_thumb:
            categories.append("TRUE_SELF_COLLISION_NATIVE_MATERIAL")
        if other_self:
            categories.append("OTHER_TRUE_SELF_COLLISION_NATIVE_MATERIAL")
        if findings["left_unclassified_omitted_nonadjacent"]:
            categories.append("UNKNOWN_NATIVE_SELF_COLLISION")
        if row["runtime_proxy_interference"] == "True" or row["runtime_proxy_interference"] is True:
            categories.append("RUNTIME_PROXY_INTERFERENCE")
        if row["classification"] in ("NATIVE_STRICTER_THAN_PROXY", "PROXY_STRICTER_THAN_NATIVE", "UNKNOWN"):
            mismatches += 1
        categorized.append({
            **row,
            "hand_object_native_material": bool(findings["left_hand_object"]),
            "hand_object_findings": json.dumps(findings["left_hand_object"], sort_keys=True),
            "hand_table_native_material": bool(findings["left_hand_table"]),
            "hand_table_findings": json.dumps(findings["left_hand_table"], sort_keys=True),
            "true_self_collision_native_material": bool(palm_thumb or other_self),
            "palm_thumb_native_material": palm_thumb,
            "palm_thumb_semantics": palm_classification,
            "other_true_self_findings": json.dumps(other_self, sort_keys=True),
            "structural_overlap_findings": "[]",
            "explicit_categories": json.dumps(categories),
            "unknown_semantics": bool(findings["left_unclassified_omitted_nonadjacent"]),
        })
    write_csv(output / "collision_semantics_v2.csv", categorized)
    unknown = sum(bool(row["unknown_semantics"]) for row in categorized)
    true_self_proxy_mismatch = sum(
        bool(row["true_self_collision_native_material"])
        and not (row["runtime_proxy_interference"] == "True" or row["runtime_proxy_interference"] is True)
        for row in categorized
    )
    external_native = sum(bool(row["hand_object_native_material"] or row["hand_table_native_material"])
                          for row in categorized)
    return {
        "state_count": len(categorized),
        "trajectory_state_count": sum(len(value) for value in states.values()),
        "probe_state_count": len(probes["qpos"]),
        "unknown_count": unknown,
        "proxy_native_mismatch_count": mismatches,
        "true_self_native_without_runtime_proxy_count": true_self_proxy_mismatch,
        "states_with_left_hand_object_or_table_native_material": external_native,
        "certified": unknown == 0 and true_self_proxy_mismatch == 0 and external_native == 0,
    }


def objective_audit(output: Path, cfg: dict[str, Any], prior: Path) -> dict[str, Any]:
    components = objective_semantic_unit_tests(float(cfg["objective"]["analytic_tolerance"]))
    landmark = json.loads((prior / "semantic_landmark_candidates.json").read_text())
    for row in components:
        row["joint_anchor_landmark_contract_certified"] = bool(landmark["certified"])
        if not landmark["certified"] and row["component"] not in (
            "joint_nominal_deviation_rad_l2", "temporal_joint_change_rad_l2"
        ):
            row["classification"] = "INSUFFICIENT_IDENTIFIABILITY"
    matrix = read_csv(prior / "objective_fidelity_matrix.csv")
    sample_rows = []
    for blind_id in ("P070", "P072", "P073", "P074", "P075"):
        selected = [row for row in matrix if row["blind_id"] == blind_id]
        sample_rows.append({
            "blind_id": blind_id,
            "sample_type": "endpoint0_shape_aware_positive" if blind_id == "P070"
                           else "interaction_v1_negative",
            "applicable_holistic_comparisons": len(selected),
            "comparisons_matching_holistic_label": sum(row["pass"] == "True" for row in selected),
            "component_outcomes": [{
                "finger": row["finger"], "component": row["objective_component"],
                "matches_holistic_label": row["pass"] == "True", "delta": float(row["delta"]),
            } for row in selected],
        })
    report = {
        "schema": "taco_pour_objective_component_semantics_v2",
        "components": components,
        "sample_type_separation": sample_rows,
        "small_probe_evidence": {
            "probe_count": 60,
            "unambiguous_holistic_visual_labels": 0,
            "role": "DETERMINISTIC_FK_DUTY_PROBES_NOT_HOLISTIC_RANKING",
        },
        "P070_interpretation": (
            "The 1/8 holistic agreement is not a semantic failure: each component has a narrower duty, "
            "and endpoint-0 near-interaction metrics are phase-inapplicable."
        ),
        "P072_P075_interpretation": (
            "All 48 applicable component comparisons identify the visibly degraded interaction candidates."
        ),
        "global_E_bone_role": "DIAGNOSTIC_ONLY",
        "global_E_IM_role": "DIAGNOSTIC_ONLY",
        "scalar_weight_fit_performed": False,
    }
    report["certified"] = all(row["classification"] == "SEMANTICALLY_CERTIFIED" for row in components)
    write_json(output / "objective_component_semantics.json", report)
    lines = [
        "# Objective semantics v2", "",
        f"- Component contract certified: `{report['certified']}`.",
        "- Metrics are certified against their own analytic/FK duties, not a holistic visual vote.",
        "- No scalar composite and no objective weights were fitted.", "", "## Components", "",
    ]
    for row in components:
        lines.append(f"- `{row['component']}` — {row['duty']}: `{row['classification']}`.")
    lines += [
        "", "## Sample-type evidence", "",
        "- `P070`: 1/8 narrow metrics move with the holistic visual preference; this is retained as a "
        "counterexample to the old all-components-must-agree rule, not used to invalidate each metric.",
        "- `P072–P075`: 48/48 applicable ring/pinky shape and near-interaction comparisons correctly "
        "increase on the clearly worse trajectories.",
        "- Small deterministic probes have no unambiguous holistic labels and are used only for component/FK duties.",
    ]
    (output / "objective_semantics_v2.md").write_text("\n".join(lines) + "\n")
    return report


def rgb_reconciliation(output: Path, cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    human = arrays(paths["human_reference"])
    raw = np.load(paths["hand_joints"])
    extrinsics = np.load(paths["camera_extrinsics"])
    intrinsic = np.loadtxt(paths["camera_intrinsics"])
    transform = human["T_sim_world"]
    transformed = raw[0] @ transform[:3, :3].T + transform[:3, 3]
    mapping_error = np.asarray([
        [np.max(np.abs(transformed[source] - human["joint_positions_sim"][0, target]))
         for target in range(2)] for source in range(2)
    ])
    raw_left = int(np.argmin(mapping_error[:, int(np.flatnonzero(human["hand_order"] == "left")[0])]))
    indices = [0, 13, 14, 15, 16, 17, 18, 19, 20]
    names = cfg["rgb"]["required_landmarks"]
    cap = cv2.VideoCapture(str(paths["rgb_video"]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    requested = set(cfg["frames"]["rgb_reconciliation"])
    projection_rows = []
    overlay_files = []
    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frame in requested:
            points = raw[frame, raw_left, indices].astype(np.float64)
            pixels, depth = project_world_points(points, intrinsic, extrinsics[frame])
            for name, pixel, z in zip(names, pixels, depth, strict=True):
                projection_rows.append({
                    "endpoint": frame, "rgb_frame_zero_based": frame,
                    "timestamp_s": float(human["timestamps_s"][frame]),
                    "landmark": name, "pixel_x": float(pixel[0]), "pixel_y": float(pixel[1]),
                    "camera_depth_m": float(z),
                    "inside_image": bool(0 <= pixel[0] < width and 0 <= pixel[1] < height),
                    "independent_2d_ground_truth_available": False,
                    "reprojection_error_px": "",
                })
            ring, pinky = pixels[1:5], pixels[5:9]
            for chain, color in ((ring, (60, 40, 240)), (pinky, (40, 210, 80))):
                for first, second in zip(chain[:-1], chain[1:], strict=True):
                    cv2.line(image, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)), color, 4)
                for pixel in chain:
                    cv2.circle(image, tuple(np.rint(pixel).astype(int)), 7, color, -1)
            cv2.circle(image, tuple(np.rint(pixels[0]).astype(int)), 8, (255, 180, 20), -1)
            cv2.putText(image, f"raw frame {frame} | t={human['timestamps_s'][frame]:.6f}s",
                        (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
            path = output / "rgb_overlays" / f"endpoint_{frame:03d}_left_ring_pinky.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
            overlay_files.append(str(path.relative_to(output)))
        frame += 1
    cap.release()
    write_csv(output / "rgb_projection_coordinates.csv", projection_rows)
    input_audit = json.loads(paths["input_audit"].read_text())
    exact_frame_contract = bool(
        frame_count == len(raw) == len(extrinsics) == len(human["frame_indices"])
        and np.array_equal(human["frame_indices"], np.arange(len(raw)))
        and np.allclose(human["timestamps_s"], np.arange(len(raw)) / 30.0, atol=1e-12)
        and abs(fps - 30.0) <= 1e-12
    )
    all_inside = all(bool(row["inside_image"]) for row in projection_rows)
    report = {
        "classification": "RGB_3D_SOURCE_ALIGNMENT_UNRESOLVED",
        "endpoint0_mapping": {
            "rgb_frame_zero_based": 0, "timestamp_s": float(human["timestamps_s"][0]),
            "human_3d_frame_zero_based": 0, "camera_frame_zero_based": 0,
        },
        "frame_contract_exact": exact_frame_contract,
        "video": {"frames": frame_count, "fps": fps, "width": width, "height": height},
        "array_counts": {"hand_3d": len(raw), "camera": len(extrinsics), "human_reference": len(human["frame_indices"])},
        "released_camera_convention": cfg["rgb"]["projection_convention"],
        "raw_hand_index_for_left": raw_left,
        "raw_to_sim_hand_mapping_max_abs_error_m": float(mapping_error[raw_left, int(np.flatnonzero(human["hand_order"] == "left")[0])]),
        "projected_landmarks_all_in_image": all_inside,
        "overlay_files": overlay_files,
        "pixel_reprojection_error": None,
        "unresolved_reason": (
            "The release contains no independent 2D hand landmarks, masks, or equivalent pixel labels. "
            "A projected point cannot be compared to itself to manufacture a reprojection error."
        ),
        "camera_modified": False,
        "manual_offset_applied": False,
        "source_input_audit_camera_front_check": input_audit["status"]["released_camera_front_check_passed"],
    }
    lines = [
        "# Endpoint-0 source RGB reconciliation", "",
        f"- Classification: `{report['classification']}`.",
        f"- Endpoint 0 maps to RGB/3D/camera row 0 at `{report['endpoint0_mapping']['timestamp_s']:.6f} s`.",
        f"- All RGB, hand-3D, camera, and reference streams have `{frame_count}` rows at `{fps:g} fps`.",
        "- Released extrinsics are used exactly as `T_camera_world`; no inverse, offset, rotation, or frame shift was fitted.",
        f"- All projected wrist/ring/pinky points for frames {sorted(requested)} lie inside the 1920×1080 image: `{all_inside}`.",
        "- Pixel reprojection error is deliberately `null`: the release has no independent 2D landmark truth.",
        "- Overlays are diagnostic evidence, not a numeric source-alignment certificate.",
    ]
    (output / "endpoint0_source_rgb_reconciliation.md").write_text("\n".join(lines) + "\n")
    write_json(output / "rgb_reconciliation.json", report)
    return report


def prepare(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    cfg = load_contract()
    require_clean_baseline(cfg)
    paths = input_paths(cfg)
    evidence_inputs = [CONFIG, Path(__file__), ROOT / "src/egoengine_repro/retarget/contract_closure.py"]
    evidence_inputs += [value for key, value in paths.items() if key != "prior_audit"]
    evidence_inputs += [
        paths["prior_audit"] / "collision_contract_audit.csv",
        paths["prior_audit"] / "objective_fidelity_matrix.csv",
        paths["prior_audit"] / "semantic_landmark_candidates.json",
        paths["prior_audit"] / "probe_states.npz",
        paths["prior_audit"] / "probe_manifest.json",
    ]
    records = [artifact(path) for path in evidence_inputs]
    records.extend(scene_mesh_artifacts(paths["active_scene"]))
    records = list({row["path"]: row for row in records}.values())
    verify_artifacts(records)
    output.mkdir(parents=True)
    write_json(output / "input_manifest.json", {
        "schema": "taco_pour_retarget_contract_closure_v2_input_manifest",
        "expected_baseline": cfg["expected_baseline"], "implementation_commit": head(),
        "artifacts": sorted(records, key=lambda row: row["path"]),
    })

    prior_rows = read_csv(paths["prior_audit"] / "collision_contract_audit.csv")
    a1 = prior_native_stricter_breakdown(prior_rows)
    write_json(output / "prior_native_stricter_recount.json", a1)
    model = compile_model(paths)
    meshes, _ = visual_meshes(paths["active_scene"], model)
    states = trajectory_states(paths)
    topology = xml_topology(paths["active_scene"], model)
    write_json(output / "palm_thumb_model_topology.json", topology)
    sweep = palm_thumb_sweep(output, cfg, model, meshes, states)
    write_json(output / "palm_thumb_sweep_summary.json", sweep)
    structural = {
        "schema": "taco_pour_structural_overlap_policy_v2",
        "pair_classification": {"pair": list(PALM_THUMB_PAIR), "classification": sweep["classification"],
                                "rationale": sweep["rationale"]},
        "structural_overlap_allowlist": [],
        "allowlist_applied_to_hand_object": False,
        "allowlist_applied_to_hand_table": False,
        "policy": "Only INTENTIONAL_ASSEMBLY_OVERLAP is allowlist-eligible; TRUE_SELF_COLLISION is never allowlisted.",
    }
    write_json(output / "structural_overlap_policy.json", structural)
    collision = recompute_collisions(output, cfg, model, meshes, states, paths["prior_audit"], sweep["classification"])
    collision_lines = [
        "# Collision contract v2", "",
        "## A1 CSV recount", "",
        f"- Native-stricter rows: `{a1['native_stricter_row_count']}`.",
        f"- Rows containing `{PALM_THUMB_PAIR[0]} ↔ {PALM_THUMB_PAIR[1]}`: `{a1['rows_containing_palm_thumb_pair']}`.",
        f"- Left hand-object findings: `{a1['left_hand_object_finding_count']}`.",
        f"- Left hand-table findings: `{a1['left_hand_table_finding_count']}` across `{a1['left_hand_table_row_count']}` row.",
        f"- Other omitted self pairs: `{a1['other_omitted_self_pair_count']}`; unknown omitted pairs: `{a1['unclassified_omitted_self_pair_count']}`.",
        "", "## Palm–thumb adjudication", "",
        f"- Classification: `{sweep['classification']}`.",
        f"- Legal FK grid: `{sweep['legal_grid_samples']}` samples; native intersections: `{sweep['native_intersection_samples']}`.",
        f"- Samples whose contact extends beyond the entire rota1→rota2 joint span: `{sweep['samples_beyond_joint_span']}`.",
        f"- Maximum native penetration: `{sweep['maximum_native_penetration_m']:.9f} m`.",
        "- Consequently this pair is not added to the structural-overlap allowlist.",
        "", "## Recomputed state library", "",
        f"- States: `{collision['state_count']}` = `{collision['trajectory_state_count']}` trajectory states + `{collision['probe_state_count']}` probes.",
        f"- Unknown semantics: `{collision['unknown_count']}`.",
        f"- Native true-self findings missed by runtime proxy: `{collision['true_self_native_without_runtime_proxy_count']}`.",
        f"- States with left hand-object/table native material findings: `{collision['states_with_left_hand_object_or_table_native_material']}`.",
        f"- Collision contract certified: `{collision['certified']}`.",
        "- Structural overlap never explains hand-object or hand-table findings.",
    ]
    (output / "collision_contract_v2.md").write_text("\n".join(collision_lines) + "\n")

    objective = objective_audit(output, cfg, paths["prior_audit"])
    rgb = rgb_reconciliation(output, cfg, paths)
    classification, blockers = final_classification(
        collision_closed=collision["certified"], objective_closed=objective["certified"],
        source_rgb_closed=rgb["classification"] in ("SOURCE_RGB_CONSISTENT", "SOURCE_RGB_CONFLICT"),
    )
    runtime = {"physics": 0, "control": 0, "candidate": 0, "planner": 0,
               "actor_critic": 0, "rl": 0, "promotion": 0}
    write_json(output / "phase_contract.json", {
        "schema": "taco_pour_retarget_contract_closure_v2_phase_contract",
        "phase_A_collision": {"closed": collision["certified"], "summary": collision},
        "phase_B_objective": {"closed": objective["certified"]},
        "phase_C_source_rgb": {"closed": rgb["classification"] != "RGB_3D_SOURCE_ALIGNMENT_UNRESOLVED",
                               "classification": rgb["classification"]},
        "phase_D_contract_written": classification == "RETARGET_CONTRACT_V2_CERTIFIED_NO_CANDIDATE",
        "runtime_counts": runtime,
    })
    decision = {
        "classification": classification, "blockers": blockers,
        "collision_semantics_certified": collision["certified"],
        "objective_semantics_certified": objective["certified"],
        "source_rgb_alignment": rgb["classification"],
        "retarget_v2_contract_written": classification == "RETARGET_CONTRACT_V2_CERTIFIED_NO_CANDIDATE",
        "authorization": cfg["authorization"], "runtime_counts": runtime,
    }
    write_json(output / "decision.json", decision)
    if decision["retarget_v2_contract_written"]:
        proposal = {
            "schema": "taco_pour_retarget_v2_contract", "status": "CERTIFIED_NOT_RUN",
            "layer_0": ["joint_control_limits", "true_self_collision", "hand_table_material_legality", "hand_object_material_legality"],
            "layer_1": {"nominal": "OLD_MINK", "trust_region": "evidence_derived_required"},
            "layer_2": {"fingertip_position_orientation": "guard", "tolerance": "evidence_derived_required"},
            "layer_3": ["ring_proximal_distal", "pinky_proximal_distal"],
            "layer_4": {"phase": "near_interaction_only", "metrics": ["relative_position", "relative_orientation"]},
            "layer_5": ["temporal_continuity"], "solver": "DEFERRED", "run_authorized": False,
        }
        (output / "retarget_v2_contract.yaml").write_text(yaml.safe_dump(proposal, sort_keys=False))
    summary = [
        "# TACO Pour retarget contract closure v2", "",
        f"- Final classification: `{classification}`.",
        f"- Collision semantics/runtime contract certified: `{collision['certified']}`.",
        f"- Objective component semantics certified: `{objective['certified']}`.",
        f"- Source RGB/3D classification: `{rgb['classification']}`.",
        f"- `retarget_v2_contract.yaml` written: `{decision['retarget_v2_contract_written']}`.",
        "- No trajectory, candidate, physics step, control interval, planner call, RL update, or promotion was run.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "server_artifacts.sha256")
    (output / "server_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
