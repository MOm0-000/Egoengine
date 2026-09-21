"""Audit hand-object and hand-floor collision semantics on all 198 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from audit_taco_pour_first40_collision_attribution import (
    _collides,
    _declared_pairs,
    _runtime_min_distance,
    _runtime_object,
    _set_transforms,
)
from build_taco_pour_initialization_candidates import (
    _bounds_overlap,
    _world_mesh,
)
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts

PROTOCOL = ROOT / "configs/taco_pour_collision_semantics_full_horizon_v1.yaml"
OUTPUT = ROOT / "runs/taco_pour_collision_semantics_full_horizon_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("protocol_name") != "taco_pour_collision_semantics_full_horizon_v1":
        raise ValueError("unsupported full-horizon protocol")
    if protocol.get("scope") != "read_only_full_horizon_collision_gate":
        raise ValueError("full-horizon audit must remain read-only")
    if protocol.get("training_ready") is not False:
        raise ValueError("collision audit cannot enable training")
    for row in protocol["inputs"].values():
        value = Path(row["path"])
        if _sha(value) != row["sha256"]:
            raise ValueError(f"frozen input changed: {value}")
    return protocol


def _is_hand_visual(name: str) -> bool:
    return name.endswith("_visual") and not name.endswith("object_visual")


def _signed_distance_scenes(meshes):
    import open3d as o3d

    scenes = {}
    samples = {}
    for geom, mesh in meshes.items():
        samples[geom] = np.concatenate([mesh.vertices, mesh.triangles_center])
        if not mesh.is_volume:
            continue
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32),
        )
        scenes[geom] = scene
    return scenes, samples


def _containment_evidence(model, data, source, target, meshes, scenes, samples, threshold):
    if target not in scenes:
        source_bounds = meshes[source].bounds
        target_bounds = meshes[target].bounds
        possible = bool(
            np.all(source_bounds[0] >= target_bounds[0])
            and np.all(source_bounds[1] <= target_bounds[1])
        )
        return {
            "classification": "unclassified_open_target" if possible else "full_containment_impossible_by_aabb",
            "samples": 0,
            "over_50um": None if possible else 0,
        }

    source_body = int(model.geom_bodyid[source])
    target_body = int(model.geom_bodyid[target])
    source_rotation = data.xmat[source_body].reshape(3, 3)
    target_rotation = data.xmat[target_body].reshape(3, 3)
    world = samples[source] @ source_rotation.T + data.xpos[source_body]
    target_local = (world - data.xpos[target_body]) @ target_rotation
    bounds = meshes[target].bounds
    keep = np.all((target_local >= bounds[0]) & (target_local <= bounds[1]), axis=1)
    target_local = target_local[keep]
    if not len(target_local):
        return {
            "classification": "closed_target_aabb_rejected_all_samples",
            "samples": 0,
            "over_50um": 0,
            "sampled_max_inside_m": 0.0,
        }
    import open3d as o3d

    signed = -scenes[target].compute_signed_distance(
        o3d.core.Tensor(target_local, dtype=o3d.core.Dtype.Float32),
        nthreads=4,
        nsamples=11,
    ).numpy().astype(np.float64)
    if not np.isfinite(signed).all():
        raise ValueError("nonfinite native signed distance")
    return {
        "classification": "closed_target_all_aabb_candidate_vertices_and_centroids",
        "samples": int(len(target_local)),
        "over_50um": int(np.count_nonzero(signed > threshold)),
        "sampled_max_inside_m": float(max(0.0, signed.max())),
    }


def _hand_object_audit(protocol, model, data, meshes, native, runtime, qpos):
    hand = sorted(
        (geom for geom in meshes if _is_hand_visual(model.geom(geom).name)),
        key=lambda geom: model.geom(geom).name,
    )
    objects = [model.geom(f"{side}_object_visual").id for side in ("right", "left")]
    threshold = float(protocol["thresholds"]["native_material_interference_m"])
    runtime_threshold = float(protocol["thresholds"]["runtime_interference_m"])
    scenes, samples = _signed_distance_scenes(meshes)

    pairs = []
    for hand_geom in hand:
        for object_geom in objects:
            hand_body = int(model.geom_bodyid[hand_geom])
            object_body = int(model.geom_bodyid[object_geom])
            declared = _declared_pairs(model, hand_body, object_body)
            changed = any(
                "_external_semantic_" in model.geom(geom).name
                for runtime_pair in declared for geom in runtime_pair
            )
            pairs.append({
                "hand_geom": hand_geom,
                "object_geom": object_geom,
                "hand_body": hand_body,
                "object_body": object_body,
                "geoms": [model.geom(hand_geom).name, model.geom(object_geom).name],
                "changed_external_semantics": changed,
                "declared": declared,
                "rows": [],
            })

    for endpoint, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        _set_transforms(model, data, native, runtime)
        world = {geom: _world_mesh(model, data, geom, meshes[geom]) for geom in hand + objects}
        for pair in pairs:
            ga, gb = pair["hand_geom"], pair["object_geom"]
            first, second = world[ga], world[gb]
            aabb_overlap = _bounds_overlap(first.bounds, second.bounds)
            surface = aabb_overlap and _collides([native[ga]], [native[gb]])
            directions = []
            first_may_be_contained = bool(
                np.all(first.bounds[0] >= second.bounds[0])
                and np.all(first.bounds[1] <= second.bounds[1])
            )
            second_may_be_contained = bool(
                np.all(second.bounds[0] >= first.bounds[0])
                and np.all(second.bounds[1] <= first.bounds[1])
            )
            # Material overlap without a triangle crossing requires complete
            # containment, whose AABB must also be contained. Avoid thousands
            # of expensive signed-distance queries for merely nearby meshes.
            if surface or first_may_be_contained:
                directions.append(_containment_evidence(
                    model, data, ga, gb, meshes, scenes, samples, threshold
                ))
            if surface or second_may_be_contained:
                directions.append(_containment_evidence(
                    model, data, gb, ga, meshes, scenes, samples, threshold
                ))
            if aabb_overlap and not directions:
                directions = [
                    {"classification": "aabb_overlap_without_surface_or_possible_containment",
                     "over_50um": 0}
                ]
            material = any(row["over_50um"] not in (None, 0) for row in directions)
            classified = not any(row["over_50um"] is None for row in directions) or surface
            minimum = _runtime_min_distance(model, data, pair["declared"])
            runtime_collision = minimum is not None and minimum <= runtime_threshold
            native_clear = not surface and not material
            pair["rows"].append({
                "endpoint": endpoint,
                "native_surface_crossing": surface,
                "native_material_interference_over_50um": material,
                "native_evidence_classified": bool(classified),
                "runtime_collision": bool(runtime_collision),
                "runtime_min_distance_m": minimum,
                "material_false_negative": material and not runtime_collision,
                "surface_only_miss": surface and not material and not runtime_collision,
                "native_clear_runtime_false_positive": native_clear and runtime_collision,
            })
        if endpoint % 20 == 0 or endpoint == len(qpos) - 1:
            print(f"hand-object endpoint {endpoint}/{len(qpos) - 1}", flush=True)

    for pair in pairs:
        rows = pair["rows"]
        pair.update({
            "declared_runtime_geom_pair_count": len(pair.pop("declared")),
            "native_surface_crossing_rows": [r["endpoint"] for r in rows if r["native_surface_crossing"]],
            "native_material_over_50um_rows": [r["endpoint"] for r in rows if r["native_material_interference_over_50um"]],
            "material_false_negative_rows": [r["endpoint"] for r in rows if r["material_false_negative"]],
            "surface_only_miss_rows": [r["endpoint"] for r in rows if r["surface_only_miss"]],
            "native_clear_runtime_false_positive_rows": [r["endpoint"] for r in rows if r["native_clear_runtime_false_positive"]],
        })
        pair.pop("hand_geom")
        pair.pop("object_geom")
        pair.pop("hand_body")
        pair.pop("object_body")
    return pairs


def _hand_floor_audit(protocol, model, data, meshes, qpos):
    hand = sorted(
        (geom for geom in meshes if _is_hand_visual(model.geom(geom).name)),
        key=lambda geom: model.geom(geom).name,
    )
    floor = model.geom("floor").id
    floor_body = int(model.geom_bodyid[floor])
    floor_normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    floor_origin = data.geom_xpos[floor].copy()
    material_threshold = float(protocol["thresholds"]["native_material_interference_m"])
    runtime_threshold = float(protocol["thresholds"]["runtime_interference_m"])
    records = []
    for geom in hand:
        body = int(model.geom_bodyid[geom])
        records.append({
            "visual": model.geom(geom).name,
            "body": model.body(body).name,
            "geom": geom,
            "declared": _declared_pairs(model, body, floor_body),
            "rows": [],
        })

    for endpoint, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        for record in records:
            geom = record["geom"]
            vertices = _world_mesh(model, data, geom, meshes[geom]).vertices
            native_distance = float(((vertices - floor_origin) @ floor_normal).min())
            minimum = _runtime_min_distance(model, data, record["declared"])
            runtime_collision = minimum is not None and minimum <= runtime_threshold
            material = native_distance < -material_threshold
            native_clear = native_distance > 0.0
            record["rows"].append({
                "endpoint": endpoint,
                "native_floor_distance_m": native_distance,
                "native_material_below_floor_over_50um": material,
                "runtime_collision": bool(runtime_collision),
                "runtime_min_distance_m": minimum,
                "material_false_negative": material and not runtime_collision,
                "native_clear_runtime_false_positive": native_clear and runtime_collision,
            })

    for record in records:
        rows = record["rows"]
        record.update({
            "declared_runtime_geom_pair_count": len(record.pop("declared")),
            "native_material_below_floor_over_50um_rows": [r["endpoint"] for r in rows if r["native_material_below_floor_over_50um"]],
            "material_false_negative_rows": [r["endpoint"] for r in rows if r["material_false_negative"]],
            "native_clear_runtime_false_positive_rows": [r["endpoint"] for r in rows if r["native_clear_runtime_false_positive"]],
        })
        record.pop("geom")
    return records


def _summary(hand_object, hand_floor):
    object_rows = [row for pair in hand_object for row in pair["rows"]]
    changed_rows = [
        row for pair in hand_object if pair["changed_external_semantics"] for row in pair["rows"]
    ]
    floor_rows = [row for link in hand_floor for row in link["rows"]]

    def count(rows, key):
        return sum(bool(row[key]) for row in rows)

    return {
        "hand_object_pair_count": len(hand_object),
        "hand_object_frame_pair_checks": len(object_rows),
        "hand_object_native_surface_crossings": count(object_rows, "native_surface_crossing"),
        "hand_object_native_material_over_50um": count(object_rows, "native_material_interference_over_50um"),
        "hand_object_material_false_negatives_over_50um": count(object_rows, "material_false_negative"),
        "hand_object_surface_only_misses": count(object_rows, "surface_only_miss"),
        "hand_object_native_clear_runtime_false_positives": count(object_rows, "native_clear_runtime_false_positive"),
        "changed_external_semantics_pair_count": sum(p["changed_external_semantics"] for p in hand_object),
        "changed_external_semantics_frame_pair_checks": len(changed_rows),
        "changed_external_semantics_material_false_negatives_over_50um": count(changed_rows, "material_false_negative"),
        "changed_external_semantics_native_clear_runtime_false_positives": count(changed_rows, "native_clear_runtime_false_positive"),
        "hand_floor_link_count": len(hand_floor),
        "hand_floor_frame_link_checks": len(floor_rows),
        "hand_floor_native_material_over_50um": count(floor_rows, "native_material_below_floor_over_50um"),
        "hand_floor_material_false_negatives_over_50um": count(floor_rows, "material_false_negative"),
        "hand_floor_native_clear_runtime_false_positives": count(floor_rows, "native_clear_runtime_false_positive"),
    }


def run(protocol_path: Path, output: Path):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    protocol = _load(protocol_path)
    scene = Path(protocol["inputs"]["scene"]["path"])
    reference = Path(protocol["inputs"]["robot_reference"]["path"])
    with np.load(reference, allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"], dtype=float)
        frame_indices = np.asarray(source["frame_indices"])
    first = int(protocol["window"]["first_endpoint"])
    last = int(protocol["window"]["last_endpoint_inclusive"])
    if len(qpos) != 198 or first != 0 or last != 197:
        raise ValueError("v1 requires the complete 198-endpoint reference")
    if not np.array_equal(frame_indices, np.arange(198)):
        raise ValueError("reference must preserve source frame indices 0..197")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    meshes, mesh_paths = visual_meshes(scene, model)
    if len(meshes) != 28:
        raise ValueError("expected 26 hand visuals and two object visuals")
    native = {geom: triangle_object(mesh) for geom, mesh in meshes.items()}
    runtime_ids = [
        geom for geom in range(model.ngeom)
        if model.geom(geom).name.startswith("collision_hand_")
        or (
            model.geom(geom).name.startswith(("right_object_", "left_object_"))
            and not model.geom(geom).name.endswith("visual")
        )
    ]
    runtime = {geom: _runtime_object(model, geom) for geom in runtime_ids}

    hand_object = _hand_object_audit(protocol, model, data, meshes, native, runtime, qpos)
    hand_floor = _hand_floor_audit(protocol, model, data, meshes, qpos)
    summary = _summary(hand_object, hand_floor)
    passed = (
        summary["hand_object_material_false_negatives_over_50um"] == 0
        and summary["hand_object_native_clear_runtime_false_positives"] == 0
        and summary["hand_floor_material_false_negatives_over_50um"] == 0
        and summary["hand_floor_native_clear_runtime_false_positives"] == 0
    )
    report = {
        "status": "full_horizon_external_floor_gate_passed" if passed else "full_horizon_external_floor_gate_failed",
        "protocol": artifact(protocol_path),
        "inputs": {
            key: artifact(Path(value["path"])) for key, value in protocol["inputs"].items()
        },
        "scene_dependencies": scene_mesh_artifacts(scene),
        "window": {"first_endpoint": first, "last_endpoint_inclusive": last, "endpoints": len(qpos)},
        "coverage": protocol["coverage"],
        "summary": summary,
        "hand_object": hand_object,
        "hand_floor": hand_floor,
        "passed": passed,
        "interpretation_limits": [
            "the native surface oracle is triangle intersection at discrete reference endpoints",
            "material evidence uses sampled native-mesh containment deeper than 50 micrometres",
            "the audit does not cover interpolation between endpoints or unseen policy states",
            "no physics step, capacity test, initialization repair, or PPO training is performed",
        ],
        "formal_scene_modified": False,
        "reference_modified": False,
        "simulation_steps_executed": 0,
        "training_ready": False,
        "code": artifact(Path(__file__)),
    }
    verify_artifacts([
        *report["inputs"].values(), *report["scene_dependencies"],
        *[artifact(path) for path in mesh_paths], report["code"], report["protocol"],
    ])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], json.dumps(summary, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
