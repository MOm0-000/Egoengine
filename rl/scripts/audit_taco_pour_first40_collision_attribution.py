"""Attribute first-40 native/runtime collision mismatches without changing physics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import fcl
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from build_taco_pour_initialization_candidates import _native_pair, _world_mesh
from egoengine_repro.retarget.paper_audit import artifact, scene_mesh_artifacts, verify_artifacts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("protocol_name") != "taco_pour_first40_collision_attribution_v1":
        raise ValueError("unsupported collision-attribution protocol")
    if protocol.get("scope") != "read_only_collision_diagnostic" or protocol.get("training_ready") is not False:
        raise ValueError("collision attribution must remain diagnostic and training-blocked")
    if protocol.get("simulation_steps") != 0 or protocol.get("repair_allowed") is not False:
        raise ValueError("v1 must not step physics or repair geometry")
    for section, path_key, hash_key in (
        ("formal_scene", "path", "sha256"),
        ("robot_reference", "path", "sha256"),
        ("native_preflight", "report", "report_sha256"),
        ("native_preflight", "checks", "checks_sha256"),
    ):
        value = Path(protocol[section][path_key])
        if _sha(value) != protocol[section][hash_key]:
            raise ValueError(f"frozen input changed: {value}")
    for probe in protocol.get("diagnostic_probes", []):
        value = Path(probe["state"])
        if _sha(value) != probe["state_sha256"]:
            raise ValueError(f"frozen probe changed: {value}")
    return protocol


def _runtime_object(model, geom):
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        shape = fcl.Sphere(float(size[0]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        shape = fcl.Capsule(float(size[0]), float(2 * size[1]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        shape = fcl.Box(*(2 * size).tolist())
    elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=float)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int32)
        geometry = fcl.BVHModel()
        geometry.beginModel(len(vertices), len(faces))
        geometry.addSubModel(vertices, faces)
        geometry.endModel()
        shape = geometry
    else:
        raise ValueError(f"unsupported runtime geom type for {model.geom(geom).name}: {kind}")
    return fcl.CollisionObject(shape)


def _set_transforms(model, data, native, runtime):
    for geom, obj in native.items():
        body = int(model.geom_bodyid[geom])
        obj.setTransform(fcl.Transform(data.xmat[body].reshape(3, 3), data.xpos[body]))
    for geom, obj in runtime.items():
        obj.setTransform(fcl.Transform(data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]))


def _collides(first, second):
    request = fcl.CollisionRequest(num_max_contacts=1)
    for a in first:
        for b in second:
            result = fcl.CollisionResult()
            fcl.collide(a, b, request, result)
            if result.is_collision:
                return True
    return False


def _proxy_geoms(model, body):
    names = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body:
            continue
        name = model.geom(geom).name
        hand = name.startswith("collision_hand_") and "guard" not in name
        obj = name.startswith(("right_object_", "left_object_")) and not name.endswith("visual")
        if hand or obj:
            names.append(geom)
    return names


def _declared_pairs(model, first_body, second_body):
    target = frozenset((first_body, second_body))
    return [(int(a), int(b)) for a, b in zip(model.pair_geom1, model.pair_geom2)
            if frozenset(map(int, model.geom_bodyid[[a, b]])) == target]


def _runtime_min_distance(model, data, pairs):
    if not pairs:
        return None
    return float(min(mujoco.mj_geomDistance(model, data, a, b, 1.0, None) for a, b in pairs))


def _classification(native, actual, declared_count, mixed_first, mixed_second):
    if not native:
        return "runtime_false_positive" if actual else "both_clear"
    if actual:
        return "runtime_detects_native_crossing"
    if not declared_count:
        return "runtime_pair_missing"
    first_missing = not mixed_first   # runtime-first x native-second is clear
    second_missing = not mixed_second  # native-first x runtime-second is clear
    if first_missing and second_missing:
        return "both_runtime_proxies_underfill_contact_region"
    if first_missing:
        return "first_runtime_proxy_underfills_contact_region"
    if second_missing:
        return "second_runtime_proxy_underfills_contact_region"
    return "combined_proxy_arrangement_gap"


def _evaluate_state(model, data, qpos, ga, gb, meshes, native_objects, runtime_objects,
                    proxies_a, proxies_b, declared, protocol):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    _set_transforms(model, data, native_objects, runtime_objects)
    native = _collides([native_objects[ga]], [native_objects[gb]])
    minimum = _runtime_min_distance(model, data, declared)
    actual = bool(minimum is not None and
                  minimum <= protocol["runtime_semantics"]["runtime_interference_threshold_m"])
    first_runtime_native_second = _collides(
        [runtime_objects[g] for g in proxies_a], [native_objects[gb]])
    native_first_runtime_second = _collides(
        [native_objects[ga]], [runtime_objects[g] for g in proxies_b])
    material = None
    if native:
        evidence = _native_pair(
            _world_mesh(model, data, ga, meshes[ga]),
            _world_mesh(model, data, gb, meshes[gb]),
            float(protocol["native_material_reporting_threshold_m"]),
            allow_surface_contact=True,
        )
        material = {
            "classified": evidence["classified"],
            "material_interference_over_50um": evidence["material_interference_over_50um"],
            "directions": evidence.get("directions", []),
        }
    return {
        "native_native_surface_crossing": native,
        "declared_runtime_pair_collision": actual,
        "declared_runtime_min_distance_m": minimum,
        "counterfactual_proxy_proxy_shape_collision": _collides(
            [runtime_objects[g] for g in proxies_a], [runtime_objects[g] for g in proxies_b]),
        "runtime_first_x_native_second": first_runtime_native_second,
        "native_first_x_runtime_second": native_first_runtime_second,
        "classification": _classification(
            native, actual, len(declared), first_runtime_native_second,
            native_first_runtime_second),
        "native_material_evidence": material,
    }


def run(protocol_path: Path, output: Path):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    protocol = load_protocol(protocol_path)
    scene = Path(protocol["formal_scene"]["path"])
    reference = Path(protocol["robot_reference"]["path"])
    preflight_path = Path(protocol["native_preflight"]["report"])
    checks_path = Path(protocol["native_preflight"]["checks"])
    preflight = json.loads(preflight_path.read_text())
    with np.load(reference, allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"])
        frame_indices = np.asarray(source["frame_indices"])
    with np.load(checks_path, allow_pickle=False) as source:
        source_flags = np.asarray(source["surface_contact"], dtype=bool)
    first = int(protocol["endpoints"]["first"])
    last = int(protocol["endpoints"]["last_inclusive"])
    if not np.array_equal(frame_indices, np.arange(len(qpos))) or last >= len(qpos):
        raise ValueError("attribution requires the uncropped source-indexed reference")

    model = mujoco.MjModel.from_xml_path(str(scene))
    meshes, mesh_paths = visual_meshes(scene, model)
    native_objects = {geom: triangle_object(mesh) for geom, mesh in meshes.items()}
    runtime_ids = [geom for geom in range(model.ngeom)
                   if model.geom(geom).name.startswith("collision_hand_")
                   or (model.geom(geom).name.startswith(("right_object_", "left_object_"))
                       and not model.geom(geom).name.endswith("visual"))]
    runtime_objects = {geom: _runtime_object(model, geom) for geom in runtime_ids}
    pair_index = {tuple(row["geoms"]): index for index, row in enumerate(preflight["native_surfaces"]["pairs"])}
    selected = []
    for row in preflight["native_surfaces"]["pairs"]:
        active = [index for index in row["source_rows"] if first <= index <= last]
        include = row["family"] == "hand_object" or (
            row["family"] == "intrahand" and not row["assembly_adjacent"])
        if active and include:
            selected.append((row, active))
    seeds = {tuple(value) for value in protocol["known_material_seed_pairs"]}
    if not seeds.issubset({tuple(row["geoms"]) for row, _ in selected}):
        raise ValueError("a frozen material seed was not selected")

    data = mujoco.MjData(model)
    records = []
    for pair_number, (source, active_rows) in enumerate(selected, 1):
        ga, gb = (model.geom(name).id for name in source["geoms"])
        ba, bb = map(int, model.geom_bodyid[[ga, gb]])
        proxies_a, proxies_b = _proxy_geoms(model, ba), _proxy_geoms(model, bb)
        declared = _declared_pairs(model, ba, bb)
        rows = []
        expected_column = pair_index[tuple(source["geoms"])]
        for endpoint in range(first, last + 1):
            result = _evaluate_state(
                model, data, qpos[endpoint], ga, gb, meshes, native_objects, runtime_objects,
                proxies_a, proxies_b, declared, protocol)
            native = result["native_native_surface_crossing"]
            expected = bool(source_flags[endpoint, expected_column])
            if native != expected:
                raise ValueError(f"native oracle differs from frozen preflight at {source['geoms']} row {endpoint}")
            rows.append({"endpoint": endpoint, **result})
        native_rows = [row for row in rows if row["native_native_surface_crossing"]]
        missed = [row for row in native_rows if not row["declared_runtime_pair_collision"]]
        material_rows = [row for row in native_rows if row["native_material_evidence"]["material_interference_over_50um"]]
        record = {
            "geoms": source["geoms"],
            "family": source["family"],
            "known_material_seed": tuple(source["geoms"]) in seeds,
            "bodies": [model.body(ba).name, model.body(bb).name],
            "candidate_runtime_proxies": [
                [model.geom(g).name for g in proxies_a],
                [model.geom(g).name for g in proxies_b],
            ],
            "declared_runtime_geom_pair_count": len(declared),
            "native_crossing_rows": [row["endpoint"] for row in native_rows],
            "native_material_interference_over_50um_rows": [row["endpoint"] for row in material_rows],
            "runtime_missed_native_crossing_rows": [row["endpoint"] for row in missed],
            "runtime_false_positive_rows": [row["endpoint"] for row in rows
                                             if row["classification"] == "runtime_false_positive"],
            "miss_attribution_counts": dict(Counter(row["classification"] for row in missed)),
            "rows": rows,
        }
        records.append(record)
        print(f"[{pair_number}/{len(selected)}] {source['geoms']}: native={len(native_rows)} "
              f"material={len(material_rows)} missed={len(missed)}", flush=True)

    all_rows = [row for record in records for row in record["rows"]]
    native_rows = [row for row in all_rows if row["native_native_surface_crossing"]]
    missed = [row for row in native_rows if not row["declared_runtime_pair_collision"]]
    material = [row for row in native_rows if row["native_material_evidence"]["material_interference_over_50um"]]
    material_missed = [row for row in material if not row["declared_runtime_pair_collision"]]
    probes = []
    by_names = {tuple(record["geoms"]): record for record in records}
    for probe in protocol.get("diagnostic_probes", []):
        with np.load(probe["state"], allow_pickle=False) as source:
            probe_qpos = np.asarray(source["qpos"])
        probe_pairs = seeds if probe["pairs"] == "known_material_seed_pairs" else {
            tuple(value) for value in probe["pairs"]}
        results = []
        for names in sorted(probe_pairs):
            source = by_names[names]
            ga, gb = (model.geom(name).id for name in names)
            ba, bb = map(int, model.geom_bodyid[[ga, gb]])
            proxies_a, proxies_b = _proxy_geoms(model, ba), _proxy_geoms(model, bb)
            declared = _declared_pairs(model, ba, bb)
            results.append({
                "geoms": list(names),
                "declared_runtime_geom_pair_count": len(declared),
                **_evaluate_state(
                    model, data, probe_qpos, ga, gb, meshes, native_objects, runtime_objects,
                    proxies_a, proxies_b, declared, protocol),
            })
        probes.append({"label": probe["label"], "state": artifact(Path(probe["state"])),
                       "pairs": results})

    report = {
        "status": "first40_collision_attribution_complete_no_repair_applied",
        "protocol": artifact(protocol_path),
        "inputs": {
            "scene": artifact(scene), "reference": artifact(reference),
            "preflight": artifact(preflight_path), "native_checks": artifact(checks_path),
            "scene_dependencies": scene_mesh_artifacts(scene),
        },
        "backends": {"mujoco": mujoco.__version__, "python_fcl": fcl.__version__},
        "window": {"first_endpoint": first, "last_endpoint_inclusive": last,
                   "endpoints": last - first + 1},
        "selection": {
            "selected_pairs": len(records),
            "rule": "hand-object or nonadjacent intrahand with native crossing in endpoints 0..40",
            "adjacent_fixed_assembly_seams_excluded": True,
            "known_material_seeds_present": len(seeds),
        },
        "summary": {
            "native_crossing_frame_pairs": len(native_rows),
            "native_material_interference_over_50um_frame_pairs": len(material),
            "runtime_detected_native_crossing_frame_pairs": len(native_rows) - len(missed),
            "runtime_missed_native_crossing_frame_pairs": len(missed),
            "runtime_missed_material_interference_frame_pairs": len(material_missed),
            "runtime_false_positive_frame_pairs_within_selected_pairs": sum(
                row["classification"] == "runtime_false_positive" for row in all_rows),
            "miss_attribution_counts": dict(Counter(row["classification"] for row in missed)),
        },
        "pairs": records,
        "diagnostic_probes": probes,
        "interpretation_limits": [
            "native FCL reports triangle-surface crossing, not penetration depth",
            "over-50um material evidence is sampled closed-solid containment, not a global maximum-depth certificate",
            "counterfactual proxy tests ignore pair-specific guard geoms unless they are present in the actual declared pair table",
            "the audit covers discrete reference endpoints 0..40, not swept interpolation or unseen policy states",
        ],
        "formal_scene_modified": False,
        "reference_modified": False,
        "simulation_steps_executed": 0,
        "repair_applied": False,
        "training_ready": False,
        "code": artifact(Path(__file__)),
    }
    verify_artifacts([
        report["inputs"]["scene"], report["inputs"]["reference"],
        report["inputs"]["preflight"], report["inputs"]["native_checks"],
        *report["inputs"]["scene_dependencies"], *[artifact(path) for path in mesh_paths],
        *[probe["state"] for probe in probes],
    ])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/taco_pour_first40_collision_attribution_v1.yaml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/taco_pour_first40_collision_attribution_v1")
    args = parser.parse_args()
    run(args.protocol, args.output)
