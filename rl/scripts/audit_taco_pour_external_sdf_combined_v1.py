#!/usr/bin/env python3
"""Build and audit one shared high-resolution SDF per Pour object."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_pour_first40_collision_attribution import (
    _declared_pairs,
    _runtime_min_distance,
)
from probe_taco_pour_external_sdf_v1 import (
    EIGHTH_LABELS,
    ENDPOINT_LABELS,
    MIDPOINT_LABELS,
    QUARTER_LABELS,
    REFERENCE,
    SIXTEENTH_LABELS,
    SOURCE,
    SPECS,
    _fraction_states,
    _label_rows,
)

OUTPUT = ROOT / "runs/taco_pour_external_sdf_combined_v1"
OBJECT_DEPTHS = {"right_visual": 8, "left_visual": 9}


def _build_xml() -> tuple[Path, dict[str, tuple[str, str]]]:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        root.insert(list(root).index(compiler) + 1 if compiler is not None else 0, option)
    option.set("sdf_iterations", "10")
    option.set("sdf_initpoints", "40")
    contact = root.find("contact")
    bodies = {body.get("name"): body for body in root.findall(".//body")}

    original_names = {
        name: {geom.get("name") for geom in body.findall("geom")}
        for name, body in bodies.items()
    }
    for spec in SPECS.values():
        hand_names = original_names[spec["hand_body"]]
        object_names = original_names[spec["object_body"]]
        for pair in list(contact.findall("pair")):
            ends = {pair.get("geom1"), pair.get("geom2")}
            if ends & hand_names and ends & object_names:
                contact.remove(pair)
        hand_body = bodies[spec["hand_body"]]
        for geom in list(hand_body.findall("geom")):
            if "_external_semantic_" in geom.get("name", ""):
                hand_body.remove(geom)

    object_sdf = {}
    for object_body, object_mesh in (("right_object", "right_visual"),
                                     ("left_object", "left_visual")):
        name = f"{object_body}_external_exact_sdf"
        ET.SubElement(
            bodies[object_body], "geom", name=name, type="sdf", mesh=object_mesh,
            group="3", density="0", contype="0", conaffinity="0", condim="3",
            friction="1 0.1 0",
        )
        object_sdf[object_body] = name

    pair_names = {}
    for name, spec in SPECS.items():
        tag = spec["hand_body"].replace("_hand_", "_")
        hand_name = f"collision_hand_{tag}_external_exact_mesh"
        ET.SubElement(
            bodies[spec["hand_body"]], "geom", name=hand_name, type="mesh",
            mesh=spec["hand_mesh"], group="3", density="0", contype="0",
            conaffinity="0", condim="3", friction="1 0.1 0",
        )
        target = object_sdf[spec["object_body"]]
        ET.SubElement(
            contact, "pair", name=f"semantic_sdf_{tag}", geom1=hand_name,
            geom2=target, condim="3", friction="1 1 0.1 0 0",
        )
        pair_names[name] = (hand_name, target)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    scene = OUTPUT / "candidate.xml"
    ET.indent(tree, space="  ")
    tree.write(scene, encoding="unicode")
    return scene, pair_names


def _compile(scene: Path, high_resolution: bool):
    started = time.monotonic()
    spec = mujoco.MjSpec.from_file(str(scene))
    if high_resolution:
        for mesh_name, depth in OBJECT_DEPTHS.items():
            mesh = spec.mesh(mesh_name)
            if not hasattr(mesh, "octree_maxdepth"):
                raise RuntimeError("MuJoCo lacks mesh.octree_maxdepth")
            mesh.needsdf = True
            mesh.octree_maxdepth = depth
    model = spec.compile()
    return model, time.monotonic() - started


def _groups(model, pair):
    endpoints = np.load(REFERENCE, allow_pickle=False)["qpos"]
    return (
        ("endpoints", endpoints, _label_rows(ENDPOINT_LABELS, pair)),
        ("midpoints", _fraction_states(model, endpoints, (0.5,)),
         _label_rows(MIDPOINT_LABELS, pair)),
        ("quarters", _fraction_states(model, endpoints, (0.25, 0.75)),
         _label_rows(QUARTER_LABELS, pair)),
        ("eighths", _fraction_states(
            model, endpoints, (0.125, 0.375, 0.625, 0.875),
         ), _label_rows(EIGHTH_LABELS, pair)),
        ("sixteenths", _fraction_states(
            model, endpoints, tuple(index / 16 for index in range(1, 16, 2)),
         ), _label_rows(SIXTEENTH_LABELS, pair)),
    )


def _validate(model, pair_names):
    data = mujoco.MjData(model)
    validation = {}
    benchmark_states = None
    for name, spec in SPECS.items():
        groups = _groups(model, spec["pair"])
        if benchmark_states is None:
            benchmark_states = np.concatenate([row[1] for row in groups])
        first, second = [model.geom(value).id for value in pair_names[name]]
        declared = _declared_pairs(
            model, int(model.geom_bodyid[first]), int(model.geom_bodyid[second]),
        )
        if len(declared) != 1:
            raise ValueError(f"{name} has {len(declared)} runtime pairs")
        records = {}
        for group_name, states, labels in groups:
            false_negative, false_positive = [], []
            closest_positive = None
            for index, (state, label) in enumerate(zip(states, labels)):
                data.qpos[:] = state
                mujoco.mj_forward(model, data)
                distance = _runtime_min_distance(model, data, declared)
                collision = distance is not None and distance <= 0.0
                if label["native_material_interference_over_50um"]:
                    closest_positive = max(
                        distance,
                        closest_positive if closest_positive is not None else -np.inf,
                    )
                    if not collision:
                        false_negative.append(index)
                if (not label["native_surface_crossing"]
                        and not label["native_material_interference_over_50um"]
                        and collision):
                    false_positive.append(index)
            records[group_name] = {
                "checks": len(states),
                "material_false_negative_indices": false_negative,
                "native_clear_runtime_false_positive_indices": false_positive,
                "closest_material_positive_runtime_distance_m": closest_positive,
            }
        validation[name] = {
            "groups": records,
            "passed": all(
                not row["material_false_negative_indices"]
                and not row["native_clear_runtime_false_positive_indices"]
                for row in records.values()
            ),
        }

    started = time.monotonic()
    for state in benchmark_states:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
    forward_seconds = time.monotonic() - started
    return validation, {
        "state_count": len(benchmark_states),
        "total_seconds": forward_seconds,
        "milliseconds_per_state": 1000 * forward_seconds / len(benchmark_states),
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scene, pair_names = _build_xml()
    # Compile the high-resolution version first. MuJoCo's in-process mesh
    # asset cache does not key the cached octree on octree_maxdepth in 3.13.
    model, compile_seconds = _compile(scene, True)
    validation, benchmark = _validate(model, pair_names)
    octree = {}
    octree_bytes = 0
    for mesh_name, depth in OBJECT_DEPTHS.items():
        mesh = model.mesh(mesh_name).id
        octree[mesh_name] = {
            "maxdepth": depth,
            "nodes": int(model.mesh_octnum[mesh]),
        }
    for field in ("oct_aabb", "oct_child", "oct_coeff", "oct_depth"):
        octree_bytes += int(getattr(model, field).nbytes)
    passed = all(row["passed"] for row in validation.values())
    report = {
        "status": "combined_external_sdf_geometry_passed" if passed
                  else "combined_external_sdf_geometry_failed",
        "mujoco_version": mujoco.__version__,
        "requires_programmatic_mjspec_compile": True,
        "object_octrees": octree,
        "combined_octree_bytes": octree_bytes,
        "compile_seconds": compile_seconds,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "compiled": {"ngeom": int(model.ngeom), "npair": int(model.npair)},
        "validation": validation,
        "forward_benchmark": benchmark,
        "passed": passed,
        "formal_scene_modified": False,
        "formal_mujoco_environment_modified": False,
        "capacity_validated": False,
        "physics_rollout_validated": False,
        "training_ready": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
