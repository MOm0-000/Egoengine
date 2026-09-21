#!/usr/bin/env python3
"""Probe one exact hand-mesh/object-SDF contact pair without changing formal assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

SOURCE = ROOT / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml"
REFERENCE = ROOT / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz"
ENDPOINT_LABELS = ROOT / "runs/taco_pour_collision_semantics_full_horizon_v1/report.json"
MIDPOINT_LABELS = ROOT / "TRASH/rejected_candidates/2026-09-21_external_ellipsoids_endpoint_fit_midpoint_failed/midpoint_holdout.json"
QUARTER_LABELS = ROOT / "TRASH/rejected_candidates/2026-09-21_external_ellipsoid_union3_fraction_holdout_failed/fraction_holdout.json"
EIGHTH_LABELS = (
    ROOT / "TRASH/rejected_candidates/"
    "2026-09-21_external_ellipsoid_all_fraction_v2_eighth_failed/"
    "eighth_holdout.json"
)
SIXTEENTH_LABELS = (
    ROOT / "runs/taco_pour_collision_semantics_full_horizon_v1/"
    "sixteenth_holdout.json"
)
OUTPUT = ROOT / "runs/taco_pour_external_sdf_probe_v1"

SPECS = {
    "left_pinky": {
        "pair": ("left_pinky_link2_visual", "left_object_visual"),
        "hand_body": "left_hand_pinky_link2",
        "object_body": "left_object",
        "hand_mesh": "left_hand_pinky_link2",
        "object_mesh": "left_visual",
    },
    "right_index": {
        "pair": ("right_index_rota2_visual", "right_object_visual"),
        "hand_body": "right_hand_index_rota_link2",
        "object_body": "right_object",
        "hand_mesh": "right_hand_index_rota_link2",
        "object_mesh": "right_visual",
    },
    "right_thumb": {
        "pair": ("right_thumb_rota2_visual", "right_object_visual"),
        "hand_body": "right_hand_thumb_rota_link2",
        "object_body": "right_object",
        "hand_mesh": "right_hand_thumb_rota_link2",
        "object_mesh": "right_visual",
    },
    "left_ring": {
        "pair": ("left_ring_link2_visual", "left_object_visual"),
        "hand_body": "left_hand_ring_link2",
        "object_body": "left_object",
        "hand_mesh": "left_hand_ring_link2",
        "object_mesh": "left_visual",
    },
}


def _fraction_states(model, endpoints, fractions):
    velocity = np.empty(model.nv, dtype=float)
    result = []
    for first, second in zip(endpoints[:-1], endpoints[1:]):
        mujoco.mj_differentiatePos(model, velocity, 1.0, first, second)
        for fraction in fractions:
            state = first.copy()
            mujoco.mj_integratePos(model, state, velocity, fraction)
            result.append(state)
    return np.asarray(result)


def _body_geom_names(root: ET.Element) -> dict[str, set[str]]:
    return {
        body.get("name", ""): {
            geom.get("name", "") for geom in body.findall("geom")
        }
        for body in root.findall(".//body")
    }


def _build(spec: dict[str, object], output: Path, mode: str,
           hand_subdivisions: int, octree_depth: int | None, sdf_iterations: int,
           sdf_initpoints: int) -> Path:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    option = root.find("option")
    assets = root.find("asset")
    contact = root.find("contact")
    if assets is None or contact is None:
        raise ValueError("source scene lacks asset or contact block")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        root.insert(list(root).index(compiler) + 1 if compiler is not None else 0, option)
    option.set("sdf_iterations", str(sdf_iterations))
    option.set("sdf_initpoints", str(sdf_initpoints))
    output.mkdir(parents=True, exist_ok=False)

    bodies = {
        body.get("name", ""): body for body in root.findall(".//body")
    }
    hand_body = bodies[str(spec["hand_body"])]
    object_body = bodies[str(spec["object_body"])]
    names = _body_geom_names(root)
    hand_names = names[str(spec["hand_body"])]
    object_names = names[str(spec["object_body"])]
    removed = 0
    for pair in list(contact.findall("pair")):
        ends = {pair.get("geom1", ""), pair.get("geom2", "")}
        if ends & hand_names and ends & object_names:
            contact.remove(pair)
            removed += 1
    if not removed:
        raise ValueError("target body pair has no source runtime contact")

    for geom in list(hand_body.findall("geom")):
        if "_external_semantic_" in geom.get("name", ""):
            hand_body.remove(geom)

    tag = str(spec["hand_body"]).replace("_hand_", "_")
    hand_geom = f"collision_hand_{tag}_external_exact_mesh"
    object_geom = f"{spec['object_body']}_{tag}_external_exact_sdf"
    hand_mesh_name = str(spec["hand_mesh"])
    if hand_subdivisions:
        compiler = root.find("compiler")
        meshdir = Path(compiler.get("meshdir", ".")).resolve()
        source_asset = next(
            row for row in assets.findall("mesh")
            if row.get("name") == hand_mesh_name
        )
        mesh = trimesh.load_mesh(
            meshdir / source_asset.get("file"), process=True,
        )
        components = [
            part for part in mesh.split(only_watertight=False)
            if part.is_volume and part.volume > 1.0e-10
        ]
        if not components:
            raise ValueError("hand mesh has no closed positive-volume component")
        mesh = max(components, key=lambda part: part.volume)
        for _ in range(hand_subdivisions):
            mesh = mesh.subdivide()
        hand_path = output / f"hand_main_subdivided_{hand_subdivisions}.obj"
        mesh.export(hand_path)
        hand_mesh_name = f"{tag}_external_exact_subdivided_mesh"
        ET.SubElement(
            assets, "mesh", name=hand_mesh_name,
            file=os.path.relpath(hand_path.resolve(), meshdir),
        )
    ET.SubElement(
        hand_body, "geom", name=hand_geom,
        type="sdf" if mode == "both_sdf" else "mesh",
        mesh=hand_mesh_name, group="3", density="0",
        contype="0", conaffinity="0", condim="3", friction="1 0.1 0",
        rgba="0.2 0.8 0.4 0.25",
    )
    ET.SubElement(
        object_body, "geom", name=object_geom, type="sdf",
        mesh=str(spec["object_mesh"]), group="3", density="0",
        contype="0", conaffinity="0", condim="3", friction="1 0.1 0",
        rgba="0.9 0.5 0.1 0.15",
    )
    ET.SubElement(
        contact, "pair", name=f"semantic_sdf_{tag}", geom1=hand_geom,
        geom2=object_geom, condim="3", friction="1 1 0.1 0 0",
    )
    scene = output / "candidate.xml"
    ET.indent(tree, space="  ")
    tree.write(scene, encoding="unicode")
    return scene


def _label_rows(path: Path, pair: tuple[str, str]) -> list[dict[str, object]]:
    report = json.loads(path.read_text())
    key = "hand_object" if "hand_object" in report else "pairs"
    return next(row["rows"] for row in report[key] if tuple(row["geoms"]) == pair)


def _evaluate(model: mujoco.MjModel, states: np.ndarray,
              labels: list[dict[str, object]], hand_geom: int,
              object_geom: int) -> dict[str, object]:
    if len(states) != len(labels):
        raise ValueError(f"state/label mismatch: {len(states)} != {len(labels)}")
    data = mujoco.MjData(model)
    rows = []
    for index, (state, label) in enumerate(zip(states, labels)):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        distance = float(mujoco.mj_geomDistance(
            model, data, hand_geom, object_geom, 1.0, None,
        ))
        collision = distance <= 0.0
        material = bool(label["native_material_interference_over_50um"])
        surface = bool(label["native_surface_crossing"])
        rows.append({
            "index": index,
            "native_surface_crossing": surface,
            "native_material_interference_over_50um": material,
            "runtime_distance_m": distance,
            "runtime_collision": collision,
            "material_false_negative": material and not collision,
            "native_clear_runtime_false_positive": (
                not surface and not material and collision
            ),
        })
    return {
        "checks": len(rows),
        "material_false_negative_indices": [
            row["index"] for row in rows if row["material_false_negative"]
        ],
        "native_clear_runtime_false_positive_indices": [
            row["index"] for row in rows
            if row["native_clear_runtime_false_positive"]
        ],
        "minimum_runtime_distance_m": min(row["runtime_distance_m"] for row in rows),
        "maximum_runtime_distance_m": max(row["runtime_distance_m"] for row in rows),
        "rows": rows,
    }


def run(name: str, mode: str, hand_subdivisions: int,
        octree_depth: int | None, sdf_iterations: int,
        sdf_initpoints: int) -> dict[str, object]:
    output = OUTPUT / (
        f"{name}_{mode}_subdiv{hand_subdivisions}_oct"
        f"{octree_depth if octree_depth is not None else 'default'}_"
        f"{sdf_iterations}x{sdf_initpoints}"
    )
    if output.exists():
        raise FileExistsError(output)
    spec = SPECS[name]
    started = time.monotonic()
    scene = _build(
        spec, output, mode, hand_subdivisions, octree_depth,
        sdf_iterations, sdf_initpoints,
    )
    compile_started = time.monotonic()
    if octree_depth is None:
        model = mujoco.MjModel.from_xml_path(str(scene))
    else:
        mj_spec = mujoco.MjSpec.from_file(str(scene))
        object_mesh = mj_spec.mesh(str(spec["object_mesh"]))
        object_mesh.needsdf = True
        if not hasattr(object_mesh, "octree_maxdepth"):
            raise RuntimeError(
                "installed MuJoCo does not expose mesh.octree_maxdepth"
            )
        object_mesh.octree_maxdepth = octree_depth
        model = mj_spec.compile()
    compile_seconds = time.monotonic() - compile_started
    hand_geom = model.geom(f"collision_hand_{str(spec['hand_body']).replace('_hand_', '_')}_external_exact_mesh").id
    object_geom = model.geom(
        f"{spec['object_body']}_{str(spec['hand_body']).replace('_hand_', '_')}_external_exact_sdf"
    ).id
    with np.load(REFERENCE, allow_pickle=False) as source:
        endpoints = np.asarray(source["qpos"], dtype=float)
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE))
    evaluations = {
        "endpoints": _evaluate(
            model, endpoints, _label_rows(ENDPOINT_LABELS, spec["pair"]),
            hand_geom, object_geom,
        ),
        "midpoints": _evaluate(
            model, _fraction_states(source_model, endpoints, (0.5,)),
            _label_rows(MIDPOINT_LABELS, spec["pair"]), hand_geom, object_geom,
        ),
        "quarters": _evaluate(
            model, _fraction_states(source_model, endpoints, (0.25, 0.75)),
            _label_rows(QUARTER_LABELS, spec["pair"]), hand_geom, object_geom,
        ),
        "eighths": _evaluate(
            model, _fraction_states(
                source_model, endpoints, (0.125, 0.375, 0.625, 0.875),
            ),
            _label_rows(EIGHTH_LABELS, spec["pair"]), hand_geom, object_geom,
        ),
        "sixteenths": _evaluate(
            model, _fraction_states(
                source_model, endpoints,
                tuple(index / 16 for index in range(1, 16, 2)),
            ),
            _label_rows(SIXTEENTH_LABELS, spec["pair"]), hand_geom, object_geom,
        ),
    }
    passed = all(
        not row["material_false_negative_indices"]
        and not row["native_clear_runtime_false_positive_indices"]
        for row in evaluations.values()
    )
    report = {
        "status": "external_exact_sdf_probe_passed" if passed else "external_exact_sdf_probe_failed",
        "spec": name,
        "pair": list(spec["pair"]),
        "representation": (
            "native hand and object mesh SDFs" if mode == "both_sdf"
            else "exact hand triangle mesh against native object mesh SDF"
        ),
        "mode": mode,
        "hand_surface_subdivisions": hand_subdivisions,
        "object_sdf_octree_maxdepth": octree_depth,
        "object_sdf_octree_nodes": int(
            model.mesh_octnum[model.mesh(str(spec["object_mesh"])).id]
        ),
        "sdf_iterations": sdf_iterations,
        "sdf_initpoints": sdf_initpoints,
        "compiled": {
            "ngeom": int(model.ngeom), "npair": int(model.npair),
            "compile_seconds": compile_seconds,
        },
        "evaluations": evaluations,
        "passed": passed,
        "formal_scene_modified": False,
        "mink_hand_object_limit_enabled": False,
        "training_ready": False,
        "total_seconds": time.monotonic() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "spec": name,
        "compile_seconds": compile_seconds,
        "errors": {
            key: {
                "fn": len(value["material_false_negative_indices"]),
                "fp": len(value["native_clear_runtime_false_positive_indices"]),
            }
            for key, value in evaluations.items()
        },
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", choices=sorted(SPECS))
    parser.add_argument("--mode", choices=("mesh_sdf", "both_sdf"),
                        default="mesh_sdf")
    parser.add_argument("--hand-subdivisions", type=int, choices=(0, 1, 2),
                        default=0)
    parser.add_argument("--octree-depth", type=int)
    parser.add_argument("--sdf-iterations", type=int, default=10)
    parser.add_argument("--sdf-initpoints", type=int, default=40)
    args = parser.parse_args()
    run(args.spec, args.mode, args.hand_subdivisions, args.octree_depth,
        args.sdf_iterations, args.sdf_initpoints)
