"""Fit a sparse palm/thumb self guard from local CAD convex parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import fcl
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization_preflight import triangle_object
from egoengine_repro.retarget.paper_audit import artifact
from fit_taco_left_palm_thumb_semantic_guard import Oracle, SCENE, REFERENCE, _grid

PALM_PARTS = ROOT / "runs/taco_pour_collision_repair/assets/left_palm_assembly"
REJECTED = ROOT / "TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard"
THUMB_PARTS = REJECTED / "assets/left_thumb_root_local"
OUTPUT = REJECTED / "left_palm_thumb_convex_guard_fit.json"


def _load_parts(directory: Path):
    paths = sorted(directory.glob("*.obj"), key=lambda path: int(path.stem))
    meshes = [trimesh.load_mesh(path, process=False) for path in paths]
    objects = [triangle_object(mesh) for mesh in meshes]
    corners = []
    for mesh in meshes:
        lo, hi = mesh.bounds
        corners.append(np.asarray([
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]))
    return paths, objects, np.asarray(corners)


class ConvexParts:
    def __init__(self, palm_dir: Path, thumb_dir: Path):
        self.palm_paths, self.palm, palm_corners = _load_parts(palm_dir)
        self.thumb_paths, self.thumb, self.thumb_corners = _load_parts(thumb_dir)
        self.palm_min = palm_corners.min(axis=1)
        self.palm_max = palm_corners.max(axis=1)
        self.nthumb = len(self.thumb)

    def active_pairs(self, rotation, translation):
        world = self.thumb_corners @ rotation.T + translation
        thumb_min, thumb_max = world.min(axis=1), world.max(axis=1)
        overlap = np.all(
            (self.palm_min[:, None, :] <= thumb_max[None, :, :])
            & (thumb_min[None, :, :] <= self.palm_max[:, None, :]),
            axis=2,
        )
        for obj in self.thumb:
            obj.setTransform(fcl.Transform(rotation, translation))
        active = set()
        request = fcl.CollisionRequest(num_max_contacts=1)
        for palm, thumb in np.argwhere(overlap):
            result = fcl.CollisionResult()
            fcl.collide(self.palm[int(palm)], self.thumb[int(thumb)], request, result)
            if result.is_collision:
                active.add(int(palm) * self.nthumb + int(thumb))
        return active

    def names(self, flattened):
        palm, thumb = divmod(int(flattened), self.nthumb)
        return {
            "palm_part": palm,
            "thumb_part": thumb,
            "palm_asset": artifact(self.palm_paths[palm]),
            "thumb_asset": artifact(self.thumb_paths[thumb]),
        }


def _evaluate(oracle: Oracle, parts: ConvexParts, samples):
    labels = []
    active = []
    for index, (bend, rota) in enumerate(samples):
        native, rotation, translation, _ = oracle.evaluate(bend, rota)
        labels.append(native)
        active.append(parts.active_pairs(rotation, translation))
        if index and index % 1000 == 0:
            print(f"convex union: {index}/{len(samples)}", flush=True)
    return np.asarray(labels, dtype=bool), active


def _fit(labels, active):
    positive = np.flatnonzero(labels)
    negative_pairs = set().union(*(active[index] for index in np.flatnonzero(~labels)))
    candidates = set().union(*(active[index] for index in positive)) - negative_pairs
    uncovered = set(map(int, positive))
    chosen = []
    while uncovered:
        best = max(
            candidates,
            key=lambda pair: sum(pair in active[index] for index in uncovered),
            default=None,
        )
        covered = {index for index in uncovered if best is not None and best in active[index]}
        if not covered:
            return chosen, sorted(uncovered), len(candidates)
        chosen.append(best)
        uncovered -= covered
        candidates.remove(best)
    return chosen, [], len(candidates) + len(chosen)


def _classification(samples, labels, active, chosen):
    predicted = np.asarray([bool(set(chosen) & pairs) for pairs in active])
    fp = np.flatnonzero(predicted & ~labels)
    fn = np.flatnonzero(~predicted & labels)
    rows = lambda indices: [
        {"bend_rad": float(samples[index][0]), "rota1_rad": float(samples[index][1])}
        for index in indices
    ]
    return {
        "samples": len(samples),
        "native_collision": int(labels.sum()),
        "guard_collision": int(predicted.sum()),
        "false_positive_count": len(fp),
        "false_negative_count": len(fn),
        "false_positives": rows(fp),
        "false_negatives": rows(fn),
    }, predicted


def run(output: Path):
    oracle = Oracle(SCENE, REFERENCE)
    parts = ConvexParts(PALM_PARTS, THUMB_PARTS)
    calibration, midpoint, third, golden_a, golden_b = _grid(oracle)
    labels, active = _evaluate(oracle, parts, calibration)
    chosen, uncovered, candidates = _fit(labels, active)
    tuning = {}
    counterexamples = []
    if not uncovered:
        for name, samples in (("midpoint_tuning", midpoint), ("one_third_tuning", third)):
            grid_labels, grid_active = _evaluate(oracle, parts, samples)
            row, predicted = _classification(samples, grid_labels, grid_active, chosen)
            tuning[name] = row
            counterexamples.extend(
                sample for sample, native, guard in zip(samples, grid_labels, predicted)
                if native != guard
            )
    if counterexamples:
        calibration.extend(counterexamples)
        calibration = list(dict.fromkeys(calibration))
        labels, active = _evaluate(oracle, parts, calibration)
        chosen, uncovered, candidates = _fit(labels, active)

    validation = {}
    if not uncovered:
        validation["calibration_after_counterexamples"], _ = _classification(
            calibration, labels, active, chosen
        )
        for name, samples in (
            ("golden_offset_holdout_a", golden_a),
            ("golden_offset_holdout_b", golden_b),
        ):
            grid_labels, grid_active = _evaluate(oracle, parts, samples)
            validation[name], _ = _classification(
                samples, grid_labels, grid_active, chosen
            )
        reference = list(zip(
            oracle.reference_qpos[:41, oracle.bend_address],
            oracle.reference_qpos[:41, oracle.rota_address],
        ))
        grid_labels, grid_active = _evaluate(oracle, parts, reference)
        validation["reference_endpoints_0_40"], _ = _classification(
            reference, grid_labels, grid_active, chosen
        )
    passed = not uncovered and all(
        not row["false_positive_count"] and not row["false_negative_count"]
        for row in validation.values()
    )
    report = {
        "status": "candidate_convex_guard_fit_passed" if passed else "candidate_convex_guard_fit_rejected",
        "scope": "offline_left_palm_thumb1_pair_specific_convex_guard_fit",
        "scene": artifact(SCENE),
        "reference": artifact(REFERENCE),
        "decompositions": {
            "palm": artifact(PALM_PARTS / "provenance.json"),
            "thumb": artifact(THUMB_PARTS / "provenance.json"),
            "palm_parts": len(parts.palm),
            "thumb_parts": len(parts.thumb),
            "requested_threshold_certified": False,
        },
        "fit": {
            "safe_candidate_part_pairs": candidates,
            "selected_part_pairs": [parts.names(pair) for pair in chosen],
            "uncovered_calibration_positive_count": len(uncovered),
            "uncovered_calibration_positives": [
                {"bend_rad": float(calibration[index][0]),
                 "rota1_rad": float(calibration[index][1]),
                 "decomposed_union_collision": bool(active[index])}
                for index in uncovered
            ],
            "counterexample_tuning": tuning,
            "counterexamples_added": len(counterexamples),
        },
        "validation": validation,
        "formal_scene_modified": False,
        "code": artifact(Path(__file__)),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], "selected", len(chosen), "uncovered", len(uncovered), flush=True)
    for name, row in validation.items():
        print(name, "FP", row["false_positive_count"], "FN", row["false_negative_count"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
