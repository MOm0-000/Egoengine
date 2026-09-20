"""Fit a sparse hybrid convex/sphere guard for left palm versus thumb root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from egoengine_repro.retarget.paper_audit import artifact
from fit_taco_left_palm_thumb_convex_guard import ConvexParts, PALM_PARTS, THUMB_PARTS
from fit_taco_left_palm_thumb_semantic_guard import (
    Oracle, SCENE, REFERENCE, _candidate_centres, _distances, _grid,
)

OUTPUT = ROOT / "TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard/left_palm_thumb_hybrid_guard_fit.json"


def _evaluate(oracle, parts, samples, contacts=0):
    labels, active, transforms, points = [], [], [], []
    for index, (bend, rota) in enumerate(samples):
        native, rotation, translation, contact = oracle.evaluate(bend, rota, contacts)
        labels.append(native)
        transforms.append((rotation, translation))
        points.append(contact)
        active.append(parts.active_pairs(rotation, translation))
        if index and index % 1000 == 0:
            print(f"hybrid oracle: {index}/{len(samples)}", flush=True)
    return np.asarray(labels, dtype=bool), active, transforms, points


def _fit(labels, active, transforms, points):
    positive_rows = np.flatnonzero(labels)
    negative_pairs = set().union(*(active[index] for index in np.flatnonzero(~labels)))
    convex = sorted(set().union(*(active[index] for index in positive_rows)) - negative_pairs)

    centres = _candidate_centres(labels, transforms, points)
    distances = _distances(centres, transforms)
    diameters = distances[:, ~labels].min(axis=1)
    sphere_valid = diameters > 0.0
    centres = [value for value, keep in zip(centres, sphere_valid) if keep]
    distances = distances[sphere_valid]
    diameters = diameters[sphere_valid]

    candidates = []
    for pair in convex:
        candidates.append(("convex", pair, np.asarray([
            pair in active[index] for index in positive_rows
        ], dtype=bool)))
    for index, coverage in enumerate(distances[:, labels] < diameters[:, None]):
        candidates.append(("sphere", index, coverage))

    uncovered = np.ones(len(positive_rows), dtype=bool)
    selected = []
    while uncovered.any():
        gains = [int(np.count_nonzero(row[2] & uncovered)) for row in candidates]
        best = int(np.argmax(gains)) if gains else -1
        if best < 0 or gains[best] == 0:
            break
        selected.append(candidates[best])
        uncovered &= ~candidates[best][2]
        candidates.pop(best)

    selected_convex = [row[1] for row in selected if row[0] == "convex"]
    selected_spheres = []
    for kind, index, _coverage in selected:
        if kind != "sphere":
            continue
        selected_spheres.append({
            "palm_center_m": centres[index][0].tolist(),
            "thumb_center_m": centres[index][1].tolist(),
            "sphere_radius_m": float(diameters[index] / 2.0),
        })
    return selected_convex, selected_spheres, int(uncovered.sum()), {
        "safe_convex_candidates": len(convex),
        "safe_sphere_candidates": len(centres),
    }


def _predict(convex, spheres, active, transforms):
    predicted = np.asarray([bool(set(convex) & row) for row in active])
    for guard in spheres:
        palm = np.asarray(guard["palm_center_m"])
        thumb = np.asarray(guard["thumb_center_m"])
        diameter = 2.0 * guard["sphere_radius_m"]
        predicted |= np.asarray([
            np.linalg.norm(palm - (translation + rotation @ thumb)) < diameter
            for rotation, translation in transforms
        ])
    return predicted


def _classification(samples, labels, predicted):
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
    }


def run(output: Path):
    oracle = Oracle(SCENE, REFERENCE)
    parts = ConvexParts(PALM_PARTS, THUMB_PARTS)
    calibration, midpoint, third, golden_a, golden_b = _grid(oracle)
    labels, active, transforms, points = _evaluate(
        oracle, parts, calibration, contacts=64
    )
    convex, spheres, uncovered, inventory = _fit(labels, active, transforms, points)

    tuning = {}
    counterexamples = []
    if not uncovered:
        for name, samples in (("midpoint_tuning", midpoint), ("one_third_tuning", third)):
            gl, ga, gt, _ = _evaluate(oracle, parts, samples)
            predicted = _predict(convex, spheres, ga, gt)
            tuning[name] = _classification(samples, gl, predicted)
            counterexamples.extend(
                sample for sample, native, guard in zip(samples, gl, predicted)
                if native != guard
            )
    if counterexamples:
        calibration.extend(counterexamples)
        calibration = list(dict.fromkeys(calibration))
        labels, active, transforms, points = _evaluate(
            oracle, parts, calibration, contacts=64
        )
        convex, spheres, uncovered, inventory = _fit(labels, active, transforms, points)

    second_tuning = {}
    second_counterexamples = []
    if not uncovered:
        for name, samples in (
            ("golden_offset_tuning_a", golden_a),
            ("golden_offset_tuning_b", golden_b),
        ):
            gl, ga, gt, _ = _evaluate(oracle, parts, samples)
            predicted = _predict(convex, spheres, ga, gt)
            second_tuning[name] = _classification(samples, gl, predicted)
            second_counterexamples.extend(
                sample for sample, native, guard in zip(samples, gl, predicted)
                if native != guard
            )
    if second_counterexamples:
        calibration.extend(second_counterexamples)
        calibration = list(dict.fromkeys(calibration))
        labels, active, transforms, points = _evaluate(
            oracle, parts, calibration, contacts=64
        )
        convex, spheres, uncovered, inventory = _fit(labels, active, transforms, points)

    bends = np.linspace(0.0, 1.83, 65)
    rotas = np.linspace(-1.05, 1.57, 97)
    def offset_grid(alpha, beta):
        return [
            (float(bends[i] + alpha * (bends[i + 1] - bends[i])),
             float(rotas[j] + beta * (rotas[j + 1] - rotas[j])))
            for i in range(len(bends) - 1)
            for j in range(len(rotas) - 1)
        ]
    final_a = offset_grid(np.sqrt(5.0) - 2.0, np.pi - 3.0)
    final_b = offset_grid(np.pi - 3.0, np.sqrt(5.0) - 2.0)

    validation = {}
    if not uncovered:
        validation["calibration_after_counterexamples"] = _classification(
            calibration, labels, _predict(convex, spheres, active, transforms)
        )
        for name, samples in (
            ("final_offset_holdout_a", final_a),
            ("final_offset_holdout_b", final_b),
        ):
            gl, ga, gt, _ = _evaluate(oracle, parts, samples)
            validation[name] = _classification(
                samples, gl, _predict(convex, spheres, ga, gt)
            )
        reference = list(zip(
            oracle.reference_qpos[:41, oracle.bend_address],
            oracle.reference_qpos[:41, oracle.rota_address],
        ))
        gl, ga, gt, _ = _evaluate(oracle, parts, reference)
        validation["reference_endpoints_0_40"] = _classification(
            reference, gl, _predict(convex, spheres, ga, gt)
        )
    passed = not uncovered and all(
        not row["false_positive_count"] and not row["false_negative_count"]
        for row in validation.values()
    )
    report = {
        "status": "candidate_hybrid_guard_fit_passed" if passed else "candidate_hybrid_guard_fit_rejected",
        "scope": "offline_left_palm_thumb1_pair_specific_hybrid_guard_fit",
        "scene": artifact(SCENE),
        "reference": artifact(REFERENCE),
        "decompositions": {
            "palm": artifact(PALM_PARTS / "provenance.json"),
            "thumb": artifact(THUMB_PARTS / "provenance.json"),
            "requested_threshold_certified": False,
        },
        "fit": {
            **inventory,
            "selected_convex_pairs": [parts.names(pair) for pair in convex],
            "selected_sphere_pairs": spheres,
            "uncovered_calibration_positive_count": uncovered,
            "counterexample_tuning": tuning,
            "counterexamples_added": len(counterexamples),
            "second_counterexample_tuning": second_tuning,
            "second_counterexamples_added": len(second_counterexamples),
        },
        "validation": validation,
        "formal_scene_modified": False,
        "code": artifact(Path(__file__)),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], "convex", len(convex), "sphere", len(spheres), flush=True)
    for name, row in validation.items():
        print(name, "FP", row["false_positive_count"], "FN", row["false_negative_count"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
