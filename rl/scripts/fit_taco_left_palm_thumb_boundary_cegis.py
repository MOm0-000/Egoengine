"""Boundary-guided CEGIS using only the existing palm/thumb candidate inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import fcl
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_left_palm_thumb_boundary_topology import (
    _boundary_comparison,
    _single_upper_boundary,
    _slice,
)
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts
from fit_taco_left_palm_thumb_convex_guard import ConvexParts, PALM_PARTS, THUMB_PARTS
from fit_taco_left_palm_thumb_hybrid_guard import _evaluate, _fit, _predict
from fit_taco_left_palm_thumb_semantic_guard import Oracle, _grid

DEFAULT_PROTOCOL = ROOT / "configs/taco_pour_left_palm_thumb_boundary_cegis_v1.yaml"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_left_palm_thumb_boundary_cegis_v1/report.json"


class SelectedCandidate:
    def __init__(self, parts: ConvexParts, convex: list[int], spheres: list[dict]):
        self.parts = parts
        self.convex = tuple(map(int, convex))
        self.spheres = tuple(spheres)
        self.request = fcl.CollisionRequest(num_max_contacts=1)
        self.thumb_indices = tuple(sorted({divmod(index, parts.nthumb)[1]
                                           for index in self.convex}))
        self.palm_centres = np.asarray(
            [row["palm_center_m"] for row in self.spheres], dtype=float
        ).reshape(-1, 3)
        self.thumb_centres = np.asarray(
            [row["thumb_center_m"] for row in self.spheres], dtype=float
        ).reshape(-1, 3)
        self.diameters = 2.0 * np.asarray(
            [row["sphere_radius_m"] for row in self.spheres], dtype=float
        )

    def evaluate_transform(self, rotation: np.ndarray, translation: np.ndarray) -> bool:
        transform = fcl.Transform(rotation, translation)
        for index in self.thumb_indices:
            self.parts.thumb[index].setTransform(transform)
        for flattened in self.convex:
            palm, thumb = divmod(flattened, self.parts.nthumb)
            result = fcl.CollisionResult()
            fcl.collide(
                self.parts.palm[palm], self.parts.thumb[thumb], self.request, result
            )
            if result.is_collision:
                return True
        if len(self.spheres):
            world = self.thumb_centres @ rotation.T + translation
            if np.any(np.linalg.norm(self.palm_centres - world, axis=1) < self.diameters):
                return True
        return False


def _boundary_samples(rows: list[dict], offsets: list[float], upper: float) -> list[tuple[float, float]]:
    samples = []
    for row in rows:
        bend = float(row["thumb_bend_rad"])
        intervals = row["native"]["collision_intervals_rad"]
        if not intervals:
            samples.append((bend, upper))
            continue
        boundary = float(intervals[0][0])
        for offset in offsets:
            for rota in (boundary - offset, boundary + offset):
                samples.append((bend, rota))
    return list(dict.fromkeys(samples))


def _labels_and_predictions(oracle: Oracle, candidate: SelectedCandidate, samples):
    labels, predicted = [], []
    for bend, rota in samples:
        native, rotation, translation, _ = oracle.evaluate(float(bend), float(rota))
        labels.append(native)
        predicted.append(candidate.evaluate_transform(rotation, translation))
    return np.asarray(labels, dtype=bool), np.asarray(predicted, dtype=bool)


def _classification(samples, labels, predicted) -> dict:
    fp = np.flatnonzero(predicted & ~labels)
    fn = np.flatnonzero(~predicted & labels)
    rows = lambda indices: [{
        "thumb_bend_rad": float(samples[index][0]),
        "thumb_rota1_rad": float(samples[index][1]),
    } for index in indices]
    return {
        "samples": len(samples),
        "native_collision": int(labels.sum()),
        "candidate_collision": int(predicted.sum()),
        "false_positive_count": len(fp),
        "false_negative_count": len(fn),
        "false_positives": rows(fp),
        "false_negatives": rows(fn),
    }


def _extract_rows(oracle: Oracle, candidate: SelectedCandidate | None,
                  bends: np.ndarray, rotas: np.ndarray,
                  tolerance: float, max_iterations: int) -> list[dict]:
    rows = []
    for bend_index, bend in enumerate(bends):
        native_labels = np.empty(len(rotas), dtype=bool)
        candidate_labels = np.empty(len(rotas), dtype=bool) if candidate else None
        for index, rota in enumerate(rotas):
            native, rotation, translation, _ = oracle.evaluate(float(bend), float(rota))
            native_labels[index] = native
            if candidate:
                candidate_labels[index] = candidate.evaluate_transform(rotation, translation)
        native_predicate = lambda rota, bend=float(bend): oracle.evaluate(bend, rota)[0]
        row = {
            "bend_slice_index": bend_index,
            "thumb_bend_rad": float(bend),
            "native": _slice(native_predicate, rotas, native_labels,
                             tolerance, max_iterations),
        }
        if candidate:
            def candidate_predicate(rota, bend=float(bend)):
                rotation, translation = oracle.transform(bend, rota)
                return candidate.evaluate_transform(rotation, translation)
            row["hybrid"] = _slice(
                candidate_predicate, rotas, candidate_labels,
                tolerance, max_iterations,
            )
        rows.append(row)
        if bend_index % 32 == 0 or bend_index == len(bends) - 1:
            print(f"boundary extraction: {bend_index + 1}/{len(bends)}", flush=True)
    return rows


def _boundary_summary(rows: list[dict], upper: float) -> dict:
    mismatches = []
    for row in rows:
        native, candidate = row["native"], row["hybrid"]
        if (
            native["collision_interval_count"] != candidate["collision_interval_count"]
            or native["lower_endpoint_collision"] != candidate["lower_endpoint_collision"]
            or native["upper_endpoint_collision"] != candidate["upper_endpoint_collision"]
        ):
            mismatches.append(row["bend_slice_index"])
    return {
        "topology_mismatch_slice_count": len(mismatches),
        "topology_mismatch_slice_indices": mismatches,
        "boundary": _boundary_comparison(rows, upper),
    }


def _fit_candidate(oracle: Oracle, parts: ConvexParts, samples):
    labels, active, transforms, contacts = _evaluate(
        oracle, parts, samples, contacts=64
    )
    convex, spheres, uncovered, inventory = _fit(
        labels, active, transforms, contacts
    )
    candidate = SelectedCandidate(parts, convex, spheres)
    calibration = _classification(
        samples, labels, _predict(convex, spheres, active, transforms)
    )
    return candidate, {
        "sample_count": len(samples),
        "native_collision_count": int(labels.sum()),
        "uncovered_positive_count": uncovered,
        **inventory,
        "selected_convex_pairs": len(convex),
        "selected_sphere_pairs": len(spheres),
        "calibration_classification": calibration,
    }


def run(protocol_path: Path, output_path: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    verify_artifacts(protocol["inputs"].values())
    scene = Path(protocol["inputs"]["scene"]["path"])
    reference = Path(protocol["inputs"]["robot_reference"]["path"])
    topology = json.loads(Path(protocol["inputs"]["topology_v1"]["path"]).read_text())
    oracle = Oracle(scene, reference)
    parts = ConvexParts(PALM_PARTS, THUMB_PARTS)

    sampling = protocol["sampling"]
    offsets = sorted(map(float, sampling["boundary_probe_offsets_rad"]))
    rota_lo, rota_hi = -1.05, 1.57
    rotas = np.linspace(rota_lo, rota_hi, int(sampling["rota1_coarse_samples"]))
    tolerance = float(sampling["transition_numerical_tolerance_rad"])
    max_iterations = int(sampling["transition_max_bisections"])
    primary_bends = np.linspace(0.0, 1.83, 257)

    base, *_ = _grid(oracle)
    base.extend(_boundary_samples(topology["slice_results"], offsets, rota_hi))
    base.extend((row["thumb_bend_rad"], row["thumb_rota1_rad"])
                for row in topology["known_counterexamples"])
    base = list(dict.fromkeys(base))
    first, first_fit = _fit_candidate(oracle, parts, base)

    tuning_fraction = float(sampling["tuning_bend_fraction"])
    tuning_bends = primary_bends[:-1] + tuning_fraction * np.diff(primary_bends)
    tuning_rows = _extract_rows(
        oracle, first, tuning_bends, rotas, tolerance, max_iterations
    )
    tuning_samples = _boundary_samples(tuning_rows, offsets, rota_hi)
    tuning_labels, tuning_predicted = _labels_and_predictions(
        oracle, first, tuning_samples
    )
    tuning_before = _classification(tuning_samples, tuning_labels, tuning_predicted)
    counterexamples = [
        sample for sample, native, predicted in zip(
            tuning_samples, tuning_labels, tuning_predicted
        ) if native != predicted
    ]
    refined_samples = list(dict.fromkeys(base + counterexamples))
    final_candidate, refined_fit = _fit_candidate(oracle, parts, refined_samples)

    tuning_labels, tuning_predicted = _labels_and_predictions(
        oracle, final_candidate, tuning_samples
    )
    tuning_after = _classification(tuning_samples, tuning_labels, tuning_predicted)

    final_fraction = float(sampling["final_holdout_bend_fraction"])
    final_bends = primary_bends[:-1] + final_fraction * np.diff(primary_bends)
    final_rows = _extract_rows(
        oracle, final_candidate, final_bends, rotas, tolerance, max_iterations
    )
    final_boundary = _boundary_summary(final_rows, rota_hi)
    final_samples = _boundary_samples(final_rows, offsets, rota_hi)
    final_labels, final_predicted = _labels_and_predictions(
        oracle, final_candidate, final_samples
    )
    final_classification = _classification(
        final_samples, final_labels, final_predicted
    )

    acceptance = protocol["local_diagnostic_acceptance"]
    boundary = final_boundary["boundary"]
    passed = (
        not refined_fit["uncovered_positive_count"]
        and final_classification["false_positive_count"]
            == int(acceptance["final_holdout_sampled_false_positives"])
        and final_classification["false_negative_count"]
            == int(acceptance["final_holdout_sampled_false_negatives"])
        and final_boundary["topology_mismatch_slice_count"]
            == int(acceptance["final_holdout_topology_mismatch_slices"])
        and boundary.get("maximum_absolute_error_rad", np.inf)
            <= float(acceptance["final_holdout_max_absolute_boundary_error_rad"])
    )
    report = {
        "status": (
            "existing_inventory_cegis_candidate_passed_local_diagnostic_not_promoted"
            if passed else
            "existing_inventory_cegis_rejected_local_decomposition_required"
        ),
        "scope": protocol["scope"],
        "protocol": artifact(protocol_path),
        "inputs": protocol["inputs"],
        "first_fit": first_fit,
        "tuning": {
            "bend_fraction": tuning_fraction,
            "counterexamples_added": len(counterexamples),
            "before_refit": tuning_before,
            "after_refit": tuning_after,
        },
        "refined_fit": refined_fit,
        "selected_candidate": {
            "convex_pairs": [parts.names(index) for index in final_candidate.convex],
            "sphere_pairs": list(final_candidate.spheres),
            "total_pair_count": len(final_candidate.convex) + len(final_candidate.spheres),
        },
        "final_holdout": {
            "bend_fraction": final_fraction,
            "never_entered_fit": True,
            "classification": final_classification,
            **final_boundary,
        },
        "local_diagnostic_acceptance": acceptance,
        "passed_local_diagnostic": passed,
        "decision": {
            "guard_promoted": False,
            "scene_xml_emitted": False,
            "formal_scene_modified": False,
            "reference_modified": False,
            "mink_retarget_run": False,
            "capacity_run": False,
            "initialization_run": False,
            "training_run": False,
            "on_rejection": protocol["on_rejection"],
        },
        "code": artifact(Path(__file__)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "first_fit": first_fit,
        "tuning_counterexamples": len(counterexamples),
        "refined_fit": refined_fit,
        "final_classification": final_classification,
        "final_boundary": final_boundary,
    }, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
