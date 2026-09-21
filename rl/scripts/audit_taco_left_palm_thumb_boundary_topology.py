"""Map native and rejected-Hybrid left palm/thumb collision boundaries.

This is a read-only diagnostic. It does not fit guards, edit MJCF, retarget,
simulate a reset, or train a policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import fcl
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization_preflight import triangle_object
from egoengine_repro.retarget.paper_audit import (
    artifact,
    resolve_artifact_path,
    verify_artifacts,
)
from fit_taco_left_palm_thumb_semantic_guard import Oracle

DEFAULT_PROTOCOL = ROOT / "configs/taco_pour_left_palm_thumb_boundary_topology_v1.yaml"
DEFAULT_OUTPUT = ROOT / "runs/taco_pour_left_palm_thumb_boundary_topology_v1/report.json"


class FrozenHybrid:
    """Evaluate only the 8 convex and 2 sphere pairs in the rejected report."""

    def __init__(self, report: dict):
        fit = report["fit"]
        self.pairs = []
        palm_objects = {}
        thumb_objects = {}
        for row in fit["selected_convex_pairs"]:
            palm_path = resolve_artifact_path(row["palm_asset"])
            thumb_path = resolve_artifact_path(row["thumb_asset"])
            palm_id, thumb_id = int(row["palm_part"]), int(row["thumb_part"])
            if palm_id not in palm_objects:
                palm_objects[palm_id] = triangle_object(
                    trimesh.load_mesh(palm_path, process=False)
                )
            if thumb_id not in thumb_objects:
                thumb_objects[thumb_id] = triangle_object(
                    trimesh.load_mesh(thumb_path, process=False)
                )
            self.pairs.append((palm_objects[palm_id], thumb_objects[thumb_id]))
        self.thumb_objects = tuple(thumb_objects.values())
        self.spheres = tuple(fit["selected_sphere_pairs"])
        self.request = fcl.CollisionRequest(num_max_contacts=1)

    def evaluate_transform(self, rotation: np.ndarray, translation: np.ndarray) -> bool:
        transform = fcl.Transform(rotation, translation)
        for obj in self.thumb_objects:
            obj.setTransform(transform)
        for palm, thumb in self.pairs:
            result = fcl.CollisionResult()
            fcl.collide(palm, thumb, self.request, result)
            if result.is_collision:
                return True
        for row in self.spheres:
            palm = np.asarray(row["palm_center_m"], dtype=float)
            thumb = np.asarray(row["thumb_center_m"], dtype=float)
            diameter = 2.0 * float(row["sphere_radius_m"])
            if np.linalg.norm(palm - (translation + rotation @ thumb)) < diameter:
                return True
        return False


def _bisect(predicate, lo: float, hi: float, left_label: bool,
            tolerance: float, max_iterations: int) -> dict:
    right_label = not left_label
    iterations = 0
    while hi - lo > tolerance and iterations < max_iterations:
        mid = (lo + hi) / 2.0
        if bool(predicate(mid)) == left_label:
            lo = mid
        else:
            hi = mid
        iterations += 1
    return {
        "left_bracket_rad": float(lo),
        "right_bracket_rad": float(hi),
        "left_collision": bool(left_label),
        "right_collision": bool(right_label),
        "estimate_rad": float((lo + hi) / 2.0),
        "numerical_bracket_width_rad": float(hi - lo),
        "bisections": iterations,
    }


def _intervals(labels: np.ndarray, transitions: list[dict], lower: float,
               upper: float) -> list[list[float]]:
    result = []
    start = lower if bool(labels[0]) else None
    for transition in transitions:
        boundary = transition["estimate_rad"]
        if not transition["left_collision"] and transition["right_collision"]:
            if start is not None:
                raise RuntimeError("clear-to-collision transition inside an open interval")
            start = boundary
        else:
            if start is None:
                raise RuntimeError("collision-to-clear transition without an open interval")
            result.append([float(start), float(boundary)])
            start = None
    if start is not None:
        result.append([float(start), float(upper)])
    return result


def _slice(predicate, coarse_rotas: np.ndarray, labels: np.ndarray,
           tolerance: float, max_iterations: int) -> dict:
    transitions = []
    for index in np.flatnonzero(labels[:-1] != labels[1:]):
        transitions.append(_bisect(
            predicate,
            float(coarse_rotas[index]),
            float(coarse_rotas[index + 1]),
            bool(labels[index]),
            tolerance,
            max_iterations,
        ))
    intervals = _intervals(
        labels, transitions, float(coarse_rotas[0]), float(coarse_rotas[-1])
    )
    return {
        "lower_endpoint_collision": bool(labels[0]),
        "upper_endpoint_collision": bool(labels[-1]),
        "coarse_collision_samples": int(labels.sum()),
        "transition_count": len(transitions),
        "collision_interval_count": len(intervals),
        "transitions": transitions,
        "collision_intervals_rad": intervals,
    }


def _histogram(rows: list[dict], key: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(row[key] for row in rows).items())}


def _collision_bend_bands(rows: list[dict], family: str) -> list[dict]:
    indices = [row["bend_slice_index"] for row in rows
               if row[family]["collision_interval_count"]]
    runs = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return [{
        "first_slice_index": run[0],
        "last_slice_index": run[-1],
        "sampled_bend_range_rad": [
            rows[run[0]]["thumb_bend_rad"], rows[run[-1]]["thumb_bend_rad"]
        ],
        "slice_count": len(run),
        "range_endpoints_are_sampled_not_bisected": True,
    } for run in runs]


def _single_upper_boundary(row: dict, upper: float) -> float | None:
    intervals = row["collision_intervals_rad"]
    if len(intervals) != 1 or not row["upper_endpoint_collision"]:
        return None
    if abs(intervals[0][1] - upper) > 1e-12:
        return None
    return float(intervals[0][0])


def _boundary_comparison(rows: list[dict], upper: float) -> dict:
    samples = []
    for row in rows:
        native = _single_upper_boundary(row["native"], upper)
        hybrid = _single_upper_boundary(row["hybrid"], upper)
        if native is None or hybrid is None:
            continue
        error = hybrid - native
        samples.append({
            "thumb_bend_rad": row["thumb_bend_rad"],
            "native_boundary_rad": native,
            "hybrid_boundary_rad": hybrid,
            "hybrid_minus_native_rad": error,
        })
    if not samples:
        return {"comparable_slice_count": 0}
    errors = np.asarray([row["hybrid_minus_native_rad"] for row in samples])
    absolute = np.abs(errors)
    worst = np.argsort(absolute)[-10:][::-1]
    return {
        "sign_convention": "negative means Hybrid starts collision too early (FP band); positive means too late (FN band)",
        "comparable_slice_count": len(samples),
        "negative_fp_band_slices": int(np.count_nonzero(errors < 0.0)),
        "positive_fn_band_slices": int(np.count_nonzero(errors > 0.0)),
        "exact_within_numerical_brackets_slices": int(np.count_nonzero(errors == 0.0)),
        "mean_signed_error_rad": float(errors.mean()),
        "rmse_rad": float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error_rad": float(np.median(absolute)),
        "p95_absolute_error_rad": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error_rad": float(absolute.max()),
        "worst_slices": [samples[int(index)] for index in worst],
    }


def _point_check(oracle: Oracle, hybrid: FrozenHybrid, row: dict) -> dict:
    bend = float(row["thumb_bend_rad"])
    rota = float(row["thumb_rota1_rad"])
    native, rotation, translation, _ = oracle.evaluate(bend, rota)
    predicted = hybrid.evaluate_transform(rotation, translation)
    return {
        **row,
        "native_collision": bool(native),
        "hybrid_collision": bool(predicted),
        "reproduced": bool(
            (row["kind"] == "false_negative" and native and not predicted)
            or (row["kind"] == "false_positive" and predicted and not native)
        ),
    }


def run(protocol_path: Path, output_path: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    verify_artifacts(protocol["inputs"].values())
    domain = protocol["domain"]
    bend_lo, bend_hi = map(float, domain["thumb_bend_rad"])
    rota_lo, rota_hi = map(float, domain["thumb_rota1_rad"])
    bends = np.linspace(bend_lo, bend_hi, int(domain["thumb_bend_slices"]))
    rotas = np.linspace(rota_lo, rota_hi, int(domain["thumb_rota1_coarse_samples"]))
    tolerance = float(domain["transition_numerical_tolerance_rad"])
    max_iterations = int(domain["transition_max_bisections"])

    scene = Path(protocol["inputs"]["scene"]["path"])
    reference = Path(protocol["inputs"]["robot_reference"]["path"])
    hybrid_report = json.loads(Path(protocol["inputs"]["rejected_hybrid"]["path"]).read_text())
    if hybrid_report["status"] != "candidate_hybrid_guard_fit_rejected":
        raise ValueError("boundary comparison requires the frozen rejected Hybrid")
    oracle = Oracle(scene, reference)
    hybrid = FrozenHybrid(hybrid_report)

    rows = []
    for bend_index, bend in enumerate(bends):
        native_labels = np.empty(len(rotas), dtype=bool)
        hybrid_labels = np.empty(len(rotas), dtype=bool)
        for index, rota in enumerate(rotas):
            native, rotation, translation, _ = oracle.evaluate(float(bend), float(rota))
            native_labels[index] = native
            hybrid_labels[index] = hybrid.evaluate_transform(rotation, translation)

        native_predicate = lambda rota, bend=float(bend): oracle.evaluate(bend, rota)[0]
        def hybrid_predicate(rota, bend=float(bend)):
            rotation, translation = oracle.transform(bend, rota)
            return hybrid.evaluate_transform(rotation, translation)

        rows.append({
            "bend_slice_index": bend_index,
            "thumb_bend_rad": float(bend),
            "native": _slice(
                native_predicate, rotas, native_labels, tolerance, max_iterations
            ),
            "hybrid": _slice(
                hybrid_predicate, rotas, hybrid_labels, tolerance, max_iterations
            ),
        })
        if bend_index % 16 == 0 or bend_index == len(bends) - 1:
            print(f"boundary slices: {bend_index + 1}/{len(bends)}", flush=True)

    native_multi = [row["bend_slice_index"] for row in rows
                    if row["native"]["collision_interval_count"] > 1]
    topology_mismatch = [row["bend_slice_index"] for row in rows if (
        row["native"]["collision_interval_count"] != row["hybrid"]["collision_interval_count"]
        or row["native"]["lower_endpoint_collision"] != row["hybrid"]["lower_endpoint_collision"]
        or row["native"]["upper_endpoint_collision"] != row["hybrid"]["upper_endpoint_collision"]
    )]
    native_all_simple = not native_multi and all(
        row["native"]["collision_interval_count"] == 0
        or _single_upper_boundary(row["native"], rota_hi) is not None
        for row in rows
    )
    recommendation = (
        "boundary_curve_cegis_allow_shrink_or_remove_fp_guards_then_add_local_fn_guard"
        if native_all_simple
        else "local_contact_region_decomposition_before_new_guard_fit"
    )
    report = {
        "status": "boundary_topology_mapped_guard_not_promoted",
        "scope": protocol["scope"],
        "protocol": artifact(protocol_path),
        "inputs": protocol["inputs"],
        "sampling": {
            "thumb_bend_slices": len(bends),
            "thumb_rota1_coarse_samples_per_slice": len(rotas),
            "coarse_evaluations_per_predicate": int(len(bends) * len(rotas)),
            "thumb_bend_step_rad": float(bends[1] - bends[0]),
            "thumb_rota1_coarse_step_rad": float(rotas[1] - rotas[0]),
            "transition_numerical_tolerance_rad": tolerance,
            "transition_max_bisections": max_iterations,
            "boundary_precision_note": "brackets describe numerical FCL transition localization, not physical CAD or manufacturing accuracy",
        },
        "topology": {
            "native_interval_count_histogram": _histogram([row["native"] for row in rows], "collision_interval_count"),
            "hybrid_interval_count_histogram": _histogram([row["hybrid"] for row in rows], "collision_interval_count"),
            "native_sampled_collision_bend_bands": _collision_bend_bands(rows, "native"),
            "hybrid_sampled_collision_bend_bands": _collision_bend_bands(rows, "hybrid"),
            "native_multi_interval_slice_indices": native_multi,
            "native_is_zero_or_one_upper_tail_interval_on_every_slice": native_all_simple,
            "native_hybrid_topology_mismatch_slice_count": len(topology_mismatch),
            "native_hybrid_topology_mismatch_slice_indices": topology_mismatch,
        },
        "boundary_comparison": _boundary_comparison(rows, rota_hi),
        "known_counterexamples": [
            _point_check(oracle, hybrid, row) for row in protocol["known_counterexamples"]
        ],
        "slice_results": rows,
        "decision": {
            "recommended_next_step": recommendation,
            "guard_promoted": False,
            "formal_scene_modified": False,
            "reference_modified": False,
            "mink_retarget_run": False,
            "capacity_run": False,
            "initialization_run": False,
            "training_run": False,
            "promotion_order": protocol["promotion_order"],
        },
        "limitations": protocol["limitations"],
        "code": artifact(Path(__file__)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "topology": report["topology"],
        "boundary_comparison": report["boundary_comparison"],
        "known_counterexamples": report["known_counterexamples"],
        "recommended_next_step": recommendation,
    }, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)
