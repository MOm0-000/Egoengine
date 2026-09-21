"""Validate the compiled MuJoCo local surface guard on an unseen bend grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from egoengine_repro.retarget.collision_audit import hand_ids
from egoengine_repro.retarget.mink import _explicit_collision_groups
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts
from fit_taco_left_palm_thumb_boundary_cegis import _extract_rows
from fit_taco_left_palm_thumb_semantic_guard import Oracle

DEFAULT_PROTOCOL = ROOT / "configs/taco_pour_left_palm_thumb_local_surface_guard_v1.yaml"
DEFAULT_RUN = ROOT / "runs/taco_pour_left_palm_thumb_local_surface_guard_v1"


def run(protocol_path: Path, run_dir: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    verify_artifacts(protocol["inputs"].values())
    build = json.loads((run_dir / "build_report.json").read_text())
    verify_artifacts(build["palm_assets"] + build["thumb_assets"])
    candidate_path = Path(build["candidate_scene"]["path"])
    if artifact(candidate_path)["sha256"] != build["candidate_scene"]["sha256"]:
        raise ValueError("candidate scene changed after build")
    model = mujoco.MjModel.from_xml_path(str(candidate_path))
    data = mujoco.MjData(model)
    with np.load(protocol["inputs"]["robot_reference"]["path"], allow_pickle=False) as source:
        seed = np.asarray(source["qpos"][0], dtype=float)
    bend_address = int(model.joint("left_hand_thumb_bend_joint").qposadr[0])
    rota_address = int(model.joint("left_hand_thumb_rota_joint1").qposadr[0])
    guard_pairs = [
        (model.geom(row["palm_geom"]).id, model.geom(row["thumb_geom"]).id)
        for row in build["guard_pairs"]
    ]
    guard_ids = {geom for pair in guard_pairs for geom in pair}
    fromto = np.empty(6, dtype=float)
    zero_distance_queries = 0

    def runtime_collision(bend: float, rota: float) -> bool:
        nonlocal zero_distance_queries
        data.qpos[:] = seed
        data.qpos[bend_address] = bend
        data.qpos[rota_address] = rota
        mujoco.mj_forward(model, data)
        distances = [
            mujoco.mj_geomDistance(model, data, first, second, 1.0, fromto)
            for first, second in guard_pairs
        ]
        zero_distance_queries += sum(distance == 0.0 for distance in distances)
        # MuJoCo can return exactly zero for separated mesh pairs for which its
        # convex distance query has no positive witness. Such pairs do not
        # enter data.contact. Negative distance is the runtime penetration
        # signal and is what the explicit pair dynamics acts on.
        return any(distance < 0.0 for distance in distances)

    holdout = protocol["independent_holdout"]
    base_bends = np.linspace(0.0, 1.83, 257)
    fraction = float(holdout["bend_fraction"])
    bends = base_bends[:-1] + fraction * np.diff(base_bends)
    rotas = np.linspace(-1.05, 1.57, int(holdout["rota1_coarse_samples"]))
    oracle = Oracle(
        Path(protocol["inputs"]["scene"]["path"]),
        Path(protocol["inputs"]["robot_reference"]["path"]),
    )
    native_rows = _extract_rows(
        oracle, None, bends, rotas,
        float(holdout["transition_numerical_tolerance_rad"]),
        int(holdout["transition_max_bisections"]),
    )
    offsets = list(map(float, holdout["boundary_probe_offsets_rad"]))
    false_positives, false_negatives, topology_mismatches, errors = [], [], [], []
    tolerance = float(holdout["transition_numerical_tolerance_rad"])
    max_iterations = int(holdout["transition_max_bisections"])
    for row in native_rows:
        bend = float(row["thumb_bend_rad"])
        intervals = row["native"]["collision_intervals_rad"]
        if not intervals:
            if runtime_collision(bend, 1.57):
                false_positives.append({"thumb_bend_rad": bend, "thumb_rota1_rad": 1.57})
                topology_mismatches.append(row["bend_slice_index"])
            continue
        boundary = float(intervals[0][0])
        for offset in offsets:
            clear = boundary - offset
            collision = boundary + offset
            if runtime_collision(bend, clear):
                false_positives.append({"thumb_bend_rad": bend, "thumb_rota1_rad": clear})
            if not runtime_collision(bend, collision):
                false_negatives.append({"thumb_bend_rad": bend, "thumb_rota1_rad": collision})
        lo, hi = boundary - 0.003, boundary + 0.003
        if runtime_collision(bend, lo) or not runtime_collision(bend, hi):
            topology_mismatches.append(row["bend_slice_index"])
            continue
        iterations = 0
        while hi - lo > tolerance and iterations < max_iterations:
            mid = (lo + hi) / 2.0
            if runtime_collision(bend, mid):
                hi = mid
            else:
                lo = mid
            iterations += 1
        errors.append((lo + hi) / 2.0 - boundary)

    error = np.abs(np.asarray(errors, dtype=float))
    known = {}
    for name, bend, rota in (
        ("old_hybrid_false_negative", 0.12112506873163462, 1.4374059661708884),
        ("old_hybrid_false_positive", 0.004048664938583147, 1.4399843552192655),
    ):
        native = oracle.evaluate(bend, rota)[0]
        runtime = runtime_collision(bend, rota)
        known[name] = {
            "thumb_bend_rad": bend, "thumb_rota1_rad": rota,
            "native_collision": bool(native), "runtime_collision": bool(runtime),
            "matched": bool(native == runtime),
        }

    all_hand = set(hand_ids(model))
    mink_groups = _explicit_collision_groups(
        model, mujoco, hand_geom_ids=all_hand, object_geom_ids=None
    )
    mink_pairs = {
        tuple(sorted((model.geom(first[0]).id, model.geom(second[0]).id)))
        for first, second in mink_groups
        if model.geom(first[0]).id in guard_ids or model.geom(second[0]).id in guard_ids
    }
    runtime_pairs = {tuple(sorted(pair)) for pair in guard_pairs}
    compiled_guard_pairs = []
    for pair_id in range(model.npair):
        pair = (int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id]))
        if pair[0] in guard_ids or pair[1] in guard_ids:
            compiled_guard_pairs.append(tuple(sorted(pair)))
    isolation_passed = set(compiled_guard_pairs) == runtime_pairs
    pair_sets_identical = mink_pairs == runtime_pairs

    acceptance = protocol["acceptance"]
    max_error = float(error.max()) if len(error) else None
    passed = (
        len(false_positives) == int(acceptance["sampled_false_positives"])
        and len(false_negatives) == int(acceptance["sampled_false_negatives"])
        and len(set(topology_mismatches)) == int(acceptance["topology_mismatch_slices"])
        and max_error is not None
        and max_error <= float(acceptance["maximum_absolute_boundary_error_rad"])
        and all(row["matched"] for row in known.values())
        and isolation_passed and pair_sets_identical
    )
    report = {
        "status": (
            "isolated_local_surface_self_guard_passed_not_combined_or_promoted"
            if passed else "isolated_local_surface_self_guard_rejected"
        ),
        "protocol": artifact(protocol_path),
        "build_report": artifact(run_dir / "build_report.json"),
        "candidate_scene": build["candidate_scene"],
        "independent_holdout": {
            "bend_fraction": fraction,
            "bend_slices": len(bends),
            "native_collision_slices": sum(
                bool(row["native"]["collision_intervals_rad"]) for row in native_rows
            ),
            "sampled_false_positive_count": len(false_positives),
            "sampled_false_negative_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "topology_mismatch_slice_indices": sorted(set(topology_mismatches)),
            "comparable_boundary_slices": len(errors),
            "boundary_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
            "boundary_median_absolute_error_rad": float(np.median(error)),
            "boundary_p95_absolute_error_rad": float(np.quantile(error, 0.95)),
            "boundary_maximum_absolute_error_rad": max_error,
        },
        "known_old_hybrid_counterexamples": known,
        "pair_contract": {
            "guard_geom_count": len(guard_ids),
            "runtime_guard_pair_count": len(runtime_pairs),
            "compiled_pairs_touching_guard_count": len(compiled_guard_pairs),
            "guards_pair_only_with_each_other": isolation_passed,
            "mink_guard_pair_count": len(mink_pairs),
            "runtime_and_mink_guard_pair_sets_identical": pair_sets_identical,
            "zero_distance_queries_not_counted_as_penetration": zero_distance_queries,
            "penetration_definition": "mj_geomDistance < 0; exact zero without a generated contact is boundary/unsupported-positive-distance, not material penetration",
        },
        "passed_local_self_guard_gate": passed,
        "formal_scene_modified": False,
        "combined_with_external_floor_candidate": False,
        "reference_modified": False,
        "mink_retarget_run": False,
        "capacity_run": False,
        "initialization_run": False,
        "training_run": False,
        "next_steps": protocol["promotion_after_acceptance"] if passed else [],
        "code": artifact(Path(__file__)),
    }
    (run_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args.protocol, args.run_dir)
