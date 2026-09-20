"""Fit pair-specific sphere guards for the left palm/thumb-root CAD pair.

This is an offline calibration tool.  Native CAD triangle intersection is the
label oracle; the emitted spheres are only candidates for one explicit
palm/thumb self-collision pair family.  Object and floor contacts are outside
this script's scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import fcl
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REJECTED = ROOT / "TRASH/rejected_candidates/2026-09-20_collision_semantics_repair_v2_self_guard"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from audit_taco_initialization import visual_meshes
from audit_taco_initialization_preflight import triangle_object
from egoengine_repro.retarget.paper_audit import artifact

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
REFERENCE = ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/robot_reference.npz"


class Oracle:
    def __init__(self, scene: Path, reference: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        meshes, _ = visual_meshes(scene, self.model)
        palm = self.model.geom("left_hand_link_visual").id
        thumb = self.model.geom("left_thumb_rota1_visual").id
        self.palm_body = int(self.model.geom_bodyid[palm])
        self.thumb_body = int(self.model.geom_bodyid[thumb])
        self.palm = triangle_object(meshes[palm])
        self.thumb = triangle_object(meshes[thumb])
        with np.load(reference, allow_pickle=False) as source:
            self.seed = np.asarray(source["qpos"][0], dtype=float)
            self.reference_qpos = np.asarray(source["qpos"], dtype=float)
        self.bend_address = int(self.model.joint("left_hand_thumb_bend_joint").qposadr[0])
        self.rota_address = int(self.model.joint("left_hand_thumb_rota_joint1").qposadr[0])

    def transform(self, bend: float, rota: float):
        self.data.qpos[:] = self.seed
        self.data.qpos[self.bend_address] = bend
        self.data.qpos[self.rota_address] = rota
        mujoco.mj_kinematics(self.model, self.data)
        rp = self.data.xmat[self.palm_body].reshape(3, 3)
        rt = self.data.xmat[self.thumb_body].reshape(3, 3)
        rotation = rp.T @ rt
        translation = rp.T @ (
            self.data.xpos[self.thumb_body] - self.data.xpos[self.palm_body]
        )
        return rotation, translation

    def evaluate(self, bend: float, rota: float, contacts: int = 0):
        rotation, translation = self.transform(bend, rota)
        self.thumb.setTransform(fcl.Transform(rotation, translation))
        result = fcl.CollisionResult()
        fcl.collide(
            self.palm,
            self.thumb,
            fcl.CollisionRequest(
                num_max_contacts=max(1, contacts), enable_contact=contacts > 0
            ),
            result,
        )
        points = np.asarray([row.pos for row in result.contacts], dtype=float)
        if points.size:
            points = points.reshape(-1, 3)
        else:
            points = np.empty((0, 3), dtype=float)
        return bool(result.is_collision), rotation, translation, points


def _adaptive_boundary_samples(oracle: Oracle):
    """Add near-boundary samples without consuming either holdout grid."""
    samples = []
    scan_rotas = np.linspace(-1.05, 1.57, 97)
    for bend in np.linspace(0.0, 1.83, 129):
        scan = [oracle.evaluate(float(bend), float(rota))[0] for rota in scan_rotas]
        for index in range(len(scan_rotas) - 1):
            if scan[index] == scan[index + 1]:
                continue
            lo, hi = float(scan_rotas[index]), float(scan_rotas[index + 1])
            lo_label = scan[index]
            for _ in range(22):
                mid = (lo + hi) / 2.0
                if oracle.evaluate(float(bend), mid)[0] == lo_label:
                    lo = mid
                else:
                    hi = mid
            boundary = (lo + hi) / 2.0
            for delta in (-1.5e-3, -5e-4, 5e-4, 1.5e-3):
                rota = boundary + delta
                if -1.05 <= rota <= 1.57:
                    samples.append((float(bend), float(rota)))
    return samples


def _grid(oracle: Oracle):
    bends = np.linspace(0.0, 1.83, 65)
    rotas = np.linspace(-1.05, 1.57, 97)
    calibration = [(float(bend), float(rota)) for bend in bends for rota in rotas]
    calibration.extend(_adaptive_boundary_samples(oracle))
    calibration = list(dict.fromkeys(calibration))
    # Both offset grids are never used by the fitter.
    midpoint = [
        (float(bend), float(rota))
        for bend in (bends[:-1] + bends[1:]) / 2
        for rota in (rotas[:-1] + rotas[1:]) / 2
    ]
    third = [
        (float(bends[i] + (bends[i + 1] - bends[i]) / 3),
         float(rotas[j] + 2 * (rotas[j + 1] - rotas[j]) / 3))
        for i in range(len(bends) - 1)
        for j in range(len(rotas) - 1)
    ]
    golden_a = [
        (float(bends[i] + (np.sqrt(2.0) - 1.0) * (bends[i + 1] - bends[i])),
         float(rotas[j] + (np.sqrt(3.0) - 1.0) * (rotas[j + 1] - rotas[j])))
        for i in range(len(bends) - 1)
        for j in range(len(rotas) - 1)
    ]
    golden_b = [
        (float(bends[i] + (np.sqrt(3.0) - 1.0) * (bends[i + 1] - bends[i])),
         float(rotas[j] + (np.sqrt(2.0) - 1.0) * (rotas[j + 1] - rotas[j])))
        for i in range(len(bends) - 1)
        for j in range(len(rotas) - 1)
    ]
    return calibration, midpoint, third, golden_a, golden_b


def _labels(oracle: Oracle, samples, contact_count=0):
    labels = []
    transforms = []
    contacts = []
    for index, (bend, rota) in enumerate(samples):
        label, rotation, translation, points = oracle.evaluate(
            bend, rota, contact_count
        )
        labels.append(label)
        transforms.append((rotation, translation))
        contacts.append(points)
        if index and index % 1000 == 0:
            print(f"native CAD oracle: {index}/{len(samples)}", flush=True)
    return np.asarray(labels, dtype=bool), transforms, contacts


def _candidate_centres(labels, transforms, contacts):
    candidates = []
    for label, (rotation, translation), points in zip(labels, transforms, contacts):
        if not label:
            continue
        for point in points:
            thumb = rotation.T @ (point - translation)
            candidates.append((point.copy(), thumb))
    # Contact triangles often return repeated points.  A quarter-millimetre
    # key removes duplicates without moving the original centres.
    unique = {}
    for palm, thumb in candidates:
        key = tuple(np.rint(np.r_[palm, thumb] / 2.5e-4).astype(int))
        unique.setdefault(key, (palm, thumb))
    return list(unique.values())


def _distances(candidates, transforms):
    result = np.empty((len(candidates), len(transforms)), dtype=float)
    for row, (palm, thumb) in enumerate(candidates):
        result[row] = [
            np.linalg.norm(palm - (translation + rotation @ thumb))
            for rotation, translation in transforms
        ]
    return result


def _fit(candidates, transforms, labels, clearance):
    distances = _distances(candidates, transforms)
    negative = ~labels
    radii = distances[:, negative].min(axis=1) - clearance
    valid = radii > 0.0
    candidates = [candidate for candidate, keep in zip(candidates, valid) if keep]
    distances = distances[valid]
    radii = radii[valid]
    coverage = distances[:, labels] < radii[:, None]
    uncovered = np.ones(int(labels.sum()), dtype=bool)
    chosen = []
    while uncovered.any():
        gains = np.count_nonzero(coverage[:, uncovered], axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            raise RuntimeError(f"sphere candidates leave {int(uncovered.sum())} calibration positives uncovered")
        chosen.append(best)
        uncovered &= ~coverage[best]
    guards = []
    for index in chosen:
        guards.append({
            "palm_center_m": candidates[index][0].tolist(),
            "thumb_center_m": candidates[index][1].tolist(),
            "sphere_radius_m": float(radii[index] / 2.0),
        })
    return guards


def _predict(guards, transforms):
    predicted = np.zeros(len(transforms), dtype=bool)
    for guard in guards:
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
    def rows(indices):
        return [
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


def run(scene: Path, reference: Path, output: Path):
    oracle = Oracle(scene, reference)
    calibration, midpoint, third, golden_a, golden_b = _grid(oracle)
    labels, transforms, contacts = _labels(oracle, calibration, contact_count=64)
    candidates = _candidate_centres(labels, transforms, contacts)
    guards = _fit(candidates, transforms, labels, clearance=0.0)

    # Counterexample-guided refinement uses two declared tuning grids.  They
    # are not reported as holdout evidence.  Only their misclassified points
    # enter the second fit; the two irrational-offset grids below stay unseen.
    tuning = {}
    counterexamples = []
    for name, samples in (("midpoint_tuning", midpoint), ("one_third_tuning", third)):
        grid_labels, grid_transforms, _ = _labels(oracle, samples)
        predicted = _predict(guards, grid_transforms)
        tuning[name] = _classification(samples, grid_labels, predicted)
        counterexamples.extend(
            sample for sample, native, guard in zip(samples, grid_labels, predicted)
            if native != guard
        )
    if counterexamples:
        calibration.extend(counterexamples)
        calibration = list(dict.fromkeys(calibration))
        labels, transforms, contacts = _labels(oracle, calibration, contact_count=64)
        candidates = _candidate_centres(labels, transforms, contacts)
        guards = _fit(candidates, transforms, labels, clearance=0.0)

    grids = {
        "calibration_after_counterexamples": _classification(
            calibration, labels, _predict(guards, transforms)
        )
    }
    for name, samples in (
        ("golden_offset_holdout_a", golden_a),
        ("golden_offset_holdout_b", golden_b),
    ):
        grid_labels, grid_transforms, _ = _labels(oracle, samples)
        grids[name] = _classification(
            samples, grid_labels, _predict(guards, grid_transforms)
        )

    reference_samples = list(zip(
        oracle.reference_qpos[:41, oracle.bend_address],
        oracle.reference_qpos[:41, oracle.rota_address],
    ))
    reference_labels, reference_transforms, _ = _labels(oracle, reference_samples)
    grids["reference_endpoints_0_40"] = _classification(
        reference_samples,
        reference_labels,
        _predict(guards, reference_transforms),
    )
    report = {
        "status": "candidate_guard_fit_passed" if all(
            not row["false_positive_count"] and not row["false_negative_count"]
            for row in grids.values()
        ) else "candidate_guard_fit_rejected",
        "scope": "offline_left_palm_thumb1_pair_specific_guard_fit",
        "scene": artifact(scene),
        "reference": artifact(reference),
        "joint_space": {
            "left_hand_thumb_bend_joint_rad": [0.0, 1.83],
            "left_hand_thumb_rota_joint1_rad": [-1.05, 1.57],
        },
        "native_oracle": "FCL triangle-surface intersection on original CAD meshes",
        "fit": {
            "candidate_contact_centres": len(candidates),
            "negative_clearance_margin_m": 0.0,
            "counterexample_tuning": tuning,
            "counterexamples_added": len(counterexamples),
            "guard_pairs": guards,
        },
        "validation": grids,
        "code": artifact(Path(__file__)),
        "formal_scene_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], "guards", len(guards), flush=True)
    for name, row in grids.items():
        print(name, "FP", row["false_positive_count"], "FN", row["false_negative_count"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument(
        "--output", type=Path,
        default=REJECTED / "left_palm_thumb_guard_fit.json",
    )
    args = parser.parse_args()
    run(args.scene, args.reference, args.output)
