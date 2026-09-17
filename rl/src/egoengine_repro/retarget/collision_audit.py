"""Independent distance audits, including collision pairs omitted by a model."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def validate_qpos(model, qpos, *, trajectory=False):
    """Reject invalid states instead of letting MuJoCo silently normalize them."""
    qpos = np.asarray(qpos, dtype=float)
    valid_shape = (qpos.ndim == 2 and len(qpos) > 0 and qpos.shape[1] == model.nq
                   if trajectory else qpos.shape == (model.nq,))
    if not valid_shape or not np.isfinite(qpos).all():
        raise ValueError("expected finite nonempty (T,nq) qpos" if trajectory else "expected one finite (nq,) qpos")
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind not in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            continue
        address = int(model.jnt_qposadr[joint]) + (3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 0)
        norms = np.linalg.norm(qpos[..., address:address + 4], axis=-1)
        if not np.all(np.abs(norms - 1) <= 1e-6):
            raise ValueError(f"expected unit quaternion for {model.joint(joint).name}; no normalization applied")
    return qpos


def hand_ids(model):
    return [i for i in range(model.ngeom)
            if model.geom(i).name.startswith("collision_hand_")]


def explicit_hand_pairs(model):
    hands = set(hand_ids(model))
    return sorted({tuple(sorted((int(a), int(b))))
                   for a, b in zip(model.pair_geom1, model.pair_geom2)
                   if a in hands and b in hands})


def nonadjacent_hand_pairs(model):
    pairs = []
    for a, b in combinations(hand_ids(model), 2):
        wa, wb = model.body_weldid[model.geom_bodyid[[a, b]]]
        pa, pb = model.body_weldid[model.body_parentid[[wa, wb]]]
        if wa != wb and wa != pb and wb != pa:
            pairs.append((a, b))
    return pairs


def collision_families(model):
    hands = hand_ids(model)
    objects = [[i for i in range(model.ngeom)
                if model.geom(i).name.startswith(f"{side}_object_")
                and not model.geom(i).name.endswith("visual")]
               for side in ("right", "left")]
    floor = model.geom("floor").id
    return {
        "self_explicit": explicit_hand_pairs(model),
        "self_nonadjacent_shells": nonadjacent_hand_pairs(model),
        "hand_tool": list(product(hands, objects[0])),
        "hand_target": list(product(hands, objects[1])),
        "tool_target": list(product(*objects)),
        "hand_floor": [(i, floor) for i in hands],
        "tool_floor": [(i, floor) for i in objects[0]],
        "target_floor": [(i, floor) for i in objects[1]],
    }


def distances(model, data, pairs, detection=0.05):
    validate_qpos(model, data.qpos)
    if not np.isfinite(detection) or detection <= 0:
        raise ValueError("positive finite distance detection bound required")
    if not np.isfinite(data.geom_xpos).all() or not np.isfinite(data.geom_xmat).all():
        raise ValueError("nonfinite geometry transforms in distance audit")
    result = np.asarray([mujoco.mj_geomDistance(model, data, a, b, detection, None)
                         for a, b in pairs])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite collision distance; cannot classify feasibility")
    return result


def audit_trajectory(model, qpos, *, tolerance=5e-5):
    qpos = validate_qpos(model, qpos, trajectory=True)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative collision tolerance required")
    data = mujoco.MjData(model)
    groups = collision_families(model)
    values = {key: np.empty((len(qpos), len(pairs))) for key, pairs in groups.items()}
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for key, pairs in groups.items():
            values[key][frame] = distances(model, data, pairs)
    report = {}
    for key, value in values.items():
        pairs = groups[key]
        worst = []
        if value.size:
            for index in np.argsort(value.min(axis=0))[:12]:
                frame = int(value[:, index].argmin())
                a, b = pairs[index]
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                # MuJoCo mesh distance uses a convex hull, not the original
                # concave CAD surface; it is a diagnostic, never a pass gate.
                meshes = [[i for i in range(model.ngeom)
                           if model.geom_bodyid[i] == model.geom_bodyid[g]
                           and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                           and model.geom_group[i] == 1] for g in (a, b)]
                hull_pairs = list(product(*meshes))
                hull = distances(model, data, hull_pairs)
                worst.append(dict(frame=frame, geom1=model.geom(a).name,
                                  geom2=model.geom(b).name,
                                  distance_m=float(value[frame, index]),
                                  native_visual_convex_hull_distance_m=(float(hull.min())
                                                                      if hull.size else None)))
        report[key] = dict(pair_count=len(pairs),
                           min_distance_m=float(value.min()) if value.size else None,
                           penetrating_frames=int(np.any(value < -tolerance, axis=1).sum()),
                           worst_pairs=worst)
    report["object_mass_kg"] = {side: float(model.body(f"{side}_object").mass[0])
                                for side in ("right", "left")}
    report["tolerance_m"] = tolerance
    report["strict_gate_passed"] = False
    report["physics_validated"] = False
    return report


def audit_intrahand_trajectory(model, qpos, *, tolerance=5e-5):
    """Audit every same-hand collision shell pair over a qpos trajectory.

    The runtime XML intentionally declares only a subset of hand self-contact
    pairs.  This report keeps those pairs separate from omitted assembly
    neighbours and omitted non-adjacent pairs, so a candidate cannot be called
    collision-free merely because MINK did not constrain a pair.
    """
    qpos = validate_qpos(model, qpos, trajectory=True)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative collision tolerance required")

    explicit = set(explicit_hand_pairs(model))
    nonadjacent = set(nonadjacent_hand_pairs(model))
    hands = hand_ids(model)
    pairs = []
    classifications = []
    for a, b in combinations(hands, 2):
        name_a, name_b = model.geom(a).name, model.geom(b).name
        if name_a.split("_")[2] != name_b.split("_")[2]:
            continue
        pair = (a, b)
        if pair in explicit:
            classification = "declared"
        elif pair in nonadjacent:
            classification = "omitted_nonadjacent"
        else:
            classification = "omitted_assembly_adjacent"
        pairs.append(pair)
        classifications.append(classification)

    values = np.empty((len(qpos), len(pairs)), dtype=float)
    data = mujoco.MjData(model)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = distances(model, data, pairs)

    details = []
    for index, (a, b) in enumerate(pairs):
        frame = int(values[:, index].argmin())
        minimum = float(values[frame, index])
        penetrating = int((values[:, index] < -tolerance).sum())
        details.append(dict(
            geom1=model.geom(a).name,
            geom2=model.geom(b).name,
            classification=classifications[index],
            declared=classifications[index] == "declared",
            omitted_assembly_adjacent=classifications[index] == "omitted_assembly_adjacent",
            omitted_nonadjacent=classifications[index] == "omitted_nonadjacent",
            minimum_distance_m=minimum,
            worst_frame=frame,
            penetrating_frames=penetrating,
        ))

    categories = ("declared", "omitted_assembly_adjacent", "omitted_nonadjacent")
    counts = {key: classifications.count(key) for key in categories}
    by_classification = {}
    for key in categories:
        indices = [index for index, value in enumerate(classifications) if value == key]
        subset = values[:, indices]
        by_classification[key] = dict(
            pair_count=len(indices),
            min_distance_m=float(subset.min()) if subset.size else None,
            penetrating_pair_count=int(np.any(subset < -tolerance, axis=0).sum()) if subset.size else 0,
            penetrating_frames=int(np.any(subset < -tolerance, axis=1).sum()) if subset.size else 0,
        )
    return dict(
        pair_count=len(pairs),
        counts=counts,
        by_classification=by_classification,
        min_distance_m=float(values.min()) if values.size else None,
        penetrating_frames=int(np.any(values < -tolerance, axis=1).sum()) if values.size else 0,
        tolerance_m=float(tolerance),
        pairs=details,
        physics_validated=False,
        strict_gate_passed=False,
    )


def source_topology_report(template: Path):
    root = ET.parse(template).getroot()
    hands = sorted(g.get("name") for g in root.findall(".//geom")
                   if g.get("name", "").startswith("collision_hand_"))
    source = {tuple(sorted((p.get("geom1"), p.get("geom2"))))
              for p in root.findall("contact/pair")
              if p.get("geom1") in hands and p.get("geom2") in hands}
    intra = [p for p in source if p[0].split("_")[2] == p[1].split("_")[2]]
    omitted = [p for p in combinations(hands, 2)
               if p[0].split("_")[2] == p[1].split("_")[2] and p not in source]
    return dict(source_hand_pairs=len(source), source_intrahand_pairs=len(intra),
                source_interhand_pairs=len(source) - len(intra),
                omitted_intrahand_pairs=[list(p) for p in omitted],
                interpretation="source models palm/distal and distal/distal self contact only; not full CAD collision coverage")
