"""Geometry selection rules shared by physics-facing evaluation code.

SPIDER object bodies contain two intentionally different geometry classes:
CoACD convex hulls used by MuJoCo contact pairs, and an ``*_object_visual``
triangle mesh used only by the renderer.  The latter must never participate in
distance, penetration, or grasp-contact metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def body_subtree_ids(model: Any, root_body_id: int) -> frozenset[int]:
    """Return one compiled body's complete rigid/kinematic subtree."""
    root = int(root_body_id)
    if not hasattr(model, "body_parentid"):
        return frozenset({root})
    parents = np.asarray(model.body_parentid, dtype=np.int64)
    if root < 0 or root >= len(parents):
        raise ValueError(f"body id is outside the compiled model: {root}")
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body, parent in enumerate(parents.tolist()):
            if body not in bodies and parent in bodies:
                bodies.add(body)
                changed = True
    return frozenset(bodies)


def physical_object_geom_ids(model: Any, mujoco: Any, object_body_id: int) -> frozenset[int]:
    """Return collision-hull geoms belonging to one object subtree.

    The generated XML explicitly disables collision on ``*_object_visual``.
    Filtering by the body alone therefore gives a visually plausible but
    physically invalid distance measurement.
    """
    object_body_id = int(object_body_id)
    object_bodies = body_subtree_ids(model, object_body_id)
    result = frozenset(
        geom
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in object_bodies
        and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
        .endswith("_object_visual")
    )
    if not result:
        raise ValueError(
            f"object body {object_body_id} has no physical collision geometry; "
            "cannot compute a physics contact metric"
        )
    return result


def explicit_collision_pairs(
    model: Any, mujoco: Any, *, object_geom_ids: frozenset[int],
    hand_sides: tuple[str, ...],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return the one runtime self/floor/object collision graph.

    SPIDER scenes use explicit pairs.  Keeping this parser here prevents the
    nominal builder, dynamics audit and residual search from quietly using
    three slightly different definitions of a hand collision geometry.
    """
    sides = tuple(str(side) for side in hand_sides)
    if not sides or set(sides) - {"left", "right"}:
        raise ValueError("hand collision sides must be a nonempty left/right tuple")
    prefixes = tuple(f"collision_hand_{side}_" for side in sides)
    hand_geoms = {
        geom
        for geom in range(int(model.ngeom))
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
        .startswith(prefixes)
    }
    if not hand_geoms:
        raise ValueError(f"runtime scene has no collision geometry for hand sides {sides}")
    families: dict[str, list[tuple[int, int]]] = {
        "self": [], "floor": [], "object": [],
    }
    for pair in range(int(model.npair)):
        first, second = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first) or ""
        second_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second) or ""
        first_hand = first_name.startswith(prefixes)
        second_hand = second_name.startswith(prefixes)
        if first_hand and second_hand:
            families["self"].append((first, second))
        elif (first_hand and second_name == "floor") or (second_hand and first_name == "floor"):
            families["floor"].append((first, second))
        elif (
            (first_hand and second in object_geom_ids)
            or (second_hand and first in object_geom_ids)
        ):
            families["object"].append((first, second))
    if not all(families.values()):
        raise ValueError(
            "runtime self/floor/object collision graph is incomplete: "
            f"{ {name: len(pairs) for name, pairs in families.items()} }",
        )
    names = {
        geom: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
        for geom in hand_geoms
    }
    self_only = {geom for geom, name in names.items() if "guard" in name}
    object_only = {geom for geom, name in names.items() if "_external_semantic_" in name}
    floor_only = {geom for geom, name in names.items() if "_floor_semantic_" in name}
    base = hand_geoms - self_only - object_only - floor_only
    floor_covered = {
        first if first in hand_geoms else second
        for first, second in families["floor"]
    }
    object_covered = {
        first if first in hand_geoms else second
        for first, second in families["object"]
    }
    # A scene may replace every legacy primitive--floor pair with a dedicated
    # native-mesh support geom.  In that case the old primitives deliberately
    # remain object/self-only and must not be required in the floor family.
    expected_floor = floor_only if floor_only else base
    expected_object = base | object_only
    if floor_covered != expected_floor or object_covered != expected_object:
        raise ValueError(
            "runtime collision graph does not cover every selected hand geom: "
            f"floor_missing={sorted(expected_floor - floor_covered)}, "
            f"floor_unexpected={sorted(floor_covered - expected_floor)}, "
            f"object_missing={sorted(expected_object - object_covered)}, "
            f"object_unexpected={sorted(object_covered - expected_object)}"
        )
    return {name: tuple(pairs) for name, pairs in families.items()}


def minimum_pair_distance(
    model: Any, data: Any, mujoco: Any, pairs: tuple[tuple[int, int], ...],
) -> float:
    """Measure the minimum signed gap over a nonempty explicit pair family."""
    if not pairs:
        raise ValueError("cannot measure an empty explicit collision-pair family")
    fromto = np.empty(6, dtype=np.float64)
    return min(float(mujoco.mj_geomDistance(
        model, data, first, second, 1.0, fromto,
    )) for first, second in pairs)
