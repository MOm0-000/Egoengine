"""Pure helpers for the bounded Replay 0->20 contact trace diagnostic.

The live MuJoCo-Warp state is sampled by the standalone script.  Everything in
this module is side-effect free so indexing, contact conventions, and the cost
ledger can be tested without advancing physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BudgetLedger:
    physics_steps: int = 0
    control_intervals: int = 0
    limit_physics_steps: int = 1000
    limit_control_intervals: int = 80

    def charge(self, *, physics_steps: int, control_intervals: int) -> None:
        if physics_steps < 0 or control_intervals < 0:
            raise ValueError("budget charges must be nonnegative")
        new_physics = self.physics_steps + physics_steps
        new_control = self.control_intervals + control_intervals
        if new_physics > self.limit_physics_steps or new_control > self.limit_control_intervals:
            raise RuntimeError("Replay contact-trace budget would be exceeded")
        self.physics_steps = new_physics
        self.control_intervals = new_control


def transition_indices(source_endpoint: int, substep: int, *, ctrl_steps: int = 10) -> dict[str, int]:
    if source_endpoint < 0 or not 0 <= substep < ctrl_steps:
        raise ValueError("invalid source endpoint or substep")
    return {
        "source_endpoint": source_endpoint,
        "outcome_endpoint": source_endpoint + 1,
        "command_reference_endpoint": source_endpoint + 1,
        "substep": substep,
        "global_substep": source_endpoint * ctrl_steps + substep + 1,
    }


def replay_command(reference_ctrl: np.ndarray, source_endpoint: int) -> np.ndarray:
    """Return the exact zero-residual command for transition t->t+1."""
    reference_ctrl = np.asarray(reference_ctrl)
    if reference_ctrl.ndim != 2 or not 0 <= source_endpoint + 1 < len(reference_ctrl):
        raise ValueError("reference control table or source endpoint is invalid")
    return reference_ctrl[source_endpoint + 1].copy()


def clone_array(value: object) -> np.ndarray:
    """Materialize an owned NumPy array; never retain a live Warp/Torch view."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    result = np.asarray(value).copy()
    if not result.flags.owndata:
        result = np.array(result, copy=True)
    return result


def valid_contact_prefix(value: object, nacon: int) -> np.ndarray:
    """Copy only active contact rows and never expose allocator padding."""
    result = clone_array(value)
    if result.ndim == 0 or not 0 <= nacon <= result.shape[0]:
        raise ValueError("invalid active-contact extent")
    return result[:nacon].copy()


def contact_frame_wrench_to_world(wrench: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Rotate MuJoCo [force, torque] row-vectors from contact to world frame.

    MuJoCo stores the contact frame as three world-space axes in rows.  The
    positive normal axis points from geom1 toward geom2.  Consequently the
    returned positive-normal wrench is the force/torque on geom2 by geom1;
    force on geom1 is its negative.
    """
    wrench = np.asarray(wrench, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    if wrench.shape != (6,) or frame.shape != (3, 3):
        raise ValueError("wrench/frame shapes must be (6,) and (3,3)")
    return np.r_[wrench[:3] @ frame, wrench[3:] @ frame]


def reference_decode_elliptic(efc_force: np.ndarray, address: int, dim: int) -> np.ndarray | None:
    """Small independent reference decoder for synthetic elliptic-cone tests."""
    force = np.asarray(efc_force, dtype=np.float64)
    if address < 0:
        return None
    if dim not in (1, 3, 4, 6) or address + dim > force.size:
        raise ValueError("invalid elliptic contact address/dimension")
    result = np.zeros(6, dtype=np.float64)
    result[:dim] = force[address : address + dim]
    return result


def reference_decode_pyramidal(
    efc_force: np.ndarray, address: int, dim: int, friction: np.ndarray
) -> np.ndarray | None:
    """Independent MuJoCo pyramidal contact-force decoding reference.

    Each non-normal dimension is represented by two pyramid edges.  The normal
    force is their sum and tangential/torque components are their difference
    multiplied by the corresponding friction coefficient.
    """
    force = np.asarray(efc_force, dtype=np.float64)
    friction = np.asarray(friction, dtype=np.float64)
    if address < 0:
        return None
    if dim not in (1, 3, 4, 6):
        raise ValueError("invalid pyramidal contact dimension")
    if dim == 1:
        if address >= force.size:
            raise ValueError("invalid pyramidal contact address")
        return np.array([force[address], 0, 0, 0, 0, 0], dtype=np.float64)
    count = 2 * (dim - 1)
    if address + count > force.size or friction.size < dim - 1:
        raise ValueError("invalid pyramidal force/friction extent")
    edges = force[address : address + count].reshape(dim - 1, 2)
    result = np.zeros(6, dtype=np.float64)
    result[0] = edges.sum()
    result[1:dim] = (edges[:, 0] - edges[:, 1]) * friction[: dim - 1]
    return result


def body_is_descendant(body_id: int, root_id: int, parents: np.ndarray) -> bool:
    body_id, root_id = int(body_id), int(root_id)
    while body_id > 0:
        if body_id == root_id:
            return True
        body_id = int(parents[body_id])
    return root_id == 0 and body_id == 0


def classify_geom_role(
    geom_id: int,
    geom_body: np.ndarray,
    body_parents: np.ndarray,
    body_names: Iterable[str],
    *,
    right_root: int,
    left_root: int,
    tool_root: int,
    target_root: int,
    floor_geom: int,
) -> str:
    if int(geom_id) == int(floor_geom):
        return "floor"
    body_id = int(geom_body[int(geom_id)])
    roots = (
        (right_root, "right_hand"),
        (left_root, "left_hand"),
        (tool_root, "tool"),
        (target_root, "target"),
    )
    for root, role in roots:
        if body_is_descendant(body_id, root, body_parents):
            if role.endswith("_hand"):
                name = list(body_names)[body_id].lower()
                finger = next((part for part in ("thumb", "index", "middle", "ring", "pinky", "palm") if part in name), "other")
                return f"{role}:{finger}"
            return role
    return "other"


def canonical_contact_group(role1: str, role2: str) -> str:
    coarse1, coarse2 = role1.split(":", 1)[0], role2.split(":", 1)[0]
    pair = frozenset((coarse1, coarse2))
    known = {
        frozenset(("right_hand", "tool")): "right_hand_tool",
        frozenset(("left_hand", "target")): "left_hand_target",
        frozenset(("right_hand", "target")): "right_hand_target",
        frozenset(("left_hand", "tool")): "left_hand_tool",
        frozenset(("tool", "target")): "tool_target",
        frozenset(("floor", "tool")): "floor_tool",
        frozenset(("floor", "target")): "floor_target",
    }
    return known.get(pair, f"{coarse1}_{coarse2}")
