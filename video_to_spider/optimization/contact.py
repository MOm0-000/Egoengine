"""Mesh-local fingertip contact inference with hysteresis and persistence."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def infer_contact(
    mesh_m: trimesh.Trimesh, T_sim_object: np.ndarray, fingertips_sim: np.ndarray,
    timestamps_s: np.ndarray, valid_hand: np.ndarray, *, enter_distance_m: float = 0.012,
    exit_distance_m: float = 0.020, max_relative_speed_m_s: float = 0.30,
    min_duration_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer per-finger binary contact; local positions are in metric object coordinates."""
    count, hands = fingertips_sim.shape[:2]
    inverse_object = np.linalg.inv(T_sim_object)
    homogeneous = np.concatenate([fingertips_sim, np.ones((*fingertips_sim.shape[:-1], 1))], axis=-1)
    local = np.einsum("tij,thfj->thfi", inverse_object, homogeneous)[..., :3]
    vertices = np.asarray(mesh_m.vertices, dtype=np.float64)
    tree = cKDTree(vertices)
    flat_local = local.reshape(-1, 3)
    distances, nearest = tree.query(flat_local)
    distances = distances.reshape(count, hands, 5)
    closest = vertices[nearest].reshape(count, hands, 5, 3)
    dt = np.diff(timestamps_s)
    local_speed = np.zeros_like(distances)
    if count > 1:
        local_speed[1:] = np.linalg.norm(np.diff(local, axis=0), axis=-1) / np.maximum(dt[:, None, None], 1e-6)
        local_speed[0] = local_speed[1]
    contact = np.zeros((count, hands, 5), dtype=bool)
    for hand in range(hands):
        for finger in range(5):
            active = False
            for index in range(count):
                if not valid_hand[index, hand]:
                    active = False
                elif active:
                    active = bool(
                        distances[index, hand, finger] <= exit_distance_m
                        and local_speed[index, hand, finger] <= max_relative_speed_m_s * 1.5
                    )
                else:
                    active = bool(
                        distances[index, hand, finger] <= enter_distance_m
                        and local_speed[index, hand, finger] <= max_relative_speed_m_s
                    )
                contact[index, hand, finger] = active
            # Remove runs shorter than the configured minimum.
            start = 0
            while start < count:
                if not contact[start, hand, finger]:
                    start += 1
                    continue
                end = start + 1
                while end < count and contact[end, hand, finger]:
                    end += 1
                if end - start < min_duration_frames:
                    contact[start:end, hand, finger] = False
                start = end
    contact_positions = np.zeros((hands, 5, 3), dtype=np.float32)
    for hand in range(hands):
        for finger in range(5):
            active = contact[:, hand, finger]
            if active.any():
                contact_positions[hand, finger] = np.median(closest[active, hand, finger], axis=0)
            else:
                best = int(np.argmin(distances[:, hand, finger]))
                contact_positions[hand, finger] = closest[best, hand, finger]
    active_distances = distances[contact]
    active_slip = local_speed[contact]
    metrics = {
        "contact_rate": float(contact.mean()),
        "contact_distance_p95_m": float(np.percentile(active_distances, 95)) if active_distances.size else None,
        "contact_local_slip_p95_m_s": float(np.percentile(active_slip, 95)) if active_slip.size else None,
        "enter_distance_m": enter_distance_m, "exit_distance_m": exit_distance_m,
        "max_relative_speed_m_s": max_relative_speed_m_s,
        "min_duration_frames": min_duration_frames,
        "distance_method": "nearest canonical mesh vertex (conservative unsigned proxy)",
    }
    return contact.astype(np.float32), contact_positions, metrics
