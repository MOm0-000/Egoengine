"""State and observation types for the active MuJoCo backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationState:
    qpos: np.ndarray
    qvel: np.ndarray
    time: float
    ctrl: np.ndarray | None = None

    def copy(self) -> "SimulationState":
        return SimulationState(
            qpos=np.asarray(self.qpos).copy(), qvel=np.asarray(self.qvel).copy(),
            time=float(self.time), ctrl=None if self.ctrl is None else np.asarray(self.ctrl).copy(),
        )


@dataclass(frozen=True)
class BackendObservation:
    object_pose: np.ndarray
    wrist_poses: np.ndarray
    hand_joint_positions: np.ndarray
    thumb_contact: bool = False
    other_finger_contact: bool = False
