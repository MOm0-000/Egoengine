"""Active free-object action primitives.

Historical chunk/MPC/RL stacks are intentionally not re-exported.  New code
must import an explicit active module so old experiment dependencies cannot be
loaded accidentally through this package initializer.
"""

from .contracts import (
    HUMAN_CONTACT_POSITION_SOURCE,
    SOURCE_TIMED_CONTROL_SCHEMA,
    SOURCE_TIMED_TRACE_SCHEMA,
    XHAND_DYNAMIC_CAGE_IMPEDANCE,
    XHAND_NOMINAL_TRACKING_IMPEDANCE,
    FingerImpedance,
    contact_reference_signature,
    reference_trajectory_signature,
)
from .replay import (
    MujocoReplayBackend,
    direct_object_actuator_names,
    require_free_object_joint,
)

__all__ = [
    "SOURCE_TIMED_CONTROL_SCHEMA",
    "SOURCE_TIMED_TRACE_SCHEMA",
    "XHAND_DYNAMIC_CAGE_IMPEDANCE",
    "XHAND_NOMINAL_TRACKING_IMPEDANCE",
    "FingerImpedance",
    "HUMAN_CONTACT_POSITION_SOURCE",
    "contact_reference_signature",
    "reference_trajectory_signature",
    "MujocoReplayBackend",
    "direct_object_actuator_names",
    "require_free_object_joint",
]
