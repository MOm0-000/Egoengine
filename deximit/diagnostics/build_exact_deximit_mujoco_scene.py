#!/usr/bin/env python3
"""Build an isolated MuJoCo scene from DexImit's exact SAPIEN robot URDF.

This is deliberately separate from the formal renderer and 3.3 chain.  The
robot kinematic tree, link inertias, joint axes, joint limits, and collision
meshes come from the same URDF loaded by DexImit's SAPIEN environment.  Only
engine-specific contact and drive behavior remains to be calibrated at
runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import trimesh


ROBOT_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2", "right_hand_index_bend_joint",
    "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
)
HAND_MESHES = (
    "right_hand_link", "right_hand_thumb_bend_link",
    "right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2",
    "right_hand_index_bend_link", "right_hand_index_rota_link1",
    "right_hand_index_rota_link2", "right_hand_mid_link1",
    "right_hand_mid_link2", "right_hand_ring_link1",
    "right_hand_ring_link2", "right_hand_pinky_link1",
    "right_hand_pinky_link2",
)
COOKED_OWNER_BY_MESH = {
    "base": "base_link_inertia",
    "shoulder": "shoulder_link",
    "upperarm": "upper_arm_link",
    "forearm": "forearm_link",
    "wrist1": "wrist_1_link",
    "wrist2": "wrist_2_link",
    "wrist3": "wrist_3_link",
    **{name: name for name in HAND_MESHES},
}
PHYSX_POSITION_ITERATIONS = 25
SOURCE_TIMESTEP_S = 1.0 / 240.0
MUJOCO_SUBSTEP_S = SOURCE_TIMESTEP_S / PHYSX_POSITION_ITERATIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_row(
    summary_path: Path, candidate: int, *, allow_failed: bool = False,
) -> tuple[dict[str, object], bool]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("original_sapien_passes", [])
        if int(row.get("source_candidate_index", -1)) == candidate
    ]
    passed = True
    if len(rows) != 1 or rows[0].get("original_sapien_pass") is not True:
        if not allow_failed:
            raise ValueError("SAPIEN summary does not contain one selected physical pass")
        rows = [
            row for row in payload.get("all_attempts", [])
            if int(row.get("source_candidate_index", -1)) == candidate
            and row.get("original_sapien_pass") is False
        ]
        passed = False
    if len(rows) != 1:
        raise ValueError("SAPIEN summary does not contain one selected candidate row")
    metrics = rows[0].get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("selected SAPIEN row lacks physical metrics")
    return metrics, passed


def cooked_collision_rows(path: Path, object_mesh: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "deximit_sapien_cooked_collisions_v1_diagnostic_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("grasp_candidate_executed") is not False
        or Path(payload.get("object_input_mesh", "")).resolve() != object_mesh
        or payload.get("object_input_mesh_sha256") != sha256(object_mesh)
    ):
        raise ValueError("SAPIEN cooked-collision report violates its diagnostic contract")
    rows = payload.get("shapes", [])
    by_owner = {str(row.get("owner")): row for row in rows}
    expected = set(COOKED_OWNER_BY_MESH.values()) | {"right_object"}
    if set(by_owner) != expected or len(rows) != len(expected):
        raise ValueError("cooked-collision report does not define each exact convex shape")
    for owner, row in by_owner.items():
        mesh = Path(str(row.get("mesh", ""))).resolve(strict=True)
        if (
            row.get("shape_index") != 0
            or row.get("shape_type") != "PhysxCollisionShapeConvexMesh"
            or row.get("mesh_sha256") != sha256(mesh)
            or int(row.get("cooked_vertex_count", 0)) < 4
        ):
            raise ValueError(f"invalid cooked collision for {owner!r}")
        row["mesh"] = str(mesh)
    return by_owner


def calibrated_contact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_contact_profile", {})
    result = {
        "time_constant_s": float(selected.get("time_constant_s", np.nan)),
        "damping_ratio": float(selected.get("damping_ratio", np.nan)),
        "constraint_impedance": float(
            selected.get("constant_constraint_impedance", np.nan),
        ),
        "table_static_speed_m_s": float(
            selected.get("static_friction_speed_threshold_m_s", np.nan),
        ),
        "integration_order": str(payload.get("integration_order", "native")),
        "friction_cone": str(payload.get("friction_cone", "")),
        "noslip_iterations": int(payload.get("noslip_iterations", -1)),
    }
    if (
        payload.get("schema") not in {
            "deximit_mujoco_contact_calibration_v3_friction_solver_diagnostic_only",
            "deximit_mujoco_contact_calibration_v4_fixed_profile_diagnostic_only",
        }
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("grasp_candidate_executed") is not False
        or not bool(selected.get("hard_conditions_passed"))
        or not np.isfinite([
            result["time_constant_s"], result["damping_ratio"],
            result["constraint_impedance"], result["table_static_speed_m_s"],
        ]).all()
        or min(
            float(result[key]) for key in (
                "time_constant_s", "damping_ratio", "constraint_impedance",
                "table_static_speed_m_s",
            )
        ) < 0.0
        or result["time_constant_s"] == 0.0
        or result["integration_order"] not in ("native", "physx_tgs")
        or result["friction_cone"] not in ("pyramidal", "elliptic")
        or result["noslip_iterations"] < 0
    ):
        raise ValueError("contact calibration is not a passing candidate-free audit")
    return result


def calibrated_pinch(
    path: Path, contact_profile: dict[str, object],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema in {
        "deximit_mujoco_pinch_audit_v2_contact_diagnostic_only",
        "deximit_mujoco_pinch_audit_v3_hand_contact_diagnostic_only",
    }:
        matching = [
            row for row in payload.get("all_profiles", [])
            if row.get("strict_gate_passed") is True
            and np.isclose(
                float(row.get("contact_time_constant_s", np.nan)),
                float(contact_profile["time_constant_s"]),
                atol=1.0e-12, rtol=0.0,
            )
            and np.isclose(
                float(row.get("contact_impedance", np.nan)),
                float(contact_profile["constraint_impedance"]),
                atol=1.0e-12, rtol=0.0,
            )
            and np.isclose(
                float(row.get(
                    "contact_damping_ratio", contact_profile["damping_ratio"],
                )),
                float(contact_profile["damping_ratio"]),
                atol=1.0e-12, rtol=0.0,
            )
            and row.get("cone") == contact_profile["friction_cone"]
            and int(row.get("noslip_iterations", -1))
            == int(contact_profile["noslip_iterations"])
        ]
        matching.sort(key=lambda row: float(row.get("score", np.inf)))
        if not matching:
            raise ValueError(
                "pinch calibration has no strict pass for the selected contact profile"
            )
        selected = matching[0]
    else:
        selected = payload.get("selected_profile", {})
    result = {
        "friction_cone": str(selected.get("cone", "")),
        "noslip_iterations": int(selected.get("noslip_iterations", -1)),
        "hand_static_speed_m_s": float(
            selected.get("hand_static_speed_threshold_m_s", np.nan),
        ),
        "contact_time_constant_s": float(
            selected.get(
                "contact_time_constant_s", contact_profile["time_constant_s"],
            ),
        ),
        "contact_impedance": float(
            selected.get(
                "contact_impedance", contact_profile["constraint_impedance"],
            ),
        ),
        "contact_damping_ratio": float(
            selected.get("contact_damping_ratio", contact_profile["damping_ratio"]),
        ),
        "score": float(selected.get("score", np.nan)),
    }
    if (
        schema not in {
            "deximit_mujoco_pinch_audit_v1_diagnostic_only",
            "deximit_mujoco_pinch_audit_v2_contact_diagnostic_only",
            "deximit_mujoco_pinch_audit_v3_hand_contact_diagnostic_only",
        }
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("grasp_candidate_executed") is not False
        or payload.get("any_strict_pass") is not True
        or selected.get("strict_gate_passed") is not True
        or result["friction_cone"] not in ("pyramidal", "elliptic")
        or result["noslip_iterations"] < 0
        or not np.isfinite([
            result["hand_static_speed_m_s"], result["contact_time_constant_s"],
            result["contact_damping_ratio"], result["contact_impedance"],
        ]).all()
        or result["hand_static_speed_m_s"] < 0.0
        or not np.isclose(
            result["contact_time_constant_s"], contact_profile["time_constant_s"],
            atol=1.0e-12, rtol=0.0,
        )
        or not np.isclose(
            result["contact_damping_ratio"], contact_profile["damping_ratio"],
            atol=1.0e-12, rtol=0.0,
        )
        or not np.isclose(
            result["contact_impedance"], contact_profile["constraint_impedance"],
            atol=1.0e-12, rtol=0.0,
        )
    ):
        raise ValueError("pinch calibration is not a passing candidate-free audit")
    return result


def validate_hand_pinch(
    path: Path, hand_profile: dict[str, object],
    friction_profile: dict[str, object],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matching = [
        row for row in payload.get("all_profiles", [])
        if row.get("strict_gate_passed") is True
        and np.isclose(
            float(row.get("contact_time_constant_s", np.nan)),
            float(hand_profile["time_constant_s"]), atol=1.0e-12, rtol=0.0,
        )
        and np.isclose(
            float(row.get("contact_damping_ratio", np.nan)),
            float(hand_profile["damping_ratio"]), atol=1.0e-12, rtol=0.0,
        )
        and np.isclose(
            float(row.get("contact_impedance", np.nan)),
            float(hand_profile["constraint_impedance"]),
            atol=1.0e-12, rtol=0.0,
        )
        and row.get("cone") == friction_profile["friction_cone"]
        and int(row.get("noslip_iterations", -1))
        == int(friction_profile["noslip_iterations"])
        and np.isclose(
            float(row.get("hand_static_speed_threshold_m_s", np.nan)),
            float(friction_profile["hand_static_speed_m_s"]),
            atol=1.0e-12, rtol=0.0,
        )
    ]
    if (
        payload.get("schema")
        != "deximit_mujoco_pinch_audit_v3_hand_contact_diagnostic_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("grasp_candidate_executed") is not False
        or len(matching) != 1
    ):
        raise ValueError("stronger hand contact lacks an exact candidate-free pinch pass")
    return matching[0]


def calibrated_hand_contact(path: Path, candidate: int) -> dict[str, object]:
    """Select the least disruptive stronger hand contact from one-step evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_trace = Path(str(payload.get("source_trace", ""))).resolve(strict=True)
    with np.load(source_trace, allow_pickle=False) as source:
        source_candidate = int(np.asarray(source["candidate_index"]).item())
    if (
        payload.get("schema")
        != "deximit_one_step_contact_audit_v4_hand_contact_grid_diagnostic_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("rendered") is not False
        or payload.get("source_trace_sha256") != sha256(source_trace)
        or source_candidate != candidate
    ):
        raise ValueError("hand-contact audit is not bound to this diagnostic candidate")
    rows = [
        row for row in payload.get("profiles", [])
        if row.get("constraint_order") == "contact_before_drive"
    ]
    baseline_rows = [
        row for row in rows if row.get("name") == "new_contact_new_friction"
    ]
    if len(baseline_rows) != 1:
        raise ValueError("hand-contact audit lacks one current-contact baseline")
    baseline_hand = float(
        baseline_rows[0]["all_contact"]["hand_qpos_rmse_rad"],
    )
    stronger = [
        row for row in rows
        if str(row.get("name", "")).startswith("hand_d")
        and float(row["all_contact"]["hand_qpos_rmse_rad"])
        <= 0.9 * baseline_hand
    ]
    stronger.sort(key=lambda row: (
        float(row["all_contact"]["linear_velocity_rmse_m_s"]),
        float(row["all_contact"]["position_rmse_m"]),
        float(row["all_contact"]["rotation_error_rad_mean"]),
    ))
    if not stronger:
        raise ValueError("hand-contact audit found no conservative stronger profile")
    selected = stronger[0]
    result = {
        "time_constant_s": float(selected.get("hand_time", np.nan)),
        "damping_ratio": float(selected.get("hand_damping", np.nan)),
        "constraint_impedance": float(selected.get("hand_impedance", np.nan)),
        "profile_name": str(selected.get("name", "")),
        "baseline_hand_qpos_rmse_rad": baseline_hand,
        "selected_hand_qpos_rmse_rad": float(
            selected["all_contact"]["hand_qpos_rmse_rad"],
        ),
        "selected_object_linear_velocity_rmse_m_s": float(
            selected["all_contact"]["linear_velocity_rmse_m_s"],
        ),
        "source_trace": str(source_trace),
    }
    if (
        not np.isfinite([
            result["time_constant_s"], result["damping_ratio"],
            result["constraint_impedance"], result["baseline_hand_qpos_rmse_rad"],
            result["selected_hand_qpos_rmse_rad"],
            result["selected_object_linear_velocity_rmse_m_s"],
        ]).all()
        or result["time_constant_s"] <= 0.0
        or result["damping_ratio"] <= 0.0
        or not 0.0 < result["constraint_impedance"] <= 1.0
    ):
        raise ValueError("selected hand-contact profile is malformed")
    return result


def validate_strong_friction_audit(
    path: Path, candidate: int,
) -> dict[str, object]:
    """Require a passing isolated loaded-hold audit at source frame timing."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "deximit_loaded_hold_audit_v1_diagnostic_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("formal_renderer_3_3_eligible") is not False
        or payload.get("grasp_candidate_advanced") is not False
        or payload.get("rendered") is not False
        or int(payload.get("state_source_candidate", -1)) != candidate
    ):
        raise ValueError("strong-friction evidence is not an isolated loaded-hold audit")
    matches = [
        profile for profile in payload.get("profiles", [])
        if profile.get("hand_friction_mode") == "strong_anchor"
        and np.isclose(
            float(profile.get("strong_anchor_time_constant_s", np.nan)),
            SOURCE_TIMESTEP_S, atol=1.0e-12, rtol=0.0,
        )
        and np.isclose(
            float(profile.get("hand_contact_damping_ratio", np.nan)),
            1.0, atol=1.0e-12, rtol=0.0,
        )
        and int(profile.get("noslip_iterations", -1)) == 1
    ]
    if len(matches) != 1:
        raise ValueError("strong-friction evidence lacks one source-frame profile")
    selected = matches[0]
    if (
        selected.get("passed") is not True
        or selected.get("opposed_contact_all_tail_frames") is not True
        or float(selected.get("minimum_hand_object_gap_m", -np.inf)) < -0.002
    ):
        raise ValueError("source-frame strong-friction profile did not pass")
    return selected


def validate_loaded_drive_probe(path: Path) -> None:
    with np.load(path, allow_pickle=False) as probe:
        if (
            str(np.asarray(probe["schema"]).item())
            != (
                "deximit_sapien_joint_drive_probe_"
                "v9_clean_loaded_limits_diagnostic_only"
            )
            or not bool(np.asarray(probe["diagnostic_only"]).item())
            or bool(np.asarray(probe["formal_renderer_3_3_eligible"]).item())
            or len(probe["joint_names"]) != 18
            or set(str(value) for value in probe["joint_names"])
            != set(ROBOT_JOINTS)
            or not np.allclose(probe["stiffness"], 1000.0)
            or not np.allclose(probe["damping"], 100.0)
            or np.asarray(probe["loaded_response_qpos"]).shape
            != (7, 3, 18, 24, 18)
            or np.asarray(probe["loaded_response_qvel"]).shape
            != (7, 3, 18, 24, 18)
            or int(np.asarray(probe["limit_response_steps"]).item()) != 24
            or np.asarray(probe["limit_response_start_qpos"]).shape
            != (18, 2, 18)
            or np.asarray(probe["limit_response_drive_target"]).shape
            != (18, 2, 18)
            or np.asarray(probe["limit_response_qpos"]).shape
            != (18, 2, 24, 18)
            or np.asarray(probe["limit_response_qvel"]).shape
            != (18, 2, 24, 18)
        ):
            raise ValueError("loaded SAPIEN drive probe violates its isolated contract")


def add_option(root: ET.Element, *, cone: str, noslip_iterations: int) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.attrib.update({
        "timestep": f"{MUJOCO_SUBSTEP_S:.17g}",
        "gravity": "0 0 -9.81",
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": "100",
        "tolerance": "1e-10",
        "cone": cone,
        "noslip_iterations": str(noslip_iterations),
    })
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    # PhysX uses a persistent contact manifold.  MuJoCo's multi-contact convex
    # collision is the closest available geometric counterpart.
    flag.set("multiccd", "enable")


def contact_distance_attributes() -> tuple[dict[str, str], str]:
    """Convert two 20 mm PhysX offsets for the installed MuJoCo semantics."""
    version = tuple(int(part) for part in mujoco.__version__.split(".")[:2])
    if version >= (3, 9):
        return {"margin": "0", "gap": "0.04"}, "mujoco_3_9_or_newer"
    return {"margin": "0.04", "gap": "0.04"}, "mujoco_3_8_or_older"


def vector(element: ET.Element | None, key: str) -> np.ndarray:
    value = "0 0 0" if element is None else element.get(key, "0 0 0")
    result = np.fromstring(value, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"invalid URDF {key} vector {value!r}")
    return result


def origin_transform(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", vector(element, "rpy")).as_matrix()
    result[:3, 3] = vector(element, "xyz")
    return result


def aggregate_fixed_link_inertias(
    root: ET.Element,
) -> dict[str, dict[str, object]]:
    """Combine fixed descendants exactly as SAPIEN's articulation does."""
    link_elements = {
        str(link.get("name")): link for link in root.findall("link")
    }
    fixed_children: dict[str, list[tuple[str, np.ndarray]]] = {}
    active_roots: list[str] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError("URDF joint lacks parent or child")
        parent_name = str(parent.get("link"))
        child_name = str(child.get("link"))
        if joint.get("type") == "fixed":
            fixed_children.setdefault(parent_name, []).append((
                child_name, origin_transform(joint.find("origin")),
            ))
        else:
            active_roots.append(child_name)

    result: dict[str, dict[str, object]] = {}
    for active_root in active_roots:
        members: list[tuple[str, np.ndarray]] = []
        stack = [(active_root, np.eye(4, dtype=np.float64))]
        while stack:
            link_name, active_from_link = stack.pop()
            members.append((link_name, active_from_link))
            for child_name, link_from_child in fixed_children.get(link_name, []):
                stack.append((child_name, active_from_link @ link_from_child))

        pieces: list[tuple[float, np.ndarray, np.ndarray]] = []
        for link_name, active_from_link in members:
            inertial = link_elements[link_name].find("inertial")
            if inertial is None:
                raise ValueError(f"preprocessed link {link_name!r} lacks inertia")
            mass_element = inertial.find("mass")
            inertia_element = inertial.find("inertia")
            if mass_element is None or inertia_element is None:
                raise ValueError(f"link {link_name!r} has incomplete inertia")
            mass = float(mass_element.get("value", "nan"))
            tensor = np.array([
                [float(inertia_element.get("ixx", "0")),
                 float(inertia_element.get("ixy", "0")),
                 float(inertia_element.get("ixz", "0"))],
                [float(inertia_element.get("ixy", "0")),
                 float(inertia_element.get("iyy", "0")),
                 float(inertia_element.get("iyz", "0"))],
                [float(inertia_element.get("ixz", "0")),
                 float(inertia_element.get("iyz", "0")),
                 float(inertia_element.get("izz", "0"))],
            ], dtype=np.float64)
            link_from_inertia = origin_transform(inertial.find("origin"))
            active_from_inertia = active_from_link @ link_from_inertia
            center = active_from_inertia[:3, 3]
            rotation = active_from_inertia[:3, :3]
            pieces.append((mass, center, rotation @ tensor @ rotation.T))

        total_mass = float(sum(piece[0] for piece in pieces))
        center = sum(piece[0] * piece[1] for piece in pieces) / total_mass
        tensor = np.zeros((3, 3), dtype=np.float64)
        for mass, piece_center, piece_tensor in pieces:
            displacement = piece_center - center
            tensor += piece_tensor + mass * (
                float(displacement @ displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        result[active_root] = {
            "mass": total_mass,
            "center": center,
            "tensor": tensor,
            "members": [name for name, _ in members],
        }
    return result


def compile_with_sapien_fallback_inertias(
    urdf: Path,
) -> tuple[
    mujoco.MjModel, list[str], list[str], dict[str, dict[str, object]],
]:
    """Mirror SAPIEN's URDF inertia loading before MuJoCo conversion.

    SAPIEN gives links without an ``inertial`` block a tiny default inertia.
    MuJoCo drops those links.  MuJoCo's URDF importer also ignores the ``rpy``
    orientation of an inertial frame, so bake that rotation into the full
    inertia tensor before asking MuJoCo to compile the URDF.
    """
    tree = ET.parse(urdf)
    root = tree.getroot()
    added: list[str] = []
    rotated: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = Path(mesh.get("filename", ""))
        if not filename.is_absolute():
            mesh.set("filename", str((urdf.parent / filename).resolve(strict=True)))
    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "1e-6"})
        ET.SubElement(inertial, "inertia", {
            "ixx": "1e-6", "iyy": "1e-6", "izz": "1e-6",
            "ixy": "0", "ixz": "0", "iyz": "0",
        })
        added.append(link.get("name", "<unnamed>"))
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        if origin is None or inertia is None:
            continue
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            raise ValueError(f"invalid inertial rpy on link {link.get('name')!r}")
        if np.max(np.abs(rpy)) <= 1.0e-12:
            continue
        tensor = np.array([
            [float(inertia.get("ixx", "0")), float(inertia.get("ixy", "0")),
             float(inertia.get("ixz", "0"))],
            [float(inertia.get("ixy", "0")), float(inertia.get("iyy", "0")),
             float(inertia.get("iyz", "0"))],
            [float(inertia.get("ixz", "0")), float(inertia.get("iyz", "0")),
             float(inertia.get("izz", "0"))],
        ], dtype=np.float64)
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        tensor = rotation @ tensor @ rotation.T
        inertia.attrib.update({
            "ixx": f"{tensor[0, 0]:.17g}",
            "iyy": f"{tensor[1, 1]:.17g}",
            "izz": f"{tensor[2, 2]:.17g}",
            "ixy": f"{tensor[0, 1]:.17g}",
            "ixz": f"{tensor[0, 2]:.17g}",
            "iyz": f"{tensor[1, 2]:.17g}",
        })
        origin.set("rpy", "0 0 0")
        rotated.append(link.get("name", "<unnamed>"))
    aggregates = aggregate_fixed_link_inertias(root)
    with tempfile.TemporaryDirectory(prefix="deximit_sapien_urdf_") as temporary:
        patched = Path(temporary) / "robot_with_sapien_fallback.urdf"
        tree.write(patched, encoding="unicode", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(patched))
    return model, added, rotated, aggregates


def compare_object_meshes(sapien_mesh: Path, mujoco_mesh: Path) -> dict[str, object]:
    """Require the MuJoCo-readable conversion to preserve SAPIEN geometry."""
    source = trimesh.load(sapien_mesh, force="mesh", process=False)
    converted = trimesh.load(mujoco_mesh, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or not isinstance(converted, trimesh.Trimesh):
        raise ValueError("object mesh input is not one triangle mesh")
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    converted_vertices = np.asarray(converted.vertices, dtype=np.float64)
    source_to_converted = cKDTree(converted_vertices).query(source_vertices)[0]
    converted_to_source = cKDTree(source_vertices).query(converted_vertices)[0]
    maximum = float(max(source_to_converted.max(), converted_to_source.max()))
    hull_volume_error = float(abs(source.convex_hull.volume - converted.convex_hull.volume))
    if maximum > 1.0e-8 or hull_volume_error > 1.0e-10:
        raise ValueError(
            "MuJoCo-readable object mesh is not geometrically equivalent to SAPIEN: "
            f"vertex_error={maximum:g}, hull_volume_error={hull_volume_error:g}"
        )
    return {
        "sapien_vertex_count": int(len(source_vertices)),
        "mujoco_vertex_count": int(len(converted_vertices)),
        "symmetric_nearest_vertex_error_m_max": maximum,
        "convex_hull_volume_m3_sapien": float(source.convex_hull.volume),
        "convex_hull_volume_m3_mujoco": float(converted.convex_hull.volume),
        "convex_hull_volume_error_m3": hull_volume_error,
        "passed": True,
    }


def build_scene(
    urdf: Path, sapien_object_mesh: Path, mujoco_object_mesh: Path,
    summary: Path, candidate: int, cooked_report: Path,
    contact_calibration: Path, pinch_calibration: Path, loaded_drive_probe: Path,
    hand_contact_audit: Path | None = None,
    hand_pinch_calibration: Path | None = None,
    strong_friction_audit: Path | None = None,
    allow_failed_candidate_properties: bool = False,
) -> tuple[ET.ElementTree, dict[str, object]]:
    cooked = cooked_collision_rows(cooked_report, sapien_object_mesh)
    contact_profile = calibrated_contact(contact_calibration)
    pinch_profile = calibrated_pinch(pinch_calibration, contact_profile)
    hand_contact_profile = (
        calibrated_hand_contact(hand_contact_audit, candidate)
        if hand_contact_audit is not None else None
    )
    if (hand_contact_profile is None) != (hand_pinch_calibration is None):
        raise ValueError(
            "candidate-specific hand contact and candidate-free pinch evidence "
            "must be supplied together"
        )
    hand_pinch_profile = (
        validate_hand_pinch(
            hand_pinch_calibration, hand_contact_profile, pinch_profile,
        )
        if hand_pinch_calibration is not None else None
    )
    strong_friction_profile = (
        validate_strong_friction_audit(strong_friction_audit, candidate)
        if strong_friction_audit is not None else None
    )
    if (
        pinch_profile["friction_cone"] != contact_profile["friction_cone"]
        or pinch_profile["noslip_iterations"]
        != contact_profile["noslip_iterations"]
    ):
        raise ValueError("object and pinch calibrations selected different friction solvers")
    validate_loaded_drive_probe(loaded_drive_probe)
    source_model, fallback_links, rotated_inertial_links, fixed_inertia_aggregates = (
        compile_with_sapien_fallback_inertias(urdf)
    )
    with tempfile.TemporaryDirectory(prefix="deximit_exact_mjcf_") as temporary:
        generated = Path(temporary) / "robot.xml"
        mujoco.mj_saveLastXML(str(generated), source_model)
        tree = ET.parse(generated)
    root = tree.getroot()
    source_joint_ranges: dict[str, tuple[float, float]] = {}
    for source_joint in ET.parse(urdf).getroot().findall("joint"):
        if source_joint.get("name") not in ROBOT_JOINTS:
            continue
        limit = source_joint.find("limit")
        if limit is None:
            raise ValueError(f"active joint {source_joint.get('name')!r} lacks limits")
        source_joint_ranges[str(source_joint.get("name"))] = (
            float(limit.get("lower", "nan")), float(limit.get("upper", "nan")),
        )
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in source_joint_ranges:
            joint.set(
                "range", " ".join(
                    f"{value:.17g}" for value in source_joint_ranges[str(name)]
                ),
            )
    if set(source_joint_ranges) != set(ROBOT_JOINTS):
        raise ValueError("source URDF does not define all 18 active joint limits")
    fixed_aggregate_rows: dict[str, dict[str, object]] = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        if name not in fixed_inertia_aggregates:
            continue
        aggregate = fixed_inertia_aggregates[str(name)]
        inertial = body.find("inertial")
        if inertial is None:
            raise ValueError(f"converted active body {name!r} lacks inertia")
        center = np.asarray(aggregate["center"], dtype=np.float64)
        tensor = np.asarray(aggregate["tensor"], dtype=np.float64)
        inertial.attrib.clear()
        inertial.attrib.update({
            "pos": " ".join(f"{value:.17g}" for value in center),
            "mass": f"{float(aggregate['mass']):.17g}",
            "fullinertia": " ".join(f"{value:.17g}" for value in (
                tensor[0, 0], tensor[1, 1], tensor[2, 2],
                tensor[0, 1], tensor[0, 2], tensor[1, 2],
            )),
        })
        fixed_aggregate_rows[str(name)] = {
            "members": aggregate["members"],
            "member_count": len(aggregate["members"]),
        }
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("MuJoCo URDF conversion omitted compiler settings")
    compiler.set("meshdir", str(urdf.parent))
    add_option(
        root, cone=str(contact_profile["friction_cone"]),
        noslip_iterations=int(contact_profile["noslip_iterations"]),
    )

    world = root.find("worldbody")
    asset = root.find("asset")
    if world is None or asset is None:
        raise ValueError("converted URDF lacks worldbody or assets")
    original_world = list(world)
    if not original_world:
        raise ValueError("converted URDF contains no robot geometry")
    robot_root = ET.Element("body", {
        "name": "robot_right_root",
        "pos": "0 -0.45 0.714",
        "gravcomp": "1",
    })
    for child in original_world:
        world.remove(child)
        robot_root.append(child)
    world.append(robot_root)

    for body in robot_root.iter("body"):
        body.set("gravcomp", "1")
    for joint in robot_root.iter("joint"):
        if joint.get("name") not in ROBOT_JOINTS:
            raise ValueError(f"unexpected active robot joint {joint.get('name')!r}")
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        joint.set("armature", "0")
        joint.set("actuatorfrclimited", "false")
        joint.attrib.pop("actuatorfrcrange", None)

    for owner, row in cooked.items():
        ET.SubElement(asset, "mesh", {
            "name": f"physx_cooked_{owner}", "file": str(row["mesh"]),
        })
    original_robot_geoms = list(robot_root.iter("geom"))
    parent_by_child = {
        child: parent for parent in robot_root.iter() for child in list(parent)
    }
    static_geom_elements = set(robot_root.findall("geom"))
    robot_geoms: list[str] = []
    static_robot_geoms: list[str] = []
    hand_geoms: list[str] = []
    for index, geom in enumerate(original_robot_geoms):
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_name not in COOKED_OWNER_BY_MESH:
            raise ValueError("exact URDF conversion produced a non-mesh robot collision")
        name = f"collision_robot_right_{mesh_name}_{index:02d}"
        if mesh_name in HAND_MESHES:
            name = f"collision_hand_{mesh_name}"
            hand_geoms.append(name)
        parent = parent_by_child.get(geom)
        if parent is None:
            raise RuntimeError("robot collision geom has no owning body")
        visual_attributes = dict(geom.attrib)
        visual_attributes.update({
            "name": name.replace("collision_", "visual_", 1),
            "mesh": mesh_name, "group": "2", "density": "0",
            "contype": "0", "conaffinity": "0",
        })
        ET.SubElement(parent, "geom", visual_attributes)
        cooked_owner = COOKED_OWNER_BY_MESH[mesh_name]
        geom.attrib.update({
            "name": name, "mesh": f"physx_cooked_{cooked_owner}",
            "group": "3", "density": "0", "contype": "0", "conaffinity": "0",
            "rgba": "0 1 0 0",
        })
        robot_geoms.append(name)
        if geom in static_geom_elements:
            static_robot_geoms.append(name)
    if len(hand_geoms) != len(HAND_MESHES) or len(set(hand_geoms)) != len(HAND_MESHES):
        raise ValueError(f"expected all 13 official hand collisions, got {hand_geoms}")

    ET.SubElement(asset, "mesh", {
        "name": "right_object_mesh", "file": str(mujoco_object_mesh),
    })
    metrics, candidate_passed = physical_row(
        summary, candidate, allow_failed=allow_failed_candidate_properties,
    )
    initial = np.asarray(metrics["object_initial_pose"], dtype=np.float64)
    inertia = np.asarray(metrics["object_inertia_kg_m2"], dtype=np.float64)
    cmass = np.asarray(metrics["object_cmass_local_pose_wxyz"], dtype=np.float64)
    if initial.shape != (4, 4) or inertia.shape != (3,) or cmass.shape != (7,):
        raise ValueError("SAPIEN object pose or inertia record is malformed")
    object_quat = Rotation.from_matrix(initial[:3, :3]).as_quat(scalar_first=True)
    object_body = ET.SubElement(world, "body", {
        "name": "right_object", "pos": " ".join(map(str, initial[:3, 3])),
        "quat": " ".join(map(str, object_quat)),
    })
    ET.SubElement(object_body, "joint", {
        "name": "right_object_joint", "type": "free", "damping": "0",
    })
    ET.SubElement(object_body, "inertial", {
        "pos": " ".join(map(str, cmass[:3])),
        "quat": " ".join(map(str, cmass[3:])),
        "mass": str(float(metrics["object_mass_kg"])),
        "diaginertia": " ".join(map(str, inertia)),
    })
    ET.SubElement(object_body, "geom", {
        "name": "right_object_visual", "type": "mesh", "mesh": "right_object_mesh",
        "density": "0", "contype": "0", "conaffinity": "0",
        "rgba": "0.78 0.32 0.16 1",
    })
    ET.SubElement(object_body, "geom", {
        "name": "right_object_single", "type": "mesh",
        "mesh": "physx_cooked_right_object",
        "density": "0", "group": "3", "contype": "0", "conaffinity": "0",
        "rgba": "0 1 0 0",
    })
    table = ET.SubElement(world, "geom", {
        "name": "table", "type": "box", "pos": "0.3 0 0.684",
        "size": "0.6 0.8 0.03", "density": "0",
        "rgba": "0.706 0.667 0.627 1", "contype": "0", "conaffinity": "0",
    })
    del table
    ET.SubElement(world, "camera", {
        "name": "front", "mode": "fixed", "pos": "0.031 0.941 1.558",
        "quat": "0.0150738 0.00673439 0.407848 0.912901",
    })

    contact = ET.SubElement(root, "contact")
    contact_distance, contact_semantics = contact_distance_attributes()
    contact_solver = {
        "solref": (
            f"{contact_profile['time_constant_s']} "
            f"{contact_profile['damping_ratio']}"
        ),
        "solimp": (
            f"{contact_profile['constraint_impedance']} "
            f"{contact_profile['constraint_impedance']} 0.001 0.5 2"
        ),
    }
    hand_contact_solver = contact_solver
    if hand_contact_profile is not None:
        hand_contact_solver = {
            "solref": (
                f"{hand_contact_profile['time_constant_s']} "
                f"{hand_contact_profile['damping_ratio']}"
            ),
            "solimp": (
                f"{hand_contact_profile['constraint_impedance']} "
                f"{hand_contact_profile['constraint_impedance']} 0.001 0.5 2"
            ),
        }
    # PhysX's default material combine rule averages the two materials.  XML
    # stores the kinetic value; runtime switches to the static value only in
    # the sticking regime because MuJoCo has no separate static coefficient.
    ET.SubElement(contact, "pair", {
        "name": "table_right_object", "geom1": "table", "geom2": "right_object_single",
        "condim": "3", "friction": "0.75 0.75 0 0 0",
        **contact_solver, **contact_distance,
    })
    for geom in hand_geoms:
        ET.SubElement(contact, "pair", {
            "name": f"{geom}_right_object", "geom1": geom,
            "geom2": "right_object_single", "condim": "3",
            "friction": "0.5 0.5 0 0 0",
            **hand_contact_solver, **contact_distance,
        })
    for geom in robot_geoms:
        # MuJoCo rejects a contact between two fixed bodies.  DexImit's base
        # collision is fixed to the world and overlaps the fixed table, so do
        # not construct that physically meaningless pair.
        if geom not in static_robot_geoms:
            ET.SubElement(contact, "pair", {
                "name": f"table_{geom}", "geom1": "table", "geom2": geom,
                "condim": "3", "friction": "0.75 0.75 0 0 0",
                **contact_solver, **contact_distance,
            })
        if geom not in hand_geoms:
            ET.SubElement(contact, "pair", {
                "name": f"{geom}_right_object", "geom1": geom,
                "geom2": "right_object_single", "condim": "3",
                "friction": "0.5 0.5 0 0 0",
                **contact_solver, **contact_distance,
            })

    actuator = ET.SubElement(root, "actuator")
    for joint in ROBOT_JOINTS:
        ET.SubElement(actuator, "general", {
            "name": f"drive_{joint}", "joint": joint,
            "gaintype": "fixed", "biastype": "affine",
            "gainprm": "1000", "biasprm": "0 -1000 -100",
            "ctrllimited": "false", "forcelimited": "false",
        })

    provenance = {
        "schema": (
            "deximit_exact_urdf_mujoco_scene_v11_failed_candidate_properties_"
            "diagnostic_only"
            if not candidate_passed else (
                "deximit_exact_urdf_mujoco_scene_v10_bound_strong_friction_"
                "diagnostic_only"
                if strong_friction_profile is not None else
                "deximit_exact_urdf_mujoco_scene_v9_bound_friction_drive_limits_"
                "diagnostic_only"
            )
        ),
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "candidate_used_only_for_recorded_object_properties": candidate,
        "candidate_original_sapien_pass": candidate_passed,
        "failed_candidate_properties_allowed": allow_failed_candidate_properties,
        "sapien_robot_urdf": str(urdf),
        "sapien_robot_urdf_sha256": sha256(urdf),
        "sapien_object_mesh": str(sapien_object_mesh),
        "sapien_object_mesh_sha256": sha256(sapien_object_mesh),
        "mujoco_object_mesh": str(mujoco_object_mesh),
        "mujoco_object_mesh_sha256": sha256(mujoco_object_mesh),
        "object_geometry_conversion": compare_object_meshes(
            sapien_object_mesh, mujoco_object_mesh,
        ),
        "sapien_cooked_collision_report": str(cooked_report),
        "sapien_cooked_collision_report_sha256": sha256(cooked_report),
        "sapien_cooked_collision_shape_count": len(cooked),
        "sapien_cooked_collision_vertex_counts": {
            owner: int(row["cooked_vertex_count"]) for owner, row in cooked.items()
        },
        "physics_mesh_policy": (
            "original high-resolution meshes remain visual-only; explicit-pair "
            "collision uses the vertices and faces exported from PhysX cooking"
        ),
        "sapien_summary": str(summary),
        "sapien_summary_sha256": sha256(summary),
        "robot_joint_tree": "same URDF as SAPIEN; no virtual Cartesian wrist",
        "active_joint_limits_restored_at_full_source_precision": True,
        "sapien_missing_inertial_fallback": {
            "mass_kg": 1.0e-6,
            "diagonal_inertia_kg_m2": [1.0e-6, 1.0e-6, 1.0e-6],
            "links": fallback_links,
            "link_count": len(fallback_links),
        },
        "urdf_inertial_frame_rotation_baked_before_mujoco_import": {
            "links": rotated_inertial_links,
            "link_count": len(rotated_inertial_links),
        },
        "fixed_link_inertias_reaggregated_from_urdf": fixed_aggregate_rows,
        "robot_gravity": "disabled by gravcomp=1, matching every SAPIEN robot link",
        "robot_passive_joint_friction": 0.0,
        "robot_passive_joint_damping": 0.0,
        "robot_force_limit": "disabled, matching PhysX force_limit=1e10",
        "drive_conversion": {
            "source": str(loaded_drive_probe),
            "source_sha256": sha256(loaded_drive_probe),
            "runtime_converter_implementation": str(
                Path(__file__).resolve().with_name("sapien_equiv.py")
            ),
            "runtime_converter_implementation_sha256": sha256(
                Path(__file__).resolve().with_name("sapien_equiv.py")
            ),
            "source_stiffness": 1000.0,
            "source_damping": 100.0,
            "method": (
                "PhysX TGS frame-persistent scalar drive impulses: cache the "
                "articulated unit response and initial error once per 1/240 s "
                "frame, accumulate impulse and joint displacement across 25 "
                "position iterations, and interleave unilateral limit rows"
            ),
            "reference": (
                "NVIDIA PhysX DyCpuGpuArticulation.h "
                "computeImplicitDriveParamsForceDrive/computeDriveImpulse"
            ),
            "native_mujoco_actuators_disabled_at_runtime": True,
            "joint_limit_conversion": (
                "after every scalar drive row, cancel predicted outward limit "
                "motion through the coupled inverse mass; project only residual "
                "floating-point position error to the measured hard bound"
            ),
            "native_mujoco_joint_limits_disabled_at_runtime": True,
        },
        "strong_friction_conversion": (
            {
                "source": str(strong_friction_audit),
                "source_sha256": sha256(strong_friction_audit),
                "runtime_converter_implementation": str(
                    Path(__file__).resolve().with_name("sapien_equiv.py")
                ),
                "runtime_converter_implementation_sha256": sha256(
                    Path(__file__).resolve().with_name("sapien_equiv.py")
                ),
                "mode": "one retained tangential anchor per convex hand-object pair",
                "static_friction": 0.7,
                "time_constant_s": SOURCE_TIMESTEP_S,
                "correlation_distance_m": 0.025,
                "runtime_hand_contact_damping_ratio": 1.0,
                "loaded_hold_passed": True,
                "loaded_hold_minimum_height_change_m": (
                    strong_friction_profile["minimum_height_change_m"]
                ),
                "loaded_hold_tail_vertical_span_m": (
                    strong_friction_profile["tail_vertical_span_m"]
                ),
                "loaded_hold_minimum_hand_object_gap_m": (
                    strong_friction_profile["minimum_hand_object_gap_m"]
                ),
            }
            if strong_friction_profile is not None else None
        ),
        "table_center_m": [0.3, 0.0, 0.684],
        "table_half_size_m": [0.6, 0.8, 0.03],
        "table_top_m": 0.714,
        "robot_root_position_m": [0.0, -0.45, 0.714],
        "official_hand_collision_mesh_count": len(hand_geoms),
        "fixed_robot_collision_pairs_omitted": static_robot_geoms,
        "convex_multicontact": True,
        "mujoco_version_used_to_generate_contact_semantics": mujoco.__version__,
        "mujoco_contact_margin_gap_semantics": contact_semantics,
        "contact_offset_conversion": {
            "sapien_contact_offset_per_shape_m": 0.02,
            "sapien_rest_offset_per_shape_m": 0.0,
            "mujoco_xml_attributes": contact_distance,
            "meaning": "generate contacts within 40 mm; activate force at zero separation",
        },
        "contact_calibration": {
            "source": str(contact_calibration),
            "source_sha256": sha256(contact_calibration),
            "mujoco_time_constant_s": contact_profile["time_constant_s"],
            "damping_ratio": contact_profile["damping_ratio"],
            "constant_constraint_impedance": contact_profile["constraint_impedance"],
            "table_static_speed_threshold_m_s": (
                contact_profile["table_static_speed_m_s"]
            ),
            "integration_order": contact_profile["integration_order"],
            "friction_cone": contact_profile["friction_cone"],
            "noslip_iterations": contact_profile["noslip_iterations"],
            "penetration_hard_condition_m": -2.5e-4,
            "net_peak_impulse_relative_error_limit": 0.25,
        },
        "pinch_friction_calibration": {
            "source": str(pinch_calibration),
            "source_sha256": sha256(pinch_calibration),
            "friction_cone": pinch_profile["friction_cone"],
            "noslip_iterations": pinch_profile["noslip_iterations"],
            "hand_static_speed_threshold_m_s": (
                pinch_profile["hand_static_speed_m_s"]
            ),
            "validated_contact_time_constant_s": (
                pinch_profile["contact_time_constant_s"]
            ),
            "validated_contact_damping_ratio": (
                pinch_profile["contact_damping_ratio"]
            ),
            "validated_contact_impedance": pinch_profile["contact_impedance"],
            "selected_profile_score": pinch_profile["score"],
        },
        "hand_contact_calibration": (
            {
                "source": str(hand_contact_audit),
                "source_sha256": sha256(hand_contact_audit),
                "candidate_specific_diagnostic": candidate,
                "source_trace": hand_contact_profile["source_trace"],
                "profile_name": hand_contact_profile["profile_name"],
                "mujoco_time_constant_s": hand_contact_profile["time_constant_s"],
                "damping_ratio": hand_contact_profile["damping_ratio"],
                "constant_constraint_impedance": (
                    hand_contact_profile["constraint_impedance"]
                ),
                "baseline_hand_qpos_rmse_rad": (
                    hand_contact_profile["baseline_hand_qpos_rmse_rad"]
                ),
                "selected_hand_qpos_rmse_rad": (
                    hand_contact_profile["selected_hand_qpos_rmse_rad"]
                ),
                "selected_object_linear_velocity_rmse_m_s": (
                    hand_contact_profile[
                        "selected_object_linear_velocity_rmse_m_s"
                    ]
                ),
                "candidate_free_pinch_source": str(hand_pinch_calibration),
                "candidate_free_pinch_source_sha256": sha256(
                    hand_pinch_calibration
                ),
                "candidate_free_pinch_strict_passed": True,
                "candidate_free_pinch_score": hand_pinch_profile["score"],
                "candidate_free_pinch_minimum_gap_m": (
                    hand_pinch_profile["minimum_paddle_object_gap_m"]
                ),
                "selection_policy": (
                    "require at least 10% lower one-step hand qpos RMSE, then "
                    "choose the lowest object linear-velocity RMSE"
                ),
            }
            if hand_contact_profile is not None else {
                "source": None,
                "inherits_object_contact_response": True,
            }
        ),
        "tgs_time_conversion": {
            "source_timestep_s": SOURCE_TIMESTEP_S,
            "source_position_iterations": PHYSX_POSITION_ITERATIONS,
            "mujoco_substep_s": MUJOCO_SUBSTEP_S,
            "mujoco_substeps_per_source_step": PHYSX_POSITION_ITERATIONS,
            "basis": (
                "PhysX implicit-drive documentation: TGS position iterations "
                "advance time as equal substeps"
            ),
        },
        "camera": {"count": 1, "name": "front", "mode": "fixed"},
        "remaining_runtime_conversions": [
            "PhysX rigid-body damping to MuJoCo generalized damping",
            "PhysX speculative contact and TGS response to MuJoCo contact response",
            (
                "PhysX strong-friction history to retained MuJoCo external-force "
                "anchors; table friction remains speed-dependent"
                if strong_friction_profile is not None else
                "separate PhysX static/dynamic friction to speed-dependent MuJoCo friction"
            ),
        ],
    }
    return tree, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sapien-urdf", type=Path, required=True)
    parser.add_argument("--sapien-object-mesh", type=Path, required=True)
    parser.add_argument("--mujoco-object-mesh", type=Path, required=True)
    parser.add_argument("--sapien-summary", type=Path, required=True)
    parser.add_argument("--sapien-cooked-collisions", type=Path, required=True)
    parser.add_argument("--contact-calibration", type=Path, required=True)
    parser.add_argument("--pinch-calibration", type=Path, required=True)
    parser.add_argument("--sapien-loaded-drive-probe", type=Path, required=True)
    parser.add_argument("--hand-contact-audit", type=Path)
    parser.add_argument("--hand-pinch-calibration", type=Path)
    parser.add_argument("--strong-friction-audit", type=Path)
    parser.add_argument(
        "--allow-failed-candidate-properties", action="store_true",
        help=(
            "Diagnostic-only: permit a unique failed SAPIEN attempt to supply "
            "object mass, inertia, and initial pose. The default still requires "
            "a passing candidate."
        ),
    )
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    urdf = args.sapien_urdf.expanduser().resolve(strict=True)
    sapien_mesh = args.sapien_object_mesh.expanduser().resolve(strict=True)
    mujoco_mesh = args.mujoco_object_mesh.expanduser().resolve(strict=True)
    summary = args.sapien_summary.expanduser().resolve(strict=True)
    cooked_report = args.sapien_cooked_collisions.expanduser().resolve(strict=True)
    contact_calibration = args.contact_calibration.expanduser().resolve(strict=True)
    pinch_calibration = args.pinch_calibration.expanduser().resolve(strict=True)
    loaded_drive_probe = args.sapien_loaded_drive_probe.expanduser().resolve(strict=True)
    hand_contact_audit = (
        args.hand_contact_audit.expanduser().resolve(strict=True)
        if args.hand_contact_audit is not None else None
    )
    hand_pinch_calibration = (
        args.hand_pinch_calibration.expanduser().resolve(strict=True)
        if args.hand_pinch_calibration is not None else None
    )
    strong_friction_audit = (
        args.strong_friction_audit.expanduser().resolve(strict=True)
        if args.strong_friction_audit is not None else None
    )
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".provenance.json").exists():
        raise FileExistsError(f"refusing to overwrite isolated scene {output}")
    tree, provenance = build_scene(
        urdf, sapien_mesh, mujoco_mesh, summary, args.candidate,
        cooked_report, contact_calibration, pinch_calibration, loaded_drive_probe,
        hand_contact_audit, hand_pinch_calibration,
        strong_friction_audit,
        args.allow_failed_candidate_properties,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(output))
    camera_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ]
    if model.nq != 25 or model.nu != 18 or camera_names != ["front"]:
        raise RuntimeError(
            f"compiled exact scene contract failed: nq={model.nq}, nu={model.nu}, "
            f"cameras={camera_names}"
        )
    provenance.update({
        "output": str(output), "output_sha256": sha256(output),
        "compiled_nq": model.nq, "compiled_nv": model.nv,
        "compiled_nu": model.nu, "compiled_ngeom": model.ngeom,
    })
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
