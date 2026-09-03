#!/usr/bin/env python3
"""Audit DexImit-to-current-XHand frames, joints, and collision geometry.

This is a diagnostic-only bridge.  It proves coordinate and kinematic
correspondence before any SAPIEN result is replayed in MuJoCo.  Equality of
the official BODex and SAPIEN collision meshes is checked separately from the
deliberately simplified MuJoCo collision proxy; the latter is measured, never
silently treated as an equivalent mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for import_root in (ROOT, HERE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bodex_triptych import (  # noqa: E402
    BASE_TO_HAND_ROTATION,
    BASE_TO_HAND_TRANSLATION_M,
    BODEX_TO_MUJOCO_JOINT_SIGN,
)
from generate_bodex_candidates import CANONICAL_JOINTS  # noqa: E402


SCHEMA = "deximit_xhand_alignment_audit_v1_diagnostic_only"
LINKS = (
    "right_hand_link",
    "right_hand_thumb_bend_link",
    "right_hand_thumb_rota_link1",
    "right_hand_thumb_rota_link2",
    "right_hand_index_bend_link",
    "right_hand_index_rota_link1",
    "right_hand_index_rota_link2",
    "right_hand_mid_link1",
    "right_hand_mid_link2",
    "right_hand_ring_link1",
    "right_hand_ring_link2",
    "right_hand_pinky_link1",
    "right_hand_pinky_link2",
)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    moving: bool


@dataclass(frozen=True)
class Urdf:
    path: Path
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    collisions: dict[str, tuple[tuple[Path, np.ndarray, np.ndarray], ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodex-urdf", type=Path, required=True)
    parser.add_argument("--sapien-urdf", type=Path, required=True)
    parser.add_argument("--mujoco-scene", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--contact-v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--surface-samples", type=int, default=1200)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def origin(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.eye(4, dtype=np.float64)
    xyz = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError("URDF origin must contain three xyz and three rpy values")
    return transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


def parse_urdf(path: Path) -> Urdf:
    source = path.resolve(strict=True)
    tree = ET.parse(source)
    root = tree.getroot()
    links = tuple(node.attrib["name"] for node in root.findall("link"))
    joints: list[Joint] = []
    collisions: dict[str, tuple[tuple[Path, np.ndarray, np.ndarray], ...]] = {}
    for node in root.findall("joint"):
        kind = node.attrib.get("type", "fixed")
        axis_node = node.find("axis")
        axis = np.fromstring(
            axis_node.attrib.get("xyz", "0 0 0") if axis_node is not None else "0 0 0",
            sep=" ", dtype=np.float64,
        )
        limit = node.find("limit")
        moving = kind in {"revolute", "continuous", "prismatic"}
        lower = float(limit.attrib.get("lower", "-inf")) if limit is not None else -np.inf
        upper = float(limit.attrib.get("upper", "inf")) if limit is not None else np.inf
        joints.append(Joint(
            name=node.attrib["name"],
            parent=node.find("parent").attrib["link"],
            child=node.find("child").attrib["link"],
            origin=origin(node.find("origin")), axis=axis,
            lower=lower, upper=upper, moving=moving,
        ))
    for node in root.findall("link"):
        items: list[tuple[Path, np.ndarray, np.ndarray]] = []
        for collision in node.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
            filename = mesh.attrib["filename"].replace("package://", "")
            mesh_path = (source.parent / filename).resolve(strict=True)
            items.append((mesh_path, scale, origin(collision.find("origin"))))
        collisions[node.attrib["name"]] = tuple(items)
    return Urdf(source, links, tuple(joints), collisions)


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("moving URDF joint has a zero axis")
    return Rotation.from_rotvec(axis / norm * angle).as_matrix()


def forward(urdf: Urdf, values: dict[str, float], root: str) -> dict[str, np.ndarray]:
    result = {root: np.eye(4, dtype=np.float64)}
    pending = list(urdf.joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            if joint.parent not in result:
                continue
            motion = np.eye(4, dtype=np.float64)
            if joint.moving:
                motion[:3, :3] = axis_rotation(joint.axis, float(values.get(joint.name, 0.0)))
            result[joint.child] = result[joint.parent] @ joint.origin @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            break
    return result


def relative(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.inv(first) @ second


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())


def mj_body_transform(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f"MuJoCo body is missing: {name}")
    return transform(data.xmat[body].reshape(3, 3), data.xpos[body])


def mj_link_poses(
    model: mujoco.MjModel, data: mujoco.MjData, bodex_values: dict[str, float],
) -> dict[str, np.ndarray]:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    for index, name in enumerate(CANONICAL_JOINTS):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"MuJoCo joint is missing: {name}")
        data.qpos[int(model.jnt_qposadr[joint])] = (
            float(bodex_values[name]) * float(BODEX_TO_MUJOCO_JOINT_SIGN[index])
        )
    mujoco.mj_forward(model, data)
    base = mj_body_transform(model, data, "right_hand_link")
    return {name: relative(base, mj_body_transform(model, data, name)) for name in LINKS}


def joint_intervals(urdf: Urdf, model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    source = {joint.name: joint for joint in urdf.joints}
    result: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(CANONICAL_JOINTS):
        if name not in source:
            raise ValueError(f"BODex URDF is missing joint {name}")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low_mj, high_mj = (float(value) for value in model.jnt_range[joint])
        sign = float(BODEX_TO_MUJOCO_JOINT_SIGN[index])
        mapped = sorted((sign * low_mj, sign * high_mj))
        low = max(source[name].lower, mapped[0])
        high = min(source[name].upper, mapped[1])
        if not np.isfinite((low, high)).all() or low >= high:
            raise ValueError(f"joint {name} has no finite common range")
        result[name] = (low, high)
    return result


def samples(intervals: dict[str, tuple[float, float]], seed: int) -> list[dict[str, float]]:
    middle = {name: 0.5 * (low + high) for name, (low, high) in intervals.items()}
    result = [middle]
    for name, (low, high) in intervals.items():
        for fraction in (0.2, 0.8):
            row = middle.copy()
            row[name] = low + fraction * (high - low)
            result.append(row)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        result.append({name: float(rng.uniform(low, high)) for name, (low, high) in intervals.items()})
    return result


def mesh_for_collision(entries: tuple[tuple[Path, np.ndarray, np.ndarray], ...]) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for path, scale, pose in entries:
        mesh = trimesh.load(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"collision asset is not one triangle mesh: {path}")
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("link has no mesh collision")
    return trimesh.util.concatenate(meshes)


def primitive_mesh(model: mujoco.MjModel, geom: int) -> trimesh.Trimesh | None:
    kind = int(model.geom_type[geom])
    size = np.asarray(model.geom_size[geom], dtype=np.float64)
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
    else:
        return None
    pose = transform(
        Rotation.from_quat(model.geom_quat[geom], scalar_first=True).as_matrix(),
        model.geom_pos[geom],
    )
    mesh.apply_transform(pose)
    return mesh


def mj_collision_mesh(model: mujoco.MjModel, link: str) -> trimesh.Trimesh | None:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    meshes: list[trimesh.Trimesh] = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body:
            continue
        # This scene uses explicit named contact pairs, so physical geoms may
        # intentionally have zero contype/conaffinity.  Geometry type, rather
        # than broad collision masks, separates the primitive proxy from the
        # non-contact visual mesh.
        mesh = primitive_mesh(model, geom)
        if mesh is not None:
            meshes.append(mesh)
    return trimesh.util.concatenate(meshes) if meshes else None


def surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    return np.asarray(trimesh.sample.sample_surface(mesh, count, seed=seed)[0], dtype=np.float64)


def distances(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    forward_distance = cKDTree(second).query(first, workers=-1)[0]
    backward_distance = cKDTree(first).query(second, workers=-1)[0]
    both = np.concatenate((forward_distance, backward_distance))
    return {
        "symmetric_mean_m": float(both.mean()),
        "symmetric_p95_m": float(np.quantile(both, 0.95)),
        "symmetric_max_m": float(both.max()),
        "official_to_proxy_p95_m": float(np.quantile(forward_distance, 0.95)),
        "proxy_to_official_p95_m": float(np.quantile(backward_distance, 0.95)),
    }


def mesh_identity(bodex: Urdf, sapien: Urdf) -> tuple[dict[str, object], bool]:
    rows: dict[str, object] = {}
    passed = True
    for link in LINKS:
        first = bodex.collisions.get(link, ())
        second = sapien.collisions.get(link, ())
        hashes_first = [sha256(item[0]) for item in first]
        hashes_second = [sha256(item[0]) for item in second]
        same = bool(first) and len(first) == len(second) and hashes_first == hashes_second
        passed &= same
        rows[link] = {
            "bodex_meshes": [str(item[0]) for item in first],
            "sapien_meshes": [str(item[0]) for item in second],
            "bodex_sha256": hashes_first,
            "sapien_sha256": hashes_second,
            "exact_file_identity": same,
        }
    return rows, bool(passed)


def contact_anchor(path: Path) -> tuple[int, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["schema"]).item()) != "taco_mano_surface_contact_v3_conservative_geometric_evidence":
            raise ValueError("alignment audit requires conservative surface-contact v3")
        states = np.asarray(data["state"], dtype=np.int8)
        hands = [str(value) for value in data["hand_order"]]
        regions = [str(value) for value in data["region_order"]]
    right = hands.index("right")
    finger_names = [name for name in regions if name != "palm"]
    state = states[:, right]
    thumb = state[:, regions.index("thumb")] == 1
    other = np.any(state[:, [regions.index(name) for name in finger_names if name != "thumb"]] == 1, axis=1)
    rows = np.flatnonzero(thumb & other)
    if not len(rows):
        raise ValueError("v3 has no right-hand thumb-plus-other surface-contact row")
    anchor = int(rows[0])
    active = [name for name in finger_names if state[anchor, regions.index(name)] == 1]
    return anchor, active


def human_prompt(path: Path, anchor: int) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        wrist = np.asarray(data["T_sim_wrist_target"], dtype=np.float64)
        hands = [str(value) for value in data["hand_order"]]
        objects = np.asarray(data["T_sim_object_reference"], dtype=np.float64)
    right = hands.index("right")
    wrist = wrist[:, right]
    obj = objects[:, 0]
    fixed = Rotation.from_euler("x", np.pi / 2.0).as_matrix() @ np.diag((-1.0, 1.0, -1.0))
    local_to_wrist = transform(fixed)
    prompt = wrist @ np.linalg.inv(local_to_wrist)[None]
    reconstructed = prompt @ local_to_wrist[None]
    position_error = np.linalg.norm(reconstructed[:, :3, 3] - wrist[:, :3, 3], axis=1)
    angle_error = np.asarray([
        rotation_error(reconstructed[row], wrist[row]) for row in range(len(wrist))
    ])
    report = {
        "definition": (
            "DexImit right-hand local-to-world prompt reconstructed from the official "
            "MANO wrist frame and DexImit's fixed right-hand canonicalization"
        ),
        "anchor_row": anchor,
        "maximum_wrist_reconstruction_position_error_m": float(position_error.max()),
        "maximum_wrist_reconstruction_rotation_error_rad": float(angle_error.max()),
        "anchor_prompt_world": prompt[anchor].tolist(),
        "anchor_prompt_object_local": (np.linalg.inv(obj[anchor]) @ prompt[anchor]).tolist(),
        "fixed_wilor_local_to_deximit_local_rotation": fixed.tolist(),
    }
    return prompt, report


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.surface_samples < 100:
        raise ValueError("surface sample count is too small for a collision-envelope audit")
    bodex = parse_urdf(args.bodex_urdf)
    sapien = parse_urdf(args.sapien_urdf)
    model = mujoco.MjModel.from_xml_path(str(args.mujoco_scene.resolve(strict=True)))
    data = mujoco.MjData(model)
    intervals = joint_intervals(bodex, model)
    test_rows = samples(intervals, args.seed)

    link_errors: dict[str, dict[str, float]] = {
        name: {"maximum_position_error_m": 0.0, "maximum_rotation_error_rad": 0.0}
        for name in LINKS
    }
    for values in test_rows:
        source = forward(bodex, values, "base")
        root = source["right_hand_link"]
        source_relative = {name: relative(root, source[name]) for name in LINKS}
        target = mj_link_poses(model, data, values)
        for name in LINKS:
            position = float(np.linalg.norm(source_relative[name][:3, 3] - target[name][:3, 3]))
            angle = rotation_error(source_relative[name], target[name])
            link_errors[name]["maximum_position_error_m"] = max(
                link_errors[name]["maximum_position_error_m"], position,
            )
            link_errors[name]["maximum_rotation_error_rad"] = max(
                link_errors[name]["maximum_rotation_error_rad"], angle,
            )
    kinematic_position = max(row["maximum_position_error_m"] for row in link_errors.values())
    kinematic_rotation = max(row["maximum_rotation_error_rad"] for row in link_errors.values())
    kinematic_pass = kinematic_position <= 1.0e-4 and kinematic_rotation <= np.deg2rad(0.1)

    base_pose = forward(bodex, {}, "base")["right_hand_link"]
    constant_pose = transform(BASE_TO_HAND_ROTATION, BASE_TO_HAND_TRANSLATION_M)
    base_position_error = float(np.linalg.norm(base_pose[:3, 3] - constant_pose[:3, 3]))
    base_rotation_error = rotation_error(base_pose, constant_pose)
    sapien_zero = forward(sapien, {}, "base")
    same_base_position = float(np.linalg.norm(
        sapien_zero["right_hand_link"][:3, 3] - base_pose[:3, 3],
    ))
    same_base_rotation = rotation_error(sapien_zero["right_hand_link"], base_pose)

    identity_rows, official_mesh_pass = mesh_identity(bodex, sapien)
    middle = test_rows[0]
    official_pose = forward(bodex, middle, "base")
    official_root = official_pose["right_hand_link"]
    official_relative = {name: relative(official_root, official_pose[name]) for name in LINKS}
    proxy_pose = mj_link_poses(model, data, middle)
    proxy_rows: dict[str, object] = {}
    all_official: list[np.ndarray] = []
    all_proxy: list[np.ndarray] = []
    for link_index, link in enumerate(LINKS):
        official_mesh = mesh_for_collision(bodex.collisions[link])
        proxy_mesh = mj_collision_mesh(model, link)
        official_points = surface(official_mesh, args.surface_samples, args.seed + link_index)
        official_world = (
            official_points @ official_relative[link][:3, :3].T
            + official_relative[link][:3, 3]
        )
        all_official.append(official_world)
        if proxy_mesh is None:
            proxy_rows[link] = {
                "current_proxy_present": False,
                "note": "official SAPIEN mesh has no physical MuJoCo proxy on this body",
            }
            continue
        proxy_points = surface(proxy_mesh, args.surface_samples, args.seed + 100 + link_index)
        proxy_world = proxy_points @ proxy_pose[link][:3, :3].T + proxy_pose[link][:3, 3]
        all_proxy.append(proxy_world)
        proxy_rows[link] = {
            "current_proxy_present": True,
            **distances(official_world, proxy_world),
        }
    union_distance = distances(np.concatenate(all_official), np.concatenate(all_proxy))

    anchor, active = contact_anchor(args.contact_v3)
    _, prompt_report = human_prompt(args.human_reference, anchor)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link")
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    hand_pose = mj_body_transform(model, data, "right_hand_link")
    palm_pose = transform(data.site_xmat[palm].reshape(3, 3), data.site_xpos[palm])
    hand_to_palm = relative(hand_pose, palm_pose)

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "inputs": {
            "bodex_urdf": str(bodex.path),
            "bodex_urdf_sha256": sha256(bodex.path),
            "sapien_urdf": str(sapien.path),
            "sapien_urdf_sha256": sha256(sapien.path),
            "mujoco_scene": str(args.mujoco_scene.resolve()),
            "mujoco_scene_sha256": sha256(args.mujoco_scene.resolve()),
            "human_reference": str(args.human_reference.resolve()),
            "human_reference_sha256": sha256(args.human_reference.resolve()),
            "contact_v3": str(args.contact_v3.resolve()),
            "contact_v3_sha256": sha256(args.contact_v3.resolve()),
        },
        "contact_derived_prompt": {
            "anchor_row": anchor,
            "active_fingers_at_anchor": active,
            "derived_finger_count": len(active),
            **prompt_report,
        },
        "wrist_and_palm": {
            "bodex_base_to_hand_position_error_m": base_position_error,
            "bodex_base_to_hand_rotation_error_rad": base_rotation_error,
            "sapien_vs_bodex_base_to_hand_position_error_m": same_base_position,
            "sapien_vs_bodex_base_to_hand_rotation_error_rad": same_base_rotation,
            "mujoco_right_hand_link_to_right_palm": hand_to_palm.tolist(),
            "note": (
                "MANO wrist, DexImit canonical hand root, robot right_hand_link, and "
                "MuJoCo right_palm are distinct frames; only declared transforms connect them."
            ),
        },
        "joint_mapping": {
            "bodex_joint_order": list(CANONICAL_JOINTS),
            "bodex_to_mujoco_sign": {
                name: float(sign) for name, sign in zip(CANONICAL_JOINTS, BODEX_TO_MUJOCO_JOINT_SIGN)
            },
            "common_ranges_bodex_convention_rad": {
                name: [float(low), float(high)] for name, (low, high) in intervals.items()
            },
            "sample_count": len(test_rows),
            "per_link_forward_kinematics": link_errors,
            "maximum_position_error_m": kinematic_position,
            "maximum_rotation_error_rad": kinematic_rotation,
            "passed": bool(kinematic_pass),
        },
        "collision_geometry": {
            "official_bodex_vs_sapien": {
                "per_link": identity_rows,
                "all_13_link_mesh_files_exact": official_mesh_pass,
            },
            "official_mesh_vs_current_mujoco_proxy": {
                "per_link": proxy_rows,
                "whole_hand_sampled_surface": union_distance,
                "equivalent": False,
                "note": (
                    "The current MuJoCo model intentionally uses boxes/capsules/spheres. "
                    "These distances quantify the model gap; they are not an alignment pass."
                ),
            },
        },
        "passed": bool(
            kinematic_pass and official_mesh_pass
            and base_position_error <= 1.0e-6 and base_rotation_error <= 1.0e-5
            and same_base_position <= 1.0e-9 and same_base_rotation <= 1.0e-9
            and prompt_report["maximum_wrist_reconstruction_position_error_m"] <= 1.0e-9
            and prompt_report["maximum_wrist_reconstruction_rotation_error_rad"] <= 1.0e-9
        ),
        "interpretation": (
            "Passing proves frame/joint alignment and exact official collision-asset identity. "
            "It does not claim that the simplified MuJoCo collision proxy is equivalent to PhysX."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "passed": report["passed"],
        "anchor_row": anchor, "derived_fingers": len(active),
        "fk_max_mm": 1000.0 * kinematic_position,
        "fk_max_deg": float(np.rad2deg(kinematic_rotation)),
        "official_mesh_identity": official_mesh_pass,
        "mujoco_proxy_surface_p95_mm": 1000.0 * union_distance["symmetric_p95_m"],
    }, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
