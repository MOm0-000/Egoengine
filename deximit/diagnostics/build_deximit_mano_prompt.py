#!/usr/bin/env python3
"""Build the exact DexImit hand prompt from released TACO MANO parameters.

This is an isolated diagnostic adapter.  It does not edit the human reference,
renderer inputs, controller inputs, or section-3.3 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egoengine_repro.evaluation.taco_surface import load_taco_mano_sequence


SCHEMA = "deximit_taco_mano_prompt_v1_diagnostic_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-v3", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--mano-pose", type=Path, required=True)
    parser.add_argument("--mano-shape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def angle_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("tji,tjk->tik", left, right)
    return Rotation.from_matrix(relative).magnitude()


def quantiles(values: np.ndarray, *, degrees: bool = False) -> dict[str, float]:
    data = np.degrees(values) if degrees else values
    levels = (0.0, 0.05, 0.5, 0.95, 1.0)
    names = ("min", "p05", "median", "p95", "max")
    return {
        name: float(value)
        for name, value in zip(names, np.quantile(data, levels), strict=True)
    }


def build_prompt(
    root_world_rotation: np.ndarray, translations: np.ndarray,
    world_to_sim: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MANO root frames, DexImit prompts, and its fixed right-hand rotation."""
    rotations = np.asarray(root_world_rotation, dtype=np.float64)
    positions = np.asarray(translations, dtype=np.float64)
    transform = np.asarray(world_to_sim, dtype=np.float64)
    if (
        rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
        or positions.shape != (len(rotations), 3) or transform.shape != (4, 4)
    ):
        raise ValueError("MANO prompt inputs have invalid shapes")
    root_sim_rotation = np.einsum("ij,tjk->tik", transform[:3, :3], rotations)
    root_sim_position = positions @ transform[:3, :3].T + transform[:3, 3]
    root_sim = np.repeat(np.eye(4, dtype=np.float64)[None], len(rotations), axis=0)
    root_sim[:, :3, :3] = root_sim_rotation
    root_sim[:, :3, 3] = root_sim_position
    canonical = (
        Rotation.from_euler("x", np.pi / 2.0).as_matrix()
        @ np.diag((-1.0, 1.0, -1.0))
    )
    prompt = root_sim.copy()
    prompt[:, :3, :3] = np.einsum("tij,jk->tik", root_sim_rotation, canonical.T)
    return root_sim, prompt, canonical


def main() -> int:
    args = parse_args()
    contact_path = args.contact_v3.resolve(strict=True)
    human_path = args.human_reference.resolve(strict=True)
    pose_path = args.mano_pose.resolve(strict=True)
    shape_path = args.mano_shape.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("choose fresh output/report paths; diagnostic evidence is immutable")

    with np.load(contact_path, allow_pickle=False) as data:
        contact_schema = str(np.asarray(data["schema"]).item())
        frames = np.asarray(data["frame_indices"], dtype=np.int64)
        hands = [str(value) for value in data["hand_order"]]
        world_object = np.asarray(data["T_world_object"], dtype=np.float64)
        world_vertices = np.asarray(data["mano_vertices_world_m"], dtype=np.float64)
        world_joints = np.asarray(data["mano_joints_world_m"], dtype=np.float64)
        declared_pose_hash = str(np.asarray(data["right_pose_sha256"]).item())
        declared_shape_hash = str(np.asarray(data["right_shape_sha256"]).item())
    if contact_schema != "taco_mano_surface_contact_v3_conservative_geometric_evidence":
        raise ValueError("the prompt adapter requires contact geometry v3")
    right = hands.index("right")

    with np.load(human_path, allow_pickle=False) as data:
        human_frames = np.asarray(data["frame_indices"], dtype=np.int64)
        human_hands = [str(value) for value in data["hand_order"]]
        xhand_wrist = np.asarray(data["T_sim_wrist_target"], dtype=np.float64)
        sim_object = np.asarray(data["T_sim_object_reference"], dtype=np.float64)
    if not np.array_equal(frames, human_frames) or human_hands != ["right"]:
        raise ValueError("human reference and contact-v3 frames/hands do not align")
    xhand_wrist = xhand_wrist[:, 0]
    sim_object = sim_object[:, 0]

    pose_hash = sha256(pose_path)
    shape_hash = sha256(shape_path)
    if pose_hash != declared_pose_hash or shape_hash != declared_shape_hash:
        raise ValueError("MANO pose/shape hashes differ from the sources frozen in contact-v3")
    poses, translations, _, keys = load_taco_mano_sequence(pose_path, shape_path)
    if len(poses) != len(frames):
        raise ValueError("MANO sequence length does not match contact-v3")
    root_error = np.linalg.norm(world_joints[:, right, 0] - translations, axis=1)
    if float(root_error.max()) > 1.0e-6:
        raise ValueError("released MANO translation does not reproduce the v3 wrist joint")

    # Recover the single world->simulation transform already used to build the
    # current reference.  Deriving it from every object frame also detects any
    # accidental time-dependent alignment or hidden re-scaling.
    world_to_sim_each = sim_object @ np.linalg.inv(world_object)
    world_to_sim = world_to_sim_each[0]
    map_translation_error = np.linalg.norm(
        world_to_sim_each[:, :3, 3] - world_to_sim[:3, 3], axis=1,
    )
    map_rotation_error = angle_between(
        np.repeat(world_to_sim[None, :3, :3], len(frames), axis=0),
        world_to_sim_each[:, :3, :3],
    )
    if float(map_translation_error.max()) > 2.0e-7 or float(map_rotation_error.max()) > 2.0e-7:
        raise ValueError("human reference does not use one fixed world-to-simulation transform")

    root_world_rotation = Rotation.from_rotvec(poses[:, :3]).as_matrix()
    # This is the exact convention in DexImit's estimate_hand_poses_new.py:
    # right MANO local -> flip X/Z, then rotate +pi/2 around X.  Its saved
    # prompt is the inverse route, local -> MANO root -> simulation world.
    root_sim, prompt, canonical = build_prompt(
        root_world_rotation, translations, world_to_sim,
    )
    root_sim_rotation = root_sim[:, :3, :3]
    root_sim_position = root_sim[:, :3, 3]

    # Reconstruct every official right-hand vertex through the canonical local
    # mesh.  A correct prompt must return the source surface point-for-point.
    local_vertices = np.einsum(
        "tvi,tij,jk->tvk",
        world_vertices[:, right] - translations[:, None],
        root_world_rotation,
        canonical.T,
    )
    rebuilt_sim_vertices = (
        np.einsum("tvi,tji->tvj", local_vertices, prompt[:, :3, :3])
        + prompt[:, None, :3, 3]
    )
    expected_sim_vertices = (
        np.einsum("tvi,ji->tvj", world_vertices[:, right], world_to_sim[:3, :3])
        + world_to_sim[:3, 3]
    )
    vertex_error = np.linalg.norm(rebuilt_sim_vertices - expected_sim_vertices, axis=-1)
    if float(vertex_error.max()) > 1.0e-6:
        raise RuntimeError("DexImit prompt fails point-wise official MANO reconstruction")

    # Reproduce the old adapter solely for the audit.  It incorrectly treated
    # the joint-derived xHand palm frame as MANO's root frame before applying
    # the DexImit canonical offset a second time.
    old_prompt_rotation = np.einsum(
        "tij,jk->tik", xhand_wrist[:, :3, :3], canonical.T,
    )
    old_prompt_error = angle_between(prompt[:, :3, :3], old_prompt_rotation)
    root_vs_xhand_error = angle_between(root_sim_rotation, xhand_wrist[:, :3, :3])
    wrist_position_error = np.linalg.norm(root_sim_position - xhand_wrist[:, :3, 3], axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(SCHEMA),
        diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False),
        frame_indices=frames,
        source_frame_keys=np.asarray(keys),
        T_world_to_sim=world_to_sim,
        T_sim_mano_root=root_sim,
        T_sim_deximit_prompt=prompt,
        T_sim_object_reference=sim_object,
        deximit_right_canonical_rotation=canonical,
        source_contact_v3=np.asarray(str(contact_path)),
        source_contact_v3_sha256=np.asarray(sha256(contact_path)),
        source_human_reference=np.asarray(str(human_path)),
        source_human_reference_sha256=np.asarray(sha256(human_path)),
        source_mano_pose=np.asarray(str(pose_path)),
        source_mano_pose_sha256=np.asarray(pose_hash),
        source_mano_shape=np.asarray(str(shape_path)),
        source_mano_shape_sha256=np.asarray(shape_hash),
    )

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "output": str(output),
        "output_sha256": sha256(output),
        "finding": (
            "the previous adapter used a joint-derived xHand palm frame where "
            "DexImit requires the released MANO global root rotation"
        ),
        "source_identity": {
            "contact_v3": str(contact_path),
            "contact_v3_sha256": sha256(contact_path),
            "human_reference": str(human_path),
            "human_reference_sha256": sha256(human_path),
            "mano_pose": str(pose_path),
            "mano_pose_sha256": pose_hash,
            "mano_shape": str(shape_path),
            "mano_shape_sha256": shape_hash,
        },
        "checks": {
            "mano_root_position_error_m": quantiles(root_error),
            "world_to_sim_translation_drift_m": quantiles(map_translation_error),
            "world_to_sim_rotation_drift_deg": quantiles(map_rotation_error, degrees=True),
            "official_mesh_roundtrip_error_m": quantiles(vertex_error.reshape(-1)),
            "mano_root_vs_xhand_palm_rotation_deg": quantiles(
                root_vs_xhand_error, degrees=True,
            ),
            "correct_vs_previous_deximit_prompt_rotation_deg": quantiles(
                old_prompt_error, degrees=True,
            ),
            "mano_root_vs_xhand_wrist_position_error_m": quantiles(wrist_position_error),
        },
        "decision": {
            "previous_sapien_ranking_valid": False,
            "reason": "all prior top-120 sets were selected with the wrong hand orientation prompt",
            "required_action": "rerun each BODex depth with T_sim_deximit_prompt from this artifact",
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
