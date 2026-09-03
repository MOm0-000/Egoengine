#!/usr/bin/env python3
"""Record DexImit/SAPIEN incoming-joint frames without running a candidate.

PhysX reports ``linkIncomingJointForce`` in its incoming-joint frame.  That
frame is not generally the URDF child-link frame because the SAPIEN loader
rotates PhysX joints onto canonical articulation axes.  This isolated probe
records the loader result so cross-engine force comparisons use like-for-like
coordinates.  It initializes no task object, executes no candidate, and does
not render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SCHEMA = "deximit_sapien_joint_frame_probe_v1_diagnostic_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deximit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    deximit = args.deximit_root.expanduser().resolve(strict=True)
    package = deximit / "third_party/any2dex/any2dex"
    config_path = package / "env/config/env_hand.yaml"
    if not package.is_dir() or not config_path.is_file():
        raise FileNotFoundError("DexImit any2dex package/config is incomplete")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite joint-frame probe {output}")

    sys.path.insert(0, str(package))
    sys.path.insert(0, str(package / "third_party/BODex_api/src"))
    import sapien
    import yaml
    import env.base_env_for_render as render_env
    from env.base_env_for_render import BaseEnv

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    original_synthetic_pc = render_env.SyntheticPC
    original_setup_render = BaseEnv.set_up_physics_and_render

    class DisabledSyntheticPC:
        def __init__(self, *unused_args: object, **unused_kwargs: object) -> None:
            pass

    def setup_raster_render(self: object) -> None:
        # Match the released BaseEnv physics configuration exactly.  The
        # default raster backend only avoids initializing the unused denoiser.
        sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
        sapien.physx.set_body_config(
            solver_position_iterations=25,
            solver_velocity_iterations=1,
            sleep_threshold=0.005,
        )
        sapien.physx.set_scene_config(
            gravity=np.array([0.0, 0.0, -9.81]),
            bounce_threshold=2.0,
            enable_pcm=True,
            enable_tgs=True,
            enable_ccd=False,
            enable_enhanced_determinism=False,
            enable_friction_every_iteration=True,
            cpu_workers=0,
        )
        sapien.physx.set_default_material(
            static_friction=0.7, dynamic_friction=0.5, restitution=0.0,
        )
        self.scene = sapien.Scene([
            sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(),
        ])
        self.scene.set_timestep(self.timestep)
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_viewer_shader_dir("default")

    render_env.SyntheticPC = DisabledSyntheticPC
    BaseEnv.set_up_physics_and_render = setup_raster_render
    try:
        env = BaseEnv(config)
    finally:
        BaseEnv.set_up_physics_and_render = original_setup_render
        render_env.SyntheticPC = original_synthetic_pc

    links = env.robot_right.get_links()
    joints = env.robot_right.get_joints()
    if len(links) != len(joints) or not links:
        raise RuntimeError("SAPIEN articulation link/joint indexing changed")
    link_names = [link.get_name() for link in links]
    joint_names = [joint.get_name() for joint in joints]
    if len(set(link_names)) != len(link_names):
        raise RuntimeError("right-hand articulation has duplicate link names")

    child_pose = np.asarray([
        np.concatenate((joint.get_pose_in_child().p, joint.get_pose_in_child().q))
        for joint in joints
    ], dtype=np.float64)
    parent_pose = np.asarray([
        np.concatenate((joint.get_pose_in_parent().p, joint.get_pose_in_parent().q))
        for joint in joints
    ], dtype=np.float64)
    if (
        child_pose.shape != (len(joints), 7)
        or parent_pose.shape != (len(joints), 7)
        or not np.isfinite(child_pose).all()
        or not np.isfinite(parent_pose).all()
        or not np.allclose(np.linalg.norm(child_pose[:, 3:], axis=1), 1.0, atol=2e-5)
    ):
        raise RuntimeError("SAPIEN joint-frame poses are malformed")

    output.mkdir(parents=True)
    trace = output / "joint_frames.npz"
    np.savez_compressed(
        trace,
        schema=np.asarray(SCHEMA),
        diagnostic_only=np.asarray(True),
        formal_renderer_3_3_eligible=np.asarray(False),
        candidate_executed=np.asarray(False),
        rendered=np.asarray(False),
        camera_following_allowed=np.asarray(False),
        link_order=np.asarray(link_names),
        incoming_joint_order=np.asarray(joint_names),
        pose_component_order=np.asarray((
            "position_x", "position_y", "position_z",
            "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
        )),
        incoming_joint_pose_in_child_link_p_wxyz=child_pose,
        incoming_joint_pose_in_parent_link_p_wxyz=parent_pose,
    )
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "candidate_executed": False,
        "rendered": False,
        "camera_following_allowed": False,
        "deximit_root": str(deximit),
        "deximit_commit": subprocess.run(
            ["git", "-C", str(deximit), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "sapien_version": sapien.__version__,
        "right_robot_link_count": len(link_names),
        "active_joint_count": len(env.robot_right.get_active_joints()),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "interpretation": (
            "pose_in_child maps incoming-joint coordinates into the child-link "
            "coordinates; cross-engine wrenches must apply its inverse rotation"
        ),
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "report.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
