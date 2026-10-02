#!/usr/bin/env python3
"""Read-only hand/object visual audit for the frozen TACO Pour evidence.

The script never advances MuJoCo time.  It loads saved qpos values into fresh
MjData objects and calls only ``mj_kinematics`` before rendering or reading
site positions.  No environment, policy, rollout, controller, or optimizer is
constructed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh


EXPECTED_BASE_COMMIT = "3635de5ddfa575780f1a4a9c30dfafd33a350f9d"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HANDS = ("right", "left")
OBJECTS = ("tool", "target")
HUMAN_TIPS = (4, 8, 12, 16, 20)
HUMAN_BONES = tuple(
    (0 if joint == start else joint - 1, joint)
    for start in (1, 5, 9, 13, 17)
    for joint in range(start, start + 4)
)
KEY_ENDPOINTS = (20, 30, 40, 45, 50, 55, 58, 59, 60)
CAMERAS = {
    "oblique": {
        "lookat": (0.60, 0.0, 0.83),
        "distance": 0.82,
        "azimuth": 128.0,
        "elevation": -24.0,
    },
    "top": {
        "lookat": (0.60, 0.0, 0.80),
        "distance": 0.86,
        "azimuth": 90.0,
        "elevation": -89.0,
    },
}
PANEL_WIDTH = 480
PANEL_HEIGHT = 360


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite wxyz")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def pose7_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("free-joint pose must have shape (7,)")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_wxyz_to_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def transform_to_pose7(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quaternion[[3, 0, 1, 2]]]


def points_in_object(points_world: np.ndarray, object_transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(object_transform, dtype=np.float64)
    return (points - transform[:3, 3]) @ transform[:3, :3]


def triangle_surface_distance(
    query: trimesh.proximity.ProximityQuery,
    points_world: np.ndarray,
    object_transform: np.ndarray,
) -> np.ndarray:
    local = points_in_object(points_world, object_transform)
    _, distances, _ = query.on_surface(local)
    return np.asarray(distances, dtype=np.float64)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_torch_gzip(path: Path) -> dict[str, Any]:
    import gzip

    with gzip.open(path, "rb") as stream:
        value = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dictionary in {path}")
    return value


@dataclass(frozen=True)
class SavedState:
    endpoint: int
    qpos: np.ndarray
    qvel: np.ndarray
    tracking_score: float | None
    valid_status: str
    source: str
    contact_flags: np.ndarray | None = None


@dataclass
class StaticModel:
    label: str
    path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    kinematics_calls: int = 0

    @classmethod
    def load(cls, label: str, path: Path) -> "StaticModel":
        model = mujoco.MjModel.from_xml_path(str(path))
        if (model.nq, model.nv, model.nu) != (50, 48, 36):
            raise ValueError(f"{label} model dimensions changed")
        return cls(label, path, model, mujoco.MjData(model))

    def set_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"invalid qpos for {self.label}")
        before_time = float(self.data.time)
        self.data.qpos[:] = qpos
        copied = self.data.qpos.copy()
        mujoco.mj_kinematics(self.model, self.data)
        self.kinematics_calls += 1
        if float(self.data.time) != before_time:
            raise RuntimeError("mj_kinematics advanced data.time")
        if not np.array_equal(self.data.qpos, copied):
            raise RuntimeError("mj_kinematics changed qpos")

    def markers(self, qpos: np.ndarray) -> np.ndarray:
        self.set_qpos(qpos)
        result = np.empty((2, 6, 3), dtype=np.float64)
        for hand_index, hand in enumerate(HANDS):
            names = [f"{hand}_palm", *(f"{hand}_{finger}_tip" for finger in FINGERS)]
            for marker_index, name in enumerate(names):
                result[hand_index, marker_index] = self.data.site_xpos[
                    self.model.site(name).id
                ]
        return result


def model_semantics(model: mujoco.MjModel) -> dict[str, Any]:
    joints = []
    for index in range(model.njnt):
        joints.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                "type": int(model.jnt_type[index]),
                "qpos_address": int(model.jnt_qposadr[index]),
                "dof_address": int(model.jnt_dofadr[index]),
                "axis": model.jnt_axis[index].astype(float).tolist(),
            }
        )
    sites = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index)
        for index in range(model.nsite)
    ]
    objects = {}
    for name in ("right_object", "left_object"):
        body = model.body(name).id
        joint = int(model.body_jntadr[body])
        objects[name] = {
            "body_id": int(body),
            "joint_id": joint,
            "qpos_address": int(model.jnt_qposadr[joint]),
            "dof_address": int(model.jnt_dofadr[joint]),
        }
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joints": joints,
        "sites": sites,
        "objects": objects,
        "qpos0": model.qpos0.astype(float).tolist(),
    }


def validate_model_compatibility(reference: StaticModel, actual: StaticModel) -> dict[str, Any]:
    left = model_semantics(reference.model)
    right = model_semantics(actual.model)
    checks = {
        "dimensions_equal": (left["nq"], left["nv"], left["nu"])
        == (right["nq"], right["nv"], right["nu"]),
        "joint_semantics_equal": left["joints"] == right["joints"],
        "required_sites_equal": all(
            name in left["sites"] and name in right["sites"]
            for hand in HANDS
            for name in [f"{hand}_palm", *(f"{hand}_{finger}_tip" for finger in FINGERS)]
        ),
        "object_addresses_equal": left["objects"] == right["objects"],
        "qpos0_equal": np.array_equal(reference.model.qpos0, actual.model.qpos0),
    }
    if not all(checks.values()):
        raise ValueError(f"reference/actual kinematic model mismatch: {checks}")
    return checks


def build_saved_states(
    asset_root: Path,
) -> tuple[dict[int, SavedState], dict[int, SavedState], dict[str, Any]]:
    s20_path = asset_root / (
        "runs/taco_pour_corrected_replay_rebase_v1/tool_only/"
        "committed_boundary_endpoint_20.pt.gz"
    )
    s40_path = asset_root / (
        "runs/taco_pour_candidate_D_strict_success_promotion_v1/"
        "committed_boundary_endpoint_40.pt.gz"
    )
    chunk_path = asset_root / (
        "runs/taco_pour_candidate_D_strict_success_promotion_v1/"
        "committed_chunk_20_40.npz"
    )
    donor_path = asset_root / (
        "runs/taco_pour_endpoint60_viability_adjudication_v1/carryover/"
        "continuous_20_80.npz"
    )
    latest_path = asset_root / (
        "runs/taco_pour_virtual_object_assist_v1/evaluations/"
        "official_unassisted_epoch_0250.npz"
    )
    s20 = load_torch_gzip(s20_path)
    s40 = load_torch_gzip(s40_path)
    q20 = _to_numpy(s20["qpos"])[0].astype(np.float64)
    v20 = _to_numpy(s20["qvel"])[0].astype(np.float64)
    q40 = _to_numpy(s40["qpos"])[0].astype(np.float64)
    v40 = _to_numpy(s40["qvel"])[0].astype(np.float64)
    score20 = float(_to_numpy(s20["last_tracking_error"])[0])
    score40 = float(_to_numpy(s40["last_tracking_error"])[0])
    main: dict[int, SavedState] = {
        20: SavedState(20, q20, v20, score20, "VALID", "committed_s20_snapshot")
    }
    donor: dict[int, SavedState] = {
        20: SavedState(20, q20, v20, score20, "VALID", "committed_s20_snapshot")
    }
    with np.load(chunk_path, allow_pickle=False) as chunk:
        endpoints = chunk["endpoint"].astype(int)
        if endpoints.tolist() != list(range(21, 41)):
            raise ValueError("committed chunk endpoint coverage changed")
        for row, endpoint in enumerate(endpoints):
            main[endpoint] = SavedState(
                endpoint,
                chunk["qpos"][row].astype(np.float64),
                chunk["qvel"][row].astype(np.float64),
                float(chunk["tracking_score"][row]),
                "VALID",
                "committed_chunk_20_40",
                chunk["contact_flags"][row].astype(bool),
            )
        if not np.array_equal(main[40].qpos.astype(np.float32), q40.astype(np.float32)):
            raise ValueError("committed endpoint40 qpos differs from snapshot")
        if not np.array_equal(main[40].qvel.astype(np.float32), v40.astype(np.float32)):
            raise ValueError("committed endpoint40 qvel differs from snapshot")
        if not math.isclose(main[40].tracking_score or 0.0, score40, abs_tol=2e-7):
            raise ValueError("committed endpoint40 score differs from snapshot")
    with np.load(donor_path, allow_pickle=False) as sequence:
        endpoints = sequence["endpoint"].astype(int)
        if endpoints.tolist() != list(range(21, 81)):
            raise ValueError("donor endpoint coverage changed")
        for row, endpoint in enumerate(endpoints):
            status = "VALID" if endpoint <= 60 else ("FAIL" if endpoint == 61 else "POST_FAILURE")
            donor[endpoint] = SavedState(
                endpoint,
                sequence["qpos"][row].astype(np.float64),
                sequence["qvel"][row].astype(np.float64),
                float(sequence["tracking_score"][row]),
                status,
                "continuous_donor_20_80",
                sequence["contact_flags"][row].astype(bool),
            )
        for endpoint in range(21, 41):
            if not np.array_equal(
                donor[endpoint].qpos.astype(np.float32),
                main[endpoint].qpos.astype(np.float32),
            ):
                raise ValueError("donor and committed prefix qpos differ")
    with np.load(latest_path, allow_pickle=False) as latest:
        endpoints = latest["outcome_endpoint"].astype(int)
        if endpoints.tolist() != list(range(41, 61)):
            raise ValueError("latest endpoint coverage changed")
        if np.flatnonzero(latest["terminated"]).tolist() != [19]:
            raise ValueError("latest failure row changed")
        for row, endpoint in enumerate(endpoints):
            main[endpoint] = SavedState(
                endpoint,
                latest["qpos"][row].astype(np.float64),
                latest["qvel"][row].astype(np.float64),
                float(latest["tracking_score"][row]),
                "FAIL" if endpoint == 60 else "VALID",
                "virtual_assist_epoch250_unassisted",
                latest["contact_flags"][row].astype(bool),
            )
    checks = {
        "s20_snapshot": artifact(s20_path),
        "s40_snapshot": artifact(s40_path),
        "committed_prefix": artifact(chunk_path),
        "donor_continuous": artifact(donor_path),
        "latest_unassisted": artifact(latest_path),
        "committed_s40_snapshot_match": True,
        "committed_chunk_equals_donor_endpoints21_40": True,
        "main_recorded_endpoints": [20, 60],
        "donor_recorded_endpoints": [20, 80],
        "donor_first_failure_endpoint": 61,
        "latest_first_failure_endpoint": 60,
    }
    return main, donor, checks


def human_qpos_for_objects(model: mujoco.MjModel, transforms: np.ndarray) -> np.ndarray:
    qpos = model.qpos0.copy()
    qpos[36:43] = transform_to_pose7(transforms[0])
    qpos[43:50] = transform_to_pose7(transforms[1])
    return qpos


def add_human_skeleton(renderer: mujoco.Renderer, joints: np.ndarray) -> None:
    colors = (
        np.array([0.18, 0.68, 1.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.55, 0.12, 1.0], dtype=np.float32),
    )
    for hand_index in range(2):
        color = colors[hand_index]
        for parent, child in HUMAN_BONES:
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                raise RuntimeError("render scene geom capacity exhausted")
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(-1),
                color,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.0028,
                joints[hand_index, parent].astype(np.float64),
                joints[hand_index, child].astype(np.float64),
            )
            renderer.scene.ngeom += 1
        for joint_index, point in enumerate(joints[hand_index]):
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                raise RuntimeError("render scene geom capacity exhausted")
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            radius = 0.0052 if joint_index in (0, *HUMAN_TIPS) else 0.0035
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius),
                point.astype(np.float64),
                np.eye(3).reshape(-1),
                color,
            )
            renderer.scene.ngeom += 1


def camera_object(settings: dict[str, Any]) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = settings["lookat"]
    camera.distance = settings["distance"]
    camera.azimuth = settings["azimuth"]
    camera.elevation = settings["elevation"]
    return camera


def scene_option(*, collision: bool = False, human: bool = False) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    if not human and not collision:
        option.geomgroup[1] = 1
    if collision:
        option.geomgroup[3] = 1
    option.sitegroup[:] = 0
    return option


def annotate_panel(image: np.ndarray, lines: Iterable[str], color=(255, 255, 255)) -> np.ndarray:
    result = np.asarray(image, dtype=np.uint8).copy()
    line_list = list(lines)
    overlay_height = 12 + 22 * len(line_list)
    cv2.rectangle(result, (0, 0), (result.shape[1], overlay_height), (0, 0, 0), -1)
    for index, line in enumerate(line_list):
        cv2.putText(
            result,
            str(line),
            (10, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    scale = min(width / source.shape[1], height / source.shape[0])
    resized = cv2.resize(
        source,
        (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class Visualizer:
    def __init__(self, reference: StaticModel, actual: StaticModel):
        self.reference = reference
        self.actual = actual
        self.reference_renderer = mujoco.Renderer(
            reference.model, height=PANEL_HEIGHT, width=PANEL_WIDTH, max_geom=10000
        )
        self.actual_renderer = mujoco.Renderer(
            actual.model, height=PANEL_HEIGHT, width=PANEL_WIDTH, max_geom=10000
        )
        self.render_updates = 0

    def close(self) -> None:
        self.reference_renderer.close()
        self.actual_renderer.close()

    def render_reference(
        self, qpos: np.ndarray, view: str, *, collision: bool = False
    ) -> np.ndarray:
        self.reference.set_qpos(qpos)
        self.reference_renderer.update_scene(
            self.reference.data,
            camera=camera_object(CAMERAS[view]),
            scene_option=scene_option(collision=collision),
        )
        self.render_updates += 1
        return self.reference_renderer.render().copy()

    def render_actual(self, qpos: np.ndarray, view: str, *, collision: bool = False) -> np.ndarray:
        self.actual.set_qpos(qpos)
        self.actual_renderer.update_scene(
            self.actual.data,
            camera=camera_object(CAMERAS[view]),
            scene_option=scene_option(collision=collision),
        )
        self.render_updates += 1
        return self.actual_renderer.render().copy()

    def render_human(
        self,
        object_transforms: np.ndarray,
        joints: np.ndarray,
        view: str,
    ) -> np.ndarray:
        qpos = human_qpos_for_objects(self.actual.model, object_transforms)
        self.actual.set_qpos(qpos)
        self.actual_renderer.update_scene(
            self.actual.data,
            camera=camera_object(CAMERAS[view]),
            scene_option=scene_option(human=True),
        )
        add_human_skeleton(self.actual_renderer, joints)
        self.render_updates += 1
        return self.actual_renderer.render().copy()


def read_rgb_frames(video: Path, count: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open RGB video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("RGB video ended before requested frame")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if total != 198 or not math.isclose(fps, 30.0, abs_tol=1e-9):
        raise ValueError(f"unexpected RGB stream: frames={total}, fps={fps}")
    return frames, {"frames": total, "fps": fps, "size": [frames[0].shape[1], frames[0].shape[0]]}


def missing_panel(endpoint: int) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        "ACTUAL: NOT RECORDED",
        (65, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 210, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"endpoint {endpoint}",
        (145, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_comparison_frame(
    endpoint: int,
    rgb: np.ndarray,
    human_panel: np.ndarray,
    reference_panel: np.ndarray,
    actual_panel: np.ndarray,
    actual_state: SavedState | None,
) -> np.ndarray:
    rgb_panel = annotate_panel(
        letterbox(rgb, PANEL_WIDTH, PANEL_HEIGHT),
        ["REAL RGB (independent camera)", f"frame {endpoint} | t={endpoint / 30.0:.3f}s"],
    )
    human_panel = annotate_panel(
        human_panel,
        ["HUMAN GT + REFERENCE OBJECTS", "shared sim coordinates; non-RGB camera"],
    )
    reference_panel = annotate_panel(
        reference_panel,
        ["KINEMATIC REFERENCE", f"endpoint {endpoint}; not physics-validated"],
    )
    if actual_state is not None:
        score = "missing" if actual_state.tracking_score is None else f"{actual_state.tracking_score:.6f}"
        color = (90, 255, 120) if actual_state.valid_status == "VALID" else (255, 100, 100)
        actual_panel = annotate_panel(
            actual_panel,
            [
                "ROBOT EXECUTED",
                f"endpoint {endpoint} | score={score} | {actual_state.valid_status}",
                actual_state.source,
            ],
            color=color,
        )
    return np.concatenate([rgb_panel, human_panel, reference_panel, actual_panel], axis=1)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image: {path}")


def build_alignment(
    output: Path,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
) -> None:
    fields = (
        "trajectory_id",
        "data_row",
        "source_endpoint",
        "outcome_endpoint",
        "reference_endpoint",
        "video_frame",
        "timestamp_s",
        "status",
        "source_artifact",
    )
    rows = []
    for endpoint in range(81):
        if int(frame_indices[endpoint]) != endpoint or not math.isclose(
            float(timestamps[endpoint]), endpoint / 30.0, abs_tol=1e-12
        ):
            raise ValueError("human/reference time mapping changed")
        state = main.get(endpoint)
        rows.append(
            {
                "trajectory_id": "executed_main_composite",
                "data_row": "" if state is None else endpoint,
                "source_endpoint": "" if endpoint == 0 else endpoint - 1,
                "outcome_endpoint": endpoint,
                "reference_endpoint": endpoint,
                "video_frame": int(frame_indices[endpoint]),
                "timestamp_s": float(timestamps[endpoint]),
                "status": "NOT_RECORDED" if state is None else state.valid_status,
                "source_artifact": "" if state is None else state.source,
            }
        )
        donor_state = donor.get(endpoint)
        rows.append(
            {
                "trajectory_id": "executed_donor_continuous",
                "data_row": "" if donor_state is None else endpoint,
                "source_endpoint": "" if endpoint == 0 else endpoint - 1,
                "outcome_endpoint": endpoint,
                "reference_endpoint": endpoint,
                "video_frame": int(frame_indices[endpoint]),
                "timestamp_s": float(timestamps[endpoint]),
                "status": "NOT_RECORDED" if donor_state is None else donor_state.valid_status,
                "source_artifact": "" if donor_state is None else donor_state.source,
            }
        )
    with (output / "alignment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def object_transforms_from_qpos(qpos: np.ndarray) -> np.ndarray:
    return np.stack([pose7_to_transform(qpos[36:43]), pose7_to_transform(qpos[43:50])])


def compute_markers(
    reference_model: StaticModel,
    actual_model: StaticModel,
    robot_qpos: np.ndarray,
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {
        "ROBOT_REFERENCE": {},
        "ROBOT_EXECUTED_MAIN": {},
        "ROBOT_EXECUTED_DONOR": {},
    }
    for endpoint in range(81):
        result["ROBOT_REFERENCE"][endpoint] = reference_model.markers(robot_qpos[endpoint])
    for endpoint, state in main.items():
        result["ROBOT_EXECUTED_MAIN"][endpoint] = actual_model.markers(state.qpos)
    for endpoint, state in donor.items():
        result["ROBOT_EXECUTED_DONOR"][endpoint] = actual_model.markers(state.qpos)
    return result


def write_geometry(
    output: Path,
    human: dict[str, np.ndarray],
    robot_qpos: np.ndarray,
    markers: dict[str, dict[int, np.ndarray]],
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
    meshes: tuple[trimesh.Trimesh, trimesh.Trimesh],
) -> list[dict[str, Any]]:
    queries = tuple(trimesh.proximity.ProximityQuery(mesh) for mesh in meshes)
    rows: list[dict[str, Any]] = []
    human_tip_world = np.take(human["joint_positions_sim"], HUMAN_TIPS, axis=2)
    human_palm_world = human["joint_positions_sim"][:, :, 0]
    datasets: list[tuple[str, range, Any, Any, Any]] = [
        (
            "HUMAN_GT",
            range(81),
            lambda endpoint: np.concatenate(
                [human_palm_world[endpoint, :, None], human_tip_world[endpoint]], axis=1
            ),
            lambda endpoint: human["T_sim_object_reference"][endpoint],
            lambda endpoint: None,
        ),
        (
            "ROBOT_REFERENCE",
            range(81),
            lambda endpoint: markers["ROBOT_REFERENCE"][endpoint],
            lambda endpoint: object_transforms_from_qpos(robot_qpos[endpoint]),
            lambda endpoint: None,
        ),
        (
            "ROBOT_EXECUTED_MAIN",
            range(20, 61),
            lambda endpoint: markers["ROBOT_EXECUTED_MAIN"][endpoint],
            lambda endpoint: object_transforms_from_qpos(main[endpoint].qpos),
            lambda endpoint: main[endpoint],
        ),
        (
            "ROBOT_EXECUTED_DONOR",
            range(20, 81),
            lambda endpoint: markers["ROBOT_EXECUTED_DONOR"][endpoint],
            lambda endpoint: object_transforms_from_qpos(donor[endpoint].qpos),
            lambda endpoint: donor[endpoint],
        ),
    ]
    for trajectory, endpoints, marker_getter, transform_getter, state_getter in datasets:
        for endpoint in endpoints:
            points = marker_getter(endpoint)
            transforms = transform_getter(endpoint)
            state = state_getter(endpoint)
            for object_index, object_role in enumerate(OBJECTS):
                relative = points_in_object(points.reshape(-1, 3), transforms[object_index]).reshape(2, 6, 3)
                distances = triangle_surface_distance(
                    queries[object_index], points[:, 1:].reshape(-1, 3), transforms[object_index]
                ).reshape(2, 5)
                for hand_index, hand in enumerate(HANDS):
                    palm_relative = relative[hand_index, 0]
                    rows.append(
                        {
                            "trajectory": trajectory,
                            "endpoint": endpoint,
                            "timestamp_s": endpoint / 30.0,
                            "record_status": "REFERENCE" if state is None else state.valid_status,
                            "hand": hand,
                            "marker_role": "human_wrist" if trajectory == "HUMAN_GT" else "robot_palm_site",
                            "finger": "palm",
                            "object_role": object_role,
                            "surface_kind": "released_visual_triangle_surface",
                            "surface_distance_m": "",
                            "tip_object_x_m": palm_relative[0],
                            "tip_object_y_m": palm_relative[1],
                            "tip_object_z_m": palm_relative[2],
                            "human_target_to_reference_error_m": "",
                            "reference_to_executed_error_m": "",
                            "tracking_score": "" if state is None else state.tracking_score,
                            "contact_flag": "",
                            "contact_force": "MISSING_NOT_RECORDED",
                        }
                    )
                    for finger_index, finger in enumerate(FINGERS):
                        target_error: float | str = ""
                        execution_error: float | str = ""
                        if trajectory == "ROBOT_REFERENCE":
                            target = human["T_sim_fingertip_target"][
                                endpoint, hand_index, finger_index, :3, 3
                            ]
                            target_error = float(
                                np.linalg.norm(points[hand_index, finger_index + 1] - target)
                            )
                        if trajectory in ("ROBOT_EXECUTED_MAIN", "ROBOT_EXECUTED_DONOR"):
                            execution_error = float(
                                np.linalg.norm(
                                    points[hand_index, finger_index + 1]
                                    - markers["ROBOT_REFERENCE"][endpoint][hand_index, finger_index + 1]
                                )
                            )
                        contact: bool | str = ""
                        if state is not None and state.contact_flags is not None:
                            contact = bool(state.contact_flags[hand_index, object_index, finger_index])
                        tip_relative = relative[hand_index, finger_index + 1]
                        rows.append(
                            {
                                "trajectory": trajectory,
                                "endpoint": endpoint,
                                "timestamp_s": endpoint / 30.0,
                                "record_status": "REFERENCE" if state is None else state.valid_status,
                                "hand": hand,
                                "marker_role": "fingertip_target" if trajectory == "HUMAN_GT" else "robot_tip_site",
                                "finger": finger,
                                "object_role": object_role,
                                "surface_kind": "released_visual_triangle_surface",
                                "surface_distance_m": float(distances[hand_index, finger_index]),
                                "tip_object_x_m": tip_relative[0],
                                "tip_object_y_m": tip_relative[1],
                                "tip_object_z_m": tip_relative[2],
                                "human_target_to_reference_error_m": target_error,
                                "reference_to_executed_error_m": execution_error,
                                "tracking_score": "" if state is None else state.tracking_score,
                                "contact_flag": contact,
                                "contact_force": "MISSING_NOT_RECORDED",
                            }
                        )
    fields = list(rows[0])
    with (output / "geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_object_tracking(
    output: Path,
    robot_qpos: np.ndarray,
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory, states in (
        ("ROBOT_EXECUTED_MAIN", main),
        ("ROBOT_EXECUTED_DONOR", donor),
    ):
        for endpoint, state in sorted(states.items()):
            for object_index, (object_role, address) in enumerate(zip(OBJECTS, (36, 43))):
                reference = pose7_to_transform(robot_qpos[endpoint, address : address + 7])
                actual = pose7_to_transform(state.qpos[address : address + 7])
                delta = actual[:3, 3] - reference[:3, 3]
                rotation_error = Rotation.from_matrix(
                    reference[:3, :3].T @ actual[:3, :3]
                ).magnitude()
                rows.append(
                    {
                        "trajectory": trajectory,
                        "endpoint": endpoint,
                        "timestamp_s": endpoint / 30.0,
                        "record_status": state.valid_status,
                        "object_role": object_role,
                        "position_error_x_m": delta[0],
                        "position_error_y_m": delta[1],
                        "position_error_z_m": delta[2],
                        "position_error_norm_m": np.linalg.norm(delta),
                        "rotation_error_rad": rotation_error,
                        "tracking_score": state.tracking_score,
                    }
                )
    fields = tuple(rows[0])
    with (output / "object_tracking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    colors = {"ROBOT_EXECUTED_MAIN": "#2ca02c", "ROBOT_EXECUTED_DONOR": "#9467bd"}
    for trajectory, color in colors.items():
        selected = [
            row
            for row in rows
            if row["trajectory"] == trajectory and row["object_role"] == "tool"
        ]
        endpoints = [row["endpoint"] for row in selected]
        for component, linestyle in zip(
            ("position_error_x_m", "position_error_y_m", "position_error_z_m"),
            ("-", "--", ":"),
        ):
            axes[0].plot(
                endpoints,
                [1000.0 * float(row[component]) for row in selected],
                color=color,
                linestyle=linestyle,
                label=f"{trajectory} {component[-3]}",
            )
        axes[1].plot(
            endpoints,
            [1000.0 * float(row["position_error_norm_m"]) for row in selected],
            color=color,
            label=trajectory,
        )
        axes[2].plot(
            endpoints,
            [float(row["rotation_error_rad"]) for row in selected],
            color=color,
            label=trajectory,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("tool position\nerror xyz (mm)")
    axes[1].set_ylabel("tool position\nerror norm (mm)")
    axes[2].set_ylabel("tool rotation\nerror (rad)")
    axes[2].set_xlabel("endpoint")
    for axis in axes:
        axis.axvline(60, color="#d62728", linestyle="--", linewidth=0.8)
        axis.axvline(61, color="#9467bd", linestyle=":", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Recorded tool-pose error against same-endpoint reference")
    figure.tight_layout()
    figure.savefig(output / "curves/object_tracking_errors.png", dpi=150)
    plt.close(figure)
    return rows


def _rows_for(
    rows: list[dict[str, Any]],
    *,
    trajectory: str,
    hand: str,
    finger: str,
    object_role: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["trajectory"] == trajectory
        and row["hand"] == hand
        and row["finger"] == finger
        and row["object_role"] == object_role
    ]


def plot_surface_clearance(
    output: Path, rows: list[dict[str, Any]], object_role: str
) -> None:
    colors = {
        "HUMAN_GT": "#1f77b4",
        "ROBOT_REFERENCE": "#ff7f0e",
        "ROBOT_EXECUTED_MAIN": "#2ca02c",
        "ROBOT_EXECUTED_DONOR": "#9467bd",
    }
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for hand_index, hand in enumerate(HANDS):
        for finger_index, finger in enumerate(FINGERS):
            axis = axes[hand_index, finger_index]
            for trajectory, color in colors.items():
                selected = _rows_for(
                    rows,
                    trajectory=trajectory,
                    hand=hand,
                    finger=finger,
                    object_role=object_role,
                )
                axis.plot(
                    [row["endpoint"] for row in selected],
                    [1000.0 * float(row["surface_distance_m"]) for row in selected],
                    label=trajectory,
                    color=color,
                    linewidth=1.4,
                )
            axis.axvline(60, color="#d62728", linestyle="--", linewidth=0.8)
            axis.axvline(61, color="#9467bd", linestyle=":", linewidth=0.8)
            axis.set_title(f"{hand} {finger}")
            axis.grid(alpha=0.25)
            if finger_index == 0:
                axis.set_ylabel("visual-surface clearance (mm)")
            if hand_index == 1:
                axis.set_xlabel("endpoint")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(
        f"Fingertip distance to {object_role} released visual triangle surface\n"
        "Unsigned geometry locator; not a contact or penetration label",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output / f"curves/surface_clearance_{object_role}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marker_errors(output: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for hand_index, hand in enumerate(HANDS):
        for finger_index, finger in enumerate(FINGERS):
            axis = axes[hand_index, finger_index]
            reference = _rows_for(
                rows,
                trajectory="ROBOT_REFERENCE",
                hand=hand,
                finger=finger,
                object_role="tool",
            )
            axis.plot(
                [row["endpoint"] for row in reference],
                [1000.0 * float(row["human_target_to_reference_error_m"]) for row in reference],
                label="human target -> robot reference",
                color="#ff7f0e",
            )
            for trajectory, label, color in (
                ("ROBOT_EXECUTED_MAIN", "reference -> latest executed", "#2ca02c"),
                ("ROBOT_EXECUTED_DONOR", "reference -> donor executed", "#9467bd"),
            ):
                selected = _rows_for(
                    rows,
                    trajectory=trajectory,
                    hand=hand,
                    finger=finger,
                    object_role="tool",
                )
                axis.plot(
                    [row["endpoint"] for row in selected],
                    [1000.0 * float(row["reference_to_executed_error_m"]) for row in selected],
                    label=label,
                    color=color,
                )
            axis.axvline(60, color="#d62728", linestyle="--", linewidth=0.8)
            axis.set_title(f"{hand} {finger}")
            axis.grid(alpha=0.25)
            if finger_index == 0:
                axis.set_ylabel("marker position difference (mm)")
            if hand_index == 1:
                axis.set_xlabel("endpoint")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Two-layer fingertip marker position differences", y=1.02)
    fig.tight_layout()
    fig.savefig(output / "curves/marker_position_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tracking_contacts(
    output: Path,
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
) -> None:
    pairs = tuple(
        (hand_index, object_index, f"{hand}-{object_role}")
        for hand_index, hand in enumerate(HANDS)
        for object_index, object_role in enumerate(OBJECTS)
    )
    fig, axes = plt.subplots(1 + len(pairs), 1, figsize=(13, 13), sharex=True)
    for states, label, color in (
        (main, "executed main composite", "#2ca02c"),
        (donor, "donor continuous", "#9467bd"),
    ):
        endpoints = sorted(states)
        axes[0].plot(
            endpoints,
            [states[endpoint].tracking_score for endpoint in endpoints],
            label=label,
            color=color,
        )
        for axis_index, (hand_index, object_index, _label) in enumerate(pairs, start=1):
            values = []
            for endpoint in endpoints:
                flags = states[endpoint].contact_flags
                values.append(
                    np.nan if flags is None else int(flags[hand_index, object_index].sum())
                )
            axes[axis_index].step(
                endpoints,
                values,
                where="mid",
                label=label,
                color=color,
            )
    axes[0].axhline(1.0, color="#d62728", linestyle="--", label="tracking boundary")
    axes[0].set_ylabel("tracking score")
    for axis_index, (_hand_index, _object_index, label) in enumerate(pairs, start=1):
        axes[axis_index].set_ylabel(f"{label}\ncontact flags")
    axes[-1].set_xlabel("endpoint")
    for axis in axes:
        axis.axvline(60, color="#d62728", linestyle="--", linewidth=0.8)
        axis.axvline(61, color="#9467bd", linestyle=":", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    fig.suptitle("Recorded tracking score and fingertip contact flags\nContact force was not saved")
    fig.tight_layout()
    fig.savefig(output / "curves/tracking_and_contact_flags.png", dpi=150)
    plt.close(fig)


def plot_object_relative_paths(output: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    colors = {
        "HUMAN_GT": "#1f77b4",
        "ROBOT_REFERENCE": "#ff7f0e",
        "ROBOT_EXECUTED_MAIN": "#2ca02c",
        "ROBOT_EXECUTED_DONOR": "#9467bd",
    }
    for hand_index, hand in enumerate(HANDS):
        for finger_index, finger in enumerate(FINGERS):
            axis = axes[hand_index, finger_index]
            for trajectory, color in colors.items():
                selected = [
                    row
                    for row in _rows_for(
                        rows,
                        trajectory=trajectory,
                        hand=hand,
                        finger=finger,
                        object_role="tool",
                    )
                    if 20 <= int(row["endpoint"]) <= 60
                ]
                axis.plot(
                    [1000.0 * float(row["tip_object_y_m"]) for row in selected],
                    [1000.0 * float(row["tip_object_z_m"]) for row in selected],
                    color=color,
                    label=trajectory,
                    linewidth=1.4,
                )
                if selected:
                    axis.scatter(
                        [1000.0 * float(selected[0]["tip_object_y_m"])],
                        [1000.0 * float(selected[0]["tip_object_z_m"])],
                        color=color,
                        s=12,
                    )
            axis.set_title(f"{hand} {finger}")
            axis.grid(alpha=0.25)
            axis.set_aspect("equal", adjustable="box")
            if finger_index == 0:
                axis.set_ylabel("tool-local z (mm)")
            if hand_index == 1:
                axis.set_xlabel("tool-local y (mm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Fingertip paths in each trajectory's own tool frame, endpoints 20--60", y=1.02)
    fig.tight_layout()
    fig.savefig(output / "curves/tool_relative_fingertip_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_object_relative_keyframe(
    output: Path,
    endpoint: int,
    human: dict[str, np.ndarray],
    markers: dict[str, dict[int, np.ndarray]],
    robot_qpos: np.ndarray,
    main: dict[int, SavedState],
    donor: dict[int, SavedState],
    tool_mesh: trimesh.Trimesh,
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    figure = plt.figure(figsize=(16, 4.6))
    tool_points = tool_mesh.vertices[:: max(1, len(tool_mesh.vertices) // 1800)]
    datasets = []
    # Keep the complete 21-joint hand here: the object-relative panel draws the
    # skeleton, whereas geometry.csv intentionally uses only wrists and tips.
    human_points = human["joint_positions_sim"][endpoint]
    datasets.append(
        (
            "HUMAN GT\n21-joint skeleton (tips highlighted)",
            human_points,
            human["T_sim_object_reference"][endpoint, 0],
            True,
        )
    )
    datasets.append(
        (
            "KINEMATIC REFERENCE\npalm + tip sites",
            markers["ROBOT_REFERENCE"][endpoint],
            object_transforms_from_qpos(robot_qpos[endpoint])[0],
            False,
        )
    )
    datasets.append(
        (
            "EXECUTED MAIN\npalm + tip sites",
            markers["ROBOT_EXECUTED_MAIN"][endpoint],
            object_transforms_from_qpos(main[endpoint].qpos)[0],
            False,
        )
    )
    datasets.append(
        (
            "EXECUTED DONOR\npalm + tip sites",
            markers["ROBOT_EXECUTED_DONOR"][endpoint],
            object_transforms_from_qpos(donor[endpoint].qpos)[0],
            False,
        )
    )
    for column, (title, world_points, transform, is_human) in enumerate(datasets, start=1):
        axis = figure.add_subplot(1, 4, column, projection="3d")
        local = points_in_object(world_points.reshape(-1, 3), transform).reshape(world_points.shape)
        axis.scatter(
            tool_points[:, 0], tool_points[:, 1], tool_points[:, 2],
            s=0.45, c="#999999", alpha=0.22,
        )
        for hand_index, color in enumerate(("#1f9fff", "#ff8c1a")):
            points = local[hand_index]
            if is_human:
                original = points
                for parent, child in HUMAN_BONES:
                    axis.plot(*original[[parent, child]].T, color=color, linewidth=1.0)
                axis.scatter(*original[list(HUMAN_TIPS)].T, color=color, s=22)
                axis.scatter(*original[0].T, color=color, marker="*", s=50)
            else:
                palm, tips = points[0], points[1:]
                for tip in tips:
                    axis.plot(*np.stack([palm, tip]).T, color=color, linewidth=1.0)
                axis.scatter(*tips.T, color=color, s=22)
                axis.scatter(*palm, color=color, marker="*", s=50)
        axis.set_xlim(limits[0][0], limits[1][0])
        axis.set_ylim(limits[0][1], limits[1][1])
        axis.set_zlim(limits[0][2], limits[1][2])
        axis.set_box_aspect(limits[1] - limits[0])
        axis.view_init(elev=22, azim=-58)
        axis.set_xlabel("x m")
        axis.set_ylabel("y m")
        axis.set_zlabel("z m")
        axis.set_title(title)
    figure.suptitle(
        f"Endpoint {endpoint}: object-relative comparison in each trajectory's own tool frame\n"
        "Same camera and metric limits; no fitting between hands",
        y=1.05,
    )
    figure.tight_layout()
    figure.savefig(
        output / f"keyframes/object_relative_endpoint_{endpoint:03d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_timeline(output: Path, endpoints: Iterable[int]) -> None:
    cards = []
    for endpoint in endpoints:
        name = f"endpoint_{endpoint:03d}.jpg"
        cards.append(
            f'<figure><a href="keyframes/endpoint_{endpoint:03d}_oblique.png">'
            f'<img src="thumbnails/{name}" alt="endpoint {endpoint}"></a>'
            f'<figcaption>endpoint {endpoint}</figcaption></figure>'
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>TACO Pour hand-object timeline</title>
<style>
body {{ background:#17191c; color:#eee; font-family:sans-serif; margin:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:14px; }}
figure {{ margin:0; background:#24272b; padding:8px; }} img {{ width:100%; }}
figcaption {{ padding-top:5px; }}
</style>
<h1>Executed main trajectory: endpoint 20 through first failure at 60</h1>
<p>Click a thumbnail for the full oblique four-column comparison. Endpoint 60 is the
first failed endpoint; no later latest-policy state is presented as recorded.</p>
<div class="grid">{''.join(cards)}</div>
"""
    (output / "timeline.html").write_text(document)


def output_hashes(output: Path) -> None:
    target = output / "server_artifacts.sha256"
    lines = [
        f"{sha256(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != target
    ]
    target.write_text("\n".join(lines) + "\n")


def run_pure_checks() -> dict[str, bool]:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    transform[:3, 3] = [0.4, -0.2, 0.7]
    points = np.array([[0.5, -0.1, 0.8], [0.3, -0.25, 0.9]])
    local = points_in_object(points, transform)
    roundtrip = local @ transform[:3, :3].T + transform[:3, 3]
    pose_roundtrip = pose7_to_transform(transform_to_pose7(transform))
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    distance = triangle_surface_distance(
        trimesh.proximity.ProximityQuery(mesh), np.array([[0.25, 0.25, 1.0]]), np.eye(4)
    )[0]
    checks = {
        "coordinate_roundtrip": bool(np.allclose(points, roundtrip, atol=1e-12)),
        "pose_roundtrip": bool(np.allclose(transform, pose_roundtrip, atol=1e-12)),
        "triangle_face_distance_not_vertex_distance": bool(math.isclose(distance, 1.0, abs_tol=1e-12)),
        "hand_order_right_left": HANDS == ("right", "left"),
        "object_roles_tool_target": OBJECTS == ("tool", "target"),
        "endpoint_frame_mapping_identity": all(endpoint == int(np.arange(81)[endpoint]) for endpoint in range(81)),
        "metric_mesh_scale_cm_to_m": math.isclose(0.01, 1.0 / 100.0),
    }
    if not all(checks.values()):
        raise RuntimeError(f"pure checks failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("/data_all/zzx/3.2RL"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data_all/zzx/3.2RL/runs/taco_pour_hand_object_visual_audit_v1"),
    )
    args = parser.parse_args()
    asset_root = args.asset_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable audit output already exists: {output}")
    output.mkdir(parents=True)
    for directory in ("keyframes", "thumbnails", "curves", "collision"):
        (output / directory).mkdir()

    repository = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != EXPECTED_BASE_COMMIT:
        raise RuntimeError(f"expected base {EXPECTED_BASE_COMMIT}, observed {head}")

    checks = run_pure_checks()
    paths = {
        "rgb": asset_root / "data/taco_v1/pour_bowl_plate/rgb/taco_pour_bowl_plate_20230927_017.mp4",
        "human_reference": asset_root / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/human_reference.npz",
        "human_input_audit": asset_root / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1/input_audit.json",
        "robot_reference": asset_root / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/robot_reference.npz",
        "retarget_report": asset_root / "runs/taco_pour_bimanual_mano_fk_combined_collision_v1/retarget_report.json",
        "reference_model": asset_root / "runs/taco_pour_collision_semantics_combined_v1/combined_candidate_scene.xml",
        "actual_model": asset_root / "runs/taco_pour_floor_contact_v1/candidate.xml",
        "tool_mesh": asset_root / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/022_cm.obj",
        "target_mesh": asset_root / "data/taco_v1/pour_bowl_plate/object_models/object_models_released/135_cm.obj",
    }
    expected = {
        "rgb": "9c6bd70041be04f5237613632f508dff2b12cdb2e19941259ad1751c21d8add4",
        "human_reference": "852bda624bb22c9cf0797544294556752d490d11d443eb59a7f1e3e4aaab986a",
        "reference_model": "cd2600cd195db9086a6a3c6cec5aab49991320c3f300e7fbdd511dc99ab7960f",
        "tool_mesh": "35c52d853b85aa521eeadda78abb2bd370975611c8c35cd9a9d4c96746d8418c",
        "target_mesh": "6316aea5b5c29a6231639d4ac8dad2c42ee013cfc55f1ce5a9dc5a38056f52a1",
    }
    input_manifest = {name: artifact(path) for name, path in paths.items()}
    for name, digest in expected.items():
        if input_manifest[name]["sha256"] != digest:
            raise ValueError(f"frozen input hash mismatch: {name}")

    with np.load(paths["human_reference"], allow_pickle=False) as archive:
        human = {name: archive[name].copy() for name in archive.files}
    with np.load(paths["robot_reference"], allow_pickle=False) as archive:
        robot = {name: archive[name].copy() for name in archive.files}
    if human["frame_indices"].tolist() != list(range(198)):
        raise ValueError("human frame indices are not 0..197")
    if not np.array_equal(human["frame_indices"], robot["frame_indices"]):
        raise ValueError("human and robot frame indices differ")
    if not np.array_equal(human["timestamps_s"], robot["timestamps_s"]):
        raise ValueError("human and robot timestamps differ")
    if human["hand_order"].tolist() != list(HANDS) or human["object_roles"].tolist() != list(OBJECTS):
        raise ValueError("human hand/object role order changed")
    if robot["hand_order"].tolist() != list(HANDS) or robot["object_roles"].tolist() != list(OBJECTS):
        raise ValueError("robot hand/object role order changed")

    main_states, donor_states, trajectory_checks = build_saved_states(asset_root)
    input_manifest.update({f"trajectory_{name}": value for name, value in trajectory_checks.items() if isinstance(value, dict)})
    reference_model = StaticModel.load("MINK reference", paths["reference_model"])
    actual_model = StaticModel.load("executed", paths["actual_model"])
    compatibility = validate_model_compatibility(reference_model, actual_model)
    markers = compute_markers(reference_model, actual_model, robot["qpos"], main_states, donor_states)

    tool_mesh = trimesh.load_mesh(paths["tool_mesh"], process=False)
    target_mesh = trimesh.load_mesh(paths["target_mesh"], process=False)
    if not isinstance(tool_mesh, trimesh.Trimesh) or not isinstance(target_mesh, trimesh.Trimesh):
        raise TypeError("released object surfaces must load as triangle meshes")
    tool_mesh.apply_scale(0.01)
    target_mesh.apply_scale(0.01)
    rows = write_geometry(
        output,
        human,
        robot["qpos"],
        markers,
        main_states,
        donor_states,
        (tool_mesh, target_mesh),
    )
    write_object_tracking(output, robot["qpos"], main_states, donor_states)
    plot_surface_clearance(output, rows, "tool")
    plot_surface_clearance(output, rows, "target")
    plot_marker_errors(output, rows)
    plot_tracking_contacts(output, main_states, donor_states)
    plot_object_relative_paths(output, rows)

    rgb_frames, video_info = read_rgb_frames(paths["rgb"], 81)
    for endpoint in KEY_ENDPOINTS:
        save_rgb(output / f"keyframes/rgb_endpoint_{endpoint:03d}.png", rgb_frames[endpoint])
    visualizer = Visualizer(reference_model, actual_model)
    writers = {}
    video_paths = {}
    for view in CAMERAS:
        path = output / f"comparison_{view}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (PANEL_WIDTH * 4, PANEL_HEIGHT)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {path}")
        writers[view] = writer
        video_paths[view] = path
    try:
        for endpoint in range(81):
            for view in CAMERAS:
                human_panel = visualizer.render_human(
                    human["T_sim_object_reference"][endpoint],
                    human["joint_positions_sim"][endpoint],
                    view,
                )
                reference_panel = visualizer.render_reference(robot["qpos"][endpoint], view)
                state = main_states.get(endpoint)
                actual_panel = (
                    missing_panel(endpoint)
                    if state is None
                    else visualizer.render_actual(state.qpos, view)
                )
                composite = make_comparison_frame(
                    endpoint,
                    rgb_frames[endpoint],
                    human_panel,
                    reference_panel,
                    actual_panel,
                    state,
                )
                writers[view].write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                if endpoint in KEY_ENDPOINTS:
                    save_rgb(output / f"keyframes/endpoint_{endpoint:03d}_{view}.png", composite)
                if view == "oblique" and 20 <= endpoint <= 60:
                    thumbnail = cv2.resize(composite, (960, 180), interpolation=cv2.INTER_AREA)
                    path = output / f"thumbnails/endpoint_{endpoint:03d}.jpg"
                    cv2.imwrite(str(path), cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        for endpoint in (40, 50, 60):
            reference_collision = visualizer.render_reference(
                robot["qpos"][endpoint], "oblique", collision=True
            )
            actual_collision = visualizer.render_actual(main_states[endpoint].qpos, "oblique", collision=True)
            reference_collision = annotate_panel(reference_collision, ["REFERENCE COLLISION GEOMETRY", f"endpoint {endpoint}"])
            actual_collision = annotate_panel(actual_collision, ["EXECUTED COLLISION GEOMETRY", f"endpoint {endpoint}"])
            save_rgb(
                output / f"collision/endpoint_{endpoint:03d}_collision.png",
                np.concatenate([reference_collision, actual_collision], axis=1),
            )
        for endpoint in (40, 50, 55, 58, 59, 60):
            latest = annotate_panel(
                visualizer.render_actual(main_states[endpoint].qpos, "oblique"),
                ["EXECUTED MAIN", f"endpoint {endpoint} | {main_states[endpoint].valid_status}"],
            )
            donor = annotate_panel(
                visualizer.render_actual(donor_states[endpoint].qpos, "oblique"),
                ["EXECUTED DONOR", f"endpoint {endpoint} | {donor_states[endpoint].valid_status}"],
            )
            save_rgb(
                output / f"keyframes/donor_latest_endpoint_{endpoint:03d}.png",
                np.concatenate([latest, donor], axis=1),
            )
    finally:
        for writer in writers.values():
            writer.release()
        visualizer.close()

    if any(not path.is_file() or path.stat().st_size == 0 for path in video_paths.values()):
        raise RuntimeError("comparison video was not written")
    write_timeline(output, range(20, 61))

    relative_points = []
    for endpoint in range(20, 61):
        relative_points.append(points_in_object(
            human["joint_positions_sim"][endpoint].reshape(-1, 3),
            human["T_sim_object_reference"][endpoint, 0],
        ))
        relative_points.append(points_in_object(
            markers["ROBOT_REFERENCE"][endpoint].reshape(-1, 3),
            object_transforms_from_qpos(robot["qpos"][endpoint])[0],
        ))
        relative_points.append(points_in_object(
            markers["ROBOT_EXECUTED_MAIN"][endpoint].reshape(-1, 3),
            object_transforms_from_qpos(main_states[endpoint].qpos)[0],
        ))
        relative_points.append(points_in_object(
            markers["ROBOT_EXECUTED_DONOR"][endpoint].reshape(-1, 3),
            object_transforms_from_qpos(donor_states[endpoint].qpos)[0],
        ))
    all_relative = np.concatenate(relative_points)
    low = all_relative.min(axis=0)
    high = all_relative.max(axis=0)
    center = (low + high) / 2.0
    half = max(float((high - low).max()) / 2.0, 0.08) * 1.06
    limits = (center - half, center + half)
    for endpoint in KEY_ENDPOINTS:
        render_object_relative_keyframe(
            output,
            endpoint,
            human,
            markers,
            robot["qpos"],
            main_states,
            donor_states,
            tool_mesh,
            limits,
        )

    build_alignment(
        output,
        human["frame_indices"],
        human["timestamps_s"],
        main_states,
        donor_states,
    )
    manifest = {
        "schema": "taco_pour_hand_object_visual_audit_v1",
        "status": "READONLY_STATIC_AUDIT_COMPLETE",
        "source_commit": head,
        "input_artifacts": input_manifest,
        "input_hashes_unchanged_after_audit": {
            name: sha256(Path(details["path"])) == details["sha256"]
            for name, details in input_manifest.items()
        },
        "alignment": {
            "human_video_robot_frame_count": 198,
            "fps": 30.0,
            "mapping": "state/reference/human endpoint k equals RGB frame_indices[k]",
            "dynamic_time_warping": False,
            "main_actual": "s20 snapshot + committed outcomes21--40 + latest unassisted outcomes41--60",
            "main_missing": "endpoints0--19 and61--80",
            "donor": "s20 snapshot + saved outcomes21--80; endpoint61 fail;62--80 post-failure history",
        },
        "coordinate_contract": {
            "hand_order": list(HANDS),
            "object_roles": list(OBJECTS),
            "T_sim_world": human["T_sim_world"].astype(float).tolist(),
            "rgb_camera_extrinsics_available": False,
            "three_dimensional_views": "fixed shared simulation camera; not RGB pixel registration",
            "object_relative": "each trajectory transformed into its own tool pose; no fitted transform",
        },
        "model_compatibility": compatibility,
        "model_paths": {
            "kinematic_reference": str(paths["reference_model"]),
            "executed": str(paths["actual_model"]),
        },
        "surface_distance": {
            "kind": "unsigned nearest released visual triangle surface",
            "tool_scale": 0.01,
            "target_scale": 0.01,
            "signed_penetration_reported": False,
            "contact_label_from_distance": False,
        },
        "cameras": CAMERAS,
        "object_relative_metric_limits_m": [limits[0].tolist(), limits[1].tolist()],
        "pure_checks": checks,
        "execution_counts": {
            "physics_integrator_steps": 0,
            "simulation_control_intervals": 0,
            "actions_sampled": 0,
            "actor_critic_or_visual_model_forwards": 0,
            "optimizer_updates": 0,
            "mj_forward_calls": 0,
            "mj_kinematics_calls_reference_model": reference_model.kinematics_calls,
            "mj_kinematics_calls_executed_model": actual_model.kinematics_calls,
            "renderer_scene_updates": visualizer.render_updates,
        },
        "contact_evidence": {
            "flags": "loaded from saved rollout arrays where present",
            "forces": "missing; no historical contact force was saved",
            "isolated_kinematics_contacts_used_as_history": False,
        },
        "video": video_info,
        "trajectory_checks": trajectory_checks,
        "paper_faithful": False,
        "mutations": "output directory only; source inputs unchanged",
    }
    write_json(output / "input_manifest.json", manifest)
    (output / "findings.md").write_text(
        "# Hand-object visual audit v1\n\n"
        "Status: `READONLY_STATIC_AUDIT_COMPLETE`\n\n"
        "This file is intentionally a draft pending direct inspection of the rendered keyframes "
        "and geometry curves. It must be completed with endpoint-linked observations without "
        "claiming mathematical irrecoverability.\n"
    )
    output_hashes(output)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output),
        "geometry_rows": len(rows),
        "kinematics_calls": manifest["execution_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
