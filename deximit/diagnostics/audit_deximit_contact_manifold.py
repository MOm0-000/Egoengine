#!/usr/bin/env python3
"""Compare SAPIEN and MuJoCo contact manifolds at identical recorded states.

This diagnostic never advances a grasp candidate.  Every sample is reset to a
recorded SAPIEN state, then MuJoCo only rebuilds collision contacts.  It thus
separates collision-manifold differences from controllers and trajectory drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np


SCHEMA = "deximit_contact_manifold_audit_v3_exact_links_diagnostic_only"
TRACE_SCHEMAS = {
    "deximit_sapien_hand_trace_v6_pair_patch_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v7_link_contact_metrics_diagnostic_only",
    "deximit_sapien_hand_trace_v8_joint_force_metrics_diagnostic_only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def summarize_manifold(
    source_detected: np.ndarray, source_loaded: np.ndarray,
    source_count: np.ndarray, source_separation: np.ndarray,
    source_position: np.ndarray, source_normal: np.ndarray,
    predicted_detected: np.ndarray, predicted_active: np.ndarray,
    predicted_count: np.ndarray, predicted_active_count: np.ndarray,
    predicted_separation: np.ndarray, predicted_position: np.ndarray,
    predicted_normal: np.ndarray, predicted_active_position: np.ndarray,
    predicted_active_normal: np.ndarray,
) -> dict[str, object]:
    """Compare one exact contact group without mixing different links."""
    both = source_detected & predicted_detected
    finite_position = (
        both & np.isfinite(source_position).all(axis=1)
        & np.isfinite(predicted_position).all(axis=1)
    )
    finite_normal = (
        finite_position & np.isfinite(source_normal).all(axis=1)
        & np.isfinite(predicted_normal).all(axis=1)
    )
    normal_dot = np.sum(
        predicted_normal[finite_normal] * source_normal[finite_normal], axis=1,
    )
    active_position_mask = (
        source_loaded & predicted_active
        & np.isfinite(source_position).all(axis=1)
        & np.isfinite(predicted_active_position).all(axis=1)
    )
    active_normal_mask = (
        active_position_mask & np.isfinite(source_normal).all(axis=1)
        & np.isfinite(predicted_active_normal).all(axis=1)
    )
    active_normal_dot = np.sum(
        predicted_active_normal[active_normal_mask]
        * source_normal[active_normal_mask], axis=1,
    )
    return {
        "source_detected_samples": int(np.count_nonzero(source_detected)),
        "source_load_bearing_samples": int(np.count_nonzero(source_loaded)),
        "mujoco_detected_on_source_detected_fraction": (
            None if not np.any(source_detected) else float(np.mean(
                predicted_detected[source_detected],
            ))
        ),
        "mujoco_active_on_source_load_bearing_fraction": (
            None if not np.any(source_loaded) else float(np.mean(
                predicted_active[source_loaded],
            ))
        ),
        "both_detected_samples": int(np.count_nonzero(both)),
        "minimum_separation_rmse_m": (
            None if not np.any(both) else rms(
                predicted_separation[both] - source_separation[both]
            )
        ),
        "minimum_separation_bias_m": (
            None if not np.any(both) else float(np.mean(
                predicted_separation[both] - source_separation[both]
            ))
        ),
        "contact_position_rmse_m": (
            None if not np.any(finite_position) else rms(
                predicted_position[finite_position]
                - source_position[finite_position]
            )
        ),
        "contact_normal_angle_rad_mean": (
            None if not np.any(finite_normal) else float(np.mean(np.arccos(
                np.clip(normal_dot, -1.0, 1.0),
            )))
        ),
        "source_point_count_mean_when_detected": (
            None if not np.any(source_detected)
            else float(np.mean(source_count[source_detected]))
        ),
        "mujoco_point_count_mean_on_same_samples": (
            None if not np.any(source_detected)
            else float(np.mean(predicted_count[source_detected]))
        ),
        "mujoco_active_point_count_mean_on_source_load_bearing_samples": (
            None if not np.any(source_loaded)
            else float(np.mean(predicted_active_count[source_loaded]))
        ),
        "active_contact_position_rmse_on_source_load_bearing_m": (
            None if not np.any(active_position_mask) else rms(
                predicted_active_position[active_position_mask]
                - source_position[active_position_mask]
            )
        ),
        "active_contact_normal_angle_on_source_load_bearing_rad_mean": (
            None if not np.any(active_normal_mask) else float(np.mean(
                np.arccos(np.clip(active_normal_dot, -1.0, 1.0)),
            ))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sapien-trace", type=Path, required=True)
    parser.add_argument("--home-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve(strict=True)
    trace_path = args.sapien_trace.expanduser().resolve(strict=True)
    probe_path = args.home_probe.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifold audit {output}")

    with np.load(probe_path, allow_pickle=False) as values:
        names = [str(value) for value in values["joint_names"]]
    with np.load(trace_path, allow_pickle=False) as values:
        if (
            str(np.asarray(values["schema"]).item()) not in TRACE_SCHEMAS
            or not bool(np.asarray(values["diagnostic_only"]).item())
            or bool(np.asarray(values["formal_renderer_3_3_eligible"]).item())
            or str(np.asarray(values["sample_semantics"]).item())
            != "post-step state under same-index drive target"
        ):
            raise ValueError("input is not an isolated detailed SAPIEN trace")
        source_robot = np.asarray(values["right_robot_qpos_sapien"], dtype=np.float64)
        source_pose = np.asarray(values["object_pose_sapien_wxyz"], dtype=np.float64)
        channels = [str(value) for value in values["contact_channel_order"]]
        source_detected = np.asarray(values["contact_channel_detected"], dtype=bool)
        source_loaded = np.asarray(values["contact_channel_load_bearing"], dtype=bool)
        source_count = np.asarray(values["contact_channel_point_count"], dtype=np.int32)
        source_separation = np.asarray(
            values["contact_channel_min_separation_m"], dtype=np.float64,
        )
        source_position = np.asarray(
            values["contact_channel_position_mean_m"], dtype=np.float64,
        )
        source_normal = np.asarray(
            values["contact_channel_normal_toward_object_mean"], dtype=np.float64,
        )
        links = [str(value) for value in values["contact_link_order"]]
        source_link_detected = np.asarray(values["contact_link_detected"], dtype=bool)
        source_link_loaded = np.asarray(
            values["contact_link_load_bearing"], dtype=bool,
        )
        source_link_count = np.asarray(
            values["contact_link_point_count"], dtype=np.int32,
        )
        source_link_separation = np.asarray(
            values["contact_link_min_separation_m"], dtype=np.float64,
        )
        source_link_position = np.asarray(
            values["contact_link_position_mean_m"], dtype=np.float64,
        )
        source_link_normal = np.asarray(
            values["contact_link_normal_toward_object_mean"], dtype=np.float64,
        )
    count = len(source_pose)
    expected_shape = (count, len(channels))
    expected_link_shape = (count, len(links))
    if (
        len(names) != 18 or source_robot.shape != (count, 18)
        or source_detected.shape != expected_shape
        or source_loaded.shape != expected_shape
        or source_count.shape != expected_shape
        or source_separation.shape != expected_shape
        or source_position.shape != expected_shape + (3,)
        or source_normal.shape != expected_shape + (3,)
        or len(set(links)) != len(links)
        or source_link_detected.shape != expected_link_shape
        or source_link_loaded.shape != expected_link_shape
        or source_link_count.shape != expected_link_shape
        or source_link_separation.shape != expected_link_shape
        or source_link_position.shape != expected_link_shape + (3,)
        or source_link_normal.shape != expected_link_shape + (3,)
    ):
        raise ValueError("SAPIEN contact manifold arrays are malformed")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_addresses = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"scene lacks source joint {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint",
    )
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_object",
    )
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_object_single",
    )
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if min(object_joint, object_body, object_geom, table_geom) < 0:
        raise ValueError("scene lacks object/table manifold bodies")
    object_qpos = int(model.jnt_qposadr[object_joint])
    hand_geoms: dict[int, str] = {}
    hand_links: dict[int, str] = {}
    for geom in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if not geom_name.startswith("collision_hand_"):
            continue
        hand_links[geom] = geom_name.removeprefix("collision_hand_").split(
            "_cell_", 1,
        )[0]
        hand_geoms[geom] = next(
            (finger for finger in ("thumb", "index", "mid", "ring", "pinky")
             if finger in geom_name),
            "palm",
        )
    channel_index = {name: index for index, name in enumerate(channels)}
    link_index = {name: index for index, name in enumerate(links)}

    predicted_detected = np.zeros(expected_shape, dtype=bool)
    predicted_active = np.zeros(expected_shape, dtype=bool)
    predicted_count = np.zeros(expected_shape, dtype=np.int32)
    predicted_active_count = np.zeros(expected_shape, dtype=np.int32)
    predicted_separation = np.full(expected_shape, np.nan, dtype=np.float64)
    predicted_position = np.full(expected_shape + (3,), np.nan, dtype=np.float64)
    predicted_normal = np.full(expected_shape + (3,), np.nan, dtype=np.float64)
    predicted_active_position = np.full(
        expected_shape + (3,), np.nan, dtype=np.float64,
    )
    predicted_active_normal = np.full(
        expected_shape + (3,), np.nan, dtype=np.float64,
    )
    link_detected = np.zeros(expected_link_shape, dtype=bool)
    link_active = np.zeros(expected_link_shape, dtype=bool)
    link_count = np.zeros(expected_link_shape, dtype=np.int32)
    link_active_count = np.zeros(expected_link_shape, dtype=np.int32)
    link_separation = np.full(expected_link_shape, np.nan, dtype=np.float64)
    link_position = np.full(expected_link_shape + (3,), np.nan, dtype=np.float64)
    link_normal = np.full(expected_link_shape + (3,), np.nan, dtype=np.float64)
    link_active_position = np.full(
        expected_link_shape + (3,), np.nan, dtype=np.float64,
    )
    link_active_normal = np.full(
        expected_link_shape + (3,), np.nan, dtype=np.float64,
    )
    for row in range(count):
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addresses] = source_robot[row]
        data.qpos[object_qpos : object_qpos + 7] = source_pose[row]
        mujoco.mj_forward(model, data)
        position_sum = np.zeros((len(channels), 3), dtype=np.float64)
        normal_sum = np.zeros_like(position_sum)
        active_position_sum = np.zeros_like(position_sum)
        active_normal_sum = np.zeros_like(position_sum)
        link_position_sum = np.zeros((len(links), 3), dtype=np.float64)
        link_normal_sum = np.zeros_like(link_position_sum)
        link_active_position_sum = np.zeros_like(link_position_sum)
        link_active_normal_sum = np.zeros_like(link_position_sum)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geoms = {int(contact.geom1), int(contact.geom2)}
            if object_geom not in geoms:
                continue
            if table_geom in geoms:
                channel = "table"
                link = "table"
            else:
                current_hand = geoms & hand_geoms.keys()
                if not current_hand:
                    continue
                hand_geom = next(iter(current_hand))
                channel = hand_geoms[hand_geom]
                link = hand_links[hand_geom]
            if channel not in channel_index:
                continue
            index = channel_index[channel]
            exact_index = link_index.get(link, link_index.get("other"))
            if exact_index is None:
                continue
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            toward_object = data.xpos[object_body] - np.asarray(contact.pos)
            if float(normal @ toward_object) < 0.0:
                normal = -normal
            predicted_detected[row, index] = True
            active = int(contact.efc_address) >= 0
            predicted_active[row, index] |= active
            predicted_count[row, index] += 1
            predicted_separation[row, index] = np.nanmin((
                predicted_separation[row, index], float(contact.dist),
            )) if np.isfinite(predicted_separation[row, index]) else float(contact.dist)
            position_sum[index] += np.asarray(contact.pos)
            normal_sum[index] += normal
            if active:
                predicted_active_count[row, index] += 1
                active_position_sum[index] += np.asarray(contact.pos)
                active_normal_sum[index] += normal
            link_detected[row, exact_index] = True
            link_active[row, exact_index] |= active
            link_count[row, exact_index] += 1
            link_separation[row, exact_index] = np.nanmin((
                link_separation[row, exact_index], float(contact.dist),
            )) if np.isfinite(
                link_separation[row, exact_index]
            ) else float(contact.dist)
            link_position_sum[exact_index] += np.asarray(contact.pos)
            link_normal_sum[exact_index] += normal
            if active:
                link_active_count[row, exact_index] += 1
                link_active_position_sum[exact_index] += np.asarray(contact.pos)
                link_active_normal_sum[exact_index] += normal
        nonzero = predicted_count[row] > 0
        predicted_position[row, nonzero] = (
            position_sum[nonzero] / predicted_count[row, nonzero, None]
        )
        norms = np.linalg.norm(normal_sum[nonzero], axis=1)
        valid_normals = norms > 1.0e-12
        active_channels = np.flatnonzero(nonzero)[valid_normals]
        predicted_normal[row, active_channels] = (
            normal_sum[active_channels]
            / np.linalg.norm(normal_sum[active_channels], axis=1, keepdims=True)
        )
        constrained = predicted_active_count[row] > 0
        predicted_active_position[row, constrained] = (
            active_position_sum[constrained]
            / predicted_active_count[row, constrained, None]
        )
        active_norms = np.linalg.norm(active_normal_sum[constrained], axis=1)
        valid_active_normals = active_norms > 1.0e-12
        constrained_channels = np.flatnonzero(constrained)[valid_active_normals]
        predicted_active_normal[row, constrained_channels] = (
            active_normal_sum[constrained_channels]
            / np.linalg.norm(
                active_normal_sum[constrained_channels], axis=1, keepdims=True,
            )
        )
        link_populated = link_count[row] > 0
        link_position[row, link_populated] = (
            link_position_sum[link_populated]
            / link_count[row, link_populated, None]
        )
        link_norms = np.linalg.norm(link_normal_sum[link_populated], axis=1)
        valid_link_normals = link_norms > 1.0e-12
        populated_links = np.flatnonzero(link_populated)[valid_link_normals]
        link_normal[row, populated_links] = (
            link_normal_sum[populated_links]
            / np.linalg.norm(link_normal_sum[populated_links], axis=1, keepdims=True)
        )
        link_constrained = link_active_count[row] > 0
        link_active_position[row, link_constrained] = (
            link_active_position_sum[link_constrained]
            / link_active_count[row, link_constrained, None]
        )
        link_active_norms = np.linalg.norm(
            link_active_normal_sum[link_constrained], axis=1,
        )
        valid_link_active_normals = link_active_norms > 1.0e-12
        constrained_links = np.flatnonzero(
            link_constrained,
        )[valid_link_active_normals]
        link_active_normal[row, constrained_links] = (
            link_active_normal_sum[constrained_links]
            / np.linalg.norm(
                link_active_normal_sum[constrained_links], axis=1, keepdims=True,
            )
        )

    per_channel: dict[str, object] = {}
    for index, name in enumerate(channels):
        per_channel[name] = summarize_manifold(
            source_detected[:, index], source_loaded[:, index],
            source_count[:, index], source_separation[:, index],
            source_position[:, index], source_normal[:, index],
            predicted_detected[:, index], predicted_active[:, index],
            predicted_count[:, index], predicted_active_count[:, index],
            predicted_separation[:, index], predicted_position[:, index],
            predicted_normal[:, index], predicted_active_position[:, index],
            predicted_active_normal[:, index],
        )

    per_link: dict[str, object] = {}
    for index, name in enumerate(links):
        per_link[name] = summarize_manifold(
            source_link_detected[:, index], source_link_loaded[:, index],
            source_link_count[:, index], source_link_separation[:, index],
            source_link_position[:, index], source_link_normal[:, index],
            link_detected[:, index], link_active[:, index],
            link_count[:, index], link_active_count[:, index],
            link_separation[:, index], link_position[:, index],
            link_normal[:, index], link_active_position[:, index],
            link_active_normal[:, index],
        )

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "formal_renderer_3_3_eligible": False,
        "grasp_candidate_advanced": False,
        "rendered": False,
        "sample_semantics": "independent collision rebuild at identical recorded state",
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256(trace_path),
        "sample_count": count,
        "coarse_channel_warning": (
            "channel metrics merge several hand links, including predictive "
            "zero-load contacts; use exact links for causal attribution"
        ),
        "channels": per_channel,
        "links": per_link,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
