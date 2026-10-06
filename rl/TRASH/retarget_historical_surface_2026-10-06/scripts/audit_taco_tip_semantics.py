"""Inspect tip landmarks and anatomical axes without selecting a new mapping."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mink/src")]
from audit_taco_initialization import visual_meshes
from diagnose_taco_retarget_objectives import load_inputs, UPSTREAM
from egoengine_repro.evaluation.taco_surface import (
    _load_model_data, load_taco_mano_sequence, reconstruct_taco_mano,
    MANO_TIP_VERTICES, MANO21_SOURCE,
)
from egoengine_repro.retarget.paper_audit import artifact, verify_artifacts
from egoengine_repro.retarget.taco_bimanual import SIDES, FINGERS, MANO_DISTALS, TIPS, DIPS, geometric_frame

HANDS = ROOT / "data/taco_v1/pour_bowl_plate/hand_poses/Hand_Poses/(pour in some, bowl, plate)/20230927_017"
MODELS = ROOT / "data/taco_v1/hand_poses_v1/mano_v1_2/models"


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("undefined axis")
    return vector / norm


def point_geometry(mesh, point, longitudinal_axis):
    nearest, distance, face = trimesh.proximity.closest_point(mesh, np.asarray(point)[None])
    axis = unit(longitudinal_axis)
    support = np.asarray(mesh.vertices) @ axis
    point_support = float(np.asarray(point) @ axis)
    return dict(point_m=np.asarray(point).tolist(), nearest_surface_m=nearest[0].tolist(),
        unsigned_surface_distance_m=float(distance[0]),
        axial_gap_to_farthest_native_vertex_m=float(support.max() - point_support),
        nearest_triangle_normal=mesh.face_normals[face[0]].tolist(),
        mesh_watertight=bool(mesh.is_watertight),
        normal_dot_longitudinal_axis=float(mesh.face_normals[face[0]] @ axis))


def inspect():
    import smplx
    import torch
    from smplx.utils import Struct
    from smplx.lbs import batch_rigid_transform

    scene, _, human, _, inputs = load_inputs()
    inputs.append(artifact(UPSTREAM / "retarget_config.yaml"))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meshes, _ = visual_meshes(scene, model)
    records, neutral_frames, endpoint_frames = {}, {}, {}
    for hi, side in enumerate(SIDES):
        pose_path = HANDS / f"{side}_hand.pkl"
        shape_path = HANDS / f"{side}_hand_shape.pkl"
        model_path = MODELS / f"MANO_{side.upper()}.pkl"
        inputs += [artifact(p) for p in (pose_path, shape_path, model_path)]
        poses, _, betas, _ = load_taco_mano_sequence(pose_path, shape_path)
        model_data = _load_model_data(model_path)
        layer = smplx.MANO("unused", data_struct=Struct(**model_data), is_rhand=side == "right",
                           use_pca=False, flat_hand_mean=True, create_transl=False)
        vertices, joints, faces, _, _ = reconstruct_taco_mano(pose_path, shape_path, model_path, side=side)
        np.testing.assert_allclose(vertices[:, MANO_TIP_VERTICES[side]], joints[:, TIPS], atol=1e-12, rtol=0)
        np.testing.assert_allclose(joints[:, TIPS] + human["T_sim_world"][:3, 3],
                                   human["T_sim_fingertip_target"][:, hi, :, :3, 3], atol=1e-6, rtol=0)
        neutral = {}
        for label, shape in (("mean_shape", np.zeros(10)), ("episode_shape", betas)):
            with torch.no_grad():
                output = layer(global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45),
                               betas=torch.tensor(shape[None], dtype=torch.float32))
            points = np.concatenate((output.joints[0].numpy(), output.vertices[0, MANO_TIP_VERTICES[side]].numpy()))[MANO21_SOURCE]
            normal = np.cross(points[5] - points[0], points[17] - points[0]) * (-1 if side == "left" else 1)
            frames = np.array([geometric_frame(normal, points[t] - points[d]) for t, d in zip(TIPS, DIPS)])
            neutral[label] = (output.vertices[0].numpy(), points, frames)
        neutral_frames[side] = neutral["mean_shape"][2]
        urdf = ET.parse(UPSTREAM / f"xhand_{side}.urdf").getroot()
        root = model.body(f"{side}_hand_link").id
        palm = model.site(f"{side}_palm").id
        middle = model.body(f"{side}_hand_mid_link1").id
        robot_frame = geometric_frame(data.site_xmat[palm].reshape(3, 3)[:, 0], data.xpos[middle] - data.xpos[root])
        per_finger = {}
        endpoint_frames[side] = []
        for fi, finger in enumerate(FINGERS):
            vid = int(MANO_TIP_VERTICES[side][fi])
            sid = model.site(f"{side}_{finger}_tip").id
            bid = int(model.site_bodyid[sid])
            geom = next(gid for gid in meshes if model.geom_bodyid[gid] == bid)
            native = meshes[geom]
            fixed = next(j for j in urdf.findall("joint") if j.get("type") == "fixed"
                and j.find("parent").get("link") == model.body(bid).name and j.find("child").get("link").endswith("tip"))
            endpoint = np.fromstring(fixed.find("origin").get("xyz"), sep=" ")
            axis = unit(endpoint)
            robot = dict(inherited_site=point_geometry(native, model.site_pos[sid], axis),
                         named_urdf_tip=point_geometry(native, endpoint, axis), longitudinal_axis_local=axis.tolist())
            end_frame = data.site_xmat[sid].reshape(3, 3).T @ geometric_frame(robot_frame[:, 0], data.xmat[bid].reshape(3, 3) @ axis)
            endpoint_frames[side].append(end_frame)
            mano = {}
            for label, (verts, points, frames) in neutral.items():
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                direction = unit(points[TIPS[fi]] - points[DIPS[fi]])
                # Restrict distal support to vertices dominated by this MANO joint.
                region = np.argmax(model_data["weights"], axis=1) == MANO_DISTALS[fi]
                maximum = (verts[region] @ direction).max()
                mano[label] = dict(vertex_id=vid, surface_point_m=verts[vid].tolist(),
                    normal_dot_distal_axis=float(mesh.vertex_normals[vid] @ direction),
                    distal_axial_gap_to_support_m=float(maximum - verts[vid] @ direction),
                    distal_joint_skinning_weight=float(model_data["weights"][vid, MANO_DISTALS[fi]]),
                    anatomical_frame=frames[fi].tolist())
            per_finger[finger] = dict(human=mano, robot=robot)
        rotations = Rotation.from_rotvec(poses.astype(float).reshape(-1, 3)).as_matrix().reshape(-1, 16, 3, 3)
        with torch.no_grad():
            _, global_t = batch_rigid_transform(torch.from_numpy(rotations), torch.zeros(len(poses), 16, 3, dtype=torch.float64), layer.parents)
        distal_rot = global_t.numpy()[:, MANO_DISTALS, :3, :3]
        mean_frames = distal_rot @ neutral["mean_shape"][2]
        shape_frames = distal_rot @ neutral["episode_shape"][2]
        angular_change = np.rad2deg(Rotation.from_matrix((mean_frames.swapaxes(-1, -2) @ shape_frames).reshape(-1, 3, 3)).magnitude()).reshape(len(poses), 5)
        # A point on a skinned surface is not a rigid distal-joint frame. Quantify
        # the difference, but do not replace the released GT with a rigid proxy.
        local_vectors = np.einsum("tfji,tfj->tfi", distal_rot, joints[:, TIPS] - joints[:, DIPS])
        vector_variation = np.linalg.norm(local_vectors - local_vectors.mean(axis=0), axis=-1)
        records[side] = dict(fingers=per_finger,
            neutral_shape_frame_change_max_deg=angular_change.max(axis=0).tolist(),
            gt_tip_to_DIP_distal_local_vector_variation_max_m=vector_variation.max(axis=0).tolist())
    verify_artifacts(inputs)
    return dict(status="geometric_semantics_audit_not_new_reference", inputs=inputs,
        code=[artifact(Path(__file__)), artifact(ROOT / "src/egoengine_repro/evaluation/taco_surface.py")],
        per_hand=records,
        definitions=dict(point="MANO released surface landmark, not an inferred task contact point",
            human_longitudinal_axis="neutral DIP-to-released-tip direction transported by distal MANO FK",
            current_robot_longitudinal_axis="distal joint to inherited XML site, not necessarily the URDF longitudinal axis",
            roll_axis="neutral projected palm normal, not a measured fingertip-pad or nail normal"),
        limitations=["Neither XML/YAML naming nor a nearest mesh point establishes the author's intended mapping.",
            "Surface normals near a rounded fingertip are not a unique nail/pad orientation convention.",
            "No new mapping, site, weight, reference or physical initial state is adopted."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for side, hand in report["per_hand"].items():
        for finger, values in hand["fingers"].items():
            human = values["human"]["episode_shape"]
            robot = values["robot"]
            print(side, finger, "human_tip normal dot / axial gap mm", human["normal_dot_distal_axis"], human["distal_axial_gap_to_support_m"] * 1000,
                "robot site/tip surface distance mm", robot["inherited_site"]["unsigned_surface_distance_m"] * 1000, robot["named_urdf_tip"]["unsigned_surface_distance_m"] * 1000)
        print(side, "shape frame change degrees", hand["neutral_shape_frame_change_max_deg"])


if __name__ == "__main__":
    main()
