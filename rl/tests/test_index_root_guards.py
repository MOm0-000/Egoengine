"""The bilateral index-root guards must stay local, reproducible, and active."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import resolve_artifact_path

SCENE = ROOT / "models/taco_xhand/xhand/bimanual/taco_pour_bowl_plate_20230927_017/scene_source_contacts_mass.xml"
RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_bilateral_guard_v1"
AUDIT_DIR = ROOT / "runs/taco_pour_bilateral_index_guard_v1"


def test_guard_audit_and_active_reference_are_consistent():
    for side in ("right", "left"):
        report = json.loads((AUDIT_DIR / f"{side}_index_guard_audit.json").read_text())
        assert report["status"] == f"{side}_index_root_guard_formal_audit_passed"
        assert report["retarget_comparison"]["pre_guard_native_interference_frames"] > 0
        assert report["retarget_comparison"]["post_guard_native_interference_frames"] == 0
        assert report["retarget_comparison"]["baseline_scene_only_rebases_compiler_meshdir"]
        assert report["retarget_comparison"]["solver_numerics_equal"]
        assert report["active_retarget_trajectory"]["native_interference_frames"] == 0
        assert min(report["active_retarget_trajectory"]["guard_minimum_distance_m"].values()) > 0
        assert report["active_retarget_trajectory"]["planning_collision_buffer_m"] == 2e-6
        assert report["active_retarget_trajectory"]["accepted_min_self_collision_distance_m"] == -1e-6
        assert min(report["active_retarget_trajectory"]["guard_minimum_distance_m"].values()) < 2e-6
        for grid in ("sampled_grid", "midpoint_holdout_grid"):
            assert report["classification_check"][grid]["false_positive_count"] == 0
            assert report["classification_check"][grid]["false_negative_count"] == 0
        assert report["classification_method"] == (
            "direct_native_FCL_predicate_and_actual_MuJoCo_guard_distance"
        )
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    ppo = yaml.safe_load((ROOT / "configs/taco_pour_bimanual_ppo.yaml").read_text())
    assert Path(protocol["inputs"]["active_robot_reference"]) == RUN / "robot_reference.npz"
    assert Path(ppo["data_path"]) == RUN / "robot_reference.npz"


def test_each_side_guard_geometries_are_isolated_to_two_explicit_pairs():
    root = ET.parse(SCENE).getroot()
    for side in ("right", "left"):
        guards = {geom.get("name"): geom for geom in root.findall(".//geom")
                  if f"collision_hand_{side}_" in geom.get("name", "")
                  and "index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")}
        assert len(guards) == 4
        assert all(geom.get("type") == "sphere" and geom.get("density") == "0"
                   and geom.get("contype") == "0" and geom.get("conaffinity") == "0"
                   for geom in guards.values())
        involved = [pair for pair in root.findall("contact/pair")
                    if pair.get("geom1") in guards or pair.get("geom2") in guards]
        assert len(involved) == 2
        assert all(pair.get("geom1") in guards and pair.get("geom2") in guards
                   for pair in involved)
        assert np.array_equal(
            np.fromstring(root.find(f".//joint[@name='{side}_hand_index_bend_joint']").get("range"), sep=" "),
            [-0.175, 0.175],
        )


def test_active_reference_keeps_all_four_guard_pairs_nonpenetrating():
    model, data = mujoco.MjModel.from_xml_path(str(SCENE)), None
    data = mujoco.MjData(model)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as archive:
        qpos = archive["qpos"]
    for side in ("right", "left"):
        for label in ("negative", "positive"):
            pair = (model.geom(f"collision_hand_{side}_palm_index_root_{label}_guard").id,
                    model.geom(f"collision_hand_{side}_index_root_{label}_guard").id)
            distances = []
            for q in qpos:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                distances.append(mujoco.mj_geomDistance(model, data, *pair, 0.05, None))
            assert min(distances) > 0


def test_formal_gpu_capacity_and_historical_relocation_are_auditable():
    capacity = json.loads((ROOT / "runs/taco_pour_bilateral_index_guard_v1/capacity_formal_4env.json").read_text())
    assert capacity["gpu"]["snapshots_complete"]
    assert capacity["gpu"]["all_snapshots_finite"]
    assert not capacity["gpu"]["overflow_seen"]
    old = json.loads((ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1/retarget_report.json").read_text())
    record = {"path": old["scene"], "sha256": old["scene_sha256"]}
    resolved = resolve_artifact_path(record)
    assert resolved == (ROOT / "TRASH/superseded_formal_models/2026-09-20_right_index_guard/scene_source_contacts_mass.before_right_index_guard.xml")
    right_only = json.loads((ROOT / "TRASH/superseded_runs/2026-09-20_left_index_guard/taco_pour_bimanual_mano_fk_right_guard_v1/retarget_report.json").read_text())
    resolved = resolve_artifact_path({"path": right_only["scene"], "sha256": right_only["scene_sha256"]})
    assert resolved == (ROOT / "TRASH/superseded_formal_models/2026-09-20_left_index_guard/scene_source_contacts_mass.right_guard_only.xml")
