"""The adopted right index-root guard must stay local, reproducible, and active."""

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
RUN = ROOT / "runs/taco_pour_bimanual_mano_fk_right_guard_v1"
AUDIT = ROOT / "runs/taco_pour_right_index_guard_v1/right_index_guard_audit.json"


def test_guard_audit_and_active_reference_are_consistent():
    report = json.loads(AUDIT.read_text())
    assert report["status"] == "right_index_root_guard_formal_audit_passed"
    assert report["retarget_comparison"]["pre_guard_native_interference_frames"] == 52
    assert report["retarget_comparison"]["post_guard_native_interference_frames"] == 0
    assert report["active_retarget_trajectory"]["native_interference_frames"] == 0
    assert min(report["active_retarget_trajectory"]["guard_minimum_distance_m"].values()) > 0
    for result in report["classification_check"].values():
        assert result["sampled_false_positive_count"] == 0
        assert result["sampled_false_negative_count"] == 0
    protocol = yaml.safe_load((ROOT / "configs/replay_rl_protocol.yaml").read_text())
    ppo = yaml.safe_load((ROOT / "configs/taco_pour_bimanual_ppo.yaml").read_text())
    assert Path(protocol["inputs"]["active_robot_reference"]) == RUN / "robot_reference.npz"
    assert Path(ppo["data_path"]) == RUN / "robot_reference.npz"


def test_guard_geometries_are_isolated_to_two_explicit_pairs():
    root = ET.parse(SCENE).getroot()
    guards = {geom.get("name"): geom for geom in root.findall(".//geom")
              if "index_root_" in geom.get("name", "") and "guard" in geom.get("name", "")}
    assert len(guards) == 4
    assert all(geom.get("type") == "sphere" and geom.get("density") == "0"
               and geom.get("contype") == "0" and geom.get("conaffinity") == "0"
               for geom in guards.values())
    involved = [pair for pair in root.findall("contact/pair")
                if pair.get("geom1") in guards or pair.get("geom2") in guards]
    assert len(involved) == 2
    assert all(pair.get("geom1") in guards and pair.get("geom2") in guards for pair in involved)
    assert np.array_equal(
        np.fromstring(root.find(".//joint[@name='right_hand_index_bend_joint']").get("range"), sep=" "),
        [-0.175, 0.175],
    )


def test_active_reference_keeps_both_guards_nonpenetrating():
    model, data = mujoco.MjModel.from_xml_path(str(SCENE)), None
    data = mujoco.MjData(model)
    with np.load(RUN / "robot_reference.npz", allow_pickle=False) as archive:
        qpos = archive["qpos"]
    for label in ("negative", "positive"):
        pair = (model.geom(f"collision_hand_right_palm_index_root_{label}_guard").id,
                model.geom(f"collision_hand_right_index_root_{label}_guard").id)
        distances = []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            distances.append(mujoco.mj_geomDistance(model, data, *pair, 0.05, None))
        assert min(distances) > 0


def test_formal_gpu_capacity_and_historical_relocation_are_auditable():
    capacity = json.loads((ROOT / "runs/taco_pour_right_index_guard_v1/capacity_formal_4env.json").read_text())
    assert capacity["gpu"]["snapshots_complete"]
    assert capacity["gpu"]["all_snapshots_finite"]
    assert not capacity["gpu"]["overflow_seen"]
    old = json.loads((ROOT / "TRASH/superseded_runs/2026-09-20_right_index_guard/taco_pour_bimanual_mano_fk_v1/retarget_report.json").read_text())
    record = {"path": old["scene"], "sha256": old["scene_sha256"]}
    resolved = resolve_artifact_path(record)
    assert resolved == (ROOT / "TRASH/superseded_formal_models/2026-09-20_right_index_guard/scene_source_contacts_mass.before_right_index_guard.xml")
