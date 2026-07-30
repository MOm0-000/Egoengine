import xml.etree.ElementTree as ET
from pathlib import Path

from video_to_spider.export.spider_runner import _inspect_hand_floor_contacts


def test_inspect_hand_floor_contacts_counts_floor_pairs(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        "<mujoco><contact>"
        '<pair name="collision_hand_right_thumb_0_floor" solref="0.02 1"/>'
        '<pair name="collision_hand_right_thumb_0_right_object_0" solref="0.02 1"/>'
        "</contact></mujoco>"
    )

    report = _inspect_hand_floor_contacts([scene])

    pairs = {
        pair.get("name"): pair.get("solref")
        for pair in ET.parse(scene).getroot().findall("./contact/pair")
    }
    assert pairs["collision_hand_right_thumb_0_floor"] == "0.02 1"
    assert pairs["collision_hand_right_thumb_0_right_object_0"] == "0.02 1"
    assert report["pair_count_by_scene"] == {"scene.xml": 1}
