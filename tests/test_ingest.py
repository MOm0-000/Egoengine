from pathlib import Path

import h5py
import numpy as np

from video_to_spider.ingest.egodex import EpisodeRef, keyword_candidates, select_instruction
from video_to_spider.ingest.egodex_ground_truth import load_hand_ground_truth


def test_instruction_selection_and_keywords_do_not_use_gt_object_attr():
    attrs = {
        "which_llm_description": "2", "llm_description": "Place a red cup on the table.",
        "llm_description2": "Remove lids from four blue cups.", "llm_objects": ["forbidden_gt_name"],
    }
    instruction, source = select_instruction(attrs, "add_remove_lid")
    assert source == "llm_description2"
    assert instruction.startswith("Remove lids")
    words = keyword_candidates(instruction, "add_remove_lid")
    assert "lid" in words and "cup" in words
    assert "four" not in words
    assert "forbidden_gt_name" not in words


def test_ground_truth_reader_is_explicit(tmp_path: Path):
    path = tmp_path / "episode.hdf5"
    names = ["rightHand", "rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip", "rightRingFingerTip", "rightLittleFingerTip"]
    with h5py.File(path, "w") as handle:
        for name in names:
            handle.create_dataset(f"transforms/{name}", data=np.repeat(np.eye(4)[None], 2, axis=0))
            handle.create_dataset(f"confidences/{name}", data=np.ones(2))
    result = load_hand_ground_truth(path, "right")
    assert result["T_world_joint"].shape == (2, 6, 4, 4)
    assert result["confidence"].shape == (2, 6)
    assert result["confidence_source"] == "recorded"


def test_ground_truth_reader_accepts_missing_optional_confidence(tmp_path: Path):
    path = tmp_path / "episode_without_confidence.hdf5"
    names = ["leftHand", "leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip", "leftRingFingerTip", "leftLittleFingerTip"]
    with h5py.File(path, "w") as handle:
        for name in names:
            handle.create_dataset(f"transforms/{name}", data=np.repeat(np.eye(4)[None], 3, axis=0))
    result = load_hand_ground_truth(path, "left")
    np.testing.assert_array_equal(result["confidence"], np.ones((3, 6)))
    assert result["confidence_source"] == "missing_assumed_valid"
