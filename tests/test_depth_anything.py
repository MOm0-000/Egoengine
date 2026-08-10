from pathlib import Path

import numpy as np

from video_to_spider.adapters.depth_anything import _load_optional_masks


def test_optional_masks_decompress_each_npz_member_once(tmp_path: Path, monkeypatch):
    segmentation = tmp_path / "segmentation"
    segmentation.mkdir()
    (segmentation / "object_masks.npz").touch()
    (segmentation / "hand_masks.npz").touch()

    frame_indices = np.array([10, 11, 12], dtype=np.int64)
    masks = np.zeros((3, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    masks[2, 1, 1] = True
    access_counts = {"frame_indices": 0, "masks": 0}

    class CountingArtifact:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, key):
            access_counts[key] += 1
            return {"frame_indices": frame_indices, "masks": masks}[key]

    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: CountingArtifact())
    rows = [{"source_frame_index": 12}, {"source_frame_index": 10}]

    object_masks, hand_masks = _load_optional_masks(tmp_path, rows)

    assert access_counts == {"frame_indices": 2, "masks": 2}
    assert np.array_equal(object_masks, masks[[2, 0]])
    assert np.array_equal(hand_masks, masks[[2, 0]])
