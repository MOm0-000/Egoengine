from types import SimpleNamespace

import numpy as np

from video_to_spider.adapters.sam3 import (
    _configure_state_offload,
    _invalid_spans,
    _load_mask_artifact,
    _mask_iou,
    _offload_session_frames,
    _rank_instances,
)


class _FakeTracker:
    def __init__(self):
        self.kwargs = None

    def init_state(self, **kwargs):
        self.kwargs = kwargs
        return {"configured": True}


def test_configure_state_offload_enables_inner_tracker_cpu_storage():
    tracker = _FakeTracker()
    model = SimpleNamespace(tracker=tracker, _init_new_sam2_state=lambda _: None)
    predictor = SimpleNamespace(model=model)
    _configure_state_offload(predictor)
    state = {
        "feature_cache": {"frame": 1}, "orig_height": 1080,
        "orig_width": 1920, "num_frames": 90,
    }
    assert model._init_new_sam2_state(state) == {"configured": True}
    assert tracker.kwargs["offload_video_to_cpu"] is True
    assert tracker.kwargs["offload_state_to_cpu"] is True
    assert tracker.kwargs["cached_features"] is state["feature_cache"]


class _FakeGpuFrames:
    is_cuda = True

    def cpu(self):
        return "cpu-frames"


def test_offload_session_frames_moves_full_clip_back_to_cpu():
    image_batch = SimpleNamespace(tensors=_FakeGpuFrames())
    state = {"input_batch": SimpleNamespace(img_batch=image_batch)}
    predictor = SimpleNamespace(
        _all_inference_states={"session": {"state": state}}
    )
    _offload_session_frames(predictor, "session")
    assert image_batch.tensors == "cpu-frames"


def test_invalid_spans_and_overlap_iou():
    valid = np.array([False, False, True, False, True, False, False])
    assert _invalid_spans(valid) == [(0, 2), (3, 4), (5, 7)]
    first = np.array([[1, 1], [0, 0]], dtype=bool)
    second = np.array([[0, 1], [1, 0]], dtype=bool)
    assert np.isclose(_mask_iou(first, second), 1.0 / 3.0)


def test_load_mask_artifact_normalizes_suppression_sentinels(tmp_path):
    path = tmp_path / "masks.npz"
    masks = np.ones((3, 2, 2), dtype=np.uint8)
    masks[2] = 0
    np.savez_compressed(
        path,
        frame_indices=np.arange(3),
        timestamps_s=np.arange(3) / 30.0,
        masks=masks,
        valid=np.ones(3, dtype=bool),
        confidence=np.array([0.8, -10000.0, 0.9], dtype=np.float32),
        object_ids=np.ones(3, dtype=np.int64),
    )
    frame_indices, timestamps, result = _load_mask_artifact(path)
    assert frame_indices.tolist() == [0, 1, 2]
    assert np.allclose(timestamps, [0.0, 1.0 / 30.0, 2.0 / 30.0])
    assert result["confidence"].tolist() == [np.float32(0.8), 0.0, np.float32(0.9)]
    assert result["valid"].tolist() == [True, False, False]


def test_instance_ranking_prefers_moving_handheld_target_over_static_container():
    shape = (100, 120)
    container = np.zeros(shape, dtype=bool)
    container[30:80, 55:115] = True
    handheld = np.zeros(shape, dtype=bool)
    handheld[45:60, 35:50] = True
    hand = np.zeros(shape, dtype=bool)
    hand[40:70, 42:72] = True
    per_frame = {}
    for frame_index, offset in enumerate((0, 15, 30, 45)):
        moving = np.zeros(shape, dtype=bool)
        moving[45:60, 35 + offset:50 + offset] = True
        per_frame[frame_index] = {
            "out_obj_ids": np.array([1, 2]),
            "out_binary_masks": np.stack([container, moving]),
            "out_probs": np.array([0.95, 0.85]),
        }
    selected, diagnostics = _rank_instances(
        np.array([1, 2]), np.stack([container, handheld]), np.array([0.95, 0.85]),
        hand, per_frame,
    )
    assert selected == 1
    assert diagnostics["motion_span_px"][1] > diagnostics["motion_span_px"][0]
    assert diagnostics["area_ratio_to_hand"][0] > diagnostics["area_ratio_to_hand"][1]
