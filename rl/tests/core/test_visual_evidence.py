from pathlib import Path

import numpy as np
import pytest

from video_to_spider.rl.core.visual_evidence import (
    contact_transitions,
    endpoint_status,
    pending_review,
    resolve_frame_map,
    semantic_contact_sets,
    top_positive_increments,
    validate_publication,
    video_frame_map,
)


def test_frame_map_supports_nonzero_source_offset_and_rejects_duplicates():
    rows = resolve_frame_map(
        np.asarray([7, 8, 9]),
        np.asarray([0.0, 0.04, 0.08]),
        [None] * 7 + [0.0, 0.04, 0.08],
        endpoint_start=0,
        endpoint_stop=2,
        provenance="fixture",
    )
    assert [row["decoded_video_frame_index"] for row in rows] == [7, 8, 9]
    assert all(row["alignment_status"] == "aligned" for row in rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_frame_map(
            np.asarray([7, 7, 9]), np.asarray([0.0, 0.04, 0.08]), [0.0] * 10,
            endpoint_start=0, endpoint_stop=2, provenance="fixture",
        )


def test_frame_map_marks_missing_and_half_interval_mismatch_unaligned():
    rows = resolve_frame_map(
        np.asarray([0, 1, 4]),
        np.asarray([0.0, 0.1, 0.2]),
        [0.0, 0.18],
        endpoint_start=0,
        endpoint_stop=2,
        provenance="fixture",
    )
    assert rows[1]["alignment_status"] == "unaligned"
    assert rows[2]["rgb_available"] is False
    assert rows[2]["alignment_status"] == "unaligned"


def test_endpoint_status_distinguishes_initial_failure_unreached_and_unrecorded():
    available = {0: 0, 1: 1, 3: 2}
    assert endpoint_status(0, available, first_failure_endpoint=3) == "INITIAL"
    assert endpoint_status(2, available, first_failure_endpoint=3) == "NOT RECORDED"
    assert endpoint_status(3, available, first_failure_endpoint=3) == "OLD TRACKING TERMINATED"
    assert endpoint_status(4, available, first_failure_endpoint=3) == "NOT REACHED"


def test_video_map_repetition_is_uniform_and_explicit():
    rows = [
        {"endpoint": 4, "reference_time_s": 0.1, "source_frame_id": 14},
        {"endpoint": 5, "reference_time_s": 0.2, "source_frame_id": 15},
    ]
    realtime = video_frame_map(rows, slowdown_factor=1)
    slow = video_frame_map(rows, slowdown_factor=4)
    assert [row["endpoint"] for row in realtime] == [4, 5]
    assert [row["endpoint"] for row in slow] == [4] * 4 + [5] * 4
    assert [row["uniform_repeat_index"] for row in slow[:4]] == [0, 1, 2, 3]


def test_events_are_stable_to_contact_row_reordering_and_ignore_unknown_groups():
    source = np.asarray([1, 0, 0, 1])
    groups = np.asarray(["right_hand_tool", "right_hand_tool", "self", "right_hand_tool"])
    roles1 = np.asarray(["right_hand:index", "right_hand:index", "x", "right_hand:thumb"])
    roles2 = np.asarray(["tool", "tool", "x", "tool"])
    states = semantic_contact_sets(
        source, groups, roles1, roles2, meaningful_groups={"right_hand_tool"},
    )
    events = contact_transitions(states, start=0, stop=2)
    assert events == [
        {"endpoint": 1, "transition": "appeared", "semantic_identity": "right_hand_tool|right_hand:index|tool"},
        {"endpoint": 2, "transition": "appeared", "semantic_identity": "right_hand_tool|right_hand:thumb|tool"},
    ]


def test_positive_increment_ranking_uses_only_positive_and_earlier_ties():
    assert top_positive_increments({0: 1.0, 1: 3.0, 2: 2.0, 3: 4.0}, 3) == [(1, 2.0), (3, 2.0)]


def test_review_is_pending_even_when_an_old_review_file_exists(tmp_path: Path):
    (tmp_path / "visual_review.json").write_text('{"status":"complete"}')
    assert pending_review("abc").status == "PENDING"


def test_publication_requires_all_endpoint_views_and_indexes(tmp_path: Path):
    for view in ("top", "oblique"):
        directory = tmp_path / "previews" / view
        directory.mkdir(parents=True)
        for endpoint in range(2):
            (directory / f"endpoint_{endpoint:06d}.jpg").write_bytes(b"jpeg")
        sheet = tmp_path / "sheets" / view
        sheet.mkdir(parents=True)
        (sheet / "page_000.jpg").write_bytes(b"jpeg")
    for name in ("VISUAL_INDEX.md", "index.html", "frame_map.csv", "events.csv", "visual_review.json"):
        (tmp_path / name).write_text("ok")
    result = validate_publication(tmp_path, endpoint_count=2, views=("top", "oblique"), expected_sheet_count=2)
    assert result["valid"]
    (tmp_path / "previews/top/endpoint_000001.jpg").unlink()
    assert not validate_publication(tmp_path, endpoint_count=2, views=("top", "oblique"), expected_sheet_count=2)["valid"]
