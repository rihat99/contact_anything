from __future__ import annotations

import numpy as np

from scripts.render_climbing_video_contacts import (
    CONTACT_COLOR,
    FREE_COLOR,
    _draw_contacts,
    select_random_scenes,
    temporal_window_requests,
)


def test_random_selection_is_reproducible_and_unique_by_source_video():
    scenes = ["video_a_0001", "video_a_0002", "video_b_0001", "video_c_0003"]
    first = select_random_scenes(scenes, 3, seed=17)
    second = select_random_scenes(scenes, 3, seed=17)
    assert first == second
    assert len({video_id for video_id, _ in first}) == 3
    assert all(scene.startswith(video_id) for video_id, scene in first)


def test_draw_contacts_uses_red_above_threshold_and_green_below():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    probs = np.array([[0.9, 0.1]], dtype=np.float32)
    points = np.array([[[25.0, 30.0], [80.0, 70.0]]], dtype=np.float32)
    _draw_contacts(frame, probs, points, threshold=0.2)
    assert tuple(int(v) for v in frame[30, 25]) == CONTACT_COLOR
    assert tuple(int(v) for v in frame[70, 80]) == FREE_COLOR


def _emitted_frames(requests):
    return [
        (person, positions[offset], seq_len)
        for seq_len, group in requests.items()
        for person, positions, offsets in group
        for offset in offsets
    ]


def test_temporal_requests_use_t5_centers_and_t3_boundaries_once():
    requests = temporal_window_requests(np.ones((1, 8), dtype=bool))
    assert requests[3] == [
        (0, (0, 1, 2), (0, 1)),
        (0, (5, 6, 7), (1, 2)),
    ]
    assert requests[5] == [
        (0, (0, 1, 2, 3, 4), (2,)),
        (0, (1, 2, 3, 4, 5), (2,)),
        (0, (2, 3, 4, 5, 6), (2,)),
        (0, (3, 4, 5, 6, 7), (2,)),
    ]
    emitted = _emitted_frames(requests)
    assert [(person, frame) for person, frame, _ in emitted] == [
        (0, 0), (0, 1), (0, 6), (0, 7),
        (0, 2), (0, 3), (0, 4), (0, 5),
    ]
    assert len({(person, frame) for person, frame, _ in emitted}) == 8


def test_temporal_requests_treat_gaps_as_boundaries_and_cover_short_tracks():
    valid = np.array([
        [1, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0],
    ], dtype=bool)
    requests = temporal_window_requests(valid)
    emitted = _emitted_frames(requests)
    emitted_pairs = {(person, frame) for person, frame, _ in emitted}
    expected = set(zip(*np.nonzero(valid)))
    assert emitted_pairs == expected
    assert len(emitted) == len(expected)
    assert {(person, frame) for person, frame, t in emitted if t == 1} == {
        (0, 0), (0, 1),
    }


def test_temporal_requests_reject_non_matrix_mask():
    with np.testing.assert_raises_regex(ValueError, "people, frames"):
        temporal_window_requests(np.ones(5, dtype=bool))
