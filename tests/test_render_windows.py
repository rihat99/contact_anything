"""Centered sliding-window planner for the ClimbingVideos overlay renderer.

Exercises :func:`scripts.render_climbing_video_contacts.plan_track_windows` — the
pure per-person planner that tiles each stride-``s`` parity's contiguous valid
track with centered windows of ``T`` sampled frames. The load-bearing invariant
is exactly-once coverage: every valid source frame is emitted by exactly one
``(window, offset)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.render_climbing_video_contacts import (
    plan_track_windows,
    sliding_window_requests,
)


def _emitted_positions(valid_row, seq_len, stride) -> list[int]:
    """Source frame positions the planner emits (with multiplicity)."""
    emitted = []
    for positions, offsets in plan_track_windows(valid_row, seq_len, stride):
        for offset in offsets:
            emitted.append(positions[offset])
    return emitted


def _check_well_formed(valid_row, seq_len, stride) -> None:
    """Assert every request is a valid stride-``s`` window over valid frames."""
    valid_row = np.asarray(valid_row, dtype=bool)
    max_len = min(seq_len, int(valid_row.sum()) or 1)
    for positions, offsets in plan_track_windows(valid_row, seq_len, stride):
        # Full windows carry T frames; a short track collapses to one smaller window.
        assert 1 <= len(positions) <= max_len
        # Frames within a window are the parity's contiguous samples: step == stride.
        steps = np.diff(positions)
        assert np.all(steps == stride), (positions, stride)
        # Every window frame is a real (valid) source frame.
        assert all(valid_row[p] for p in positions), positions
        # Emitted offsets index into the window.
        assert all(0 <= off < len(positions) for off in offsets), (offsets, positions)


@pytest.mark.parametrize("length", [1, 5, 16, 33, 100])
def test_exactly_once_all_valid(length):
    """T=16 s=2: every valid frame of a full track is emitted exactly once."""
    valid_row = np.ones(length, dtype=bool)
    emitted = _emitted_positions(valid_row, seq_len=16, stride=2)
    assert sorted(emitted) == list(range(length))       # covers all, no gaps
    assert len(emitted) == len(set(emitted))             # no duplicates
    _check_well_formed(valid_row, 16, 2)


@pytest.mark.parametrize("length", [33, 100])
def test_exactly_once_with_holes(length):
    """T=16 s=2 with invalid frames: valid frames covered once, holes never."""
    rng = np.random.default_rng(length)
    valid_row = rng.random(length) > 0.25                # ~25% holes
    valid_row[0] = valid_row[-1] = True                  # keep both boundaries valid
    expected = sorted(int(p) for p in np.flatnonzero(valid_row))
    emitted = _emitted_positions(valid_row, seq_len=16, stride=2)
    assert sorted(emitted) == expected
    assert len(emitted) == len(set(emitted))
    _check_well_formed(valid_row, 16, 2)


def test_reduces_to_per_frame():
    """T=1 s=1 emits one single-frame window per valid frame (per-frame path)."""
    valid_row = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    requests = plan_track_windows(valid_row, seq_len=1, stride=1)
    assert all(positions == (int(np.flatnonzero(valid_row)[i]),) and offsets == (0,)
               for i, (positions, offsets) in enumerate(requests))
    assert [positions[0] for positions, _ in requests] == \
        [int(p) for p in np.flatnonzero(valid_row)]


def test_edge_window_clamping():
    """Long track: first/last windows clamp to the track boundary; interior full."""
    length = 33
    valid_row = np.ones(length, dtype=bool)
    requests = plan_track_windows(valid_row, seq_len=16, stride=1)
    # Every window of a track longer than T carries exactly T frames.
    assert all(len(positions) == 16 for positions, _ in requests)
    first_positions = requests[0][0]
    last_positions = requests[-1][0]
    assert first_positions[0] == 0                       # left edge clamp
    assert last_positions[-1] == length - 1              # right edge clamp
    # Boundary windows own their uncovered edge rows (frame 0 and the last frame).
    assert 0 in (first_positions[o] for o in requests[0][1])
    assert length - 1 in (last_positions[o] for o in requests[-1][1])


def test_short_track_single_window():
    """A track shorter than T is one window emitting every row (down to T=1)."""
    valid_row = np.ones(5, dtype=bool)
    requests = plan_track_windows(valid_row, seq_len=16, stride=1)
    assert len(requests) == 1
    positions, offsets = requests[0]
    assert positions == (0, 1, 2, 3, 4)
    assert offsets == (0, 1, 2, 3, 4)


def test_parities_cover_all_source_frames():
    """Stride-2 splits into two parities; together they cover every valid frame."""
    valid_row = np.ones(20, dtype=bool)
    valid_row[7] = False
    emitted = _emitted_positions(valid_row, seq_len=16, stride=2)
    assert sorted(emitted) == [p for p in range(20) if p != 7]
    assert len(emitted) == len(set(emitted))


def test_sliding_window_requests_grouping():
    """Multi-person requests group by window length and carry the person index."""
    valid_mask = np.ones((2, 40), dtype=bool)
    grouped = sliding_window_requests(valid_mask, seq_len=16, stride=2)
    # T=16 s=2 over 40 frames: each parity is 20 samples > T, so every window is a
    # full T=16 window (short-track fallbacks would appear under other keys).
    assert set(grouped) == {16}
    persons = sorted(person for person, _, _ in grouped[16])
    assert persons.count(0) == persons.count(1)          # symmetric across people
    emitted = [
        positions[off]
        for person, positions, offsets in grouped[16]
        if person == 0
        for off in offsets
    ]
    assert sorted(emitted) == list(range(40))
