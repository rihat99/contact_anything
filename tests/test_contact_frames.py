"""Unit tests of the contact sets (CPU).

* the ``frames35`` table in :mod:`model.contact_frames` is the authored JSON's own
  ``{name: vertex}`` and BetterHuman's own parent resolution — the table is what every
  loader, head and loss indexes by, so a silent drift would mislabel the whole corpus;
* ``parent_joint22`` folds the finger parents onto their wrist;
* ``group_of`` / ``fold_max`` / ``fold_sum``: the six-group fold of a 35-slot prediction,
  and the identity on ``kindyn6``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from model.contact_frames import (FRAMES35, KINDYN_GROUP_JOINTS, KINDYN_GROUP_NAMES,
                                  NUM_KINDYN_GROUPS, body22_parent, contact_set,
                                  frames35_json_vertices)

REPO = Path(__file__).resolve().parents[1]


def test_the_table_is_the_authored_json():
    slots = contact_set("frames35")
    authored = frames35_json_vertices()
    assert slots.count == 35 and len(authored) == 35
    assert list(authored) == list(slots.slot_names)              # order included
    assert [authored[name] for name in slots.slot_names] == list(slots.vertex_ids)


def test_the_table_is_betterhumans_own_frame_set():
    import better_human as bh
    slots = contact_set("frames35")
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    body = bh.SMPLX(model_path=cfg["model"]["smplx"]["model_path"], gender="neutral",
                    num_betas=10, use_hands=True, use_face=False, compute_mass=False,
                    contact_frames=str(slots.frame_json), dtype=torch.float32, device="cpu")
    frames = body.contact_frames
    assert tuple(frames.names) == slots.slot_names
    assert tuple(frames.parent_joints) == slots.parent_joint52
    assert tuple(frames.vertex_ids) == slots.vertex_ids
    assert len(frames.frame_ids) == 35


def test_parent22_folds_the_fingers_onto_the_wrists():
    slots = contact_set("frames35")
    assert body22_parent(21) == 21 and body22_parent(36) == 20 and body22_parent(51) == 21
    for (name, parent52, _), parent22 in zip(FRAMES35, slots.parent_joint22):
        assert 0 <= parent22 < 22, name
        assert parent22 == parent52 if parent52 < 22 else parent22 in (20, 21), name
    # thumb_left (36) hangs off the left wrist, thumb_right (51) off the right one.
    by_name = dict(zip(slots.slot_names, slots.parent_joint22))
    assert by_name["thumb_left"] == 20 and by_name["thumb_right"] == 21
    assert by_name["palm_distal_left"] == 20 and by_name["finger_right"] == 21


def test_kindyn6_is_its_own_identity():
    slots = contact_set("kindyn6")
    assert slots.count == NUM_KINDYN_GROUPS and not slots.uses_frames
    assert slots.slot_names == KINDYN_GROUP_NAMES
    assert slots.parent_joint52 == KINDYN_GROUP_JOINTS == slots.parent_joint22
    assert slots.group_of.tolist() == list(range(NUM_KINDYN_GROUPS))
    values = np.arange(2 * 6, dtype=np.float64).reshape(2, 6)
    assert np.array_equal(slots.fold_max(values), values)
    vectors = np.arange(2 * 6 * 3, dtype=np.float64).reshape(2, 6, 3)
    assert np.array_equal(slots.fold_sum(vectors), vectors)


def test_frames35_folds_onto_the_six_groups():
    slots = contact_set("frames35")
    group_of = slots.group_of
    by_name = dict(zip(slots.slot_names, group_of.tolist()))
    # The hands take the palms and every finger frame, the toe groups the foot joint's
    # frames, the heel groups the ankle's; torso / head / knee frames belong to none.
    assert by_name["palm_left"] == by_name["thumb_left"] == KINDYN_GROUP_NAMES.index("left_hand")
    assert by_name["big_toe_right"] == by_name["ball_inner_right"] == \
        KINDYN_GROUP_NAMES.index("right_foot")
    assert by_name["heel_left"] == KINDYN_GROUP_NAMES.index("left_ankle")
    assert by_name["chest"] == by_name["head"] == by_name["knee_left"] == -1
    assert int((group_of >= 0).sum()) == 18 and int((group_of < 0).sum()) == 17

    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 35))
    folded = slots.fold_max(values)
    assert folded.shape == (4, 6)
    for group in range(6):
        members = np.flatnonzero(group_of == group)
        assert np.allclose(folded[:, group], values[:, members].max(axis=-1))
    vectors = rng.normal(size=(4, 35, 3))
    summed = slots.fold_sum(vectors)
    assert summed.shape == (4, 6, 3)
    for group in range(6):
        members = np.flatnonzero(group_of == group)
        assert np.allclose(summed[:, group], vectors[:, members].sum(axis=1))


def test_unknown_set_raises():
    with pytest.raises(ValueError):
        contact_set("frames36")
