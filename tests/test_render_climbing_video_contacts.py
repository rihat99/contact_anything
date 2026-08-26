from __future__ import annotations

import numpy as np

from contact.targets import JOINT_SET_GROUPS
from scripts.render_climbing_video_contacts import (
    CONTACT_COLOR,
    FREE_COLOR,
    _draw_contacts,
    _scene_ground_truth,
    select_random_scenes,
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


def test_draw_contacts_uses_inner_label_and_outer_prediction_ring():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    probs = np.array([[0.9]], dtype=np.float32)
    points = np.array([[[50.0, 50.0]]], dtype=np.float32)
    labels = np.array([[False]])
    label_mask = np.array([[True]])

    _draw_contacts(
        frame, probs, points, threshold=0.5,
        frame_labels=labels, frame_label_mask=label_mask)

    assert tuple(int(v) for v in frame[50, 50]) == FREE_COLOR
    patch = frame[38:63, 38:63]
    assert np.any(np.all(patch == np.asarray(CONTACT_COLOR), axis=-1))


def test_scene_ground_truth_reduces_hands_and_ankle_or_foot():
    contact = np.zeros((1, 2, 22), dtype=bool)
    contact[0, 0, 20] = True      # left hand
    contact[0, 0, 10] = True      # left foot via foot/toe
    contact[0, 1, 8] = True       # right foot via ankle
    labels, known = _scene_ground_truth({
        "joint_contact": contact,
        "contact_conf": np.ones_like(contact, dtype=np.float32),
        "valid_mask": np.array([[True, False]]),
        "annotated": None,
    }, JOINT_SET_GROUPS["extremities_4"])

    np.testing.assert_array_equal(labels[0, 0], [True, False, True, False])
    np.testing.assert_array_equal(known[0, 0], [True, True, True, True])
    np.testing.assert_array_equal(known[0, 1], [False, False, False, False])
