"""Loader tests for the 35-slot contact set and the BEDLAM2_our corpus (CPU, real data).

* ClimbingVideos under ``frames35``: a train frame carries the file's own 35
  contact-frame labels (positives, supervision and the NaN -> 0 confidence rule),
  a test frame carries those AND the manual six-group block;
* ClimbingVideos under ``kindyn6`` is unchanged: the six-group forces are the
  independent fold of the raw kindyn wrenches, and the group labels stay the
  52 -> 22 -> 6 reduction;
* the BEDLAM split: deterministic, ~``test_fraction`` per render job, every job
  represented, train and test disjoint and exhaustive;
* one real BEDLAM scene: the emitted person mask IS the instance map's ``p + 1``,
  the GT joints project inside the visible box under the emitted camera, and the
  forces follow kindyn's body-weight / root-frame convention;
* ``video_of`` per corpus and the ``epoch_fraction`` prefix of the
  video-interleaved sampler.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.bedlam2 import Bedlam2Dataset, scene as bedlam_scene, split_scenes
from data.climbing_videos import ClimbingVideosDataset
from data.climbing_videos.kindyn import GRAVITY_MAG, quat_xyzw_to_matrix
from data.climbing_videos.scene import scene_shard
from data.loaders import VideoInterleavedSampler
from model.contact_frames import KINDYN_GROUPS_52, contact_set

CLIMBING_ROOT = Path("/home/rikhat.akizhanov/better/data/ClimbingVideos")
BEDLAM_ROOT = Path("/home/rikhat.akizhanov/better/data/BEDLAM2_our")
#: A BEDLAM scene whose boxes and instance masks are generated (10 people, 149 frames).
BEDLAM_SCENE = "0l000002"
CONTACT_LEVEL = 2

pytestmark = pytest.mark.skipif(
    not (CLIMBING_ROOT / "scenes" / "scenes.db").is_file()
    or not (BEDLAM_ROOT / "scenes" / "scenes.db").is_file(),
    reason="the corpora are not mounted")


def _climbing(scene: str, split: str, contact_set_name: str) -> ClimbingVideosDataset:
    return ClimbingVideosDataset(
        CLIMBING_ROOT, scenes=[scene], split=split, clip_frames=4, jitter=False,
        contact_level=CONTACT_LEVEL, contact_set=contact_set_name,
        load={"forces", "smplx"}, full_scenes=split == "test", max_frames=8)


def _contacts_npz(scene: str):
    return np.load(
        CLIMBING_ROOT / "features" / "human_optim" / scene_shard(scene) / scene
        / f"contacts_{CONTACT_LEVEL}.npz", allow_pickle=True)


def _kindyn_npz(scene: str):
    return np.load(
        CLIMBING_ROOT / "features" / "human_optim" / scene_shard(scene) / scene
        / "kindyn_1.npz", allow_pickle=True)


@pytest.fixture(scope="module")
def climbing_train_scene() -> str:
    return ClimbingVideosDataset.list_scenes(CLIMBING_ROOT, "train")[0]


@pytest.fixture(scope="module")
def climbing_test_scene() -> str:
    return ClimbingVideosDataset.list_scenes(CLIMBING_ROOT, "test")[0]


# ------------------------------------------------------------------ ClimbingVideos

def test_frames35_train_labels_are_the_files_own(climbing_train_scene):
    """The 35 slot labels, their supervision and confidence come straight from the npz."""
    slots = contact_set("frames35")
    dataset = _climbing(climbing_train_scene, "train", "frames35")
    raw = _contacts_npz(climbing_train_scene)
    assert tuple(str(x) for x in raw["contact_frame_names"]) == slots.slot_names
    contact = np.asarray(raw["frame_contact"], bool)
    conf = np.asarray(raw["frame_label_confidence"], np.float32)
    data = dataset.scene_data(climbing_train_scene)

    for frame in dataset[0]:
        person, position = 0, frame["frame_index"]
        assert frame["contact_gt"].shape == (35,)
        assert np.array_equal(
            frame["contact_gt"] > 0.5, contact[person, position])
        # supervised wherever the person is tracked
        assert np.array_equal(
            frame["contact_valid"] > 0,
            np.full(35, bool(data["valid_mask"][person, position])))
        expected_conf = np.where(
            np.isfinite(conf[person, position]), conf[person, position], 0.0)
        assert np.allclose(frame["contact_conf"], expected_conf)
    # a not-assessed row is never a positive label
    assert not bool((contact & ~np.isfinite(conf)).any())


def test_frames35_test_frame_carries_both_label_blocks(climbing_test_scene):
    """A frames35 test frame has the 35 slots AND the manual six-group labels."""
    slot_frame = _climbing(climbing_test_scene, "test", "frames35")[0][0]
    group_frame = _climbing(climbing_test_scene, "test", "kindyn6")[0][0]
    assert slot_frame["contact_gt"].shape == (35,)
    assert slot_frame["contact_gt_groups"].shape == (6,)
    assert slot_frame["contact_valid_groups"].shape == (6,)
    # the group block IS the kindyn6 manual label of the same frame
    assert np.array_equal(slot_frame["contact_gt_groups"], group_frame["contact_gt"])
    assert np.array_equal(
        slot_frame["contact_valid_groups"], group_frame["contact_valid"])
    # ... and the slot labels are the file's, not the annotation's
    raw = _contacts_npz(climbing_test_scene)
    assert np.array_equal(
        slot_frame["contact_gt"] > 0.5,
        np.asarray(raw["frame_contact"], bool)[0, slot_frame["frame_index"]])


def test_kindyn6_frame_has_no_group_block(climbing_test_scene):
    """Under kindyn6 the test labels ARE ``contact_gt``; no second block is emitted."""
    frame = _climbing(climbing_test_scene, "test", "kindyn6")[0][0]
    assert "contact_gt_groups" not in frame and "contact_valid_groups" not in frame
    assert frame["contact_gt"].shape == (6,) and frame["force_gt"].shape == (6, 3)


def test_kindyn6_forces_are_the_independent_fold(climbing_train_scene):
    """The six-group forces / levers match a from-scratch fold of the kindyn npz."""
    frame = _climbing(climbing_train_scene, "train", "kindyn6")[0][0]
    raw = _kindyn_npz(climbing_train_scene)
    position = frame["frame_index"]
    parents = np.asarray(raw["contact_frame_parents"], np.int64)
    forces = np.asarray(raw["frame_forces"], np.float32)[0, position]        # [35, 3] N
    joints = np.asarray(raw["joints_world"], np.float32)[0, position]        # [52, 3] m
    mass = float(np.asarray(raw["total_mass"], np.float32).reshape(-1)[0])
    rot = quat_xyzw_to_matrix(np.asarray(raw["q"], np.float32)[0, position, 3:7])

    expected = np.stack([
        forces[np.isin(parents, list(members))].sum(axis=0)
        for members in KINDYN_GROUPS_52]) / (mass * GRAVITY_MAG)
    lever = joints[list(contact_set("kindyn6").parent_joint52)] - joints[0]
    assert np.allclose(frame["force_gt"], expected @ rot, atol=1e-6)
    assert np.allclose(frame["force_lever"], lever @ rot, atol=1e-5)


def test_frames35_forces_fold_onto_the_kindyn6_forces(climbing_train_scene):
    """Summing a group's member slots reproduces the six-group forces exactly."""
    slots = contact_set("frames35")
    slot_frame = _climbing(climbing_train_scene, "train", "frames35")[0][0]
    group_frame = _climbing(climbing_train_scene, "train", "kindyn6")[0][0]
    assert slot_frame["force_gt"].shape == (35, 3)
    assert np.allclose(
        slots.fold_sum(np.asarray(slot_frame["force_gt"])),
        np.asarray(group_frame["force_gt"]), atol=1e-6)
    # each slot's lever is its PARENT joint's offset; a group's own joint is its lever
    group_slots = [int(np.flatnonzero(slots.group_of == g)[0]) for g in range(6)]
    for group, slot in enumerate(group_slots):
        parent = slots.parent_joint52[slot]
        if parent == contact_set("kindyn6").parent_joint52[group]:
            assert np.allclose(
                slot_frame["force_lever"][slot], group_frame["force_lever"][group],
                atol=1e-5)


# ------------------------------------------------------------------ BEDLAM2_our

def test_bedlam_split_is_deterministic_and_stratified():
    """2 % of every render job, one seeded draw, train and test disjoint + exhaustive."""
    jobs = bedlam_scene.scenes_by_job(BEDLAM_ROOT)
    split = split_scenes(BEDLAM_ROOT, test_fraction=0.02, split_seed=0)
    assert split == split_scenes(BEDLAM_ROOT, test_fraction=0.02, split_seed=0)
    assert split != split_scenes(BEDLAM_ROOT, test_fraction=0.02, split_seed=1)

    all_scenes = sorted(scene for ids in jobs.values() for scene in ids)
    assert sorted(split["train"] + split["test"]) == all_scenes
    assert not set(split["train"]) & set(split["test"])
    test = set(split["test"])
    for job, ids in jobs.items():
        picked = test.intersection(ids)
        assert len(picked) == max(1, int(round(0.02 * len(ids)))), job
    assert abs(len(test) / len(all_scenes) - 0.02) < 0.005


def test_bedlam_scene_masks_cameras_and_forces():
    """The instance-map mask, the camera and the force convention of one real scene."""
    slots = contact_set("frames35")
    dataset = Bedlam2Dataset(
        BEDLAM_ROOT, scenes=[BEDLAM_SCENE], split="train", clip_frames=4, jitter=False,
        contact_set="frames35", load={"forces", "smplx"})
    data = dataset.scene_data(BEDLAM_SCENE)
    directory = bedlam_scene.gt_dir(BEDLAM_ROOT, BEDLAM_SCENE)
    forces_npz = np.load(directory / "forces.npz", allow_pickle=True)

    people = [clip.person for clip in dataset.clips]
    assert set(people) == set(range(len(data["object_ids"])))
    for person in sorted(set(people))[:3]:
        index = people.index(person)
        frame = dataset[index][0]
        position = frame["frame_index"]

        instances = np.array(
            Image.open(directory / "masks" / f"{position:06d}.png"), np.uint8)
        assert np.array_equal(frame["mask"] > 0, instances == person + 1)

        # GT joints project inside the visible box under the emitted camera
        joints = frame["smplx_joints_world"][:22]
        cam = joints @ frame["cam_from_world"][:3, :3].T + frame["cam_from_world"][:3, 3]
        pixels = (cam @ frame["cam_int"].T)[:, :2] / cam[:, 2:3]
        box = frame["bbox"]
        assert (cam[:, 2] > 0).all()
        inside = ((pixels[:, 0] >= box[0] - 5) & (pixels[:, 0] <= box[2] + 5)
                  & (pixels[:, 1] >= box[1] - 5) & (pixels[:, 1] <= box[3] + 5))
        assert inside.mean() > 0.9, f"person {person}: {inside.mean():.2f} joints in box"

        # forces: world newtons / (mass g), rotated into the root frame
        mass = float(np.asarray(forces_npz["total_mass"], np.float32)[person])
        rot = quat_xyzw_to_matrix(
            np.asarray(forces_npz["q"], np.float32)[person, position, 3:7])
        world = np.asarray(forces_npz["frame_forces"], np.float32)[person, position]
        assert np.allclose(frame["force_gt"], (world / (mass * GRAVITY_MAG)) @ rot,
                           atol=1e-6)
        joints_world = np.asarray(forces_npz["joints_world"], np.float32)[person, position]
        lever = joints_world[list(slots.parent_joint52)] - joints_world[0]
        assert np.allclose(frame["force_lever"], lever @ rot, atol=1e-5)
        assert frame["force_conf"] == 1.0 and frame["gravity_measured"]


def test_bedlam_frame_schema_matches_climbing_videos(climbing_train_scene):
    """Both corpora emit the same frame keys, so a batch can mix them."""
    bedlam = Bedlam2Dataset(
        BEDLAM_ROOT, scenes=[BEDLAM_SCENE], split="train", clip_frames=4, jitter=False,
        contact_set="frames35", load={"forces", "smplx"})[0][0]
    climbing = _climbing(climbing_train_scene, "train", "frames35")[0][0]
    assert set(bedlam) == set(climbing)
    for key, value in bedlam.items():
        # the frame sizes differ between corpora; the per-frame RANK must not
        assert np.ndim(value) == np.ndim(climbing[key]), key
        if key not in ("image", "mask"):
            assert np.shape(value) == np.shape(climbing[key]), key


def test_bedlam_kindyn6_folds_the_35_frames():
    """Under kindyn6 a BEDLAM frame carries the six-group fold of its 35 slots."""
    slots = contact_set("frames35")
    common = dict(scenes=[BEDLAM_SCENE], split="train", clip_frames=4, jitter=False,
                  load={"forces"})
    slot_frame = Bedlam2Dataset(BEDLAM_ROOT, contact_set="frames35", **common)[0][0]
    group_frame = Bedlam2Dataset(BEDLAM_ROOT, contact_set="kindyn6", **common)[0][0]
    assert group_frame["contact_gt"].shape == (6,)
    assert np.array_equal(
        group_frame["contact_gt"] > 0.5,
        slots.fold_max(np.asarray(slot_frame["contact_gt"])) > 0.5)
    assert np.allclose(
        group_frame["force_gt"], slots.fold_sum(np.asarray(slot_frame["force_gt"])),
        atol=1e-6)


# ------------------------------------------------------------------ sampling

def test_video_of_per_corpus():
    """ClimbingVideos scenes share a video; a BEDLAM scene is its own."""
    assert ClimbingVideosDataset.video_of("-4dCpR9i6tQ_0025") == "-4dCpR9i6tQ"
    assert Bedlam2Dataset.video_of("0l000002") == "0l000002"


def test_epoch_fraction_keeps_a_different_video_diverse_subset_each_epoch():
    """``fraction`` truncates the per-epoch stream; the subset moves with the epoch."""
    videos = [f"video{i // 5}" for i in range(200)]
    full = VideoInterleavedSampler(videos, batch_size=4, seed=3)
    half = VideoInterleavedSampler(videos, batch_size=4, seed=3, fraction=0.5)
    assert len(full) == 200 and len(half) == 100

    epoch0 = list(half)
    half.set_epoch(1)
    epoch1 = list(half)
    assert len(set(epoch0)) == len(epoch0) == 100
    assert sorted(epoch0) != sorted(epoch1)
    # the kept prefix is still the video-diverse stream: consecutive clips differ
    assert all(videos[a] != videos[b] for a, b in zip(epoch0, epoch0[1:]))
    # over enough epochs every clip is eventually drawn
    seen: set[int] = set()
    for epoch in range(12):
        half.set_epoch(epoch)
        seen.update(half)
    assert seen == set(range(200))
