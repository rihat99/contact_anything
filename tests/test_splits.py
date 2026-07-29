"""Splits: video-group disjointness, stills determinism, stateless jitter resume."""
from __future__ import annotations

from pathlib import Path

import pytest

from contact.data.splits import (
    group_train_val_split,
    train_val_indices,
    video_id_from_scene,
)

_CORPUS_ROOT = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
_HAS_CORPUS = (_CORPUS_ROOT / "scenes" / "scenes.db").is_file()


def test_video_id_strips_only_chunk_suffix():
    assert video_id_from_scene("0aow0AvNZ2A_0004") == "0aow0AvNZ2A"
    assert video_id_from_scene("2nwG1Qa5k_c_0000") == "2nwG1Qa5k_c"   # video id keeps its underscore
    assert video_id_from_scene("noChunkHere") == "noChunkHere"        # nothing to strip


def test_group_split_disjoint_and_covers_all():
    groups = [f"vid{i}" for i in range(40)]
    train, val = group_train_val_split(groups, val_ratio=0.25, seed=42)
    assert train.isdisjoint(val)
    assert train | val == set(groups)
    assert len(val) == 10


def test_group_split_keeps_all_members_of_a_group_together():
    # duplicated group ids (chunks) must never straddle the split
    scenes = [f"vidA_{c:04d}" for c in range(5)] + [f"vidB_{c:04d}" for c in range(3)]
    groups = [video_id_from_scene(s) for s in scenes]
    train, val = group_train_val_split(groups, val_ratio=0.5, seed=7)
    assert train.isdisjoint(val)
    for split in (train, val):
        assert split <= {"vidA", "vidB"}


def test_group_split_is_seed_deterministic():
    groups = [f"vid{i}" for i in range(50)]
    a = group_train_val_split(groups, 0.2, seed=123)
    b = group_train_val_split(groups, 0.2, seed=123)
    assert a == b
    c = group_train_val_split(groups, 0.2, seed=999)
    assert a != c


def test_stills_split_deterministic_and_partition():
    train, val = train_val_indices(1000, 0.15, 42)
    assert train == train_val_indices(1000, 0.15, 42)[0]         # reproducible
    assert len(val) == 150
    assert set(train) | set(val) == set(range(1000))
    assert set(train).isdisjoint(val)


@pytest.mark.skipif(not _HAS_CORPUS, reason="ClimbingVideos corpus not present")
def test_video_group_split_no_video_crosses():
    from contact.data.climbing_corpus import list_corpus_scenes
    scenes = list_corpus_scenes(_CORPUS_ROOT, "train")
    vids = [video_id_from_scene(s) for s in scenes]
    train, val = group_train_val_split(vids, 0.15, 42)
    train_scene_vids = {video_id_from_scene(s) for s in scenes if video_id_from_scene(s) in train}
    val_scene_vids = {video_id_from_scene(s) for s in scenes if video_id_from_scene(s) in val}
    assert train_scene_vids.isdisjoint(val_scene_vids)


@pytest.mark.skipif(not _HAS_CORPUS, reason="ClimbingVideos corpus not present")
def test_stateless_jitter_reproducible_across_fresh_instances():
    # Two *independent* dataset instances (a resume) must pick identical windows
    # for the same (seed, epoch) — the jitter must not depend on process state.
    from contact.data.climbing_corpus import ClimbingCorpusDataset, list_corpus_scenes
    scenes = list_corpus_scenes(_CORPUS_ROOT, "train")[:6]

    def keys_for_epoch(epoch):
        ds = ClimbingCorpusDataset(_CORPUS_ROOT, scenes=scenes, split="train",
                                   frames_per_clip=8, frame_stride=2, jitter=True,
                                   seed=42, load_images=False)
        ds.set_epoch(epoch)
        return [ds[i][0]["key"] for i in range(min(len(ds), 12))]

    assert keys_for_epoch(0) == keys_for_epoch(0)        # resume-safe
    assert keys_for_epoch(0) != keys_for_epoch(1)        # epoch actually re-jitters


@pytest.mark.skipif(not _HAS_CORPUS, reason="ClimbingVideos corpus not present")
def test_val_windows_are_deterministic_sliding():
    from contact.data.climbing_corpus import ClimbingCorpusDataset, list_corpus_scenes
    scenes = list_corpus_scenes(_CORPUS_ROOT, "train")[:6]
    ds = ClimbingCorpusDataset(_CORPUS_ROOT, scenes=scenes, split="val",
                               frames_per_clip=8, frame_stride=2, load_images=False)
    ds.set_epoch(0)
    first = [ds[i][0]["key"] for i in range(min(len(ds), 12))]
    ds.set_epoch(5)                                      # epoch must not move val windows
    assert [ds[i][0]["key"] for i in range(min(len(ds), 12))] == first
