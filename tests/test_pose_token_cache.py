"""The cached-pose-token path: loader frames and the network's decoder bypass."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.climbing_videos.dataset import ClimbingVideosDataset
from data.climbing_videos.scene import list_test_scenes, pose_token_path
from data.collate import make_collate

CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")
needs_corpus = pytest.mark.skipif(
    not (CORPUS / "features" / "pose_token").is_dir(), reason="corpus pose-token cache not on this box")


@needs_corpus
def test_cached_frames_carry_the_token_and_no_pixels():
    scene = list_test_scenes(CORPUS)[0]
    ds = ClimbingVideosDataset(
        CORPUS, [scene], split="test", clip_frames=4, stride=1, jitter=False,
        full_scenes=True, max_frames=4, pose_token_dir=CORPUS / "features" / "pose_token")
    clip = ds[0]
    frame = clip[0]
    assert frame["image"] is None and frame["mask"] is None and frame["geometry_only"]
    assert frame["pose_token"].dtype == torch.bfloat16 and frame["pose_token"].shape == (1024,)
    assert frame["pose_token"].float().abs().sum() > 0
    # img_wh comes from the cache and must equal the JPEG's real size
    data = ds._scenes[scene] if scene in ds._scenes else ds._load_scene(scene)
    position = int(frame["key"].split("@")[1])
    with Image.open(data["frames_dir"] / f"{position:06d}.jpg") as im:
        assert tuple(frame["img_wh"]) == im.size
    batch = make_collate((512, 512))([clip])
    assert "img" not in batch and "mask" not in batch and "embedding" not in batch
    assert batch["pose_token"].shape == (4, 1024) and batch["bbox_center"].shape == (4, 2)
    assert np.array_equal(batch["ori_img_size"][0].numpy(), np.array(frame["img_wh"]))


@needs_corpus
def test_cache_layout_matches_the_scene():
    scene = list_test_scenes(CORPUS)[0]
    cache = np.load(pose_token_path(CORPUS / "features" / "pose_token", scene))
    assert cache["tokens"].dtype == np.int16 and cache["tokens"].shape[-1] == 1024
    assert cache["valid"].shape == cache["tokens"].shape[:2]
    assert cache["img_wh"].shape == (cache["tokens"].shape[1], 2)


def test_pose_token_and_embedding_caches_are_exclusive():
    with pytest.raises(ValueError, match="supersedes"):
        ClimbingVideosDataset(
            CORPUS, ["x"], split="test", embedding_dir="a", pose_token_dir="b")
