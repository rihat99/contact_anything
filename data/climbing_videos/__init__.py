"""ClimbingVideos corpus loader: scenes, kindyn GT, MHR GT, clip dataset."""
from __future__ import annotations

from .dataset import ClimbingVideosDataset
from .scene import (
    GROUP_NAMES,
    NUM_GROUPS,
    embedding_path,
    list_scenes,
    list_test_scenes,
    list_train_scenes,
    scene_shard,
)

__all__ = [
    "ClimbingVideosDataset",
    "GROUP_NAMES",
    "NUM_GROUPS",
    "embedding_path",
    "list_scenes",
    "list_test_scenes",
    "list_train_scenes",
    "scene_shard",
]
