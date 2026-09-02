"""Datasets: clip loaders over the ClimbingVideos corpus.

The frame and batch schemas live in :mod:`data.base` and :mod:`data.collate`.
A dataset yaml names the source and its on-disk options only
(``configs/datasets/climbing_videos.yaml``); WHICH ground-truth signals load is
derived from the enabled losses and passed in as ``needs``.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import Clip, ClipDataset
from .climbing_videos import ClimbingVideosDataset
from .collate import batch_to_device, make_collate
from .loaders import build_loaders, set_epoch

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {ClimbingVideosDataset.name: ClimbingVideosDataset}

__all__ = [
    "Clip", "ClipDataset", "ClimbingVideosDataset", "DATASETS", "REPO_ROOT",
    "batch_to_device", "build_datasets", "build_loaders", "make_collate", "set_epoch",
]


def build_datasets(
    cfg: dict, needs: set[str], *, limit_scenes: int | None = None,
) -> tuple[list[ClipDataset], list[ClipDataset]]:
    """Build the train and test datasets listed in ``data.datasets``.

    :param needs: signal groups the enabled losses require
        (``forces`` / ``motion`` / ``pose`` / ``keypoints``).
    :param limit_scenes: keep only the first N scenes of every split (smoke runs).
    :returns: ``(train_sets, test_sets)`` — one of each per listed dataset yaml.
    """
    dcfg = cfg["data"]
    clip = dcfg["clip"]
    train_sets: list[ClipDataset] = []
    test_sets: list[ClipDataset] = []
    for entry in dcfg["datasets"]:
        path = Path(entry)
        spec = yaml.safe_load(
            (path if path.is_absolute() else REPO_ROOT / path).read_text())
        name = spec["name"]
        if name not in DATASETS:
            raise ValueError(
                f"{entry}: unknown dataset {name!r}; known: {sorted(DATASETS)}")
        root = Path(spec["root"])
        common = dict(
            root=root,
            contact_level=int(spec["contact_level"]),
            clip_frames=int(clip["frames"]),
            stride=clip["stride"],
            seed=int(dcfg["seed"]),
            load=set(needs),
            embedding_dir=(root / "features" / "embedding"
                           if bool(dcfg["embedding_cache"]) else None),
            motion_smooth_sec=float(cfg["motion_supervision"]["target_smooth_sec"]),
            motion_outlier_acc_ms2=float(
                cfg["motion_supervision"]["outlier_acc_ms2"]),
        )
        cls = DATASETS[name]

        def scenes(split: str):
            return None if limit_scenes is None else cls.list_scenes(root, split)[:limit_scenes]

        train_sets.append(cls(
            scenes=scenes("train"), split="train", jitter=bool(clip["jitter"]), **common))
        test_sets.append(cls(
            scenes=scenes("test"), split="test", jitter=False, full_scenes=True,
            max_frames=int(dcfg["eval_max_frames"]), **common))
    return train_sets, test_sets
