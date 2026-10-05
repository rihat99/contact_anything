"""Datasets: clip loaders over the annotated corpora.

The frame and batch schemas live in :mod:`data.base` and :mod:`data.collate`.
A dataset yaml names the source and its own on-disk options
(``configs/datasets/*.yaml``) and is handed to its class's ``from_spec``; WHICH
ground-truth signals load is derived from the enabled losses and passed in as
``needs``, and which SLOTS contact and force are indexed by comes from
``data.contact_set``.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import Clip, ClipDataset
from .bedlam2 import Bedlam2Dataset
from .climbing_videos import ClimbingVideosDataset
from .collate import batch_to_device, make_collate
from .loaders import build_loaders, set_epoch

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {cls.name: cls for cls in (ClimbingVideosDataset, Bedlam2Dataset)}

__all__ = [
    "Bedlam2Dataset", "Clip", "ClipDataset", "ClimbingVideosDataset", "DATASETS",
    "REPO_ROOT", "batch_to_device", "build_datasets", "build_loaders", "make_collate",
    "set_epoch",
]


def build_datasets(
    cfg: dict, needs: set[str], *, limit_scenes: int | None = None,
) -> tuple[list[ClipDataset], list[ClipDataset]]:
    """Build the train and test datasets listed in ``data.datasets``.

    The backbone cache is read on a corpus only when both the run
    (``data.embedding_cache``) and its dataset yaml (``embedding_cache``) say so.

    :param needs: signal groups the enabled losses require (``forces`` / ``smplx``).
    :param limit_scenes: keep only the first N scenes of every split (smoke runs).
    :returns: ``(train_sets, test_sets)`` — one of each per listed dataset yaml.
    """
    dcfg = cfg["data"]
    clip = dcfg["clip"]
    train_sets: list[ClipDataset] = []
    test_sets: list[ClipDataset] = []
    common = dict(
        clip_frames=int(clip["frames"]),
        stride=clip["stride"],
        seed=int(dcfg["seed"]),
        load=set(needs),
        contact_set=str(dcfg["contact_set"]),
        pose_token_cache=bool(dcfg["pose_token_cache"]),
    )
    for entry in dcfg["datasets"]:
        path = Path(entry)
        spec = yaml.safe_load(
            (path if path.is_absolute() else REPO_ROOT / path).read_text())
        name = spec["name"]
        if name not in DATASETS:
            raise ValueError(
                f"{entry}: unknown dataset {name!r}; known: {sorted(DATASETS)}")
        cls = DATASETS[name]

        def scenes(split: str):
            if limit_scenes is None:
                return None
            return cls.scene_ids(spec, split)[:limit_scenes]

        cache = bool(dcfg["embedding_cache"]) and bool(spec["embedding_cache"])
        train_sets.append(cls.from_spec(
            spec, scenes=scenes("train"), split="train", embedding_cache=cache,
            jitter=bool(clip["jitter"]), **common))
        test_sets.append(cls.from_spec(
            spec, scenes=scenes("test"), split="test", embedding_cache=cache, jitter=False,
            full_scenes=True, max_frames=int(dcfg["eval_max_frames"]), **common))
    return train_sets, test_sets
