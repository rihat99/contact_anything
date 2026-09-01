"""Deterministic train/val split of corpus scenes, grouped by source video.

The single source of truth for the seed-42 split. ``make_loaders`` (training)
and the corpus loader both call it, so they issue identical RNG calls and
therefore produce identical splits. Chunks of one source video never straddle
the split.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def video_id_from_scene(scene_id: str) -> str:
    """Source-video id of a ClimbingVideos scene: drop a trailing ``_NNNN`` chunk.

    ``scene_id`` is ``{video}_{chunk:04d}`` where the video id may itself contain
    underscores (e.g. ``2nwG1Qa5k_c_0000`` -> ``2nwG1Qa5k_c``). Only the last
    ``_<digits>`` suffix is stripped.
    """
    head, sep, tail = scene_id.rpartition("_")
    return head if (sep and head and tail.isdigit()) else scene_id


def group_train_val_split(
    groups: Iterable[str], val_ratio: float = 0.15, seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split *unique* group ids into ``(train_groups, val_groups)`` — no group crosses.

    Permutes the sorted unique group ids with a NumPy ``default_rng`` seeded by
    ``seed`` and assigns the first ``round(n_groups * val_ratio)`` to validation.
    Duplicate ids in ``groups`` are collapsed first, so every member of a group
    lands in the same split.
    """
    unique = sorted(set(groups))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(unique))
    n_val = int(round(len(unique) * float(val_ratio)))
    val = {unique[int(i)] for i in perm[:n_val]}
    train = {unique[int(i)] for i in perm[n_val:]}
    return train, val

