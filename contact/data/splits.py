"""Deterministic train/val index split for the still-image contact datasets.

The single source of truth for the seed-42 random split. ``make_loaders``
(training) uses it directly; evaluate/demo re-derive held-out sets by calling
the same function, so all three callers issue identical RNG calls and therefore
produce identical splits.
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


def train_val_indices(n: int, val_ratio: float = 0.15, seed: int = 42) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)`` for a random split of ``range(n)``.

    Permutes ``range(n)`` with a NumPy ``default_rng`` seeded by ``seed`` and
    takes the first ``round(n * val_ratio)`` items as validation, the rest as
    train. This exact call sequence is what every caller must reproduce for the
    splits to match.
    """
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(int(n))
    n_val = int(round(int(n) * float(val_ratio)))
    return idx[n_val:].tolist(), idx[:n_val].tolist()
