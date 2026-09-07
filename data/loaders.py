"""DataLoaders over clip datasets: batch budget, DDP sharding, epoch handoff.

Training draws ``frames_per_batch // clip.frames`` clips per batch (per GPU), so
memory stays flat as ``T`` changes. With ``data.interleave_videos`` the clips of
one epoch are dealt out round-robin over the SOURCE VIDEOS
(:class:`VideoInterleavedSampler`), so the global batch of a step comes from as
many different videos as it has clips; otherwise clips are drawn uniformly at
random. Evaluation runs the whole-scene protocol: one clip per batch, sharded
across ranks without padding, in dataset order.

Workers are re-forked every epoch (``persistent_workers=False``) so the
stateless window jitter sees the epoch :func:`set_epoch` just set; the re-fork
is cheap because the scene index is built once in the parent.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .base import ClipDataset
from .collate import make_collate
from .transforms import crop_size


class DistributedEvalSampler(Sampler[int]):
    """Exact, non-padding evaluation shard: rank ``r`` takes rows ``r::world``."""

    def __init__(self, dataset, *, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} outside [0, {self.num_replicas})")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def clip_videos(dataset) -> list[str]:
    """Source-video id of every clip of a (concatenated) clip dataset.

    Scene ids are ``<video_id>_<idx:04d>`` (the corpus DB's convention), so
    the video is everything before the last underscore.
    """
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    return [clip.scene.rsplit("_", 1)[0] for part in parts for clip in part.clips]


class VideoInterleavedSampler(Sampler[int]):
    """Epoch permutation that spreads consecutive clips over distinct source videos.

    Per epoch the clips of each video are shuffled, then dealt out in rounds: a
    round visits every video that still has clips (in a fresh random order) and
    takes one clip from each. Consecutive positions of the resulting stream come
    from different videos for as long as enough videos remain, so a batch of
    ``B`` clips — or, under DDP, the ``world x B`` clips of one step — samples
    ``B`` (``world x B``) different videos whenever the corpus allows. Every
    clip is visited exactly once per epoch (``drop_last`` aside).

    Under DDP the stream is cut into blocks of ``world x batch_size``; rank
    ``r`` takes the ``r``-th slice of ``batch_size`` of every block, so the
    per-rank batches of one step are disjoint slices of one video-diverse
    block. Deterministic in ``(seed, epoch)``.

    :param videos: the source video of every dataset index.
    :param batch_size: clips per rank per step.
    :param num_replicas: DDP world size.
    :param rank: this rank.
    :param seed: permutation seed.
    """

    def __init__(self, videos: Sequence[str], batch_size: int, *, num_replicas: int = 1,
                 rank: int = 0, seed: int = 0):
        self.videos = list(videos)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} outside [0, {self.num_replicas})")
        block = self.batch_size * self.num_replicas
        self.num_blocks = len(self.videos) // block

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _stream(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch])
        groups: dict[str, list[int]] = {}
        for index, video in enumerate(self.videos):
            groups.setdefault(video, []).append(index)
        queues = {video: list(rng.permutation(indices)) for video, indices in groups.items()}
        stream: list[int] = []
        while queues:
            for video in rng.permutation(sorted(queues)):
                stream.append(int(queues[video].pop()))
                if not queues[video]:
                    del queues[video]
        return np.asarray(stream, dtype=np.int64)

    def __iter__(self):
        block = self.batch_size * self.num_replicas
        stream = self._stream()[: self.num_blocks * block].reshape(self.num_blocks, block)
        mine = stream[:, self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
        return iter(mine.reshape(-1).tolist())

    def __len__(self) -> int:
        return self.num_blocks * self.batch_size


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Announce the epoch to a loader's sampler and to every clip dataset."""
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    dataset = loader.dataset
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for part in parts:
        if hasattr(part, "set_epoch"):
            part.set_epoch(epoch)


def build_loaders(
    cfg: dict,
    train_sets: Sequence[ClipDataset],
    test_sets: Sequence[ClipDataset],
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader | None, DataLoader]:
    """Train and test loaders over the built datasets (``None`` for an empty train list).

    :param train_sets: shuffled, ``drop_last``, ``DistributedSampler`` under DDP.
    :param test_sets: one clip per batch, sharded exactly, dataset order.
    """
    dcfg = cfg["data"]
    collate = make_collate(crop_size(cfg["model"]["checkpoint_path"]))
    clips_per_batch = max(1, int(dcfg["frames_per_batch"]) // int(dcfg["clip"]["frames"]))
    seed = int(dcfg["seed"])
    num_workers = int(dcfg["num_workers"])

    def loader(datasets: Sequence[ClipDataset], batch_size: int, shuffle: bool):
        dataset = (datasets[0] if len(datasets) == 1 else ConcatDataset(datasets))
        sampler = None
        if shuffle and bool(dcfg["interleave_videos"]):
            sampler = VideoInterleavedSampler(
                clip_videos(dataset), batch_size, num_replicas=world_size, rank=rank, seed=seed)
        elif world_size > 1:
            sampler = (
                DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                   shuffle=True, seed=seed, drop_last=True)
                if shuffle else
                DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank))
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
            sampler=sampler, num_workers=num_workers, drop_last=shuffle,
            collate_fn=collate, pin_memory=True, persistent_workers=False)

    return (loader(train_sets, clips_per_batch, True) if train_sets else None,
            loader(test_sets, 1, False))
