"""DataLoaders over clip datasets: batch budget, DDP sharding, epoch handoff.

Training draws ``frames_per_batch // clip.frames`` clips per batch (per GPU), so
memory stays flat as ``T`` changes. With ``data.interleave_videos`` the clips of
one epoch are dealt out round-robin over the SOURCE VIDEOS
(:class:`VideoInterleavedSampler`), so the global batch of a step comes from as
many different videos as it has clips; otherwise clips are drawn uniformly at
random. Evaluation runs the whole-scene protocol: one clip per batch, sharded
across ranks without padding, in dataset order.

With ``data.epoch_dataset`` the train micro-batches ALTERNATE over the listed
datasets (:class:`AlternatingSampler`): each dataset keeps its own video-interleaved
stream, cut to the epoch dataset's length, and micro-batch ``k`` of an epoch comes
from dataset ``k mod D`` — an optimizer step of ``D`` accumulated micro-batches
holds one from each. The test split is one loader per dataset.

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

    Each dataset maps its own scene ids onto videos (``video_of``): ClimbingVideos
    scenes are ``<video_id>_<idx:04d>``, a BEDLAM scene is its own render.
    """
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    return [part.video_of(clip.scene) for part in parts for clip in part.clips]


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

    With ``fraction`` < 1 an epoch keeps only the first ``round(fraction * n)``
    entries of that stream — it is already a video-diverse random permutation, so
    the kept prefix is a video-diverse random SUBSET, and a different one every
    epoch (the stream is seeded by ``(seed, epoch)``).

    :param videos: the source video of every dataset index.
    :param batch_size: clips per rank per step.
    :param num_replicas: DDP world size.
    :param rank: this rank.
    :param seed: permutation seed.
    :param fraction: share of the clips one epoch draws.
    :param max_blocks: cap on the epoch's ``world x batch_size`` blocks (the
        alternating loader cuts every stream to the epoch dataset's).
    """

    def __init__(self, videos: Sequence[str], batch_size: int, *, num_replicas: int = 1,
                 rank: int = 0, seed: int = 0, fraction: float = 1.0,
                 max_blocks: int | None = None):
        self.videos = list(videos)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.fraction = float(fraction)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} outside [0, {self.num_replicas})")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction!r}")
        block = self.batch_size * self.num_replicas
        self.num_blocks = int(round(self.fraction * len(self.videos))) // block
        if max_blocks is not None:
            self.num_blocks = min(self.num_blocks, int(max_blocks))

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


class AlternatingSampler(Sampler[int]):
    """Per-rank index stream over a :class:`ConcatDataset` whose micro-batches alternate
    over its parts: batch ``k`` is the ``k // D``-th batch of part ``k mod D``.

    Every part's sampler yields the same number of batches (``max_blocks``), so an
    epoch is ``D x blocks`` micro-batches on every rank.

    :param samplers: one per-rank sampler per part, in part order.
    :param offsets: each part's first index in the concatenated dataset.
    :param batch_size: clips per micro-batch.
    """

    def __init__(self, samplers: Sequence[VideoInterleavedSampler], offsets: Sequence[int],
                 batch_size: int):
        self.samplers = list(samplers)
        self.offsets = [int(o) for o in offsets]
        self.batch_size = int(batch_size)
        blocks = {s.num_blocks for s in self.samplers}
        if len(blocks) != 1:
            raise ValueError(f"the parts' samplers disagree on the epoch length: {blocks}")
        self.num_blocks = blocks.pop()

    def set_epoch(self, epoch: int) -> None:
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def __iter__(self):
        streams = [np.asarray(list(s), dtype=np.int64).reshape(self.num_blocks, self.batch_size)
                   + offset for s, offset in zip(self.samplers, self.offsets)]
        return iter(np.stack(streams, axis=1).reshape(-1).tolist())   # (blocks, D, B)

    def __len__(self) -> int:
        return self.num_blocks * len(self.samplers) * self.batch_size


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
) -> tuple[DataLoader | None, list[tuple[str, DataLoader]]]:
    """Train loader and one named test loader per dataset over the built datasets.

    :param train_sets: shuffled, ``drop_last``, ``DistributedSampler`` under DDP; with
        ``data.epoch_dataset`` the micro-batches alternate over the sets.
    :param test_sets: one clip per batch, sharded exactly, dataset order.
    :returns: ``(train_loader | None, [(dataset name, test loader), ...])`` — the
        epoch dataset (else the first listed) first.
    """
    dcfg = cfg["data"]
    collate = make_collate(crop_size(cfg["model"]["checkpoint_path"]))
    clips_per_batch = max(1, int(dcfg["frames_per_batch"]) // int(dcfg["clip"]["frames"]))
    seed = int(dcfg["seed"])
    num_workers = int(dcfg["num_workers"])
    fraction = float(dcfg["epoch_fraction"])
    epoch_dataset = dcfg["epoch_dataset"]
    if fraction < 1.0 and not bool(dcfg["interleave_videos"]):
        raise ValueError(
            "data.epoch_fraction < 1 draws the subset from the video-interleaved "
            "stream: data.interleave_videos must be on")

    def make(dataset, batch_size: int, shuffle: bool, sampler) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
            sampler=sampler, num_workers=num_workers, drop_last=shuffle,
            collate_fn=collate, pin_memory=False, persistent_workers=False)

    def interleaved(dataset, max_blocks=None, part_fraction=fraction):
        return VideoInterleavedSampler(
            clip_videos(dataset), clips_per_batch, num_replicas=world_size, rank=rank,
            seed=seed, fraction=part_fraction, max_blocks=max_blocks)

    def train_loader() -> DataLoader:
        if epoch_dataset is not None:
            anchor = next(i for i, d in enumerate(train_sets) if d.name == epoch_dataset)
            blocks = interleaved(train_sets[anchor]).num_blocks
            samplers = [interleaved(d, blocks, fraction if i == anchor else 1.0)
                        for i, d in enumerate(train_sets)]
            dataset = ConcatDataset(train_sets)
            offsets = [0, *dataset.cumulative_sizes[:-1]]
            return make(dataset, clips_per_batch, True,
                        AlternatingSampler(samplers, offsets, clips_per_batch))
        dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        if bool(dcfg["interleave_videos"]):
            sampler = interleaved(dataset)
        elif world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=True, seed=seed, drop_last=True)
        else:
            sampler = None
        return make(dataset, clips_per_batch, True, sampler)

    def test_loader(dataset) -> DataLoader:
        sampler = (DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
                   if world_size > 1 else None)
        return make(dataset, 1, False, sampler)

    tests = sorted(test_sets, key=lambda d: d.name != epoch_dataset)   # the epoch dataset first
    return (train_loader() if train_sets else None,
            [(dataset.name, test_loader(dataset)) for dataset in tests])
