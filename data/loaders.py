"""DataLoaders over clip datasets: batch budget, DDP sharding, epoch handoff.

Training draws ``frames_per_batch // clip.frames`` clips per batch (per GPU), so
memory stays flat as ``T`` changes. Evaluation runs the whole-scene protocol:
one clip per batch, sharded across ranks without padding, in dataset order.

Workers are re-forked every epoch (``persistent_workers=False``) so the
stateless window jitter sees the epoch :func:`set_epoch` just set; the re-fork
is cheap because the scene index is built once in the parent.
"""
from __future__ import annotations

from typing import Sequence

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
        if world_size > 1:
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
