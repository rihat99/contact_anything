"""Unified contact dataset — concatenates DAMON, LEMON, and RICH.

All sub-datasets share the SMPL 6890-vertex topology and return the
same dict schema (see ``damon.py`` for the contract). The unified
dataset just wraps them in an index-translation layer so a single
DataLoader can iterate the combined corpus.

Subsets are addressable by name (``"damon"``, ``"lemon"``, ``"rich"``);
pass ``names=[...]`` to use only a subset.
"""

from typing import Optional, Sequence

from torch.utils.data import Dataset

from .damon import DamonDataset
from .lemon import LemonDataset
from .rich import RichDataset


_BUILDERS = {
    "damon": DamonDataset,
    "lemon": LemonDataset,
    "rich":  RichDataset,
}


class ContactDataset(Dataset):
    def __init__(
        self,
        names: Sequence[str] = ("damon", "lemon", "rich"),
        roots: Optional[dict[str, str]] = None,
        splits: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        roots = roots or {}
        splits = splits or {}

        self.names: list[str] = []
        self.subsets: list[Dataset] = []
        for n in names:
            if n not in _BUILDERS:
                raise ValueError(f"Unknown dataset {n!r}; choose from {list(_BUILDERS)}")
            kwargs = {}
            if n in roots:
                kwargs["root"] = roots[n]
            if n in splits:
                kwargs["split"] = splits[n]
            self.subsets.append(_BUILDERS[n](**kwargs))
            self.names.append(n)

        self.cum_sizes = []
        total = 0
        for ds in self.subsets:
            total += len(ds)
            self.cum_sizes.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        prev = 0
        for i, end in enumerate(self.cum_sizes):
            if idx < end:
                return i, idx - prev
            prev = end
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> dict:
        sub_idx, local_idx = self._locate(idx)
        return self.subsets[sub_idx][local_idx]

    def sizes(self) -> dict[str, int]:
        return {n: len(ds) for n, ds in zip(self.names, self.subsets)}
