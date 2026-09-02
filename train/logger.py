"""Rank-0 tensorboard scalar logging for a run.

One :class:`SummaryWriter` under ``<run>/tensorboard``. Non-main ranks build the
same object and drop everything, so callers never branch on rank.
"""
from __future__ import annotations

from pathlib import Path


class Logger:
    """Write flat ``{tag: float}`` scalar dicts to tensorboard on rank 0.

    :param out_dir: run directory; the event files go to ``out_dir/tensorboard``.
    :param enabled: ``False`` on non-main ranks — every call becomes a no-op.
    """

    def __init__(self, out_dir: str | Path, enabled: bool = True):
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(Path(out_dir) / "tensorboard"))

    def log(self, scalars: dict, step: int) -> None:
        """Log every ``{tag: value}`` pair at ``step``."""
        if self.writer is None:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
