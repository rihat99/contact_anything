"""Rank-0 tensorboard scalar logging for a run.

One :class:`SummaryWriter` under ``<run>/tensorboard``. Non-main ranks build the
same object and drop everything, so callers never branch on rank.

Tag layout (the first path segment is a tensorboard section):

* ``optim/`` — lr, gradient norm, throughput, epoch time.
* ``loss_train/`` / ``loss_test/`` — ``total`` and one ``<loss>.<term>`` per
  weighted term (the same names on both splits, so overfitting reads off
  directly).
* ``metric_<group>/`` — the reported metrics of each loss (pose: mpjpe,
  pa_mpjpe, pve, accel; contact: f1, precision, recall, iou, per-group f1).

A frozen-baseline dict (``output.frozen_metrics``) is written as a SECOND run,
``<run>/tensorboard/frozen``, at every evaluation step with the same
``metric_*`` tags — tensorboard overlays it on the live curves as a flat line
without adding a single card.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class Logger:
    """Write flat ``{tag: float}`` scalar dicts to tensorboard on rank 0.

    :param out_dir: run directory; the event files go to ``out_dir/tensorboard``.
    :param enabled: ``False`` on non-main ranks — every call becomes a no-op.
    :param frozen: ``{tag: value}`` constants drawn as the ``frozen`` run.
    """

    def __init__(self, out_dir: str | Path, enabled: bool = True,
                 frozen: Optional[dict] = None):
        self.writer = None
        self.frozen_writer = None
        self.frozen = dict(frozen or {})
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(Path(out_dir) / "tensorboard"))
            if self.frozen:
                self.frozen_writer = SummaryWriter(
                    str(Path(out_dir) / "tensorboard" / "frozen"))

    def log(self, scalars: dict, step: int) -> None:
        """Log every ``{tag: value}`` pair at ``step``."""
        if self.writer is None:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, int(step))

    def log_frozen(self, step: int) -> None:
        """Re-emit the frozen-baseline constants at ``step`` (one flat line per tag)."""
        if self.frozen_writer is None:
            return
        for tag, value in self.frozen.items():
            self.frozen_writer.add_scalar(tag, value, int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.frozen_writer is not None:
            self.frozen_writer.close()
